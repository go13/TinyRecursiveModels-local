"""
LBVS - Latent Beam Value Search

A simple beam search model that operates in latent space:
1. Generate K candidate latent states via learned branch projections
2. Score each with a value head (cheap evaluation)
3. Keep top-B beams based on value scores
4. Decode only the best beam at the end

This is designed to be simple and extensible for future beam search variants.
"""

from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, RotaryEmbedding, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


class BranchStrategy(str, Enum):
    """Strategies for generating beam candidates."""
    LEARNED_PERTURBATION = "learned_perturbation"  # K learned delta projections
    STOCHASTIC = "stochastic"  # Add learned noise, sample K times
    # Future: ATTENTION_MODES, MIXTURE_OF_EXPERTS, etc.


@dataclass
class LBVSCarry:
    """Carry state for beam search."""
    z: torch.Tensor  # Current latent states [B, num_beams, L, D]
    beam_values: torch.Tensor  # Value scores for each beam [B, num_beams]
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class LBVSConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0

    # Architecture
    hidden_size: int
    num_heads: int
    expansion: float
    num_layers: int

    # Beam search config
    num_candidates: int = 4  # K: number of candidate beams to generate
    num_beams: int = 2  # B: number of beams to keep
    branch_strategy: str = "learned_perturbation"

    # Iteration config
    max_iterations: int
    halt_exploration_prob: float

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 8


class LBVSBlock(nn.Module):
    """Simple transformer block for processing latent states."""

    def __init__(self, config: LBVSConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        # Temporal MLP
        self.mlp_t = SwiGLU(
            hidden_size=config.seq_len + config.puzzle_emb_len,
            expansion=config.expansion,
        )

        # Channel MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B*num_beams, L, D] or [B, L, D]

        # Temporal mixing
        h = hidden_states.transpose(1, 2)
        h = self.mlp_t(h)
        h = h.transpose(1, 2)
        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        # Channel mixing
        h = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        return hidden_states


class BranchGenerator(nn.Module):
    """Generates K candidate latent states from current state."""

    def __init__(self, config: LBVSConfig):
        super().__init__()
        self.config = config
        self.strategy = BranchStrategy(config.branch_strategy)

        if self.strategy == BranchStrategy.LEARNED_PERTURBATION:
            # K different learned perturbation directions
            self.branch_projs = nn.ModuleList([
                CastedLinear(config.hidden_size, config.hidden_size, bias=True)
                for _ in range(config.num_candidates)
            ])
            # Initialize with small weights for small perturbations
            for proj in self.branch_projs:
                with torch.no_grad():
                    proj.weight.mul_(0.1)
                    proj.bias.zero_()

        elif self.strategy == BranchStrategy.STOCHASTIC:
            # Learned noise scale per position
            self.noise_scale = nn.Parameter(torch.ones(config.hidden_size) * 0.1)
            self.noise_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate K candidate latent states.

        Args:
            z: [B, L, D] current latent state

        Returns:
            candidates: [B, K, L, D] candidate latent states
        """
        B, L, D = z.shape
        K = self.config.num_candidates

        if self.strategy == BranchStrategy.LEARNED_PERTURBATION:
            # Apply each learned perturbation
            candidates = []
            for proj in self.branch_projs:
                delta = proj(z)  # [B, L, D]
                candidate = z + 0.1 * delta  # Small perturbation
                candidates.append(candidate)
            candidates = torch.stack(candidates, dim=1)  # [B, K, L, D]

        elif self.strategy == BranchStrategy.STOCHASTIC:
            # Sample K different noise vectors
            noise = torch.randn(B, K, L, D, device=z.device, dtype=z.dtype)
            noise = noise * self.noise_scale.view(1, 1, 1, -1)
            projected_noise = self.noise_proj(noise)
            candidates = z.unsqueeze(1) + projected_noise  # [B, K, L, D]

        return candidates


class ValueHead(nn.Module):
    """Evaluates the promise of a latent state."""

    def __init__(self, config: LBVSConfig):
        super().__init__()
        self.config = config

        # Pool latent state and predict value
        self.value_net = nn.Sequential(
            CastedLinear(config.hidden_size, config.hidden_size // 2, bias=True),
            nn.GELU(),
            CastedLinear(config.hidden_size // 2, 1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute value for latent states.

        Args:
            z: [B, K, L, D] or [B, L, D] latent states

        Returns:
            values: [B, K] or [B] value scores
        """
        # Global average pooling over sequence
        if z.dim() == 4:
            pooled = z.mean(dim=2)  # [B, K, D]
        else:
            pooled = z.mean(dim=1)  # [B, D]

        values = self.value_net(pooled).squeeze(-1)  # [B, K] or [B]
        return values


class LBVSInner(nn.Module):
    """Core LBVS model."""

    def __init__(self, config: LBVSConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.puzzle_emb_len = config.puzzle_emb_len

        # Embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Processing blocks
        self.blocks = nn.ModuleList([
            LBVSBlock(config) for _ in range(config.num_layers)
        ])

        # Beam search components
        self.branch_generator = BranchGenerator(config)
        self.value_head = ValueHead(config)

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Q-head for halt decision (compatible with ACT loss)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Initial state
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Initialize Q head conservatively
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat([
                puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
                embedding
            ], dim=1)

        return self.embed_scale * embedding

    def forward(
        self,
        z: torch.Tensor,  # [B, num_beams, L, D]
        beam_values: torch.Tensor,  # [B, num_beams]
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single iteration with beam search.

        Returns:
            new_z: [B, num_beams, L, D] updated latent states
            new_beam_values: [B, num_beams] updated value scores
            logits: [B, seq_len, vocab_size] predictions from best beam
            q_halt_logits: [B] halt logits
            q_continue_logits: [B] continue logits
        """
        B = z.shape[0]
        num_beams = z.shape[1]
        L = z.shape[2]
        D = z.shape[3]
        K = self.config.num_candidates

        # Get input embeddings
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Process each beam
        # Reshape for batch processing: [B * num_beams, L, D]
        z_flat = z.view(B * num_beams, L, D)

        # Add input embeddings
        input_emb_expanded = input_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)
        input_emb_flat = input_emb_expanded.reshape(B * num_beams, L, D)
        z_flat = z_flat + input_emb_flat

        # Process through blocks
        for block in self.blocks:
            z_flat = block(z_flat)

        # Reshape back: [B, num_beams, L, D]
        z_processed = z_flat.view(B, num_beams, L, D)

        # Generate candidates from each beam
        # For simplicity, only branch from the best beam
        best_beam_idx = beam_values.argmax(dim=1)  # [B]
        best_beam = z_processed[torch.arange(B, device=z.device), best_beam_idx]  # [B, L, D]

        # Generate K candidates
        candidates = self.branch_generator(best_beam)  # [B, K, L, D]

        # Score all candidates with value head
        candidate_values = self.value_head(candidates)  # [B, K]

        # Select top-B beams
        top_values, top_indices = candidate_values.topk(num_beams, dim=1)  # [B, num_beams]

        # Gather selected beams
        top_indices_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
        new_z = torch.gather(candidates, 1, top_indices_expanded)  # [B, num_beams, L, D]
        new_beam_values = top_values

        # Get predictions from best beam (index 0 after sorting)
        best_z = new_z[:, 0]  # [B, L, D]
        logits = self.lm_head(best_z)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]

        # Q-values for halting
        q_logits = self.q_head(best_z[:, 0]).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        return new_z.detach(), new_beam_values.detach(), logits, q_halt_logits, q_continue_logits


class LBVS(nn.Module):
    """
    Latent Beam Value Search model.

    Simple beam search in latent space with value-based pruning.
    Extensible for future beam search strategies.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LBVSConfig(**config_dict)
        self.inner = LBVSInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> LBVSCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        seq_len = self.config.seq_len + self.config.puzzle_emb_len
        num_beams = self.config.num_beams

        # Initialize all beams with same state
        z_init = self.inner.z_init.expand(batch_size, seq_len, -1).clone()
        z = z_init.unsqueeze(1).expand(-1, num_beams, -1, -1).clone()

        return LBVSCarry(
            z=z.to(device),
            beam_values=torch.zeros(batch_size, num_beams, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: LBVSCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[LBVSCarry, Dict[str, torch.Tensor]]:

        device = batch["inputs"].device
        B = batch["inputs"].shape[0]
        num_beams = self.config.num_beams
        seq_len = self.config.seq_len + self.config.puzzle_emb_len

        # Reset state for halted sequences
        z_init = self.inner.z_init.expand(B, seq_len, -1)
        z_init_beams = z_init.unsqueeze(1).expand(-1, num_beams, -1, -1)

        z = torch.where(
            carry.halted.view(-1, 1, 1, 1),
            z_init_beams.to(device),
            carry.z.to(device)
        )
        beam_values = torch.where(
            carry.halted.view(-1, 1),
            torch.zeros_like(carry.beam_values),
            carry.beam_values.to(device)
        )
        steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v.to(device)
            )
            for k, v in carry.current_data.items()
        }

        # Run beam search iteration
        new_z, new_beam_values, logits, q_halt, q_continue = self.inner(
            z, beam_values, current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue,
            "beam_values": new_beam_values,
        }

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)

            new_steps = steps + 1
            is_last_step = new_steps >= self.config.max_iterations

            halted = is_last_step

            if self.training and self.config.max_iterations > 1:
                halted = halted | (q_halt > 0)

                # Exploration
                min_halt = (
                    (torch.rand_like(q_halt) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.max_iterations + 1)
                )
                halted = halted & (new_steps >= min_halt)

        new_carry = LBVSCarry(
            z=new_z,
            beam_values=new_beam_values,
            steps=new_steps,
            halted=halted,
            current_data=current_data
        )

        return new_carry, outputs
