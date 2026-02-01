"""
LBVS - Latent Beam Value Search

A beam search model that operates in latent space:
1. Generate K candidate latent states via learned branch projections
2. Score candidates by blended q-head/value-head (q shared for halting)
3. Keep top-B beams based on scores
4. Decode best beam at the end

Key design:
- Branches from ALL beams for diversity
- Q-head used for halting; beam selection blends q and value
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


class BranchStrategy(str, Enum):
    """Strategies for generating beam candidates."""
    LEARNED_PERTURBATION = "learned_perturbation"  # K learned delta projections
    STOCHASTIC = "stochastic"  # Add learned noise, sample K times
    FILM = "film"  # Per-branch scale/shift modulation
    BRANCH_TOKENS = "branch_tokens"  # Per-branch token injection
    # Future: ATTENTION_MODES, MIXTURE_OF_EXPERTS, etc.


@dataclass
class LBVSCarry:
    """Carry state for beam search."""
    z: torch.Tensor  # Current latent states [B, num_beams, L, D]
    pred_embedding: torch.Tensor  # Prediction embeddings [B, num_beams, L, D]
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
    num_candidates: int = 4  # K: number of candidate beams to generate per beam
    num_beams: int = 2  # B: number of beams to keep
    branch_strategy: str = "learned_perturbation"

    # Iteration config
    max_iterations: int
    halt_exploration_prob: float

    # Value loss weight
    value_loss_weight: float = 0.1
    q_value_blend_alpha: float = 0.5
    beam_rank_loss_weight: float = 0.1
    ga_num_candidates: int = 0  # Extra crossover candidates per batch item
    ga_mix_alpha: float = 0.5  # Fixed mix ratio for crossover
    use_ind_token: bool = True

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 8


class LBVSBlock(nn.Module):
    """Attention-based transformer block for processing latent states."""

    def __init__(self, config: LBVSConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )

        # Channel MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(self, cos_sin, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: [B*num_beams, L, D] or [B, L, D]

        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attn_bias=attn_bias),
            variance_epsilon=self.norm_eps,
        )

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
        elif self.strategy == BranchStrategy.FILM:
            # Per-branch scale/shift modulation
            self.film_scale = nn.Parameter(torch.zeros(config.num_candidates, config.hidden_size))
            self.film_shift = nn.Parameter(torch.zeros(config.num_candidates, config.hidden_size))
        elif self.strategy == BranchStrategy.BRANCH_TOKENS:
            # Learned tokens that represent distinct hypotheses
            self.branch_tokens = nn.Parameter(torch.zeros(config.num_candidates, config.hidden_size))
            nn.init.normal_(self.branch_tokens, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate K candidate latent states from each input beam.

        Args:
            z: [B, num_beams, L, D] current latent states

        Returns:
            candidates: [B, num_beams * K, L, D] candidate latent states
        """
        B, num_beams, L, D = z.shape
        K = self.config.num_candidates

        # Flatten beams for processing: [B * num_beams, L, D]
        z_flat = z.view(B * num_beams, L, D)

        if self.strategy == BranchStrategy.LEARNED_PERTURBATION:
            # Apply each learned perturbation to all beams
            candidates = []
            for proj in self.branch_projs:
                delta = proj(z_flat)  # [B * num_beams, L, D]
                candidate = z_flat + 0.1 * delta  # Small perturbation
                candidates.append(candidate)
            # Stack: [K, B * num_beams, L, D] -> [B * num_beams, K, L, D]
            candidates = torch.stack(candidates, dim=1)
            # Reshape to [B, num_beams * K, L, D]
            candidates = candidates.view(B, num_beams * K, L, D)

        elif self.strategy == BranchStrategy.STOCHASTIC:
            # Sample K different noise vectors for each beam
            noise = torch.randn(B * num_beams, K, L, D, device=z.device, dtype=z.dtype)
            noise = noise * self.noise_scale.view(1, 1, 1, -1)
            projected_noise = self.noise_proj(noise)
            candidates = z_flat.unsqueeze(1) + projected_noise  # [B * num_beams, K, L, D]
            candidates = candidates.view(B, num_beams * K, L, D)
        elif self.strategy == BranchStrategy.FILM:
            scales = 1.0 + self.film_scale.view(1, K, 1, 1, D)
            shifts = self.film_shift.view(1, K, 1, 1, D)
            z_expanded = z_flat.unsqueeze(1).unsqueeze(2)  # [B*num_beams, 1, 1, L, D]
            modulated = z_expanded * scales + shifts  # [B*num_beams, K, 1, L, D]
            candidates = modulated.squeeze(2).view(B, num_beams * K, L, D)
        elif self.strategy == BranchStrategy.BRANCH_TOKENS:
            tokens = self.branch_tokens.view(1, K, 1, D)
            z_expanded = z_flat.unsqueeze(1)  # [B*num_beams, K, L, D] after broadcast
            candidates = z_expanded.clone()
            candidates[:, :, 0, :] = candidates[:, :, 0, :] + tokens  # inject into first position
            candidates = candidates.view(B, num_beams * K, L, D)

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
        # Use induction token if enabled, otherwise global average pooling
        if z.dim() == 4:
            if self.config.use_ind_token:
                pooled = z[:, :, 0, :]  # [B, K, D]
            else:
                pooled = z.mean(dim=2)  # [B, K, D]
        else:
            if self.config.use_ind_token:
                pooled = z[:, 0, :]  # [B, D]
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

        # Prediction embedding for feedback loop
        self.pred_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Processing blocks (pre/post around branching)
        pre_layers = max(1, config.num_layers // 2)
        post_layers = max(1, config.num_layers - pre_layers)
        self.pre_blocks = nn.ModuleList([
            LBVSBlock(config) for _ in range(pre_layers)
        ])
        self.post_blocks = nn.ModuleList([
            LBVSBlock(config) for _ in range(post_layers)
        ])

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + self.puzzle_emb_len,
            base=config.rope_theta,
        )
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(config.num_heads, config.seq_len + self.puzzle_emb_len, config.seq_len + self.puzzle_emb_len)
        )

        # Beam search components
        self.branch_generator = BranchGenerator(config)
        self.value_head = ValueHead(config)

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Q-head for halt decision (compatible with ACT loss)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Fusion gate for combining input, prediction, and state
        self.fusion_gate = CastedLinear(config.hidden_size * 2, config.hidden_size, bias=True)

        # Initial state
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.ind_token = nn.Parameter(torch.zeros(config.hidden_size))
        nn.init.normal_(self.ind_token, std=0.02)

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
        pred_embedding: torch.Tensor,  # [B, num_beams, L, D]
        beam_values: torch.Tensor,  # [B, num_beams]
        batch: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single iteration with beam search.

        Key insight: Use a blend of q-head and value-head for beam selection,
        while q-head still drives halting.

        Returns:
            new_z: [B, num_beams, L, D] updated latent states (detached for carry)
            new_pred_embedding: [B, num_beams, L, D] prediction embeddings (detached)
            new_beam_values: [B, num_beams] updated value scores (detached)
            logits: [B, seq_len, vocab_size] predictions from best beam
            q_halt_logits: [B] halt logits
            q_continue_logits: [B] continue logits
            value_loss: scalar, loss for training value head
            q_value_loss: scalar, loss for training q head on beam quality
            rank_loss: scalar, pairwise ranking loss for beam scores
        """
        B = z.shape[0]
        num_beams = z.shape[1]
        if beam_values.shape[1] != num_beams:
            num_beams = min(num_beams, beam_values.shape[1])
            z = z[:, :num_beams]
            pred_embedding = pred_embedding[:, :num_beams]
            beam_values = beam_values[:, :num_beams]
        L = z.shape[2]
        D = z.shape[3]
        K = self.config.num_candidates

        # Get input embeddings
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Expand input embeddings for all beams: [B, num_beams, L, D]
        input_emb_expanded = input_emb.unsqueeze(1).expand(-1, num_beams, -1, -1)

        # Fuse input, prediction feedback, and current state
        # Reshape for processing: [B * num_beams, L, D]
        z_flat = z.view(B * num_beams, L, D)
        input_flat = input_emb_expanded.reshape(B * num_beams, L, D)
        pred_flat = pred_embedding.view(B * num_beams, L, D)

        # Simple fusion: gate between (input + pred) and current state
        fusion_input = torch.cat([input_flat + pred_flat, z_flat], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        z_fused = gate * (input_flat + pred_flat) + (1 - gate) * z_flat
        if self.config.use_ind_token:
            z_fused[:, 0, :] = z_fused[:, 0, :] + self.ind_token.to(z_fused.dtype)

        # Process through pre-branch blocks
        h = z_fused
        cos_sin = self.rotary_emb()
        attn_bias = self.rel_pos_bias[:, :L, :L].unsqueeze(0).to(h.dtype)
        for block in self.pre_blocks:
            h = block(cos_sin, h, attn_bias=attn_bias)

        # Reshape back: [B, num_beams, L, D]
        z_processed = h.view(B, num_beams, L, D)

        # Generate candidates from ALL beams (maintains diversity)
        candidates = self.branch_generator(z_processed)  # [B, num_beams * K, L, D]
        total_candidates = num_beams * K

        # Optional GA-style crossover candidates
        parent_scores = (
            beam_values.unsqueeze(-1)
            .expand(-1, num_beams, K)
            .reshape(B, total_candidates)
        )
        if self.config.ga_num_candidates > 0:
            ga_k = self.config.ga_num_candidates
            idx_a = torch.randint(0, num_beams, (B, ga_k), device=z.device, dtype=torch.long)
            idx_b = torch.randint(0, num_beams, (B, ga_k), device=z.device, dtype=torch.long)
            if num_beams > 0:
                idx_a = idx_a.clamp_max(num_beams - 1)
                idx_b = idx_b.clamp_max(num_beams - 1)

            gather_a = idx_a.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
            gather_b = idx_b.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
            z_a = torch.gather(z_processed, 1, gather_a)
            z_b = torch.gather(z_processed, 1, gather_b)

            alpha = self.config.ga_mix_alpha
            ga_candidates = alpha * z_a + (1.0 - alpha) * z_b  # [B, ga_k, L, D]

            score_a = torch.gather(beam_values, 1, idx_a)
            score_b = torch.gather(beam_values, 1, idx_b)
            ga_parent_scores = alpha * score_a + (1.0 - alpha) * score_b  # [B, ga_k]

            candidates = torch.cat([candidates, ga_candidates], dim=1)
            parent_scores = torch.cat([parent_scores, ga_parent_scores], dim=1)
            total_candidates = candidates.shape[1]

        # Process candidates through post-branch blocks
        candidates_flat = candidates.view(B * total_candidates, L, D)
        for block in self.post_blocks:
            candidates_flat = block(cos_sin, candidates_flat, attn_bias=attn_bias)
        candidates = candidates_flat.view(B, total_candidates, L, D)

        # === SCORE CANDIDATES ===
        # Blend q-head and value-head for beam selection; q-head still drives halting.

        # Use q-halt logits from candidate first token as beam quality
        candidates_flat = candidates.view(B * total_candidates, L, D)
        candidate_q_logits = self.q_head(candidates_flat[:, 0]).to(torch.float32)
        candidate_q_scores = candidate_q_logits[:, 0].view(B, total_candidates)

        # Value head provides an auxiliary beam signal
        candidate_values = self.value_head(candidates).to(torch.float32)  # [B, num_beams * K]

        alpha = self.config.q_value_blend_alpha
        candidate_scores = alpha * candidate_q_scores + (1.0 - alpha) * candidate_values + parent_scores
        top_scores, top_indices = candidate_scores.topk(num_beams, dim=1)

        # Optional value/q supervision from LM loss (if labels available)
        labels = batch.get("labels", None)
        if labels is not None:
            vocab_size = self.config.vocab_size
            seq_len = self.config.seq_len  # actual sequence length without puzzle prefix

            all_logits = self.lm_head(candidates_flat)[:, self.puzzle_emb_len:]  # [B*cand, seq_len, vocab]
            all_logits = all_logits.view(B, total_candidates, seq_len, vocab_size)

            mask = (labels != IGNORE_LABEL_ID)  # [B, seq_len]
            labels_expanded = labels.unsqueeze(1).expand(-1, total_candidates, -1)  # [B, cand, seq_len]
            mask_expanded = mask.unsqueeze(1).expand(-1, total_candidates, -1)  # [B, cand, seq_len]

            all_logits_flat = all_logits.reshape(B * total_candidates * seq_len, vocab_size)
            labels_flat = labels_expanded.reshape(B * total_candidates * seq_len)

            per_token_loss = F.cross_entropy(
                all_logits_flat.to(torch.float32),
                labels_flat.to(torch.long),
                ignore_index=IGNORE_LABEL_ID,
                reduction='none'
            ).view(B, total_candidates, seq_len)

            denom = mask_expanded.sum(-1).clamp(min=1).to(per_token_loss.dtype)
            candidate_losses = (per_token_loss * mask_expanded).sum(-1) / denom  # [B, cand]
            candidate_step_scores = -candidate_losses

            with torch.no_grad():
                score_min = candidate_step_scores.min(dim=1, keepdim=True).values
                score_max = candidate_step_scores.max(dim=1, keepdim=True).values
                score_range = (score_max - score_min).clamp(min=1e-6)
                normalized_targets = (candidate_step_scores - score_min) / score_range

            value_loss = F.mse_loss(
                torch.sigmoid(candidate_values),
                normalized_targets.to(torch.float32)
            )
            q_value_loss = F.mse_loss(
                torch.sigmoid(candidate_q_scores),
                normalized_targets.to(torch.float32)
            )

            # Pairwise ranking loss: encourage blended scores to rank LM-top above LM-bottom
            top_idx = torch.argmax(candidate_step_scores, dim=1)
            bottom_idx = torch.argmin(candidate_step_scores, dim=1)
            top_scores = candidate_scores.gather(1, top_idx.unsqueeze(1))
            bottom_scores = candidate_scores.gather(1, bottom_idx.unsqueeze(1))
            rank_loss = F.softplus(-(top_scores - bottom_scores)).mean()
            rank_loss = rank_loss * self.config.beam_rank_loss_weight
        else:
            value_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)
            q_value_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)
            rank_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        # Gather selected beams
        top_indices_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
        new_z = torch.gather(candidates, 1, top_indices_expanded)  # [B, num_beams, L, D]

        # Get predictions from best beam (index 0 after sorting by score)
        best_z = new_z[:, 0]  # [B, L, D]
        logits = self.lm_head(best_z)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]

        # Create prediction embeddings for feedback
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # [B, seq_len]

        # Embed predictions for all beams (use best beam's predictions)
        pred_emb_seq = self.pred_embed(preds.to(torch.int32))  # [B, seq_len, D]
        # Add puzzle prefix zeros and expand to all beams
        new_pred_emb_single = torch.cat([
            torch.zeros(B, self.puzzle_emb_len, D, device=pred_emb_seq.device, dtype=pred_emb_seq.dtype),
            pred_emb_seq
        ], dim=1)  # [B, L, D]
        new_pred_embedding = new_pred_emb_single.unsqueeze(1).expand(-1, num_beams, -1, -1)

        # Q-values for halting (based on best beam quality)
        q_logits = self.q_head(best_z[:, 0]).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        # Detach carry states to prevent gradient explosion across iterations
        return (
            new_z.detach(),
            new_pred_embedding.detach(),
            top_scores.detach(),
            logits,
            q_halt_logits,
            q_continue_logits,
            value_loss,
            q_value_loss,
            rank_loss,
        )


class LBVS(nn.Module):
    """
    Latent Beam Value Search model.

    Beam search in latent space:
    - Beam selection blends q-head and value head
    - Q-head used for halting
    - Branches from ALL beams for diversity
    - Prediction feedback for iterative refinement
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LBVSConfig(**config_dict)
        self.inner = LBVSInner(self.config)
        # Store value loss for the loss head to access
        self._last_value_loss = None

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

        # Initialize prediction embeddings to zeros
        pred_embedding = torch.zeros(
            batch_size, num_beams, seq_len, self.config.hidden_size,
            dtype=self.inner.forward_dtype, device=device
        )

        return LBVSCarry(
            z=z.to(device),
            pred_embedding=pred_embedding,
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
        pred_embedding = torch.where(
            carry.halted.view(-1, 1, 1, 1),
            torch.zeros_like(carry.pred_embedding, device=device),
            carry.pred_embedding.to(device)
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
        new_z, new_pred_embedding, new_beam_values, logits, q_halt, q_continue, value_loss, q_value_loss, rank_loss = self.inner(
            z, pred_embedding, beam_values, current_data, training=self.training
        )

        # Store value loss for loss head
        self._last_value_loss = value_loss

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue,
            "beam_values": new_beam_values,
            "value_loss": value_loss,
            "q_value_loss": q_value_loss,
            "rank_loss": rank_loss,
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
            pred_embedding=new_pred_embedding,
            beam_values=new_beam_values,
            steps=new_steps,
            halted=halted,
            current_data=current_data
        )

        return new_carry, outputs
