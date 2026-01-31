"""
Iterative Error-Correcting Transformer (IECT)

Key innovations over TRM:
1. Prediction Feedback Loop - predictions are fed back as embeddings for refinement
2. Confidence-Weighted Updates - uncertain positions get more attention
3. Residual Refinement - each iteration refines the previous prediction
4. Multi-Head Ensemble - multiple prediction heads with voting
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class IECTCarry:
    """Carry state between iterations."""
    z: torch.Tensor  # Main latent state [B, L, D]
    pred_embedding: torch.Tensor  # Embedding of current predictions [B, L, D]
    confidence: torch.Tensor  # Per-position confidence [B, L]
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class IECTConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0

    # Architecture
    hidden_size: int
    num_heads: int
    expansion: float
    num_layers: int  # Depth per iteration

    # Iteration config
    max_iterations: int  # Maximum refinement iterations
    halt_exploration_prob: float

    # Multi-head ensemble
    num_pred_heads: int = 3
    use_learned_confidence: bool = False

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 8


class IECTBlock(nn.Module):
    """Single transformer block with confidence-aware processing."""

    def __init__(self, config: IECTConfig):
        super().__init__()
        self.config = config

        # Temporal MLP (across sequence positions)
        self.mlp_t = SwiGLU(
            hidden_size=config.seq_len + config.puzzle_emb_len,
            expansion=config.expansion,
        )

        # Channel MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        # Confidence gate - modulates update based on uncertainty
        self.confidence_gate = nn.Sequential(
            CastedLinear(config.hidden_size + 1, config.hidden_size, bias=True),
            nn.Sigmoid()
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # Temporal mixing
        h = hidden_states.transpose(1, 2)  # [B, D, L]
        h = self.mlp_t(h)
        h = h.transpose(1, 2)  # [B, L, D]
        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        # Channel mixing with confidence gating
        h = self.mlp(hidden_states)

        # Pad confidence to match sequence length (includes puzzle prefix)
        if confidence.shape[1] < L:
            pad_len = L - confidence.shape[1]
            confidence = F.pad(confidence, (pad_len, 0), value=0.5)  # Neutral confidence for puzzle prefix

        # Gate update by inverse confidence (uncertain positions update more)
        gate_input = torch.cat([hidden_states, (1 - confidence).unsqueeze(-1)], dim=-1)
        gate = self.confidence_gate(gate_input)
        h = h * gate

        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        return hidden_states


class IECTInner(nn.Module):
    """Core IECT model."""

    def __init__(self, config: IECTConfig):
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

        # Prediction embedding - converts predictions back to embeddings
        self.pred_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Puzzle embeddings
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + config.puzzle_emb_len,
            base=config.rope_theta
        )

        # Processing blocks
        self.blocks = nn.ModuleList([
            IECTBlock(config) for _ in range(config.num_layers)
        ])

        # Multi-head prediction ensemble
        self.pred_heads = nn.ModuleList([
            CastedLinear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_pred_heads)
        ])

        # Confidence head
        self.confidence_head = nn.Sequential(
            CastedLinear(config.hidden_size, config.hidden_size // 2, bias=True),
            nn.GELU(),
            CastedLinear(config.hidden_size // 2, 1, bias=True),
            nn.Sigmoid()
        )

        # Halt Q-head
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Initial latent state
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Input-prediction fusion gate
        self.fusion_gate = nn.Sequential(
            CastedLinear(config.hidden_size * 3, config.hidden_size, bias=True),
            nn.Sigmoid()
        )

        # Q head init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """Create input embeddings with puzzle prefix."""
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
        z: torch.Tensor,
        pred_embedding: torch.Tensor,
        confidence: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single iteration of refinement.

        Returns:
            new_z: Updated latent state
            new_pred_embedding: Embedding of new predictions
            new_confidence: Updated confidence scores
            logits: Prediction logits
            q_halt_logits: Halting Q-value
            q_continue_logits: Continue Q-value
        """
        # Get input embeddings
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Fuse input, previous prediction, and current state
        # This lets the model see: what was asked, what it predicted, current reasoning state
        fusion_input = torch.cat([input_emb, pred_embedding, z], dim=-1)
        fusion_weight = self.fusion_gate(fusion_input)

        # Weighted combination: emphasize input for uncertain positions, prediction for confident ones
        combined = fusion_weight * input_emb + (1 - fusion_weight) * pred_embedding + z

        # Process through blocks with confidence awareness
        h = combined
        for block in self.blocks:
            h = block(h, confidence)

        # Update latent state (residual)
        new_z = z + 0.1 * (h - z)  # Slow update for stability

        # Multi-head predictions
        all_logits = torch.stack([head(h) for head in self.pred_heads], dim=0)  # [num_heads, B, L, V]

        # Ensemble: average logits (could also do voting)
        logits = all_logits.mean(dim=0)

        # Get predictions and their embeddings
        preds = torch.argmax(logits, dim=-1)

        # Create prediction embeddings (with puzzle prefix zeros)
        pred_emb_seq = self.pred_embed(preds[:, self.puzzle_emb_len:].to(torch.int32))
        new_pred_embedding = torch.cat([
            torch.zeros_like(pred_embedding[:, :self.puzzle_emb_len]),
            pred_emb_seq
        ], dim=1)

        # Confidence from prediction entropy and head agreement
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # Lower = more confident
        max_entropy = math.log(self.config.vocab_size)
        entropy_confidence = 1 - (entropy / max_entropy)

        # Head agreement confidence
        head_preds = torch.argmax(all_logits, dim=-1)  # [num_heads, B, L]
        agreement = (head_preds == head_preds[0:1]).float().mean(dim=0)  # How many heads agree

        # Learned confidence (if enabled)
        learned_confidence = self.confidence_head(h).squeeze(-1)

        # Combined confidence
        if self.config.use_learned_confidence:
            new_confidence = learned_confidence
        else:
            new_confidence = 0.5 * entropy_confidence + 0.5 * agreement

        # Q values for halting decision
        q_logits = self.q_head(h[:, 0]).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        # Return without puzzle prefix for logits
        return (
            new_z.detach(),
            new_pred_embedding.detach(),
            new_confidence[:, self.puzzle_emb_len:].detach(),
            logits[:, self.puzzle_emb_len:],
            q_halt_logits,
            q_continue_logits
        )


class IterativeErrorCorrectingTransformer(nn.Module):
    """
    Main IECT model with ACT wrapper.

    Key innovations:
    1. Predictions are fed back as embeddings for refinement
    2. Confidence-weighted updates focus on uncertain positions
    3. Multi-head ensemble for robust predictions
    4. Residual refinement preserves good predictions
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = IECTConfig(**config_dict)
        self.inner = IECTInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> IECTCarry:
        batch_size = batch["inputs"].shape[0]
        seq_len = self.config.seq_len + self.config.puzzle_emb_len

        return IECTCarry(
            z=self.inner.z_init.expand(batch_size, seq_len, -1).clone(),
            pred_embedding=torch.zeros(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.inner.forward_dtype, device=batch["inputs"].device
            ),
            confidence=torch.zeros(
                batch_size, self.config.seq_len,
                dtype=torch.float32, device=batch["inputs"].device
            ),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: IECTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[IECTCarry, Dict[str, torch.Tensor]]:

        # Reset state for halted sequences
        z = torch.where(
            carry.halted.view(-1, 1, 1),
            self.inner.z_init.expand_as(carry.z),
            carry.z
        )
        pred_embedding = torch.where(
            carry.halted.view(-1, 1, 1),
            torch.zeros_like(carry.pred_embedding),
            carry.pred_embedding
        )
        confidence = torch.where(
            carry.halted.view(-1, 1),
            torch.zeros_like(carry.confidence),
            carry.confidence
        )
        steps = torch.where(carry.halted, 0, carry.steps)
        current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Run refinement iteration
        new_z, new_pred_embedding, new_confidence, logits, q_halt, q_continue = self.inner(
            z, pred_embedding, confidence, current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue,
            "confidence": new_confidence
        }

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)

            new_steps = steps + 1
            is_last_step = new_steps >= self.config.max_iterations

            halted = is_last_step

            if self.training and self.config.max_iterations > 1:
                # Halt based on Q-value
                halted = halted | (q_halt > 0)

                # Exploration
                min_halt = (
                    (torch.rand_like(q_halt) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.max_iterations + 1)
                )
                halted = halted & (new_steps >= min_halt)

        new_carry = IECTCarry(
            z=new_z,
            pred_embedding=new_pred_embedding,
            confidence=new_confidence,
            steps=new_steps,
            halted=halted,
            current_data=current_data
        )

        return new_carry, outputs
