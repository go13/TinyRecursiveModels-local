"""
FIECT - FiLM-modulated Iterative Error-Correcting Transformer with Memory

Combines the best innovations from existing models:
1. Prediction Feedback Loop (from IECT) - predictions embedded and fed back for refinement
2. FiLM Modulation (from TRM_MoR) - dynamic layer conditioning based on global state
3. Lightweight Memory (from TRM_MoR) - compact slot-based memory for iteration history
4. Multi-Head Ensemble (from IECT) - multiple prediction heads with cross-head communication
5. Dual Confidence (IECT + IECT_V2) - both entropy-based AND learned confidence
6. Progressive Focus - uncertain positions get progressively more attention
7. Residual Gating - learned gating for stable iterative refinement
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class FIECTCarry:
    """Carry state between iterations."""
    z: torch.Tensor  # Main latent state [B, L, D]
    pred_embedding: torch.Tensor  # Embedding of current predictions [B, L, D]
    confidence: torch.Tensor  # Per-position confidence [B, L]
    memory: torch.Tensor  # Slot memory [B, K, D_mem]
    planner_state: torch.Tensor  # Planner hidden state [B, D_plan]
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class FIECTConfig(BaseModel):
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
    max_iterations: int
    halt_exploration_prob: float
    min_iterations: int = 2

    # Multi-head ensemble
    num_pred_heads: int = 3

    # Memory config
    mem_slots: int = 4
    mem_dim: int = 64

    # Planner config
    planner_size: int = 64
    use_film: bool = True

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 8

    # Refinement dynamics
    residual_scale: float = 0.15  # Scale for residual updates
    confidence_temp: float = 1.0  # Temperature for confidence weighting


class FIECTBlock(nn.Module):
    """Transformer block with FiLM modulation and confidence-aware processing."""

    def __init__(self, config: FIECTConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

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

        # FiLM modulation projections (will receive external FiLM params)
        self.film_norm = nn.LayerNorm(config.hidden_size)

        # Confidence-aware gating
        self.conf_gate = nn.Sequential(
            CastedLinear(config.hidden_size + 1, config.hidden_size // 2, bias=True),
            nn.GELU(),
            CastedLinear(config.hidden_size // 2, config.hidden_size, bias=True),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        confidence: torch.Tensor,
        film_scale: Optional[torch.Tensor] = None,
        film_shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # Temporal mixing
        h = hidden_states.transpose(1, 2)  # [B, D, L]
        h = self.mlp_t(h)
        h = h.transpose(1, 2)  # [B, L, D]
        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        # Apply FiLM modulation after temporal mixing
        if film_scale is not None and film_shift is not None:
            hidden_states = self.film_norm(hidden_states)
            hidden_states = hidden_states * (1 + film_scale) + film_shift

        # Channel mixing with confidence-aware gating
        h = self.mlp(hidden_states)

        # Pad confidence to match sequence length (includes puzzle prefix)
        if confidence.shape[1] < L:
            pad_len = L - confidence.shape[1]
            confidence = F.pad(confidence, (pad_len, 0), value=0.5)

        # Gate: uncertain positions get larger updates (inverse confidence)
        gate_input = torch.cat([hidden_states, (1 - confidence).unsqueeze(-1)], dim=-1)
        gate = self.conf_gate(gate_input)
        h = h * gate

        hidden_states = rms_norm(hidden_states + h, variance_epsilon=self.norm_eps)

        return hidden_states


class CrossHeadAttention(nn.Module):
    """Lightweight attention between prediction heads to resolve disagreements."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.out_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_outputs: [num_pred_heads, B, L, D]
        Returns:
            refined: [num_pred_heads, B, L, D]
        """
        H, B, L, D = head_outputs.shape

        # Reshape for cross-head attention: treat each position independently
        # [H, B, L, D] -> [B*L, H, D]
        x = head_outputs.permute(1, 2, 0, 3).reshape(B * L, H, D)

        q = self.q_proj(x)  # [B*L, H, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Attention across heads dimension
        scale = 1.0 / math.sqrt(D)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # [B*L, H, H]
        out = torch.bmm(attn, v)  # [B*L, H, D]
        out = self.out_proj(out)

        # Residual and reshape back
        out = x + 0.1 * out
        out = out.reshape(B, L, H, D).permute(2, 0, 1, 3)  # [H, B, L, D]

        return out


class FIECTInner(nn.Module):
    """Core FIECT model."""

    def __init__(self, config: FIECTConfig):
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

        # Processing blocks
        self.blocks = nn.ModuleList([
            FIECTBlock(config) for _ in range(config.num_layers)
        ])

        # Multi-head prediction ensemble with cross-head communication
        self.pred_heads = nn.ModuleList([
            CastedLinear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_pred_heads)
        ])
        self.cross_head_attn = CrossHeadAttention(config.hidden_size, config.num_pred_heads)

        # Dual confidence heads
        self.entropy_weight = nn.Parameter(torch.tensor(0.5))  # Learnable balance
        self.confidence_head = nn.Sequential(
            CastedLinear(config.hidden_size, config.hidden_size // 2, bias=True),
            nn.GELU(),
            CastedLinear(config.hidden_size // 2, 1, bias=True),
            nn.Sigmoid()
        )

        # Planner (produces FiLM parameters and halt signal)
        self.planner = nn.Sequential(
            CastedLinear(config.hidden_size + config.planner_size + config.mem_dim, config.planner_size, bias=True),
            nn.GELU(),
            CastedLinear(config.planner_size, config.planner_size, bias=True),
        )
        self.planner_delta = CastedLinear(config.planner_size, config.planner_size, bias=True)

        # FiLM projections from planner
        self.film_proj = CastedLinear(config.planner_size, config.hidden_size * 2, bias=True)

        # Memory system
        self.mem_query = CastedLinear(config.planner_size, config.mem_dim, bias=True)
        self.mem_key = CastedLinear(config.hidden_size, config.mem_dim, bias=True)
        self.mem_value = CastedLinear(config.hidden_size, config.mem_dim, bias=True)
        self.mem_gate = CastedLinear(config.planner_size, 1, bias=True)
        self.mem_out = CastedLinear(config.mem_dim, config.hidden_size, bias=True)

        # Halt head (from planner)
        self.halt_head = CastedLinear(config.planner_size, 1, bias=True)

        # Q-head for ACT loss compatibility
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Initial states
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.planner_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.planner_size, dtype=self.forward_dtype), std=0.5),
            persistent=True
        )
        self.mem_init = nn.Buffer(
            torch.zeros(config.mem_slots, config.mem_dim, dtype=self.forward_dtype),
            persistent=True
        )

        # Fusion gate (input, prediction, state combination)
        self.fusion_gate = nn.Sequential(
            CastedLinear(config.hidden_size * 3, config.hidden_size, bias=True),
            nn.GELU(),
            CastedLinear(config.hidden_size, config.hidden_size * 2, bias=True),
        )

        # Initialize halt/Q heads conservatively
        with torch.no_grad():
            self.halt_head.weight.zero_()
            self.halt_head.bias.fill_(-3)
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

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """Global average pooling."""
        return z.mean(dim=1)

    def _memory_read(self, mem: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Soft attention read from memory."""
        # mem: [B, K, D_mem], query: [B, D_mem]
        attn = torch.softmax(
            (mem * query.unsqueeze(1)).sum(-1) / math.sqrt(mem.shape[-1]),
            dim=-1
        )  # [B, K]
        return (mem * attn.unsqueeze(-1)).sum(dim=1)  # [B, D_mem]

    def _memory_write(self, mem: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Gated memory write."""
        # key, value: [B, D_mem], gate: [B, 1]
        attn = torch.softmax(
            (mem * key.unsqueeze(1)).sum(-1) / math.sqrt(mem.shape[-1]),
            dim=-1
        ).unsqueeze(-1)  # [B, K, 1]
        gate = torch.sigmoid(gate).unsqueeze(-1)  # [B, 1, 1]
        return mem * (1 - gate * attn) + (gate * attn) * value.unsqueeze(1)

    def forward(
        self,
        z: torch.Tensor,
        pred_embedding: torch.Tensor,
        confidence: torch.Tensor,
        memory: torch.Tensor,
        planner_state: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single iteration of refinement.

        Returns:
            new_z, new_pred_embedding, new_confidence, new_memory, new_planner_state,
            logits, q_halt_logits, q_continue_logits
        """
        B = z.shape[0]

        # Get input embeddings
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Pool current state for planner
        pooled = self._pool(z)

        # Memory read
        mem_query = self.mem_query(planner_state)
        mem_context = self._memory_read(memory, mem_query)

        # Update planner
        planner_input = torch.cat([pooled, planner_state, mem_context], dim=-1)
        planner_mid = self.planner(planner_input)
        new_planner_state = planner_state + self.planner_delta(planner_mid)

        # FiLM parameters from planner
        film_params = self.film_proj(planner_mid)
        film_scale = film_params[:, :self.config.hidden_size].unsqueeze(1)  # [B, 1, D]
        film_shift = film_params[:, self.config.hidden_size:].unsqueeze(1)  # [B, 1, D]

        # Fusion: combine input, previous prediction, and current state
        # Weight shifts based on iteration (early: more input, later: more prediction)
        iter_weight = min(iteration / self.config.max_iterations, 0.7)  # Cap at 0.7

        fusion_input = torch.cat([input_emb, pred_embedding, z], dim=-1)
        fusion_out = self.fusion_gate(fusion_input)
        alpha = torch.sigmoid(fusion_out[:, :, :self.config.hidden_size])  # Input vs pred weight
        beta = torch.sigmoid(fusion_out[:, :, self.config.hidden_size:])  # State mixing weight

        # Adaptive fusion based on confidence
        conf_expanded = confidence.unsqueeze(-1)
        if conf_expanded.shape[1] < input_emb.shape[1]:
            pad_len = input_emb.shape[1] - conf_expanded.shape[1]
            conf_expanded = F.pad(conf_expanded, (0, 0, pad_len, 0), value=0.5)

        # Low confidence -> more input, high confidence -> more prediction
        adaptive_alpha = alpha * (1 - conf_expanded * iter_weight)
        combined = adaptive_alpha * input_emb + (1 - adaptive_alpha) * pred_embedding
        combined = beta * combined + (1 - beta) * z

        # Memory context broadcast to all positions
        mem_hidden = self.mem_out(mem_context).unsqueeze(1)
        combined = combined + 0.1 * mem_hidden  # Soft memory injection

        # Process through blocks with FiLM and confidence
        h = combined
        for block in self.blocks:
            h = block(h, confidence, film_scale if self.config.use_film else None,
                     film_shift if self.config.use_film else None)

        # Update latent state with residual scaling
        new_z = z + self.config.residual_scale * (h - z)

        # Multi-head predictions with cross-head communication
        head_features = h.unsqueeze(0).expand(self.config.num_pred_heads, -1, -1, -1)
        head_features = self.cross_head_attn(head_features)

        all_logits = torch.stack([
            self.pred_heads[i](head_features[i])
            for i in range(self.config.num_pred_heads)
        ], dim=0)  # [num_heads, B, L, V]

        # Ensemble: weighted average based on head confidence
        head_max_probs = F.softmax(all_logits.float(), dim=-1).max(dim=-1).values  # [H, B, L]
        head_weights = F.softmax(head_max_probs.mean(dim=-1), dim=0)  # [H, B]
        head_weights = head_weights.unsqueeze(-1).unsqueeze(-1)  # [H, B, 1, 1]
        logits = (all_logits * head_weights).sum(dim=0)

        # Get predictions and their embeddings
        preds = torch.argmax(logits, dim=-1)

        # Create prediction embeddings (with puzzle prefix zeros)
        pred_emb_seq = self.pred_embed(preds[:, self.puzzle_emb_len:].to(torch.int32))
        new_pred_embedding = torch.cat([
            torch.zeros_like(pred_embedding[:, :self.puzzle_emb_len]),
            pred_emb_seq
        ], dim=1)

        # Dual confidence: entropy-based AND learned
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)
        max_entropy = math.log(self.config.vocab_size)
        entropy_confidence = 1 - (entropy / max_entropy)

        # Head agreement
        head_preds = torch.argmax(all_logits, dim=-1)  # [H, B, L]
        agreement = (head_preds == head_preds[0:1]).float().mean(dim=0)  # [B, L]

        # Learned confidence
        learned_confidence = self.confidence_head(h).squeeze(-1)

        # Combine confidences with learnable weight
        w = torch.sigmoid(self.entropy_weight)
        entropy_agreement_conf = 0.5 * entropy_confidence + 0.5 * agreement
        new_confidence = w * entropy_agreement_conf + (1 - w) * learned_confidence

        # Memory write
        mem_key = self.mem_key(pooled)
        mem_value = self.mem_value(pooled)
        mem_gate = self.mem_gate(planner_mid)
        new_memory = self._memory_write(memory, mem_key, mem_value, mem_gate)

        # Halt logits from planner
        halt_logits = self.halt_head(planner_mid).squeeze(-1)

        # Q values for ACT loss compatibility
        q_logits = self.q_head(h[:, 0]).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        return (
            new_z.detach(),
            new_pred_embedding.detach(),
            new_confidence[:, self.puzzle_emb_len:].detach(),
            new_memory.detach(),
            new_planner_state.detach(),
            logits[:, self.puzzle_emb_len:],
            q_halt_logits,
            q_continue_logits,
        )


class FIECT(nn.Module):
    """
    FiLM-modulated Iterative Error-Correcting Transformer with Memory.

    Key innovations:
    1. Prediction feedback loop with adaptive fusion
    2. FiLM modulation from planner for dynamic layer behavior
    3. Lightweight slot memory for iteration history
    4. Multi-head ensemble with cross-head attention
    5. Dual confidence (entropy + learned) for better halt decisions
    6. Progressive refinement focusing on uncertain positions
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FIECTConfig(**config_dict)
        self.inner = FIECTInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> FIECTCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        seq_len = self.config.seq_len + self.config.puzzle_emb_len

        return FIECTCarry(
            z=self.inner.z_init.expand(batch_size, seq_len, -1).clone(),
            pred_embedding=torch.zeros(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.inner.forward_dtype, device=device
            ),
            confidence=torch.zeros(
                batch_size, self.config.seq_len,
                dtype=torch.float32, device=device
            ),
            memory=self.inner.mem_init.expand(batch_size, -1, -1).clone().to(device),
            planner_state=self.inner.planner_init.expand(batch_size, -1).clone().to(device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: FIECTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[FIECTCarry, Dict[str, torch.Tensor]]:

        device = batch["inputs"].device

        # Reset state for halted sequences
        z = torch.where(
            carry.halted.view(-1, 1, 1),
            self.inner.z_init.expand_as(carry.z).to(device),
            carry.z.to(device)
        )
        pred_embedding = torch.where(
            carry.halted.view(-1, 1, 1),
            torch.zeros_like(carry.pred_embedding, device=device),
            carry.pred_embedding.to(device)
        )
        confidence = torch.where(
            carry.halted.view(-1, 1),
            torch.zeros_like(carry.confidence, device=device),
            carry.confidence.to(device)
        )
        memory = torch.where(
            carry.halted.view(-1, 1, 1),
            self.inner.mem_init.expand_as(carry.memory).to(device),
            carry.memory.to(device)
        )
        planner_state = torch.where(
            carry.halted.view(-1, 1),
            self.inner.planner_init.expand_as(carry.planner_state).to(device),
            carry.planner_state.to(device)
        )
        steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v.to(device)
            )
            for k, v in carry.current_data.items()
        }

        # Run refinement iteration
        # Note: Use mean step count as approximation (all active sequences should be at similar steps)
        iteration = int(steps.float().mean().item()) if steps.numel() > 0 else 0
        (new_z, new_pred_embedding, new_confidence, new_memory, new_planner_state,
         logits, q_halt, q_continue) = self.inner(
            z, pred_embedding, confidence, memory, planner_state, current_data, iteration
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
                # Halt based on Q-value (after minimum iterations)
                can_halt = new_steps >= self.config.min_iterations
                halted = halted | (can_halt & (q_halt > 0))

                # Exploration
                min_halt = (
                    (torch.rand_like(q_halt) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=self.config.min_iterations, high=self.config.max_iterations + 1)
                )
                halted = halted & (new_steps >= min_halt)

        new_carry = FIECTCarry(
            z=new_z,
            pred_embedding=new_pred_embedding,
            confidence=new_confidence,
            memory=new_memory,
            planner_state=new_planner_state,
            steps=new_steps,
            halted=halted,
            current_data=current_data
        )

        return new_carry, outputs
