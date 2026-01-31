from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from torch.utils.checkpoint import checkpoint as ckpt

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class TRM_MoR_InnerCarry:
    z_L: torch.Tensor
    h: torch.Tensor
    mem: torch.Tensor


@dataclass
class TRM_MoRCarry:
    inner_carry: TRM_MoR_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRM_MoRConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Worker config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    L_layers: int
    mlp_t: bool = True
    puzzle_emb_len: int = 12

    # Planner / memory
    planner_size: int
    mem_slots: int
    mem_dim: int
    use_planner: bool = True
    use_memory: bool = True
    use_film: bool = True
    film_per_layer: bool = False

    # Halting
    halt_max_steps: int
    min_steps: int = 2

    # Training helpers
    bptt_steps: int = 0  # if >0, detach state every N steps
    checkpoint_worker: bool = False

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"


class MoRWorkerBlock(nn.Module):
    def __init__(self, config: TRM_MoRConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        film_scale: Optional[torch.Tensor],
        film_shift: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.config.mlp_t:
            h = hidden_states.transpose(1, 2)
            out = self.mlp_t(h)
            hidden_states = rms_norm(h + out, variance_epsilon=self.norm_eps).transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        if film_scale is not None and film_shift is not None:
            hidden_states = hidden_states * (1 + film_scale) + film_shift

        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRM_MoR_Inner(nn.Module):
    def __init__(self, config: TRM_MoRConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )

        self.worker_blocks = nn.ModuleList([MoRWorkerBlock(self.config) for _ in range(self.config.L_layers)])

        # Planner
        planner_in = self.config.hidden_size + self.config.planner_size
        self.planner = nn.Sequential(
            CastedLinear(planner_in, self.config.planner_size, bias=True),
            nn.GELU(),
            CastedLinear(self.config.planner_size, self.config.planner_size, bias=True),
        )
        self.h_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.planner_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Planner outputs
        film_out = self.config.hidden_size * 2
        if self.config.film_per_layer:
            film_out *= self.config.L_layers
        self.film_proj = CastedLinear(self.config.planner_size, film_out, bias=True)
        self.h_delta_proj = CastedLinear(self.config.planner_size, self.config.planner_size, bias=True)
        self.halt_proj = CastedLinear(self.config.planner_size, 1, bias=True)

        # Memory
        self.mem_query_proj = CastedLinear(self.config.planner_size, self.config.mem_dim, bias=True)
        self.mem_write_key = CastedLinear(self.config.planner_size, self.config.mem_dim, bias=True)
        self.mem_write_val = CastedLinear(self.config.planner_size, self.config.mem_dim, bias=True)
        self.mem_write_gate = CastedLinear(self.config.planner_size, 1, bias=True)
        self.mem_out_proj = (
            CastedLinear(self.config.mem_dim, self.config.hidden_size, bias=True)
            if self.config.mem_dim != self.config.hidden_size
            else None
        )

        # Fallback controllers when planner disabled
        self.pool_to_mem = CastedLinear(self.config.hidden_size, self.config.mem_dim, bias=True)
        self.pool_to_halt = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.pool_to_film = CastedLinear(self.config.hidden_size, film_out, bias=True)

        # Initial states
        self.z_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype),
                std=1,
            ),
            persistent=True,
        )
        self.mem_init = nn.Buffer(
            torch.zeros(self.config.mem_slots, self.config.mem_dim, dtype=self.forward_dtype),
            persistent=True,
        )

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        return self.embed_scale * embedding

    def _pool(self, z_L: torch.Tensor) -> torch.Tensor:
        return z_L.mean(dim=1)

    def _memory_read(self, mem: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # mem: B x K x Dm, q: B x Dm
        attn = torch.softmax((mem * q.unsqueeze(1)).sum(-1) / math.sqrt(mem.shape[-1]), dim=-1)
        return torch.sum(mem * attn.unsqueeze(-1), dim=1)

    def _memory_write(self, mem: torch.Tensor, key: torch.Tensor, val: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # key/val: B x Dm, gate: B x 1
        attn = torch.softmax((mem * key.unsqueeze(1)).sum(-1) / math.sqrt(mem.shape[-1]), dim=-1)
        attn = attn.unsqueeze(-1)
        gate = torch.sigmoid(gate).unsqueeze(-1)
        return mem * (1 - gate * attn) + (gate * attn) * val.unsqueeze(1)

    def forward(
        self,
        carry: TRM_MoR_InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRM_MoR_InnerCarry, List[torch.Tensor], List[torch.Tensor]]:
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        z_L = carry.z_L
        h = carry.h
        mem = carry.mem

        logits_steps: List[torch.Tensor] = []
        halt_logits_steps: List[torch.Tensor] = []

        for step in range(self.config.halt_max_steps):
            pooled = self._pool(z_L)

            if self.config.use_planner:
                planner_in = torch.cat([pooled, h], dim=-1)
                h_mid = self.planner(planner_in)
                h = h + self.h_delta_proj(h_mid)
                film_params = self.film_proj(h_mid)
                halt_logits = self.halt_proj(h_mid).squeeze(-1)
                mem_q = self.mem_query_proj(h_mid)
                mem_k = self.mem_write_key(h_mid)
                mem_v = self.mem_write_val(h_mid)
                mem_g = self.mem_write_gate(h_mid)
            else:
                film_params = self.pool_to_film(pooled)
                halt_logits = self.pool_to_halt(pooled).squeeze(-1)
                mem_q = self.pool_to_mem(pooled)
                mem_k = mem_q
                mem_v = mem_q
                mem_g = self.pool_to_halt(pooled)

            if self.config.use_memory:
                mem_context = self._memory_read(mem, mem_q)
                if self.mem_out_proj is not None:
                    mem_context = self.mem_out_proj(mem_context)
                mem = self._memory_write(mem, mem_k, mem_v, mem_g)
            else:
                mem_context = torch.zeros_like(pooled)

            # Expand mem context to tokens
            mem_context = mem_context.unsqueeze(1).expand_as(z_L)
            z_L = z_L + input_embeddings + mem_context

            # FiLM
            if self.config.use_film:
                if self.config.film_per_layer:
                    film_params = film_params.view(
                        z_L.shape[0], self.config.L_layers, 2, self.config.hidden_size
                    )
                else:
                    film_params = film_params.view(z_L.shape[0], 1, 2, self.config.hidden_size)
            else:
                film_params = None

            for i, block in enumerate(self.worker_blocks):
                if film_params is None:
                    scale = shift = None
                else:
                    scale = film_params[:, min(i, film_params.shape[1] - 1), 0, :].unsqueeze(1)
                    shift = film_params[:, min(i, film_params.shape[1] - 1), 1, :].unsqueeze(1)
                if self.config.checkpoint_worker and self.training:
                    z_L = ckpt(block, z_L, cos_sin, scale, shift, use_reentrant=False)
                else:
                    z_L = block(z_L, cos_sin, scale, shift)

            logits = self.lm_head(z_L)[:, self.puzzle_emb_len:]
            logits_steps.append(logits)
            halt_logits_steps.append(halt_logits)

            if self.config.bptt_steps > 0 and (step + 1) % self.config.bptt_steps == 0:
                z_L = z_L.detach()
                h = h.detach()
                mem = mem.detach()

        return TRM_MoR_InnerCarry(z_L=z_L, h=h, mem=mem), logits_steps, halt_logits_steps


class TRM_MoR(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_MoRConfig(**config_dict)
        self.inner = TRM_MoR_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        seq_len = self.config.seq_len + self.inner.puzzle_emb_len

        return TRM_MoRCarry(
            inner_carry=TRM_MoR_InnerCarry(
                z_L=self.inner.z_init.expand(batch_size, seq_len, -1).clone(),
                h=self.inner.h_init.expand(batch_size, -1).clone(),
                mem=self.inner.mem_init.expand(batch_size, -1, -1).clone(),
            ),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: TRM_MoRCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_MoRCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        inner_carry, logits_steps, halt_logits_steps = self.inner(carry.inner_carry, new_current_data)
        T = len(logits_steps)
        halt_logits = torch.stack(halt_logits_steps, dim=0)  # T x B
        halt_probs = torch.sigmoid(halt_logits)

        # Determine halting step per sample
        halted = torch.zeros_like(new_steps, dtype=torch.bool)
        halt_step = torch.full_like(new_steps, T - 1)
        for t in range(T):
            can_halt = t + 1 >= self.config.min_steps
            step_halt = (halt_probs[t] > 0.5) if can_halt else torch.zeros_like(halt_probs[t], dtype=torch.bool)
            step_halt = step_halt | (t == T - 1)
            newly = (~halted) & step_halt
            halt_step = torch.where(newly, torch.full_like(halt_step, t), halt_step)
            halted = halted | step_halt

        new_steps = halt_step + 1

        # Gather final logits
        logits_stack = torch.stack(logits_steps, dim=0)  # T x B x L x V
        gather_idx = halt_step.view(1, -1, 1, 1).expand(1, -1, logits_stack.shape[2], logits_stack.shape[3])
        final_logits = torch.gather(logits_stack, 0, gather_idx).squeeze(0)

        outputs = {
            "logits": final_logits,
            "logits_steps": logits_steps,
            "halt_logits_steps": halt_logits_steps,
            "halt_step": halt_step,
        }

        return TRM_MoRCarry(inner_carry, new_steps, halted, new_current_data), outputs
