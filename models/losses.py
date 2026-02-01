from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Value loss (for LBVS beam search)
        value_loss = 0
        if "value_loss" in outputs:
            value_loss = outputs["value_loss"]
            metrics["value_loss"] = value_loss.detach()
        # Q-value loss (for LBVS beam search)
        q_value_loss = 0
        if "q_value_loss" in outputs:
            q_value_loss = outputs["q_value_loss"]
            metrics["q_value_loss"] = q_value_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + value_loss + q_value_loss
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class MoRLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, halt_margin: float = 1e-3, halt_loss_weight: float = 1.0, step_lambda: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.halt_margin = halt_margin
        self.halt_loss_weight = halt_loss_weight
        self.step_lambda = step_lambda

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def _per_sample_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        denom = mask.sum(-1).clamp_min(1)
        return (loss.sum(-1) / denom)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        mask = (labels != IGNORE_LABEL_ID)

        logits_steps = outputs["logits_steps"]
        halt_logits_steps = outputs["halt_logits_steps"]
        halt_step = outputs["halt_step"]
        T = len(logits_steps)

        # Per-step losses (per sample)
        step_losses = [self._per_sample_loss(lg, labels, mask) for lg in logits_steps]

        # Progress target: loss_{t+1} < loss_t - margin
        progress = []
        for t in range(T - 1):
            progress.append((step_losses[t + 1] < (step_losses[t] - self.halt_margin)).to(torch.float32))

        # Halt targets
        halt_targets = []
        for t in range(T):
            if t == T - 1:
                halt_targets.append(torch.ones_like(step_losses[t]))
            else:
                halt_targets.append(1.0 - progress[t])
        halt_targets = torch.stack(halt_targets, dim=0)

        # Do not allow halting before min steps if available
        min_steps = getattr(self.model.config, "min_steps", 1)
        if min_steps > 1:
            halt_targets[: min_steps - 1] = 0.0

        halt_logits = torch.stack(halt_logits_steps, dim=0)
        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, halt_targets, reduction="mean")

        # Final logits loss
        final_logits = outputs["logits"]
        lm_loss = (self.loss_fn(final_logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)).sum() / mask.sum().clamp_min(1)

        # Step penalty
        step_penalty = self.step_lambda * new_carry.steps.to(torch.float32).mean()

        loss = lm_loss + self.halt_loss_weight * halt_loss + step_penalty

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(final_logits, dim=-1)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            metrics = {
                "count": (loss_counts > 0).sum(),
                "accuracy": torch.where(loss_counts > 0, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": seq_is_correct.sum(),
                "avg_steps": new_carry.steps.to(torch.float32).mean(),
                "halt_loss": halt_loss.detach(),
                "lm_loss": lm_loss.detach(),
            }

            # Halt quality: did it halt when progress ceased?
            halt_target_at_step = halt_targets.gather(0, halt_step.view(1, -1)).squeeze(0)
            metrics["halt_correct"] = halt_target_at_step.mean()

            # Compute/accuracy frontier at step 4 and max
            step4 = min(3, T - 1)
            preds4 = torch.argmax(logits_steps[step4], dim=-1)
            is_correct4 = mask & (preds4 == labels)
            seq_is_correct4 = is_correct4.sum(-1) == loss_counts
            metrics["exact_accuracy_step4"] = seq_is_correct4.sum()
            metrics["exact_accuracy_stepmax"] = seq_is_correct.sum()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, loss, metrics, detached_outputs, new_carry.halted.all()
