from __future__ import annotations

import torch
import torch.nn as nn


class GradNormController(nn.Module):
    """GradNorm controller for observation losses (ww/hosp/cases/deaths).

    The controller keeps learnable log-weights and optional EMA-smoothed task
    statistics for stable periodic updates.
    """

    task_names = ("ww", "hosp", "cases", "deaths")

    def __init__(
        self,
        *,
        alpha: float = 1.5,
        obs_weight_sum: float = 0.95,
        min_weight: float = 1.0e-3,
        warmup_steps: int = 50,
        update_every: int = 16,
        ema_decay: float = 0.9,
        probe: str = "obs_context",
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.obs_weight_sum = float(obs_weight_sum)
        self.min_weight = float(min_weight)
        self.warmup_steps = int(warmup_steps)
        self.update_every = int(update_every)
        self.ema_decay = float(ema_decay)
        self.probe = str(probe)
        self.eps = float(eps)

        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.obs_weight_sum <= 0:
            raise ValueError(
                f"obs_weight_sum must be positive, got {self.obs_weight_sum}"
            )
        if self.min_weight < 0:
            raise ValueError(f"min_weight must be non-negative, got {self.min_weight}")
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {self.update_every}")
        if not (0 <= self.ema_decay < 1):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.min_weight * len(self.task_names) > self.obs_weight_sum:
            raise ValueError(
                "min_weight is too large for obs_weight_sum and number of tasks"
            )

        self.log_weights = nn.Parameter(torch.zeros(len(self.task_names)))
        self.register_buffer("l0", torch.ones(len(self.task_names)))
        self.register_buffer(
            "l0_initialized", torch.zeros(len(self.task_names), dtype=torch.bool)
        )
        self.register_buffer("ema_losses", torch.zeros(len(self.task_names)))
        self.register_buffer("ema_grad_norms", torch.zeros(len(self.task_names)))
        self.register_buffer(
            "ema_initialized", torch.zeros(len(self.task_names), dtype=torch.bool)
        )

    def _normalize_active_weights(
        self, raw_weights: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            return torch.zeros_like(raw_weights)

        if self.min_weight * active_count > self.obs_weight_sum:
            raise ValueError(
                "min_weight * active_count exceeds obs_weight_sum; cannot normalize"
            )

        weights = torch.where(active_mask, raw_weights.clamp_min(self.min_weight), 0.0)
        weights_sum = weights.sum().clamp_min(self.eps)
        weights = weights * (self.obs_weight_sum / weights_sum)

        if self.min_weight > 0:
            below = active_mask & (weights < self.min_weight)
            if below.any():
                fixed_total = float(self.min_weight) * int(below.sum().item())
                remaining_budget = self.obs_weight_sum - fixed_total
                remaining_mask = active_mask & (~below)
                adjusted = torch.where(
                    below,
                    torch.full_like(weights, self.min_weight),
                    torch.zeros_like(weights),
                )

                if remaining_mask.any() and remaining_budget > 0:
                    remaining = torch.where(remaining_mask, weights, 0.0)
                    remaining_sum = remaining.sum().clamp_min(self.eps)
                    adjusted = adjusted + remaining * (remaining_budget / remaining_sum)
                weights = adjusted

        return torch.where(active_mask, weights, 0.0)

    def weights(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        raw = torch.exp(self.log_weights.float())
        if active_mask is None:
            active_mask = torch.ones_like(raw, dtype=torch.bool)
        active_mask = active_mask.to(device=raw.device, dtype=torch.bool)
        return self._normalize_active_weights(raw, active_mask)

    def current_weights(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.weights(active_mask)

    def maybe_init_l0(
        self,
        losses: torch.Tensor,
        *,
        step: int,
        active_mask: torch.Tensor,
    ) -> bool:
        if step < self.warmup_steps:
            return False

        losses_detached = losses.detach().float()
        active_mask = active_mask.to(device=losses_detached.device, dtype=torch.bool)
        needs_init = active_mask & (~self.l0_initialized)
        if not needs_init.any():
            return False

        safe_losses = losses_detached.clamp_min(self.eps)
        self.l0 = torch.where(needs_init, safe_losses.to(self.l0.dtype), self.l0)
        self.l0_initialized = torch.where(
            needs_init,
            torch.ones_like(self.l0_initialized),
            self.l0_initialized,
        )
        return True

    def _update_ema(
        self,
        *,
        losses: torch.Tensor,
        grad_norms: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        losses = losses.detach().float()
        grad_norms = grad_norms.detach().float()
        active_mask = active_mask.to(device=losses.device, dtype=torch.bool)

        first_update = active_mask & (~self.ema_initialized)
        if first_update.any():
            self.ema_losses = torch.where(first_update, losses, self.ema_losses)
            self.ema_grad_norms = torch.where(
                first_update, grad_norms, self.ema_grad_norms
            )
            self.ema_initialized = torch.where(
                first_update,
                torch.ones_like(self.ema_initialized),
                self.ema_initialized,
            )

        steady = active_mask & self.ema_initialized
        if steady.any():
            d = self.ema_decay
            self.ema_losses = torch.where(
                steady,
                d * self.ema_losses + (1.0 - d) * losses,
                self.ema_losses,
            )
            self.ema_grad_norms = torch.where(
                steady,
                d * self.ema_grad_norms + (1.0 - d) * grad_norms,
                self.ema_grad_norms,
            )

    def _compute_probe_grad_norms(
        self,
        *,
        losses: torch.Tensor,
        probe: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        base_grad_norms = torch.zeros_like(losses)
        for idx in range(losses.numel()):
            if not bool(active_mask[idx]):
                continue
            grad = torch.autograd.grad(
                losses[idx],
                probe,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            base_grad_norms[idx] = grad.float().pow(2).sum().sqrt()
        return base_grad_norms

    def compute_gradnorm_terms(
        self,
        losses: torch.Tensor,
        *,
        probe: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute GradNorm terms from task losses and a shared probe tensor."""
        device = losses.device
        losses_f32 = losses.float()
        active_mask = active_mask.to(device=device, dtype=torch.bool)

        if active_mask.sum() == 0:
            zeros = torch.zeros_like(losses_f32)
            return {
                "weights": zeros,
                "base_grad_norms": zeros,
                "grad_norms": zeros,
                "target_grad_norms": zeros,
                "gradnorm_loss": losses_f32.new_zeros(()),
                "rates": zeros,
                "ema_losses": zeros,
                "ema_grad_norms": zeros,
            }

        if not torch.all(self.l0_initialized[active_mask]).item():
            raise RuntimeError("GradNorm L0 not initialized for active tasks")

        weights = self.weights(active_mask)
        base_grad_norms = self._compute_probe_grad_norms(
            losses=losses_f32,
            probe=probe,
            active_mask=active_mask,
        )
        grad_norms = torch.where(active_mask, weights * base_grad_norms, 0.0)

        self._update_ema(
            losses=losses_f32,
            grad_norms=grad_norms,
            active_mask=active_mask,
        )

        ema_ready = self.ema_initialized.to(device=device, dtype=torch.bool)
        smooth_losses = torch.where(ema_ready, self.ema_losses.to(device), losses_f32)
        smooth_grad_norms = torch.where(
            ema_ready,
            self.ema_grad_norms.to(device),
            grad_norms.detach(),
        )

        l0 = self.l0.to(device=device, dtype=losses_f32.dtype).clamp_min(self.eps)
        rel_train = torch.zeros_like(losses_f32)
        rel_train = torch.where(active_mask, smooth_losses.detach() / l0, rel_train)

        mean_rel = rel_train[active_mask].mean().clamp_min(self.eps)
        rates = torch.zeros_like(losses_f32)
        rates = torch.where(active_mask, rel_train / mean_rel, rates)

        mean_grad = smooth_grad_norms[active_mask].mean().detach()
        target_grad_norms = torch.zeros_like(losses_f32)
        target_grad_norms = torch.where(
            active_mask,
            mean_grad * torch.pow(rates.clamp_min(self.eps), self.alpha),
            target_grad_norms,
        )

        gradnorm_loss = torch.abs(
            grad_norms[active_mask] - target_grad_norms[active_mask]
        ).sum()

        return {
            "weights": weights,
            "base_grad_norms": base_grad_norms,
            "grad_norms": grad_norms,
            "target_grad_norms": target_grad_norms,
            "gradnorm_loss": gradnorm_loss,
            "rates": rates,
            "ema_losses": smooth_losses,
            "ema_grad_norms": smooth_grad_norms,
        }
