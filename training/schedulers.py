"""
Learning rate schedulers for training stability.

This module provides schedulers that combine warmup with decay strategies,
specifically designed for transformer-style forecasting models where early
training stability is critical.
"""

import math

import torch
from torch.optim import Optimizer


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by cosine decay.

    This scheduler implements the most robust default for transformer-style
    forecasting models:

    - **Warmup phase** (t <= warmup_steps):
        LR(t) = base_lr * (t / warmup_steps)

      During warmup, the learning rate increases linearly from 0 to base_lr.
      This stabilizes early updates when:
        - LayerNorm statistics are uncalibrated
        - Adam's second-moment estimates are near zero
        - Attention weights are effectively random

    - **Decay phase** (t > warmup_steps):
        LR(t) = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * progress))

      Where progress = (t - warmup_steps) / (total_steps - warmup_steps)

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps (0 = no warmup).
        total_steps: Total number of training steps (for decay).
        eta_min: Minimum learning rate at end of training. Default: 0.
        last_epoch: The index of last epoch. Default: -1.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer,
        ...     warmup_steps=500,
        ...     total_steps=16000,
        ... )
        >>> for step in range(total_steps):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()

    Note:
        Warmup duration recommendations (as percentage of total steps):
        - Fine-tuning pretrained backbone: 1-3%
        - Training from scratch (10-100M params): 3-5%
        - Very deep stacks or small batch sizes: up to 8-10%

        Diagnostic signals warmup is too short:
        - Training loss spikes in first few hundred steps
        - Gradient norm clipping triggers heavily early on
        - Validation loss degrades immediately after initial improvement

        Diagnostic signals warmup is too long:
        - Slow initial loss drop
        - Model underfits early horizons while later ones lag
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current step."""
        if self.warmup_steps <= 0:
            return self._cosine_decay_lr()

        if self._step_count <= self.warmup_steps:
            return self._warmup_lr()
        else:
            return self._cosine_decay_lr()

    def _warmup_lr(self) -> list[float]:
        """Linear warmup from 0 to base_lr."""
        warmup_progress = self._step_count / self.warmup_steps
        return [base_lr * warmup_progress for base_lr in self.base_lrs]

    def _cosine_decay_lr(self) -> list[float]:
        """Cosine decay from base_lr to eta_min."""
        if self.total_steps <= self.warmup_steps:
            return [self.eta_min for _ in self.base_lrs]

        decay_steps = self.total_steps - self.warmup_steps
        current_decay_step = self._step_count - self.warmup_steps
        progress = min(current_decay_step / decay_steps, 1.0)

        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def compute_scheduler_steps(
    epochs: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int = 1,
    warmup_batches: int = 0,
) -> tuple[int, int]:
    """Compute scheduler steps accounting for gradient accumulation.

    The scheduler steps once per optimizer update, not per batch. When using
    gradient accumulation, the optimizer steps every N batches, so both
    total_steps and warmup_steps must be divided accordingly.

    Args:
        epochs: Total number of training epochs.
        batches_per_epoch: Number of batches in one epoch (len(train_loader)).
        gradient_accumulation_steps: Number of batches to accumulate gradients
            before optimizer step. Default: 1 (no accumulation).
        warmup_batches: Number of warmup batches from config. Default: 0.

    Returns:
        Tuple of (total_scheduler_steps, warmup_scheduler_steps).

    Example:
        >>> # 10 epochs, 100 batches/epoch, accumulate every 4 batches
        >>> total, warmup = compute_scheduler_steps(
        ...     10, 100, gradient_accumulation_steps=4, warmup_batches=200
        ... )
        >>> total
        250  # = (10 * 100) // 4
        >>> warmup
        50   # = 200 // 4
    """
    total_batches = epochs * batches_per_epoch
    total_steps = total_batches // gradient_accumulation_steps
    warmup_steps = warmup_batches // gradient_accumulation_steps
    return total_steps, warmup_steps
