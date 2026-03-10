"""Optimizer and scheduler factory functions.

This module provides factory functions for creating optimizers and learning rate
schedulers for training EpiForecaster models.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    optimizer_eps: float,
    device: torch.device,
    status_callback: Any = None,
) -> torch.optim.Optimizer:
    """Create an optimizer with weight decay filtering.

    Applies weight decay only to non-bias, non-normalization parameters
    following best practices for transformer training.

    Args:
        model: The model to optimize
        optimizer_name: Name of optimizer ("adam" or "adamw")
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_eps: Epsilon for numerical stability
        device: Device for checking CUDA availability
        status_callback: Optional callback for status messages

    Returns:
        Configured optimizer instance
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        normalized_name = name.lower()
        if (
            name.endswith("bias")
            or "norm" in normalized_name
            or "alpha_" in normalized_name
            or param.ndim < 2
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_lower = optimizer_name.lower()
    optimizer_cls: type[torch.optim.Optimizer]
    if optimizer_lower == "adam":
        optimizer_cls = torch.optim.Adam
    elif optimizer_lower == "adamw":
        optimizer_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_name}")

    optimizer_kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "eps": optimizer_eps,
    }

    # CUDA fast path: fused AdamW reduces optimizer kernel launch overhead.
    if optimizer_lower == "adamw" and device.type == "cuda":
        try:
            return optimizer_cls(
                param_groups,
                fused=True,
                capturable=True,
                **optimizer_kwargs,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            msg = f"Fused AdamW unavailable, falling back to standard AdamW ({exc})"
            if status_callback is not None:
                status_callback(msg, logging.WARNING)
            else:
                logger.warning(msg)

    return optimizer_cls(param_groups, **optimizer_kwargs)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    eta_min: float = 0.0,
    epochs: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "step", or "none")
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps (only for cosine scheduler)
        epochs: Number of epochs (required for step scheduler)

    Returns:
        Configured scheduler instance, or None for scheduler_type="none"
    """
    if scheduler_type == "cosine":
        if warmup_steps > 0:
            from training.schedulers import WarmupCosineScheduler

            return WarmupCosineScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                eta_min=eta_min,
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=eta_min
            )
    elif scheduler_type == "step":
        if warmup_steps > 0:
            logger.warning(
                "warmup_steps is set but StepLR scheduler does not support warmup. "
                "Warmup will be ignored. Use scheduler_type='cosine' for warmup support."
            )
        if epochs is None:
            raise ValueError("epochs is required for step scheduler")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1
        )
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
