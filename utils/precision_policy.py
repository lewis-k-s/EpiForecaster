"""Centralized precision policy for EpiForecaster.

This module provides a single source of truth for precision handling,
supporting only:
- float32 parameters (always)
- Optional bfloat16 autocast on CUDA (when enabled and available)

Float16 weights and GradScaler are intentionally not supported to reduce
complexity and precision-related bugs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from models.configs import TrainingParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Immutable precision configuration for training.

    Attributes:
        param_dtype: Parameter dtype (always float32)
        autocast_enabled: Whether BF16 autocast is active
        autocast_dtype: Autocast dtype (bfloat16 when enabled)
        optimizer_eps: Adam epsilon value
        device_type: Device type for validation
    """

    param_dtype: torch.dtype
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    optimizer_eps: float
    device_type: str

    def division_eps(self, dtype: torch.dtype | None = None) -> float:
        """Get safe epsilon for division operations.

        Since we only support float32 and bfloat16 autocast,
        we use consistent epsilon values:
        - float32: 1e-8 (standard)
        - bfloat16: 1e-4 (safe for reduced precision)

        Args:
            dtype: Optional dtype to check. If None, uses autocast_dtype
                   when autocast is enabled, otherwise param_dtype.

        Returns:
            Safe epsilon value
        """
        check_dtype = dtype or (
            self.autocast_dtype if self.autocast_enabled else self.param_dtype
        )

        if check_dtype == torch.bfloat16:
            return 1e-4
        elif check_dtype == torch.float16:
            # Should never happen with current policy, but handle gracefully
            return 1e-4
        else:
            # float32 and others
            return 1e-8


def resolve_precision_policy(
    training_config: TrainingParams,
    device: torch.device,
) -> PrecisionPolicy:
    """Resolve precision policy from configuration and device.

    Enforces strict rules:
    1. Parameters are always float32
    2. BF16 autocast only on CUDA when enabled and supported
    3. FP16 is explicitly rejected

    Args:
        training_config: Training configuration
        device: Target device

    Returns:
        Resolved precision policy

    Raises:
        ValueError: If unsupported precision configuration is detected
    """
    # Always use float32 for parameters
    param_dtype = torch.float32

    # Determine autocast settings
    autocast_enabled = False
    autocast_dtype = None

    if training_config.enable_mixed_precision:
        # Check for FP16 (explicitly rejected)
        mp_dtype = training_config.mixed_precision_dtype.lower()
        if mp_dtype == "float16":
            raise ValueError(
                "FP16 is no longer supported. "
                "Use 'bfloat16' for autocast or disable mixed precision. "
                "Set training.mixed_precision_dtype='bfloat16' or "
                "training.enable_mixed_precision=false"
            )

        # BF16 is the only supported autocast dtype
        if mp_dtype != "bfloat16":
            raise ValueError(
                f"Unsupported mixed_precision_dtype: '{mp_dtype}'. "
                "Only 'bfloat16' is supported."
            )

        # BF16 autocast only on CUDA
        if device.type != "cuda":
            raise ValueError(
                f"BF16 autocast requested but device '{device.type}' doesn't support it. "
                "BF16 autocast is only available on CUDA devices with compute capability >= 8.0. "
                "Either switch to a CUDA device or disable mixed precision."
            )

        # Check CUDA BF16 support
        if not torch.cuda.is_available():
            raise ValueError(
                "BF16 autocast requested but CUDA is not available. "
                "Either install CUDA or disable mixed precision."
            )

        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "BF16 autocast requested but this GPU doesn't support BF16. "
                "Requires compute capability >= 8.0 (Ampere or newer). "
                "Either upgrade GPU or disable mixed precision."
            )

        autocast_enabled = True
        autocast_dtype = torch.bfloat16
        logger.info(f"BF16 autocast enabled on {torch.cuda.get_device_name(device)}")

    # Resolve optimizer epsilon
    if training_config.optimizer_eps is not None:
        optimizer_eps = training_config.optimizer_eps
    else:
        # Default epsilon for FP32 parameters
        optimizer_eps = 1e-8

    return PrecisionPolicy(
        param_dtype=param_dtype,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        optimizer_eps=optimizer_eps,
        device_type=device.type,
    )


def validate_old_checkpoint_compatible(
    checkpoint: dict,
    current_policy: PrecisionPolicy,
) -> None:
    """Validate that a checkpoint is compatible with current precision policy.

    Args:
        checkpoint: Loaded checkpoint dictionary
        current_policy: Current precision policy

    Raises:
        ValueError: If checkpoint uses incompatible precision settings
    """
    # Check for scaler state (indicates old FP16 training)
    if "scaler_state_dict" in checkpoint:
        raise ValueError(
            "Checkpoint contains GradScaler state, indicating FP16 training. "
            "FP16 is no longer supported. Please retrain with current precision settings "
            "(FP32 parameters + optional BF16 autocast)."
        )

    # Check precision signature if present
    if "precision_signature" in checkpoint:
        sig = checkpoint["precision_signature"]

        # Only accept FP32 parameters
        if sig.get("param_dtype") != "float32":
            raise ValueError(
                f"Checkpoint uses unsupported parameter dtype: {sig.get('param_dtype')}. "
                "Only float32 parameters are supported. Please retrain."
            )

        # Check autocast compatibility
        ckpt_autocast = sig.get("autocast_dtype")
        if ckpt_autocast is not None and ckpt_autocast not in (None, "bfloat16"):
            raise ValueError(
                f"Checkpoint uses unsupported autocast dtype: {ckpt_autocast}. "
                "Only bfloat16 autocast is supported. Please retrain."
            )


def create_precision_signature(policy: PrecisionPolicy) -> dict:
    """Create serializable signature for checkpoint compatibility checking.

    Args:
        policy: Current precision policy

    Returns:
        Dictionary with precision settings
    """
    return {
        "param_dtype": str(policy.param_dtype).split(".")[-1],
        "autocast_enabled": policy.autocast_enabled,
        "autocast_dtype": (
            str(policy.autocast_dtype).split(".")[-1] if policy.autocast_dtype else None
        ),
        "optimizer_eps": policy.optimizer_eps,
    }
