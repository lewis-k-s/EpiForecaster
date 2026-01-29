"""Utilities for configuring Tensor Core optimizations (TF32, mixed precision)."""

import logging
from typing import Literal

import torch

PrecisionMode = Literal["tf32", "ieee"]


def setup_tensor_core_optimizations(
    device: torch.device,
    enable_tf32: bool = True,
    enable_mixed_precision: bool = True,
    mixed_precision_dtype: str = "bfloat16",
    logger: logging.Logger | None = None,
) -> None:
    """Configure TF32 and mixed precision settings for CUDA devices.

    Args:
        device: The torch device being used for training.
        enable_tf32: Whether to enable TF32 precision on Ampere+ GPUs.
        enable_mixed_precision: Whether mixed precision (AMP) is enabled.
        mixed_precision_dtype: The dtype for mixed precision ('bfloat16' or 'float16').
        logger: Optional logger instance for status messages.
    """
    if device.type != "cuda":
        _log(logger, "Tensor Core optimizations skipped (non-CUDA device)")
        return

    # Use new PyTorch 2.9+ API for TF32 control
    # See: https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    precision_mode: PrecisionMode = "tf32" if enable_tf32 else "ieee"

    torch.backends.cuda.matmul.fp32_precision = precision_mode  # type: ignore[attr-defined]
    torch.backends.cudnn.conv.fp32_precision = precision_mode  # type: ignore[attr-defined]

    if enable_tf32:
        _log(logger, "TF32 optimizations enabled")
    else:
        _log(logger, "TF32 optimizations disabled")

    if enable_mixed_precision:
        _log(logger, f"Mixed precision ({mixed_precision_dtype}): will be used in forward pass")
    else:
        _log(logger, "Mixed precision disabled")


def _log(logger: logging.Logger | None, message: str) -> None:
    """Helper to log to either the provided logger or print."""
    if logger:
        logger.info(message)
    else:
        print(message)
