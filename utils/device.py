"""
Centralized device and computation dtype utilities for EpiForecaster.

This module isolates model execution, dtype coercion, and numerical stability
from the raw data schemas.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from data.epi_batch import EpiBatch

PrecisionMode = Literal["tf32", "ieee"]


@dataclass(frozen=True)
class DeviceStreams:
    """CUDA stream handles used to overlap transfer and compute."""

    compute: torch.cuda.Stream | None = None
    transfer: torch.cuda.Stream | None = None


# =============================================================================
# MODEL DTYPES (used for computation)
# =============================================================================

# Model computation dtypes - different for CPU vs GPU due to numerical stability
# CPU: float32 for stability (PyTorch Adam has float16 issues on CPU)
# GPU: float16 for memory efficiency and speed
MODEL_DTYPE_CPU = torch.float32
MODEL_DTYPE_GPU = torch.float16

# Dtype for autocast during mixed precision training
AUTOCAST_DTYPE_CPU = torch.float32
AUTOCAST_DTYPE_GPU = torch.float16


def get_model_dtype_for_device(device: torch.device | str) -> torch.dtype:
    """Get appropriate model dtype for the given device.

    Args:
        device: Target device (torch.device or string like 'cpu', 'cuda:0')

    Returns:
        torch.dtype: float32 for CPU, float16 for CUDA/MPS
    """
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if device_obj.type == "cpu":
        return MODEL_DTYPE_CPU
    else:
        return MODEL_DTYPE_GPU


def get_autocast_dtype_for_device(device: torch.device | str) -> torch.dtype:
    """Get appropriate autocast dtype for the given device.

    Args:
        device: Target device (torch.device or string like 'cpu', 'cuda:0')

    Returns:
        torch.dtype: float32 for CPU, float16 for CUDA/MPS
    """
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if device_obj.type == "cpu":
        return AUTOCAST_DTYPE_CPU
    else:
        return AUTOCAST_DTYPE_GPU


# Legacy constant for backward compatibility - defaults to GPU dtype
MODEL_DTYPE = MODEL_DTYPE_GPU
AUTOCAST_DTYPE = AUTOCAST_DTYPE_GPU


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================


def coerce_to_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Coerce a tensor to the target dtype.

    Only converts bool and integer types to the target dtype.
    Float tensors are converted to the target dtype to ensure consistency.

    Args:
        tensor: Input tensor with any dtype
        target_dtype: Target dtype for coercion (e.g., from model parameters)

    Returns:
        Tensor with target dtype
    """
    if tensor.dtype == target_dtype:
        return tensor

    # Convert bool and int types to target dtype
    if tensor.dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return tensor.to(target_dtype)

    # For float types, convert to target dtype to ensure consistency
    return tensor.to(target_dtype)


def coerce_batch_to_dtype(
    batch: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Coerce all tensors in a batch dict to target dtype.

    Args:
        batch: Dictionary of tensors
        target_dtype: Target dtype for coercion

    Returns:
        Dictionary with tensors converted to target_dtype
    """
    return {key: coerce_to_dtype(value, target_dtype) for key, value in batch.items()}


def get_model_dtype() -> torch.dtype:
    """Get the current model computation dtype."""
    return MODEL_DTYPE


# =============================================================================
# NUMERICAL STABILITY UTILITIES
# =============================================================================


def get_dtype_safe_eps(
    dtype: torch.dtype,
    base_eps: float = 1e-8,
    float16_eps: float = 1e-4,
) -> float:
    """Return epsilon value safe for the given dtype to prevent underflow.

    In float16, small epsilon values (like 1e-8) can underflow to 0, causing
    division by zero in normalization operations. This function returns an
    appropriate epsilon based on the dtype.

    Args:
        dtype: Target dtype (e.g., torch.float16, torch.float32)
        base_eps: Epsilon for float32 and other higher precision types
        float16_eps: Epsilon for float16 (must be >= 6e-5 to avoid underflow)

    Returns:
        float: Epsilon value safe for the given dtype

    Example:
        >>> eps = get_dtype_safe_eps(torch.float16)
        >>> normalized = x / (y + eps)  # Safe division in float16
    """
    if dtype == torch.float16:
        return float16_eps
    return base_eps


def get_model_eps(device: torch.device | str, base_eps: float = 1e-8) -> float:
    """Get appropriate epsilon for the model's current dtype.

    Returns epsilon appropriate for the device's model dtype.

    Args:
        device: Target device to determine dtype
        base_eps: Base epsilon value (default 1e-8)

    Returns:
        float: Epsilon value safe for the device's model dtype
    """
    model_dtype = get_model_dtype_for_device(device)
    return get_dtype_safe_eps(model_dtype, base_eps)


def sync_to_device(
    *tensors: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, ...]:
    """Sync tensors to a target device.

    Moves tensors to the specified device if they are not already there.
    Commonly used to move DataLoader outputs (CPU) to model device (GPU/MPS).

    Args:
        *tensors: Variable number of tensors (or None) to sync
        device: Target device to move tensors to

    Returns:
        Tuple of tensors on the target device (or None for None inputs)

    Example:
        >>> target, mask, mean = sync_to_device(target, mask, mean, device=pred.device)
    """
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif t.device != device:
            result.append(t.to(device))
        else:
            result.append(t)
    return tuple(result)


def resolve_device(device: str) -> torch.device:
    """Resolve the torch device string using the same priority as training."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def _log(logger: logging.Logger | None, message: str) -> None:
    if logger:
        logger.info(message)


def prefetch_enabled(prefetch_factor: int | None) -> bool:
    """Return True when prefetching should be active for loaders and CUDA staging."""
    return prefetch_factor is not None and prefetch_factor > 0


def setup_device_streams(
    device: torch.device,
    logger: logging.Logger | None = None,
) -> DeviceStreams:
    """Create explicit compute/transfer CUDA streams for the active device."""
    if device.type != "cuda":
        _log(logger, "Device stream setup skipped (non-CUDA device)")
        return DeviceStreams()

    streams = DeviceStreams(
        compute=torch.cuda.current_stream(device=device),
        transfer=torch.cuda.Stream(device=device),
    )
    _log(logger, "Configured CUDA compute and transfer streams")
    return streams


def prepare_batch_for_device(
    batch_data: "EpiBatch",
    *,
    dataset: object | None,
    device: torch.device,
    non_blocking: bool = True,
) -> "EpiBatch":
    """Inject mobility and move a dtype-normalized batch to the target device."""
    if dataset is not None:
        from utils.training_utils import inject_gpu_mobility

        inject_gpu_mobility(batch_data, dataset, device)

    return batch_data.to(
        device=device,
        non_blocking=non_blocking,
    )


def iter_device_ready_batches(
    loader: object,
    *,
    device: torch.device,
    prefetch_factor: int | None = None,
    streams: DeviceStreams | None = None,
) -> Iterator["EpiBatch"]:
    """Yield batches after mobility injection and device transfer.

    On CUDA with enabled prefetch, the next batch is staged on a dedicated transfer
    stream while the current batch remains available to the compute stream.
    """
    dataset = getattr(loader, "dataset", None)

    if device.type != "cuda" or not prefetch_enabled(prefetch_factor):
        for batch_data in loader:
            yield prepare_batch_for_device(
                batch_data,
                dataset=dataset,
                device=device,
            )
        return

    active_streams = streams or setup_device_streams(device)
    compute_stream = active_streams.compute or torch.cuda.current_stream(device=device)
    transfer_stream = active_streams.transfer
    if transfer_stream is None:
        for batch_data in loader:
            yield prepare_batch_for_device(
                batch_data,
                dataset=dataset,
                device=device,
            )
        return

    iterator = iter(loader)

    def _preload() -> "EpiBatch" | None:
        try:
            batch_data = next(iterator)
        except StopIteration:
            return None

        with torch.cuda.stream(transfer_stream):
            return prepare_batch_for_device(
                batch_data,
                dataset=dataset,
                device=device,
                non_blocking=True,
            )

    next_batch = _preload()
    while next_batch is not None:
        compute_stream.wait_stream(transfer_stream)
        batch_data = next_batch
        batch_data.record_stream(compute_stream)
        next_batch = _preload()
        yield batch_data


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
        _log(
            logger,
            f"Mixed precision ({mixed_precision_dtype}): will be used in forward pass",
        )
    else:
        _log(logger, "Mixed precision disabled")
