"""
Training utilities for the EpiForecaster project.
"""

import torch


def drop_nowcast(prediction: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
    """
    Remove the t=0 nowcast from predictions to match forecast horizon.

    Model predictions now include a nowcast at t=0 (the initial state prediction),
    resulting in shape [B, H+1] instead of [B, H]. This utility slices off the
    first timestep to align predictions with targets during loss computation.

    Args:
        prediction: Tensor of shape [B, T] where T >= horizon (typically H+1)
        horizon: Expected forecast horizon. If None, assumes T-1.

    Returns:
        Sliced tensor of shape [B, horizon] with t=0 removed.

    Example:
        >>> pred = torch.randn(2, 8)  # [B=2, H+1=8] includes nowcast
        >>> pred_forecast = drop_nowcast(pred, horizon=7)
        >>> pred_forecast.shape
        torch.Size([2, 7])
    """
    if prediction.ndim < 2:
        return prediction

    current_len = prediction.shape[1]
    if horizon is None:
        horizon = current_len - 1

    if current_len <= horizon:
        # Already correct shape or smaller, return as-is
        return prediction

    # Slice off t=0, keep horizon steps
    return prediction[:, 1 : 1 + horizon]


def get_effective_optimizer_step(batch_step: int, accumulation_steps: int) -> int:
    """
    Compute the effective optimizer step given the global batch step and gradient accumulation steps.

    Args:
        batch_step: The current global step count (increments every batch).
        accumulation_steps: Number of batches to accumulate gradients before an optimizer step.

    Returns:
        The effective optimizer step count.
    """
    if accumulation_steps <= 0:
        return batch_step
    return batch_step // accumulation_steps


def should_log_step(
    batch_step: int, accumulation_steps: int, log_frequency: int
) -> bool:
    """
    Determine whether to log at the current batch step, accounting for gradient accumulation.

    Ensures logging only occurs at accumulation boundaries to avoid duplicate step logs
    when multiple batches map to the same effective optimizer step.

    Args:
        batch_step: The current global batch step count (0-indexed, increments every batch).
        accumulation_steps: Number of batches to accumulate gradients before an optimizer step.
        log_frequency: Log every N effective steps.

    Returns:
        True if this batch should trigger logging, False otherwise.
    """
    effective_step = get_effective_optimizer_step(batch_step, accumulation_steps)
    if effective_step <= 0:
        return False
    if effective_step % log_frequency != 0:
        return False
    # Only log at the first batch of the accumulation window (boundary)
    if accumulation_steps <= 1:
        return True
    return batch_step % accumulation_steps == 0
