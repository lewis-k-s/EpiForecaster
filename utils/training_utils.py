"""
Training utilities for the EpiForecaster project.
"""


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
