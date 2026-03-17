"""
Training utilities for the EpiForecaster project.
"""

import torch
from data.epi_batch import EpiBatch
from utils.precision_policy import MODEL_PARAM_DTYPE


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


def inject_gpu_mobility(
    batch_data: EpiBatch,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> None:
    """
    Constructs the dense mobility adjacency matrix directly on the GPU.

    This avoids passing huge dense matrices through the DataLoader's
    multiprocessing queue, preventing severe worker OOMs on Linux systems.

    Args:
        batch_data: The batch data containing 'mob_batch'.
        dataset: The Dataset yielding the batches (contains preloaded_mobility).
        device: The target GPU device.
    """
    mob_batch = batch_data.mob_batch
    if mob_batch is None or not hasattr(mob_batch, "global_t"):
        return

    # Handle ConcatDataset wrappers safely: resolve the active sub-dataset
    # from batch run_id provenance when available.
    base_ds = dataset
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        base_ds = dataset.datasets[0]

        batch_run_id = getattr(mob_batch, "run_id", None)
        if batch_run_id is not None:
            run_id = str(batch_run_id).strip()
            if run_id:
                if not hasattr(dataset, "_run_id_to_dataset_index"):
                    run_id_to_dataset_index = {}
                    for i, sub_dataset in enumerate(dataset.datasets):
                        sub_run_id = getattr(sub_dataset, "run_id", None)
                        if sub_run_id is None:
                            continue
                        sub_run_id = str(sub_run_id).strip()
                        if sub_run_id and sub_run_id not in run_id_to_dataset_index:
                            run_id_to_dataset_index[sub_run_id] = i
                    dataset._run_id_to_dataset_index = run_id_to_dataset_index

                dataset_index = dataset._run_id_to_dataset_index.get(run_id)
                if dataset_index is not None:
                    base_ds = dataset.datasets[dataset_index]

    if not hasattr(base_ds, "preloaded_mobility") or base_ds.preloaded_mobility is None:
        return

    # Cache GPU mobility directly on the dataset object keyed by device
    if not hasattr(base_ds, "_gpu_mobility_cache"):
        base_ds._gpu_mobility_cache = {}

    cache_key = (device, MODEL_PARAM_DTYPE)
    if cache_key not in base_ds._gpu_mobility_cache:
        node_ids = torch.where(base_ds._get_graph_node_mask())[0]
        # Slice original CPU tensor to context nodes: resulting size [TotalT, N_ctx, N_ctx]
        # Using unsqueeze is faster and creates an intermediate of correct shape directly
        sliced_mob = base_ds.preloaded_mobility[:, node_ids.unsqueeze(-1), node_ids].to(
            MODEL_PARAM_DTYPE
        )

        # Transfer to GPU
        gpu_mob = sliced_mob.to(device, non_blocking=True)

        # Enforce self-loops (diagonal >= 1.0) on the GPU
        eye = torch.eye(
            len(node_ids), dtype=MODEL_PARAM_DTYPE, device=device
        ).unsqueeze(0)
        gpu_mob = torch.maximum(gpu_mob, eye)

        base_ds._gpu_mobility_cache[cache_key] = gpu_mob
    else:
        gpu_mob = base_ds._gpu_mobility_cache[cache_key]

    # Extract base global_t for this chunk (global_t_gpu is shape [B*T])
    global_t_gpu = mob_batch.global_t.to(device, non_blocking=True)

    # If strict mode is enabled, add bounds assertions to catch
    # out-of-bounds gathers before they corrupt the CUDA context.
    strict = getattr(base_ds, "config", None) is not None and getattr(
        base_ds.config.model, "strict", False
    )
    if strict:
        max_t = gpu_mob.shape[0] - 1
        if (global_t_gpu > max_t).any() or (global_t_gpu < 0).any():
            import logging

            logger = logging.getLogger(__name__)
            msg = (
                f"CRITICAL: global_t_gpu indices out of bounds for gpu_mob! "
                f"Max allowed: {max_t}, Found min: {global_t_gpu.min().item()}, "
                f"Found max: {global_t_gpu.max().item()}. "
                f"Resolved base_ds length: {len(base_ds)}. "
                f"Batch run_id: {getattr(mob_batch, 'run_id', 'None')}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

    # Reconstruct adj_dense inside the batch!
    mob_batch.adj_dense = gpu_mob[global_t_gpu]


def ensure_mobility_adj_dense_ready(
    batch_data: EpiBatch, *, required: bool, context: str = ""
) -> None:
    """
    Validate that dense mobility adjacency exists when required.

    This guard is intended for pre-compiled call sites (e.g., trainer loop) so
    we can fail fast before entering a compiled graph.

    Args:
        batch_data: Batch data expected to contain ``mob_batch``.
        required: Whether mobility adjacency is required for the current path.
        context: Optional context label included in error messages.
    """
    if not required:
        return

    suffix = f" ({context})" if context else ""
    mob_batch = batch_data.mob_batch
    if mob_batch is None:
        raise ValueError(f"Missing 'mob_batch' in batch data{suffix}.")

    if not hasattr(mob_batch, "adj_dense") or mob_batch.adj_dense is None:
        raise ValueError(
            "Missing 'mob_batch.adj_dense' before compiled forward"
            f"{suffix}. Ensure inject_gpu_mobility() (or equivalent prep) runs first."
        )
