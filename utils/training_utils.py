"""
Training utilities for the EpiForecaster project.
"""

from typing import Any

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


def inject_gpu_mobility(
    batch_data: dict[str, Any],
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> None:
    """
    Constructs the dense mobility adjacency matrix directly on the GPU.

    This avoids passing huge float16 matrices through the DataLoader's
    multiprocessing queue, preventing severe worker OOMs on Linux systems.

    Args:
        batch_data: The batch dictionary containing 'MobBatch'.
        dataset: The Dataset yielding the batches (contains preloaded_mobility).
        device: The target GPU device.
    """
    mob_batch = batch_data.get("MobBatch")
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

    if device not in base_ds._gpu_mobility_cache:
        node_ids = torch.where(base_ds._get_graph_node_mask())[0]
        # Slice original CPU tensor to context nodes: resulting size [TotalT, N_ctx, N_ctx]
        # Using unsqueeze is faster and creates an intermediate of correct shape directly
        sliced_mob = base_ds.preloaded_mobility[:, node_ids.unsqueeze(-1), node_ids].to(
            torch.float16
        )

        # Transfer to GPU
        gpu_mob = sliced_mob.to(device, non_blocking=True)

        # Enforce self-loops (diagonal >= 1.0) on the GPU
        eye = torch.eye(len(node_ids), dtype=torch.float32, device=device).unsqueeze(0)
        gpu_mob = torch.maximum(gpu_mob.float(), eye).to(torch.float16)

        base_ds._gpu_mobility_cache[device] = gpu_mob
    else:
        gpu_mob = base_ds._gpu_mobility_cache[device]

    # Extract base global_t for this chunk (global_t_gpu is shape [B*T])
    global_t_gpu = mob_batch.global_t.to(device, non_blocking=True)

    # Reconstruct adj_dense inside the batch!
    mob_batch.adj_dense = gpu_mob[global_t_gpu]


def ensure_mobility_adj_dense_ready(
    batch_data: dict[str, Any], *, required: bool, context: str = ""
) -> None:
    """
    Validate that dense mobility adjacency exists when required.

    This guard is intended for pre-compiled call sites (e.g., trainer loop) so
    we can fail fast before entering a compiled graph.

    Args:
        batch_data: Batch dictionary expected to contain ``MobBatch``.
        required: Whether mobility adjacency is required for the current path.
        context: Optional context label included in error messages.
    """
    if not required:
        return

    suffix = f" ({context})" if context else ""
    mob_batch = batch_data.get("MobBatch")
    if mob_batch is None:
        raise ValueError(f"Missing 'MobBatch' in batch data{suffix}.")

    if not hasattr(mob_batch, "adj_dense") or mob_batch.adj_dense is None:
        raise ValueError(
            "Missing 'MobBatch.adj_dense' before compiled forward"
            f"{suffix}. Ensure inject_gpu_mobility() (or equivalent prep) runs first."
        )


def mask_ablated_inputs(
    batch_data: dict[str, Any],
    *,
    mask_cases: bool = False,
    mask_ww: bool = False,
    mask_hosp: bool = False,
    mask_deaths: bool = False,
) -> None:
    """
    Apply zero-masking to input data series corresponding to disabled clinical heads.

    This ensures that when an observation head's loss is ablated, its input data
    is also zeroed out in the batch so it does not leak information to other heads
    through the backbone representation.

    Args:
        batch_data: The batch data dictionary. Modified in-place.
        mask_cases: If True, zero out 'CasesHist'.
        mask_ww: If True, zero out 'BioNode'.
        mask_hosp: If True, zero out 'HospHist'.
        mask_deaths: If True, zero out 'DeathsHist'.
    """
    if mask_ww and "BioNode" in batch_data and batch_data["BioNode"] is not None:
        batch_data["BioNode"] = torch.zeros_like(batch_data["BioNode"])
    if mask_hosp and "HospHist" in batch_data and batch_data["HospHist"] is not None:
        batch_data["HospHist"] = torch.zeros_like(batch_data["HospHist"])
    if mask_cases and "CasesHist" in batch_data and batch_data["CasesHist"] is not None:
        batch_data["CasesHist"] = torch.zeros_like(batch_data["CasesHist"])
    if (
        mask_deaths
        and "DeathsHist" in batch_data
        and batch_data["DeathsHist"] is not None
    ):
        batch_data["DeathsHist"] = torch.zeros_like(batch_data["DeathsHist"])
