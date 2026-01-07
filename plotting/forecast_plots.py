from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from data.preprocess.config import TEMPORAL_COORD
from utils.normalization import unscale_forecasts


def indices_for_target_nodes_in_window(
    *,
    target_nodes: list[int],
    target_node_ids: list[int],
    window_idx: int,
    dataset: EpiDataset | None = None,
) -> list[int]:
    """
    Compute dataset indices for given target nodes in a particular window.

    If a dataset is provided, look up indices using its window mapping (which
    accounts for missingness filtering). Otherwise, fall back to the legacy
    assumption that every node has every window.
    """
    indices: list[int] = []
    if dataset is not None:
        for target_node in target_node_ids:
            try:
                indices.append(
                    dataset.index_for_target_node_window(
                        target_node=target_node, window_idx=window_idx
                    )
                )
            except (KeyError, IndexError):
                continue
        return indices

    node_to_local = {n: i for i, n in enumerate(target_nodes)}
    N = len(target_nodes)
    for target_node in target_node_ids:
        local_idx = node_to_local.get(target_node)
        if local_idx is None:
            continue
        indices.append((window_idx * N) + local_idx)
    return indices


def collect_forecast_samples_for_target_nodes(
    *,
    target_node_ids: list[int],
    model: torch.nn.Module,
    loader: DataLoader,
    window: str = "last",
    context_pre: int = 30,
    context_post: int = 30,
) -> list[dict[str, Any]]:
    """
    Run model inference for a specific subset of target nodes and return raw series.

    This intentionally contains no plotting logic; it only materializes inputs
    for chosen target-node IDs from the loader's dataset and runs a single
    forward pass.

    Args:
        context_pre: Number of days to include before forecast start (default: 30).
        context_post: Number of days to include after forecast end (default: 30).
    """
    import os
    import random

    from torch.utils.data import Subset
    from data.collate import collate_epidataset_batch

    dataset = loader.dataset
    if not isinstance(dataset, EpiDataset):
        raise TypeError(
            "collect_forecast_samples_for_target_nodes currently expects an EpiDataset."
        )

    if dataset.num_windows() == 0:
        return []

    indices: list[int] = []
    start_times: list[Any] = []
    start_indices: list[int] = []

    for target_node in target_node_ids:
        valid_starts = dataset._valid_window_starts_by_node.get(target_node, [])
        if not valid_starts:
            continue

        start_idx = -1
        if window == "last":
            start_idx = valid_starts[-1]
        elif window == "random":
            start_idx = random.choice(valid_starts)
        else:
            raise ValueError(f"Unknown window spec: {window}")

        idx = dataset._index_lookup.get((target_node, start_idx))
        if idx is not None:
            indices.append(idx)
            start_indices.append(start_idx)
            try:
                time_val = dataset.dataset[TEMPORAL_COORD].values[start_idx]
                start_times.append(str(time_val).split("T")[0])
            except Exception:
                start_times.append(f"t={start_idx}")

    if not indices:
        return []

    subset_dataset = Subset(dataset, indices)

    avail_cores = (os.cpu_count() or 1) - 1
    num_workers = min(avail_cores, 4)

    sample_loader = DataLoader(
        subset_dataset,
        batch_size=len(indices),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_epidataset_batch,
    )

    batch = next(iter(sample_loader))

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    try:
        mob_batch = batch["MobBatch"].to(device)
        region_embeddings = getattr(dataset, "region_embeddings", None)
        if region_embeddings is not None:
            region_embeddings = region_embeddings.to(device)

        with torch.no_grad():
            predictions = model.forward(
                cases_norm=batch["CaseNode"].to(device),
                cases_mean=batch["CaseMean"].to(device),
                cases_std=batch["CaseStd"].to(device),
                biomarkers_hist=batch["BioNode"].to(device),
                mob_graphs=mob_batch,
                target_nodes=batch["TargetNode"].to(device),
                region_embeddings=region_embeddings,
                population=batch["Population"].to(device),
            )

            pred_unscaled, targets_unscaled = unscale_forecasts(
                predictions,
                batch["Target"].to(device),
                batch["TargetMean"].to(device),
                batch["TargetScale"].to(device),
            )

            history_norm = batch["CaseNode"].to(device)
            case_mean = batch["CaseMean"].to(device)
            case_std = batch["CaseStd"].to(device)
            history_unscaled = history_norm * case_std + case_mean

        samples: list[dict[str, Any]] = []
        L = dataset.config.model.history_length
        H = dataset.config.model.forecast_horizon
        T_total = dataset.precomputed_cases.shape[0]

        for i in range(int(predictions.shape[0])):
            target_node = int(batch["TargetNode"][i].item())
            node_idx = target_node
            start_idx = start_indices[i] if i < len(start_indices) else -1
            t0 = start_idx + L

            t_min = max(0, t0 - context_pre)
            t_max = min(T_total, t0 + H + context_post)

            actual_context_full = dataset.precomputed_cases[
                t_min:t_max, node_idx, 0
            ].numpy()
            t_rel = np.arange(t_min, t_max, dtype=np.int64) - t0

            pred_series = pred_unscaled[i].detach().cpu().numpy().reshape(-1)
            target_series = targets_unscaled[i].detach().cpu().numpy().reshape(-1)
            history_series = history_unscaled[i].detach().cpu().numpy().reshape(-1)

            samples.append(
                {
                    "node_id": target_node,
                    "node_label": str(batch["NodeLabels"][i]),
                    "actual_context": actual_context_full.astype(np.float32),
                    "prediction": np.asarray(pred_series, dtype=np.float32),
                    "target": np.asarray(target_series, dtype=np.float32),
                    "history": np.asarray(history_series, dtype=np.float32),
                    "t_rel": t_rel,
                    "t0_idx_in_context": t0 - t_min,
                    "start_time": start_times[i] if i < len(start_times) else "",
                    "L": L,
                    "H": H,
                }
            )
        return samples
    finally:
        if model_was_training:
            model.train()


def make_forecast_figure(
    *,
    samples: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    history_length: int,
    forecast_horizon: int,
    context_pre: int = 30,
    context_post: int = 30,
):
    """
    Build a figure showing actual series vs forecasts with wider context.

    Supports either a flat list of samples (single column) or a dict of {group: samples}
    for grid layout (rows=groups).

    The plot shows:
    - Actual: Extended time series from dataset (wider than just history+horizon)
    - Forecast: Only over the forecast horizon [0, H)
    - Receptive field: Shaded region [-history_length, 0]
    - Forecast boundary: Vertical line at t=0
    """
    if not samples:
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if isinstance(samples, list):
        groups = {"All": samples}
    else:
        groups = samples

    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return None

    row_names = list(groups.keys())
    nrows = len(row_names)
    ncols = max(len(v) for v in groups.values())

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for i, row_name in enumerate(row_names):
        row_samples = groups[row_name]
        for j, sample in enumerate(row_samples):
            ax = axes[i, j]
            actual_context = np.asarray(sample["actual_context"]).reshape(-1)
            pred_series = np.asarray(sample["prediction"]).reshape(-1)
            t_rel = np.asarray(sample["t_rel"]).reshape(-1)
            L = sample["L"]
            H = sample["H"]

            forecast_series_full = np.full(len(t_rel), np.nan, dtype=np.float32)
            
            # Mask for the forecast horizon: 0 <= t < H
            horizon_mask = (t_rel >= 0) & (t_rel < H)
            # Only take as many predictions as we have matching time points (handles truncation)
            points_to_plot = horizon_mask.sum()
            forecast_series_full[horizon_mask] = pred_series[:points_to_plot]

            df = pd.DataFrame(
                {
                    "t": np.concatenate([t_rel, t_rel], axis=0),
                    "value": np.concatenate(
                        [actual_context, forecast_series_full], axis=0
                    ),
                    "series": ["Actual"] * len(t_rel) + ["Forecast"] * len(t_rel),
                }
            )

            sns.lineplot(
                data=df,
                x="t",
                y="value",
                hue="series",
                ax=ax,
                legend=(j == 0 and i == 0),
            )

            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.axvspan(-history_length, 0, color="gray", alpha=0.15)

            node_label = sample.get("node_label", "")
            start_time = sample.get("start_time", "")
            title_parts = [node_label]
            if start_time:
                title_parts.append(f"({start_time})")

            if j == 0:
                ax.set_ylabel(f"{row_name}\nCases", fontweight="bold")
            else:
                ax.set_ylabel("")

            if i == nrows - 1:
                ax.set_xlabel("Time (days relative to forecast start)")
            else:
                ax.set_xlabel("")

            ax.set_title(" ".join(title_parts), fontsize=10)

        for j in range(len(row_samples), ncols):
            axes[i, j].set_visible(False)

    fig.tight_layout()
    return fig
