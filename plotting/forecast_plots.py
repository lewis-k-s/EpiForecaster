from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from data.preprocess.config import TEMPORAL_COORD
from utils.device import prepare_batch_for_device

logger = logging.getLogger(__name__)


TARGET_PLOT_SPECS: dict[str, dict[str, str]] = {
    "hosp": {
        "model_output": "pred_hosp",
        "batch_target": "hosp_target",
        "dataset_attr": "precomputed_hosp",
        "label": "Hospitalizations",
    },
    "ww": {
        "model_output": "pred_ww",
        "batch_target": "ww_target",
        "dataset_attr": "precomputed_ww",
        "label": "Wastewater",
    },
    "cases": {
        "model_output": "pred_cases",
        "batch_target": "cases_target",
        "dataset_attr": "precomputed_cases_target",
        "label": "Cases",
    },
    "deaths": {
        "model_output": "pred_deaths",
        "batch_target": "deaths_target",
        "dataset_attr": "precomputed_deaths",
        "label": "Deaths",
    },
}

DEFAULT_PLOT_TARGETS = ["hosp", "ww", "cases", "deaths"]


def _resolve_target_names(target_names: list[str] | None) -> list[str]:
    requested = target_names or list(DEFAULT_PLOT_TARGETS)
    out: list[str] = []
    for target in requested:
        if target not in TARGET_PLOT_SPECS:
            raise ValueError(
                f"Unknown target '{target}'. Valid targets: {sorted(TARGET_PLOT_SPECS)}"
            )
        out.append(target)
    return out


def _as_numpy_1d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _dataset_series_window(
    series_tensor: torch.Tensor,
    *,
    t_min: int,
    t_max: int,
    node_idx: int,
) -> np.ndarray:
    return series_tensor[t_min:t_max, node_idx].detach().cpu().numpy().reshape(-1)


def _history_window(
    series_tensor: torch.Tensor,
    *,
    start_idx: int,
    input_window_length: int,
    node_idx: int,
) -> np.ndarray:
    return (
        series_tensor[start_idx : start_idx + input_window_length, node_idx]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


def _extract_target_payload(
    sample: dict[str, Any], *, target: str | None
) -> tuple[np.ndarray, np.ndarray]:
    if target is not None and "targets" in sample and target in sample["targets"]:
        payload = sample["targets"][target]
        return (
            np.asarray(payload["actual_context"]).reshape(-1),
            np.asarray(payload["prediction"]).reshape(-1),
        )
    return (
        np.asarray(sample["actual_context"]).reshape(-1),
        np.asarray(sample["prediction"]).reshape(-1),
    )


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
    target_names: list[str] | None = None,
    validation_mode: str = "all",
    required_targets: list[str] | None = None,
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
    from data.epi_batch import collate_epiforecaster_batch

    dataset = loader.dataset
    if not isinstance(dataset, EpiDataset):
        raise TypeError(
            "collect_forecast_samples_for_target_nodes currently expects an EpiDataset."
        )

    if dataset.num_windows() == 0:
        return []
    resolved_targets = _resolve_target_names(target_names)

    # Compute valid windows specifically for plotting using the requested mode
    dataset_name_map = {
        "hosp": "hospitalizations",
        "ww": "wastewater",
        "cases": "cases",
        "deaths": "deaths",
    }

    if required_targets is None:
        mapped_required = [dataset_name_map[t] for t in resolved_targets]
    else:
        mapped_required = [
            dataset_name_map[t] for t in required_targets if t in dataset_name_map
        ]

    valid_starts_by_node = dataset.get_valid_window_starts_dict(
        mode=validation_mode, required_targets=mapped_required
    )

    indices: list[int] = []
    start_times: list[Any] = []
    start_indices: list[int] = []

    for target_node in target_node_ids:
        valid_starts = valid_starts_by_node.get(target_node, [])
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
        collate_fn=collate_epiforecaster_batch,
    )

    batch = next(iter(sample_loader))

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    try:
        batch = cast(
            Any,
            prepare_batch_for_device(
                batch,
                dataset=dataset,
                device=device,
            ),
        )

        region_embeddings = getattr(dataset, "region_embeddings", None)
        if region_embeddings is not None:
            region_embeddings = region_embeddings.to(device)

        with torch.no_grad():
            if hasattr(model, "forward_batch"):
                model_outputs, _targets_dict = model.forward_batch(
                    batch_data=batch,
                    region_embeddings=region_embeddings,
                )
            else:
                target_nodes = (
                    batch.target_region_index
                    if getattr(batch, "target_region_index", None) is not None
                    else batch.target_node
                )
                model_outputs = model.forward(
                    hosp_hist=batch.hosp_hist,
                    deaths_hist=batch.deaths_hist,
                    cases_hist=batch.cases_hist,
                    biomarkers_hist=batch.bio_node,
                    mob_graphs=batch.mob_batch,
                    target_nodes=target_nodes,
                    region_embeddings=region_embeddings,
                    population=batch.population,
                )
            if not isinstance(model_outputs, dict):
                raise ValueError(
                    "Joint inference plotting expects model outputs as a dict."
                )

        samples: list[dict[str, Any]] = []
        L = dataset.config.model.input_window_length
        H = dataset.config.model.forecast_horizon
        T_total = dataset.precomputed_cases_hist.shape[0]

        batch_size = int(batch.target_node.shape[0])
        for i in range(batch_size):
            target_node = int(batch.target_node[i].item())
            node_idx = target_node
            start_idx = start_indices[i] if i < len(start_indices) else -1
            t0 = start_idx + L

            t_min = max(0, t0 - context_pre)
            t_max = min(T_total, t0 + H + context_post)
            t_rel = np.arange(t_min, t_max, dtype=np.int64) - t0

            target_payloads: dict[str, dict[str, np.ndarray]] = {}
            for target_name in resolved_targets:
                spec = TARGET_PLOT_SPECS[target_name]
                pred_key = spec["model_output"]
                batch_key = spec["batch_target"]
                dataset_attr = spec["dataset_attr"]

                if pred_key not in model_outputs or not hasattr(batch, batch_key):
                    continue
                if not hasattr(dataset, dataset_attr):
                    continue

                pred_series = _as_numpy_1d(model_outputs[pred_key][i])
                target_series = _as_numpy_1d(batch[batch_key][i])
                dataset_tensor = getattr(dataset, dataset_attr)
                actual_context_full = _dataset_series_window(
                    dataset_tensor,
                    t_min=t_min,
                    t_max=t_max,
                    node_idx=node_idx,
                ).astype(np.float32)
                history_series = _history_window(
                    dataset_tensor,
                    start_idx=start_idx,
                    input_window_length=L,
                    node_idx=node_idx,
                ).astype(np.float32)

                target_payloads[target_name] = {
                    "actual_context": actual_context_full,
                    "prediction": np.asarray(pred_series, dtype=np.float32),
                    "target": np.asarray(target_series, dtype=np.float32),
                    "history": history_series,
                }

            if not target_payloads:
                logger.debug(
                    "[plot] Skipping node %s because no target payload was available",
                    target_node,
                )
                continue

            primary_target = resolved_targets[0]
            if primary_target not in target_payloads:
                primary_target = next(iter(target_payloads.keys()))
            primary = target_payloads[primary_target]

            samples.append(
                {
                    "node_id": target_node,
                    "node_label": str(batch.node_labels[i]),
                    "actual_context": primary["actual_context"],
                    "prediction": primary["prediction"],
                    "target": primary["target"],
                    "history": primary["history"],
                    "t_rel": t_rel,
                    "t0_idx_in_context": t0 - t_min,
                    "start_time": start_times[i] if i < len(start_times) else "",
                    "L": L,
                    "H": H,
                    "targets": target_payloads,
                }
            )
        return samples
    finally:
        if model_was_training:
            model.train()


def make_forecast_figure(
    *,
    samples: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    input_window_length: int,
    forecast_horizon: int,
    context_pre: int = 30,
    context_post: int = 30,
    target: str | None = "hosp",
    target_label: str | None = None,
):
    """
    Build a figure showing actual series vs forecasts with wider context.

    Supports either a flat list of samples (single column) or a dict of {group: samples}
    for grid layout (rows=groups).

    The plot shows:
    - Actual: Extended time series from dataset (wider than just history+horizon)
    - Forecast: Only over the forecast horizon [0, H)
    - Receptive field: Shaded region [-input_window_length, 0]
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
            actual_context, pred_series = _extract_target_payload(sample, target=target)
            t_rel = np.asarray(sample["t_rel"]).reshape(-1)
            H = sample["H"]

            forecast_series_full = np.full(len(t_rel), np.nan, dtype=np.float32)

            # Mask for the forecast horizon: 0 <= t < H
            horizon_mask = (t_rel >= 0) & (t_rel < H)
            # Only take as many predictions as we have matching time points (handles truncation)
            points_to_plot = horizon_mask.sum()
            forecast_series_full[horizon_mask] = pred_series[:points_to_plot]

            # Include last history point (t=-1) so forecast line connects from history endpoint
            history_end_mask = t_rel == -1
            if history_end_mask.any():
                forecast_series_full[history_end_mask] = actual_context[
                    history_end_mask
                ]

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
            ax.axvspan(-input_window_length, 0, color="gray", alpha=0.15)

            node_label = sample.get("node_label", "")
            start_time = sample.get("start_time", "")
            title_parts = [node_label]
            if start_time:
                title_parts.append(f"({start_time})")

            if j == 0:
                if target_label is None and target in TARGET_PLOT_SPECS:
                    axis_label = TARGET_PLOT_SPECS[target]["label"]
                elif target_label is not None:
                    axis_label = target_label
                else:
                    axis_label = "Target"
                ax.set_ylabel(f"{row_name}\n{axis_label}", fontweight="bold")
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


def make_joint_forecast_figure(
    *,
    samples: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    input_window_length: int,
    forecast_horizon: int,
    context_pre: int = 30,
    context_post: int = 30,
    target_names: list[str] | None = None,
):
    """Build a single figure containing all requested targets by group."""
    if isinstance(samples, list):
        base_groups = {"All": samples}
    else:
        base_groups = samples

    base_groups = {k: v for k, v in base_groups.items() if v}
    if not base_groups:
        return None

    resolved_targets = _resolve_target_names(target_names)
    expanded_groups: dict[str, list[dict[str, Any]]] = {}
    for target_name in resolved_targets:
        target_label = TARGET_PLOT_SPECS[target_name]["label"]
        for group_name, group_samples in base_groups.items():
            materialized: list[dict[str, Any]] = []
            for sample in group_samples:
                if "targets" in sample and target_name not in sample["targets"]:
                    continue
                row_sample = dict(sample)
                if "targets" in sample and target_name in sample["targets"]:
                    payload = sample["targets"][target_name]
                    row_sample["actual_context"] = payload["actual_context"]
                    row_sample["prediction"] = payload["prediction"]
                    row_sample["target"] = payload["target"]
                    row_sample["history"] = payload["history"]
                materialized.append(row_sample)
            if materialized:
                expanded_groups[f"{group_name} | {target_label}"] = materialized

    if not expanded_groups:
        return None

    return make_forecast_figure(
        samples=expanded_groups,
        input_window_length=input_window_length,
        forecast_horizon=forecast_horizon,
        context_pre=context_pre,
        context_post=context_post,
        target=None,
        target_label=None,
    )


def generate_forecast_plots(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    node_groups: dict[str, list[int]],
    window: str = "last",
    context_pre: int = 30,
    context_post: int = 30,
    output_path: Path | None = None,
    log_dir: Path | None = None,
    target_names: list[str] | None = None,
    wandb_prefix: str = "forecasts",
) -> dict[str, Any]:
    """
    Generate forecast plots for given node groups (generic).

    This is an orchestration function that coordinates plotting via
    collect_forecast_samples_for_target_nodes, make_forecast_figure,
    and make_joint_forecast_figure.

    Args:
        model: The trained model
        loader: Original DataLoader for data access
        node_groups: Dict mapping group name -> list of node IDs
                     (could be quartiles, topk, worst, random, anything!)
        window: Which time window to plot ("last" or "random")
        context_pre: Days before forecast start
        context_post: Days after forecast end
        output_path: Optional path to save figure
        log_dir: Optional W&B run directory for eval metrics
        target_names: List of targets to plot (default: all)
        wandb_prefix: Prefix for W&B logged images

    Returns:
        Dict with figure, all_samples, selected_nodes, node_groups
    """
    import wandb

    from evaluation.eval_loop import _ensure_wandb_run

    # Flatten all nodes to collect samples once
    all_selected_nodes: list[int] = []
    for group_nodes in node_groups.values():
        all_selected_nodes.extend(group_nodes)

    if not all_selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "all_samples": [],
            "selected_nodes": [],
            "node_groups": {},
        }

    logger.info(
        f"[plot] Collecting forecast samples for {len(all_selected_nodes)} nodes..."
    )

    # Use existing function - it handles subset creation internally
    resolved_targets = target_names or list(DEFAULT_PLOT_TARGETS)

    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=all_selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
    )

    # Group samples by original group names
    node_to_group: dict[int, str] = {}
    for group_name, nodes in node_groups.items():
        for node_id in nodes:
            node_to_group[node_id] = group_name

    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        node_id = sample["node_id"]
        if node_id in node_to_group:
            group_name = node_to_group[node_id]
            if group_name not in grouped_samples:
                grouped_samples[group_name] = []
            grouped_samples[group_name].append(sample)

    # Generate figure using existing generic function
    dataset = cast(EpiDataset, loader.dataset)
    config = dataset.config
    fig = make_joint_forecast_figure(
        samples=grouped_samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    separate_figures: dict[str, Any] = {}
    for target_name in resolved_targets:
        target_fig = make_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=context_pre,
            context_post=context_post,
            target=target_name,
        )
        if target_fig is None:
            continue
        separate_figures[target_name] = target_fig
        if output_path is not None:
            target_output_path = output_path.with_name(
                f"{output_path.stem}_{target_name}{output_path.suffix}"
            )
            target_fig.savefig(target_output_path, dpi=200, bbox_inches="tight")
            logger.info(f"[plot] Saved figure to: {target_output_path}")

    if fig is not None and (log_dir is not None or wandb.run is not None):
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        _ensure_wandb_run(
            config=dataset.config,
            log_dir=log_dir,
            name="forecast_plots",
            job_type="eval",
        )
        if wandb.run is not None:
            log_payload: dict[str, Any] = {}
            if fig is not None:
                log_payload[f"{wandb_prefix}/joint"] = wandb.Image(fig)
            for target_name, target_fig in separate_figures.items():
                log_payload[f"{wandb_prefix}/{target_name}"] = wandb.Image(target_fig)
            if log_payload:
                wandb.log(log_payload, step=0)

    return {
        "figure": fig,
        "joint_figure": fig,
        "separate_figures": separate_figures,
        "all_samples": samples,
        "selected_nodes": all_selected_nodes,
        "node_groups": node_groups,
    }
