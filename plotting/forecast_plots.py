from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from data.preprocess.config import TEMPORAL_COORD
from utils.device import prepare_batch_for_device

if TYPE_CHECKING:
    from evaluation.selection import WindowSelectionSpec

logger = logging.getLogger(__name__)


TARGET_PLOT_SPECS: dict[str, dict[str, str]] = {
    "hosp": {
        "model_output": "pred_hosp",
        "batch_target": "hosp_target",
        "batch_mask": "hosp_target_mask",
        "dataset_attr": "precomputed_hosp",
        "label": "Hospitalizations",
    },
    "ww": {
        "model_output": "pred_ww",
        "batch_target": "ww_target",
        "batch_mask": "ww_target_mask",
        "dataset_attr": "precomputed_ww",
        "label": "Wastewater",
    },
    "cases": {
        "model_output": "pred_cases",
        "batch_target": "cases_target",
        "batch_mask": "cases_target_mask",
        "dataset_attr": "precomputed_cases_target",
        "label": "Cases",
    },
    "deaths": {
        "model_output": "pred_deaths",
        "batch_target": "deaths_target",
        "batch_mask": "deaths_target_mask",
        "dataset_attr": "precomputed_deaths",
        "label": "Deaths",
    },
}

DEFAULT_PLOT_TARGETS = ["hosp", "ww", "cases", "deaths"]
LATENT_PLOT_SPECS: dict[str, dict[str, str]] = {
    "latent_s": {
        "model_output": "S_trajectory",
        "label": "Latent S",
    },
    "latent_i": {
        "model_output": "I_trajectory",
        "label": "Latent I",
    },
    "latent_h": {
        "model_output": "H_trajectory",
        "label": "Latent H",
    },
    "latent_r": {
        "model_output": "R_trajectory",
        "label": "Latent R",
    },
    "latent_d": {
        "model_output": "D_trajectory",
        "label": "Latent D",
    },
}


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


def _resolve_latent_series(
    model_outputs: dict[str, Any],
) -> dict[str, torch.Tensor | np.ndarray]:
    latents: dict[str, torch.Tensor | np.ndarray] = {}
    for latent_name, spec in LATENT_PLOT_SPECS.items():
        model_output = spec["model_output"]
        if model_output in model_outputs:
            latents[latent_name] = model_outputs[model_output]
    return latents


def _build_prediction_only_context(
    *,
    t_rel: np.ndarray,
    horizon: int,
    prediction: np.ndarray,
) -> np.ndarray:
    context = np.full(len(t_rel), np.nan, dtype=np.float32)
    horizon_mask = (t_rel >= 0) & (t_rel < horizon)
    points_to_plot = min(int(horizon_mask.sum()), int(prediction.shape[0]))
    if points_to_plot > 0:
        context[np.where(horizon_mask)[0][:points_to_plot]] = prediction[
            :points_to_plot
        ]
    return context


def _as_numpy_1d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _align_prediction_to_target(
    prediction: torch.Tensor | np.ndarray,
    *,
    target_length: int,
) -> np.ndarray:
    aligned = _as_numpy_1d(prediction).astype(np.float32)
    if aligned.shape[0] == target_length + 1:
        return aligned[1:]
    if aligned.shape[0] >= target_length:
        return aligned[:target_length]
    return aligned


def _dataset_series_window(
    series_tensor: torch.Tensor,
    *,
    t_min: int,
    t_max: int,
    node_idx: int,
) -> np.ndarray:
    return series_tensor[t_min:t_max, node_idx].detach().cpu().numpy().reshape(-1)


def _dataset_series_full(
    series_tensor: torch.Tensor,
    *,
    node_idx: int,
) -> np.ndarray:
    return series_tensor[:, node_idx].detach().cpu().numpy().reshape(-1)


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


def _extract_series_payload(
    sample: dict[str, Any],
    *,
    target: str | None,
    payload_collection: str = "targets",
) -> tuple[np.ndarray, np.ndarray, float | None]:
    payloads = sample.get(payload_collection)
    if target is not None and isinstance(payloads, dict) and target in payloads:
        payload = payloads[target]
        return (
            np.asarray(payload["actual_context"]).reshape(-1),
            np.asarray(payload["prediction"]).reshape(-1),
            float(payload["window_mae"])
            if payload.get("window_mae") is not None
            else None,
        )
    return (
        np.asarray(sample["actual_context"]).reshape(-1),
        np.asarray(sample["prediction"]).reshape(-1),
        float(sample["window_mae"]) if sample.get("window_mae") is not None else None,
    )


def _compute_window_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    observed_mask: np.ndarray | None,
) -> float | None:
    pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
    actual = np.asarray(target, dtype=np.float32).reshape(-1)
    if pred.shape[0] != actual.shape[0]:
        size = min(pred.shape[0], actual.shape[0])
        pred = pred[:size]
        actual = actual[:size]
    if observed_mask is None:
        if pred.size == 0:
            return None
        return float(np.abs(pred - actual).mean())
    mask = np.asarray(observed_mask, dtype=np.float32).reshape(-1)
    mask = mask[: pred.shape[0]] > 0
    if not np.any(mask):
        return None
    return float(np.abs(pred[mask] - actual[mask]).mean())


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
            start_times.append(_format_start_time(dataset, start_idx))

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
            latent_payloads: dict[str, dict[str, np.ndarray]] = {}
            for target_name in resolved_targets:
                spec = TARGET_PLOT_SPECS[target_name]
                pred_key = spec["model_output"]
                batch_key = spec["batch_target"]
                mask_key = spec["batch_mask"]
                dataset_attr = spec["dataset_attr"]

                if pred_key not in model_outputs or not hasattr(batch, batch_key):
                    continue
                if not hasattr(dataset, dataset_attr):
                    continue

                target_series = _as_numpy_1d(getattr(batch, batch_key)[i])
                pred_series = _align_prediction_to_target(
                    model_outputs[pred_key][i],
                    target_length=target_series.shape[0],
                )
                target_mask = (
                    _as_numpy_1d(getattr(batch, mask_key)[i])
                    if hasattr(batch, mask_key)
                    else None
                )
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
                full_series = _dataset_series_full(
                    dataset_tensor,
                    node_idx=node_idx,
                ).astype(np.float32)

                target_payloads[target_name] = {
                    "actual_context": actual_context_full,
                    "prediction": np.asarray(pred_series, dtype=np.float32),
                    "target": np.asarray(target_series, dtype=np.float32),
                    "history": history_series,
                    "full_series": full_series,
                    "window_mae": _compute_window_mae(
                        pred_series,
                        target_series,
                        target_mask,
                    ),
                }

            latent_outputs = _resolve_latent_series(model_outputs)
            for latent_name, latent_series_raw in latent_outputs.items():
                pred_series = _align_prediction_to_target(
                    latent_series_raw[i],
                    target_length=H,
                ).astype(np.float32)
                latent_payloads[latent_name] = {
                    "actual_context": _build_prediction_only_context(
                        t_rel=t_rel,
                        horizon=H,
                        prediction=pred_series,
                    ),
                    "prediction": pred_series,
                    "target": np.full_like(pred_series, np.nan),
                    "history": np.empty(0, dtype=np.float32),
                    "full_series": np.full(T_total, np.nan, dtype=np.float32),
                    "window_mae": None,
                }

            if not target_payloads and not latent_payloads:
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
                    "window_start": start_idx,
                    "window_mae": primary.get("window_mae"),
                    "L": L,
                    "H": H,
                    "t0": t0,
                    "series_t": np.arange(T_total, dtype=np.int64),
                    "targets": target_payloads,
                    "latents": latent_payloads,
                }
            )
        return samples
    finally:
        if model_was_training:
            model.train()


def _format_start_time(dataset: Any, start_idx: int) -> str:
    temporal_coords = getattr(dataset, "_temporal_coords", None)
    if temporal_coords is not None and 0 <= start_idx < len(temporal_coords):
        return str(temporal_coords[start_idx]).split("T")[0]
    try:
        time_val = dataset.dataset[TEMPORAL_COORD].values[start_idx]
        return str(time_val).split("T")[0]
    except Exception:
        return f"t={start_idx}"


def _resolve_dataset_index_for_window_start(
    dataset: Any,
    *,
    target_node: int,
    window_start: int,
) -> int | None:
    index_lookup = getattr(dataset, "_index_lookup", None)
    if index_lookup is not None:
        dataset_idx = index_lookup.get((target_node, window_start))
        if dataset_idx is not None:
            return int(dataset_idx)

    window_starts = getattr(dataset, "window_starts", None)
    if window_starts is None:
        return None

    try:
        window_idx = list(window_starts).index(window_start)
    except ValueError:
        return None

    try:
        return int(
            dataset.index_for_target_node_window(
                target_node=target_node,
                window_idx=window_idx,
            )
        )
    except Exception:
        return None


def collect_forecast_samples_for_window_specs(
    *,
    window_specs: list[WindowSelectionSpec],
    model: torch.nn.Module,
    loader: DataLoader,
    context_pre: int = 30,
    context_post: int = 30,
    target_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    import os

    from torch.utils.data import Subset

    from data.epi_batch import collate_epiforecaster_batch

    dataset = loader.dataset
    if not isinstance(dataset, EpiDataset):
        raise TypeError(
            "collect_forecast_samples_for_window_specs currently expects an EpiDataset."
        )

    if not window_specs:
        return []

    resolved_targets = _resolve_target_names(target_names)
    indices: list[int] = []
    start_indices: list[int] = []
    start_times: list[str] = []
    ordered_specs: list[WindowSelectionSpec] = []

    for spec in window_specs:
        dataset_idx = _resolve_dataset_index_for_window_start(
            dataset,
            target_node=int(spec.node_id),
            window_start=int(spec.window_start),
        )
        if dataset_idx is None:
            continue
        indices.append(int(dataset_idx))
        start_indices.append(int(spec.window_start))
        start_times.append(_format_start_time(dataset, int(spec.window_start)))
        ordered_specs.append(spec)

    if not indices:
        return []

    avail_cores = (os.cpu_count() or 1) - 1
    num_workers = min(avail_cores, 4)
    subset_dataset = Subset(dataset, indices)
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
                raise ValueError("Exact-window plotting expects forward_batch support.")

        results: list[dict[str, Any]] = []
        L = dataset.config.model.input_window_length
        H = dataset.config.model.forecast_horizon
        T_total = dataset.precomputed_cases_hist.shape[0]
        for i, spec in enumerate(ordered_specs):
            node_idx = int(spec.node_id)
            start_idx = int(spec.window_start)
            t0 = start_idx + L
            t_min = max(0, t0 - context_pre)
            t_max = min(T_total, t0 + H + context_post)
            t_rel = np.arange(t_min, t_max, dtype=np.int64) - t0

            target_payloads: dict[str, dict[str, np.ndarray]] = {}
            latent_payloads: dict[str, dict[str, np.ndarray]] = {}
            for target_name in resolved_targets:
                plot_spec = TARGET_PLOT_SPECS[target_name]
                pred_key = plot_spec["model_output"]
                batch_key = plot_spec["batch_target"]
                mask_key = plot_spec["batch_mask"]
                dataset_attr = plot_spec["dataset_attr"]
                if pred_key not in model_outputs or not hasattr(batch, batch_key):
                    continue
                if not hasattr(dataset, dataset_attr):
                    continue
                target_series = _as_numpy_1d(getattr(batch, batch_key)[i])
                pred_series = _align_prediction_to_target(
                    model_outputs[pred_key][i],
                    target_length=target_series.shape[0],
                )
                target_mask = (
                    _as_numpy_1d(getattr(batch, mask_key)[i])
                    if hasattr(batch, mask_key)
                    else None
                )
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
                full_series = _dataset_series_full(
                    dataset_tensor,
                    node_idx=node_idx,
                ).astype(np.float32)
                target_payloads[target_name] = {
                    "actual_context": actual_context_full,
                    "prediction": np.asarray(pred_series, dtype=np.float32),
                    "target": np.asarray(target_series, dtype=np.float32),
                    "history": history_series,
                    "full_series": full_series,
                    "window_mae": _compute_window_mae(
                        pred_series,
                        target_series,
                        target_mask,
                    ),
                }

            latent_outputs = _resolve_latent_series(model_outputs)
            for latent_name, latent_series_raw in latent_outputs.items():
                pred_series = _align_prediction_to_target(
                    latent_series_raw[i],
                    target_length=H,
                ).astype(np.float32)
                latent_payloads[latent_name] = {
                    "actual_context": _build_prediction_only_context(
                        t_rel=t_rel,
                        horizon=H,
                        prediction=pred_series,
                    ),
                    "prediction": pred_series,
                    "target": np.full_like(pred_series, np.nan),
                    "history": np.empty(0, dtype=np.float32),
                    "full_series": np.full(T_total, np.nan, dtype=np.float32),
                    "window_mae": None,
                }

            if not target_payloads and not latent_payloads:
                continue
            primary_target = resolved_targets[0]
            if primary_target not in target_payloads:
                primary_target = next(iter(target_payloads.keys()), None)
            if primary_target is not None:
                primary = target_payloads[primary_target]
            else:
                first_latent = next(iter(latent_payloads.values()))
                primary = first_latent
            results.append(
                {
                    "node_id": node_idx,
                    "node_label": str(batch.node_labels[i]),
                    "actual_context": primary["actual_context"],
                    "prediction": primary["prediction"],
                    "target": primary["target"],
                    "history": primary["history"],
                    "t_rel": t_rel,
                    "t0_idx_in_context": t0 - t_min,
                    "start_time": start_times[i],
                    "window_start": start_idx,
                    "window_mae": primary.get("window_mae"),
                    "L": L,
                    "H": H,
                    "t0": t0,
                    "series_t": np.arange(T_total, dtype=np.int64),
                    "targets": target_payloads,
                    "latents": latent_payloads,
                }
            )
        return results
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
    figure_title: str | None = None,
    shared_xlabel: str | None = None,
    payload_collection: str = "targets",
    connect_from_history: bool = True,
    overlay_target: str | None = None,
    overlay_payload_collection: str = "latents",
    overlay_label: str | None = None,
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
            actual_context, pred_series, window_mae = _extract_series_payload(
                sample,
                target=target,
                payload_collection=payload_collection,
            )
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
            if connect_from_history and history_end_mask.any():
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

            payloads = sample.get(overlay_payload_collection)
            if (
                overlay_target is not None
                and isinstance(payloads, dict)
                and overlay_target in payloads
            ):
                _overlay_context, overlay_prediction, _overlay_mae = (
                    _extract_series_payload(
                        sample,
                        target=overlay_target,
                        payload_collection=overlay_payload_collection,
                    )
                )
                overlay_series_full = np.full(len(t_rel), np.nan, dtype=np.float32)
                overlay_horizon_mask = (t_rel >= 0) & (t_rel < H)
                overlay_points = min(
                    int(overlay_horizon_mask.sum()),
                    int(overlay_prediction.shape[0]),
                )
                if overlay_points > 0:
                    overlay_series_full[
                        np.where(overlay_horizon_mask)[0][:overlay_points]
                    ] = overlay_prediction[:overlay_points]

                overlay_ax = ax.twinx()
                overlay_name = overlay_label or overlay_target
                overlay_df = pd.DataFrame(
                    {
                        "t": t_rel,
                        "value": overlay_series_full,
                    }
                )
                sns.lineplot(
                    data=overlay_df,
                    x="t",
                    y="value",
                    ax=overlay_ax,
                    color="#C44E52",
                    linewidth=1.5,
                    legend=False,
                )
                overlay_ax.set_ylabel(overlay_name if j == ncols - 1 else "")
                overlay_ax.grid(False)

            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.axvspan(-input_window_length, 0, color="gray", alpha=0.15)

            node_label = sample.get("node_label", "")
            start_time = sample.get("start_time", "")
            title_parts = [node_label]
            if start_time:
                title_parts.append(f"({start_time})")
            if window_mae is not None:
                title_parts.append(f"MAE={window_mae:.3f}")

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
                ax.set_xlabel(
                    ""
                    if shared_xlabel is not None
                    else "Time (days relative to forecast start)"
                )
            else:
                ax.set_xlabel("")

            ax.set_title(" ".join(title_parts), fontsize=10)

        for j in range(len(row_samples), ncols):
            axes[i, j].set_visible(False)

    if figure_title is not None:
        fig.suptitle(figure_title)

    if shared_xlabel is not None:
        fig.supxlabel(shared_xlabel)

    if figure_title is not None or shared_xlabel is not None:
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    else:
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
                    row_sample["window_mae"] = payload.get("window_mae")
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


def make_joint_latent_forecast_figure(
    *,
    samples: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    input_window_length: int,
    forecast_horizon: int,
    context_pre: int = 30,
    context_post: int = 30,
    latent_names: list[str] | None = None,
):
    """Build a single figure containing all available SIRHD latent trajectories."""
    if isinstance(samples, list):
        base_groups = {"All": samples}
    else:
        base_groups = samples

    base_groups = {k: v for k, v in base_groups.items() if v}
    if not base_groups:
        return None

    requested_latents = latent_names or list(LATENT_PLOT_SPECS)
    unknown_latents = [
        latent_name
        for latent_name in requested_latents
        if latent_name not in LATENT_PLOT_SPECS
    ]
    if unknown_latents:
        raise ValueError(
            f"Unknown latent series {unknown_latents}. "
            f"Valid latents: {sorted(LATENT_PLOT_SPECS)}"
        )

    expanded_groups: dict[str, list[dict[str, Any]]] = {}
    for latent_name in requested_latents:
        latent_label = LATENT_PLOT_SPECS[latent_name]["label"]
        for group_name, group_samples in base_groups.items():
            materialized: list[dict[str, Any]] = []
            for sample in group_samples:
                if "latents" not in sample or latent_name not in sample["latents"]:
                    continue
                row_sample = dict(sample)
                payload = sample["latents"][latent_name]
                row_sample["actual_context"] = payload["actual_context"]
                row_sample["prediction"] = payload["prediction"]
                row_sample["target"] = payload["target"]
                row_sample["history"] = payload["history"]
                row_sample["window_mae"] = payload.get("window_mae")
                materialized.append(row_sample)
            if materialized:
                expanded_groups[f"{group_name} | {latent_label}"] = materialized

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
        payload_collection="latents",
        connect_from_history=False,
    )


def make_forecast_history_figure(
    *,
    samples: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    input_window_length: int,
    forecast_horizon: int,
    target: str | None = "hosp",
    target_label: str | None = None,
    figure_title: str | None = None,
    shared_xlabel: str | None = None,
    payload_collection: str = "targets",
    max_forecasts_per_region: int | None = None,
):
    """Build a figure with a long observed series and multiple forecast windows per node."""
    if not samples:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(samples, list):
        groups = {"All": samples}
    else:
        groups = samples

    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return None

    grouped_nodes: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for group_name, group_samples in groups.items():
        node_map: dict[int, list[dict[str, Any]]] = {}
        for sample in group_samples:
            node_map.setdefault(int(sample["node_id"]), []).append(sample)
        ordered_nodes = sorted(
            node_map.items(),
            key=lambda item: (
                np.nanmean(
                    [
                        float(s.get("window_mae"))
                        for s in item[1]
                        if s.get("window_mae") is not None
                    ]
                )
                if any(s.get("window_mae") is not None for s in item[1])
                else np.inf,
                item[0],
            ),
        )
        grouped_nodes[group_name] = ordered_nodes

    row_names = list(grouped_nodes.keys())
    nrows = len(row_names)
    ncols = max(len(v) for v in grouped_nodes.values())

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    forecast_color = sns.color_palette()[1]
    actual_color = sns.color_palette()[0]

    for i, row_name in enumerate(row_names):
        row_nodes = grouped_nodes[row_name]
        for j, (_node_id, node_samples) in enumerate(row_nodes):
            ax = axes[i, j]
            sorted_samples = sorted(
                node_samples,
                key=lambda sample: int(sample.get("window_start", sample.get("t0", 0))),
            )
            if (
                max_forecasts_per_region is not None
                and len(sorted_samples) > max_forecasts_per_region
            ):
                selected_idx = np.linspace(
                    0,
                    len(sorted_samples) - 1,
                    num=max_forecasts_per_region,
                    dtype=int,
                )
                sorted_samples = [sorted_samples[idx] for idx in selected_idx]

            full_series, _pred_series, _window_mae = _extract_series_payload(
                sorted_samples[0],
                target=target,
                payload_collection=payload_collection,
            )
            payloads = sorted_samples[0].get(payload_collection)
            if (
                target is not None
                and isinstance(payloads, dict)
                and target in payloads
                and "full_series" in payloads[target]
            ):
                full_series = np.asarray(payloads[target]["full_series"]).reshape(-1)
            series_t = np.asarray(
                sorted_samples[0].get(
                    "series_t", np.arange(len(full_series), dtype=np.int64)
                )
            ).reshape(-1)

            ax.plot(
                series_t, full_series, color=actual_color, linewidth=1.5, label="Actual"
            )

            mae_values = [
                float(sample["window_mae"])
                for sample in sorted_samples
                if sample.get("window_mae") is not None
            ]
            mean_mae = float(np.mean(mae_values)) if mae_values else None

            for forecast_idx, sample in enumerate(sorted_samples):
                _actual_context, pred_series, _sample_mae = _extract_series_payload(
                    sample,
                    target=target,
                    payload_collection=payload_collection,
                )
                t0 = int(
                    sample.get("t0", int(sample["window_start"]) + forecast_horizon)
                )
                pred_t = t0 + np.arange(len(pred_series), dtype=np.int64)
                valid_mask = pred_t < len(full_series)
                pred_t = pred_t[valid_mask]
                pred_values = pred_series[: pred_t.shape[0]]
                if pred_t.size == 0:
                    continue
                ax.plot(
                    pred_t,
                    pred_values,
                    color=forecast_color,
                    linewidth=1.8,
                    alpha=0.45
                    + (0.45 * ((forecast_idx + 1) / max(len(sorted_samples), 1))),
                    label="Forecast" if forecast_idx == 0 else None,
                )
                ax.axvline(
                    t0, color=forecast_color, linestyle="--", alpha=0.12, linewidth=0.8
                )

            node_label = sorted_samples[0].get("node_label", "")
            title_parts = [str(node_label), f"n={len(sorted_samples)} forecasts"]
            if mean_mae is not None:
                title_parts.append(f"mean MAE={mean_mae:.3f}")
            ax.set_title(" | ".join(title_parts), fontsize=10)

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
                ax.set_xlabel("" if shared_xlabel is not None else "Time index")
            else:
                ax.set_xlabel("")

            if i == 0 and j == 0:
                ax.legend(loc="upper left")

        for j in range(len(row_nodes), ncols):
            axes[i, j].set_visible(False)

    if figure_title is not None:
        fig.suptitle(figure_title)
    if shared_xlabel is not None:
        fig.supxlabel(shared_xlabel)
    if figure_title is not None or shared_xlabel is not None:
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def generate_forecast_plots(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    node_groups: dict[str, list[int]] | None = None,
    window_groups: dict[str, list[WindowSelectionSpec]] | None = None,
    window: str = "last",
    context_pre: int = 30,
    context_post: int = 30,
    output_path: Path | None = None,
    log_dir: Path | None = None,
    target_names: list[str] | None = None,
    wandb_prefix: str = "forecasts",
    include_sird_latents: bool = False,
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
        window_groups: Dict mapping group name -> exact `(node, window)` specs
        window: Which time window to plot ("last" or "random")
        context_pre: Days before forecast start
        context_post: Days after forecast end
        output_path: Optional path to save figure
        log_dir: Optional W&B run directory for eval metrics
        target_names: List of targets to plot (default: all)
        wandb_prefix: Prefix for W&B logged images
        include_sird_latents: Save/log an additional joint SIRHD latent forecast figure

    Returns:
        Dict with figure, all_samples, selected_nodes, node_groups, window_groups
    """
    import wandb

    from evaluation.eval_loop import _ensure_wandb_run

    if window_groups is None and node_groups is None:
        raise ValueError(
            "generate_forecast_plots requires node_groups or window_groups"
        )

    if window_groups is not None:
        all_selected_windows: list[WindowSelectionSpec] = []
        for group_specs in window_groups.values():
            all_selected_windows.extend(group_specs)
        if not all_selected_windows:
            logger.warning("[plot] No windows selected for plotting")
            return {
                "figure": None,
                "all_samples": [],
                "selected_nodes": [],
                "node_groups": {},
                "window_groups": {},
            }
    else:
        all_selected_windows = []

    all_selected_nodes: list[int] = []
    if node_groups is not None:
        for group_nodes in node_groups.values():
            all_selected_nodes.extend(group_nodes)
    elif window_groups is not None:
        all_selected_nodes = [spec.node_id for spec in all_selected_windows]

    if not all_selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "all_samples": [],
            "selected_nodes": [],
            "node_groups": {},
            "window_groups": {},
        }

    logger.info(
        f"[plot] Collecting forecast samples for {len(all_selected_nodes)} nodes..."
    )

    # Use existing function - it handles subset creation internally
    resolved_targets = target_names or list(DEFAULT_PLOT_TARGETS)

    if window_groups is not None:
        samples = collect_forecast_samples_for_window_specs(
            window_specs=all_selected_windows,
            model=model,
            loader=loader,
            context_pre=context_pre,
            context_post=context_post,
            target_names=resolved_targets,
        )
        sample_key_to_group: dict[tuple[int, int], str] = {}
        for group_name, specs in window_groups.items():
            for spec in specs:
                sample_key_to_group[(spec.node_id, spec.window_start)] = group_name
        grouped_samples: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            group_name = sample_key_to_group.get(
                (int(sample["node_id"]), int(sample.get("window_start", -1)))
            )
            if group_name is None:
                continue
            grouped_samples.setdefault(group_name, []).append(sample)
    else:
        samples = collect_forecast_samples_for_target_nodes(
            target_node_ids=all_selected_nodes,
            model=model,
            loader=loader,
            window=window,
            context_pre=context_pre,
            context_post=context_post,
            target_names=resolved_targets,
        )
        node_to_group: dict[int, str] = {}
        for group_name, nodes in (node_groups or {}).items():
            for node_id in nodes:
                node_to_group[node_id] = group_name

        grouped_samples = {}
        for sample in samples:
            node_id = sample["node_id"]
            if node_id in node_to_group:
                group_name = node_to_group[node_id]
                grouped_samples.setdefault(group_name, []).append(sample)

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

    latent_fig = (
        make_joint_latent_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=context_pre,
            context_post=context_post,
        )
        if include_sird_latents
        else None
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    latent_output_path: Path | None = None
    if latent_fig is not None and output_path is not None:
        latent_output_path = output_path.with_name(
            f"{output_path.stem}_sirhd_latents{output_path.suffix}"
        )
        latent_fig.savefig(latent_output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved SIRHD latent figure to: {latent_output_path}")

    separate_figures: dict[str, Any] = {}
    if output_path is not None:
        for target_name in resolved_targets:
            legacy_target_output = output_path.with_name(
                f"{output_path.stem}_{target_name}{output_path.suffix}"
            )
            if legacy_target_output.exists():
                legacy_target_output.unlink()
                logger.info(
                    f"[plot] Removed legacy target figure: {legacy_target_output}"
                )

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
            if latent_fig is not None:
                log_payload[f"{wandb_prefix}/sirhd_latents"] = wandb.Image(latent_fig)
            if log_payload:
                wandb.log(log_payload, step=0)

    return {
        "figure": fig,
        "joint_figure": fig,
        "latent_figure": latent_fig,
        "latent_output_path": latent_output_path,
        "separate_figures": separate_figures,
        "all_samples": samples,
        "selected_nodes": all_selected_nodes,
        "node_groups": node_groups or {},
        "window_groups": window_groups or {},
    }
