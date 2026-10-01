"""Evaluation loop implementation for EpiForecaster models.

This module provides the core evaluation loop that computes loss and metrics
over a DataLoader, matching the trainer's evaluation behavior.
"""

from __future__ import annotations

import csv as csv_lib
import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset

from data.epi_dataset import EpiDataset
from data.latent_supervision import latent_scales_from_checkpoint
from data.native_batch import NativeEpiBatch
from data.native_dataset import CONFIRMATION_WINDOW, NativeSupportDataset
from evaluation.aggregate_export import (
    NativeAggregateAccumulator,
    write_native_aggregate_outputs,
)
from evaluation.granular_export import (
    GRANULAR_SCHEMA_VERSION,
    GranularEvalWriter,
    infer_granular_eval_format,
    write_metadata_sidecar,
    write_granular_metadata_sidecar,
)
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.latent_diagnostics import build_latent_diagnostic_protocols
from evaluation.losses import (
    JointInferenceLoss,
    NativeJointInferenceLoss,
    _binary_target_mask,
    _count_once_slots,
    _log1p_per100k_targets,
    get_loss_from_config,
)
from evaluation.metrics import (
    ProbabilisticMetricResult,
    TorchDistributionMetricAccumulator,
    TorchMaskedMetricAccumulator,
    TorchWeightedIntervalScoreAccumulator,
)
from evaluation.pathway_metrics import (
    ObservationLatentMetricResult,
    PathwayMetricResult,
    TorchLatentCoherenceAccumulator,
    TorchLatentProtocolAccumulator,
    TorchObservationLatentCoherenceAccumulator,
    TorchPathwayMetricAccumulator,
    WindowSpearmanMetricResult,
)
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from models.observation_heads import DelayKernel, SheddingConvolution
from utils.device import (
    iter_device_ready_batches,
    prefetch_enabled,
    setup_device_streams,
)
from utils.log_keys import CORE_EVAL_METRICS, build_eval_metric_key, build_loss_key
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.training_utils import drop_nowcast

logger = logging.getLogger(__name__)

PER_HEAD_NODE_METRICS_SCHEMA_VERSION = "1"
PER_HEAD_NODE_METRICS_FIELDNAMES = [
    "target",
    "node_id",
    "region_id",
    "region_label",
    "population",
    "observed_count",
    "mae",
    "rmse",
]
_HEAD_CANONICAL_NAMES = {
    "hosp": "hospitalizations",
    "ww": "wastewater",
    "cases": "cases",
    "deaths": "deaths",
}
_DEFAULT_RELEVANT_SOURCE_MAPPING = {
    "hosp": "hospitalization_flow",
    "ww": "I",
    "cases": "I",
    "deaths": "death_flow",
}
_RELEVANT_SOURCE_OUTPUT_KEYS = {
    "hospitalization_flow": "hospitalization_flow",
    "H": "H_trajectory",
    "I": "I_trajectory",
    "death_flow": "death_flow",
    "D": "D_trajectory",
}
_ALLOWED_RELEVANT_SOURCES = {
    "hosp": {"hospitalization_flow", "H"},
    "ww": {"I"},
    "cases": {"I"},
    "deaths": {"death_flow", "D"},
}


def _resolve_relevant_source_mapping(
    relevant_source_mapping: dict[str, str] | None,
) -> dict[str, str]:
    if relevant_source_mapping is None:
        return dict(_DEFAULT_RELEVANT_SOURCE_MAPPING)

    expected_heads = set(_HEAD_CANONICAL_NAMES)
    actual_heads = set(relevant_source_mapping)
    if actual_heads != expected_heads:
        raise ValueError(
            "relevant_source_mapping must define exactly "
            f"{sorted(expected_heads)}, got {sorted(actual_heads)}"
        )

    resolved = dict(relevant_source_mapping)
    for head_name, source_name in resolved.items():
        allowed_sources = _ALLOWED_RELEVANT_SOURCES[head_name]
        if source_name not in allowed_sources:
            raise ValueError(
                f"Relevant source for {head_name!r} must be one of "
                f"{sorted(allowed_sources)}, got {source_name!r}"
            )
    return resolved


def _forecast_aligned_output(
    model_outputs: dict[str, torch.Tensor],
    key: str,
    horizon: int,
) -> torch.Tensor | None:
    value = model_outputs.get(key)
    if value is None:
        return None
    return drop_nowcast(value, horizon)


def _accumulate_per_head_node_metrics(
    *,
    accumulator: dict[str, dict[int, dict[str, float]]],
    target_name: str,
    target_nodes: torch.Tensor,
    diff: torch.Tensor,
    abs_diff: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    target_stats = accumulator.setdefault(target_name, {})
    target_node_ids = target_nodes.detach().cpu().tolist()
    per_sample_abs_sum = (abs_diff * weights).sum(dim=1).detach().cpu().tolist()
    per_sample_sq_sum = ((diff**2) * weights).sum(dim=1).detach().cpu().tolist()
    per_sample_count = (weights > 0).sum(dim=1).detach().cpu().tolist()
    for node_id, abs_sum, sq_sum, observed_count in zip(
        target_node_ids,
        per_sample_abs_sum,
        per_sample_sq_sum,
        per_sample_count,
        strict=False,
    ):
        count_value = int(observed_count)
        if count_value <= 0:
            continue
        node_stats = target_stats.setdefault(
            int(node_id),
            {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "observed_count": 0.0},
        )
        node_stats["abs_error_sum"] += float(abs_sum)
        node_stats["sq_error_sum"] += float(sq_sum)
        node_stats["observed_count"] += float(count_value)


def _write_per_head_node_metrics_csv(
    *,
    output_path: Path,
    per_head_node_metrics: dict[str, dict[int, dict[str, float]]],
    region_ids: list[str] | None,
    region_labels: list[str] | None,
    population_by_node: list[float] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_lib.DictWriter(f, fieldnames=PER_HEAD_NODE_METRICS_FIELDNAMES)
        writer.writeheader()
        for target in _HEAD_CANONICAL_NAMES.values():
            target_stats = per_head_node_metrics.get(target, {})
            for node_id in sorted(target_stats):
                stats = target_stats[node_id]
                observed_count = int(stats["observed_count"])
                if observed_count <= 0:
                    continue
                region_id = ""
                region_label = ""
                population = float("nan")
                if region_ids is not None and 0 <= node_id < len(region_ids):
                    region_id = str(region_ids[node_id])
                if region_labels is not None and 0 <= node_id < len(region_labels):
                    region_label = str(region_labels[node_id])
                if not region_label:
                    region_label = region_id
                if population_by_node is not None and 0 <= node_id < len(
                    population_by_node
                ):
                    population = float(population_by_node[node_id])

                writer.writerow(
                    {
                        "target": target,
                        "node_id": node_id,
                        "region_id": region_id,
                        "region_label": region_label,
                        "population": population,
                        "observed_count": observed_count,
                        "mae": stats["abs_error_sum"] / observed_count,
                        "rmse": math.sqrt(stats["sq_error_sum"] / observed_count),
                    }
                )


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""

    def _fmt(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"

    rows = [
        ("Loss", _fmt(loss)),
        ("Joint Obs Loss", _fmt(metrics.get("mae"))),
        ("Hosp MAE", _fmt(metrics.get("mae_hosp_log1p_per_100k"))),
        ("WW MAE", _fmt(metrics.get("mae_ww_log1p_per_100k"))),
    ]
    table = ["| Metric | Value |", "|---|---|"]
    for name, value in rows:
        table.append(f"| {name} | {value} |")
    return "\n".join(table)


def _add_probabilistic_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: ProbabilisticMetricResult,
) -> None:
    metrics[f"nll_{head_name}_log1p_per_100k"] = summary.nll
    metrics[f"crps_{head_name}_log1p_per_100k"] = summary.crps
    metrics[f"nll_{head_name}_per_h"] = summary.nll_per_h
    metrics[f"crps_{head_name}_per_h"] = summary.crps_per_h
    metrics[f"probabilistic_effective_count_{head_name}"] = summary.effective_count
    for level_name, coverage in summary.coverage_by_level.items():
        metrics[f"coverage_{level_name}_{head_name}"] = coverage
        metrics[f"interval_width_{level_name}_{head_name}_log1p_per_100k"] = (
            summary.interval_width_by_level[level_name]
        )
        metrics[f"coverage_{level_name}_{head_name}_per_h"] = (
            summary.coverage_per_h_by_level[level_name]
        )
        metrics[f"interval_width_{level_name}_{head_name}_per_h"] = (
            summary.interval_width_per_h_by_level[level_name]
        )


def _add_pathway_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: PathwayMetricResult,
    mechanistic_probabilistic_summary: ProbabilisticMetricResult,
) -> None:
    prefix = f"pathway_{head_name}"
    metrics[f"{prefix}_observed_count"] = summary.observed_count
    metrics[f"{prefix}_effective_count"] = summary.effective_count
    metrics[f"{prefix}_mean_abs_mechanistic_log1p_per_100k"] = (
        summary.mean_abs_mechanistic
    )
    metrics[f"{prefix}_mean_abs_residual_log1p_per_100k"] = summary.mean_abs_residual
    metrics[f"{prefix}_residual_abs_fraction"] = summary.residual_abs_fraction
    metrics[f"{prefix}_residual_to_mechanistic_abs_ratio"] = (
        summary.residual_to_mechanistic_abs_ratio
    )
    metrics[f"{prefix}_mechanistic_variance"] = summary.mechanistic_variance
    metrics[f"{prefix}_residual_variance"] = summary.residual_variance
    metrics[f"{prefix}_full_variance"] = summary.full_variance
    metrics[f"{prefix}_mechanistic_variance_share"] = summary.mechanistic_variance_share
    metrics[f"{prefix}_residual_variance_share"] = summary.residual_variance_share
    metrics[f"{prefix}_mechanistic_only_mae_log1p_per_100k"] = (
        summary.mechanistic_only_mae
    )
    metrics[f"{prefix}_mechanistic_only_rmse_log1p_per_100k"] = (
        summary.mechanistic_only_rmse
    )
    metrics[f"{prefix}_mechanistic_only_nll_log1p_per_100k"] = (
        mechanistic_probabilistic_summary.nll
    )
    metrics[f"{prefix}_mechanistic_only_crps_log1p_per_100k"] = (
        mechanistic_probabilistic_summary.crps
    )
    metrics[f"{prefix}_reconstruction_max_abs_error"] = (
        summary.reconstruction_max_abs_error
    )


def _add_window_spearman_metrics(
    *,
    metrics: dict[str, Any],
    prefix: str,
    summary: WindowSpearmanMetricResult,
) -> None:
    metrics[f"{prefix}_mean_window_spearman"] = summary.mean
    metrics[f"{prefix}_window_spearman_total_count"] = summary.total_count
    metrics[f"{prefix}_window_spearman_candidate_count"] = summary.candidate_count
    metrics[f"{prefix}_window_spearman_eligible_count"] = summary.eligible_count
    metrics[f"{prefix}_window_spearman_candidate_fraction"] = summary.candidate_fraction
    metrics[f"{prefix}_window_spearman_eligible_fraction"] = summary.eligible_fraction
    metrics[f"{prefix}_window_spearman_eligible_given_candidate_fraction"] = (
        summary.eligible_given_candidate_fraction
    )


def _add_observation_latent_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: ObservationLatentMetricResult,
) -> None:
    prefix = f"observation_latent_{head_name}"
    metrics[f"{prefix}_observed_count"] = summary.observed_count
    _add_window_spearman_metrics(
        metrics=metrics,
        prefix=f"{prefix}_full",
        summary=summary.full_window_spearman,
    )
    _add_window_spearman_metrics(
        metrics=metrics,
        prefix=f"{prefix}_mechanistic_only",
        summary=summary.mechanistic_only_window_spearman,
    )


def _add_delay_kernel_metrics(
    *,
    metrics: dict[str, Any],
    model: torch.nn.Module,
) -> None:
    """Add final kernel-state diagnostics for the observation heads.

    Run-level model-state metrics for ``test_summary.json``; the granular row
    schema is unaffected. Clinical heads route through ``DelayKernel`` and the
    wastewater head through ``SheddingConvolution``; both expose the same
    normalized-kernel and Gamma diagnostics.
    """
    model = getattr(model, "_orig_mod", model)
    kernels: dict[str, DelayKernel | SheddingConvolution] = {}
    for head_name in ("hosp", "cases", "deaths"):
        kernel = getattr(
            getattr(model, f"{head_name}_head", None), "delay_kernel", None
        )
        if isinstance(kernel, DelayKernel):
            kernels[head_name] = kernel
    ww_kernel = getattr(getattr(model, "ww_head", None), "shedding_conv", None)
    if isinstance(ww_kernel, SheddingConvolution):
        kernels["ww"] = ww_kernel
    for head_name, kernel in kernels.items():
        if kernel.parameterization not in {"simplex", "gamma"}:
            continue
        trainable_count = sum(
            parameter.numel()
            for parameter in kernel.parameters()
            if parameter.requires_grad
        )
        metrics[f"kernel_{head_name}_trainable_parameter_count"] = trainable_count
        metrics[f"kernel_{head_name}_mean_delay_days"] = kernel.get_mean_delay()
        metrics[f"kernel_{head_name}_delay_std_days"] = kernel.get_delay_std()
        if kernel.parameterization == "gamma":
            shape, scale = kernel.get_gamma_parameters()
            metrics[f"kernel_{head_name}_gamma_shape"] = shape
            metrics[f"kernel_{head_name}_gamma_scale"] = scale
            boundary_hits = kernel.get_gamma_boundary_hits()
            metrics[f"kernel_{head_name}_gamma_shape_lower_bound_hit"] = int(
                boundary_hits["shape_lower"]
            )
            metrics[f"kernel_{head_name}_gamma_shape_upper_bound_hit"] = int(
                boundary_hits["shape_upper"]
            )
            metrics[f"kernel_{head_name}_gamma_scale_lower_bound_hit"] = int(
                boundary_hits["scale_lower"]
            )
            metrics[f"kernel_{head_name}_gamma_scale_upper_bound_hit"] = int(
                boundary_hits["scale_upper"]
            )


def assert_native_confirmation_period_safe(
    *,
    queries: list[Any],
    date_grid: np.ndarray,
    confirmation_guard: bool = True,
) -> None:
    """Fail when a native evaluation target window touches confirmation data."""
    if not confirmation_guard:
        return
    dates = np.asarray(date_grid).astype("datetime64[D]")
    reserved_start, reserved_end = CONFIRMATION_WINDOW
    for query in queries:
        target_start_index = int(query.forecast_origin) + 1
        target_end_index = int(query.forecast_origin) + int(query.horizon_days)
        if target_start_index < 0 or target_end_index >= len(dates):
            raise ValueError(
                f"native query target window is outside the date grid: {query}"
            )
        target_start = dates[target_start_index]
        target_end = dates[target_end_index]
        if target_start <= reserved_end and target_end >= reserved_start:
            raise ValueError(
                "native evaluation target window intersects the confirmation "
                f"period {reserved_start}..{reserved_end}; set "
                "confirmation_guard=False only for explicit calibration work"
            )


def _native_dataset_from_loader(loader: DataLoader) -> NativeSupportDataset:
    dataset: Any = loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, NativeSupportDataset):
        raise TypeError(
            "native evaluation requires a NativeSupportDataset or Subset thereof"
        )
    return dataset


def _native_daily_eval_view(
    *,
    head: str,
    model_outputs: dict[str, Any],
    batch: NativeEpiBatch,
    criterion: NativeJointInferenceLoss,
    device: torch.device,
) -> dict[str, Any] | None:
    rows = batch.head_index.get(head)
    raw_target = batch.head_target.get(head)
    target_mask = batch.head_target_mask.get(head)
    population = batch.head_population.get(head)
    if rows is None or raw_target is None or target_mask is None or rows.numel() == 0:
        return None
    slots = _count_once_slots(batch, rows)
    rows = rows.index_select(0, slots).to(device=device, dtype=torch.long)
    raw_target = raw_target.index_select(0, slots).to(device=device).float()
    target_mask = target_mask.index_select(0, slots).to(device)
    if population is None:
        population = batch.source_pop.index_select(0, rows)
    else:
        population = population.index_select(0, slots)
    population = population.to(device=device).float()
    target_width = int(raw_target.shape[1])
    loc_tensor = model_outputs.get(f"dist_{head}_loc")
    if loc_tensor is None:
        loc_tensor = model_outputs[f"pred_{head}"]
    loc = criterion._align_daily_head_tensor(
        loc_tensor.to(device).index_select(0, rows),
        head_name=head,
        target_width=target_width,
    )
    scale_tensor = model_outputs.get(f"dist_{head}_scale")
    scale = (
        torch.ones_like(loc)
        if scale_tensor is None
        else criterion._align_daily_head_tensor(
            scale_tensor.to(device).index_select(0, rows),
            head_name=head,
            target_width=target_width,
        )
    )
    weights = _binary_target_mask(target_mask).float()
    target = _log1p_per100k_targets(raw_target, population, weights)
    raw_prediction = torch.expm1(loc).clamp_min(0.0) * (
        population.unsqueeze(1) / 100_000.0
    )
    expected_deaths = None
    if head == "deaths":
        expected_block = batch.head_extra.get("deaths", {}).get("expected_deaths")
        if expected_block is None:
            raise ValueError("ABS deaths evaluation requires expected_deaths")
        expected_deaths = expected_block.index_select(0, slots).to(device).float()
        if expected_deaths.shape != raw_target.shape:
            raise ValueError("ABS expected deaths shape differs from target")
    row_positions = rows.tolist()
    queries = [batch.queries[int(position)] for position in row_positions]
    target_indices = torch.tensor(
        [
            [
                int(query.forecast_origin) + horizon + 1
                for horizon in range(target_width)
            ]
            for query in queries
        ],
        device=device,
        dtype=torch.long,
    )
    return {
        "queries": queries,
        "loc": loc,
        "scale": scale,
        "target": target,
        "raw_target": raw_target,
        "raw_prediction": raw_prediction,
        "expected_deaths": expected_deaths,
        "population": population,
        "weights": weights,
        "target_indices": target_indices,
    }


def _native_hosp_eval_view(
    *,
    model_outputs: dict[str, Any],
    batch: NativeEpiBatch,
    criterion: NativeJointInferenceLoss,
    device: torch.device,
) -> dict[str, Any] | None:
    rows = batch.head_index.get("hosp")
    raw_target = batch.head_target.get("hosp")
    target_mask = batch.head_target_mask.get("hosp")
    population = batch.head_population.get("hosp")
    if (
        rows is None
        or raw_target is None
        or target_mask is None
        or population is None
        or rows.numel() == 0
    ):
        return None
    slots = _count_once_slots(batch, rows)
    row_positions_tensor = rows.index_select(0, slots)
    queries = [
        batch.queries[int(position)] for position in row_positions_tensor.tolist()
    ]
    raw_target = raw_target.index_select(0, slots).to(device=device).float()
    target_mask = target_mask.index_select(0, slots).to(device)
    population = population.index_select(0, slots).to(device=device).float()
    loc = model_outputs["dist_hosp_weekly_loc"].to(device).index_select(0, slots)
    scale = model_outputs["dist_hosp_weekly_scale"].to(device).index_select(0, slots)
    weekly_mask = model_outputs["hosp_weekly_mask"].to(device).index_select(0, slots)
    if raw_target.shape[1] < criterion.weekly_slots:
        pad = criterion.weekly_slots - int(raw_target.shape[1])
        raw_target = torch.cat(
            [raw_target, raw_target.new_zeros(raw_target.shape[0], pad)], dim=1
        )
        target_mask = torch.cat(
            [
                target_mask,
                torch.zeros(
                    target_mask.shape[0],
                    pad,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )
    weights = (
        _binary_target_mask(target_mask) & weekly_mask.to(dtype=torch.bool)
    ).float()
    target = _log1p_per100k_targets(raw_target, population, weights)
    raw_prediction = torch.expm1(loc).clamp_min(0.0) * (
        population.unsqueeze(1) / 100_000.0
    )
    target_indices: list[list[int]] = []
    origin_dows = batch.origin_dow.index_select(0, row_positions_tensor).tolist()
    for query, origin_dow in zip(queries, origin_dows, strict=True):
        origin = int(query.forecast_origin)
        first_monday = (-(int(origin_dow) + 1)) % 7
        target_indices.append(
            [
                origin + 1 + first_monday + 7 * week + 6
                for week in range(criterion.weekly_slots)
            ]
        )
    return {
        "queries": queries,
        "loc": loc,
        "scale": scale,
        "target": target,
        "raw_target": raw_target,
        "raw_prediction": raw_prediction,
        "population": population,
        "weights": weights,
        "target_indices": torch.tensor(target_indices, device=device, dtype=torch.long),
    }


def _native_wastewater_eval_view(
    *,
    model_outputs: dict[str, Any],
    batch: NativeEpiBatch,
    dataset: NativeSupportDataset,
    criterion: NativeJointInferenceLoss,
    device: torch.device,
) -> dict[str, Any] | None:
    """Flatten ``[EDAR, day, assay]`` into assay-specific evaluation rows."""
    rows = batch.head_index.get("ww")
    raw_target = batch.head_target.get("ww")
    target_mask = batch.head_target_mask.get("ww")
    extras = batch.head_extra.get("ww", {})
    detection_limit = extras.get("detection_limit")
    censor_code = extras.get("censor_code")
    population = batch.head_population.get("ww")
    if (
        rows is None
        or raw_target is None
        or target_mask is None
        or detection_limit is None
        or censor_code is None
        or rows.numel() == 0
    ):
        return None
    slots = _count_once_slots(batch, rows)
    row_positions = rows.index_select(0, slots)
    queries = [batch.queries[int(position)] for position in row_positions.tolist()]
    raw_target = raw_target.index_select(0, slots).to(device=device).float()
    target_mask = target_mask.index_select(0, slots).to(device)
    detection_limit = detection_limit.index_select(0, slots).to(device).float()
    censor_code = censor_code.index_select(0, slots).to(device)
    loc = model_outputs["dist_ww_loc"].to(device).index_select(0, slots)
    scale = model_outputs["dist_ww_scale"].to(device).index_select(0, slots)
    if population is None:
        population = batch.source_pop.index_select(0, row_positions)
    else:
        population = population.index_select(0, slots)
    population = population.to(device=device).float()

    assay_count = int(raw_target.shape[2])
    if assay_count != len(dataset.wastewater_assays):
        raise ValueError("wastewater assay coordinate does not match target block")

    def flatten_assays(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1).reshape(-1, tensor.shape[1])

    loc = flatten_assays(loc)
    scale = flatten_assays(scale)
    raw_target = flatten_assays(raw_target)
    target_mask = flatten_assays(target_mask)
    detection_limit = flatten_assays(detection_limit)
    censor_code = flatten_assays(censor_code)
    target = torch.log1p(raw_target.clamp_min(0.0))
    uncensored = _binary_target_mask(target_mask) & (censor_code == 0)
    censored = _binary_target_mask(target_mask) & (censor_code == 1)
    weights = uncensored.float()
    raw_prediction = torch.expm1(loc).clamp_min(0.0)
    expanded_queries = [query for query in queries for _ in dataset.wastewater_assays]
    source_ids = [
        f"{query.source_id}::{assay}"
        for query in queries
        for assay in dataset.wastewater_assays
    ]
    target_indices = torch.tensor(
        [
            [
                int(query.forecast_origin) + horizon + 1
                for horizon in range(criterion.horizon_days)
            ]
            for query in expanded_queries
        ],
        device=device,
        dtype=torch.long,
    )
    scores = criterion.left_censored_nll(
        loc,
        scale,
        torch.where(censored, detection_limit, torch.zeros_like(detection_limit)),
    )
    censored_nll = torch.where(censored, scores, torch.zeros_like(scores))
    return {
        "queries": expanded_queries,
        "source_ids": source_ids,
        "loc": loc,
        "scale": scale,
        "target": target,
        "raw_target": raw_target,
        "raw_prediction": raw_prediction,
        "population": population.repeat_interleave(assay_count),
        "weights": weights,
        "target_indices": target_indices,
        "censored_nll_sum": censored_nll.sum(),
        "censored_count": censored.sum(),
    }


@torch.no_grad()
def evaluate_native_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: NativeJointInferenceLoss,
    device: torch.device,
    split_name: str = "Eval",
    max_batches: int | None = None,
    granular_output_path: Path | None = None,
    granular_eval_format: str | None = None,
    aggregate_output_dir: Path | None = None,
    confirmation_guard: bool = True,
    granular_metadata: dict[str, Any] | None = None,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[float, dict[str, Any], Any]:
    """Evaluate routed native heads and write canonical EVL-1 artifacts."""
    dataset = _native_dataset_from_loader(loader)
    date_grid = dataset.date_grid
    head_horizons = {
        "ww": criterion.horizon_days,
        "cases": criterion.horizon_days,
        "hosp": criterion.weekly_slots,
        "deaths": criterion.horizon_days,
    }
    point_accumulators = {
        head: TorchMaskedMetricAccumulator(device=device, horizon=horizon)
        for head, horizon in head_horizons.items()
    }
    raw_accumulators = {
        head: TorchMaskedMetricAccumulator(device=device, horizon=horizon)
        for head, horizon in head_horizons.items()
    }
    distribution_accumulators = {
        head: TorchDistributionMetricAccumulator(
            device=device,
            horizon=horizon,
            distribution_family=criterion.distribution_family,
            distribution_df=criterion.distribution_df,
            scale_floor=criterion.distribution_scale_floor,
            interval_levels=criterion.interval_levels,
        )
        for head, horizon in head_horizons.items()
    }
    wis_accumulators = {
        head: TorchWeightedIntervalScoreAccumulator(
            device=device,
            horizon=horizon,
            interval_levels=criterion.interval_levels,
        )
        for head, horizon in head_horizons.items()
    }
    aggregate_accumulator = NativeAggregateAccumulator()
    excess_abs_error = 0.0
    excess_observed = 0
    excess_handle = None
    excess_writer = None
    if aggregate_output_dir is not None:
        aggregate_output_dir.mkdir(parents=True, exist_ok=True)
        excess_path = (
            aggregate_output_dir / f"{split_name.lower()}_native_excess_deaths.csv"
        )
        excess_handle = excess_path.open("w", newline="")
        excess_writer = csv_lib.writer(excess_handle)
        excess_writer.writerow(
            [
                "split",
                "abs_id",
                "origin_index",
                "horizon",
                "target_index",
                "target_date",
                "expected_all_cause",
                "observed_all_cause",
                "predicted_all_cause",
                "observed_excess",
                "predicted_excess",
            ]
        )
    granular_writer = (
        GranularEvalWriter(
            path=granular_output_path,
            split_name=split_name,
            output_format=granular_eval_format,
        )
        if granular_output_path is not None
        else None
    )
    model_was_training = model.training
    model.eval()
    loss_sum = 0.0
    processed_batches = 0
    censored_ww_nll_sum = 0.0
    censored_ww_count = 0
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            assert_native_confirmation_period_safe(
                queries=batch.queries,
                date_grid=date_grid,
                confirmation_guard=confirmation_guard,
            )
            batch = batch.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(batch)
                components = criterion(outputs, batch)
            loss_sum += float(components["total"].item())
            processed_batches += 1
            for head in ("ww", "cases", "hosp", "deaths"):
                view = (
                    _native_wastewater_eval_view(
                        model_outputs=outputs,
                        batch=batch,
                        dataset=dataset,
                        criterion=criterion,
                        device=device,
                    )
                    if head == "ww"
                    else _native_hosp_eval_view(
                        model_outputs=outputs,
                        batch=batch,
                        criterion=criterion,
                        device=device,
                    )
                    if head == "hosp"
                    else _native_daily_eval_view(
                        head=head,
                        model_outputs=outputs,
                        batch=batch,
                        criterion=criterion,
                        device=device,
                    )
                )
                if view is None:
                    continue
                if head == "deaths":
                    expected = view["expected_deaths"]
                    observed_excess = view["raw_target"] - expected
                    predicted_excess = view["raw_prediction"] - expected
                    active = view["weights"] > 0
                    excess_abs_error += float(
                        ((predicted_excess - observed_excess).abs() * active).sum()
                    )
                    excess_observed += int(active.sum())
                    if excess_writer is not None:
                        expected_cpu = expected.detach().cpu().numpy()
                        observed_cpu = view["raw_target"].detach().cpu().numpy()
                        predicted_cpu = view["raw_prediction"].detach().cpu().numpy()
                        active_cpu = active.detach().cpu().numpy()
                        indices_cpu = view["target_indices"].detach().cpu().numpy()
                        for row, query in enumerate(view["queries"]):
                            for day in range(active_cpu.shape[1]):
                                if not active_cpu[row, day]:
                                    continue
                                target_index = int(indices_cpu[row, day])
                                baseline = float(expected_cpu[row, day])
                                observed_count = float(observed_cpu[row, day])
                                predicted_count = float(predicted_cpu[row, day])
                                excess_writer.writerow(
                                    [
                                        split_name.lower(),
                                        str(query.source_id),
                                        int(query.forecast_origin),
                                        day + 1,
                                        target_index,
                                        str(date_grid[target_index]),
                                        baseline,
                                        observed_count,
                                        predicted_count,
                                        observed_count - baseline,
                                        predicted_count - baseline,
                                    ]
                                )
                if head == "ww":
                    censored_ww_nll_sum += float(view["censored_nll_sum"].item())
                    censored_ww_count += int(view["censored_count"].item())
                point_accumulators[head].update(
                    view["loc"], view["target"], None, view["weights"]
                )
                raw_accumulators[head].update(
                    view["raw_prediction"],
                    view["raw_target"],
                    None,
                    view["weights"],
                )
                prob_batch = distribution_accumulators[head].update(
                    loc=view["loc"],
                    scale=view["scale"],
                    targets=view["target"],
                    observed_mask=None,
                    sample_weights=view["weights"],
                )
                wis_batch = wis_accumulators[head].update(
                    median=view["loc"],
                    targets=view["target"],
                    intervals=prob_batch.intervals,
                    observed_mask=None,
                    sample_weights=view["weights"],
                )
                wis_components = {
                    "wis": wis_batch.wis,
                    "dispersion": wis_batch.dispersion,
                    "underprediction": wis_batch.underprediction,
                    "overprediction": wis_batch.overprediction,
                    "point": wis_batch.point,
                }
                source_types = [str(query.source_type) for query in view["queries"]]
                source_ids = view.get(
                    "source_ids",
                    [str(query.source_id) for query in view["queries"]],
                )
                aggregate_accumulator.update(
                    head=head,
                    source_types=source_types,
                    source_ids=source_ids,
                    predictions=view["loc"],
                    targets=view["target"],
                    raw_predictions=view["raw_prediction"],
                    raw_targets=view["raw_target"],
                    weights=view["weights"],
                    nll=prob_batch.nll,
                    crps=prob_batch.crps,
                    intervals=prob_batch.intervals,
                    wis_components=wis_components,
                )
                if granular_writer is not None:
                    granular_writer.write_native_rows(
                        target_name=head,
                        queries=view["queries"],
                        source_ids=source_ids,
                        node_ids=[
                            dataset.graph_node_index(query.source_type, query.source_id)
                            for query in view["queries"]
                        ],
                        predictions=view["loc"],
                        targets=view["target"],
                        raw_targets=view["raw_target"],
                        raw_predictions=view["raw_prediction"],
                        populations=view["population"],
                        weights=view["weights"],
                        target_indices=view["target_indices"],
                        temporal_coords=date_grid,
                        input_window_length=batch.t,
                        nll=prob_batch.nll,
                        crps=prob_batch.crps,
                        intervals=prob_batch.intervals,
                        wis_components=wis_components,
                    )
    finally:
        if excess_handle is not None:
            excess_handle.close()
        if granular_writer is not None:
            granular_writer.close()
        if model_was_training:
            model.train()
    if processed_batches == 0:
        raise ValueError("native evaluation processed no batches")

    metrics: dict[str, Any] = {}
    metrics["mae_excess_deaths_raw"] = (
        excess_abs_error / excess_observed if excess_observed else float("nan")
    )
    metrics["observed_excess_deaths"] = excess_observed
    for head in ("ww", "cases", "hosp", "deaths"):
        point = point_accumulators[head].finalize()
        raw = raw_accumulators[head].finalize()
        probabilistic = distribution_accumulators[head].finalize()
        wis = wis_accumulators[head].finalize()
        metrics.update(
            {
                f"mae_{head}_log1p_per_100k": point.mae,
                f"rmse_{head}_log1p_per_100k": point.rmse,
                f"raw_count_mae_{head}": raw.mae,
                f"nll_{head}": probabilistic.nll,
                f"crps_{head}": probabilistic.crps,
                f"wis_{head}": wis.wis,
                f"wis_dispersion_{head}": wis.dispersion,
                f"wis_underprediction_{head}": wis.underprediction,
                f"wis_overprediction_{head}": wis.overprediction,
                f"wis_point_{head}": wis.point,
                f"observed_count_{head}": wis.observed_count,
                f"wis_per_h_{head}": wis.wis_per_h,
            }
        )
        if head == "ww":
            metrics["mae_ww_log1p_concentration"] = point.mae
            metrics["rmse_ww_log1p_concentration"] = point.rmse
            metrics["raw_concentration_mae_ww"] = raw.mae
    metrics["censored_nll_ww"] = (
        censored_ww_nll_sum / censored_ww_count
        if censored_ww_count > 0
        else float("nan")
    )
    metrics["censored_count_ww"] = censored_ww_count
    active_wis = [
        float(metrics[f"wis_{head}"])
        for head in ("ww", "cases", "hosp", "deaths")
        if metrics[f"observed_count_{head}"] > 0
    ]
    if active_wis and not all(math.isfinite(value) for value in active_wis):
        raise ValueError("Observed native head has no finite WIS")
    metrics["joint_wis"] = float(np.mean(active_wis)) if active_wis else float("nan")
    metrics["joint_wis_head_count"] = len(active_wis)
    aggregate = aggregate_accumulator.finalize()
    if aggregate_output_dir is not None:
        write_native_aggregate_outputs(
            output_dir=aggregate_output_dir,
            split=split_name,
            aggregate=aggregate,
        )
    if granular_output_path is not None:
        metadata = {
            "split": split_name.strip().lower(),
            "native_support": True,
            "confirmation_guard": confirmation_guard,
            "observed_only": True,
            "metric_space": "target_specific",
            "metric_space_by_target": {
                "cases": "log1p_per_100k",
                "hospitalizations": "log1p_per_100k_week",
                "deaths": "log1p_per_100k",
                "wastewater": "log1p_concentration",
            },
            "raw_count_metrics": ["raw_count_mae"],
            "interval_levels": list(criterion.interval_levels),
        }
        if granular_metadata is not None:
            metadata.update(granular_metadata)
        write_granular_metadata_sidecar(
            granular_output_path,
            metadata,
        )
    return loss_sum / processed_batches, metrics, aggregate


def evaluate_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: JointInferenceLoss,
    horizon: int,
    device: torch.device,
    region_embeddings: torch.Tensor | None = None,
    split_name: str = "Eval",
    max_batches: int | None = None,
    node_metrics_csv_path: Path | None = None,
    per_head_node_metrics_csv_path: Path | None = None,
    granular_output_path: Path | None = None,
    granular_csv_path: Path | None = None,
    granular_eval_format: str | None = None,
    granular_metadata: dict[str, Any] | None = None,
    relevant_source_mapping: dict[str, str] | None = None,
    output_csv_path: Path | None = None,
    node_metrics_target: str = "hospitalizations",
    log_every: int = 10,
) -> tuple[float, dict[str, Any], dict[str, dict[int, float]]]:
    """Evaluate a loader and compute loss/metrics matching trainer behavior.

    Uses device-local metric accumulation to minimize CPU-GPU synchronization.

    Args:
        model: The EpiForecaster model to evaluate
        loader: DataLoader providing evaluation batches
        criterion: Loss function for computing evaluation loss
        horizon: Forecast horizon for dropping nowcast period
        device: Device for tensor operations
        region_embeddings: Optional pre-loaded region embeddings
        split_name: Name of the split for logging (e.g., "Val", "Test")
        max_batches: Optional limit on number of batches to evaluate
        node_metrics_csv_path: Optional path to save per-node MAE metrics as CSV
        per_head_node_metrics_csv_path: Optional path to save per-head node metrics CSV
        granular_output_path: Optional path to save granular per-example error rows
        granular_csv_path: Deprecated alias for granular_output_path
        granular_eval_format: Optional output format override ("parquet" or "csv")
        granular_metadata: Optional metadata to write alongside granular output
        relevant_source_mapping: Exact target-to-mechanistic-source mapping used
            for quantitative observation-latent coherence metrics and persisted
            qualitative plot overlays. This does not alter pathway attribution.
        output_csv_path: Deprecated alias for node_metrics_csv_path
        node_metrics_target: Canonical target name for the legacy node metrics CSV
        log_every: Progress logging cadence in batches

    Returns:
        Tuple of (mean_loss, metrics_dict, node_mae_dict) where:
        - mean_loss: Average loss per batch
        - metrics_dict: Dictionary of computed metrics
        - node_mae_dict: Dictionary mapping target name -> node_id -> average MAE
    """
    logger.info(f"{split_name} evaluation started...")
    if node_metrics_csv_path is None:
        node_metrics_csv_path = output_csv_path
    if granular_output_path is None:
        granular_output_path = granular_csv_path
    if node_metrics_target not in _HEAD_CANONICAL_NAMES.values():
        raise ValueError(
            f"Unknown node_metrics_target {node_metrics_target!r}; expected one of "
            f"{tuple(_HEAD_CANONICAL_NAMES.values())}."
        )
    resolved_relevant_sources = _resolve_relevant_source_mapping(
        relevant_source_mapping
    )

    # Device-local accumulators - avoid sync until end
    total_loss = torch.tensor(0.0, device=device)
    hosp_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    ww_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    cases_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    deaths_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    distribution_metric_kwargs = {
        "device": device,
        "horizon": horizon,
        "distribution_family": criterion.distribution_family,
        "distribution_df": criterion.distribution_df,
        "scale_floor": criterion.distribution_scale_floor,
        "interval_levels": criterion.interval_levels,
    }
    hosp_prob_metrics = TorchDistributionMetricAccumulator(**distribution_metric_kwargs)
    ww_prob_metrics = TorchDistributionMetricAccumulator(**distribution_metric_kwargs)
    cases_prob_metrics = TorchDistributionMetricAccumulator(
        **distribution_metric_kwargs
    )
    deaths_prob_metrics = TorchDistributionMetricAccumulator(
        **distribution_metric_kwargs
    )
    pathway_metrics = {
        head_name: TorchPathwayMetricAccumulator(device=device)
        for head_name in _HEAD_CANONICAL_NAMES
    }
    observation_latent_metrics = {
        head_name: TorchObservationLatentCoherenceAccumulator(device=device)
        for head_name in _HEAD_CANONICAL_NAMES
    }
    mechanistic_prob_metrics = {
        head_name: TorchDistributionMetricAccumulator(**distribution_metric_kwargs)
        for head_name in _HEAD_CANONICAL_NAMES
    }
    latent_metrics = TorchLatentCoherenceAccumulator(device=device)
    latent_diagnostic_metrics = {
        protocol: TorchLatentProtocolAccumulator(protocol=protocol, device=device)
        for protocol in ("oracle_x0", "teacher_forced_one_step")
    }
    loss_ww_sum = torch.tensor(0.0, device=device)
    loss_hosp_sum = torch.tensor(0.0, device=device)
    loss_cases_sum = torch.tensor(0.0, device=device)
    loss_deaths_sum = torch.tensor(0.0, device=device)
    loss_sird_supervision_sum = torch.tensor(0.0, device=device)
    loss_ww_weighted_sum = torch.tensor(0.0, device=device)
    loss_hosp_weighted_sum = torch.tensor(0.0, device=device)
    loss_cases_weighted_sum = torch.tensor(0.0, device=device)
    loss_deaths_weighted_sum = torch.tensor(0.0, device=device)
    loss_sird_supervision_weighted_sum = torch.tensor(0.0, device=device)
    loss_latent_s_sum = torch.tensor(0.0, device=device)
    loss_latent_i_sum = torch.tensor(0.0, device=device)
    loss_latent_h_sum = torch.tensor(0.0, device=device)
    loss_latent_r_sum = torch.tensor(0.0, device=device)
    loss_latent_death_flow_sum = torch.tensor(0.0, device=device)
    latent_loss_tasks = ("s", "i", "h", "r", "death_flow")
    loss_latent_raw_sums = {
        task: torch.tensor(0.0, device=device) for task in latent_loss_tasks
    }
    loss_latent_component_weighted_sums = {
        task: torch.tensor(0.0, device=device) for task in latent_loss_tasks
    }

    per_head_node_metrics: dict[str, dict[int, dict[str, float]]] = {}
    num_batches = len(loader)
    eval_iter = loader
    effective_log_every = max(1, log_every)
    dataset = getattr(loader, "dataset", None)
    temporal_coords = getattr(dataset, "_temporal_coords", None)
    region_ids = getattr(dataset, "_region_ids", None)
    region_labels = getattr(dataset, "_region_labels", None)
    resolved_granular_format = (
        infer_granular_eval_format(granular_output_path, granular_eval_format)
        if granular_output_path is not None
        else None
    )
    granular_writer = (
        GranularEvalWriter(
            path=granular_output_path,
            split_name=split_name,
            output_format=resolved_granular_format,
        )
        if granular_output_path is not None
        else None
    )

    model_was_training = model.training
    model.eval()
    forward_model = cast(EpiForecaster, model)
    dataset_config = getattr(dataset, "config", None)
    prefetch_factor = (
        dataset_config.training.prefetch_factor if dataset_config is not None else None
    )
    use_prefetch = prefetch_enabled(prefetch_factor)
    eval_iter = iter_device_ready_batches(
        loader,
        device=device,
        prefetch_factor=prefetch_factor,
        streams=setup_device_streams(device) if use_prefetch else None,
    )
    current_batch_idx: int | None = None
    current_stage = "init"
    processed_batches = 0
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            for batch_idx, batch_data in enumerate(eval_iter):
                if max_batches and batch_idx >= max_batches:
                    break
                current_batch_idx = batch_idx
                current_stage = "batch_fetched"
                should_log_batch = batch_idx % effective_log_every == 0
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                current_stage = "forward_batch"
                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                    mask_cases=criterion.mask_input_cases,
                    mask_ww=criterion.mask_input_ww,
                    mask_hosp=criterion.mask_input_hosp,
                    mask_deaths=criterion.mask_input_deaths,
                )
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                # Create sliced model outputs for metric computation
                sliced_model_outputs = {
                    k: drop_nowcast(v, horizon)
                    if k.startswith("pred_") and isinstance(v, torch.Tensor)
                    else v
                    for k, v in model_outputs.items()
                }

                # Compute loss components
                current_stage = "loss_components"
                components = criterion.compute_components(
                    model_outputs, targets_dict, batch_data
                )
                metric_supervision = criterion.compute_observation_supervision(
                    targets_dict,
                    device=device,
                    batch_data=batch_data,
                )
                loss = components["total"]
                total_loss += loss.detach()
                loss_ww_sum += components["ww"].detach()
                loss_hosp_sum += components["hosp"].detach()
                loss_cases_sum += components["cases"].detach()
                loss_deaths_sum += components["deaths"].detach()
                loss_sird_supervision_sum += components["sird_supervision"].detach()
                loss_latent_s_sum += components["latent_s"].detach()
                loss_latent_i_sum += components["latent_i"].detach()
                loss_latent_h_sum += components["latent_h"].detach()
                loss_latent_r_sum += components["latent_r"].detach()
                loss_latent_death_flow_sum += components["latent_death_flow"].detach()
                for task in latent_loss_tasks:
                    loss_latent_raw_sums[task] += components[
                        f"latent_{task}_raw"
                    ].detach()
                    loss_latent_component_weighted_sums[task] += components[
                        f"latent_{task}_weighted"
                    ].detach()
                loss_ww_weighted_sum += components["ww_weighted"].detach()
                loss_hosp_weighted_sum += components["hosp_weighted"].detach()
                loss_cases_weighted_sum += components["cases_weighted"].detach()
                loss_deaths_weighted_sum += components["deaths_weighted"].detach()
                loss_sird_supervision_weighted_sum += components[
                    "sird_supervision_weighted"
                ].detach()
                latent_metrics.update(
                    model_outputs=model_outputs,
                    targets=targets_dict,
                )
                for protocol, (
                    diagnostic_predictions,
                    diagnostic_targets,
                ) in build_latent_diagnostic_protocols(
                    model=forward_model,
                    model_outputs=model_outputs,
                    targets=targets_dict,
                ).items():
                    latent_diagnostic_metrics[protocol].update(
                        predictions=diagnostic_predictions,
                        targets=diagnostic_targets,
                    )
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                # Log sparsity-loss correlation during evaluation (moved from training)
                if batch_idx % 10 == 0:
                    log_sparsity_loss_correlation(
                        batch=batch_data,
                        model_outputs=model_outputs,
                        targets=targets_dict,
                        wandb_run=None,
                        step=batch_idx,
                        epoch=0,
                    )

                pred_hosp = sliced_model_outputs.get("pred_hosp")
                hosp_targets = metric_supervision["hosp"]["target"]
                hosp_mask = targets_dict.get("hosp_mask")
                hosp_weights = metric_supervision["hosp"]["weights"]
                if (
                    pred_hosp is not None
                    and hosp_targets is not None
                    and hosp_weights is not None
                ):
                    _unused_diff, abs_diff, weights = hosp_metrics.update(
                        predictions=pred_hosp,
                        targets=hosp_targets,
                        observed_mask=hosp_mask,
                        sample_weights=hosp_weights,
                    )
                    target_nodes = batch_data.target_node

                    _accumulate_per_head_node_metrics(
                        accumulator=per_head_node_metrics,
                        target_name=_HEAD_CANONICAL_NAMES["hosp"],
                        target_nodes=target_nodes,
                        diff=_unused_diff,
                        abs_diff=abs_diff,
                        weights=weights,
                    )
                    hosp_loc, hosp_scale = criterion._get_head_distribution_params(
                        model_outputs,
                        head_name="hosp",
                        horizon=int(hosp_targets.shape[1]),
                    )
                    hosp_prob_batch = hosp_prob_metrics.update(
                        loc=hosp_loc,
                        scale=hosp_scale,
                        targets=hosp_targets,
                        observed_mask=hosp_mask,
                        sample_weights=hosp_weights,
                    )
                    hosp_mechanistic = _forecast_aligned_output(
                        model_outputs,
                        "mechanistic_hosp",
                        int(hosp_targets.shape[1]),
                    )
                    hosp_residual = _forecast_aligned_output(
                        model_outputs,
                        "residual_hosp",
                        int(hosp_targets.shape[1]),
                    )
                    hosp_relevant_source_name = resolved_relevant_sources["hosp"]
                    hosp_relevant_source = _forecast_aligned_output(
                        model_outputs,
                        _RELEVANT_SOURCE_OUTPUT_KEYS[hosp_relevant_source_name],
                        int(hosp_targets.shape[1]),
                    )
                    observation_latent_metrics["hosp"].update(
                        full_prediction=pred_hosp,
                        relevant_latent=hosp_relevant_source,
                        weights=hosp_weights,
                        mechanistic_projection=hosp_mechanistic,
                    )
                    if hosp_mechanistic is not None and hosp_residual is not None:
                        pathway_metrics["hosp"].update(
                            full=hosp_loc,
                            mechanistic=hosp_mechanistic,
                            residual=hosp_residual,
                            targets=hosp_targets,
                            weights=hosp_weights,
                        )
                        mechanistic_prob_metrics["hosp"].update(
                            loc=hosp_mechanistic,
                            scale=hosp_scale,
                            targets=hosp_targets,
                            observed_mask=hosp_mask,
                            sample_weights=hosp_weights,
                        )

                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="hosp",
                            predictions=pred_hosp,
                            targets=hosp_targets,
                            weights=weights,
                            nll=hosp_prob_batch.nll,
                            crps=hosp_prob_batch.crps,
                            intervals=hosp_prob_batch.intervals,
                            mechanistic_predictions=hosp_mechanistic,
                            residual_contributions=hosp_residual,
                            relevant_latent_name=hosp_relevant_source_name,
                            relevant_latent_values=hosp_relevant_source,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                    del (
                        _unused_diff,
                        abs_diff,
                        weights,
                        target_nodes,
                        hosp_loc,
                        hosp_scale,
                        hosp_prob_batch,
                        hosp_mechanistic,
                        hosp_residual,
                        hosp_relevant_source,
                    )
                del pred_hosp, hosp_targets, hosp_mask, hosp_weights

                pred_ww = sliced_model_outputs.get("pred_ww")
                ww_targets = metric_supervision["ww"]["target"]
                ww_mask = targets_dict.get("ww_mask")
                ww_weights = metric_supervision["ww"]["weights"]
                if (
                    pred_ww is not None
                    and ww_targets is not None
                    and ww_weights is not None
                ):
                    ww_diff, ww_abs_diff, ww_effective_weights = ww_metrics.update(
                        predictions=pred_ww,
                        targets=ww_targets,
                        observed_mask=ww_mask,
                        sample_weights=ww_weights,
                    )
                    _accumulate_per_head_node_metrics(
                        accumulator=per_head_node_metrics,
                        target_name=_HEAD_CANONICAL_NAMES["ww"],
                        target_nodes=batch_data.target_node,
                        diff=ww_diff,
                        abs_diff=ww_abs_diff,
                        weights=ww_effective_weights,
                    )
                    ww_loc, ww_scale = criterion._get_head_distribution_params(
                        model_outputs,
                        head_name="ww",
                        horizon=int(ww_targets.shape[1]),
                    )
                    ww_prob_batch = ww_prob_metrics.update(
                        loc=ww_loc,
                        scale=ww_scale,
                        targets=ww_targets,
                        observed_mask=ww_mask,
                        sample_weights=ww_weights,
                    )
                    ww_mechanistic = _forecast_aligned_output(
                        model_outputs,
                        "mechanistic_ww",
                        int(ww_targets.shape[1]),
                    )
                    ww_residual = _forecast_aligned_output(
                        model_outputs,
                        "residual_ww",
                        int(ww_targets.shape[1]),
                    )
                    ww_relevant_source_name = resolved_relevant_sources["ww"]
                    ww_relevant_source = _forecast_aligned_output(
                        model_outputs,
                        _RELEVANT_SOURCE_OUTPUT_KEYS[ww_relevant_source_name],
                        int(ww_targets.shape[1]),
                    )
                    observation_latent_metrics["ww"].update(
                        full_prediction=pred_ww,
                        relevant_latent=ww_relevant_source,
                        weights=ww_weights,
                        mechanistic_projection=ww_mechanistic,
                    )
                    if ww_mechanistic is not None and ww_residual is not None:
                        pathway_metrics["ww"].update(
                            full=ww_loc,
                            mechanistic=ww_mechanistic,
                            residual=ww_residual,
                            targets=ww_targets,
                            weights=ww_weights,
                        )
                        mechanistic_prob_metrics["ww"].update(
                            loc=ww_mechanistic,
                            scale=ww_scale,
                            targets=ww_targets,
                            observed_mask=ww_mask,
                            sample_weights=ww_weights,
                        )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="ww",
                            predictions=pred_ww,
                            targets=ww_targets,
                            weights=ww_effective_weights,
                            nll=ww_prob_batch.nll,
                            crps=ww_prob_batch.crps,
                            intervals=ww_prob_batch.intervals,
                            mechanistic_predictions=ww_mechanistic,
                            residual_contributions=ww_residual,
                            relevant_latent_name=ww_relevant_source_name,
                            relevant_latent_values=ww_relevant_source,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                    del (
                        ww_diff,
                        ww_abs_diff,
                        ww_effective_weights,
                        ww_loc,
                        ww_scale,
                        ww_prob_batch,
                        ww_mechanistic,
                        ww_residual,
                        ww_relevant_source,
                    )
                del pred_ww, ww_targets, ww_mask, ww_weights

                pred_cases = sliced_model_outputs.get("pred_cases")
                cases_targets = metric_supervision["cases"]["target"]
                cases_mask = targets_dict.get("cases_mask")
                cases_weights = metric_supervision["cases"]["weights"]
                if (
                    pred_cases is not None
                    and cases_targets is not None
                    and cases_weights is not None
                ):
                    cases_diff, cases_abs_diff, cases_effective_weights = (
                        cases_metrics.update(
                            predictions=pred_cases,
                            targets=cases_targets,
                            observed_mask=cases_mask,
                            sample_weights=cases_weights,
                        )
                    )
                    _accumulate_per_head_node_metrics(
                        accumulator=per_head_node_metrics,
                        target_name=_HEAD_CANONICAL_NAMES["cases"],
                        target_nodes=batch_data.target_node,
                        diff=cases_diff,
                        abs_diff=cases_abs_diff,
                        weights=cases_effective_weights,
                    )
                    cases_loc, cases_scale = criterion._get_head_distribution_params(
                        model_outputs,
                        head_name="cases",
                        horizon=int(cases_targets.shape[1]),
                    )
                    cases_prob_batch = cases_prob_metrics.update(
                        loc=cases_loc,
                        scale=cases_scale,
                        targets=cases_targets,
                        observed_mask=cases_mask,
                        sample_weights=cases_weights,
                    )
                    cases_mechanistic = _forecast_aligned_output(
                        model_outputs,
                        "mechanistic_cases",
                        int(cases_targets.shape[1]),
                    )
                    cases_residual = _forecast_aligned_output(
                        model_outputs,
                        "residual_cases",
                        int(cases_targets.shape[1]),
                    )
                    cases_relevant_source_name = resolved_relevant_sources["cases"]
                    cases_relevant_source = _forecast_aligned_output(
                        model_outputs,
                        _RELEVANT_SOURCE_OUTPUT_KEYS[cases_relevant_source_name],
                        int(cases_targets.shape[1]),
                    )
                    observation_latent_metrics["cases"].update(
                        full_prediction=pred_cases,
                        relevant_latent=cases_relevant_source,
                        weights=cases_weights,
                        mechanistic_projection=cases_mechanistic,
                    )
                    if cases_mechanistic is not None and cases_residual is not None:
                        pathway_metrics["cases"].update(
                            full=cases_loc,
                            mechanistic=cases_mechanistic,
                            residual=cases_residual,
                            targets=cases_targets,
                            weights=cases_weights,
                        )
                        mechanistic_prob_metrics["cases"].update(
                            loc=cases_mechanistic,
                            scale=cases_scale,
                            targets=cases_targets,
                            observed_mask=cases_mask,
                            sample_weights=cases_weights,
                        )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="cases",
                            predictions=pred_cases,
                            targets=cases_targets,
                            weights=cases_effective_weights,
                            nll=cases_prob_batch.nll,
                            crps=cases_prob_batch.crps,
                            intervals=cases_prob_batch.intervals,
                            mechanistic_predictions=cases_mechanistic,
                            residual_contributions=cases_residual,
                            relevant_latent_name=cases_relevant_source_name,
                            relevant_latent_values=cases_relevant_source,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                    del (
                        cases_diff,
                        cases_abs_diff,
                        cases_effective_weights,
                        cases_loc,
                        cases_scale,
                        cases_prob_batch,
                        cases_mechanistic,
                        cases_residual,
                        cases_relevant_source,
                    )
                del pred_cases, cases_targets, cases_mask, cases_weights

                pred_deaths = sliced_model_outputs.get("pred_deaths")
                deaths_targets = metric_supervision["deaths"]["target"]
                deaths_mask = targets_dict.get("deaths_mask")
                deaths_weights = metric_supervision["deaths"]["weights"]
                if (
                    pred_deaths is not None
                    and deaths_targets is not None
                    and deaths_weights is not None
                ):
                    deaths_diff, deaths_abs_diff, deaths_effective_weights = (
                        deaths_metrics.update(
                            predictions=pred_deaths,
                            targets=deaths_targets,
                            observed_mask=deaths_mask,
                            sample_weights=deaths_weights,
                        )
                    )
                    _accumulate_per_head_node_metrics(
                        accumulator=per_head_node_metrics,
                        target_name=_HEAD_CANONICAL_NAMES["deaths"],
                        target_nodes=batch_data.target_node,
                        diff=deaths_diff,
                        abs_diff=deaths_abs_diff,
                        weights=deaths_effective_weights,
                    )
                    deaths_loc, deaths_scale = criterion._get_head_distribution_params(
                        model_outputs,
                        head_name="deaths",
                        horizon=int(deaths_targets.shape[1]),
                    )
                    deaths_prob_batch = deaths_prob_metrics.update(
                        loc=deaths_loc,
                        scale=deaths_scale,
                        targets=deaths_targets,
                        observed_mask=deaths_mask,
                        sample_weights=deaths_weights,
                    )
                    deaths_mechanistic = _forecast_aligned_output(
                        model_outputs,
                        "mechanistic_deaths",
                        int(deaths_targets.shape[1]),
                    )
                    deaths_residual = _forecast_aligned_output(
                        model_outputs,
                        "residual_deaths",
                        int(deaths_targets.shape[1]),
                    )
                    deaths_relevant_source_name = resolved_relevant_sources["deaths"]
                    deaths_relevant_source = _forecast_aligned_output(
                        model_outputs,
                        _RELEVANT_SOURCE_OUTPUT_KEYS[deaths_relevant_source_name],
                        int(deaths_targets.shape[1]),
                    )
                    observation_latent_metrics["deaths"].update(
                        full_prediction=pred_deaths,
                        relevant_latent=deaths_relevant_source,
                        weights=deaths_weights,
                        mechanistic_projection=deaths_mechanistic,
                    )
                    if deaths_mechanistic is not None and deaths_residual is not None:
                        pathway_metrics["deaths"].update(
                            full=deaths_loc,
                            mechanistic=deaths_mechanistic,
                            residual=deaths_residual,
                            targets=deaths_targets,
                            weights=deaths_weights,
                        )
                        mechanistic_prob_metrics["deaths"].update(
                            loc=deaths_mechanistic,
                            scale=deaths_scale,
                            targets=deaths_targets,
                            observed_mask=deaths_mask,
                            sample_weights=deaths_weights,
                        )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="deaths",
                            predictions=pred_deaths,
                            targets=deaths_targets,
                            weights=deaths_effective_weights,
                            nll=deaths_prob_batch.nll,
                            crps=deaths_prob_batch.crps,
                            intervals=deaths_prob_batch.intervals,
                            mechanistic_predictions=deaths_mechanistic,
                            residual_contributions=deaths_residual,
                            relevant_latent_name=deaths_relevant_source_name,
                            relevant_latent_values=deaths_relevant_source,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                    del (
                        deaths_diff,
                        deaths_abs_diff,
                        deaths_effective_weights,
                        deaths_loc,
                        deaths_scale,
                        deaths_prob_batch,
                        deaths_mechanistic,
                        deaths_residual,
                        deaths_relevant_source,
                    )
                del pred_deaths, deaths_targets, deaths_mask, deaths_weights
                current_stage = "granular_write_complete"
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )
                processed_batches += 1
                del components, metric_supervision, model_outputs, targets_dict
                del sliced_model_outputs, batch_data, loss

    except Exception:
        logger.error(
            "[eval] Evaluation failed: split=%s batch=%s stage=%s",
            split_name,
            current_batch_idx if current_batch_idx is not None else "n/a",
            current_stage,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    finally:
        if granular_writer is not None:
            granular_writer.close()
        if model_was_training:
            model.train()

    mean_loss = (total_loss / max(1, processed_batches)).item()
    hosp_summary = hosp_metrics.finalize()
    ww_summary = ww_metrics.finalize()
    cases_summary = cases_metrics.finalize()
    deaths_summary = deaths_metrics.finalize()
    hosp_prob_summary = hosp_prob_metrics.finalize()
    ww_prob_summary = ww_prob_metrics.finalize()
    cases_prob_summary = cases_prob_metrics.finalize()
    deaths_prob_summary = deaths_prob_metrics.finalize()
    pathway_summaries = {
        head_name: accumulator.finalize()
        for head_name, accumulator in pathway_metrics.items()
    }
    mechanistic_prob_summaries = {
        head_name: accumulator.finalize()
        for head_name, accumulator in mechanistic_prob_metrics.items()
    }
    observation_latent_summaries = {
        head_name: accumulator.finalize()
        for head_name, accumulator in observation_latent_metrics.items()
    }
    latent_summary = latent_metrics.finalize()
    latent_diagnostic_summary = {
        key: value
        for accumulator in latent_diagnostic_metrics.values()
        for key, value in accumulator.finalize().items()
    }

    node_mae_by_target: dict[str, dict[int, float]] = {}
    node_mae_count_by_target: dict[str, dict[int, int]] = {}
    for target_name, target_stats in per_head_node_metrics.items():
        node_mae_by_target[target_name] = {}
        node_mae_count_by_target[target_name] = {}
        for node_id, stats in target_stats.items():
            observed_count = int(stats["observed_count"])
            if observed_count <= 0:
                continue
            node_mae_by_target[target_name][node_id] = (
                stats["abs_error_sum"] / observed_count
            )
            node_mae_count_by_target[target_name][node_id] = observed_count

    if node_metrics_csv_path is not None:
        node_metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(node_metrics_csv_path, "w", newline="") as f:
            writer = csv_lib.writer(f)
            writer.writerow(["target", "node_id", "mae", "num_samples"])
            selected_node_mae = node_mae_by_target.get(node_metrics_target, {})
            selected_node_counts = node_mae_count_by_target.get(node_metrics_target, {})
            for node_id in sorted(selected_node_mae.keys()):
                writer.writerow(
                    [
                        node_metrics_target,
                        node_id,
                        selected_node_mae[node_id],
                        selected_node_counts[node_id],
                    ]
                )

    population_by_node: list[float] | None = None
    if dataset is not None and hasattr(dataset, "node_static_covariates"):
        population_tensor = dataset.node_static_covariates.get("Pop")
        if population_tensor is not None:
            population_by_node = population_tensor.detach().cpu().tolist()

    if per_head_node_metrics_csv_path is not None:
        _write_per_head_node_metrics_csv(
            output_path=per_head_node_metrics_csv_path,
            per_head_node_metrics=per_head_node_metrics,
            region_ids=region_ids,
            region_labels=region_labels,
            population_by_node=population_by_node,
        )
        per_head_metadata = {
            "dataset_path": getattr(dataset, "aligned_data_path", None),
            "run_id": getattr(dataset, "_run_id_value", None)
            if dataset is not None
            else None,
            "split": split_name.lower(),
        }
        if granular_metadata is not None:
            per_head_metadata.update(granular_metadata)
        per_head_metadata["relevant_source_mapping"] = resolved_relevant_sources
        write_metadata_sidecar(
            per_head_node_metrics_csv_path,
            per_head_metadata,
            schema_version=PER_HEAD_NODE_METRICS_SCHEMA_VERSION,
        )

    if granular_output_path is not None:
        metadata = {
            "dataset_path": getattr(dataset, "aligned_data_path", None),
            "granular_csv_path": granular_output_path,
            "granular_eval_path": granular_output_path,
            "granular_eval_format": resolved_granular_format,
            "observed_only": True,
            "region_name_source": getattr(dataset, "_region_name_source", None),
            "run_id": getattr(dataset, "run_id", None),
            "schema_version": GRANULAR_SCHEMA_VERSION,
            "split": split_name.lower(),
            "training_seed": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "seed", None),
            "node_split_strategy": getattr(
                getattr(dataset, "config", None), "training", None
            )
            and getattr(dataset.config.training, "node_split_strategy", None),
            "node_split_population_bins": getattr(
                getattr(dataset, "config", None), "training", None
            )
            and getattr(dataset.config.training, "node_split_population_bins", None),
            "val_split": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "val_split", None),
            "test_split": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "test_split", None),
        }
        if granular_metadata is not None:
            metadata.update(granular_metadata)
        metadata["relevant_source_mapping"] = resolved_relevant_sources
        write_granular_metadata_sidecar(
            granular_output_path,
            metadata,
        )

    joint_obs_loss_total = float(
        (
            loss_ww_weighted_sum
            + loss_hosp_weighted_sum
            + loss_cases_weighted_sum
            + loss_deaths_weighted_sum
        )
        / max(1, processed_batches)
    )
    metrics = {
        # Generic primary metric: aliased joint observation loss
        "mae": joint_obs_loss_total,
        "joint_obs_loss_total": joint_obs_loss_total,
        # Explicit hospitalization metrics
        "mae_hosp_per_h": hosp_summary.mae_per_h,
        "rmse_hosp_per_h": hosp_summary.rmse_per_h,
        # Hospitalization metrics in log1p(per-100k) space
        "mae_hosp_log1p_per_100k": hosp_summary.mae,
        "rmse_hosp_log1p_per_100k": hosp_summary.rmse,
        "r2_hosp_log1p_per_100k": hosp_summary.r2,
        "observed_count_hosp": hosp_summary.observed_count,
        "effective_count_hosp": hosp_summary.effective_count,
        # Wastewater metrics in log1p(per-100k) space
        "mae_ww_log1p_per_100k": ww_summary.mae,
        "rmse_ww_log1p_per_100k": ww_summary.rmse,
        "r2_ww_log1p_per_100k": ww_summary.r2,
        "observed_count_ww": ww_summary.observed_count,
        "effective_count_ww": ww_summary.effective_count,
        # Cases metrics in log1p(per-100k) space
        "mae_cases_log1p_per_100k": cases_summary.mae,
        "rmse_cases_log1p_per_100k": cases_summary.rmse,
        "r2_cases_log1p_per_100k": cases_summary.r2,
        "observed_count_cases": cases_summary.observed_count,
        "effective_count_cases": cases_summary.effective_count,
        # Deaths metrics in log1p(per-100k) space
        "mae_deaths_log1p_per_100k": deaths_summary.mae,
        "rmse_deaths_log1p_per_100k": deaths_summary.rmse,
        "r2_deaths_log1p_per_100k": deaths_summary.r2,
        "observed_count_deaths": deaths_summary.observed_count,
        "effective_count_deaths": deaths_summary.effective_count,
        # Joint loss components (averaged per batch, same reduction as mean_loss)
        "loss_ww": (loss_ww_sum / max(1, processed_batches)).item(),
        "loss_hosp": (loss_hosp_sum / max(1, processed_batches)).item(),
        "loss_cases": (loss_cases_sum / max(1, processed_batches)).item(),
        "loss_deaths": (loss_deaths_sum / max(1, processed_batches)).item(),
        "loss_sird_supervision": (
            loss_sird_supervision_sum / max(1, processed_batches)
        ).item(),
        "loss_latent_s": (loss_latent_s_sum / max(1, processed_batches)).item(),
        "loss_latent_i": (loss_latent_i_sum / max(1, processed_batches)).item(),
        "loss_latent_h": (loss_latent_h_sum / max(1, processed_batches)).item(),
        "loss_latent_r": (loss_latent_r_sum / max(1, processed_batches)).item(),
        "loss_latent_death_flow": (
            loss_latent_death_flow_sum / max(1, processed_batches)
        ).item(),
        "loss_ww_weighted": (loss_ww_weighted_sum / max(1, processed_batches)).item(),
        "loss_hosp_weighted": (
            loss_hosp_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_cases_weighted": (
            loss_cases_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_deaths_weighted": (
            loss_deaths_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_sird_supervision_weighted": (
            loss_sird_supervision_weighted_sum / max(1, processed_batches)
        ).item(),
    }
    for task in latent_loss_tasks:
        metrics[f"loss_latent_{task}_raw"] = (
            loss_latent_raw_sums[task] / max(1, processed_batches)
        ).item()
        metrics[f"loss_latent_{task}_component_weighted"] = (
            loss_latent_component_weighted_sums[task] / max(1, processed_batches)
        ).item()
        metrics[f"loss_latent_{task}_total_weighted"] = (
            criterion.latent_total_weight
            * loss_latent_component_weighted_sums[task]
            / max(1, processed_batches)
        ).item()
    _add_probabilistic_metrics(
        metrics=metrics,
        head_name="hosp",
        summary=hosp_prob_summary,
    )
    _add_probabilistic_metrics(
        metrics=metrics,
        head_name="ww",
        summary=ww_prob_summary,
    )
    _add_probabilistic_metrics(
        metrics=metrics,
        head_name="cases",
        summary=cases_prob_summary,
    )
    _add_probabilistic_metrics(
        metrics=metrics,
        head_name="deaths",
        summary=deaths_prob_summary,
    )
    for head_name in _HEAD_CANONICAL_NAMES:
        _add_pathway_metrics(
            metrics=metrics,
            head_name=head_name,
            summary=pathway_summaries[head_name],
            mechanistic_probabilistic_summary=mechanistic_prob_summaries[head_name],
        )
    for head_name in _HEAD_CANONICAL_NAMES:
        _add_observation_latent_metrics(
            metrics=metrics,
            head_name=head_name,
            summary=observation_latent_summaries[head_name],
        )
    metrics.update(latent_summary)
    metrics.update(latent_diagnostic_summary)
    _add_delay_kernel_metrics(metrics=metrics, model=model)

    logger.info("EVAL COMPLETE")
    return mean_loss, metrics, node_mae_by_target


def _ensure_wandb_run(
    *,
    config: EpiForecasterConfig | None,
    log_dir: Path | None,
    name: str,
    job_type: str,
) -> Any:
    """Ensure a W&B run exists, creating one if needed."""
    if wandb.run is not None:
        return wandb.run
    if log_dir is None:
        return None
    project = config.output.wandb_project if config is not None else "epiforecaster"
    entity = config.output.wandb_entity if config is not None else None
    group = None
    mode: Any = "online"
    if config is not None:
        group = config.output.wandb_group or config.output.experiment_name
        mode = config.output.wandb_mode
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        dir=str(log_dir),
        config=config.to_dict() if config is not None else None,
        job_type=job_type,
        mode=mode,
    )


def eval_checkpoint(
    *,
    checkpoint_path: Path,
    split: str = "val",
    device: str = "auto",
    log_dir: Path | None = None,
    overrides: list[str] | None = None,
    node_metrics_csv_path: Path | None = None,
    per_head_node_metrics_csv_path: Path | None = None,
    granular_output_path: Path | None = None,
    granular_csv_path: Path | None = None,
    granular_eval_format: str | None = None,
    output_csv_path: Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    node_metrics_target: str = "hospitalizations",
    log_every: int = 10,
) -> dict[str, Any]:
    """
    Evaluate checkpoint - pure evaluation, no selection or plotting.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Which split to evaluate ("val", "test", or "full")
        device: Device to use for evaluation (overridden by training.device in overrides)
        log_dir: Optional W&B run directory for forecast plots
        overrides: Optional list of dotted-key config overrides (e.g., ["training.val_workers=4"])
        node_metrics_csv_path: Optional path to save node-level metrics CSV
        per_head_node_metrics_csv_path: Optional path to save per-head node metrics CSV
        granular_output_path: Optional path to save granular per-example error rows
        granular_csv_path: Deprecated alias for granular_output_path
        granular_eval_format: Optional output format override ("parquet" or "csv")
        output_csv_path: Deprecated alias for node_metrics_csv_path
        batch_size: Optional batch size override
        num_workers: Optional DataLoader worker override
        pin_memory: Optional pin-memory override
        prefetch_factor: Optional prefetch override. Use 0 or None to disable.
        node_metrics_target: Canonical target name for node-level CSV export
        log_every: Progress logging cadence in batches

    Returns:
        Dict with: checkpoint, config, model, loader, node_mae_dict,
                   eval_loss, eval_metrics
    """
    # Extract training.device from overrides if present
    if node_metrics_csv_path is None:
        node_metrics_csv_path = output_csv_path
    explicit_granular_output_path = (
        granular_output_path is not None or granular_csv_path is not None
    )
    if granular_output_path is None:
        granular_output_path = granular_csv_path

    resolved_device = device
    if overrides:
        for ov in overrides:
            if ov.startswith("training.device="):
                resolved_device = ov.split("=", 1)[1]
                break

    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        overrides=list(overrides) if overrides else None,
    )
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        "[eval] Building %s loader from dataset=%s resolved_device=%s batch_size=%s "
        "num_workers=%s prefetch_factor=%s pin_memory=%s",
        split,
        config.data.dataset_path,
        resolved_device,
        batch_size if batch_size is not None else config.training.batch_size,
        num_workers if num_workers is not None else "config",
        (
            prefetch_factor
            if prefetch_factor is not None
            else config.training.prefetch_factor
        ),
        pin_memory if pin_memory is not None else config.training.pin_memory,
    )
    loader, region_embeddings = build_loader_from_config(
        config,
        split=split,
        device=resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(
        "[eval] Environment: split=%s dataset=%s samples=%d batch_size=%s num_workers=%d "
        "prefetch_factor=%s pin_memory=%s device=%s",
        split,
        config.data.dataset_path,
        len(dataset),
        loader.batch_size,
        loader.num_workers,
        loader.prefetch_factor,
        loader.pin_memory,
        next(model.parameters()).device,
    )

    resolved_granular_format = granular_eval_format
    if granular_output_path is None and config.output.write_granular_eval:
        granular_output_path = (
            checkpoint_path.parent.parent
            / config.output.resolve_granular_eval_filename(split=split)
        )
    if resolved_granular_format is None and not explicit_granular_output_path:
        resolved_granular_format = getattr(config.output, "granular_eval_format", None)

    criterion = get_loss_from_config(
        config.training.loss,
        latent_scales=latent_scales_from_checkpoint(checkpoint),
        data_config=config.data,
        observation_heads_config=config.model.observation_heads,
        input_window_length=config.model.input_window_length,
        forecast_horizon=config.model.forecast_horizon,
    )
    try:
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            node_metrics_csv_path=node_metrics_csv_path,
            per_head_node_metrics_csv_path=per_head_node_metrics_csv_path,
            granular_output_path=granular_output_path,
            granular_eval_format=resolved_granular_format,
            node_metrics_target=node_metrics_target,
            log_every=log_every,
            granular_metadata={
                "batch_size": batch_size
                if batch_size is not None
                else loader.batch_size,
                "checkpoint_path": checkpoint_path,
                "config_path": checkpoint_path.parent.parent / "config.yaml",
                "forecast_horizon": int(config.model.forecast_horizon),
                "input_window_length": int(config.model.input_window_length),
                "max_batches": None,
                "model_experiment_name": config.output.experiment_name,
                "node_split_population_bins": int(
                    config.training.node_split_population_bins
                ),
                "node_split_strategy": config.training.node_split_strategy,
                "node_metrics_target": node_metrics_target,
                "overrides": list(overrides) if overrides else [],
                "run_dir": checkpoint_path.parent.parent,
                "split": split.lower(),
                "test_split": float(config.training.test_split),
                "training_seed": config.training.seed,
                "val_split": float(config.training.val_split),
                "window_stride": int(config.data.window_stride),
                "eval_log_every": int(log_every),
                "eval_num_workers": int(loader.num_workers),
                "eval_prefetch_factor": loader.prefetch_factor,
                "eval_pin_memory": bool(loader.pin_memory),
            },
            relevant_source_mapping=(
                config.model.observation_heads.relevant_source_mapping()
            ),
        )
    except Exception:
        logger.exception("[eval] Metrics evaluation failed")
        raise

    if log_dir is not None or wandb.run is not None:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"eval_{split}_{checkpoint_path.parent.parent.name}"
        _ensure_wandb_run(
            config=config, log_dir=log_dir, name=run_name, job_type="eval"
        )
        if wandb.run is not None:
            log_data: dict[str, Any] = {}
            if math.isfinite(eval_loss):
                log_data[build_loss_key(split=split)] = eval_loss
            for key in CORE_EVAL_METRICS:
                if key in eval_metrics:
                    log_data[build_eval_metric_key(key, split)] = eval_metrics[key]
            if log_data:
                wandb.log(log_data, step=0)

    return {
        "checkpoint": checkpoint,
        "config": config,
        "model": model,
        "loader": loader,
        "node_mae": node_mae_dict,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
    }
