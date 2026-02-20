from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.epi_dataset import EpiDataset
from evaluation.baseline_models import (
    BaselinePredictionResult,
    predict_with_exponential_smoothing_fallback,
    predict_with_tiered_fallback,
    predict_with_var_cross_target_fallback,
)
from evaluation.epiforecaster_eval import build_loader_from_config
from evaluation.metrics import compute_masked_metrics_numpy
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)

_TARGET_SPECS = {
    "wastewater": ("precomputed_ww", "precomputed_ww_mask"),
    "hospitalizations": ("precomputed_hosp", "precomputed_hosp_mask"),
    "cases": ("precomputed_cases_target", "precomputed_cases_mask"),
    "deaths": ("precomputed_deaths", "precomputed_deaths_mask"),
}
_SUPPORTED_BASELINE_MODELS = ["tiered", "exp_smoothing", "var_cross_target"]
_VAR_TARGET_ORDER = ["hospitalizations", "wastewater", "cases", "deaths"]


@dataclass
class RollingFold:
    fold: int
    train_start: int
    train_end: int
    forecast_start: int
    forecast_end: int

    def to_dict(self, temporal_coords: list[Any]) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_start_idx": self.train_start,
            "train_end_idx": self.train_end,
            "forecast_start_idx": self.forecast_start,
            "forecast_end_idx": self.forecast_end,
            "train_start_date": str(temporal_coords[self.train_start]),
            "train_end_date": str(temporal_coords[self.train_end - 1]),
            "forecast_start_date": str(temporal_coords[self.forecast_start]),
            "forecast_end_date": str(temporal_coords[self.forecast_end - 1]),
        }


@dataclass
class TargetSeriesView:
    values: np.ndarray
    mask: np.ndarray
    node_to_bin: dict[int, int]


def _torch_to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _resolve_split_bounds(dataset: EpiDataset) -> tuple[int, int]:
    if dataset.time_range is not None:
        return int(dataset.time_range[0]), int(dataset.time_range[1])
    max_lag = max(dataset.mobility_lags, default=0) if dataset.mobility_lags else 0
    return int(max_lag), int(len(dataset._temporal_coords))


def _generate_rolling_folds(
    *,
    dataset: EpiDataset,
    rolling_folds: int,
) -> list[RollingFold]:
    L = int(dataset.config.model.history_length)
    H = int(dataset.config.model.forecast_horizon)
    split_start, split_end = _resolve_split_bounds(dataset)

    min_forecast_start = split_start + L + 2 * H
    max_forecast_start = split_end - H
    if min_forecast_start > max_forecast_start:
        return []

    forecast_starts = list(range(min_forecast_start, max_forecast_start + 1, H))
    selected = (
        forecast_starts[-rolling_folds:]
        if len(forecast_starts) > rolling_folds
        else forecast_starts
    )

    folds: list[RollingFold] = []
    for idx, forecast_start in enumerate(selected):
        folds.append(
            RollingFold(
                fold=idx,
                train_start=split_start,
                train_end=forecast_start,
                forecast_start=forecast_start,
                forecast_end=forecast_start + H,
            )
        )
    return folds


def _resolve_calendar_exog(dataset: EpiDataset) -> np.ndarray | None:
    cov = _torch_to_numpy_2d(dataset.temporal_covariates)
    if cov.ndim != 2 or cov.shape[1] == 0:
        return None

    ds = dataset._dataset
    if ds is not None and "temporal_covariates" in ds:
        da = ds["temporal_covariates"]
        if "covariate" in da.coords:
            cov_names = [str(x).lower() for x in da.coords["covariate"].values.tolist()]
            idx = [
                i
                for i, name in enumerate(cov_names)
                if ("dow" in name) or ("holiday" in name)
            ]
            if idx:
                return cov[:, idx]
    return cov


def _compute_valid_node_mask_for_target(
    *,
    target_mask: np.ndarray,
    forecast_start: int,
    history_length: int,
    horizon: int,
    permit: int,
) -> np.ndarray:
    history_slice = target_mask[forecast_start - history_length : forecast_start]
    target_slice = target_mask[forecast_start : forecast_start + horizon]
    history_counts = history_slice.sum(axis=0)
    target_counts = target_slice.sum(axis=0)
    history_threshold = max(0, history_length - permit)
    target_threshold = max(0, horizon - permit)
    return (history_counts >= history_threshold) & (target_counts >= target_threshold)


def _compute_sparsity_bins(
    *,
    mask: np.ndarray,
    split_start: int,
    split_end: int,
    target_nodes: list[int],
) -> dict[int, int]:
    view = mask[split_start:split_end, :]
    total_steps = max(1, view.shape[0])
    rows: list[dict[str, Any]] = []
    for node in target_nodes:
        observed = int(view[:, node].sum())
        sparsity_pct = 100.0 * (1.0 - (observed / total_steps))
        rows.append({"node": node, "sparsity_pct": sparsity_pct})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    if frame["sparsity_pct"].nunique() <= 1:
        return {int(node): 0 for node in frame["node"].tolist()}
    frame["bin"] = pd.qcut(frame["sparsity_pct"], q=10, labels=False, duplicates="drop")
    return {
        int(row["node"]): int(row["bin"]) if pd.notna(row["bin"]) else 0
        for row in frame.to_dict(orient="records")
    }


def _resolve_baseline_models(models: list[str] | None) -> list[str]:
    if not models:
        return ["tiered"]

    resolved: list[str] = []
    for model in models:
        key = model.lower()
        if key == "all":
            for baseline in _SUPPORTED_BASELINE_MODELS:
                if baseline not in resolved:
                    resolved.append(baseline)
            continue
        if key not in _SUPPORTED_BASELINE_MODELS:
            raise ValueError(
                f"Unsupported baseline model '{model}'. Supported: "
                f"{_SUPPORTED_BASELINE_MODELS + ['all']}"
            )
        if key not in resolved:
            resolved.append(key)
    return resolved


def _prepare_target_views(
    *,
    dataset: EpiDataset,
    split_start: int,
    split_end: int,
    target_nodes: list[int],
    include_sparsity_bins: bool,
) -> dict[str, TargetSeriesView]:
    out: dict[str, TargetSeriesView] = {}
    for target_name, (value_attr, mask_attr) in _TARGET_SPECS.items():
        values = _torch_to_numpy_2d(getattr(dataset, value_attr))
        mask = _torch_to_numpy_2d(getattr(dataset, mask_attr))
        mask = (mask > 0).astype(np.float64)
        node_to_bin = (
            _compute_sparsity_bins(
                mask=mask,
                split_start=split_start,
                split_end=split_end,
                target_nodes=target_nodes,
            )
            if include_sparsity_bins
            else {}
        )
        out[target_name] = TargetSeriesView(values=values, mask=mask, node_to_bin=node_to_bin)
    return out


def _global_median_for_target(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    train_start: int,
    train_end: int,
) -> float:
    train_view = values[train_start:train_end]
    train_mask_view = mask[train_start:train_end]
    global_obs = train_view[train_mask_view > 0]
    return float(np.nanmedian(global_obs)) if global_obs.size > 0 else 0.0


def _predict_univariate_baseline(
    *,
    baseline_model: str,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int,
    exog_train: np.ndarray | None,
    exog_future: np.ndarray | None,
) -> BaselinePredictionResult:
    if baseline_model == "tiered":
        return predict_with_tiered_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            global_train_median=global_train_median,
            seasonal_period=seasonal_period,
            exog_train=exog_train,
            exog_future=exog_future,
        )
    if baseline_model == "exp_smoothing":
        return predict_with_exponential_smoothing_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            global_train_median=global_train_median,
            seasonal_period=seasonal_period,
        )
    raise ValueError(f"Unsupported univariate baseline model: {baseline_model}")


def _evaluate_univariate_baseline_model(
    *,
    baseline_model: str,
    target_views: dict[str, TargetSeriesView],
    folds: list[RollingFold],
    target_nodes: list[int],
    permit_map: dict[str, int],
    history_length: int,
    horizon: int,
    seasonal_period: int,
    calendar_exog: np.ndarray | None,
    include_sparsity_bins: bool,
    fold_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    sparsity_rows: list[dict[str, Any]],
    model_orders: list[dict[str, Any]],
) -> None:
    for target_name, target_view in target_views.items():
        values = target_view.values
        mask = target_view.mask
        node_to_bin = target_view.node_to_bin

        for fold in folds:
            valid_nodes_mask = _compute_valid_node_mask_for_target(
                target_mask=mask,
                forecast_start=fold.forecast_start,
                history_length=history_length,
                horizon=horizon,
                permit=int(permit_map[target_name]),
            )

            global_median = _global_median_for_target(
                values=values,
                mask=mask,
                train_start=fold.train_start,
                train_end=fold.train_end,
            )

            pred_rows: list[np.ndarray] = []
            target_rows: list[np.ndarray] = []
            score_mask_rows: list[np.ndarray] = []
            model_usage: dict[str, int] = {}
            fit_success = 0
            node_pairs = 0
            scored_points = 0

            per_bin_preds: dict[int, list[np.ndarray]] = {}
            per_bin_targets: dict[int, list[np.ndarray]] = {}
            per_bin_masks: dict[int, list[np.ndarray]] = {}

            for node in target_nodes:
                if not bool(valid_nodes_mask[node]):
                    continue

                node_pairs += 1
                train_values = values[fold.train_start : fold.train_end, node]
                train_mask = mask[fold.train_start : fold.train_end, node]
                target_values = values[fold.forecast_start : fold.forecast_end, node]
                target_mask = mask[fold.forecast_start : fold.forecast_end, node]
                scored_points += int(target_mask.sum())

                exog_train = None
                exog_future = None
                if baseline_model == "tiered" and calendar_exog is not None:
                    exog_train = calendar_exog[fold.train_start : fold.train_end]
                    exog_future = calendar_exog[fold.forecast_start : fold.forecast_end]

                pred = _predict_univariate_baseline(
                    baseline_model=baseline_model,
                    train_values=train_values,
                    train_mask=train_mask,
                    horizon=horizon,
                    global_train_median=global_median,
                    seasonal_period=seasonal_period,
                    exog_train=exog_train,
                    exog_future=exog_future,
                )
                if pred.fit_status == "fit_success":
                    fit_success += 1
                model_usage[pred.model_name] = model_usage.get(pred.model_name, 0) + 1
                if pred.model_order:
                    model_orders.append(
                        {
                            "baseline_model": baseline_model,
                            "target": target_name,
                            "fold": fold.fold,
                            "node": int(node),
                            "model": pred.model_name,
                            "order": pred.model_order,
                        }
                    )

                pred_rows.append(pred.predictions.astype(np.float64))
                target_rows.append(target_values.astype(np.float64))
                score_mask_rows.append(target_mask.astype(np.float64))
                pair_rows.append(
                    {
                        "model": baseline_model,
                        "target": target_name,
                        "fold": fold.fold,
                        "node_id": int(node),
                        "selected_model": pred.model_name,
                        "fit_status": pred.fit_status,
                        "fallback_reason": pred.fallback_reason,
                        "observed_count": int(target_mask.sum()),
                    }
                )

                if include_sparsity_bins:
                    bin_idx = node_to_bin.get(int(node), 0)
                    per_bin_preds.setdefault(bin_idx, []).append(pred.predictions)
                    per_bin_targets.setdefault(bin_idx, []).append(target_values)
                    per_bin_masks.setdefault(bin_idx, []).append(target_mask)

            if node_pairs == 0:
                metric = compute_masked_metrics_numpy(
                    predictions=np.zeros((1, horizon), dtype=np.float64),
                    targets=np.zeros((1, horizon), dtype=np.float64),
                    observed_mask=np.zeros((1, horizon), dtype=np.float64),
                )
            else:
                metric = compute_masked_metrics_numpy(
                    predictions=np.vstack(pred_rows),
                    targets=np.vstack(target_rows),
                    observed_mask=np.vstack(score_mask_rows),
                )

            fold_rows.append(
                {
                    "model": baseline_model,
                    "target": target_name,
                    "fold": fold.fold,
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "smape": metric.smape,
                    "r2": metric.r2,
                    "observed_count": metric.observed_count,
                    "node_target_pairs": node_pairs,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "forecast_start": fold.forecast_start,
                    "forecast_end": fold.forecast_end,
                }
            )

            coverage_row = {
                "model": baseline_model,
                "target": target_name,
                "fold": fold.fold,
                "node_target_pairs": node_pairs,
                "fit_success_rate": (fit_success / node_pairs) if node_pairs else 0.0,
                "scored_points": scored_points,
                "total_points": node_pairs * horizon,
                "scored_point_coverage": (scored_points / (node_pairs * horizon))
                if node_pairs
                else 0.0,
            }
            for used_model, count in model_usage.items():
                coverage_row[f"used_{used_model}"] = count
            coverage_rows.append(coverage_row)

            if include_sparsity_bins:
                for bin_idx in sorted(per_bin_preds.keys()):
                    bin_metric = compute_masked_metrics_numpy(
                        predictions=np.vstack(per_bin_preds[bin_idx]),
                        targets=np.vstack(per_bin_targets[bin_idx]),
                        observed_mask=np.vstack(per_bin_masks[bin_idx]),
                    )
                    sparsity_rows.append(
                        {
                            "model": baseline_model,
                            "target": target_name,
                            "fold": fold.fold,
                            "sparsity_bin": int(bin_idx),
                            "mae": bin_metric.mae,
                            "rmse": bin_metric.rmse,
                            "smape": bin_metric.smape,
                            "r2": bin_metric.r2,
                            "observed_count": bin_metric.observed_count,
                            "node_count": len(per_bin_preds[bin_idx]),
                        }
                    )


def _evaluate_var_cross_target_model(
    *,
    target_views: dict[str, TargetSeriesView],
    folds: list[RollingFold],
    target_nodes: list[int],
    permit_map: dict[str, int],
    history_length: int,
    horizon: int,
    seasonal_period: int,
    include_sparsity_bins: bool,
    var_maxlags: int,
    fold_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    sparsity_rows: list[dict[str, Any]],
    model_orders: list[dict[str, Any]],
) -> None:
    var_targets = [t for t in _VAR_TARGET_ORDER if t in target_views]
    if not var_targets:
        return

    for fold in folds:
        valid_nodes_by_target: dict[str, np.ndarray] = {}
        global_medians_by_target: dict[str, float] = {}
        for target_name in var_targets:
            target_view = target_views[target_name]
            valid_nodes_by_target[target_name] = _compute_valid_node_mask_for_target(
                target_mask=target_view.mask,
                forecast_start=fold.forecast_start,
                history_length=history_length,
                horizon=horizon,
                permit=int(permit_map[target_name]),
            )
            global_medians_by_target[target_name] = _global_median_for_target(
                values=target_view.values,
                mask=target_view.mask,
                train_start=fold.train_start,
                train_end=fold.train_end,
            )

        pred_rows_by_target: dict[str, list[np.ndarray]] = {t: [] for t in var_targets}
        target_rows_by_target: dict[str, list[np.ndarray]] = {t: [] for t in var_targets}
        score_mask_by_target: dict[str, list[np.ndarray]] = {t: [] for t in var_targets}
        model_usage_by_target: dict[str, dict[str, int]] = {t: {} for t in var_targets}
        fit_success_by_target: dict[str, int] = {t: 0 for t in var_targets}
        node_pairs_by_target: dict[str, int] = {t: 0 for t in var_targets}
        scored_points_by_target: dict[str, int] = {t: 0 for t in var_targets}

        per_bin_preds: dict[str, dict[int, list[np.ndarray]]] = {t: {} for t in var_targets}
        per_bin_targets: dict[str, dict[int, list[np.ndarray]]] = {t: {} for t in var_targets}
        per_bin_masks: dict[str, dict[int, list[np.ndarray]]] = {t: {} for t in var_targets}

        for node in target_nodes:
            if not any(bool(valid_nodes_by_target[t][node]) for t in var_targets):
                continue

            train_matrix = np.column_stack(
                [
                    target_views[t].values[fold.train_start : fold.train_end, node]
                    for t in var_targets
                ]
            )
            train_mask_matrix = np.column_stack(
                [
                    target_views[t].mask[fold.train_start : fold.train_end, node]
                    for t in var_targets
                ]
            )
            global_median_vec = np.asarray(
                [global_medians_by_target[t] for t in var_targets],
                dtype=np.float64,
            )

            preds = predict_with_var_cross_target_fallback(
                train_values=train_matrix,
                train_mask=train_mask_matrix,
                horizon=horizon,
                target_names=var_targets,
                global_train_medians=global_median_vec,
                seasonal_period=seasonal_period,
                maxlags=var_maxlags,
            )

            for target_name in var_targets:
                if not bool(valid_nodes_by_target[target_name][node]):
                    continue

                target_view = target_views[target_name]
                target_values = target_view.values[
                    fold.forecast_start : fold.forecast_end,
                    node,
                ]
                target_mask = target_view.mask[
                    fold.forecast_start : fold.forecast_end,
                    node,
                ]
                pred = preds[target_name]

                node_pairs_by_target[target_name] += 1
                scored_points_by_target[target_name] += int(target_mask.sum())
                if pred.fit_status == "fit_success":
                    fit_success_by_target[target_name] += 1
                model_usage = model_usage_by_target[target_name]
                model_usage[pred.model_name] = model_usage.get(pred.model_name, 0) + 1

                if pred.model_order:
                    model_orders.append(
                        {
                            "baseline_model": "var_cross_target",
                            "target": target_name,
                            "fold": fold.fold,
                            "node": int(node),
                            "model": pred.model_name,
                            "order": pred.model_order,
                        }
                    )

                pred_rows_by_target[target_name].append(pred.predictions.astype(np.float64))
                target_rows_by_target[target_name].append(target_values.astype(np.float64))
                score_mask_by_target[target_name].append(target_mask.astype(np.float64))
                pair_rows.append(
                    {
                        "model": "var_cross_target",
                        "target": target_name,
                        "fold": fold.fold,
                        "node_id": int(node),
                        "selected_model": pred.model_name,
                        "fit_status": pred.fit_status,
                        "fallback_reason": pred.fallback_reason,
                        "observed_count": int(target_mask.sum()),
                    }
                )

                if include_sparsity_bins:
                    bin_idx = target_view.node_to_bin.get(int(node), 0)
                    per_bin_preds[target_name].setdefault(bin_idx, []).append(pred.predictions)
                    per_bin_targets[target_name].setdefault(bin_idx, []).append(target_values)
                    per_bin_masks[target_name].setdefault(bin_idx, []).append(target_mask)

        for target_name in var_targets:
            node_pairs = node_pairs_by_target[target_name]
            if node_pairs == 0:
                metric = compute_masked_metrics_numpy(
                    predictions=np.zeros((1, horizon), dtype=np.float64),
                    targets=np.zeros((1, horizon), dtype=np.float64),
                    observed_mask=np.zeros((1, horizon), dtype=np.float64),
                )
            else:
                metric = compute_masked_metrics_numpy(
                    predictions=np.vstack(pred_rows_by_target[target_name]),
                    targets=np.vstack(target_rows_by_target[target_name]),
                    observed_mask=np.vstack(score_mask_by_target[target_name]),
                )

            fold_rows.append(
                {
                    "model": "var_cross_target",
                    "target": target_name,
                    "fold": fold.fold,
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "smape": metric.smape,
                    "r2": metric.r2,
                    "observed_count": metric.observed_count,
                    "node_target_pairs": node_pairs,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "forecast_start": fold.forecast_start,
                    "forecast_end": fold.forecast_end,
                }
            )

            scored_points = scored_points_by_target[target_name]
            coverage_row = {
                "model": "var_cross_target",
                "target": target_name,
                "fold": fold.fold,
                "node_target_pairs": node_pairs,
                "fit_success_rate": (fit_success_by_target[target_name] / node_pairs)
                if node_pairs
                else 0.0,
                "scored_points": scored_points,
                "total_points": node_pairs * horizon,
                "scored_point_coverage": (scored_points / (node_pairs * horizon))
                if node_pairs
                else 0.0,
            }
            for used_model, count in model_usage_by_target[target_name].items():
                coverage_row[f"used_{used_model}"] = count
            coverage_rows.append(coverage_row)

            if include_sparsity_bins:
                for bin_idx in sorted(per_bin_preds[target_name].keys()):
                    bin_metric = compute_masked_metrics_numpy(
                        predictions=np.vstack(per_bin_preds[target_name][bin_idx]),
                        targets=np.vstack(per_bin_targets[target_name][bin_idx]),
                        observed_mask=np.vstack(per_bin_masks[target_name][bin_idx]),
                    )
                    sparsity_rows.append(
                        {
                            "model": "var_cross_target",
                            "target": target_name,
                            "fold": fold.fold,
                            "sparsity_bin": int(bin_idx),
                            "mae": bin_metric.mae,
                            "rmse": bin_metric.rmse,
                            "smape": bin_metric.smape,
                            "r2": bin_metric.r2,
                            "observed_count": bin_metric.observed_count,
                            "node_count": len(per_bin_preds[target_name][bin_idx]),
                        }
                    )


def run_baseline_evaluation(
    *,
    config: EpiForecasterConfig,
    output_dir: Path,
    models: list[str] | None = None,
    config_path: str | None = None,
    split: str = "test",
    rolling_folds: int = 5,
    seasonal_period: int = 7,
    include_sparsity_bins: bool = True,
    var_maxlags: int = 14,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_models = _resolve_baseline_models(models)

    loader, _region_embeddings = build_loader_from_config(
        config=config,
        split=split,
        batch_size=1,
        device="cpu",
    )
    dataset = loader.dataset

    folds = _generate_rolling_folds(dataset=dataset, rolling_folds=rolling_folds)
    if not folds:
        raise ValueError(
            "No valid rolling-origin folds found. Check split boundaries and horizon."
        )

    split_start, split_end = _resolve_split_bounds(dataset)
    temporal_coords = list(dataset._temporal_coords)
    target_nodes = list(dataset.target_nodes)
    horizon = int(config.model.forecast_horizon)
    history_length = int(config.model.history_length)
    permit_map = config.data.resolve_missing_permit_map()
    calendar_exog = _resolve_calendar_exog(dataset)
    target_views = _prepare_target_views(
        dataset=dataset,
        split_start=split_start,
        split_end=split_end,
        target_nodes=target_nodes,
        include_sparsity_bins=include_sparsity_bins,
    )

    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    sparsity_rows: list[dict[str, Any]] = []
    model_orders: list[dict[str, Any]] = []

    for model_name in resolved_models:
        if model_name in {"tiered", "exp_smoothing"}:
            _evaluate_univariate_baseline_model(
                baseline_model=model_name,
                target_views=target_views,
                folds=folds,
                target_nodes=target_nodes,
                permit_map=permit_map,
                history_length=history_length,
                horizon=horizon,
                seasonal_period=seasonal_period,
                calendar_exog=calendar_exog,
                include_sparsity_bins=include_sparsity_bins,
                fold_rows=fold_rows,
                coverage_rows=coverage_rows,
                pair_rows=pair_rows,
                sparsity_rows=sparsity_rows,
                model_orders=model_orders,
            )
        elif model_name == "var_cross_target":
            _evaluate_var_cross_target_model(
                target_views=target_views,
                folds=folds,
                target_nodes=target_nodes,
                permit_map=permit_map,
                history_length=history_length,
                horizon=horizon,
                seasonal_period=seasonal_period,
                include_sparsity_bins=include_sparsity_bins,
                var_maxlags=var_maxlags,
                fold_rows=fold_rows,
                coverage_rows=coverage_rows,
                pair_rows=pair_rows,
                sparsity_rows=sparsity_rows,
                model_orders=model_orders,
            )

    fold_df = pd.DataFrame(fold_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    pair_df = pd.DataFrame(pair_rows)
    sparsity_df = pd.DataFrame(sparsity_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for (model_name, target_name), group in fold_df.groupby(["model", "target"]):
        row: dict[str, Any] = {
            "model": model_name,
            "target": target_name,
            "folds": int(group["fold"].nunique()),
        }
        for metric_name in ["mae", "rmse", "smape", "r2", "observed_count"]:
            values = pd.to_numeric(group[metric_name], errors="coerce").dropna()
            if values.empty:
                row[f"{metric_name}_median"] = float("nan")
                row[f"{metric_name}_iqr"] = float("nan")
            else:
                row[f"{metric_name}_median"] = float(values.median())
                row[f"{metric_name}_iqr"] = float(
                    values.quantile(0.75) - values.quantile(0.25)
                )
        aggregate_rows.append(row)
    aggregate_df = pd.DataFrame(aggregate_rows)

    baseline_fold_metrics = output_dir / "baseline_fold_metrics.csv"
    baseline_aggregate_metrics = output_dir / "baseline_aggregate_metrics.csv"
    baseline_coverage = output_dir / "baseline_coverage.csv"
    baseline_pair_details = output_dir / "baseline_pair_details.csv"
    baseline_sparsity = output_dir / "baseline_sparsity_stratified_metrics.csv"
    baseline_vs_model_deltas = output_dir / "baseline_vs_model_deltas.csv"
    baseline_metadata = output_dir / "baseline_metadata.json"

    fold_df.to_csv(baseline_fold_metrics, index=False)
    aggregate_df.to_csv(baseline_aggregate_metrics, index=False)
    coverage_df.to_csv(baseline_coverage, index=False)
    pair_df.to_csv(baseline_pair_details, index=False)
    if include_sparsity_bins:
        sparsity_df.to_csv(baseline_sparsity, index=False)
    pd.DataFrame(
        columns=[
            "target",
            "fold",
            "metric",
            "model_value",
            "baseline_value",
            "delta_model_minus_baseline",
        ]
    ).to_csv(baseline_vs_model_deltas, index=False)

    metadata = {
        "split": split,
        "rolling_folds_requested": rolling_folds,
        "rolling_folds_used": len(folds),
        "history_length": history_length,
        "forecast_horizon": horizon,
        "seasonal_period": seasonal_period,
        "var_maxlags": var_maxlags,
        "models": resolved_models,
        "missing_permit_map": permit_map,
        "split_start": split_start,
        "split_end": split_end,
        "folds": [f.to_dict(temporal_coords) for f in folds],
        "model_orders": model_orders,
        "config_path": config_path,
        "run_id": config.data.run_id,
    }
    baseline_metadata.write_text(json.dumps(metadata, indent=2))

    result = {
        "baseline_fold_metrics": baseline_fold_metrics,
        "baseline_aggregate_metrics": baseline_aggregate_metrics,
        "baseline_coverage": baseline_coverage,
        "baseline_pair_details": baseline_pair_details,
        "baseline_vs_model_deltas": baseline_vs_model_deltas,
        "baseline_metadata": baseline_metadata,
    }
    if include_sparsity_bins:
        result["baseline_sparsity_stratified_metrics"] = baseline_sparsity
    return result


def run_tiered_baseline_evaluation(
    *,
    config: EpiForecasterConfig,
    config_path: str | None = None,
    output_dir: Path,
    split: str = "test",
    rolling_folds: int = 5,
    seasonal_period: int = 7,
    include_sparsity_bins: bool = True,
) -> dict[str, Path]:
    return run_baseline_evaluation(
        config=config,
        output_dir=output_dir,
        models=["tiered"],
        config_path=config_path,
        split=split,
        rolling_folds=rolling_folds,
        seasonal_period=seasonal_period,
        include_sparsity_bins=include_sparsity_bins,
    )


def compare_model_metrics_against_baselines(
    *,
    eval_metrics: dict[str, Any],
    baseline_results_csv: Path,
    output_csv: Path,
) -> Path:
    baseline_df = pd.read_csv(baseline_results_csv)
    if "fold" in baseline_df.columns:
        agg_rows: list[dict[str, Any]] = []
        for (model_name, target_name), group in baseline_df.groupby(
            ["model", "target"]
        ):
            agg_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "mae_median": pd.to_numeric(group["mae"], errors="coerce").median(),
                    "rmse_median": pd.to_numeric(
                        group["rmse"], errors="coerce"
                    ).median(),
                    "smape_median": pd.to_numeric(
                        group["smape"], errors="coerce"
                    ).median(),
                    "r2_median": pd.to_numeric(group["r2"], errors="coerce").median(),
                }
            )
        baseline_df = pd.DataFrame(agg_rows)

    model_target_metrics = {
        "hospitalizations": {
            "mae": eval_metrics.get("mae_hosp_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_hosp_log1p_per_100k"),
            "smape": eval_metrics.get("smape_hosp_log1p_per_100k"),
            "r2": eval_metrics.get("r2_hosp_log1p_per_100k"),
        },
        "wastewater": {
            "mae": eval_metrics.get("mae_ww_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_ww_log1p_per_100k"),
            "smape": eval_metrics.get("smape_ww_log1p_per_100k"),
            "r2": eval_metrics.get("r2_ww_log1p_per_100k"),
        },
        "cases": {
            "mae": eval_metrics.get("mae_cases_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_cases_log1p_per_100k"),
            "smape": eval_metrics.get("smape_cases_log1p_per_100k"),
            "r2": eval_metrics.get("r2_cases_log1p_per_100k"),
        },
        "deaths": {
            "mae": eval_metrics.get("mae_deaths_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_deaths_log1p_per_100k"),
            "smape": eval_metrics.get("smape_deaths_log1p_per_100k"),
            "r2": eval_metrics.get("r2_deaths_log1p_per_100k"),
        },
    }

    rows: list[dict[str, Any]] = []
    for baseline_row in baseline_df.to_dict(orient="records"):
        target = str(baseline_row["target"])
        model_name = str(baseline_row["model"])
        if target not in model_target_metrics:
            continue
        for metric_name in ["mae", "rmse", "smape", "r2"]:
            baseline_value = baseline_row.get(f"{metric_name}_median")
            model_value = model_target_metrics[target].get(metric_name)
            if baseline_value is None or model_value is None:
                continue
            rows.append(
                {
                    "target": target,
                    "baseline_model": model_name,
                    "metric": metric_name,
                    "model_value": float(model_value),
                    "baseline_value": float(baseline_value),
                    "delta_model_minus_baseline": float(model_value)
                    - float(baseline_value),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv
