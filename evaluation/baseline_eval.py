from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.epi_dataset import EpiDataset
from evaluation.baseline_models import (
    BaselinePredictionResult,
    predict_with_exponential_smoothing_fallback,
    predict_with_tiered_fallback,
    predict_with_var_cross_target_fallback,
)
from evaluation.epiforecaster_eval import (
    JointInferenceLoss,
    build_loader_from_config,
    get_loss_from_config,
)
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
_JOINT_LOSS_VALUE_CLAMP = 1.0e6


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


@dataclass
class JointObservationLossSpec:
    w_ww: float
    w_hosp: float
    w_cases: float
    w_deaths: float
    ww_imputed_weight: float
    hosp_imputed_weight: float
    cases_imputed_weight: float
    deaths_imputed_weight: float
    ww_min_observed: int
    hosp_min_observed: int
    cases_min_observed: int
    deaths_min_observed: int


def _torch_to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        # Upcast float16/bfloat16 to float32 before numpy conversion to prevent overflow
        if value.dtype in (torch.float16, torch.bfloat16):
            value = value.float()
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
    L = int(dataset.config.model.input_window_length)
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
    input_window_length: int,
    horizon: int,
    permit: int,
) -> np.ndarray:
    history_slice = target_mask[forecast_start - input_window_length : forecast_start]
    target_slice = target_mask[forecast_start : forecast_start + horizon]
    history_counts = history_slice.sum(axis=0)
    target_counts = target_slice.sum(axis=0)
    history_threshold = max(0, input_window_length - permit)
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
        out[target_name] = TargetSeriesView(
            values=values, mask=mask, node_to_bin=node_to_bin
        )
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


def _sanitize_metric_inputs(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    mask = np.clip(
        np.nan_to_num(np.asarray(observed_mask, dtype=np.float64), nan=0.0),
        0.0,
        1.0,
    )
    invalid = (~np.isfinite(pred)) | (~np.isfinite(target))
    dropped_invalid_observed = int((invalid & (mask > 0)).sum())
    if dropped_invalid_observed > 0:
        mask = mask.copy()
        mask[invalid] = 0.0
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return pred, target, mask, dropped_invalid_observed


def _resolve_joint_observation_loss_spec(
    *,
    config: EpiForecasterConfig,
    horizon: int,
) -> JointObservationLossSpec | None:
    if not hasattr(config, "training") or not hasattr(config.training, "loss"):
        return None
    criterion = get_loss_from_config(
        config.training.loss,
        data_config=config.data,
        forecast_horizon=horizon,
    )
    if not isinstance(criterion, JointInferenceLoss):
        return None

    return JointObservationLossSpec(
        w_ww=float(criterion.w_ww),
        w_hosp=float(criterion.w_hosp),
        w_cases=float(criterion.w_cases),
        w_deaths=float(criterion.w_deaths),
        ww_imputed_weight=float(criterion.ww_imputed_weight),
        hosp_imputed_weight=float(criterion.hosp_imputed_weight),
        cases_imputed_weight=float(criterion.cases_imputed_weight),
        deaths_imputed_weight=float(criterion.deaths_imputed_weight),
        ww_min_observed=int(criterion.ww_min_observed),
        hosp_min_observed=int(criterion.hosp_min_observed),
        cases_min_observed=int(criterion.cases_min_observed),
        deaths_min_observed=int(criterion.deaths_min_observed),
    )


def _empty_metric_matrix(horizon: int) -> np.ndarray:
    return np.zeros((1, horizon), dtype=np.float64)


def _joint_weighted_masked_mse_numpy(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    observed_mask: np.ndarray,
    imputed_weight: float,
    min_observed: int,
) -> float:
    pred_t = torch.as_tensor(prediction, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    observed_t = torch.as_tensor(observed_mask, dtype=torch.float32)

    finite_mask = torch.isfinite(target_t).to(pred_t.dtype)
    observed = torch.nan_to_num(observed_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(
        min=0.0, max=1.0
    )
    weights = (observed + (1.0 - observed) * float(imputed_weight)) * finite_mask
    observed_binary = (observed > 0.5).to(pred_t.dtype) * finite_mask

    if min_observed > 0:
        observed_counts = observed_binary.sum(dim=1, keepdim=True)
        eligible = (observed_counts >= float(min_observed)).to(pred_t.dtype)
        weights = weights * eligible

    active = weights > 0
    prediction_clean = torch.nan_to_num(
        pred_t,
        nan=0.0,
        posinf=_JOINT_LOSS_VALUE_CLAMP,
        neginf=-_JOINT_LOSS_VALUE_CLAMP,
    ).clamp(min=-_JOINT_LOSS_VALUE_CLAMP, max=_JOINT_LOSS_VALUE_CLAMP)
    target_clean = torch.nan_to_num(
        target_t,
        nan=0.0,
        posinf=_JOINT_LOSS_VALUE_CLAMP,
        neginf=-_JOINT_LOSS_VALUE_CLAMP,
    ).clamp(min=-_JOINT_LOSS_VALUE_CLAMP, max=_JOINT_LOSS_VALUE_CLAMP)
    prediction_safe = torch.where(
        active, prediction_clean, torch.zeros_like(prediction_clean)
    )
    target_safe = torch.where(active, target_clean, torch.zeros_like(target_clean))

    sq = (prediction_safe - target_safe) ** 2
    numerator = (sq * weights).sum()
    denominator = weights.sum().clamp_min(1e-8)
    return float((numerator / denominator).item())


def _compute_joint_observation_loss_for_fold(
    *,
    fold_target_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    horizon: int,
    joint_spec: JointObservationLossSpec,
) -> dict[str, Any]:
    target_aliases = {
        "wastewater": "ww",
        "hospitalizations": "hosp",
        "cases": "cases",
        "deaths": "deaths",
    }
    target_weights = {
        "wastewater": joint_spec.w_ww,
        "hospitalizations": joint_spec.w_hosp,
        "cases": joint_spec.w_cases,
        "deaths": joint_spec.w_deaths,
    }
    target_imputed_weights = {
        "wastewater": joint_spec.ww_imputed_weight,
        "hospitalizations": joint_spec.hosp_imputed_weight,
        "cases": joint_spec.cases_imputed_weight,
        "deaths": joint_spec.deaths_imputed_weight,
    }
    target_min_observed = {
        "wastewater": joint_spec.ww_min_observed,
        "hospitalizations": joint_spec.hosp_min_observed,
        "cases": joint_spec.cases_min_observed,
        "deaths": joint_spec.deaths_min_observed,
    }

    components: dict[str, Any] = {"joint_obs_loss_total": 0.0}
    for target_name, alias in target_aliases.items():
        pred, target, mask = fold_target_data.get(
            target_name,
            (
                _empty_metric_matrix(horizon),
                _empty_metric_matrix(horizon),
                _empty_metric_matrix(horizon),
            ),
        )
        loss_value = _joint_weighted_masked_mse_numpy(
            prediction=pred,
            target=target,
            observed_mask=mask,
            imputed_weight=target_imputed_weights[target_name],
            min_observed=target_min_observed[target_name],
        )
        weighted_value = target_weights[target_name] * loss_value
        components[f"joint_loss_{alias}"] = loss_value
        components[f"joint_loss_{alias}_weighted"] = weighted_value
        components[f"joint_observed_count_{alias}"] = int((mask > 0).sum())
        components["joint_obs_loss_total"] += weighted_value

    return components


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
    permit_map: dict[str, dict[str, int]],
    input_window_length: int,
    horizon: int,
    seasonal_period: int,
    calendar_exog: np.ndarray | None,
    include_sparsity_bins: bool,
    fold_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    sparsity_rows: list[dict[str, Any]],
    model_orders: list[dict[str, Any]],
    joint_spec: JointObservationLossSpec | None,
    joint_rows: list[dict[str, Any]],
) -> None:
    joint_target_buffers: dict[
        int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}

    for target_name, target_view in target_views.items():
        values = target_view.values
        mask = target_view.mask
        node_to_bin = target_view.node_to_bin

        for fold in folds:
            valid_nodes_mask = _compute_valid_node_mask_for_target(
                target_mask=mask,
                forecast_start=fold.forecast_start,
                input_window_length=input_window_length,
                horizon=horizon,
                permit=int(permit_map["horizon"][target_name]),
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
            dropped_invalid_observed = 0

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

                cleaned_pred, cleaned_target, cleaned_mask, dropped = (
                    _sanitize_metric_inputs(
                        predictions=pred.predictions,
                        targets=target_values,
                        observed_mask=target_mask,
                    )
                )
                dropped_invalid_observed += dropped
                scored_points += int(cleaned_mask.sum())

                pred_rows.append(cleaned_pred)
                target_rows.append(cleaned_target)
                score_mask_rows.append(cleaned_mask)
                pair_rows.append(
                    {
                        "model": baseline_model,
                        "target": target_name,
                        "fold": fold.fold,
                        "node_id": int(node),
                        "selected_model": pred.model_name,
                        "fit_status": pred.fit_status,
                        "fallback_reason": pred.fallback_reason,
                        "observed_count": int(cleaned_mask.sum()),
                        "invalid_observed_dropped": dropped,
                    }
                )

                if include_sparsity_bins:
                    bin_idx = node_to_bin.get(int(node), 0)
                    per_bin_preds.setdefault(bin_idx, []).append(cleaned_pred)
                    per_bin_targets.setdefault(bin_idx, []).append(cleaned_target)
                    per_bin_masks.setdefault(bin_idx, []).append(cleaned_mask)

            if dropped_invalid_observed > 0:
                logger.warning(
                    "Dropped %d invalid observed points while scoring baseline=%s "
                    "target=%s fold=%d",
                    dropped_invalid_observed,
                    baseline_model,
                    target_name,
                    fold.fold,
                )

            if node_pairs == 0:
                metric = compute_masked_metrics_numpy(
                    predictions=np.zeros((1, horizon), dtype=np.float64),
                    targets=np.zeros((1, horizon), dtype=np.float64),
                    observed_mask=np.zeros((1, horizon), dtype=np.float64),
                    horizon=horizon,
                )
            else:
                metric = compute_masked_metrics_numpy(
                    predictions=np.vstack(pred_rows),
                    targets=np.vstack(target_rows),
                    observed_mask=np.vstack(score_mask_rows),
                    horizon=horizon,
                )

            fold_row: dict[str, Any] = {
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
            for h_idx, (mae_h, rmse_h) in enumerate(
                zip(metric.mae_per_h, metric.rmse_per_h, strict=False)
            ):
                fold_row[f"mae_h{h_idx + 1}"] = mae_h
                fold_row[f"rmse_h{h_idx + 1}"] = rmse_h
            fold_rows.append(fold_row)

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
                        horizon=horizon,
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

            if joint_spec is not None:
                fold_target_data = joint_target_buffers.setdefault(fold.fold, {})
                if pred_rows:
                    fold_target_data[target_name] = (
                        np.vstack(pred_rows),
                        np.vstack(target_rows),
                        np.vstack(score_mask_rows),
                    )
                else:
                    fold_target_data[target_name] = (
                        _empty_metric_matrix(horizon),
                        _empty_metric_matrix(horizon),
                        _empty_metric_matrix(horizon),
                    )

    if joint_spec is not None:
        for fold in folds:
            components = _compute_joint_observation_loss_for_fold(
                fold_target_data=joint_target_buffers.get(fold.fold, {}),
                horizon=horizon,
                joint_spec=joint_spec,
            )
            joint_rows.append(
                {
                    "model": baseline_model,
                    "fold": fold.fold,
                    **components,
                }
            )


def _evaluate_var_cross_target_model(
    *,
    target_views: dict[str, TargetSeriesView],
    folds: list[RollingFold],
    target_nodes: list[int],
    permit_map: dict[str, dict[str, int]],
    input_window_length: int,
    horizon: int,
    seasonal_period: int,
    include_sparsity_bins: bool,
    var_maxlags: int,
    fold_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    sparsity_rows: list[dict[str, Any]],
    model_orders: list[dict[str, Any]],
    joint_spec: JointObservationLossSpec | None,
    joint_rows: list[dict[str, Any]],
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
                input_window_length=input_window_length,
                horizon=horizon,
                permit=int(permit_map["horizon"][target_name]),
            )
            global_medians_by_target[target_name] = _global_median_for_target(
                values=target_view.values,
                mask=target_view.mask,
                train_start=fold.train_start,
                train_end=fold.train_end,
            )

        pred_rows_by_target: dict[str, list[np.ndarray]] = {t: [] for t in var_targets}
        target_rows_by_target: dict[str, list[np.ndarray]] = {
            t: [] for t in var_targets
        }
        score_mask_by_target: dict[str, list[np.ndarray]] = {t: [] for t in var_targets}
        model_usage_by_target: dict[str, dict[str, int]] = {t: {} for t in var_targets}
        fit_success_by_target: dict[str, int] = {t: 0 for t in var_targets}
        node_pairs_by_target: dict[str, int] = {t: 0 for t in var_targets}
        scored_points_by_target: dict[str, int] = {t: 0 for t in var_targets}
        dropped_invalid_by_target: dict[str, int] = {t: 0 for t in var_targets}

        per_bin_preds: dict[str, dict[int, list[np.ndarray]]] = {
            t: {} for t in var_targets
        }
        per_bin_targets: dict[str, dict[int, list[np.ndarray]]] = {
            t: {} for t in var_targets
        }
        per_bin_masks: dict[str, dict[int, list[np.ndarray]]] = {
            t: {} for t in var_targets
        }

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

                cleaned_pred, cleaned_target, cleaned_mask, dropped = (
                    _sanitize_metric_inputs(
                        predictions=pred.predictions,
                        targets=target_values,
                        observed_mask=target_mask,
                    )
                )

                node_pairs_by_target[target_name] += 1
                scored_points_by_target[target_name] += int(cleaned_mask.sum())
                dropped_invalid_by_target[target_name] += dropped
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

                pred_rows_by_target[target_name].append(cleaned_pred)
                target_rows_by_target[target_name].append(cleaned_target)
                score_mask_by_target[target_name].append(cleaned_mask)
                pair_rows.append(
                    {
                        "model": "var_cross_target",
                        "target": target_name,
                        "fold": fold.fold,
                        "node_id": int(node),
                        "selected_model": pred.model_name,
                        "fit_status": pred.fit_status,
                        "fallback_reason": pred.fallback_reason,
                        "observed_count": int(cleaned_mask.sum()),
                        "invalid_observed_dropped": dropped,
                    }
                )

                if include_sparsity_bins:
                    bin_idx = target_view.node_to_bin.get(int(node), 0)
                    per_bin_preds[target_name].setdefault(bin_idx, []).append(
                        cleaned_pred
                    )
                    per_bin_targets[target_name].setdefault(bin_idx, []).append(
                        cleaned_target
                    )
                    per_bin_masks[target_name].setdefault(bin_idx, []).append(
                        cleaned_mask
                    )

        for target_name in var_targets:
            node_pairs = node_pairs_by_target[target_name]
            if dropped_invalid_by_target[target_name] > 0:
                logger.warning(
                    "Dropped %d invalid observed points while scoring baseline=%s "
                    "target=%s fold=%d",
                    dropped_invalid_by_target[target_name],
                    "var_cross_target",
                    target_name,
                    fold.fold,
                )
            if node_pairs == 0:
                metric = compute_masked_metrics_numpy(
                    predictions=np.zeros((1, horizon), dtype=np.float64),
                    targets=np.zeros((1, horizon), dtype=np.float64),
                    observed_mask=np.zeros((1, horizon), dtype=np.float64),
                    horizon=horizon,
                )
            else:
                metric = compute_masked_metrics_numpy(
                    predictions=np.vstack(pred_rows_by_target[target_name]),
                    targets=np.vstack(target_rows_by_target[target_name]),
                    observed_mask=np.vstack(score_mask_by_target[target_name]),
                    horizon=horizon,
                )

            fold_row: dict[str, Any] = {
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
            for h_idx, (mae_h, rmse_h) in enumerate(
                zip(metric.mae_per_h, metric.rmse_per_h, strict=False)
            ):
                fold_row[f"mae_h{h_idx + 1}"] = mae_h
                fold_row[f"rmse_h{h_idx + 1}"] = rmse_h
            fold_rows.append(fold_row)

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
                        horizon=horizon,
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

        if joint_spec is not None:
            fold_target_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for target_name in _TARGET_SPECS:
                pred_rows = pred_rows_by_target.get(target_name, [])
                target_rows = target_rows_by_target.get(target_name, [])
                mask_rows = score_mask_by_target.get(target_name, [])
                if pred_rows:
                    fold_target_data[target_name] = (
                        np.vstack(pred_rows),
                        np.vstack(target_rows),
                        np.vstack(mask_rows),
                    )
                else:
                    fold_target_data[target_name] = (
                        _empty_metric_matrix(horizon),
                        _empty_metric_matrix(horizon),
                        _empty_metric_matrix(horizon),
                    )
            components = _compute_joint_observation_loss_for_fold(
                fold_target_data=fold_target_data,
                horizon=horizon,
                joint_spec=joint_spec,
            )
            joint_rows.append(
                {
                    "model": "var_cross_target",
                    "fold": fold.fold,
                    **components,
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
    input_window_length = int(config.model.input_window_length)
    permit_map = config.data.resolve_missing_permit_map()
    calendar_exog = _resolve_calendar_exog(dataset)
    joint_spec = _resolve_joint_observation_loss_spec(config=config, horizon=horizon)
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
    joint_rows: list[dict[str, Any]] = []

    for model_name in resolved_models:
        if model_name in {"tiered", "exp_smoothing"}:
            _evaluate_univariate_baseline_model(
                baseline_model=model_name,
                target_views=target_views,
                folds=folds,
                target_nodes=target_nodes,
                permit_map=permit_map,
                input_window_length=input_window_length,
                horizon=horizon,
                seasonal_period=seasonal_period,
                calendar_exog=calendar_exog,
                include_sparsity_bins=include_sparsity_bins,
                fold_rows=fold_rows,
                coverage_rows=coverage_rows,
                pair_rows=pair_rows,
                sparsity_rows=sparsity_rows,
                model_orders=model_orders,
                joint_spec=joint_spec,
                joint_rows=joint_rows,
            )
        elif model_name == "var_cross_target":
            _evaluate_var_cross_target_model(
                target_views=target_views,
                folds=folds,
                target_nodes=target_nodes,
                permit_map=permit_map,
                input_window_length=input_window_length,
                horizon=horizon,
                seasonal_period=seasonal_period,
                include_sparsity_bins=include_sparsity_bins,
                var_maxlags=var_maxlags,
                fold_rows=fold_rows,
                coverage_rows=coverage_rows,
                pair_rows=pair_rows,
                sparsity_rows=sparsity_rows,
                model_orders=model_orders,
                joint_spec=joint_spec,
                joint_rows=joint_rows,
            )

    fold_df = pd.DataFrame(fold_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    pair_df = pd.DataFrame(pair_rows)
    sparsity_df = pd.DataFrame(sparsity_rows)
    joint_cols = [
        "model",
        "fold",
        "joint_obs_loss_total",
        "joint_loss_ww",
        "joint_loss_hosp",
        "joint_loss_cases",
        "joint_loss_deaths",
        "joint_loss_ww_weighted",
        "joint_loss_hosp_weighted",
        "joint_loss_cases_weighted",
        "joint_loss_deaths_weighted",
        "joint_observed_count_ww",
        "joint_observed_count_hosp",
        "joint_observed_count_cases",
        "joint_observed_count_deaths",
    ]
    joint_fold_df = (
        pd.DataFrame(joint_rows) if joint_rows else pd.DataFrame(columns=joint_cols)
    )

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
                row[f"{metric_name}_mean"] = float("nan")
                row[f"{metric_name}_std"] = float("nan")
            else:
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(values.std(ddof=1))

        per_h_cols = [
            c for c in group.columns if c.startswith("mae_h") or c.startswith("rmse_h")
        ]
        if per_h_cols:
            max_h = max(int(c.split("_h")[1]) for c in per_h_cols if "_h" in c)
            for start_idx in range(0, max_h, 7):
                end_idx = min(start_idx + 7, max_h)
                week_num = (start_idx // 7) + 1
                week_mae_cols = [f"mae_h{h}" for h in range(start_idx + 1, end_idx + 1)]
                week_rmse_cols = [
                    f"rmse_h{h}" for h in range(start_idx + 1, end_idx + 1)
                ]

                week_mae_values: list[float] = []
                week_rmse_values: list[float] = []
                for col in week_mae_cols:
                    if col in group.columns:
                        vals = pd.to_numeric(group[col], errors="coerce").dropna()
                        week_mae_values.extend(vals.tolist())
                for col in week_rmse_cols:
                    if col in group.columns:
                        vals = pd.to_numeric(group[col], errors="coerce").dropna()
                        week_rmse_values.extend(vals.tolist())

                if week_mae_values:
                    week_mae_arr = np.array(week_mae_values)
                    row[f"mae_w{week_num}_mean"] = float(np.nanmean(week_mae_arr))
                    row[f"mae_w{week_num}_median"] = float(np.nanmedian(week_mae_arr))
                    row[f"mae_w{week_num}_std"] = float(np.nanstd(week_mae_arr))
                else:
                    row[f"mae_w{week_num}_mean"] = float("nan")
                    row[f"mae_w{week_num}_median"] = float("nan")
                    row[f"mae_w{week_num}_std"] = float("nan")

                if week_rmse_values:
                    week_rmse_arr = np.array(week_rmse_values)
                    row[f"rmse_w{week_num}_mean"] = float(np.nanmean(week_rmse_arr))
                    row[f"rmse_w{week_num}_median"] = float(np.nanmedian(week_rmse_arr))
                    row[f"rmse_w{week_num}_std"] = float(np.nanstd(week_rmse_arr))
                else:
                    row[f"rmse_w{week_num}_mean"] = float("nan")
                    row[f"rmse_w{week_num}_median"] = float("nan")
                    row[f"rmse_w{week_num}_std"] = float("nan")

        aggregate_rows.append(row)
    aggregate_df = pd.DataFrame(aggregate_rows)
    joint_aggregate_rows: list[dict[str, Any]] = []
    if not joint_fold_df.empty:
        value_cols = [
            col for col in joint_fold_df.columns if col not in {"model", "fold"}
        ]
        for model_name, group in joint_fold_df.groupby("model"):
            row: dict[str, Any] = {
                "model": model_name,
                "folds": int(group["fold"].nunique()),
            }
            for value_col in value_cols:
                values = pd.to_numeric(group[value_col], errors="coerce").dropna()
                if values.empty:
                    row[f"{value_col}_mean"] = float("nan")
                    row[f"{value_col}_std"] = float("nan")
                else:
                    row[f"{value_col}_mean"] = float(values.mean())
                    row[f"{value_col}_std"] = float(values.std(ddof=1))
            joint_aggregate_rows.append(row)
    joint_aggregate_df = pd.DataFrame(joint_aggregate_rows)
    if joint_aggregate_df.empty:
        joint_aggregate_df = pd.DataFrame(columns=["model", "folds"])

    baseline_fold_metrics = output_dir / "baseline_fold_metrics.csv"
    baseline_aggregate_metrics = output_dir / "baseline_aggregate_metrics.csv"
    baseline_joint_loss_fold = output_dir / "baseline_joint_loss_fold.csv"
    baseline_joint_loss_aggregate = output_dir / "baseline_joint_loss_aggregate.csv"
    baseline_coverage = output_dir / "baseline_coverage.csv"
    baseline_pair_details = output_dir / "baseline_pair_details.csv"
    baseline_model_usage = output_dir / "baseline_model_usage.csv"
    baseline_sparsity = output_dir / "baseline_sparsity_stratified_metrics.csv"
    baseline_vs_model_deltas = output_dir / "baseline_vs_model_deltas.csv"
    baseline_metadata = output_dir / "baseline_metadata.json"

    fold_df.to_csv(baseline_fold_metrics, index=False)
    aggregate_df.to_csv(baseline_aggregate_metrics, index=False)
    joint_fold_df.to_csv(baseline_joint_loss_fold, index=False)
    joint_aggregate_df.to_csv(baseline_joint_loss_aggregate, index=False)
    coverage_df.to_csv(baseline_coverage, index=False)
    pair_df.to_csv(baseline_pair_details, index=False)
    usage_df = (
        pair_df.groupby(["model", "selected_model"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    usage_df["count"] = usage_df["count"].astype(int)
    usage_df = usage_df.sort_values(
        by=["model", "count", "selected_model"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    usage_df.to_csv(baseline_model_usage, index=False)
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
        "input_window_length": input_window_length,
        "forecast_horizon": horizon,
        "seasonal_period": seasonal_period,
        "var_maxlags": var_maxlags,
        "models": resolved_models,
        "weekly_horizon_aggregation": {
            "enabled": True,
            "days_per_week": 7,
            "metrics": ["mae", "rmse"],
            "statistics": ["mean", "median", "std"],
        },
        "joint_observation_loss_enabled": joint_spec is not None,
        "joint_observation_loss_spec": None
        if joint_spec is None
        else {
            "w_ww": joint_spec.w_ww,
            "w_hosp": joint_spec.w_hosp,
            "w_cases": joint_spec.w_cases,
            "w_deaths": joint_spec.w_deaths,
            "ww_imputed_weight": joint_spec.ww_imputed_weight,
            "hosp_imputed_weight": joint_spec.hosp_imputed_weight,
            "cases_imputed_weight": joint_spec.cases_imputed_weight,
            "deaths_imputed_weight": joint_spec.deaths_imputed_weight,
            "ww_min_observed": joint_spec.ww_min_observed,
            "hosp_min_observed": joint_spec.hosp_min_observed,
            "cases_min_observed": joint_spec.cases_min_observed,
            "deaths_min_observed": joint_spec.deaths_min_observed,
        },
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
        "baseline_joint_loss_fold": baseline_joint_loss_fold,
        "baseline_joint_loss_aggregate": baseline_joint_loss_aggregate,
        "baseline_coverage": baseline_coverage,
        "baseline_pair_details": baseline_pair_details,
        "baseline_model_usage": baseline_model_usage,
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
    has_fold = "fold" in baseline_df.columns
    has_target_metrics = {
        "target",
        "mae",
        "rmse",
        "smape",
        "r2",
    }.issubset(set(baseline_df.columns))
    has_joint_fold_metrics = "joint_obs_loss_total" in baseline_df.columns

    if has_fold and has_target_metrics:
        agg_rows: list[dict[str, Any]] = []
        for (model_name, target_name), group in baseline_df.groupby(
            ["model", "target"]
        ):
            agg_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "mae_mean": pd.to_numeric(group["mae"], errors="coerce").mean(),
                    "rmse_mean": pd.to_numeric(group["rmse"], errors="coerce").mean(),
                    "smape_mean": pd.to_numeric(group["smape"], errors="coerce").mean(),
                    "r2_mean": pd.to_numeric(group["r2"], errors="coerce").mean(),
                }
            )
        baseline_df = pd.DataFrame(agg_rows)
    elif has_fold and has_joint_fold_metrics:
        joint_value_cols = [
            col for col in baseline_df.columns if col not in {"model", "fold"}
        ]
        agg_rows = []
        for model_name, group in baseline_df.groupby("model"):
            row: dict[str, Any] = {"model": model_name}
            for col in joint_value_cols:
                row[f"{col}_mean"] = pd.to_numeric(group[col], errors="coerce").mean()
            agg_rows.append(row)
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
    joint_component_metrics = {
        "joint_loss_ww_weighted": eval_metrics.get("loss_ww_weighted"),
        "joint_loss_hosp_weighted": eval_metrics.get("loss_hosp_weighted"),
        "joint_loss_cases_weighted": eval_metrics.get("loss_cases_weighted"),
        "joint_loss_deaths_weighted": eval_metrics.get("loss_deaths_weighted"),
    }
    joint_obs_total = 0.0
    has_joint_obs_component = False
    for value in joint_component_metrics.values():
        if value is None:
            continue
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            continue
        joint_obs_total += numeric_value
        has_joint_obs_component = True

    rows: list[dict[str, Any]] = []
    joint_total_recorded: set[str] = set()
    joint_component_recorded: set[tuple[str, str]] = set()
    for baseline_row in baseline_df.to_dict(orient="records"):
        target = str(baseline_row.get("target"))
        model_name = str(baseline_row["model"])
        if target in model_target_metrics:
            for metric_name in ["mae", "rmse", "smape", "r2"]:
                baseline_value = baseline_row.get(f"{metric_name}_mean")
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

        if has_joint_obs_component and model_name not in joint_total_recorded:
            baseline_joint_total = baseline_row.get("joint_obs_loss_total_mean")
            if baseline_joint_total is not None and pd.notna(baseline_joint_total):
                rows.append(
                    {
                        "target": "joint_observation",
                        "baseline_model": model_name,
                        "metric": "joint_obs_loss_total",
                        "model_value": float(joint_obs_total),
                        "baseline_value": float(baseline_joint_total),
                        "delta_model_minus_baseline": float(joint_obs_total)
                        - float(baseline_joint_total),
                    }
                )
                joint_total_recorded.add(model_name)

        for baseline_key, model_metric in joint_component_metrics.items():
            if (model_name, baseline_key) in joint_component_recorded:
                continue
            if model_metric is None:
                continue
            metric_value = float(model_metric)
            if not np.isfinite(metric_value):
                continue
            baseline_value = baseline_row.get(f"{baseline_key}_mean")
            if baseline_value is None or not pd.notna(baseline_value):
                continue
            rows.append(
                {
                    "target": "joint_observation",
                    "baseline_model": model_name,
                    "metric": baseline_key,
                    "model_value": metric_value,
                    "baseline_value": float(baseline_value),
                    "delta_model_minus_baseline": metric_value - float(baseline_value),
                }
            )
            joint_component_recorded.add((model_name, baseline_key))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv
