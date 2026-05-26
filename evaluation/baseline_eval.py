from __future__ import annotations

import json
import logging
import time
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
    predict_with_last_observed_fallback,
    predict_with_sarima_fallback,
    predict_with_var_fallback,
    predict_with_varmax_fallback,
)
from evaluation.losses import get_loss_from_config
from evaluation.epiforecaster_eval import (
    build_loader_from_config,
)
from evaluation.granular_export import (
    GRANULAR_FIELDNAMES,
    _format_temporal_coord,
    write_granular_metadata_sidecar,
)
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)

_WEEKLY_CADENCE_DAYS = 7
_BASELINE_PROGRESS_LOG_BATCH_INTERVAL = 10
_BASELINE_PROGRESS_LOG_TIME_INTERVAL_SEC = 30.0


@dataclass(frozen=True)
class BaselineTargetSpec:
    target_name: str
    value_attr: str
    mask_attr: str
    cadence_mode: str
    joint_alias: str


_TARGET_SPECS = {
    "wastewater": BaselineTargetSpec(
        target_name="wastewater",
        value_attr="precomputed_ww",
        mask_attr="precomputed_ww_mask",
        cadence_mode="weekly_observed_dates",
        joint_alias="ww",
    ),
    "hospitalizations": BaselineTargetSpec(
        target_name="hospitalizations",
        value_attr="precomputed_hosp",
        mask_attr="precomputed_hosp_mask",
        cadence_mode="weekly_observed_dates",
        joint_alias="hosp",
    ),
    "cases": BaselineTargetSpec(
        target_name="cases",
        value_attr="precomputed_cases_target",
        mask_attr="precomputed_cases_mask",
        cadence_mode="daily",
        joint_alias="cases",
    ),
    "deaths": BaselineTargetSpec(
        target_name="deaths",
        value_attr="precomputed_deaths",
        mask_attr="precomputed_deaths_mask",
        cadence_mode="daily",
        joint_alias="deaths",
    ),
}
_SUPPORTED_BASELINE_MODELS = [
    "exp_smoothing",
    "last_observed",
    "sarima",
    "var",
    "varmax",
]
_DEFAULT_ALL_BASELINE_MODELS = [
    "exp_smoothing",
    "last_observed",
    "sarima",
    "var",
]
_SAME_SLICE_BASELINE_SCHEMA_VERSION = "1"
_SAME_SLICE_COMPARISON_SCOPE = "same_eval_slice"
_MAX_SAFE_ABS_ERROR = float(np.sqrt(np.finfo(np.float64).max) / 4.0)
_BASELINE_GRANULAR_FIELDNAMES = [
    "model",
    "selected_model",
    "fit_status",
    "fallback_reason",
    *GRANULAR_FIELDNAMES,
]
_BASELINE_FAILURE_FIELDNAMES = [
    "model",
    "selected_model",
    "target",
    "node_id",
    "region_id",
    "region_label",
    "window_start",
    "window_start_date",
    "fit_status",
    "error_reason",
]


@dataclass
class TargetSeriesView:
    spec: BaselineTargetSpec
    values: np.ndarray
    mask: np.ndarray
    node_to_bin: dict[int, int]


@dataclass
class JointObservationLossSpec:
    obs_weight_sum: float
    ww_min_observed: int
    hosp_min_observed: int
    cases_min_observed: int
    deaths_min_observed: int


@dataclass(frozen=True)
class SameSliceTargetSpec:
    canonical_name: str
    history_attr: str
    target_attr: str
    target_mask_attr: str
    prediction_key: str


_SAME_SLICE_TARGET_SPECS = {
    "hospitalizations": SameSliceTargetSpec(
        canonical_name="hospitalizations",
        history_attr="hosp_hist",
        target_attr="hosp_target",
        target_mask_attr="hosp_target_mask",
        prediction_key="pred_hosp",
    ),
    "wastewater": SameSliceTargetSpec(
        canonical_name="wastewater",
        history_attr="ww_hist",
        target_attr="ww_target",
        target_mask_attr="ww_target_mask",
        prediction_key="pred_ww",
    ),
    "cases": SameSliceTargetSpec(
        canonical_name="cases",
        history_attr="cases_hist",
        target_attr="cases_target",
        target_mask_attr="cases_target_mask",
        prediction_key="pred_cases",
    ),
    "deaths": SameSliceTargetSpec(
        canonical_name="deaths",
        history_attr="deaths_hist",
        target_attr="deaths_target",
        target_mask_attr="deaths_target_mask",
        prediction_key="pred_deaths",
    ),
}
_SAME_SLICE_TARGET_ORDER = list(_SAME_SLICE_TARGET_SPECS)


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
        return ["sarima"]

    resolved: list[str] = []
    for model in models:
        key = model.lower()
        if key == "all":
            for baseline in _DEFAULT_ALL_BASELINE_MODELS:
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
    for target_name, spec in _TARGET_SPECS.items():
        values = _torch_to_numpy_2d(getattr(dataset, spec.value_attr))
        mask = _torch_to_numpy_2d(getattr(dataset, spec.mask_attr))
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
            spec=spec,
            values=values,
            mask=mask,
            node_to_bin=node_to_bin,
        )
    return out


def _required_observed_points(
    *,
    window_length: int,
    permit: int,
    cadence_mode: str,
) -> int:
    dense_required = max(0, int(window_length) - int(permit))
    if cadence_mode == "daily":
        return dense_required
    if cadence_mode == "weekly_observed_dates":
        return int(np.ceil(dense_required / _WEEKLY_CADENCE_DAYS))
    raise ValueError(f"Unsupported cadence mode: {cadence_mode}")


def _compute_valid_node_mask_for_target(
    *,
    target_view: TargetSeriesView,
    forecast_start: int,
    input_window_length: int,
    horizon: int,
    permit_map: dict[str, dict[str, int]],
) -> np.ndarray:
    target_name = target_view.spec.target_name
    cadence_mode = target_view.spec.cadence_mode
    history_slice = target_view.mask[
        forecast_start - input_window_length : forecast_start
    ]
    target_slice = target_view.mask[forecast_start : forecast_start + horizon]
    history_counts = history_slice.sum(axis=0)
    target_counts = target_slice.sum(axis=0)
    history_threshold = _required_observed_points(
        window_length=input_window_length,
        permit=int(permit_map["input"][target_name]),
        cadence_mode=cadence_mode,
    )
    target_threshold = _required_observed_points(
        window_length=horizon,
        permit=int(permit_map["horizon"][target_name]),
        cadence_mode=cadence_mode,
    )
    return (history_counts >= history_threshold) & (target_counts >= target_threshold)


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


def _resolve_joint_observation_loss_spec(
    *,
    config: EpiForecasterConfig,
    horizon: int,
) -> JointObservationLossSpec:
    training_cfg = getattr(config, "training", None)
    loss_cfg = getattr(training_cfg, "loss", None)
    criterion = get_loss_from_config(
        loss_cfg,
        data_config=config.data,
        forecast_horizon=horizon,
    )

    return JointObservationLossSpec(
        obs_weight_sum=float(criterion.obs_weight_sum),
        ww_min_observed=int(criterion.ww_min_observed),
        hosp_min_observed=int(criterion.hosp_min_observed),
        cases_min_observed=int(criterion.cases_min_observed),
        deaths_min_observed=int(criterion.deaths_min_observed),
    )


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
    if baseline_model == "last_observed":
        return predict_with_last_observed_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            global_train_median=global_train_median,
        )
    if baseline_model == "sarima":
        return predict_with_sarima_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            global_train_median=global_train_median,
            seasonal_period=seasonal_period,
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


def _predict_var_baseline(
    *,
    batch_data: Any,
    horizon: int,
    global_medians: dict[str, float],
    seasonal_period: int,
) -> dict[str, list[BaselinePredictionResult]]:
    history_values_by_target: dict[str, np.ndarray] = {}
    history_masks_by_target: dict[str, np.ndarray] = {}
    for target_name in _SAME_SLICE_TARGET_ORDER:
        spec = _SAME_SLICE_TARGET_SPECS[target_name]
        history_values, history_mask = _extract_same_slice_history(
            batch_data,
            spec=spec,
        )
        history_values_by_target[target_name] = history_values
        history_masks_by_target[target_name] = history_mask

    batch_size = history_values_by_target[_SAME_SLICE_TARGET_ORDER[0]].shape[0]
    results = {target_name: [] for target_name in _SAME_SLICE_TARGET_ORDER}
    global_train_medians = np.asarray(
        [global_medians[target_name] for target_name in _SAME_SLICE_TARGET_ORDER],
        dtype=np.float64,
    )

    for batch_index in range(batch_size):
        train_values = np.column_stack(
            [
                history_values_by_target[target_name][batch_index]
                for target_name in _SAME_SLICE_TARGET_ORDER
            ]
        ).astype(np.float64)
        train_mask = np.column_stack(
            [
                history_masks_by_target[target_name][batch_index]
                for target_name in _SAME_SLICE_TARGET_ORDER
            ]
        ).astype(np.float64)
        per_target_result = predict_with_var_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            target_names=list(_SAME_SLICE_TARGET_ORDER),
            global_train_medians=global_train_medians,
            seasonal_period=seasonal_period,
        )
        for target_name in _SAME_SLICE_TARGET_ORDER:
            results[target_name].append(per_target_result[target_name])
    return results


def _future_temporal_covariates_from_batch(
    *,
    batch_data: Any,
    temporal_covariates: np.ndarray,
    horizon: int,
) -> np.ndarray:
    window_starts = batch_data.window_start.detach().cpu().tolist()
    history_length = int(getattr(batch_data, "hosp_hist").shape[1])
    covariates = np.asarray(temporal_covariates, dtype=np.float64)
    if covariates.ndim != 2:
        raise ValueError(f"Expected temporal covariates to be 2D, got {covariates.shape}")
    future = np.zeros((len(window_starts), horizon, covariates.shape[1]), dtype=np.float64)
    for batch_index, window_start in enumerate(window_starts):
        start = int(window_start) + history_length
        end = start + horizon
        if end > covariates.shape[0]:
            raise ValueError(
                "Temporal covariates do not cover the full forecast horizon: "
                f"end={end}, available={covariates.shape[0]}"
            )
        future[batch_index] = covariates[start:end]
    return future


def _predict_varmax_baseline(
    *,
    batch_data: Any,
    horizon: int,
    global_medians: dict[str, float],
    temporal_covariates: np.ndarray,
) -> dict[str, list[BaselinePredictionResult]]:
    history_values_by_target: dict[str, np.ndarray] = {}
    history_masks_by_target: dict[str, np.ndarray] = {}
    for target_name in _SAME_SLICE_TARGET_ORDER:
        spec = _SAME_SLICE_TARGET_SPECS[target_name]
        history_values, history_mask = _extract_same_slice_history(
            batch_data,
            spec=spec,
        )
        history_values_by_target[target_name] = history_values
        history_masks_by_target[target_name] = history_mask

    history_temporal_covariates = _torch_to_numpy_2d(
        getattr(batch_data, "temporal_covariates")
    )
    future_temporal_covariates = _future_temporal_covariates_from_batch(
        batch_data=batch_data,
        temporal_covariates=temporal_covariates,
        horizon=horizon,
    )

    batch_size = history_values_by_target[_SAME_SLICE_TARGET_ORDER[0]].shape[0]
    results = {target_name: [] for target_name in _SAME_SLICE_TARGET_ORDER}
    global_train_medians = np.asarray(
        [global_medians[target_name] for target_name in _SAME_SLICE_TARGET_ORDER],
        dtype=np.float64,
    )

    for batch_index in range(batch_size):
        train_values = np.column_stack(
            [
                history_values_by_target[target_name][batch_index]
                for target_name in _SAME_SLICE_TARGET_ORDER
            ]
        ).astype(np.float64)
        train_mask = np.column_stack(
            [
                history_masks_by_target[target_name][batch_index]
                for target_name in _SAME_SLICE_TARGET_ORDER
            ]
        ).astype(np.float64)
        future_mask = np.zeros((horizon, train_mask.shape[1]), dtype=np.float64)
        exog_train = np.concatenate(
            [train_mask, history_temporal_covariates[batch_index]],
            axis=1,
        )
        exog_future = np.concatenate(
            [future_mask, future_temporal_covariates[batch_index]],
            axis=1,
        )
        per_target_result = predict_with_varmax_fallback(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            target_names=list(_SAME_SLICE_TARGET_ORDER),
            global_train_medians=global_train_medians,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        for target_name in _SAME_SLICE_TARGET_ORDER:
            results[target_name].append(per_target_result[target_name])
    return results


def run_baseline_evaluation(
    *,
    config: EpiForecasterConfig,
    output_dir: Path,
    models: list[str] | None = None,
    config_path: str | None = None,
    split: str = "test",
    seasonal_period: int = 7,
) -> dict[str, Path]:
    return run_same_slice_baseline_evaluation(
        config=config,
        output_dir=output_dir,
        models=models,
        config_path=config_path,
        split=split,
        seasonal_period=seasonal_period,
    )


def compare_model_metrics_against_baselines(
    *,
    eval_metrics: dict[str, Any],
    baseline_results_csv: Path,
    output_csv: Path,
) -> Path:
    baseline_df = _normalize_baseline_target_metrics_csv(baseline_results_csv)

    model_target_metrics = {
        "hospitalizations": {
            "mae": eval_metrics.get("mae_hosp_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_hosp_log1p_per_100k"),
            "r2": eval_metrics.get("r2_hosp_log1p_per_100k"),
        },
        "wastewater": {
            "mae": eval_metrics.get("mae_ww_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_ww_log1p_per_100k"),
            "r2": eval_metrics.get("r2_ww_log1p_per_100k"),
        },
        "cases": {
            "mae": eval_metrics.get("mae_cases_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_cases_log1p_per_100k"),
            "r2": eval_metrics.get("r2_cases_log1p_per_100k"),
        },
        "deaths": {
            "mae": eval_metrics.get("mae_deaths_log1p_per_100k"),
            "rmse": eval_metrics.get("rmse_deaths_log1p_per_100k"),
            "r2": eval_metrics.get("r2_deaths_log1p_per_100k"),
        },
    }
    lo_baseline_by_target_metric: dict[tuple[str, str], float] = {}
    for baseline_row in baseline_df.to_dict(orient="records"):
        if str(baseline_row.get("model")) != "last_observed":
            continue
        target = str(baseline_row.get("target"))
        for metric_name in ("mae", "rmse"):
            baseline_value = baseline_row.get(f"{metric_name}_mean")
            if baseline_value is None:
                continue
            baseline_float = float(baseline_value)
            if np.isfinite(baseline_float):
                lo_baseline_by_target_metric[(target, metric_name)] = baseline_float

    rows: list[dict[str, Any]] = []
    for baseline_row in baseline_df.to_dict(orient="records"):
        target = str(baseline_row.get("target"))
        model_name = str(baseline_row["model"])
        if target in model_target_metrics:
            for metric_name in ["mae", "rmse", "r2"]:
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
            for metric_name in ("mae", "rmse"):
                lo_baseline_value = lo_baseline_by_target_metric.get((target, metric_name))
                model_value = model_target_metrics[target].get(metric_name)
                if lo_baseline_value is None or model_value is None:
                    continue
                skill_value = _safe_skill_ratio(
                    numerator=float(model_value),
                    denominator=lo_baseline_value,
                )
                if skill_value is None:
                    continue
                eval_metrics[_build_skill_metric_key(target=target, metric_name=metric_name)] = (
                    skill_value
                )
                if model_name == "last_observed":
                    rows.append(
                        {
                            "target": target,
                            "baseline_model": model_name,
                            "metric": f"skill_{metric_name}",
                            "model_value": float(skill_value),
                            "baseline_value": 1.0,
                            "delta_model_minus_baseline": float(skill_value) - 1.0,
                        }
                    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def _build_skill_metric_key(*, target: str, metric_name: str) -> str:
    target_alias = {
        "hospitalizations": "hosp",
        "wastewater": "ww",
        "cases": "cases",
        "deaths": "deaths",
    }[target]
    return f"skill_{metric_name}_{target_alias}_log1p_per_100k_vs_lo"


def _safe_skill_ratio(*, numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _normalize_baseline_target_metrics_csv(
    baseline_results_csv: Path,
) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline_results_csv)
    has_aggregate_target_metrics = {
        "target",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
    }.issubset(set(baseline_df.columns))

    if has_aggregate_target_metrics:
        return baseline_df

    return pd.DataFrame(columns=["model", "target", "mae_mean", "rmse_mean", "r2_mean"])


def _compute_metrics_from_granular_rows(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "observed_count": 0.0,
        }

    observed = pd.to_numeric(rows["observed"], errors="coerce")
    abs_error = pd.to_numeric(rows["abs_error"], errors="coerce")
    sq_error = pd.to_numeric(rows["sq_error"], errors="coerce")
    valid = np.isfinite(observed) & np.isfinite(abs_error) & np.isfinite(sq_error)
    observed = observed[valid].astype(float)
    abs_error = abs_error[valid].astype(float)
    sq_error = sq_error[valid].astype(float)
    if observed.empty:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "observed_count": 0.0,
        }

    weight_sum = float(len(observed))
    mae = float(abs_error.sum() / weight_sum)
    rmse = float(np.sqrt(sq_error.sum() / weight_sum))
    ss_res = float(sq_error.sum())
    target_weighted_sum = float(observed.sum())
    target_weighted_sq_sum = float((observed**2).sum())
    ss_tot = target_weighted_sq_sum - (target_weighted_sum**2) / max(weight_sum, 1.0)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "observed_count": weight_sum,
    }


def _extract_same_slice_history(
    batch_data: Any,
    *,
    spec: SameSliceTargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    history = getattr(batch_data, spec.history_attr, None)
    if history is None:
        raise ValueError(f"Missing history tensor for target {spec.canonical_name}")

    if spec.canonical_name == "wastewater":
        history_values = _torch_to_numpy_2d(history)
        history_mask_tensor = getattr(batch_data, "ww_hist_mask", None)
        if history_mask_tensor is None:
            raise ValueError("Missing ww_hist_mask for wastewater same-slice baseline")
        history_mask = _torch_to_numpy_2d(history_mask_tensor)
        return history_values, (history_mask > 0).astype(np.float64)

    history_np = np.asarray(history.detach().cpu().numpy(), dtype=np.float64)
    if history_np.ndim != 3 or history_np.shape[2] < 2:
        raise ValueError(f"Unexpected history shape for {spec.canonical_name}: {history_np.shape}")
    history_values = history_np[:, :, 0]
    history_mask = (history_np[:, :, 1] > 0).astype(np.float64)
    return history_values, history_mask


def _targets_dict_from_batch(batch_data: Any) -> dict[str, torch.Tensor | None]:
    return {
        "ww": getattr(batch_data, "ww_target", None),
        "hosp": getattr(batch_data, "hosp_target", None),
        "cases": getattr(batch_data, "cases_target", None),
        "deaths": getattr(batch_data, "deaths_target", None),
        "S_target": getattr(batch_data, "S_target", None),
        "I_target": getattr(batch_data, "I_target", None),
        "R_target": getattr(batch_data, "R_target", None),
        "D_target": getattr(batch_data, "D_target", None),
        "ww_mask": getattr(batch_data, "ww_target_mask", None),
        "hosp_mask": getattr(batch_data, "hosp_target_mask", None),
        "cases_mask": getattr(batch_data, "cases_target_mask", None),
        "deaths_mask": getattr(batch_data, "deaths_target_mask", None),
        "S_target_mask": getattr(batch_data, "S_target_mask", None),
        "I_target_mask": getattr(batch_data, "I_target_mask", None),
        "R_target_mask": getattr(batch_data, "R_target_mask", None),
        "D_target_mask": getattr(batch_data, "D_target_mask", None),
    }


def _build_same_slice_granular_rows(
    *,
    batch_data: Any,
    spec: SameSliceTargetSpec,
    predictions: np.ndarray,
    targets: torch.Tensor,
    weights: torch.Tensor,
    temporal_coords: list[Any] | None,
    region_ids: list[str] | None,
    region_labels: list[str] | None,
    split: str,
    model_name: str,
    selected_model: str | list[str],
    fit_status: str | list[str],
    fallback_reason: str | list[str],
    batch_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    pred = np.asarray(predictions, dtype=np.float64)
    target = (
        torch.nan_to_num(targets.detach().float(), nan=0.0).cpu().numpy().astype(np.float64)
    )
    observed_weights = (
        torch.nan_to_num(weights.detach().float(), nan=0.0).cpu().numpy().astype(np.float64)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        error = pred - target
        abs_error = np.abs(error)
        sq_error = np.square(error)
    smape_num = 2.0 * abs_error
    smape_den = np.abs(pred) + np.abs(target) + 1.0e-6

    target_nodes = batch_data.target_node.detach().cpu().tolist()
    window_starts = batch_data.window_start.detach().cpu().tolist()
    input_window_length = int(getattr(batch_data, "hosp_hist").shape[1])

    rows: list[dict[str, Any]] = []
    batch_size, horizon_size = target.shape
    source_indices = batch_indices or list(range(batch_size))
    for batch_index, source_index in enumerate(source_indices):
        selected_model_value = (
            selected_model[source_index]
            if isinstance(selected_model, list)
            else selected_model
        )
        fit_status_value = (
            fit_status[source_index] if isinstance(fit_status, list) else fit_status
        )
        fallback_reason_value = (
            fallback_reason[source_index]
            if isinstance(fallback_reason, list)
            else fallback_reason
        )
        node_id = int(target_nodes[source_index])
        window_start = int(window_starts[source_index])
        window_start_date = _format_temporal_coord(temporal_coords, window_start)
        region_id = ""
        region_label = ""
        if region_ids is not None and 0 <= node_id < len(region_ids):
            region_id = str(region_ids[node_id])
        if region_labels is not None and 0 <= node_id < len(region_labels):
            region_label = str(region_labels[node_id])
        if not region_label:
            region_label = region_id

        for horizon_index in range(horizon_size):
            if not bool(observed_weights[batch_index, horizon_index] > 0):
                continue
            pred_value = float(pred[batch_index, horizon_index])
            target_value = float(target[batch_index, horizon_index])
            abs_error_value = float(abs_error[batch_index, horizon_index])
            sq_error_value = float(sq_error[batch_index, horizon_index])
            smape_num_value = float(smape_num[batch_index, horizon_index])
            smape_den_value = float(smape_den[batch_index, horizon_index])
            if not all(
                np.isfinite(value)
                for value in (
                    pred_value,
                    target_value,
                    abs_error_value,
                    sq_error_value,
                    smape_num_value,
                    smape_den_value,
                )
            ):
                continue
            if abs_error_value > _MAX_SAFE_ABS_ERROR:
                continue
            target_index = window_start + input_window_length + horizon_index
            rows.append(
                {
                    "model": model_name,
                    "selected_model": selected_model_value,
                    "fit_status": fit_status_value,
                    "fallback_reason": fallback_reason_value,
                    "split": split,
                    "target": spec.canonical_name,
                    "node_id": node_id,
                    "region_id": region_id,
                    "region_label": region_label,
                    "window_start": window_start,
                    "window_start_date": window_start_date,
                    "horizon": horizon_index + 1,
                    "target_index": target_index,
                    "target_date": _format_temporal_coord(temporal_coords, target_index),
                    "observed": target_value,
                    "abs_error": abs_error_value,
                    "sq_error": sq_error_value,
                    "smape_num": smape_num_value,
                    "smape_den": smape_den_value,
                }
            )
    return rows


def _build_baseline_failure_rows(
    *,
    batch_data: Any,
    spec: SameSliceTargetSpec,
    results: list[BaselinePredictionResult],
    temporal_coords: list[Any] | None,
    region_ids: list[str] | None,
    region_labels: list[str] | None,
    model_name: str,
) -> list[dict[str, Any]]:
    target_nodes = batch_data.target_node.detach().cpu().tolist()
    window_starts = batch_data.window_start.detach().cpu().tolist()

    rows: list[dict[str, Any]] = []
    for batch_index, result in enumerate(results):
        if result.fit_status == "fit_success":
            continue
        node_id = int(target_nodes[batch_index])
        window_start = int(window_starts[batch_index])
        region_id = ""
        region_label = ""
        if region_ids is not None and 0 <= node_id < len(region_ids):
            region_id = str(region_ids[node_id])
        if region_labels is not None and 0 <= node_id < len(region_labels):
            region_label = str(region_labels[node_id])
        if not region_label:
            region_label = region_id
        rows.append(
            {
                "model": model_name,
                "selected_model": result.model_name,
                "target": spec.canonical_name,
                "node_id": node_id,
                "region_id": region_id,
                "region_label": region_label,
                "window_start": window_start,
                "window_start_date": _format_temporal_coord(
                    temporal_coords, window_start
                ),
                "fit_status": result.fit_status,
                "error_reason": result.fallback_reason,
            }
        )
    return rows


def _aggregate_same_slice_granular_rows(granular_df: pd.DataFrame) -> pd.DataFrame:
    if granular_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "target",
                "mae_mean",
                "rmse_mean",
                "r2_mean",
                "observed_count_mean",
            ]
        )

    aggregate_rows: list[dict[str, Any]] = []
    for (model_name, target_name), group in granular_df.groupby(["model", "target"]):
        metrics = _compute_metrics_from_granular_rows(group)
        aggregate_rows.append(
            {
                "model": str(model_name),
                "target": str(target_name),
                "mae_mean": metrics["mae"],
                "rmse_mean": metrics["rmse"],
                "r2_mean": metrics["r2"],
                "observed_count_mean": metrics["observed_count"],
            }
        )

    return pd.DataFrame(aggregate_rows)


def _safe_len(obj: Any) -> int | None:
    try:
        return int(len(obj))
    except (TypeError, ValueError):
        return None


def run_same_slice_baseline_evaluation(
    *,
    config: EpiForecasterConfig,
    output_dir: Path,
    models: list[str] | None = None,
    config_path: str | None = None,
    split: str = "test",
    seasonal_period: int = 7,
) -> dict[str, Path]:
    eval_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_models = _resolve_baseline_models(models)
    logger.info(
        "[baseline_eval] Starting same-slice baseline evaluation: split=%s models=%s output_dir=%s",
        split.lower(),
        ",".join(resolved_models),
        output_dir,
    )

    loader_build_start = time.perf_counter()
    loader, _region_embeddings = build_loader_from_config(
        config=config,
        split=split,
        batch_size=1,
        device="cpu",
    )
    logger.info(
        "[baseline_eval] Loader ready in %.2fs",
        time.perf_counter() - loader_build_start,
    )
    dataset = loader.dataset
    split_start, split_end = _resolve_split_bounds(dataset)
    target_nodes = list(dataset.target_nodes)
    total_batches = _safe_len(loader)
    logger.info(
        "[baseline_eval] Dataset summary: split_bounds=(%d,%d) target_nodes=%d loader_batches=%s forecast_horizon=%d",
        split_start,
        split_end,
        len(target_nodes),
        total_batches if total_batches is not None else "unknown",
        int(config.model.forecast_horizon),
    )
    criterion = get_loss_from_config(
        config.training.loss,
        data_config=config.data,
        forecast_horizon=config.model.forecast_horizon,
    )
    temporal_coords = list(getattr(dataset, "_temporal_coords", []))
    dataset_temporal_covariates = _torch_to_numpy_2d(
        getattr(
            dataset,
            "temporal_covariates",
            torch.zeros((len(temporal_coords), 0), dtype=torch.float32),
        )
    )
    region_ids = list(getattr(dataset, "_region_ids", []) or [])
    region_labels = list(getattr(dataset, "_region_labels", []) or [])
    target_views = _prepare_target_views(
        dataset=dataset,
        split_start=split_start,
        split_end=split_end,
        target_nodes=target_nodes,
        include_sparsity_bins=False,
    )
    global_medians = {
        target_name: _global_median_for_target(
            values=target_view.values,
            mask=target_view.mask,
            train_start=split_start,
            train_end=split_end,
        )
        for target_name, target_view in target_views.items()
    }
    logger.info(
        "[baseline_eval] Prepared target views and global medians in %.2fs",
        time.perf_counter() - eval_start,
    )

    granular_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    processed_batches = 0
    completed_series = 0
    last_progress_log = time.perf_counter()
    current_model_name = ""
    current_target_name = ""
    for batch_data in loader:
        batch_start = time.perf_counter()
        if hasattr(batch_data, "to"):
            batch_data = batch_data.to(torch.device("cpu"))
        targets_dict = _targets_dict_from_batch(batch_data)
        obs_supervision = criterion.compute_observation_supervision(
            targets_dict,
            device=torch.device("cpu"),
        )

        for model_name in resolved_models:
            current_model_name = model_name
            if model_name == "var":
                per_target_results = _predict_var_baseline(
                    batch_data=batch_data,
                    horizon=int(config.model.forecast_horizon),
                    global_medians=global_medians,
                    seasonal_period=seasonal_period,
                )
                for target_name, spec in _SAME_SLICE_TARGET_SPECS.items():
                    current_target_name = target_name
                    supervision_key = {
                        "hospitalizations": "hosp",
                        "wastewater": "ww",
                        "cases": "cases",
                        "deaths": "deaths",
                    }[target_name]
                    supervision = obs_supervision[supervision_key]
                    target_tensor = supervision["target"]
                    weights = supervision["weights"]
                    if target_tensor is None or weights is None:
                        continue

                    target_results = per_target_results[target_name]
                    predictions = np.stack(
                        [pred.predictions for pred in target_results],
                        axis=0,
                    )
                    selected_models = [pred.model_name for pred in target_results]
                    fit_statuses = [pred.fit_status for pred in target_results]
                    fallback_reasons = [
                        pred.fallback_reason for pred in target_results
                    ]
                    completed_series += len(target_results)
                    failure_rows.extend(
                        _build_baseline_failure_rows(
                            batch_data=batch_data,
                            spec=spec,
                            results=target_results,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                            model_name=model_name,
                        )
                    )
                    granular_rows.extend(
                        _build_same_slice_granular_rows(
                            batch_data=batch_data,
                            spec=spec,
                            predictions=predictions,
                            targets=target_tensor,
                            weights=weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                            split=split.lower(),
                            model_name=model_name,
                            selected_model=selected_models,
                            fit_status=fit_statuses,
                            fallback_reason=fallback_reasons,
                        )
                    )
                continue

            if model_name == "varmax":
                per_target_results = _predict_varmax_baseline(
                    batch_data=batch_data,
                    horizon=int(config.model.forecast_horizon),
                    global_medians=global_medians,
                    temporal_covariates=dataset_temporal_covariates,
                )
                for target_name, spec in _SAME_SLICE_TARGET_SPECS.items():
                    current_target_name = target_name
                    supervision_key = {
                        "hospitalizations": "hosp",
                        "wastewater": "ww",
                        "cases": "cases",
                        "deaths": "deaths",
                    }[target_name]
                    supervision = obs_supervision[supervision_key]
                    target_tensor = supervision["target"]
                    weights = supervision["weights"]
                    if target_tensor is None or weights is None:
                        continue

                    target_results = per_target_results[target_name]
                    predictions = np.stack(
                        [pred.predictions for pred in target_results],
                        axis=0,
                    )
                    selected_models = [pred.model_name for pred in target_results]
                    fit_statuses = [pred.fit_status for pred in target_results]
                    fallback_reasons = [
                        pred.fallback_reason for pred in target_results
                    ]
                    completed_series += len(target_results)
                    failure_rows.extend(
                        _build_baseline_failure_rows(
                            batch_data=batch_data,
                            spec=spec,
                            results=target_results,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                            model_name=model_name,
                        )
                    )
                    granular_rows.extend(
                        _build_same_slice_granular_rows(
                            batch_data=batch_data,
                            spec=spec,
                            predictions=predictions,
                            targets=target_tensor,
                            weights=weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                            split=split.lower(),
                            model_name=model_name,
                            selected_model=selected_models,
                            fit_status=fit_statuses,
                            fallback_reason=fallback_reasons,
                        )
                    )
                continue

            if model_name not in {"exp_smoothing", "last_observed", "sarima"}:
                continue
            for target_name, spec in _SAME_SLICE_TARGET_SPECS.items():
                current_target_name = target_name
                supervision_key = {
                    "hospitalizations": "hosp",
                    "wastewater": "ww",
                    "cases": "cases",
                    "deaths": "deaths",
                }[target_name]
                supervision = obs_supervision[supervision_key]
                target_tensor = supervision["target"]
                weights = supervision["weights"]
                if target_tensor is None or weights is None:
                    continue

                history_values, history_mask = _extract_same_slice_history(
                    batch_data,
                    spec=spec,
                )
                batch_size = history_values.shape[0]
                predictions = np.zeros_like(target_tensor.detach().cpu().numpy(), dtype=np.float64)
                selected_models: list[str] = []
                fit_statuses: list[str] = []
                fallback_reasons: list[str] = []
                target_results: list[BaselinePredictionResult] = []
                for batch_index in range(batch_size):
                    pred = _predict_univariate_baseline(
                        baseline_model=model_name,
                        train_values=history_values[batch_index],
                        train_mask=history_mask[batch_index],
                        horizon=int(target_tensor.shape[1]),
                        global_train_median=global_medians[target_name],
                        seasonal_period=seasonal_period,
                        exog_train=None,
                        exog_future=None,
                    )
                    predictions[batch_index] = pred.predictions
                    selected_models.append(pred.model_name)
                    fit_statuses.append(pred.fit_status)
                    fallback_reasons.append(pred.fallback_reason)
                    target_results.append(pred)
                completed_series += batch_size
                failure_rows.extend(
                    _build_baseline_failure_rows(
                        batch_data=batch_data,
                        spec=spec,
                        results=target_results,
                        temporal_coords=temporal_coords,
                        region_ids=region_ids,
                        region_labels=region_labels,
                        model_name=model_name,
                    )
                )
                granular_rows.extend(
                    _build_same_slice_granular_rows(
                        batch_data=batch_data,
                        spec=spec,
                        predictions=predictions,
                        targets=target_tensor,
                        weights=weights,
                        temporal_coords=temporal_coords,
                        region_ids=region_ids,
                        region_labels=region_labels,
                        split=split.lower(),
                        model_name=model_name,
                        selected_model=selected_models,
                        fit_status=fit_statuses,
                        fallback_reason=fallback_reasons,
                    )
                )
        processed_batches += 1
        now = time.perf_counter()
        should_log_progress = (
            processed_batches == 1
            or processed_batches % _BASELINE_PROGRESS_LOG_BATCH_INTERVAL == 0
            or (now - last_progress_log) >= _BASELINE_PROGRESS_LOG_TIME_INTERVAL_SEC
        )
        if should_log_progress:
            batch_suffix = (
                f"/{total_batches}" if total_batches is not None else ""
            )
            logger.info(
                "[baseline_eval] Progress: batches=%d%s series=%d rows=%d current_model=%s current_target=%s elapsed=%.2fs last_batch=%.2fs",
                processed_batches,
                batch_suffix,
                completed_series,
                len(granular_rows),
                current_model_name or "unknown",
                current_target_name or "unknown",
                now - eval_start,
                now - batch_start,
            )
            last_progress_log = now

    aggregate_start = time.perf_counter()
    granular_df = pd.DataFrame(granular_rows, columns=_BASELINE_GRANULAR_FIELDNAMES)
    failure_df = pd.DataFrame(failure_rows, columns=_BASELINE_FAILURE_FIELDNAMES)
    aggregate_df = _aggregate_same_slice_granular_rows(granular_df)
    logger.info(
        "[baseline_eval] Aggregated granular rows in %.2fs: rows=%d failures=%d aggregate_rows=%d",
        time.perf_counter() - aggregate_start,
        len(granular_df),
        len(failure_df),
        len(aggregate_df),
    )
    if not failure_df.empty:
        failure_summary = (
            failure_df.groupby(["model", "target", "error_reason"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for row in failure_summary.itertuples(index=False):
            logger.warning(
                "[baseline_eval] Baseline failures: model=%s target=%s reason=%s count=%d",
                row.model,
                row.target,
                row.error_reason,
                int(row.count),
            )

    baseline_granular = output_dir / "baseline_granular.csv"
    baseline_aggregate_metrics = output_dir / "baseline_aggregate_metrics.csv"
    baseline_metadata = output_dir / "baseline_metadata.json"
    baseline_model_usage = output_dir / "baseline_model_usage.csv"
    baseline_failures = output_dir / "baseline_failures.csv"

    granular_df.to_csv(baseline_granular, index=False)
    failure_df.to_csv(baseline_failures, index=False)
    aggregate_df.to_csv(baseline_aggregate_metrics, index=False)
    usage_source = pd.concat(
        [
            granular_df[["model", "selected_model"]],
            failure_df[["model", "selected_model"]],
        ],
        ignore_index=True,
    )
    usage_df = (
        usage_source.groupby(["model", "selected_model"], dropna=False)
        .size()
        .reset_index(name="count")
        if not usage_source.empty
        else pd.DataFrame(columns=["model", "selected_model", "count"])
    )
    usage_df.to_csv(baseline_model_usage, index=False)
    logger.info(
        "[baseline_eval] Wrote artifacts in %.2fs: granular=%s aggregate=%s usage=%s failures=%s total_elapsed=%.2fs",
        time.perf_counter() - aggregate_start,
        baseline_granular,
        baseline_aggregate_metrics,
        baseline_model_usage,
        baseline_failures,
        time.perf_counter() - eval_start,
    )

    write_granular_metadata_sidecar(
        baseline_granular,
        {
            "comparison_scope": _SAME_SLICE_COMPARISON_SCOPE,
            "split": split.lower(),
            "config_path": config_path,
        },
    )
    baseline_metadata.write_text(
        json.dumps(
            {
                "comparison_scope": _SAME_SLICE_COMPARISON_SCOPE,
                "schema_version": _SAME_SLICE_BASELINE_SCHEMA_VERSION,
                "split": split.lower(),
                "models": resolved_models,
                "config_path": config_path,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "baseline_granular": baseline_granular,
        "baseline_aggregate_metrics": baseline_aggregate_metrics,
        "baseline_model_usage": baseline_model_usage,
        "baseline_failures": baseline_failures,
        "baseline_metadata": baseline_metadata,
    }
