"""Equal-target validation scores used for checkpoint selection and endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

_TARGET_MAE_METRIC_KEYS = [
    "mae_ww_log1p_per_100k",
    "mae_hosp_log1p_per_100k",
    "mae_cases_log1p_per_100k",
    "mae_deaths_log1p_per_100k",
]
_TARGET_CRPS_METRIC_KEYS = [
    "crps_ww_log1p_per_100k",
    "crps_hosp_log1p_per_100k",
    "crps_cases_log1p_per_100k",
    "crps_deaths_log1p_per_100k",
]
_TARGET_NLL_METRIC_KEYS = [
    "nll_ww_log1p_per_100k",
    "nll_hosp_log1p_per_100k",
    "nll_cases_log1p_per_100k",
    "nll_deaths_log1p_per_100k",
]
_SELECTION_METRIC_NAME = "mean_target_mae"
_CRPS_SELECTION_METRIC_NAME = "mean_target_crps"
_NLL_SELECTION_METRIC_NAME = "mean_target_nll"


def compute_mean_target_mae(metrics: dict[str, Any]) -> float:
    """Compute the locked validation endpoint score from per-target MAEs."""
    values: list[float] = []
    for key in _TARGET_MAE_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            values.append(value_float)
    if not values:
        return float("inf")
    return float(np.mean(values))


def compute_mean_target_crps(metrics: dict[str, Any]) -> float:
    """Compute the validation endpoint score from per-target CRPS."""
    values: list[float] = []
    for key in _TARGET_CRPS_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            values.append(value_float)
    if not values:
        return float("inf")
    return float(np.mean(values))


def compute_mean_target_nll(metrics: dict[str, Any]) -> float:
    """Compute the validation endpoint score from per-target NLL."""
    values: list[float] = []
    for key in _TARGET_NLL_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            values.append(value_float)
    if not values:
        return float("inf")
    return float(np.mean(values))


def compute_validation_selection_score(
    metrics: dict[str, Any],
    *,
    observation_loss: str,
    selection_metric: str = "auto",
    required_heads: tuple[str, ...] | list[str] = ("ww", "hosp", "cases", "deaths"),
) -> tuple[float, str]:
    """Choose the checkpoint-selection score for the current observation loss."""
    if selection_metric != "auto":
        metric_names = {
            "mean_target_wis": "wis",
            "mean_target_crps": "crps",
            "mean_target_nll": "nll",
            "mean_target_mae": "mae",
        }
        if selection_metric not in metric_names or not required_heads:
            raise ValueError(
                "An explicit selection metric requires a valid metric and heads"
            )
        values = []
        for head in required_heads:
            key = f"{metric_names[selection_metric]}_{head}_log1p_per_100k"
            value = metrics.get(key)
            if (
                value is None
                or not np.isfinite(float(value))
                or metrics.get(f"invalid_prediction_count_{head}", 0) != 0
            ):
                raise ValueError(
                    f"Required checkpoint-selection score is invalid: {key}"
                )
            values.append(float(value))
        return float(np.mean(values)), selection_metric
    selection_score = compute_mean_target_mae(metrics)
    selection_metric = _SELECTION_METRIC_NAME
    if observation_loss in {"nll", "crps", "wis"}:
        crps_selection_score = compute_mean_target_crps(metrics)
        if np.isfinite(crps_selection_score):
            selection_score = crps_selection_score
            selection_metric = _CRPS_SELECTION_METRIC_NAME
        elif observation_loss == "nll":
            nll_selection_score = compute_mean_target_nll(metrics)
            if np.isfinite(nll_selection_score):
                selection_score = nll_selection_score
                selection_metric = _NLL_SELECTION_METRIC_NAME
    return selection_score, selection_metric
