from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

_TARGET_AGGREGATE_COLUMNS = [
    "model",
    "target",
    "folds",
    "mae_median",
    "mae_iqr",
    "rmse_median",
    "rmse_iqr",
    "r2_median",
    "r2_iqr",
    "observed_count_median",
    "observed_count_iqr",
]

_JOINT_AGGREGATE_COLUMNS = [
    "model",
    "folds",
    "joint_obs_loss_total_median",
    "joint_obs_loss_total_iqr",
    "joint_loss_ww_median",
    "joint_loss_ww_iqr",
    "joint_loss_hosp_median",
    "joint_loss_hosp_iqr",
    "joint_loss_cases_median",
    "joint_loss_cases_iqr",
    "joint_loss_deaths_median",
    "joint_loss_deaths_iqr",
    "joint_loss_ww_weighted_median",
    "joint_loss_ww_weighted_iqr",
    "joint_loss_hosp_weighted_median",
    "joint_loss_hosp_weighted_iqr",
    "joint_loss_cases_weighted_median",
    "joint_loss_cases_weighted_iqr",
    "joint_loss_deaths_weighted_median",
    "joint_loss_deaths_weighted_iqr",
    "joint_observed_count_ww_median",
    "joint_observed_count_ww_iqr",
    "joint_observed_count_hosp_median",
    "joint_observed_count_hosp_iqr",
    "joint_observed_count_cases_median",
    "joint_observed_count_cases_iqr",
    "joint_observed_count_deaths_median",
    "joint_observed_count_deaths_iqr",
]


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numeric):
        return float("nan")
    return numeric


def _median_and_iqr(value: float) -> tuple[float, float]:
    if math.isfinite(value):
        return value, 0.0
    return float("nan"), float("nan")


def build_main_model_target_aggregate(
    eval_metrics: dict[str, Any],
    model_name: str = "epiforecaster",
) -> pd.DataFrame:
    target_to_keys = {
        "hospitalizations": {
            "mae": "mae_hosp_log1p_per_100k",
            "rmse": "rmse_hosp_log1p_per_100k",
            "r2": "r2_hosp_log1p_per_100k",
            "observed_count": "observed_count_hosp",
        },
        "wastewater": {
            "mae": "mae_ww_log1p_per_100k",
            "rmse": "rmse_ww_log1p_per_100k",
            "r2": "r2_ww_log1p_per_100k",
            "observed_count": "observed_count_ww",
        },
        "cases": {
            "mae": "mae_cases_log1p_per_100k",
            "rmse": "rmse_cases_log1p_per_100k",
            "r2": "r2_cases_log1p_per_100k",
            "observed_count": "observed_count_cases",
        },
        "deaths": {
            "mae": "mae_deaths_log1p_per_100k",
            "rmse": "rmse_deaths_log1p_per_100k",
            "r2": "r2_deaths_log1p_per_100k",
            "observed_count": "observed_count_deaths",
        },
    }

    rows: list[dict[str, Any]] = []
    for target_name, key_map in target_to_keys.items():
        row: dict[str, Any] = {
            "model": model_name,
            "target": target_name,
            "folds": 1,
        }
        for metric_name in ["mae", "rmse", "r2", "observed_count"]:
            value = _finite_or_nan(eval_metrics.get(key_map[metric_name]))
            median, iqr = _median_and_iqr(value)
            row[f"{metric_name}_median"] = median
            row[f"{metric_name}_iqr"] = iqr
        rows.append(row)

    return pd.DataFrame(rows, columns=_TARGET_AGGREGATE_COLUMNS)


def build_main_model_joint_observation_aggregate(
    eval_metrics: dict[str, Any],
    model_name: str = "epiforecaster",
) -> pd.DataFrame:
    component_to_metric_key = {
        "joint_loss_ww": "loss_ww",
        "joint_loss_hosp": "loss_hosp",
        "joint_loss_cases": "loss_cases",
        "joint_loss_deaths": "loss_deaths",
        "joint_loss_ww_weighted": "loss_ww_weighted",
        "joint_loss_hosp_weighted": "loss_hosp_weighted",
        "joint_loss_cases_weighted": "loss_cases_weighted",
        "joint_loss_deaths_weighted": "loss_deaths_weighted",
        "joint_observed_count_ww": "observed_count_ww",
        "joint_observed_count_hosp": "observed_count_hosp",
        "joint_observed_count_cases": "observed_count_cases",
        "joint_observed_count_deaths": "observed_count_deaths",
    }

    component_values = {
        component_name: _finite_or_nan(eval_metrics.get(metric_key))
        for component_name, metric_key in component_to_metric_key.items()
    }

    weighted_names = [
        "joint_loss_ww_weighted",
        "joint_loss_hosp_weighted",
        "joint_loss_cases_weighted",
        "joint_loss_deaths_weighted",
    ]
    weighted_values = [component_values[name] for name in weighted_names]
    if all(math.isfinite(value) for value in weighted_values):
        joint_obs_total = float(sum(weighted_values))
    else:
        joint_obs_total = float("nan")

    row: dict[str, Any] = {"model": model_name, "folds": 1}

    total_median, total_iqr = _median_and_iqr(joint_obs_total)
    row["joint_obs_loss_total_median"] = total_median
    row["joint_obs_loss_total_iqr"] = total_iqr

    ordered_components = [
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
    for component_name in ordered_components:
        value = component_values[component_name]
        median, iqr = _median_and_iqr(value)
        row[f"{component_name}_median"] = median
        row[f"{component_name}_iqr"] = iqr

    return pd.DataFrame([row], columns=_JOINT_AGGREGATE_COLUMNS)


def write_main_model_aggregate_csvs(
    run_dir: Path,
    split: str,
    eval_metrics: dict[str, Any],
    model_name: str = "epiforecaster",
) -> dict[str, Path]:
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'")

    run_dir.mkdir(parents=True, exist_ok=True)
    target_df = build_main_model_target_aggregate(
        eval_metrics=eval_metrics,
        model_name=model_name,
    )
    joint_df = build_main_model_joint_observation_aggregate(
        eval_metrics=eval_metrics,
        model_name=model_name,
    )

    target_path = run_dir / f"{split_key}_main_model_aggregate_metrics.csv"
    joint_path = run_dir / f"{split_key}_main_model_joint_loss_aggregate.csv"
    target_df.to_csv(target_path, index=False)
    joint_df.to_csv(joint_path, index=False)

    return {
        f"{split_key}_main_model_aggregate_metrics": target_path,
        f"{split_key}_main_model_joint_loss_aggregate": joint_path,
    }
