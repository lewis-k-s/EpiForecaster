from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

_NATIVE_TARGET_NAMES = {
    "ww": "wastewater",
    "cases": "cases",
    "hosp": "hospitalizations",
    "deaths": "deaths",
}
_NATIVE_INTERVAL_LEVELS = ("50", "80", "95")
_NATIVE_AGGREGATE_COLUMNS = [
    "aggregation",
    "target",
    "source_type",
    "source_id",
    "horizon",
    "observed_count",
    "mae_log1p_per_100k",
    "rmse_log1p_per_100k",
    "raw_count_mae",
    "nll",
    "crps",
    "wis",
    "wis_dispersion",
    "wis_underprediction",
    "wis_overprediction",
    "wis_point",
    *[
        field
        for level in _NATIVE_INTERVAL_LEVELS
        for field in (f"coverage_{level}", f"interval_width_{level}")
    ],
]

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


@dataclass
class _NativeMetricSums:
    count: float = 0.0
    observed_count: int = 0
    abs_error: float = 0.0
    sq_error: float = 0.0
    raw_abs_error: float = 0.0
    nll: float = 0.0
    crps: float = 0.0
    wis: float = 0.0
    wis_dispersion: float = 0.0
    wis_underprediction: float = 0.0
    wis_overprediction: float = 0.0
    wis_point: float = 0.0
    coverage: dict[str, float] = field(default_factory=dict)
    interval_width: dict[str, float] = field(default_factory=dict)


class NativeAggregateAccumulator:
    """Online native metrics keyed by head, horizon, and source identity."""

    def __init__(self, *, origin_block_days: int = 0) -> None:
        self.origin_block_days = origin_block_days
        self._groups: dict[tuple[str, str, str, str, int], _NativeMetricSums] = {}

    def update(
        self,
        *,
        head: str,
        source_types: list[str],
        source_ids: list[str],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        raw_predictions: torch.Tensor,
        raw_targets: torch.Tensor,
        weights: torch.Tensor,
        nll: torch.Tensor,
        crps: torch.Tensor,
        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]],
        wis_components: dict[str, torch.Tensor],
        origin_indices: list[int] | None = None,
    ) -> None:
        target_name = _NATIVE_TARGET_NAMES[head]
        arrays = {
            "prediction": predictions.detach().float().cpu(),
            "target": targets.detach().float().cpu(),
            "raw_prediction": raw_predictions.detach().float().cpu(),
            "raw_target": raw_targets.detach().float().cpu(),
            "weights": weights.detach().float().cpu(),
            "nll": nll.detach().float().cpu(),
            "crps": crps.detach().float().cpu(),
            **{
                name: value.detach().float().cpu()
                for name, value in wis_components.items()
            },
        }
        interval_cpu = {
            level: (low.detach().float().cpu(), high.detach().float().cpu())
            for level, (low, high) in intervals.items()
        }
        batch_size, horizon_size = targets.shape
        if len(source_types) != batch_size or len(source_ids) != batch_size:
            raise ValueError("native aggregate identities do not match tensor rows")
        if self.origin_block_days and (
            origin_indices is None or len(origin_indices) != batch_size
        ):
            raise ValueError("origin-block aggregation requires one origin per row")
        for row in range(batch_size):
            for horizon_index in range(horizon_size):
                weight = float(arrays["weights"][row, horizon_index].item())
                if weight <= 0:
                    continue
                keys = (
                    (
                        "source_horizon",
                        target_name,
                        source_types[row],
                        source_ids[row],
                        horizon_index + 1,
                    ),
                    (
                        "head_horizon",
                        target_name,
                        "",
                        "",
                        horizon_index + 1,
                    ),
                    ("head_all", target_name, "", "", 0),
                )
                if self.origin_block_days:
                    assert origin_indices is not None
                    start = (
                        origin_indices[row] // self.origin_block_days
                    ) * self.origin_block_days
                    keys += (
                        ("origin_block", target_name, "origin_index", str(start), 0),
                    )
                prediction = float(arrays["prediction"][row, horizon_index].item())
                target = float(arrays["target"][row, horizon_index].item())
                raw_prediction = float(
                    arrays["raw_prediction"][row, horizon_index].item()
                )
                raw_target = float(arrays["raw_target"][row, horizon_index].item())
                for key in keys:
                    sums = self._groups.setdefault(key, _NativeMetricSums())
                    sums.count += weight
                    sums.observed_count += 1
                    sums.abs_error += abs(prediction - target) * weight
                    sums.sq_error += (prediction - target) ** 2 * weight
                    sums.raw_abs_error += abs(raw_prediction - raw_target) * weight
                    for name in (
                        "nll",
                        "crps",
                        "wis",
                        "wis_dispersion",
                        "wis_underprediction",
                        "wis_overprediction",
                        "wis_point",
                    ):
                        source_name = (
                            name.removeprefix("wis_") if name != "wis" else "wis"
                        )
                        sums_value = (
                            arrays[source_name][row, horizon_index]
                            if source_name in arrays
                            else arrays[name][row, horizon_index]
                        )
                        setattr(
                            sums,
                            name,
                            getattr(sums, name) + float(sums_value.item()) * weight,
                        )
                    for level in _NATIVE_INTERVAL_LEVELS:
                        bounds = interval_cpu.get(level)
                        if bounds is None:
                            continue
                        low = float(bounds[0][row, horizon_index].item())
                        high = float(bounds[1][row, horizon_index].item())
                        sums.coverage[level] = sums.coverage.get(level, 0.0) + (
                            float(low <= target <= high) * weight
                        )
                        sums.interval_width[level] = (
                            sums.interval_width.get(level, 0.0)
                            + max(0.0, high - low) * weight
                        )

    def finalize(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for key in sorted(self._groups):
            aggregation, target, source_type, source_id, horizon = key
            sums = self._groups[key]
            denominator = max(sums.count, 1e-12)
            row: dict[str, Any] = {
                "aggregation": aggregation,
                "target": target,
                "source_type": source_type,
                "source_id": source_id,
                "horizon": horizon,
                "observed_count": sums.observed_count,
                "mae_log1p_per_100k": sums.abs_error / denominator,
                "rmse_log1p_per_100k": math.sqrt(sums.sq_error / denominator),
                "raw_count_mae": sums.raw_abs_error / denominator,
                "nll": sums.nll / denominator,
                "crps": sums.crps / denominator,
                "wis": sums.wis / denominator,
                "wis_dispersion": sums.wis_dispersion / denominator,
                "wis_underprediction": sums.wis_underprediction / denominator,
                "wis_overprediction": sums.wis_overprediction / denominator,
                "wis_point": sums.wis_point / denominator,
            }
            for level in _NATIVE_INTERVAL_LEVELS:
                row[f"coverage_{level}"] = (
                    sums.coverage[level] / denominator
                    if level in sums.coverage
                    else float("nan")
                )
                row[f"interval_width_{level}"] = (
                    sums.interval_width[level] / denominator
                    if level in sums.interval_width
                    else float("nan")
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=_NATIVE_AGGREGATE_COLUMNS)


def write_native_aggregate_outputs(
    *,
    output_dir: Path,
    split: str,
    aggregate: pd.DataFrame,
) -> dict[str, Path]:
    """Write the canonical native aggregate CSV and head-level WIS JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_key = split.strip().lower()
    aggregate_path = output_dir / f"{split_key}_native_aggregate_metrics.csv"
    summary_path = output_dir / f"{split_key}_native_wis_summary.json"
    aggregate.to_csv(aggregate_path, index=False)
    head_rows = aggregate[aggregate["aggregation"] == "head_all"]
    summary = {
        str(row.target): {
            key: float(getattr(row, key))
            for key in (
                "wis",
                "wis_dispersion",
                "wis_underprediction",
                "wis_overprediction",
                "wis_point",
            )
        }
        for row in head_rows.itertuples(index=False)
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "native_aggregate_metrics": aggregate_path,
        "native_wis_summary": summary_path,
    }


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
