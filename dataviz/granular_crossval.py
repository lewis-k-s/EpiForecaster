from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataviz.granular_comparison import (
    _format_target_label,
    _load_granular_csv,
    _ordered_targets,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


@dataclass(frozen=True)
class CrossvalGranularRun:
    fold: int
    seed: int | None
    run_dir: Path
    granular_csv_path: Path


def _dataframe_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def _append_frame(path: Path, df: pd.DataFrame) -> None:
    write_header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=write_header)


def _read_fold_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"region_id": str})
    if "target_date" in df.columns:
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    return df


def _load_sidecar(granular_csv_path: Path) -> dict[str, Any]:
    sidecar_path = granular_csv_path.with_suffix(f"{granular_csv_path.suffix}.meta.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing granular metadata sidecar: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Granular metadata sidecar is not a JSON object: {sidecar_path}")
    return payload


def _load_annotated_granular_df(
    run: CrossvalGranularRun,
    *,
    split: str,
) -> pd.DataFrame:
    df = _load_granular_csv(run.granular_csv_path)
    sidecar = _load_sidecar(run.granular_csv_path)
    sidecar_split = str(sidecar.get("split", "")).strip().lower()
    if sidecar_split and sidecar_split != split:
        raise ValueError(
            f"Granular sidecar split mismatch for {run.granular_csv_path}: "
            f"expected {split}, got {sidecar_split}"
        )
    sidecar_seed = sidecar.get("training_seed")
    if (
        run.seed is not None
        and sidecar_seed is not None
        and not pd.isna(sidecar_seed)
        and int(sidecar_seed) != int(run.seed)
    ):
        raise ValueError(
            f"Granular sidecar seed mismatch for {run.granular_csv_path}: "
            f"run seed={run.seed}, sidecar seed={sidecar_seed}"
        )

    df["fold"] = int(run.fold)
    df["seed"] = run.seed
    df["smape"] = df["smape_num"] / df["smape_den"].clip(lower=1e-6)
    logger.info(
        "Loaded granular run: path=%s fold=%d seed=%s rows=%d memory_mb=%.2f",
        run.granular_csv_path,
        run.fold,
        run.seed,
        len(df),
        _dataframe_memory_mb(df),
    )
    return df


def _build_fold_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(["fold", "seed", *group_cols], dropna=False)
        .agg(
            count=("abs_error", "size"),
            observed_mean=("observed", "mean"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            smape=("smape", "mean"),
        )
        .reset_index()
    )
    grouped["rmse"] = grouped["mse"].clip(lower=0.0).map(math.sqrt)
    grouped = grouped.drop(columns=["mse"])
    return _sort_target_frame(grouped)


def _aggregate_fold_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False, observed=False)
        .agg(
            folds=("fold", "nunique"),
            count_mean=("count", "mean"),
            count_std=("count", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            observed_mean_mean=("observed_mean", "mean"),
            observed_mean_std=(
                "observed_mean",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            smape_mean=("smape", "mean"),
            smape_std=(
                "smape",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
        )
        .reset_index()
    )
    return _sort_target_frame(grouped)


def _log_table_build(
    *,
    table_name: str,
    phase: str,
    group_cols: list[str],
    input_rows: int,
    output_rows: int,
    df: pd.DataFrame,
) -> None:
    logger.info(
        "Granular %s table=%s groups=%s input_rows=%d output_rows=%d memory_mb=%.2f",
        phase,
        table_name,
        group_cols,
        input_rows,
        output_rows,
        _dataframe_memory_mb(df),
    )


def _sort_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns or df.empty:
        return df
    ordered_targets = _ordered_targets(df["target"])
    sorted_df = df.copy()
    sorted_df["target"] = pd.Categorical(
        sorted_df["target"],
        categories=ordered_targets,
        ordered=True,
    )
    sort_cols = ["target"]
    for candidate in ["horizon", "target_date", "region_label", "region_id", "node_id"]:
        if candidate in sorted_df.columns:
            sort_cols.append(candidate)
    return sorted_df.sort_values(sort_cols).reset_index(drop=True)


def _save_horizon_curve(horizon_df: pd.DataFrame, output_path: Path) -> Path:
    targets = _ordered_targets(horizon_df["target"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for target in targets:
        target_df = horizon_df[horizon_df["target"] == target].sort_values("horizon")
        if target_df.empty:
            continue
        ax.plot(
            target_df["horizon"],
            target_df["mae_mean"],
            marker="o",
            linewidth=2.0,
            label=_format_target_label(str(target)),
        )
        lower = (target_df["mae_mean"] - target_df["mae_std"]).clip(lower=0.0)
        upper = target_df["mae_mean"] + target_df["mae_std"]
        ax.fill_between(
            target_df["horizon"],
            lower,
            upper,
            alpha=0.15,
        )
    ax.set_title("Cross-val MAE by forecast horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.legend(title="Target")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_rolling_time_curve(time_df: pd.DataFrame, output_path: Path) -> Path:
    targets = _ordered_targets(time_df["target"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for target in targets:
        target_df = time_df[time_df["target"] == target].sort_values("target_date")
        if target_df.empty:
            continue
        ax.plot(
            target_df["target_date"],
            target_df["mae_mean"],
            linewidth=2.0,
            label=_format_target_label(str(target)),
        )
        lower = (target_df["mae_mean"] - target_df["mae_std"]).clip(lower=0.0)
        upper = target_df["mae_mean"] + target_df["mae_std"]
        ax.fill_between(
            target_df["target_date"],
            lower,
            upper,
            alpha=0.15,
        )
    ax.set_title("Cross-val MAE over target date")
    ax.set_xlabel("Target date")
    ax.set_ylabel("MAE")
    ax.legend(title="Target")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_region_summary(region_df: pd.DataFrame, output_path: Path, *, top_n: int = 8) -> Path:
    targets = _ordered_targets(region_df["target"])
    fig, axes = plt.subplots(
        len(targets),
        2,
        figsize=(14, max(4, len(targets) * 3.5)),
        squeeze=False,
    )
    for row_idx, target in enumerate(targets):
        target_df = region_df[region_df["target"] == target].copy()
        if target_df.empty:
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)
            continue

        best_df = target_df.nsmallest(top_n, "mae_mean").sort_values("mae_mean")
        worst_df = target_df.nlargest(top_n, "mae_mean").sort_values("mae_mean")

        axes[row_idx, 0].barh(best_df["region_label"], best_df["mae_mean"], color="#2ca25f")
        axes[row_idx, 1].barh(
            worst_df["region_label"],
            worst_df["mae_mean"],
            color="#de2d26",
        )

        axes[row_idx, 0].set_title(f"{_format_target_label(str(target))}: Best regions")
        axes[row_idx, 1].set_title(f"{_format_target_label(str(target))}: Worst regions")
        axes[row_idx, 0].set_xlabel("Mean MAE")
        axes[row_idx, 1].set_xlabel("Mean MAE")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analyze_crossval_granular(
    *,
    runs: list[CrossvalGranularRun],
    split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    split_key = split.strip().lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs:
        raise ValueError("No granular runs provided for cross-val analysis")

    table_specs = {
        "target": ["target"],
        "horizon": ["target", "horizon"],
        "region": ["target", "node_id", "region_id", "region_label"],
        "time": ["target", "target_date"],
        "region_time": ["target", "node_id", "region_id", "region_label", "target_date"],
    }

    table_paths: dict[str, Path] = {}
    fold_paths: dict[str, Path] = {}
    aggregate_paths: dict[str, Path] = {}
    for name in table_specs:
        fold_path = output_dir / f"{split_key}_crossval_{name}_fold_metrics.csv"
        aggregate_path = output_dir / f"{split_key}_crossval_{name}_aggregates.csv"
        fold_path.unlink(missing_ok=True)
        aggregate_path.unlink(missing_ok=True)
        fold_paths[name] = fold_path
        aggregate_paths[name] = aggregate_path
        table_paths[f"{name}_fold_metrics"] = fold_path
        table_paths[f"{name}_aggregates"] = aggregate_path

    total_rows = 0
    for run in runs:
        run_df = _load_annotated_granular_df(run, split=split_key)
        total_rows += len(run_df)
        for name, group_cols in table_specs.items():
            fold_df = _build_fold_table(run_df, group_cols)
            _log_table_build(
                table_name=name,
                phase="fold-build",
                group_cols=group_cols,
                input_rows=len(run_df),
                output_rows=len(fold_df),
                df=fold_df,
            )
            _append_frame(fold_paths[name], fold_df)
            del fold_df
        del run_df

    aggregate_frames: dict[str, pd.DataFrame] = {}
    for name, group_cols in table_specs.items():
        fold_df = _read_fold_metrics_csv(fold_paths[name])
        aggregate_df = _aggregate_fold_table(fold_df, group_cols)
        _log_table_build(
            table_name=name,
            phase="aggregate-build",
            group_cols=group_cols,
            input_rows=len(fold_df),
            output_rows=len(aggregate_df),
            df=aggregate_df,
        )
        aggregate_df.to_csv(aggregate_paths[name], index=False)
        aggregate_frames[name] = aggregate_df
        del fold_df

    plots = {
        "horizon_curve": _save_horizon_curve(
            aggregate_frames["horizon"],
            output_dir / f"{split_key}_crossval_horizon_curve.png",
        ),
        "rolling_time_curve": _save_rolling_time_curve(
            aggregate_frames["time"],
            output_dir / f"{split_key}_crossval_rolling_time_curve.png",
        ),
        "region_summary": _save_region_summary(
            aggregate_frames["region"],
            output_dir / f"{split_key}_crossval_region_summary.png",
        ),
    }
    logger.info(
        "Crossval granular analysis complete: runs=%d rows=%d tables=%s plots=%s",
        len(runs),
        total_rows,
        {name: str(path) for name, path in table_paths.items()},
        {name: str(path) for name, path in plots.items()},
    )

    return {
        "rows": int(total_rows),
        "runs": len(runs),
        "tables": table_paths,
        "plots": plots,
    }
