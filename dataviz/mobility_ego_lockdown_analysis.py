"""Vectorized ego-neighborhood lockdown analysis.

This analysis keeps one row per valid ``(window_start, target_municipality)``
instead of collapsing all municipalities into one row per time window. It reads
canonical Zarr arrays directly and reproduces the ego-mask behavior used by
``EpiDataset`` without constructing dataset items.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

logger = logging.getLogger("mobility_ego_lockdown_analysis")

MIN_REGRESSION_TIMEPOINTS = 2
DEFAULT_MOBILITY_THRESHOLD = 0.0


@dataclass(frozen=True)
class LockdownPeriod:
    """Definition of a named restriction period."""

    name: str
    start: str
    end: str

    def bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.start), pd.Timestamp(self.end)


LOCKDOWN_PERIODS = [
    LockdownPeriod(
        name="Spain-wide lockdown",
        start="2020-03-15",
        end="2020-06-21",
    ),
    LockdownPeriod(
        name="Catalunya perimeter+weekend",
        start="2020-10-30",
        end="2020-11-14",
    ),
    LockdownPeriod(
        name="Catalunya perimeter+municipal",
        start="2021-01-07",
        end="2021-01-18",
    ),
]


@dataclass
class EgoAnalysisConfig:
    """Runtime configuration for the ego-neighborhood analyzer."""

    config_path: Path
    output_dir: Path
    baseline_start: str = "2020-03-01"
    baseline_end: str = "2020-03-14"
    mobility_threshold: float = DEFAULT_MOBILITY_THRESHOLD
    window_length: int | None = None
    window_stride: int = 1
    min_timepoints: int = MIN_REGRESSION_TIMEPOINTS
    max_windows: int | None = None
    log_every: int = 25


def open_canonical_dataset(config: EpiForecasterConfig) -> xr.Dataset:
    """Open the configured canonical Zarr and select the configured run_id."""
    ds = xr.open_zarr(config.data.dataset_path, chunks={"run_id": 1}, zarr_format=2)
    run_id = config.data.run_id
    if not run_id:
        raise ValueError("config.data.run_id must be set")
    if "run_id" not in ds.coords:
        raise ValueError("Canonical dataset must contain a run_id coordinate")

    available_runs = [str(value).strip() for value in ds.run_id.values]
    if run_id not in available_runs:
        raise ValueError(
            f"run_id '{run_id}' not found in {config.data.dataset_path}; "
            f"available runs: {available_runs[:10]}"
        )
    return ds.sel(run_id=ds.run_id.str.strip() == run_id).squeeze(drop=True)


def load_analysis_arrays(ds: xr.Dataset) -> dict[str, Any]:
    """Load the canonical arrays required by the vectorized analysis."""
    required = {"cases", "population", "mobility", REGION_COORD, TEMPORAL_COORD}
    missing = required - set(ds.variables)
    if missing:
        raise ValueError(f"Canonical dataset missing required variables: {missing}")

    cases_da = ds["cases"]
    if cases_da.ndim == 3:
        cases_da = cases_da.squeeze(drop=True)
    cases = cases_da.transpose(TEMPORAL_COORD, REGION_COORD).values.astype(np.float32)

    if "cases_mask" in ds:
        mask_da = ds["cases_mask"]
        if mask_da.ndim == 3:
            mask_da = mask_da.squeeze(drop=True)
        cases_mask = (
            mask_da.transpose(TEMPORAL_COORD, REGION_COORD).values.astype(bool)
        )
    else:
        cases_mask = np.isfinite(cases)
    cases_mask &= np.isfinite(cases)

    mobility = resolve_mobility_array(ds["mobility"]).astype(np.float32)
    mobility = np.nan_to_num(mobility, nan=0.0, posinf=0.0, neginf=0.0)

    mobility_time_mask = None
    if "mobility_time_mask" in ds:
        mobility_time_mask = ds["mobility_time_mask"].values.astype(bool)

    return {
        "cases": cases,
        "cases_mask": cases_mask,
        "population": ds["population"].values.astype(np.float32),
        "mobility": mobility,
        "mobility_time_mask": mobility_time_mask,
        "dates": pd.DatetimeIndex(ds[TEMPORAL_COORD].values),
        "region_ids": np.asarray([str(value) for value in ds[REGION_COORD].values]),
    }


def resolve_mobility_array(mobility_da: xr.DataArray) -> np.ndarray:
    """Return mobility in ``(date, origin, destination)`` order."""
    dims = list(mobility_da.dims)
    time_dim = TEMPORAL_COORD if TEMPORAL_COORD in dims else None
    if time_dim is None:
        raise ValueError(f"Mobility data missing {TEMPORAL_COORD} dimension")

    if "origin" in dims and "destination" in dims:
        return mobility_da.transpose(time_dim, "origin", "destination").values
    if "origin" in dims and "target" in dims:
        return mobility_da.transpose(time_dim, "origin", "target").values
    if dims.count(REGION_COORD) == 2:
        return mobility_da.transpose(time_dim, REGION_COORD, REGION_COORD).values

    raise ValueError(
        "Mobility data must include ('origin', 'destination'), ('origin', 'target'), "
        f"or two '{REGION_COORD}' dimensions"
    )


def compute_window_starts(
    num_timesteps: int, window_length: int, window_stride: int
) -> np.ndarray:
    """Return stride-based history-window starts."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if window_stride <= 0:
        raise ValueError("window_stride must be positive")
    if num_timesteps < window_length:
        return np.array([], dtype=np.int64)
    return np.arange(0, num_timesteps - window_length + 1, window_stride, dtype=np.int64)


def build_ego_mask(
    mobility: np.ndarray,
    mobility_threshold: float,
    mobility_time_mask: np.ndarray | None = None,
    include_self: bool = True,
) -> np.ndarray:
    """Build time-varying incoming ego masks.

    For exploratory raw canonical analysis, a non-positive threshold means
    positive-flow edges are included. Positive thresholds use the canonical
    ``>= threshold`` comparison.
    """
    if mobility_threshold <= 0:
        mask = mobility > 0
    else:
        mask = mobility >= mobility_threshold

    mask = np.asarray(mask, dtype=bool)
    if mobility_time_mask is not None:
        time_mask = np.asarray(mobility_time_mask, dtype=bool)
        if time_mask.shape != (mask.shape[0],):
            raise ValueError(
                f"mobility_time_mask shape {time_mask.shape} does not match "
                f"time dimension {mask.shape[0]}"
            )
        mask &= time_mask[:, None, None]

    if include_self:
        diag = np.arange(mask.shape[1])
        mask[:, diag, diag] = True
    return mask


def compute_valid_history_mask(
    cases_mask: np.ndarray,
    starts: np.ndarray,
    window_length: int,
    missing_permit: int,
) -> np.ndarray:
    """Compute per-window, per-target validity from case observation masks."""
    observed = np.asarray(cases_mask, dtype=np.int32)
    cumsum = np.concatenate(
        [np.zeros((1, observed.shape[1]), dtype=np.int32), np.cumsum(observed, axis=0)],
        axis=0,
    )
    counts = cumsum[window_length:] - cumsum[:-window_length]
    counts = counts[starts]
    threshold = max(0, window_length - int(missing_permit))
    return counts >= threshold


def compute_mobility_reduction(
    mobility: np.ndarray,
    dates: pd.DatetimeIndex,
    baseline_start: str,
    baseline_end: str,
) -> pd.Series:
    """Compute smoothed mobility reduction from a baseline period."""
    start = pd.Timestamp(datetime.fromisoformat(baseline_start))
    end = pd.Timestamp(datetime.fromisoformat(baseline_end))
    baseline_mask = (dates >= start) & (dates <= end)
    if baseline_mask.sum() == 0:
        raise ValueError(f"No dates found in baseline period {baseline_start} to {baseline_end}")

    daily_totals = np.nansum(mobility.astype(np.float64), axis=(1, 2))
    daily = pd.Series(daily_totals, index=dates)
    rolling_baseline = daily.rolling(window=14, center=True, min_periods=7).median()
    rolling_baseline[baseline_mask] = daily[baseline_mask]
    valid_baseline = rolling_baseline.where(rolling_baseline > 0)
    reduction = (valid_baseline - daily) / valid_baseline * 100.0
    reduction = reduction.replace([np.inf, -np.inf], np.nan)
    return reduction.rolling(window=7, center=True, min_periods=1).mean()


def vectorized_window_regression(
    cases_window: np.ndarray,
    cases_mask_window: np.ndarray,
    population: np.ndarray,
    ego_mask_window: np.ndarray,
    min_timepoints: int = MIN_REGRESSION_TIMEPOINTS,
) -> dict[str, np.ndarray]:
    """Fit ``global_trend ~ ego_trend`` for all targets in one window."""
    cases_window = np.asarray(cases_window, dtype=np.float32)
    cases_mask_window = np.asarray(cases_mask_window, dtype=bool)
    population = np.asarray(population, dtype=np.float32)
    ego_mask_window = np.asarray(ego_mask_window, dtype=bool)

    valid_pop = np.isfinite(population) & (population > 0)
    valid_cases = cases_mask_window & np.isfinite(cases_window) & valid_pop[None, :]
    safe_cases = np.where(valid_cases, cases_window, 0.0)

    weights = np.where(valid_pop, population, 0.0).astype(np.float64)
    weighted_cases = safe_cases.astype(np.float64) * weights[None, :]
    global_den = np.einsum("tn,n->t", valid_cases.astype(np.float64), weights)
    global_num = weighted_cases.sum(axis=1)
    global_trend = np.divide(
        global_num,
        global_den,
        out=np.full(global_num.shape, np.nan, dtype=np.float64),
        where=global_den > 0,
    )

    valid_float = valid_cases.astype(np.float32)
    ego_counts = np.einsum("to,tod->td", valid_float, ego_mask_window, optimize=True)
    ego_sums = np.einsum("to,tod->td", safe_cases, ego_mask_window, optimize=True)
    ego_trend = np.divide(
        ego_sums,
        ego_counts,
        out=np.full(ego_sums.shape, np.nan, dtype=np.float32),
        where=ego_counts > 0,
    )

    y = global_trend.astype(np.float32)[:, None]
    valid_xy = np.isfinite(ego_trend) & np.isfinite(y)
    n_timepoints = valid_xy.sum(axis=0).astype(np.int32)

    x_sum = np.where(valid_xy, ego_trend, 0.0).sum(axis=0)
    y_sum = np.where(valid_xy, y, 0.0).sum(axis=0)
    x_mean = np.divide(
        x_sum,
        n_timepoints,
        out=np.full_like(x_sum, np.nan, dtype=np.float32),
        where=n_timepoints > 0,
    )
    y_mean = np.divide(
        y_sum,
        n_timepoints,
        out=np.full_like(y_sum, np.nan, dtype=np.float32),
        where=n_timepoints > 0,
    )

    dx = np.where(valid_xy, ego_trend - x_mean[None, :], 0.0)
    dy = np.where(valid_xy, y - y_mean[None, :], 0.0)
    ss_xx = np.sum(dx * dx, axis=0)
    ss_xy = np.sum(dx * dy, axis=0)
    ss_yy = np.sum(dy * dy, axis=0)
    valid_fit = (n_timepoints >= min_timepoints) & (ss_xx > 0) & (ss_yy > 0)

    slope = np.divide(
        ss_xy,
        ss_xx,
        out=np.full_like(ss_xy, np.nan, dtype=np.float32),
        where=valid_fit,
    )
    intercept = y_mean - slope * x_mean
    rss = ss_yy - slope * ss_xy
    r2 = 1.0 - np.divide(
        rss,
        ss_yy,
        out=np.full_like(ss_yy, np.nan, dtype=np.float32),
        where=valid_fit,
    )
    r2 = np.where(valid_fit, np.clip(r2, -np.inf, 1.0), np.nan)

    neighbor_counts = ego_mask_window.sum(axis=1).astype(np.float32)
    return {
        "slope": slope.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "r2": r2.astype(np.float32),
        "n_timepoints": n_timepoints,
        "n_neighbors_mean": neighbor_counts.mean(axis=0),
        "n_neighbors_min": neighbor_counts.min(axis=0),
        "n_neighbors_max": neighbor_counts.max(axis=0),
    }


def classify_lockdown(date: pd.Timestamp) -> tuple[str, str]:
    """Classify a date by lockdown status and named period."""
    for period in LOCKDOWN_PERIODS:
        start, end = period.bounds()
        if start <= date <= end:
            return "during_lockdown", period.name
    return "outside_lockdown", "outside_lockdown"


def run_ego_analysis(
    *,
    cases: np.ndarray,
    cases_mask: np.ndarray,
    population: np.ndarray,
    mobility: np.ndarray,
    dates: pd.DatetimeIndex,
    region_ids: np.ndarray,
    mobility_time_mask: np.ndarray | None,
    mobility_threshold: float,
    window_length: int,
    window_stride: int,
    missing_permit: int,
    mobility_reduction: pd.Series,
    min_timepoints: int,
    max_windows: int | None = None,
    log_every: int = 25,
) -> pd.DataFrame:
    """Run vectorized ego regressions for all valid windows and targets."""
    starts = compute_window_starts(cases.shape[0], window_length, window_stride)
    if max_windows is not None:
        starts = starts[:max_windows]
    if len(starts) == 0:
        return pd.DataFrame()

    valid_history = compute_valid_history_mask(
        cases_mask, starts, window_length, missing_permit
    )
    logger.info("Building ego mask for mobility threshold %.3f", mobility_threshold)
    ego_mask = build_ego_mask(mobility, mobility_threshold, mobility_time_mask)

    rows: list[pd.DataFrame] = []
    target_nodes = np.arange(cases.shape[1], dtype=np.int32)

    for w_idx, start in enumerate(starts):
        end = int(start + window_length)
        center_idx = int(start + window_length // 2)
        center_date = pd.Timestamp(dates[center_idx])
        lockdown_status, lockdown_period = classify_lockdown(center_date)

        stats = vectorized_window_regression(
            cases[start:end],
            cases_mask[start:end],
            population,
            ego_mask[start:end],
            min_timepoints=min_timepoints,
        )
        valid_targets = valid_history[w_idx] & np.isfinite(stats["r2"])
        if np.any(valid_targets):
            frame = pd.DataFrame(
                {
                    "window_start": int(start),
                    "window_center_date": center_date,
                    "target_node": target_nodes[valid_targets],
                    "region_id": region_ids[valid_targets],
                    "lockdown_status": lockdown_status,
                    "lockdown_period": lockdown_period,
                    "slope": stats["slope"][valid_targets],
                    "intercept": stats["intercept"][valid_targets],
                    "r2": stats["r2"][valid_targets],
                    "n_timepoints": stats["n_timepoints"][valid_targets],
                    "n_neighbors_mean": stats["n_neighbors_mean"][valid_targets],
                    "n_neighbors_min": stats["n_neighbors_min"][valid_targets],
                    "n_neighbors_max": stats["n_neighbors_max"][valid_targets],
                    "mobility_reduction_at_center": float(
                        mobility_reduction.iloc[center_idx]
                    ),
                }
            )
            rows.append(frame)

        if (w_idx + 1) % log_every == 0 or w_idx + 1 == len(starts):
            logger.info("Processed %d/%d windows", w_idx + 1, len(starts))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize ego regression rows by lockdown status and period."""
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["lockdown_status", "lockdown_period"], dropna=False)
        .agg(
            n_samples=("r2", "size"),
            n_windows=("window_start", "nunique"),
            n_targets=("target_node", "nunique"),
            r2_mean=("r2", "mean"),
            r2_median=("r2", "median"),
            r2_std=("r2", "std"),
            slope_mean=("slope", "mean"),
            slope_median=("slope", "median"),
            mobility_reduction_mean=("mobility_reduction_at_center", "mean"),
            n_neighbors_mean=("n_neighbors_mean", "mean"),
        )
        .reset_index()
    )


def generate_plots(results: pd.DataFrame, output_dir: Path) -> None:
    """Generate distribution, density, and count plots."""
    if results.empty:
        return

    plot_df = results[np.isfinite(results["r2"])].copy()
    period_order = ["outside_lockdown"] + [period.name for period in LOCKDOWN_PERIODS]
    periods = [p for p in period_order if p in set(plot_df["lockdown_period"])]

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [
        plot_df.loc[plot_df["lockdown_period"] == period, "r2"].values
        for period in periods
    ]
    positions = np.arange(1, len(periods) + 1)
    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(periods)
    ax.set_ylabel("Ego regression R²")
    ax.set_title("Ego-Neighborhood R² Distribution by Period")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "ego_r2_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(
        plot_df["mobility_reduction_at_center"],
        plot_df["r2"],
        gridsize=35,
        mincnt=1,
        cmap="viridis",
    )
    ax.set_xlabel("Mobility reduction at window center (%)")
    ax.set_ylabel("Ego regression R²")
    ax.set_title("Mobility Reduction vs Ego R² Density")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Ego windows")
    fig.tight_layout()
    fig.savefig(output_dir / "ego_mobility_vs_r2_hexbin.png", dpi=200)
    plt.close(fig)

    counts = (
        plot_df.groupby(["window_center_date", "lockdown_status"])
        .size()
        .rename("n_samples")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    for status, group in counts.groupby("lockdown_status"):
        ax.plot(group["window_center_date"], group["n_samples"], label=status)
    for period in LOCKDOWN_PERIODS:
        start, end = period.bounds()
        ax.axvspan(start, end, color="#c44e52", alpha=0.12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_xlabel("Window center date")
    ax.set_ylabel("Finite ego samples")
    ax.set_title("Finite Ego-Sample Counts by Window")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "ego_window_counts.png", dpi=200)
    plt.close(fig)


def write_report(
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    config: EgoAnalysisConfig,
    window_length: int,
    missing_permit: int,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> None:
    """Write a compact markdown report and JSON metadata."""
    metadata = {
        "n_samples": int(len(results)),
        "n_windows": int(results["window_start"].nunique()) if not results.empty else 0,
        "n_targets": int(results["target_node"].nunique()) if not results.empty else 0,
        "window_length": int(window_length),
        "window_stride": int(config.window_stride),
        "missing_permit_cases": int(missing_permit),
        "mobility_threshold": float(config.mobility_threshold),
        "date_start": str(date_start.date()),
        "date_end": str(date_end.date()),
    }
    with (output_dir / "ego_analysis_summary.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    lines = [
        "# Vectorized Ego-Neighborhood Lockdown Analysis",
        "",
        f"Dataset period: {metadata['date_start']} to {metadata['date_end']}.",
        f"Rows: {metadata['n_samples']} ego windows across "
        f"{metadata['n_windows']} temporal windows and {metadata['n_targets']} targets.",
        "",
        "Each row is one valid `(window_start, target_municipality)` pair. "
        "Rows are more numerous than the aggregate analysis, but they are not fully "
        "independent because windows overlap and all targets share the same global "
        "trend within a window.",
        "",
        "## Key Outputs",
        "",
        "- `ego_regression_results.parquet`",
        "- `ego_lockdown_summary.csv`",
        "- `ego_r2_distribution.png`",
        "- `ego_mobility_vs_r2_hexbin.png`",
        "- `ego_window_counts.png`",
        "",
        "## Settings",
        "",
        f"- Window length: {metadata['window_length']}",
        f"- Window stride: {metadata['window_stride']}",
        f"- Cases missing permit: {metadata['missing_permit_cases']}",
        f"- Mobility threshold: {metadata['mobility_threshold']:.3f}",
        "- Non-positive mobility thresholds use positive-flow edges (`mobility > 0`).",
    ]
    if not summary.empty:
        lines.extend(["", "## Summary", "", format_markdown_table(summary)])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without optional pandas dependencies."""
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    headers = [str(col) for col in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.astype(str).itertuples(index=False):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized ego-neighborhood lockdown analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_epifor_real_local.yaml"),
        help="Training config path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/mobility_ego_lockdown_analysis"),
        help="Directory for report artifacts",
    )
    parser.add_argument("--baseline-start", type=str, default="2020-03-01")
    parser.add_argument("--baseline-end", type=str, default="2020-03-14")
    parser.add_argument(
        "--mobility-threshold",
        type=float,
        default=DEFAULT_MOBILITY_THRESHOLD,
        help=(
            "Minimum incoming mobility for ego membership. Non-positive values "
            "use positive-flow edges."
        ),
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="History window length; defaults to config.model.input_window_length",
    )
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--min-timepoints", type=int, default=MIN_REGRESSION_TIMEPOINTS)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional debug limit on number of windows",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    runtime_config = EgoAnalysisConfig(
        config_path=args.config,
        output_dir=args.output_dir,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        mobility_threshold=args.mobility_threshold,
        window_length=args.window_length,
        window_stride=args.window_stride,
        min_timepoints=args.min_timepoints,
        max_windows=args.max_windows,
        log_every=args.log_every,
    )

    if runtime_config.output_dir.exists():
        shutil.rmtree(runtime_config.output_dir)
    runtime_config.output_dir.mkdir(parents=True, exist_ok=True)

    config = EpiForecasterConfig.from_file(str(runtime_config.config_path))
    window_length = (
        int(runtime_config.window_length)
        if runtime_config.window_length is not None
        else int(config.model.input_window_length)
    )
    missing_permit = int(config.data.missing_permit.input.get("cases", 0))

    logger.info("Loading canonical arrays from %s", config.data.dataset_path)
    ds = open_canonical_dataset(config)
    try:
        arrays = load_analysis_arrays(ds)
    finally:
        ds.close()

    mobility_reduction = compute_mobility_reduction(
        arrays["mobility"],
        arrays["dates"],
        runtime_config.baseline_start,
        runtime_config.baseline_end,
    )
    results = run_ego_analysis(
        cases=arrays["cases"],
        cases_mask=arrays["cases_mask"],
        population=arrays["population"],
        mobility=arrays["mobility"],
        dates=arrays["dates"],
        region_ids=arrays["region_ids"],
        mobility_time_mask=arrays["mobility_time_mask"],
        mobility_threshold=runtime_config.mobility_threshold,
        window_length=window_length,
        window_stride=runtime_config.window_stride,
        missing_permit=missing_permit,
        mobility_reduction=mobility_reduction,
        min_timepoints=runtime_config.min_timepoints,
        max_windows=runtime_config.max_windows,
        log_every=runtime_config.log_every,
    )

    results_path = runtime_config.output_dir / "ego_regression_results.parquet"
    results.to_parquet(results_path, index=False)
    summary = summarize_results(results)
    summary.to_csv(runtime_config.output_dir / "ego_lockdown_summary.csv", index=False)
    generate_plots(results, runtime_config.output_dir)
    write_report(
        results=results,
        summary=summary,
        output_dir=runtime_config.output_dir,
        config=runtime_config,
        window_length=window_length,
        missing_permit=missing_permit,
        date_start=arrays["dates"][0],
        date_end=arrays["dates"][-1],
    )
    logger.info("Wrote ego-neighborhood report to %s", runtime_config.output_dir)


if __name__ == "__main__":
    main()
