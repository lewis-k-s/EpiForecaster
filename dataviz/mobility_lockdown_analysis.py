"""
Mobility Reduction and Regression Quality During Lockdown Periods

Analyzes whether mobility reductions during lockdowns caused a delinking between
regional neighborhood trends and global epidemic trends.

Visualizes:
- Total mobility volume reduction percentage over time
- Regression quality (R²) of neighborhood vs global trends
- Lockdown periods as shaded regions
- Correlation between mobility reduction and regression quality
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.epi_dataset import EpiDataset  # noqa: E402
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from models.configs import EpiForecasterConfig  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class LockdownPeriod:
    """Definition of a lockdown period."""

    name: str
    start: str
    end: str

    def to_datetime(self) -> tuple[datetime, datetime]:
        """Convert start/end strings to datetime objects."""
        start_dt = datetime.fromisoformat(self.start)
        end_dt = datetime.fromisoformat(self.end)
        return start_dt, end_dt


# Lockdown periods with confirmed dates (Catalunya-wide only)
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


def resolve_mobility_array(mobility_da: xr.DataArray | xr.Dataset) -> np.ndarray:
    """Return mobility array in (time, origin, destination) order."""
    if isinstance(mobility_da, xr.Dataset):
        if "mobility" not in mobility_da:
            raise ValueError("Mobility dataset missing 'mobility' variable")
        mobility_da = mobility_da["mobility"]

    dims = list(mobility_da.dims)
    time_dim = TEMPORAL_COORD if TEMPORAL_COORD in dims else None
    if time_dim is None:
        raise ValueError(f"Mobility data missing {TEMPORAL_COORD} dimension")

    if "origin" in dims and "destination" in dims:
        ordered = mobility_da.transpose(time_dim, "origin", "destination")
        return ordered.values

    if dims.count(REGION_COORD) == 2:
        ordered = mobility_da.transpose(time_dim, REGION_COORD, REGION_COORD)
        return ordered.values

    raise ValueError(
        "Mobility data must include ('origin','destination') or two "
        f"'{REGION_COORD}' dims"
    )


def compute_mobility_reduction(
    mobility: np.ndarray,
    dates: pd.DatetimeIndex,
    baseline_start: str,
    baseline_end: str,
) -> tuple[pd.Series, pd.Series]:
    """Compute percentage reduction in total mobility volume from rolling baseline.

    Uses a 14-day rolling median baseline to account for seasonal and
    weekly patterns, and applies 7-day smoothing to the reduction.

    Args:
        mobility: (time, origin, destination) mobility array
        dates: DatetimeIndex for time dimension
        baseline_start: ISO date string for baseline period start
        baseline_end: ISO date string for baseline period end

    Returns:
        Tuple of (raw_reduction, smoothed_reduction) Series with datetime index
    """
    baseline_start_dt = datetime.fromisoformat(baseline_start)
    baseline_end_dt = datetime.fromisoformat(baseline_end)

    baseline_mask = (dates >= baseline_start_dt) & (dates <= baseline_end_dt)

    if baseline_mask.sum() == 0:
        raise ValueError(
            f"No dates found in baseline period {baseline_start} to {baseline_end}"
        )

    # Compute total volume per day
    daily_totals = mobility.sum(axis=(1, 2))
    daily_series = pd.Series(daily_totals, index=dates)

    # Compute rolling median baseline (14-day window)
    rolling_baseline = daily_series.rolling(
        window=14, center=True, min_periods=7
    ).median()

    # Backfill baseline period with the actual values for stability
    rolling_baseline[baseline_mask] = daily_series[baseline_mask]

    # Compute reduction percentage
    reduction_pct = (rolling_baseline - daily_series) / rolling_baseline * 100

    # Apply 7-day smoothing to reduction
    reduction_smoothed = reduction_pct.rolling(
        window=7, center=True, min_periods=1
    ).mean()

    return reduction_pct, reduction_smoothed


def load_or_compute_regression_results(
    results_path: Path | None,
    config: EpiForecasterConfig,
    dataset: xr.Dataset,
    mobility: np.ndarray,
    population: np.ndarray,
    window_stride: int,
    mobility_threshold: float,
) -> pd.DataFrame:
    """Load regression results from CSV or recompute them."""
    from dataviz.neighborhood_global_regression import (
        compute_valid_window_mask,
        run_regression_analysis,
    )

    if results_path is not None and results_path.exists():
        logger.info("Loading regression results from %s", results_path)
        return pd.read_csv(results_path)

    logger.info("Computing regression results from scratch...")

    cases_da = dataset["cases"]

    history_len = int(config.model.history_length)
    horizon = int(config.model.forecast_horizon)
    window_len = history_len + horizon
    missing_permit = int(config.model.missing_permit)

    starts, valid_mask = compute_valid_window_mask(
        cases_da, history_len, horizon, window_stride, missing_permit
    )

    logger.info("Found %d valid windows", len(starts))

    num_nodes = dataset[REGION_COORD].size
    target_nodes = list(range(num_nodes))

    results_df = run_regression_analysis(
        cases_da,
        mobility,
        population,
        starts,
        valid_mask,
        target_nodes,
        window_len,
        mobility_threshold,
        include_self=False,
    )

    return results_df


def aggregate_regression_by_window(
    results_df: pd.DataFrame, window_len: int
) -> pd.DataFrame:
    """Aggregate regression statistics per window, using center date.

    Args:
        results_df: DataFrame with regression results
        window_len: Length of each window in days

    Returns:
        DataFrame with aggregated statistics per window
    """
    agg_df = (
        results_df.groupby("window_start")
        .agg(
            {
                "slope": ["mean", "median", "std"],
                "r2": ["mean", "std"],
                "p_value": "mean",
                "n_neighbors": "mean",
            }
        )
        .round(3)
    )

    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]

    # Calculate center date for each window
    agg_df["window_center"] = agg_df.index + window_len // 2

    return agg_df.reset_index()


def add_lockdown_shading(ax, dates: pd.DatetimeIndex) -> None:
    """Add shaded regions for lockdown periods to a plot.

    Args:
        ax: Matplotlib axis
        dates: DatetimeIndex for time dimension
    """
    for lockdown in LOCKDOWN_PERIODS:
        start_dt, end_dt = lockdown.to_datetime()

        # Find indices for lockdown period - convert to numeric for comparison
        dates_numpy = np.array(dates, dtype="datetime64[ns]")
        dates_numeric = dates_numpy.astype("int64")
        start_numeric = pd.Timestamp(start_dt).value
        end_numeric = pd.Timestamp(end_dt).value

        start_idx = int(np.searchsorted(dates_numeric, start_numeric))
        end_idx = int(np.searchsorted(dates_numeric, end_numeric))

        ax.axvspan(
            start_idx,
            end_idx,
            alpha=0.2,
            color="red",
            label=lockdown.name if not hasattr(ax, "_lockdown_labeled") else None,
        )

        if not hasattr(ax, "_lockdown_labeled"):
            ax._lockdown_labeled = True

    # Add legend once
    handles, labels = ax.get_legend_handles_labels()
    if not hasattr(ax, "_lockdown_added_to_legend"):
        # Filter to unique lockdown labels
        lockdown_labels = [ld.name for ld in LOCKDOWN_PERIODS]
        lockdown_handles = [
            mpatches.Rectangle((0, 0), 1, 1, alpha=0.2, color="red")
            for _ in LOCKDOWN_PERIODS
        ]

        # Remove duplicates and add to existing legend
        ax.legend(
            handles + lockdown_handles,
            labels + lockdown_labels,
            loc="upper right",
            fontsize=8,
        )
        ax._lockdown_added_to_legend = True


def plot_mobility_reduction(
    mobility_reduction_raw: pd.Series,
    mobility_reduction_smoothed: pd.Series,
    output_path: Path,
) -> None:
    """Plot mobility reduction time series with lockdown periods shaded.

    Args:
        mobility_reduction_raw: Raw reduction Series with datetime index
        mobility_reduction_smoothed: Smoothed (7-day) reduction Series
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x_vals = np.arange(len(mobility_reduction_raw))

    # Plot raw reduction with transparency
    ax.plot(
        x_vals,
        mobility_reduction_raw.values.astype(float),
        linewidth=0.5,
        color="#1f77b4",
        alpha=0.3,
        label="Raw reduction",
    )

    # Plot smoothed reduction as main line
    ax.plot(
        x_vals,
        mobility_reduction_smoothed.values.astype(float),
        linewidth=2,
        color="#1f77b4",
        label="Smoothed reduction (7-day)",
    )

    add_lockdown_shading(ax, mobility_reduction_raw.index)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mobility reduction (%)", fontsize=12)
    ax.set_title(
        "Total Mobility Volume Reduction from Rolling 14-Day Baseline", fontsize=14
    )
    ax.grid(True, alpha=0.3)

    # Set x-axis to date format
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved mobility reduction plot to %s", output_path)


def plot_regression_quality(
    agg_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    output_path: Path,
) -> None:
    """Plot regression quality (R²) time series with lockdown periods shaded.

    Args:
        agg_df: DataFrame with aggregated regression statistics per window
        dates: DatetimeIndex for time dimension
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Map window center indices to dates
    window_centers = agg_df["window_center"].values

    # Plot mean R²
    ax.plot(
        window_centers,
        agg_df["r2_mean"].values.astype(float),
        linewidth=2,
        color="#2ca02c",
        label="Mean R²",
    )

    # Plot R² ± std as shaded region
    ax.fill_between(
        window_centers,
        (agg_df["r2_mean"] - agg_df["r2_std"]).values.astype(float),
        (agg_df["r2_mean"] + agg_df["r2_std"]).values.astype(float),
        alpha=0.3,
        color="#2ca02c",
        label="R² ± std",
    )

    # Add reference lines
    ax.axhline(
        y=0.5,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="R² = 0.5",
    )
    ax.axhline(
        y=0.8, color="blue", linestyle="--", alpha=0.7, linewidth=1.5, label="R² = 0.8"
    )
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.5, linewidth=1)

    add_lockdown_shading(ax, dates)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Neighborhood vs Global Trend Regression Quality", fontsize=14)
    ax.set_ylim((-0.1, 1.1))
    ax.grid(True, alpha=0.3)

    # Set x-axis to date format
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved regression quality plot to %s", output_path)


def plot_lockdown_correlation_scatter(
    mobility_reduction_smoothed: pd.Series,
    agg_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    output_path: Path,
) -> None:
    """Plot scatter of mobility reduction vs regression quality, colored by lockdown status.

    Args:
        mobility_reduction_smoothed: Smoothed (7-day) reduction Series
        agg_df: DataFrame with aggregated regression statistics per window
        dates: DatetimeIndex for time dimension
        output_path: Path to save plot
    """
    # Map window centers to dates and get corresponding mobility reduction
    window_centers = agg_df["window_center"].values
    window_dates = []
    for wc in window_centers:
        w_ts = pd.to_datetime(dates[wc])
        if hasattr(w_ts, "to_pydatetime"):
            w_date = w_ts.to_pydatetime()
        else:
            w_date = w_ts
        window_dates.append(w_date)

    # Get mobility reduction values for each window center
    mobility_values = []
    for wc in window_centers:
        w_ts = pd.to_datetime(dates[wc])
        # Use get_loc to find the nearest date
        if w_ts in mobility_reduction_smoothed.index:
            idx = mobility_reduction_smoothed.index.get_loc(w_ts)
        else:
            # Find nearest date by searchsorted
            idx = int(np.searchsorted(mobility_reduction_smoothed.index.values, w_ts))
            if idx >= len(mobility_reduction_smoothed.index):
                idx = len(mobility_reduction_smoothed.index) - 1
        mobility_values.append(mobility_reduction_smoothed.iloc[idx])

    mobility_values = np.array(mobility_values, dtype=float)

    # Determine which windows are during lockdowns
    lockdown_status = []
    for w_date in window_dates:
        in_lockdown = any(
            start_dt <= w_date <= end_dt
            for lockdown in LOCKDOWN_PERIODS
            for start_dt, end_dt in [lockdown.to_datetime()]
        )
        lockdown_status.append(in_lockdown)

    lockdown_status = np.array(lockdown_status, dtype=bool)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot non-lockdown windows
    mask = ~lockdown_status
    if mask.sum() > 0:
        ax.scatter(
            mobility_values[mask],
            agg_df["r2_mean"].values[mask].astype(float),
            alpha=0.6,
            color="#1f77b4",
            label="Outside lockdown",
            s=60,
        )

        # Add trend line for non-lockdown
        if mask.sum() >= 2:
            x_vals = mobility_values[mask]
            y_vals = agg_df["r2_mean"].values[mask].astype(float)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                color="#1f77b4",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Non-lockdown trend: slope={z[0]:.3f}",
            )

    # Plot lockdown windows
    mask = lockdown_status
    if mask.sum() > 0:
        ax.scatter(
            mobility_values[mask],
            agg_df["r2_mean"].values[mask].astype(float),
            alpha=0.8,
            color="#d62728",
            label="During lockdown",
            s=80,
            edgecolors="black",
            linewidth=1.5,
        )

        # Add trend line for lockdown
        if mask.sum() >= 2:
            x_vals = mobility_values[mask]
            y_vals = agg_df["r2_mean"].values[mask].astype(float)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                color="#d62728",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Lockdown trend: slope={z[0]:.3f}",
            )

    ax.set_xlabel("Mobility reduction (%)", fontsize=12)
    ax.set_ylabel("Mean R²", fontsize=12)
    ax.set_title(
        "Mobility Reduction vs Regression Quality\n(Windows during vs outside lockdowns)",
        fontsize=14,
    )
    ax.set_ylim((-0.1, 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add reference lines
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.8, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved lockdown correlation scatter to %s", output_path)


def compute_lockdown_statistics(
    mobility_reduction_smoothed: pd.Series,
    agg_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, dict[str, Any]]:
    """Compute statistics for each lockdown period.

    Args:
        mobility_reduction_smoothed: Smoothed Series with datetime index and reduction percentages
        agg_df: DataFrame with aggregated regression statistics per window
        dates: DatetimeIndex for time dimension

    Returns:
        Dictionary with statistics per lockdown period
    """
    stats = {}

    # Map window centers to dates
    window_dates = dates[agg_df["window_center"].values]

    for lockdown in LOCKDOWN_PERIODS:
        start_dt, end_dt = lockdown.to_datetime()

        # Mobility reduction stats during lockdown
        mobility_during = mobility_reduction_smoothed[
            (mobility_reduction_smoothed.index >= start_dt)
            & (mobility_reduction_smoothed.index <= end_dt)
        ]

        # Pre-lockdown baseline (7 days before)
        pre_start = start_dt - pd.Timedelta(days=7)
        pre_end = start_dt - pd.Timedelta(days=1)
        mobility_pre = mobility_reduction_smoothed[
            (mobility_reduction_smoothed.index >= pre_start)
            & (mobility_reduction_smoothed.index <= pre_end)
        ]

        # Regression quality stats during lockdown
        lockdown_mask = (window_dates >= start_dt) & (window_dates <= end_dt)
        r2_during = agg_df["r2_mean"].values[lockdown_mask]

        # Pre-lockdown regression quality
        pre_mask = (window_dates >= pre_start) & (window_dates <= pre_end)
        r2_pre = agg_df["r2_mean"].values[pre_mask]

        mobility_mean_during = (
            float(mobility_during.mean()) if len(mobility_during) > 0 else np.nan
        )
        mobility_mean_pre = (
            float(mobility_pre.mean()) if len(mobility_pre) > 0 else np.nan
        )

        stats[lockdown.name] = {
            "mobility_reduction_during_mean": mobility_mean_during,
            "mobility_reduction_during_std": float(mobility_during.std())
            if len(mobility_during) > 0
            else np.nan,
            "mobility_reduction_pre_mean": mobility_mean_pre,
            "mobility_reduction_pre_std": float(mobility_pre.std())
            if len(mobility_pre) > 0
            else np.nan,
            "mobility_delta": mobility_mean_during - mobility_mean_pre
            if not np.isnan(mobility_mean_during) and not np.isnan(mobility_mean_pre)
            else np.nan,
            "r2_during_mean": float(np.nanmean(r2_during))
            if len(r2_during) > 0
            else np.nan,
            "r2_during_std": float(np.nanstd(r2_during))
            if len(r2_during) > 0
            else np.nan,
            "r2_pre_mean": float(np.nanmean(r2_pre)) if len(r2_pre) > 0 else np.nan,
            "r2_pre_std": float(np.nanstd(r2_pre)) if len(r2_pre) > 0 else np.nan,
            "r2_delta": float(np.nanmean(r2_during) - np.nanmean(r2_pre))
            if len(r2_during) > 0 and len(r2_pre) > 0
            else np.nan,
        }

    # Overall statistics
    overall_lockdown_mask = np.array(
        [
            any(
                start_dt <= d <= end_dt
                for start_dt, end_dt in [ld.to_datetime() for ld in LOCKDOWN_PERIODS]
            )
            for d in window_dates
        ]
    )
    overall_non_lockdown_mask = ~overall_lockdown_mask

    r2_during = agg_df["r2_mean"].values[overall_lockdown_mask]
    r2_outside = agg_df["r2_mean"].values[overall_non_lockdown_mask]

    stats["overall"] = {
        "r2_mean_during_lockdown": float(np.nanmean(r2_during))
        if overall_lockdown_mask.sum() > 0
        else np.nan,
        "r2_mean_outside_lockdown": float(np.nanmean(r2_outside))
        if overall_non_lockdown_mask.sum() > 0
        else np.nan,
        "r2_delta": float(np.nanmean(r2_during) - np.nanmean(r2_outside))
        if overall_lockdown_mask.sum() > 0 and overall_non_lockdown_mask.sum() > 0
        else np.nan,
        "n_windows_during_lockdown": int(overall_lockdown_mask.sum()),
        "n_windows_outside_lockdown": int(overall_non_lockdown_mask.sum()),
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mobility reduction and regression quality during lockdown periods"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config path",
    )
    parser.add_argument(
        "--baseline-start",
        type=str,
        default="2020-01-01",
        help="Baseline period start date (ISO format, default: 2020-01-01)",
    )
    parser.add_argument(
        "--baseline-end",
        type=str,
        default="2020-03-01",
        help="Baseline period end date (ISO format, default: 2020-03-01)",
    )
    parser.add_argument(
        "--regression-results",
        type=Path,
        default=None,
        help="Path to saved regression CSV from neighborhood_global_regression.py",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=14,
        help="Stride for sliding windows (default: 14 days)",
    )
    parser.add_argument(
        "--mobility-threshold",
        type=float,
        default=None,
        help="Minimum incoming flow for neighbors (default: config value)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/mobility_lockdown_analysis"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Mobility Lockdown Analysis")
    logger.info("=" * 60)

    # Load config and dataset
    logger.info("Loading config from %s", args.config)
    config = EpiForecasterConfig.from_file(str(args.config))

    logger.info("Loading dataset from %s", config.data.dataset_path)
    dataset = EpiDataset.load_canonical_dataset(Path(config.data.dataset_path))

    dates = pd.DatetimeIndex(dataset[TEMPORAL_COORD].values)
    logger.info(
        "Dataset: %d time steps, %d regions", len(dates), dataset[REGION_COORD].size
    )

    # Get mobility and population data
    logger.info("Extracting mobility and population data...")
    mobility = resolve_mobility_array(dataset["mobility"])
    population = dataset["population"].values

    # Compute mobility reduction
    logger.info(
        "Computing mobility reduction from baseline %s to %s...",
        args.baseline_start,
        args.baseline_end,
    )
    mobility_reduction_raw, mobility_reduction_smoothed = compute_mobility_reduction(
        mobility, dates, args.baseline_start, args.baseline_end
    )

    # Save mobility reduction data (both raw and smoothed)
    mobility_path = output_dir / "mobility_reduction.csv"
    df_mobility = pd.DataFrame(
        {
            "mobility_reduction_raw": mobility_reduction_raw,
            "mobility_reduction_smoothed": mobility_reduction_smoothed,
        },
        index=mobility_reduction_raw.index,
    )
    df_mobility.to_csv(mobility_path)
    logger.info("Saved mobility reduction to %s", mobility_path)

    # Load or compute regression results
    mobility_threshold = (
        float(args.mobility_threshold)
        if args.mobility_threshold is not None
        else float(config.data.mobility_threshold)
    )

    logger.info("Processing regression results...")
    results_df = load_or_compute_regression_results(
        args.regression_results,
        config,
        dataset,
        mobility,
        population,
        args.window_stride,
        mobility_threshold,
    )

    logger.info("Computed %d regression results", len(results_df))

    # Save regression results if computed from scratch
    if args.regression_results is None:
        regression_path = output_dir / "regression_results.csv"
        results_df.to_csv(regression_path, index=False)
        logger.info("Saved regression results to %s", regression_path)

    # Aggregate regression statistics by window
    window_len = int(config.model.history_length) + int(config.model.forecast_horizon)
    agg_df = aggregate_regression_by_window(results_df, window_len)

    # Save aggregated results
    agg_path = output_dir / "regression_quality_windows.csv"
    agg_df.to_csv(agg_path, index=False)
    logger.info("Saved aggregated regression quality to %s", agg_path)

    # Generate plots
    logger.info("Generating visualizations...")

    plot_mobility_reduction(
        mobility_reduction_raw,
        mobility_reduction_smoothed,
        output_dir / "mobility_reduction.png",
    )

    plot_regression_quality(
        agg_df,
        dates,
        output_dir / "regression_quality.png",
    )

    plot_lockdown_correlation_scatter(
        mobility_reduction_smoothed,
        agg_df,
        dates,
        output_dir / "lockdown_correlation_scatter.png",
    )

    # Compute and print lockdown statistics
    logger.info("Computing lockdown statistics...")
    stats = compute_lockdown_statistics(mobility_reduction_smoothed, agg_df, dates)

    stats_path = output_dir / "lockdown_statistics.txt"
    with open(stats_path, "w") as f:
        f.write("Lockdown Statistics\n")
        f.write("=" * 60 + "\n\n")

        for lockdown_name, lockdown_stats in stats.items():
            if lockdown_name == "overall":
                f.write("Overall Statistics\n")
                f.write("-" * 60 + "\n")
            else:
                f.write(f"{lockdown_name}\n")
                f.write("-" * 60 + "\n")

            for key, value in lockdown_stats.items():
                if isinstance(value, float) and not np.isnan(value):
                    f.write(f"  {key}: {value:.3f}\n")
                elif isinstance(value, float):
                    f.write(f"  {key}: NaN\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    logger.info("Saved lockdown statistics to %s", stats_path)

    # Print summary to console
    logger.info("=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)

    for lockdown_name, lockdown_stats in stats.items():
        if lockdown_name == "overall":
            logger.info("\nOverall Statistics:")
        else:
            logger.info("\n%s:", lockdown_name)

        if "r2_delta" in lockdown_stats:
            r2_delta = lockdown_stats["r2_delta"]
            if not np.isnan(r2_delta):
                direction = "decreased" if r2_delta < 0 else "increased"
                logger.info("  R² %s by %.3f during lockdown", direction, abs(r2_delta))

        if "mobility_delta" in lockdown_stats:
            mob_delta = lockdown_stats["mobility_delta"]
            if not np.isnan(mob_delta):
                logger.info(
                    "  Mobility reduction %.3f%% vs pre-lockdown baseline", mob_delta
                )

    logger.info("\nDone! Plots saved to %s", output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
