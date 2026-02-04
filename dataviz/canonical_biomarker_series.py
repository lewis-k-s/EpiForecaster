"""
Plot canonical biomarker series from a preprocessed dataset.

Outputs:
- Region time series with global mean/median
- Heatmap of log1p biomarker values
- Distribution of log1p biomarker values
- Per-region boxplots
- Time window zoom view
- Summary statistics table (console and CSV)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from utils.plotting import (  # noqa: E402
    Colors,
    Style,
    format_date_axis,
    robust_bounds,
    save_figure,
)

logger = logging.getLogger(__name__)


def _load_biomarkers(
    dataset_path: Path,
) -> tuple[dict[str, xr.DataArray], np.ndarray | None]:
    dataset = xr.open_zarr(dataset_path)
    variant_vars: dict[str, xr.DataArray] = {}
    for name in dataset.data_vars:
        name_str = str(name)
        if name_str.startswith("edar_biomarker_"):
            variant_vars[name_str] = dataset[name]
    if not variant_vars:
        if "edar_biomarker" not in dataset:
            raise ValueError("Dataset missing EDAR biomarker variables")
        variant_vars = {"edar_biomarker": dataset["edar_biomarker"]}

    data_start = dataset.get("biomarker_data_start")

    if "edar_has_source" in dataset:
        mask = dataset["edar_has_source"].astype(bool)
        valid_regions = dataset[REGION_COORD].values[mask.values]
        if len(valid_regions) == 0:
            raise ValueError("No regions with edar_has_source == 1")
        for name, biomarker in variant_vars.items():
            variant_vars[name] = biomarker.sel({REGION_COORD: valid_regions})
        if data_start is not None:
            data_start = data_start.sel({REGION_COORD: valid_regions})
        logger.info(
            "Filtered biomarker to %d regions with EDAR sources",
            len(valid_regions),
        )

    for name, biomarker in variant_vars.items():
        variant_vars[name] = biomarker.transpose(TEMPORAL_COORD, REGION_COORD)

    data_start_values = data_start.values if data_start is not None else None

    logger.info("Loaded %d biomarker variants", len(variant_vars))
    return variant_vars, data_start_values


def _select_region_indices(
    region_ids: np.ndarray,
    *,
    requested_ids: list[str] | None,
    max_regions: int,
    seed: int,
) -> list[int]:
    if requested_ids:
        id_to_index = {str(rid): idx for idx, rid in enumerate(region_ids)}
        missing = [rid for rid in requested_ids if rid not in id_to_index]
        if missing:
            raise ValueError(f"Unknown region ids: {missing}")
        return [id_to_index[rid] for rid in requested_ids]

    rng = np.random.default_rng(seed)
    n_regions = min(max_regions, len(region_ids))
    return rng.choice(len(region_ids), size=n_regions, replace=False).tolist()


def plot_region_series(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
    variant_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx in indices:
        ax.plot(dates, values[:, idx], alpha=0.35, linewidth=1)

    values_positive = np.where(values > 0, values, np.nan)
    if np.isfinite(values_positive).any():
        with np.errstate(all="ignore"):
            global_mean = np.nanmean(values_positive, axis=1)
            global_median = np.nanmedian(values_positive, axis=1)
        ax.plot(dates, global_mean, color=Colors.GLOBAL_MEAN, linewidth=2, label="Global mean")
        ax.plot(
            dates, global_median, color=Colors.GLOBAL_MEDIAN, linewidth=2, label="Global median"
        )

    bounds = robust_bounds(values[:, indices], positive_only=True)
    if bounds is not None:
        _, upper = bounds
        ax.set_ylim(bottom=0, top=upper * 1.05)

    format_date_axis(ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("log1p biomarker")
    ax.set_title(f"Canonical Biomarker Series ({variant_label})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved time series plot")


def plot_heatmap(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
    variant_label: str,
) -> None:
    subset = values[:, indices]
    heatmap_df = pd.DataFrame(subset, index=dates, columns=region_ids[indices])

    fig, ax = plt.subplots(figsize=(14, 6))
    bounds = robust_bounds(subset, positive_only=True)
    vmin = 0.0
    vmax = bounds[1] if bounds is not None else None
    sns.heatmap(
        heatmap_df.T,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "log1p biomarker"},
    )
    step = max(1, len(dates) // 12)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Region")
    ax.set_title(f"Biomarker Heatmap ({variant_label})")

    plt.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved heatmap")


def plot_distribution(
    values: np.ndarray, output_path: Path, variant_label: str
) -> None:
    flat_values = values[np.isfinite(values)]
    positive_values = flat_values[flat_values > 0]
    zero_fraction = 1.0 - (positive_values.size / max(flat_values.size, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    if positive_values.size > 0:
        sns.histplot(positive_values, bins=60, kde=True, ax=ax)
    else:
        logger.warning("No positive values for distribution plot")
    ax.set_xlabel("log1p biomarker")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Biomarker Distribution ({variant_label}) | zeros: {zero_fraction:.1%}"
    )

    plt.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved distribution plot")


def compute_summary_statistics(
    biomarker: xr.DataArray,
    region_ids: np.ndarray,
    indices: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-region summary statistics.

    Args:
        biomarker: Raw (untransformed) biomarker data array
        region_ids: Array of region IDs
        indices: Optional subset of region indices to analyze

    Returns:
        DataFrame with per-region statistics including coverage,
        central tendency, dispersion, and data quality flags
    """
    if indices is None:
        indices = list(range(len(region_ids)))

    stats = []
    raw_values = biomarker.values
    n_total_days = raw_values.shape[0]

    for idx in indices:
        region_data = raw_values[:, idx]
        region_id = str(region_ids[idx])

        # Valid (non-NaN) data
        valid_mask = np.isfinite(region_data)
        valid_data = region_data[valid_mask]
        n_valid = valid_mask.sum()
        coverage = n_valid / n_total_days

        # Sparsity: fraction of zeros (among valid values)
        sparsity = (valid_data == 0).sum() / n_valid if n_valid > 0 else np.nan

        # Central tendency (log1p scale for reporting)
        log_valid = np.log1p(valid_data) if n_valid > 0 else np.array([])
        mean_val = float(np.mean(log_valid)) if n_valid > 0 else np.nan
        median_val = float(np.median(log_valid)) if n_valid > 0 else np.nan

        # Dispersion
        std_val = float(np.std(log_valid)) if n_valid > 0 else np.nan
        min_val = float(np.min(log_valid)) if n_valid > 0 else np.nan
        max_val = float(np.max(log_valid)) if n_valid > 0 else np.nan

        # Percentiles
        p25 = float(np.percentile(log_valid, 25)) if n_valid > 0 else np.nan
        p75 = float(np.percentile(log_valid, 75)) if n_valid > 0 else np.nan

        # Data quality flags
        quality_flag = "OK"
        if coverage < 0.5:
            quality_flag = "LOW_COVERAGE"
        elif sparsity > 0.8:
            quality_flag = "VERY_SPARSE"
        elif sparsity > 0.5:
            quality_flag = "SPARSE"

        stats.append(
            {
                "region_id": region_id,
                "n_valid_days": n_valid,
                "total_days": n_total_days,
                "coverage": coverage,
                "sparsity": sparsity,
                "mean_log1p": mean_val,
                "median_log1p": median_val,
                "std_log1p": std_val,
                "min_log1p": min_val,
                "max_log1p": max_val,
                "p25_log1p": p25,
                "p75_log1p": p75,
                "quality_flag": quality_flag,
            }
        )

    df = pd.DataFrame(stats)
    df = df.sort_values("coverage", ascending=False).reset_index(drop=True)
    return df


def print_summary_statistics(stats_df: pd.DataFrame, variant_label: str) -> None:
    """Print summary statistics table to console."""
    print("\n" + "=" * 100)
    print(f"SUMMARY STATISTICS ({variant_label})")
    print("=" * 100)

    # Display columns
    display_cols = [
        "region_id",
        "coverage",
        "sparsity",
        "mean_log1p",
        "median_log1p",
        "std_log1p",
        "quality_flag",
    ]

    # Format for display
    display_df = stats_df[display_cols].copy()
    display_df["coverage"] = display_df["coverage"].apply(lambda x: f"{x:.2%}")
    display_df["sparsity"] = display_df["sparsity"].apply(
        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
    )
    display_df["mean_log1p"] = display_df["mean_log1p"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display_df["median_log1p"] = display_df["median_log1p"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display_df["std_log1p"] = display_df["std_log1p"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )

    print(display_df.to_string(index=False))

    # Summary counts
    print("\n" + "-" * 100)
    quality_counts = stats_df["quality_flag"].value_counts()
    print("Quality flag distribution:")
    for flag, count in quality_counts.items():
        print(f"  {flag}: {count}")
    print("=" * 100 + "\n")


def plot_per_region_boxplot(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    stats_df: pd.DataFrame,
    output_path: Path,
    variant_label: str,
) -> None:
    """Create boxplots comparing distributions across regions.

    Highlights regions with highest and lowest coverage.
    """
    n_regions = len(indices)
    if n_regions == 0:
        logger.warning("No regions to plot in boxplot")
        return

    # Prepare data for plotting
    plot_data = []
    labels = []
    colors = []

    # Get top and bottom coverage regions for highlighting
    top_idx = stats_df.index[0]
    bottom_idx = stats_df.index[-1]
    top_region = str(stats_df.loc[top_idx, "region_id"])
    bottom_region = str(stats_df.loc[bottom_idx, "region_id"])

    for idx in indices:
        region_data = values[:, idx]
        valid_data = region_data[np.isfinite(region_data) & (region_data > 0)]
        if len(valid_data) > 0:
            plot_data.append(valid_data)
            labels.append(str(region_ids[idx]))
            if str(region_ids[idx]) == top_region:
                colors.append("tab:green")
            elif str(region_ids[idx]) == bottom_region:
                colors.append("tab:red")
            else:
                colors.append("tab:blue")

    if not plot_data:
        logger.warning("No valid data for boxplot")
        return

    fig, ax = plt.subplots(figsize=(max(12, n_regions * 0.6), 6))

    parts = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, vert=False)

    # Color the boxes
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("log1p biomarker")
    ax.set_ylabel("Region")
    ax.set_title(f"Per-Region Biomarker Distribution ({variant_label})")
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend for highlighted regions
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="tab:green", alpha=0.6, label=f"Highest coverage ({top_region})"
        ),
        Patch(
            facecolor="tab:red", alpha=0.6, label=f"Lowest coverage ({bottom_region})"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved boxplot")


def plot_time_window_zoom(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    start_date: str | None,
    end_date: str | None,
    output_path: Path,
    variant_label: str,
) -> None:
    """Show zoomed view of specific time period.

    If start_date or end_date is None, uses the first/last 20% of the date range.
    """
    # Determine date range
    if start_date is None:
        n_dates = len(dates)
        start_idx = 0
        end_idx = int(n_dates * 0.2)
    elif end_date is None:
        n_dates = len(dates)
        start_idx = int(n_dates * 0.8)
        end_idx = n_dates
    else:
        start_date_parsed = pd.Timestamp(start_date)
        end_date_parsed = pd.Timestamp(end_date)
        mask = (dates >= start_date_parsed) & (dates <= end_date_parsed)
        if not mask.any():
            logger.warning("No dates in specified range, using full range")
            start_idx = 0
            end_idx = len(dates)
        else:
            start_idx = mask.argmax()
            end_idx = len(dates) - mask[::-1].argmax()

    zoom_dates = dates[start_idx:end_idx]
    zoom_values = values[start_idx:end_idx, :]

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx in indices:
        ax.plot(zoom_dates, zoom_values[:, idx], alpha=0.4, linewidth=1.2)

    zoom_positive = np.where(zoom_values > 0, zoom_values, np.nan)
    if np.isfinite(zoom_positive).any():
        with np.errstate(all="ignore"):
            zoom_mean = np.nanmean(zoom_positive, axis=1)
            zoom_median = np.nanmedian(zoom_positive, axis=1)
        ax.plot(zoom_dates, zoom_mean, color=Colors.GLOBAL_MEAN, linewidth=2.5, label="Mean")
        ax.plot(
            zoom_dates, zoom_median, color=Colors.GLOBAL_MEDIAN, linewidth=2.5, label="Median"
        )

    bounds = robust_bounds(zoom_values, positive_only=True)
    if bounds is not None:
        _, upper = bounds
        ax.set_ylim(bottom=0, top=upper * 1.05)

    format_date_axis(ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("log1p biomarker")
    ax.set_title(
        f"Biomarker Series ({variant_label}): {zoom_dates[0].strftime('%Y-%m-%d')} "
        f"to {zoom_dates[-1].strftime('%Y-%m-%d')}"
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved time window zoom")


def _select_regions_by_coverage(
    biomarker: xr.DataArray,
    region_ids: np.ndarray,
    top_n: int | None,
    bottom_n: int | None,
) -> list[int]:
    """Select regions by coverage ranking.

    Args:
        biomarker: Raw biomarker data
        region_ids: Array of region IDs
        top_n: Number of highest-coverage regions to select
        bottom_n: Number of lowest-coverage regions to select

    Returns:
        List of region indices
    """
    if top_n is None and bottom_n is None:
        return []

    raw_values = biomarker.values
    n_days = raw_values.shape[0]

    # Compute coverage for each region
    coverage = []
    for idx in range(len(region_ids)):
        valid_mask = np.isfinite(raw_values[:, idx])
        coverage.append((idx, valid_mask.sum() / n_days))

    coverage.sort(key=lambda x: x[1], reverse=True)

    selected = []
    if top_n is not None:
        selected.extend([idx for idx, _ in coverage[:top_n]])
    if bottom_n is not None:
        selected.extend([idx for idx, _ in coverage[-bottom_n:]])

    return list(dict.fromkeys(selected))  # Remove duplicates while preserving order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to Zarr dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/canonical_biomarker_series"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=12,
        help="Max regions to include in line plot",
    )
    parser.add_argument(
        "--heatmap-max-regions",
        type=int,
        default=30,
        help="Max regions to include in heatmap",
    )
    parser.add_argument(
        "--region-ids",
        type=str,
        default=None,
        help="Comma-separated region IDs to plot",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Select top N regions by coverage",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=None,
        help="Select bottom N regions by coverage (sparse data analysis)",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path to save CSV statistics",
    )
    parser.add_argument(
        "--focus-period-start",
        type=str,
        default=None,
        help="Start date for focused analysis (YYYY-MM-DD). If only this is set, shows first 20%% of data.",
    )
    parser.add_argument(
        "--focus-period-end",
        type=str,
        default=None,
        help="End date for focused analysis (YYYY-MM-DD). If only this is set, shows last 20%% of data.",
    )
    parser.add_argument(
        "--no-boxplot",
        action="store_true",
        help="Disable boxplot generation",
    )
    parser.add_argument(
        "--no-zoom",
        action="store_true",
        help="Disable time window zoom plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    biomarker_variants, data_start = _load_biomarkers(args.dataset)
    variant_names = list(biomarker_variants.keys())

    reference = biomarker_variants[variant_names[0]]
    region_ids = reference[REGION_COORD].values
    dates = pd.DatetimeIndex(reference[TEMPORAL_COORD].values)

    # Determine which region indices to analyze
    series_indices: list[int] = []
    heatmap_indices: list[int] = []

    # Priority: coverage-based selection > specific region IDs > random sampling
    coverage_indices = _select_regions_by_coverage(
        reference, region_ids, args.top_n, args.bottom_n
    )

    if coverage_indices:
        series_indices = coverage_indices
        heatmap_indices = coverage_indices
        logger.info(
            "Selected %d regions by coverage: %s",
            len(series_indices),
            region_ids[series_indices][:10].tolist(),
        )
    else:
        requested_ids = None
        if args.region_ids:
            requested_ids = [
                rid.strip() for rid in args.region_ids.split(",") if rid.strip()
            ]

        series_indices = _select_region_indices(
            region_ids,
            requested_ids=requested_ids,
            max_regions=args.max_regions,
            seed=args.seed,
        )

        heatmap_indices = _select_region_indices(
            region_ids,
            requested_ids=requested_ids,
            max_regions=args.heatmap_max_regions,
            seed=args.seed,
        )

    for variant_name in variant_names:
        biomarker = biomarker_variants[variant_name]
        raw_values = biomarker.values.astype(float)
        if data_start is not None:
            for idx, start in enumerate(data_start):
                if start < 0:
                    raw_values[:, idx] = np.nan
                else:
                    raw_values[: int(start), idx] = np.nan
        values = np.log1p(raw_values)
        variant_label = variant_name.replace("edar_biomarker_", "")

        # Compute and display summary statistics
        stats_df = compute_summary_statistics(biomarker, region_ids, series_indices)
        print_summary_statistics(stats_df, variant_label)

        # Save statistics to CSV if requested
        if args.stats_output:
            args.stats_output.parent.mkdir(parents=True, exist_ok=True)
            suffix = args.stats_output.suffix or ".csv"
            if len(variant_names) == 1:
                output_path = args.stats_output
            else:
                output_path = args.stats_output.with_name(
                    f"{args.stats_output.stem}_{variant_label}{suffix}"
                )
            stats_df.to_csv(output_path, index=False)
            logger.info("Saved statistics to %s", output_path)

        # Generate plots
        plot_region_series(
            dates,
            values,
            region_ids,
            series_indices,
            args.output_dir / f"biomarker_series_{variant_label}.png",
            variant_label,
        )

        plot_heatmap(
            dates,
            values,
            region_ids,
            heatmap_indices,
            args.output_dir / f"biomarker_heatmap_{variant_label}.png",
            variant_label,
        )

        plot_distribution(
            values,
            args.output_dir / f"biomarker_distribution_{variant_label}.png",
            variant_label,
        )

        # New plots
        if not args.no_boxplot:
            plot_per_region_boxplot(
                dates,
                values,
                region_ids,
                series_indices,
                stats_df,
                args.output_dir / f"biomarker_boxplots_{variant_label}.png",
                variant_label,
            )

        if not args.no_zoom:
            plot_time_window_zoom(
                dates,
                values,
                region_ids,
                series_indices,
                args.focus_period_start,
                args.focus_period_end,
                args.output_dir / f"biomarker_zoom_{variant_label}.png",
                variant_label,
            )


if __name__ == "__main__":
    main()
