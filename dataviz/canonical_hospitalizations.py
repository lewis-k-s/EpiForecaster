"""
Visualize canonical daily municipality-level hospitalization data from Zarr.

Outputs:
- Daily time series by municipality (line plots with mean/median)
- Weekly aggregation comparison (raw weekly vs daily sum)
- Municipality heatmap (municipality × date)
- Age channel visualization (heatmap overlay)
- Mask coverage visualization (where data is available)
- Summary statistics table (console and CSV)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402

logger = logging.getLogger(__name__)


def _robust_bounds(
    values: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    positive_only: bool = True,
) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if positive_only:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [lower, upper])
    return float(low), float(high)


def _load_hospitalizations(dataset_path: Path, run_id: str = "real") -> xr.Dataset:
    """Load hospitalization data from Zarr dataset with run_id filtering."""
    from data.epi_dataset import EpiDataset

    dataset = EpiDataset.load_canonical_dataset(
        aligned_data_path=dataset_path,
        run_id=run_id,
        run_id_chunk_size=1,
    )

    if "hospitalizations" not in dataset:
        raise ValueError("Dataset missing hospitalizations variable")

    logger.info(
        "Loaded hospitalizations (run_id=%s): %d dates x %d regions",
        run_id,
        dataset.sizes[TEMPORAL_COORD],
        dataset.sizes[REGION_COORD],
    )

    return dataset


def _load_config(config_path: Path) -> dict:
    """Load training config to get dataset path."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_dataset_path_from_config(config: dict) -> Path:
    """Extract dataset path from config."""
    # Try different possible locations for dataset path
    dataset_path = config.get("dataset", {}).get("path")
    if dataset_path is None:
        dataset_path = config.get("data", {}).get("dataset_path")
    if dataset_path is None:
        # Try to construct from preprocessing config
        dataset_path = config.get("preprocessing", {}).get("output_path")

    if dataset_path is None:
        raise ValueError("Could not find dataset path in config")

    return Path(dataset_path)


def plot_municipality_series(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    """Plot daily time series for selected municipalities."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx in indices:
        ax.plot(dates, values[:, idx], alpha=0.35, linewidth=1)

    values_positive = np.where(values > 0, values, np.nan)
    if np.isfinite(values_positive).any():
        with np.errstate(all="ignore"):
            global_mean = np.nanmean(values_positive, axis=1)
            global_median = np.nanmedian(values_positive, axis=1)
        ax.plot(dates, global_mean, color="black", linewidth=2, label="Global mean")
        ax.plot(
            dates, global_median, color="tab:blue", linewidth=2, label="Global median"
        )

    bounds = _robust_bounds(values[:, indices])
    if bounds is not None:
        _, upper = bounds
        ax.set_ylim(bottom=0, top=upper * 1.05)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Hospitalizations")
    ax.set_title("Daily Hospitalizations by Municipality")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved municipality series to %s", output_path)


def plot_weekly_comparison(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    """Compare daily smoothed data with weekly aggregation."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx in indices[:5]:  # Show fewer for clarity
        region_values = values[:, idx]
        axes[0].plot(
            dates,
            region_values,
            alpha=0.6,
            linewidth=1.5,
            label=f"Muni {region_ids[idx]}",
        )

    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Daily Hospitalizations")
    axes[0].set_title("Daily Hospitalizations (Smoothed)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    weekly_values = []
    weekly_dates = []
    for i in range(0, len(dates), 7):
        end_idx = min(i + 7, len(dates))
        week_sum = np.nansum(values[i:end_idx, indices[:5]], axis=0)
        weekly_values.append(week_sum)
        weekly_dates.append(dates[i])

    weekly_values = np.array(weekly_values)

    for j, idx in enumerate(indices[:5]):
        axes[1].plot(
            weekly_dates,
            weekly_values[:, j],
            alpha=0.6,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=f"Muni {region_ids[idx]}",
        )

    axes[1].set_xlabel("Week Start Date")
    axes[1].set_ylabel("Weekly Hospitalizations (Sum)")
    axes[1].set_title("Weekly Aggregated Hospitalizations (7-day Sum)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved weekly comparison to %s", output_path)


def plot_municipality_heatmap(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    """Plot heatmap of daily hospitalizations (municipality × date)."""
    subset = values[:, indices]
    heatmap_df = pd.DataFrame(subset, index=dates, columns=region_ids[indices])

    fig, ax = plt.subplots(figsize=(14, max(6, len(indices) * 0.3)))

    bounds = _robust_bounds(subset)
    vmin = 0.0
    vmax = bounds[1] if bounds is not None else None

    sns.heatmap(
        heatmap_df.T,
        ax=ax,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Daily Hospitalizations"},
    )

    step = max(1, len(dates) // 12)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Municipality")
    ax.set_title("Hospitalizations Heatmap by Municipality")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved municipality heatmap to %s", output_path)


def plot_age_channel(
    dates: pd.DatetimeIndex,
    age_values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    """Plot age channel as heatmap overlay."""
    subset = age_values[:, indices]
    heatmap_df = pd.DataFrame(subset, index=dates, columns=region_ids[indices])

    fig, ax = plt.subplots(figsize=(14, max(6, len(indices) * 0.3)))

    sns.heatmap(
        heatmap_df.T,
        ax=ax,
        cmap="viridis",
        vmin=1,
        vmax=7,
        cbar_kws={"label": "Age (days since last observation)"},
    )

    step = max(1, len(dates) // 12)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Municipality")
    ax.set_title(
        "Age Channel: Days Since Last Observation (1=original, 7=interpolated)"
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved age channel to %s", output_path)


def plot_mask_coverage(
    dates: pd.DatetimeIndex,
    mask_values: np.ndarray,
    region_ids: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    """Plot mask coverage showing data availability."""
    subset = mask_values[:, indices]
    heatmap_df = pd.DataFrame(subset, index=dates, columns=region_ids[indices])

    fig, ax = plt.subplots(figsize=(14, max(6, len(indices) * 0.3)))

    sns.heatmap(
        heatmap_df.T,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mask (1=available, 0=missing)"},
    )

    step = max(1, len(dates) // 12)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Municipality")
    ax.set_title("Data Availability Mask")

    coverage = np.mean(mask_values[:, indices])
    ax.text(
        0.02,
        0.98,
        f"Overall coverage: {coverage:.1%}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved mask coverage to %s", output_path)


def compute_summary_statistics(
    hosp_da: xr.DataArray,
    region_ids: np.ndarray,
    indices: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-municipality summary statistics."""
    if indices is None:
        indices = list(range(len(region_ids)))

    stats = []
    raw_values = hosp_da.values
    n_total_days = raw_values.shape[0]

    for idx in indices:
        region_data = raw_values[:, idx]
        region_id = str(region_ids[idx])

        valid_mask = np.isfinite(region_data)
        valid_data = region_data[valid_mask]
        n_valid = valid_mask.sum()
        coverage = n_valid / n_total_days

        sparsity = (valid_data == 0).sum() / n_valid if n_valid > 0 else np.nan

        log_valid = np.log1p(valid_data) if n_valid > 0 else np.array([])
        mean_val = float(np.mean(log_valid)) if n_valid > 0 else np.nan
        median_val = float(np.median(log_valid)) if n_valid > 0 else np.nan
        std_val = float(np.std(log_valid)) if n_valid > 0 else np.nan
        min_val = float(np.min(log_valid)) if n_valid > 0 else np.nan
        max_val = float(np.max(log_valid)) if n_valid > 0 else np.nan

        p25 = float(np.percentile(log_valid, 25)) if n_valid > 0 else np.nan
        p75 = float(np.percentile(log_valid, 75)) if n_valid > 0 else np.nan

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


def print_summary_statistics(stats_df: pd.DataFrame) -> None:
    """Print summary statistics table to console."""
    print("\n" + "=" * 100)
    print("CANONICAL HOSPITALIZATIONS SUMMARY STATISTICS")
    print("=" * 100)

    display_cols = [
        "region_id",
        "coverage",
        "sparsity",
        "mean_log1p",
        "median_log1p",
        "std_log1p",
        "quality_flag",
    ]

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

    print("\n" + "-" * 100)
    quality_counts = stats_df["quality_flag"].value_counts()
    print("Quality flag distribution:")
    for flag, count in quality_counts.items():
        print(f"  {flag}: {count}")
    print("=" * 100 + "\n")


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


def _select_regions_by_coverage(
    hosp_da: xr.DataArray,
    region_ids: np.ndarray,
    top_n: int | None,
    bottom_n: int | None,
) -> list[int]:
    """Select regions by coverage ranking."""
    if top_n is None and bottom_n is None:
        return []

    raw_values = hosp_da.values
    n_days = raw_values.shape[0]

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

    return list(dict.fromkeys(selected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to Zarr dataset (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (to extract dataset path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/canonical_hospitalizations"),
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
        help="Select bottom N regions by coverage",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path to save CSV statistics",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to filter data (default: 'real' or from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset path and run_id
    if args.dataset:
        dataset_path = args.dataset
    elif args.config:
        config = _load_config(args.config)
        dataset_path = _get_dataset_path_from_config(config)
    else:
        raise ValueError("Must provide either --dataset or --config")

    # Get run_id from CLI, config, or default to "real"
    run_id = args.run_id
    if run_id is None and args.config:
        config = _load_config(args.config)
        run_id = config.get("data", {}).get("run_id", "real")
    if run_id is None:
        run_id = "real"

    dataset = _load_hospitalizations(dataset_path, run_id=run_id)
    hosp_da = dataset["hospitalizations"]

    region_ids = hosp_da[REGION_COORD].values
    dates = pd.DatetimeIndex(hosp_da[TEMPORAL_COORD].values)

    # Load mask and age channels if available (already filtered by run_id)
    mask_da = dataset.get("hospitalizations_mask")
    age_da = dataset.get("hospitalizations_age")

    # Determine which region indices to analyze
    series_indices: list[int] = []
    heatmap_indices: list[int] = []

    coverage_indices = _select_regions_by_coverage(
        hosp_da, region_ids, args.top_n, args.bottom_n
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

    # Compute and display summary statistics
    stats_df = compute_summary_statistics(hosp_da, region_ids, series_indices)
    print_summary_statistics(stats_df)

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(args.stats_output, index=False)
        logger.info("Saved statistics to %s", args.stats_output)

    raw_values = hosp_da.values.astype(float)
    values = np.log1p(raw_values)

    # Generate plots
    plot_municipality_series(
        dates,
        values,
        region_ids,
        series_indices,
        args.output_dir / "municipality_series.png",
    )

    plot_weekly_comparison(
        dates,
        raw_values,
        region_ids,
        series_indices,
        args.output_dir / "weekly_comparison.png",
    )

    plot_municipality_heatmap(
        dates,
        values,
        region_ids,
        heatmap_indices,
        args.output_dir / "municipality_heatmap.png",
    )

    if age_da is not None:
        age_values = age_da.values
        plot_age_channel(
            dates,
            age_values,
            region_ids,
            heatmap_indices,
            args.output_dir / "age_channel.png",
        )

    if mask_da is not None:
        mask_values = mask_da.values
        plot_mask_coverage(
            dates,
            mask_values,
            region_ids,
            heatmap_indices,
            args.output_dir / "mask_coverage.png",
        )


if __name__ == "__main__":
    main()
