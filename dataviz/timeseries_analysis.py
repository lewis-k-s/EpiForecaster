"""
Time series visualization of per-region and global epidemic data.

Analyzes normalization, interpolation, and smoothing effects on case data
by plotting a subset of regions alongside aggregated global series.
"""

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

from data.epi_dataset import EpiDataset  # noqa: E402
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from models.configs import EpiForecasterConfig  # noqa: E402

logger = logging.getLogger(__name__)


def compute_global_aggregation(
    cases_df: pd.DataFrame, population: pd.Series, method: str = "weighted_mean"
) -> pd.Series:
    """Aggregate region-level cases to global series.

    Args:
        cases_df: DataFrame with regions as columns
        population: Series with population per region
        method: Aggregation method ('sum', 'mean', 'weighted_mean')

    Returns:
        1D global series
    """
    if method == "sum":
        return cases_df.sum(axis=1)
    elif method == "mean":
        return cases_df.mean(axis=1)
    elif method == "weighted_mean":
        weights = population / population.sum()
        return (cases_df * weights).sum(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def per_100k_scaling(cases_df: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    """Scale cases to per-100k using population."""
    per_100k = 100000.0 / population.values
    return cases_df * per_100k


def normalize_per_series(values: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize time series per region.

    Args:
        values: (time, region) array
        method: 'standard' (z-score), 'minmax', 'robust' (median/IQR)

    Returns:
        Normalized array of same shape
    """
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {values.shape}")

    if method == "standard":
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        return (values - mean) / std
    elif method == "minmax":
        min_val = np.nanmin(values, axis=0, keepdims=True)
        max_val = np.nanmax(values, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val = np.maximum(range_val, 1e-6)
        return (values - min_val) / range_val
    elif method == "robust":
        median = np.nanmedian(values, axis=0, keepdims=True)
        q75 = np.nanpercentile(values, 75, axis=0, keepdims=True)
        q25 = np.nanpercentile(values, 25, axis=0, keepdims=True)
        iqr = q75 - q25
        iqr = np.maximum(iqr, 1e-6)
        return (values - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_series(
    values: np.ndarray, window: int = 7, method: str = "rolling"
) -> np.ndarray:
    """Smooth time series.

    Args:
        values: (time, region) or (time,) array
        window: Window size for smoothing
        method: 'rolling' (moving average), 'ewm' (exponential weighted)

    Returns:
        Smoothed array
    """
    if method == "rolling":
        df = pd.DataFrame(values)
        return df.rolling(window=window, center=True, min_periods=1).mean().values
    elif method == "ewm":
        df = pd.DataFrame(values)
        return df.ewm(halflife=window / 2, min_periods=1).mean().values
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def interpolate_missing(
    values: np.ndarray, method: str = "linear", limit: int = None
) -> np.ndarray:
    """Interpolate missing values.

    Args:
        values: Array with NaN values
        method: 'linear', 'forward', 'backward'
        limit: Max consecutive NaN to fill

    Returns:
        Array with interpolated values
    """
    df = pd.DataFrame(values)

    if method == "linear":
        return df.interpolate(method="linear", axis=0, limit=limit).values
    elif method == "forward":
        return df.ffill(limit=limit).values
    elif method == "backward":
        return df.bfill(limit=limit).values
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def plot_region_series(
    cases_df: pd.DataFrame,
    global_series: pd.Series,
    output_path: Path,
    title: str,
    max_regions: int = 15,
) -> None:
    """Plot subset of region series and global aggregate.

    Args:
        cases_df: DataFrame with dates as index, regions as columns
        global_series: Global aggregated series
        output_path: Path to save plot
        title: Plot title
        max_regions: Maximum number of regions to plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    n_regions = min(max_regions, cases_df.shape[1])
    regions_to_plot = cases_df.columns[:n_regions]

    for region in regions_to_plot:
        ax.plot(cases_df.index, cases_df[region], alpha=0.3, linewidth=1)

    ax.plot(
        cases_df.index,
        global_series,
        color="black",
        linewidth=2,
        label="Global aggregate",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


def plot_normalization_comparison(
    cases_np: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: Path,
    region_subset: list[int],
    region_names: list[str],
) -> None:
    """Compare normalization methods.

    Args:
        cases_np: (time, region) raw case values
        dates: Date index
        output_dir: Directory to save plots
        region_subset: Indices of regions to highlight
        region_names: Names of highlighted regions
    """
    methods = ["standard", "minmax", "robust"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for idx, method in enumerate(methods):
        normalized = normalize_per_series(cases_np, method=method)
        ax = axes[idx]

        n_highlight = min(5, len(region_subset))
        for _i, (r_idx, r_name) in enumerate(
            zip(region_subset[:n_highlight], region_names[:n_highlight], strict=False)
        ):
            ax.plot(dates, normalized[:, r_idx], alpha=0.7, linewidth=1.5, label=r_name)

        ax.set_ylabel(f"{method.capitalize()} normalized")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        if idx == 0:
            ax.set_title("Normalization Comparison")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    output_path = output_dir / "normalization_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved normalization comparison to %s", output_path)


def plot_interpolation_effects(
    cases_np: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: Path,
    region_idx: int = 0,
) -> None:
    """Visualize interpolation of missing values.

    Args:
        cases_np: (time, region) case values
        dates: Date index
        output_dir: Directory to save plot
        region_idx: Index of region to analyze
    """
    series = cases_np[:, region_idx].copy()

    num_missing = int(len(series) * 0.1)
    missing_indices = np.random.choice(len(series), num_missing, replace=False)
    series_with_gaps = series.copy()
    series_with_gaps[missing_indices] = np.nan

    methods = ["linear", "forward", "backward"]
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 8), sharex=True)

    for idx, method in enumerate(methods):
        interpolated = interpolate_missing(series_with_gaps, method=method)
        ax = axes[idx]

        ax.plot(dates, series, color="black", alpha=0.5, linewidth=2, label="Original")
        ax.plot(
            dates,
            interpolated,
            color="red",
            alpha=0.7,
            linewidth=1.5,
            label=f"Interpolated ({method})",
        )
        ax.scatter(
            dates[missing_indices],
            series[missing_indices],
            color="blue",
            s=10,
            alpha=0.5,
            label="Artificial gaps",
        )

        ax.set_ylabel("Cases")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_title(f"Interpolation Methods (Region {region_idx})")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    output_path = output_dir / "interpolation_effects.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved interpolation effects to %s", output_path)


def plot_smoothing_effects(
    cases_np: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: Path,
    region_subset: list[int],
    region_names: list[str],
) -> None:
    """Compare smoothing methods.

    Args:
        cases_np: (time, region) case values
        dates: Date index
        output_dir: Directory to save plot
        region_subset: Indices of regions to highlight
        region_names: Names of highlighted regions
    """
    windows = [3, 7, 14]
    methods = ["rolling", "ewm"]

    fig, axes = plt.subplots(len(windows), len(methods), figsize=(14, 12))

    n_highlight = min(2, len(region_subset))

    for w_idx, window in enumerate(windows):
        for m_idx, method in enumerate(methods):
            ax = axes[w_idx, m_idx]

            for r_idx, r_name in zip(
                region_subset[:n_highlight], region_names[:n_highlight], strict=False
            ):
                original = cases_np[:, r_idx]
                smoothed = smooth_series(original, window=window, method=method)

                ax.plot(
                    dates,
                    original,
                    color="black",
                    alpha=0.3,
                    linewidth=1,
                    label="Original" if r_idx == 0 else None,
                )
                ax.plot(
                    dates,
                    smoothed,
                    alpha=0.8,
                    linewidth=2,
                    label=f"{r_name} (w={window})",
                )

            ax.set_ylabel("Cases")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

            if w_idx == 0:
                ax.set_title(f"{method.capitalize()} smoothing")

    fig.suptitle("Smoothing Effects Comparison", y=1.005)
    plt.tight_layout()

    output_path = output_dir / "smoothing_effects.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved smoothing effects to %s", output_path)


def plot_missingness_heatmap(
    cases_da: xr.DataArray,
    output_dir: Path,
    max_regions: int = 50,
) -> None:
    """Visualize missing data pattern.

    Args:
        cases_da: (time, region) cases DataArray
        output_dir: Directory to save plot
        max_regions: Maximum number of regions to show
    """
    cases_subset = cases_da.isel({REGION_COORD: slice(0, max_regions)})
    missing_mask = cases_subset.isnull().values

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        missing_mask.T,
        cbar_kws={"label": "Missing"},
        cmap="Reds",
        xticklabels=30,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(f"Missing Data Pattern ({max_regions} regions)")
    ax.set_xlabel("Time index")

    output_path = output_dir / "missingness_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved missingness heatmap to %s", output_path)


def plot_scale_statistics(
    cases_df: pd.DataFrame,
    population: pd.Series,
    output_dir: Path,
) -> None:
    """Plot scale-related statistics.

    Args:
        cases_df: DataFrame with cases data
        population: Series with population per region
        output_dir: Directory to save plot
    """
    cases_np = cases_df.values
    per_100k_np = (cases_np * (100000.0 / np.array(population.values))).T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mean_cases = np.nanmean(cases_np, axis=0)
    std_cases = np.nanstd(cases_np, axis=0)

    mean_per100k = np.nanmean(per_100k_np, axis=0)
    std_per100k = np.nanstd(per_100k_np, axis=0)

    sns.histplot(mean_cases, bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Mean Cases (Raw)")
    axes[0, 0].set_xlabel("Mean cases")
    axes[0, 0].set_ylabel("Regions")

    sns.histplot(mean_per100k, bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Mean Cases (per-100k)")
    axes[0, 1].set_xlabel("Mean cases (per-100k)")
    axes[0, 1].set_ylabel("Regions")

    sns.scatterplot(x=mean_cases, y=std_cases, alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Mean vs Std (Raw)")
    axes[1, 0].set_xlabel("Mean cases")
    axes[1, 0].set_ylabel("Std cases")

    sns.scatterplot(x=mean_per100k, y=std_per100k, alpha=0.5, ax=axes[1, 1])
    axes[1, 1].set_title("Mean vs Std (per-100k)")
    axes[1, 1].set_xlabel("Mean cases (per-100k)")
    axes[1, 1].set_ylabel("Std cases (per-100k)")

    plt.tight_layout()

    output_path = output_dir / "scale_statistics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved scale statistics to %s", output_path)


def analyze_normalization(cases_np: np.ndarray) -> dict[str, str]:
    """Analyze normalization characteristics.

    Args:
        cases_np: (time, region) case values

    Returns:
        Analysis findings as dict
    """
    findings = {}

    mean_std = np.nanstd(cases_np, axis=0).mean()
    min_std = np.nanstd(cases_np, axis=0).min()
    max_std = np.nanstd(cases_np, axis=0).max()

    findings["scale_variability"] = (
        f"Region std dev ranges from {min_std:.2f} to {max_std:.2f} (mean: {mean_std:.2f})"
    )

    skewness = ((cases_np - np.nanmean(cases_np, axis=0, keepdims=True)) ** 3).mean(
        axis=0
    ) / (np.nanstd(cases_np, axis=0) ** 3 + 1e-6)
    findings["skewness"] = (
        f"Mean skewness: {np.nanmean(skewness):.2f}, Max skewness: {np.nanmax(skewness):.2f}"
    )

    zero_fraction = (cases_np == 0).sum(axis=0) / len(cases_np)
    findings["sparsity"] = (
        f"Mean zero fraction: {zero_fraction.mean():.3f}, Max: {zero_fraction.max():.3f}"
    )

    return findings


def analyze_interpolation(cases_da: xr.DataArray) -> dict[str, str]:
    """Analyze interpolation needs.

    Args:
        cases_da: (time, region) cases DataArray

    Returns:
        Analysis findings as dict
    """
    findings = {}

    missing_mask = cases_da.isnull().values
    missing_per_region = missing_mask.sum(axis=0)
    missing_per_time = missing_mask.sum(axis=1)

    findings["missing_by_region"] = (
        f"Mean missing per region: {missing_per_region.mean():.1f}, Max: {missing_per_region.max():.1f}"
    )
    findings["missing_by_time"] = (
        f"Mean missing per time: {missing_per_time.mean():.1f}, Max: {missing_per_time.max():.1f}"
    )

    total_missing = missing_mask.sum()
    total_values = missing_mask.size
    findings["overall_missing"] = f"Overall missing: {total_missing / total_values:.3%}"

    consecutive_missing = []
    for region in range(missing_mask.shape[1]):
        series = missing_mask[:, region]
        consecutive = 0
        max_consecutive = 0
        for val in series:
            if val:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        consecutive_missing.append(max_consecutive)

    findings["consecutive_missing"] = (
        f"Max consecutive missing per region: mean {np.mean(consecutive_missing):.1f}, max {np.max(consecutive_missing):.1f}"
    )

    return findings


def analyze_smoothing(cases_np: np.ndarray) -> dict[str, str]:
    """Analyze smoothing characteristics.

    Args:
        cases_np: (time, region) case values

    Returns:
        Analysis findings as dict
    """
    findings = {}

    autocorr_7d = []
    for region in range(cases_np.shape[1]):
        series = pd.Series(cases_np[:, region])
        autocorr_7d.append(series.autocorr(lag=7))

    autocorr_7d = np.array([a for a in autocorr_7d if not np.isnan(a)])
    findings["autocorr_7d"] = (
        f"Mean 7-day autocorrelation: {np.mean(autocorr_7d):.3f}, Range: [{np.min(autocorr_7d):.3f}, {np.max(autocorr_7d):.3f}]"
    )

    weekly_variation = []
    for region in range(cases_np.shape[1]):
        series = pd.Series(cases_np[:, region])
        if len(series) >= 7:
            std_rolling = (
                series.rolling(window=7, center=True, min_periods=1).std().mean()
            )
            std_total = series.std()
            if std_total > 0:
                weekly_variation.append(std_rolling / std_total)

    weekly_variation = np.array([v for v in weekly_variation if not np.isnan(v)])
    findings["weekly_variation"] = (
        f"Mean weekly/total std ratio: {np.mean(weekly_variation):.3f}"
    )

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time series visualization and analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_epifor_full.yaml"),
        help="Training config path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/timeseries_analysis"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=15,
        help="Maximum number of regions to plot",
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

    cfg = EpiForecasterConfig.from_file(str(args.config))

    dataset = xr.open_zarr(cfg.data.dataset_path)
    num_nodes = dataset[REGION_COORD].size
    dataset.close()

    target_nodes = list(range(num_nodes))
    context_nodes = list(range(num_nodes))

    epi_dataset = EpiDataset(
        cfg,
        target_nodes=target_nodes,
        context_nodes=context_nodes,
    )

    cases_da = epi_dataset.dataset.cases
    population = epi_dataset.dataset.population
    dates = pd.DatetimeIndex(epi_dataset.dataset[TEMPORAL_COORD].values)
    regions = list(epi_dataset.dataset[REGION_COORD].values)

    if "valid_targets" in epi_dataset.dataset:
        valid_mask = epi_dataset.dataset.valid_targets.values.astype(bool)
        cases_da = cases_da.isel({REGION_COORD: valid_mask})
        population = population.isel({REGION_COORD: valid_mask})
        regions = [r for r, v in zip(regions, valid_mask, strict=False) if v]
        logger.info(
            "Using valid_targets filter: %d regions (out of %d total)",
            valid_mask.sum(),
            valid_mask.size,
        )
    else:
        logger.info("No valid_targets filter found, using all regions")

    logger.info("Time steps: %d, Regions: %d", len(dates), len(regions))

    population_series = pd.Series(population.values, index=regions)

    cases_df = pd.DataFrame(cases_da.values, index=dates, columns=regions)
    cases_per100k_df = per_100k_scaling(cases_df, population_series)

    global_weighted = compute_global_aggregation(
        cases_per100k_df, population_series, method="weighted_mean"
    )

    plot_region_series(
        cases_per100k_df,
        pd.Series(global_weighted.values, index=dates),
        output_dir / "cases_per100k.png",
        f"Cases per-100k with Global Weighted Mean ({args.max_regions} regions)",
        max_regions=args.max_regions,
    )

    region_subset = list(range(min(10, len(regions))))
    region_names = [str(regions[i]) for i in region_subset]

    plot_normalization_comparison(
        cases_per100k_df.values,
        dates,
        output_dir,
        region_subset,
        region_names,
    )

    plot_interpolation_effects(
        cases_per100k_df.values,
        dates,
        output_dir,
        region_idx=0,
    )

    plot_smoothing_effects(
        cases_per100k_df.values,
        dates,
        output_dir,
        region_subset,
        region_names,
    )

    plot_missingness_heatmap(
        cases_da,
        output_dir,
        max_regions=50,
    )

    plot_scale_statistics(
        cases_df,
        population_series,
        output_dir,
    )

    norm_findings = analyze_normalization(cases_per100k_df.values)
    interp_findings = analyze_interpolation(cases_da)
    smooth_findings = analyze_smoothing(cases_per100k_df.values)

    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Time Series Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("NORMALIZATION ANALYSIS\n")
        f.write("-" * 50 + "\n")
        for key, value in norm_findings.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("INTERPOLATION ANALYSIS\n")
        f.write("-" * 50 + "\n")
        for key, value in interp_findings.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("SMOOTHING ANALYSIS\n")
        f.write("-" * 50 + "\n")
        for key, value in smooth_findings.items():
            f.write(f"{key}: {value}\n")

    logger.info("Saved analysis summary to %s", summary_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
