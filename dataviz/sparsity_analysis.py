"""
High sparsity region analysis with geographic visualization.

Creates choropleth maps and time-aggregated scatter plots to analyze
sparsity patterns across regions and time windows.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
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


def compute_windowed_sparsity(
    cases_da: xr.DataArray, window_size: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sparsity per region over time windows.

    Args:
        cases_da: (time, region) cases data
        window_size: Size of time window for aggregation

    Returns:
        sparsity_matrix: (num_windows, num_regions) sparsity percentages
        window_starts: Array of window start indices
    """
    missing_mask = cases_da.isnull().values
    T, N = missing_mask.shape

    num_windows = T // window_size
    sparsity_matrix = np.full((num_windows, N), np.nan, dtype=np.float32)

    for w in range(num_windows):
        start = w * window_size
        end = min((w + 1) * window_size, T)
        window_mask = missing_mask[start:end]
        sparsity_matrix[w] = window_mask.sum(axis=0) / window_size * 100

    window_starts = np.arange(num_windows) * window_size
    return sparsity_matrix, window_starts


def compute_overall_sparsity(cases_da: xr.DataArray) -> pd.Series:
    """Compute overall sparsity percentage per region.

    Args:
        cases_da: (time, region) cases data

    Returns:
        Series with region_id as index, sparsity percentage as values
    """
    missing_mask = cases_da.isnull().values
    sparsity_per_region = missing_mask.sum(axis=0) / missing_mask.shape[0] * 100

    return pd.Series(
        sparsity_per_region,
        index=cases_da[REGION_COORD].values,
        name="sparsity_percent",
    )


def compute_consecutive_missing(cases_da: xr.DataArray) -> pd.Series:
    """Compute max consecutive missing values per region.

    Args:
        cases_da: (time, region) cases data

    Returns:
        Series with region_id as index, max consecutive missing as values
    """
    missing_mask = cases_da.isnull().values

    max_consecutive = []
    for region in range(missing_mask.shape[1]):
        series = missing_mask[:, region]
        consecutive = 0
        max_consecutive_region = 0
        for val in series:
            if val:
                consecutive += 1
                max_consecutive_region = max(max_consecutive_region, consecutive)
            else:
                consecutive = 0
        max_consecutive.append(max_consecutive_region)

    return pd.Series(
        max_consecutive,
        index=cases_da[REGION_COORD].values,
        name="max_consecutive_missing",
    )


def plot_choropleth(
    geo_df: gpd.GeoDataFrame,
    sparsity_series: pd.Series,
    output_path: Path,
    title: str,
    vmin: float = 0,
    vmax: float = 100,
) -> None:
    """Create choropleth map of sparsity.

    Args:
        geo_df: GeoDataFrame with region geometries
        sparsity_series: Series mapping region_id to sparsity percentage
        output_path: Path to save plot
        title: Plot title
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    merged = geo_df.merge(
        sparsity_series.rename("sparsity_percent"),
        left_index=True,
        right_index=True,
        how="left",
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    merged.plot(
        column="sparsity_percent",
        cmap="Reds",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "lightgray", "label": "No data"},
        edgecolor="black",
        linewidth=0.2,
        ax=ax,
    )

    cbar = ax.get_figure().axes[-1]
    cbar.set_ylabel("Sparsity (%)", rotation=270, labelpad=20)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved choropleth map to %s", output_path)


def plot_windowed_sparsity_scatter(
    sparsity_matrix: np.ndarray,
    window_starts: np.ndarray,
    region_ids: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: Path,
    top_n_regions: int = 20,
) -> None:
    """Create scatter plot of sparsity over time windows for high-sparsity regions.

    Args:
        sparsity_matrix: (num_windows, num_regions) sparsity percentages
        window_starts: Array of window start indices
        region_ids: Array of region IDs
        dates: Date index for the full time series
        output_dir: Directory to save plot
        top_n_regions: Number of highest-sparsity regions to highlight
    """
    num_windows, num_regions = sparsity_matrix.shape

    mean_sparsity = np.nanmean(sparsity_matrix, axis=0)
    top_indices = np.argsort(mean_sparsity)[-top_n_regions:]

    fig, ax = plt.subplots(figsize=(14, 8))

    for idx in top_indices:
        region_id = region_ids[idx]
        sparsity_values = sparsity_matrix[:, idx]
        window_dates = dates[window_starts + window_starts[0] // 2]
        ax.scatter(
            window_dates,
            sparsity_values,
            alpha=0.7,
            s=20,
            label=f"{region_id} ({mean_sparsity[idx]:.1f}%)",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title(f"Sparsity Over Time (Top {top_n_regions} Highest-Sparsity Regions)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "windowed_sparsity_scatter.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved windowed sparsity scatter to %s", output_path)


def plot_sparsity_distribution(
    sparsity_series: pd.Series,
    consecutive_series: pd.Series,
    output_dir: Path,
) -> None:
    """Plot distribution of sparsity and consecutive missing values.

    Args:
        sparsity_series: Series with sparsity percentages
        consecutive_series: Series with max consecutive missing
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(sparsity_series, bins=50, ax=axes[0])
    axes[0].set_title("Distribution of Sparsity Across Regions")
    axes[0].set_xlabel("Sparsity (%)")
    axes[0].set_ylabel("Number of Regions")
    axes[0].axvline(
        sparsity_series.median(),
        color="red",
        linestyle="--",
        label=f"Median: {sparsity_series.median():.1f}%",
    )
    axes[0].legend()

    sns.histplot(consecutive_series, bins=50, ax=axes[1])
    axes[1].set_title("Distribution of Max Consecutive Missing")
    axes[1].set_xlabel("Max Consecutive Missing (days)")
    axes[1].set_ylabel("Number of Regions")
    axes[1].axvline(
        consecutive_series.median(),
        color="red",
        linestyle="--",
        label=f"Median: {consecutive_series.median():.0f}",
    )
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / "sparsity_distribution.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved sparsity distribution plot to %s", output_path)


def plot_sparsity_heatmap(
    sparsity_matrix: np.ndarray,
    region_ids: np.ndarray,
    dates: pd.DatetimeIndex,
    window_starts: np.ndarray,
    output_dir: Path,
    max_regions: int = 50,
) -> None:
    """Create heatmap of sparsity over time and regions.

    Args:
        sparsity_matrix: (num_windows, num_regions) sparsity percentages
        region_ids: Array of region IDs
        dates: Date index for the full time series
        window_starts: Array of window start indices
        output_dir: Directory to save plot
        max_regions: Maximum number of regions to show
    """
    num_regions = min(max_regions, sparsity_matrix.shape[1])

    sorted_mean_sparsity = np.nanmean(sparsity_matrix, axis=0)
    top_indices = np.argsort(sorted_mean_sparsity)[-num_regions:]

    sparsity_subset = sparsity_matrix[:, top_indices]
    region_ids_subset = region_ids[top_indices]

    window_dates = [dates[start + window_starts[0] // 2] for start in window_starts]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        sparsity_subset.T,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(0, len(window_dates), len(window_dates) // 10))
    ax.set_xticklabels(
        [str(d.date()) for d in window_dates[:: len(window_dates) // 10]],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(num_regions))
    ax.set_yticklabels(region_ids_subset, fontsize=8)

    ax.set_xlabel("Time Window")
    ax.set_ylabel("Region")
    ax.set_title(f"Sparsity Heatmap (Top {num_regions} Highest-Sparsity Regions)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sparsity (%)")

    plt.tight_layout()
    output_path = output_dir / "sparsity_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sparsity heatmap to %s", output_path)


def plot_sparsity_vs_population(
    sparsity_series: pd.Series,
    population: xr.DataArray,
    output_dir: Path,
) -> None:
    """Plot relationship between sparsity and population.

    Args:
        sparsity_series: Series with sparsity percentages
        population: (region,) population array
        output_dir: Directory to save plot
    """
    pop_df = pd.DataFrame(
        {"population": population.values, "sparsity": sparsity_series.values},
        index=sparsity_series.index,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pop_df["population"], pop_df["sparsity"], alpha=0.5, s=30)

    ax.set_xlabel("Population")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Sparsity vs Population")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "sparsity_vs_population.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved sparsity vs population plot to %s", output_path)


def write_high_sparsity_report(
    sparsity_series: pd.Series,
    consecutive_series: pd.Series,
    sparsity_matrix: np.ndarray,
    region_ids: np.ndarray,
    dates: pd.DatetimeIndex,
    window_starts: np.ndarray,
    output_dir: Path,
    top_n: int = 30,
) -> None:
    """Write detailed report on high-sparsity regions.

    Args:
        sparsity_series: Series with sparsity percentages
        consecutive_series: Series with max consecutive missing
        sparsity_matrix: (num_windows, num_regions) sparsity percentages
        region_ids: Array of region IDs
        dates: Date index for the full time series
        window_starts: Array of window start indices
        output_dir: Directory to save report
        top_n: Number of top sparsity regions to report
    """
    report_path = output_dir / "high_sparsity_report.txt"

    with open(report_path, "w") as f:
        f.write("High Sparsity Region Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        top_indices = sparsity_series.nlargest(top_n).index

        f.write(f"Top {top_n} Highest Sparsity Regions\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Region':<20} {'Sparsity (%)':<15} {'Max Consecutive':<20}\n")
        f.write("-" * 60 + "\n")

        for region_id in top_indices:
            f.write(
                f"{region_id:<20} {sparsity_series[region_id]:<15.2f} "
                f"{consecutive_series[region_id]:<20}\n"
            )

        f.write("\n\n")
        f.write("Summary Statistics\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total regions: {len(sparsity_series)}\n")
        f.write(f"Mean sparsity: {sparsity_series.mean():.2f}%\n")
        f.write(f"Median sparsity: {sparsity_series.median():.2f}%\n")
        f.write(f"Max sparsity: {sparsity_series.max():.2f}%\n")
        f.write(f"Regions with >50% sparsity: {(sparsity_series > 50).sum()}\n")
        f.write(f"Regions with >75% sparsity: {(sparsity_series > 75).sum()}\n")
        f.write(f"Regions with >90% sparsity: {(sparsity_series > 90).sum()}\n")

        f.write("\n")
        f.write(f"Mean max consecutive missing: {consecutive_series.mean():.1f}\n")
        f.write(f"Median max consecutive missing: {consecutive_series.median():.0f}\n")
        f.write(f"Max consecutive missing: {consecutive_series.max():.0f}\n")

        f.write(f"\nTime windows analyzed: {len(window_starts)}\n")
        f.write(f"Date range: {dates[0].date()} to {dates[-1].date()}\n")

    logger.info("Saved high sparsity report to %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="High sparsity region analysis")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_epifor_full.yaml"),
        help="Training config path for dataset loading",
    )
    parser.add_argument(
        "--geo-path",
        type=str,
        default="data/files/geo/fl_municipios_catalonia.geojson",
        help="Path to geojson file with region boundaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/sparsity_analysis"),
        help="Directory to save plots and reports",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=14,
        help="Time window size in days for aggregation",
    )
    parser.add_argument(
        "--top-regions",
        type=int,
        default=20,
        help="Number of top sparsity regions to highlight in plots",
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

    # Load dataset to get node count
    dataset = xr.open_zarr(cfg.data.dataset_path)
    num_nodes = dataset[REGION_COORD].size
    dataset.close()

    # Use all nodes as targets and context
    target_nodes = list(range(num_nodes))
    context_nodes = list(range(num_nodes))

    epi_dataset = EpiDataset(
        cfg,
        target_nodes=target_nodes,
        context_nodes=context_nodes,
    )

    dataset = epi_dataset.dataset
    logger.info("Loaded dataset: %s", cfg.data.dataset_path)
    logger.info("Variables: %s", list(dataset.keys()))

    cases_da = dataset.cases
    population = dataset.population
    dates = pd.DatetimeIndex(dataset[TEMPORAL_COORD].values)
    regions = list(dataset[REGION_COORD].values)

    # Filter by valid_targets if available
    if "valid_targets" in dataset:
        valid_mask = dataset.valid_targets.values.astype(bool)
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

    geo_df = gpd.read_file(args.geo_path)
    logger.info("Loaded geo data: %d features", len(geo_df))

    geo_df.index = geo_df["id"].astype(str)

    sparsity_series = compute_overall_sparsity(cases_da)
    consecutive_series = compute_consecutive_missing(cases_da)
    sparsity_matrix, window_starts = compute_windowed_sparsity(
        cases_da, args.window_size
    )

    plot_choropleth(
        geo_df,
        sparsity_series,
        output_dir / "sparsity_choropleth.png",
        "Overall Sparsity Across Regions",
    )

    plot_choropleth(
        geo_df,
        consecutive_series.rename("max_consecutive_missing"),
        output_dir / "consecutive_missing_choropleth.png",
        "Max Consecutive Missing Days Across Regions",
    )

    plot_windowed_sparsity_scatter(
        sparsity_matrix,
        window_starts,
        np.array(regions),
        dates,
        output_dir,
        top_n_regions=args.top_regions,
    )

    plot_sparsity_distribution(sparsity_series, consecutive_series, output_dir)

    plot_sparsity_heatmap(
        sparsity_matrix,
        np.array(regions),
        dates,
        window_starts,
        output_dir,
        max_regions=args.top_regions,
    )

    plot_sparsity_vs_population(sparsity_series, population, output_dir)

    write_high_sparsity_report(
        sparsity_series,
        consecutive_series,
        sparsity_matrix,
        np.array(regions),
        dates,
        window_starts,
        output_dir,
        top_n=args.top_regions,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
