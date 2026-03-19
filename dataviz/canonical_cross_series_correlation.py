"""
Canonical cross-series correlation plots for observation series.

Generates scatter plots showing baseline correlations between observation series
in canonical data (cases, hospitalizations, deaths, wastewater). This establishes
expected correlation patterns to help interpret ablation knockout results.

Key design:
- Wastewater: Mean of biomarker variants (N1, N2, IP4)
- Masking: Use only mask=1 (truly observed) points, exclude interpolated/imputed
- Aggregation: Per-region means (not temporal correlations)

Outputs:
- scatter_grid_per_region.png: 4x4 scatter grid (per-region means)
- scatter_grid_pooled.png: 4x4 scatter grid (all points pooled)
- correlation_heatmap.png: Pearson/Spearman correlation heatmap
- correlations.csv: Correlation coefficients table
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from scipy import stats

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from utils.plotting import save_figure  # noqa: E402

logger = logging.getLogger(__name__)

# Default series names and their dataset variable names
SERIES_VARIABLES = {
    "cases": "cases",
    "hospitalizations": "hospitalizations",
    "deaths": "deaths",
    "wastewater": "biomarker_mean",  # Computed from edar_biomarker_* variants
}

# Human-readable labels with scale information
SERIES_LABELS = {
    "cases": "Cases log₁₊(per-100k)",
    "hospitalizations": "Hospitalizations log₁₊(per-100k)",
    "deaths": "Deaths log₁₊(per-100k)",
    "wastewater": "Wastewater log₁₊(raw)",
}


@dataclass
class CorrelationResult:
    """Container for correlation statistics between two series."""

    series_a: str
    series_b: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_points: int


@dataclass
class CorrelationData:
    """Container for loaded observation series data."""

    values: dict[str, np.ndarray]  # series_name -> 2D array (time, region)
    masks: dict[str, np.ndarray]  # series_name -> 2D bool array (time, region)
    region_ids: np.ndarray
    dates: np.ndarray


# =============================================================================
# DATA LOADING
# =============================================================================


def _discover_biomarker_variants(dataset: xr.Dataset) -> list[str]:
    """Find all edar_biomarker_* variables excluding _mask, _age, _censor suffixes."""
    variants = []
    for name in dataset.data_vars:
        name_str = str(name)
        if name_str.startswith("edar_biomarker_"):
            if not any(name_str.endswith(s) for s in ["_mask", "_age", "_censor"]):
                variants.append(name_str)
    return sorted(variants)


def _aggregate_biomarker_mean(
    dataset: xr.Dataset,
    variants: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked mean across variants, matching training pipeline.

    This matches the aggregation in epi_dataset.py:_precompute_wastewater_target()
    which uses masked mean respecting observation masks, not simple nanmean.

    Returns:
        Tuple of (values, valid_mask) where valid_mask indicates points
        with at least one observed variant.
    """
    if not variants:
        raise ValueError("No biomarker variants found for aggregation")

    component_tensors = []
    component_masks = []

    for v in variants:
        values = dataset[v].values.astype(float)
        mask = _get_observed_mask(dataset, v)
        component_tensors.append(values)
        component_masks.append(mask)

    stacked_values = np.stack(component_tensors, axis=0)  # (C, T, N)
    stacked_masks = np.stack(component_masks, axis=0)  # (C, T, N)

    # Match training pipeline: valid = mask & finite
    valid = stacked_masks & np.isfinite(stacked_values)
    valid_count = valid.sum(axis=0)

    # Masked mean: sum of valid / count of valid
    weighted_sum = np.where(valid, stacked_values, 0.0).sum(axis=0)
    combined_values = np.divide(
        weighted_sum,
        np.where(valid_count > 0, valid_count, 1.0),
    )

    # Union mask: any variant observed
    combined_mask = np.any(valid, axis=0)

    return combined_values, combined_mask


def _get_observed_mask(dataset: xr.Dataset, var_name: str) -> np.ndarray:
    """Get mask for observed values. Fallback to finite check if no mask."""
    mask_name = f"{var_name}_mask"
    if mask_name in dataset:
        return dataset[mask_name].values.astype(bool)
    logger.warning("No %s found, using np.isfinite()", mask_name)
    return np.isfinite(dataset[var_name].values)


def _load_canonical_data(
    dataset_path: Path,
    run_id: str = "real",
    series_subset: list[str] | None = None,
) -> CorrelationData:
    """Load observation series from canonical Zarr dataset.

    Args:
        dataset_path: Path to Zarr dataset
        run_id: Run ID to filter (default: "real")
        series_subset: Optional list of series to load (default: all 4)

    Returns:
        CorrelationData container with values, masks, region_ids, and dates
    """
    dataset = xr.open_zarr(dataset_path)

    # Filter by run_id if present
    if "run_id" in dataset.dims:
        dataset = dataset.sel(run_id=run_id)

    region_ids = dataset[REGION_COORD].values
    dates = dataset[TEMPORAL_COORD].values

    series_to_load = series_subset or list(SERIES_VARIABLES.keys())
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for series_name in series_to_load:
        if series_name not in SERIES_VARIABLES:
            logger.warning("Unknown series: %s, skipping", series_name)
            continue

        var_name = SERIES_VARIABLES[series_name]

        if series_name == "wastewater":
            # Special handling: aggregate biomarker variants with masked mean
            variants = _discover_biomarker_variants(dataset)
            if not variants:
                logger.warning("No biomarker variants found, skipping wastewater")
                continue
            logger.info(
                "Aggregating %d biomarker variants: %s", len(variants), variants
            )
            # Use masked mean matching training pipeline
            values[series_name], masks[series_name] = _aggregate_biomarker_mean(
                dataset, variants
            )
        else:
            if var_name not in dataset:
                logger.warning(
                    "Variable %s not found in dataset, skipping %s",
                    var_name,
                    series_name,
                )
                continue
            values[series_name] = dataset[var_name].values.astype(float)
            masks[series_name] = _get_observed_mask(dataset, var_name)

    if not values:
        raise ValueError("No valid series found in dataset")

    logger.info(
        "Loaded %d series: %s",
        len(values),
        list(values.keys()),
    )

    return CorrelationData(
        values=values,
        masks=masks,
        region_ids=region_ids,
        dates=dates,
    )


# =============================================================================
# PROCESSING
# =============================================================================


def _extract_observed_values_flattened(
    data: CorrelationData,
) -> dict[str, np.ndarray]:
    """Flatten 2D arrays to 1D, setting unobserved values to NaN.

    This preserves the index correspondence across series for pairwise correlation.

    Returns:
        Dict mapping series_name -> 1D array (flattened time*region with NaN for unobserved)
    """
    flattened: dict[str, np.ndarray] = {}

    for series_name, series_values in data.values.items():
        mask = data.masks.get(series_name, np.isfinite(series_values))
        # Flatten and set unobserved to NaN
        flat = series_values.flatten()
        flat_mask = mask.flatten() & np.isfinite(flat)
        flat[~flat_mask] = np.nan
        flattened[series_name] = flat

    return flattened


def _aggregate_per_region_means(
    data: CorrelationData,
) -> dict[str, np.ndarray]:
    """Compute per-region mean (over time) for observed values only.

    Returns:
        Dict mapping series_name -> 1D array of per-region means
    """
    region_means: dict[str, np.ndarray] = {}
    n_regions = len(data.region_ids)

    for series_name, series_values in data.values.items():
        mask = data.masks.get(series_name, np.isfinite(series_values))
        region_mean_list = []

        for r in range(n_regions):
            region_data = series_values[:, r]
            region_mask = mask[:, r] & np.isfinite(region_data)
            if region_mask.any():
                region_mean_list.append(np.mean(region_data[region_mask]))
            else:
                region_mean_list.append(np.nan)

        region_means[series_name] = np.array(region_mean_list)

    return region_means


# =============================================================================
# CORRELATION COMPUTATION
# =============================================================================


def compute_pairwise_correlations(
    values: dict[str, np.ndarray],
) -> dict[tuple[str, str], CorrelationResult]:
    """Compute pairwise correlations between all series.

    Args:
        values: Dict mapping series_name -> 1D array of values

    Returns:
        Dict mapping (series_a, series_b) -> CorrelationResult
    """
    series_names = list(values.keys())
    results: dict[tuple[str, str], CorrelationResult] = {}

    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i >= j:
                continue  # Only compute upper triangle

            arr_a = values[name_a]
            arr_b = values[name_b]

            # Find common valid indices
            valid = np.isfinite(arr_a) & np.isfinite(arr_b)
            n_points = valid.sum()

            if n_points < 3:
                logger.warning(
                    "Insufficient overlap (%d points) for %s vs %s",
                    n_points,
                    name_a,
                    name_b,
                )
                continue

            a_valid = arr_a[valid]
            b_valid = arr_b[valid]

            pearson_r, pearson_p = stats.pearsonr(a_valid, b_valid)
            spearman_r, spearman_p = stats.spearmanr(a_valid, b_valid)

            results[(name_a, name_b)] = CorrelationResult(
                series_a=name_a,
                series_b=name_b,
                pearson_r=float(pearson_r),
                pearson_p=float(pearson_p),
                spearman_r=float(spearman_r),
                spearman_p=float(spearman_p),
                n_points=n_points,
            )

    return results


# =============================================================================
# PLOTTING
# =============================================================================


def _add_regression_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Add regression line with confidence band to scatter plot."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return

    x_valid = x[valid]
    y_valid = y[valid]

    # Fit line (linregress returns a result object in modern scipy)
    result = stats.linregress(x_valid, y_valid)
    slope = result.slope
    intercept = result.intercept

    # Plot line
    x_line = np.array([x_valid.min(), x_valid.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="red", linewidth=1.5, alpha=0.8)


def plot_pairwise_scatter_grid(
    values: dict[str, np.ndarray],
    output_path: Path,
    alpha: float = 0.3,
    title_suffix: str = "",
) -> None:
    """Create scatter grid with histograms on diagonal and correlation in upper triangle.

    Layout:
    - Diagonal: Histograms
    - Lower triangle (i > j): Scatter plots with regression lines
    - Upper triangle (i < j): Correlation coefficient only (no scatter)

    Args:
        values: Dict mapping series_name -> 1D array of values
        output_path: Path to save figure
        alpha: Scatter point transparency
        title_suffix: Suffix for figure title
    """
    series_names = list(values.keys())
    n = len(series_names)

    if n == 0:
        logger.warning("No series to plot")
        return

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    # Handle single series case
    if n == 1:
        axes = np.array([[axes]])

    for i, name_i in enumerate(series_names):
        for j, name_j in enumerate(series_names):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                data = values[name_i]
                valid_data = data[np.isfinite(data)]
                if len(valid_data) > 0:
                    ax.hist(
                        valid_data,
                        bins=30,
                        alpha=0.7,
                        color="steelblue",
                        edgecolor="white",
                    )
                ax.set_xlabel(SERIES_LABELS.get(name_i, name_i))
                ax.set_ylabel("Count")
            elif i > j:
                # Lower triangle: scatter plot with regression line
                x = values[name_j]  # x-axis is column index
                y = values[name_i]  # y-axis is row index

                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() > 0:
                    ax.scatter(
                        x[valid],
                        y[valid],
                        alpha=alpha,
                        s=10,
                        color="steelblue",
                        edgecolors="none",
                    )
                    _add_regression_line(ax, x[valid], y[valid])

                    # Add correlation annotation
                    r, _ = stats.pearsonr(x[valid], y[valid])
                    ax.text(
                        0.05,
                        0.95,
                        f"r={r:.2f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                ax.set_xlabel(SERIES_LABELS.get(name_j, name_j))
                ax.set_ylabel(SERIES_LABELS.get(name_i, name_i))
            else:
                # Upper triangle (i < j): correlation coefficient only
                x = values[name_j]
                y = values[name_i]

                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 3:
                    r, p = stats.pearsonr(x[valid], y[valid])
                    # Display correlation centered in the cell
                    ax.text(
                        0.5,
                        0.5,
                        f"r = {r:.2f}\np = {p:.2e}",
                        transform=ax.transAxes,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=12,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                    )

                # Hide ticks for upper triangle cells
                ax.set_xticks([])
                ax.set_yticks([])

            ax.grid(True, alpha=0.3)

    title = f"Cross-Series Correlation{title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_path, dpi=150, log_msg="Saved scatter grid")


def plot_correlation_heatmap(
    correlations: dict[tuple[str, str], CorrelationResult],
    series_names: list[str],
    output_path: Path,
) -> None:
    """Create correlation heatmap with Pearson (lower) and Spearman (upper).

    Args:
        correlations: Dict of correlation results
        series_names: Ordered list of series names
        output_path: Path to save figure
    """
    n = len(series_names)

    # Build matrices
    pearson_matrix = np.ones((n, n))
    spearman_matrix = np.ones((n, n))

    for (a, b), result in correlations.items():
        i = series_names.index(a)
        j = series_names.index(b)
        # Ensure i < j for matrix indexing
        if i > j:
            i, j = j, i
        pearson_matrix[i, j] = result.pearson_r
        pearson_matrix[j, i] = result.pearson_r
        spearman_matrix[i, j] = result.spearman_r
        spearman_matrix[j, i] = result.spearman_r

    # Combine: lower triangle = Pearson, upper triangle = Spearman
    combined = np.tril(pearson_matrix) + np.triu(spearman_matrix, k=1)

    # Map series names to labeled versions for display
    labeled_names = [SERIES_LABELS.get(name, name) for name in series_names]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom annotation
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = "1.00"
            elif i < j:
                annot[i, j] = f"{spearman_matrix[i, j]:.2f}"
            else:
                annot[i, j] = f"{pearson_matrix[i, j]:.2f}"

    sns.heatmap(
        combined,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        xticklabels=labeled_names,
        yticklabels=labeled_names,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title("Cross-Series Correlation Heatmap\n(Lower: Pearson, Upper: Spearman)")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    save_figure(fig, output_path, dpi=150, log_msg="Saved correlation heatmap")


# =============================================================================
# OUTPUT
# =============================================================================


def export_correlation_csv(
    correlations: dict[tuple[str, str], CorrelationResult],
    output_path: Path,
) -> None:
    """Export correlation results to CSV."""
    rows = []
    for (a, b), result in correlations.items():
        rows.append(
            {
                "series_a": a,
                "series_b": b,
                "pearson_r": result.pearson_r,
                "pearson_p": result.pearson_p,
                "spearman_r": result.spearman_r,
                "spearman_p": result.spearman_p,
                "n_points": result.n_points,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["series_a", "series_b"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved correlations to %s", output_path)


def print_summary(
    correlations: dict[tuple[str, str], CorrelationResult],
    aggregation: str,
) -> None:
    """Print correlation summary to console."""
    print(f"\n{'=' * 80}")
    print(f"CROSS-SERIES CORRELATIONS ({aggregation})")
    print("=" * 80)

    for (a, b), result in correlations.items():
        print(f"\n{a} vs {b}:")
        print(f"  Pearson r:  {result.pearson_r:+.3f} (p={result.pearson_p:.2e})")
        print(f"  Spearman r: {result.spearman_r:+.3f} (p={result.spearman_p:.2e})")
        print(f"  N points:   {result.n_points}")

    print("\n" + "=" * 80 + "\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to Zarr dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/cross_series_correlation"),
        help="Output directory",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="real",
        help="Run ID filter",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["per_region_mean", "all_pooled", "both"],
        default="both",
        help="Aggregation strategy",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Scatter point transparency",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Comma-separated subset of series (default: all 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse series subset
    series_subset = None
    if args.series:
        series_subset = [s.strip() for s in args.series.split(",") if s.strip()]

    # Load data
    data = _load_canonical_data(
        args.dataset,
        run_id=args.run_id,
        series_subset=series_subset,
    )

    series_names = list(data.values.keys())
    logger.info("Analyzing series: %s", series_names)

    aggregation_modes = []
    if args.aggregation in ("per_region_mean", "both"):
        aggregation_modes.append("per_region_mean")
    if args.aggregation in ("all_pooled", "both"):
        aggregation_modes.append("all_pooled")

    for agg_mode in aggregation_modes:
        if agg_mode == "per_region_mean":
            values = _aggregate_per_region_means(data)
            suffix = " (per-region means)"
            file_suffix = "per_region"
        else:
            values = _extract_observed_values_flattened(data)
            suffix = " (all pooled)"
            file_suffix = "pooled"

        # Compute correlations
        correlations = compute_pairwise_correlations(values)

        # Print summary
        print_summary(correlations, agg_mode)

        # Generate scatter grid
        plot_pairwise_scatter_grid(
            values,
            args.output_dir / f"scatter_grid_{file_suffix}.png",
            alpha=args.alpha,
            title_suffix=suffix,
        )

        # Only generate heatmap and CSV once (use first aggregation mode)
        if agg_mode == aggregation_modes[0]:
            # Generate heatmap
            plot_correlation_heatmap(
                correlations,
                series_names,
                args.output_dir / "correlation_heatmap.png",
            )

            # Export CSV
            export_correlation_csv(
                correlations,
                args.output_dir / "correlations.csv",
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
