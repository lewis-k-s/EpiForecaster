"""Generate age/mask sparsity visualizations for all data sources.

This script loads a training config and generates age/mask sparsity
visualizations for all epidemiological data sources:
- Cases
- Hospitalizations
- Deaths
- Wastewater (if available)

Outputs:
    cases_age_sparsity.png: Cases age/mask heatmap
    hospitalizations_age_sparsity.png: Hospitalizations age/mask heatmap
    deaths_age_sparsity.png: Deaths age/mask heatmap
    wastewater_age_sparsity.png: Wastewater age/mask heatmap (if available)
    *_summary.csv: Summary statistics tables
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.figure import Figure

from data.preprocess.config import REGION_COORD
from models.configs import EpiForecasterConfig
from utils.logging import setup_logging, suppress_zarr_warnings

suppress_zarr_warnings()
logger = logging.getLogger(__name__)


def load_dataset_for_viz(config: EpiForecasterConfig) -> xr.Dataset | None:
    """Load the dataset from config for visualization."""
    try:
        zarr_path = Path(config.data.dataset_path).resolve()
        if not zarr_path.exists():
            logger.error(f"Dataset not found: {zarr_path}")
            return None
        return xr.open_zarr(zarr_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def prepare_2d_dataarray(da: xr.DataArray, run_id: int | None = 0) -> xr.DataArray:
    """Prepare a DataArray for visualization by handling run_id dimension.

    If the DataArray has a run_id dimension, select the specified run_id.
    Returns a 2D DataArray with dimensions (date, region_id).
    """
    if "run_id" in da.dims:
        if run_id is None:
            run_id = 0
        da = da.isel(run_id=run_id)
    return da


def compute_sparsity_stats(
    da: xr.DataArray,
    da_mask: xr.DataArray | None = None,
    include_all_regions: bool = False,
) -> pd.DataFrame:
    """Compute sparsity statistics for each region.

    Args:
        da: 2D DataArray with dimensions (date, region_id)
        da_mask: Optional 2D binary observation mask (1=observed, 0=missing)
        include_all_regions: If True, include regions with zero observations

    Returns:
        DataFrame with sparsity statistics per region
    """
    # Ensure 2D
    if "run_id" in da.dims:
        da = prepare_2d_dataarray(da)

    values = da.values
    mask_values = None
    if da_mask is not None:
        if "run_id" in da_mask.dims:
            da_mask = prepare_2d_dataarray(da_mask)
        mask_values = (da_mask.values > 0).astype(bool)

    n_time, n_regions = values.shape

    rows = []
    region_ids = da[REGION_COORD].values

    for i, region_id in enumerate(region_ids):
        series = values[:, i]

        if mask_values is not None:
            observed = mask_values[:, i]
        else:
            observed = np.isfinite(series)

        # Skip regions with no observations unless explicitly requested.
        if (not include_all_regions) and (not np.any(observed)):
            continue

        n_measurements = int(np.sum(observed))
        sparsity_pct = 100 * (1 - n_measurements / n_time)

        # Compute max consecutive missing
        missing_mask = ~observed
        max_consecutive = 0
        current = 0
        for is_missing in missing_mask:
            if is_missing:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        rows.append(
            {
                "region_id": region_id,
                "n_measurements": n_measurements,
                "total_timesteps": n_time,
                "sparsity_pct": round(sparsity_pct, 2),
                "max_consecutive_missing": max_consecutive,
            }
        )

    return pd.DataFrame(rows)


def compute_age_from_mask(mask_2d: np.ndarray, age_max: int = 14) -> np.ndarray:
    """Compute normalized age [0,1] from binary observation mask (time, region)."""
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {mask_2d.shape}")

    T, N = mask_2d.shape
    observed = mask_2d > 0

    time_idx = np.arange(T, dtype=np.float32)[:, None]
    last_seen = np.where(observed, time_idx, np.nan)
    last_seen_filled = pd.DataFrame(last_seen).ffill().values

    has_history = ~np.isnan(last_seen_filled)
    current_age = np.zeros((T, N), dtype=np.float32)
    current_age[has_history] = (
        time_idx * np.ones((1, N), dtype=np.float32) - last_seen_filled
    )[has_history]
    final_age = np.where(has_history, np.minimum(current_age, age_max), age_max)
    return (final_age / age_max).astype(np.float32)


def plot_age_heatmap(
    ax: Any,
    da_values: np.ndarray,
    da_age: np.ndarray,
    region_ids: np.ndarray,
    observed_mask: np.ndarray | None = None,
    history_length: int | None = None,
    age_max: int = 14,
    title: str = "Data",
) -> None:
    """Plot age/staleness heatmap.

    Args:
        ax: Matplotlib axes
        da_values: 2D array (time, region) of values
        da_age: 2D array (time, region) of age in days
        region_ids: Array of region IDs
        history_length: Optional separator line position
        age_max: Maximum age for normalization
        title: Plot title
    """
    # Filter to regions with any data
    if observed_mask is not None:
        has_data = observed_mask.any(axis=0)
    else:
        has_data = np.isfinite(da_values).any(axis=0)
    age_filtered = da_age[:, has_data].T

    # Normalize age to [0, 1]. Some channels are already normalized (e.g. biomarkers).
    finite_age = age_filtered[np.isfinite(age_filtered)]
    if finite_age.size > 0 and float(np.nanmax(finite_age)) <= 1.0:
        age_normalized = np.clip(age_filtered, 0, 1)
    else:
        age_normalized = np.clip(age_filtered / age_max, 0, 1)

    # Plot heatmap
    sns.heatmap(
        age_normalized,
        cbar_kws={"label": f"Age (days, max={age_max})"},
        cmap="Reds",
        vmin=0,
        vmax=1,
        xticklabels=30,
        yticklabels=False,
        ax=ax,
    )

    # Add history/horizon separator
    if history_length is not None:
        ax.axvline(history_length, color="blue", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_title(f"{title} - Data Age/Staleness", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Region")


def make_age_sparsity_figure(
    da_values: xr.DataArray,
    da_age: xr.DataArray,
    da_mask: xr.DataArray | None = None,
    history_length: int | None = None,
    age_max: int = 14,
    title: str = "Data",
) -> Figure | None:
    """Create multi-panel age/sparsity figure.

    Args:
        da_values: DataArray with values (2D: date, region)
        da_age: DataArray with age in days (2D: date, region)
        history_length: Optional separator position
        age_max: Maximum age for normalization
        title: Figure title

    Returns:
        Matplotlib Figure or None if no data
    """
    # Prepare data
    da_values = prepare_2d_dataarray(da_values)
    da_age = prepare_2d_dataarray(da_age)
    if da_mask is not None:
        da_mask = prepare_2d_dataarray(da_mask)

    values = da_values.values
    age_values = da_age.values
    observed_mask = None if da_mask is None else (da_mask.values > 0)

    # Check if there's any data
    if not np.isfinite(values).any():
        return None

    # Compute sparsity stats
    stats_df = compute_sparsity_stats(da_values, da_mask=da_mask)

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Age heatmap
    ax1 = fig.add_subplot(gs[0, :])
    plot_age_heatmap(
        ax1,
        da_values=values,
        da_age=age_values,
        region_ids=da_values[REGION_COORD].values,
        observed_mask=observed_mask,
        history_length=history_length,
        age_max=age_max,
        title=title,
    )

    # Panel 2: Sparsity distribution
    ax2 = fig.add_subplot(gs[1, 0])
    if not stats_df.empty:
        ax2.hist(
            stats_df["sparsity_pct"],
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.set_xlabel("Sparsity (%)")
        ax2.set_ylabel("Number of regions")
        ax2.set_title(
            f"{title} Sparsity Distribution", fontsize=10, fontweight="semibold"
        )
        ax2.axvline(
            stats_df["sparsity_pct"].median(),
            color="red",
            linestyle="--",
            label=f"Median: {stats_df['sparsity_pct'].median():.1f}%",
        )
        ax2.legend(fontsize=8)
    else:
        ax2.text(
            0.5,
            0.5,
            "No sparsity data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # Panel 3: Age distribution
    ax3 = fig.add_subplot(gs[1, 1])
    age_finite = age_values[np.isfinite(age_values)]
    if len(age_finite) > 0:
        ax3.hist(age_finite, bins=20, color="coral", edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Age (days)")
        ax3.set_ylabel("Count")
        ax3.set_title(f"{title} Age Distribution", fontsize=10, fontweight="semibold")
    else:
        ax3.text(
            0.5, 0.5, "No age data", ha="center", va="center", transform=ax3.transAxes
        )

    return fig


def generate_age_mask_plots(
    config_path: str,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate age/mask sparsity visualizations for all data sources."""
    setup_logging(logging.INFO)

    logger.info(f"Loading config from: {config_path}")
    config = EpiForecasterConfig.from_file(config_path)

    ds = load_dataset_for_viz(config)
    if ds is None:
        logger.error("Failed to load dataset, aborting")
        return {}

    if output_dir is None:
        output_dir = Path("outputs/reports/age_mask")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}
    data_vars = list(ds.data_vars)
    logger.info(
        f"Available variables: {[v for v in data_vars if not v.startswith('edar_')]}"
    )

    # Primary data sources with their age variables
    sources = [
        ("cases", "cases_age", "Cases"),
        ("hospitalizations", "hospitalizations_age", "Hospitalizations"),
        ("deaths", "deaths_age", "Deaths"),
    ]

    for value_var, age_var, title in sources:
        if value_var not in data_vars:
            logger.warning(f"Variable '{value_var}' not found in dataset, skipping")
            continue

        if age_var not in data_vars:
            logger.warning(f"Age variable '{age_var}' not found, skipping {title}")
            continue

        logger.info(f"Generating {title} age/mask visualization...")

        try:
            da_values = ds[value_var]
            da_age = ds[age_var]
            da_mask = ds.get(f"{value_var}_mask")

            # Compute sparsity stats
            stats_df = compute_sparsity_stats(da_values, da_mask=da_mask)

            # Generate figure
            fig = make_age_sparsity_figure(
                da_values=da_values,
                da_age=da_age,
                da_mask=da_mask,
                history_length=config.model.history_length,
                title=title,
            )

            # Save figure
            if fig is not None:
                figure_path = output_dir / f"{value_var}_age_sparsity.png"
                fig.savefig(figure_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                exported_files[f"{value_var}_figure"] = figure_path
                logger.info(f"Saved {title} figure to {figure_path}")
            else:
                logger.warning(f"No figure generated for {title} (no data)")

            # Export summary table
            summary_path = output_dir / f"{value_var}_summary.csv"
            if not stats_df.empty:
                stats_df.to_csv(summary_path, index=False)
                exported_files[f"{value_var}_summary"] = summary_path
                logger.info(f"Saved {title} summary to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to generate {title} visualization: {e}")
            continue

    # Biomarkers - combine all variants with OR logic
    biomarker_variants = [
        "edar_biomarker_N1",
        "edar_biomarker_N2",
        "edar_biomarker_IP4",
    ]
    available_variants = [v for v in biomarker_variants if v in data_vars]

    if available_variants:
        logger.info(
            f"Generating Biomarkers age/mask visualization (variants: {available_variants})..."
        )

        try:
            source_mask = None
            if "edar_has_source" in ds:
                source_mask = ds["edar_has_source"].values.astype(bool)

            # Combine all variants with OR logic for sparsity
            # A timestep has data if ANY variant has data
            combined_values = None

            for variant in available_variants:
                da_variant = ds[variant]
                da_variant_mask = ds.get(f"{variant}_mask")

                # Prepare 2D data
                da_variant = prepare_2d_dataarray(da_variant)
                if da_variant_mask is not None:
                    da_variant_mask = prepare_2d_dataarray(da_variant_mask)

                if da_variant_mask is not None:
                    variant_observed = (da_variant_mask.values > 0).astype(float)
                else:
                    variant_observed = np.isfinite(da_variant.values).astype(float)

                if combined_values is None:
                    combined_values = variant_observed.copy()
                else:
                    # OR logic: has data if any variant has data.
                    combined_values = np.maximum(combined_values, variant_observed)
            assert combined_values is not None
            # Combined age should be derived from the combined mask over time.
            combined_age = compute_age_from_mask(combined_values, age_max=14)

            # Create DataArrays for the combined data
            template = prepare_2d_dataarray(ds[available_variants[0]])
            if source_mask is not None:
                template = template.isel({REGION_COORD: source_mask})
                combined_values = combined_values[:, source_mask]
                combined_age = combined_age[:, source_mask]

            da_combined_values = xr.DataArray(
                combined_values,
                dims=template.dims,
                coords=template.coords,
                name="biomarkers_combined",
            )
            da_combined_age = xr.DataArray(
                combined_age,
                dims=template.dims,
                coords=template.coords,
                name="biomarkers_age_combined",
            )

            # Compute sparsity (treating 1.0 as has data, 0.0 as missing)
            stats_df = compute_sparsity_stats(
                da_combined_values,
                da_mask=da_combined_values,
                include_all_regions=True,
            )

            # Generate figure
            fig = make_age_sparsity_figure(
                da_values=da_combined_values,
                da_age=da_combined_age,
                da_mask=da_combined_values,
                history_length=config.model.history_length,
                title="Biomarkers (combined)",
            )

            if fig is not None:
                figure_path = output_dir / "biomarkers_age_sparsity.png"
                fig.savefig(figure_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                exported_files["biomarkers_figure"] = figure_path
                logger.info(f"Saved Biomarkers figure to {figure_path}")

            summary_path = output_dir / "biomarkers_summary.csv"
            if not stats_df.empty:
                stats_df.to_csv(summary_path, index=False)
                exported_files["biomarkers_summary"] = summary_path
                logger.info(f"Saved Biomarkers summary to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to generate Biomarkers visualization: {e}")

    # Check for wastewater data
    ww_vars = [v for v in data_vars if "wastewater" in v.lower() or "ww" in v.lower()]
    for ww_var in ww_vars:
        logger.info(f"Found wastewater variable: {ww_var}")
        # Could add wastewater handling here if needed

    ds.close()
    logger.info(f"Generated {len(exported_files)} output files")
    return exported_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate age/mask sparsity visualizations for all data sources"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports/age_mask",
        help="Output directory for plots (default: outputs/reports/age_mask)",
    )

    args = parser.parse_args()

    generate_age_mask_plots(
        config_path=args.config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
