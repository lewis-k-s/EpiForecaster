"""
Sparsity analysis with population scatter plots.

Creates per-variable sparsity vs population scatter plots using explicit
mask variables from the canonical dataset.

Uses explicit mask variables to distinguish truly observed values from
interpolated/imputed values.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from utils.plotting import Style  # noqa: E402

logger = logging.getLogger(__name__)

# Wastewater biomarker variants
WW_BIOMARKERS = ["edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4"]

# Variables with weekly reporting (use 7-day window aggregation)
WEEKLY_VARIABLES = {"hospitalizations", "wastewater"}


def get_sparsity_mask(dataset: xr.Dataset, var_name: str) -> np.ndarray:
    """Extract sparsity mask from dataset for a given variable.

    Uses explicit {var_name}_mask if available, where mask=1 means observed
    and mask=0 means sparse/interpolated. Returns a boolean array where
    True = sparse/interpolated.

    Args:
        dataset: xarray Dataset with mask variables
        var_name: Variable name (e.g., "cases", "hospitalizations")

    Returns:
        Boolean array (date, region_id) where True = sparse
    """
    mask_name = f"{var_name}_mask"

    if mask_name not in dataset:
        raise ValueError(f"Mask variable {mask_name} not found in dataset")

    mask = dataset[mask_name]

    # Handle run_id dimension if present (select run_id=0)
    if "run_id" in mask.dims:
        mask = mask.isel(run_id=0)

    # mask=1 means observed, mask=0 means sparse
    # Return True where sparse (mask == 0)
    return (mask.values == 0).astype(bool)


def get_wastewater_sparsity_mask(dataset: xr.Dataset) -> np.ndarray:
    """Compute combined sparsity mask from all wastewater biomarker variants.

    A point is considered sparse only if ALL biomarker variants are sparse.
    This is because any single biomarker observation provides value.

    Args:
        dataset: xarray Dataset with biomarker mask variables

    Returns:
        Boolean array (date, region_id) where True = sparse (all variants missing)
    """
    masks = []
    available_biomarkers = []

    for biomarker in WW_BIOMARKERS:
        mask_name = f"{biomarker}_mask"
        if mask_name in dataset:
            mask = dataset[mask_name]
            if "run_id" in mask.dims:
                mask = mask.isel(run_id=0)
            # mask=1 means observed, so sparse where mask==0
            masks.append(mask.values == 0)
            available_biomarkers.append(biomarker)

    if not masks:
        raise ValueError("No wastewater biomarker masks found in dataset")

    logger.info(
        "Using wastewater biomarkers: %s",
        ", ".join(b.replace("edar_biomarker_", "") for b in available_biomarkers),
    )

    # Stack masks and compute combined sparsity
    # A point is sparse only if ALL biomarkers are sparse
    stacked = np.stack(masks, axis=0)  # (num_biomarkers, date, region_id)
    combined_sparse = np.all(stacked, axis=0)  # (date, region_id)

    return combined_sparse.astype(bool)


def compute_weekly_sparsity(mask: np.ndarray) -> np.ndarray:
    """Compute sparsity using 7-day windows for weekly-reported variables.

    For weekly data, a window is considered sparse only if there are ZERO
    observations in that 7-day period. This avoids penalizing regions that
    report consistently on a weekly schedule.

    Args:
        mask: Boolean array (date, region_id) where True = observed (mask==1)

    Returns:
        Array of sparsity percentages per region (N,)
    """
    # mask is True where observed, False where sparse
    T, N = mask.shape
    window_size = 7
    num_windows = T // window_size

    if num_windows == 0:
        # Not enough data for even one window, fall back to daily
        logger.warning(
            "Dataset too short for weekly windowing (%d days), using daily", T
        )
        return (~mask).mean(axis=0) * 100

    # Reshape to (num_windows, window_size, N)
    # Truncate to full windows
    truncated_mask = mask[: num_windows * window_size]
    windowed = truncated_mask.reshape(num_windows, window_size, N)

    # A window has data if ANY day in that window has an observation
    window_has_data = windowed.any(axis=1)  # (num_windows, N)

    # Sparsity = % of windows with NO data
    sparsity_pct = (~window_has_data).mean(axis=0) * 100  # (N,)

    return sparsity_pct


def compute_daily_sparsity(mask: np.ndarray) -> np.ndarray:
    """Compute sparsity using daily granularity.

    Args:
        mask: Boolean array (date, region_id) where True = observed (mask==1)

    Returns:
        Array of sparsity percentages per region (N,)
    """
    # mask is True where observed, False where sparse
    return (~mask).mean(axis=0) * 100


def plot_sparsity_vs_population(
    sparsity_pct: np.ndarray,
    population: np.ndarray,
    output_dir: Path,
    var_name: str,
    frequency: str = "daily",
) -> None:
    """Plot relationship between sparsity and population for a variable.

    Args:
        sparsity_pct: Array of sparsity percentages per region (N,)
        population: Array of population values per region (N,)
        output_dir: Directory to save plot
        var_name: Variable name for filename and title
        frequency: Reporting frequency note for title (e.g., "daily", "weekly")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(population, sparsity_pct, alpha=0.5, s=30)

    ax.set_xlabel("Population")
    ax.set_ylabel("Sparsity (%)")
    ax.set_ylim(0, 100)
    # Capitalize first letter for title, include frequency note
    title_var = var_name.capitalize()
    ax.set_title(f"{title_var} Sparsity vs Population ({frequency})")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{var_name}_sparsity_vs_population.png"
    fig.savefig(output_path, dpi=Style.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s sparsity vs population plot to %s", var_name, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparsity analysis with population scatter plots"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to canonical Zarr dataset",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["cases", "hospitalizations", "deaths", "wastewater"],
        help="Variables to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/sparsity_analysis"),
        help="Directory to save plots",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = xr.open_zarr(args.dataset_path)
    logger.info("Loaded dataset: %s", args.dataset_path)

    # Get population
    population = dataset["population"].values

    logger.info("Regions: %d", len(population))

    for var_name in args.variables:
        logger.info("Processing variable: %s", var_name)

        if var_name == "wastewater":
            # Get combined sparse mask (True = sparse)
            sparse_mask = get_wastewater_sparsity_mask(dataset)
            # Convert to observed mask (True = observed)
            obs_mask = ~sparse_mask
        else:
            # Get sparse mask (True = sparse)
            sparse_mask = get_sparsity_mask(dataset, var_name)
            # Convert to observed mask (True = observed)
            obs_mask = ~sparse_mask

        # Use weekly windowing for weekly-reported variables
        if var_name in WEEKLY_VARIABLES:
            logger.info("  Using 7-day window aggregation (weekly reporting)")
            sparsity_pct = compute_weekly_sparsity(obs_mask)
            frequency = "weekly windows"
        else:
            sparsity_pct = compute_daily_sparsity(obs_mask)
            frequency = "daily"

        plot_sparsity_vs_population(
            sparsity_pct, population, output_dir, var_name, frequency
        )

    logger.info("Analysis complete. Output saved to: %s", output_dir)


if __name__ == "__main__":
    main()
