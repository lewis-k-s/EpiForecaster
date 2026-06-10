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

BIOMARKER_VARIANTS = ("N1", "N2", "IP4")
BIOMARKER_SUFFIXES = ("_mask", "_age", "_censor")


def _is_biomarker_value_var(name: str) -> bool:
    return name.startswith("edar_biomarker_") and not name.endswith(BIOMARKER_SUFFIXES)


def _load_biomarkers(
    dataset_path: Path,
) -> tuple[dict[str, xr.DataArray], np.ndarray | None]:
    dataset = xr.open_zarr(dataset_path)
    variant_vars: dict[str, xr.DataArray] = {}
    for name in dataset.data_vars:
        name_str = str(name)
        if _is_biomarker_value_var(name_str):
            variant_vars[name_str] = dataset[name]
    if not variant_vars:
        if "edar_biomarker" not in dataset:
            raise ValueError("Dataset missing EDAR biomarker variables")
        variant_vars = {"edar_biomarker": dataset["edar_biomarker"]}

    data_start = dataset.get("biomarker_data_start")
    if data_start is not None and "run_id" in data_start.dims:
        data_start = data_start.isel(run_id=0)

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
        # Handle run_id dimension if present (multi-run datasets)
        if "run_id" in biomarker.dims:
            biomarker = biomarker.isel(run_id=0)  # Select first run
        variant_vars[name] = biomarker.transpose(TEMPORAL_COORD, REGION_COORD)

    data_start_values = data_start.values if data_start is not None else None

    logger.info("Loaded %d biomarker variants", len(variant_vars))
    return variant_vars, data_start_values


def _ordered_variant_names(dataset: xr.Dataset | None = None) -> list[str]:
    if dataset is None:
        return [f"edar_biomarker_{variant}" for variant in BIOMARKER_VARIANTS]
    names = {
        str(name) for name in dataset.data_vars if _is_biomarker_value_var(str(name))
    }
    ordered = [f"edar_biomarker_{variant}" for variant in BIOMARKER_VARIANTS]
    ordered = [name for name in ordered if name in names]
    ordered.extend(sorted(names.difference(ordered)))
    return ordered


def _load_mask(dataset: xr.Dataset, variant_name: str) -> xr.DataArray | None:
    mask_name = f"{variant_name}_mask"
    if mask_name not in dataset:
        return None
    mask = dataset[mask_name]
    if "run_id" in mask.dims:
        mask = mask.isel(run_id=0)
    return mask.transpose(TEMPORAL_COORD, REGION_COORD)


def _mask_pre_data_start(
    values: np.ndarray, data_start: np.ndarray | None
) -> np.ndarray:
    masked = values.astype(float, copy=True)
    if data_start is None:
        return masked
    starts = np.asarray(data_start).reshape(-1)
    for idx, start in enumerate(starts):
        if start < 0:
            masked[:, idx] = np.nan
        else:
            masked[: int(start), idx] = np.nan
    return masked


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
        ax.plot(
            dates,
            global_mean,
            color=Colors.GLOBAL_MEAN,
            linewidth=2,
            label="Global mean",
        )
        ax.plot(
            dates,
            global_median,
            color=Colors.GLOBAL_MEDIAN,
            linewidth=2,
            label="Global median",
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
        ax.plot(
            zoom_dates, zoom_mean, color=Colors.GLOBAL_MEAN, linewidth=2.5, label="Mean"
        )
        ax.plot(
            zoom_dates,
            zoom_median,
            color=Colors.GLOBAL_MEDIAN,
            linewidth=2.5,
            label="Median",
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


def plot_manuscript_support_summary(dataset_path: Path, output_path: Path) -> None:
    dataset = xr.open_zarr(dataset_path)
    source = dataset["edar_has_source"].values.astype(bool)
    total_regions = int(dataset.sizes[REGION_COORD])
    source_regions = int(source.sum())
    no_source_regions = total_regions - source_regions
    no_source_pct = no_source_regions / total_regions * 100

    variant_support: dict[str, int] = {}
    for variant_name in _ordered_variant_names(dataset):
        label = variant_name.replace("edar_biomarker_", "")
        mask = _load_mask(dataset, variant_name)
        if mask is None:
            continue
        region_support = mask.values.astype(bool).any(axis=0)
        variant_support[label] = int((region_support & source).sum())

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.6),
        height_ratios=(1.0, 1.45),
        constrained_layout=True,
    )

    support_color = "#2a9d8f"
    no_source_color = "#6c757d"
    variant_colors = {
        "N1": "#1f77b4",
        "N2": "#e76f51",
        "IP4": "#7a5195",
    }

    ax_top.barh(
        ["Municipalities"],
        [no_source_regions],
        color=no_source_color,
        label="No biomarker source support",
    )
    ax_top.barh(
        ["Municipalities"],
        [source_regions],
        left=[no_source_regions],
        color=support_color,
        label="EDAR-supported",
    )
    ax_top.set_xlim(0, total_regions)
    ax_top.set_xlabel("Municipalities in canonical panel")
    ax_top.set_title("Wastewater Source Support")
    ax_top.text(
        no_source_regions / 2,
        0,
        f"No source support\n{no_source_regions} ({no_source_pct:.1f}%)",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )
    ax_top.text(
        no_source_regions + source_regions / 2,
        0,
        f"EDAR-supported\n{source_regions}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )

    labels = list(variant_support)
    counts = [variant_support[label] for label in labels]
    colors = [variant_colors.get(label, "#4c78a8") for label in labels]
    ax_bottom.bar(labels, counts, color=colors)
    ax_bottom.axhline(source_regions, color="#333333", linewidth=1.2, linestyle="--")
    ax_bottom.text(
        len(labels) - 0.15,
        source_regions + 4,
        f"{source_regions} EDAR-supported municipalities",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    for x_pos, count in enumerate(counts):
        ax_bottom.text(
            x_pos,
            count + 4,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_bottom.set_ylim(0, max(source_regions * 1.18, max(counts, default=0) + 25))
    ax_bottom.set_ylabel("Municipalities with direct observations")
    ax_bottom.set_xlabel("Biomarker target")
    ax_bottom.set_title("Direct Observation Support Within EDAR-Supported Regions")
    ax_bottom.grid(True, axis="y", alpha=0.25)
    ax_bottom.set_axisbelow(True)

    save_figure(
        fig,
        output_path,
        dpi=Style.DPI,
        log_msg="Saved manuscript support summary",
    )


def plot_manuscript_variant_panel(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    mask: np.ndarray | None,
    region_ids: np.ndarray,
    output_path: Path,
    variant_label: str,
    *,
    max_heatmap_regions: int,
) -> None:
    if mask is None:
        direct_counts = np.isfinite(values).sum(axis=0)
    else:
        direct_counts = mask.astype(bool).sum(axis=0)
    supported_indices = np.flatnonzero(direct_counts > 0)
    if supported_indices.size == 0:
        raise ValueError(f"No directly supported regions for {variant_label}")
    ordered_indices = supported_indices[
        np.argsort(direct_counts[supported_indices])[::-1]
    ]
    heatmap_indices = ordered_indices[:max_heatmap_regions].tolist()

    values_positive = np.where(values > 0, values, np.nan)
    supported_values = values_positive[:, supported_indices]
    valid_times = np.isfinite(supported_values).any(axis=1)
    global_mean = np.full(len(dates), np.nan, dtype=float)
    global_median = np.full(len(dates), np.nan, dtype=float)
    with np.errstate(all="ignore"):
        global_mean[valid_times] = np.nanmean(supported_values[valid_times], axis=1)
        global_median[valid_times] = np.nanmedian(supported_values[valid_times], axis=1)

    fig, (ax_series, ax_heatmap) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        height_ratios=(1.05, 1.45),
        constrained_layout=True,
    )

    ax_series.plot(
        dates,
        global_mean,
        color=Colors.GLOBAL_MEAN,
        linewidth=2.2,
        label="Mean across supported municipalities",
    )
    ax_series.plot(
        dates,
        global_median,
        color=Colors.GLOBAL_MEDIAN,
        linewidth=2.2,
        label="Median across supported municipalities",
    )
    if mask is not None:
        daily_observed = mask[:, supported_indices].astype(bool).sum(axis=1)
        ax_obs = ax_series.twinx()
        ax_obs.fill_between(
            dates,
            daily_observed,
            color="#495057",
            alpha=0.16,
            linewidth=0,
            label="Daily direct-observation support",
        )
        ax_obs.set_ylabel("Observed municipalities")
        ax_obs.set_ylim(0, max(int(daily_observed.max()) + 5, 10))
        ax_obs.grid(False)

    format_date_axis(ax_series)
    ax_series.set_ylabel("Processed concentration")
    ax_series.set_title(
        f"{variant_label}: Canonical Biomarker Series "
        f"({supported_indices.size} EDAR-supported municipalities)"
    )
    ax_series.grid(True, alpha=0.3)
    ax_series.legend(loc="upper left")

    subset = values[:, heatmap_indices]
    bounds = robust_bounds(subset, positive_only=True)
    vmax = bounds[1] if bounds is not None else None
    heatmap_df = pd.DataFrame(
        subset,
        index=dates,
        columns=[str(region_ids[idx]) for idx in heatmap_indices],
    )
    sns.heatmap(
        heatmap_df.T,
        ax=ax_heatmap,
        cmap="mako",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Processed concentration"},
    )
    step = max(1, len(dates) // 8)
    ax_heatmap.set_xticks(np.arange(0, len(dates), step))
    ax_heatmap.set_xticklabels(
        [d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right"
    )
    ax_heatmap.set_xlabel("Date")
    ax_heatmap.set_ylabel("Municipality")
    ax_heatmap.set_title(
        f"Top {len(heatmap_indices)} municipalities by direct-observation count"
    )

    save_figure(
        fig,
        output_path,
        dpi=Style.DPI,
        log_msg=f"Saved manuscript {variant_label} panel",
    )


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
    parser.add_argument(
        "--manuscript-plots",
        action="store_true",
        help=(
            "Also generate stable manuscript assets: canonical_biomarker_support.png "
            "and canonical_biomarker_series_{N1,N2,IP4}.png"
        ),
    )
    parser.add_argument(
        "--manuscript-heatmap-regions",
        type=int,
        default=40,
        help="Number of top direct-observation regions in manuscript heatmaps",
    )
    parser.add_argument(
        "--manuscript-only",
        action="store_true",
        help="Generate only stable manuscript assets and skip exploratory report plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manuscript_only:
        args.manuscript_plots = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    biomarker_variants, data_start = _load_biomarkers(args.dataset)
    variant_names = [
        name for name in _ordered_variant_names() if name in biomarker_variants
    ]
    variant_names.extend(sorted(set(biomarker_variants).difference(variant_names)))

    reference = biomarker_variants[variant_names[0]]
    region_ids = reference[REGION_COORD].values
    dates = pd.DatetimeIndex(reference[TEMPORAL_COORD].values)

    if args.manuscript_plots:
        plot_manuscript_support_summary(
            args.dataset,
            args.output_dir / "canonical_biomarker_support.png",
        )

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
        raw_values = _mask_pre_data_start(biomarker.values, data_start)
        values = np.log1p(raw_values)
        variant_label = variant_name.replace("edar_biomarker_", "")

        manuscript_mask: np.ndarray | None = None
        if args.manuscript_plots:
            dataset = xr.open_zarr(args.dataset)
            mask = _load_mask(dataset, variant_name)
            if mask is not None:
                mask = mask.sel({REGION_COORD: region_ids})
                manuscript_mask = mask.values.astype(bool)
            plot_manuscript_variant_panel(
                dates,
                raw_values,
                manuscript_mask,
                region_ids,
                args.output_dir / f"canonical_biomarker_series_{variant_label}.png",
                variant_label,
                max_heatmap_regions=args.manuscript_heatmap_regions,
            )

        if args.manuscript_only:
            continue

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
