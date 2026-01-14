from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def collect_case_window_samples(
    loader: DataLoader,
    n: int = 5,
    cases_feature_idx: int = 0,
    biomarker_feature_idx: int | None = 0,
    include_biomarkers: bool = True,
    include_mobility: bool = True,
    include_biomarkers_locf: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Collect case windows (history + target horizon) from the DataLoader's dataset.

    We intentionally avoid collation/batching here: for plotting we only need
    single samples, and collation would only stack tensors without changing
    values. Using the dataset ensures we capture exactly what the DataLoader
    would yield (pre-stack) for each sample.
    """
    samples: list[dict[str, Any]] = []

    dataset = loader.dataset
    k = min(int(n), len(dataset))
    if shuffle:
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), k=k)
    else:
        indices = list(range(k))

    for idx in indices:
        item = dataset[idx]
        case_hist = item["case_node"]
        targets = item["target"]

        if not isinstance(case_hist, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            raise TypeError(
                "collect_case_window_samples expects dataset items with torch tensors."
            )

        if case_hist.ndim != 2:
            raise ValueError("Expected `case_node` with shape (L, C).")

        history = case_hist[:, cases_feature_idx]
        if targets.ndim == 1:
            future = targets
        elif targets.ndim == 2:
            future = targets[:, cases_feature_idx]
        else:
            raise ValueError("Expected `target` with shape (H,) or (H, C).")

        window = torch.cat([history, future], dim=0).detach().cpu().numpy()
        sample: dict[str, Any] = {
            "node_id": int(item["target_node"]),
            "node_label": str(item.get("node_label", "")),
            "series": np.asarray(window, dtype=np.float32),
        }

        if include_biomarkers_locf:
            biomarkers = item.get("bio_node")
            if isinstance(biomarkers, torch.Tensor):
                # bio_node has shape (L, 4) with channels: [value, mask, age, has_data]
                if biomarkers.ndim != 2 or biomarkers.shape[1] != 4:
                    raise ValueError("Expected `bio_node` with shape (L, 4).")
                # Extract channels: [value, mask, age]
                value_channel = biomarkers[:, 0]
                mask_channel = biomarkers[:, 1]
                age_channel = biomarkers[:, 2]
                sample["biomarkers_locf"] = {
                    "value": value_channel.detach().cpu().numpy().astype(np.float32),
                    "mask": mask_channel.detach().cpu().numpy().astype(np.float32),
                    "age": age_channel.detach().cpu().numpy().astype(np.float32),
                }

        if include_biomarkers:
            biomarkers = item.get("bio_node")
            if isinstance(biomarkers, torch.Tensor):
                # bio_node has shape (L, 4) with channels: [value, mask, age, has_data]
                if biomarkers.ndim != 2 or biomarkers.shape[1] != 4:
                    raise ValueError("Expected `bio_node` with shape (L, 4).")
                # Use the value channel (index 0) for biomarker visualization
                biomarker_series = biomarkers[:, 0]
                sample["biomarkers"] = (
                    biomarker_series.detach().cpu().numpy().astype(np.float32)
                )

        if include_mobility:
            mob_graphs = item.get("mob")
            if isinstance(mob_graphs, list):
                means = []
                stds = []
                counts = []
                for g in mob_graphs:
                    edge_weight = getattr(g, "edge_weight", None)
                    if edge_weight is None:
                        edge_weight = getattr(g, "edge_attr", None)
                    if edge_weight is None:
                        means.append(0.0)
                        stds.append(0.0)
                        counts.append(0)
                    else:
                        weights = edge_weight.detach().cpu()
                        means.append(weights.mean().item())
                        stds.append(weights.std().item())
                        counts.append(len(weights))
                sample["mobility_mean"] = np.asarray(means, dtype=np.float32)
                sample["mobility_std"] = np.asarray(stds, dtype=np.float32)
                sample["mobility_count"] = np.asarray(counts, dtype=np.int32)

        samples.append(sample)

    return samples


def make_cases_window_figure(samples: list[dict[str, Any]], history_length: int):
    """
    Build a seaborn figure to visualize case windows (history + horizon).

    If samples include biomarker or mobility series, those are plotted alongside
    cases and each series is normalized to the same 0-1 scale for comparability.

    Returns a matplotlib Figure suitable for saving or TensorBoard logging.
    """
    if not samples:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    n = len(samples)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(12, max(2.5 * n, 3.0)), sharex=True
    )
    if n == 1:
        axes = [axes]

    colors = {
        "Cases": "#1f77b4",
        "Biomarkers": "#ff7f0e",
        "Incoming mobility": "#2ca02c",
    }

    for ax, sample in zip(axes, samples, strict=False):
        series = np.asarray(sample["series"], dtype=np.float32).reshape(-1)
        total_len = series.shape[0]

        biomarker_series = sample.get("biomarkers")
        mobility_mean = sample.get("mobility_mean")
        mobility_std = sample.get("mobility_std")

        plot_series: dict[str, np.ndarray] = {"Cases": series}

        if biomarker_series is not None:
            biomarker_series = np.asarray(biomarker_series, dtype=np.float32).reshape(
                -1
            )
            biomarker_aligned = np.full(total_len, np.nan, dtype=np.float32)
            hist_len = min(history_length, biomarker_series.shape[0])
            biomarker_aligned[:hist_len] = biomarker_series[:hist_len]
            plot_series["Biomarkers"] = biomarker_aligned

        mobility_mean_aligned = None
        mobility_std_aligned = None
        mobility_vmin, mobility_vmax = 0.0, 1.0
        if mobility_mean is not None:
            mobility_mean = np.asarray(mobility_mean, dtype=np.float32).reshape(-1)
            mobility_std = np.asarray(mobility_std, dtype=np.float32).reshape(-1)
            mobility_mean_aligned = np.full(total_len, np.nan, dtype=np.float32)
            mobility_std_aligned = np.full(total_len, np.nan, dtype=np.float32)
            hist_len = min(history_length, mobility_mean.shape[0])
            mobility_mean_aligned[:hist_len] = mobility_mean[:hist_len]
            mobility_std_aligned[:hist_len] = mobility_std[:hist_len]
            plot_series["Incoming mobility"] = mobility_mean_aligned

        # Normalize all series to 0-1 for comparability
        for label, values in plot_series.items():
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                vmin, vmax = valid.min(), valid.max()
                if vmax > vmin:
                    plot_series[label] = (values - vmin) / (vmax - vmin)
                    # Store vmin, vmax for mobility std normalization
                    if label == "Incoming mobility":
                        mobility_vmin, mobility_vmax = vmin, vmax
                else:
                    plot_series[label] = values - vmin
                    if label == "Incoming mobility":
                        mobility_vmin, mobility_vmax = vmin, vmin

        t = np.arange(total_len)

        # Plot cases
        ax.plot(t, plot_series["Cases"], label="Cases", color=colors["Cases"], linewidth=1.5)

        # Plot biomarkers
        if biomarker_series is not None and "Biomarkers" in plot_series:
            biomarker_aligned = plot_series["Biomarkers"]
            ax.plot(
                t,
                biomarker_aligned,
                label="Biomarkers",
                color=colors["Biomarkers"],
                linewidth=1.5,
                alpha=0.8,
            )

        # Plot mobility with std dev band
        if mobility_mean_aligned is not None and "Incoming mobility" in plot_series:
            mobility_mean_norm = plot_series["Incoming mobility"]
            # Normalize std by the same scale used for mean
            if mobility_vmax > mobility_vmin:
                mobility_std_norm = mobility_std_aligned / (mobility_vmax - mobility_vmin)
            else:
                mobility_std_norm = mobility_std_aligned

            ax.plot(
                t,
                mobility_mean_norm,
                label="Incoming mobility (mean)",
                color=colors["Incoming mobility"],
                linewidth=1.5,
                alpha=0.9,
            )
            # Plot shaded std dev band
            ax.fill_between(
                t,
                mobility_mean_norm - mobility_std_norm,
                mobility_mean_norm + mobility_std_norm,
                color=colors["Incoming mobility"],
                alpha=0.25,
                label="±1 std",
            )

        # Draw history/horizon separator
        ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5)

        # Title
        node_label = sample.get("node_label", "")
        node_id = sample.get("node_id", None)
        title = f"{node_label}".strip()
        if node_id is not None:
            title = f"{title} (id={node_id})" if title else f"id={node_id}"
        ax.set_title(title, fontsize=10, fontweight="semibold")

        ax.set_ylabel("Scaled value", fontsize=9)
        legend = ax.get_legend()
        if legend is not None and ax is not axes[0]:
            legend.remove()

    axes[-1].set_xlabel("Time index (history → horizon)")
    fig.tight_layout()
    return fig


def make_biomarker_locf_figure(
    samples: list[dict[str, Any]],
    history_length: int,
    age_visualization: str = "ribbon",
):
    """
    Build a figure to visualize LOCF biomarker data with age indicators.

    The age_visualization parameter controls how measurement staleness is shown:
    - "ribbon": Semi-transparent gray ribbon showing staleness (darker = older)
    - "scatter": Scatter points colored by age

    Returns a matplotlib Figure suitable for saving or TensorBoard logging.
    """
    if not samples:
        return None

    import matplotlib.pyplot as plt

    sns = _import_seaborn()
    if sns is None:
        return None

    sns.set_theme(style="whitegrid")
    n = len(samples)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(12, max(2.5 * n, 3.0)), sharex=True
    )
    if n == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples, strict=False):
        locf_data = sample.get("biomarkers_locf")
        if locf_data is None:
            ax.text(0.5, 0.5, "No LOCF data", ha="center", va="center")
            continue

        values = locf_data["value"]
        mask = locf_data["mask"]
        age = locf_data["age"]

        total_len = values.shape[0]
        t = np.arange(total_len)

        # Plot the LOCF values
        ax.plot(t, values, color="steelblue", linewidth=1.5, label="LOCF value")

        # Add age visualization
        if age_visualization == "ribbon":
            # Semi-transparent ribbon: darker = more stale
            for i in range(total_len):
                alpha = min(age[i], 1.0) * 0.7
                ax.axvspan(i - 0.4, i + 0.4, color="gray", alpha=alpha, zorder=-1)

            # Add a vertical line at history/horizon boundary
            ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5)

        elif age_visualization == "scatter":
            # Scatter points colored by age
            measured = mask > 0.5
            if np.any(measured):
                scatter = ax.scatter(
                    t[measured],
                    values[measured],
                    c=age[measured],
                    cmap="YlOrRd",
                    s=30,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                    label="Fresh measurement",
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Age (days)")

        # Title
        node_label = sample.get("node_label", "")
        node_id = sample.get("node_id", None)
        title = f"{node_label}".strip()
        if node_id is not None:
            title = f"{title} (id={node_id})" if title else f"id={node_id}"
        ax.set_title(title, fontsize=10, fontweight="semibold")

        ax.set_ylabel("Biomarker value (scaled)", fontsize=9)
        legend = ax.get_legend()
        if legend is not None and ax is not axes[0]:
            legend.remove()

    axes[-1].set_xlabel("Time index (history → horizon)")
    fig.tight_layout()
    return fig


def _import_seaborn():
    """Import seaborn optionally, returning None if not available."""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        return None


# =============================================================================
# Sample-Level Summary Statistics Functions
# =============================================================================


def compute_biomarker_sparsity_stats(
    samples: list[dict[str, Any]], age_max_days: int = 14
) -> pd.DataFrame:
    """Compute per-sample biomarker sparsity statistics.

    Args:
        samples: List of sample dicts containing biomarkers_locf data
        age_max_days: Maximum age in days (for unnormalizing age channel)

    Returns:
        DataFrame with columns: node_id, node_label, measurements_count,
            sparsity_pct, max_consecutive_missing, mean_staleness_days
    """
    rows = []

    for sample in samples:
        locf_data = sample.get("biomarkers_locf")
        if locf_data is None:
            continue

        mask = locf_data["mask"]
        age = locf_data["age"]

        n_measurements = int(np.sum(mask))
        total_timesteps = len(mask)
        sparsity_pct = 100 * (1 - n_measurements / total_timesteps)

        # Compute max consecutive missing
        max_consecutive_missing = 0
        current = 0
        for val in mask:
            if val < 0.5:
                current += 1
                max_consecutive_missing = max(max_consecutive_missing, current)
            else:
                current = 0

        # Compute mean staleness (only for non-mask positions)
        valid_age = age[mask > 0.5]
        if len(valid_age) > 0:
            mean_staleness_days = float(np.mean(valid_age) * age_max_days)
        else:
            mean_staleness_days = np.nan

        rows.append({
            "node_id": sample.get("node_id", ""),
            "node_label": sample.get("node_label", ""),
            "measurements_count": n_measurements,
            "total_timesteps": total_timesteps,
            "sparsity_pct": round(sparsity_pct, 2),
            "max_consecutive_missing": max_consecutive_missing,
            "mean_staleness_days": round(mean_staleness_days, 1) if not np.isnan(mean_staleness_days) else None,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_locf_age_stats(
    samples: list[dict[str, Any]], age_max_days: int = 14, n_bins: int = 10
) -> pd.DataFrame:
    """Compute distribution of LOCF age/staleness across samples.

    Args:
        samples: List of sample dicts containing biomarkers_locf data
        age_max_days: Maximum age in days (for unnormalizing age channel)
        n_bins: Number of bins for age distribution

    Returns:
        DataFrame with columns: age_bin, count, pct
    """
    all_ages = []

    for sample in samples:
        locf_data = sample.get("biomarkers_locf")
        if locf_data is not None:
            age = locf_data["age"]
            # Unnormalize to days
            age_days = age * age_max_days
            all_ages.extend(age_days[~np.isnan(age_days)])

    if not all_ages:
        return pd.DataFrame(columns=["age_bin_days", "count", "pct"])

    all_ages = np.array(all_ages)
    total = len(all_ages)

    # Create bins
    bin_edges = np.linspace(0, age_max_days, n_bins + 1)

    counts, _ = np.histogram(all_ages, bins=bin_edges)

    rows = []
    for i, count in enumerate(counts):
        bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        rows.append({
            "age_bin_days": bin_label,
            "count": int(count),
            "pct": round(100 * count / total, 2),
        })

    return pd.DataFrame(rows)


def export_summary_tables(
    samples: list[dict[str, Any]], output_dir: Path, skip_biomarker: bool = False
) -> dict[str, Path]:
    """Export sample-level summary statistics to CSV files.

    Args:
        samples: List of sample dicts
        output_dir: Directory to save CSV files
        skip_biomarker: If True, skip biomarker tables (use when exporting all-region biomarker tables separately)

    Returns:
        Dict mapping table name to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Sample summary table
    rows = []
    for sample in samples:
        row = {
            "node_id": sample.get("node_id", ""),
            "node_label": sample.get("node_label", ""),
        }

        # Case stats
        series = sample.get("series")
        if series is not None:
            series = np.asarray(series)
            row["case_mean"] = round(float(np.mean(series)), 2)
            row["case_std"] = round(float(np.std(series)), 2)
            row["case_min"] = round(float(np.min(series)), 2)
            row["case_max"] = round(float(np.max(series)), 2)

        # Biomarker stats
        locf_data = sample.get("biomarkers_locf")
        if locf_data is not None:
            values = locf_data["value"]
            row["biomarker_mean"] = round(float(np.mean(values)), 2)
            row["biomarker_std"] = round(float(np.std(values)), 2)
        else:
            row["biomarker_mean"] = None
            row["biomarker_std"] = None

        # Mobility stats
        mobility = sample.get("mobility_incoming")
        if mobility is not None:
            mobility = np.asarray(mobility)
            row["mobility_mean"] = round(float(np.mean(mobility)), 2)
            row["mobility_std"] = round(float(np.std(mobility)), 2)
        else:
            row["mobility_mean"] = None
            row["mobility_std"] = None

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / "samples_summary.csv"
        df.to_csv(path, index=False)
        exported["samples_summary"] = path

    # Biomarker tables (if not skipped)
    if not skip_biomarker:
        sparsity_df = compute_biomarker_sparsity_stats(samples)
        if not sparsity_df.empty:
            path = output_dir / "biomarker_sparsity.csv"
            sparsity_df.to_csv(path, index=False)
            exported["biomarker_sparsity"] = path

        age_df = compute_locf_age_stats(samples)
        if not age_df.empty:
            path = output_dir / "biomarker_age_dist.csv"
            age_df.to_csv(path, index=False)
            exported["biomarker_age_dist"] = path

    return exported


# =============================================================================
# Biomarker Sparsity Analysis (All Regions)
# =============================================================================


def plot_biomarker_age_heatmap(
    ax, biomarker_da, history_length: int | None = None, age_max_days: int = 14
) -> None:
    """Plot biomarker LOCF age (staleness) heatmap across ALL regions.

    Since LOCF imputation means every region that has ever had a measurement
    will have a value for all timesteps, the meaningful visualization is
    the AGE channel (days since last actual measurement), not binary missingness.

    Args:
        ax: Matplotlib axes to plot on
        biomarker_da: (time, region) biomarker DataArray
        history_length: Optional history/horizon separator position
        age_max_days: Maximum age in days for LOCF (for normalization)
    """
    import seaborn as sns

    from data.biomarker_preprocessor import BiomarkerPreprocessor

    # Convert to dataset format expected by preprocessor
    if biomarker_da.name is None:
        biomarker_ds = biomarker_da.to_dataset(name="edar_biomarker")
    else:
        biomarker_ds = biomarker_da.to_dataset()

    # Create preprocessor and process to get age channel
    preprocessor = BiomarkerPreprocessor(age_max=age_max_days)
    preprocessor.scaler_params = type(
        "obj",
        (object,),
        {"center": 0.0, "scale": 1.0, "is_fitted": True},
    )()

    processed = preprocessor.preprocess_dataset(biomarker_ds)
    # Shape: (time, regions, 3) -> [value, mask, age]
    age_channel = processed[:, :, 2]  # (time, regions)

    # Filter to regions with any finite values in original data
    has_data = np.isfinite(biomarker_da.values).any(axis=0)
    age_filtered = age_channel[:, has_data].T  # (regions, time)

    # Plot heatmap with regions as rows, time as columns
    # Light/White = Fresh (low age), Red/Dark = Stale (high age)
    sns.heatmap(
        age_filtered,
        cbar_kws={"label": f"Age (days, max={age_max_days})"},
        cmap="Reds",
        vmin=0,
        vmax=1,
        xticklabels=30,
        yticklabels=False,
        ax=ax,
    )

    # Draw history/horizon separator if provided
    if history_length is not None:
        ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5, linewidth=2)

    n_regions = age_filtered.shape[0]
    ax.set_title(f"Biomarker LOCF Age/Staleness ({n_regions} regions with data)", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Time index")


def compute_biomarker_sparsity_stats_all(
    biomarker_da,
) -> pd.DataFrame:
    """Compute sparsity statistics for ALL biomarker regions.

    Args:
        biomarker_da: (time, region) biomarker DataArray

    Returns:
        DataFrame with one row per region that has biomarker data
    """
    from data.preprocess.config import REGION_COORD

    # Filter to regions with any data
    has_data = np.isfinite(biomarker_da.values).any(axis=0)

    rows = []
    for region_idx, region_id in enumerate(biomarker_da[REGION_COORD].values):
        if not has_data[region_idx]:
            continue

        series = biomarker_da.isel({REGION_COORD: region_idx}).values

        # Compute sparsity
        n_measurements = int(np.sum(np.isfinite(series)))
        total_timesteps = len(series)
        sparsity_pct = 100 * (1 - n_measurements / total_timesteps)

        # Compute max consecutive missing
        missing_mask = np.isnan(series)
        max_consecutive_missing = 0
        current = 0
        for val in missing_mask:
            if val:
                current += 1
                max_consecutive_missing = max(max_consecutive_missing, current)
            else:
                current = 0

        rows.append({
            "node_id": int(region_id),
            "node_label": str(region_id),
            "measurements_count": n_measurements,
            "total_timesteps": total_timesteps,
            "sparsity_pct": round(sparsity_pct, 2),
            "max_consecutive_missing": max_consecutive_missing,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_locf_age_stats_from_biomarker_da(
    biomarker_da,
    age_max_days: int = 14,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute LOCF age distribution from raw biomarker data.

    Args:
        biomarker_da: (time, region) biomarker DataArray
        age_max_days: Maximum age in days for LOCF
        n_bins: Number of bins for age distribution

    Returns:
        DataFrame with age bin distribution
    """
    from data.biomarker_preprocessor import BiomarkerPreprocessor

    # Convert to dataset format expected by preprocessor
    if biomarker_da.name is None:
        biomarker_ds = biomarker_da.to_dataset(name="edar_biomarker")
    else:
        biomarker_ds = biomarker_da.to_dataset()

    # Create preprocessor and process to get age channel
    preprocessor = BiomarkerPreprocessor(age_max=age_max_days)
    preprocessor.scaler_params = type(
        "obj",
        (object,),
        {"center": 0.0, "scale": 1.0, "is_fitted": True},
    )()

    processed = preprocessor.preprocess_dataset(biomarker_ds)
    # Shape: (time, regions, 3) -> [value, mask, age]
    age_channel = processed[:, :, 2]  # (time, regions)

    # Collect all age values where there's data (mask > 0)
    all_ages = []
    for region_idx in range(age_channel.shape[1]):
        region_ages = age_channel[:, region_idx]
        all_ages.extend(region_ages)

    all_ages = np.array(all_ages)
    # Unnormalize to days
    age_days = all_ages * age_max_days

    # Filter out any NaN
    age_days = age_days[~np.isnan(age_days)]

    if len(age_days) == 0:
        return pd.DataFrame(columns=["age_bin_days", "count", "pct"])

    total = len(age_days)

    # Create bins
    bin_edges = np.linspace(0, age_max_days, n_bins + 1)

    counts, _ = np.histogram(age_days, bins=bin_edges)

    rows = []
    for i, count in enumerate(counts):
        bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        rows.append({
            "age_bin_days": bin_label,
            "count": int(count),
            "pct": round(100 * count / total, 2),
        })

    return pd.DataFrame(rows)


def export_biomarker_sparsity_tables(
    biomarker_da, output_dir: Path
) -> dict[str, Path]:
    """Export biomarker sparsity statistics for ALL regions to CSV files.

    Args:
        biomarker_da: (time, region) biomarker DataArray
        output_dir: Directory to save CSV files

    Returns:
        Dict mapping table name to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # 1. Biomarker sparsity stats (all regions)
    sparsity_df = compute_biomarker_sparsity_stats_all(biomarker_da)
    if not sparsity_df.empty:
        path = output_dir / "biomarker_sparsity.csv"
        sparsity_df.to_csv(path, index=False)
        exported["sparsity"] = path

    # 2. Age distribution (all regions)
    age_df = compute_locf_age_stats_from_biomarker_da(biomarker_da)
    if not age_df.empty:
        path = output_dir / "biomarker_age_dist.csv"
        age_df.to_csv(path, index=False)
        exported["age_dist"] = path

    return exported


def make_biomarker_sparsity_figure_all(
    biomarker_da,
    history_length: int | None = None,
):
    """Create multi-panel biomarker sparsity figure using ALL regions.

    This version uses LOCF age (staleness) for the heatmap since with LOCF
    imputation, every region that has ever had a measurement will have a value
    for all timesteps. The age channel shows how stale each value is.

    Args:
        biomarker_da: Raw biomarker DataArray (time, region)
        history_length: Optional history/horizon separator position

    Returns a matplotlib Figure suitable for saving.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: LOCF age heatmap (ALL regions)
    ax1 = fig.add_subplot(gs[0, :])
    plot_biomarker_age_heatmap(ax1, biomarker_da, history_length)

    # Panel 2: Sparsity distribution (ALL regions)
    ax2 = fig.add_subplot(gs[1, 0])
    sparsity_df = compute_biomarker_sparsity_stats_all(biomarker_da)
    if not sparsity_df.empty:
        ax2.hist(sparsity_df["sparsity_pct"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Sparsity (%)")
        ax2.set_ylabel("Number of regions")
        ax2.set_title("Sparsity Distribution (all regions)", fontsize=10, fontweight="semibold")
        ax2.axvline(
            sparsity_df["sparsity_pct"].median(),
            color="red",
            linestyle="--",
            label=f"Median: {sparsity_df['sparsity_pct'].median():.1f}%",
        )
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No sparsity data", ha="center", va="center")

    # Panel 3: Age distribution (ALL regions)
    ax3 = fig.add_subplot(gs[1, 1])
    age_df = compute_locf_age_stats_from_biomarker_da(biomarker_da)
    if not age_df.empty:
        ax3.bar(age_df["age_bin_days"], age_df["count"], color="coral", edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Age (days)")
        ax3.set_ylabel("Count")
        ax3.set_title("LOCF Value Age Distribution (all regions)", fontsize=10, fontweight="semibold")
        ax3.tick_params(axis="x", rotation=45)
    else:
        ax3.text(0.5, 0.5, "No age data", ha="center", va="center")

    return fig
