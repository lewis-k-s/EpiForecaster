from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.plotting import Colors, FigureSizes


def _split_biomarker_channels(
    biomarkers: torch.Tensor,
    variant_names: list[str] | None,
) -> dict[str, dict[str, np.ndarray]]:
    if biomarkers.ndim != 2:
        raise ValueError("Expected `bio_node` with shape (L, B).")

    total_channels = biomarkers.shape[1]
    if total_channels % 4 != 0:
        raise ValueError(
            "Expected biomarker channels in 4-channel blocks [value, mask, censor, age]."
        )

    variant_count = total_channels // 4
    if variant_count <= 0:
        raise ValueError("Unexpected biomarker channel layout.")

    if not variant_names or len(variant_names) != variant_count:
        variant_names = [f"variant_{i + 1}" for i in range(variant_count)]

    output: dict[str, dict[str, np.ndarray]] = {}
    for idx, name in enumerate(variant_names):
        base = idx * 4
        output[name] = {
            "value": biomarkers[:, base].detach().cpu().numpy().astype(np.float32),
            "mask": biomarkers[:, base + 1].detach().cpu().numpy().astype(np.float32),
            "censor": biomarkers[:, base + 2].detach().cpu().numpy().astype(np.float32),
            "age": biomarkers[:, base + 3].detach().cpu().numpy().astype(np.float32),
        }

    return output


def _split_precomputed_variants(
    precomputed_biomarkers: torch.Tensor,
    variant_names: list[str] | None,
) -> dict[str, dict[str, np.ndarray]]:
    if precomputed_biomarkers.ndim != 3:
        raise ValueError("Expected precomputed biomarkers with shape (T, N, B).")

    total_channels = precomputed_biomarkers.shape[2]
    if total_channels % 4 != 0:
        raise ValueError(
            "Expected biomarker channels in 4-channel blocks [value, mask, censor, age]."
        )

    variant_count = total_channels // 4
    if not variant_names or len(variant_names) != variant_count:
        variant_names = [f"variant_{i + 1}" for i in range(variant_count)]

    output: dict[str, dict[str, np.ndarray]] = {}
    for idx, name in enumerate(variant_names):
        base = idx * 4
        output[name] = {
            "value": precomputed_biomarkers[:, :, base].cpu().numpy(),
            "mask": precomputed_biomarkers[:, :, base + 1].cpu().numpy(),
            "censor": precomputed_biomarkers[:, :, base + 2].cpu().numpy(),
            "age": precomputed_biomarkers[:, :, base + 3].cpu().numpy(),
        }

    return output


def collect_case_window_samples(
    loader: DataLoader,
    n: int = 5,
    cases_feature_idx: int = 0,
    biomarker_feature_idx: int | None = 0,
    include_biomarkers: bool = True,
    include_mobility: bool = True,
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

    variant_names = getattr(dataset, "biomarker_variants", None)

    for idx in indices:
        item = dataset[idx]
        # Use cases_hist instead of deprecated case_node
        cases_hist = item["cases_hist"]
        targets = item["cases_target"]

        if not isinstance(cases_hist, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            raise TypeError(
                "collect_case_window_samples expects dataset items with torch tensors."
            )

        if cases_hist.ndim != 2:
            raise ValueError("Expected `cases_hist` with shape (L, 3).")

        # Extract value channel (index 0) for the series
        history = cases_hist[:, cases_feature_idx]
        # Extract cases age channel (index 2) for ribbon visualization
        cases_age = cases_hist[:, 2].detach().cpu().numpy().astype(np.float32)

        if targets.ndim == 1:
            future = targets
        elif targets.ndim == 2:
            future = targets[:, cases_feature_idx]
        else:
            raise ValueError("Expected `cases_target` with shape (H,) or (H, C).")

        window = torch.cat([history, future], dim=0).detach().cpu().numpy()
        sample: dict[str, Any] = {
            "node_id": int(item["target_node"]),
            "node_label": str(item.get("node_label", "")),
            "window_start": int(item.get("window_start", -1)),
            "series": np.asarray(window, dtype=np.float32),
            "cases_age": cases_age,  # History-only cases age for ribbon
        }

        if include_biomarkers:
            biomarkers = item.get("bio_node")
            if isinstance(biomarkers, torch.Tensor):
                # Exclude the last channel (has_data) which is not part of variant blocks
                # Shape: (L, 13) -> (L, 12) where 12 = 3 variants * 4 channels
                biomarkers_for_splitting = biomarkers[:, :-1]
                biomarker_channels = _split_biomarker_channels(
                    biomarkers_for_splitting, variant_names
                )
                sample["biomarkers"] = {
                    name: channels["value"]
                    for name, channels in biomarker_channels.items()
                }
                sample["biomarkers_age"] = {
                    name: channels["age"]
                    for name, channels in biomarker_channels.items()
                }
                sample["biomarkers_censor"] = {
                    name: channels["censor"]
                    for name, channels in biomarker_channels.items()
                }

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
        nrows=n,
        ncols=1,
        figsize=(FigureSizes.TIME_SERIES[0], max(2.5 * n, 3.0)),
        sharex=True,
    )
    if n == 1:
        axes = [axes]

    colors = {
        "Cases": Colors.CASES,
        "Biomarkers": Colors.BIOMARKER,
        "Incoming mobility": Colors.MOBILITY,
    }

    for ax, sample in zip(axes, samples, strict=False):
        # Series already includes history + horizon
        series = np.asarray(sample["series"], dtype=np.float32).reshape(-1)
        total_len = series.shape[0]
        horizon_length = total_len - history_length

        cases_age = sample.get("cases_age")
        biomarker_series = sample.get("biomarkers") or {}
        biomarker_age = sample.get("biomarkers_age") or {}
        biomarker_censor = sample.get("biomarkers_censor") or {}
        mobility_mean = sample.get("mobility_mean")
        mobility_std = sample.get("mobility_std")

        plot_series: dict[str, np.ndarray] = {"Cases": series}

        # Extract and pad biomarkers for plotting (history only)
        biomarker_padded: dict[str, np.ndarray] = {}
        biomarker_age_padded: dict[str, np.ndarray] = {}
        if biomarker_series:
            for name, series_values in biomarker_series.items():
                series_values = np.asarray(series_values, dtype=np.float32).reshape(-1)
                assert series_values.shape[0] == history_length, (
                    f"Biomarker length {series_values.shape[0]} != history_length {history_length}"
                )
                padded = np.concatenate(
                    [series_values, np.full(horizon_length, np.nan)]
                )
                # Use shorthand label: N1, N2, IP4
                label = str(name).replace("edar_biomarker_", "")
                biomarker_padded[label] = padded
                plot_series[label] = padded

                age_values = biomarker_age.get(name)
                if age_values is not None:
                    age_values = np.asarray(age_values, dtype=np.float32).reshape(-1)
                    assert age_values.shape[0] == history_length
                    biomarker_age_padded[label] = np.concatenate(
                        [age_values, np.full(horizon_length, np.nan)]
                    )

        # Extract and pad mobility for plotting (history only)
        mobility_mean_padded = None
        mobility_std_padded = None
        mobility_vmin, mobility_vmax = 0.0, 1.0
        if mobility_mean is not None:
            mobility_mean = np.asarray(mobility_mean, dtype=np.float32).reshape(-1)
            mobility_std = np.asarray(mobility_std, dtype=np.float32).reshape(-1)
            assert mobility_mean.shape[0] == history_length, (
                f"Mobility length {mobility_mean.shape[0]} != history_length {history_length}"
            )
            assert mobility_std.shape[0] == history_length
            # Pad with NaNs for horizon portion
            mobility_mean_padded = np.concatenate(
                [mobility_mean, np.full(horizon_length, np.nan)]
            )
            mobility_std_padded = np.concatenate(
                [mobility_std, np.full(horizon_length, np.nan)]
            )
            plot_series["Incoming mobility"] = mobility_mean_padded

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

        # Add cases age ribbon at top of plot (just below top edge)
        if cases_age is not None:
            # cases_age is history-only, pad with NaNs for horizon
            cases_age_padded = np.concatenate(
                [cases_age, np.full(horizon_length, np.nan)]
            )
            valid_mask = ~np.isnan(cases_age_padded)
            valid_t = t[valid_mask]
            valid_age = cases_age_padded[valid_mask]

            if len(valid_t) > 0:
                # Draw ribbon for each timestep with color based on age
                # ymin/ymax are in axes coordinates (0-1), so 0.95-1.0 is top 5%
                for ti, age in zip(valid_t, valid_age):
                    # Use RdYlGn colormap: green=fresh (age=0), red=stale (age=1)
                    color = plt.get_cmap("RdYlGn_r")(min(age, 1.0))
                    ax.axvspan(
                        ti - 0.5,
                        ti + 0.5,
                        ymin=0.95,
                        ymax=1.0,
                        color=color,
                        alpha=0.7,
                        zorder=-1,
                    )

                # Add label to the right of the ribbon, aligned with history/horizon line
                ax.text(
                    history_length - 0.5,
                    0.975,  # Center of the ribbon (ymin=0.95, ymax=1.0)
                    " cases age",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=7,
                    fontweight="semibold",
                    color="#333333",
                )

        # Plot cases
        ax.plot(
            t, plot_series["Cases"], label="Cases", color=colors["Cases"], linewidth=1.5
        )

        # Plot biomarkers (per-variant lines)
        if biomarker_padded:
            palette = sns.color_palette("tab10", n_colors=len(biomarker_padded))
            biomarker_colors = dict(zip(biomarker_padded.keys(), palette, strict=False))
            for label, series_values in biomarker_padded.items():
                biomarker_norm = plot_series[label]
                ax.plot(
                    t,
                    biomarker_norm,
                    label=label,
                    color=biomarker_colors[label],
                    linewidth=1.5,
                    alpha=0.85,
                )

            # Add age ribbon at bottom of plot (just above x-axis)
            if biomarker_age_padded:
                stacked = np.vstack(list(biomarker_age_padded.values()))
                combined_age = np.nanmin(stacked, axis=0)
                valid_mask = ~np.isnan(combined_age)
                valid_t = t[valid_mask]
                valid_age = combined_age[valid_mask]

                if len(valid_t) > 0:
                    cmap = plt.get_cmap("RdYlGn_r")
                    for ti, age in zip(valid_t, valid_age):
                        color = cmap(min(age, 1.0))
                        ax.axvspan(
                            ti - 0.5,
                            ti + 0.5,
                            ymin=0,
                            ymax=0.05,
                            color=color,
                            alpha=0.7,
                            zorder=-1,
                        )

                    ax.text(
                        history_length - 0.5,
                        0.025,
                        " biomarker age (min)",
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=7,
                        fontweight="semibold",
                        color="#333333",
                    )

            # Add censor ribbon just above age ribbon (ymin=0.05, ymax=0.10)
            if biomarker_censor:
                for variant, flags in biomarker_censor.items():
                    flags_padded = np.concatenate(
                        [flags, np.full(horizon_length, np.nan)]
                    )
                    valid_mask = ~np.isnan(flags_padded)
                    valid_t = t[valid_mask]
                    valid_flags = flags_padded[valid_mask]

                    if len(valid_t) > 0:
                        # Use purple colormap for censor ribbon (distinct from age ribbon)
                        cmap = plt.get_cmap("Purples")
                        for ti, flag in zip(valid_t, valid_flags):
                            if flag == 1.0:
                                # censored=1, so darker purple
                                ax.axvspan(
                                    ti - 0.5,
                                    ti + 0.5,
                                    ymin=0.05,
                                    ymax=0.10,
                                    color=cmap(0.8),
                                    alpha=0.7,
                                    zorder=-1,
                                )

                if biomarker_censor:
                    ax.text(
                        history_length - 0.5,
                        0.075,
                        " LD censored",
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=7,
                        fontweight="semibold",
                        color="#333333",
                    )

        # Plot mobility with std dev band
        if mobility_mean_padded is not None and "Incoming mobility" in plot_series:
            mobility_mean_norm = plot_series["Incoming mobility"]
            # Normalize std by the same scale used for mean
            if mobility_vmax > mobility_vmin:
                mobility_std_norm = mobility_std_padded / (
                    mobility_vmax - mobility_vmin
                )
            else:
                mobility_std_norm = mobility_std_padded

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
        window_start = sample.get("window_start", None)
        title = f"{node_label}".strip()
        if node_id is not None:
            title = f"{title} (id={node_id})" if title else f"id={node_id}"
        if window_start is not None and window_start >= 0:
            title = f"{title} | window_start={window_start}"
        ax.set_title(title, fontsize=10, fontweight="semibold")

        ax.set_ylabel("Scaled value", fontsize=9)

    # Add figure-level legend (shared across all subplots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

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
        DataFrame with columns: node_id, node_label, variant, measurements_count,
            sparsity_pct, max_consecutive_missing, mean_staleness_days
    """
    rows = []

    for sample in samples:
        locf_data = sample.get("biomarkers_locf") or {}
        if not locf_data:
            continue

        for variant, channels in locf_data.items():
            mask = channels["mask"]
            age = channels["age"]

            n_measurements = int(np.sum(mask))
            total_timesteps = len(mask)
            sparsity_pct = 100 * (1 - n_measurements / total_timesteps)

            max_consecutive_missing = 0
            current = 0
            for val in mask:
                if val < 0.5:
                    current += 1
                    max_consecutive_missing = max(max_consecutive_missing, current)
                else:
                    current = 0

            valid_age = age[mask > 0.5]
            if len(valid_age) > 0:
                mean_staleness_days = float(np.mean(valid_age) * age_max_days)
            else:
                mean_staleness_days = np.nan

            rows.append(
                {
                    "node_id": sample.get("node_id", ""),
                    "node_label": sample.get("node_label", ""),
                    "variant": variant,
                    "measurements_count": n_measurements,
                    "total_timesteps": total_timesteps,
                    "sparsity_pct": round(sparsity_pct, 2),
                    "max_consecutive_missing": max_consecutive_missing,
                    "mean_staleness_days": round(mean_staleness_days, 1)
                    if not np.isnan(mean_staleness_days)
                    else None,
                }
            )

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
        DataFrame with columns: variant, age_bin_days, count, pct
    """
    rows = []

    for sample in samples:
        locf_data = sample.get("biomarkers_locf") or {}
        for variant, channels in locf_data.items():
            age = channels["age"]
            age_days = age * age_max_days
            age_days = age_days[~np.isnan(age_days)]
            if len(age_days) == 0:
                continue

            total = len(age_days)
            bin_edges = np.linspace(0, age_max_days, n_bins + 1)
            counts, _ = np.histogram(age_days, bins=bin_edges)

            for i, count in enumerate(counts):
                bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
                rows.append(
                    {
                        "variant": variant,
                        "age_bin_days": bin_label,
                        "count": int(count),
                        "pct": round(100 * count / total, 2),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["variant", "age_bin_days", "count", "pct"])

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
        locf_data = sample.get("biomarkers_locf") or {}
        if locf_data:
            values = np.concatenate(
                [channels["value"].reshape(-1) for channels in locf_data.values()]
            )
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
# Biomarker Sparsity Analysis from Precomputed Data (Model-Visible Metrics)
# =============================================================================


def compute_biomarker_sparsity_stats_from_precomputed(
    precomputed_biomarkers: torch.Tensor,
    biomarker_available_mask: torch.Tensor,
    region_ids: np.ndarray,
    age_max: int = 14,
    variant_names: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Compute per-region biomarker sparsity from precomputed tensor.

    This function uses the model-visible data (mask channel) to compute
    sparsity statistics. Only regions with biomarker data (has_data > 0)
    are included in the sparsity analysis.

    Args:
        precomputed_biomarkers: (T, N,4 * V) tensor with [value, mask, censor, age] blocks
        biomarker_available_mask: (N, B) tensor indicating region-level availability
        region_ids: Array of region IDs
        age_max: Maximum age in days (for reference, sparsity uses mask only)
        variant_names: Optional list of biomarker variant names

    Returns:
        Tuple of (DataFrame, context_string):
        - DataFrame with columns: node_id, node_label, variant, measurements_count,
            total_timesteps, sparsity_pct, max_consecutive_missing,
            censored_count, censored_pct
        - Context string: "n of N regions have biomarker data"
    """
    T, N, _ = precomputed_biomarkers.shape

    # Filter to regions with biomarker data
    has_data = biomarker_available_mask[:, 0].cpu().numpy() > 0.5
    n_with_data = int(has_data.sum())
    n_total = len(region_ids)

    variants = _split_precomputed_variants(precomputed_biomarkers, variant_names)

    rows = []
    for variant, channels in variants.items():
        mask_channel = channels["mask"]
        censor_channel = channels["censor"]

        for region_idx, region_id in enumerate(region_ids):
            if not has_data[region_idx]:
                continue

            mask = mask_channel[:, region_idx]
            n_measurements = int((mask == 1.0).sum())
            sparsity_pct = 100 * (1 - n_measurements / T)

            max_consecutive_missing = 0
            current = 0
            for val in mask:
                if val < 0.5:
                    current += 1
                    max_consecutive_missing = max(max_consecutive_missing, current)
                else:
                    current = 0

            # Compute censored count from precomputed censor channel
            # Assert censor data is valid for regions with biomarker data
            censor = censor_channel[:, region_idx]
            if np.any(mask == 1.0) and np.all(np.isnan(censor[mask == 1.0])):
                raise ValueError(
                    f"Censor data is all NaN for region {region_id}, variant {variant} "
                    "with non-zero mask values. This indicates missing required data."
                )

            measured_mask = mask == 1.0
            censored_mask = measured_mask & (censor == 1.0)
            censored_count = int(censored_mask.sum())
            n_valid_measured = int(measured_mask.sum())

            row = {
                "node_id": int(region_id),
                "node_label": str(region_id),
                "variant": variant,
                "measurements_count": n_measurements,
                "total_timesteps": T,
                "sparsity_pct": round(sparsity_pct, 2),
                "max_consecutive_missing": max_consecutive_missing,
                "censored_count": censored_count,
                "censored_pct": round(100 * (censored_count / n_valid_measured), 2)
                if n_valid_measured > 0
                else None,
            }

            rows.append(row)

    context = f"{n_with_data} of {n_total} regions have biomarker data"

    if not rows:
        return pd.DataFrame(), context

    return pd.DataFrame(rows), context


def compute_biomarker_age_dist_from_precomputed(
    precomputed_biomarkers: torch.Tensor,
    age_max: int = 14,
    n_bins: int = 10,
    variant_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute LOCF age distribution from precomputed biomarker tensor.

    Args:
        precomputed_biomarkers: (T, N, 3 * V) tensor with [value, mask, age] blocks
        age_max: Maximum age in days for unnormalization
        n_bins: Number of bins for age distribution
        variant_names: Optional list of biomarker variant names

    Returns:
        DataFrame with columns: variant, age_bin_days, count, pct
    """
    variants = _split_precomputed_variants(precomputed_biomarkers, variant_names)
    rows = []

    for variant, channels in variants.items():
        age_channel = channels["age"]
        age_days = age_channel * age_max
        all_ages = age_days.flatten()
        all_ages = all_ages[~np.isnan(all_ages)]

        if len(all_ages) == 0:
            continue

        total = len(all_ages)
        bin_edges = np.linspace(0, age_max, n_bins + 1)
        counts, _ = np.histogram(all_ages, bins=bin_edges)

        for i, count in enumerate(counts):
            bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
            rows.append(
                {
                    "variant": variant,
                    "age_bin_days": bin_label,
                    "count": int(count),
                    "pct": round(100 * count / total, 2),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variant", "age_bin_days", "count", "pct"])

    return pd.DataFrame(rows)


def plot_biomarker_age_heatmap_from_precomputed(
    ax,
    precomputed_biomarkers: torch.Tensor,
    biomarker_available_mask: torch.Tensor,
    history_length: int | None = None,
    age_max: int = 14,
    variant_names: list[str] | None = None,
) -> None:
    """Plot biomarker LOCF age (staleness) heatmap from precomputed data.

    Args:
        ax: Matplotlib axes to plot on
        precomputed_biomarkers: (T, N, 3 * V) tensor with [value, mask, age] blocks
        biomarker_available_mask: (N, B) tensor indicating region-level availability
        history_length: Optional history/horizon separator position
        age_max: Maximum age in days (for normalization label only)
        variant_names: Optional list of biomarker variant names
    """
    import seaborn as sns

    variants = _split_precomputed_variants(precomputed_biomarkers, variant_names)
    age_stack = np.stack([channels["age"] for channels in variants.values()], axis=0)
    age_channel = np.nanmin(age_stack, axis=0)

    # Filter to regions with biomarker availability (has_data > 0)
    has_data = biomarker_available_mask[:, 0].cpu().numpy() > 0.5
    age_filtered = age_channel[:, has_data].T  # (regions, time)

    # Plot heatmap with regions as rows, time as columns
    sns.heatmap(
        age_filtered,
        cbar_kws={"label": f"Age (normalized 0-1, max={age_max} days)"},
        cmap="Reds",
        vmin=0,
        vmax=1,
        xticklabels=30,
        yticklabels=False,
        ax=ax,
    )

    # Draw history/horizon separator if provided
    if history_length is not None:
        ax.axvline(
            history_length - 0.5, color="black", linestyle="--", alpha=0.5, linewidth=2
        )

    n_regions = age_filtered.shape[0]
    ax.set_title(
        f"Biomarker LOCF Age/Staleness ({n_regions} regions with data)",
        fontsize=10,
        fontweight="semibold",
    )
    ax.set_xlabel("Time index")


def export_biomarker_sparsity_tables_from_precomputed(
    precomputed_biomarkers: torch.Tensor,
    biomarker_available_mask: torch.Tensor,
    region_ids: np.ndarray,
    output_dir: Path,
    age_max: int = 14,
    variant_names: list[str] | None = None,
) -> tuple[dict[str, Path], str]:
    """Export biomarker sparsity statistics from precomputed data to CSV files.

    Args:
        precomputed_biomarkers: (T, N,4 * V) tensor with [value, mask, censor, age] blocks
        biomarker_available_mask: (N, B) tensor indicating region-level availability
        region_ids: Array of region IDs
        output_dir: Directory to save CSV files
        age_max: Maximum age in days
        variant_names: Optional list of biomarker variant names

    Returns:
        Tuple of (exported_files_dict, context_string)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    sparsity_df, context = compute_biomarker_sparsity_stats_from_precomputed(
        precomputed_biomarkers,
        biomarker_available_mask,
        region_ids,
        age_max,
        variant_names,
    )
    if not sparsity_df.empty:
        path = output_dir / "biomarker_sparsity.csv"
        sparsity_df.to_csv(path, index=False)
        exported["sparsity"] = path

    age_df = compute_biomarker_age_dist_from_precomputed(
        precomputed_biomarkers, age_max, variant_names=variant_names
    )
    if not age_df.empty:
        path = output_dir / "biomarker_age_dist.csv"
        age_df.to_csv(path, index=False)
        exported["age_dist"] = path

    return exported, context


def make_biomarker_sparsity_figure_all_from_precomputed(
    precomputed_biomarkers: torch.Tensor,
    biomarker_available_mask: torch.Tensor,
    region_ids: np.ndarray,
    history_length: int | None = None,
    age_max: int = 14,
    variant_names: list[str] | None = None,
):
    """Create multi-panel biomarker sparsity figure using precomputed data."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    plot_biomarker_age_heatmap_from_precomputed(
        ax1,
        precomputed_biomarkers,
        biomarker_available_mask,
        history_length,
        age_max,
        variant_names=variant_names,
    )

    ax2 = fig.add_subplot(gs[1, 0])
    sparsity_df, context = compute_biomarker_sparsity_stats_from_precomputed(
        precomputed_biomarkers,
        biomarker_available_mask,
        region_ids,
        age_max,
        variant_names,
    )
    if not sparsity_df.empty:
        ax2.hist(
            sparsity_df["sparsity_pct"],
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Sparsity (%)")
        ax2.set_ylabel("Number of regions")
        ax2.set_title(
            f"Sparsity Distribution ({context})",
            fontsize=10,
            fontweight="semibold",
        )
        ax2.axvline(
            sparsity_df["sparsity_pct"].median(),
            color="red",
            linestyle="--",
            label=f"Median: {sparsity_df['sparsity_pct'].median():.1f}%",
        )
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No sparsity data", ha="center", va="center")

    ax3 = fig.add_subplot(gs[1, 1])
    if not sparsity_df.empty and "censored_pct" in sparsity_df.columns:
        # Plot censored percentage distribution
        censored_data = sparsity_df.dropna(subset=["censored_pct"])
        if not censored_data.empty:
            if "variant" in censored_data.columns:
                sns = _import_seaborn()
                if sns is not None:
                    sns.boxplot(
                        data=censored_data,
                        x="variant",
                        y="censored_pct",
                        ax=ax3,
                        color="lightcoral",
                    )
                ax3.set_ylim(0, 100)
                ax3.set_ylabel("Censored (%)")
                ax3.set_title(
                    "LD Censoring Distribution (by variant)",
                    fontsize=10,
                    fontweight="semibold",
                )
                ax3.tick_params(axis="x", rotation=45)
            else:
                ax3.hist(
                    censored_data["censored_pct"],
                    bins=20,
                    color="lightcoral",
                    edgecolor="black",
                    alpha=0.7,
                )
                ax3.set_xlabel("Censored (%)")
                ax3.set_ylabel("Number of regions")
                ax3.set_title(
                    "LD Censoring Distribution",
                    fontsize=10,
                    fontweight="semibold",
                )
    else:
        # Fallback to age distribution if no censor data
        age_df = compute_biomarker_age_dist_from_precomputed(
            precomputed_biomarkers, age_max, variant_names=variant_names
        )
        if not age_df.empty:
            if "variant" in age_df.columns:
                sns = _import_seaborn()
                if sns is not None:
                    sns.barplot(
                        data=age_df,
                        x="age_bin_days",
                        y="count",
                        hue="variant",
                        ax=ax3,
                    )
            else:
                ax3.bar(
                    age_df["age_bin_days"],
                    age_df["count"],
                    color="coral",
                    edgecolor="black",
                    alpha=0.7,
                )
            ax3.set_xlabel("Age (days)")
            ax3.set_ylabel("Count")
            ax3.set_title(
                "LOCF Value Age Distribution (all regions)",
                fontsize=10,
                fontweight="semibold",
            )
            ax3.tick_params(axis="x", rotation=45)

    return fig
