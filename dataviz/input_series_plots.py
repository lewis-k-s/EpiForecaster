"""Generate input series window plots from EpiForecaster config.

This script loads a training config, creates a dataloader, and generates
window visualizations showing all 4 input series (cases, biomarkers x3,
hospitalizations, deaths) with their age ribbons.

Each window is visualized as 4 subplots:
- Cases: 1 line + age ribbon
- Biomarkers: 3 lines (variants) + 3 age ribbons
- Hospitalizations: 1 line + age ribbon
- Deaths: 1 line + age ribbon

Outputs:
    input_series_ordered.png: First N windows in dataset order
    input_series_shuffled.png: N random windows
    samples_summary.csv: Summary statistics for collected samples
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.preprocess.config import REGION_COORD
from models.configs import EpiForecasterConfig
from utils.logging import setup_logging, suppress_zarr_warnings

suppress_zarr_warnings()
logger = logging.getLogger(__name__)

# Channel indices for 4-channel format [value, mask, censor, age]
VALUE_IDX = 0
MASK_IDX = 1
CENSOR_IDX = 2
AGE_IDX = 3

# Channel indices for 3-channel format [value, mask, age]
CLINICAL_VALUE_IDX = 0
CLINICAL_MASK_IDX = 1
CLINICAL_AGE_IDX = 2


def collect_window_samples(
    loader: DataLoader,
    n: int = 5,
    shuffle: bool = False,
    seed: int | None = None,
    prefer_ww_horizon: bool = True,
) -> list[dict[str, Any]]:
    """Collect window samples from DataLoader's dataset.

    Collects all 4 input series (cases, biomarkers, hosp, deaths) with their
    targets and age channels for visualization.
    """
    samples: list[dict[str, Any]] = []
    dataset = loader.dataset
    k = min(int(n), len(dataset))

    ww_positive_indices: list[int] = []
    if prefer_ww_horizon and hasattr(dataset, "_index_map"):
        L = int(dataset.config.model.history_length)
        H = int(dataset.config.model.forecast_horizon)
        for sample_idx, (target_idx, start_idx) in enumerate(dataset._index_map):
            ww_slice = dataset.precomputed_ww_mask[
                start_idx + L : start_idx + L + H,
                target_idx,
            ]
            if bool(torch.any(ww_slice > 0).item()):
                ww_positive_indices.append(sample_idx)

    if ww_positive_indices:
        if shuffle:
            rng = random.Random(seed)
            indices = rng.sample(
                ww_positive_indices, k=min(k, len(ww_positive_indices))
            )
        else:
            indices = ww_positive_indices[:k]
    else:
        if shuffle:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), k=k)
        else:
            indices = list(range(k))

    # Get biomarker variant names from dataset if available
    variant_names = getattr(dataset, "biomarker_variants", None)
    if not variant_names:
        # Default to 3 variants
        variant_names = ["N1", "N2", "IP4"]

    for idx in indices:
        item = dataset[idx]

        # Extract cases (3 channels: value, mask, age)
        cases_hist = item["cases_hist"]  # (L, 3)
        cases_target = item["cases_target"]  # (H,)
        cases_target_mask = item["cases_target_mask"]  # (H,)

        # Extract hospitalizations (3 channels: value, mask, age)
        hosp_hist = item["hosp_hist"]  # (L, 3)
        hosp_target = item["hosp_target"]  # (H,)
        hosp_target_mask = item["hosp_target_mask"]  # (H,)

        # Extract deaths (3 channels: value, mask, age)
        deaths_hist = item["deaths_hist"]  # (L, 3)
        deaths_target = item["deaths_target"]  # (H,)
        deaths_target_mask = item["deaths_target_mask"]  # (H,)
        ww_target = item["ww_target"]  # (H,)
        ww_target_mask = item["ww_target_mask"]  # (H,)

        # Extract biomarkers (4 channels per variant: value, mask, censor, age)
        bio_node = item.get("bio_node")  # (L, 12) for 3 variants

        sample: dict[str, Any] = {
            "node_id": int(item["target_node"]),
            "node_label": str(item.get("node_label", "")),
            "window_start": int(item.get("window_start", -1)),
            # Cases series: concat history + target
            "cases_series": torch.cat(
                [cases_hist[:, CLINICAL_VALUE_IDX], cases_target], dim=0
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "cases_age": cases_hist[:, CLINICAL_AGE_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "cases_mask": cases_hist[:, CLINICAL_MASK_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "cases_target_mask": cases_target_mask.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            # Hospitalizations series: concat history + target
            "hosp_series": torch.cat(
                [hosp_hist[:, CLINICAL_VALUE_IDX], hosp_target], dim=0
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "hosp_age": hosp_hist[:, CLINICAL_AGE_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "hosp_mask": hosp_hist[:, CLINICAL_MASK_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "hosp_target_mask": hosp_target_mask.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            # Deaths series: concat history + target
            "deaths_series": torch.cat(
                [deaths_hist[:, CLINICAL_VALUE_IDX], deaths_target], dim=0
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "deaths_age": deaths_hist[:, CLINICAL_AGE_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "deaths_mask": deaths_hist[:, CLINICAL_MASK_IDX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "deaths_target_mask": deaths_target_mask.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "ww_target": ww_target.detach().cpu().numpy().astype(np.float32),
            "ww_target_mask": ww_target_mask.detach().cpu().numpy().astype(np.float32),
        }

        # Full-window observation masks used to color observed vs interpolated points.
        sample["cases_obs_mask_full"] = np.concatenate(
            [sample["cases_mask"], sample["cases_target_mask"]]
        ).astype(np.float32)
        sample["hosp_obs_mask_full"] = np.concatenate(
            [sample["hosp_mask"], sample["hosp_target_mask"]]
        ).astype(np.float32)
        sample["deaths_obs_mask_full"] = np.concatenate(
            [sample["deaths_mask"], sample["deaths_target_mask"]]
        ).astype(np.float32)

        # Extract biomarkers (3 variants, 4 channels each)
        if isinstance(bio_node, torch.Tensor):
            bio_hist = bio_node.detach().cpu().numpy().astype(np.float32)  # (L, 12)

            # Split into 3 variants, 4 channels each
            biomarkers: dict[str, dict[str, np.ndarray]] = {}
            for i, name in enumerate(variant_names[:3]):  # Ensure max 3 variants
                base = i * 4
                biomarkers[name] = {
                    "value": bio_hist[:, base + VALUE_IDX],
                    "mask": bio_hist[:, base + MASK_IDX],
                    "censor": bio_hist[:, base + CENSOR_IDX],
                    "age": bio_hist[:, base + AGE_IDX],
                }
            sample["biomarkers"] = biomarkers

            # Use precomputed WW target-space trajectory for both history+horizon.
            # This keeps WW on a consistent scale across the full window.
            target_node = int(item["target_node"])
            window_start = int(item["window_start"])
            ww_full = dataset.precomputed_ww[
                window_start : window_start + len(cases_hist) + len(cases_target),
                target_node,
            ]
            ww_full_mask = dataset.precomputed_ww_mask[
                window_start : window_start + len(cases_hist) + len(cases_target),
                target_node,
            ]
            ww_series_full = ww_full.detach().cpu().numpy().astype(np.float32)
            ww_mask_full = ww_full_mask.detach().cpu().numpy().astype(np.float32)
            sample["ww_series"] = ww_series_full.astype(np.float32)
            sample["ww_obs_mask_full"] = ww_mask_full.astype(np.float32)

        samples.append(sample)

    return samples


def make_input_series_figure(
    samples: list[dict[str, Any]],
    history_length: int,
) -> Any:
    """Build figure with 4 subplots per window (cases, biomarkers, hosp, deaths).

    Each subplot shows:
    - 1 series line (or 3 lines for biomarkers)
    - Age ribbon at top (per-series for biomarkers)
    - History/horizon separator

    Args:
        samples: List of window samples from collect_window_samples
        history_length: Length of history window (for separator line)

    Returns:
        Matplotlib Figure
    """
    if not samples:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    n_samples = len(samples)
    n_series = 4  # cases, biomarkers, hosp, deaths

    fig, axes = plt.subplots(
        nrows=n_samples,
        ncols=n_series,
        figsize=(20, 3.5 * n_samples),
        sharex="col",
        squeeze=False,
    )

    colors = {
        "cases": "#1f77b4",  # blue
        "hosp": "#d62728",  # red
        "deaths": "#9467bd",  # purple
    }

    for row_idx, sample in enumerate(samples):
        # Get series lengths
        cases_series = sample["cases_series"]
        total_len = len(cases_series)
        horizon_length = total_len - history_length
        t = np.arange(total_len)

        # Column 0: Cases
        ax_cases = axes[row_idx, 0]
        _plot_single_series(
            ax=ax_cases,
            series=sample["cases_series"],
            age=sample["cases_age"],
            observed_mask_full=sample["cases_obs_mask_full"],
            history_length=history_length,
            horizon_length=horizon_length,
            t=t,
            color=colors["cases"],
            label="Cases",
        )
        ax_cases.set_ylabel("Cases (log1p per-100k)", fontsize=9)

        # Column 1: Biomarkers (3 variants)
        ax_bio = axes[row_idx, 1]
        biomarkers = sample.get("biomarkers", {})
        _plot_biomarkers(
            ax=ax_bio,
            biomarkers=biomarkers,
            ww_series=sample.get("ww_series"),
            ww_obs_mask_full=sample.get("ww_obs_mask_full"),
            history_length=history_length,
            horizon_length=horizon_length,
            t=t,
        )
        ax_bio.set_ylabel("Biomarkers (log1p per-100k)", fontsize=9)

        # Column 2: Hospitalizations
        ax_hosp = axes[row_idx, 2]
        _plot_single_series(
            ax=ax_hosp,
            series=sample["hosp_series"],
            age=sample["hosp_age"],
            observed_mask_full=sample["hosp_obs_mask_full"],
            history_length=history_length,
            horizon_length=horizon_length,
            t=t,
            color=colors["hosp"],
            label="Hosp",
        )
        ax_hosp.set_ylabel("Hosp (log1p per-100k)", fontsize=9)

        # Column 3: Deaths
        ax_deaths = axes[row_idx, 3]
        _plot_single_series(
            ax=ax_deaths,
            series=sample["deaths_series"],
            age=sample["deaths_age"],
            observed_mask_full=sample["deaths_obs_mask_full"],
            history_length=history_length,
            horizon_length=horizon_length,
            t=t,
            color=colors["deaths"],
            label="Deaths",
        )
        ax_deaths.set_ylabel("Deaths (log1p per-100k)", fontsize=9)

        # Add title to first column only
        node_label = sample.get("node_label", "")
        node_id = sample.get("node_id", None)
        window_start = sample.get("window_start", None)
        title_parts = []
        if node_label:
            title_parts.append(node_label)
        if node_id is not None:
            title_parts.append(f"id={node_id}")
        if window_start is not None and window_start >= 0:
            title_parts.append(f"start={window_start}")
        title = " | ".join(title_parts)
        ax_cases.set_title(title, fontsize=10, fontweight="semibold")

    # Set column titles
    axes[0, 0].set_title(
        f"{axes[0, 0].get_title()}\nCases", fontsize=10, fontweight="semibold"
    )
    axes[0, 1].set_title("Biomarkers\n(3 variants)", fontsize=10, fontweight="semibold")
    axes[0, 2].set_title("Hospitalizations", fontsize=10, fontweight="semibold")
    axes[0, 3].set_title("Deaths", fontsize=10, fontweight="semibold")

    # Add x-label to bottom row
    for col in range(n_series):
        axes[-1, col].set_xlabel("Time index (history → horizon)", fontsize=9)

    plt.tight_layout()
    return fig


def _plot_single_series(
    ax: Any,
    series: np.ndarray,
    age: np.ndarray,
    observed_mask_full: np.ndarray,
    history_length: int,
    horizon_length: int,
    t: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Plot a single series with age ribbon."""
    import matplotlib.pyplot as plt

    # Pad age to match series length (history only)
    age_padded = np.concatenate([age, np.full(horizon_length, np.nan)])

    # Plot age ribbon at top (5% of plot height)
    valid_mask = ~np.isnan(age_padded)
    valid_t = t[valid_mask]
    valid_age = age_padded[valid_mask]

    if len(valid_t) > 0:
        cmap = plt.get_cmap("RdYlGn_r")
        for ti, age_val in zip(valid_t, valid_age):
            ribbon_color = cmap(min(age_val, 1.0))
            ax.axvspan(
                ti - 0.5,
                ti + 0.5,
                ymin=0.95,
                ymax=1.0,
                color=ribbon_color,
                alpha=0.7,
                zorder=-1,
            )

    # Plot smoothed/interpolated series line
    ax.plot(t, series, color=color, linewidth=1.5, label=label)

    # Mark observed vs interpolated points using the mask.
    finite = np.isfinite(series)
    observed = (observed_mask_full > 0.5) & finite
    interpolated = (observed_mask_full <= 0.5) & finite

    if observed.any():
        ax.scatter(
            t[observed],
            series[observed],
            s=10,
            color=color,
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )
    if interpolated.any():
        ax.scatter(
            t[interpolated],
            series[interpolated],
            s=14,
            color="#ff7f0e",
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    # Draw history/horizon separator
    ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5)

    # Set y-limits with padding for ribbon
    if np.all(np.isnan(series)):
        return
    ymin, ymax = np.nanmin(series), np.nanmax(series)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        padding = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - padding, ymax + padding * 2)  # Extra padding for ribbon


def _plot_biomarkers(
    ax: Any,
    biomarkers: dict[str, dict[str, np.ndarray]],
    ww_series: np.ndarray | None,
    ww_obs_mask_full: np.ndarray | None,
    history_length: int,
    horizon_length: int,
    t: np.ndarray,
) -> None:
    """Plot biomarkers with 3 lines and 3 age ribbons."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not biomarkers:
        ax.text(
            0.5, 0.5, "No biomarkers", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Color palette for 3 variants
    palette = sns.color_palette("tab10", n_colors=len(biomarkers))

    for idx, (name, channels) in enumerate(biomarkers.items()):
        color = palette[idx]
        values = channels["value"]
        age = channels["age"]

        # Pad age to match full window length
        age_padded = np.concatenate([age, np.full(horizon_length, np.nan)])

        # Plot age ribbon for this variant (stacked vertically)
        # Each variant gets a portion of the top 15% of the plot
        ribbon_bottom = 0.95 - (idx * 0.05)
        ribbon_top = ribbon_bottom + 0.04

        valid_mask = ~np.isnan(age_padded)
        valid_t = t[valid_mask]
        valid_age = age_padded[valid_mask]

        if len(valid_t) > 0:
            cmap = plt.get_cmap("RdYlGn_r")
            for ti, age_val in zip(valid_t, valid_age):
                ribbon_color = cmap(min(age_val, 1.0))
                ax.axvspan(
                    ti - 0.5,
                    ti + 0.5,
                    ymin=ribbon_bottom,
                    ymax=ribbon_top,
                    color=ribbon_color,
                    alpha=0.7,
                    zorder=-1,
                )

        # Plot series line
        ax.plot(t[: len(values)], values, color=color, linewidth=1.5, label=name)

        # Mark observed vs interpolated points using the mask channel
        mask = channels["mask"]
        finite = np.isfinite(values)
        observed = (mask > 0.5) & finite
        interpolated = (mask <= 0.5) & finite

        if observed.any():
            ax.scatter(
                t[: len(values)][observed],
                values[observed],
                s=10,
                color=color,
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
        if interpolated.any():
            ax.scatter(
                t[: len(values)][interpolated],
                values[interpolated],
                s=14,
                color="#ff7f0e",
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )

    # Draw history/horizon separator
    ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5)

    # Overlay WW target/trajectory as the biomarker aggregate used in loss.
    if ww_series is not None and ww_obs_mask_full is not None:
        ax.plot(
            t, ww_series, color="black", linewidth=1.2, linestyle="--", label="WW mean"
        )
        finite = np.isfinite(ww_series)
        observed = (ww_obs_mask_full > 0.5) & finite
        interpolated = (ww_obs_mask_full <= 0.5) & finite
        if observed.any():
            ax.scatter(
                t[observed],
                ww_series[observed],
                s=10,
                color="black",
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
        if interpolated.any():
            ax.scatter(
                t[interpolated],
                ww_series[interpolated],
                s=14,
                color="#ff7f0e",
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )

    # Add legend
    ax.legend(loc="upper left", fontsize=8)

    # Set y-limits with padding for ribbons
    all_values = np.concatenate([ch["value"] for ch in biomarkers.values()])
    if ww_series is not None:
        ww_finite = ww_series[np.isfinite(ww_series)]
        if ww_finite.size > 0:
            all_values = np.concatenate([all_values, ww_finite])
    ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        padding = (ymax - ymin) * 0.15  # Extra padding for 3 ribbons
        ax.set_ylim(ymin - padding, ymax + padding)


def export_samples_summary(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Export summary statistics for collected samples to CSV."""
    rows = []
    for sample in samples:
        row: dict[str, Any] = {
            "node_id": sample["node_id"],
            "node_label": sample["node_label"],
            "window_start": sample["window_start"],
            "cases_mean": np.nanmean(sample["cases_series"]),
            "cases_std": np.nanstd(sample["cases_series"]),
            "hosp_mean": np.nanmean(sample["hosp_series"]),
            "hosp_std": np.nanstd(sample["hosp_series"]),
            "deaths_mean": np.nanmean(sample["deaths_series"]),
            "deaths_std": np.nanstd(sample["deaths_series"]),
        }

        # Add biomarker stats
        biomarkers = sample.get("biomarkers", {})
        for name, channels in biomarkers.items():
            row[f"{name}_mean"] = np.nanmean(channels["value"])
            row[f"{name}_std"] = np.nanstd(channels["value"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported samples summary to {output_path}")


def generate_input_series_plots(
    config_path: str,
    output_dir: str | Path | None = None,
    num_samples: int = 5,
    seed: int = 42,
    biomarker_source_only: bool = True,
) -> tuple[list, dict[str, Path]]:
    """Generate input series window plots from a training config.

    Args:
        config_path: Path to training config YAML
        output_dir: Output directory for plots
        num_samples: Number of samples to plot
        seed: Random seed for shuffling

    Returns:
        Tuple of (samples, exported_files)
    """
    setup_logging(logging.INFO)

    # Load config
    logger.info(f"Loading config from: {config_path}")
    config = EpiForecasterConfig.from_file(config_path)

    # Get all node IDs from dataset
    import xarray as xr

    from data.epi_dataset import EpiDataset

    zarr_path = Path(config.data.dataset_path).resolve()
    dataset = xr.open_zarr(zarr_path)
    all_nodes = list(range(dataset[REGION_COORD].size))
    biomarker_source_nodes: set[int] | None = None
    if biomarker_source_only and "edar_has_source" in dataset:
        source_da = dataset["edar_has_source"]
        if "run_id" in source_da.dims:
            source_da = source_da.isel(run_id=0)
        source_mask = source_da.values.astype(bool)
        biomarker_source_nodes = set(np.where(source_mask)[0].tolist())
    dataset.close()

    # Split nodes (train/val/test) - use a simple split for visualization
    rng = random.Random(seed)
    rng.shuffle(all_nodes)

    n_val = int(len(all_nodes) * config.training.val_split)
    n_test = int(len(all_nodes) * config.training.test_split)

    val_nodes = all_nodes[:n_val]
    train_nodes = all_nodes[n_val + n_test :]
    context_nodes = train_nodes + val_nodes

    if biomarker_source_nodes is not None:
        train_source_nodes = [n for n in train_nodes if n in biomarker_source_nodes]
        if train_source_nodes:
            logger.info(
                "Restricting input-series target sampling to biomarker source regions: "
                "%d -> %d train nodes",
                len(train_nodes),
                len(train_source_nodes),
            )
            train_nodes = train_source_nodes
        else:
            logger.warning(
                "No train nodes overlap with edar_has_source==1; using unfiltered train nodes."
            )

    logger.info(f"Total nodes: {len(all_nodes)}")
    logger.info(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}")

    # Create dataset with train nodes for fitting preprocessors
    epi_dataset = EpiDataset(
        config=config,
        target_nodes=train_nodes,
        context_nodes=context_nodes,
    )

    # Create dataloader
    loader = DataLoader(
        epi_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Collect samples
    logger.info(f"Collecting {num_samples} samples (ordered and shuffled)...")
    ordered_samples = collect_window_samples(
        loader=loader,
        n=num_samples,
        shuffle=False,
        seed=seed,
    )
    shuffled_samples = collect_window_samples(
        loader=loader,
        n=num_samples,
        shuffle=True,
        seed=seed,
    )

    # Setup output directory
    if output_dir is None:
        output_dir = Path("outputs/reports/input_series")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}

    # Generate ordered plots
    logger.info("Generating ordered window figure...")
    fig_ordered = make_input_series_figure(
        ordered_samples,
        history_length=config.model.history_length,
    )
    if fig_ordered:
        ordered_path = output_dir / "input_series_ordered.png"
        fig_ordered.savefig(ordered_path, dpi=200, bbox_inches="tight")
        plt.close(fig_ordered)
        exported_files["ordered"] = ordered_path
        logger.info(f"Saved ordered figure to {ordered_path}")

    # Generate shuffled plots
    logger.info("Generating shuffled window figure...")
    fig_shuffled = make_input_series_figure(
        shuffled_samples,
        history_length=config.model.history_length,
    )
    if fig_shuffled:
        shuffled_path = output_dir / "input_series_shuffled.png"
        fig_shuffled.savefig(shuffled_path, dpi=200, bbox_inches="tight")
        plt.close(fig_shuffled)
        exported_files["shuffled"] = shuffled_path
        logger.info(f"Saved shuffled figure to {shuffled_path}")

    # Export summary
    summary_path = output_dir / "samples_summary.csv"
    export_samples_summary(shuffled_samples, summary_path)
    exported_files["summary"] = summary_path

    return shuffled_samples, exported_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate input series window plots from EpiForecaster config"
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
        default="outputs/reports/input_series",
        help="Output directory for plots (default: outputs/reports/input_series)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to plot (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--include-all-targets",
        action="store_true",
        help=(
            "Disable biomarker-source filtering and sample target nodes from all regions "
            "(default behavior is biomarker source nodes only when edar_has_source exists)."
        ),
    )

    args = parser.parse_args()

    generate_input_series_plots(
        config_path=args.config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        biomarker_source_only=not args.include_all_targets,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
