"""
Neighborhood trace density analysis for windowed case series.

Computes, per window start, the fraction of incoming neighbors whose case traces
have <= k missing values in the (history + horizon) window. Aggregates across
valid target nodes and plots mean with std-dev bands for each k.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (time, region, feature), adding trailing dim if needed."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def resolve_mobility_array(mobility_da: xr.DataArray | xr.Dataset) -> np.ndarray:
    """Return mobility array in (time, origin, destination) order."""
    if isinstance(mobility_da, xr.Dataset):
        if "mobility" not in mobility_da:
            raise ValueError("Mobility dataset missing 'mobility' variable")
        mobility_da = mobility_da["mobility"]

    dims = list(mobility_da.dims)
    time_dim = TEMPORAL_COORD if TEMPORAL_COORD in dims else None
    if time_dim is None:
        raise ValueError(f"Mobility data missing {TEMPORAL_COORD} dimension")

    if "origin" in dims and "destination" in dims:
        ordered = mobility_da.transpose(time_dim, "origin", "destination")
        return ordered.values

    if dims.count(REGION_COORD) == 2:
        ordered = mobility_da.transpose(time_dim, REGION_COORD, REGION_COORD)
        return ordered.values

    raise ValueError(
        "Mobility data must include ('origin','destination') or two "
        f"'{REGION_COORD}' dims"
    )


def compute_valid_window_mask(
    cases_da: xr.DataArray,
    history_length: int,
    horizon: int,
    window_stride: int,
    missing_permit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute window starts and validity mask per node.

    Returns:
        starts: 1D array of window start indices
        valid_mask: 2D boolean array (num_windows, num_nodes)
    """
    if TEMPORAL_COORD not in cases_da.dims or REGION_COORD not in cases_da.dims:
        raise ValueError(
            f"Cases data must include {TEMPORAL_COORD} and {REGION_COORD} dims"
        )

    other_dims = [d for d in cases_da.dims if d not in (TEMPORAL_COORD, REGION_COORD)]
    cases_da = cases_da.transpose(TEMPORAL_COORD, REGION_COORD, *other_dims)
    cases_np = _ensure_3d(cases_da.values)
    if cases_np.ndim != 3:
        raise ValueError(f"Expected cases array with 2 or 3 dims, got {cases_np.shape}")

    T = cases_np.shape[0]
    seg = history_length + horizon
    if T < seg:
        starts = np.array([], dtype=np.int64)
        valid_mask = np.zeros((0, cases_np.shape[1]), dtype=bool)
        return starts, valid_mask

    valid = np.isfinite(cases_np).all(axis=2)
    valid_int = valid.astype(np.int32)

    cumsum = np.concatenate(
        [
            np.zeros((1, valid_int.shape[1]), dtype=np.int32),
            np.cumsum(valid_int, axis=0),
        ],
        axis=0,
    )

    history_counts = cumsum[history_length:] - cumsum[:-history_length]

    starts = np.arange(0, T - seg + 1, window_stride, dtype=np.int64)
    history_counts = history_counts[starts]

    history_threshold = max(0, history_length - missing_permit)
    history_ok = history_counts >= history_threshold
    valid_mask = history_ok

    return starts, valid_mask


def compute_missing_counts(
    cases_da: xr.DataArray,
    window_len: int,
    window_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return window starts and missing counts per node for each window."""
    other_dims = [d for d in cases_da.dims if d not in (TEMPORAL_COORD, REGION_COORD)]
    cases_da = cases_da.transpose(TEMPORAL_COORD, REGION_COORD, *other_dims)
    cases_np = _ensure_3d(cases_da.values)
    if cases_np.ndim != 3:
        raise ValueError(f"Expected cases array with 2 or 3 dims, got {cases_np.shape}")

    T = cases_np.shape[0]
    if T < window_len:
        starts = np.array([], dtype=np.int64)
        missing_counts = np.zeros((0, cases_np.shape[1]), dtype=np.int32)
        return starts, missing_counts

    valid = np.isfinite(cases_np).all(axis=2)
    missing = (~valid).astype(np.int32)

    cumsum = np.concatenate(
        [np.zeros((1, missing.shape[1]), dtype=np.int32), np.cumsum(missing, axis=0)],
        axis=0,
    )

    counts = cumsum[window_len:] - cumsum[:-window_len]
    starts = np.arange(0, T - window_len + 1, window_stride, dtype=np.int64)
    counts = counts[starts]

    return starts, counts


def split_nodes(
    num_nodes: int, val_split: float, test_split: float, seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Split nodes into train/val/test lists using a fixed RNG seed."""
    train_split = 1 - val_split - test_split
    all_nodes = np.arange(num_nodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_nodes)

    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * val_split)

    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]

    return list(train_nodes), list(val_nodes), list(test_nodes)


def plot_density_traces(
    time_index: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: list[int],
    output_path: Path,
    title: str,
) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    for i, k in enumerate(thresholds):
        ax.plot(time_index, mean[:, i], label=f"<= {k} missing")
        ax.fill_between(
            time_index,
            mean[:, i] - std[:, i],
            mean[:, i] + std[:, i],
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Window start")
    ax.set_ylabel("Neighborhood density")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=9, frameon=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_neighbor_sparsity(
    mobility: np.ndarray,
    starts: np.ndarray,
    valid_mask: np.ndarray,
    history_length: int,
    mobility_threshold: float,
    neighbor_timestep: str,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbor sparsity summaries for valid windows.

    Returns:
        mean_neighbor_counts: Mean neighbor count per node across valid windows
        zero_neighbor_fraction: Fraction of valid windows with zero neighbors per node
    """
    if neighbor_timestep not in {"start", "end"}:
        raise ValueError("neighbor_timestep must be 'start' or 'end'")

    if mobility.ndim != 3:
        raise ValueError(
            f"Expected mobility (time, origin, dest), got {mobility.shape}"
        )

    offset = 0 if neighbor_timestep == "start" else max(0, history_length - 1)
    times = starts + offset
    time_mask = times < mobility.shape[0]
    times = times[time_mask]
    valid_mask = valid_mask[time_mask]

    num_windows = len(times)
    num_nodes = mobility.shape[1]
    neighbor_counts = np.full((num_windows, num_nodes), np.nan, dtype=np.float32)

    for i, t in enumerate(times):
        inflow = mobility[t]
        neighbors = inflow >= mobility_threshold
        if not include_self:
            np.fill_diagonal(neighbors, False)
        neighbor_counts[i] = neighbors.sum(axis=0)

    neighbor_counts = np.where(valid_mask, neighbor_counts, np.nan)
    mean_counts = np.nanmean(neighbor_counts, axis=0)
    zero_frac = np.nanmean(neighbor_counts == 0, axis=0)
    return mean_counts, zero_frac


def plot_neighbor_sparsity(
    mean_counts: pd.Series,
    zero_fraction: pd.Series,
    output_dir: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.histplot(mean_counts.dropna(), bins=60, ax=axes[0])
    axes[0].set_title(f"Mean neighbor count per node{title_suffix}")
    axes[0].set_xlabel("Mean neighbor count")
    axes[0].set_ylabel("Nodes")

    sns.histplot(zero_fraction.dropna(), bins=60, ax=axes[1])
    axes[1].set_title(f"Zero-neighbor window fraction per node{title_suffix}")
    axes[1].set_xlabel("Fraction of valid windows")
    axes[1].set_ylabel("Nodes")

    output_path = output_dir / "neighborhood_sparsity_hist.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved neighbor sparsity histogram to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot neighborhood trace density across window starts."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Target node split to analyze.",
    )
    parser.add_argument(
        "--neighbor-timestep",
        default="end",
        choices=["start", "end"],
        help="Mobility snapshot used for neighbors.",
    )
    parser.add_argument(
        "--neighbor-missing-max",
        type=int,
        default=None,
        help="Maximum missing values threshold for neighbors (k in 1..k).",
    )
    parser.add_argument(
        "--neighbor-missing-step",
        type=int,
        default=1,
        help="Step size for missing thresholds (default: 1).",
    )
    parser.add_argument(
        "--mobility-threshold",
        type=float,
        default=None,
        help="Minimum incoming flow to include neighbor (default: config value).",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include target node as a neighbor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataviz"),
    )
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    config = EpiForecasterConfig.from_file(str(args.config))
    dataset = EpiDataset.load_canonical_dataset(Path(config.data.dataset_path))

    cases_da = dataset["cases"]
    mobility = resolve_mobility_array(dataset["mobility"])

    num_nodes = dataset[REGION_COORD].size
    train_nodes, val_nodes, test_nodes = split_nodes(
        num_nodes, config.training.val_split, config.training.test_split
    )

    if args.split == "train":
        target_nodes = train_nodes
    elif args.split == "val":
        target_nodes = val_nodes
    elif args.split == "test":
        target_nodes = test_nodes
    else:
        target_nodes = list(range(num_nodes))

    history_len = int(config.model.history_length)
    horizon = int(config.model.forecast_horizon)
    window_stride = int(config.data.window_stride)
    missing_permit = int(config.data.missing_permit)
    window_len = history_len + horizon

    starts, valid_mask = compute_valid_window_mask(
        cases_da, history_len, horizon, window_stride, missing_permit
    )
    count_starts, missing_counts = compute_missing_counts(
        cases_da, window_len, window_stride
    )

    if not np.array_equal(starts, count_starts):
        raise ValueError("Window starts mismatch between validity and missing counts")

    if starts.size == 0:
        raise ValueError("No valid window starts found.")

    max_missing = args.neighbor_missing_max
    if max_missing is None:
        max_missing = max(1, int(0.5 * window_len))

    thresholds = list(range(1, max_missing + 1, args.neighbor_missing_step))

    mobility_threshold = (
        float(args.mobility_threshold)
        if args.mobility_threshold is not None
        else float(config.data.mobility_threshold)
    )

    # Save original starts and valid_mask before filtering for density computation
    starts_orig = starts.copy()
    valid_mask_orig = valid_mask.copy()

    offset = 0 if args.neighbor_timestep == "start" else max(0, history_len - 1)
    times = starts + offset
    time_mask = times < mobility.shape[0]
    times = times[time_mask]
    valid_mask = valid_mask[time_mask]
    missing_counts = missing_counts[time_mask]
    starts = starts[time_mask]

    num_windows = len(starts)
    mean_density = np.full((num_windows, len(thresholds)), np.nan, dtype=np.float32)
    std_density = np.full((num_windows, len(thresholds)), np.nan, dtype=np.float32)

    target_nodes_arr = np.asarray(target_nodes, dtype=np.int64)

    for i, t in enumerate(times):
        valid_targets = target_nodes_arr[valid_mask[i, target_nodes_arr]]
        if valid_targets.size == 0:
            continue

        inflow = mobility[t]  # (origin, dest)
        dens = np.full((valid_targets.size, len(thresholds)), np.nan, dtype=np.float32)

        for j, node in enumerate(valid_targets):
            neighbors = inflow[:, node] >= mobility_threshold
            if not args.include_self:
                neighbors[node] = False
            if not neighbors.any():
                continue

            node_missing = missing_counts[i, neighbors]
            for k_idx, k in enumerate(thresholds):
                dens[j, k_idx] = np.mean(node_missing <= k)

        mean_density[i] = np.nanmean(dens, axis=0)
        std_density[i] = np.nanstd(dens, axis=0)

    if TEMPORAL_COORD in dataset.coords:
        time_index = dataset[TEMPORAL_COORD].values[starts]
    else:
        time_index = starts

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or (
        f"Neighborhood trace density ({args.split} split, window={window_len})"
    )
    output_path = output_dir / f"neighborhood_trace_density_{args.split}.png"

    plot_density_traces(
        time_index, mean_density, std_density, thresholds, output_path, title
    )

    logger.info("Saved plot to %s", output_path)

    # Compute and plot neighbor sparsity histograms
    mean_neighbors, zero_fraction = compute_neighbor_sparsity(
        mobility,
        starts_orig,
        valid_mask_orig,
        history_length=history_len,
        mobility_threshold=mobility_threshold,
        neighbor_timestep=args.neighbor_timestep,
        include_self=args.include_self,
    )
    region_ids = dataset[REGION_COORD].values
    mean_series = pd.Series(mean_neighbors, index=region_ids)
    zero_series = pd.Series(zero_fraction, index=region_ids)
    plot_neighbor_sparsity(mean_series, zero_series, output_dir, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
