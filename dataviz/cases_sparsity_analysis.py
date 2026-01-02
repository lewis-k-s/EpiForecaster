"""
Cases sparsity analysis for raw and canonical datasets.

Computes valid overlapping window counts per node using the same history/horizon
logic as the training dataset, and summarizes neighborhood sparsity using the
mobility graph at a chosen timestep within each window.
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
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from data.preprocess.processors.cases_processor import CasesProcessor
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (time, region, feature), adding trailing dim if needed."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def load_raw_cases(preprocess_config_path: Path) -> xr.DataArray:
    """Load raw cases via the preprocessor loader and return a (time, region) DataArray."""
    pcfg = PreprocessingConfig.from_file(str(preprocess_config_path))
    processor = CasesProcessor(pcfg)
    cases_df = processor._load_cases_data(pcfg.cases_file)
    cases_df = cases_df[
        (cases_df["date"] >= pcfg.start_date) & (cases_df["date"] <= pcfg.end_date)
    ]
    if cases_df.empty:
        raise ValueError("No raw cases found in configured date range")

    cases_pivot = cases_df.pivot_table(
        index="date", columns=REGION_COORD, values="cases", aggfunc="sum"
    )
    date_range = pd.date_range(start=pcfg.start_date, end=pcfg.end_date, freq="D")
    cases_pivot = cases_pivot.reindex(date_range)
    cases_pivot.index.name = TEMPORAL_COORD
    cases_pivot.columns.name = REGION_COORD

    return xr.DataArray(
        cases_pivot.values,
        dims=[TEMPORAL_COORD, REGION_COORD],
        coords={TEMPORAL_COORD: cases_pivot.index, REGION_COORD: cases_pivot.columns},
        name="cases",
    )


def compute_valid_window_mask(
    cases_da: xr.DataArray,
    history_length: int,
    horizon: int,
    window_stride: int,
    missing_permit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute window starts and validity mask per node.

    Returns:
        starts: 1D array of window start indices
        valid_mask: 2D boolean array (num_windows, num_nodes)
        region_ids: 1D array of region IDs in order
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
        return starts, valid_mask, cases_da[REGION_COORD].values

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
    target_counts = cumsum[history_length + horizon :] - cumsum[history_length:-horizon]

    starts = np.arange(0, T - seg + 1, window_stride, dtype=np.int64)
    history_counts = history_counts[starts]
    target_counts = target_counts[starts]

    history_threshold = max(0, history_length - missing_permit)
    history_ok = history_counts >= history_threshold
    target_ok = target_counts >= horizon
    valid_mask = history_ok & target_ok

    return starts, valid_mask, cases_da[REGION_COORD].values


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
        f"Mobility data must include ('origin','destination') or two '{REGION_COORD}' dims"
    )


def plot_valid_window_hist(
    raw_counts: pd.Series,
    canon_counts: pd.Series,
    output_dir: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.histplot(raw_counts, bins=60, ax=axes[0])
    axes[0].set_title(f"Raw cases valid windows per node{title_suffix}")
    axes[0].set_xlabel("Valid window count")
    axes[0].set_ylabel("Nodes")

    sns.histplot(canon_counts, bins=60, ax=axes[1])
    axes[1].set_title(f"Canonical cases valid windows per node{title_suffix}")
    axes[1].set_xlabel("Valid window count")
    axes[1].set_ylabel("Nodes")

    output_path = output_dir / "cases_valid_window_hist.png"
    fig.savefig(output_path, dpi=200)
    logger.info("Saved valid window histogram to %s", output_path)


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

    output_path = output_dir / "cases_neighbor_sparsity_hist.png"
    fig.savefig(output_path, dpi=200)
    logger.info("Saved neighbor sparsity histogram to %s", output_path)


def write_summary_tables(
    raw_counts: pd.Series,
    canon_counts: pd.Series,
    mean_neighbors: pd.Series,
    zero_fraction: pd.Series,
    output_dir: Path,
) -> None:
    raw_df = raw_counts.rename("valid_window_count").to_frame()
    raw_df.index.name = "region_id"
    raw_path = output_dir / "cases_raw_valid_window_counts.csv"
    raw_df.to_csv(raw_path)

    canon_df = pd.DataFrame(
        {
            "valid_window_count": canon_counts,
            "mean_neighbor_count": mean_neighbors,
            "zero_neighbor_fraction": zero_fraction,
        }
    )
    canon_df.index.name = "region_id"
    canon_path = output_dir / "cases_canon_neighbor_summary.csv"
    canon_df.to_csv(canon_path)

    logger.info("Saved raw window counts to %s", raw_path)
    logger.info("Saved canonical neighbor summary to %s", canon_path)

    no_valid = int((canon_counts == 0).sum())
    logger.info("Canonical nodes with 0 valid windows: %d", no_valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cases sparsity analysis")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train_epifor_full.yaml"),
        help="Training config path for canonical dataset",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=Path("configs/preprocess_full.yaml"),
        help="Preprocessing config path for raw cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/cases_sparsity"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--neighbor-timestep",
        choices=["start", "end"],
        default="end",
        help="Which timestep within the window to compute neighborhoods",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include self-edges when counting neighbors",
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

    cfg = EpiForecasterConfig.from_file(str(args.train_config))

    raw_cases = load_raw_cases(args.preprocess_config)
    canon_ds = EpiDataset.load_canonical_dataset(Path(cfg.data.dataset_path))
    canon_cases = (
        canon_ds.cases_normalized if "cases_normalized" in canon_ds else canon_ds.cases
    )

    starts_raw, valid_raw, raw_regions = compute_valid_window_mask(
        raw_cases,
        history_length=cfg.model.history_length,
        horizon=cfg.model.forecast_horizon,
        window_stride=cfg.model.window_stride,
        missing_permit=cfg.model.missing_permit,
    )
    starts_canon, valid_canon, canon_regions = compute_valid_window_mask(
        canon_cases,
        history_length=cfg.model.history_length,
        horizon=cfg.model.forecast_horizon,
        window_stride=cfg.model.window_stride,
        missing_permit=cfg.model.missing_permit,
    )

    raw_counts = pd.Series(valid_raw.sum(axis=0), index=raw_regions)
    canon_counts = pd.Series(valid_canon.sum(axis=0), index=canon_regions)

    title_suffix = (
        f" (L={cfg.model.history_length}, H={cfg.model.forecast_horizon}, "
        f"stride={cfg.model.window_stride}, permit={cfg.model.missing_permit})"
    )
    plot_valid_window_hist(raw_counts, canon_counts, output_dir, title_suffix)

    mobility_threshold = float(cfg.data.mobility_threshold)
    mobility = resolve_mobility_array(canon_ds["mobility"])
    mean_neighbors, zero_fraction = compute_neighbor_sparsity(
        mobility,
        starts_canon,
        valid_canon,
        history_length=cfg.model.history_length,
        mobility_threshold=mobility_threshold,
        neighbor_timestep=args.neighbor_timestep,
        include_self=args.include_self,
    )

    mean_series = pd.Series(mean_neighbors, index=canon_regions)
    zero_series = pd.Series(zero_fraction, index=canon_regions)
    neighbor_suffix = (
        f" (timestep={args.neighbor_timestep}, threshold={mobility_threshold}, "
        f"include_self={args.include_self})"
    )
    plot_neighbor_sparsity(mean_series, zero_series, output_dir, neighbor_suffix)
    write_summary_tables(raw_counts, canon_counts, mean_series, zero_series, output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
