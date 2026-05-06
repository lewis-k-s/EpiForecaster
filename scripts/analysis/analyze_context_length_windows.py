#!/usr/bin/env python
"""Analyze valid forecasting windows as input context length varies.

This diagnostic quantifies how the fixed real-data calendar and sparse
observation masks constrain the number of supervised forecasting examples.
It mirrors the window-start and missing-permit logic used by ``EpiDataset``
without constructing graph batches or running training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from data.dataset_factory import split_nodes_by_ratio
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

DEFAULT_CONTEXT_LENGTHS = (
    "14 28 42 60 84 112 140 168 196 224 252 280 308 336 364 392 420"
)
WW_COMPONENTS = ("edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4")


@dataclass(frozen=True)
class SplitCounts:
    context_length: int
    split: str
    raw_stride_windows: int
    valid_global_windows: int
    valid_node_windows: int
    target_nodes: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(",", " ").split()]
    if not values:
        raise ValueError("At least one context length is required")
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"Context lengths must be positive: {invalid}")
    return values


def _select_run_id(dataset: xr.Dataset, run_id: str) -> xr.Dataset:
    if "run_id" in dataset.dims:
        return dataset.sel(run_id=run_id)
    return dataset


def _time_region_values(dataset: xr.Dataset, var_name: str) -> np.ndarray | None:
    if var_name not in dataset:
        return None

    data_array = dataset[var_name]
    if "run_id" in data_array.dims:
        data_array = data_array.squeeze("run_id", drop=True)
    data_array = data_array.transpose(TEMPORAL_COORD, REGION_COORD)
    return np.asarray(data_array.values)


def _target_mask(dataset: xr.Dataset, var_name: str) -> np.ndarray | None:
    values = _time_region_values(dataset, var_name)
    if values is None:
        return None

    mask_name = f"{var_name}_mask"
    mask_values = _time_region_values(dataset, mask_name)
    if mask_values is None:
        observed = np.isfinite(values)
    else:
        observed = mask_values > 0

    return (observed & np.isfinite(values) & (values >= 0.0)).astype(bool)


def _wastewater_mask(dataset: xr.Dataset) -> np.ndarray | None:
    component_masks = [
        mask
        for var_name in WW_COMPONENTS
        if (mask := _target_mask(dataset, var_name)) is not None
    ]
    if not component_masks:
        return None
    return np.any(np.stack(component_masks, axis=0), axis=0)


def _valid_targets(dataset: xr.Dataset, config: EpiForecasterConfig) -> list[int]:
    num_nodes = int(dataset.sizes[REGION_COORD])
    if not config.data.use_valid_targets or "valid_targets" not in dataset:
        return list(range(num_nodes))

    valid_targets = dataset["valid_targets"]
    if "run_id" in valid_targets.dims:
        valid_targets = valid_targets.squeeze("run_id", drop=True)
    valid_mask = np.asarray(valid_targets.values).astype(bool)
    return np.flatnonzero(valid_mask).astype(int).tolist()


def _count_valid_windows(
    *,
    mask_by_target: dict[str, np.ndarray],
    target_nodes: list[int],
    context_length: int,
    forecast_horizon: int,
    window_stride: int,
    max_lag: int,
    missing_permit_map: dict[str, dict[str, int]],
    total_time_steps: int,
) -> tuple[int, int, int]:
    segment_length = context_length + forecast_horizon
    if total_time_steps < segment_length:
        return 0, 0, 0

    starts = np.asarray(
        list(range(max_lag, total_time_steps - segment_length + 1, window_stride)),
        dtype=np.int64,
    )
    if starts.size == 0:
        return 0, 0, 0

    num_nodes = next(iter(mask_by_target.values())).shape[1]
    valid_mask = np.zeros((len(starts), num_nodes), dtype=bool)

    for target_name, target_mask in mask_by_target.items():
        observed = target_mask.astype(np.int32)
        cumsum = np.concatenate(
            [
                np.zeros((1, num_nodes), dtype=np.int32),
                np.cumsum(observed, axis=0),
            ],
            axis=0,
        )

        history_counts = cumsum[context_length:] - cumsum[:-context_length]
        target_counts = (
            cumsum[context_length + forecast_horizon :]
            - cumsum[context_length:-forecast_horizon]
        )
        history_counts = history_counts[starts]
        target_counts = target_counts[starts]

        input_permit = int(missing_permit_map["input"].get(target_name, 0))
        horizon_permit = int(missing_permit_map["horizon"].get(target_name, 0))
        history_threshold = max(0, context_length - input_permit)
        target_threshold = max(0, forecast_horizon - horizon_permit)
        valid_mask |= (history_counts >= history_threshold) & (
            target_counts >= target_threshold
        )

    split_valid = valid_mask[:, target_nodes]
    valid_global_windows = int(split_valid.any(axis=1).sum())
    valid_node_windows = int(split_valid.sum())
    return int(starts.size), valid_global_windows, valid_node_windows


def compute_counts(
    *,
    config: EpiForecasterConfig,
    context_lengths: list[int],
) -> list[SplitCounts]:
    dataset = _select_run_id(xr.open_zarr(config.data.dataset_path), config.data.run_id)
    valid_targets = _valid_targets(dataset, config)

    if config.training.split_strategy == "node":
        train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)
    elif config.training.split_strategy == "time":
        train_nodes = val_nodes = test_nodes = valid_targets
    else:
        raise ValueError(
            f"Unsupported split_strategy: {config.training.split_strategy}"
        )

    mask_by_target: dict[str, np.ndarray] = {}
    for target_name, var_name in {
        "cases": "cases",
        "hospitalizations": "hospitalizations",
        "deaths": "deaths",
    }.items():
        mask = _target_mask(dataset, var_name)
        if mask is not None:
            mask_by_target[target_name] = mask

    wastewater_mask = _wastewater_mask(dataset)
    if wastewater_mask is not None:
        mask_by_target["wastewater"] = wastewater_mask

    if not mask_by_target:
        raise ValueError("No target masks found in dataset")

    total_time_steps = int(dataset.sizes[TEMPORAL_COORD])
    forecast_horizon = int(config.model.forecast_horizon)
    window_stride = int(config.data.window_stride)
    max_lag = max(config.data.mobility_lags, default=0)
    missing_permit_map = config.data.resolve_missing_permit_map()

    split_nodes = {
        "train": train_nodes,
        "val": val_nodes,
        "test": test_nodes,
    }
    counts: list[SplitCounts] = []
    for context_length in context_lengths:
        for split, nodes in split_nodes.items():
            raw_windows, global_windows, node_windows = _count_valid_windows(
                mask_by_target=mask_by_target,
                target_nodes=nodes,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                window_stride=window_stride,
                max_lag=max_lag,
                missing_permit_map=missing_permit_map,
                total_time_steps=total_time_steps,
            )
            counts.append(
                SplitCounts(
                    context_length=context_length,
                    split=split,
                    raw_stride_windows=raw_windows,
                    valid_global_windows=global_windows,
                    valid_node_windows=node_windows,
                    target_nodes=len(nodes),
                )
            )
    return counts


def write_counts_csv(counts: list[SplitCounts], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "context_length",
                "split",
                "raw_stride_windows",
                "valid_global_windows",
                "valid_node_windows",
                "target_nodes",
            ],
        )
        writer.writeheader()
        for row in counts:
            writer.writerow(row.__dict__)


def plot_counts(counts: list[SplitCounts], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    split_colors = {
        "train": "#2A6F97",
        "val": "#D9822B",
        "test": "#7A4EA3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    for split in ("train", "val", "test"):
        rows = [row for row in counts if row.split == split]
        rows.sort(key=lambda row: row.context_length)
        x = [row.context_length for row in rows]
        y_samples = [row.valid_node_windows for row in rows]
        y_samples_positive = [value if value > 0 else np.nan for value in y_samples]
        y_windows = [row.valid_global_windows for row in rows]

        color = split_colors[split]
        axes[0].scatter(x, y_samples_positive, label=split, color=color, s=34)
        axes[0].plot(x, y_samples_positive, color=color, linewidth=1.3, alpha=0.75)
        axes[1].scatter(x, y_windows, label=split, color=color, s=34)
        axes[1].plot(x, y_windows, color=color, linewidth=1.3, alpha=0.75)

    axes[0].set_title("Valid node-window samples")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    axes[1].set_title("Valid stride windows")
    axes[1].set_ylabel("Count")

    zero_contexts = [
        row.context_length for row in counts if row.valid_node_windows == 0
    ]
    if zero_contexts:
        zero_at = min(zero_contexts)
        for axis in axes:
            axis.axvline(
                zero_at,
                color="#666666",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
        axes[0].text(
            zero_at,
            0.95,
            "zero support",
            rotation=90,
            va="top",
            ha="right",
            color="#4A4A4A",
            transform=axes[0].get_xaxis_transform(),
        )

    for axis in axes:
        axis.set_xlabel("Historical context length (days)")
        axis.grid(True, color="#D8D8D8", linewidth=0.8, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(title="Split", frameon=False)
    fig.suptitle("Longer historical context reduces supervised real-data support")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze valid EpiDataset windows across context lengths."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/production_only/train_epifor_mn5_full.yaml"),
        help="Training config to inspect.",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default=DEFAULT_CONTEXT_LENGTHS,
        help="Space- or comma-separated context lengths to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/reports/context_length_window_counts.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("tex/EpiForecaster/plots/context_length_window_sparsity.png"),
        help="Scatter plot output path.",
    )
    args = parser.parse_args()

    config = EpiForecasterConfig.load(args.config)
    context_lengths = parse_int_list(args.context_lengths)
    counts = compute_counts(config=config, context_lengths=context_lengths)
    write_counts_csv(counts, args.output_csv)
    plot_counts(counts, args.output_png)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote plot: {args.output_png}")


if __name__ == "__main__":
    main()
