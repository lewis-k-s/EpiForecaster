#!/usr/bin/env python
"""Persist unit-correct mobility edge-selection and weighting diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.dataset_factory import split_nodes_by_ratio
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

RAW_THRESHOLDS = np.asarray(
    [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0],
    dtype=np.float64,
)


def new_scope_accumulator(name: str, indices: np.ndarray) -> dict[str, object]:
    """Create a bounded accumulator for one square regional graph scope."""
    return {
        "scope": name,
        "indices": np.asarray(indices, dtype=np.int64),
        "date_count": 0,
        "positive_edge_count": 0,
        "selected_edge_count": 0,
        "clipped_edge_count": 0,
        "raw_flow_sum": 0.0,
        "clipped_raw_flow_sum": 0.0,
        "unclipped_log_weight_sum": 0.0,
        "effective_log_weight_sum": 0.0,
        "non_finite_weight_count": 0,
        "negative_weight_count": 0,
        "degrees_unthresholded": [],
        "degrees_effective": [],
        "threshold_counts": np.zeros(len(RAW_THRESHOLDS), dtype=np.int64),
    }


def _clear_diagonal(mask: np.ndarray) -> None:
    diagonal = np.arange(mask.shape[-1])
    mask[..., diagonal, diagonal] = False


def update_scope(
    accumulator: dict[str, object],
    stored_log_flow: np.ndarray,
    *,
    min_flow_raw: float,
    weight_clip_max: float,
) -> None:
    """Update diagnostics from a time chunk in canonical log1p flow space."""
    indices = np.asarray(accumulator["indices"], dtype=np.int64)
    scoped = stored_log_flow[:, indices][:, :, indices].astype(np.float32, copy=False)
    finite = np.isfinite(scoped)
    positive = finite & (scoped > 0.0)
    _clear_diagonal(positive)
    selected = positive & (scoped >= np.log1p(min_flow_raw))
    clipped = selected & (scoped > weight_clip_max)

    weights = np.where(
        selected,
        np.clip(
            np.nan_to_num(
                scoped,
                nan=0.0,
                posinf=weight_clip_max,
                neginf=0.0,
            ),
            0.0,
            weight_clip_max,
        ),
        0.0,
    )
    raw_flow = np.zeros_like(scoped, dtype=np.float64)
    raw_flow[positive] = np.expm1(scoped[positive].astype(np.float64))

    accumulator["date_count"] = int(accumulator["date_count"]) + scoped.shape[0]
    accumulator["positive_edge_count"] = int(accumulator["positive_edge_count"]) + int(
        positive.sum()
    )
    accumulator["selected_edge_count"] = int(accumulator["selected_edge_count"]) + int(
        selected.sum()
    )
    accumulator["clipped_edge_count"] = int(accumulator["clipped_edge_count"]) + int(
        clipped.sum()
    )
    accumulator["raw_flow_sum"] = float(accumulator["raw_flow_sum"]) + float(
        raw_flow[positive].sum()
    )
    accumulator["clipped_raw_flow_sum"] = float(
        accumulator["clipped_raw_flow_sum"]
    ) + float(raw_flow[clipped].sum())
    accumulator["unclipped_log_weight_sum"] = float(
        accumulator["unclipped_log_weight_sum"]
    ) + float(scoped[selected].sum(dtype=np.float64))
    accumulator["effective_log_weight_sum"] = float(
        accumulator["effective_log_weight_sum"]
    ) + float(weights[selected].sum(dtype=np.float64))
    accumulator["non_finite_weight_count"] = int(
        accumulator["non_finite_weight_count"]
    ) + int((~np.isfinite(weights)).sum())
    accumulator["negative_weight_count"] = int(
        accumulator["negative_weight_count"]
    ) + int((weights < 0.0).sum())
    accumulator["degrees_unthresholded"].extend(
        positive.sum(axis=1).reshape(-1).tolist()
    )
    accumulator["degrees_effective"].extend(selected.sum(axis=1).reshape(-1).tolist())

    threshold_counts = np.asarray(accumulator["threshold_counts"])
    for index, raw_threshold in enumerate(RAW_THRESHOLDS):
        threshold_mask = positive & (scoped >= np.log1p(raw_threshold))
        threshold_counts[index] += int(threshold_mask.sum())


def finalize_scope(accumulator: dict[str, object]) -> dict[str, object]:
    """Convert a scope accumulator into one machine-readable record."""
    node_count = len(np.asarray(accumulator["indices"]))
    date_count = int(accumulator["date_count"])
    possible_edges = date_count * node_count * max(node_count - 1, 0)
    positive_count = int(accumulator["positive_edge_count"])
    selected_count = int(accumulator["selected_edge_count"])
    clipped_count = int(accumulator["clipped_edge_count"])
    raw_flow_sum = float(accumulator["raw_flow_sum"])
    unclipped_weight_sum = float(accumulator["unclipped_log_weight_sum"])
    degrees_unthresholded = np.asarray(
        accumulator["degrees_unthresholded"],
        dtype=np.float64,
    )
    degrees_effective = np.asarray(
        accumulator["degrees_effective"],
        dtype=np.float64,
    )

    def percentile(values: np.ndarray, quantile: float) -> float:
        return float(np.quantile(values, quantile)) if values.size else 0.0

    return {
        "scope": str(accumulator["scope"]),
        "node_count": node_count,
        "date_count": date_count,
        "positive_edge_count": positive_count,
        "selected_edge_count": selected_count,
        "positive_edge_density": (
            positive_count / possible_edges if possible_edges else 0.0
        ),
        "selected_edge_density": (
            selected_count / possible_edges if possible_edges else 0.0
        ),
        "positive_observation_retained_fraction": (
            selected_count / positive_count if positive_count else 0.0
        ),
        "mean_incoming_degree_unthresholded": (
            float(degrees_unthresholded.mean()) if degrees_unthresholded.size else 0.0
        ),
        "mean_incoming_degree_effective": (
            float(degrees_effective.mean()) if degrees_effective.size else 0.0
        ),
        "incoming_degree_p10": percentile(degrees_effective, 0.10),
        "incoming_degree_p50": percentile(degrees_effective, 0.50),
        "incoming_degree_p90": percentile(degrees_effective, 0.90),
        "clipped_edge_fraction": (
            clipped_count / selected_count if selected_count else 0.0
        ),
        "clipped_raw_flow_fraction": (
            float(accumulator["clipped_raw_flow_sum"]) / raw_flow_sum
            if raw_flow_sum
            else 0.0
        ),
        "log_weight_reduction_fraction": (
            (unclipped_weight_sum - float(accumulator["effective_log_weight_sum"]))
            / unclipped_weight_sum
            if unclipped_weight_sum
            else 0.0
        ),
        "non_finite_weight_count": int(accumulator["non_finite_weight_count"]),
        "negative_weight_count": int(accumulator["negative_weight_count"]),
    }


def _daily_metrics(
    stored_log_flow: np.ndarray,
    dates: np.ndarray,
    train_indices: np.ndarray,
    *,
    min_flow_raw: float,
    weight_clip_max: float,
) -> list[dict[str, object]]:
    scoped = stored_log_flow[:, train_indices][:, :, train_indices].astype(
        np.float32,
        copy=False,
    )
    positive = np.isfinite(scoped) & (scoped > 0.0)
    _clear_diagonal(positive)
    selected = positive & (scoped >= np.log1p(min_flow_raw))
    clipped = selected & (scoped > weight_clip_max)
    rows = []
    for index, date in enumerate(dates):
        positive_count = int(positive[index].sum())
        selected_count = int(selected[index].sum())
        rows.append(
            {
                "date": np.datetime_as_string(date, unit="D"),
                "positive_edge_count": positive_count,
                "selected_edge_count": selected_count,
                "positive_observation_retained_fraction": (
                    selected_count / positive_count if positive_count else 0.0
                ),
                "mean_incoming_degree_unthresholded": float(
                    positive[index].sum(axis=0).mean()
                ),
                "mean_incoming_degree_effective": float(
                    selected[index].sum(axis=0).mean()
                ),
                "clipped_edge_fraction": (
                    float(clipped[index].sum()) / selected_count
                    if selected_count
                    else 0.0
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--date-chunk-size", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_chunk_size <= 0:
        raise ValueError("--date-chunk-size must be positive")

    config = EpiForecasterConfig.load(args.config)
    dataset_path = args.dataset or Path(config.data.dataset_path)
    run_id = args.run_id or config.data.run_id
    train_nodes, validation_nodes, test_nodes = split_nodes_by_ratio(
        config,
        dataset_path=dataset_path,
        run_id=run_id,
    )
    train_indices = np.asarray(train_nodes, dtype=np.int64)

    dataset = xr.open_zarr(dataset_path)
    try:
        selected_dataset = (
            dataset.sel(run_id=run_id, drop=True)
            if "run_id" in dataset.dims
            else dataset
        )
        mobility = selected_dataset["mobility"]
        spatial_dims = [dim for dim in mobility.dims if dim != TEMPORAL_COORD]
        if len(spatial_dims) != 2:
            raise ValueError(
                f"Expected mobility with two spatial dimensions, got {mobility.dims}"
            )
        mobility = mobility.transpose(TEMPORAL_COORD, *spatial_dims)
        dates = np.asarray(selected_dataset[TEMPORAL_COORD].values)
        if "mobility_time_mask" in selected_dataset:
            time_mask = np.asarray(
                selected_dataset["mobility_time_mask"].values,
                dtype=bool,
            ).reshape(-1)
            mobility = mobility.isel({TEMPORAL_COORD: np.flatnonzero(time_mask)})
            dates = dates[time_mask]

        region_count = selected_dataset[REGION_COORD].size
        valid_targets = (
            np.asarray(selected_dataset["valid_targets"].values, dtype=bool)
            if "valid_targets" in selected_dataset
            else np.ones(region_count, dtype=bool)
        )
        scopes = {
            "All regions": new_scope_accumulator(
                "All regions",
                np.arange(region_count),
            ),
            "Valid targets": new_scope_accumulator(
                "Valid targets",
                np.flatnonzero(valid_targets),
            ),
            "Training context": new_scope_accumulator(
                "Training context",
                train_indices,
            ),
        }
        daily_rows: list[dict[str, object]] = []
        for start in range(0, len(dates), args.date_chunk_size):
            stop = min(start + args.date_chunk_size, len(dates))
            chunk = np.asarray(
                mobility.isel({TEMPORAL_COORD: slice(start, stop)}).values,
                dtype=np.float32,
            )
            for accumulator in scopes.values():
                update_scope(
                    accumulator,
                    chunk,
                    min_flow_raw=config.data.mobility_min_flow_raw,
                    weight_clip_max=config.data.mobility_weight_clip_max,
                )
            daily_rows.extend(
                _daily_metrics(
                    chunk,
                    dates[start:stop],
                    train_indices,
                    min_flow_raw=config.data.mobility_min_flow_raw,
                    weight_clip_max=config.data.mobility_weight_clip_max,
                )
            )
    finally:
        dataset.close()

    scope_metrics = pd.DataFrame(
        [finalize_scope(accumulator) for accumulator in scopes.values()]
    )
    daily_metrics = pd.DataFrame(daily_rows)
    threshold_rows = []
    for scope_name, accumulator in scopes.items():
        positive_count = int(accumulator["positive_edge_count"])
        for raw_threshold, count in zip(
            RAW_THRESHOLDS,
            np.asarray(accumulator["threshold_counts"]),
            strict=True,
        ):
            threshold_rows.append(
                {
                    "scope": scope_name,
                    "raw_flow_threshold": raw_threshold,
                    "stored_log1p_threshold": float(np.log1p(raw_threshold)),
                    "retained_edge_count": int(count),
                    "positive_observation_retained_fraction": (
                        int(count) / positive_count if positive_count else 0.0
                    ),
                }
            )
    threshold_sensitivity = pd.DataFrame(threshold_rows)

    summary = {
        "schema_version": 1,
        "config": str(args.config),
        "dataset": str(dataset_path),
        "run_id": run_id,
        "split_strategy": config.training.node_split_strategy,
        "split_seed": config.training.split_seed,
        "split_sizes": {
            "train": len(train_nodes),
            "validation": len(validation_nodes),
            "test": len(test_nodes),
        },
        "valid_target_count": int(valid_targets.sum()),
        "total_region_count": int(region_count),
        "valid_mobility_dates": len(dates),
        "stored_value_units": "log1p(raw mobility flow)",
        "mobility_min_flow_raw": float(config.data.mobility_min_flow_raw),
        "mobility_min_flow_stored_log1p": float(
            np.log1p(config.data.mobility_min_flow_raw)
        ),
        "mobility_weight_clip_max": float(config.data.mobility_weight_clip_max),
        "scope_metrics": scope_metrics.to_dict(orient="records"),
        "mask_sources": {
            "gnn_weights": "masked_log1p_mobility_weights",
            "k_hop_topology": "mobility_edge_mask",
            "clinical_context": "mobility_edge_mask",
            "imported_risk": "masked_log1p_mobility_weights",
            "gpu_adjacency": "masked_log1p_mobility_weights",
            "self_loops": "separate_gpu_identity",
            "static_graph_clinical_context": "selected_static_adjacency",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope_metrics.to_csv(args.output_dir / "scope_metrics.csv", index=False)
    daily_metrics.to_csv(
        args.output_dir / "daily_training_metrics.csv",
        index=False,
    )
    threshold_sensitivity.to_csv(
        args.output_dir / "threshold_sensitivity.csv",
        index=False,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(scope_metrics.to_string(index=False))
    print(f"Wrote mobility QC artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
