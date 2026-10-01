"""Canonical native dataset construction and split input identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from data.native_source_sets import QueryEligibilityRecord
from models.configs import EpiForecasterConfig


def native_dataset(config: EpiForecasterConfig) -> NativeSupportDataset:
    """Use the same eligibility rules for split generation, training, and evaluation."""
    return NativeSupportDataset(
        native_root=config.native.root,
        real_with_id_path=config.data.dataset_path,
        config=NativeDatasetConfig(
            input_window_length=config.model.input_window_length,
            forecast_horizon=config.model.forecast_horizon,
            window_stride=config.data.window_stride,
            min_history_observed_fraction=config.data.min_history_observed_fraction,
            min_horizon_observed_fraction=config.data.min_horizon_observed_fraction,
            confirmation_guard=config.native.confirmation_guard,
            spatial_encoder=config.native.spatial_encoder,
            od_source=config.native.od_source,
            od_min_flow_raw=config.native.od_min_flow_raw,
            od_self_flows=config.native.od_self_flows,
            mobility_enabled=config.native.mobility_enabled,
            wastewater_enabled=config.native.wastewater_enabled,
            include_vaccination=config.native.include_vaccination,
            include_od_context=True,
        ),
    )


def query_inventory(dataset: NativeSupportDataset) -> list[QueryEligibilityRecord]:
    return [
        QueryEligibilityRecord(
            str(q.source_type),
            str(q.source_id),
            dataset.head_at(i),
            int(q.forecast_origin),
        )
        for i, q in enumerate(dataset.queries)
    ]


def _content_hash(path: Path) -> str:
    """Hash actual materialized bytes, including Zarr metadata and chunks."""
    digest = hashlib.sha256()
    files = (
        sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    )
    if not files:
        raise ValueError(f"Split input is empty: {path}")
    for file in files:
        relative = str(file.relative_to(path) if path.is_dir() else file.name)
        file_hash = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(block)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.digest())
        digest.update(b"\0")
    return digest.hexdigest()


def split_identity(config: EpiForecasterConfig, dataset: NativeSupportDataset) -> dict:
    """Bind a split to all processed inputs, eligibility, and assignment policy."""
    root = Path(config.native.root)
    inputs = {"clinical": Path(config.data.dataset_path)}
    for name in (
        "native_graph.zarr",
        "obs_admissions.zarr",
        "obs_deaths_abs_excess.zarr",
        "overlap_edges.parquet",
    ):
        inputs[name] = root / name
    if config.native.wastewater_enabled:
        for name in ("obs_wastewater.zarr", "edar_catchment_edges.parquet"):
            inputs[name] = root / name
    if config.native.mobility_enabled:
        name = (
            "od_mitma.zarr"
            if config.native.od_source == "mitma"
            else "od_municipal.zarr"
        )
        inputs[name] = root / name
    queries = [
        (q.source_type, q.source_id, q.head, q.origin) for q in query_inventory(dataset)
    ]
    return {
        "builder_version": 4,
        "builder_sha256": hashlib.sha256(
            b"".join(
                (Path(__file__).parent / name).read_bytes()
                for name in (
                    "build_native_source_sets.py",
                    "native_source_sets.py",
                    "native_dataset.py",
                    "native_inputs.py",
                )
            )
        ).hexdigest(),
        "inputs": {name: _content_hash(path) for name, path in inputs.items()},
        "eligibility": {
            "input_window_length": config.model.input_window_length,
            "forecast_horizon": config.model.forecast_horizon,
            "window_stride": config.data.window_stride,
            "min_history_observed_fraction": config.data.min_history_observed_fraction,
            "min_horizon_observed_fraction": config.data.min_horizon_observed_fraction,
            "confirmation_guard": config.native.confirmation_guard,
            "wastewater_enabled": config.native.wastewater_enabled,
        },
        "policy": {
            "seed": config.training.split_seed,
            "temporal_validation_fraction": config.training.val_split,
            "num_folds": config.native.num_folds,
            "validation_fraction": config.native.validation_fraction,
            "test_fraction": config.native.test_fraction,
        },
        "date_start": str(dataset.date_grid[0]),
        "date_end": str(dataset.date_grid[-1]),
        "query_inventory_sha256": hashlib.sha256(
            json.dumps(queries, separators=(",", ":")).encode()
        ).hexdigest(),
        "query_inventory_count": len(queries),
    }
