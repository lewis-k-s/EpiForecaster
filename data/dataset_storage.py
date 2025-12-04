"""
Zarr-based storage for canonical EpiForecaster datasets.

This module provides efficient storage and loading of preprocessed epidemiological
datasets using Zarr format. It handles the persistence of EpiBatch objects and
associated metadata, supporting chunked access patterns suitable for temporal
graph data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BytesCodec
from zarr.errors import ContainsGroupError

from .epi_batch import EpiBatch


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_compression_codecs(
    compression: str = "blosc", compression_opts: dict | None = None
) -> list:
    """Create a list of codecs compatible with zarr v3."""
    # For now, just return bytes codec to avoid compression issues
    return [BytesCodec()]


def _assign_dimensions(zarr_array, dims: tuple[str, ...]) -> None:
    """Attach xarray-compatible metadata for array dimensions."""
    dim_names = list(dims)
    zarr_array.attrs["dimension_names"] = dim_names
    zarr_array.attrs["_ARRAY_DIMENSIONS"] = dim_names


def _normalize_timestamp(ts: datetime) -> datetime:
    """Return a timezone-naive UTC timestamp for consistent serialization."""
    if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


class DatasetStorage:
    """
    Handles loading and saving canonical EpiForecaster datasets using Zarr.

    This class provides a high-level interface for persisting and retrieving
    preprocessed epidemiological datasets. It uses Zarr for efficient chunked
    storage of large tensor data and JSON for metadata.

    Storage schema:
    dataset.zarr/
    ├── node_features/          # [time, num_nodes, feature_dim]
    ├── edge_index/            # [2, num_edges] (static)
    ├── edge_attr/             # [time, num_edges, edge_dim] (optional)
    ├── target_sequences/      # [time, num_nodes, forecast_horizon]
    ├── region_embeddings/     # [num_nodes, embed_dim] (optional)
    ├── edar_features/         # [time, num_edars, edar_dim] (optional)
    ├── edar_attention_mask/   # [num_nodes, num_edars] (optional)
    ├── time_index/            # [time] datetime coordinates
    └── metadata/              # Dataset information and preprocessing config
    """

    # Zarr chunk sizes optimized for temporal access patterns
    DEFAULT_CHUNKS = {
        "node_features": (100, 100, 64),  # (time, nodes, features)
        "edge_attr": (100, 1000, 32),  # (time, edges, edge_features)
        "target_sequences": (100, 100, 7),  # (time, nodes, horizon)
        "edar_features": (100, 50, 16),  # (time, edar_sites, features)
    }

    @staticmethod
    def save_dataset(
        dataset: list[EpiBatch],
        path: Path,
        dataset_name: str,
        compression: str = "blosc",
        compression_opts: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a dataset of EpiBatch objects to Zarr format.

        Args:
            dataset: List of EpiBatch objects to save
            path: Directory path for the Zarr dataset
            dataset_name: Human-readable name for the dataset
            compression: Compression algorithm (default: "blosc")
            compression_opts: Compression options
        """
        if not dataset:
            raise ValueError("Cannot save empty dataset")

        path = Path(path)
        if path.suffix != ".zarr":
            path = path.with_suffix(".zarr")

        # Create Zarr group
        try:
            root = zarr.group(str(path))
        except ContainsGroupError as e:
            raise ValueError(f"Dataset already exists at {path}") from e

        # Extract dimensions from first batch
        first_batch = dataset[0]
        num_timepoints = len(dataset)
        num_nodes = first_batch.num_nodes
        feature_dim = first_batch.feature_dim
        num_edges = first_batch.num_edges
        forecast_horizon = first_batch.forecast_horizon

        # Initialize data arrays
        arrays = {}

        # Core data (always present)
        arrays["node_features"] = root.zeros(
            name="node_features",
            shape=(num_timepoints, num_nodes, feature_dim),
            chunks=DatasetStorage.DEFAULT_CHUNKS["node_features"],
            dtype=np.float32,
            codecs=get_compression_codecs(compression, compression_opts),
            dimension_names=("time", "region", "feature"),
        )
        _assign_dimensions(arrays["node_features"], ("time", "region", "feature"))

        arrays["edge_index"] = root.zeros(
            name="edge_index",
            shape=(2, num_edges),
            dtype=np.int64,
            # No compression for small static data
            dimension_names=("coordinate", "edge"),
        )
        _assign_dimensions(arrays["edge_index"], ("coordinate", "edge"))

        arrays["target_sequences"] = root.zeros(
            name="target_sequences",
            shape=(num_timepoints, num_nodes, forecast_horizon),
            chunks=DatasetStorage.DEFAULT_CHUNKS["target_sequences"],
            dtype=np.float32,
            codecs=get_compression_codecs(compression, compression_opts),
            dimension_names=("time", "region", "horizon"),
        )
        _assign_dimensions(arrays["target_sequences"], ("time", "region", "horizon"))

        # Optional data (check if present in first batch)
        if first_batch.edge_attr is not None:
            edge_attr_dim = first_batch.edge_attr.shape[1]
            arrays["edge_attr"] = root.zeros(
                name="edge_attr",
                shape=(num_timepoints, num_edges, edge_attr_dim),
                chunks=DatasetStorage.DEFAULT_CHUNKS["edge_attr"],
                dtype=np.float32,
                codecs=get_compression_codecs(compression, compression_opts),
                dimension_names=("time", "edge", "edge_feature"),
            )
            _assign_dimensions(arrays["edge_attr"], ("time", "edge", "edge_feature"))

        if first_batch.region_embeddings is not None:
            embed_dim = first_batch.region_embeddings.shape[1]
            arrays["region_embeddings"] = root.zeros(
                name="region_embeddings",
                shape=(num_nodes, embed_dim),
                chunks=(num_nodes, embed_dim),
                dtype=np.float32,
                codecs=get_compression_codecs(compression, compression_opts),
                dimension_names=("region", "embedding"),
            )
            _assign_dimensions(arrays["region_embeddings"], ("region", "embedding"))

        if first_batch.edar_features is not None:
            num_edars = first_batch.edar_features.shape[0]
            edar_dim = first_batch.edar_features.shape[1]
            arrays["edar_features"] = root.zeros(
                name="edar_features",
                shape=(num_timepoints, num_edars, edar_dim),
                chunks=DatasetStorage.DEFAULT_CHUNKS["edar_features"],
                dtype=np.float32,
                codecs=get_compression_codecs(compression, compression_opts),
                dimension_names=("time", "edar_site", "feature"),
            )
            _assign_dimensions(arrays["edar_features"], ("time", "edar_site", "feature"))

        if first_batch.edar_attention_mask is not None:
            arrays["edar_attention_mask"] = root.zeros(
                name="edar_attention_mask",
                shape=(num_nodes, num_edars),
                chunks=(num_nodes, num_edars),
                dtype=np.float32,
                dimension_names=("region", "edar_site"),
            )
            _assign_dimensions(arrays["edar_attention_mask"], ("region", "edar_site"))

        # Time index stored as datetime64 for xarray compatibility
        arrays["time_index"] = root.zeros(
            name="time_index",
            shape=(num_timepoints,),
            dtype=np.dtype("datetime64[ns]"),
            dimension_names=("time",),
        )
        _assign_dimensions(arrays["time_index"], ("time",))

        # Fill arrays with data
        for i, batch in enumerate(dataset):
            # Validate consistency across batches
            if batch.num_nodes != num_nodes:
                raise ValueError(
                    f"Inconsistent num_nodes at batch {i}: "
                    f"{batch.num_nodes} != {num_nodes}"
                )
            if batch.feature_dim != feature_dim:
                raise ValueError(
                    f"Inconsistent feature_dim at batch {i}: "
                    f"{batch.feature_dim} != {feature_dim}"
                )
            if batch.num_edges != num_edges:
                raise ValueError(
                    f"Inconsistent num_edges at batch {i}: "
                    f"{batch.num_edges} != {num_edges}"
                )
            if batch.forecast_horizon != forecast_horizon:
                raise ValueError(
                    f"Inconsistent forecast_horizon at batch {i}: "
                    f"{batch.forecast_horizon} != {forecast_horizon}"
                )

            # Store core data
            arrays["node_features"][i] = batch.node_features.cpu().numpy()
            arrays["target_sequences"][i] = batch.target_sequences.cpu().numpy()
            normalized_ts = _normalize_timestamp(batch.timestamp)
            arrays["time_index"][i] = np.datetime64(normalized_ts.isoformat())

            # Store optional data if present
            if batch.edge_attr is not None:
                arrays["edge_attr"][i] = batch.edge_attr.cpu().numpy()

        # Store static data (same for all batches)
        arrays["edge_index"][:] = first_batch.edge_index.cpu().numpy()

        if first_batch.region_embeddings is not None:
            arrays["region_embeddings"][:] = first_batch.region_embeddings.cpu().numpy()

        if (
            first_batch.edar_features is not None
            and first_batch.edar_attention_mask is not None
        ):
            arrays["edar_attention_mask"][:] = (
                first_batch.edar_attention_mask.cpu().numpy()
            )

        # Store metadata
        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "num_timepoints": num_timepoints,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "feature_dim": feature_dim,
            "forecast_horizon": forecast_horizon,
            "has_edge_attr": first_batch.edge_attr is not None,
            "has_region_embeddings": first_batch.region_embeddings is not None,
            "has_edar_data": first_batch.edar_features is not None,
            "time_range": {
                "start": _normalize_timestamp(dataset[0].timestamp).isoformat(),
                "end": _normalize_timestamp(dataset[-1].timestamp).isoformat(),
            },
            "preprocessing_config": first_batch.metadata.get(
                "preprocessing_config", {}
            ),
            "schema_version": "1.0",
        }

        root.attrs["metadata"] = json.dumps(metadata, indent=2, cls=DateTimeEncoder)
        zarr.consolidate_metadata(str(path))

        print(f"Dataset saved to {path}")
        print(f"  - {num_timepoints} timepoints")
        print(f"  - {num_nodes} nodes, {num_edges} edges")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Forecast horizon: {forecast_horizon}")

    @staticmethod
    def load_dataset(path: Path) -> dict[str, Any]:
        """
        Load dataset metadata and array handles from Zarr storage.

        Args:
            path: Path to the Zarr dataset

        Returns:
            Dictionary containing dataset metadata and array references
        """
        path = Path(path)
        if path.suffix != ".zarr":
            path = path.with_suffix(".zarr")

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        # Open Zarr group
        root = zarr.open(str(path), mode="r")

        # Load metadata
        metadata_json = root.attrs.get("metadata", "{}")
        metadata = json.loads(metadata_json)

        # Create array references
        arrays = {
            "node_features": root["node_features"],
            "edge_index": root["edge_index"],
            "target_sequences": root["target_sequences"],
            "time_index": root["time_index"],
        }

        # Load optional arrays if present
        if "edge_attr" in root:
            arrays["edge_attr"] = root["edge_attr"]
        if "region_embeddings" in root:
            arrays["region_embeddings"] = root["region_embeddings"]
        if "edar_features" in root:
            arrays["edar_features"] = root["edar_features"]
        if "edar_attention_mask" in root:
            arrays["edar_attention_mask"] = root["edar_attention_mask"]

        return {"metadata": metadata, "arrays": arrays, "path": path}

    @staticmethod
    def create_dataset_index(data_dir: Path) -> dict[str, dict[str, Any]]:
        """
        Create an index of all available datasets in a directory.

        Args:
            data_dir: Directory containing Zarr datasets

        Returns:
            Dictionary mapping dataset names to metadata
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return {}

        index = {}
        for zarr_path in data_dir.glob("*.zarr"):
            try:
                dataset_info = DatasetStorage.load_dataset(zarr_path)
                metadata = dataset_info["metadata"]
                dataset_name = metadata.get("dataset_name", zarr_path.stem)

                index[dataset_name] = {
                    "path": str(zarr_path),
                    "num_timepoints": metadata["num_timepoints"],
                    "num_nodes": metadata["num_nodes"],
                    "num_edges": metadata["num_edges"],
                    "feature_dim": metadata["feature_dim"],
                    "forecast_horizon": metadata["forecast_horizon"],
                    "has_edge_attr": metadata["has_edge_attr"],
                    "has_region_embeddings": metadata["has_region_embeddings"],
                    "has_edar_data": metadata["has_edar_data"],
                    "time_range": metadata["time_range"],
                    "created_at": metadata["created_at"],
                }
            except Exception as e:
                print(f"Warning: Failed to load dataset {zarr_path}: {e}")

        return index

    @staticmethod
    def validate_dataset(path: Path) -> dict[str, Any]:
        """
        Validate dataset integrity and return validation report.

        Args:
            path: Path to the Zarr dataset

        Returns:
            Validation report with any issues found
        """
        try:
            dataset_info = DatasetStorage.load_dataset(path)
            metadata = dataset_info["metadata"]
            arrays = dataset_info["arrays"]

            issues = []

            # Check required arrays
            required_arrays = [
                "node_features",
                "edge_index",
                "target_sequences",
                "time_index",
            ]
            for array_name in required_arrays:
                if array_name not in arrays:
                    issues.append(f"Missing required array: {array_name}")

            # Check array shapes match metadata
            if "node_features" in arrays:
                actual_shape = arrays["node_features"].shape
                expected_shape = (
                    metadata["num_timepoints"],
                    metadata["num_nodes"],
                    metadata["feature_dim"],
                )
                if actual_shape != expected_shape:
                    issues.append(
                        f"node_features shape mismatch: {actual_shape} != {expected_shape}"
                    )

            # Check time range consistency
            if "time_index" in arrays:
                time_array = arrays["time_index"][:]
                actual_time_range = DatasetStorage._format_time_range_from_array(
                    time_array
                )
                expected_range = metadata.get("time_range")
                if expected_range and (
                    actual_time_range.get("start") != expected_range.get("start")
                    or actual_time_range.get("end") != expected_range.get("end")
                ):
                    issues.append(
                        f"time_range mismatch: {actual_time_range} != {metadata['time_range']}"
                    )

            # Check for NaN values
            for array_name, array in arrays.items():
                if array.dtype.kind in ["f", "i"]:  # Numeric arrays
                    if np.isnan(array[:]).any():
                        issues.append(f"NaN values found in {array_name}")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "metadata": metadata,
                "validation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Failed to load dataset: {str(e)}"],
                "metadata": None,
                "validation_timestamp": datetime.now().isoformat(),
            }

    @staticmethod
    def _format_time_range_from_array(time_array: np.ndarray) -> dict[str, str]:
        """Convert stored time index values to ISO-8601 strings."""
        if time_array.size == 0:
            return {"start": "", "end": ""}

        if np.issubdtype(time_array.dtype, np.datetime64):
            start_iso = np.datetime_as_string(time_array.min(), unit="s")
            end_iso = np.datetime_as_string(time_array.max(), unit="s")
        else:
            start_iso = datetime.utcfromtimestamp(float(time_array.min())).isoformat()
            end_iso = datetime.utcfromtimestamp(float(time_array.max())).isoformat()
        return {"start": start_iso, "end": end_iso}
