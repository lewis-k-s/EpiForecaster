"""
Dataset interface for loading and configuring canonical EpiBatch objects.

This module provides the EpiDataset class which serves as the primary interface
for loading preprocessed datasets and applying variant-specific configurations.
It handles lazy loading, batching, and variant masking to support different
model configurations from a single canonical dataset.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.configs import ModelVariant

from .dataset_storage import DatasetStorage
from .ego_graph_dataset import GraphEgoDataset
from .epi_batch import EpiBatch


class EpiDataset(Dataset):
    """
    Canonical dataset that loads and configures EpiBatch objects.

    This class provides a PyTorch Dataset interface for loading preprocessed
    epidemiological data stored in Zarr format. It supports variant-specific
    configuration through masking of optional features like region embeddings
    and EDAR wastewater data.

    Args:
        zarr_path: Path to the Zarr dataset
        variant_config: Configuration for masking features (e.g., which
            data streams to use for a specific model variant)
        batch_size: Number of timepoints per batch (default: 1)
        shuffle_timepoints: Whether to shuffle temporal order (default: False)
        sequence_length: Length of input sequences (default: 1)
        forecast_horizon: Override forecast horizon from dataset (optional)
        ego_graph_params: Parameters forwarded to the ego-graph dataset factory
    """

    def __init__(
        self,
        zarr_path: Path,
        variant_config: ModelVariant,
        batch_size: int = 1,
        shuffle_timepoints: bool = False,
        sequence_length: int = 1,
        forecast_horizon: int | None = None,
    ):
        self.zarr_path = Path(zarr_path)
        self.variant_config = variant_config or {}
        self.batch_size = batch_size
        self.shuffle_timepoints = shuffle_timepoints
        self.sequence_length = sequence_length
        # Enable ego-graph view by default since trainer uses forward_ego_graph
        self.enable_ego_graph_view = True
        self._ego_graph_dataset: GraphEgoDataset | None = None
        self._ego_graph_signature: tuple[Any, ...] | None = None

        # Load dataset
        self.dataset_info = DatasetStorage.load_dataset(self.zarr_path)
        self.metadata = self.dataset_info["metadata"]
        self.arrays = self.dataset_info["arrays"]

        self._validate_dataset_compatibility()

        # Set dimensions
        self.num_timepoints = self.metadata["num_timepoints"]
        self.num_nodes = self.metadata["num_nodes"]
        self.num_edges = self.metadata["num_edges"]
        self.feature_dim = self.metadata["feature_dim"]
        self.forecast_horizon = forecast_horizon or self.metadata["forecast_horizon"]

        # Create timepoint indices
        self.timepoint_indices = np.arange(self.num_timepoints)
        if self.shuffle_timepoints:
            np.random.shuffle(self.timepoint_indices)

        # Pre-calculate batch indices
        self._create_batch_indices()

        print(f"Loaded dataset: {self.metadata['dataset_name']}")
        print(f"  - {self.num_timepoints} timepoints")
        print(f"  - {self.num_nodes} nodes, {self.num_edges} edges")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Forecast horizon: {self.forecast_horizon}")
        print(f"  - Variant config: {self.variant_config}")
        if self.enable_ego_graph_view:
            # Lazily initialize so we fail fast if parameters are invalid
            self.get_ego_graph_dataset()

    def _validate_dataset_compatibility(self):
        """Validate that dataset is compatible with current configuration."""
        # Check required arrays exist
        required_arrays = [
            "node_features",
            "edge_index",
            "target_sequences",
            "time_index",
        ]
        for array_name in required_arrays:
            if array_name not in self.arrays:
                raise ValueError(f"Dataset missing required array: {array_name}")

        # Validate array shapes
        expected_shapes = {
            "node_features": (
                self.metadata["num_timepoints"],
                self.metadata["num_nodes"],
                self.metadata["feature_dim"],
            ),
            "edge_index": (2, self.metadata["num_edges"]),
            "target_sequences": (
                self.metadata["num_timepoints"],
                self.metadata["num_nodes"],
                self.metadata["forecast_horizon"],
            ),
            "time_index": (self.metadata["num_timepoints"],),
        }

        for array_name, expected_shape in expected_shapes.items():
            actual_shape = self.arrays[array_name].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Dataset {array_name} shape mismatch: "
                    f"{actual_shape} != {expected_shape}"
                )

    def _create_batch_indices(self):
        """Create indices for batching timepoints."""
        if self.batch_size == 1:
            # Single timepoint batches
            self.batch_indices = [[i] for i in range(self.num_timepoints)]
        else:
            # Multi-timepoint batches
            num_batches = self.num_timepoints // self.batch_size
            self.batch_indices = [
                list(range(i * self.batch_size, (i + 1) * self.batch_size))
                for i in range(num_batches)
            ]

            # Handle remaining timepoints
            remaining = self.num_timepoints % self.batch_size
            if remaining > 0:
                self.batch_indices.append(
                    list(range(num_batches * self.batch_size, self.num_timepoints))
                )

        # Shuffle batch order if requested
        if self.shuffle_timepoints:
            np.random.shuffle(self.batch_indices)

    def __len__(self) -> int:
        """Return number of region-timepoint combinations."""
        return self.num_timepoints * self.num_nodes

    def __getitem__(self, idx: int) -> EpiBatch:
        """
        Load data for a single region with sequence history.

        Args:
            idx: Index representing (timepoint_idx * num_nodes + region_idx)

        Returns:
            EpiBatch for one region with:
            - node_features: [sequence_length, feature_dim]
            - target_sequences: [forecast_horizon]
            - time_index: [sequence_length]
        """
        # Convert idx to timepoint and region indices
        timepoint_idx = idx // self.num_nodes
        region_idx = idx % self.num_nodes

        # Validate indices
        if timepoint_idx >= self.num_timepoints:
            raise IndexError(f"Timepoint index {timepoint_idx} out of range")
        if region_idx >= self.num_nodes:
            raise IndexError(f"Region index {region_idx} out of range")

        # Load static data (same for all timepoints)
        edge_index = torch.from_numpy(self.arrays["edge_index"][:])

        # Load sequence data for this region
        if timepoint_idx >= self.sequence_length:
            # Have enough history for full sequence
            seq_start = timepoint_idx - self.sequence_length + 1
            node_features = torch.from_numpy(
                self.arrays["node_features"][seq_start:timepoint_idx+1, region_idx, :]
            )  # [sequence_length, feature_dim]
        else:
            # Handle early timepoints with zero padding
            padding_size = self.sequence_length - timepoint_idx - 1
            padding = torch.zeros(padding_size, self.feature_dim)
            available_data = torch.from_numpy(
                self.arrays["node_features"][:timepoint_idx+1, region_idx, :]
            )
            node_features = torch.cat([padding, available_data], dim=0)

        # Target sequences for this region: [forecast_horizon]
        target_sequences = torch.from_numpy(
            self.arrays["target_sequences"][timepoint_idx, region_idx, :]
        )  # [forecast_horizon]

        # Time indices for the sequence: [sequence_length]
        time_index = torch.arange(
            timepoint_idx - self.sequence_length + 1,
            timepoint_idx + 1,
            dtype=torch.long
        )
        time_index = torch.clamp(time_index, 0, self.num_timepoints - 1)

        # Get timestamp for primary timepoint
        timestamp_np = self.arrays["time_index"][timepoint_idx]
        timestamp = datetime.fromtimestamp(
            timestamp_np.astype("datetime64[s]").astype(int)
        )

        # Load optional data for this region and timepoint
        edge_attr = None
        if "edge_attr" in self.arrays and self.variant_config.mobility:
            # Use the current timepoint's edge features
            edge_attr = torch.from_numpy(
                self.arrays["edge_attr"][timepoint_idx, :, :]
            )  # [num_edges, edge_dim]

        region_embeddings = None
        if "region_embeddings" in self.arrays and self.variant_config.regions:
            # Get embedding for this specific region
            region_embeddings = torch.from_numpy(
                self.arrays["region_embeddings"][region_idx:region_idx+1, :]
            )  # [1, embed_dim]

        edar_features = None
        edar_attention_mask = None
        if "edar_features" in self.arrays and self.variant_config.biomarkers:
            # Use the current timepoint's EDAR features
            edar_features = torch.from_numpy(
                self.arrays["edar_features"][timepoint_idx]
            ).mean(dim=0)  # [edar_dim] - average across EDARs
            edar_attention_mask = torch.from_numpy(
                self.arrays["edar_attention_mask"][region_idx:region_idx+1, :]
            )  # [1, num_edars]

        # Create EpiBatch with num_nodes=1 for single region
        batch = EpiBatch(
            batch_id=f"{self.metadata['dataset_name']}_region{region_idx}_t{timepoint_idx}",
            timestamp=timestamp,
            num_nodes=1,  # Single region
            node_features=node_features,  # [sequence_length, feature_dim]
            edge_index=edge_index,  # Full graph structure
            edge_attr=edge_attr,
            time_index=time_index,  # [sequence_length]
            sequence_length=self.sequence_length,
            target_sequences=target_sequences,  # [forecast_horizon]
            region_embeddings=region_embeddings,  # [1, embed_dim] or None
            edar_features=edar_features,  # [edar_dim] or None
            edar_attention_mask=edar_attention_mask,  # [1, num_edars] or None
            metadata={
                "dataset_name": self.metadata["dataset_name"],
                "variant_config": asdict(self.variant_config),
                "preprocessing_config": self.metadata.get("preprocessing_config", {}),
                "timepoint_idx": timepoint_idx,
                "region_idx": region_idx,
            },
        )

        return batch

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Get the time range of the dataset."""
        time_array = self.arrays["time_index"][:]
        start_time = datetime.fromtimestamp(
            time_array[0].astype("datetime64[s]").astype(int)
        )
        end_time = datetime.fromtimestamp(
            time_array[-1].astype("datetime64[s]").astype(int)
        )
        return start_time, end_time

    def summary(self) -> dict[str, Any]:
        """Get summary of dataset information."""
        start_time, end_time = self.get_time_range()

        return {
            "dataset_name": self.metadata["dataset_name"],
            "path": str(self.zarr_path),
            "num_timepoints": self.num_timepoints,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "feature_dim": self.feature_dim,
            "forecast_horizon": self.forecast_horizon,
            "num_batches": len(self),
            "batch_size": self.batch_size,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_days": (end_time - start_time).days,
            },
            "variant_config": self.variant_config.as_dict(),
            "ego_graph_view_enabled": self.enable_ego_graph_view,
            "ego_graph_initialized": self._ego_graph_dataset is not None,
            "ego_graph_params": self._normalize_ego_graph_params(),
            "available_features": {
                "edge_attr": "edge_attr" in self.arrays,
                "region_embeddings": "region_embeddings" in self.arrays,
                "edar_data": "edar_features" in self.arrays,
            },
            "created_at": self.metadata.get("created_at"),
            "schema_version": self.metadata.get("schema_version"),
        }

    def get_ego_graph_dataset(self) -> dict[str, GraphEgoDataset]:
        """Create and return GraphEgoDataset from the loaded data."""
        if self._ego_graph_dataset is None:
            # Extract data for GraphEgoDataset
            # GraphEgoDataset expects: cases [num_nodes, num_timesteps]
            # Our data is: node_features [num_timesteps, num_nodes, feature_dim]
            cases = torch.from_numpy(self.arrays["node_features"][:, :, 0]).T  # Transpose to [num_nodes, num_timesteps]

            # Handle biomarkers (remaining features)
            if self.feature_dim > 1:
                biomarkers = torch.from_numpy(self.arrays["node_features"][:, :, 1:]).transpose(0, 1)  # [num_nodes, num_timesteps, biomarker_dim]
            else:
                # Create dummy biomarkers if not available
                biomarkers = torch.zeros(self.num_nodes, self.num_timepoints, 1)

            # Handle mobility data
            mobility = []
            if "edge_attr" in self.arrays and self.arrays["edge_attr"].shape[0] > 0:
                # Convert edge_attr to mobility matrices
                for t in range(self.num_timepoints):
                    mobility.append(torch.from_numpy(self.arrays["edge_attr"][t]))
            else:
                # Create identity mobility if not available
                for t in range(self.num_timepoints):
                    mobility.append(torch.eye(self.num_nodes))

            # Create GraphEgoDataset
            self._ego_graph_dataset = GraphEgoDataset(
                cases=cases,
                biomarkers=biomarkers,
                mobility=mobility,
                L=self.sequence_length,
                H=self.forecast_horizon,
                min_flow_threshold=10.0,
                max_neighbors=20,
                include_target_in_graph=True,
            )

        return {"train": self._ego_graph_dataset, "val": self._ego_graph_dataset}

    def get_dataset_info(self) -> dict[str, Any]:
        """Compatibility helper mirroring GraphEgoDataset API."""
        info = self.summary()
        if self._ego_graph_dataset is not None:
            info["ego_graph_dataset"] = self._ego_graph_dataset.get_dataset_info()
        return info

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"EpiDataset(name='{self.metadata['dataset_name']}', "
            f"timepoints={self.num_timepoints}, "
            f"nodes={self.num_nodes}, "
            f"batches={len(self)})"
        )
