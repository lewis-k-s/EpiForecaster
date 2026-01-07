import json
from pathlib import Path
from typing import Any

import torch
import zarr
from torch.utils.data import Dataset

from utils.logging import suppress_zarr_warnings


class RegionGraphDataset(Dataset):
    """Dataset for region graph data stored in zarr format.

    This dataset provides access to region features, adjacency, flows, and IDs for training region embeddings.
    """

    def __init__(
        self, zarr_path: Path, normalize_features: bool = True, device: str = "auto"
    ):
        suppress_zarr_warnings()
        self.zarr_path = zarr_path
        self.normalize_features = normalize_features
        self.device = torch.device(
            device
            if device != "auto"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load zarr root
        self.root = zarr.open(str(zarr_path), mode="r")

        metadata_raw = self.root.attrs.get("metadata")
        if isinstance(metadata_raw, str):
            self.metadata = json.loads(metadata_raw)
        elif isinstance(metadata_raw, dict):
            self.metadata = metadata_raw
        else:
            self.metadata = {}
        self.flow_source = self.metadata.get("flow_source")

        # Store array references for lazy loading
        self._features = self.root["features"] if "features" in self.root else None
        self._edge_index = (
            self.root["edge_index"] if "edge_index" in self.root else None
        )
        self._flows = self.root["flows"] if "flows" in self.root else None
        self._region_ids = (
            self.root["region_ids"] if "region_ids" in self.root else None
        )

        # Validate required arrays exist
        if self._features is None:
            raise ValueError("'features' not found in zarr file")
        if self._edge_index is None:
            raise ValueError("'edge_index' not found in zarr file")

        # Store dataset info
        self.num_regions = self._features.shape[0]
        self.feature_dim = self._features.shape[1]

        # Precompute normalization stats if needed
        if self.normalize_features:
            features_array = self._features[:]
            self.feature_mean = torch.from_numpy(
                features_array.mean(axis=0, keepdims=True)
            )
            self.feature_std = torch.from_numpy(
                features_array.std(axis=0, keepdims=True)
            )

    def __len__(self) -> int:
        """Returns the number of regions."""
        return self.num_regions

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get data for a single region.

        Returns:
            Dictionary containing:
            - features: torch.Tensor of shape [feature_dim]
            - region_id: str
        """
        # Load features for this region
        features = torch.from_numpy(self._features[idx])

        if self.normalize_features:
            features = (features - self.feature_mean.squeeze()) / (
                self.feature_std.squeeze() + 1e-6
            )

        # Get region ID
        if self._region_ids is not None:
            region_id = str(self._region_ids[idx])
        else:
            region_id = f"region_{idx}"

        return {
            "features": features,
            "region_id": region_id,
            "idx": idx,
        }

    def get_all_features(self) -> torch.Tensor:
        """Load all features at once (for training)."""
        features = torch.from_numpy(self._features[:])

        if self.normalize_features:
            features = (features - self.feature_mean) / (self.feature_std + 1e-6)

        return features.to(self.device)

    def get_edge_index(self) -> torch.Tensor:
        """Load edge index (converted to long)."""
        edge_index = torch.from_numpy(self._edge_index[:]).long()
        return edge_index.to(self.device)

    def get_flow_matrix(self) -> torch.Tensor | None:
        """Load flow matrix if available."""
        if self._flows is None:
            return None
        return torch.from_numpy(self._flows[:]).to(self.device)

    def get_region_ids(self) -> list[str]:
        """Get all region IDs."""
        if self._region_ids is not None:
            return [str(rid) for rid in self._region_ids[:]]
        return [f"region_{i}" for i in range(self.num_regions)]
