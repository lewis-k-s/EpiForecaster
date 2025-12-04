"""
Canonical batch representation for epidemiological forecasting.

This module defines the EpiBatch dataclass, which serves as the universal
batch representation across all model variants and training pipelines in
EpiForecaster. It provides a consistent interface for handling epidemiological
data including case counts, mobility patterns, wastewater biomarkers, and
regional embeddings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class EpiBatch:
    """
    Canonical batch representation for all epidemiological forecasting.

    This dataclass provides a unified format for handling epidemiological data
    across different model variants. All trainers and models should expect
    EpiBatch objects as input, making it easy to switch between variants
    without changing model implementations.

    Attributes:
        batch_id: Unique identifier for this batch
        timestamp: When this batch was created or processed
        num_nodes: Number of geographical regions/municipalities
        node_features: Feature matrix [num_nodes, feature_dim] containing
            case data and mobility statistics
        edge_index: Graph connectivity [2, num_edges] for municipality network
        edge_attr: Optional temporal edge features [num_edges, edge_dim]
        time_index: Temporal indices [batch_size] for this batch
        sequence_length: Length of temporal sequences (default: 1)
        target_sequences: Target case forecasts [batch_size, forecast_horizon]
        region_embeddings: Optional spatial embeddings [num_nodes, embed_dim]
        edar_features: Optional wastewater biomarker features [num_edars, edar_dim]
        edar_attention_mask: Optional bipartite connectivity [num_nodes, num_edars]

        # New fields for three-layer EpiForecaster architecture
        cases_hist: Historical case sequences [batch_size, seq_len, num_nodes]
        biomarkers_hist: Historical biomarker sequences [batch_size, seq_len, num_nodes, biomarker_dim]
        mobility_hist: Historical mobility matrices [batch_size, seq_len, num_nodes, num_nodes]
        region_ids: Region indices for batch samples [batch_size]

        metadata: Additional information about data sources, preprocessing, etc.
    """

    # Core identifiers (required)
    batch_id: str
    timestamp: datetime

    # Graph structure (required)
    num_nodes: int
    node_features: torch.Tensor  # [num_nodes, feature_dim]
    edge_index: torch.Tensor  # [2, num_edges]

    # Temporal context (required)
    time_index: torch.Tensor  # [batch_size] time indices
    target_sequences: torch.Tensor  # [batch_size, forecast_horizon]

    # Optional fields with defaults
    edge_attr: torch.Tensor | None = None  # [num_edges, edge_dim]
    sequence_length: int = 1
    region_embeddings: torch.Tensor | None = None  # [num_nodes, embed_dim]
    edar_features: torch.Tensor | None = None  # [num_edars, edar_dim]
    edar_attention_mask: torch.Tensor | None = None  # [num_nodes, num_edars]

    # New fields for three-layer EpiForecaster architecture
    cases_hist: torch.Tensor | None = None  # [batch_size, seq_len, num_nodes]
    biomarkers_hist: torch.Tensor | None = (
        None  # [batch_size, seq_len, num_nodes, biomarker_dim]
    )
    mobility_hist: torch.Tensor | None = (
        None  # [batch_size, seq_len, num_nodes, num_nodes]
    )
    region_ids: torch.Tensor | None = None  # [batch_size]

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        self._validate_shapes()
        self._validate_devices()

    def _validate_shapes(self):
        """Validate tensor shapes are consistent."""
        batch_size = len(self.time_index)

        # [sequence_length, feature_dim]
        if len(self.node_features.shape) != 2:
            raise ValueError(
                f"node_features must be 2D [sequence_length, feature_dim] for single region, "
                f"got shape {self.node_features.shape}"
            )
        # sequence_length should match time_index length
        if self.node_features.shape[0] != batch_size:
            raise ValueError(
                f"node_features sequence length ({self.node_features.shape[0]}) "
                f"must match time_index length ({batch_size})"
            )

        # Validate edge index
        if len(self.edge_index.shape) != 2:
            raise ValueError(
                f"edge_index must be 2D [2, num_edges], "
                f"got shape {self.edge_index.shape}"
            )
        if self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index first dimension must be 2, got {self.edge_index.shape[0]}"
            )

        # Validate optional edge attributes
        if self.edge_attr is not None:
            if self.edge_attr.shape[0] != self.edge_index.shape[1]:
                raise ValueError(
                    f"edge_attr first dimension ({self.edge_attr.shape[0]}) "
                    f"must match num_edges ({self.edge_index.shape[1]})"
                )

        # Validate optional region embeddings
        if self.region_embeddings is not None:
            if self.region_embeddings.shape[0] != self.num_nodes:
                raise ValueError(
                    f"region_embeddings first dimension ({self.region_embeddings.shape[0]}) "
                    f"must match num_nodes ({self.num_nodes})"
                )

        # Validate optional EDAR features
        if self.edar_features is not None and self.edar_attention_mask is not None:
            if self.edar_attention_mask.shape[0] != self.num_nodes:
                raise ValueError(
                    f"edar_attention_mask first dimension ({self.edar_attention_mask.shape[0]}) "
                    f"must match num_nodes ({self.num_nodes})"
                )
            if self.edar_attention_mask.shape[1] != self.edar_features.shape[0]:
                raise ValueError(
                    f"edar_attention_mask second dimension ({self.edar_attention_mask.shape[1]}) "
                    f"must match edar_features first dimension ({self.edar_features.shape[0]})"
                )

        # Validate new EpiForecaster fields
        self._validate_epiforecaster_fields(batch_size)

    def _validate_epiforecaster_fields(self, batch_size: int):
        """Validate fields required for three-layer EpiForecaster architecture."""
        # Validate cases_hist if provided
        if self.cases_hist is not None:
            expected_shape = (batch_size, self.sequence_length, self.num_nodes)
            if self.cases_hist.shape != expected_shape:
                raise ValueError(
                    f"cases_hist shape {self.cases_hist.shape} != expected {expected_shape}"
                )

            assert self.cases_hist.shape == (
                batch_size,
                self.sequence_length,
                self.num_nodes,
            )

        # Validate biomarkers_hist if provided
        if self.biomarkers_hist is not None:
            # Allow variable biomarker dimension
            expected_dims = (batch_size, self.sequence_length, self.num_nodes)
            if len(self.biomarkers_hist.shape) != 4:
                raise ValueError(
                    f"biomarkers_hist should be 4D (batch, seq_len, num_nodes, biomarker_dim), "
                    f"got {self.biomarkers_hist.shape}"
                )
            if self.biomarkers_hist.shape[:3] != expected_dims:
                raise ValueError(
                    f"biomarkers_hist first three dimensions {self.biomarkers_hist.shape[:3]} "
                    f"!= expected {expected_dims}"
                )

            biomarker_dim = self.biomarkers_hist.shape[3]
            assert self.biomarkers_hist.shape == (
                batch_size,
                self.sequence_length,
                self.num_nodes,
                biomarker_dim,
            )

        # Validate mobility_hist if provided
        if self.mobility_hist is not None:
            expected_shape = (
                batch_size,
                self.sequence_length,
                self.num_nodes,
                self.num_nodes,
            )
            if self.mobility_hist.shape != expected_shape:
                raise ValueError(
                    f"mobility_hist shape {self.mobility_hist.shape} != expected {expected_shape}"
                )

            assert self.mobility_hist.shape == (
                batch_size,
                self.sequence_length,
                self.num_nodes,
                self.num_nodes,
            )

        # Validate region_ids if provided
        if self.region_ids is not None:
            if (
                len(self.region_ids.shape) != 1
                or self.region_ids.shape[0] != batch_size
            ):
                raise ValueError(
                    f"region_ids should be 1D with shape ({batch_size},), "
                    f"got {self.region_ids.shape}"
                )

            # Check that region IDs are valid
            if torch.any(self.region_ids < 0) or torch.any(
                self.region_ids >= self.num_nodes
            ):
                raise ValueError(
                    f"region_ids contain invalid values outside range [0, {self.num_nodes})"
                )

            # Use simple tensor shape assertions
            assert self.region_ids.shape == (batch_size,)

    def _validate_devices(self):
        """Ensure all tensors are on the same device."""
        if not self.node_features.is_cpu:
            device = self.node_features.device
            tensors_to_check = [self.edge_index, self.time_index, self.target_sequences]

            if self.edge_attr is not None:
                tensors_to_check.append(self.edge_attr)
            if self.region_embeddings is not None:
                tensors_to_check.append(self.region_embeddings)
            if self.edar_features is not None:
                tensors_to_check.append(self.edar_features)
            if self.edar_attention_mask is not None:
                tensors_to_check.append(self.edar_attention_mask)

            for tensor in tensors_to_check:
                if tensor.device != device:
                    raise ValueError(
                        f"All tensors must be on the same device. "
                        f"Found tensors on {device} and {tensor.device}"
                    )

    def to(self, device: torch.device) -> "EpiBatch":
        """Move all tensors to specified device."""
        # Move all tensors to the specified device
        node_features = self.node_features.to(device)
        edge_index = self.edge_index.to(device)
        time_index = self.time_index.to(device)
        target_sequences = self.target_sequences.to(device)

        edge_attr = self.edge_attr.to(device) if self.edge_attr is not None else None
        region_embeddings = (
            self.region_embeddings.to(device)
            if self.region_embeddings is not None
            else None
        )
        edar_features = (
            self.edar_features.to(device) if self.edar_features is not None else None
        )
        edar_attention_mask = (
            self.edar_attention_mask.to(device)
            if self.edar_attention_mask is not None
            else None
        )

        return EpiBatch(
            batch_id=self.batch_id,
            timestamp=self.timestamp,
            num_nodes=self.num_nodes,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            time_index=time_index,
            sequence_length=self.sequence_length,
            target_sequences=target_sequences,
            region_embeddings=region_embeddings,
            edar_features=edar_features,
            edar_attention_mask=edar_attention_mask,
            metadata=self.metadata.copy(),
        )

    def pin_memory(self) -> "EpiBatch":
        """Pin all tensors for faster GPU transfer."""
        return EpiBatch(
            batch_id=self.batch_id,
            timestamp=self.timestamp,
            num_nodes=self.num_nodes,
            node_features=self.node_features.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_attr=self.edge_attr.pin_memory()
            if self.edge_attr is not None
            else None,
            time_index=self.time_index.pin_memory(),
            sequence_length=self.sequence_length,
            target_sequences=self.target_sequences.pin_memory(),
            region_embeddings=self.region_embeddings.pin_memory()
            if self.region_embeddings is not None
            else None,
            edar_features=self.edar_features.pin_memory()
            if self.edar_features is not None
            else None,
            edar_attention_mask=self.edar_attention_mask.pin_memory()
            if self.edar_attention_mask is not None
            else None,
            metadata=self.metadata.copy(),
        )

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension of node features."""
        return self.node_features.shape[1]

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph."""
        return self.edge_index.shape[1]

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return len(self.time_index)

    @property
    def forecast_horizon(self) -> int:
        """Get the forecast horizon."""
        return self.target_sequences.shape[1]

    @property
    def has_region_embeddings(self) -> bool:
        """Check if batch includes region embeddings."""
        return self.region_embeddings is not None

    @property
    def has_edar_data(self) -> bool:
        """Check if batch includes EDAR wastewater data."""
        return self.edar_features is not None

    def mask_variant(
        self, use_region_embeddings: bool = True, use_edar_data: bool = True
    ) -> "EpiBatch":
        """
        Create variant-specific batch by masking out unused features.

        Args:
            use_region_embeddings: Whether to keep region embeddings
            use_edar_data: Whether to keep EDAR wastewater data

        Returns:
            EpiBatch with masked features for specified variant
        """
        return EpiBatch(
            batch_id=self.batch_id,
            timestamp=self.timestamp,
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            time_index=self.time_index,
            sequence_length=self.sequence_length,
            target_sequences=self.target_sequences,
            region_embeddings=self.region_embeddings if use_region_embeddings else None,
            edar_features=self.edar_features if use_edar_data else None,
            edar_attention_mask=self.edar_attention_mask if use_edar_data else None,
            metadata={
                **self.metadata,
                "variant_config": {
                    "use_region_embeddings": use_region_embeddings,
                    "use_edar_data": use_edar_data,
                },
            },
        )

    def summary(self) -> dict[str, Any]:
        """Get summary of batch statistics and metadata."""
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "batch_size": self.batch_size,
            "feature_dim": self.feature_dim,
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "has_region_embeddings": self.has_region_embeddings,
            "has_edar_data": self.has_edar_data,
            "time_range": {
                "start": self.time_index.min().item(),
                "end": self.time_index.max().item(),
            },
            "metadata_keys": list(self.metadata.keys()),
        }
