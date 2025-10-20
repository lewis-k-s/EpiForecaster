"""
Mobility flow edge feature processing for epidemiological graphs.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MobilityFlowProcessor:
    """
    Processes mobility flow features for epidemiological graph edges.

    Handles normalization and smoothing of mobility flow data between zones,
    creating meaningful edge representations for graph neural networks.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        normalize_method: str = "minmax",
        temporal_smoothing: bool = True,
        spatial_smoothing: bool = False,
        handle_missing: str = "interpolate",
    ):
        """
        Initialize mobility flow processor.

        Args:
            embedding_dim: Dimension for flow embeddings
            normalize_method: Normalization method ('minmax', 'zscore', 'log')
            temporal_smoothing: Whether to apply temporal smoothing
            spatial_smoothing: Whether to apply spatial smoothing
            handle_missing: Strategy for missing values ('interpolate', 'zero', 'mean')
        """
        self.embedding_dim = embedding_dim
        self.normalize_method = normalize_method
        self.temporal_smoothing = temporal_smoothing
        self.spatial_smoothing = spatial_smoothing
        self.handle_missing = handle_missing

        # Normalization parameters (fitted during training)
        self.flow_stats = {}
        self.fitted = False

        # Smoothing parameters
        self.temporal_window = 7  # Days for temporal smoothing
        self.spatial_decay = 1.0  # Spatial decay factor

        # Processing hooks for custom flow processing
        self.processing_hooks = []

    def register_processing_hook(self, func: callable):
        """
        Register a custom mobility flow processing function.

        Args:
            func: Function that takes and returns mobility flow tensor
        """
        self.processing_hooks.append(func)
        logger.info(f"Registered flow processing hook: {func.__name__}")

    def fit(self, flow_data: torch.Tensor):
        """
        Fit normalization parameters on mobility flow data.

        Args:
            flow_data: Tensor with mobility flow data [time, origin, destination]
        """
        logger.info("Fitting mobility flow processors")

        # Apply preprocessing hooks
        for hook in self.processing_hooks:
            logger.info(f"Applying preprocessing hook: {hook.__name__}")
            flow_data = hook(flow_data)

        # Handle missing values first
        flow_data = self._handle_missing_values(flow_data)

        # Compute normalization statistics
        if self.normalize_method == "minmax":
            self.flow_stats["min"] = torch.min(flow_data)
            self.flow_stats["max"] = torch.max(flow_data)
        elif self.normalize_method == "zscore":
            self.flow_stats["mean"] = torch.mean(flow_data)
            self.flow_stats["std"] = torch.std(flow_data)
        elif self.normalize_method == "log":
            # For log normalization, we just need to ensure positive values
            self.flow_stats["offset"] = torch.abs(torch.min(flow_data)) + 1e-8

        self.fitted = True
        logger.info("Mobility flow processors fitted successfully")

    def transform(self, flow_data: torch.Tensor) -> torch.Tensor:
        """
        Transform mobility flow data into normalized edge features.

        Args:
            flow_data: Tensor with mobility flow data [time, origin, destination]

        Returns:
            Normalized edge feature tensor
        """
        if not self.fitted:
            raise ValueError("Processor must be fitted before transforming")

        logger.info("Transforming mobility flow data to edge features")

        # Apply preprocessing hooks
        for hook in self.processing_hooks:
            flow_data = hook(flow_data)

        # Handle missing values
        flow_data = self._handle_missing_values(flow_data)

        # Apply normalization
        normalized_flows = self._normalize_flows(flow_data)

        # Apply temporal smoothing if enabled
        if self.temporal_smoothing:
            normalized_flows = self._apply_temporal_smoothing(normalized_flows)

        # Apply spatial smoothing if enabled
        if self.spatial_smoothing:
            normalized_flows = self._apply_spatial_smoothing(normalized_flows)

        # Create edge features from processed flows
        edge_features = self._create_edge_features(normalized_flows)

        logger.info(f"Created edge features with shape: {edge_features.shape}")
        return edge_features

    def fit_transform(self, flow_data: torch.Tensor) -> torch.Tensor:
        """
        Fit processors and transform data in one step.

        Args:
            flow_data: Tensor with mobility flow data [time, origin, destination]

        Returns:
            Normalized edge feature tensor
        """
        self.fit(flow_data)
        return self.transform(flow_data)

    def _handle_missing_values(self, flow_data: torch.Tensor) -> torch.Tensor:
        """Handle missing values in mobility flow data."""
        if self.handle_missing == "interpolate":
            # Linear interpolation for time series data
            mask = torch.isnan(flow_data)
            if mask.any():
                # Simple forward fill then backward fill
                flow_data = flow_data.clone()
                for t in range(1, flow_data.shape[0]):
                    flow_data[t][mask[t]] = flow_data[t - 1][mask[t]]
                # Backward fill for any remaining NaNs at the beginning
                for t in range(flow_data.shape[0] - 2, -1, -1):
                    flow_data[t][mask[t]] = flow_data[t + 1][mask[t]]
        elif self.handle_missing == "zero":
            flow_data = torch.nan_to_num(flow_data, nan=0.0)
        elif self.handle_missing == "mean":
            # Replace with mean flow for that origin-destination pair
            mean_flows = torch.nanmean(flow_data, dim=0, keepdim=True)
            mask = torch.isnan(flow_data)
            flow_data = torch.where(mask, mean_flows, flow_data)

        return flow_data

    def _normalize_flows(self, flow_data: torch.Tensor) -> torch.Tensor:
        """Apply normalization to mobility flows."""
        if self.normalize_method == "minmax":
            min_val = self.flow_stats["min"]
            max_val = self.flow_stats["max"]
            if max_val > min_val:
                return (flow_data - min_val) / (max_val - min_val)
            else:
                return torch.zeros_like(flow_data)
        elif self.normalize_method == "zscore":
            mean_val = self.flow_stats["mean"]
            std_val = self.flow_stats["std"]
            if std_val > 0:
                return (flow_data - mean_val) / std_val
            else:
                return torch.zeros_like(flow_data)
        elif self.normalize_method == "log":
            offset = self.flow_stats["offset"]
            return torch.log(flow_data + offset)
        else:
            return flow_data

    def _apply_temporal_smoothing(self, flow_data: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing using moving averages."""
        if flow_data.shape[0] < self.temporal_window:
            return flow_data

        smoothed = flow_data.clone()
        half_window = self.temporal_window // 2

        # Apply moving average
        for t in range(half_window, flow_data.shape[0] - half_window):
            window_start = t - half_window
            window_end = t + half_window + 1
            smoothed[t] = torch.mean(flow_data[window_start:window_end], dim=0)

        return smoothed

    def _apply_spatial_smoothing(self, flow_data: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing (placeholder - requires distance matrix)."""
        # This would require a distance matrix between zones
        # For now, return the data unchanged
        logger.warning("Spatial smoothing not implemented - requires distance matrix")
        return flow_data

    def _create_edge_features(self, flow_data: torch.Tensor) -> torch.Tensor:
        """Create edge features from processed mobility flows."""
        # For now, we'll aggregate temporal dimension and create basic features
        T, O, D = flow_data.shape

        # Create edge list from non-zero flows
        edges = []
        edge_features = []

        for o in range(O):
            for d in range(D):
                if o != d:  # Skip self-loops
                    # Time series of flows for this edge
                    flow_series = flow_data[:, o, d]

                    # Basic flow statistics as features
                    features = [
                        torch.mean(flow_series).item(),  # Average flow
                        torch.std(flow_series).item(),  # Flow variability
                        torch.max(flow_series).item(),  # Peak flow
                        torch.sum(flow_series > 0).float().item() / T,  # Activity ratio
                    ]

                    edges.append((o, d))
                    edge_features.append(features)

        if len(edge_features) == 0:
            return torch.empty(0, 4)

        return torch.tensor(edge_features, dtype=torch.float32)

    def get_feature_dimension(self) -> int:
        """Get the total dimension of processed edge features."""
        if not self.fitted:
            raise ValueError("Processor must be fitted to determine feature dimension")

        # Basic mobility flow features
        return 4  # average_flow, flow_variability, peak_flow, activity_ratio


class EdgeAttributeEmbedder(nn.Module):
    """
    Neural network module for embedding edge attributes.

    Processes demographic edge features and creates learnable embeddings
    that can be used in graph neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """
        Initialize edge attribute embedder.

        Args:
            input_dim: Input dimension of edge features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build layers
        layers = []

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            # Add activation and dropout (except for last layer)
            if i < num_layers - 1:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())

                layers.append(nn.Dropout(dropout))

        self.embedder = nn.Sequential(*layers)

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through edge embedder.

        Args:
            edge_attr: Edge attributes [num_edges, input_dim]

        Returns:
            Edge embeddings [num_edges, output_dim]
        """
        embeddings = self.embedder(edge_attr)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class EdgeFeatureProcessor:
    """
    Complete edge feature processing pipeline.

    Combines mobility flow processing with optional geographic and temporal features
    to create comprehensive edge representations.
    """

    def __init__(
        self,
        flow_processor: MobilityFlowProcessor,
        include_geographic: bool = False,
        include_temporal: bool = False,
        embedding_dim: int = 16,
    ):
        """
        Initialize complete edge feature processor.

        Args:
            flow_processor: Fitted mobility flow processor
            include_geographic: Whether to include geographic features
            include_temporal: Whether to include temporal features
            embedding_dim: Dimension for neural embeddings
        """
        self.flow_processor = flow_processor
        self.include_geographic = include_geographic
        self.include_temporal = include_temporal
        self.embedding_dim = embedding_dim

        # Neural embedder (will be initialized after knowing input dimension)
        self.embedder = None

    def process_all_features(
        self,
        flow_data: torch.Tensor,
        geographic_data: Optional[pd.DataFrame] = None,
        temporal_data: Optional[pd.DataFrame] = None,
    ) -> torch.Tensor:
        """
        Process all types of edge features.

        Args:
            flow_data: Mobility flow tensor [time, origin, destination]
            geographic_data: Geographic features (distance, direction, etc.)
            temporal_data: Temporal features (time of day, day of week, etc.)

        Returns:
            Combined edge feature tensor
        """
        logger.info("Processing all edge features")

        features = []

        # Process mobility flow features
        flow_features = self.flow_processor.transform(flow_data)
        features.append(flow_features)

        # Process geographic features
        if self.include_geographic and geographic_data is not None:
            geo_features = self._process_geographic_features(geographic_data)
            features.append(geo_features)

        # Process temporal features
        if self.include_temporal and temporal_data is not None:
            temp_features = self._process_temporal_features(temporal_data)
            features.append(temp_features)

        # Combine all features
        combined_features = torch.cat(features, dim=1)

        # Initialize embedder if not done yet
        if self.embedder is None:
            self.embedder = EdgeAttributeEmbedder(
                input_dim=combined_features.shape[1], output_dim=self.embedding_dim
            )

        # Create neural embeddings
        edge_embeddings = self.embedder(combined_features)

        logger.info(f"Created edge embeddings with shape: {edge_embeddings.shape}")
        return edge_embeddings

    def _process_geographic_features(self, geo_data: pd.DataFrame) -> torch.Tensor:
        """Process geographic edge features."""
        geo_features = []

        # Distance features
        if "distance_km" in geo_data.columns:
            distances = torch.tensor(
                geo_data["distance_km"].values, dtype=torch.float
            ).unsqueeze(1)
            geo_features.append(distances)

            # Log distance (helps with skewed distributions)
            log_distances = torch.log(distances + 1.0)
            geo_features.append(log_distances)

        # Direction/bearing features
        if "bearing_degrees" in geo_data.columns:
            bearings = torch.tensor(
                geo_data["bearing_degrees"].values, dtype=torch.float
            )
            # Convert to sine/cosine for circular feature
            bearing_sin = torch.sin(torch.deg2rad(bearings)).unsqueeze(1)
            bearing_cos = torch.cos(torch.deg2rad(bearings)).unsqueeze(1)
            geo_features.extend([bearing_sin, bearing_cos])

        # Urban/rural indicators
        if "origin_urban" in geo_data.columns and "dest_urban" in geo_data.columns:
            origin_urban = torch.tensor(
                geo_data["origin_urban"].values, dtype=torch.float
            ).unsqueeze(1)
            dest_urban = torch.tensor(
                geo_data["dest_urban"].values, dtype=torch.float
            ).unsqueeze(1)
            geo_features.extend([origin_urban, dest_urban])

        if len(geo_features) == 0:
            return torch.empty(len(geo_data), 0)

        return torch.cat(geo_features, dim=1)

    def _process_temporal_features(self, temp_data: pd.DataFrame) -> torch.Tensor:
        """Process temporal edge features."""
        temp_features = []

        # Time of day (cyclical encoding)
        if "hour" in temp_data.columns:
            hours = torch.tensor(temp_data["hour"].values, dtype=torch.float)
            hour_sin = torch.sin(2 * np.pi * hours / 24).unsqueeze(1)
            hour_cos = torch.cos(2 * np.pi * hours / 24).unsqueeze(1)
            temp_features.extend([hour_sin, hour_cos])

        # Day of week (cyclical encoding)
        if "day_of_week" in temp_data.columns:
            days = torch.tensor(temp_data["day_of_week"].values, dtype=torch.float)
            day_sin = torch.sin(2 * np.pi * days / 7).unsqueeze(1)
            day_cos = torch.cos(2 * np.pi * days / 7).unsqueeze(1)
            temp_features.extend([day_sin, day_cos])

        # Weekend indicator
        if "is_weekend" in temp_data.columns:
            weekend = torch.tensor(
                temp_data["is_weekend"].values, dtype=torch.float
            ).unsqueeze(1)
            temp_features.append(weekend)

        # Rush hour indicators
        if "is_rush_hour" in temp_data.columns:
            rush_hour = torch.tensor(
                temp_data["is_rush_hour"].values, dtype=torch.float
            ).unsqueeze(1)
            temp_features.append(rush_hour)

        if len(temp_features) == 0:
            return torch.empty(len(temp_data), 0)

        return torch.cat(temp_features, dim=1)


def example_mobility_hooks():
    """
    Example preprocessing functions for mobility flow data.
    """

    def remove_outliers(flow_data: torch.Tensor) -> torch.Tensor:
        """Example: Remove extreme outliers using IQR method."""
        flow_data = flow_data.clone()

        # Flatten spatial dimensions for outlier detection
        flat_flows = flow_data.flatten()
        q75, q25 = torch.quantile(flat_flows, torch.tensor([0.75, 0.25]))
        iqr = q75 - q25

        # Define outlier thresholds
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # Cap outliers
        flow_data = torch.clamp(flow_data, lower_bound, upper_bound)
        return flow_data

    def apply_weekend_adjustment(flow_data: torch.Tensor) -> torch.Tensor:
        """Example: Adjust flows based on weekday patterns."""
        # This would require temporal information about which days are weekends
        # For now, return unchanged
        return flow_data

    return {
        "remove_outliers": remove_outliers,
        "apply_weekend_adjustment": apply_weekend_adjustment,
    }


if __name__ == "__main__":
    # Example usage
    processor = MobilityFlowProcessor(
        embedding_dim=16,
        normalize_method="minmax",
        temporal_smoothing=True,
        spatial_smoothing=False,
    )

    # Register example hooks
    hooks = example_mobility_hooks()
    processor.register_processing_hook(hooks["remove_outliers"])

    # Create example mobility flow data (time=30, origin=5, destination=5)
    torch.manual_seed(42)
    flow_data = torch.abs(torch.randn(30, 5, 5)) * 100  # Positive flows
    flow_data.fill_diagonal_(0)  # No self-loops

    # Add some missing values for testing
    flow_data[0, 1, 2] = float("nan")
    flow_data[5:8, 3, 4] = float("nan")

    # Process features
    edge_features = processor.fit_transform(flow_data)
    print(f"Mobility edge features shape: {edge_features.shape}")
    print(f"Feature dimension: {processor.get_feature_dimension()}")
    print("MobilityFlowProcessor demonstration completed")
