"""
EDAR Biomarker Data Loader

Loads and processes temporal wastewater biomarker data from EDAR stations.
This loader handles the time series data for wastewater surveillance signals
that serve as leading indicators for epidemiological forecasting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .wastewater_preprocessing import preprocess_wastewater_pipeline

logger = logging.getLogger(__name__)


class EDARBiomarkerLoader:
    """
    Loader for temporal EDAR biomarker data.

    Processes wastewater surveillance data from EDAR treatment plants,
    including viral load measurements (LD, N1, N2, IP4, E genes) and
    associated metadata (flow rates, rainfall).
    """

    def __init__(
        self,
        biomarker_csv_path: str,
        biomarker_features: list[str] = None,
        normalize_features: bool = True,
        handle_missing: str = "interpolate",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        device: str = "cpu",
        enable_advanced_preprocessing: bool = False,
        preprocessing_start_date: str = "2020-08-15",
        enable_duplicate_removal: bool = True,
        enable_variant_selection: bool = True,
        enable_resampling: bool = True,
        enable_flow_calculation: bool = True,
    ):
        """
        Initialize EDAR biomarker loader.

        Args:
            biomarker_csv_path: Path to wastewater biomarkers CSV file
            biomarker_features: List of biomarker columns to use
            normalize_features: Whether to normalize biomarker concentrations
            handle_missing: How to handle missing values ('interpolate', 'forward_fill', 'drop')
            min_date: Minimum date for data filtering (YYYY-MM-DD)
            max_date: Maximum date for data filtering (YYYY-MM-DD)
            device: Device for tensors ('cpu' or 'cuda')
            enable_advanced_preprocessing: Enable advanced wastewater preprocessing pipeline
            preprocessing_start_date: Start date for advanced preprocessing (YYYY-MM-DD)
            enable_duplicate_removal: Enable duplicate removal in preprocessing
            enable_variant_selection: Enable variant selection (N2/IP4) in preprocessing
            enable_resampling: Enable resampling to daily frequency in preprocessing
            enable_flow_calculation: Enable total COVID flow calculation in preprocessing
        """
        self.biomarker_csv_path = Path(biomarker_csv_path)
        self.device = device
        self.normalize_features = normalize_features
        self.handle_missing = handle_missing

        # Advanced preprocessing configuration
        self.enable_advanced_preprocessing = enable_advanced_preprocessing
        self.preprocessing_start_date = preprocessing_start_date
        self.enable_duplicate_removal = enable_duplicate_removal
        self.enable_variant_selection = enable_variant_selection
        self.enable_resampling = enable_resampling
        self.enable_flow_calculation = enable_flow_calculation

        # Default biomarker features if not specified
        if biomarker_features is None:
            self.biomarker_features = [
                "LD(CG/L)",
                "N1(CG/L)",
                "N2(CG/L)",
                "IP4(CG/L)",
                "E(CG/L)",
            ]
        else:
            self.biomarker_features = biomarker_features

        self.auxiliary_features = ["Cabal últimes 24h(m3)", "Pluja(mm)"]

        # Date filtering
        self.min_date = datetime.strptime(min_date, "%Y-%m-%d") if min_date else None
        self.max_date = datetime.strptime(max_date, "%Y-%m-%d") if max_date else None

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.edar_ids = None
        self.time_index = None
        self.feature_stats = {}

        self._load_and_process_data()

    def _load_and_process_data(self):
        """Load and preprocess the biomarker data."""
        if not self.biomarker_csv_path.exists():
            raise FileNotFoundError(
                f"Biomarker file not found: {self.biomarker_csv_path}"
            )

        logger.info(f"Loading EDAR biomarker data from {self.biomarker_csv_path}")

        # Load raw data
        self.raw_data = pd.read_csv(self.biomarker_csv_path)

        # Extract date from sample ID (format: DABR-YYYY-MM-DD)
        self.raw_data["date"] = pd.to_datetime(
            self.raw_data["id mostra"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        )

        # Apply advanced preprocessing pipeline if enabled
        if self.enable_advanced_preprocessing:
            logger.info("Applying advanced wastewater preprocessing pipeline")
            self.raw_data = preprocess_wastewater_pipeline(
                self.raw_data,
                start_date=self.preprocessing_start_date,
                enable_date_filter=True,
                enable_duplicate_removal=self.enable_duplicate_removal,
                enable_variant_selection=self.enable_variant_selection,
                enable_resampling=self.enable_resampling,
                enable_flow_calculation=self.enable_flow_calculation,
            )

            # If resampling was applied, the EDAR column name might have changed
            if "depuradora" in self.raw_data.columns:
                pass
            else:
                # Check if column was renamed during preprocessing
                possible_edar_cols = [
                    col for col in self.raw_data.columns if "edar" in col.lower()
                ]
                possible_edar_cols[0] if possible_edar_cols else "depuradora"

            # Update biomarker features if variant selection was enabled
            if (
                self.enable_variant_selection
                and "best_variant" in self.raw_data.columns
            ):
                # Add best_variant to features and update feature list
                if "best_variant" not in self.biomarker_features:
                    # Replace N2 and IP4 with best_variant for processing
                    updated_features = []
                    for feature in self.biomarker_features:
                        if feature not in ["N2(CG/L)", "IP4(CG/L)"]:
                            updated_features.append(feature)
                    updated_features.append("best_variant")
                    self.biomarker_features = updated_features
                    logger.info("Updated biomarker features to include best_variant")

        else:
            # Standard date filtering if advanced preprocessing not enabled
            if self.min_date:
                self.raw_data = self.raw_data[self.raw_data["date"] >= self.min_date]
            if self.max_date:
                self.raw_data = self.raw_data[self.raw_data["date"] <= self.max_date]

        if len(self.raw_data) == 0:
            raise ValueError("No data remains after date filtering")

        # Get unique EDARs and time points
        self.edar_ids = sorted(self.raw_data["depuradora"].unique())
        self.time_index = sorted(self.raw_data["date"].unique())

        logger.info(
            f"Found {len(self.edar_ids)} EDARs and {len(self.time_index)} time points"
        )
        logger.info(
            f"Date range: {self.time_index[0].date()} to {self.time_index[-1].date()}"
        )

        # Process features
        self._process_features()

        # Handle missing values (skip if advanced preprocessing with resampling was applied)
        if not (self.enable_advanced_preprocessing and self.enable_resampling):
            self._handle_missing_values()

        # Normalize if requested
        if self.normalize_features:
            self._normalize_features()

        logger.info("EDAR biomarker data preprocessing completed")

    def _process_features(self):
        """Process and clean biomarker features."""
        # Create pivot table: rows=time, columns=EDAR, values=features
        feature_matrices = {}

        all_features = self.biomarker_features + self.auxiliary_features

        for feature in all_features:
            if feature in self.raw_data.columns:
                # Convert to numeric, handling non-numeric values (e.g., "148" represents below detection limit)
                feature_data = self.raw_data[feature].copy()

                # Handle special values that appear in the data
                feature_data = feature_data.replace(
                    "148", 0.0
                )  # Below detection limit -> 0
                feature_data = pd.to_numeric(feature_data, errors="coerce")

                # Create pivot table with the processed data
                raw_data_copy = self.raw_data.copy()
                raw_data_copy[feature] = feature_data

                pivot = raw_data_copy.pivot_table(
                    index="date",
                    columns="depuradora",
                    values=feature,
                    aggfunc="mean",  # Handle duplicates if any
                )

                # Ensure all EDARs and times are present
                pivot = pivot.reindex(index=self.time_index, columns=self.edar_ids)

                feature_matrices[feature] = pivot

                # Store statistics
                non_null_data = feature_data.dropna()
                if len(non_null_data) > 0:
                    self.feature_stats[feature] = {
                        "mean": non_null_data.mean(),
                        "std": non_null_data.std(),
                        "min": non_null_data.min(),
                        "max": non_null_data.max(),
                        "missing_rate": feature_data.isna().mean(),
                    }
                    logger.debug(
                        f"Feature {feature}: mean={self.feature_stats[feature]['mean']:.2f}, "
                        f"missing={self.feature_stats[feature]['missing_rate']:.1%}"
                    )

        self.processed_data = feature_matrices

    def _handle_missing_values(self):
        """Handle missing values in the biomarker data."""
        for feature, data in self.processed_data.items():
            if self.handle_missing == "interpolate":
                # Linear interpolation along time axis
                self.processed_data[feature] = data.interpolate(method="time", axis=0)
                # Forward fill remaining NaN values
                self.processed_data[feature] = self.processed_data[feature].ffill()
                # Backward fill any remaining NaN values
                self.processed_data[feature] = self.processed_data[feature].bfill()

            elif self.handle_missing == "forward_fill":
                self.processed_data[feature] = data.ffill()

            elif self.handle_missing == "drop":
                # This would require more complex handling for time series
                pass

            # Replace any remaining NaN with feature mean
            if feature in self.feature_stats:
                mean_value = self.feature_stats[feature]["mean"]
                self.processed_data[feature] = self.processed_data[feature].fillna(
                    mean_value
                )

    def _normalize_features(self):
        """Normalize biomarker features."""
        for feature in self.biomarker_features:
            if feature in self.processed_data and feature in self.feature_stats:
                stats = self.feature_stats[feature]
                if stats["std"] > 0:
                    # Z-score normalization
                    self.processed_data[feature] = (
                        self.processed_data[feature] - stats["mean"]
                    ) / stats["std"]
                    logger.debug(f"Normalized feature: {feature}")

    def get_temporal_features_tensor(self, time_index: int) -> torch.Tensor:
        """
        Get biomarker features for all EDARs at a specific time point.

        Args:
            time_index: Index in the time series

        Returns:
            Feature tensor [n_edars, n_features]
        """
        if time_index >= len(self.time_index):
            raise IndexError(
                f"Time index {time_index} out of range (max: {len(self.time_index) - 1})"
            )

        features = []
        all_features = self.biomarker_features + self.auxiliary_features

        for feature in all_features:
            if feature in self.processed_data:
                # Get feature values for all EDARs at this time point
                feature_values = self.processed_data[feature].iloc[time_index].values
                features.append(feature_values)

        if len(features) == 0:
            # Return zero features if no data available
            return torch.zeros(
                len(self.edar_ids), 1, dtype=torch.float32, device=self.device
            )

        # Stack features: [n_edars, n_features]
        feature_matrix = np.column_stack(features)

        return torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)

    def create_edar_graph(
        self, time_index: int, create_edges: bool = True, edge_threshold: float = 0.5
    ) -> Data:
        """
        Create PyTorch Geometric graph for EDARs at a specific time point.

        Args:
            time_index: Time point index
            create_edges: Whether to create edges between EDARs
            edge_threshold: Threshold for creating edges based on biomarker correlation

        Returns:
            PyG Data object for EDAR graph
        """
        # Get node features
        node_features = self.get_temporal_features_tensor(time_index)

        if create_edges and len(self.edar_ids) > 1:
            # Create edges based on biomarker similarity or geographic proximity
            # For now, create a simple correlation-based graph

            # Get biomarker data for correlation
            biomarker_data = []
            for feature in self.biomarker_features:
                if feature in self.processed_data:
                    # Use a window around current time for correlation
                    start_idx = max(0, time_index - 7)  # 7-day window
                    end_idx = min(len(self.time_index), time_index + 1)

                    window_data = self.processed_data[feature].iloc[start_idx:end_idx]
                    biomarker_data.append(
                        window_data.T.values
                    )  # Transpose for EDAR x time

            if biomarker_data:
                # Compute correlation matrix
                combined_data = np.concatenate(biomarker_data, axis=1)
                correlation_matrix = np.corrcoef(combined_data)

                # Create edges where correlation > threshold
                edge_indices = np.where(
                    (correlation_matrix > edge_threshold) & (correlation_matrix < 0.99)
                )  # Exclude self-loops

                if len(edge_indices[0]) > 0:
                    edge_index = torch.tensor(
                        np.vstack(edge_indices), dtype=torch.long, device=self.device
                    )

                    # Edge attributes (correlation values)
                    edge_attr = torch.tensor(
                        correlation_matrix[edge_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)
                else:
                    # No edges found, create empty edge structure
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                    edge_attr = torch.zeros(
                        0, 1, dtype=torch.float32, device=self.device
                    )
            else:
                # No biomarker data available, no edges
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                edge_attr = torch.zeros(0, 1, dtype=torch.float32, device=self.device)
        else:
            # No edges requested or only one EDAR
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_attr = torch.zeros(0, 1, dtype=torch.float32, device=self.device)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.edar_ids),
        )

    def stream_temporal_graphs(
        self, start_time_idx: int = 0, end_time_idx: Optional[int] = None
    ) -> list[Data]:
        """
        Generate a sequence of EDAR graphs over time.

        Args:
            start_time_idx: Starting time index
            end_time_idx: Ending time index (None for all remaining)

        Returns:
            List of PyG Data objects
        """
        if end_time_idx is None:
            end_time_idx = len(self.time_index)

        temporal_graphs = []

        for t in range(start_time_idx, end_time_idx):
            graph = self.create_edar_graph(time_index=t)
            temporal_graphs.append(graph)

        return temporal_graphs

    def align_with_dates(self, target_dates: list[datetime]) -> list[int]:
        """
        Find time indices that best match target dates.

        Args:
            target_dates: List of target datetime objects

        Returns:
            List of time indices
        """
        time_indices = []

        for target_date in target_dates:
            # Find closest date
            time_diffs = [abs((t - target_date).days) for t in self.time_index]
            closest_idx = np.argmin(time_diffs)
            time_indices.append(closest_idx)

        return time_indices

    def get_statistics(self) -> dict:
        """Get statistics about the biomarker data."""
        stats = {
            "n_edars": len(self.edar_ids),
            "n_timepoints": len(self.time_index),
            "date_range": (self.time_index[0].date(), self.time_index[-1].date()),
            "features": list(self.biomarker_features),
            "auxiliary_features": list(self.auxiliary_features),
            "feature_statistics": self.feature_stats,
        }

        return stats

    def close(self):
        """Clean up resources."""
        pass


def create_edar_biomarker_loader(
    data_dir: str,
    biomarker_features: list[str] = None,
    date_range: tuple[Optional[str], Optional[str]] = (None, None),
    normalize: bool = True,
    enable_advanced_preprocessing: bool = False,
    preprocessing_config: Optional[dict] = None,
    biomarker_path: str | None = None,
) -> EDARBiomarkerLoader:
    """
    Factory function to create EDAR biomarker loader.

    Args:
        data_dir: Directory containing data files
        biomarker_features: List of biomarker features to use
        date_range: Tuple of (start_date, end_date) strings
        normalize: Whether to normalize features
        enable_advanced_preprocessing: Enable advanced wastewater preprocessing pipeline
        preprocessing_config: Dictionary with preprocessing configuration options
        biomarker_path: Optional override for the wastewater biomarker CSV path

    Returns:
        Configured EDARBiomarkerLoader
    """
    data_base = Path(data_dir)

    def _resolve_path(candidate: str | Path) -> Path:
        path_candidate = Path(candidate)
        if path_candidate.is_absolute():
            return path_candidate
        base_parts = data_base.parts
        candidate_parts = path_candidate.parts
        if candidate_parts[: len(base_parts)] == base_parts:
            return path_candidate
        return data_base / path_candidate

    if biomarker_path is None:
        resolved_path = data_base / "wastewater_biomarkers_icra.csv"
    else:
        resolved_path = _resolve_path(biomarker_path)

    # Default preprocessing configuration
    default_preprocessing_config = {
        "preprocessing_start_date": "2020-08-15",
        "enable_duplicate_removal": True,
        "enable_variant_selection": True,
        "enable_resampling": True,
        "enable_flow_calculation": True,
    }

    # Update with user-provided config
    if preprocessing_config:
        default_preprocessing_config.update(preprocessing_config)

    loader = EDARBiomarkerLoader(
        biomarker_csv_path=str(resolved_path),
        biomarker_features=biomarker_features,
        normalize_features=normalize,
        min_date=date_range[0],
        max_date=date_range[1],
        enable_advanced_preprocessing=enable_advanced_preprocessing,
        **default_preprocessing_config,
    )

    # Log statistics
    stats = loader.get_statistics()
    logger.info("EDAR Biomarker Loader Statistics:")
    logger.info(f"  EDARs: {stats['n_edars']}")
    logger.info(f"  Time points: {stats['n_timepoints']}")
    logger.info(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    logger.info(f"  Biomarker features: {len(stats['features'])}")

    return loader


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    # Create loader
    loader = create_edar_biomarker_loader("data")

    # Test graph creation
    if len(loader.time_index) > 0:
        test_graph = loader.create_edar_graph(time_index=0)
        print("\nTest EDAR graph:")
        print(f"  Nodes: {test_graph.num_nodes}")
        print(f"  Edges: {test_graph.edge_index.shape[1]}")
        print(f"  Node features: {test_graph.x.shape}")

        # Test temporal sequence
        temporal_graphs = loader.stream_temporal_graphs(
            end_time_idx=min(5, len(loader.time_index))
        )
        print(f"  Temporal sequence length: {len(temporal_graphs)}")

    loader.close()
    print("\nEDAR Biomarker Loader test completed successfully!")
