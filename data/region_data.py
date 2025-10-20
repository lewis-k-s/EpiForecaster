"""
Region data processing for embedding training.

Handles spatial adjacency matrix construction, mobility flow integration,
and node attribute preprocessing for region2vec-style embedding training.
"""

import logging
from typing import Any, Optional, Union

import esda
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

# PySAL imports for enhanced spatial analysis
from libpysal import weights
from libpysal.weights import KNN, Delaunay, DistanceBand, Gabriel, Kernel, Queen, Rook
from libpysal.weights.util import attach_islands
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from .mobility_loader import MobilityDataLoader

logger = logging.getLogger(__name__)


class SpatialAdjacencyBuilder:
    """
    Builds spatial adjacency matrices from geographic data.

    Supports rook/queen contiguity, k-nearest neighbors, and distance-based
    adjacency construction methods.
    """

    def __init__(
        self,
        adjacency_type: str = "queen",
        k_neighbors: int = 5,
        distance_threshold: float = 50000.0,  # 50km in meters
        include_self: bool = False,
        kernel_function: str = "triangular",  # for kernel weights
        gabriel_relative: bool = True,  # for Gabriel graph
        handle_islands: bool = True,  # attach disconnected regions
    ):
        """
        Initialize spatial adjacency builder.

        Args:
            adjacency_type: Type of adjacency ('rook', 'queen', 'knn', 'distance',
                'gabriel', 'delaunay', 'kernel')
            k_neighbors: Number of neighbors for k-NN adjacency
            distance_threshold: Maximum distance for distance-based adjacency (meters)
            include_self: Whether to include self-loops
            kernel_function: Kernel function for kernel weights ('triangular', 'gaussian')
            gabriel_relative: Use relative Gabriel graph if True
            handle_islands: Attach isolated regions to nearest neighbors
        """
        self.adjacency_type = adjacency_type
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.include_self = include_self
        self.kernel_function = kernel_function
        self.gabriel_relative = gabriel_relative
        self.handle_islands = handle_islands

    def build_contiguity_adjacency(
        self, gdf: gpd.GeoDataFrame, contiguity_type: str = "queen"
    ) -> csr_matrix:
        """
        Build contiguity-based adjacency matrix using PySAL.

        Args:
            gdf: GeoDataFrame with geometry column
            contiguity_type: 'rook' or 'queen' contiguity

        Returns:
            Sparse adjacency matrix
        """
        # Use PySAL's optimized contiguity weights
        if contiguity_type == "queen":
            w = Queen.from_dataframe(gdf, silence_warnings=True)
        elif contiguity_type == "rook":
            w = Rook.from_dataframe(gdf, silence_warnings=True)
        else:
            raise ValueError(f"Unknown contiguity type: {contiguity_type}")

        # Handle disconnected regions if requested
        if self.handle_islands and w.islands:
            logger.info(
                f"Found {len(w.islands)} disconnected regions, attaching to nearest neighbors"
            )
            w = attach_islands(w, gdf)

        # Convert to sparse matrix
        adjacency_matrix = w.sparse

        # Add self-loops if requested
        if self.include_self:
            adjacency_matrix.setdiag(1)

        return adjacency_matrix

    def build_knn_adjacency(self, coordinates: np.ndarray, k: int = 5) -> csr_matrix:
        """
        Build k-nearest neighbors adjacency matrix using PySAL.

        Args:
            coordinates: Region coordinates [n_regions, 2]
            k: Number of nearest neighbors

        Returns:
            Sparse adjacency matrix
        """
        try:
            # Use PySAL's KNN weights
            w = KNN.from_array(coordinates, k=k)

            adjacency_matrix = w.sparse

            # Add self-loops if requested
            if self.include_self:
                adjacency_matrix.setdiag(1)

            return adjacency_matrix

        except Exception as e:
            logger.warning(f"PySAL KNN failed: {e}, falling back to sklearn")
            # Fallback to sklearn's kneighbors_graph
            adjacency_matrix = kneighbors_graph(
                coordinates,
                n_neighbors=k,
                mode="connectivity",
                include_self=self.include_self,
            )
            return adjacency_matrix

    def build_distance_adjacency(
        self, coordinates: np.ndarray, threshold: float
    ) -> csr_matrix:
        """
        Build distance-based adjacency matrix using PySAL.

        Args:
            coordinates: Region coordinates [n_regions, 2]
            threshold: Distance threshold

        Returns:
            Sparse adjacency matrix
        """
        try:
            # Use PySAL's DistanceBand weights
            w = DistanceBand.from_array(
                coordinates, threshold=threshold, binary=True, silence_warnings=True
            )

            adjacency_matrix = w.sparse

            # Add self-loops if requested
            if self.include_self:
                adjacency_matrix.setdiag(1)

            return adjacency_matrix

        except Exception as e:
            logger.warning(f"PySAL DistanceBand failed: {e}, falling back to scipy")
            # Fallback to original scipy-based method
            distances = squareform(pdist(coordinates))
            adjacency_matrix = csr_matrix(distances <= threshold)

            if not self.include_self:
                adjacency_matrix.setdiag(0)

            return adjacency_matrix

    def build_gabriel_adjacency(self, coordinates: np.ndarray) -> csr_matrix:
        """
        Build Gabriel graph adjacency matrix using PySAL.

        Args:
            coordinates: Region coordinates [n_regions, 2]

        Returns:
            Sparse adjacency matrix
        """
        try:
            w = Gabriel.from_array(coordinates, relative=self.gabriel_relative)

            adjacency_matrix = w.sparse

            if self.include_self:
                adjacency_matrix.setdiag(1)

            return adjacency_matrix

        except Exception as e:
            logger.error(f"Gabriel graph construction failed: {e}")
            # Fallback to KNN if Gabriel fails
            return self.build_knn_adjacency(coordinates, self.k_neighbors)

    def build_delaunay_adjacency(self, coordinates: np.ndarray) -> csr_matrix:
        """
        Build Delaunay triangulation adjacency matrix using PySAL.

        Args:
            coordinates: Region coordinates [n_regions, 2]

        Returns:
            Sparse adjacency matrix
        """
        try:
            w = Delaunay.from_array(coordinates)

            adjacency_matrix = w.sparse

            if self.include_self:
                adjacency_matrix.setdiag(1)

            return adjacency_matrix

        except Exception as e:
            logger.error(f"Delaunay triangulation failed: {e}")
            # Fallback to KNN if Delaunay fails
            return self.build_knn_adjacency(coordinates, self.k_neighbors)

    def build_kernel_adjacency(
        self, coordinates: np.ndarray, bandwidth: Optional[float] = None
    ) -> csr_matrix:
        """
        Build kernel-based adjacency matrix using PySAL.

        Args:
            coordinates: Region coordinates [n_regions, 2]
            bandwidth: Kernel bandwidth (auto-computed if None)

        Returns:
            Sparse adjacency matrix
        """
        try:
            # Auto-compute bandwidth if not provided
            if bandwidth is None:
                bandwidth = self.distance_threshold / 2.0

            w = Kernel.from_array(
                coordinates,
                bandwidth=bandwidth,
                function=self.kernel_function,
                silence_warnings=True,
            )

            adjacency_matrix = w.sparse

            if self.include_self:
                adjacency_matrix.setdiag(1)

            return adjacency_matrix

        except Exception as e:
            logger.error(f"Kernel weights construction failed: {e}")
            # Fallback to distance-based adjacency
            return self.build_distance_adjacency(coordinates, self.distance_threshold)

    def build_adjacency(
        self,
        gdf: Optional[gpd.GeoDataFrame] = None,
        coordinates: Optional[np.ndarray] = None,
    ) -> csr_matrix:
        """
        Build adjacency matrix based on configured type.

        Args:
            gdf: GeoDataFrame (for contiguity-based adjacency)
            coordinates: Region coordinates (for coordinate-based adjacency)

        Returns:
            Sparse adjacency matrix
        """
        if self.adjacency_type in ["rook", "queen"]:
            if gdf is None:
                raise ValueError(
                    f"{self.adjacency_type} adjacency requires GeoDataFrame"
                )
            return self.build_contiguity_adjacency(gdf, self.adjacency_type)

        elif self.adjacency_type == "knn":
            if coordinates is None:
                raise ValueError("k-NN adjacency requires coordinates")
            return self.build_knn_adjacency(coordinates, self.k_neighbors)

        elif self.adjacency_type == "distance":
            if coordinates is None:
                raise ValueError("Distance adjacency requires coordinates")
            return self.build_distance_adjacency(coordinates, self.distance_threshold)

        elif self.adjacency_type == "gabriel":
            if coordinates is None:
                raise ValueError("Gabriel graph adjacency requires coordinates")
            return self.build_gabriel_adjacency(coordinates)

        elif self.adjacency_type == "delaunay":
            if coordinates is None:
                raise ValueError("Delaunay adjacency requires coordinates")
            return self.build_delaunay_adjacency(coordinates)

        elif self.adjacency_type == "kernel":
            if coordinates is None:
                raise ValueError("Kernel adjacency requires coordinates")
            return self.build_kernel_adjacency(coordinates)

        else:
            raise ValueError(f"Unknown adjacency type: {self.adjacency_type}")


class SpatialFeatureEngineer:
    """
    Creates spatial lag features and other derived spatial variables using PySAL.

    Enhances region attributes with neighborhood context, spatial autocorrelation
    measures, and multi-order spatial relationships.
    """

    def __init__(
        self,
        adjacency_builder: Optional[SpatialAdjacencyBuilder] = None,
        max_lag_order: int = 2,
        include_local_stats: bool = True,
        standardize_features: bool = True,
    ):
        """
        Initialize spatial feature engineer.

        Args:
            adjacency_builder: SpatialAdjacencyBuilder for weights matrices
            max_lag_order: Maximum order of spatial lags (1st, 2nd, etc.)
            include_local_stats: Whether to include local spatial statistics
            standardize_features: Whether to standardize spatial features
        """
        self.adjacency_builder = adjacency_builder or SpatialAdjacencyBuilder()
        self.max_lag_order = max_lag_order
        self.include_local_stats = include_local_stats
        self.standardize_features = standardize_features
        self.scaler = StandardScaler() if standardize_features else None

    def create_spatial_lag_features(
        self,
        features: np.ndarray,
        spatial_weights: weights.W,
        feature_names: Optional[list] = None,
    ) -> tuple[np.ndarray, list]:
        """
        Create spatial lag features using PySAL.

        Args:
            features: Original feature matrix [n_regions, n_features]
            spatial_weights: PySAL weights object
            feature_names: Names of original features

        Returns:
            Tuple of (augmented_features, feature_names)
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(features.shape[1])]

        augmented_features = [features]
        augmented_names = feature_names.copy()

        # Create spatial lags up to max_lag_order
        current_features = features
        current_weights = spatial_weights

        for lag_order in range(1, self.max_lag_order + 1):
            lag_features = []
            lag_names = []

            # Compute spatial lag for each feature
            for i, feat_name in enumerate(feature_names):
                try:
                    lag_values = weights.lag_spatial(
                        current_weights, current_features[:, i]
                    )
                    lag_features.append(lag_values)
                    lag_names.append(f"{feat_name}_lag{lag_order}")
                except Exception as e:
                    logger.warning(
                        f"Failed to compute lag {lag_order} for {feat_name}: {e}"
                    )
                    # Use zeros as fallback
                    lag_features.append(np.zeros(len(current_features)))
                    lag_names.append(f"{feat_name}_lag{lag_order}")

            if lag_features:
                lag_matrix = np.column_stack(lag_features)
                augmented_features.append(lag_matrix)
                augmented_names.extend(lag_names)

                # Update for next iteration (higher-order lags)
                current_features = lag_matrix

        # Combine all features
        final_features = np.column_stack(augmented_features)

        return final_features, augmented_names

    def compute_spatial_autocorrelation(
        self,
        features: np.ndarray,
        spatial_weights: weights.W,
        feature_names: Optional[list] = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute spatial autocorrelation measures using ESDA.

        Args:
            features: Feature matrix [n_regions, n_features]
            spatial_weights: PySAL weights object
            feature_names: Names of features

        Returns:
            Dictionary with autocorrelation measures
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(features.shape[1])]

        autocorr_measures = {}

        for i, feat_name in enumerate(feature_names):
            try:
                # Global Moran's I
                moran = esda.Moran(features[:, i], spatial_weights)
                autocorr_measures[f"{feat_name}_moran_i"] = moran.I
                autocorr_measures[f"{feat_name}_moran_p"] = moran.p_norm

                # Local Moran's I (LISA)
                if self.include_local_stats:
                    lisa = esda.Moran_Local(features[:, i], spatial_weights)
                    autocorr_measures[f"{feat_name}_lisa_i"] = lisa.Is
                    autocorr_measures[f"{feat_name}_lisa_p"] = lisa.p_sim
                    autocorr_measures[f"{feat_name}_lisa_quadrant"] = lisa.q

            except Exception as e:
                logger.warning(
                    f"Failed to compute autocorrelation for {feat_name}: {e}"
                )
                # Add placeholder values
                autocorr_measures[f"{feat_name}_moran_i"] = 0.0
                autocorr_measures[f"{feat_name}_moran_p"] = 1.0
                if self.include_local_stats:
                    n_regions = features.shape[0]
                    autocorr_measures[f"{feat_name}_lisa_i"] = np.zeros(n_regions)
                    autocorr_measures[f"{feat_name}_lisa_p"] = np.ones(n_regions)
                    autocorr_measures[f"{feat_name}_lisa_quadrant"] = np.zeros(
                        n_regions
                    )

        return autocorr_measures

    def create_neighborhood_context_features(
        self,
        features: np.ndarray,
        spatial_weights: weights.W,
        feature_names: Optional[list] = None,
    ) -> tuple[np.ndarray, list]:
        """
        Create neighborhood context features (diversity, heterogeneity, etc.).

        Args:
            features: Feature matrix
            spatial_weights: PySAL weights object
            feature_names: Names of features

        Returns:
            Tuple of (context_features, context_names)
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(features.shape[1])]

        context_features = []
        context_names = []

        n_regions = features.shape[0]

        for i, feat_name in enumerate(feature_names):
            # Neighborhood diversity (coefficient of variation)
            neighborhood_diversity = []
            # Neighborhood range
            neighborhood_range = []
            # Neighborhood size (number of neighbors)
            neighborhood_size = []

            for region_id in range(n_regions):
                # Get neighbors
                neighbors = spatial_weights.neighbors.get(region_id, [])

                if len(neighbors) > 0:
                    neighbor_values = features[neighbors, i]

                    # Diversity (CV)
                    if np.std(neighbor_values) > 0:
                        diversity = np.std(neighbor_values) / (
                            np.mean(neighbor_values) + 1e-8
                        )
                    else:
                        diversity = 0.0

                    # Range
                    range_val = np.max(neighbor_values) - np.min(neighbor_values)

                    # Size
                    size = len(neighbors)
                else:
                    diversity = 0.0
                    range_val = 0.0
                    size = 0

                neighborhood_diversity.append(diversity)
                neighborhood_range.append(range_val)
                neighborhood_size.append(size)

            context_features.extend(
                [neighborhood_diversity, neighborhood_range, neighborhood_size]
            )

            context_names.extend(
                [
                    f"{feat_name}_neighbor_diversity",
                    f"{feat_name}_neighbor_range",
                    f"{feat_name}_neighbor_size",
                ]
            )

        context_matrix = np.column_stack(context_features)

        return context_matrix, context_names

    def engineer_spatial_features(
        self,
        features: np.ndarray,
        gdf: gpd.GeoDataFrame,
        feature_names: Optional[list] = None,
        coordinates: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Create comprehensive spatial features from input data.

        Args:
            features: Original feature matrix
            gdf: GeoDataFrame for spatial relationships
            feature_names: Names of original features
            coordinates: Region coordinates (optional)

        Returns:
            Dictionary with all spatial features and metadata
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(features.shape[1])]

        # Build spatial weights matrix
        try:
            adjacency_matrix = self.adjacency_builder.build_adjacency(
                gdf=gdf, coordinates=coordinates
            )
            # Convert to PySAL weights format
            spatial_weights = weights.W(adjacency_matrix)
        except Exception as e:
            logger.error(f"Failed to build spatial weights: {e}")
            # Return original features as fallback
            return {
                "features": features,
                "feature_names": feature_names,
                "spatial_weights": None,
                "autocorrelation": {},
            }

        # Create spatial lag features
        lag_features, lag_names = self.create_spatial_lag_features(
            features, spatial_weights, feature_names
        )

        # Create neighborhood context features
        context_features, context_names = self.create_neighborhood_context_features(
            features, spatial_weights, feature_names
        )

        # Compute spatial autocorrelation measures
        autocorr_measures = self.compute_spatial_autocorrelation(
            features, spatial_weights, feature_names
        )

        # Extract local statistics as features if available
        local_features = []
        local_names = []
        for key, values in autocorr_measures.items():
            if isinstance(values, np.ndarray) and len(values) == features.shape[0]:
                local_features.append(values)
                local_names.append(key)

        # Combine all features
        all_features = [lag_features, context_features]
        all_names = lag_names + context_names

        if local_features:
            local_matrix = np.column_stack(local_features)
            all_features.append(local_matrix)
            all_names.extend(local_names)

        final_features = np.column_stack(all_features)

        # Optional standardization
        if self.standardize_features:
            final_features = self.scaler.fit_transform(final_features)

        return {
            "features": final_features,
            "feature_names": all_names,
            "spatial_weights": spatial_weights,
            "autocorrelation": autocorr_measures,
            "original_features": features,
            "original_names": feature_names,
        }


class RegionDataProcessor:
    """
    Processes region data for embedding training.

    Integrates spatial adjacency, mobility flows, and node attributes
    into format suitable for region2vec training.
    """

    def __init__(
        self,
        mobility_loader: Optional[MobilityDataLoader] = None,
        adjacency_builder: Optional[SpatialAdjacencyBuilder] = None,
        feature_scaler: Optional[StandardScaler] = None,
        spatial_engineer: Optional[SpatialFeatureEngineer] = None,
        normalize_flows: bool = True,
        flow_aggregation: str = "mean",  # 'mean', 'sum', 'max'
        min_flow_threshold: float = 1.0,
        use_spatial_features: bool = True,
    ):
        """
        Initialize region data processor.

        Args:
            mobility_loader: MobilityDataLoader for flow data
            adjacency_builder: SpatialAdjacencyBuilder for adjacency
            feature_scaler: StandardScaler for node features
            spatial_engineer: SpatialFeatureEngineer for spatial lag features
            normalize_flows: Whether to normalize flow values
            flow_aggregation: How to aggregate temporal flows
            min_flow_threshold: Minimum flow to consider
            use_spatial_features: Whether to use spatial feature engineering
        """
        self.mobility_loader = mobility_loader or MobilityDataLoader()
        self.adjacency_builder = adjacency_builder or SpatialAdjacencyBuilder()
        self.feature_scaler = feature_scaler or StandardScaler()
        self.spatial_engineer = spatial_engineer or SpatialFeatureEngineer(
            adjacency_builder
        )
        self.normalize_flows = normalize_flows
        self.flow_aggregation = flow_aggregation
        self.min_flow_threshold = min_flow_threshold
        self.use_spatial_features = use_spatial_features

    def load_mobility_flows(
        self,
        flow_data_path: str,
        time_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """
        Load and aggregate mobility flows.

        Args:
            flow_data_path: Path to mobility data
            time_slice: Time slice for temporal aggregation

        Returns:
            Aggregated flow matrix [num_regions, num_regions]
        """
        # Load temporal flow data
        temporal_graphs = self.mobility_loader.stream_dataset(
            flow_data_path, time_slice=time_slice
        )

        # Aggregate flows over time
        aggregated_flows = None

        for graph_data in temporal_graphs:
            # Convert edge attributes to dense flow matrix
            num_nodes = graph_data.x.size(0)
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr

            # Create dense adjacency with flow weights
            flow_matrix = torch.zeros(num_nodes, num_nodes)
            flow_matrix[edge_index[0], edge_index[1]] = edge_attr.squeeze()

            # Aggregate over time
            if aggregated_flows is None:
                aggregated_flows = flow_matrix
            else:
                if self.flow_aggregation == "mean":
                    aggregated_flows = aggregated_flows + flow_matrix
                elif self.flow_aggregation == "sum":
                    aggregated_flows = aggregated_flows + flow_matrix
                elif self.flow_aggregation == "max":
                    aggregated_flows = torch.maximum(aggregated_flows, flow_matrix)

        # Final aggregation (mean requires division)
        if aggregated_flows is not None and self.flow_aggregation == "mean":
            aggregated_flows = aggregated_flows / len(temporal_graphs)

        # Apply threshold
        if aggregated_flows is not None:
            aggregated_flows[aggregated_flows < self.min_flow_threshold] = 0.0

        return aggregated_flows or torch.zeros(1, 1)

    def load_spatial_adjacency(
        self,
        geospatial_data: Union[str, gpd.GeoDataFrame],
        coordinate_cols: Optional[tuple[str, str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load spatial adjacency matrix from geospatial data.

        Args:
            geospatial_data: Path to geospatial file or GeoDataFrame
            coordinate_cols: Column names for coordinates (lon, lat)

        Returns:
            Tuple of (edge_index, edge_weights) for spatial adjacency
        """
        # Load geospatial data
        if isinstance(geospatial_data, str):
            gdf = gpd.read_file(geospatial_data)
        else:
            gdf = geospatial_data.copy()

        # Extract geometries and coordinates
        geometries = gdf.geometry

        # Get coordinates (centroids or specified columns)
        if coordinate_cols is not None:
            coordinates = gdf[list(coordinate_cols)].values
        else:
            # Use centroids
            centroids = geometries.centroid
            coordinates = np.array([[geom.x, geom.y] for geom in centroids])

        # Build adjacency matrix
        adjacency_matrix = self.adjacency_builder.build_adjacency(
            gdf=gdf, coordinates=coordinates
        )

        # Convert to PyTorch Geometric format
        edge_index, edge_weights = from_scipy_sparse_matrix(adjacency_matrix)

        return edge_index, edge_weights

    def process_node_attributes(
        self,
        attribute_data: Union[str, pd.DataFrame],
        feature_cols: Optional[list] = None,
        categorical_cols: Optional[list] = None,
        id_col: str = "region_id",
        fit_scaler: bool = True,
        geospatial_data: Optional[Union[str, gpd.GeoDataFrame]] = None,
        coordinate_cols: Optional[tuple[str, str]] = None,
    ) -> torch.Tensor:
        """
        Process node attributes for regions with optional spatial feature engineering.

        Args:
            attribute_data: Path to attribute file or DataFrame
            feature_cols: Columns to use as features
            categorical_cols: Categorical columns to encode
            id_col: Region ID column name
            fit_scaler: Whether to fit feature scaler
            geospatial_data: Geospatial data for spatial feature engineering
            coordinate_cols: Column names for coordinates (lon, lat)

        Returns:
            Processed node features [num_regions, feature_dim]
        """
        # Load attribute data
        if isinstance(attribute_data, str):
            df = pd.read_csv(attribute_data)
        else:
            df = attribute_data.copy()

        # Select feature columns
        if feature_cols is not None:
            feature_df = df[feature_cols].copy()
        else:
            # Use all numeric columns except ID
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != id_col]
            feature_df = df[feature_cols].copy()

        # Handle categorical variables
        if categorical_cols is not None:
            for col in categorical_cols:
                if col in df.columns:
                    # One-hot encode
                    encoded = pd.get_dummies(df[col], prefix=col)
                    feature_df = pd.concat([feature_df, encoded], axis=1)

        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())

        # Apply spatial feature engineering if requested and data available
        final_features = feature_df.values
        feature_names = list(feature_df.columns)

        if self.use_spatial_features and geospatial_data is not None:
            logger.info("Applying spatial feature engineering")
            try:
                # Load geospatial data
                if isinstance(geospatial_data, str):
                    gdf = gpd.read_file(geospatial_data)
                else:
                    gdf = geospatial_data.copy()

                # Get coordinates if needed
                if coordinate_cols is not None:
                    coordinates = gdf[list(coordinate_cols)].values
                else:
                    centroids = gdf.geometry.centroid
                    coordinates = np.array([[geom.x, geom.y] for geom in centroids])

                # Engineer spatial features
                spatial_results = self.spatial_engineer.engineer_spatial_features(
                    features=final_features,
                    gdf=gdf,
                    feature_names=feature_names,
                    coordinates=coordinates,
                )

                final_features = spatial_results["features"]
                logger.info(
                    f"Enhanced features from {len(feature_names)} to {final_features.shape[1]} dimensions"
                )

            except Exception as e:
                logger.warning(
                    f"Spatial feature engineering failed: {e}, using original features"
                )
                final_features = feature_df.values

        # Scale features
        if fit_scaler:
            features_scaled = self.feature_scaler.fit_transform(final_features)
        else:
            features_scaled = self.feature_scaler.transform(final_features)

        return torch.from_numpy(features_scaled).float()

    def create_region_data(
        self,
        flow_data_path: str,
        geospatial_data: Union[str, gpd.GeoDataFrame],
        attribute_data: Union[str, pd.DataFrame],
        time_slice: Optional[slice] = None,
        feature_cols: Optional[list] = None,
        categorical_cols: Optional[list] = None,
        coordinate_cols: Optional[tuple[str, str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Create complete region dataset for embedding training.

        Args:
            flow_data_path: Path to mobility flow data
            geospatial_data: Geospatial data for adjacency
            attribute_data: Node attribute data
            time_slice: Time slice for flow aggregation
            feature_cols: Feature columns to use
            categorical_cols: Categorical columns to encode
            coordinate_cols: Coordinate column names

        Returns:
            Dictionary with processed data tensors
        """
        logger.info("Creating region dataset for embedding training")

        # Load mobility flows
        logger.info("Loading mobility flows")
        flow_matrix = self.load_mobility_flows(flow_data_path, time_slice)

        # Load spatial adjacency
        logger.info("Building spatial adjacency")
        edge_index, edge_weights = self.load_spatial_adjacency(
            geospatial_data, coordinate_cols
        )

        # Process node attributes with spatial feature engineering
        logger.info("Processing node attributes")
        node_features = self.process_node_attributes(
            attribute_data,
            feature_cols,
            categorical_cols,
            geospatial_data=geospatial_data,
            coordinate_cols=coordinate_cols,
        )

        # Create dataset dictionary
        dataset = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
            "flow_matrix": flow_matrix,
            "num_nodes": node_features.size(0),
            "feature_dim": node_features.size(1),
        }

        logger.info(f"Created region dataset with {dataset['num_nodes']} regions")
        logger.info(f"Feature dimension: {dataset['feature_dim']}")
        logger.info(f"Number of spatial edges: {edge_index.size(1)}")
        logger.info(f"Flow matrix shape: {flow_matrix.shape}")

        return dataset

    def create_pytorch_geometric_data(self, dataset: dict[str, torch.Tensor]) -> Data:
        """
        Convert dataset to PyTorch Geometric Data object.

        Args:
            dataset: Region dataset dictionary

        Returns:
            PyTorch Geometric Data object
        """
        return Data(
            x=dataset["node_features"],
            edge_index=dataset["edge_index"],
            edge_attr=dataset["edge_weights"],
            flow_matrix=dataset["flow_matrix"],
            num_nodes=dataset["num_nodes"],
        )


def create_region_data_processor(config: dict[str, Any]) -> RegionDataProcessor:
    """
    Factory function to create RegionDataProcessor from configuration.

    Args:
        config: Data processing configuration

    Returns:
        Configured RegionDataProcessor
    """
    # Mobility loader config
    mobility_config = config.get("mobility", {})
    mobility_loader = MobilityDataLoader(**mobility_config)

    # Adjacency builder config
    adjacency_config = config.get("adjacency", {})
    adjacency_builder = SpatialAdjacencyBuilder(**adjacency_config)

    # Feature scaling config
    scaler_config = config.get("feature_scaling", {})
    feature_scaler = StandardScaler(**scaler_config)

    # Spatial feature engineering config
    spatial_config = config.get("spatial_features", {})
    spatial_engineer = SpatialFeatureEngineer(
        adjacency_builder=adjacency_builder, **spatial_config
    )

    # Processing config
    processing_config = config.get("processing", {})

    return RegionDataProcessor(
        mobility_loader=mobility_loader,
        adjacency_builder=adjacency_builder,
        feature_scaler=feature_scaler,
        spatial_engineer=spatial_engineer,
        normalize_flows=processing_config.get("normalize_flows", True),
        flow_aggregation=processing_config.get("flow_aggregation", "mean"),
        min_flow_threshold=processing_config.get("min_flow_threshold", 1.0),
        use_spatial_features=processing_config.get("use_spatial_features", True),
    )


if __name__ == "__main__":
    # Example usage and testing

    # Create processor
    processor = RegionDataProcessor()

    # Test spatial adjacency builder
    adjacency_builder = SpatialAdjacencyBuilder(adjacency_type="knn", k_neighbors=3)

    # Create dummy coordinates
    coordinates = np.random.randn(20, 2) * 10
    adjacency_matrix = adjacency_builder.build_knn_adjacency(coordinates)

    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"Number of edges: {adjacency_matrix.nnz}")

    print("Region data processing initialized successfully")
