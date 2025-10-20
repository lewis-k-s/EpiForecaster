"""
Geometric feature extraction for epidemiological forecasting.
"""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """
    Extracts geometric and spatial features from geographic data for graph neural networks.

    This class handles the extraction of various geometric features including distances,
    spatial densities, centrality measures, and geographic embeddings that can improve
    epidemiological forecasting performance.
    """

    def __init__(
        self,
        coordinate_system: str = "WGS84",
        distance_metric: str = "haversine",
        normalize_features: bool = True,
        k_nearest: int = 10,
    ):
        """
        Initialize the geometric feature extractor.

        Args:
            coordinate_system: Coordinate reference system ('WGS84', 'UTM', etc.)
            distance_metric: Distance calculation method ('haversine', 'euclidean')
            normalize_features: Whether to normalize extracted features
            k_nearest: Number of nearest neighbors for spatial features
        """
        self.coordinate_system = coordinate_system
        self.distance_metric = distance_metric
        self.normalize_features = normalize_features
        self.k_nearest = k_nearest

        # Feature scalers for normalization
        self.scalers = {}

        # Preprocessing hooks for custom feature engineering
        self.feature_hooks = []

    def register_feature_hook(self, func: callable):
        """
        Register a custom feature extraction function.

        Args:
            func: Function that takes coordinates and returns additional features
        """
        self.feature_hooks.append(func)
        logger.info(f"Registered feature extraction hook: {func.__name__}")

    def extract_distance_features(
        self, coordinates: np.ndarray, region_ids: list[int]
    ) -> dict[str, np.ndarray]:
        """
        Extract distance-based features from geographic coordinates.

        Args:
            coordinates: Array of [lat, lon] coordinates
            region_ids: List of region identifiers

        Returns:
            Dictionary of distance-based features
        """
        logger.info("Extracting distance features")

        n_regions = len(coordinates)
        features = {}

        # Calculate pairwise distances
        if self.distance_metric == "haversine":
            distance_matrix = self._haversine_distance_matrix(coordinates)
        else:
            distance_matrix = self._euclidean_distance_matrix(coordinates)

        # Mean distance to all other regions
        features["mean_distance"] = np.mean(distance_matrix, axis=1)

        # Distance to k-nearest neighbors
        nn_distances = []
        for i in range(n_regions):
            sorted_distances = np.sort(distance_matrix[i])
            # Exclude self (distance = 0)
            k_nearest_dist = sorted_distances[
                1 : min(self.k_nearest + 1, len(sorted_distances))
            ]
            nn_distances.append(
                np.mean(k_nearest_dist) if len(k_nearest_dist) > 0 else 0.0
            )

        features["knn_mean_distance"] = np.array(nn_distances)

        # Distance to geographic centroid
        centroid = np.mean(coordinates, axis=0)
        features["distance_to_centroid"] = np.array(
            [
                self._haversine_distance(coord, centroid)
                if self.distance_metric == "haversine"
                else np.linalg.norm(coord - centroid)
                for coord in coordinates
            ]
        )

        # Isolation measure (distance to nearest neighbor)
        features["isolation"] = np.array(
            [
                np.min(distance_matrix[i][distance_matrix[i] > 0])
                for i in range(n_regions)
            ]
        )

        logger.info(f"Extracted {len(features)} distance-based feature types")
        return features

    def extract_density_features(
        self,
        coordinates: np.ndarray,
        populations: Optional[np.ndarray] = None,
        case_counts: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract spatial density features.

        Args:
            coordinates: Array of [lat, lon] coordinates
            populations: Population counts for each region (optional)
            case_counts: Epidemiological case counts (optional)

        Returns:
            Dictionary of density-based features
        """
        logger.info("Extracting density features")

        features = {}
        n_regions = len(coordinates)

        # Build k-nearest neighbors model for density estimation
        if self.distance_metric == "haversine":
            # For geographic coordinates, use ball tree with haversine metric
            nbrs = NearestNeighbors(
                n_neighbors=self.k_nearest, metric="haversine", algorithm="ball_tree"
            )
            # Convert to radians for haversine
            coords_rad = np.radians(coordinates)
            nbrs.fit(coords_rad)
            distances, indices = nbrs.kneighbors(coords_rad)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k_nearest, metric="euclidean")
            nbrs.fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)

        # Spatial point density (inverse of mean distance to k neighbors)
        mean_distances = np.mean(distances, axis=1)
        features["spatial_density"] = 1.0 / (mean_distances + 1e-8)  # Add small epsilon

        # Population density in neighborhood (if population data available)
        if populations is not None:
            neighborhood_pop = []
            for i in range(n_regions):
                neighbor_pops = populations[indices[i]]
                neighborhood_pop.append(np.sum(neighbor_pops))
            features["neighborhood_population"] = np.array(neighborhood_pop)

            # Population-weighted centrality
            total_pop = np.sum(populations)
            features["population_centrality"] = (
                populations / total_pop if total_pop > 0 else populations * 0
            )

        # Case density in neighborhood (if case data available)
        if case_counts is not None:
            neighborhood_cases = []
            for i in range(n_regions):
                neighbor_cases = case_counts[indices[i]]
                neighborhood_cases.append(np.sum(neighbor_cases))
            features["neighborhood_cases"] = np.array(neighborhood_cases)

            # Local case rate relative to population
            if populations is not None:
                local_case_rates = []
                for i in range(n_regions):
                    neighbor_cases = case_counts[indices[i]]
                    neighbor_pops = populations[indices[i]]
                    total_cases = np.sum(neighbor_cases)
                    total_pop = np.sum(neighbor_pops)
                    rate = total_cases / total_pop if total_pop > 0 else 0
                    local_case_rates.append(rate)
                features["local_case_rate"] = np.array(local_case_rates)

        logger.info(f"Extracted {len(features)} density-based feature types")
        return features

    def extract_centrality_features(
        self, coordinates: np.ndarray, flow_matrix: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        """
        Extract centrality and connectivity features.

        Args:
            coordinates: Array of [lat, lon] coordinates
            flow_matrix: Optional mobility flow matrix for connectivity measures

        Returns:
            Dictionary of centrality features
        """
        logger.info("Extracting centrality features")

        features = {}
        len(coordinates)

        # Geographic centrality measures
        centroid = np.mean(coordinates, axis=0)

        # Distance-based centrality (inverse of distance to centroid)
        distances_to_center = np.array(
            [
                self._haversine_distance(coord, centroid)
                if self.distance_metric == "haversine"
                else np.linalg.norm(coord - centroid)
                for coord in coordinates
            ]
        )

        max_distance = np.max(distances_to_center)
        features["geographic_centrality"] = 1.0 - (distances_to_center / max_distance)

        # Degree centrality based on spatial proximity
        distance_threshold = np.percentile(
            distances_to_center, 25
        )  # Connect to nearby regions
        adjacency = self._compute_spatial_adjacency(coordinates, distance_threshold)
        features["spatial_degree"] = np.sum(adjacency, axis=1)

        # Flow-based centrality measures (if flow data available)
        if flow_matrix is not None:
            # In-degree: total incoming flows
            features["flow_in_degree"] = np.sum(flow_matrix, axis=0)

            # Out-degree: total outgoing flows
            features["flow_out_degree"] = np.sum(flow_matrix, axis=1)

            # Betweenness approximation: regions with high bidirectional flows
            bidirectional_flows = np.minimum(flow_matrix, flow_matrix.T)
            features["flow_betweenness"] = np.sum(bidirectional_flows, axis=1)

            # PageRank-style importance (flows as votes)
            features["flow_importance"] = self._compute_flow_pagerank(flow_matrix)

        logger.info(f"Extracted {len(features)} centrality-based feature types")
        return features

    def extract_geometric_embeddings(
        self, coordinates: np.ndarray, embedding_dim: int = 8
    ) -> dict[str, np.ndarray]:
        """
        Create geometric embeddings for spatial locations.

        Args:
            coordinates: Array of [lat, lon] coordinates
            embedding_dim: Dimension of resulting embeddings

        Returns:
            Dictionary containing geometric embeddings
        """
        logger.info(f"Creating {embedding_dim}-dimensional geometric embeddings")

        features = {}

        # Sinusoidal position encoding (similar to transformers)
        lat_rad = np.radians(coordinates[:, 0])
        lon_rad = np.radians(coordinates[:, 1])

        embeddings = []
        for i in range(embedding_dim):
            if i % 4 == 0:
                embeddings.append(np.sin(lat_rad * (10 ** (i // 4))))
            elif i % 4 == 1:
                embeddings.append(np.cos(lat_rad * (10 ** (i // 4))))
            elif i % 4 == 2:
                embeddings.append(np.sin(lon_rad * (10 ** (i // 4))))
            else:
                embeddings.append(np.cos(lon_rad * (10 ** (i // 4))))

        features["position_embedding"] = np.column_stack(embeddings)

        # Grid-based embeddings
        lat_min, lat_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
        lon_min, lon_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

        # Normalize to [0, 1] grid
        normalized_coords = np.column_stack(
            [
                (coordinates[:, 0] - lat_min) / (lat_max - lat_min + 1e-8),
                (coordinates[:, 1] - lon_min) / (lon_max - lon_min + 1e-8),
            ]
        )

        features["normalized_coordinates"] = normalized_coords

        logger.info("Geometric embeddings created")
        return features

    def extract_all_features(
        self,
        coordinates: np.ndarray,
        region_ids: list[int],
        populations: Optional[np.ndarray] = None,
        case_counts: Optional[np.ndarray] = None,
        flow_matrix: Optional[np.ndarray] = None,
        embedding_dim: int = 8,
    ) -> torch.Tensor:
        """
        Extract all geometric features and combine into a single tensor.

        Args:
            coordinates: Array of [lat, lon] coordinates
            region_ids: List of region identifiers
            populations: Population counts (optional)
            case_counts: Case counts (optional)
            flow_matrix: Mobility flows (optional)
            embedding_dim: Dimension for geometric embeddings

        Returns:
            Combined feature tensor of shape [n_regions, n_features]
        """
        logger.info("Extracting all geometric features")

        # Extract different types of features
        distance_features = self.extract_distance_features(coordinates, region_ids)
        density_features = self.extract_density_features(
            coordinates, populations, case_counts
        )
        centrality_features = self.extract_centrality_features(coordinates, flow_matrix)
        embedding_features = self.extract_geometric_embeddings(
            coordinates, embedding_dim
        )

        # Combine all features
        feature_dict = {**distance_features, **density_features, **centrality_features}

        # Add embeddings
        for key, values in embedding_features.items():
            if values.ndim == 1:
                feature_dict[key] = values
            else:
                # Multi-dimensional embeddings: add each dimension separately
                for i in range(values.shape[1]):
                    feature_dict[f"{key}_{i}"] = values[:, i]

        # Apply custom feature hooks
        for hook in self.feature_hooks:
            logger.info(f"Applying feature hook: {hook.__name__}")
            custom_features = hook(coordinates, region_ids)
            if isinstance(custom_features, dict):
                feature_dict.update(custom_features)

        # Stack features into matrix
        feature_names = sorted(feature_dict.keys())
        feature_matrix = np.column_stack([feature_dict[name] for name in feature_names])

        # Normalize features if requested
        if self.normalize_features:
            feature_matrix = self._normalize_features(feature_matrix, feature_names)

        # Convert to PyTorch tensor
        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

        logger.info(
            f"Extracted {feature_tensor.shape[1]} geometric features for {feature_tensor.shape[0]} regions"
        )
        return feature_tensor

    def _haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate haversine distance between two coordinates."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth's radius in kilometers
        R = 6371.0
        return R * c

    def _haversine_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise haversine distance matrix."""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._haversine_distance(coordinates[i], coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _euclidean_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise euclidean distance matrix."""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _compute_spatial_adjacency(
        self, coordinates: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Compute spatial adjacency matrix based on distance threshold."""
        if self.distance_metric == "haversine":
            distance_matrix = self._haversine_distance_matrix(coordinates)
        else:
            distance_matrix = self._euclidean_distance_matrix(coordinates)

        # Create adjacency matrix (1 if distance <= threshold, 0 otherwise)
        adjacency = (distance_matrix <= threshold).astype(int)

        # Remove self-connections
        np.fill_diagonal(adjacency, 0)

        return adjacency

    def _compute_flow_pagerank(
        self, flow_matrix: np.ndarray, alpha: float = 0.85, max_iter: int = 100
    ) -> np.ndarray:
        """Compute PageRank-style importance scores from flow matrix."""
        n = flow_matrix.shape[0]

        # Normalize flow matrix to get transition probabilities
        row_sums = np.sum(flow_matrix, axis=1)
        transition_matrix = flow_matrix / (row_sums[:, np.newaxis] + 1e-8)

        # Initialize PageRank scores
        scores = np.ones(n) / n

        # Iterative computation
        for _ in range(max_iter):
            new_scores = (1 - alpha) / n + alpha * transition_matrix.T @ scores

            # Check convergence
            if np.allclose(scores, new_scores, rtol=1e-6):
                break

            scores = new_scores

        return scores

    def _normalize_features(
        self, feature_matrix: np.ndarray, feature_names: list[str]
    ) -> np.ndarray:
        """Normalize feature matrix using stored scalers."""
        normalized_matrix = feature_matrix.copy()

        for i, name in enumerate(feature_names):
            if name not in self.scalers:
                # Use StandardScaler for most features, MinMaxScaler for embeddings
                if "embedding" in name or "normalized" in name:
                    self.scalers[name] = MinMaxScaler()
                else:
                    self.scalers[name] = StandardScaler()

                # Fit scaler on this feature
                feature_col = feature_matrix[:, i].reshape(-1, 1)
                self.scalers[name].fit(feature_col)

            # Transform feature
            feature_col = feature_matrix[:, i].reshape(-1, 1)
            normalized_matrix[:, i] = (
                self.scalers[name].transform(feature_col).flatten()
            )

        return normalized_matrix


def example_custom_features():
    """
    Example custom feature extraction functions.
    """

    def urban_rural_indicator(
        coordinates: np.ndarray, region_ids: list[int]
    ) -> dict[str, np.ndarray]:
        """
        Example: Create urban/rural indicator based on coordinate density.
        In practice, this could use land use data or population density.
        """
        # Simple heuristic: regions with many nearby neighbors are "urban"
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(
            n_neighbors=10, metric="haversine", algorithm="ball_tree"
        )
        coords_rad = np.radians(coordinates)
        nbrs.fit(coords_rad)
        distances, _ = nbrs.kneighbors(coords_rad)

        # Mean distance to 10 nearest neighbors
        mean_distances = np.mean(distances, axis=1)

        # Threshold for urban classification (smaller distance = more urban)
        urban_threshold = np.percentile(mean_distances, 33)  # Bottom 33% are urban
        urban_indicator = (mean_distances <= urban_threshold).astype(float)

        return {"urban_indicator": urban_indicator}

    def border_distance_feature(
        coordinates: np.ndarray, region_ids: list[int]
    ) -> dict[str, np.ndarray]:
        """
        Example: Distance to study area border (could be useful for edge effects).
        """
        # Find bounding box of study area
        lat_min, lat_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
        lon_min, lon_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

        # Calculate distance to nearest border
        border_distances = []
        for coord in coordinates:
            lat, lon = coord

            # Distance to each border
            dist_to_north = lat_max - lat
            dist_to_south = lat - lat_min
            dist_to_east = lon_max - lon
            dist_to_west = lon - lon_min

            # Minimum distance to any border
            min_border_dist = min(
                dist_to_north, dist_to_south, dist_to_east, dist_to_west
            )
            border_distances.append(min_border_dist)

        return {"border_distance": np.array(border_distances)}

    return {
        "urban_rural_indicator": urban_rural_indicator,
        "border_distance_feature": border_distance_feature,
    }


if __name__ == "__main__":
    # Example usage
    feature_extractor = GeometricFeatureExtractor()

    # Register custom feature hooks
    custom_features = example_custom_features()
    feature_extractor.register_feature_hook(custom_features["urban_rural_indicator"])
    feature_extractor.register_feature_hook(custom_features["border_distance_feature"])

    # Example coordinates (latitude, longitude)
    coordinates = np.array(
        [
            [40.7128, -74.0060],  # New York
            [34.0522, -118.2437],  # Los Angeles
            [41.8781, -87.6298],  # Chicago
            [29.7604, -95.3698],  # Houston
        ]
    )

    region_ids = [1, 2, 3, 4]

    # Extract features
    features = feature_extractor.extract_all_features(coordinates, region_ids)
    print(f"Extracted feature tensor shape: {features.shape}")
    print("GeometricFeatureExtractor demonstration completed")
