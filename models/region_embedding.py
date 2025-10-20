"""
Region embedding models based on region2vec approach.

Implements 2-layer GCN with community-oriented loss for unsupervised
geography-aware pretraining of static region embeddings.
"""

import logging
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libpysal import weights
from sklearn.cluster import AgglomerativeClustering

# PySAL imports for advanced regionalization
from spopt.region import MaxPHeuristic, Skater, Spenc
from torch_geometric.nn import GCNConv, SAGEConv

from .region_losses import create_community_loss

logger = logging.getLogger(__name__)


class RegionGCN(nn.Module):
    """
    2-layer Graph Convolutional Network for region embedding.

    Encodes regions using spatial adjacency and node attributes following
    the region2vec architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        conv_type: str = "gcn",
        activation: str = "relu",
        dropout: float = 0.5,
        normalize: bool = True,
        bias: bool = True,
    ):
        """
        Initialize RegionGCN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            conv_type: Convolution type ('gcn' or 'sage')
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
            normalize: Whether to normalize outputs
            bias: Whether to use bias in conv layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv_type = conv_type
        self.normalize = normalize

        # Choose convolution layers
        if conv_type == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim, bias=bias, normalize=False)
            self.conv2 = GCNConv(hidden_dim, output_dim, bias=bias, normalize=False)
        elif conv_type == "sage":
            self.conv1 = SAGEConv(input_dim, hidden_dim, bias=bias, normalize=False)
            self.conv2 = SAGEConv(hidden_dim, output_dim, bias=bias, normalize=False)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2-layer GCN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # First layer
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.dropout(h)

        # Second layer
        h = self.conv2(h, edge_index)

        # Optional normalization
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)

        return h

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get region embeddings (alias for forward).

        Args:
            x: Node features
            edge_index: Graph edges

        Returns:
            Node embeddings
        """
        return self.forward(x, edge_index)


class SpatialRegionalizer:
    """
    Advanced spatial regionalization using PySAL algorithms.

    Supports SKATER, Max-P, and SPENC regionalization methods for creating
    spatially contiguous and homogeneous regions from embeddings.
    """

    def __init__(
        self,
        regionalization_method: str = "skater",
        n_clusters: Optional[int] = 10,
        min_size: int = 5,
        connectivity_matrix: Optional[torch.Tensor] = None,
        spatial_weights: Optional[weights.W] = None,
        gdf: Optional[gpd.GeoDataFrame] = None,
        homogeneity_threshold: Optional[float] = None,
        alpha: float = 0.1,  # For SPENC
        random_state: int = 42,
    ):
        """
        Initialize spatial regionalizer.

        Args:
            regionalization_method: Algorithm to use ('skater', 'maxp', 'spenc', 'agglomerative')
            n_clusters: Number of clusters (for SKATER and agglomerative)
            min_size: Minimum region size (for Max-P)
            connectivity_matrix: Spatial connectivity matrix (tensor format)
            spatial_weights: PySAL spatial weights object
            gdf: GeoDataFrame for spatial context
            homogeneity_threshold: Homogeneity threshold for Max-P
            alpha: Spatial regularization parameter for SPENC
            random_state: Random seed for reproducible results
        """
        self.regionalization_method = regionalization_method
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.connectivity_matrix = connectivity_matrix
        self.spatial_weights = spatial_weights
        self.gdf = gdf
        self.homogeneity_threshold = homogeneity_threshold
        self.alpha = alpha
        self.random_state = random_state
        self._regionalizer = None

        # Validate parameters
        if regionalization_method not in ["skater", "maxp", "spenc", "agglomerative"]:
            raise ValueError(
                f"Unknown regionalization method: {regionalization_method}"
            )

    def _prepare_spatial_weights(self, n_regions: int) -> Optional[weights.W]:
        """
        Prepare spatial weights for regionalization algorithms.

        Args:
            n_regions: Number of regions

        Returns:
            PySAL weights object or None
        """
        if self.spatial_weights is not None:
            return self.spatial_weights

        if self.connectivity_matrix is not None:
            # Convert connectivity matrix to PySAL weights format
            try:
                connectivity_np = self.connectivity_matrix.detach().cpu().numpy()
                w = weights.W(connectivity_np)
                return w
            except Exception as e:
                logger.warning(f"Failed to convert connectivity matrix to weights: {e}")

        # Fallback: create simple weights if GDF is available
        if self.gdf is not None:
            try:
                from libpysal.weights import Queen

                w = Queen.from_dataframe(self.gdf, silence_warnings=True)
                return w
            except Exception as e:
                logger.warning(f"Failed to create weights from GDF: {e}")

        return None

    def fit_predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fit regionalization and predict region labels.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]

        Returns:
            Region labels [num_nodes]
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        n_regions, embed_dim = embeddings_np.shape

        # Prepare spatial weights
        w = self._prepare_spatial_weights(n_regions)

        try:
            if self.regionalization_method == "skater":
                return self._fit_skater(embeddings_np, w)
            elif self.regionalization_method == "maxp":
                return self._fit_maxp(embeddings_np, w)
            elif self.regionalization_method == "spenc":
                return self._fit_spenc(embeddings_np, w)
            else:  # fallback to agglomerative
                return self._fit_agglomerative(embeddings_np)

        except Exception as e:
            logger.error(
                f"Regionalization method {self.regionalization_method} failed: {e}"
            )
            # Fallback to simple agglomerative clustering
            return self._fit_agglomerative(embeddings_np)

    def _fit_skater(
        self, embeddings: np.ndarray, w: Optional[weights.W]
    ) -> torch.Tensor:
        """Fit SKATER regionalization."""
        if w is None:
            raise ValueError("SKATER requires spatial weights")

        try:
            skater = Skater(
                gdf=self.gdf,
                w=w,
                attrs_name=None,  # Will use embeddings directly
                n_clusters=self.n_clusters or 10,
                random_state=self.random_state,
            )

            # Create temporary dataframe with embeddings as attributes
            import pandas as pd

            embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
            embed_df = pd.DataFrame(embeddings, columns=embed_cols)

            if self.gdf is not None:
                temp_gdf = self.gdf.copy()
                for col in embed_cols:
                    temp_gdf[col] = embed_df[col]

                skater = Skater(
                    gdf=temp_gdf,
                    w=w,
                    attrs_name=embed_cols,
                    n_clusters=self.n_clusters or 10,
                    random_state=self.random_state,
                )
                skater.fit()
                labels = skater.labels_
            else:
                # Fallback approach
                raise ValueError("SKATER requires GeoDataFrame")

        except Exception as e:
            logger.warning(f"SKATER failed: {e}, using agglomerative fallback")
            return self._fit_agglomerative(embeddings)

        return torch.from_numpy(labels).to(torch.long)

    def _fit_maxp(self, embeddings: np.ndarray, w: Optional[weights.W]) -> torch.Tensor:
        """Fit Max-P regionalization."""
        if w is None:
            raise ValueError("Max-P requires spatial weights")

        try:
            # Calculate homogeneity threshold if not provided
            threshold = self.homogeneity_threshold
            if threshold is None:
                # Use embedding variance as proxy for homogeneity
                threshold = np.var(embeddings) * embeddings.shape[1]

            maxp = MaxPHeuristic(
                gdf=self.gdf,
                w=w,
                attrs_name=None,
                threshold_name=None,
                threshold=threshold,
                top_n=self.min_size,
                random_state=self.random_state,
            )

            # Similar approach as SKATER - requires GDF with embedding attributes
            if self.gdf is not None:
                import pandas as pd

                embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
                embed_df = pd.DataFrame(embeddings, columns=embed_cols)

                temp_gdf = self.gdf.copy()
                for col in embed_cols:
                    temp_gdf[col] = embed_df[col]

                # Use sum of embeddings as threshold variable
                temp_gdf["embed_sum"] = temp_gdf[embed_cols].sum(axis=1)

                maxp = MaxPHeuristic(
                    gdf=temp_gdf,
                    w=w,
                    attrs_name=embed_cols,
                    threshold_name="embed_sum",
                    threshold=threshold,
                    top_n=self.min_size,
                    random_state=self.random_state,
                )
                maxp.fit()
                labels = maxp.labels_
            else:
                raise ValueError("Max-P requires GeoDataFrame")

        except Exception as e:
            logger.warning(f"Max-P failed: {e}, using agglomerative fallback")
            return self._fit_agglomerative(embeddings)

        return torch.from_numpy(labels).to(torch.long)

    def _fit_spenc(
        self, embeddings: np.ndarray, w: Optional[weights.W]
    ) -> torch.Tensor:
        """Fit SPENC regionalization."""
        try:
            spenc = Spenc(
                n_clusters=self.n_clusters or 10,
                gamma=self.alpha,
                random_state=self.random_state,
            )

            # SPENC can work directly with embeddings and connectivity
            if w is not None:
                connectivity = w.sparse.toarray()
            else:
                connectivity = None

            labels = spenc.fit_predict(embeddings, connectivity=connectivity)

        except Exception as e:
            logger.warning(f"SPENC failed: {e}, using agglomerative fallback")
            return self._fit_agglomerative(embeddings)

        return torch.from_numpy(labels).to(torch.long)

    def _fit_agglomerative(self, embeddings: np.ndarray) -> torch.Tensor:
        """Fallback agglomerative clustering."""
        connectivity = None
        if self.connectivity_matrix is not None:
            connectivity = self.connectivity_matrix.detach().cpu().numpy()

        clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters or 10,
            connectivity=connectivity,
            linkage="ward",
        )

        labels = clusterer.fit_predict(embeddings)
        return torch.from_numpy(labels).to(torch.long)

    def get_cluster_centers(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute region center embeddings.

        Args:
            embeddings: Node embeddings
            labels: Region labels

        Returns:
            Region center embeddings [n_regions, embed_dim]
        """
        unique_labels = torch.unique(labels)
        centers = []

        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                center = embeddings[mask].mean(dim=0)
                centers.append(center)
            else:
                # Handle empty regions
                centers.append(torch.zeros_like(embeddings[0]))

        return torch.stack(centers)

    def get_regionalization_stats(self, labels: torch.Tensor) -> dict[str, float]:
        """
        Compute regionalization quality statistics.

        Args:
            labels: Region labels

        Returns:
            Dictionary with quality metrics
        """
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)

        # Basic statistics
        stats = {
            "n_regions": len(unique_labels),
            "mean_region_size": len(labels_np) / len(unique_labels),
            "min_region_size": min(
                np.sum(labels_np == label) for label in unique_labels
            ),
            "max_region_size": max(
                np.sum(labels_np == label) for label in unique_labels
            ),
        }

        # Size balance (coefficient of variation)
        region_sizes = [np.sum(labels_np == label) for label in unique_labels]
        stats["size_cv"] = np.std(region_sizes) / np.mean(region_sizes)

        return stats


class RegionEmbedder(nn.Module):
    """
    Main region embedding model that orchestrates GCN training with community loss.

    Implements the full region2vec pipeline: GCN encoding + community-oriented
    loss + optional clustering for hierarchical modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        conv_type: str = "gcn",
        activation: str = "relu",
        dropout: float = 0.5,
        normalize_embeddings: bool = True,
        loss_config: Optional[dict[str, Any]] = None,
        clustering_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize RegionEmbedder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embed_dim: Output embedding dimension
            conv_type: GCN convolution type
            activation: Activation function
            dropout: Dropout probability
            normalize_embeddings: Whether to normalize embeddings
            loss_config: Community loss configuration
            clustering_config: Clustering configuration
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.normalize_embeddings = normalize_embeddings

        # GCN encoder
        self.gcn = RegionGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            conv_type=conv_type,
            activation=activation,
            dropout=dropout,
            normalize=normalize_embeddings,
        )

        # Community-oriented loss
        self.loss_fn = create_community_loss(loss_config or {})

        # Optional regionalization
        self.clustering_config = (
            clustering_config  # Keep name for backward compatibility
        )
        self._regionalizer = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get embeddings.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            Node embeddings [num_nodes, embed_dim]
        """
        return self.gcn(x, edge_index)

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        flow_matrix: torch.Tensor,
        num_negative_samples: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute community-oriented loss for training.

        Args:
            x: Node features
            edge_index: Graph edges
            flow_matrix: Flow matrix S̄
            num_negative_samples: Number of negative samples

        Returns:
            Dictionary with loss components
        """
        # Get embeddings
        embeddings = self.forward(x, edge_index)

        # Compute community loss
        loss_dict = self.loss_fn(
            embeddings, flow_matrix, edge_index, num_negative_samples
        )

        return loss_dict

    def get_static_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get static region embeddings for downstream forecasting.

        Args:
            x: Node features
            edge_index: Graph edges

        Returns:
            Static embeddings matrix Z [num_nodes, embed_dim]
        """
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings

    def fit_clustering(
        self,
        embeddings: torch.Tensor,
        connectivity_matrix: Optional[torch.Tensor] = None,
        spatial_weights: Optional[weights.W] = None,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit spatial regionalization on embeddings using PySAL methods.

        Args:
            embeddings: Node embeddings
            connectivity_matrix: Spatial connectivity matrix
            spatial_weights: PySAL spatial weights object
            gdf: GeoDataFrame for spatial context

        Returns:
            Tuple of (region_labels, region_centers)
        """
        if self.clustering_config is None:
            raise ValueError("Regionalization not configured")

        # Initialize regionalizer if needed
        if self._regionalizer is None:
            self._regionalizer = SpatialRegionalizer(
                regionalization_method=self.clustering_config.get("method", "spenc"),
                n_clusters=self.clustering_config.get("n_clusters", 10),
                min_size=self.clustering_config.get("min_size", 5),
                connectivity_matrix=connectivity_matrix,
                spatial_weights=spatial_weights,
                gdf=gdf,
                homogeneity_threshold=self.clustering_config.get(
                    "homogeneity_threshold"
                ),
                alpha=self.clustering_config.get("alpha", 0.1),
                random_state=self.clustering_config.get("random_state", 42),
            )

        # Fit and predict
        region_labels = self._regionalizer.fit_predict(embeddings)
        region_centers = self._regionalizer.get_cluster_centers(
            embeddings, region_labels
        )

        return region_labels, region_centers

    def precompute_embeddings_and_clusters(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        connectivity_matrix: Optional[torch.Tensor] = None,
        spatial_weights: Optional[weights.W] = None,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Precompute embeddings and optional regionalization for downstream use.

        Args:
            x: Node features
            edge_index: Graph edges
            connectivity_matrix: Spatial connectivity matrix
            spatial_weights: PySAL spatial weights object
            gdf: GeoDataFrame for spatial context

        Returns:
            Dictionary with embeddings and optional regionalization results
        """
        # Get embeddings
        embeddings = self.get_static_embeddings(x, edge_index)

        results = {"embeddings": embeddings}

        # Optional regionalization
        if self.clustering_config is not None:
            region_labels, region_centers = self.fit_clustering(
                embeddings, connectivity_matrix, spatial_weights, gdf
            )
            results.update(
                {
                    "cluster_labels": region_labels,  # Keep key name for compatibility
                    "cluster_centers": region_centers,
                    "n_clusters": len(region_centers),
                }
            )

            # Add regionalization quality statistics if possible
            if hasattr(self._regionalizer, "get_regionalization_stats"):
                stats = self._regionalizer.get_regionalization_stats(region_labels)
                results["regionalization_stats"] = stats

        return results


def create_region_embedder(config: dict[str, Any]) -> RegionEmbedder:
    """
    Factory function to create RegionEmbedder from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured RegionEmbedder instance
    """
    # Model architecture
    model_config = config.get("model", {})

    # Loss configuration
    loss_config = config.get("loss", {})

    # Clustering configuration
    clustering_config = config.get("clustering")

    return RegionEmbedder(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config.get("hidden_dim", 128),
        embed_dim=model_config.get("embed_dim", 64),
        conv_type=model_config.get("conv_type", "gcn"),
        activation=model_config.get("activation", "relu"),
        dropout=model_config.get("dropout", 0.5),
        normalize_embeddings=model_config.get("normalize_embeddings", True),
        loss_config=loss_config,
        clustering_config=clustering_config,
    )


if __name__ == "__main__":
    # Example usage and testing

    # Create dummy data
    num_nodes = 100
    input_dim = 32
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    flow_matrix = torch.rand(num_nodes, num_nodes) * 50

    # Model configuration with enhanced PySAL features
    config = {
        "model": {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "embed_dim": 32,
        },
        "loss": {
            "temperature": 0.1,
            "spatial_weight": 0.5,
            # Enhanced spatial autocorrelation parameters
            "autocorrelation_weight": 0.3,
            "moran_weight": 1.0,
            "lisa_weight": 0.5,
            "target_moran_i": 0.3,
            "smoothness_weight": 0.1,
        },
        "clustering": {
            "method": "spenc",  # PySAL regionalization method
            "n_clusters": 5,
            "min_size": 3,
            "alpha": 0.1,
            "random_state": 42,
        },
    }

    # Create model
    embedder = create_region_embedder(config)

    # Test forward pass
    embeddings = embedder(x, edge_index)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test loss computation
    loss_dict = embedder.compute_loss(x, edge_index, flow_matrix)
    print(f"Total loss: {loss_dict['total_loss']:.4f}")

    # Test static embeddings and clustering
    results = embedder.precompute_embeddings_and_clusters(x, edge_index)
    print(f"Static embeddings shape: {results['embeddings'].shape}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Cluster labels shape: {results['cluster_labels'].shape}")

    print("Region embedding models initialized successfully")
