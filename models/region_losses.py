"""
Loss functions for region embedding training based on region2vec approach.

Implements community-oriented contrastive loss with spatial contiguity priors
for unsupervised geography-aware pretraining.
"""

import logging
from typing import Optional

import esda
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PySAL imports for spatial autocorrelation
from libpysal import weights
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# from torch_geometric.utils import shortest_path  # Not available in current PyTorch Geometric version

logger = logging.getLogger(__name__)


def _convert_edge_index_to_scipy_sparse(
    edge_index: torch.Tensor, num_nodes: int
) -> csr_matrix:
    """
    Convert PyTorch Geometric edge_index format to SciPy sparse matrix.

    Args:
        edge_index: Edge connectivity tensor [2, num_edges]
        num_nodes: Number of nodes in the graph

    Returns:
        SciPy CSR sparse matrix representing the adjacency matrix
    """
    # Convert to numpy arrays
    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    # Create data array (all ones for unweighted graph)
    data = np.ones(len(row), dtype=np.float64)

    # Create SciPy sparse matrix
    adjacency_matrix = csr_matrix(
        (data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float64
    )

    return adjacency_matrix


class SpatialContiguityPrior:
    """
    Computes hop distances between nodes for spatial contiguity enforcement.
    """

    def __init__(self, max_hops: int = 5, cache_distances: bool = True):
        """
        Initialize spatial contiguity prior.

        Args:
            max_hops: Maximum hop distance to compute
            cache_distances: Whether to cache hop distance matrix
        """
        self.max_hops = max_hops
        self.cache_distances = cache_distances
        self._cached_distances = None
        self._cached_edge_index = None

    def compute_hop_distances(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute hop distances between all node pairs.

        Args:
            edge_index: Graph edges [2, num_edges]

        Returns:
            Hop distance matrix [num_nodes, num_nodes]
        """
        # Check cache
        if (
            self.cache_distances
            and self._cached_distances is not None
            and torch.equal(edge_index, self._cached_edge_index)
        ):
            return self._cached_distances

        num_nodes = edge_index.max().item() + 1

        # Convert edge_index to SciPy sparse matrix
        adjacency_matrix = _convert_edge_index_to_scipy_sparse(edge_index, num_nodes)

        # Compute shortest paths using SciPy's optimized algorithms
        try:
            # Use Floyd-Warshall for all-pairs shortest paths
            distances_np = shortest_path(
                csgraph=adjacency_matrix,
                method="FW",  # Floyd-Warshall algorithm
                directed=False,  # Treat as undirected graph
                unweighted=True,  # Compute hop distances
                return_predecessors=False,
            )

            # Convert back to PyTorch tensor
            distances = torch.from_numpy(distances_np).float()

            # Handle unreachable nodes and clamp to max_hops
            distances[distances == np.inf] = self.max_hops + 1
            distances = torch.clamp(distances, max=self.max_hops + 1)

        except Exception as e:
            logger.warning(
                f"SciPy shortest_path failed: {e}. Using fallback implementation."
            )
            # Fallback: use simple adjacency-based distances
            distances = torch.full(
                (num_nodes, num_nodes), self.max_hops + 1, dtype=torch.float
            )
            distances.fill_diagonal_(0)

            # Set direct neighbors to distance 1
            row, col = edge_index
            distances[row, col] = 1
            distances[col, row] = 1

        # Cache if enabled
        if self.cache_distances:
            self._cached_distances = distances
            self._cached_edge_index = edge_index.clone()

        return distances

    def get_negative_pairs_by_distance(
        self, edge_index: torch.Tensor, hop_threshold: int = 2
    ) -> torch.Tensor:
        """
        Get node pairs separated by more than hop_threshold hops.

        Args:
            edge_index: Graph edges
            hop_threshold: Minimum hop distance for negative pairs

        Returns:
            Negative pair indices [2, num_negative_pairs]
        """
        distances = self.compute_hop_distances(edge_index)

        # Find pairs with distance > hop_threshold
        negative_mask = distances > hop_threshold
        negative_indices = negative_mask.nonzero(as_tuple=False).T

        return negative_indices


class FlowWeightedContrastiveLoss(nn.Module):
    """
    Contrastive loss weighted by flow intensities (log-transformed).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Initialize flow-weighted contrastive loss.

        Args:
            temperature: Temperature parameter for contrastive loss
            margin: Margin for negative pairs
            reduction: Loss reduction method
            eps: Small value to avoid log(0)
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        flow_weights: torch.Tensor,
        negative_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow-weighted contrastive loss.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            positive_pairs: Positive pair indices [2, num_pos_pairs]
            flow_weights: Flow weights for positive pairs [num_pos_pairs]
            negative_pairs: Negative pair indices [2, num_neg_pairs]

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Positive pair similarities
        pos_i, pos_j = positive_pairs
        pos_similarities = (embeddings[pos_i] * embeddings[pos_j]).sum(dim=1)
        pos_similarities = pos_similarities / self.temperature

        # Negative pair similarities
        neg_i, neg_j = negative_pairs
        neg_similarities = (embeddings[neg_i] * embeddings[neg_j]).sum(dim=1)
        neg_similarities = neg_similarities / self.temperature

        # Positive loss (attraction) weighted by log-flow
        log_flow_weights = torch.log(flow_weights + self.eps)
        pos_loss = -pos_similarities * log_flow_weights

        # Negative loss (repulsion) with margin
        neg_loss = torch.relu(neg_similarities - self.margin)

        # Combine losses
        total_pos_loss = pos_loss.sum() if pos_loss.numel() > 0 else 0.0
        total_neg_loss = neg_loss.sum() if neg_loss.numel() > 0 else 0.0

        total_loss = total_pos_loss + total_neg_loss

        # Apply reduction
        if self.reduction == "mean":
            total_pairs = positive_pairs.size(1) + negative_pairs.size(1)
            if total_pairs > 0:
                total_loss = total_loss / total_pairs
        elif self.reduction == "sum":
            pass  # Keep as is
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return total_loss


class SpatialAutocorrelationLoss(nn.Module):
    """
    Spatial autocorrelation loss based on PySAL Moran's I and local statistics.

    Encourages spatial autocorrelation in embeddings by penalizing low global
    Moran's I values and adding LISA-based local smoothness terms.
    """

    def __init__(
        self,
        moran_weight: float = 1.0,
        lisa_weight: float = 0.5,
        target_moran_i: float = 0.3,
        smoothness_weight: float = 0.1,
        spatial_weights: Optional[weights.W] = None,
        device: str = "cpu",
    ):
        """
        Initialize spatial autocorrelation loss.

        Args:
            moran_weight: Weight for global Moran's I term
            lisa_weight: Weight for local LISA term
            target_moran_i: Target value for global Moran's I
            smoothness_weight: Weight for embedding smoothness term
            spatial_weights: PySAL spatial weights object
            device: Device for computations
        """
        super().__init__()
        self.moran_weight = moran_weight
        self.lisa_weight = lisa_weight
        self.target_moran_i = target_moran_i
        self.smoothness_weight = smoothness_weight
        self.spatial_weights = spatial_weights
        self.device = device

    def _compute_moran_i_loss(
        self, embeddings: torch.Tensor, spatial_weights: weights.W
    ) -> torch.Tensor:
        """
        Compute loss based on global Moran's I statistics.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            spatial_weights: PySAL spatial weights

        Returns:
            Moran's I based loss
        """
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            n_nodes, embed_dim = embeddings_np.shape

            moran_losses = []

            # Compute Moran's I for each embedding dimension
            for dim in range(embed_dim):
                try:
                    moran = esda.Moran(embeddings_np[:, dim], spatial_weights)
                    # Penalty: encourage positive spatial autocorrelation
                    moran_penalty = (self.target_moran_i - moran.I) ** 2
                    moran_losses.append(moran_penalty)
                except Exception as e:
                    logger.warning(f"Moran's I computation failed for dim {dim}: {e}")
                    moran_losses.append(0.0)

            # Average across dimensions
            avg_moran_loss = np.mean(moran_losses)
            return torch.tensor(avg_moran_loss, device=embeddings.device)

        except Exception as e:
            logger.warning(f"Global Moran's I loss computation failed: {e}")
            return torch.tensor(0.0, device=embeddings.device)

    def _compute_lisa_loss(
        self, embeddings: torch.Tensor, spatial_weights: weights.W
    ) -> torch.Tensor:
        """
        Compute loss based on local indicators of spatial association (LISA).

        Args:
            embeddings: Node embeddings
            spatial_weights: PySAL spatial weights

        Returns:
            LISA-based loss
        """
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            n_nodes, embed_dim = embeddings_np.shape

            lisa_losses = []

            # Compute LISA for each embedding dimension
            for dim in range(embed_dim):
                try:
                    lisa = esda.Moran_Local(embeddings_np[:, dim], spatial_weights)

                    # LISA loss: penalize negative local autocorrelation
                    # Focus on high-high and low-low clusters (positive autocorr)
                    negative_lisa_mask = lisa.Is < 0
                    if negative_lisa_mask.sum() > 0:
                        lisa_penalty = np.mean(np.abs(lisa.Is[negative_lisa_mask]))
                        lisa_losses.append(lisa_penalty)
                    else:
                        lisa_losses.append(0.0)

                except Exception as e:
                    logger.warning(f"LISA computation failed for dim {dim}: {e}")
                    lisa_losses.append(0.0)

            # Average across dimensions
            avg_lisa_loss = np.mean(lisa_losses)
            return torch.tensor(avg_lisa_loss, device=embeddings.device)

        except Exception as e:
            logger.warning(f"LISA loss computation failed: {e}")
            return torch.tensor(0.0, device=embeddings.device)

    def _compute_embedding_smoothness_loss(
        self, embeddings: torch.Tensor, spatial_weights: weights.W
    ) -> torch.Tensor:
        """
        Compute embedding smoothness loss using spatial lags.

        Args:
            embeddings: Node embeddings
            spatial_weights: PySAL spatial weights

        Returns:
            Smoothness loss
        """
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            n_nodes, embed_dim = embeddings_np.shape

            smoothness_losses = []

            # Compute spatial lag differences for each dimension
            for dim in range(embed_dim):
                try:
                    # Compute spatial lag
                    spatial_lag = weights.lag_spatial(
                        spatial_weights, embeddings_np[:, dim]
                    )

                    # Smoothness: minimize difference between value and spatial lag
                    differences = embeddings_np[:, dim] - spatial_lag
                    smoothness_loss = np.mean(differences**2)
                    smoothness_losses.append(smoothness_loss)

                except Exception as e:
                    logger.warning(f"Smoothness computation failed for dim {dim}: {e}")
                    smoothness_losses.append(0.0)

            # Average across dimensions
            avg_smoothness_loss = np.mean(smoothness_losses)
            return torch.tensor(avg_smoothness_loss, device=embeddings.device)

        except Exception as e:
            logger.warning(f"Smoothness loss computation failed: {e}")
            return torch.tensor(0.0, device=embeddings.device)

    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_weights: Optional[weights.W] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute spatial autocorrelation loss components.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            edge_index: Graph edges [2, num_edges]
            spatial_weights: Optional spatial weights (uses self.spatial_weights if None)

        Returns:
            Dictionary with loss components
        """
        # Use provided spatial weights or create from edge_index
        w = spatial_weights or self.spatial_weights

        if w is None:
            # Create weights from edge_index as fallback
            try:
                num_nodes = embeddings.size(0)
                adjacency_matrix = _convert_edge_index_to_scipy_sparse(
                    edge_index, num_nodes
                )
                w = weights.W(adjacency_matrix)
            except Exception as e:
                logger.warning(f"Failed to create spatial weights: {e}")
                # Return zero losses if no weights available
                return {
                    "spatial_autocorr_loss": torch.tensor(
                        0.0, device=embeddings.device
                    ),
                    "moran_loss": torch.tensor(0.0, device=embeddings.device),
                    "lisa_loss": torch.tensor(0.0, device=embeddings.device),
                    "smoothness_loss": torch.tensor(0.0, device=embeddings.device),
                }

        # Compute loss components
        moran_loss = self._compute_moran_i_loss(embeddings, w)
        lisa_loss = self._compute_lisa_loss(embeddings, w)
        smoothness_loss = self._compute_embedding_smoothness_loss(embeddings, w)

        # Combine losses
        total_spatial_loss = (
            self.moran_weight * moran_loss
            + self.lisa_weight * lisa_loss
            + self.smoothness_weight * smoothness_loss
        )

        return {
            "spatial_autocorr_loss": total_spatial_loss,
            "moran_loss": moran_loss,
            "lisa_loss": lisa_loss,
            "smoothness_loss": smoothness_loss,
        }


class CommunityOrientedLoss(nn.Module):
    """
    Community-oriented loss function from region2vec approach.

    Combines flow-weighted contrastive loss with spatial contiguity prior.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
        spatial_weight: float = 1.0,
        hop_threshold: int = 2,
        max_hops: int = 5,
        min_flow_threshold: float = 1.0,
        reduction: str = "mean",
        # Spatial autocorrelation parameters
        autocorrelation_weight: float = 0.5,
        moran_weight: float = 1.0,
        lisa_weight: float = 0.5,
        target_moran_i: float = 0.3,
        smoothness_weight: float = 0.1,
        spatial_weights: Optional[weights.W] = None,
    ):
        """
        Initialize community-oriented loss with spatial autocorrelation.

        Args:
            temperature: Temperature for contrastive loss
            margin: Margin for negative pairs
            spatial_weight: Weight for spatial contiguity penalty
            hop_threshold: Hop threshold for spatial negative pairs
            max_hops: Maximum hops for distance computation
            min_flow_threshold: Minimum flow to consider positive pairs
            reduction: Loss reduction method
            autocorrelation_weight: Weight for spatial autocorrelation loss
            moran_weight: Weight for global Moran's I term
            lisa_weight: Weight for local LISA term
            target_moran_i: Target value for global Moran's I
            smoothness_weight: Weight for embedding smoothness term
            spatial_weights: PySAL spatial weights object
        """
        super().__init__()

        self.temperature = temperature
        self.margin = margin
        self.spatial_weight = spatial_weight
        self.hop_threshold = hop_threshold
        self.min_flow_threshold = min_flow_threshold
        self.reduction = reduction
        self.autocorrelation_weight = autocorrelation_weight

        # Components
        self.contrastive_loss = FlowWeightedContrastiveLoss(
            temperature=temperature, margin=margin, reduction=reduction
        )
        self.spatial_prior = SpatialContiguityPrior(max_hops=max_hops)

        # Spatial autocorrelation component
        self.spatial_autocorr_loss = SpatialAutocorrelationLoss(
            moran_weight=moran_weight,
            lisa_weight=lisa_weight,
            target_moran_i=target_moran_i,
            smoothness_weight=smoothness_weight,
            spatial_weights=spatial_weights,
        )

    def _get_positive_pairs(
        self, flow_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract positive pairs and their flow weights from flow matrix.

        Args:
            flow_matrix: Flow matrix S̄ [num_nodes, num_nodes]

        Returns:
            Tuple of (positive_pairs [2, num_pairs], flow_weights [num_pairs])
        """
        # Find positive pairs where flow > min_threshold
        positive_mask = flow_matrix > self.min_flow_threshold
        positive_indices = positive_mask.nonzero(as_tuple=False).T

        if positive_indices.size(1) == 0:
            # Return empty tensors if no positive pairs
            return (
                torch.empty((2, 0), dtype=torch.long, device=flow_matrix.device),
                torch.empty((0,), dtype=torch.float, device=flow_matrix.device),
            )

        # Extract flow weights for positive pairs
        i_indices, j_indices = positive_indices
        flow_weights = flow_matrix[i_indices, j_indices]

        return positive_indices, flow_weights

    def _get_negative_pairs(
        self,
        flow_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        num_negative_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get negative pairs based on zero flow and/or spatial distance.

        Args:
            flow_matrix: Flow matrix S̄
            edge_index: Graph adjacency edges
            num_negative_samples: Number of negative samples (None = all)

        Returns:
            Negative pair indices [2, num_neg_pairs]
        """
        num_nodes = flow_matrix.size(0)

        # Zero-flow negative pairs
        zero_flow_mask = flow_matrix <= self.min_flow_threshold
        zero_flow_indices = zero_flow_mask.nonzero(as_tuple=False).T

        # Spatial distance negative pairs
        spatial_negatives = self.spatial_prior.get_negative_pairs_by_distance(
            edge_index, hop_threshold=self.hop_threshold
        )

        # Combine negative pairs
        if zero_flow_indices.size(1) > 0 and spatial_negatives.size(1) > 0:
            # Union of zero-flow and spatial negatives
            all_negatives = torch.cat([zero_flow_indices, spatial_negatives], dim=1)
            # Remove duplicates
            all_negatives = torch.unique(all_negatives, dim=1)
        elif zero_flow_indices.size(1) > 0:
            all_negatives = zero_flow_indices
        elif spatial_negatives.size(1) > 0:
            all_negatives = spatial_negatives
        else:
            # Fallback: random negative pairs
            all_pairs = torch.combinations(torch.arange(num_nodes), r=2).T
            all_negatives = all_pairs

        # Sample if requested
        if (
            num_negative_samples is not None
            and all_negatives.size(1) > num_negative_samples
        ):
            perm = torch.randperm(all_negatives.size(1))[:num_negative_samples]
            all_negatives = all_negatives[:, perm]

        return all_negatives

    def forward(
        self,
        embeddings: torch.Tensor,
        flow_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        num_negative_samples: Optional[int] = None,
    ) -> dict:
        """
        Compute community-oriented loss.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            flow_matrix: Flow matrix S̄ [num_nodes, num_nodes]
            edge_index: Graph adjacency edges [2, num_edges]
            num_negative_samples: Number of negative samples

        Returns:
            Dictionary with loss components
        """
        # Get positive and negative pairs
        positive_pairs, flow_weights = self._get_positive_pairs(flow_matrix)
        negative_pairs = self._get_negative_pairs(
            flow_matrix, edge_index, num_negative_samples
        )

        # Compute contrastive loss
        if positive_pairs.size(1) > 0 and negative_pairs.size(1) > 0:
            contrastive_loss = self.contrastive_loss(
                embeddings, positive_pairs, flow_weights, negative_pairs
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=embeddings.device)

        # Spatial contiguity penalty (optional additional term)
        spatial_penalty = self._compute_spatial_penalty(embeddings, edge_index)

        # Spatial autocorrelation loss
        spatial_autocorr_dict = self.spatial_autocorr_loss(embeddings, edge_index)

        # Total loss
        total_loss = (
            contrastive_loss
            + self.spatial_weight * spatial_penalty
            + self.autocorrelation_weight
            * spatial_autocorr_dict["spatial_autocorr_loss"]
        )

        # Combine all loss components
        loss_dict = {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "spatial_penalty": spatial_penalty,
            "num_positive_pairs": positive_pairs.size(1),
            "num_negative_pairs": negative_pairs.size(1),
        }

        # Add spatial autocorrelation components
        loss_dict.update(spatial_autocorr_dict)

        return loss_dict

    def _compute_spatial_penalty(
        self, embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial contiguity penalty to encourage smooth embeddings.

        Args:
            embeddings: Node embeddings
            edge_index: Graph edges

        Returns:
            Spatial penalty value
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute embedding differences for adjacent nodes
        source_nodes, target_nodes = edge_index
        source_embeds = embeddings[source_nodes]
        target_embeds = embeddings[target_nodes]

        # L2 difference between adjacent nodes
        embedding_diffs = torch.norm(source_embeds - target_embeds, p=2, dim=1)

        # Mean squared difference (encourages smoothness)
        spatial_penalty = torch.mean(embedding_diffs.pow(2))

        return spatial_penalty


def create_community_loss(config: dict) -> CommunityOrientedLoss:
    """
    Factory function to create community-oriented loss from config.

    Args:
        config: Loss configuration dictionary

    Returns:
        Configured CommunityOrientedLoss instance
    """
    return CommunityOrientedLoss(
        temperature=config.get("temperature", 0.1),
        margin=config.get("margin", 1.0),
        spatial_weight=config.get("spatial_weight", 1.0),
        hop_threshold=config.get("hop_threshold", 2),
        max_hops=config.get("max_hops", 5),
        min_flow_threshold=config.get("min_flow_threshold", 1.0),
        reduction=config.get("reduction", "mean"),
        # Spatial autocorrelation parameters
        autocorrelation_weight=config.get("autocorrelation_weight", 0.5),
        moran_weight=config.get("moran_weight", 1.0),
        lisa_weight=config.get("lisa_weight", 0.5),
        target_moran_i=config.get("target_moran_i", 0.3),
        smoothness_weight=config.get("smoothness_weight", 0.1),
        spatial_weights=config.get("spatial_weights"),
    )


if __name__ == "__main__":
    # Example usage and testing

    # Create dummy data
    num_nodes = 50
    embed_dim = 64
    embeddings = torch.randn(num_nodes, embed_dim)
    flow_matrix = torch.rand(num_nodes, num_nodes) * 100
    edge_index = torch.randint(0, num_nodes, (2, 200))

    # Create loss function
    loss_fn = CommunityOrientedLoss()

    # Compute loss
    loss_dict = loss_fn(embeddings, flow_matrix, edge_index)

    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print(f"Contrastive loss: {loss_dict['contrastive_loss']:.4f}")
    print(f"Spatial penalty: {loss_dict['spatial_penalty']:.4f}")
    print(f"Positive pairs: {loss_dict['num_positive_pairs']}")
    print(f"Negative pairs: {loss_dict['num_negative_pairs']}")

    print("Region embedding losses initialized successfully")
