"""
Loss functions for region embedding training based on region2vec approach.

Implements community-oriented contrastive loss with spatial contiguity priors
for unsupervised geography-aware pretraining.
"""

import logging
from collections import deque

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


def _create_spatial_weights_from_edge_index(
    edge_index: torch.Tensor, num_nodes: int | None = None
) -> weights.W:
    """Create PySAL weights from an ``edge_index`` tensor.

    When the contiguity graph contains disconnected components, PySAL emits a
    warning about ``islands``. We silence the built-in warning and surface a
    single structured log message instead so training logs stay readable.
    """

    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    inferred_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    total_nodes = num_nodes if num_nodes is not None else inferred_nodes
    if total_nodes == 0:
        raise ValueError("edge_index is empty; cannot build spatial weights")

    neighbors: dict[int, list[int]] = {i: [] for i in range(total_nodes)}
    for src, dst in zip(row, col, strict=False):
        src_i, dst_i = int(src), int(dst)
        if src_i == dst_i:
            continue
        if dst_i not in neighbors[src_i]:
            neighbors[src_i].append(dst_i)
        if src_i not in neighbors[dst_i]:
            neighbors[dst_i].append(src_i)

    w = weights.W(neighbors, silence_warnings=True)
    if getattr(w, "islands", None):
        logger.warning(
            "Spatial weights contain %d island component(s) with no contiguity edges: %s",
            len(w.islands),
            w.islands,
        )
    return w


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
        self._cached_num_nodes: int | None = None

    def compute_hop_distances(
        self, edge_index: torch.Tensor, num_nodes: int | None = None
    ) -> torch.Tensor:
        """
        Compute hop distances between all node pairs.

        Args:
            edge_index: Graph edges [2, num_edges]
            num_nodes: Total number of nodes in the graph. When ``None`` the
                value is inferred from ``edge_index``.

        Returns:
            Hop distance matrix [num_nodes, num_nodes]
        """
        # Check cache
        if (
            self.cache_distances
            and self._cached_distances is not None
            and self._cached_edge_index is not None
            and torch.equal(edge_index, self._cached_edge_index)  # type: ignore[arg-type]
            and (num_nodes is None or num_nodes == self._cached_num_nodes)
        ):
            return self._cached_distances

        if num_nodes is None:
            if edge_index.numel() == 0:
                raise ValueError("Cannot infer num_nodes from an empty edge_index")
            num_nodes = int(edge_index.max().item()) + 1

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

        except Exception as e:
            logger.warning(
                f"SciPy shortest_path failed: {e}. Using fallback implementation."
            )
            distances = torch.full(
                (num_nodes, num_nodes), float("inf"), dtype=torch.float
            )
            distances.fill_diagonal_(0)

            adjacency_lists: list[list[int]] = [[] for _ in range(num_nodes)]
            row, col = edge_index
            for src, dst in zip(row.tolist(), col.tolist(), strict=False):
                if src == dst:
                    continue
                adjacency_lists[src].append(dst)
                adjacency_lists[dst].append(src)

            for start in range(num_nodes):
                queue: deque[int] = deque([start])
                while queue:
                    node = queue.popleft()
                    for neighbor in adjacency_lists[node]:
                        if torch.isfinite(distances[start, neighbor]):
                            continue
                        distances[start, neighbor] = distances[start, node] + 1
                        queue.append(neighbor)

        finite_mask = torch.isfinite(distances)
        distances[finite_mask] = torch.clamp(
            distances[finite_mask], max=float(self.max_hops)
        )

        # Cache if enabled
        if self.cache_distances:
            self._cached_distances = distances
            self._cached_edge_index = edge_index.clone()
            self._cached_num_nodes = num_nodes

        return distances

    def get_negative_pairs_by_distance(
        self,
        edge_index: torch.Tensor,
        hop_threshold: int = 2,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """
        Get node pairs separated by more than hop_threshold hops.

        Args:
            edge_index: Graph edges
            hop_threshold: Minimum hop distance for negative pairs
            num_nodes: Optional explicit node count

        Returns:
            Negative pair indices [2, num_negative_pairs]
        """
        distances = self.compute_hop_distances(edge_index, num_nodes=num_nodes)

        finite = torch.isfinite(distances)
        within_max = distances <= self.max_hops
        negative_mask = (distances > hop_threshold) & finite & within_max
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
        total_pos_loss = pos_loss.sum() if pos_loss.numel() > 0 else torch.tensor(0.0)
        total_neg_loss = neg_loss.sum() if neg_loss.numel() > 0 else torch.tensor(0.0)

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
        spatial_weights: weights.W | None = None,
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

            # Early exit for insufficient nodes
            if n_nodes < 2:
                logger.debug(
                    f"Moran's I requires at least 2 regions, got {n_nodes}. Skipping."
                )
                return torch.tensor(0.0, device=embeddings.device)

            # Check for islands - they cause seI_sim = 0 in permutation simulation
            # which triggers RuntimeWarning: division by zero in esda/moran.py:1354
            if hasattr(spatial_weights, "islands") and spatial_weights.islands:
                logger.debug(
                    f"Skipping Moran's I: spatial weights contain {len(spatial_weights.islands)} "
                    f"island(s) {spatial_weights.islands} which cause unstable seI_sim computation"
                )
                return torch.tensor(0.0, device=embeddings.device)

            moran_losses = []
            eps = 1e-10  # Numerical stability threshold

            # Compute Moran's I for each embedding dimension
            for dim in range(embed_dim):
                values = embeddings_np[:, dim]

                # Pre-computation validation checks to prevent division by zero warnings
                value_std = float(np.std(values))
                n_unique = int(np.unique(values).size)

                # Skip dimensions with no variance (causes seI_sim = 0 or NaN)
                if value_std < eps:
                    logger.debug(
                        f"Skipping dim {dim}: constant values (std={value_std:.2e})"
                    )
                    moran_losses.append(0.0)
                    continue

                # Skip dimensions with insufficient unique values
                if n_unique < 2:
                    logger.debug(
                        f"Skipping dim {dim}: only {n_unique} unique value(s)"
                    )
                    moran_losses.append(0.0)
                    continue

                # Now safe to compute Moran's I
                try:
                    moran = esda.Moran(values, spatial_weights)
                    # Penalty: encourage positive spatial autocorrelation
                    moran_penalty = (self.target_moran_i - moran.I) ** 2
                    moran_losses.append(moran_penalty)
                except Exception as e:
                    logger.warning(f"Moran's I computation failed for dim {dim}: {e}")
                    moran_losses.append(0.0)

            # Average across dimensions
            avg_moran_loss = np.mean(moran_losses) if moran_losses else 0.0
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

            # Early exit for insufficient nodes
            if n_nodes < 2:
                logger.debug(
                    f"LISA requires at least 2 regions, got {n_nodes}. Skipping."
                )
                return torch.tensor(0.0, device=embeddings.device)

            # Check for islands - they cause unstable LISA computation
            if hasattr(spatial_weights, "islands") and spatial_weights.islands:
                logger.debug(
                    f"Skipping LISA: spatial weights contain {len(spatial_weights.islands)} "
                    f"island(s) {spatial_weights.islands}"
                )
                return torch.tensor(0.0, device=embeddings.device)

            lisa_losses = []
            eps = 1e-10  # Numerical stability threshold

            # Compute LISA for each embedding dimension
            for dim in range(embed_dim):
                values = embeddings_np[:, dim]

                # Pre-computation validation checks to prevent division by zero warnings
                value_std = float(np.std(values))
                n_unique = int(np.unique(values).size)

                # Skip dimensions with no variance (causes unstable LISA computation)
                if value_std < eps:
                    logger.debug(
                        f"Skipping dim {dim}: constant values (std={value_std:.2e})"
                    )
                    lisa_losses.append(0.0)
                    continue

                # Skip dimensions with insufficient unique values
                if n_unique < 2:
                    logger.debug(
                        f"Skipping dim {dim}: only {n_unique} unique value(s)"
                    )
                    lisa_losses.append(0.0)
                    continue

                # Now safe to compute LISA
                try:
                    lisa = esda.Moran_Local(values, spatial_weights)

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
            avg_lisa_loss = np.mean(lisa_losses) if lisa_losses else 0.0
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
            _n_nodes, embed_dim = embeddings_np.shape

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
        spatial_weights: weights.W | None = None,
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
            w = _create_spatial_weights_from_edge_index(
                edge_index, num_nodes=embeddings.size(0)
            )

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


class SpatialOnlyLoss(nn.Module):
    """
    Spatial-only loss function for region embedding training without mobility data.

    Uses K-nearest neighbors adjacency, spatial autocorrelation, and smoothness
    to drive learning of spatially meaningful embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
        knn_positive_weight: float = 0.8,
        hop_threshold: int = 2,
        max_hops: int = 5,
        k_neighbors: int = 8,
        reduction: str = "mean",
        # Enhanced spatial autocorrelation parameters
        autocorrelation_weight: float = 1.0,  # Main driving force
        moran_weight: float = 1.0,
        lisa_weight: float = 0.5,
        target_moran_i: float = 0.3,
        smoothness_weight: float = 0.2,  # Enhanced smoothness
        spatial_weights: weights.W | None = None,
        device: str = "cpu",
    ):
        """
        Initialize spatial-only loss with KNN adjacency and enhanced spatial autocorrelation.

        Args:
            temperature: Temperature for contrastive loss
            margin: Margin for negative pairs
            knn_positive_weight: Weight for KNN-based positive pairs
            hop_threshold: Hop threshold for spatial negative pairs
            max_hops: Maximum hops for distance computation
            k_neighbors: Number of nearest neighbors for positive pairs
            reduction: Loss reduction method
            autocorrelation_weight: Weight for spatial autocorrelation loss (main driver)
            moran_weight: Weight for global Moran's I term
            lisa_weight: Weight for local LISA term
            target_moran_i: Target value for global Moran's I
            smoothness_weight: Weight for embedding smoothness term (enhanced)
            spatial_weights: PySAL spatial weights object
            device: Device for computation
        """
        super().__init__()

        self.temperature = temperature
        self.margin = margin
        self.knn_positive_weight = knn_positive_weight
        self.hop_threshold = hop_threshold
        self.k_neighbors = k_neighbors
        self.reduction = reduction

        # Spatial components
        self.spatial_prior = SpatialContiguityPrior(max_hops=max_hops)
        self.autocorrelation_loss = SpatialAutocorrelationLoss(
            moran_weight=moran_weight * autocorrelation_weight,
            lisa_weight=lisa_weight * autocorrelation_weight,
            target_moran_i=target_moran_i,
            smoothness_weight=smoothness_weight,
            spatial_weights=spatial_weights,
            device=device,
        )

    def _get_knn_positive_pairs(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get positive pairs using K-nearest neighbors in embedding space.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]

        Returns:
            Tuple of (positive_pairs, positive_weights)
        """
        from sklearn.neighbors import NearestNeighbors

        embeddings_np = embeddings.detach().cpu().numpy()

        # Use sklearn for efficient KNN
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(embeddings_np)
        distances, indices = nbrs.kneighbors(embeddings_np)

        # Extract positive pairs (skip self connections)
        positive_pairs = []
        positive_weights = []

        for i, neighbors in enumerate(indices):
            for j, dist in zip(
                neighbors[1:], distances[i][1:], strict=False
            ):  # Skip self
                positive_pairs.append([i, j])
                # Use inverse distance as weight (closer = stronger positive)
                weight = 1.0 / (dist + 1e-8)
                positive_weights.append(weight)

        if not positive_pairs:
            # Fallback: use identity pairs
            n_nodes = embeddings.size(0)
            positive_pairs = [[i, i] for i in range(n_nodes)]
            positive_weights = [1.0] * n_nodes

        positive_pairs = torch.tensor(positive_pairs).t().to(embeddings.device)
        positive_weights = torch.tensor(positive_weights).to(embeddings.device)

        return positive_pairs, positive_weights

    def _get_distance_negative_pairs(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get negative pairs using hop distances from spatial graph.

        Args:
            edge_index: Graph edges [2, num_edges]

        Returns:
            Tuple of (negative_pairs, negative_weights)
        """
        negative_pairs = self.spatial_prior.get_negative_pairs_by_distance(
            edge_index,
            hop_threshold=self.hop_threshold,
            num_nodes=num_nodes,
        )

        # Uniform weights for negative pairs
        negative_weights = torch.ones(negative_pairs.size(1)).to(edge_index.device)

        return negative_pairs, negative_weights

    def _compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        positive_weights: torch.Tensor,
        negative_pairs: torch.Tensor,
        negative_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss using KNN positive pairs and distance negative pairs.

        Args:
            embeddings: Node embeddings
            positive_pairs: Positive pair indices [2, num_positive]
            positive_weights: Positive pair weights
            negative_pairs: Negative pair indices [2, num_negative]
            negative_weights: Negative pair weights

        Returns:
            Contrastive loss value
        """
        # Extract embeddings for positive and negative pairs
        pos_i, pos_j = positive_pairs
        neg_i, neg_j = negative_pairs

        pos_emb_i = embeddings[pos_i]
        pos_emb_j = embeddings[pos_j]
        neg_emb_i = embeddings[neg_i]
        neg_emb_j = embeddings[neg_j]

        # Compute similarities
        pos_similarities = F.cosine_similarity(pos_emb_i, pos_emb_j, dim=1)
        neg_similarities = F.cosine_similarity(neg_emb_i, neg_emb_j, dim=1)

        # Apply temperature scaling
        pos_similarities = pos_similarities / self.temperature
        neg_similarities = neg_similarities / self.temperature

        # Weighted positive loss (attraction)
        pos_loss = -pos_similarities * positive_weights

        # Weighted negative loss (repulsion with margin)
        neg_loss = F.relu(neg_similarities + self.margin) * negative_weights

        # Combine losses
        total_loss = (pos_loss.mean() + neg_loss.mean()) * self.knn_positive_weight

        return total_loss

    def forward(
        self,
        embeddings: torch.Tensor,
        flow_matrix: torch.Tensor,  # Ignored, kept for interface compatibility
        edge_index: torch.Tensor,
        num_negative_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute spatial-only loss for embeddings.

        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            flow_matrix: Flow matrix (ignored in spatial-only mode)
            edge_index: Graph edges [2, num_edges]
            num_negative_samples: Number of negative samples (ignored)

        Returns:
            Dictionary with loss components
        """
        # Get positive and negative pairs
        positive_pairs, positive_weights = self._get_knn_positive_pairs(embeddings)
        negative_pairs, negative_weights = self._get_distance_negative_pairs(
            edge_index, embeddings.size(0)
        )

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(
            embeddings,
            positive_pairs,
            positive_weights,
            negative_pairs,
            negative_weights,
        )

        # Compute spatial autocorrelation loss
        autocorr_loss_dict = self.autocorrelation_loss(embeddings)
        autocorr_loss = autocorr_loss_dict["total_loss"]

        # Combine losses
        total_loss = contrastive_loss + autocorr_loss

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "spatial_penalty": autocorr_loss,
            "num_positive_pairs": torch.tensor(positive_pairs.size(1)),
            "num_negative_pairs": torch.tensor(negative_pairs.size(1)),
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
        spatial_weights: weights.W | None = None,
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
        num_negative_samples: int | None = None,
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
            edge_index,
            hop_threshold=self.hop_threshold,
            num_nodes=num_nodes,
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
        num_negative_samples: int | None = None,
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


def create_community_loss(config: dict) -> SpatialOnlyLoss:
    """
    Factory function to create spatial-only loss from config.

    Args:
        config: Loss configuration dictionary

    Returns:
        Configured SpatialOnlyLoss instance
    """
    return SpatialOnlyLoss(
        temperature=config.get("temperature", 0.1),
        margin=config.get("margin", 1.0),
        knn_positive_weight=config.get("spatial_weight", 0.8),  # Reuse spatial_weight
        hop_threshold=config.get("hop_threshold", 2),
        max_hops=config.get("max_hops", 5),
        k_neighbors=config.get("k_neighbors", 8),
        reduction=config.get("reduction", "mean"),
        # Enhanced spatial autocorrelation parameters
        autocorrelation_weight=config.get("autocorrelation_weight", 1.0),  # Enhanced
        moran_weight=config.get("moran_weight", 1.0),
        lisa_weight=config.get("lisa_weight", 0.5),
        target_moran_i=config.get("target_moran_i", 0.3),
        smoothness_weight=config.get("smoothness_weight", 0.2),  # Enhanced
        spatial_weights=config.get("spatial_weights"),
        device=config.get("device", "cpu"),
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
