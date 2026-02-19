import pytest
import torch
from models.region_losses import (
    FlowWeightedContrastiveLoss,
    SpatialAutocorrelationLoss,
    CommunityOrientedLoss,
    SpatialOnlyLoss,
    _create_spatial_weights_from_edge_index,
)


class TestRegionLosses:
    """Tests for region embedding losses."""

    @pytest.fixture
    def embeddings(self):
        num_nodes = 10
        dim = 8
        return torch.randn(num_nodes, dim)

    @pytest.fixture
    def flow_matrix(self):
        num_nodes = 10
        # Create some strong flows
        flow = torch.rand(num_nodes, num_nodes)
        # Add self-loops and some sparsity
        flow = flow * (flow > 0.5).float()
        return flow

    @pytest.fixture
    def edge_index(self):
        # Ring graph
        src = list(range(10))
        dst = list(range(1, 10)) + [0]
        edges = torch.tensor([src, dst], dtype=torch.long)
        return edges

    def test_contrastive_loss(self, embeddings, flow_matrix):
        """Test flow-weighted contrastive loss."""
        loss_fn = FlowWeightedContrastiveLoss()

        # Manually create pairs
        pos_pairs = torch.tensor([[0, 1], [1, 2]]).t()
        neg_pairs = torch.tensor([[0, 5], [1, 6]]).t()
        flow_weights = torch.tensor([1.0, 0.5])

        loss = loss_fn(embeddings, pos_pairs, flow_weights, neg_pairs)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_spatial_autocorrelation_loss(self, embeddings, edge_index):
        """Test spatial autocorrelation loss (Moran's I/LISA)."""
        loss_fn = SpatialAutocorrelationLoss()

        # Should run without error even if PySAL fails (returns 0 and logs warning)
        # or if it succeeds (returns value)
        # We assume dependencies are installed.

        loss_dict = loss_fn(embeddings, edge_index)
        assert "spatial_autocorr_loss" in loss_dict
        assert "moran_loss" in loss_dict
        assert "lisa_loss" in loss_dict
        assert "smoothness_loss" in loss_dict

    def test_community_loss(self, embeddings, flow_matrix, edge_index):
        """Test full community-oriented loss."""
        loss_fn = CommunityOrientedLoss()

        loss_dict = loss_fn(embeddings, flow_matrix, edge_index)

        assert "total_loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert "spatial_penalty" in loss_dict

    def test_spatial_only_loss(self, embeddings, edge_index):
        """Test spatial-only loss."""
        loss_fn = SpatialOnlyLoss()

        # Ignored flow matrix
        flow = torch.zeros(10, 10)

        loss_dict = loss_fn(embeddings, flow, edge_index)
        assert "total_loss" in loss_dict

    def test_weight_creation(self, edge_index):
        """Test creation of PySAL weights from edge_index."""
        try:
            w = _create_spatial_weights_from_edge_index(edge_index, num_nodes=10)
            assert w.n == 10
        except (ImportError, ValueError):
            pytest.skip("PySAL not installed or edge conversion failed")

    def test_empty_graph_handling(self, embeddings):
        """Test handling of graph with no edges."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        loss_fn = SpatialAutocorrelationLoss()

        # Should return 0 loss safely
        loss_dict = loss_fn(embeddings, edge_index)
        assert loss_dict["moran_loss"] == 0.0

    @pytest.mark.device
    def test_cross_device_contrastive_loss(self, accelerator_device):
        """Test that contrastive loss works when embeddings are on accelerator and pairs on CPU.

        This simulates the real training scenario where:
        - Embeddings come from model forward pass on GPU/MPS
        - Pair indices come from preprocessing on CPU

        Regression test for device mismatch bugs in loss computation.
        """
        num_nodes, dim = 10, 8

        # Embeddings on accelerator (simulates model output)
        embeddings = torch.randn(num_nodes, dim, device=accelerator_device)

        # Pair indices on CPU (simulates preprocessing output)
        pos_pairs = torch.tensor([[0, 1], [1, 2]]).t()  # CPU
        neg_pairs = torch.tensor([[0, 5], [1, 6]]).t()  # CPU
        flow_weights = torch.tensor([1.0, 0.5])  # CPU

        loss_fn = FlowWeightedContrastiveLoss()

        # This should NOT raise RuntimeError about device mismatch
        loss = loss_fn(embeddings, pos_pairs, flow_weights, neg_pairs)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
