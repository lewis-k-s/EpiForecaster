"""Test edge cases that trigger esda/moran.py RuntimeWarning."""

import warnings

import pytest
import torch
from libpysal import weights

from models.region_losses import (
    SpatialAutocorrelationLoss,
    _create_spatial_weights_from_edge_index,
)


@pytest.mark.region
class TestMoranEdgeCases:
    """Test scenarios that cause zero standard error in Moran's I."""

    @pytest.fixture
    def basic_weights(self):
        """Create basic spatial weights for 10 fully connected regions."""
        neighbors = {i: [j for j in range(10) if j != i] for i in range(10)}
        return weights.W(neighbors, silence_warnings=True)

    def test_constant_embeddings_all_zeros(self, basic_weights):
        """Test with all-zero embeddings (zero variance)."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.zeros(10, 4)  # 10 regions, 4 dims, all zeros

        # Should NOT raise RuntimeWarning, should gracefully handle
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            result = loss_fn._compute_moran_i_loss(embeddings, basic_weights)

        # Should return 0.0 loss instead of crashing
        assert torch.is_tensor(result)
        assert result.item() == 0.0

    def test_constant_embeddings_all_ones(self, basic_weights):
        """Test with all-one embeddings (zero variance)."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.ones(10, 4)  # 10 regions, 4 dims, all ones

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_moran_i_loss(embeddings, basic_weights)

        assert torch.is_tensor(result)

    def test_single_region_graph(self):
        """Test with only 1 region (Moran's I undefined)."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.randn(1, 4)  # Only 1 region
        w = weights.W({0: []}, silence_warnings=True)  # Island region

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_moran_i_loss(embeddings, w)

        assert torch.is_tensor(result)
        assert result.item() == 0.0

    def test_two_identical_regions(self):
        """Test with two regions having identical values."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.tensor([[1.0, 2.0], [1.0, 2.0]])  # Identical rows
        w = weights.W({0: [1], 1: [0]}, silence_warnings=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_moran_i_loss(embeddings, w)

        assert torch.is_tensor(result)

    def test_graph_with_islands(self):
        """Test with disconnected island regions (like region 119)."""
        loss_fn = SpatialAutocorrelationLoss()
        # 5 regions, but region 4 is an island (no connections)
        embeddings = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

        w = _create_spatial_weights_from_edge_index(edge_index, num_nodes=5)

        # Should log island warning but not crash
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_moran_i_loss(embeddings, w)

        assert torch.is_tensor(result)
        # Verify island was detected
        assert hasattr(w, "islands")
        assert 4 in w.islands

    def test_single_dimension_constant(self, basic_weights):
        """Test where only one dimension is constant."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.randn(10, 4)
        embeddings[:, 2] = 5.0  # Make dim 2 constant

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_moran_i_loss(embeddings, basic_weights)

        assert torch.is_tensor(result)
        # Should compute loss for non-constant dims

    def test_lisa_constant_embeddings(self, basic_weights):
        """Test LISA computation with constant embeddings."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.zeros(10, 4)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn._compute_lisa_loss(embeddings, basic_weights)

        assert torch.is_tensor(result)

    def test_forward_with_islands(self):
        """Test full forward pass with island regions."""
        loss_fn = SpatialAutocorrelationLoss()
        embeddings = torch.randn(10, 4)
        # Region 9 is an island
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loss_fn(embeddings, edge_index)

        assert isinstance(result, dict)
        assert "spatial_autocorr_loss" in result
        assert torch.is_tensor(result["spatial_autocorr_loss"])
