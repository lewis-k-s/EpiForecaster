import torch
from models.mobility_gnn import MobilityPyGEncoder


class TestMobilityGNN:
    """Tests for MobilityPyGEncoder."""

    def test_gnn_stack_forward(self):
        """Test full GNN stack."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gcn",
        )

        num_nodes = 10
        x = torch.randn(num_nodes, 8)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_weight = torch.rand(20)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)

    def test_gnn_single_layer(self):
        """Test single-layer GNN."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=1,
            module_type="gcn",
        )

        num_nodes = 10
        x = torch.randn(num_nodes, 8)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_weight = torch.rand(20)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)

    def test_gnn_depth_3(self):
        """Test 3-layer GNN."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=3,
            module_type="gcn",
        )

        num_nodes = 10
        x = torch.randn(num_nodes, 8)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_weight = torch.rand(20)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)

    def test_pyg_encoder_skip_linear_bias_zero(self):
        """Skip linear layers in the PyG encoder should use explicit zero bias init."""
        encoder = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=3,
            module_type="gcn",
        )

        linear_skips = [
            layer for layer in encoder.skips if isinstance(layer, torch.nn.Linear)
        ]
        assert linear_skips, "Expected at least one linear skip layer."
        for layer in linear_skips:
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_gnn_output_is_finite_after_initialization(self):
        """Forward pass should stay finite immediately after initialization."""
        gnn = MobilityPyGEncoder(
            in_dim=8, hidden_dim=16, out_dim=8, depth=3, module_type="gcn"
        )
        x = torch.randn(10, 8)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_weight = torch.rand(20)

        out = gnn(x, edge_index, edge_weight)
        assert torch.all(torch.isfinite(out))

    def test_gat_variant(self):
        """Test GAT variant of the encoder."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gat",
            heads=2,
        )

        num_nodes = 10
        x = torch.randn(num_nodes, 8)
        edge_index = torch.randint(0, num_nodes, (2, 20))

        out = gnn(x, edge_index)
        assert out.shape == (num_nodes, 8)
        assert torch.all(torch.isfinite(out))
