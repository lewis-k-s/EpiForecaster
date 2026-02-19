import pytest
import torch
from models.mobility_gnn import MobilityPyGEncoder


@pytest.mark.device
class TestMobilityGNN:
    """Tests for MobilityPyGEncoder."""

    def test_gnn_stack_forward(self, accelerator_device):
        """Test full GNN stack on accelerator."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(num_nodes, 8, device=accelerator_device)
        edge_index = torch.randint(0, num_nodes, (2, 20), device=accelerator_device)
        edge_weight = torch.rand(20, device=accelerator_device)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)
        assert out.device.type == accelerator_device.type

    def test_gnn_single_layer(self, accelerator_device):
        """Test single-layer GNN on accelerator."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=1,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(num_nodes, 8, device=accelerator_device)
        edge_index = torch.randint(0, num_nodes, (2, 20), device=accelerator_device)
        edge_weight = torch.rand(20, device=accelerator_device)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)
        assert out.device.type == accelerator_device.type

    def test_gnn_depth_3(self, accelerator_device):
        """Test 3-layer GNN on accelerator."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=3,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(num_nodes, 8, device=accelerator_device)
        edge_index = torch.randint(0, num_nodes, (2, 20), device=accelerator_device)
        edge_weight = torch.rand(20, device=accelerator_device)

        out = gnn(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 8)
        assert out.device.type == accelerator_device.type

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

    def test_gnn_output_is_finite_after_initialization(self, accelerator_device):
        """Forward pass should stay finite immediately after initialization."""
        gnn = MobilityPyGEncoder(
            in_dim=8, hidden_dim=16, out_dim=8, depth=3, module_type="gcn"
        ).to(accelerator_device)
        x = torch.randn(10, 8, device=accelerator_device)
        edge_index = torch.randint(0, 10, (2, 20), device=accelerator_device)
        edge_weight = torch.rand(20, device=accelerator_device)

        out = gnn(x, edge_index, edge_weight)
        assert torch.all(torch.isfinite(out))

    def test_gat_variant(self, accelerator_device):
        """Test GAT variant of the encoder on accelerator."""
        gnn = MobilityPyGEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gat",
            heads=2,
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(num_nodes, 8, device=accelerator_device)
        edge_index = torch.randint(0, num_nodes, (2, 20), device=accelerator_device)

        out = gnn(x, edge_index)
        assert out.shape == (num_nodes, 8)
        assert torch.all(torch.isfinite(out))
        assert out.device.type == accelerator_device.type
