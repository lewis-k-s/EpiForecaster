import pytest
import torch
from models.mobility_gnn import MobilityDenseEncoder


def _make_dense_adj(
    num_nodes: int, num_edges: int, device: torch.device
) -> torch.Tensor:
    """Create a random dense adjacency matrix [1, N, N] with guaranteed self-loops."""
    adj = torch.eye(num_nodes, device=device).unsqueeze(0)
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    adj[0, src, dst] = 1.0
    return adj


@pytest.mark.device
class TestMobilityGNN:
    """Tests for MobilityDenseEncoder."""

    def test_gnn_stack_forward(self, accelerator_device):
        """Test full GNN stack on accelerator."""
        gnn = MobilityDenseEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(1, num_nodes, 8, device=accelerator_device)
        adj = _make_dense_adj(num_nodes, 20, accelerator_device)

        out = gnn(x, adj)
        assert out.shape == (1, num_nodes, 8)
        assert out.device.type == accelerator_device.type

    def test_gnn_single_layer(self, accelerator_device):
        """Test single-layer GNN on accelerator."""
        gnn = MobilityDenseEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=1,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(1, num_nodes, 8, device=accelerator_device)
        adj = _make_dense_adj(num_nodes, 20, accelerator_device)

        out = gnn(x, adj)
        assert out.shape == (1, num_nodes, 8)
        assert out.device.type == accelerator_device.type

    def test_gnn_depth_3(self, accelerator_device):
        """Test 3-layer GNN on accelerator."""
        gnn = MobilityDenseEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=3,
            module_type="gcn",
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(1, num_nodes, 8, device=accelerator_device)
        adj = _make_dense_adj(num_nodes, 20, accelerator_device)

        out = gnn(x, adj)
        assert out.shape == (1, num_nodes, 8)
        assert out.device.type == accelerator_device.type

    def test_dense_encoder_skip_linear_bias_zero(self):
        """Skip linear layers in the dense encoder should use explicit zero bias init."""
        encoder = MobilityDenseEncoder(
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
        gnn = MobilityDenseEncoder(
            in_dim=8, hidden_dim=16, out_dim=8, depth=3, module_type="gcn"
        ).to(accelerator_device)
        x = torch.randn(1, 10, 8, device=accelerator_device)
        adj = _make_dense_adj(10, 20, accelerator_device)

        out = gnn(x, adj)
        assert torch.all(torch.isfinite(out))

    def test_gat_variant(self, accelerator_device):
        """Test GAT variant of the encoder on accelerator."""
        gnn = MobilityDenseEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gat",
            heads=2,
        ).to(accelerator_device)

        num_nodes = 10
        x = torch.randn(1, num_nodes, 8, device=accelerator_device)
        adj = _make_dense_adj(num_nodes, 20, accelerator_device)

        out = gnn(x, adj)
        assert out.shape == (1, num_nodes, 8)
        assert torch.all(torch.isfinite(out))
        assert out.device.type == accelerator_device.type

    def test_batch_of_graphs(self, accelerator_device):
        """Test processing a batch of graphs simultaneously."""
        gnn = MobilityDenseEncoder(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            depth=2,
            module_type="gcn",
        ).to(accelerator_device)

        batch_size = 4
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, 8, device=accelerator_device)
        adj = torch.zeros(batch_size, num_nodes, num_nodes, device=accelerator_device)
        for b in range(batch_size):
            src = torch.randint(0, num_nodes, (20,), device=accelerator_device)
            dst = torch.randint(0, num_nodes, (20,), device=accelerator_device)
            adj[b, src, dst] = 1.0

        out = gnn(x, adj)
        assert out.shape == (batch_size, num_nodes, 8)
        assert out.device.type == accelerator_device.type
