import pytest
import torch
from models.mobility_gnn import MobilityGNNLayer, MobilityGNN

class TestMobilityGNN:
    """Tests for MobilityGNN."""

    @pytest.fixture
    def layer_config(self):
        return {
            "input_dim": 8,
            "output_dim": 8,
            "aggregator_type": "mean",
        }

    def test_layer_normalization_logic(self, layer_config):
        """Test _normalize_incoming_flows logic."""
        layer = MobilityGNNLayer(**layer_config)
        
        # 3 nodes
        # Flow matrix M[j, i] is flow FROM j TO i
        # Col 0: flows TO node 0
        mobility = torch.tensor([
            [0.0, 10.0, 5.0],  # From 0 -> 0, 1, 2
            [2.0, 0.0, 5.0],   # From 1 -> 0, 1, 2
            [8.0, 10.0, 0.0],  # From 2 -> 0, 1, 2
        ])
        
        norm_mob = layer._normalize_incoming_flows(mobility)
        
        # Check col sums (incoming flow sums)
        # But wait, the method divides by incoming sum.
        # So A[j, i] = M[j, i] / sum_k M[k, i]
        # Sum over k A[k, i] should be 1.0 (sum of col i)
        
        col_sums = norm_mob.sum(dim=0)
        expected_sums = torch.ones(3)
        assert torch.allclose(col_sums, expected_sums, atol=1e-5)

    def test_layer_forward(self, layer_config):
        """Test layer forward pass."""
        layer = MobilityGNNLayer(**layer_config)
        batch_size = 5 # num_nodes
        x = torch.randn(batch_size, layer_config["input_dim"])
        mobility = torch.rand(batch_size, batch_size)
        
        out = layer(x, mobility)
        assert out.shape == (batch_size, layer_config["output_dim"])

    def test_gnn_stack_forward(self):
        """Test full GNN stack."""
        gnn = MobilityGNN(
            in_dim=8,
            hidden_dim=16,
            out_dim=8,
            num_layers=2,
            aggregator_type="mean"
        )
        
        num_nodes = 10
        x = torch.randn(num_nodes, 8)
        mobility = torch.rand(num_nodes, num_nodes)
        
        out = gnn(x, mobility)
        assert out.shape == (num_nodes, 8)

    def test_gnn_batch_forward(self):
        """Test forward_batch."""
        gnn = MobilityGNN(
            in_dim=8,
            hidden_dim=16,
            out_dim=8
        )
        
        batch_size = 2
        num_nodes = 5
        x_batch = torch.randn(batch_size, num_nodes, 8)
        mob_batch = torch.rand(batch_size, num_nodes, num_nodes)
        
        out = gnn.forward_batch(x_batch, mob_batch)
        assert out.shape == (batch_size, num_nodes, 8)

    def test_sparse_conversion(self, layer_config):
        """Test dense to sparse conversion."""
        layer = MobilityGNNLayer(**layer_config)
        
        # Triangle graph 0->1->2->0
        mobility = torch.zeros(3, 3)
        mobility[0, 1] = 1.0 # 0->1
        mobility[1, 2] = 1.0 # 1->2
        mobility[2, 0] = 1.0 # 2->0
        
        edge_index, edge_attr = layer._mobility_to_edges(mobility, threshold=0.1)
        
        # Should have 3 edges
        assert edge_index.shape[1] == 3
        
        # Check edges (source -> target)
        # PyG edge_index is [2, E] where [0] is source, [1] is target
        # Our mobility is [j, i] = from j to i
        # So row indices are sources, col indices are targets
        # indices of non-zero: (0,1), (1,2), (2,0)
        
        expected_src = {0, 1, 2}
        expected_dst = {1, 2, 0}
        
        assert set(edge_index[0].tolist()) == expected_src
        assert set(edge_index[1].tolist()) == expected_dst

