import pytest
import torch
from models.aggregators import (
    MeanAggregator,
    AttentionAggregator,
    MaxPoolAggregator,
    create_aggregator,
)


class TestAggregators:
    """Tests for GraphSAGE aggregators."""

    @pytest.fixture
    def graph_data(self):
        num_nodes = 5
        input_dim = 4
        output_dim = 4

        x = torch.randn(num_nodes, input_dim)

        # Edge index: Node 0 connected to 1, 2. Node 3 connected to 4.
        edge_index = torch.tensor(
            [
                [1, 2, 4],  # Sources
                [0, 0, 3],  # Targets
            ],
            dtype=torch.long,
        )

        edge_attr = torch.randn(edge_index.shape[1], 2)

        return x, edge_index, edge_attr, input_dim, output_dim

    def test_mean_aggregator(self, graph_data):
        x, edge_index, edge_attr, in_dim, out_dim = graph_data
        agg = MeanAggregator(in_dim, out_dim, normalize=False)

        # Manually compute expected for node 0
        # Neighbors: 1, 2
        # transformed neighbors: mean(W_n * x[1], W_n * x[2])
        # transformed self: W_s * x[0]
        # combined: concat -> linear

        # Run forward
        out = agg(x, edge_index, edge_attr)
        assert out.shape == (x.shape[0], out_dim)

        # Check deterministic
        out2 = agg(x, edge_index, edge_attr)
        assert torch.allclose(out, out2)

    def test_max_pool_aggregator(self, graph_data):
        x, edge_index, edge_attr, in_dim, out_dim = graph_data
        agg = MaxPoolAggregator(in_dim, out_dim, normalize=False)

        out = agg(x, edge_index, edge_attr)
        assert out.shape == (x.shape[0], out_dim)

    def test_attention_aggregator(self, graph_data):
        x, edge_index, edge_attr, in_dim, out_dim = graph_data
        # Attention requires output_dim divisible by heads (default 4)
        agg = AttentionAggregator(in_dim, out_dim, num_heads=2, normalize=False)

        out = agg(x, edge_index, edge_attr)
        assert out.shape == (x.shape[0], out_dim)

    def test_attention_aggregator_with_edge_attr(self, graph_data):
        x, edge_index, edge_attr, in_dim, out_dim = graph_data
        edge_dim = edge_attr.shape[1]

        agg = AttentionAggregator(in_dim, out_dim, edge_dim=edge_dim, num_heads=2)

        out = agg(x, edge_index, edge_attr)
        assert out.shape == (x.shape[0], out_dim)

    def test_factory_function(self):
        agg = create_aggregator("mean", 16, 16)
        assert isinstance(agg, MeanAggregator)

        agg = create_aggregator("max", 16, 16)
        assert isinstance(agg, MaxPoolAggregator)

        agg = create_aggregator("attention", 16, 16, num_heads=4)
        assert isinstance(agg, AttentionAggregator)

    def test_normalization(self, graph_data):
        x, edge_index, edge_attr, in_dim, out_dim = graph_data
        agg = MeanAggregator(in_dim, out_dim, normalize=True)

        out = agg(x, edge_index)

        # Norms should be close to 1 (or 0 if output is 0 vector)
        norms = torch.norm(out, p=2, dim=1)
        is_unit = torch.isclose(norms, torch.ones_like(norms), atol=1e-5)
        is_zero = torch.isclose(norms, torch.zeros_like(norms), atol=1e-5)
        assert torch.all(is_unit | is_zero)
