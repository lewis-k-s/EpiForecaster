import pytest
import torch
import torch.nn.functional as F

from models.mobility_gnn import MobilityDenseEncoder


def _reference_forward(encoder: MobilityDenseEncoder, x, adj):
    """Per-layer PyG forward, as before the shared normalization."""
    h = x
    for i, layer in enumerate(encoder.layers):
        h_in = h
        h = layer(h, adj, add_loop=False)
        if i < len(encoder.layers) - 1:
            h = encoder.norms[i](h)
            h = encoder.activation(h)
            h = F.dropout(h, p=encoder.dropout_val, training=encoder.training)
        if i < len(encoder.skips):
            h = h + encoder.skips[i](h_in)
    return h


def _inputs(num_graphs=6, num_nodes=9, in_dim=5):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(num_graphs, num_nodes, in_dim, generator=generator)
    adj = torch.rand(num_graphs, num_nodes, num_nodes, generator=generator)
    adj = adj * (adj > 0.6)
    adj = torch.maximum(adj, torch.eye(num_nodes).unsqueeze(0))
    target_index = torch.randint(0, num_nodes, (num_graphs,), generator=generator)
    return x, adj, target_index


@pytest.mark.epiforecaster
@pytest.mark.parametrize("module_type", ["gcn", "gat"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_encoder_matches_per_layer_reference(module_type, depth):
    torch.manual_seed(0)
    encoder = MobilityDenseEncoder(
        in_dim=5, hidden_dim=8, out_dim=4, depth=depth, module_type=module_type
    ).eval()
    x, adj, target_index = _inputs()

    expected = _reference_forward(encoder, x, adj)
    graph_ids = torch.arange(x.size(0))

    torch.testing.assert_close(encoder(x, adj), expected)
    torch.testing.assert_close(
        encoder(x, adj, target_index=target_index),
        expected[graph_ids, target_index],
    )


@pytest.mark.epiforecaster
def test_target_rows_backward_matches_reference():
    torch.manual_seed(0)
    encoder = MobilityDenseEncoder(
        in_dim=5, hidden_dim=8, out_dim=4, depth=2, module_type="gcn", dropout=0.0
    )
    x, adj, target_index = _inputs()
    graph_ids = torch.arange(x.size(0))

    encoder(x, adj, target_index=target_index).square().sum().backward()
    actual = [p.grad.clone() for p in encoder.parameters()]
    encoder.zero_grad()
    _reference_forward(encoder, x, adj)[
        graph_ids, target_index
    ].square().sum().backward()

    for got, want in zip(actual, (p.grad for p in encoder.parameters()), strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.epiforecaster
@pytest.mark.parametrize("module_type", ["gcn", "gat"])
def test_shared_adjacency_matches_expanded_adjacency(module_type):
    torch.manual_seed(0)
    encoder = MobilityDenseEncoder(
        in_dim=5, hidden_dim=8, out_dim=4, depth=2, module_type=module_type
    ).eval()
    samples, steps = 3, 4
    x, adj, target_index = _inputs(num_graphs=samples * steps)
    shared = adj[:steps]
    # Sample-major graphs g = b * steps + t use shared[t].
    expanded = shared.repeat(samples, 1, 1)

    torch.testing.assert_close(encoder(x, shared), encoder(x, expanded))
    torch.testing.assert_close(
        encoder(x, shared, target_index=target_index),
        encoder(x, expanded, target_index=target_index),
    )


@pytest.mark.epiforecaster
def test_shared_adjacency_backward_matches_expanded_adjacency():
    torch.manual_seed(0)
    encoder = MobilityDenseEncoder(
        in_dim=5, hidden_dim=8, out_dim=4, depth=2, module_type="gcn", dropout=0.0
    )
    x, adj, target_index = _inputs(num_graphs=12)
    shared = adj[:4]

    encoder(x, shared, target_index=target_index).square().sum().backward()
    actual = [p.grad.clone() for p in encoder.parameters()]
    encoder.zero_grad()
    encoder(
        x, shared.repeat(3, 1, 1), target_index=target_index
    ).square().sum().backward()

    for got, want in zip(actual, (p.grad for p in encoder.parameters()), strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.epiforecaster
def test_adjacency_count_must_divide_graph_count():
    encoder = MobilityDenseEncoder(in_dim=5, hidden_dim=8, out_dim=4, depth=2)
    x, adj, _ = _inputs(num_graphs=6)
    with pytest.raises(ValueError, match="must divide"):
        encoder(x, adj[:4])
