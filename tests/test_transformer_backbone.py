import pytest
import torch

from models.transformer_backbone import (
    LearnedPositionalEncoding,
    PositionalEncoding,
    create_transformer_backbone,
)


def test_sinusoidal_positional_encoding_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="d_model must be even"):
        PositionalEncoding(d_model=7)


def test_learned_positional_encoding_matches_per_sample_lookup() -> None:
    torch.manual_seed(0)
    encoding = LearnedPositionalEncoding(d_model=16, max_len=40, dropout=0.0)
    x = torch.randn(5, 12, 16)
    position_ids = torch.arange(12).unsqueeze(0).expand(5, -1)
    expected = x + encoding.pos_embedding(position_ids)
    output = encoding(x)
    assert torch.equal(output, expected)

    upstream = torch.randn_like(output)
    weight = encoding.pos_embedding.weight
    (expected_grad,) = torch.autograd.grad(expected, weight, upstream)
    (grad,) = torch.autograd.grad(output, weight, upstream)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-6)
    assert torch.count_nonzero(grad[12:]) == 0


def test_backbone_initializes_rates_and_states_near_epidemiological_priors() -> None:
    model = create_transformer_backbone(
        in_dim=16,
        d_model=32,
        n_heads=2,
        num_layers=2,
        horizon=7,
        obs_context_dim=8,
    )
    model.eval()
    outputs = model(torch.zeros(2, 14, 16))

    expected_rates = {
        "beta_t": 0.25,
        "gamma_t": 0.14,
        "mortality_t": 0.0005,
        "hospitalization_rate_t": 0.02,
        "hospital_recovery_t": 0.1,
        "hospital_mortality_t": 0.005,
    }
    for name, expected in expected_rates.items():
        values = outputs[name]
        assert torch.isfinite(values).all()
        torch.testing.assert_close(
            values, torch.full_like(values, expected), atol=5e-2, rtol=0
        )

    initial_states = torch.softmax(outputs["initial_states_logits"], dim=-1)
    expected_states = torch.tensor([0.9945, 0.004, 0.0005, 0.001])
    torch.testing.assert_close(
        initial_states,
        expected_states.expand_as(initial_states),
        atol=5e-3,
        rtol=0,
    )
