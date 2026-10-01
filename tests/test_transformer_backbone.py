import pytest
import torch

from models.transformer_backbone import PositionalEncoding, create_transformer_backbone


def test_sinusoidal_positional_encoding_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="d_model must be even"):
        PositionalEncoding(d_model=7)


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
