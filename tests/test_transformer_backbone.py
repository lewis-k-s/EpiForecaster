import pytest
import torch
from models.transformer_backbone import (
    TransformerBackbone,
    MultiHorizonForecaster,
    create_transformer_backbone,
)


class TestTransformerBackbone:
    """Tests for TransformerBackbone."""

    @pytest.fixture
    def backbone_config(self):
        return {
            "in_dim": 16,
            "d_model": 32,
            "n_heads": 2,
            "num_layers": 2,
            "horizon": 7,
            "obs_context_dim": 8,
        }

    def test_initialization(self, backbone_config):
        """Test initialization via factory function."""
        model = create_transformer_backbone(**backbone_config)
        assert isinstance(model, TransformerBackbone)
        assert model.horizon == backbone_config["horizon"]
        assert model.count_parameters() > 0

    def test_forward_shapes(self, backbone_config):
        """Test forward pass output shapes."""
        model = create_transformer_backbone(**backbone_config)
        batch_size = 4
        seq_len = 14
        in_dim = backbone_config["in_dim"]

        x_seq = torch.randn(batch_size, seq_len, in_dim)
        outputs = model(x_seq)

        horizon = backbone_config["horizon"]

        # Check output keys
        expected_keys = {
            "beta_t",
            "mortality_t",
            "gamma_t",
            "initial_states_logits",
            "obs_context",
        }
        assert set(outputs.keys()) == expected_keys

        # Check shapes
        assert outputs["beta_t"].shape == (batch_size, horizon)
        assert outputs["mortality_t"].shape == (batch_size, horizon)
        assert outputs["gamma_t"].shape == (batch_size, horizon)
        assert outputs["initial_states_logits"].shape == (batch_size, 3)

        # Obs context shape: [B, H, C_obs]
        assert outputs["obs_context"].shape == (
            batch_size,
            horizon,
            backbone_config["obs_context_dim"],
        )

    def test_short_sequence_handling(self, backbone_config):
        """Test handling of sequences shorter than horizon."""
        model = create_transformer_backbone(**backbone_config)
        batch_size = 2
        horizon = backbone_config["horizon"]
        seq_len = horizon - 2

        x_seq = torch.randn(batch_size, seq_len, backbone_config["in_dim"])
        outputs = model(x_seq)

        assert outputs["obs_context"].shape == (
            batch_size,
            horizon,
            backbone_config["obs_context_dim"],
        )
        assert torch.all(torch.isfinite(outputs["obs_context"]))

    def test_padding_mask(self, backbone_config):
        """Test that padding mask is respected."""
        model = create_transformer_backbone(**backbone_config)
        batch_size = 2
        seq_len = 10
        in_dim = backbone_config["in_dim"]

        x_seq = torch.randn(batch_size, seq_len, in_dim)

        # mask: True=Keep, False=Ignore (based on ~mask implementation in model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = False

        outputs = model(x_seq, mask=mask)
        assert outputs["beta_t"].shape == (batch_size, backbone_config["horizon"])

    def test_pos_encoding_types(self, backbone_config):
        """Test both sinusoidal and learned positional encodings."""
        for pe_type in ["sinusoidal", "learned"]:
            config = backbone_config.copy()
            config["positional_encoding"] = pe_type
            model = create_transformer_backbone(**config)
            x = torch.randn(2, 10, config["in_dim"])
            assert model(x)["beta_t"].shape == (2, config["horizon"])

    def test_conservative_rate_priors_are_finite_and_not_clamped(self, backbone_config):
        """Zero input should produce prior-centered rates away from clamp boundaries."""
        model = create_transformer_backbone(**backbone_config)
        model.eval()

        x_seq = torch.zeros(4, 14, backbone_config["in_dim"])
        outputs = model(x_seq)

        expected_priors = {
            "beta_t": 0.25,
            "gamma_t": 0.14,
            "mortality_t": 0.002,
        }

        for key, expected in expected_priors.items():
            values = outputs[key]
            assert torch.all(torch.isfinite(values))
            assert torch.allclose(values, torch.full_like(values, expected), atol=1e-5)

        cfg = model.sir_physics
        assert torch.all(outputs["beta_t"] > cfg.beta_min)
        assert torch.all(outputs["beta_t"] < cfg.beta_max)
        assert torch.all(outputs["gamma_t"] > cfg.gamma_min)
        assert torch.all(outputs["gamma_t"] < cfg.gamma_max)
        assert torch.all(outputs["mortality_t"] > cfg.mortality_min)
        assert torch.all(outputs["mortality_t"] < cfg.mortality_max)

    def test_initial_states_start_from_epidemiological_prior(self, backbone_config):
        """Initial state logits should map to the conservative S/I/R prior."""
        model = create_transformer_backbone(**backbone_config)
        model.eval()

        x_seq = torch.zeros(3, 12, backbone_config["in_dim"])
        outputs = model(x_seq)
        initial_states = torch.softmax(outputs["initial_states_logits"], dim=-1)

        expected = torch.tensor([0.995, 0.004, 0.001], dtype=initial_states.dtype)
        expected = expected.unsqueeze(0).expand_as(initial_states)
        assert torch.allclose(initial_states, expected, atol=1e-5)

    def test_obs_context_starts_neutral_at_initialization(self, backbone_config):
        """Observation context projection should start as an exact zero map."""
        model = create_transformer_backbone(**backbone_config)
        model.eval()

        x_seq = torch.randn(2, 10, backbone_config["in_dim"])
        outputs = model(x_seq)
        assert torch.allclose(
            outputs["obs_context"], torch.zeros_like(outputs["obs_context"]), atol=1e-8
        )


class TestMultiHorizonForecaster:
    """Tests for MultiHorizonForecaster."""

    def test_forward(self):
        in_dim = 10
        horizons = [1, 7, 14]
        model = MultiHorizonForecaster(in_dim=in_dim, horizons=horizons, d_model=16)

        batch_size = 5
        seq_len = 10
        x = torch.randn(batch_size, seq_len, in_dim)

        forecasts = model(x)

        assert len(forecasts) == len(horizons)
        for h in horizons:
            assert forecasts[h].shape == (batch_size, h)
