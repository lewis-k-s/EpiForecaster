import pytest
import torch
import torch.nn as nn
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
            "beta_t", "mortality_t", "gamma_t", 
            "initial_states_logits", "obs_context"
        }
        assert set(outputs.keys()) == expected_keys
        
        # Check shapes
        assert outputs["beta_t"].shape == (batch_size, horizon)
        assert outputs["mortality_t"].shape == (batch_size, horizon)
        assert outputs["gamma_t"].shape == (batch_size, horizon)
        assert outputs["initial_states_logits"].shape == (batch_size, 3)
        
        # Obs context shape: [B, H, C_obs]
        assert outputs["obs_context"].shape == (
            batch_size, horizon, backbone_config["obs_context_dim"]
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
            batch_size, horizon, backbone_config["obs_context_dim"]
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

class TestMultiHorizonForecaster:
    """Tests for MultiHorizonForecaster."""
    
    def test_forward(self):
        in_dim = 10
        horizons = [1, 7, 14]
        model = MultiHorizonForecaster(
            in_dim=in_dim,
            horizons=horizons,
            d_model=16
        )
        
        batch_size = 5
        seq_len = 10
        x = torch.randn(batch_size, seq_len, in_dim)
        
        forecasts = model(x)
        
        assert len(forecasts) == len(horizons)
        for h in horizons:
            assert forecasts[h].shape == (batch_size, h)
