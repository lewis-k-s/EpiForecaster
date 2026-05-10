import torch

from models.transformer_backbone import create_transformer_backbone
from utils.model_diagnostics import (
    capture_model_diagnostics,
    should_capture_model_diagnostics,
)


def test_should_capture_model_diagnostics_respects_frequency() -> None:
    assert should_capture_model_diagnostics(step=0, frequency=0) is False
    assert should_capture_model_diagnostics(step=0, frequency=10) is True
    assert should_capture_model_diagnostics(step=9, frequency=10) is False
    assert should_capture_model_diagnostics(step=10, frequency=10) is True


def test_capture_model_diagnostics_logs_backbone_scalars() -> None:
    model = create_transformer_backbone(
        in_dim=4,
        d_model=8,
        n_heads=2,
        num_layers=2,
        horizon=3,
        obs_context_dim=5,
    )
    x = torch.randn(2, 6, 4)

    with capture_model_diagnostics(model) as capture:
        outputs = model(x)
        loss = (
            outputs["beta_t"].sum()
            + outputs["gamma_t"].sum()
            + outputs["obs_context"].sum()
        )
        loss.backward()

    log_data = capture.build_log_data(model)

    assert "model_diag/backbone/input_projection/rms" in log_data
    assert "model_diag/backbone/encoder_first/rms" in log_data
    assert "model_diag/backbone/encoder_last/rms" in log_data
    assert "model_diag/backbone/final_norm/rms" in log_data
    assert "model_diag/backbone/obs_context/rms" in log_data
    assert "model_diag/norm_weight/grad_norm" in log_data
    assert "model_diag/rezero_alpha/grad_norm" in log_data
    assert all(torch.isfinite(torch.tensor(value)) for value in log_data.values())
