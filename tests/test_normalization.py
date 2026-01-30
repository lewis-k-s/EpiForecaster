import pytest
import torch
from utils.normalization import unscale_forecasts


@pytest.mark.epiforecaster
def test_unscale_forecasts_shapes():
    B, H = 2, 5
    predictions = torch.randn(B, H)
    targets = torch.randn(B, H)
    target_mean = torch.randn(B, 1)
    target_scale = torch.abs(torch.randn(B, 1))

    pred_unscaled, targets_unscaled = unscale_forecasts(
        predictions, targets, target_mean, target_scale
    )

    assert pred_unscaled.shape == (B, H)
    assert targets_unscaled.shape == (B, H)


@pytest.mark.epiforecaster
def test_unscale_forecasts_correctness():
    # Simple case
    # Mean = 10, Scale = 2
    # Pred (norm) = 1.0 -> Unscaled = 1.0 * 2 + 10 = 12.0

    predictions = torch.tensor([[1.0]])
    targets = torch.tensor([[0.0]])
    target_mean = torch.tensor([10.0])
    target_scale = torch.tensor([2.0])

    pred_unscaled, targets_unscaled = unscale_forecasts(
        predictions, targets, target_mean, target_scale
    )

    assert torch.allclose(pred_unscaled, torch.tensor([[12.0]]))
    assert torch.allclose(targets_unscaled, torch.tensor([[10.0]]))


@pytest.mark.epiforecaster
def test_unscale_forecasts_broadcasting():
    # Mean/Scale are 1D (B,)
    B, H = 2, 3
    predictions = torch.ones(B, H)
    targets = torch.zeros(B, H)
    target_mean = torch.tensor([10.0, 20.0])
    target_scale = torch.tensor([2.0, 3.0])

    pred_unscaled, _targets_unscaled = unscale_forecasts(
        predictions, targets, target_mean, target_scale
    )

    expected_pred = torch.tensor([[12.0, 12.0, 12.0], [23.0, 23.0, 23.0]])

    assert torch.allclose(pred_unscaled, expected_pred)
