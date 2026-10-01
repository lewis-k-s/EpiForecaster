import pytest
import torch

from utils.normalization import unscale_forecasts


@pytest.mark.epiforecaster
def test_unscale_forecasts_broadcasts_each_sample_scale() -> None:
    predictions = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    targets = torch.zeros_like(predictions)
    means = torch.tensor([10.0, 20.0])
    scales = torch.tensor([2.0, 3.0])

    pred_unscaled, targets_unscaled = unscale_forecasts(
        predictions, targets, means, scales
    )

    torch.testing.assert_close(
        pred_unscaled, torch.tensor([[12.0, 8.0], [23.0, 17.0]])
    )
    torch.testing.assert_close(
        targets_unscaled, torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    )
