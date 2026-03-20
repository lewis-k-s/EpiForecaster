import numpy as np
import torch

from evaluation.metrics import TorchMaskedMetricAccumulator, compute_masked_metrics_numpy


def test_compute_masked_metrics_numpy_uses_sample_weights() -> None:
    pred = np.array([[2.0, 0.0]], dtype=np.float64)
    target = np.array([[0.0, 2.0]], dtype=np.float64)
    weights = np.array([[1.0, 0.5]], dtype=np.float64)

    metrics = compute_masked_metrics_numpy(
        predictions=pred,
        targets=target,
        observed_mask=None,
        sample_weights=weights,
        horizon=2,
    )

    assert metrics.observed_count == 2
    assert np.isclose(metrics.effective_count, 1.5)
    assert np.isclose(metrics.mae, 2.0)
    assert np.isclose(metrics.rmse, 2.0)
    assert np.isclose(metrics.smape, 2.0)
    assert np.isclose(metrics.r2, -3.5)


def test_torch_metric_accumulator_matches_weighted_numpy() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    weights = torch.tensor([[1.0, 0.25], [0.0, 0.5]])

    acc = TorchMaskedMetricAccumulator(device=torch.device("cpu"), horizon=2)
    acc.update(
        predictions=pred,
        targets=target,
        observed_mask=torch.ones_like(target),
        sample_weights=weights,
    )
    torch_metrics = acc.finalize()

    numpy_metrics = compute_masked_metrics_numpy(
        predictions=pred.numpy(),
        targets=target.numpy(),
        observed_mask=np.ones((2, 2), dtype=np.float64),
        sample_weights=weights.numpy(),
        horizon=2,
    )

    assert np.isclose(torch_metrics.mae, numpy_metrics.mae)
    assert np.isclose(torch_metrics.rmse, numpy_metrics.rmse)
    assert np.isclose(torch_metrics.smape, numpy_metrics.smape)
    assert np.isclose(torch_metrics.r2, numpy_metrics.r2)
    assert torch_metrics.observed_count == numpy_metrics.observed_count
    assert np.isclose(torch_metrics.effective_count, numpy_metrics.effective_count)
    assert np.allclose(torch_metrics.mae_per_h, numpy_metrics.mae_per_h)
    assert np.allclose(torch_metrics.rmse_per_h, numpy_metrics.rmse_per_h)


def test_sparse_horizons_remain_nan_in_torch_and_numpy() -> None:
    pred = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]])
    target = torch.zeros_like(pred)
    weights = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    acc = TorchMaskedMetricAccumulator(device=torch.device("cpu"), horizon=7)
    acc.update(
        predictions=pred,
        targets=target,
        observed_mask=None,
        sample_weights=weights,
    )
    torch_metrics = acc.finalize()

    numpy_metrics = compute_masked_metrics_numpy(
        predictions=pred.numpy(),
        targets=target.numpy(),
        observed_mask=None,
        sample_weights=weights.numpy(),
        horizon=7,
    )

    assert np.isnan(torch_metrics.mae_per_h[:6]).all()
    assert np.isnan(torch_metrics.rmse_per_h[:6]).all()
    assert torch_metrics.mae_per_h[6] == 2.0
    assert torch_metrics.rmse_per_h[6] == 2.0
    assert np.allclose(
        torch_metrics.mae_per_h,
        numpy_metrics.mae_per_h,
        equal_nan=True,
    )
    assert np.allclose(
        torch_metrics.rmse_per_h,
        numpy_metrics.rmse_per_h,
        equal_nan=True,
    )
