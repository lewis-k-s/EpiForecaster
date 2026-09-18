import numpy as np
import pytest
import torch
from scipy import integrate, stats

from evaluation.distribution_scores import (
    WIS_INTERVAL_LEVELS,
    distribution_wis,
    normal_crps,
    standard_wis_half_widths,
    student_t_cdf,
    student_t_crps,
)
from evaluation.losses import JointInferenceLoss
from evaluation.metrics import TorchDistributionMetricAccumulator
from training.epiforecaster_trainer import compute_validation_selection_score


@pytest.mark.parametrize("df", [2.1, 2.5, 3.0, 5.0, 10.0, 30.0, 100.0, 101.0])
def test_student_cdf_matches_scipy_and_density_gradient(df):
    z = torch.tensor(
        [-100.0, -10, -2, -0.1, 0, 0.1, 2, 10, 100],
        dtype=torch.float64,
        requires_grad=True,
    )
    cdf = student_t_cdf(z, df)
    np.testing.assert_allclose(
        cdf.detach().numpy(), stats.t.cdf(z.detach().numpy(), df), atol=2e-9, rtol=1e-7
    )
    gradient = torch.autograd.grad(cdf.sum(), z)[0]
    np.testing.assert_allclose(
        gradient.numpy(), stats.t.pdf(z.detach().numpy(), df), atol=2e-9, rtol=1e-5
    )


@pytest.mark.parametrize("df", [2.5, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("target", [-8.0, 0.0, 2.0])
def test_student_crps_matches_integrated_definition(df, target):
    loc, scale = 0.7, 1.3
    dist = stats.t(df, loc=loc, scale=scale)
    reference = integrate.quad(lambda x: dist.cdf(x) ** 2, -np.inf, target)[0]
    reference += integrate.quad(lambda x: dist.sf(x) ** 2, target, np.inf)[0]
    args = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (loc, scale, target)
    ]
    value = student_t_crps(*args, df)
    assert value.item() == pytest.approx(reference, rel=1e-7, abs=1e-8)
    assert torch.autograd.gradcheck(
        lambda a, b, c: student_t_crps(a, b, c, df), tuple(args)
    )


def test_student_crps_normal_limit_and_extreme_scale_gradients():
    loc = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([0.001, 1, 100], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([-100.0, 0, 100.0], dtype=torch.float64)
    result = student_t_crps(loc, scale, target, 1000.0)
    torch.testing.assert_close(
        result, normal_crps(loc, scale, target), rtol=0.002, atol=1e-5
    )
    result.sum().backward()
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(scale.grad).all()


@pytest.mark.parametrize("family", ["normal", "student_t"])
def test_wis_matches_independent_pinball_quantile_score(family):
    loc = torch.tensor([[0.2, 2.0]], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([[0.1, 3.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[4.0, -1.0]], dtype=torch.float64)
    half_widths = torch.tensor(
        standard_wis_half_widths(family, 5.0), dtype=torch.float64
    )
    result = distribution_wis(loc, scale, target, half_widths)
    dist = stats.norm if family == "normal" else stats.t(5.0)
    levels = np.asarray(WIS_INTERVAL_LEVELS)
    probabilities = np.concatenate(((1 - levels) / 2, [0.5], (1 + levels) / 2))
    quantiles = loc.detach().numpy()[..., None] + scale.detach().numpy()[
        ..., None
    ] * dist.ppf(probabilities)
    errors = target.numpy()[..., None] - quantiles
    reference = 2 * np.maximum(
        probabilities * errors, (probabilities - 1) * errors
    ).mean(-1)
    np.testing.assert_allclose(result.detach().numpy(), reference, rtol=1e-12)
    result.sum().backward()
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(scale.grad).all()


@pytest.mark.parametrize("family", ["normal", "student_t"])
@pytest.mark.parametrize("objective", ["nll", "crps", "wis"])
def test_training_and_canonical_metrics_agree_on_observed_support(family, objective):
    loc = torch.tensor([[0.2, 0.0, 2.0]], requires_grad=True)
    scale = torch.tensor([[0.4, 1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[2.0, float("nan"), -1.0]])
    weights = torch.tensor([[1.0, 0.0, 1.0]])
    criterion = JointInferenceLoss(
        observation_loss=objective, distribution_family=family
    )
    loss = criterion._weighted_masked_distribution_loss_from_weights(
        loc=loc, scale=scale, target=target, weights=weights
    )
    accumulator = TorchDistributionMetricAccumulator(
        device=torch.device("cpu"), horizon=3, distribution_family=family
    )
    accumulator.update(loc=loc, scale=scale, targets=target, observed_mask=weights)
    result = accumulator.finalize()
    assert loss.item() == pytest.approx(getattr(result, objective), rel=1e-6)
    assert result.observed_count == 2
    assert np.isnan(result.wis_per_h[1])
    loss.backward()
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(scale.grad).all()
    assert loc.grad[0, 1] == 0
    assert scale.grad[0, 1] == 0


def test_invalid_observed_prediction_invalidates_selection_not_support():
    accumulator = TorchDistributionMetricAccumulator(
        device=torch.device("cpu"), horizon=2
    )
    accumulator.update(
        loc=torch.tensor([[float("nan"), 0.0]]),
        scale=torch.ones(1, 2),
        targets=torch.zeros(1, 2),
        observed_mask=torch.ones(1, 2),
    )
    result = accumulator.finalize()
    assert result.invalid_prediction_count == 1
    assert result.observed_count == 2
    assert np.isnan(result.wis)
    with pytest.raises(ValueError, match="Required checkpoint-selection"):
        compute_validation_selection_score(
            {"wis_cases_log1p_per_100k": result.wis},
            observation_loss="nll",
            selection_metric="mean_target_wis",
            required_heads=["cases"],
        )


def test_explicit_selection_never_uses_another_metric_or_drops_a_head():
    metrics = {
        f"wis_{head}_log1p_per_100k": value
        for head, value in zip(
            ("ww", "hosp", "cases", "deaths"), (1, 2, 3, 4), strict=True
        )
    }
    score, name = compute_validation_selection_score(
        metrics, observation_loss="nll", selection_metric="mean_target_wis"
    )
    assert score == 2.5
    assert name == "mean_target_wis"
    del metrics["wis_deaths_log1p_per_100k"]
    metrics["crps_deaths_log1p_per_100k"] = 0.01
    with pytest.raises(ValueError, match="wis_deaths"):
        compute_validation_selection_score(
            metrics, observation_loss="nll", selection_metric="mean_target_wis"
        )
