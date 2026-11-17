import numpy as np
import pytest
import torch
from scipy import integrate, stats

from evaluation.distribution_scores import (
    WIS_INTERVAL_LEVELS,
    censored_normal_crps,
    censored_normal_nll,
    distribution_wis,
    normal_crps,
    standard_wis_half_widths,
    student_t_cdf,
    student_t_crps,
)
from evaluation.losses import JointInferenceLoss
from evaluation.metrics import TorchDistributionMetricAccumulator
from evaluation.selection_scores import compute_validation_selection_score


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
def test_wis_matches_pinball_score_of_quantiles_clipped_at_zero(family):
    # Columns: no quantile below zero; lower quantiles clipped; location on the
    # bound with an observed zero; every quantile below zero.
    loc = torch.tensor([[4.0, 2.0, 0.0, -9.0]], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(
        [[0.1, 3.0, 1.5, 0.5]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[0.2, 5.0, 0.0, 1.0]], dtype=torch.float64)
    half_widths = torch.tensor(
        standard_wis_half_widths(family, 5.0), dtype=torch.float64
    )
    result = distribution_wis(loc, scale, target, half_widths)
    dist = stats.norm if family == "normal" else stats.t(5.0)
    levels = np.asarray(WIS_INTERVAL_LEVELS)
    probabilities = np.concatenate(((1 - levels) / 2, [0.5], (1 + levels) / 2))
    unclipped = loc.detach().numpy()[..., None] + scale.detach().numpy()[
        ..., None
    ] * dist.ppf(probabilities)

    def pinball(quantiles: np.ndarray) -> np.ndarray:
        errors = target.numpy()[..., None] - quantiles
        return 2 * np.maximum(
            probabilities * errors, (probabilities - 1) * errors
        ).mean(-1)

    reference = pinball(np.maximum(unclipped, 0.0))
    np.testing.assert_allclose(result.detach().numpy(), reference, rtol=1e-12)
    plain = pinball(unclipped)
    assert reference[0, 0] == pytest.approx(plain[0, 0], rel=1e-12)
    assert (reference[0, 1:3] < plain[0, 1:3]).all()
    assert reference[0, 3] == pytest.approx(1.0)
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


@pytest.mark.parametrize(
    ("loc", "scale", "target"),
    [
        (0.5, 1.0, 0.0),
        (-1.0, 0.7, 0.0),
        (-1.0, 0.7, 2.0),
        (2.0, 1.5, 3.1),
        (8.0, 0.5, 7.5),
        (-6.0, 0.3, 0.0),
    ],
)
def test_censored_normal_scores_match_the_censored_distribution(loc, scale, target):
    def censored_cdf(x: float) -> float:
        return 0.0 if x < 0 else float(stats.norm.cdf((x - loc) / scale))

    crps_reference = integrate.quad(
        lambda x: (censored_cdf(x) - (x >= target)) ** 2,
        -1.0,
        max(target, loc) + 12 * scale + 5,
        points=[0.0, target],
        limit=400,
    )[0]
    nll_reference = (
        -stats.norm.logcdf(-loc / scale)
        if target == 0
        else -stats.norm.logpdf(target, loc, scale)
    )
    args = [torch.tensor(v, dtype=torch.float64) for v in (loc, scale, target)]
    assert censored_normal_crps(*args).item() == pytest.approx(crps_reference, abs=1e-7)
    assert censored_normal_nll(*args).item() == pytest.approx(nll_reference, rel=1e-9)


def test_censored_nll_gives_no_reward_for_a_collapsed_scale_on_a_zero():
    loc = torch.zeros(3, requires_grad=True)
    scale = torch.tensor([1.0, 1e-2, 1e-6], requires_grad=True)
    nll = censored_normal_nll(loc, scale, torch.zeros(3))
    torch.testing.assert_close(nll, torch.full((3,), float(np.log(2.0))))
    nll.sum().backward()
    assert torch.isfinite(loc.grad).all()
    assert (scale.grad == 0).all()


@pytest.mark.parametrize("objective", ["nll", "crps"])
def test_censored_training_and_canonical_metrics_agree(objective):
    loc = torch.tensor([[-1.5, 0.0, 2.0]], requires_grad=True)
    scale = torch.tensor([[0.4, 1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan"), 3.0]])
    weights = torch.tensor([[1.0, 0.0, 1.0]])
    criterion = JointInferenceLoss(observation_loss=objective, censored=True)
    loss = criterion._weighted_masked_distribution_loss_from_weights(
        loc=loc, scale=scale, target=target, weights=weights
    )
    accumulator = TorchDistributionMetricAccumulator(
        device=torch.device("cpu"), horizon=3, censored=True
    )
    accumulator.update(loc=loc, scale=scale, targets=target, observed_mask=weights)
    result = accumulator.finalize()
    assert loss.item() == pytest.approx(getattr(result, objective), rel=1e-6)
    plain = JointInferenceLoss(observation_loss=objective)
    plain_loss = plain._weighted_masked_distribution_loss_from_weights(
        loc=loc, scale=scale, target=target, weights=weights
    )
    # The confident zero at loc=-1.5 is a miss for the uncensored Normal.
    assert loss.item() < plain_loss.item()


def test_censored_scores_reject_student_t():
    with pytest.raises(ValueError, match="censored_location requires"):
        JointInferenceLoss(
            observation_loss="nll", distribution_family="student_t", censored=True
        )
    with pytest.raises(ValueError, match="censored scores require"):
        TorchDistributionMetricAccumulator(
            device=torch.device("cpu"), distribution_family="student_t", censored=True
        )


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
