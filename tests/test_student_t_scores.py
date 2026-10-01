"""Check the analytic Student-t score and its gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.special import beta

from evaluation.student_t_scores import student_t5_cdf, student_t5_crps


@pytest.mark.epiforecaster
def test_student_t5_scores_match_scipy_and_have_finite_gradients() -> None:
    target = torch.tensor([-8.0, -2.0, 0.0, 1.0, 8.0], dtype=torch.float64)
    loc = torch.zeros_like(target, requires_grad=True)
    scale = torch.ones_like(target, requires_grad=True)
    cdf = student_t5_cdf(target)
    np.testing.assert_allclose(
        cdf.detach().numpy(), stats.t.cdf(target.numpy(), 5), rtol=1e-10, atol=1e-10
    )
    tails = torch.tensor([-100.0, -20.0, 20.0, 100.0])
    np.testing.assert_allclose(
        student_t5_cdf(tails).numpy(),
        stats.t.cdf(tails.numpy(), 5),
        rtol=1e-5,
        atol=1e-8,
    )
    score = student_t5_crps(loc=loc, scale=scale, target=target)
    reference = []
    for value in target.tolist():
        z = value
        pdf = stats.t.pdf(z, 5)
        ratio = beta(0.5, 4.5) / beta(0.5, 2.5) ** 2
        reference.append(
            z * (2 * stats.t.cdf(z, 5) - 1)
            + 0.5 * (pdf * (5 + z * z) - np.sqrt(5) * ratio)
        )
    np.testing.assert_allclose(
        score.detach().numpy(), reference, rtol=1e-10, atol=1e-10
    )
    score.sum().backward()
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(scale.grad).all()
