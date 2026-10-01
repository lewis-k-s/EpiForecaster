"""Differentiable scores for the fixed five-degree Student-t forecast family."""

from __future__ import annotations

import math

import torch


def student_t5_cdf(z: torch.Tensor) -> torch.Tensor:
    """Stable CDF for a standard Student-t variable with five degrees of freedom."""
    angle = torch.atan(z / math.sqrt(5.0))
    central = (
        0.5
        + angle / math.pi
        + 2.0 * torch.sin(2.0 * angle) / (3.0 * math.pi)
        + torch.sin(4.0 * angle) / (12.0 * math.pi)
    )
    # The central expression loses precision when its terms almost cancel.
    # Integrate the polynomial tail expansion for |z| > 8 instead.
    absolute = z.abs().clamp_min(1.0)
    inverse_square = absolute.reciprocal().square()
    tail = (
        (8.0 / (3.0 * math.pi * math.sqrt(5.0)))
        * 125.0
        * absolute.reciprocal().pow(5)
        * (
            1.0 / 5.0
            - 15.0 / 7.0 * inverse_square
            + 150.0 / 9.0 * inverse_square.square()
            - 1250.0 / 11.0 * inverse_square.pow(3)
            + 9375.0 / 13.0 * inverse_square.pow(4)
        )
    )
    return torch.where(z < -8.0, tail, torch.where(z > 8.0, 1.0 - tail, central)).clamp(
        0.0, 1.0
    )


def student_t5_crps(
    *, loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Analytic Student-t CRPS, including gradients through loc and scale."""
    z = (target - loc) / scale
    cdf = student_t5_cdf(z)
    pdf = (8.0 / (3.0 * math.pi * math.sqrt(5.0))) * (1.0 + z.square() / 5.0).pow(-3)
    beta_ratio = math.exp(
        math.lgamma(0.5)
        + math.lgamma(4.5)
        - math.lgamma(5.0)
        - 2.0 * (math.lgamma(0.5) + math.lgamma(2.5) - math.lgamma(3.0))
    )
    score = z * (2.0 * cdf - 1.0) + 0.5 * (
        pdf * (5.0 + z.square()) - math.sqrt(5.0) * beta_ratio
    )
    return (scale * score).clamp_min(0.0)
