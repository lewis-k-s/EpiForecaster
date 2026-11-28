"""Differentiable scores in log1p(per-100k) target space.

Student-t CRPS follows Jordan et al. (2019), doi:10.18637/jss.v090.i12.
Its CDF uses an exact trigonometric recurrence for integer df <= 100 and
64-point Gauss-Legendre integration otherwise. Degrees of freedom are fixed.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from scipy import stats

WIS_INTERVAL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
# log1p(per-100k) and log1p(concentration) targets cannot be negative.
TARGET_LOWER_BOUND = 0.0
_GL_NODES_ARRAY, _GL_WEIGHTS_ARRAY = np.polynomial.legendre.leggauss(64)
_GL_NODES = tuple(float(x) for x in _GL_NODES_ARRAY)
_GL_WEIGHTS = tuple(float(x) for x in _GL_WEIGHTS_ARRAY)


def student_t_cdf(z: torch.Tensor, df: float) -> torch.Tensor:
    """CDF of a standard Student-t with fixed, finite df > 2."""
    df = float(df)
    if not math.isfinite(df) or df <= 2:
        raise ValueError("Student-t df must be finite and > 2")
    theta = torch.atan(z / math.sqrt(df))
    normalizer = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(
        math.pi
    )
    if df.is_integer() and df <= 100:
        power = int(df) - 1
        sine, cosine = torch.sin(theta), torch.cos(theta)
        integral = theta if power % 2 == 0 else sine
        for n in range(2 if power % 2 == 0 else 3, power + 1, 2):
            integral = (sine * cosine.pow(n - 1) + (n - 1) * integral) / n
    else:
        nodes = torch.as_tensor(_GL_NODES, dtype=z.dtype, device=z.device)
        weights = torch.as_tensor(_GL_WEIGHTS, dtype=z.dtype, device=z.device)
        angles = theta.unsqueeze(-1) * (nodes + 1) / 2
        integral = theta / 2 * (angles.cos().pow(df - 1) * weights).sum(-1)
    return (0.5 + normalizer * integral).clamp(0, 1)


def normal_crps(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    z = (target - loc) / scale
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    return (scale * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))).clamp_min(0)


def censored_normal_nll(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """NLL of a Normal censored at the target lower bound.

    A target on the bound scores the probability mass at or below it, which
    cannot exceed one, so a collapsing scale earns nothing on an observed zero.
    A target above the bound scores the Normal density.
    """
    z = (target - loc) / scale
    density = 0.5 * z.square() + scale.log() + 0.5 * math.log(2 * math.pi)
    mass = -torch.special.log_ndtr((TARGET_LOWER_BOUND - loc) / scale)
    return torch.where(target <= TARGET_LOWER_BOUND, mass, density)


def censored_normal_crps(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """CRPS of a Normal censored at the target lower bound.

    Closed form of Jordan et al. (2019) for a point mass on the lower bound.
    Targets below the bound are scored as the bound.
    """
    bound = (TARGET_LOWER_BOUND - loc) / scale
    z = torch.maximum((target - loc) / scale, bound)
    bound_cdf = torch.special.ndtr(bound)
    bound_pdf = torch.exp(-0.5 * bound.square()) / math.sqrt(2 * math.pi)
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    standard = (
        z * (2 * torch.special.ndtr(z) - 1)
        + 2 * pdf
        - 2 * bound_pdf * bound_cdf
        - bound * bound_cdf.square()
        - (1 - torch.special.ndtr(bound * math.sqrt(2))) / math.sqrt(math.pi)
    )
    return (scale * standard).clamp_min(0)


def student_t_crps(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, df: float
) -> torch.Tensor:
    z = (target - loc) / scale
    cdf = student_t_cdf(z, float(df))
    log_pdf = (
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - (df + 1) / 2 * torch.log1p(z.square() / df)
    )
    log_beta1 = math.lgamma(0.5) + math.lgamma(df - 0.5) - math.lgamma(df)
    log_beta2 = math.lgamma(0.5) + math.lgamma(df / 2) - math.lgamma((df + 1) / 2)
    constant = 2 * math.sqrt(df) / (df - 1) * math.exp(log_beta1 - 2 * log_beta2)
    return (
        scale
        * (
            z * (2 * cdf - 1)
            + 2 * log_pdf.exp() * (df + z.square()) / (df - 1)
            - constant
        )
    ).clamp_min(0)


@lru_cache(maxsize=64)
def standard_wis_half_widths(family: str, df: float) -> tuple[float, ...]:
    """Fixed standard quantiles; never estimate these from forecast samples."""
    probabilities = (1 + np.asarray(WIS_INTERVAL_LEVELS)) / 2
    if family == "normal":
        values = stats.norm.ppf(probabilities)
    elif family == "student_t":
        if not math.isfinite(df) or df <= 2:
            raise ValueError("Student-t df must be finite and > 2")
        values = stats.t.ppf(probabilities, df)
    else:
        raise ValueError(f"Unsupported distribution family: {family}")
    return tuple(float(x) for x in values)


def clip_to_support[T: (torch.Tensor, np.ndarray)](values: T) -> T:
    """Clip forecast quantiles to the non-negative support of log1p targets."""
    return values.clip(min=TARGET_LOWER_BOUND)


def clipped_interval[T: (torch.Tensor, np.ndarray)](
    loc: T, half_width: T | float
) -> tuple[T, T]:
    """Central interval of the forecast censored at the target lower bound."""
    return clip_to_support(loc - half_width), clip_to_support(loc + half_width)


def distribution_wis(
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    half_widths: torch.Tensor,
) -> torch.Tensor:
    """WIS for the fixed 23-quantile grid, per forecast element.

    Every quantile is clipped at the target lower bound, which scores the
    forecast censored there: interval mass below the bound is not width.
    """
    low, high = clipped_interval(loc.unsqueeze(-1), scale.unsqueeze(-1) * half_widths)
    observed = target.unsqueeze(-1)
    alpha = 1 - torch.as_tensor(WIS_INTERVAL_LEVELS, dtype=loc.dtype, device=loc.device)
    weighted_intervals = (
        0.5 * alpha * (high - low)
        + (low - observed).clamp_min(0)
        + (observed - high).clamp_min(0)
    )
    median_error = (target - clip_to_support(loc)).abs()
    return (0.5 * median_error + weighted_intervals.sum(-1)) / (
        len(WIS_INTERVAL_LEVELS) + 0.5
    )
