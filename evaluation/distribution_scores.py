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


def distribution_wis(
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    half_widths: torch.Tensor,
) -> torch.Tensor:
    """Standard WIS for the fixed 23-quantile grid, per forecast element."""
    error = (target - loc).abs().unsqueeze(-1)
    widths = scale.unsqueeze(-1) * half_widths
    alpha = 1 - torch.as_tensor(WIS_INTERVAL_LEVELS, dtype=loc.dtype, device=loc.device)
    weighted_intervals = alpha * widths + (error - widths).clamp_min(0)
    return (0.5 * error.squeeze(-1) + weighted_intervals.sum(-1)) / (
        len(WIS_INTERVAL_LEVELS) + 0.5
    )
