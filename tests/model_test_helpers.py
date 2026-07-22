"""Shared setup helpers for model-focused tests."""

from typing import NamedTuple

import torch

from models.observation_heads import (
    ClinicalObservationHead,
    WastewaterObservationHead,
)
from models.sir_rollforward import SIRRollForward


class SIRScenario(NamedTuple):
    """Outputs and inputs for the standard two-node SIR scenario."""

    module: SIRRollForward
    result: dict[str, torch.Tensor]
    beta_t: torch.Tensor
    gamma_t: torch.Tensor
    mortality_t: torch.Tensor
    population: torch.Tensor
    S0: torch.Tensor
    I0: torch.Tensor
    R0: torch.Tensor


def run_standard_sir_scenario(
    *,
    horizon: int = 10,
    beta: float | None = None,
    module: SIRRollForward | None = None,
    beta_requires_grad: bool = False,
) -> SIRScenario:
    """Run the repeated two-node, 90% susceptible SIR test scenario."""
    module = module or SIRRollForward()
    beta_t = (
        torch.rand(2, horizon) * 0.5 if beta is None else torch.full((2, horizon), beta)
    )
    if beta_requires_grad:
        beta_t.requires_grad_(True)
    gamma_t = torch.full((2, horizon), 0.2)
    mortality_t = torch.zeros(2, horizon)
    population = torch.full((2,), 1000.0)
    S0 = torch.full((2,), 900.0)
    I0 = torch.full((2,), 100.0)
    R0 = torch.zeros(2)
    result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
    return SIRScenario(
        module,
        result,
        beta_t,
        gamma_t,
        mortality_t,
        population,
        S0,
        I0,
        R0,
    )


def make_observation_heads(
    *, frozen: bool = False, **kwargs: object
) -> tuple[ClinicalObservationHead, WastewaterObservationHead]:
    """Create matching clinical and wastewater heads."""
    if frozen:
        kwargs = {"learnable_kernel": False, "learnable_scale": False, **kwargs}
    return ClinicalObservationHead(**kwargs), WastewaterObservationHead(**kwargs)
