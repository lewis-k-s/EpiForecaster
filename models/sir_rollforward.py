"""
SIR Roll-Forward Module for Joint Inference-Observation Framework.

Implements differentiable discrete-time SIR dynamics using Euler integration.
This is the physics core (Stage 2) of the Joint Inference architecture.

The module rolls forward latent epidemiological states (S, I, R) given:
- Time-varying transmission rate beta_t
- Initial compartment states (S0, I0, R0)
- Fixed recovery rate gamma
- Population size N

Equations (discrete Euler):
    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - gamma * I
    dR/dt = gamma * I
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SIRRollForward(nn.Module):
    """
    Differentiable SIR roll-forward using discrete Euler integration.

    This module generates latent trajectories (S, I, R) from initial states
    and time-varying transmission rates. It enforces biological constraints
    (non-negativity, mass conservation) and provides physics residuals for
    training regularization.

    Args:
        gamma: Fixed recovery rate (default 0.2 = 5-day infectious period)
        dt: Time step size in days (default 1.0)
        enforce_nonnegativity: If True, clamps S, I, R to [0, N] after each step
        enforce_mass_conservation: If True, normalizes (S+I+R) to N after each step
    """

    def __init__(
        self,
        dt: float = 1.0,
        enforce_nonnegativity: bool = True,
        enforce_mass_conservation: bool = True,
        residual_clip: float = 1e4,
    ):
        super().__init__()
        self.dt = dt
        self.enforce_nonnegativity = enforce_nonnegativity
        self.enforce_mass_conservation = enforce_mass_conservation
        self.residual_clip = residual_clip

        logger.info(
            f"Initialized SIRRollForward: dt={dt}, "
            f"nonneg={enforce_nonnegativity}, mass_cons={enforce_mass_conservation}, "
            f"residual_clip={residual_clip}"
        )

    def forward(
        self,
        beta_t: torch.Tensor,
        gamma_t: torch.Tensor,
        mortality_t: torch.Tensor,
        S0: torch.Tensor,
        I0: torch.Tensor,
        R0: torch.Tensor,
        population: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Roll forward SIRD dynamics from initial states.

        Args:
            beta_t: Time-varying transmission rate [batch_size, horizon]
            gamma_t: Time-varying recovery rate [batch_size, horizon]
            mortality_t: Time-varying mortality rate [batch_size, horizon]
            S0: Initial susceptible population [batch_size]
            I0: Initial infected population [batch_size]
            R0: Initial recovered population [batch_size]
            population: Total population per batch element [batch_size]

        Returns:
            Dictionary containing:
                - S_trajectory: [batch_size, horizon+1] (includes S0)
                - I_trajectory: [batch_size, horizon+1] (includes I0)
                - R_trajectory: [batch_size, horizon+1] (includes R0)
                - D_trajectory: [batch_size, horizon+1] (accumulated deaths, starts at 0)
                - death_flow: [batch_size, horizon] (new deaths per step)
                - physics_residual: [batch_size, horizon] L2 residual of SIRD dynamics
                - beta_t: [batch_size, horizon] (echoed back for convenience)
        """
        if beta_t.ndim != 2:
            raise ValueError(
                f"beta_t must be 2D [batch_size, horizon], got {beta_t.shape}"
            )
        if gamma_t.ndim != 2:
            raise ValueError(
                f"gamma_t must be 2D [batch_size, horizon], got {gamma_t.shape}"
            )
        if mortality_t.ndim != 2:
            raise ValueError(
                f"mortality_t must be 2D [batch_size, horizon], got {mortality_t.shape}"
            )

        batch_size, horizon = beta_t.shape

        # Validate shapes
        if S0.shape != (batch_size,):
            raise ValueError(f"S0 shape {S0.shape} != ({batch_size},)")
        if I0.shape != (batch_size,):
            raise ValueError(f"I0 shape {I0.shape} != ({batch_size},)")
        if R0.shape != (batch_size,):
            raise ValueError(f"R0 shape {R0.shape} != ({batch_size},)")
        if population.shape != (batch_size,):
            raise ValueError(f"population shape {population.shape} != ({batch_size},)")

        if torch.any(population <= 0):
            raise ValueError("Population must be positive for all batch elements")

        # Validate and normalize initial states
        S0, I0, R0 = self._sanitize_initial_states(S0, I0, R0, population)

        # Initialize Deceased state at 0 (representing *new* deaths since start of window)
        # or we could require D0. For now, assuming D starts at 0 relative to window.
        D0 = torch.zeros_like(S0)

        # Use lists to collect states (avoids in-place operations for gradient flow)
        S_list = [S0]
        I_list = [I0]
        R_list = [R0]
        D_list = [D0]

        death_flow_list = []
        physics_residual_list = []

        # Roll forward
        for t in range(horizon):
            # Current states
            S_t = S_list[-1]
            I_t = I_list[-1]
            R_t = R_list[-1]
            D_t = D_list[-1]

            # Current rates
            beta = beta_t[:, t]
            gamma = gamma_t[:, t]
            mu = mortality_t[:, t]

            # SIRD derivatives
            # dS/dt = -beta * S * I / N
            # dI/dt = beta * S * I / N - gamma * I - mu * I
            # dR/dt = gamma * I
            # dD/dt = mu * I

            beta_SI_over_N = beta * S_t * I_t / population
            recovery_flow = gamma * I_t
            death_flow = mu * I_t

            dS_dt = -beta_SI_over_N
            dI_dt = beta_SI_over_N - recovery_flow - death_flow
            dR_dt = recovery_flow
            dD_dt = death_flow

            # Euler step
            S_next_raw = S_t + self.dt * dS_dt
            I_next_raw = I_t + self.dt * dI_dt
            R_next_raw = R_t + self.dt * dR_dt
            D_next_raw = D_t + self.dt * dD_dt

            # Apply constraints
            S_next = S_next_raw
            I_next = I_next_raw
            R_next = R_next_raw
            D_next = D_next_raw

            if self.enforce_nonnegativity:
                S_next = torch.clamp(S_next, min=0.0)
                I_next = torch.clamp(I_next, min=0.0)
                R_next = torch.clamp(R_next, min=0.0)
                D_next = torch.clamp(D_next, min=0.0)

            if self.enforce_mass_conservation:
                total = S_next + I_next + R_next + D_next
                # Scale to maintain S + I + R + D = N
                scale = population / (total + 1e-8)
                S_next = S_next * scale
                I_next = I_next * scale
                R_next = R_next * scale
                D_next = D_next * scale

            # Store next states
            S_list.append(S_next)
            I_list.append(I_next)
            R_list.append(R_next)
            D_list.append(D_next)

            # Store flow for observation heads
            death_flow_list.append(death_flow)

            # Compute physics residual on constrained step (clip at source for stability)
            dI_expected = beta_SI_over_N - recovery_flow - death_flow
            dI_actual = (I_next - I_t) / self.dt
            residual = (dI_actual - dI_expected) ** 2
            physics_residual_list.append(torch.clamp(residual, max=self.residual_clip))

        # Stack lists into tensors
        S_traj = torch.stack(S_list, dim=1)  # [B, H+1]
        I_traj = torch.stack(I_list, dim=1)  # [B, H+1]
        R_traj = torch.stack(R_list, dim=1)  # [B, H+1]
        D_traj = torch.stack(D_list, dim=1)  # [B, H+1]
        death_flow_traj = torch.stack(death_flow_list, dim=1)  # [B, H]
        physics_residual = torch.stack(physics_residual_list, dim=1)  # [B, H]

        return {
            "S_trajectory": S_traj,
            "I_trajectory": I_traj,
            "R_trajectory": R_traj,
            "D_trajectory": D_traj,
            "death_flow": death_flow_traj,
            "physics_residual": physics_residual,
            "beta_t": beta_t,
        }

    def _sanitize_initial_states(
        self,
        S0: torch.Tensor,
        I0: torch.Tensor,
        R0: torch.Tensor,
        population: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and normalize initial states to satisfy SIR constraints."""
        # Ensure consistent dtype between states and population
        target_dtype = S0.dtype
        if population.dtype != target_dtype:
            population = population.to(target_dtype)

        total = S0 + I0 + R0
        # Note: D0 is assumed 0 for the start of the window, so we check S+I+R against N.
        # If we start tracking accumulated D, we'd need D0 input.

        # Check non-negativity
        if torch.any(S0 < 0) or torch.any(I0 < 0) or torch.any(R0 < 0):
            raise ValueError("Initial states must be non-negative")

        # Check mass conservation (with small tolerance)
        if not torch.allclose(total, population, rtol=1e-3, atol=1e-3):
            if not self.enforce_mass_conservation:
                logger.warning(
                    "Initial states do not sum to population; proceeding without normalization."
                )
                return S0, I0, R0

            scale = population / (total + 1e-8)
            S0 = S0 * scale
            I0 = I0 * scale
            R0 = R0 * scale

        return S0, I0, R0

    def compute_sir_loss(
        self,
        S_trajectory: torch.Tensor,
        I_trajectory: torch.Tensor,
        R_trajectory: torch.Tensor,
        beta_t: torch.Tensor,
        gamma_t: torch.Tensor,
        mortality_t: torch.Tensor,
        population: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SIR consistency loss (physics regularizer).

        Args:
            S_trajectory: [batch_size, horizon+1]
            I_trajectory: [batch_size, horizon+1]
            R_trajectory: [batch_size, horizon+1]
            beta_t: [batch_size, horizon]
            gamma_t: [batch_size, horizon]
            mortality_t: [batch_size, horizon]
            population: [batch_size]

        Returns:
            Scalar loss value (mean squared residual across compartments)
        """
        # Extract states at each step
        S_t = S_trajectory[:, :-1]
        I_t = I_trajectory[:, :-1]
        R_t = R_trajectory[:, :-1]

        S_next = S_trajectory[:, 1:]
        I_next = I_trajectory[:, 1:]
        R_next = R_trajectory[:, 1:]

        N = population.unsqueeze(-1)

        # Compute actual derivatives
        dS_actual = (S_next - S_t) / self.dt
        dI_actual = (I_next - I_t) / self.dt
        dR_actual = (R_next - R_t) / self.dt

        # Compute expected derivatives (SIRD)
        beta_SI_over_N = beta_t * S_t * I_t / N
        recovery_flow = gamma_t * I_t
        death_flow = mortality_t * I_t

        dS_expected = -beta_SI_over_N
        dI_expected = beta_SI_over_N - recovery_flow - death_flow
        dR_expected = recovery_flow
        # Note: We don't check D residual here as D wasn't passed,
        # but constraining S, I, R effectively constrains D via mass conservation and flow.

        residual_S = dS_actual - dS_expected
        residual_I = dI_actual - dI_expected
        residual_R = dR_actual - dR_expected

        loss = (residual_S**2 + residual_I**2 + residual_R**2).mean()

        return loss

    def get_basic_reproduction_number(
        self,
        beta_t: torch.Tensor,
        gamma_t: torch.Tensor,
        mortality_t: torch.Tensor,
        S_trajectory: torch.Tensor,
        population: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute effective reproduction number R_t = beta_t * S_t / ((gamma + mu_t) * N).

        Note: With mortality, R0 depends on removal rate (gamma + mu).
        """
        S_t = S_trajectory[:, :-1]
        N = population.unsqueeze(-1)

        # Rate of removal = gamma + mortality_rate
        removal_rate = gamma_t + mortality_t
        # Clamp removal rate to avoid division by zero or negative
        removal_rate = torch.clamp(removal_rate, min=1e-8)

        R_t = beta_t * S_t / (removal_rate * N)
        return R_t

    def __repr__(self) -> str:
        return (
            f"SIRRollForward(dt={self.dt}, "
            f"nonneg={self.enforce_nonnegativity}, mass_cons={self.enforce_mass_conservation})"
        )
