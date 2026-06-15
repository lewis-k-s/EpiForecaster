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
        strict: bool = True,
    ):
        super().__init__()
        self.dt = dt
        self.enforce_nonnegativity = enforce_nonnegativity
        self.enforce_mass_conservation = enforce_mass_conservation
        self.residual_clip = residual_clip
        self.strict = strict

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
        hospitalization_rate_t: torch.Tensor | None = None,
        hospital_recovery_t: torch.Tensor | None = None,
        hospital_mortality_t: torch.Tensor | None = None,
        H0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Roll forward SIRHD dynamics from initial states.

        Args:
            beta_t: Time-varying transmission rate [batch_size, horizon]
            gamma_t: Time-varying recovery rate [batch_size, horizon]
            mortality_t: Time-varying mortality rate [batch_size, horizon]
            S0: Initial susceptible population [batch_size]
            I0: Initial infected population [batch_size]
            R0: Initial recovered population [batch_size]
            population: Total population per batch element [batch_size]
            hospitalization_rate_t: Optional I -> H rate [batch_size, horizon]
            hospital_recovery_t: Optional H -> R rate [batch_size, horizon]
            hospital_mortality_t: Optional H -> D rate [batch_size, horizon]
            H0: Optional initial hospitalized population [batch_size]

        Returns:
            Dictionary containing:
                - S_trajectory: [batch_size, horizon+1] (includes S0)
                - I_trajectory: [batch_size, horizon+1] (includes I0)
                - H_trajectory: [batch_size, horizon+1] (includes H0)
                - R_trajectory: [batch_size, horizon+1] (includes R0)
                - D_trajectory: [batch_size, horizon+1] (accumulated deaths, starts at 0)
                - hospitalization_flow: [batch_size, horizon] (new hospitalizations per step)
                - death_flow: [batch_size, horizon] (new deaths per step)
                - hospital_death_flow: [batch_size, horizon] (H -> D deaths per step)
                - direct_death_flow: [batch_size, horizon] (I -> D deaths per step)
                - physics_residual: [batch_size, horizon] L2 residual of SIRHD dynamics
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
        if hospitalization_rate_t is None:
            hospitalization_rate_t = torch.zeros_like(beta_t)
        if hospital_recovery_t is None:
            hospital_recovery_t = torch.zeros_like(beta_t)
        if hospital_mortality_t is None:
            hospital_mortality_t = torch.zeros_like(beta_t)
        if H0 is None:
            H0 = torch.zeros_like(S0)

        # Validate shapes (always safe in Dynamo relative to static shapes if trace permits,
        # but best to skip if not strict to avoid runtime checks during execute)
        if self.strict:
            if S0.shape != (batch_size,):
                raise ValueError(f"S0 shape {S0.shape} != ({batch_size},)")
            if I0.shape != (batch_size,):
                raise ValueError(f"I0 shape {I0.shape} != ({batch_size},)")
            if R0.shape != (batch_size,):
                raise ValueError(f"R0 shape {R0.shape} != ({batch_size},)")
            if H0.shape != (batch_size,):
                raise ValueError(f"H0 shape {H0.shape} != ({batch_size},)")
            if population.shape != (batch_size,):
                raise ValueError(
                    f"population shape {population.shape} != ({batch_size},)"
                )
            for name, value in [
                ("hospitalization_rate_t", hospitalization_rate_t),
                ("hospital_recovery_t", hospital_recovery_t),
                ("hospital_mortality_t", hospital_mortality_t),
            ]:
                if value.shape != (batch_size, horizon):
                    raise ValueError(
                        f"{name} shape {value.shape} != ({batch_size}, {horizon})"
                    )

            if torch.any(population <= 0):
                raise ValueError("Population must be positive for all batch elements")

        # Validate and normalize initial states
        S0, I0, H0, R0 = self._sanitize_initial_states(S0, I0, H0, R0, population)

        # Initialize Deceased state at 0 (representing *new* deaths since start of window)
        # or we could require D0. For now, assuming D starts at 0 relative to window.
        D0 = torch.zeros_like(S0)

        # Use lists to collect states (avoids in-place operations for gradient flow)
        S_list = [S0]
        I_list = [I0]
        H_list = [H0]
        R_list = [R0]
        D_list = [D0]

        hospitalization_flow_list = []
        death_flow_list = []
        hospital_death_flow_list = []
        direct_death_flow_list = []
        physics_residual_list = []

        # Roll forward
        for t in range(horizon):
            # Current states
            S_t = S_list[-1]
            I_t = I_list[-1]
            H_t = H_list[-1]
            R_t = R_list[-1]
            D_t = D_list[-1]

            # Current rates
            beta = beta_t[:, t]
            gamma = gamma_t[:, t]
            mu = mortality_t[:, t]
            alpha = hospitalization_rate_t[:, t]
            rho = hospital_recovery_t[:, t]
            mu_h = hospital_mortality_t[:, t]

            # SIRHD derivatives
            # dS/dt = -beta * S * I / N
            # dI/dt = beta * S * I / N - gamma * I - alpha * I - mu * I
            # dH/dt = alpha * I - rho * H - mu_h * H
            # dR/dt = gamma * I + rho * H
            # dD/dt = mu * I + mu_h * H

            beta_SI_over_N = beta * S_t * I_t / population
            recovery_flow = gamma * I_t
            hospitalization_flow = alpha * I_t
            hospital_recovery_flow = rho * H_t
            direct_death_flow = mu * I_t
            hospital_death_flow = mu_h * H_t
            death_flow = direct_death_flow + hospital_death_flow

            dS_dt = -beta_SI_over_N
            dI_dt = (
                beta_SI_over_N
                - recovery_flow
                - hospitalization_flow
                - direct_death_flow
            )
            dH_dt = hospitalization_flow - hospital_recovery_flow - hospital_death_flow
            dR_dt = recovery_flow + hospital_recovery_flow
            dD_dt = death_flow

            # Euler step
            S_next_raw = S_t + self.dt * dS_dt
            I_next_raw = I_t + self.dt * dI_dt
            H_next_raw = H_t + self.dt * dH_dt
            R_next_raw = R_t + self.dt * dR_dt
            D_next_raw = D_t + self.dt * dD_dt

            # Apply constraints
            S_next = S_next_raw
            I_next = I_next_raw
            H_next = H_next_raw
            R_next = R_next_raw
            D_next = D_next_raw

            if self.enforce_nonnegativity:
                S_next = torch.clamp(S_next, min=0.0)
                I_next = torch.clamp(I_next, min=0.0)
                H_next = torch.clamp(H_next, min=0.0)
                R_next = torch.clamp(R_next, min=0.0)
                D_next = torch.clamp(D_next, min=0.0)

            if self.enforce_mass_conservation:
                total = S_next + I_next + H_next + R_next + D_next
                # Scale to maintain S + I + H + R + D = N
                scale = population / (total + 1e-8)
                S_next = S_next * scale
                I_next = I_next * scale
                H_next = H_next * scale
                R_next = R_next * scale
                D_next = D_next * scale

            # Store next states
            S_list.append(S_next)
            I_list.append(I_next)
            H_list.append(H_next)
            R_list.append(R_next)
            D_list.append(D_next)

            # Store flow for observation heads
            hospitalization_flow_list.append(hospitalization_flow)
            death_flow_list.append(death_flow)
            hospital_death_flow_list.append(hospital_death_flow)
            direct_death_flow_list.append(direct_death_flow)

            # Compute physics residual on constrained step (clip at source for stability)
            dI_expected = (
                beta_SI_over_N
                - recovery_flow
                - hospitalization_flow
                - direct_death_flow
            )
            dI_actual = (I_next - I_t) / self.dt
            dH_expected = (
                hospitalization_flow - hospital_recovery_flow - hospital_death_flow
            )
            dH_actual = (H_next - H_t) / self.dt
            residual = (dI_actual - dI_expected) ** 2 + (dH_actual - dH_expected) ** 2
            physics_residual_list.append(torch.clamp(residual, max=self.residual_clip))

        # Stack lists into tensors
        S_traj = torch.stack(S_list, dim=1)  # [B, H+1]
        I_traj = torch.stack(I_list, dim=1)  # [B, H+1]
        H_traj = torch.stack(H_list, dim=1)  # [B, H+1]
        R_traj = torch.stack(R_list, dim=1)  # [B, H+1]
        D_traj = torch.stack(D_list, dim=1)  # [B, H+1]
        hospitalization_flow_traj = torch.stack(
            hospitalization_flow_list, dim=1
        )  # [B, H]
        death_flow_traj = torch.stack(death_flow_list, dim=1)  # [B, H]
        hospital_death_flow_traj = torch.stack(
            hospital_death_flow_list, dim=1
        )  # [B, H]
        direct_death_flow_traj = torch.stack(direct_death_flow_list, dim=1)  # [B, H]
        physics_residual = torch.stack(physics_residual_list, dim=1)  # [B, H]

        return {
            "S_trajectory": S_traj,
            "I_trajectory": I_traj,
            "H_trajectory": H_traj,
            "R_trajectory": R_traj,
            "D_trajectory": D_traj,
            "hospitalization_flow": hospitalization_flow_traj,
            "death_flow": death_flow_traj,
            "hospital_death_flow": hospital_death_flow_traj,
            "direct_death_flow": direct_death_flow_traj,
            "physics_residual": physics_residual,
            "beta_t": beta_t,
        }

    def _sanitize_initial_states(
        self,
        S0: torch.Tensor,
        I0: torch.Tensor,
        H0: torch.Tensor,
        R0: torch.Tensor,
        population: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and normalize initial states to satisfy SIRHD constraints."""
        # Ensure consistent dtype between states and population
        target_dtype = S0.dtype
        if population.dtype != target_dtype:
            population = population.to(target_dtype)

        total = S0 + I0 + H0 + R0
        # Note: D0 is assumed 0 for the start of the window, so we check S+I+H+R against N.
        # If we start tracking accumulated D, we'd need D0 input.

        if self.strict:
            # Check non-negativity
            if (
                torch.any(S0 < 0)
                or torch.any(I0 < 0)
                or torch.any(H0 < 0)
                or torch.any(R0 < 0)
            ):
                raise ValueError("Initial states must be non-negative")

            # Check mass conservation (with small tolerance)
            if not torch.allclose(total, population, rtol=1e-3, atol=1e-3):
                if not self.enforce_mass_conservation:
                    logger.warning(
                        "Initial states do not sum to population; proceeding without normalization."
                    )
                    return S0, I0, H0, R0

                scale = population / (total + 1e-8)
                S0 = S0 * scale
                I0 = I0 * scale
                H0 = H0 * scale
                R0 = R0 * scale
        else:
            if self.enforce_mass_conservation:
                # Branchless scaling for Dynamo
                scale = population / (total + 1e-8)
                S0 = S0 * scale
                I0 = I0 * scale
                H0 = H0 * scale
                R0 = R0 * scale

        return S0, I0, H0, R0

    def compute_sir_loss(
        self,
        S_trajectory: torch.Tensor,
        I_trajectory: torch.Tensor,
        R_trajectory: torch.Tensor,
        beta_t: torch.Tensor,
        gamma_t: torch.Tensor,
        mortality_t: torch.Tensor,
        population: torch.Tensor,
        H_trajectory: torch.Tensor | None = None,
        hospitalization_rate_t: torch.Tensor | None = None,
        hospital_recovery_t: torch.Tensor | None = None,
        hospital_mortality_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SIR/SIRHD consistency loss (physics regularizer).

        Args:
            S_trajectory: [batch_size, horizon+1]
            I_trajectory: [batch_size, horizon+1]
            R_trajectory: [batch_size, horizon+1]
            beta_t: [batch_size, horizon]
            gamma_t: [batch_size, horizon]
            mortality_t: [batch_size, horizon]
            population: [batch_size]
            H_trajectory: Optional hospitalized trajectory [batch_size, horizon+1]
            hospitalization_rate_t: Optional I -> H rate [batch_size, horizon]
            hospital_recovery_t: Optional H -> R rate [batch_size, horizon]
            hospital_mortality_t: Optional H -> D rate [batch_size, horizon]

        Returns:
            Scalar loss value (mean squared residual across compartments)
        """
        # Extract states at each step
        S_t = S_trajectory[:, :-1]
        I_t = I_trajectory[:, :-1]
        R_t = R_trajectory[:, :-1]
        H_t = H_trajectory[:, :-1] if H_trajectory is not None else None

        S_next = S_trajectory[:, 1:]
        I_next = I_trajectory[:, 1:]
        R_next = R_trajectory[:, 1:]
        H_next = H_trajectory[:, 1:] if H_trajectory is not None else None

        N = population.unsqueeze(-1)

        # Compute actual derivatives
        dS_actual = (S_next - S_t) / self.dt
        dI_actual = (I_next - I_t) / self.dt
        dR_actual = (R_next - R_t) / self.dt
        dH_actual = (H_next - H_t) / self.dt if H_t is not None else None

        # Compute expected derivatives. When H/rate inputs are omitted, this
        # reduces to the historical SIRD helper behavior.
        beta_SI_over_N = beta_t * S_t * I_t / N
        recovery_flow = gamma_t * I_t
        direct_death_flow = mortality_t * I_t
        if H_t is None:
            hospitalization_flow = torch.zeros_like(direct_death_flow)
            hospital_recovery_flow = torch.zeros_like(direct_death_flow)
            hospital_death_flow = torch.zeros_like(direct_death_flow)
        else:
            if hospitalization_rate_t is None:
                hospitalization_rate_t = torch.zeros_like(beta_t)
            if hospital_recovery_t is None:
                hospital_recovery_t = torch.zeros_like(beta_t)
            if hospital_mortality_t is None:
                hospital_mortality_t = torch.zeros_like(beta_t)
            hospitalization_flow = hospitalization_rate_t * I_t
            hospital_recovery_flow = hospital_recovery_t * H_t
            hospital_death_flow = hospital_mortality_t * H_t

        dS_expected = -beta_SI_over_N
        dI_expected = (
            beta_SI_over_N - recovery_flow - hospitalization_flow - direct_death_flow
        )
        dR_expected = recovery_flow + hospital_recovery_flow

        residual_S = dS_actual - dS_expected
        residual_I = dI_actual - dI_expected
        residual_R = dR_actual - dR_expected
        residual_terms = residual_S**2 + residual_I**2 + residual_R**2
        if dH_actual is not None:
            dH_expected = (
                hospitalization_flow - hospital_recovery_flow - hospital_death_flow
            )
            residual_H = dH_actual - dH_expected
            residual_terms = residual_terms + residual_H**2

        loss = residual_terms.mean()

        return loss

    def get_basic_reproduction_number(
        self,
        beta_t: torch.Tensor,
        gamma_t: torch.Tensor,
        mortality_t: torch.Tensor,
        S_trajectory: torch.Tensor,
        population: torch.Tensor,
        hospitalization_rate_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute effective reproduction number R_t = beta_t * S_t / (removal_t * N).

        Note: With hospitalization and mortality, R_t depends on the total removal
        rate from infectious state: gamma + mortality + hospitalization.
        """
        S_t = S_trajectory[:, :-1]
        N = population.unsqueeze(-1)

        # Rate of removal from I = gamma + direct mortality + hospitalization.
        if hospitalization_rate_t is None:
            hospitalization_rate_t = torch.zeros_like(gamma_t)
        removal_rate = gamma_t + mortality_t + hospitalization_rate_t
        # Clamp removal rate to avoid division by zero or negative
        removal_rate = torch.clamp(removal_rate, min=1e-8)

        R_t = beta_t * S_t / (removal_rate * N)
        return R_t

    def __repr__(self) -> str:
        return (
            f"SIRRollForward(dt={self.dt}, "
            f"nonneg={self.enforce_nonnegativity}, mass_cons={self.enforce_mass_conservation})"
        )
