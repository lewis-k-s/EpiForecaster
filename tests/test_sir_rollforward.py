"""
Isolated checks for SIR dynamics, constraints, gradients, and edge cases.
"""

import pytest
import torch

from models.sir_rollforward import SIRRollForward
from tests.model_test_helpers import run_standard_sir_scenario


class TestSIRPhysics:
    """Tests for SIR dynamics correctness."""

    def test_initial_state_in_trajectory(self):
        """Test that initial states appear at index 0 of trajectories."""
        batch_size = 3
        horizon = 5

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.3
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 2000.0, 5000.0])
        S0 = torch.tensor([950.0, 1900.0, 4750.0])
        I0 = torch.tensor([50.0, 100.0, 250.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Check initial states are preserved
        assert torch.allclose(result["S_trajectory"][:, 0], S0)
        assert torch.allclose(result["I_trajectory"][:, 0], I0)
        assert torch.allclose(result["R_trajectory"][:, 0], R0)

    def test_monotonic_susceptible_decrease(self):
        """Test that S never increases (in deterministic SIR)."""
        horizon = 10
        scenario = run_standard_sir_scenario(horizon=horizon, beta=0.5)
        S_traj = scenario.result["S_trajectory"]

        # Check that S is non-increasing
        for t in range(horizon):
            assert torch.all(S_traj[:, t + 1] <= S_traj[:, t] + 1e-6)

    def test_monotonic_recovered_increase(self):
        """Test that R never decreases."""
        horizon = 10
        scenario = run_standard_sir_scenario(horizon=horizon, beta=0.3)
        R_traj = scenario.result["R_trajectory"]

        # Check that R is non-decreasing
        for t in range(horizon):
            assert torch.all(R_traj[:, t + 1] >= R_traj[:, t] - 1e-6)

    def test_zero_beta_no_transmission(self):
        """Test that beta=0 means no new infections."""
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        # Zero transmission rate
        beta_t = torch.zeros(batch_size, horizon)
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # S should remain constant (no new infections)
        assert torch.allclose(
            result["S_trajectory"], S0.unsqueeze(1).expand(-1, horizon + 1)
        )

        # I should decrease exponentially (only recovery)
        # With gamma=0.2, dt=1: I_t = I0 * (1 - gamma)^t = I0 * 0.8^t
        expected_I = I0.unsqueeze(1) * (
            0.8 ** torch.arange(horizon + 1).float()
        ).unsqueeze(0)
        assert torch.allclose(result["I_trajectory"], expected_I, rtol=1e-4, atol=1e-4)


class TestSIRConstraints:
    """Tests for constraint enforcement."""

    def test_mass_conservation(self):
        """Test that S + I + R = N throughout trajectory."""
        batch_size = 3
        horizon = 20

        module = SIRRollForward(enforce_mass_conservation=True)

        beta_t = torch.rand(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 2000.0, 5000.0])
        S0 = population * 0.9
        I0 = population * 0.1
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        total = result["S_trajectory"] + result["I_trajectory"] + result["R_trajectory"]
        # Allow small deviation due to D not being summed here if D > 0
        # In SIRD, S+I+R+D = N.
        # Since mu=0 here, D=0 so S+I+R=N should hold.
        expected_total = population.unsqueeze(1).expand(-1, horizon + 1)

        assert torch.allclose(total, expected_total, rtol=1e-4, atol=1e-4)

    def test_non_negativity(self):
        """Test that S, I, R are always non-negative."""
        batch_size = 2
        horizon = 20

        module = SIRRollForward(enforce_nonnegativity=True)

        # Very high beta to stress test
        beta_t = torch.ones(batch_size, horizon) * 1.0
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([100.0, 100.0])
        S0 = torch.tensor([50.0, 50.0])
        I0 = torch.tensor([50.0, 50.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        assert torch.all(result["S_trajectory"] >= -1e-6)
        assert torch.all(result["I_trajectory"] >= -1e-6)
        assert torch.all(result["R_trajectory"] >= -1e-6)


class TestSIRGradients:
    """Tests for gradient flow and differentiability."""

    def test_initial_state_gradient_flow(self):
        """Test that gradients flow back through initial states."""
        batch_size = 2
        horizon = 5

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])

        S0 = torch.tensor([900.0, 900.0], requires_grad=True)
        I0 = torch.tensor([100.0, 100.0], requires_grad=True)
        R0 = torch.zeros(
            batch_size
        )  # R0 must be N - S0 - I0, so don't require grad independently

        # R0 is derived to maintain mass conservation
        R0 = population - S0 - I0

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        loss = result["S_trajectory"].sum() + result["I_trajectory"].sum()
        loss.backward()

        assert S0.grad is not None
        assert I0.grad is not None
        assert not torch.all(S0.grad == 0)
        assert not torch.all(I0.grad == 0)

    def test_gradient_through_sir_loss(self):
        """Test that SIR loss provides gradients."""
        scenario = run_standard_sir_scenario(beta_requires_grad=True)

        # Compute SIR loss
        loss = scenario.module.compute_sir_loss(
            scenario.result["S_trajectory"],
            scenario.result["I_trajectory"],
            scenario.result["R_trajectory"],
            scenario.beta_t,
            scenario.gamma_t,
            scenario.mortality_t,
            scenario.population,
        )

        loss.backward()

        assert scenario.beta_t.grad is not None
        assert not torch.all(scenario.beta_t.grad == 0)


class TestSIRLoss:
    """Tests for the SIR consistency loss computation."""

    def test_sir_loss_zero_for_perfect_dynamics(self):
        """Test that SIR loss is near zero when dynamics are perfectly satisfied."""
        # This is a bit tricky to test directly, but we can verify that
        # for small dt, the loss should be small
        batch_size = 2
        horizon = 5

        # Use very small dt for accurate Euler integration
        module = SIRRollForward(dt=0.01)

        beta_t = torch.ones(batch_size, horizon) * 0.3
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        loss = module.compute_sir_loss(
            result["S_trajectory"],
            result["I_trajectory"],
            result["R_trajectory"],
            beta_t,
            gamma_t,
            mortality_t,
            population,
        )

        # Loss should be very small with small dt
        assert loss.item() < 1.0


class TestReproductionNumber:
    """Tests for R_t computation."""

    def test_reproduction_number_formula(self):
        """Test that R_t = beta * S / (gamma * N)."""
        batch_size = 1
        horizon = 5

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.4
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0])
        S0 = torch.tensor([500.0])
        I0 = torch.tensor([500.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        R_t = module.get_basic_reproduction_number(
            beta_t,
            gamma_t,
            mortality_t,
            result["S_trajectory"],
            population,
        )

        # Expected: R_t = beta * S / (gamma * N) = 0.4 * S / (0.2 * 1000) = 0.4 * S / 200
        S_t = result["S_trajectory"][:, :-1]
        expected_R_t = beta_t * S_t / (0.2 * population.unsqueeze(1))

        assert torch.allclose(R_t, expected_R_t, rtol=1e-5)


class TestValidation:
    """Tests for input validation."""

    def test_invalid_initial_states_negative(self):
        """Test that negative initial states raise error."""
        module = SIRRollForward()

        beta_t = torch.rand(1, 5) * 0.5
        gamma_t = torch.ones(1, 5) * 0.2
        mortality_t = torch.zeros(1, 5)
        population = torch.tensor([1000.0])
        S0 = torch.tensor([900.0])
        I0 = torch.tensor([-100.0])  # Negative!
        R0 = torch.tensor([200.0])

        with pytest.raises(ValueError, match="non-negative"):
            module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

    def test_initial_states_mass_violation_normalized(self):
        """Test that S+I+R != N gets normalized when mass conservation is on."""
        module = SIRRollForward(enforce_mass_conservation=True)

        beta_t = torch.rand(1, 5) * 0.5
        gamma_t = torch.ones(1, 5) * 0.2
        mortality_t = torch.zeros(1, 5)
        population = torch.tensor([1000.0])
        S0 = torch.tensor([900.0])
        I0 = torch.tensor([200.0])  # Too high!
        R0 = torch.zeros(1)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        total = (
            result["S_trajectory"][:, 0]
            + result["I_trajectory"][:, 0]
            + result["R_trajectory"][:, 0]
        )
        assert torch.allclose(total, population, rtol=1e-4, atol=1e-4)

    def test_shape_mismatch_beta(self):
        """Test that mismatched beta shape raises error."""
        module = SIRRollForward()

        beta_t = torch.rand(2, 5)  # batch_size=2
        gamma_t = torch.ones(2, 5) * 0.2
        mortality_t = torch.zeros(2, 5)
        population = torch.tensor([1000.0])  # batch_size=1
        S0 = torch.tensor([900.0])
        I0 = torch.tensor([100.0])
        R0 = torch.zeros(1)

        with pytest.raises(ValueError):
            module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

    def test_invalid_population_non_positive(self):
        """Test that non-positive population raises error."""
        module = SIRRollForward()

        beta_t = torch.rand(1, 5) * 0.5
        gamma_t = torch.ones(1, 5) * 0.2
        mortality_t = torch.zeros(1, 5)
        population = torch.tensor([0.0])
        S0 = torch.tensor([0.0])
        I0 = torch.tensor([0.0])
        R0 = torch.tensor([0.0])

        with pytest.raises(ValueError, match="Population must be positive"):
            module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_initial_infected(self):
        """Test behavior when I0 = 0 (no epidemic)."""
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = population.clone()
        I0 = torch.zeros(batch_size)
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # With no infected, nothing should change
        assert torch.allclose(
            result["S_trajectory"], S0.unsqueeze(1).expand(-1, horizon + 1)
        )
        assert torch.allclose(
            result["I_trajectory"], torch.zeros(batch_size, horizon + 1)
        )
        assert torch.allclose(
            result["R_trajectory"], torch.zeros(batch_size, horizon + 1)
        )

    def test_all_initial_infected(self):
        """Test behavior when everyone initially infected."""
        batch_size = 1
        horizon = 10

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0])
        S0 = torch.zeros(batch_size)
        I0 = population.clone()
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Everyone should recover (no new infections possible)
        # With gamma=0.2, dt=1: I_t = 1000 * 0.8^t
        expected_I = population * (0.8 ** torch.arange(horizon + 1).float())

        assert torch.allclose(result["S_trajectory"], torch.zeros(1, horizon + 1))
        assert torch.allclose(result["I_trajectory"][0], expected_I, rtol=1e-4)

    def test_time_varying_beta(self):
        """Test with time-varying transmission rate."""
        batch_size = 1

        module = SIRRollForward()

        # Beta starts high, then drops
        beta_t = torch.cat(
            [
                torch.ones(1, 10) * 0.5,  # High transmission
                torch.ones(1, 10) * 0.1,  # Low transmission (interventions)
            ],
            dim=1,
        )
        gamma_t = torch.ones(1, 20) * 0.2
        mortality_t = torch.zeros(1, 20)

        population = torch.tensor([10000.0])
        S0 = torch.tensor([9900.0])
        I0 = torch.tensor([100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
        I_traj = result["I_trajectory"][0]

        # Should see peak during high-beta period
        peak_idx = torch.argmax(I_traj)

        # Peak should occur in first half (when beta is high)
        assert peak_idx <= 10

class TestSIRHDFlows:
    """Tests for hospitalized latent state and H -> D mortality path."""

    def test_sirhd_outputs_and_mass_conservation(self):
        batch_size = 2
        horizon = 4
        module = SIRRollForward(enforce_mass_conservation=True)

        beta_t = torch.zeros(batch_size, horizon)
        gamma_t = torch.ones(batch_size, horizon) * 0.1
        mortality_t = torch.zeros(batch_size, horizon)
        hospitalization_rate_t = torch.ones(batch_size, horizon) * 0.2
        hospital_recovery_t = torch.ones(batch_size, horizon) * 0.05
        hospital_mortality_t = torch.ones(batch_size, horizon) * 0.03
        population = torch.tensor([1000.0, 2000.0])
        S0 = population * 0.8
        I0 = population * 0.1
        H0 = population * 0.05
        R0 = population * 0.05

        result = module(
            beta_t=beta_t,
            gamma_t=gamma_t,
            mortality_t=mortality_t,
            S0=S0,
            I0=I0,
            R0=R0,
            population=population,
            hospitalization_rate_t=hospitalization_rate_t,
            hospital_recovery_t=hospital_recovery_t,
            hospital_mortality_t=hospital_mortality_t,
            H0=H0,
        )

        assert result["H_trajectory"].shape == (batch_size, horizon + 1)
        assert result["hospitalization_flow"].shape == (batch_size, horizon)
        assert result["hospital_death_flow"].shape == (batch_size, horizon)
        assert result["direct_death_flow"].shape == (batch_size, horizon)

        total = (
            result["S_trajectory"]
            + result["I_trajectory"]
            + result["H_trajectory"]
            + result["R_trajectory"]
            + result["D_trajectory"]
        )
        expected_total = population.unsqueeze(1).expand(-1, horizon + 1)
        assert torch.allclose(total, expected_total, rtol=1e-4, atol=1e-4)

    def test_hospital_death_flow_feeds_deceased_compartment(self):
        module = SIRRollForward(enforce_mass_conservation=False)
        beta_t = torch.zeros(1, 1)
        gamma_t = torch.zeros(1, 1)
        mortality_t = torch.zeros(1, 1)
        hospitalization_rate_t = torch.zeros(1, 1)
        hospital_recovery_t = torch.zeros(1, 1)
        hospital_mortality_t = torch.tensor([[0.25]])
        population = torch.tensor([1000.0])
        S0 = torch.tensor([800.0])
        I0 = torch.tensor([100.0])
        H0 = torch.tensor([80.0])
        R0 = torch.tensor([20.0])

        result = module(
            beta_t=beta_t,
            gamma_t=gamma_t,
            mortality_t=mortality_t,
            S0=S0,
            I0=I0,
            R0=R0,
            population=population,
            hospitalization_rate_t=hospitalization_rate_t,
            hospital_recovery_t=hospital_recovery_t,
            hospital_mortality_t=hospital_mortality_t,
            H0=H0,
        )

        expected_hospital_deaths = torch.tensor([[20.0]])
        assert torch.allclose(result["hospital_death_flow"], expected_hospital_deaths)
        assert torch.allclose(result["death_flow"], expected_hospital_deaths)
        assert torch.allclose(result["D_trajectory"][:, 1], torch.tensor([20.0]))
        assert torch.allclose(result["H_trajectory"][:, 1], torch.tensor([60.0]))
