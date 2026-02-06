"""
Tests for SIR Roll-Forward Module.

Tests cover:
1. Basic forward pass and shape validation
2. Physics correctness (SIR dynamics)
3. Constraint enforcement (non-negativity, mass conservation)
4. Gradient flow (differentiability)
5. Edge cases (zero initial infected, very high beta, etc.)
"""

import pytest
import torch

from models.sir_rollforward import SIRRollForward


class TestSIRRollForwardBasics:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test module initialization with default parameters."""
        module = SIRRollForward()
        # Gamma and mortality are no longer in init
        assert module.dt == 1.0
        assert module.enforce_nonnegativity is True
        assert module.enforce_mass_conservation is True

    def test_initialization_custom_params(self):
        """Test module initialization with custom parameters."""
        module = SIRRollForward(
            dt=0.5,
            enforce_nonnegativity=False,
            enforce_mass_conservation=False,
        )
        assert module.dt == 0.5
        assert module.enforce_nonnegativity is False
        assert module.enforce_mass_conservation is False

    def test_forward_output_shapes(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        horizon = 7

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5  # Random beta in [0, 0.5]
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 2000.0, 5000.0, 10000.0])

        # Initial state: mostly susceptible, some infected
        S0 = population * 0.95
        I0 = population * 0.05
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Check all outputs exist
        assert "S_trajectory" in result
        assert "I_trajectory" in result
        assert "R_trajectory" in result
        assert "physics_residual" in result
        assert "beta_t" in result

        # Check shapes (trajectories include initial state)
        assert result["S_trajectory"].shape == (batch_size, horizon + 1)
        assert result["I_trajectory"].shape == (batch_size, horizon + 1)
        assert result["R_trajectory"].shape == (batch_size, horizon + 1)
        assert result["physics_residual"].shape == (batch_size, horizon)
        assert result["beta_t"].shape == (batch_size, horizon)

    def test_forward_device_consistency(self):
        """Test that all outputs are on the same device as inputs."""
        batch_size = 2
        horizon = 5

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 2000.0])
        S0 = population * 0.9
        I0 = population * 0.1
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # All outputs should be on CPU (same as inputs)
        for key, tensor in result.items():
            assert tensor.device == beta_t.device


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
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        # High transmission rate
        beta_t = torch.ones(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
        S_traj = result["S_trajectory"]

        # Check that S is non-increasing
        for t in range(horizon):
            assert torch.all(S_traj[:, t + 1] <= S_traj[:, t] + 1e-6)

    def test_monotonic_recovered_increase(self):
        """Test that R never decreases."""
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.3
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
        R_traj = result["R_trajectory"]

        # Check that R is non-decreasing
        for t in range(horizon):
            assert torch.all(R_traj[:, t + 1] >= R_traj[:, t] - 1e-6)

    def test_epidemic_peak_pattern(self):
        """Test that I increases then decreases during epidemic."""
        batch_size = 1
        horizon = 50

        module = SIRRollForward()

        # High transmission rate initially
        beta_t = torch.ones(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([10000.0])
        S0 = torch.tensor([9900.0])
        I0 = torch.tensor([100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
        I_traj = result["I_trajectory"][0]

        # Find peak
        peak_idx = torch.argmax(I_traj)

        # Should have initial increase
        assert peak_idx > 0

        # Should eventually decrease (if horizon is long enough)
        if horizon > 20:
            assert I_traj[-1] < I_traj[peak_idx]

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

    def test_beta_gradient_flow(self):
        """Test that gradients flow back through beta_t."""
        batch_size = 2
        horizon = 5

        module = SIRRollForward()

        # Create beta as a leaf tensor for gradient tracking
        beta_raw = torch.rand(batch_size, horizon)
        beta_t = (beta_raw * 0.5).requires_grad_(True)
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Compute some loss on the output
        loss = result["I_trajectory"].sum()
        loss.backward()

        # Check that beta_t received gradients
        assert beta_t.grad is not None
        assert beta_t.grad.shape == beta_t.shape
        assert not torch.all(beta_t.grad == 0)

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
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        # Create beta as a leaf tensor for gradient tracking
        beta_raw = torch.rand(batch_size, horizon)
        beta_t = (beta_raw * 0.5).requires_grad_(True)
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Compute SIR loss
        loss = module.compute_sir_loss(
            result["S_trajectory"],
            result["I_trajectory"],
            result["R_trajectory"],
            beta_t,
            gamma_t,
            mortality_t,
            population,
        )

        loss.backward()

        assert beta_t.grad is not None
        assert not torch.all(beta_t.grad == 0)


class TestSIRLoss:
    """Tests for the SIR consistency loss computation."""

    def test_sir_loss_non_negative(self):
        """Test that SIR loss is always non-negative."""
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5
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

        assert loss.item() >= 0

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

    def test_basic_reproduction_number_shape(self):
        """Test that R_t has correct shape."""
        batch_size = 2
        horizon = 10

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        R_t = module.get_basic_reproduction_number(
            beta_t,
            gamma_t,
            mortality_t,
            result["S_trajectory"],
            population,
        )

        assert R_t.shape == (batch_size, horizon)

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

    def test_very_long_horizon(self):
        """Test stability with long trajectories."""
        batch_size = 1
        horizon = 200

        module = SIRRollForward()

        beta_t = torch.ones(batch_size, horizon) * 0.3
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([10000.0])
        S0 = torch.tensor([9900.0])
        I0 = torch.tensor([100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Check mass conservation throughout
        total = result["S_trajectory"] + result["I_trajectory"] + result["R_trajectory"]
        expected_total = population.unsqueeze(1).expand(-1, horizon + 1)

        assert torch.allclose(total, expected_total, rtol=1e-4, atol=1e-4)

        # In the long run, epidemic should die out (I -> 0)
        assert result["I_trajectory"][0, -1] < result["I_trajectory"][0, 0]

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

    def test_single_step(self):
        """Test with horizon=1 (single step)."""
        batch_size = 2
        horizon = 1

        module = SIRRollForward()

        beta_t = torch.rand(batch_size, horizon) * 0.5
        gamma_t = torch.ones(batch_size, horizon) * 0.2
        mortality_t = torch.zeros(batch_size, horizon)
        population = torch.tensor([1000.0, 1000.0])
        S0 = torch.tensor([900.0, 900.0])
        I0 = torch.tensor([100.0, 100.0])
        R0 = torch.zeros(batch_size)

        result = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)

        # Should still produce correct shapes
        assert result["S_trajectory"].shape == (batch_size, 2)
        assert result["I_trajectory"].shape == (batch_size, 2)
        assert result["R_trajectory"].shape == (batch_size, 2)
