"""
Tests for Observation Heads (DelayKernel, SheddingConvolution, ClinicalObservationHead, WastewaterObservationHead).

Tests cover:
1. Basic forward pass and shape validation
2. Causality (no future information leakage)
3. Kernel normalization (sums to 1)
4. Dilution physics (population scaling in wastewater)
5. Gradient flow through learnable parameters
6. Edge cases (zero infections, constant population, time-varying population)
"""

import pytest
import torch


from models.observation_heads import (
    _GAMMA_SCALE_MAX,
    _GAMMA_SCALE_MIN,
    _GAMMA_SCALE_WW_MAX,
    _GAMMA_SCALE_WW_MIN,
    _GAMMA_SHAPE_MAX,
    _GAMMA_SHAPE_MIN,
    ClinicalObservationHead,
    CompositeObservationLoss,
    DelayKernel,
    MAELoss,
    MSELoss,
    SheddingConvolution,
    UnscaledMSELoss,
    WastewaterObservationHead,
    _inverse_bounded_sigmoid,
)
from tests.model_test_helpers import make_observation_heads


def _rand_tensor(*shape):
    """Create random tensor with model dtype."""
    return torch.rand(*shape, dtype=torch.float32)


def _zeros_tensor(*shape):
    """Create zeros tensor with model dtype."""
    return torch.zeros(*shape, dtype=torch.float32)


def _ones_tensor(*shape):
    """Create ones tensor with model dtype."""
    return torch.ones(*shape, dtype=torch.float32)


def _zeros_like(tensor):
    """Create zeros like tensor with same dtype."""
    return torch.zeros_like(tensor, dtype=torch.float32)


class TestDelayKernelBasics:
    """Basic functionality tests for DelayKernel."""

    def test_initialization_default(self):
        """Test default Gamma(5, 2) initialization."""
        kernel = DelayKernel()
        assert kernel.kernel_length == 21
        assert kernel.gamma_shape == 5.0
        assert kernel.gamma_scale == 2.0
        assert kernel.learnable is True
        assert isinstance(kernel.kernel, torch.nn.Parameter)

    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        kernel = DelayKernel(
            kernel_length=14,
            gamma_shape=3.0,
            gamma_scale=1.5,
            learnable=False,
        )
        assert kernel.kernel_length == 14
        assert kernel.gamma_shape == 3.0
        assert kernel.gamma_scale == 1.5
        assert kernel.learnable is False
        # When not learnable, kernel should be a buffer
        assert not isinstance(kernel.kernel, torch.nn.Parameter)

    def test_forward_output_shape(self):
        """Test that forward produces correct output shape."""
        batch_size = 4
        time_steps = 30

        kernel = DelayKernel(kernel_length=21)
        I_trajectory = _rand_tensor(batch_size, time_steps) * 100

        output = kernel(I_trajectory)

        assert output.shape == (batch_size, time_steps)

    def test_kernel_normalization(self):
        """Test that kernel is normalized to sum to 1."""
        kernel = DelayKernel(kernel_length=21)

        weights = kernel.get_kernel_weights()
        assert torch.allclose(
            weights.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5
        )

    def test_mean_delay_computation(self):
        """Test mean delay computation is reasonable."""
        kernel = DelayKernel(kernel_length=21, gamma_shape=5.0, gamma_scale=2.0)

        mean_delay = kernel.get_mean_delay()
        # Expected mean: shape × scale = 10 days
        assert 8 < mean_delay < 12  # Allow some tolerance

    def test_free_parameterization_preserves_signed_weights(self):
        """Free kernels should expose raw signed weights instead of simplex weights."""
        kernel = DelayKernel(kernel_length=3, learnable=True, parameterization="free")

        with torch.no_grad():
            kernel.kernel.copy_(
                torch.tensor([-0.5, 0.25, 1.5], dtype=kernel.kernel.dtype)
            )

        weights = kernel.get_kernel_weights()
        expected = torch.tensor([-0.5, 0.25, 1.5], dtype=weights.dtype)

        assert torch.allclose(weights, expected)

    def test_mean_delay_requires_simplex_kernel(self):
        """Mean delay is only well-defined for normalized positive kernels."""
        kernel = DelayKernel(kernel_length=5, learnable=True, parameterization="free")

        with pytest.raises(ValueError, match="only defined for normalized kernels"):
            kernel.get_mean_delay()


class TestDelayKernelCausality:
    """Tests for causal properties (no future leakage)."""

    def test_causality_zero_initial_conditions(self):
        """Test that output at t=0 only depends on I_0 (no future info)."""
        kernel = DelayKernel(kernel_length=5, learnable=False)

        batch_size = 1
        time_steps = 10

        # Create trajectory with all zeros except at position 5
        I_trajectory = _zeros_tensor(batch_size, time_steps)
        I_trajectory[0, 5] = 100.0

        output = kernel(I_trajectory)

        # Output at positions 0-4 should be 0 (causality)
        # The impulse at t=5 should only affect outputs t >= 5
        assert torch.all(output[0, 0:5] < 1e-6)
        assert output[0, 5:].sum() > 1e-3  # Should have some downstream signal

    def test_causality_impulse_response(self):
        """Test impulse response shows correct delay."""
        kernel = DelayKernel(
            kernel_length=7, gamma_shape=3.0, gamma_scale=1.0, learnable=False
        )

        batch_size = 1
        time_steps = 20

        # Delta function at t=5
        I_trajectory = _zeros_tensor(batch_size, time_steps)
        I_trajectory[0, 5] = 100.0

        output = kernel(I_trajectory)

        # Find first significant output (above threshold)
        threshold = 1.0
        first_response = torch.where(output[0] > threshold)[0]

        # Response should start at or after t=5 (causality)
        assert len(first_response) > 0
        assert first_response[0] >= 5

    def test_impulse_response_matches_forward_kernel_order(self):
        """An impulse should reproduce the kernel in increasing lag order."""
        kernel = DelayKernel(
            kernel_length=7,
            gamma_shape=3.0,
            gamma_scale=1.0,
            learnable=False,
        )

        I_trajectory = _zeros_tensor(1, 20)
        I_trajectory[0, 5] = 1.0

        output = kernel(I_trajectory)
        weights = kernel.get_kernel_weights()

        assert torch.allclose(
            output[0, 5 : 5 + kernel.kernel_length],
            weights,
            atol=1e-6,
        )


class TestDelayKernelGradients:
    """Tests for gradient flow."""

    def test_learnable_kernel_gradient_flow(self):
        """Test gradients flow through learnable kernel."""
        kernel = DelayKernel(kernel_length=7, learnable=True)

        I_trajectory = _rand_tensor(2, 10) * 100
        I_trajectory.requires_grad = True

        output = kernel(I_trajectory)
        loss = output.sum()
        loss.backward()

        # Kernel should have gradients
        assert kernel.kernel.grad is not None
        assert not torch.all(kernel.kernel.grad == 0)
        # Input should also have gradients
        assert I_trajectory.grad is not None

    def test_frozen_kernel_no_gradient(self):
        """Test frozen kernel doesn't compute gradients."""
        kernel = DelayKernel(kernel_length=7, learnable=False)

        # Create leaf tensor properly
        I_raw = _rand_tensor(2, 10)
        I_trajectory = (I_raw * 100).requires_grad_(True)
        output = kernel(I_trajectory)
        loss = output.sum()
        loss.backward()

        # Kernel is a buffer, not a parameter, so no grad accumulated on it
        assert not isinstance(kernel.kernel, torch.nn.Parameter)
        # Input should have grad (check the leaf tensor)
        assert I_trajectory.grad is not None

    def test_learnable_kernel_sanitizes_non_finite_logits(self):
        """NaN/Inf kernel logits should not create non-finite forward/backward values."""
        kernel = DelayKernel(kernel_length=7, learnable=True)
        with torch.no_grad():
            kernel.kernel[0] = torch.tensor(float("nan"), dtype=kernel.kernel.dtype)
            kernel.kernel[1] = torch.tensor(float("inf"), dtype=kernel.kernel.dtype)
            kernel.kernel[2] = torch.tensor(float("-inf"), dtype=kernel.kernel.dtype)

        I_trajectory = _rand_tensor(2, 10).requires_grad_(True)
        output = kernel(I_trajectory)
        assert torch.isfinite(output).all()

        loss = output.sum()
        loss.backward()
        assert kernel.kernel.grad is not None
        assert torch.isfinite(kernel.kernel.grad).all()


class TestDelayKernelGamma:
    """Two-parameter learnable Gamma delay kernel."""

    _MATCHED_SHAPE = 12.25
    _MATCHED_SCALE = 4.0 / 7.0

    def test_gamma_learns_exactly_two_parameters(self):
        kernel = DelayKernel(
            kernel_length=21,
            gamma_shape=self._MATCHED_SHAPE,
            gamma_scale=self._MATCHED_SCALE,
            learnable=True,
            parameterization="gamma",
        )

        named = dict(kernel.named_parameters())
        assert set(named) == {"gamma_shape_raw", "gamma_scale_raw"}
        assert named["gamma_shape_raw"].numel() == 1
        assert named["gamma_scale_raw"].numel() == 1

    def test_gamma_kernel_normalized_and_generator_matched(self):
        kernel = DelayKernel(
            kernel_length=21,
            gamma_shape=self._MATCHED_SHAPE,
            gamma_scale=self._MATCHED_SCALE,
            learnable=True,
            parameterization="gamma",
        )

        weights = kernel.get_kernel_weights()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert (weights >= 0).all()

        shape, scale = kernel.get_gamma_parameters()
        assert shape == pytest.approx(self._MATCHED_SHAPE)
        assert scale == pytest.approx(self._MATCHED_SCALE)

        # Truncated discrete Gamma(12.25, 4/7) over lags 0..20 has mean ~7.
        mean_delay = kernel.get_mean_delay()
        assert 6.5 < mean_delay < 7.5
        std_delay = kernel.get_delay_std()
        assert 1.5 < std_delay < 2.5

    def test_gamma_frozen_buffer_matches_learnable_initialization(self):
        frozen = DelayKernel(
            kernel_length=21,
            gamma_shape=self._MATCHED_SHAPE,
            gamma_scale=self._MATCHED_SCALE,
            learnable=False,
            parameterization="gamma",
        )
        learnable = DelayKernel(
            kernel_length=21,
            gamma_shape=self._MATCHED_SHAPE,
            gamma_scale=self._MATCHED_SCALE,
            learnable=True,
            parameterization="gamma",
        )

        assert not hasattr(frozen, "kernel")
        assert "gamma_kernel" in frozen.state_dict()
        assert torch.allclose(
            frozen.get_kernel_weights(), learnable.get_kernel_weights()
        )
        assert frozen.get_gamma_parameters() == (
            pytest.approx(self._MATCHED_SHAPE),
            pytest.approx(self._MATCHED_SCALE),
        )

    def test_frozen_gamma_rejects_simplex_checkpoint_state(self):
        simplex = DelayKernel(
            kernel_length=21,
            learnable=False,
            parameterization="simplex",
        )
        gamma = DelayKernel(
            kernel_length=21,
            gamma_shape=self._MATCHED_SHAPE,
            gamma_scale=self._MATCHED_SCALE,
            learnable=False,
            parameterization="gamma",
        )

        with pytest.raises(RuntimeError, match="Missing key"):
            gamma.load_state_dict(simplex.state_dict())

    def test_gamma_causality_no_future_leakage(self):
        kernel = DelayKernel(kernel_length=5, learnable=True, parameterization="gamma")

        base = _rand_tensor(1, 12)
        perturbed = base.clone()
        perturbed[0, -1] += 10.0

        out_base = kernel(base)
        out_perturbed = kernel(perturbed)

        assert out_base.shape == base.shape
        assert torch.allclose(out_base[0, :-1], out_perturbed[0, :-1])

    def test_gamma_gradient_flows_to_both_parameters(self):
        kernel = DelayKernel(kernel_length=7, learnable=True, parameterization="gamma")

        I_trajectory = _rand_tensor(2, 10).requires_grad_(True)
        output = kernel(I_trajectory)
        loss = output.sum()
        loss.backward()

        assert kernel.gamma_shape_raw.grad is not None
        assert kernel.gamma_scale_raw.grad is not None
        assert torch.isfinite(kernel.gamma_shape_raw.grad).all()
        assert torch.isfinite(kernel.gamma_scale_raw.grad).all()
        assert I_trajectory.grad is not None

    def test_gamma_smooth_bounds_remain_open_at_extreme_coordinates(self):
        kernel = DelayKernel(kernel_length=21, learnable=True, parameterization="gamma")
        with torch.no_grad():
            kernel.gamma_shape_raw.fill_(12.0)
            kernel.gamma_scale_raw.fill_(-12.0)

        shape, scale = kernel.get_gamma_parameters()
        assert _GAMMA_SHAPE_MIN < shape < _GAMMA_SHAPE_MAX
        assert _GAMMA_SCALE_MIN < scale < _GAMMA_SCALE_MAX
        weights = kernel.get_kernel_weights()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_gamma_sanitizes_non_finite_parameters(self):
        kernel = DelayKernel(kernel_length=7, learnable=True, parameterization="gamma")
        with torch.no_grad():
            kernel.gamma_shape_raw.fill_(float("nan"))
            kernel.gamma_scale_raw.fill_(float("inf"))

        I_trajectory = _rand_tensor(2, 10).requires_grad_(True)
        output = kernel(I_trajectory)
        assert torch.isfinite(output).all()

        loss = output.sum()
        loss.backward()
        assert torch.isfinite(kernel.gamma_shape_raw.grad).all()
        assert torch.isfinite(kernel.gamma_scale_raw.grad).all()

    @pytest.mark.parametrize("raw_value", [-12.0, 12.0])
    def test_gamma_gradients_remain_nonzero_near_both_limits(self, raw_value):
        kernel = DelayKernel(kernel_length=21, learnable=True, parameterization="gamma")
        with torch.no_grad():
            kernel.gamma_shape_raw.fill_(raw_value)
            kernel.gamma_scale_raw.fill_(raw_value)

        positions = torch.arange(21, dtype=torch.float32)
        loss = (kernel.get_kernel_weights() * positions.square()).sum()
        loss.backward()

        for parameter in (kernel.gamma_shape_raw, kernel.gamma_scale_raw):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().item() > 0.0

    def test_gamma_get_parameters_rejects_other_parameterizations(self):
        kernel = DelayKernel(kernel_length=5, learnable=True)

        with pytest.raises(ValueError, match="only defined for gamma kernels"):
            kernel.get_gamma_parameters()

    def test_gamma_boundary_hits_report_open_interval_proximity(self):
        cases = [
            ("gamma_shape_raw", -12.0, "shape_lower"),
            ("gamma_shape_raw", 12.0, "shape_upper"),
            ("gamma_scale_raw", -12.0, "scale_lower"),
            ("gamma_scale_raw", 12.0, "scale_upper"),
        ]
        for raw_name, raw_value, expected_key in cases:
            kernel = DelayKernel(
                kernel_length=7, learnable=True, parameterization="gamma"
            )
            with torch.no_grad():
                getattr(kernel, raw_name).fill_(raw_value)
            hits = kernel.get_gamma_boundary_hits(atol=1.0e-5)
            assert hits[expected_key] is True, (raw_name, expected_key)
            assert sum(hits.values()) == 1

    def test_gamma_boundary_hits_rejects_other_parameterizations(self):
        kernel = DelayKernel(
            kernel_length=7, learnable=True, parameterization="simplex"
        )
        with pytest.raises(ValueError, match="only defined for gamma kernels"):
            kernel.get_gamma_boundary_hits()


class TestSheddingConvolutionBasics:
    """Basic functionality tests for SheddingConvolution."""

    def test_initialization_default(self):
        """Test default initialization."""
        conv = SheddingConvolution()
        assert conv.kernel_length == 14
        assert conv.get_sensitivity_scale().item() == pytest.approx(1.0, abs=1e-6)
        assert conv.learnable_kernel is True
        assert conv.learnable_scale is True

    def test_initialization_frozen(self):
        """Test frozen (validation mode) initialization."""
        conv = SheddingConvolution(
            learnable_kernel=False,
            learnable_scale=False,
        )
        assert conv.learnable_kernel is False
        assert conv.learnable_scale is False
        assert not isinstance(conv.kernel, torch.nn.Parameter)
        assert not isinstance(conv.sensitivity_scale, torch.nn.Parameter)

    def test_forward_output_shape(self):
        """Test output shape matches input."""
        batch_size = 3
        time_steps = 20

        conv = SheddingConvolution(kernel_length=7)
        I_trajectory = _rand_tensor(batch_size, time_steps) * 100
        population = _rand_tensor(batch_size, time_steps) * 10000 + 1000

        output = conv(I_trajectory, population)

        assert output.shape == (batch_size, time_steps)

    def test_kernel_normalization(self):
        """Test shedding kernel is normalized."""
        conv = SheddingConvolution(kernel_length=14)

        weights = conv.get_kernel_weights()
        assert torch.allclose(
            weights.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5
        )

    def test_impulse_response_matches_forward_kernel_order(self):
        """A shedding impulse should follow the biologically ordered kernel."""
        conv = SheddingConvolution(
            kernel_length=7,
            learnable_kernel=False,
            learnable_scale=False,
        )

        I_trajectory = _zeros_tensor(1, 20)
        I_trajectory[0, 4] = 1.0
        population = _ones_tensor(1, 20)

        output = conv(I_trajectory, population)
        weights = conv.get_kernel_weights()

        assert torch.allclose(
            output[0, 4 : 4 + conv.kernel_length],
            weights,
            atol=1e-6,
        )


class TestSheddingConvolutionDilution:
    """Tests for dilution physics (core insight: pop division = dilution)."""

    def test_dilution_same_infections_different_populations(self):
        """Test that same infections produce lower concentration in larger population."""
        conv = SheddingConvolution(
            kernel_length=7,
            sensitivity_scale=1.0,
            learnable_scale=False,
            learnable_kernel=False,
        )

        batch_size = 1
        time_steps = 10

        # Same infections in both cases (fraction space to avoid float16 overflow)
        I_trajectory = _ones_tensor(batch_size, time_steps) * 0.1

        # Village: 5k population (fraction)
        pop_village = _ones_tensor(batch_size, time_steps) * 0.005

        # Metropolis: 500k population (100x larger, fraction)
        pop_metropolis = _ones_tensor(batch_size, time_steps) * 0.5

        conc_village = conv(I_trajectory, pop_village)
        conc_metropolis = conv(I_trajectory, pop_metropolis)

        # Village concentration should be ~100x higher (dilution physics)
        # Allow for edge effects at the beginning
        mid_point = time_steps // 2
        ratio = conc_village[0, mid_point] / conc_metropolis[0, mid_point]

        assert 90 < ratio < 110  # Should be approximately 100x

    def test_dilution_village_vs_metropolis_detectability(self):
        """
        Test key insight: 30 cases detectable in village (5k) but invisible in metropolis (500k).

        This demonstrates the dilution physics:
        - Village (5k): 30/5000 = 0.6% infected → HIGH concentration
        - Metropolis (500k): 30/500000 = 0.006% infected → LOW concentration
        """
        conv = SheddingConvolution(
            kernel_length=7,
            sensitivity_scale=1.0,
            learnable_scale=False,
            learnable_kernel=False,
        )

        batch_size = 1
        time_steps = 15

        # Fraction space to avoid float16 overflow (30/5000 = 0.006)
        I_trajectory = _zeros_tensor(batch_size, time_steps)
        I_trajectory[0, 3:10] = 0.006  # 0.6% infection rate

        # Village: fraction
        pop_village = _ones_tensor(batch_size, time_steps) * 0.005

        # Metropolis: 100x larger
        pop_metropolis = _ones_tensor(batch_size, time_steps) * 0.5

        conc_village = conv(I_trajectory, pop_village)
        conc_metropolis = conv(I_trajectory, pop_metropolis)

        # At peak (around t=7), village should have ~100x higher concentration
        peak_time = 8
        village_peak = conc_village[0, peak_time]
        metropolis_peak = conc_metropolis[0, peak_time]

        ratio = village_peak / metropolis_peak
        assert 50 < ratio < 150  # Should be approximately 100x due to population ratio

        # Village signal should be clearly detectable (non-negligible)
        assert village_peak > 1e-6

        # Metropolis signal should be ~100x smaller (potentially near detection limit)
        assert metropolis_peak < village_peak / 50

    def test_time_varying_population(self):
        """Test with time-varying population (e.g., mobility-adjusted)."""
        conv = SheddingConvolution(
            kernel_length=5,
            sensitivity_scale=1.0,
            learnable_scale=False,
            learnable_kernel=False,
        )

        batch_size = 1
        time_steps = 40

        # Create a sustained period of infections (longer than kernel)
        # This ensures convolution has full history at comparison points
        I_trajectory = _zeros_tensor(batch_size, time_steps)
        I_trajectory[0, 10:25] = 100.0  # Sustained infections from t=10 to t=24

        # Population increases over time (from 5k to 10k)
        population = torch.linspace(
            5000, 10000, time_steps, dtype=torch.float32
        ).unsqueeze(0)

        output = conv(I_trajectory, population)

        # Compare outputs in the middle of the sustained period
        # where convolution has full history and population is different
        # t_early=15: pop ~6711, infections ongoing for 5 steps
        # t_late=20: pop ~7692, infections ongoing for 10 steps
        t_early, t_late = 15, 20
        pop_early = population[0, t_early].item()
        pop_late = population[0, t_late].item()

        # Both should have signal
        assert output[0, t_early] > 1e-6
        assert output[0, t_late] > 1e-6

        # The concentration should be roughly inversely proportional to population
        # output_early / output_late ≈ pop_late / pop_early
        ratio_output = output[0, t_early].item() / output[0, t_late].item()
        ratio_pop = pop_late / pop_early

        # The ratio should be in the ballpark (allowing for some variation due to
        # different amounts of shedding history at each point)
        assert 0.6 < ratio_output / ratio_pop < 2.0


class TestSheddingConvolutionGamma:
    """Two-parameter learnable Gamma shedding kernel."""

    _LIT_SHAPE = 2.0
    _LIT_SCALE = 6.0

    def test_gamma_learns_exactly_two_kernel_parameters(self):
        conv = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
            learnable_scale=False,
        )

        named = {
            name: parameter
            for name, parameter in conv.named_parameters()
            if name.startswith("gamma_")
        }
        assert set(named) == {"gamma_shape_raw", "gamma_scale_raw"}
        assert all(parameter.numel() == 1 for parameter in named.values())

    def test_gamma_kernel_normalized_and_prior_matched(self):
        conv = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
            learnable_scale=False,
        )

        weights = conv.get_kernel_weights()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert (weights >= 0).all()

        shape, scale = conv.get_gamma_parameters()
        assert shape == pytest.approx(self._LIT_SHAPE)
        assert scale == pytest.approx(self._LIT_SCALE)

        # Literature anchor: mean 12 d, CV 0.7 → std 8.4 d.
        assert conv.get_mean_delay() == pytest.approx(12.0, abs=1.5)
        assert conv.get_delay_std() == pytest.approx(8.4, abs=2.0)

    def test_gamma_frozen_buffer_matches_learnable_initialization(self):
        frozen = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=False,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
        )
        learnable = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
        )

        assert not hasattr(frozen, "gamma_shape_raw")
        assert not hasattr(frozen, "kernel")
        assert "gamma_kernel" in frozen.state_dict()
        assert torch.allclose(
            frozen.get_kernel_weights(), learnable.get_kernel_weights()
        )
        assert frozen.get_gamma_parameters() == pytest.approx(
            learnable.get_gamma_parameters(), rel=1e-5
        )

    def test_frozen_gamma_rejects_simplex_checkpoint_state(self):
        simplex = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=False,
            kernel_parameterization="simplex",
        )
        gamma = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=False,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
        )

        with pytest.raises(RuntimeError, match="Missing key"):
            gamma.load_state_dict(simplex.state_dict())

    def test_gamma_forward_preserves_dilution_scaling(self):
        conv = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
            learnable_scale=False,
        )

        I_trajectory = _rand_tensor(2, 40) * 0.1
        small_pop = _ones_tensor(2, 40) * 1000.0
        large_pop = _ones_tensor(2, 40) * 100000.0

        output_small = conv(I_trajectory, small_pop)
        output_large = conv(I_trajectory, large_pop)

        assert torch.isfinite(output_small).all()
        assert torch.allclose(
            output_small / output_large,
            torch.full_like(output_small, 100.0),
            rtol=1e-3,
        )

    def test_gamma_gradient_flows_to_both_parameters(self):
        conv = SheddingConvolution(
            kernel_length=21,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            gamma_shape=self._LIT_SHAPE,
            gamma_scale=self._LIT_SCALE,
            learnable_scale=False,
        )

        I_trajectory = _rand_tensor(2, 30).requires_grad_(True)
        population = _ones_tensor(2, 30) * 10000.0
        output = conv(I_trajectory, population)
        output.sum().backward()

        assert conv.gamma_shape_raw.grad is not None
        assert conv.gamma_scale_raw.grad is not None
        assert torch.isfinite(conv.gamma_shape_raw.grad).all()
        assert torch.isfinite(conv.gamma_scale_raw.grad).all()
        assert I_trajectory.grad is not None
        assert torch.isfinite(I_trajectory.grad).all()

    def test_gamma_smooth_bounds_remain_open_at_extreme_coordinates(self):
        conv = SheddingConvolution(
            kernel_length=35,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            learnable_scale=False,
        )
        with torch.no_grad():
            conv.gamma_shape_raw.fill_(20.0)
            conv.gamma_scale_raw.fill_(20.0)

        shape, scale = conv.get_gamma_parameters()
        assert _GAMMA_SHAPE_MIN < shape < _GAMMA_SHAPE_MAX
        assert _GAMMA_SCALE_WW_MIN < scale < _GAMMA_SCALE_WW_MAX
        weights = conv.get_kernel_weights()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.isfinite(conv(_rand_tensor(1, 40), _ones_tensor(1, 40))).all()
        output = conv(_rand_tensor(1, 40), _ones_tensor(1, 40))
        output.sum().backward()
        assert conv.gamma_shape_raw.grad is not None
        assert conv.gamma_shape_raw.grad.abs().item() > 0.0
        assert conv.gamma_scale_raw.grad is not None
        assert conv.gamma_scale_raw.grad.abs().item() > 0.0

    def test_gamma_causality_no_future_leakage(self):
        conv = SheddingConvolution(
            kernel_length=5,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            learnable_scale=False,
        )

        base = _rand_tensor(1, 12)
        perturbed = base.clone()
        perturbed[0, -1] += 10.0
        population = _ones_tensor(1, 12) * 1000.0

        assert torch.allclose(
            conv(base, population)[0, :-1], conv(perturbed, population)[0, :-1]
        )

    def test_gamma_sanitizes_non_finite_parameters(self):
        conv = SheddingConvolution(
            kernel_length=7,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            learnable_scale=False,
        )
        with torch.no_grad():
            conv.gamma_shape_raw.fill_(float("nan"))
            conv.gamma_scale_raw.fill_(float("inf"))

        weights = conv.get_kernel_weights()
        assert torch.isfinite(weights).all()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_gamma_get_parameters_rejects_other_parameterizations(self):
        conv = SheddingConvolution(kernel_length=5, learnable_kernel=True)

        with pytest.raises(ValueError, match="only defined for gamma kernels"):
            conv.get_gamma_parameters()

    def test_gamma_boundary_hits_use_wastewater_scale_box(self):
        conv = SheddingConvolution(
            kernel_length=21,
            learnable_kernel=True,
            kernel_parameterization="gamma",
            learnable_scale=False,
        )

        # A clinical-scale prior is far inside the wider wastewater scale box.
        with torch.no_grad():
            conv.gamma_scale_raw.fill_(
                _inverse_bounded_sigmoid(1.0, lower=0.2, upper=15.0)
            )
        assert not any(conv.get_gamma_boundary_hits(atol=1.0e-5).values())

        cases = [
            ("gamma_shape_raw", -12.0, "shape_lower"),
            ("gamma_shape_raw", 12.0, "shape_upper"),
            ("gamma_scale_raw", -12.0, "scale_lower"),
            ("gamma_scale_raw", 12.0, "scale_upper"),
        ]
        for raw_name, raw_value, expected_key in cases:
            kernel = SheddingConvolution(
                kernel_length=21,
                learnable_kernel=True,
                kernel_parameterization="gamma",
                learnable_scale=False,
            )
            with torch.no_grad():
                getattr(kernel, raw_name).fill_(raw_value)
            hits = kernel.get_gamma_boundary_hits(atol=1.0e-5)
            assert hits[expected_key] is True, (raw_name, expected_key)
            assert sum(hits.values()) == 1


class TestSheddingConvolutionGradients:
    """Tests for gradient flow in shedding convolution."""

    def test_learnable_kernel_and_scale_gradients(self):
        """Test gradients flow through both learnable kernel and scale."""
        conv = SheddingConvolution(
            kernel_length=5, learnable_kernel=True, learnable_scale=True
        )

        I_trajectory = _rand_tensor(2, 10) * 100
        population = _rand_tensor(2, 10) * 10000 + 1000

        output = conv(I_trajectory, population)
        loss = output.sum()
        loss.backward()

        # Both should have gradients
        assert conv.kernel.grad is not None
        assert conv.sensitivity_scale.grad is not None
        assert not torch.all(conv.kernel.grad == 0)
        assert not torch.all(conv.sensitivity_scale.grad == 0)

    def test_frozen_no_gradients(self):
        """Test frozen parameters don't have gradients."""
        conv = SheddingConvolution(
            kernel_length=5, learnable_kernel=False, learnable_scale=False
        )

        # Create leaf tensor properly
        I_raw = _rand_tensor(2, 10)
        I_trajectory = (I_raw * 100).requires_grad_(True)
        population = _rand_tensor(2, 10) * 10000 + 1000

        output = conv(I_trajectory, population)
        loss = output.sum()
        loss.backward()

        # No gradients for frozen parameters (they are buffers, not parameters)
        assert not isinstance(conv.kernel, torch.nn.Parameter)
        assert not isinstance(conv.sensitivity_scale, torch.nn.Parameter)
        # Input should still have gradients
        assert I_trajectory.grad is not None


class TestClinicalObservationHead:
    """Tests for ClinicalObservationHead wrapper."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        head = ClinicalObservationHead()
        assert head.delay_kernel.kernel_length == 21
        assert head.get_scale().item() == pytest.approx(1.0, abs=1e-6)
        assert head.learnable_scale is True

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        head = ClinicalObservationHead()

        batch_size = 2
        time_steps = 20

        I_trajectory = (
            _rand_tensor(batch_size, time_steps) * 0.1
        )  # Fractions, not counts

        output = head(I_trajectory)

        assert output.shape == (batch_size, time_steps)

    def test_scale_effect(self):
        """Test that scale parameter scales output correctly."""
        # Frozen kernel, different scales
        head_low = ClinicalObservationHead(
            kernel_length=7,
            learnable_kernel=False,
            scale_init=0.5,
            learnable_scale=False,
        )
        head_high = ClinicalObservationHead(
            kernel_length=7,
            learnable_kernel=False,
            scale_init=2.0,
            learnable_scale=False,
        )

        I_trajectory = _rand_tensor(1, 10) * 0.1  # Fractions

        output_low = head_low(I_trajectory)
        output_high = head_high(I_trajectory)

        # High scale should produce higher log1p values
        assert output_high.sum() > output_low.sum()

    def test_gradients_learnable_scale(self):
        """Test gradients flow through learnable scale."""
        head = ClinicalObservationHead(learnable_scale=True)

        I_trajectory = _rand_tensor(2, 10) * 0.1  # Fractions
        output = head(I_trajectory)
        loss = output.sum()
        loss.backward()

        assert head.scale.grad is not None
        assert not torch.all(head.scale.grad == 0)

    def test_log1p_output_range(self):
        """Test that output is in log1p space (non-negative)."""
        head = ClinicalObservationHead()

        I_trajectory = _rand_tensor(2, 10) * 0.1  # Fractions
        output = head(I_trajectory)

        # log1p(x) >= 0 for x >= 0
        assert torch.all(output >= 0)

    def test_residual_path_starts_neutral(self):
        """Zero-initialized residual projection should not affect startup output."""
        head = ClinicalObservationHead(residual_dim=4, alpha_init=0.2)
        head = head.to(torch.float32)  # Match model dtype

        I_trajectory = _ones_tensor(2, 12) * 0.001  # Fixed small value
        obs_context = torch.ones(2, 12, 4, dtype=torch.float32) * 0.1  # Fixed context

        no_context = head(I_trajectory, obs_context=None)
        with_context = head(I_trajectory, obs_context=obs_context)
        # Relaxed tolerance for small random init in residual_proj
        assert torch.allclose(no_context, with_context, atol=1e-2)

    @pytest.mark.device
    def test_forward_on_accelerator_after_module_transfer(self, accelerator_device):
        """Clinical head params/buffers should follow module transfer to accelerator."""
        head = ClinicalObservationHead(
            learnable_kernel=False,
            learnable_scale=False,
            residual_dim=4,
        ).to(accelerator_device)

        assert head.scale.device.type == accelerator_device.type
        assert head.alpha is not None
        assert head.alpha.device.type == accelerator_device.type
        assert head.delay_kernel.kernel.device.type == accelerator_device.type

        I_trajectory = _rand_tensor(2, 12).to(accelerator_device) * 0.1
        obs_context = _rand_tensor(2, 12, 4).to(accelerator_device)

        output = head(I_trajectory, obs_context)

        assert output.device.type == accelerator_device.type
        assert torch.isfinite(output).all()

    def test_delta_forecasting_uses_explicit_mask(self):
        """Anchoring should follow the provided mask rather than value finiteness."""
        head = ClinicalObservationHead(
            learnable_kernel=False,
            learnable_scale=False,
            delta_forecasting=True,
        )
        I_trajectory = _ones_tensor(1, 6) * 0.001

        anchored = head(
            I_trajectory,
            last_observed=torch.tensor([2.0]),
            last_observed_mask=torch.tensor([1.0]),
        )
        unanchored = head(
            I_trajectory,
            last_observed=torch.tensor([2.0]),
            last_observed_mask=torch.tensor([0.0]),
        )

        assert anchored[0, 0].item() == pytest.approx(2.0, abs=1e-6)
        assert anchored[0, 0].item() != pytest.approx(unanchored[0, 0].item(), abs=1e-6)

    def test_delta_forecasting_can_be_disabled_even_with_anchor_inputs(self):
        head = ClinicalObservationHead(
            learnable_kernel=False,
            learnable_scale=False,
            delta_forecasting=True,
            anchor_mode="disabled",
        )
        I_trajectory = _ones_tensor(1, 6) * 0.001
        output = head(
            I_trajectory,
            last_observed=torch.tensor([5.0]),
            last_observed_mask=torch.tensor([1.0]),
        )
        baseline = head(I_trajectory)
        assert torch.allclose(output, baseline)

    def test_distribution_exposes_exact_anchored_path_decomposition(self):
        head = ClinicalObservationHead(
            kernel_length=3,
            learnable_kernel=False,
            residual_dim=1,
            alpha_init=1.0,
            delta_forecasting=True,
        )
        assert head.residual_proj is not None
        with torch.no_grad():
            head.residual_proj.weight.fill_(1.0)
            head.residual_proj.bias.zero_()

        trajectory = torch.tensor([[0.001, 0.002, 0.004, 0.003]])
        context = torch.tensor([[[0.0], [0.5], [-0.25], [1.0]]])
        anchor = torch.tensor([2.0])
        anchor_mask = torch.tensor([1.0])
        distribution = head.forward_distribution(
            trajectory,
            obs_context=context,
            last_observed=anchor,
            last_observed_mask=anchor_mask,
        )
        residual_disabled = head(
            trajectory,
            obs_context=None,
            last_observed=anchor,
            last_observed_mask=anchor_mask,
        )

        assert torch.allclose(
            distribution["loc"],
            distribution["mechanistic_loc"] + distribution["residual_loc"],
        )
        assert distribution["residual_input"].shape == distribution["loc"].shape
        assert torch.allclose(distribution["mechanistic_loc"], residual_disabled)
        assert torch.allclose(
            head(
                trajectory,
                obs_context=context,
                last_observed=anchor,
                last_observed_mask=anchor_mask,
            ),
            distribution["loc"],
        )


class TestWastewaterObservationHead:
    """Tests for WastewaterObservationHead wrapper."""

    def test_initialization(self):
        """Test initialization."""
        head = WastewaterObservationHead()
        assert head.shedding_conv.kernel_length == 14
        assert head.get_scale().item() == pytest.approx(1.0, abs=1e-6)

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        head = WastewaterObservationHead()

        batch_size = 2
        time_steps = 20

        I_trajectory = _rand_tensor(batch_size, time_steps) * 0.1  # Fractions
        population = _rand_tensor(batch_size, time_steps) * 10000 + 1000

        output = head(I_trajectory, population)

        assert output.shape == (batch_size, time_steps)

    def test_mlp_kernel_accepts_history_length_population(self):
        """Static MLP population features must not require the forecast length."""
        head = WastewaterObservationHead(kernel_mlp=True)
        infections = _rand_tensor(3, 29) * 0.1
        population_history = _ones_tensor(3, 32, 1) * 10000.0

        output = head(infections, population_history)

        assert output.shape == infections.shape
        assert torch.isfinite(output).all()

    def test_scale_effect(self):
        """Test that scale parameter affects output magnitude."""
        # Fixed kernel, different scales
        head_low = WastewaterObservationHead(
            kernel_length=7,
            scale_init=0.5,
            learnable_kernel=False,
            learnable_scale=False,
        )
        head_high = WastewaterObservationHead(
            kernel_length=7,
            scale_init=2.0,
            learnable_kernel=False,
            learnable_scale=False,
        )

        I_trajectory = _rand_tensor(1, 10) * 0.1  # Fractions
        population = _ones_tensor(1, 10) * 10000.0

        output_low = head_low(I_trajectory, population)
        output_high = head_high(I_trajectory, population)

        # High scale should produce higher log1p values
        assert output_high.sum() > output_low.sum()

    def test_log1p_output_range(self):
        """Test that output is in log1p space (non-negative)."""
        head = WastewaterObservationHead()

        I_trajectory = _rand_tensor(2, 10) * 0.1  # Fractions
        population = _ones_tensor(2, 10) * 10000.0
        output = head(I_trajectory, population)

        # log1p(x) >= 0 for x >= 0
        assert torch.all(output >= 0)

    def test_residual_path_starts_neutral(self):
        """Zero-initialized residual projection should be neutral at startup."""
        head = WastewaterObservationHead(residual_dim=6, alpha_init=0.2)
        head = head.to(torch.float32)  # Match model dtype

        I_trajectory = _ones_tensor(2, 12) * 0.001  # Fixed small value
        population = _ones_tensor(2, 12)  # Normalized population
        obs_context = torch.ones(2, 12, 6, dtype=torch.float32) * 0.1  # Fixed context

        no_context = head(I_trajectory, population, obs_context=None)
        with_context = head(I_trajectory, population, obs_context=obs_context)
        # Relaxed tolerance for small random init in residual_proj
        assert torch.allclose(no_context, with_context, atol=1e-2)

    @pytest.mark.device
    def test_forward_on_accelerator_after_module_transfer(self, accelerator_device):
        """Wastewater head params/buffers should follow module transfer to accelerator."""
        head = WastewaterObservationHead(
            learnable_kernel=False,
            learnable_scale=False,
            residual_dim=6,
        ).to(accelerator_device)

        assert head.scale.device.type == accelerator_device.type
        assert head.alpha is not None
        assert head.alpha.device.type == accelerator_device.type
        assert head.shedding_conv.kernel.device.type == accelerator_device.type
        assert (
            head.shedding_conv.sensitivity_scale.device.type == accelerator_device.type
        )

        I_trajectory = _rand_tensor(2, 12).to(accelerator_device) * 0.1
        population = (_ones_tensor(2, 12) * 1000.0).to(accelerator_device)
        obs_context = _rand_tensor(2, 12, 6).to(accelerator_device)

        output = head(I_trajectory, population, obs_context)

        assert output.device.type == accelerator_device.type
        assert torch.isfinite(output).all()

    def test_delta_forecasting_skips_when_anchor_mask_missing(self):
        head = WastewaterObservationHead(
            learnable_kernel=False,
            learnable_scale=False,
            delta_forecasting=True,
        )
        I_trajectory = _ones_tensor(1, 6) * 0.001
        population = _ones_tensor(1, 6)
        output = head(
            I_trajectory,
            population,
            last_observed=torch.tensor([4.0]),
            last_observed_mask=torch.tensor([0.0]),
        )
        baseline = head(I_trajectory, population)
        assert torch.allclose(output, baseline)

    def test_distribution_exposes_exact_anchored_path_decomposition(self):
        head = WastewaterObservationHead(
            kernel_length=3,
            learnable_kernel=False,
            residual_dim=1,
            alpha_init=1.0,
            delta_forecasting=True,
        )
        assert head.residual_proj is not None
        with torch.no_grad():
            head.residual_proj.weight.fill_(1.0)
            head.residual_proj.bias.zero_()

        trajectory = torch.tensor([[0.001, 0.002, 0.004, 0.003]])
        population = torch.ones_like(trajectory)
        context = torch.tensor([[[0.0], [0.5], [-0.25], [1.0]]])
        anchor = torch.tensor([2.0])
        anchor_mask = torch.tensor([1.0])
        distribution = head.forward_distribution(
            trajectory,
            population,
            obs_context=context,
            last_observed=anchor,
            last_observed_mask=anchor_mask,
        )
        residual_disabled = head(
            trajectory,
            population,
            obs_context=None,
            last_observed=anchor,
            last_observed_mask=anchor_mask,
        )

        assert torch.allclose(
            distribution["loc"],
            distribution["mechanistic_loc"] + distribution["residual_loc"],
        )
        assert distribution["residual_input"].shape == distribution["loc"].shape
        assert torch.allclose(distribution["mechanistic_loc"], residual_disabled)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_epidemic_wave_observation(self):
        """Test observing a full epidemic wave."""
        batch_size = 1
        time_steps = 60

        # Simulate SIR-like epidemic wave (fractions, not counts)
        t = torch.arange(time_steps, dtype=torch.float32)
        I_trajectory = 0.01 * torch.exp(
            -((t - 30) ** 2) / 200
        )  # Gaussian wave centered at t=30 (1% peak infection rate)
        I_trajectory = I_trajectory.unsqueeze(0)  # Add batch dimension

        population = _ones_tensor(batch_size, time_steps)  # Normalized population (1.0)

        # Clinical observation (hospitalizations)
        hosp_head, ww_head = make_observation_heads(
            frozen=True, kernel_length=14, scale_init=1.0
        )
        hosp_obs = hosp_head(I_trajectory)

        # Wastewater observation
        ww_obs = ww_head(I_trajectory, population)

        # Both should show the epidemic wave
        # Find peaks
        hosp_peak = torch.argmax(hosp_obs[0])
        ww_peak = torch.argmax(ww_obs[0])
        infection_peak = torch.argmax(I_trajectory[0])

        # Hospitalizations should peak later than infections (delay)
        assert hosp_peak > infection_peak

        # Wastewater should track infections more closely (shedding is immediate-ish)
        # but still has some delay from the convolution
        assert ww_peak >= infection_peak

    def test_gradient_flow_end_to_end(self):
        """Test full gradient flow through observation heads."""
        batch_size = 2
        time_steps = 20

        # Learnable parameters everywhere
        hosp_head, ww_head = make_observation_heads(kernel_length=7)

        # Infections and population as "inputs" from upstream model (fractions)
        I_trajectory = _rand_tensor(batch_size, time_steps) * 0.1
        population = _rand_tensor(batch_size, time_steps) * 10000 + 1000

        # Forward pass
        hosp_pred = hosp_head(I_trajectory)
        ww_pred = ww_head(I_trajectory, population)

        # Simulated targets (log1p per-100k space)
        hosp_target = _rand_tensor(batch_size, time_steps) * 5  # log1p values
        ww_target = _rand_tensor(batch_size, time_steps) * 5

        # Loss
        hosp_loss = ((hosp_pred - hosp_target) ** 2).mean()
        ww_loss = ((ww_pred - ww_target) ** 2).mean()
        total_loss = hosp_loss + ww_loss

        # Backward
        total_loss.backward()

        # Check all parameters have gradients
        assert hosp_head.delay_kernel.kernel.grad is not None
        assert hosp_head.scale.grad is not None
        assert ww_head.shedding_conv.kernel.grad is not None
        assert ww_head.scale.grad is not None


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_infections(self):
        """Test behavior with zero infections."""
        hosp_head, ww_head = make_observation_heads(frozen=True)

        batch_size = 1
        time_steps = 10

        I_zero = _zeros_tensor(batch_size, time_steps)
        population = _ones_tensor(batch_size, time_steps) * 10000.0

        hosp_output = hosp_head(I_zero)
        ww_output = ww_head(I_zero, population)

        # Both should be zero (log1p(0) = 0)
        assert torch.allclose(hosp_output, _zeros_like(hosp_output))
        assert torch.allclose(ww_output, _zeros_like(ww_output))

    def test_very_small_population(self):
        """Test with moderately small population (avoid float16 overflow)."""
        ww_head = WastewaterObservationHead(
            learnable_kernel=False, learnable_scale=False
        )

        batch_size = 1
        time_steps = 5

        I_trajectory = (
            _ones_tensor(batch_size, time_steps) * 0.001
        )  # Very small fraction
        small_pop = _ones_tensor(batch_size, time_steps) * 0.1  # Moderately small

        # Should not crash
        output = ww_head(I_trajectory, small_pop)

        # Should be finite
        assert torch.all(torch.isfinite(output))
        # Should have positive output
        assert torch.all(output > 0)

    def test_population_broadcast(self):
        """Test population broadcasting from [batch] to [batch, time]."""
        ww_head = WastewaterObservationHead(
            learnable_kernel=False, learnable_scale=False
        )

        batch_size = 2
        time_steps = 6

        I_trajectory = _rand_tensor(batch_size, time_steps) * 100
        population = torch.tensor([1000.0, 2000.0], dtype=torch.float32)

        output = ww_head(I_trajectory, population)
        assert output.shape == (batch_size, time_steps)

    def test_population_non_positive_raises(self):
        """Test non-positive population raises error."""
        ww_head = WastewaterObservationHead(
            learnable_kernel=False, learnable_scale=False
        )

        I_trajectory = _rand_tensor(1, 5) * 10
        population = _zeros_tensor(1, 5)

        with pytest.raises(ValueError, match="population must be positive"):
            ww_head(I_trajectory, population)

    def test_short_trajectory(self):
        """Test with trajectory shorter than kernel length."""
        kernel_length = 21
        hosp_head = ClinicalObservationHead(
            kernel_length=kernel_length, learnable_kernel=False, learnable_scale=False
        )

        batch_size = 1
        time_steps = 5  # Shorter than kernel

        I_trajectory = _rand_tensor(batch_size, time_steps) * 0.1  # Fractions

        # Should not crash
        output = hosp_head(I_trajectory)

        # Should produce correct shape
        assert output.shape == (batch_size, time_steps)


class TestNormalizedLosses:
    """Tests for MSELoss and MAELoss in normalized space."""

    @pytest.mark.parametrize(
        ("loss_type", "target", "expected"),
        [
            (MSELoss, [1.0, 2.0, 4.0], 1.0 / 3.0),
            (MAELoss, [1.0, 2.0, 5.0], 2.0 / 3.0),
        ],
        ids=["mse", "mae"],
    )
    def test_basic_loss(self, loss_type, target, expected):
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_type()(pred, torch.tensor([target]))
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        ("loss_type", "target"),
        [(MSELoss, 3.0), (MAELoss, 4.0)],
        ids=["mse", "mae"],
    )
    def test_loss_ignores_mean_scale(self, loss_type, target):
        loss_fn = loss_type()
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0, target]])
        stats = {
            "target_mean": torch.tensor([[10.0]]),
            "target_scale": torch.tensor([[2.0]]),
        }

        assert loss_fn(pred, target, **stats).item() == pytest.approx(
            loss_fn(pred, target).item(), abs=1e-6
        )

    @pytest.mark.parametrize(
        ("loss_type", "target", "mask"),
        [
            (MSELoss, [1.0, 5.0, 3.0], [True, False, True]),
            (MAELoss, [10.0, 2.0, 30.0], [False, True, False]),
        ],
        ids=["mse", "mae"],
    )
    def test_loss_with_mask(self, loss_type, target, mask):
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_type()(
            pred,
            torch.tensor([target]),
            mask=torch.tensor([mask]),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_gradient_flow(self):
        """Test gradients flow through MSE loss."""
        loss_fn = MSELoss()
        pred = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0, 4.0]])

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)


class TestUnscaledMSELoss:
    """Tests for UnscaledMSELoss."""

    def test_unscaled_mse_with_stats(self):
        """Test unscaled MSE with target_mean/target_scale."""
        loss_fn = UnscaledMSELoss()
        # Normalized values: pred=1.0, target=2.0
        # With mean=100, scale=10:
        # unscaled_pred = 100 + 1.0 * 10 = 110
        # unscaled_target = 100 + 2.0 * 10 = 120
        pred_norm = torch.tensor([[1.0]])
        target_norm = torch.tensor([[2.0]])
        target_mean = torch.tensor([[100.0]])
        target_scale = torch.tensor([[10.0]])

        loss = loss_fn(
            pred_norm, target_norm, target_mean=target_mean, target_scale=target_scale
        )

        # Expected: (110 - 120)^2 = 100
        assert loss.item() == pytest.approx(100.0, abs=1e-5)

    def test_unscaled_mse_without_stats(self):
        """Test unscaled MSE without stats (falls back to normalized)."""
        loss_fn = UnscaledMSELoss()
        pred = torch.tensor([[1.0, 3.0]])
        target = torch.tensor([[1.0, 5.0]])

        loss = loss_fn(pred, target)

        # Expected: (0^2 + 2^2) / 2 = 2
        assert loss.item() == pytest.approx(2.0, abs=1e-6)

    def test_unscaled_mse_vs_normalized_mse(self):
        """Compare unscaled MSE with normalized MSE to show the difference."""
        pred_norm = torch.tensor([[1.0]])
        target_norm = torch.tensor([[2.0]])
        target_mean = torch.tensor([[100.0]])
        target_scale = torch.tensor([[10.0]])

        # Normalized MSE
        mse_loss = MSELoss()(pred_norm, target_norm)
        # Expected: (1 - 2)^2 = 1
        assert mse_loss.item() == pytest.approx(1.0, abs=1e-6)

        # Unscaled MSE
        unscaled_loss = UnscaledMSELoss()(
            pred_norm, target_norm, target_mean=target_mean, target_scale=target_scale
        )
        # Expected: (110 - 120)^2 = 100
        assert unscaled_loss.item() == pytest.approx(100.0, abs=1e-5)

    def test_unscaled_mse_with_mask(self):
        """Test unscaled MSE with mask."""
        loss_fn = UnscaledMSELoss()
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[2.0, 2.0, 4.0]])
        mask = torch.tensor([[False, True, False]])
        target_mean = torch.tensor([[10.0]])
        target_scale = torch.tensor([[1.0]])

        loss = loss_fn(
            pred, target, target_mean=target_mean, target_scale=target_scale, mask=mask
        )

        # Only position 1 is unmasked, and it has 0 error
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestCompositeObservationLoss:
    """Tests for CompositeObservationLoss."""

    def test_basic_composite(self):
        """Test basic composite loss computation."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
            "head_b": (MSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "head_a": torch.tensor([[1.0, 2.0]]),
            "head_b": torch.tensor([[1.0, 3.0]]),
        }
        targets = {
            "head_a": torch.tensor([[1.0, 3.0]]),  # MSE = 1.0/2 = 0.5
            "head_b": torch.tensor([[1.0, 5.0]]),  # MSE = 4.0/2 = 2.0
        }

        total_loss, per_head = loss_fn(preds, targets)

        # Expected: 1.0 * 0.5 + 0.5 * 2.0 = 0.5 + 1.0 = 1.5
        assert total_loss.item() == pytest.approx(1.5, abs=1e-6)
        assert "head_a" in per_head
        assert "head_b" in per_head
        assert per_head["head_a"].item() == pytest.approx(0.5, abs=1e-6)
        assert per_head["head_b"].item() == pytest.approx(2.0, abs=1e-6)

    def test_skip_none_targets(self):
        """Test that heads with None targets are skipped."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
            "head_b": (MSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "head_a": torch.tensor([[1.0, 2.0]]),
            "head_b": torch.tensor([[1.0, 3.0]]),
        }
        targets = {
            "head_a": torch.tensor([[1.0, 3.0]]),  # MSE = 0.5
            "head_b": None,  # Skip this head
        }

        total_loss, per_head = loss_fn(preds, targets)

        # Expected: only head_a contributes = 1.0 * 0.5 = 0.5
        assert total_loss.item() == pytest.approx(0.5, abs=1e-6)
        assert "head_a" in per_head
        assert (
            "head_b" not in per_head
        )  # Should not be in per_head since target was None

    def test_all_none_targets(self):
        """Test behavior when all targets are None."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {"head_a": torch.tensor([[1.0, 2.0]])}
        targets = {"head_a": None}

        total_loss, per_head = loss_fn(preds, targets)

        assert total_loss.item() == pytest.approx(0.0, abs=1e-6)
        assert per_head == {}

    def test_weight_zero_skips_computation(self):
        """Test that weight=0 heads contribute 0 to total loss (but still computed for logging)."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
            "head_b": (MSELoss(), 0.0),  # Zero weight
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "head_a": torch.tensor([[1.0, 3.0]]),  # MSE = 4/2 = 2.0
            "head_b": torch.tensor([[1.0, 5.0]]),  # MSE = 16/2 = 8.0, but weight=0
        }
        targets = {
            "head_a": torch.tensor([[1.0, 1.0]]),
            "head_b": torch.tensor([[1.0, 1.0]]),
        }

        total_loss, per_head = loss_fn(preds, targets)

        # Expected: only head_a contributes = 1.0 * 2.0 = 2.0
        assert total_loss.item() == pytest.approx(2.0, abs=1e-6)
        # Both should be in per_head for logging
        assert "head_a" in per_head
        assert "head_b" in per_head
        assert per_head["head_b"].item() == pytest.approx(8.0, abs=1e-6)

    def test_composite_with_stats(self):
        """Test composite loss with per-head unscaling stats."""
        head_specs = {
            "ww": (UnscaledMSELoss(), 1.0),
            "hosp": (UnscaledMSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "ww": torch.tensor([[0.0, 1.0]]),  # Will be unscaled
            "hosp": torch.tensor([[1.0, 2.0]]),  # Will be unscaled
        }
        targets = {
            "ww": torch.tensor([[0.0, 2.0]]),
            "hosp": torch.tensor([[2.0, 4.0]]),
        }
        stats = {
            "ww": (torch.tensor([[100.0]]), torch.tensor([[10.0]])),
            "hosp": (torch.tensor([[50.0]]), torch.tensor([[5.0]])),
        }

        total_loss, per_head = loss_fn(preds, targets, stats=stats)

        # WW: unscaled pred=[100, 110], target=[100, 120]
        # MSE: (0^2 + 10^2) / 2 = 50
        expected_ww = 50.0

        # Hosp: unscaled pred=[55, 60], target=[60, 70]
        # MSE at t=0: (55-60)^2 = 25, at t=1: (60-70)^2 = 100
        # Mean: (25+100)/2 = 62.5
        expected_hosp = 62.5

        assert per_head["ww"].item() == pytest.approx(expected_ww, abs=1e-4)
        assert per_head["hosp"].item() == pytest.approx(expected_hosp, abs=1e-4)
        assert total_loss.item() == pytest.approx(
            1.0 * expected_ww + 0.5 * expected_hosp, abs=1e-4
        )

    def test_composite_with_masks(self):
        """Test composite loss with per-head masks."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
            "head_b": (MSELoss(), 1.0),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "head_a": torch.tensor([[1.0, 2.0, 3.0]]),
            "head_b": torch.tensor([[1.0, 2.0, 3.0]]),
        }
        targets = {
            "head_a": torch.tensor([[10.0, 2.0, 30.0]]),  # Errors: 81, 0, 729
            "head_b": torch.tensor([[1.0, 20.0, 3.0]]),  # Errors: 0, 324, 0
        }
        masks = {
            "head_a": torch.tensor([[False, True, False]]),  # Only t=1 contributes
            "head_b": torch.tensor([[True, False, True]]),  # t=0 and t=2 contribute
        }

        total_loss, per_head = loss_fn(preds, targets, masks=masks)

        # head_a: only position 1 unmasked, error=0
        assert per_head["head_a"].item() == pytest.approx(0.0, abs=1e-6)

        # head_b: positions 0 and 2 unmasked, both have error=0
        assert per_head["head_b"].item() == pytest.approx(0.0, abs=1e-6)

        assert total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_missing_stats_defaults_to_none(self):
        """Test that missing stats entries default to None (no unscaling)."""
        head_specs = {
            "head_with_stats": (UnscaledMSELoss(), 1.0),
            "head_without_stats": (UnscaledMSELoss(), 1.0),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        preds = {
            "head_with_stats": torch.tensor([[100.0, 200.0]]),
            "head_without_stats": torch.tensor([[1.0, 2.0]]),
        }
        targets = {
            "head_with_stats": torch.tensor([[100.0, 300.0]]),
            "head_without_stats": torch.tensor([[1.0, 3.0]]),
        }
        stats = {
            # Only provide stats for head_with_stats
            "head_with_stats": (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
        }

        total_loss, per_head = loss_fn(preds, targets, stats=stats)

        assert total_loss.item() > 0
        assert "head_with_stats" in per_head
        assert "head_without_stats" in per_head

    def test_gradient_flow_composite(self):
        """Test gradients flow through composite loss."""
        head_specs = {
            "head_a": (MSELoss(), 1.0),
            "head_b": (MSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        pred_a = torch.tensor([[1.0, 2.0]], requires_grad=True)
        pred_b = torch.tensor([[1.0, 3.0]], requires_grad=True)
        preds = {"head_a": pred_a, "head_b": pred_b}
        targets = {
            "head_a": torch.tensor([[1.0, 3.0]]),
            "head_b": torch.tensor([[1.0, 5.0]]),
        }

        total_loss, per_head = loss_fn(preds, targets)
        total_loss.backward()

        assert pred_a.grad is not None
        assert pred_b.grad is not None
        assert not torch.all(pred_a.grad == 0)
        assert not torch.all(pred_b.grad == 0)

    def test_repr(self):
        """Test string representation."""
        head_specs = {
            "ww": (UnscaledMSELoss(), 1.0),
            "hosp": (MSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        repr_str = repr(loss_fn)
        assert "CompositeObservationLoss" in repr_str
        assert "UnscaledMSELoss" in repr_str
        assert "MSELoss" in repr_str

    @pytest.mark.device
    def test_cross_device_loss_computation(self, accelerator_device):
        """Test that CompositeObservationLoss works when preds are on accelerator and targets on CPU.

        This simulates the real training scenario where:
        - Predictions come from model forward pass on GPU/MPS
        - Targets come from DataLoader on CPU

        Regression test for device mismatch bugs in loss computation.
        """
        head_specs = {
            "hosp": (MSELoss(), 1.0),
            "ww": (MSELoss(), 0.5),
        }
        loss_fn = CompositeObservationLoss(head_specs)

        # Predictions on accelerator (simulates GPU forward pass)
        preds = {
            "hosp": torch.ones(2, 3, device=accelerator_device),
            "ww": torch.ones(2, 3, device=accelerator_device),
        }

        # Targets on CPU (simulates DataLoader output)
        targets = {
            "hosp": torch.zeros(2, 3),  # CPU
            "ww": torch.zeros(2, 3),  # CPU
        }

        # This should NOT raise RuntimeError about device mismatch
        total_loss, per_head = loss_fn(preds, targets)

        # Verify loss is computed correctly
        assert torch.isfinite(total_loss)
        # MSE of ones vs zeros = 1.0 for each head
        # Total = 1.0 * 1.0 + 0.5 * 1.0 = 1.5
        assert total_loss.item() == pytest.approx(1.5, abs=1e-5)
