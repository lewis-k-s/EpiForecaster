"""
Tests for Observation Heads (DelayKernel, SheddingConvolution, ClinicalObservationHead, WastewaterObservationHead).

These isolated checks cover mathematical behavior and edge cases that the
bounded training and evaluation test cannot resolve precisely.
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
    DelayKernel,
    SheddingConvolution,
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
