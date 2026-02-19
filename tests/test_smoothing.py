"""Unit tests for causal smoothing utilities."""

import numpy as np
import pandas as pd
import pytest

from data.preprocess.smoothing import (
    DampedHoltSmoother,
    KalmanSmoother,
    TobitKalmanSmoother,
    fit_kalman_params,
)


class TestFitKalmanParams:
    """Tests for fit_kalman_params function."""

    def test_fit_kalman_params_basic(self):
        np.random.seed(42)
        n = 50
        log_values = np.cumsum(np.random.randn(n) * 0.1)
        values = np.exp(log_values) + 1

        series = pd.Series(values, index=pd.date_range("2022-01-01", periods=n))
        process_var, measurement_var = fit_kalman_params(series)

        assert process_var > 0
        assert measurement_var > 0
        assert process_var < 10
        assert measurement_var < 10

    def test_fit_kalman_params_empty_raises(self):
        series = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="No finite observations"):
            fit_kalman_params(series)


class TestKalmanSmoother:
    """Tests for KalmanSmoother class."""

    def test_filter_series_basic(self):
        np.random.seed(42)
        n = 20
        values = np.exp(np.cumsum(np.random.randn(n) * 0.1)) + 1

        kf = KalmanSmoother(process_var=0.05, measurement_var=0.5)
        filtered, flags = kf.filter_series(values)

        assert len(filtered) == n
        assert len(flags) == n
        assert all(f == 0 for f in flags)
        assert all(np.isfinite(f) for f in filtered)

    def test_predict_missing_policy_has_no_momentum_drift(self):
        values = np.array([1.0, 2.0, np.nan, np.nan, np.nan], dtype=float)

        kf = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="predict",
        )
        filtered, flags = kf.filter_series(values)

        assert flags == [0, 0, 2, 2, 2]
        gap = np.array(filtered[2:])
        assert np.all(np.isfinite(gap))
        assert np.allclose(gap, gap[0])

    def test_momentum_missing_policy_drifts_then_caps(self):
        values = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan], dtype=float)

        kf = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="momentum",
            momentum_decay=0.5,
            momentum_max_steps=2,
        )
        filtered, flags = kf.filter_series(values)

        assert flags == [0, 0, 2, 2, 2, 2]
        # First two missing updates should drift; afterwards should settle to predict-only.
        assert filtered[2] != filtered[3]
        assert filtered[4] == pytest.approx(filtered[5])

    def test_reentry_gain_cap_reduces_jump_after_gap(self):
        values = np.array([2.0, 2.1, 2.0, np.nan, np.nan, 6.0], dtype=float)

        kf_loose = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="predict",
            innovation_clip_sigma=100.0,
            reentry_gain_cap=1.0,
            reentry_steps=2,
        )
        loose, _ = kf_loose.filter_series(values)

        kf_capped = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="predict",
            innovation_clip_sigma=100.0,
            reentry_gain_cap=0.1,
            reentry_steps=2,
        )
        capped, _ = kf_capped.filter_series(values)

        loose_jump = abs(loose[5] - loose[4])
        capped_jump = abs(capped[5] - capped[4])
        assert capped_jump < loose_jump

    def test_filter_series_all_nan(self):
        values = np.full(10, np.nan)
        kf = KalmanSmoother(process_var=0.05, measurement_var=0.5)
        filtered, flags = kf.filter_series(values)

        assert len(filtered) == 10
        assert all(f == 2 for f in flags)
        assert all(np.isfinite(f) for f in filtered)


class TestTobitKalmanSmoother:
    """Tests for TobitKalmanSmoother class."""

    def test_filter_series_censored_detection(self):
        np.random.seed(42)
        n = 20
        values = np.exp(np.cumsum(np.random.randn(n) * 0.1)) + 0.5
        limits = np.full(n, 1.5)
        values[5:10] = 0.8

        kf = TobitKalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            censor_inflation=4.0,
        )
        filtered, flags = kf.filter_series(values, limits)

        assert len(filtered) == n
        assert all(np.isfinite(f) for f in filtered)
        assert all(f == 1 for f in flags[5:10])

    def test_predict_missing_policy_stays_flat_for_tobit(self):
        values = np.array([2.0, 2.5, np.nan, np.nan], dtype=float)
        limits = np.full(values.shape[0], 0.5)

        kf = TobitKalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="predict",
        )
        filtered, flags = kf.filter_series(values, limits)

        assert flags == [0, 0, 2, 2]
        assert filtered[2] == pytest.approx(filtered[3])

    def test_momentum_policy_drifts_for_tobit(self):
        values = np.array([1.0, 2.0, np.nan, np.nan, np.nan], dtype=float)
        limits = np.full(values.shape[0], 0.5)

        kf = TobitKalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="momentum",
            momentum_max_steps=3,
        )
        filtered, flags = kf.filter_series(values, limits)

        assert flags == [0, 0, 2, 2, 2]
        assert filtered[2] != filtered[3]

    def test_zero_limits_treated_as_uncensored(self):
        values = np.array([1.0, 2.0, 3.0])
        limits = np.zeros(3)

        kf = TobitKalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            censor_inflation=4.0,
        )
        _, flags = kf.filter_series(values, limits)

        assert all(f == 0 for f in flags)


class TestDampedHoltSmoother:
    """Tests for DampedHoltSmoother class."""

    def test_basic_filtering(self):
        values = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        smoother = DampedHoltSmoother(alpha=0.3, beta=0.05, phi=0.9)
        filtered, flags = smoother.filter_series(values)

        assert len(filtered) == len(values)
        assert all(f == 0 for f in flags)
        assert all(np.isfinite(v) for v in filtered)

    def test_missing_values_do_not_plateau(self):
        values = np.array([1.0, 2.0, np.nan, np.nan, np.nan], dtype=float)
        smoother = DampedHoltSmoother(alpha=0.3, beta=0.1, phi=0.9)
        filtered, flags = smoother.filter_series(values)

        assert flags == [0, 0, 2, 2, 2]
        gap = filtered[2:]
        assert all(np.isfinite(v) for v in gap)
        # Gap predictions should evolve due to damped trend carry, not flat plateau.
        assert len(set(np.round(gap, 8))) > 1

    def test_all_missing(self):
        values = np.array([np.nan, np.nan, np.nan])
        smoother = DampedHoltSmoother()
        filtered, flags = smoother.filter_series(values)

        assert flags == [2, 2, 2]
        assert all(np.isfinite(v) for v in filtered)


class TestSmoothingRegressionQuality:
    """Regression checks for gap handling quality targets."""

    def test_predict_reentry_jump_and_observed_distortion_bounds(self):
        values = np.array(
            [10.0, 11.0, 12.0, np.nan, np.nan, np.nan, 18.0, 19.0, 20.0],
            dtype=float,
        )

        baseline = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="momentum",
            momentum_decay=0.5,
            momentum_max_steps=6,
            innovation_clip_sigma=100.0,
            reentry_gain_cap=1.0,
            reentry_steps=1,
        )
        baseline_filtered, _ = baseline.filter_series(values)

        candidate = KalmanSmoother(
            process_var=0.05,
            measurement_var=0.5,
            missing_policy="predict",
            innovation_clip_sigma=2.5,
            reentry_gain_cap=0.35,
            reentry_steps=2,
        )
        candidate_filtered, flags = candidate.filter_series(values)

        # Jump measured on first observed point after missing span.
        reentry_idx = 6
        baseline_jump = abs(
            baseline_filtered[reentry_idx] - baseline_filtered[reentry_idx - 1]
        )
        candidate_jump = abs(
            candidate_filtered[reentry_idx] - candidate_filtered[reentry_idx - 1]
        )
        assert candidate_jump <= baseline_jump * 0.70

        # Candidate must remain finite and non-negative after inverse transform.
        candidate_values = np.exp(np.array(candidate_filtered))
        assert np.isfinite(candidate_values).all()
        assert (candidate_values >= 0.0).all()

        # On observed points, smoothed log values should stay close to observed log values.
        observed_mask = np.array(flags) == 0
        raw_log = np.log(values[observed_mask] + 1e-9)
        smooth_log = np.array(candidate_filtered)[observed_mask]
        median_distortion = np.median(np.abs(smooth_log - raw_log))
        assert median_distortion <= 0.15
