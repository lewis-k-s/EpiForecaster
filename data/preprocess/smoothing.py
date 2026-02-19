"""
Shared smoothing utilities for preprocessing pipeline.

This module centralizes causal time-series smoothing algorithms to avoid code
duplication across processors. Includes Kalman filtering for both censored
(Tobit) and non-censored data, a damped Holt smoother, and parameter estimation
utilities.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.statespace.structural import UnobservedComponents

if TYPE_CHECKING:
    import pandas as pd


def fit_kalman_params(series: pd.Series) -> tuple[float, float]:
    """
    Fit Kalman filter process/measurement variances from time series data.

    Uses statsmodels UnobservedComponents to fit a local level model
    and extract the process and measurement variances. Works on log-transformed
    positive values.

    Args:
        series: Time series of positive values (cases, deaths, hospitalizations, etc.)

    Returns:
        Tuple of (process_var, measurement_var) as floats

    Raises:
        ValueError: If no finite positive observations available for fitting
    """
    series = series.where(series > 0)
    series_log = np.log(series)

    if series_log.dropna().empty:
        raise ValueError("No finite observations to fit Kalman params")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UnobservedComponents(series_log, level="local level")
        result = model.fit(disp=False)

    params = dict(zip(result.param_names, result.params, strict=False))
    process_var = float(params.get("sigma2.level", 0.0))
    measurement_var = float(params.get("sigma2.irregular", 0.0))

    # Ensure positive variances
    process_var = max(process_var, 1e-6)
    measurement_var = max(measurement_var, 1e-6)

    return process_var, measurement_var


def _clip_innovation(
    innovation: float,
    *,
    innovation_sigma: float,
    innovation_clip_sigma: float,
) -> float:
    """Clip residual shocks to reduce hard re-entry transitions after missing runs."""
    if innovation_clip_sigma <= 0.0:
        return innovation
    sigma = float(max(innovation_sigma, 1e-12))
    clip = innovation_clip_sigma * sigma
    return float(np.clip(innovation, -clip, clip))


class KalmanSmoother:
    """
    Causal Kalman filter for non-censored clinical time series.

    State space model:
    - State transition: x_t = x_{t-1} + w_t,  w_t ~ N(0, process_var)
    - Measurement: z_t = x_t + v_t,  v_t ~ N(0, measurement_var)

    Missing policy:
    - "predict": pure one-step prediction through missing spans
    - "momentum": decayed slope extrapolation using last two non-missing updates
    """

    def __init__(
        self,
        *,
        process_var: float,
        measurement_var: float,
        missing_policy: str = "predict",
        momentum_decay: float = 0.5,
        momentum_max_steps: int = 4,
        innovation_clip_sigma: float = 2.5,
        reentry_gain_cap: float = 0.35,
        reentry_steps: int = 2,
        process_var_floor: float = 1e-6,
        measurement_var_floor: float = 1e-6,
    ) -> None:
        if missing_policy not in {"predict", "momentum"}:
            raise ValueError("missing_policy must be 'predict' or 'momentum'")

        self.missing_policy = missing_policy
        self.momentum_decay = float(momentum_decay)
        self.momentum_max_steps = int(momentum_max_steps)
        self.innovation_clip_sigma = float(innovation_clip_sigma)
        self.reentry_gain_cap = float(reentry_gain_cap)
        self.reentry_steps = int(reentry_steps)

        self.process_var_floor = float(process_var_floor)
        self.measurement_var_floor = float(measurement_var_floor)
        self.process_var = float(max(process_var, self.process_var_floor))
        self.measurement_var = float(max(measurement_var, self.measurement_var_floor))

        self.state = 0.0
        self.state_var = 1.0
        self.initialized = False

        self._last_actual_value: float | None = None
        self._second_last_actual_value: float | None = None
        self._actual_count = 0
        self._consecutive_missing = 0
        self._reentry_remaining = 0

    def _initialize(self, first_log_value: float) -> None:
        self.state = float(first_log_value)
        self.state_var = float(self.measurement_var)
        self.initialized = True

    def _apply_missing_update(self, *, pred_state: float, pred_var: float) -> None:
        """Advance the state through missing observations causally."""
        use_momentum = (
            self.missing_policy == "momentum"
            and self.initialized
            and self._actual_count >= 2
            and self._consecutive_missing < self.momentum_max_steps
            and self._last_actual_value is not None
            and self._second_last_actual_value is not None
        )

        if use_momentum:
            slope = self._last_actual_value - self._second_last_actual_value
            power = self._consecutive_missing + 1
            momentum = self.momentum_decay**power
            self.state = pred_state + slope * momentum
            self.state_var = pred_var + self.process_var * 2.0
        else:
            self.state = pred_state
            self.state_var = pred_var

        self._consecutive_missing += 1
        self._reentry_remaining = self.reentry_steps

    def _apply_observed_update(
        self,
        *,
        pred_state: float,
        pred_var: float,
        log_observation: float,
    ) -> None:
        """Update state from an observed point with robust re-entry controls."""
        innovation = log_observation - pred_state
        innovation_sigma = float(np.sqrt(pred_var + self.measurement_var))
        innovation = _clip_innovation(
            innovation,
            innovation_sigma=innovation_sigma,
            innovation_clip_sigma=self.innovation_clip_sigma,
        )

        s = pred_var + self.measurement_var
        k_gain = pred_var / s if s > 0.0 else 0.0

        had_actual_before = self._actual_count > 0
        if (
            had_actual_before
            and self._consecutive_missing > 0
            and self._reentry_remaining > 0
        ):
            k_gain = min(k_gain, self.reentry_gain_cap)
            self._reentry_remaining -= 1

        self.state = pred_state + k_gain * innovation
        self.state_var = max((1.0 - k_gain) * pred_var, self.process_var_floor)

        self._second_last_actual_value = self._last_actual_value
        self._last_actual_value = float(self.state)
        self._actual_count += 1
        self._consecutive_missing = 0

    def filter_series(self, values: np.ndarray) -> tuple[list[float], list[int]]:
        """
        Apply causal Kalman filtering to a time series.

        Args:
            values: Array of measurements (can contain NaN for missing)

        Returns:
            Tuple of (filtered_log_values, flags)
            flags: 0=observed, 2=missing
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(values) & (values > 0)
        flags = np.where(finite_mask, 0, 2).astype(int).tolist()

        if finite_mask.any():
            first_value = float(np.log(values[finite_mask][0] + 1e-9))
            self._initialize(first_value)

        filtered: list[float] = []

        for value in values:
            pred_state = self.state
            pred_var = self.state_var + self.process_var

            if not np.isfinite(value) or value <= 0:
                self._apply_missing_update(pred_state=pred_state, pred_var=pred_var)
                filtered.append(float(self.state))
                continue

            log_value = float(np.log(value + 1e-9))
            if not self.initialized:
                self._initialize(log_value)
                pred_state = self.state
                pred_var = self.state_var + self.process_var

            self._apply_observed_update(
                pred_state=pred_state,
                pred_var=pred_var,
                log_observation=log_value,
            )
            filtered.append(float(self.state))

        return filtered, flags


class TobitKalmanSmoother:
    """
    Causal Tobit-Kalman filter for censored wastewater time series.

    Extends Kalman filtering to handle left-censored observations
    (values below detection limit) while preserving causal updates.
    """

    def __init__(
        self,
        *,
        process_var: float,
        measurement_var: float,
        censor_inflation: float = 4.0,
        missing_policy: str = "predict",
        momentum_decay: float = 0.5,
        momentum_max_steps: int = 4,
        innovation_clip_sigma: float = 2.5,
        reentry_gain_cap: float = 0.35,
        reentry_steps: int = 2,
        process_var_floor: float = 1e-6,
        measurement_var_floor: float = 1e-6,
    ) -> None:
        if missing_policy not in {"predict", "momentum"}:
            raise ValueError("missing_policy must be 'predict' or 'momentum'")

        self.missing_policy = missing_policy
        self.momentum_decay = float(momentum_decay)
        self.momentum_max_steps = int(momentum_max_steps)
        self.innovation_clip_sigma = float(innovation_clip_sigma)
        self.reentry_gain_cap = float(reentry_gain_cap)
        self.reentry_steps = int(reentry_steps)

        self.process_var_floor = float(process_var_floor)
        self.measurement_var_floor = float(measurement_var_floor)
        self.process_var = float(max(process_var, self.process_var_floor))
        self.measurement_var = float(max(measurement_var, self.measurement_var_floor))
        self.censor_inflation = float(censor_inflation)

        self.state = 0.0
        self.state_var = 1.0
        self.initialized = False

        self._last_actual_value: float | None = None
        self._second_last_actual_value: float | None = None
        self._actual_count = 0
        self._consecutive_missing = 0
        self._reentry_remaining = 0

    def _initialize(self, first_log_value: float) -> None:
        self.state = float(first_log_value)
        self.state_var = float(self.measurement_var)
        self.initialized = True

    def _apply_missing_update(self, *, pred_state: float, pred_var: float) -> None:
        use_momentum = (
            self.missing_policy == "momentum"
            and self.initialized
            and self._actual_count >= 2
            and self._consecutive_missing < self.momentum_max_steps
            and self._last_actual_value is not None
            and self._second_last_actual_value is not None
        )

        if use_momentum:
            slope = self._last_actual_value - self._second_last_actual_value
            power = self._consecutive_missing + 1
            momentum = self.momentum_decay**power
            self.state = pred_state + slope * momentum
            self.state_var = pred_var + self.process_var * 2.0
        else:
            self.state = pred_state
            self.state_var = pred_var

        self._consecutive_missing += 1
        self._reentry_remaining = self.reentry_steps

    def _apply_measurement_update(
        self,
        *,
        pred_state: float,
        pred_var: float,
        z_eff: float,
        r_eff: float,
        flag: int,
    ) -> None:
        innovation = z_eff - pred_state
        if flag == 0:
            innovation_sigma = float(np.sqrt(pred_var + r_eff))
            innovation = _clip_innovation(
                innovation,
                innovation_sigma=innovation_sigma,
                innovation_clip_sigma=self.innovation_clip_sigma,
            )

        s = pred_var + r_eff
        k_gain = pred_var / s if s > 0.0 else 0.0

        had_actual_before = self._actual_count > 0
        if (
            flag == 0
            and had_actual_before
            and self._consecutive_missing > 0
            and self._reentry_remaining > 0
        ):
            k_gain = min(k_gain, self.reentry_gain_cap)
            self._reentry_remaining -= 1

        self.state = pred_state + k_gain * innovation
        self.state_var = max((1.0 - k_gain) * pred_var, self.process_var_floor)

        self._second_last_actual_value = self._last_actual_value
        self._last_actual_value = float(self.state)
        self._actual_count += 1
        self._consecutive_missing = 0

    def filter_series(
        self, values: np.ndarray, limits: np.ndarray
    ) -> tuple[list[float], list[int]]:
        """
        Apply causal Tobit-Kalman filtering to a censored time series.

        Args:
            values: Array of measurements (can contain NaN for missing)
            limits: Array of detection limits for censoring

        Returns:
            Tuple of (filtered_log_values, flags)
            flags: 0=normal, 1=censored, 2=missing
        """
        filtered: list[float] = []
        flags: list[int] = []

        values = np.asarray(values, dtype=float).reshape(-1)
        limits = np.asarray(limits, dtype=float).reshape(-1)

        finite_mask = np.isfinite(values) & (values > 0)
        if finite_mask.any():
            first_value = float(np.log(values[finite_mask][0] + 1e-9))
            self._initialize(first_value)

        for value, limit in zip(values, limits, strict=False):
            pred_state = self.state
            pred_var = self.state_var + self.process_var
            pred_sigma = float(np.sqrt(pred_var + self.measurement_var))

            limit_valid = np.isfinite(limit) and limit > 0

            if not np.isfinite(value) or value <= 0:
                self._apply_missing_update(pred_state=pred_state, pred_var=pred_var)
                filtered.append(float(self.state))
                flags.append(2)
                continue

            if limit_valid and value <= limit:
                if not self.initialized:
                    self._initialize(float(np.log(limit + 1e-9) - 0.5))
                    pred_state = self.state
                    pred_var = self.state_var + self.process_var
                    pred_sigma = float(np.sqrt(pred_var + self.measurement_var))

                log_limit = float(np.log(limit + 1e-9))
                alpha = (log_limit - pred_state) / pred_sigma
                pdf = float(norm.pdf(alpha))
                cdf = float(max(norm.cdf(alpha), 1e-9))
                z_eff = pred_state - pred_sigma * (pdf / cdf)
                r_eff = max(
                    self.measurement_var * self.censor_inflation,
                    self.measurement_var_floor,
                )
                flag = 1
            else:
                log_value = float(np.log(value + 1e-9))
                if not self.initialized:
                    self._initialize(log_value)
                    pred_state = self.state
                    pred_var = self.state_var + self.process_var

                z_eff = log_value
                r_eff = max(self.measurement_var, self.measurement_var_floor)
                flag = 0

            self._apply_measurement_update(
                pred_state=pred_state,
                pred_var=pred_var,
                z_eff=z_eff,
                r_eff=r_eff,
                flag=flag,
            )

            filtered.append(float(self.state))
            flags.append(flag)

        return filtered, flags


class DampedHoltSmoother:
    """Causal damped Holt smoother operating in log space."""

    def __init__(
        self,
        *,
        alpha: float = 0.30,
        beta: float = 0.05,
        phi: float = 0.90,
        missing_trend_decay: float = 0.85,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.phi = float(phi)
        self.missing_trend_decay = float(missing_trend_decay)

        self.level = 0.0
        self.trend = 0.0
        self.initialized = False

    def _initialize(self, first_log_value: float) -> None:
        self.level = float(first_log_value)
        self.trend = 0.0
        self.initialized = True

    def filter_series(self, values: np.ndarray) -> tuple[list[float], list[int]]:
        """
        Apply causal damped Holt filtering to a time series.

        Args:
            values: Array of measurements (can contain NaN for missing)

        Returns:
            Tuple of (filtered_log_values, flags)
            flags: 0=observed, 2=missing
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(values) & (values > 0)
        flags = np.where(finite_mask, 0, 2).astype(int).tolist()

        if finite_mask.any():
            first_value = float(np.log(values[finite_mask][0] + 1e-9))
            self._initialize(first_value)

        filtered: list[float] = []
        for value in values:
            if not np.isfinite(value) or value <= 0:
                if not self.initialized:
                    filtered.append(0.0)
                    continue
                pred_level = self.level + self.phi * self.trend
                self.level = float(pred_level)
                self.trend = float(self.phi * self.trend * self.missing_trend_decay)
                filtered.append(float(self.level))
                continue

            log_value = float(np.log(value + 1e-9))
            if not self.initialized:
                self._initialize(log_value)
                filtered.append(float(self.level))
                continue

            pred_level = self.level + self.phi * self.trend
            innovation = log_value - pred_level
            self.level = float(pred_level + self.alpha * innovation)
            self.trend = float(self.phi * self.trend + self.beta * innovation)
            filtered.append(float(self.level))

        return filtered, flags
