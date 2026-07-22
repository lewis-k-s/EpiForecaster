from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

# --- Fit policies -----------------------------------------------------------
# Broadened (p,d,q) grid + higher maxiter so SARIMA/SARIMAX are not
# systematically under-fit on short (60d) histories. We no longer discard fit
# candidates merely because they raised a RuntimeWarning; instead we keep any
# candidate with a finite AIC and rely on the prediction-stability guard to
# reject numerical blow-ups.
_DEFAULT_ORDERS = [(1, 0, 0), (0, 1, 1), (1, 0, 1)]
_DEFAULT_SEASONAL_ORDERS = [(0, 0, 0, 7), (1, 0, 0, 7)]
_SARIMAX_MAXITER = 30
# VAR lag selection: search lags 1.._VAR_MAXLAGS by AIC (capped by sample size).
_VAR_MAXLAGS = 10
_VARMAX_ORDER = (1, 0)
_VARMAX_MAXITER = 50

# --- Stability guards -------------------------------------------------------
# Predictions beyond this magnitude are treated as numerical blow-ups and the
# baseline falls back to last-observed. The multiplier is intentionally loose:
# it must catch true explosions without clipping legitimately large series.
_FORECAST_ABS_SCALE_MULTIPLIER = 100.0
_FORECAST_ABS_MIN_LIMIT = 100.0
_MAX_SAFE_ABS_PREDICTION = float(np.sqrt(np.finfo(np.float64).max) / 4.0)

# Fit-status vocabulary. ``fit_success`` means the primary method produced the
# forecast; the ``fallback_*`` values mean the primary method could not fit and a
# deterministic fallback was substituted so the comparison stays on common
# support. ``selected_model`` (``BaselinePredictionResult.model_name``) records
# which method actually produced the numbers, so the fallback rate is auditable.
_FIT_SUCCESS = "fit_success"
_FALLBACK_LAST_OBSERVED = "fallback_last_observed"
_FALLBACK_GLOBAL_MEDIAN = "fallback_global_median"


@dataclass
class BaselinePredictionResult:
    model_name: str
    predictions: np.ndarray
    fit_status: str
    fallback_reason: str
    model_order: str
    prediction_scale: np.ndarray | None = None


def _prediction_abs_limit(
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> float:
    observed = np.asarray(train_values, dtype=np.float64)[np.asarray(train_mask) > 0]
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return _FORECAST_ABS_MIN_LIMIT
    baseline_scale = max(float(np.max(np.abs(observed))), 1.0)
    return min(
        _MAX_SAFE_ABS_PREDICTION,
        max(_FORECAST_ABS_MIN_LIMIT, baseline_scale * _FORECAST_ABS_SCALE_MULTIPLIER),
    )


def _predictions_are_stable(
    preds: np.ndarray,
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> bool:
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(preds)):
        return False
    return bool(
        np.all(np.abs(preds) <= _prediction_abs_limit(train_values, train_mask))
    )


def _multivariate_predictions_are_stable(
    preds: np.ndarray,
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> bool:
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2:
        return False
    if train_values.ndim != 2 or train_mask.ndim != 2:
        return False
    if train_values.shape != train_mask.shape:
        return False
    if preds.shape[1] != train_values.shape[1]:
        return False
    return all(
        _predictions_are_stable(
            preds[:, col],
            train_values[:, col],
            train_mask[:, col],
        )
        for col in range(preds.shape[1])
    )


def _impute_univariate_for_fit(
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray | None:
    y = train_values.astype(np.float64).copy()
    y[train_mask <= 0] = np.nan
    if np.all(~np.isfinite(y)):
        return None

    # Forward fill.
    last: float | None = None
    for i in range(y.shape[0]):
        if np.isfinite(y[i]):
            last = float(y[i])
        elif last is not None:
            y[i] = last

    # Backward fill.
    nxt: float | None = None
    for i in range(y.shape[0] - 1, -1, -1):
        if np.isfinite(y[i]):
            nxt = float(y[i])
        elif nxt is not None:
            y[i] = nxt

    if np.any(~np.isfinite(y)):
        finite = y[np.isfinite(y)]
        fill = float(np.nanmedian(finite)) if finite.size > 0 else 0.0
        y[~np.isfinite(y)] = fill
    return y.astype(np.float64)


def _impute_multivariate_for_fit(
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray | None:
    if train_values.ndim != 2 or train_mask.ndim != 2:
        return None
    if train_values.shape != train_mask.shape:
        return None

    cols: list[np.ndarray] = []
    for col in range(train_values.shape[1]):
        col_fit = _impute_univariate_for_fit(
            train_values=train_values[:, col],
            train_mask=train_mask[:, col],
        )
        if col_fit is None:
            col_fit = np.zeros(train_values.shape[0], dtype=np.float64)
        cols.append(col_fit)
    return np.column_stack(cols).astype(np.float64)


def _last_observed_value(
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> float | None:
    observed_idx = np.flatnonzero(train_mask > 0)
    if observed_idx.size == 0:
        return None
    return float(train_values[int(observed_idx[-1])])


def _predict_last_observed(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
) -> np.ndarray | None:
    value = _last_observed_value(train_values, train_mask)
    if value is None:
        return None
    return np.full(horizon, value, dtype=np.float64)


def _primary_success(
    predictions: np.ndarray,
    order_repr: str,
    *,
    model_name: str,
    prediction_scale: np.ndarray | None = None,
) -> BaselinePredictionResult:
    return BaselinePredictionResult(
        model_name=model_name,
        predictions=np.asarray(predictions, dtype=np.float64),
        fit_status=_FIT_SUCCESS,
        fallback_reason="",
        model_order=order_repr,
        prediction_scale=prediction_scale,
    )


def _empirical_residual_scale(
    train_values: np.ndarray,
    train_mask: np.ndarray,
) -> float:
    """Estimate a conservative one-step residual scale for normal baselines."""
    y = _impute_univariate_for_fit(train_values=train_values, train_mask=train_mask)
    if y is None or y.shape[0] < 2:
        observed = np.asarray(train_values, dtype=np.float64)[
            np.asarray(train_mask) > 0
        ]
        observed = observed[np.isfinite(observed)]
        if observed.size >= 2:
            scale = float(np.std(observed, ddof=1))
        else:
            scale = 1.0
    else:
        residuals = np.diff(np.asarray(y, dtype=np.float64))
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size >= 2:
            scale = float(np.std(residuals, ddof=1))
        elif residuals.size == 1:
            scale = float(abs(residuals[0]))
        else:
            scale = 1.0
    return max(scale, 1.0e-3)


def _with_empirical_normal_scale(
    result: BaselinePredictionResult,
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    model_name: str,
) -> BaselinePredictionResult:
    scale = _empirical_residual_scale(train_values, train_mask)
    return BaselinePredictionResult(
        model_name=model_name
        if result.fit_status == _FIT_SUCCESS
        else result.model_name,
        predictions=result.predictions,
        fit_status=result.fit_status,
        fallback_reason=result.fallback_reason,
        model_order=result.model_order,
        prediction_scale=np.full_like(result.predictions, scale, dtype=np.float64),
    )


def _fallback_to_last_observed(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    reason: str,
) -> BaselinePredictionResult:
    """Last-observed fallback, then global-median backstop. Never returns NaNs.

    Every baseline flows through this on primary-fit failure so that all methods
    are scored on the same (target, node, time) cells (common support). The
    ``model_name`` reflects which method actually produced the forecast, making
    the fallback rate auditable downstream.
    """
    last_obs_pred = _predict_last_observed(train_values, train_mask, horizon)
    if last_obs_pred is not None and np.all(np.isfinite(last_obs_pred)):
        return BaselinePredictionResult(
            model_name="last_observed",
            predictions=last_obs_pred.astype(np.float64),
            fit_status=_FALLBACK_LAST_OBSERVED,
            fallback_reason=reason,
            model_order="",
        )
    fill = float(global_train_median) if np.isfinite(global_train_median) else 0.0
    return BaselinePredictionResult(
        model_name="global_median",
        predictions=np.full(horizon, fill, dtype=np.float64),
        fit_status=_FALLBACK_GLOBAL_MEDIAN,
        fallback_reason=reason,
        model_order="",
    )


def _select_best_sarimax_fit(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    *,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
    horizon: int,
    orders: list[tuple[int, int, int]] | None = None,
    seasonal_orders: list[tuple[int, int, int, int]] | None = None,
) -> tuple[Any, str] | None:
    observed_count = int((train_mask > 0).sum())
    if observed_count < 8:
        return None

    orders = orders or list(_DEFAULT_ORDERS)
    seasonal_orders = seasonal_orders or list(_DEFAULT_SEASONAL_ORDERS)
    y = train_values.astype(np.float64).copy()
    y[train_mask <= 0] = np.nan

    exog_train_fit = _sanitize_exog(exog_train)
    exog_future_fit = _sanitize_exog(exog_future)
    if exog_train_fit is not None and exog_train_fit.shape[0] != y.shape[0]:
        return None
    if exog_future_fit is not None and exog_future_fit.shape[0] != horizon:
        return None
    if (
        exog_train_fit is not None
        and exog_future_fit is not None
        and exog_train_fit.shape[1] != exog_future_fit.shape[1]
    ):
        return None

    best_aic = float("inf")
    best_fit = None
    best_order_repr = ""

    for order in orders:
        for seasonal in seasonal_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y,
                        exog=exog_train_fit,
                        order=order,
                        seasonal_order=seasonal,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=_SARIMAX_MAXITER)
                if not np.isfinite(result.aic):
                    continue
                if result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_fit = result
                    best_order_repr = f"order={order};seasonal_order={seasonal}"
            except Exception:
                continue

    if best_fit is None:
        return None
    return best_fit, best_order_repr


def _sanitize_exog(exog: np.ndarray | None) -> np.ndarray | None:
    if exog is None:
        return None
    return np.nan_to_num(
        np.asarray(exog, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _fit_best_sarimax(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    *,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
    orders: list[tuple[int, int, int]] | None = None,
    seasonal_orders: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, str] | None:
    selected = _select_best_sarimax_fit(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
        orders=orders,
        seasonal_orders=seasonal_orders,
    )
    if selected is None:
        return None
    best_fit, best_order_repr = selected
    exog_future_fit = _sanitize_exog(exog_future)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = best_fit.get_forecast(steps=horizon, exog=exog_future_fit)
            preds = np.asarray(forecast.predicted_mean, dtype=np.float64)
        if preds.shape[0] != horizon:
            return None
        if not _predictions_are_stable(preds, train_values, train_mask):
            return None
        return preds, best_order_repr
    except Exception:
        return None


def _fit_best_sarimax_normal(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    *,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    selected = _select_best_sarimax_fit(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
    )
    if selected is None:
        return None
    best_fit, best_order_repr = selected
    exog_future_fit = _sanitize_exog(exog_future)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = best_fit.get_forecast(steps=horizon, exog=exog_future_fit)
            preds = np.asarray(forecast.predicted_mean, dtype=np.float64).reshape(-1)
            scale = np.asarray(forecast.se_mean, dtype=np.float64).reshape(-1)
        if preds.shape[0] != horizon or scale.shape[0] != horizon:
            return None
        empirical_scale = _empirical_residual_scale(train_values, train_mask)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, empirical_scale)
        scale = np.maximum(scale, 1.0e-3)
        if not _predictions_are_stable(preds, train_values, train_mask):
            return None
        return preds, scale, best_order_repr
    except Exception:
        return None


def _fit_best_exponential_smoothing(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    *,
    seasonal_period: int = 7,
) -> tuple[np.ndarray, str] | None:
    observed_count = int((train_mask > 0).sum())
    if observed_count < 3:
        return None

    y = _impute_univariate_for_fit(train_values=train_values, train_mask=train_mask)
    if y is None:
        return None

    use_seasonal = (
        seasonal_period > 1
        and y.shape[0] >= 2 * seasonal_period
        and observed_count >= 2 * seasonal_period
    )

    candidates: list[tuple[str | None, str | None, bool]] = []
    trends: list[str | None] = [None, "add"]
    seasonal_opts: list[str | None] = [None, "add"] if use_seasonal else [None]
    for trend in trends:
        for seasonal in seasonal_opts:
            for damped in [False, True]:
                if trend is None and damped:
                    continue
                candidates.append((trend, seasonal, damped))

    best_fit = None
    best_score = (float("inf"), float("inf"))
    best_order_repr = ""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for trend, seasonal, damped in candidates:
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_period if seasonal is not None else None,
                    initialization_method="estimated",
                )
                fit_result = model.fit(optimized=True, use_brute=False)
                aic = (
                    float(fit_result.aic)
                    if np.isfinite(getattr(fit_result, "aic", np.nan))
                    else float("inf")
                )
                sse = (
                    float(fit_result.sse)
                    if np.isfinite(getattr(fit_result, "sse", np.nan))
                    else float("inf")
                )
                score = (aic, sse)
                if score < best_score:
                    best_score = score
                    best_fit = fit_result
                    best_order_repr = (
                        f"trend={trend or 'none'};seasonal={seasonal or 'none'};"
                        f"damped={damped};seasonal_period={seasonal_period}"
                    )
            except Exception:
                continue

    if best_fit is None:
        return None

    try:
        preds = np.asarray(best_fit.forecast(horizon), dtype=np.float64).reshape(-1)
        if preds.shape[0] != horizon:
            return None
        if not _predictions_are_stable(preds, train_values, train_mask):
            return None
        return preds, best_order_repr
    except Exception:
        return None


def _fit_var_joint(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    *,
    maxlags: int = _VAR_MAXLAGS,
) -> tuple[np.ndarray, str] | None:
    y = _impute_multivariate_for_fit(train_values=train_values, train_mask=train_mask)
    if y is None:
        return None

    n_obs, n_targets = y.shape
    if n_targets < 2 or n_obs < 6:
        return None

    # AIC lag selection over a feasible range. VAR.fit selects the best lag by
    # the information criterion, replacing the old fixed lag=1 policy that
    # systematically mis-specified the dynamics.
    max_feasible_lags = min(int(maxlags), max(1, n_obs - n_targets - 1))
    if max_feasible_lags < 1:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            fit_result = VAR(y).fit(maxlags=max_feasible_lags, ic="aic", trend="c")
            if int(fit_result.k_ar) < 1:
                return None
            forecast_input = y[-int(fit_result.k_ar) :, :]
            preds = np.asarray(
                fit_result.forecast(y=forecast_input, steps=horizon),
                dtype=np.float64,
            )
            if preds.shape != (horizon, n_targets):
                return None
            if np.any(~np.isfinite(preds)):
                return None
            order_repr = (
                f"k_ar={int(fit_result.k_ar)};trend=c;"
                f"maxlags={max_feasible_lags};n_targets={n_targets}"
            )
            return preds, order_repr
        except Exception:
            return None


def _fit_varmax_cross_target(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    *,
    exog_train: np.ndarray,
    exog_future: np.ndarray,
) -> tuple[np.ndarray, str] | None:
    y = _impute_multivariate_for_fit(train_values=train_values, train_mask=train_mask)
    if y is None:
        return None

    n_obs, n_targets = y.shape
    if n_targets < 2 or n_obs < 6:
        return None
    if exog_train.ndim != 2 or exog_future.ndim != 2:
        return None
    if exog_train.shape[0] != n_obs or exog_future.shape[0] != horizon:
        return None
    if exog_train.shape[1] != exog_future.shape[1]:
        return None

    exog_train = np.nan_to_num(
        np.asarray(exog_train, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    exog_future = np.nan_to_num(
        np.asarray(exog_future, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    active_cols = np.flatnonzero(
        np.asarray(train_mask, dtype=np.float64).sum(axis=0) > 0
    )
    if active_cols.size == 0:
        # No target has any observed history: defer to per-target fallback.
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            y_active = y[:, active_cols]
            if active_cols.size == 1:
                model = SARIMAX(
                    y_active[:, 0],
                    exog=exog_train,
                    order=(1, 0, 0),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_result = model.fit(disp=False, maxiter=_VARMAX_MAXITER)
                active_preds = np.asarray(
                    fit_result.forecast(steps=horizon, exog=exog_future),
                    dtype=np.float64,
                ).reshape(horizon, 1)
            else:
                # trend="c" supplies the level that the previous trend="n" fit
                # could not represent, which is what made VARMAX diverge on
                # non-zero-mean log-scale targets.
                model = VARMAX(
                    y_active,
                    exog=exog_train,
                    order=_VARMAX_ORDER,
                    trend="c",
                    error_cov_type="diagonal",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_result = model.fit(disp=False, maxiter=_VARMAX_MAXITER)
                active_preds = np.asarray(
                    fit_result.forecast(steps=horizon, exog=exog_future),
                    dtype=np.float64,
                )
            if active_preds.shape != (horizon, active_cols.size):
                return None
            preds = np.zeros((horizon, n_targets), dtype=np.float64)
            preds[:, active_cols] = active_preds
            if np.any(~np.isfinite(preds)):
                return None
            order_repr = (
                f"order={_VARMAX_ORDER};trend=c;n_targets={n_targets};"
                f"n_exog={exog_train.shape[1]};active_targets={active_cols.size};"
                f"active_target_indices={','.join(str(int(i)) for i in active_cols)}"
            )
            return preds, order_repr
        except Exception:
            return None


def predict_with_sarima_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
) -> BaselinePredictionResult:
    """Pure-series SARIMA (no exogenous covariates), last-observed on failure."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fit = _fit_best_sarimax(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=None,
        exog_future=None,
    )
    if fit is not None:
        preds, order_repr = fit
        if _predictions_are_stable(preds, train_values, train_mask):
            return _primary_success(preds, order_repr, model_name="sarima")
    return _fallback_to_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        global_train_median=global_train_median,
        reason="sarima_unavailable",
    )


def predict_with_sarimax_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> BaselinePredictionResult:
    """SARIMAX with exogenous temporal covariates (calendar/holiday/lockdown).

    This is the covariate-aware counterpart to :func:`predict_with_sarima_fallback`
    so that classical baselines get the same calendar context the neural model
    is permitted. Mobility and region embeddings are intentionally not fed: they
    are cross-sectional representations that do not map onto per-series exogenous
    regressors, and isolating them is the job of the ablations.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fit = _fit_best_sarimax(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
    )
    if fit is not None:
        preds, order_repr = fit
        if _predictions_are_stable(preds, train_values, train_mask):
            return _primary_success(preds, order_repr, model_name="sarimax")
    return _fallback_to_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        global_train_median=global_train_median,
        reason="sarimax_unavailable",
    )


def predict_with_sarimax_normal_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> BaselinePredictionResult:
    """SARIMAX mean forecast with normal predictive scale.

    The primary scale comes from statsmodels forecast standard errors. If SARIMAX
    cannot fit, this falls back to last-observed/global-median means with an
    empirical one-step residual scale from the history window.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fit = _fit_best_sarimax_normal(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
    )
    if fit is not None:
        preds, scale, order_repr = fit
        return _primary_success(
            preds,
            order_repr,
            model_name="sarimax_normal",
            prediction_scale=scale,
        )

    fallback = _fallback_to_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        global_train_median=global_train_median,
        reason="sarimax_normal_unavailable",
    )
    return _with_empirical_normal_scale(
        fallback,
        train_values=train_values,
        train_mask=train_mask,
        model_name="sarimax_normal",
    )


def predict_with_exponential_smoothing_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
) -> BaselinePredictionResult:
    """Exponential smoothing, last-observed on failure."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fit = _fit_best_exponential_smoothing(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )
    if fit is not None:
        preds, order_repr = fit
        return _primary_success(preds, order_repr, model_name="exp_smoothing")
    return _fallback_to_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        global_train_median=global_train_median,
        reason="exp_smoothing_unavailable",
    )


def predict_with_last_observed_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
) -> BaselinePredictionResult:
    """Last-observed persistence, global-median backstop when no history exists."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    last_obs_pred = _predict_last_observed(train_values, train_mask, horizon)
    if last_obs_pred is not None and np.all(np.isfinite(last_obs_pred)):
        return _primary_success(last_obs_pred, "", model_name="last_observed")
    fill = float(global_train_median) if np.isfinite(global_train_median) else 0.0
    return BaselinePredictionResult(
        model_name="global_median",
        predictions=np.full(horizon, fill, dtype=np.float64),
        fit_status=_FALLBACK_GLOBAL_MEDIAN,
        fallback_reason="last_observed_unavailable",
        model_order="",
    )


def predict_with_var_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    target_names: list[str],
    global_train_medians: np.ndarray,
    seasonal_period: int = 7,
    maxlags: int = _VAR_MAXLAGS,
) -> dict[str, BaselinePredictionResult]:
    """Joint multivariate VAR (no exogenous covariates), per-target fallback."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if train_values.ndim != 2 or train_mask.ndim != 2:
        raise ValueError("train_values and train_mask must be 2D for VAR baseline")
    if train_values.shape != train_mask.shape:
        raise ValueError("train_values and train_mask must have the same shape")
    if train_values.shape[1] != len(target_names):
        raise ValueError("target_names length must match train_values target dimension")
    if global_train_medians.shape[0] != len(target_names):
        raise ValueError("global_train_medians length must match target_names")

    fit = _fit_var_joint(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        maxlags=maxlags,
    )
    if fit is not None:
        preds, order_repr = fit
        if _multivariate_predictions_are_stable(preds, train_values, train_mask):
            return {
                name: _primary_success(preds[:, idx], order_repr, model_name="var")
                for idx, name in enumerate(target_names)
            }

    return {
        name: _fallback_to_last_observed(
            train_values=train_values[:, idx],
            train_mask=train_mask[:, idx],
            horizon=horizon,
            global_train_median=float(global_train_medians[idx]),
            reason="var_unavailable",
        )
        for idx, name in enumerate(target_names)
    }


def predict_with_varmax_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    target_names: list[str],
    global_train_medians: np.ndarray,
    exog_train: np.ndarray,
    exog_future: np.ndarray,
) -> dict[str, BaselinePredictionResult]:
    """Joint multivariate VARMAX with exogenous temporal covariates.

    Covariate-aware counterpart to :func:`predict_with_var_fallback`. Exogenous
    inputs are the temporal (calendar/holiday/lockdown) covariates only.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if train_values.ndim != 2 or train_mask.ndim != 2:
        raise ValueError("train_values and train_mask must be 2D for VARMAX baseline")
    if train_values.shape != train_mask.shape:
        raise ValueError("train_values and train_mask must have the same shape")
    if train_values.shape[1] != len(target_names):
        raise ValueError("target_names length must match train_values target dimension")
    if global_train_medians.shape[0] != len(target_names):
        raise ValueError("global_train_medians length must match target_names")

    fit = _fit_varmax_cross_target(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
    )
    if fit is not None:
        preds, order_repr = fit
        if _multivariate_predictions_are_stable(preds, train_values, train_mask):
            return {
                name: _primary_success(preds[:, idx], order_repr, model_name="varmax")
                for idx, name in enumerate(target_names)
            }

    return {
        name: _fallback_to_last_observed(
            train_values=train_values[:, idx],
            train_mask=train_mask[:, idx],
            horizon=horizon,
            global_train_median=float(global_train_medians[idx]),
            reason="varmax_unavailable",
        )
        for idx, name in enumerate(target_names)
    }
