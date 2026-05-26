from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

# Short-window SARIMA policy for 60d history / 28d horizon evaluation.
_DEFAULT_ORDERS = [(1, 0, 0), (0, 1, 1)]
_DEFAULT_SEASONAL_ORDERS = [(0, 0, 0, 7), (1, 0, 0, 7)]
_SARIMAX_MAXITER = 20
_VAR_FIXED_LAG = 1
_VARMAX_ORDER = (1, 0)
_VARMAX_MAXITER = 50
_FORECAST_ABS_SCALE_MULTIPLIER = 100.0
_FORECAST_ABS_MIN_LIMIT = 100.0
_MAX_SAFE_ABS_PREDICTION = float(np.sqrt(np.finfo(np.float64).max) / 4.0)


@dataclass
class BaselinePredictionResult:
    model_name: str
    predictions: np.ndarray
    fit_status: str
    fallback_reason: str
    model_order: str


def _failed_prediction(
    *,
    model_name: str,
    horizon: int,
    reason: str,
) -> BaselinePredictionResult:
    return BaselinePredictionResult(
        model_name=model_name,
        predictions=np.full(horizon, np.nan, dtype=np.float64),
        fit_status="fit_failed",
        fallback_reason=reason,
        model_order="",
    )


def _has_runtime_warning(caught_warnings: list[warnings.WarningMessage]) -> bool:
    return any(issubclass(w.category, RuntimeWarning) for w in caught_warnings)


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
    return bool(np.all(np.abs(preds) <= _prediction_abs_limit(train_values, train_mask)))


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


def _predict_seasonal_naive(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray | None:
    if train_values.shape[0] < seasonal_period:
        return None

    preds = np.zeros(horizon, dtype=np.float64)
    last_observed = _last_observed_value(train_values, train_mask)
    if last_observed is None:
        return None

    train_len = train_values.shape[0]
    for step in range(horizon):
        idx = train_len - seasonal_period + step
        if 0 <= idx < train_len and train_mask[idx] > 0:
            preds[step] = float(train_values[idx])
        else:
            preds[step] = last_observed
    return preds


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
    observed_count = int((train_mask > 0).sum())
    if observed_count < 8:
        return None

    orders = orders or list(_DEFAULT_ORDERS)
    seasonal_orders = seasonal_orders or list(_DEFAULT_SEASONAL_ORDERS)
    y = train_values.astype(np.float64).copy()
    y[train_mask <= 0] = np.nan

    if exog_train is not None and exog_train.shape[0] != y.shape[0]:
        return None
    if exog_future is not None and exog_future.shape[0] != horizon:
        return None

    best_aic = float("inf")
    best_fit = None
    best_order_repr = ""

    for order in orders:
        for seasonal in seasonal_orders:
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", RuntimeWarning)
                    model = SARIMAX(
                        y,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=_SARIMAX_MAXITER)
                if _has_runtime_warning(caught_warnings):
                    continue
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_fit = result
                    best_order_repr = f"order={order};seasonal_order={seasonal}"
            except Exception:
                continue

    if best_fit is None:
        return None

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            forecast = best_fit.get_forecast(steps=horizon, exog=exog_future)
            preds = np.asarray(forecast.predicted_mean, dtype=np.float64)
        if _has_runtime_warning(caught_warnings):
            return None
        if preds.shape[0] != horizon:
            return None
        if not _predictions_are_stable(preds, train_values, train_mask):
            return None
        return preds, best_order_repr
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
    maxlags: int = 14,
) -> tuple[np.ndarray, str] | None:
    y = _impute_multivariate_for_fit(train_values=train_values, train_mask=train_mask)
    if y is None:
        return None

    n_obs, n_targets = y.shape
    if n_targets < 2 or n_obs < 6:
        return None

    lag_order = min(_VAR_FIXED_LAG, int(maxlags), max(1, n_obs - 2))
    if lag_order < 1:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            fit_result = VAR(y).fit(maxlags=lag_order, ic=None, trend="n")
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
                f"k_ar={int(fit_result.k_ar)};trend=n;"
                f"maxlags={lag_order};n_targets={n_targets}"
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

    active_cols = np.flatnonzero(np.asarray(train_mask, dtype=np.float64).sum(axis=0) > 0)
    if active_cols.size == 0:
        order_repr = (
            f"order={_VARMAX_ORDER};trend=n;n_targets={n_targets};"
            f"n_exog={exog_train.shape[1]};active_targets=0"
        )
        return np.zeros((horizon, n_targets), dtype=np.float64), order_repr

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            y_active = y[:, active_cols]
            if active_cols.size == 1:
                model = SARIMAX(
                    y_active[:, 0],
                    exog=exog_train,
                    order=(1, 0, 0),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_result = model.fit(disp=False, maxiter=_VARMAX_MAXITER)
                active_preds = np.asarray(
                    fit_result.forecast(steps=horizon, exog=exog_future),
                    dtype=np.float64,
                ).reshape(horizon, 1)
            else:
                model = VARMAX(
                    y_active,
                    exog=exog_train,
                    order=_VARMAX_ORDER,
                    trend="n",
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
                f"order={_VARMAX_ORDER};trend=n;n_targets={n_targets};"
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
    """Predict with pure-series SARIMA, without substituting another baseline."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    sarima_out = _fit_best_sarimax(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=None,
        exog_future=None,
    )
    if sarima_out is not None:
        preds, order_repr = sarima_out
        if _predictions_are_stable(preds, train_values, train_mask):
            return BaselinePredictionResult(
                model_name="sarima",
                predictions=preds,
                fit_status="fit_success",
                fallback_reason="",
                model_order=order_repr,
            )
    return _failed_prediction(
        model_name="sarima",
        horizon=horizon,
        reason="sarima_unavailable",
    )


def predict_with_exponential_smoothing_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
) -> BaselinePredictionResult:
    """Predict with exponential smoothing, without substituting another baseline."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    exp_smooth_out = _fit_best_exponential_smoothing(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )
    if exp_smooth_out is not None:
        preds, order_repr = exp_smooth_out
        return BaselinePredictionResult(
            model_name="exp_smoothing",
            predictions=preds,
            fit_status="fit_success",
            fallback_reason="",
            model_order=order_repr,
        )
    return _failed_prediction(
        model_name="exp_smoothing",
        horizon=horizon,
        reason="exp_smoothing_unavailable",
    )


def predict_with_last_observed_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
) -> BaselinePredictionResult:
    """Predict with the last observed value, without substituting another baseline."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    last_obs_pred = _predict_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
    )
    if last_obs_pred is not None:
        return BaselinePredictionResult(
            model_name="last_observed",
            predictions=last_obs_pred,
            fit_status="fit_success",
            fallback_reason="",
            model_order="",
        )

    return _failed_prediction(
        model_name="last_observed",
        horizon=horizon,
        reason="last_observed_unavailable",
    )


def predict_with_var_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    target_names: list[str],
    global_train_medians: np.ndarray,
    seasonal_period: int = 7,
    maxlags: int = 14,
) -> dict[str, BaselinePredictionResult]:
    """Predict all targets jointly with VAR, without substituting another baseline."""
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

    var_out = _fit_var_joint(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        maxlags=maxlags,
    )
    if var_out is not None:
        preds, order_repr = var_out
        if _multivariate_predictions_are_stable(preds, train_values, train_mask):
            return {
                name: BaselinePredictionResult(
                    model_name="var",
                    predictions=preds[:, idx].astype(np.float64),
                    fit_status="fit_success",
                    fallback_reason="",
                    model_order=order_repr,
                )
                for idx, name in enumerate(target_names)
            }

    return {
        name: _failed_prediction(
            model_name="var",
            horizon=horizon,
            reason="var_unavailable",
        )
        for name in target_names
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
    """Predict all targets jointly with mask/calendar-aware VARMAX."""
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

    varmax_out = _fit_varmax_cross_target(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=exog_train,
        exog_future=exog_future,
    )
    if varmax_out is not None:
        preds, order_repr = varmax_out
        if _multivariate_predictions_are_stable(preds, train_values, train_mask):
            return {
                name: BaselinePredictionResult(
                    model_name="varmax",
                    predictions=preds[:, idx].astype(np.float64),
                    fit_status="fit_success",
                    fallback_reason="",
                    model_order=order_repr,
                )
                for idx, name in enumerate(target_names)
            }

    return {
        name: _failed_prediction(
            model_name="varmax",
            horizon=horizon,
            reason="varmax_unavailable",
        )
        for name in target_names
    }
