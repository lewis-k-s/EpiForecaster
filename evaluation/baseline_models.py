from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

_DEFAULT_ORDERS = [(1, 0, 0), (0, 1, 1), (1, 1, 1)]
_DEFAULT_SEASONAL_ORDERS = [(0, 0, 0, 7), (1, 0, 0, 7)]


@dataclass
class BaselinePredictionResult:
    model_name: str
    predictions: np.ndarray
    fit_status: str
    fallback_reason: str
    model_order: str


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
            return None
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for order in orders:
            for seasonal in seasonal_orders:
                try:
                    model = SARIMAX(
                        y,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=50)
                    if np.isfinite(result.aic) and result.aic < best_aic:
                        best_aic = float(result.aic)
                        best_fit = result
                        best_order_repr = f"order={order};seasonal_order={seasonal}"
                except Exception:
                    continue

    if best_fit is None:
        return None

    try:
        forecast = best_fit.get_forecast(steps=horizon, exog=exog_future)
        preds = np.asarray(forecast.predicted_mean, dtype=np.float64)
        if preds.shape[0] != horizon:
            return None
        if np.any(~np.isfinite(preds)):
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
        if np.any(~np.isfinite(preds)):
            return None
        return preds, best_order_repr
    except Exception:
        return None


def _fit_var_cross_target(
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

    maxlags_capped = min(int(maxlags), max(1, n_obs - 2), max(1, n_obs // 3))
    if maxlags_capped < 1:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            fit_result = VAR(y).fit(maxlags=maxlags_capped, ic="aic", trend="c")
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
                f"k_ar={int(fit_result.k_ar)};maxlags={maxlags_capped};"
                f"n_targets={n_targets}"
            )
            return preds, order_repr
        except Exception:
            return None


def predict_with_tiered_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
    exog_train: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> BaselinePredictionResult:
    """Predict with deterministic sparse fallback chain.

    Chain:
      sarimax_calendar -> sarima -> seasonal_naive_7d -> last_observed -> global_train_median
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fallback_reasons: list[str] = []

    if exog_train is not None and exog_future is not None:
        sarimax_out = _fit_best_sarimax(
            train_values=train_values,
            train_mask=train_mask,
            horizon=horizon,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        if sarimax_out is not None:
            preds, order_repr = sarimax_out
            return BaselinePredictionResult(
                model_name="sarimax_calendar",
                predictions=preds,
                fit_status="fit_success",
                fallback_reason="",
                model_order=order_repr,
            )
        fallback_reasons.append("sarimax_unavailable")
    else:
        fallback_reasons.append("sarimax_no_exog")

    sarima_out = _fit_best_sarimax(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        exog_train=None,
        exog_future=None,
    )
    if sarima_out is not None:
        preds, order_repr = sarima_out
        return BaselinePredictionResult(
            model_name="sarima",
            predictions=preds,
            fit_status="fit_success",
            fallback_reason="|".join(fallback_reasons),
            model_order=order_repr,
        )
    fallback_reasons.append("sarima_unavailable")

    seasonal_pred = _predict_seasonal_naive(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )
    if seasonal_pred is not None:
        return BaselinePredictionResult(
            model_name="seasonal_naive_7d",
            predictions=seasonal_pred,
            fit_status="fallback",
            fallback_reason="|".join(fallback_reasons),
            model_order="",
        )
    fallback_reasons.append("seasonal_naive_unavailable")

    last_obs_pred = _predict_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
    )
    if last_obs_pred is not None:
        return BaselinePredictionResult(
            model_name="last_observed",
            predictions=last_obs_pred,
            fit_status="fallback",
            fallback_reason="|".join(fallback_reasons),
            model_order="",
        )

    fallback_reasons.append("last_observed_unavailable")
    return BaselinePredictionResult(
        model_name="global_train_median",
        predictions=np.full(horizon, float(global_train_median), dtype=np.float64),
        fit_status="fallback",
        fallback_reason="|".join(fallback_reasons),
        model_order="",
    )


def predict_with_exponential_smoothing_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    global_train_median: float,
    seasonal_period: int = 7,
) -> BaselinePredictionResult:
    """Predict with exponential smoothing followed by deterministic sparse fallbacks."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fallback_reasons: list[str] = []
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
    fallback_reasons.append("exp_smoothing_unavailable")

    seasonal_pred = _predict_seasonal_naive(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )
    if seasonal_pred is not None:
        return BaselinePredictionResult(
            model_name="seasonal_naive_7d",
            predictions=seasonal_pred,
            fit_status="fallback",
            fallback_reason="|".join(fallback_reasons),
            model_order="",
        )
    fallback_reasons.append("seasonal_naive_unavailable")

    last_obs_pred = _predict_last_observed(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
    )
    if last_obs_pred is not None:
        return BaselinePredictionResult(
            model_name="last_observed",
            predictions=last_obs_pred,
            fit_status="fallback",
            fallback_reason="|".join(fallback_reasons),
            model_order="",
        )
    fallback_reasons.append("last_observed_unavailable")

    return BaselinePredictionResult(
        model_name="global_train_median",
        predictions=np.full(horizon, float(global_train_median), dtype=np.float64),
        fit_status="fallback",
        fallback_reason="|".join(fallback_reasons),
        model_order="",
    )


def predict_with_var_cross_target_fallback(
    *,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
    target_names: list[str],
    global_train_medians: np.ndarray,
    seasonal_period: int = 7,
    maxlags: int = 14,
) -> dict[str, BaselinePredictionResult]:
    """Predict all targets jointly with VAR, fallback to per-target smoothing chain."""
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

    var_out = _fit_var_cross_target(
        train_values=train_values,
        train_mask=train_mask,
        horizon=horizon,
        maxlags=maxlags,
    )
    if var_out is not None:
        preds, order_repr = var_out
        return {
            name: BaselinePredictionResult(
                model_name="var_cross_target",
                predictions=preds[:, idx].astype(np.float64),
                fit_status="fit_success",
                fallback_reason="",
                model_order=order_repr,
            )
            for idx, name in enumerate(target_names)
        }

    output: dict[str, BaselinePredictionResult] = {}
    for idx, target_name in enumerate(target_names):
        uni_result = predict_with_exponential_smoothing_fallback(
            train_values=train_values[:, idx],
            train_mask=train_mask[:, idx],
            horizon=horizon,
            global_train_median=float(global_train_medians[idx]),
            seasonal_period=seasonal_period,
        )
        fallback_reason = "var_unavailable"
        if uni_result.fallback_reason:
            fallback_reason = f"{fallback_reason}|{uni_result.fallback_reason}"
        output[target_name] = BaselinePredictionResult(
            model_name=uni_result.model_name,
            predictions=uni_result.predictions,
            fit_status=uni_result.fit_status,
            fallback_reason=fallback_reason,
            model_order=uni_result.model_order,
        )
    return output
