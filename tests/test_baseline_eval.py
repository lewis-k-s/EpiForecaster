from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.baseline_eval import (
    _compute_metrics_from_granular_rows,
    _compute_valid_node_mask_for_target,
    _resolve_baseline_models,
    compare_model_metrics_against_baselines,
    run_baseline_evaluation,
    run_same_slice_baseline_evaluation,
)
from evaluation.baseline_models import (
    predict_with_exponential_smoothing_fallback,
    predict_with_last_observed_fallback,
    predict_with_sarima_fallback,
    predict_with_var_fallback,
    predict_with_varmax_fallback,
)
from evaluation.metrics import compute_masked_metrics_numpy


class _DummyDataConfig:
    def __init__(self, permits: dict[str, dict[str, int]] | None = None) -> None:
        self.run_id = "real"
        self._permits = permits or {
            "input": {
                "cases": 0,
                "hospitalizations": 0,
                "deaths": 0,
                "wastewater": 0,
            },
            "horizon": {
                "cases": 0,
                "hospitalizations": 0,
                "deaths": 0,
                "wastewater": 0,
            },
        }

    def resolve_missing_permit_map(self) -> dict[str, dict[str, int]]:
        return self._permits


class _DummyModelConfig:
    def __init__(self, *, input_window_length: int = 4, forecast_horizon: int = 3) -> None:
        self.input_window_length = input_window_length
        self.forecast_horizon = forecast_horizon


class _DummyTrainingConfig:
    def __init__(self) -> None:
        self.loss = None


class _DummyConfig:
    def __init__(
        self,
        *,
        permits: dict[str, dict[str, int]] | None = None,
        input_window_length: int = 4,
        forecast_horizon: int = 3,
    ) -> None:
        self.data = _DummyDataConfig(permits=permits)
        self.model = _DummyModelConfig(
            input_window_length=input_window_length,
            forecast_horizon=forecast_horizon,
        )
        self.training = _DummyTrainingConfig()


class _DummyDataset:
    def __init__(
        self,
        *,
        T: int = 40,
        N: int = 2,
        config: _DummyConfig | None = None,
        masks: dict[str, np.ndarray] | None = None,
    ):
        self.config = config or _DummyConfig()
        self.time_range = (0, T)
        self.mobility_lags: list[int] = []
        self._temporal_coords = list(range(T))
        self.target_nodes = list(range(N))
        self.temporal_covariates = torch.arange(T * 3, dtype=torch.float32).reshape(T, 3)
        self._dataset = None

        values = np.linspace(0.1, 2.0, T * N, dtype=np.float64).reshape(T, N)
        default_mask = np.ones((T, N), dtype=np.float64)
        masks = masks or {}
        hosp_mask = masks.get("hospitalizations", default_mask)
        ww_mask = masks.get("wastewater", default_mask)
        cases_mask = masks.get("cases", default_mask)
        deaths_mask = masks.get("deaths", default_mask)

        self.precomputed_hosp = torch.tensor(values, dtype=torch.float32)
        self.precomputed_hosp_mask = torch.tensor(hosp_mask, dtype=torch.float32)
        self.precomputed_ww = torch.tensor(values + 0.2, dtype=torch.float32)
        self.precomputed_ww_mask = torch.tensor(ww_mask, dtype=torch.float32)
        self.precomputed_cases_target = torch.tensor(values + 0.4, dtype=torch.float32)
        self.precomputed_cases_mask = torch.tensor(cases_mask, dtype=torch.float32)
        self.precomputed_deaths = torch.tensor(values + 0.6, dtype=torch.float32)
        self.precomputed_deaths_mask = torch.tensor(deaths_mask, dtype=torch.float32)


def _weekly_mask(T: int, N: int, *, offset: int = 0) -> np.ndarray:
    mask = np.zeros((T, N), dtype=np.float64)
    mask[offset::7, :] = 1.0
    return mask


class _SameSliceLoader:
    def __init__(self, batches: list[object]) -> None:
        self._batches = batches
        self.dataset = _DummyDataset(T=8, N=2)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def _make_same_slice_batch() -> SimpleNamespace:
    batch = SimpleNamespace(
        target_node=torch.tensor([0, 1], dtype=torch.long),
        window_start=torch.tensor([0, 1], dtype=torch.long),
        hosp_hist=torch.tensor(
            [
                [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        cases_hist=torch.tensor(
            [
                [[1.0, 1.0, 0.0], [1.5, 1.0, 0.0], [2.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [2.5, 1.0, 0.0], [3.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        deaths_hist=torch.tensor(
            [
                [[0.5, 1.0, 0.0], [0.8, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.2, 1.0, 0.0], [1.4, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        ww_hist=torch.tensor(
            [[0.5, 0.7, 1.0], [1.0, 1.2, 1.4]],
            dtype=torch.float32,
        ),
        ww_hist_mask=torch.ones((2, 3), dtype=torch.float32),
        temporal_covariates=torch.tensor(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            dtype=torch.float32,
        ),
        hosp_target=torch.tensor([[3.5, 4.0], [4.5, 5.0]], dtype=torch.float32),
        hosp_target_mask=torch.ones((2, 2), dtype=torch.float32),
        cases_target=torch.tensor([[2.5, 3.0], [3.5, 4.0]], dtype=torch.float32),
        cases_target_mask=torch.ones((2, 2), dtype=torch.float32),
        deaths_target=torch.tensor([[1.1, 1.2], [1.5, 1.6]], dtype=torch.float32),
        deaths_target_mask=torch.ones((2, 2), dtype=torch.float32),
        ww_target=torch.tensor([[1.1, 1.2], [1.5, 1.6]], dtype=torch.float32),
        ww_target_mask=torch.ones((2, 2), dtype=torch.float32),
    )
    batch.to = lambda device, **_: batch
    return batch


def test_exp_smoothing_reports_failure_when_sparse(monkeypatch):
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(
        baseline_models,
        "_fit_best_exponential_smoothing",
        lambda **kwargs: None,
    )
    train_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    train_mask = np.zeros(3, dtype=np.float64)
    result = predict_with_exponential_smoothing_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        global_train_median=2.5,
        seasonal_period=7,
    )
    assert result.model_name == "exp_smoothing"
    assert result.fit_status == "fit_failed"
    assert result.fallback_reason == "exp_smoothing_unavailable"
    assert np.all(np.isnan(result.predictions))


def test_exp_smoothing_fits_on_simple_seasonal_series():
    train_values = np.array(
        [10, 12, 15, 13, 11, 9, 8, 10, 12, 16, 14, 12, 10, 9, 8, 10, 12, 15, 13, 11],
        dtype=np.float64,
    )
    train_mask = np.ones_like(train_values, dtype=np.float64)
    result = predict_with_exponential_smoothing_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=4,
        global_train_median=float(np.median(train_values)),
        seasonal_period=7,
    )
    assert result.model_name == "exp_smoothing"
    assert result.fit_status == "fit_success"
    assert result.predictions.shape == (4,)
    assert np.all(np.isfinite(result.predictions))


def test_sarima_reports_failure_when_sparse() -> None:
    train_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    train_mask = np.zeros(3, dtype=np.float64)
    result = predict_with_sarima_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        global_train_median=2.5,
        seasonal_period=7,
    )
    assert result.model_name == "sarima"
    assert result.fit_status == "fit_failed"
    assert result.fallback_reason == "sarima_unavailable"
    assert np.all(np.isnan(result.predictions))


def test_last_observed_baseline_uses_latest_observed_value() -> None:
    train_values = np.array([1.0, 5.0, 3.0, 7.0], dtype=np.float64)
    train_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    result = predict_with_last_observed_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=3,
        global_train_median=2.5,
    )
    assert result.model_name == "last_observed"
    assert result.fit_status == "fit_success"
    assert np.allclose(result.predictions, np.array([3.0, 3.0, 3.0], dtype=np.float64))


def test_sarima_reports_failure_when_fit_returns_huge_finite_forecast(
    monkeypatch,
) -> None:
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(
        baseline_models,
        "_fit_best_sarimax",
        lambda **kwargs: (
            np.array([1.0e200, 1.0e200], dtype=np.float64),
            "order=(1, 1, 1);seasonal_order=(0, 0, 0, 7)",
        ),
    )

    train_values = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5], dtype=np.float64)
    train_mask = np.ones_like(train_values, dtype=np.float64)
    result = predict_with_sarima_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        global_train_median=2.5,
        seasonal_period=7,
    )
    assert result.model_name == "sarima"
    assert result.fit_status == "fit_failed"
    assert result.fallback_reason == "sarima_unavailable"
    assert np.all(np.isnan(result.predictions))


def test_var_predicts_jointly():
    rng = np.random.default_rng(42)
    train_values = np.zeros((48, 4), dtype=np.float64)
    train_values[0] = np.array([0.4, 0.2, 0.3, 0.1], dtype=np.float64)
    for step in range(1, train_values.shape[0]):
        prev = train_values[step - 1]
        noise = rng.normal(loc=0.0, scale=0.05, size=4)
        train_values[step, 0] = 0.65 * prev[0] + 0.20 * prev[1] + 0.05 + noise[0]
        train_values[step, 1] = 0.30 * prev[0] + 0.55 * prev[1] + 0.04 + noise[1]
        train_values[step, 2] = 0.35 * prev[0] + 0.45 * prev[2] + 0.06 + noise[2]
        train_values[step, 3] = 0.25 * prev[1] + 0.50 * prev[3] + 0.03 + noise[3]
    train_mask = np.ones_like(train_values, dtype=np.float64)
    target_names = ["hospitalizations", "wastewater", "cases", "deaths"]
    result = predict_with_var_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=3,
        target_names=target_names,
        global_train_medians=np.median(train_values, axis=0),
        seasonal_period=7,
        maxlags=6,
    )
    assert set(result.keys()) == set(target_names)
    for target in target_names:
        pred = result[target]
        assert pred.model_name == "var"
        assert pred.fit_status == "fit_success"
        assert "trend=n" in pred.model_order
        assert pred.predictions.shape == (3,)
        assert np.all(np.isfinite(pred.predictions))


def test_predict_with_var_fallback_matches_canonical_target_order():
    train_values = np.arange(48, dtype=np.float64).reshape(12, 4)
    train_mask = np.ones_like(train_values, dtype=np.float64)
    target_names = ["hospitalizations", "wastewater", "cases", "deaths"]
    result = predict_with_var_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        target_names=target_names,
        global_train_medians=np.median(train_values, axis=0),
        seasonal_period=7,
        maxlags=4,
    )
    assert list(result) == target_names


def test_var_accepts_fully_missing_target_column_as_zero_history(monkeypatch):
    import evaluation.baseline_models as baseline_models

    captured: dict[str, np.ndarray] = {}

    def _fit_var(train_values, train_mask, horizon, maxlags):
        y = baseline_models._impute_multivariate_for_fit(
            train_values=train_values,
            train_mask=train_mask,
        )
        assert y is not None
        captured["fit_values"] = y
        return np.ones((horizon, train_values.shape[1]), dtype=np.float64), "k_ar=1"

    monkeypatch.setattr(baseline_models, "_fit_var_joint", _fit_var)

    train_values = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float64),
            np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64),
        ]
    )
    train_mask = np.ones_like(train_values, dtype=np.float64)
    train_mask[:, 1] = 0.0
    target_names = ["hospitalizations", "wastewater", "cases", "deaths"]
    result = predict_with_var_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        target_names=target_names,
        global_train_medians=np.median(train_values, axis=0),
    )

    assert np.allclose(captured["fit_values"][:, 1], 0.0)
    assert all(pred.fit_status == "fit_success" for pred in result.values())


def test_varmax_passes_mask_calendar_exog(monkeypatch):
    import evaluation.baseline_models as baseline_models

    captured: dict[str, np.ndarray] = {}

    def _fit_varmax(train_values, train_mask, horizon, exog_train, exog_future):
        captured["train_values"] = train_values
        captured["train_mask"] = train_mask
        captured["exog_train"] = exog_train
        captured["exog_future"] = exog_future
        return np.ones((horizon, train_values.shape[1]), dtype=np.float64), "order=(1,0)"

    monkeypatch.setattr(baseline_models, "_fit_varmax_cross_target", _fit_varmax)

    train_values = np.arange(48, dtype=np.float64).reshape(12, 4)
    train_mask = np.ones_like(train_values, dtype=np.float64)
    exog_train = np.ones((12, 7), dtype=np.float64)
    exog_future = np.ones((2, 7), dtype=np.float64) * 2.0
    target_names = ["hospitalizations", "wastewater", "cases", "deaths"]
    result = predict_with_varmax_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        target_names=target_names,
        global_train_medians=np.median(train_values, axis=0),
        exog_train=exog_train,
        exog_future=exog_future,
    )

    assert np.array_equal(captured["exog_train"], exog_train)
    assert np.array_equal(captured["exog_future"], exog_future)
    assert list(result) == target_names
    assert all(pred.model_name == "varmax" for pred in result.values())
    assert all(pred.fit_status == "fit_success" for pred in result.values())


def test_varmax_all_missing_history_returns_zero_predictions() -> None:
    import evaluation.baseline_models as baseline_models

    train_values = np.ones((8, 4), dtype=np.float64)
    train_mask = np.zeros_like(train_values, dtype=np.float64)
    out = baseline_models._fit_varmax_cross_target(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        exog_train=np.zeros((8, 7), dtype=np.float64),
        exog_future=np.zeros((2, 7), dtype=np.float64),
    )

    assert out is not None
    preds, order_repr = out
    assert np.allclose(preds, 0.0)
    assert "active_targets=0" in order_repr


def test_var_reports_failures_when_var_unavailable(monkeypatch):
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(baseline_models, "_fit_var_joint", lambda **kwargs: None)

    train_values = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float64),
            np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64),
        ]
    )
    train_mask = np.zeros_like(train_values, dtype=np.float64)
    target_names = ["hospitalizations", "wastewater", "cases", "deaths"]
    result = predict_with_var_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        target_names=target_names,
        global_train_medians=np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float64),
        seasonal_period=7,
        maxlags=4,
    )
    for target in target_names:
        pred = result[target]
        assert pred.model_name == "var"
        assert pred.fit_status == "fit_failed"
        assert pred.fallback_reason == "var_unavailable"
        assert np.all(np.isnan(pred.predictions))


def test_resolve_baseline_models_includes_var() -> None:
    assert _resolve_baseline_models(None) == ["sarima"]
    assert _resolve_baseline_models(["last_observed"]) == ["last_observed"]
    assert _resolve_baseline_models(["var"]) == ["var"]
    assert _resolve_baseline_models(["all"]) == [
        "exp_smoothing",
        "last_observed",
        "sarima",
        "var",
    ]
    assert _resolve_baseline_models(["varmax"]) == ["varmax"]
    with pytest.raises(ValueError, match="Unsupported baseline model"):
        _resolve_baseline_models(["nope"])


def test_masked_metrics_ignore_unobserved_points():
    pred = np.array([[10.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    target = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    mask = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    metrics = compute_masked_metrics_numpy(pred, target, mask)
    assert metrics.observed_count == 1
    assert metrics.effective_count == 1.0
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0


def test_compute_valid_node_mask_for_target_preserves_daily_behavior():
    mask = np.ones((16, 2), dtype=np.float64)
    target_view = SimpleNamespace(
        spec=SimpleNamespace(target_name="cases", cadence_mode="daily"),
        mask=mask,
    )
    permit_map = {
        "input": {"cases": 0},
        "horizon": {"cases": 0},
    }
    valid = _compute_valid_node_mask_for_target(
        target_view=target_view,
        forecast_start=8,
        input_window_length=4,
        horizon=4,
        permit_map=permit_map,
    )
    assert valid.tolist() == [True, True]


def test_compute_valid_node_mask_for_target_maps_weekly_permits():
    weekly = _weekly_mask(84, 2)
    target_view = SimpleNamespace(
        spec=SimpleNamespace(
            target_name="hospitalizations",
            cadence_mode="weekly_observed_dates",
        ),
        mask=weekly,
    )
    permit_map = {
        "input": {"hospitalizations": 21},
        "horizon": {"hospitalizations": 21},
    }
    valid = _compute_valid_node_mask_for_target(
        target_view=target_view,
        forecast_start=56,
        input_window_length=28,
        horizon=28,
        permit_map=permit_map,
    )
    assert valid.tolist() == [True, True]

    strict_map = {
        "input": {"hospitalizations": 0},
        "horizon": {"hospitalizations": 0},
    }
    strict_weekly = weekly.copy()
    strict_weekly[56 + 14, 1] = 0.0
    strict_view = SimpleNamespace(
        spec=SimpleNamespace(
            target_name="hospitalizations",
            cadence_mode="weekly_observed_dates",
        ),
        mask=strict_weekly,
    )
    strict_valid = _compute_valid_node_mask_for_target(
        target_view=strict_view,
        forecast_start=56,
        input_window_length=28,
        horizon=28,
        permit_map=strict_map,
    )
    assert strict_valid.tolist() == [True, False]


def test_run_same_slice_baseline_evaluation_writes_eval_aligned_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    loader = _SameSliceLoader([_make_same_slice_batch()])

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice",
        models=["exp_smoothing", "sarima"],
        split="test",
    )

    granular_df = pd.read_csv(artifacts["baseline_granular"], dtype={"region_id": str})
    failures_df = pd.read_csv(artifacts["baseline_failures"])
    assert set(
        [
            "model",
            "selected_model",
            "split",
            "target",
            "node_id",
            "window_start",
            "horizon",
            "target_index",
            "target_date",
        ]
    ).issubset(granular_df.columns)
    assert set(granular_df["model"].unique()) == {"exp_smoothing"}
    assert set(failures_df["model"].unique()) == {"sarima"}

    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    assert {"model", "target", "mae_mean", "rmse_mean", "r2_mean"}.issubset(
        aggregate_df.columns
    )
    assert set(aggregate_df["model"].astype(str)) == {"exp_smoothing"}

    metadata = json.loads(artifacts["baseline_metadata"].read_text(encoding="utf-8"))
    assert metadata["comparison_scope"] == "same_eval_slice"


def test_run_same_slice_baseline_evaluation_supports_last_observed(
    tmp_path: Path,
    monkeypatch,
):
    loader = _SameSliceLoader([_make_same_slice_batch()])

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice_lo",
        models=["last_observed"],
        split="test",
    )

    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    assert set(aggregate_df["model"].astype(str)) == {"last_observed"}


def test_run_same_slice_baseline_evaluation_writes_var_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    loader = _SameSliceLoader([_make_same_slice_batch()])

    import evaluation.baseline_eval as baseline_eval
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )
    monkeypatch.setattr(
        baseline_models,
        "_fit_var_joint",
        lambda train_values, horizon, **kwargs: (
            np.ones((horizon, train_values.shape[1]), dtype=np.float64),
            "k_ar=1",
        ),
    )

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice_var",
        models=["var"],
        split="test",
    )

    granular_df = pd.read_csv(artifacts["baseline_granular"], dtype={"region_id": str})
    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    usage_df = pd.read_csv(artifacts["baseline_model_usage"])
    metadata = json.loads(artifacts["baseline_metadata"].read_text(encoding="utf-8"))

    assert set(granular_df["model"].unique()) == {"var"}
    assert "selected_model" in granular_df.columns
    assert set(aggregate_df["model"].unique()) == {"var"}
    assert set(aggregate_df["target"].unique()) == {
        "hospitalizations",
        "wastewater",
        "cases",
        "deaths",
    }
    assert set(usage_df["model"].unique()) == {"var"}
    assert metadata["models"] == ["var"]


def test_run_same_slice_baseline_evaluation_writes_varmax_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    batch = _make_same_slice_batch()
    batch.hosp_target_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    batch.ww_target_mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    batch.cases_target_mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    batch.deaths_target_mask = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    loader = _SameSliceLoader([batch])

    import evaluation.baseline_eval as baseline_eval
    import evaluation.baseline_models as baseline_models

    captured: dict[str, np.ndarray] = {}

    def _fit_varmax(train_values, train_mask, horizon, exog_train, exog_future):
        captured.setdefault("exog_train", exog_train.copy())
        captured.setdefault("exog_future", exog_future.copy())
        return (
            np.ones((horizon, train_values.shape[1]), dtype=np.float64),
            "order=(1,0);trend=n",
        )

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )
    monkeypatch.setattr(baseline_models, "_fit_varmax_cross_target", _fit_varmax)

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice_varmax",
        models=["varmax"],
        split="test",
    )

    granular_df = pd.read_csv(artifacts["baseline_granular"], dtype={"region_id": str})
    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    usage_df = pd.read_csv(artifacts["baseline_model_usage"])
    metadata = json.loads(artifacts["baseline_metadata"].read_text(encoding="utf-8"))

    assert set(granular_df["model"].unique()) == {"varmax"}
    assert set(aggregate_df["model"].unique()) == {"varmax"}
    assert set(usage_df["model"].unique()) == {"varmax"}
    assert metadata["models"] == ["varmax"]
    assert captured["exog_train"].shape == (3, 7)
    assert captured["exog_future"].shape == (2, 7)
    np.testing.assert_array_equal(captured["exog_train"][:, :4], np.ones((3, 4)))
    np.testing.assert_array_equal(captured["exog_future"][:, :4], np.zeros((2, 4)))
    np.testing.assert_array_equal(
        captured["exog_train"][:, 4:],
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    )
    np.testing.assert_array_equal(
        captured["exog_future"][:, 4:],
        np.array([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]),
    )


def test_run_same_slice_baseline_evaluation_var_usage_counts_failures(
    tmp_path: Path,
    monkeypatch,
):
    loader = _SameSliceLoader([_make_same_slice_batch()])

    import evaluation.baseline_eval as baseline_eval
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )
    monkeypatch.setattr(baseline_models, "_fit_var_joint", lambda **kwargs: None)

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice_var_failure",
        models=["var"],
        split="test",
    )

    usage_df = pd.read_csv(artifacts["baseline_model_usage"])
    failures_df = pd.read_csv(artifacts["baseline_failures"])
    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    assert set(usage_df["model"].unique()) == {"var"}
    assert set(usage_df["selected_model"].unique()) == {"var"}
    assert set(failures_df["model"].unique()) == {"var"}
    assert set(failures_df["error_reason"].unique()) == {"var_unavailable"}
    assert aggregate_df.empty


def test_run_same_slice_baseline_evaluation_skips_unstable_granular_rows(
    tmp_path: Path,
    monkeypatch,
):
    loader = _SameSliceLoader([_make_same_slice_batch()])

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (loader, None),
    )

    def _huge_pred(*, train_values, horizon, **kwargs):
        return SimpleNamespace(
            model_name="sarima",
            predictions=np.full(horizon, 1.0e200, dtype=np.float64),
            fit_status="fit_success",
            fallback_reason="",
            model_order="order=(1, 1, 1)",
        )

    monkeypatch.setattr(baseline_eval, "_predict_univariate_baseline", _huge_pred)

    artifacts = run_same_slice_baseline_evaluation(
        config=_DummyConfig(input_window_length=3, forecast_horizon=2),
        output_dir=tmp_path / "same_slice_unstable",
        models=["sarima"],
        split="test",
    )

    granular_df = pd.read_csv(artifacts["baseline_granular"])
    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    assert granular_df.empty
    assert aggregate_df.empty


def test_compute_metrics_from_granular_rows_filters_non_finite_values() -> None:
    rows = pd.DataFrame(
        [
            {"observed": 1.0, "abs_error": 0.5, "sq_error": 0.25},
            {"observed": np.inf, "abs_error": 1.0, "sq_error": 1.0},
            {"observed": 2.0, "abs_error": np.nan, "sq_error": 4.0},
            {"observed": 3.0, "abs_error": 1.5, "sq_error": np.inf},
        ]
    )

    metrics = _compute_metrics_from_granular_rows(rows)
    assert metrics["observed_count"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5)


def test_run_baseline_evaluation_delegates_to_same_slice_outputs(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "baseline_multi"
    import evaluation.baseline_eval as baseline_eval

    captured: dict[str, object] = {}

    def _fake_run_same_slice_baseline_evaluation(**kwargs):
        captured["kwargs"] = kwargs
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        granular = output_dir / "baseline_granular.csv"
        aggregate = output_dir / "baseline_aggregate_metrics.csv"
        usage = output_dir / "baseline_model_usage.csv"
        failures = output_dir / "baseline_failures.csv"
        metadata = output_dir / "baseline_metadata.json"
        granular.write_text("model,target\n", encoding="utf-8")
        aggregate.write_text("model,target,mae_mean,rmse_mean,r2_mean\n", encoding="utf-8")
        usage.write_text("model,selected_model,count\n", encoding="utf-8")
        failures.write_text("model,selected_model,target,error_reason\n", encoding="utf-8")
        metadata.write_text('{"comparison_scope":"same_eval_slice"}', encoding="utf-8")
        return {
            "baseline_granular": granular,
            "baseline_aggregate_metrics": aggregate,
            "baseline_model_usage": usage,
            "baseline_failures": failures,
            "baseline_metadata": metadata,
        }

    monkeypatch.setattr(
        baseline_eval,
        "run_same_slice_baseline_evaluation",
        _fake_run_same_slice_baseline_evaluation,
    )

    artifacts = run_baseline_evaluation(
        config=_DummyConfig(),
        output_dir=out_dir,
        models=["exp_smoothing", "sarima"],
        split="test",
    )
    assert set(artifacts) == {
        "baseline_granular",
        "baseline_aggregate_metrics",
        "baseline_model_usage",
        "baseline_failures",
        "baseline_metadata",
    }
    assert captured["kwargs"]["models"] == ["exp_smoothing", "sarima"]


def test_run_baseline_evaluation_weekly_heads_use_observed_cadence(
    tmp_path: Path,
    monkeypatch,
):
    permits = {
        "input": {
            "cases": 0,
            "hospitalizations": 21,
            "deaths": 0,
            "wastewater": 21,
        },
        "horizon": {
            "cases": 0,
            "hospitalizations": 21,
            "deaths": 0,
            "wastewater": 21,
        },
    }
    config = _DummyConfig(
        permits=permits,
        input_window_length=28,
        forecast_horizon=28,
    )
    dataset = _DummyDataset(
        T=140,
        N=3,
        config=config,
        masks={
            "hospitalizations": _weekly_mask(140, 3),
            "wastewater": _weekly_mask(140, 3, offset=2),
        },
    )
    dummy_loader = _SameSliceLoader([_make_same_slice_batch()])
    dummy_loader.dataset = dataset

    import evaluation.baseline_eval as baseline_eval
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (dummy_loader, None),
    )
    monkeypatch.setattr(
        baseline_models,
        "_fit_best_sarimax",
        lambda horizon, **kwargs: (np.ones(horizon, dtype=np.float64), "order=(1,0,0)"),
    )

    out_dir = tmp_path / "baseline_weekly"
    artifacts = run_baseline_evaluation(
        config=config,
        output_dir=out_dir,
        models=["sarima"],
        split="test",
    )

    aggregate_df = pd.read_csv(artifacts["baseline_aggregate_metrics"])
    metadata = json.loads(artifacts["baseline_metadata"].read_text(encoding="utf-8"))
    hosp_row = aggregate_df[
        (aggregate_df["model"] == "sarima")
        & (aggregate_df["target"] == "hospitalizations")
    ].iloc[0]
    ww_row = aggregate_df[
        (aggregate_df["model"] == "sarima")
        & (aggregate_df["target"] == "wastewater")
    ].iloc[0]
    assert hosp_row["observed_count_mean"] > 0
    assert ww_row["observed_count_mean"] > 0
    assert metadata["comparison_scope"] == "same_eval_slice"
    assert metadata["models"] == ["sarima"]


def test_compare_model_metrics_against_baselines(tmp_path: Path):
    baseline_csv = tmp_path / "baseline_aggregate_metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "sarima",
                "target": "hospitalizations",
                "mae_mean": 1.0,
                "rmse_mean": 2.0,
                "r2_mean": 0.1,
            },
            {
                "model": "sarima",
                "target": "cases",
                "mae_mean": 1.5,
                "rmse_mean": 2.5,
                "r2_mean": 0.2,
            },
            {
                "model": "exp_smoothing",
                "target": "hospitalizations",
                "mae_mean": 0.8,
                "rmse_mean": 1.8,
                "r2_mean": 0.3,
            },
            {
                "model": "last_observed",
                "target": "hospitalizations",
                "mae_mean": 1.2,
                "rmse_mean": 2.4,
                "r2_mean": 0.0,
            },
        ]
    ).to_csv(baseline_csv, index=False)

    output_csv = tmp_path / "deltas.csv"
    eval_metrics = {
        "mae_hosp_log1p_per_100k": 0.9,
        "rmse_hosp_log1p_per_100k": 1.9,
        "r2_hosp_log1p_per_100k": 0.2,
        "mae_ww_log1p_per_100k": 0.0,
        "rmse_ww_log1p_per_100k": 0.0,
        "r2_ww_log1p_per_100k": 0.0,
        "mae_cases_log1p_per_100k": 1.0,
        "rmse_cases_log1p_per_100k": 2.0,
        "r2_cases_log1p_per_100k": 0.3,
        "mae_deaths_log1p_per_100k": 0.0,
        "rmse_deaths_log1p_per_100k": 0.0,
        "r2_deaths_log1p_per_100k": 0.0,
    }
    compare_model_metrics_against_baselines(
        eval_metrics=eval_metrics,
        baseline_results_csv=baseline_csv,
        output_csv=output_csv,
    )
    deltas = pd.read_csv(output_csv)
    assert not deltas.empty
    hosp_mae = deltas[
        (deltas["target"] == "hospitalizations")
        & (deltas["metric"] == "mae")
        & (deltas["baseline_model"] == "sarima")
    ]["delta_model_minus_baseline"].iloc[0]
    assert hosp_mae == pytest.approx(-0.1)
    hosp_skill = deltas[
        (deltas["target"] == "hospitalizations")
        & (deltas["metric"] == "skill_mae")
        & (deltas["baseline_model"] == "last_observed")
    ]["model_value"].iloc[0]
    assert hosp_skill == pytest.approx(0.9 / 1.2)
    assert eval_metrics["skill_mae_hosp_log1p_per_100k_vs_lo"] == pytest.approx(0.9 / 1.2)
    assert eval_metrics["skill_rmse_hosp_log1p_per_100k_vs_lo"] == pytest.approx(1.9 / 2.4)


def test_compare_model_metrics_against_same_slice_baselines_uses_direct_aggregate(
    tmp_path: Path,
):
    baseline_dir = tmp_path / "baseline_same_slice"
    baseline_dir.mkdir()
    baseline_csv = baseline_dir / "baseline_aggregate_metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "sarima",
                "target": "cases",
                "mae_mean": 0.8,
                "rmse_mean": 1.2,
                "r2_mean": 0.4,
            }
        ]
    ).to_csv(baseline_csv, index=False)
    (baseline_dir / "baseline_metadata.json").write_text(
        json.dumps({"comparison_scope": "same_eval_slice"}),
        encoding="utf-8",
    )

    output_csv = tmp_path / "same_slice_deltas.csv"
    compare_model_metrics_against_baselines(
        eval_metrics={
            "mae_cases_log1p_per_100k": 0.5,
            "rmse_cases_log1p_per_100k": 0.9,
            "r2_cases_log1p_per_100k": 0.6,
        },
        baseline_results_csv=baseline_csv,
        output_csv=output_csv,
    )

    deltas = pd.read_csv(output_csv)
    cases_r2 = deltas[
        (deltas["target"] == "cases")
        & (deltas["baseline_model"] == "sarima")
        & (deltas["metric"] == "r2")
    ]["delta_model_minus_baseline"].iloc[0]
    assert cases_r2 == pytest.approx(0.2)
