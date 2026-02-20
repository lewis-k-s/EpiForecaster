from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.baseline_eval import (
    _generate_rolling_folds,
    compare_model_metrics_against_baselines,
    run_baseline_evaluation,
    run_tiered_baseline_evaluation,
)
from evaluation.baseline_models import (
    predict_with_exponential_smoothing_fallback,
    predict_with_tiered_fallback,
    predict_with_var_cross_target_fallback,
)
from evaluation.metrics import compute_masked_metrics_numpy


class _DummyDataConfig:
    run_id = "real"

    @staticmethod
    def resolve_missing_permit_map() -> dict[str, int]:
        return {
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 0,
            "wastewater": 0,
        }


class _DummyModelConfig:
    history_length = 4
    forecast_horizon = 3


class _DummyConfig:
    data = _DummyDataConfig()
    model = _DummyModelConfig()


class _DummyDataset:
    def __init__(self, *, T: int = 40, N: int = 2):
        self.config = _DummyConfig()
        self.time_range = (0, T)
        self.mobility_lags: list[int] = []
        self._temporal_coords = list(range(T))
        self.target_nodes = list(range(N))
        self.temporal_covariates = torch.zeros((T, 0), dtype=torch.float32)
        self._dataset = None

        values = np.linspace(0.1, 2.0, T * N, dtype=np.float64).reshape(T, N)
        mask = np.ones((T, N), dtype=np.float64)

        self.precomputed_hosp = torch.tensor(values, dtype=torch.float32)
        self.precomputed_hosp_mask = torch.tensor(mask, dtype=torch.float32)
        self.precomputed_ww = torch.tensor(values + 0.2, dtype=torch.float32)
        self.precomputed_ww_mask = torch.tensor(mask, dtype=torch.float32)
        self.precomputed_cases_target = torch.tensor(values + 0.4, dtype=torch.float32)
        self.precomputed_cases_mask = torch.tensor(mask, dtype=torch.float32)
        self.precomputed_deaths = torch.tensor(values + 0.6, dtype=torch.float32)
        self.precomputed_deaths_mask = torch.tensor(mask, dtype=torch.float32)


def test_generate_rolling_folds_has_no_leakage():
    dataset = _DummyDataset(T=50)
    folds = _generate_rolling_folds(dataset=dataset, rolling_folds=5)
    assert folds, "expected at least one fold"
    for fold in folds:
        assert fold.train_end == fold.forecast_start
        assert fold.forecast_start > fold.train_start
        assert fold.forecast_end > fold.train_end


def test_tiered_fallback_chain_deterministic_when_sparse(monkeypatch):
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(baseline_models, "_fit_best_sarimax", lambda **kwargs: None)
    train_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    train_mask = np.zeros(3, dtype=np.float64)
    result = predict_with_tiered_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        global_train_median=1.25,
        exog_train=None,
        exog_future=None,
    )
    assert result.model_name == "global_train_median"
    assert result.fit_status == "fallback"
    assert np.allclose(result.predictions, np.array([1.25, 1.25], dtype=np.float64))


def test_exp_smoothing_fallback_chain_deterministic_when_sparse(monkeypatch):
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
    assert result.model_name == "global_train_median"
    assert result.fit_status == "fallback"
    assert np.allclose(result.predictions, np.array([2.5, 2.5], dtype=np.float64))


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


def test_var_cross_target_predicts_jointly():
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
    result = predict_with_var_cross_target_fallback(
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
        assert pred.model_name == "var_cross_target"
        assert pred.fit_status == "fit_success"
        assert pred.predictions.shape == (3,)
        assert np.all(np.isfinite(pred.predictions))


def test_var_cross_target_falls_back_to_univariate_when_var_unavailable(monkeypatch):
    import evaluation.baseline_models as baseline_models

    monkeypatch.setattr(baseline_models, "_fit_var_cross_target", lambda **kwargs: None)

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
    result = predict_with_var_cross_target_fallback(
        train_values=train_values,
        train_mask=train_mask,
        horizon=2,
        target_names=target_names,
        global_train_medians=np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float64),
        seasonal_period=7,
        maxlags=4,
    )
    for idx, target in enumerate(target_names):
        pred = result[target]
        assert pred.model_name == "global_train_median"
        assert "var_unavailable" in pred.fallback_reason
        assert np.allclose(
            pred.predictions,
            np.array([1.1 + idx, 1.1 + idx], dtype=np.float64),
        )


def test_masked_metrics_ignore_unobserved_points():
    pred = np.array([[10.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    target = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    mask = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    metrics = compute_masked_metrics_numpy(pred, target, mask)
    assert metrics.observed_count == 1
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0


def test_run_tiered_baseline_writes_artifacts(tmp_path: Path, monkeypatch):
    dataset = _DummyDataset(T=48, N=3)
    dummy_loader = SimpleNamespace(dataset=dataset)

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (dummy_loader, None),
    )

    out_dir = tmp_path / "baseline"
    artifacts = run_tiered_baseline_evaluation(
        config=_DummyConfig(),
        output_dir=out_dir,
        split="test",
        rolling_folds=3,
        include_sparsity_bins=True,
    )

    expected = [
        "baseline_fold_metrics",
        "baseline_aggregate_metrics",
        "baseline_coverage",
        "baseline_pair_details",
        "baseline_vs_model_deltas",
        "baseline_metadata",
        "baseline_sparsity_stratified_metrics",
    ]
    for key in expected:
        assert key in artifacts
        assert artifacts[key].exists()

    fold_df = pd.read_csv(artifacts["baseline_fold_metrics"])
    assert set(["model", "target", "fold", "mae", "rmse", "smape", "r2"]).issubset(
        fold_df.columns
    )


def test_run_baseline_evaluation_multiple_models_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    dataset = _DummyDataset(T=60, N=3)
    dummy_loader = SimpleNamespace(dataset=dataset)

    import evaluation.baseline_eval as baseline_eval

    monkeypatch.setattr(
        baseline_eval,
        "build_loader_from_config",
        lambda **kwargs: (dummy_loader, None),
    )

    out_dir = tmp_path / "baseline_multi"
    artifacts = run_baseline_evaluation(
        config=_DummyConfig(),
        output_dir=out_dir,
        models=["tiered", "exp_smoothing", "var_cross_target"],
        split="test",
        rolling_folds=2,
        include_sparsity_bins=True,
    )
    assert artifacts["baseline_fold_metrics"].exists()
    fold_df = pd.read_csv(artifacts["baseline_fold_metrics"])
    assert {"tiered", "exp_smoothing", "var_cross_target"}.issubset(
        set(fold_df["model"].unique())
    )


def test_compare_model_metrics_against_baselines(tmp_path: Path):
    baseline_csv = tmp_path / "baseline_aggregate_metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "tiered",
                "target": "hospitalizations",
                "mae_median": 1.0,
                "rmse_median": 2.0,
                "smape_median": 0.5,
                "r2_median": 0.1,
            },
            {
                "model": "tiered",
                "target": "cases",
                "mae_median": 1.5,
                "rmse_median": 2.5,
                "smape_median": 0.6,
                "r2_median": 0.2,
            },
        ]
    ).to_csv(baseline_csv, index=False)

    output_csv = tmp_path / "deltas.csv"
    eval_metrics = {
        "mae_hosp_log1p_per_100k": 0.9,
        "rmse_hosp_log1p_per_100k": 1.9,
        "smape_hosp_log1p_per_100k": 0.4,
        "r2_hosp_log1p_per_100k": 0.2,
        "mae_ww_log1p_per_100k": 0.0,
        "rmse_ww_log1p_per_100k": 0.0,
        "smape_ww_log1p_per_100k": 0.0,
        "r2_ww_log1p_per_100k": 0.0,
        "mae_cases_log1p_per_100k": 1.0,
        "rmse_cases_log1p_per_100k": 2.0,
        "smape_cases_log1p_per_100k": 0.5,
        "r2_cases_log1p_per_100k": 0.3,
        "mae_deaths_log1p_per_100k": 0.0,
        "rmse_deaths_log1p_per_100k": 0.0,
        "smape_deaths_log1p_per_100k": 0.0,
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
        (deltas["target"] == "hospitalizations") & (deltas["metric"] == "mae")
    ]["delta_model_minus_baseline"].iloc[0]
    assert hosp_mae == pytest.approx(-0.1)
