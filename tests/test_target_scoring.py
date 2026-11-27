from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evaluation.losses import JointInferenceLoss
from evaluation.target_scoring import (
    TargetForecast,
    TargetScorer,
    build_target_forecast,
)

pytestmark = pytest.mark.epiforecaster

_BATCH = SimpleNamespace(target_node=torch.tensor([0, 1]))


def _scorer(head: str = "hosp") -> TargetScorer:
    return TargetScorer(
        head,
        criterion=JointInferenceLoss(),
        device=torch.device("cpu"),
        horizon=2,
    )


def _forecast(
    head: str = "hosp",
    *,
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mechanistic: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> TargetForecast:
    return TargetForecast(
        head=head,
        prediction=prediction,
        loc=prediction,
        scale=torch.ones_like(prediction),
        observed=observed,
        observed_mask=None,
        weights=torch.ones_like(observed),
        mechanistic=mechanistic,
        residual=residual,
        latent_source="I",
        latent=None,
        true_latent=None,
    )


def test_exact_forecast_scores_and_node_statistics() -> None:
    observed = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    scorer = _scorer("cases")
    scorer.update(
        _forecast("cases", prediction=observed.clone(), observed=observed),
        batch_data=_BATCH,
    )
    score = scorer.finalize()

    metrics = score.metrics
    assert metrics["mae_cases_log1p_per_100k"] == 0.0
    assert metrics["observed_count_cases"] == 4
    assert metrics["mae_cases_per_h"] == [0.0, 0.0]
    assert metrics["rmse_cases_per_h"] == [0.0, 0.0]
    assert metrics["nll_cases_log1p_per_100k"] == pytest.approx(
        0.5 * torch.log(torch.tensor(2.0 * torch.pi)).item()
    )
    exact_normal_crps = 2.0 / (2.0 * torch.pi) ** 0.5 - 1.0 / torch.pi**0.5
    assert metrics["crps_cases_log1p_per_100k"] == pytest.approx(
        float(exact_normal_crps)
    )
    assert metrics["coverage_80_cases"] == pytest.approx(1.0)
    assert len(metrics["coverage_80_cases_per_h"]) == 2
    assert score.node_stats == {
        0: {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "observed_count": 2.0},
        1: {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "observed_count": 2.0},
    }


def test_black_box_target_reports_the_same_keys_without_pathway_support() -> None:
    observed = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    prediction = observed + 0.5
    black_box = _scorer()
    black_box.update(
        _forecast(prediction=prediction, observed=observed), batch_data=_BATCH
    )
    mechanistic = _scorer()
    mechanistic.update(
        _forecast(
            prediction=prediction,
            observed=observed,
            mechanistic=prediction - 0.25,
            residual=torch.full_like(prediction, 0.25),
        ),
        batch_data=_BATCH,
    )

    black_box_metrics = black_box.finalize().metrics
    mechanistic_metrics = mechanistic.finalize().metrics
    assert black_box_metrics.keys() == mechanistic_metrics.keys()
    assert black_box_metrics["pathway_hosp_observed_count"] == 0
    assert mechanistic_metrics["pathway_hosp_observed_count"] == 4
    assert black_box_metrics["wis_hosp_log1p_per_100k"] == pytest.approx(
        mechanistic_metrics["wis_hosp_log1p_per_100k"]
    )


def test_target_without_supervision_is_not_scored() -> None:
    model_outputs = {"pred_deaths": torch.zeros(2, 2)}
    forecast = build_target_forecast(
        "deaths",
        model_outputs=model_outputs,
        targets={"deaths": torch.zeros(2, 2)},
        supervision={"target": None, "weights": None},
        criterion=JointInferenceLoss(),
        latent_source="death_flow",
        horizon=2,
    )
    assert forecast is None

    score = _scorer("deaths").finalize()
    assert score.node_stats is None
    assert score.metrics["observed_count_deaths"] == 0
