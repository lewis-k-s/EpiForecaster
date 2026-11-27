from __future__ import annotations

import math

import pytest
import torch

from evaluation.aggregate_export import build_main_model_target_aggregate
from evaluation.eval_loop import _add_region_set_metrics
from evaluation.losses import JointInferenceLoss
from evaluation.metrics import TorchDistributionMetricAccumulator
from evaluation.target_scoring import (
    CANONICAL_TARGET_NAMES,
    TargetForecast,
    TargetScorer,
    region_set_members,
)

pytestmark = pytest.mark.epiforecaster


def _forecast(
    head: str,
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> TargetForecast:
    return TargetForecast(
        head=head,
        prediction=loc,
        loc=loc,
        scale=scale,
        observed=targets,
        observed_mask=None,
        weights=weights,
        mechanistic=None,
        residual=None,
        latent_source="I",
        latent=None,
        true_latent=None,
    )


def _region_set_scores(
    members: dict[str, torch.Tensor],
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    nodes: torch.Tensor,
) -> dict:
    """Score every target with the same tensors and merge their region sets."""
    region_set_metrics: dict[str, dict] = {name: {} for name in members}
    for head in CANONICAL_TARGET_NAMES:
        scorer = TargetScorer(
            head,
            criterion=JointInferenceLoss(),
            device=torch.device("cpu"),
            horizon=int(targets.shape[1]),
            region_set_members=members,
        )
        scorer.update(
            _forecast(head, loc=loc, scale=scale, targets=targets, weights=weights),
            batch_data=type("Batch", (), {"target_node": nodes})(),
        )
        for name, set_metrics in scorer.finalize().region_set_metrics.items():
            region_set_metrics[name].update(set_metrics)
    metrics: dict = {}
    _add_region_set_metrics(metrics=metrics, region_set_metrics=region_set_metrics)
    return metrics


def test_region_set_scores_equal_scoring_only_member_windows() -> None:
    generator = torch.Generator().manual_seed(0)
    loc = torch.randn(4, 3, generator=generator)
    scale = torch.rand(4, 3, generator=generator) + 0.5
    targets = torch.randn(4, 3, generator=generator)
    weights = torch.ones(4, 3)
    weights[3, 1] = 0.0  # an unsupervised point stays excluded
    nodes = torch.tensor([0, 2, 1, 2])  # node index into the dataset regions
    members = region_set_members(
        {"core": frozenset({"r2"})}, ["r0", "r1", "r2"], torch.device("cpu")
    )
    metrics = _region_set_scores(
        members, loc=loc, scale=scale, targets=targets, weights=weights, nodes=nodes
    )

    reference = TorchDistributionMetricAccumulator(
        device=torch.device("cpu"), horizon=3
    )
    rows = nodes == 2
    reference.update(
        loc=loc[rows],
        scale=scale[rows],
        targets=targets[rows],
        observed_mask=None,
        sample_weights=weights[rows],
    )
    expected = reference.finalize()
    assert metrics["region_set_core_wis_cases_log1p_per_100k"] == pytest.approx(
        expected.wis
    )
    assert metrics["region_set_core_probabilistic_effective_count_ww"] == 5.0
    assert metrics["region_set_core_mean_target_wis"] == pytest.approx(expected.wis)


def test_region_set_without_support_has_nan_endpoint_and_export_columns() -> None:
    members = region_set_members(
        {"empty": frozenset({"r9"})}, ["r0", "r9"], torch.device("cpu")
    )
    metrics = _region_set_scores(
        members,
        loc=torch.zeros(1, 3),
        scale=torch.ones(1, 3),
        targets=torch.zeros(1, 3),
        weights=torch.ones(1, 3),
        nodes=torch.tensor([0]),
    )
    assert math.isnan(metrics["region_set_empty_mean_target_wis"])

    table = build_main_model_target_aggregate(metrics)
    assert {
        "region_set_empty_wis",
        "region_set_empty_probabilistic_effective_count",
        "region_set_empty_mean_target_wis",
    } <= set(table.columns)
    assert table["region_set_empty_probabilistic_effective_count"].eq(0.0).all()
