from __future__ import annotations

import math

import pytest
import torch

from evaluation.aggregate_export import build_main_model_target_aggregate
from evaluation.eval_loop import (
    _HEAD_CANONICAL_NAMES,
    _add_region_set_metrics,
    _region_set_members,
    _update_region_set_metrics,
)
from evaluation.metrics import TorchDistributionMetricAccumulator

pytestmark = pytest.mark.epiforecaster


def _accumulator() -> TorchDistributionMetricAccumulator:
    return TorchDistributionMetricAccumulator(device=torch.device("cpu"), horizon=3)


def test_region_set_scores_equal_scoring_only_member_windows() -> None:
    generator = torch.Generator().manual_seed(0)
    loc = torch.randn(4, 3, generator=generator)
    scale = torch.rand(4, 3, generator=generator) + 0.5
    targets = torch.randn(4, 3, generator=generator)
    weights = torch.ones(4, 3)
    weights[3, 1] = 0.0  # an unsupervised point stays excluded
    nodes = torch.tensor([0, 2, 1, 2])  # node index into the dataset regions
    members = _region_set_members(
        {"core": frozenset({"r2"})}, ["r0", "r1", "r2"], torch.device("cpu")
    )
    accumulators = {"core": {head: _accumulator() for head in _HEAD_CANONICAL_NAMES}}
    for head in _HEAD_CANONICAL_NAMES:
        _update_region_set_metrics(
            accumulators,
            members,
            head_name=head,
            loc=loc,
            scale=scale,
            targets=targets,
            observed_mask=None,
            sample_weights=weights,
            nodes=nodes,
        )
    metrics: dict = {}
    _add_region_set_metrics(metrics=metrics, accumulators=accumulators)

    reference = _accumulator()
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
    members = _region_set_members(
        {"empty": frozenset({"r9"})}, ["r0", "r9"], torch.device("cpu")
    )
    accumulators = {"empty": {head: _accumulator() for head in _HEAD_CANONICAL_NAMES}}
    for head in _HEAD_CANONICAL_NAMES:
        _update_region_set_metrics(
            accumulators,
            members,
            head_name=head,
            loc=torch.zeros(1, 3),
            scale=torch.ones(1, 3),
            targets=torch.zeros(1, 3),
            observed_mask=None,
            sample_weights=torch.ones(1, 3),
            nodes=torch.tensor([0]),
        )
    metrics: dict = {}
    _add_region_set_metrics(metrics=metrics, accumulators=accumulators)
    assert math.isnan(metrics["region_set_empty_mean_target_wis"])

    table = build_main_model_target_aggregate(metrics)
    assert {
        "region_set_empty_wis",
        "region_set_empty_probabilistic_effective_count",
        "region_set_empty_mean_target_wis",
    } <= set(table.columns)
    assert table["region_set_empty_probabilistic_effective_count"].eq(0.0).all()
