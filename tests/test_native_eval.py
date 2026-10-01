"""Native evaluation, WIS, and schema-v4 export tests (slice EVL-1)."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from data.query_contract import ForecastQuery
from evaluation.eval_loop import (
    assert_native_confirmation_period_safe,
)
from evaluation.metrics import TorchWeightedIntervalScoreAccumulator

pytestmark = pytest.mark.epiforecaster


def test_wis_matches_hand_computation_and_component_identity() -> None:
    accumulator = TorchWeightedIntervalScoreAccumulator(
        device=torch.device("cpu"), interval_levels=(0.5, 0.8), horizon=2
    )
    targets = torch.tensor([[0.0, 4.0]])
    median = torch.tensor([[1.0, 1.0]])
    intervals = {
        "50": (torch.tensor([[0.0, 0.0]]), torch.tensor([[2.0, 2.0]])),
        "80": (torch.tensor([[-1.0, -1.0]]), torch.tensor([[3.0, 3.0]])),
    }

    batch = accumulator.update(
        median=median,
        targets=targets,
        intervals=intervals,
        observed_mask=torch.ones_like(targets),
    )
    result = accumulator.finalize()

    # K=2 -> normalizer=2.5. Point components are 0.5*|error|/2.5.
    # Dispersion is ((0.5/2)*2 + (0.2/2)*4)/2.5 = 0.36.
    # The second target lies 1 above the 80% upper bound and 2 above the
    # 50% upper bound, so underprediction=(1+2)/2.5=1.2.
    assert batch.point.tolist()[0] == pytest.approx([0.2, 0.6])
    assert batch.dispersion.tolist()[0] == pytest.approx([0.36, 0.36])
    assert batch.underprediction.tolist()[0] == pytest.approx([0.0, 1.2])
    assert batch.overprediction.tolist()[0] == pytest.approx([0.0, 0.0])
    assert batch.wis.tolist()[0] == pytest.approx([0.56, 2.16])
    assert result.wis == pytest.approx(1.36)
    assert result.wis == pytest.approx(
        result.point
        + result.dispersion
        + result.underprediction
        + result.overprediction
    )
    assert result.wis_per_h == pytest.approx([0.56, 2.16])


def test_wis_zero_width_intervals_and_masked_rows() -> None:
    accumulator = TorchWeightedIntervalScoreAccumulator(
        device=torch.device("cpu"), interval_levels=(0.5,), horizon=1
    )
    target = torch.tensor([[2.0], [100.0]])
    point = torch.tensor([[2.0], [0.0]])
    bounds = torch.tensor([[2.0], [0.0]])
    batch = accumulator.update(
        median=point,
        targets=target,
        intervals={"50": (bounds, bounds)},
        observed_mask=torch.tensor([[1.0], [0.0]]),
    )
    result = accumulator.finalize()

    assert batch.wis[0, 0].item() == 0.0
    assert result.wis == 0.0
    assert result.dispersion == 0.0
    assert result.observed_count == 1
    assert result.effective_count == 1.0


def test_confirmation_period_guard_raises_unless_disabled() -> None:
    dates = np.datetime64("2020-12-28", "D") + np.arange(14, dtype="timedelta64[D]")
    query = ForecastQuery(
        source_type="municipality",
        source_id="01001",
        forecast_origin=1,
        horizon_days=7,
        requested_heads=frozenset({"cases"}),
    )
    with pytest.raises(ValueError, match="confirmation period"):
        assert_native_confirmation_period_safe(
            queries=[query], date_grid=dates, confirmation_guard=True
        )
    assert_native_confirmation_period_safe(
        queries=[query], date_grid=dates, confirmation_guard=False
    )
