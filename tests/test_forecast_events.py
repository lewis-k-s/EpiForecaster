import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evaluation.forecast_events import (
    event_metrics,
    exceedance_probability,
    fit_decision_threshold,
    fit_event_definition,
    fit_scale_multiplier,
)


def _fit_frame():
    return pd.DataFrame(
        {
            "role": ["fit"] * 5,
            "target": ["cases"] * 5,
            "source_id": ["a"] * 5,
            "target_date": ["d1", "d2", "d3", "d4", "d4"],
            "observed": [1.0, 2.0, 3.0, 100.0, 100.0],
            "observed_mask": [True] * 5,
        }
    )


def test_event_threshold_uses_unique_observations_and_stable_identity():
    frame = _fit_frame()
    fitted = fit_event_definition(frame, head="cases", quantile=0.5)
    assert fitted.threshold == 3.0
    assert fitted.unique_source_observations == 4
    assert (
        fit_event_definition(
            frame.sample(frac=1, random_state=42), head="cases", quantile=0.5
        )
        == fitted
    )
    frame.loc[4, "observed"] = 200
    with pytest.raises(ValueError, match="Conflicting"):
        fit_event_definition(frame, head="cases")


def test_event_threshold_rejects_nonfit_roles_and_ignores_missing_targets():
    frame = _fit_frame()
    frame.loc[0, "role"] = "calibration"
    with pytest.raises(ValueError, match="fit-role"):
        fit_event_definition(frame, head="cases")
    frame.loc[0, "role"] = "fit"
    frame.loc[3:4, "observed_mask"] = False
    frame.loc[3:4, "observed"] = np.nan
    assert fit_event_definition(frame, head="cases").unique_source_observations == 3


@pytest.mark.parametrize("family", ["normal", "student_t"])
def test_event_probability_is_upper_tail_and_respects_scale(family):
    p = exceedance_probability(
        np.array([1.0, 1.0]), np.array([1.0, 2.0]), 2.0, family=family
    )
    assert 0 < p[0] < p[1] < 0.5
    np.testing.assert_allclose(
        exceedance_probability(np.ones(2), np.ones(2), 1.0, family=family), 0.5
    )


def test_event_metrics_counts_brier_and_threshold_ties():
    result = event_metrics(np.array([0.9, 0.7, 0.5, 0.1]), np.array([1, 0, 1, 0]))
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (2, 1, 1, 0)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert result["brier"] == pytest.approx(0.19)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["false_alerts_per_100"] == 25
    assert sum(row["count"] for row in result["reliability_bins"]) == 4


def test_event_metrics_preserve_undefined_support_and_perfect_probabilities():
    result = event_metrics(np.array([0.0, 0.0]), np.array([0, 0]))
    assert (
        result["recall"] is result["precision"] is result["average_precision"] is None
    )
    assert result["brier"] == 0
    result = event_metrics(np.array([1.0, 1.0]), np.array([1, 1]))
    assert result["average_precision"] == result["precision"] == result["recall"] == 1
    assert result["reliability_bins"][-1]["count"] == 2
    assert event_metrics(np.array([]), np.array([]))["count"] == 0


@pytest.mark.parametrize(
    "probabilities, labels",
    [([np.nan], [1]), ([1.1], [1]), ([0.5], [np.nan]), ([0.5], [2])],
)
def test_invalid_event_inputs_are_not_silently_dropped(probabilities, labels):
    with pytest.raises(ValueError):
        event_metrics(np.array(probabilities), np.array(labels))


def test_operating_point_respects_probability_ties_and_support():
    p = np.array([0.9, 0.8, 0.8, 0.2])
    y = np.array([1, 1, 0, 1])
    fitted = fit_decision_threshold(
        p, y, precision_floor=0.8, min_alerts=1, min_positives=1
    )
    assert (
        fitted["threshold"] == 0.9
    )  # Cannot split the equal-probability observations.
    fitted = fit_decision_threshold(
        p, y, precision_floor=0.5, min_alerts=1, min_positives=1
    )
    assert fitted["threshold"] == 0.2
    assert fit_decision_threshold(p, y)["reason"] == "insufficient_positive_support"
    assert (
        fit_decision_threshold(
            p, y, precision_floor=0.9, min_alerts=2, min_positives=1
        )["reason"]
        == "no_feasible_operating_point"
    )


@pytest.mark.parametrize("objective", ["coverage_80", "wis"])
def test_scale_calibration_recovers_known_misscaling(objective):
    target = 2 * stats.norm.ppf(np.linspace(0.0005, 0.9995, 1000))
    result = fit_scale_multiplier(
        np.zeros(1000), np.ones(1000), target, family="normal", objective=objective
    )
    assert result["multiplier"] == pytest.approx(2.0, abs=0.02)
    assert result["count"] == 1000
    assert not result["at_bound"]
    assert len(result["fit_values_sha256"]) == 64


def test_scale_calibration_reports_bounds_and_invalid_inputs():
    result = fit_scale_multiplier(
        np.zeros(2), np.ones(2), np.array([100.0, 200.0]), family="normal"
    )
    assert result["multiplier"] == 4
    assert result["at_bound"]
    with pytest.raises(ValueError, match="finite"):
        fit_scale_multiplier(np.zeros(2), np.zeros(2), np.ones(2), family="normal")
