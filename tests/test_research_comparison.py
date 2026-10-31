import numpy as np
import pandas as pd
import pytest

from evaluation.research_calibration import HEADS
from evaluation.research_comparison import compare_scored_forecasts


def scores(days):
    return pd.DataFrame(
        [
            {
                "target": head,
                "region_id": "a",
                "window_start": i,
                "horizon": 1,
                "target_date": date,
                "role": "selection",
                "control": "C0",
                "observed": 1.0,
                "observed_mask": True,
                "wis": 2.0,
                "absolute_error": 1.0,
                "covered_80": True,
                "covered_95": True,
            }
            for i, date in enumerate(pd.date_range("2020-01-01", periods=days))
            for head in HEADS
        ]
    )


def test_short_role_is_point_evidence_only():
    baseline = scores(33)
    candidate = baseline.assign(wis=1.0)
    result = compare_scored_forecasts(baseline, candidate)
    assert result["relative_wis_improvement"] == 0.5
    assert result["wis_difference_candidate_minus_baseline"] == -1.0
    assert result["uncertainty"]["difference_interval_95"] is None
    assert result["uncertainty"]["unsupported_reason"] == "insufficient_temporal_blocks"
    assert result["promotion_status"] == "not_assessed_screening_evidence_only"


def test_temporal_resampling_keeps_all_heads_together_and_is_seeded():
    baseline = scores(112)
    candidate = baseline.copy()
    candidate["wis"] -= np.repeat(np.repeat([0.2, 0.4, 0.6, 0.8], 28), 4)
    result = compare_scored_forecasts(baseline, candidate)
    again = compare_scored_forecasts(baseline.sample(frac=1, random_state=1), candidate)
    assert result == again
    interval = result["uncertainty"]["difference_interval_95"]
    assert interval[0] < -0.5 < interval[1] < 0
    assert result["uncertainty"]["blocks"] == 4


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "labels", "role", "control", "finite"]
)
def test_comparison_rejects_invalid_pairing(fault):
    baseline = scores(5)
    candidate = baseline.copy()
    if fault == "missing":
        candidate = candidate.iloc[1:]
    elif fault == "duplicate":
        candidate = pd.concat([candidate, candidate.iloc[:1]])
    elif fault == "labels":
        candidate.loc[0, "observed"] = 2.0
    elif fault == "role":
        candidate["role"] = "calibration"
    elif fault == "control":
        candidate.loc[0, "control"] = "C2"
    else:
        candidate.loc[0, "wis"] = np.nan
    with pytest.raises(ValueError):
        compare_scored_forecasts(baseline, candidate)
