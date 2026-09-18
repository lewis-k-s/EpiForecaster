import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.research_calibration import (
    HEADS,
    fit_research_calibrators,
    observed_role_forecasts,
    score_research_calibrators,
)


@pytest.fixture
def manifest():
    return json.loads(Path("configs/research/loss_roles_v1.json").read_text())


@pytest.fixture
def calibration():
    return pd.DataFrame(
        [
            {
                "target": head,
                "region_id": str(i),
                "window_start": 240,
                "horizon": 1,
                "target_date": "2020-12-29",
                "role": "calibration",
                "observed": residual,
                "observed_mask": True,
                "observation_weight": 1.0,
                "distribution_loc": 0.0,
                "distribution_scale": 1.0,
                "distribution_family": "normal",
                "distribution_df": 5.0,
            }
            for head in HEADS
            for i, residual in enumerate(np.linspace(-3, 3, 20))
        ]
    )


def test_calibration_artifact_roundtrip_and_out_of_fit_evaluation(
    calibration, manifest
):
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256="a" * 64
    )
    artifact = json.loads(json.dumps(artifact, allow_nan=False))
    assert artifact["heads"]["cases"]["C1"]["multiplier"] > 1
    selection = calibration.assign(role="selection", target_date="2020-10-29")
    rows, summary = score_research_calibrators(
        selection, artifact, manifest, role="selection", checkpoint_sha256="a" * 64
    )
    assert len(rows) == len(selection) * 3
    assert set(summary["controls"]) == {"C0", "C1", "C2"}
    assert summary["controls"]["C1"]["heads"]["cases"]["coverage_80"] >= 0.8
    assert (
        summary["controls"]["C2"]["mean_target_wis"]
        < summary["controls"]["C0"]["mean_target_wis"]
    )
    c0 = rows.loc[rows.control.eq("C0")]
    assert np.allclose(c0.calibrated_scale, 1.0)
    assert (rows.distribution_loc == 0).all()
    assert (rows.crps >= 0).all() and (rows.wis >= 0).all()
    assert rows.pit.between(0, 1).all()
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match="cannot supply evaluation evidence"):
        score_research_calibrators(
            calibration,
            artifact,
            manifest,
            role="calibration",
            checkpoint_sha256="a" * 64,
        )


def test_student_calibration_scores_remain_finite(calibration, manifest):
    calibration["distribution_family"] = "student_t"
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256="a" * 64
    )
    rows, _ = score_research_calibrators(
        calibration.assign(role="selection", target_date="2020-10-29"),
        artifact,
        manifest,
        role="selection",
        checkpoint_sha256="a" * 64,
    )
    assert np.isfinite(rows[["wis", "crps", "nll", "pit"]].to_numpy()).all()


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "future",
        "window",
        "duplicate",
        "missing_head",
        "nan",
        "weights",
        "family",
    ],
)
def test_role_forecasts_fail_closed(calibration, manifest, fault):
    if fault == "role":
        calibration.loc[0, "role"] = "fit"
    elif fault == "future":
        calibration.loc[0, "target_date"] = "2021-03-01"
    elif fault == "window":
        calibration.loc[0, "target_date"] = (
            "2021-01-30"  # Its 28-day window exceeds calibration.
        )
    elif fault == "duplicate":
        calibration = pd.concat([calibration, calibration.iloc[:1]])
    elif fault == "missing_head":
        calibration = calibration.loc[~calibration.target.eq("cases")]
    elif fault == "nan":
        calibration.loc[0, "distribution_loc"] = np.nan
    elif fault == "weights":
        calibration.loc[0, "observation_weight"] = 0.5
    else:
        calibration.loc[0, "distribution_family"] = "student_t"
    with pytest.raises(ValueError):
        observed_role_forecasts(calibration, manifest, role="calibration")


def test_calibration_rejects_changed_artifacts_and_checkpoint(calibration, manifest):
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256="a" * 64
    )
    selection = calibration.assign(role="selection", target_date="2020-10-29")
    with pytest.raises(ValueError, match="checkpoint or protocol"):
        score_research_calibrators(
            selection, artifact, manifest, role="selection", checkpoint_sha256="b" * 64
        )
    changed = copy.deepcopy(artifact)
    changed["heads"]["cases"]["C1"]["multiplier"] = 100
    with pytest.raises(ValueError, match="integrity"):
        score_research_calibrators(
            selection, changed, manifest, role="selection", checkpoint_sha256="a" * 64
        )
