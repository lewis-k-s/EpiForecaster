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
    return json.loads(Path("tests/fixtures/research_roles.json").read_text())


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
    calibration["interval_low_95"] = -999.0
    calibration["interval_high_95"] = 999.0
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
    quantile_columns = sorted(c for c in rows if c.startswith("quantile_"))
    assert len(quantile_columns) == 23
    assert (np.diff(rows[quantile_columns].to_numpy(), axis=1) > 0).all()
    assert np.allclose(rows.quantile_500, rows.distribution_loc)
    assert np.allclose(rows.interval_low_95, rows.quantile_025)
    assert np.allclose(rows.interval_high_95, rows.quantile_975)
    assert np.allclose(rows.width_95, rows.interval_high_95 - rows.interval_low_95)
    assert np.array_equal(
        rows.covered_95,
        rows.observed.between(rows.interval_low_95, rows.interval_high_95),
    )
    assert np.allclose(rows.prediction, rows.quantile_500)
    assert np.allclose(rows.abs_error, rows.absolute_error)
    assert rows.forecast_origin_date.eq("2020-10-28").all()
    assert (
        rows.loc[rows.target.eq("wastewater"), "target_units"]
        .eq("log1p_concentration")
        .all()
    )
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


def test_research_event_artifacts_use_fit_thresholds_and_frozen_operating_rules(
    calibration, manifest
):
    from evaluation.research_events import score_research_events

    calibration.loc[calibration.target.eq("cases"), "observed"] = np.linspace(1, 3, 20)
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256="a" * 64
    )
    selection = calibration.assign(role="selection", target_date="2020-10-29")
    scored, _ = score_research_calibrators(
        selection, artifact, manifest, role="selection", checkpoint_sha256="a" * 64
    )
    fit = pd.DataFrame(
        {
            "role": "fit",
            "target": "cases",
            "source_id": "case_municipality",
            "target_date": pd.date_range("2020-09-01", periods=10),
            "observed": np.linspace(0, 0.5, 10),
            "observed_mask": True,
        }
    )
    events, summary = score_research_events(
        fit, calibration, scored, artifact, manifest
    )
    assert len(events) == 20 * 3 * 2
    assert not events.event_decision_fixed_0_5.any()
    assert events.event_decision_operating.all()
    c0 = summary["definitions"]["q90"]["controls"]["C0"]
    assert c0["horizon_bands"]["all"]["fixed_0_5"]["recall"] == 0
    assert c0["horizon_bands"]["all"]["operating"]["recall"] == 1
    assert c0["horizon_bands"]["days_8_14"]["fixed_0_5"]["count"] == 0
    assert summary["hospitalizations"]["status"] == "unsupported"
    json.dumps(summary, allow_nan=False)
    unsupported_rows, low_support = score_research_events(
        fit, calibration, scored, artifact, manifest, min_support=21
    )
    assert unsupported_rows.event_decision_operating.isna().all()
    assert (
        low_support["definitions"]["q90"]["controls"]["C0"]["operating_rule"]["reason"]
        == "insufficient_unique_positive_support"
    )
    fit["target_date"] = "2020-12-29"
    with pytest.raises(ValueError, match="fit role"):
        score_research_events(fit, calibration, scored, artifact, manifest)


def test_pipeline_writes_all_canonical_research_artifacts(
    monkeypatch, tmp_path, calibration, manifest
):
    from evaluation import research_pipeline as pipeline
    from models.configs import EpiForecasterConfig

    config = EpiForecasterConfig.load(
        "configs/train_epifor_smoke.yaml",
        overrides=[
            "training.split_strategy=research",
            "training.research_role_manifest_path=tests/fixtures/research_roles.json",
            "model.forecast_horizon=28",
        ],
    )
    fit = pd.DataFrame(
        {
            "role": "fit",
            "target": "cases",
            "source_id": "a",
            "target_date": pd.date_range("2020-09-01", periods=20),
            "observed": np.linspace(0, 2, 20),
            "observed_mask": True,
        }
    )
    monkeypatch.setattr(pipeline, "load_fit_cases", lambda *args: fit)
    cal_path = tmp_path / "cal.parquet"
    sel_path = tmp_path / "sel.parquet"
    calibration.to_parquet(cal_path)
    calibration.assign(role="selection", target_date="2020-10-29").to_parquet(sel_path)
    output = tmp_path / "outputs"
    paths = pipeline.write_research_outputs(
        config=config,
        manifest=manifest,
        checkpoint_sha256="a" * 64,
        selection_path=sel_path,
        calibration_path=cal_path,
        output_dir=output,
    )
    assert {path.name for path in paths} == set(pipeline.RESEARCH_ARTIFACTS)
    for path in paths:
        if path.suffix == ".json":
            json.loads(path.read_text())
        else:
            assert not pd.read_parquet(path).empty
    scores = pd.read_parquet(output / "research_scored.parquet")
    assert set(scores.role) == {"selection"}
    assert set(scores.control) == {"C0", "C1", "C2"}
    events = json.loads((output / "research_events.json").read_text())
    assert events["hospitalizations"]["status"] == "unsupported"
