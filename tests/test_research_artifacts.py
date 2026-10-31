import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.research_artifacts import read_research_evidence
from evaluation.research_calibration import (
    HEADS,
    fit_research_calibrators,
    score_research_calibrators,
)
from evaluation.research_events import score_research_events
from scripts.workflow.run_config import RESEARCH_OUTPUT_FILES


ROLES = Path("tests/fixtures/research_roles.json")
CHECKPOINT_HASH = "a" * 64


@pytest.fixture
def evidence(tmp_path):
    manifest = json.loads(ROLES.read_text())
    calibration = pd.DataFrame(
        [
            {
                "role": "calibration",
                "target": head,
                "region_id": "a",
                "window_start": 243,
                "horizon": 1,
                "target_date": "2020-12-29",
                "observed": 1.0,
                "observed_mask": True,
                "observation_weight": 1.0,
                "distribution_loc": 0.0,
                "distribution_scale": 1.0,
                "distribution_family": "normal",
                "distribution_df": 5.0,
            }
            for head in HEADS
        ]
    )
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256=CHECKPOINT_HASH
    )
    scores, summary = score_research_calibrators(
        calibration.assign(role="selection", target_date="2020-10-29"),
        artifact,
        manifest,
        role="selection",
        checkpoint_sha256=CHECKPOINT_HASH,
    )
    fit = pd.DataFrame(
        {
            "role": ["fit"],
            "target": ["cases"],
            "source_id": ["a"],
            "target_date": ["2020-09-01"],
            "observed": [0.0],
            "observed_mask": [True],
        }
    )
    events, event_summary = score_research_events(
        fit, calibration, scores, artifact, manifest
    )
    required = {
        "val_summary.json",
        "val_node_metrics.csv",
        "val_node_metrics_per_head.csv",
        "val_granular.parquet",
        *RESEARCH_OUTPUT_FILES,
    }
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raw evaluation fixture")
    for name, frame in [
        ("research_scored.parquet", scores),
        ("research_events.parquet", events),
        ("research_fit_cases.parquet", fit),
    ]:
        frame.to_parquet(tmp_path / name)
    for name, content in [
        ("research_calibration.json", artifact),
        ("research_summary.json", summary),
        ("research_events.json", event_summary),
    ]:
        (tmp_path / name).write_text(json.dumps(content, allow_nan=False))
    record = {
        "schema_version": 1,
        "research_outputs": True,
        "split": "val",
        "granular": True,
        "data_roles": {"val": "selection", "test": "calibration"},
        "role_manifest_sha256": hashlib.sha256(ROLES.read_bytes()).hexdigest(),
        "checkpoint_sha256": CHECKPOINT_HASH,
        "artifacts": [
            {
                "path": name,
                "required": True,
                "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            }
            for name in sorted(required)
        ],
    }
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(record))
    (tmp_path / ".done").write_text(json.dumps({"checkpoint_sha256": CHECKPOINT_HASH}))
    return tmp_path


def read(directory):
    return read_research_evidence(
        directory, expected_checkpoint_sha256=CHECKPOINT_HASH, role_manifest_path=ROLES
    )


def test_complete_research_evidence_roundtrip(evidence):
    result = read(evidence)
    assert len(result.scores) == 12
    assert len(result.events) == 6
    assert result.checkpoint_sha256 == CHECKPOINT_HASH
    assert len(result.evaluation_manifest_sha256) == 64
    assert result.summary["role"] == "selection"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_marker",
        "changed_file",
        "wrong_checkpoint",
        "missing_inventory",
        "traversal",
        "duplicate",
    ],
)
def test_research_reader_rejects_incomplete_or_changed_evidence(evidence, fault):
    path = evidence / "evaluation_manifest.json"
    record = json.loads(path.read_text())
    if fault == "missing_marker":
        (evidence / ".done").unlink()
    elif fault == "changed_file":
        (evidence / "research_summary.json").write_text("{}")
    elif fault == "wrong_checkpoint":
        record["checkpoint_sha256"] = "b" * 64
    elif fault == "missing_inventory":
        record["artifacts"].pop()
    elif fault == "traversal":
        record["artifacts"][0]["path"] = "../outside.json"
    else:
        record["artifacts"].append(record["artifacts"][0])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read(evidence)


def test_reader_checks_inner_calibration_identity_even_if_outer_hash_is_updated(
    evidence,
):
    path = evidence / "research_calibration.json"
    calibration = json.loads(path.read_text())
    calibration["heads"]["cases"]["C1"]["multiplier"] *= 2
    path.write_text(json.dumps(calibration))
    manifest_path = evidence / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["artifacts"]:
        if item["path"] == path.name:
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Calibration artifact identity"):
        read(evidence)


def test_reader_rejects_undeclared_required_artifact(evidence):
    path = evidence / "evaluation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][0]["required"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not declared required"):
        read(evidence)
