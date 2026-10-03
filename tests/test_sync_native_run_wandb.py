"""Tests for post-hoc W&B import evidence validation."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.analysis.sync_native_run_wandb import build_history, read_evidence


def _write_evidence(tmp_path):
    evidence = tmp_path / "evidence"
    model = evidence / "model"
    logs = evidence / "logs"
    model.mkdir(parents=True)
    logs.mkdir()
    dynamics = model / "training_dynamics.jsonl"
    dynamics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "epoch": 0,
                "step": 100,
                "train_loss": 1.0,
                "grad_norm": 2.0,
                "nonfinite_fields": [],
            }
        )
        + "\n"
    )
    dynamics_manifest = model / "training_dynamics_manifest.json"
    dynamics_manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "total_epochs": 1,
                    "global_step": 100,
                    "val_loss_trace": [0.25],
                }
            }
        )
    )
    (model / "training_manifest.json").write_text(
        json.dumps(
            {
                "training_dynamics": {
                    "status": "complete",
                    "artifact_sha256": hashlib.sha256(
                        dynamics.read_bytes()
                    ).hexdigest(),
                    "manifest_sha256": hashlib.sha256(
                        dynamics_manifest.read_bytes()
                    ).hexdigest(),
                    "row_count": 1,
                }
            }
        )
    )
    (model / "training_provenance.json").write_text(
        json.dumps({"run": {"run_id": "immutable-test-run"}})
    )
    (logs / "train_model.log").write_text(
        "native epoch 1/1 mean_train_loss=0.500000 "
        "val_loss=0.250000 selection=mean_target_crps=0.300000\n"
    )
    csv_text = "aggregation,target,horizon,observed_count,crps,wis,coverage_80\n"
    csv_text += "head_all,wastewater,0,3,0.4,0.3,0.7\n"
    for split in ("val", "test"):
        (model / f"{split}_native_aggregate_metrics.csv").write_text(csv_text)
    return evidence


def test_read_evidence_validates_complete_run(tmp_path):
    evidence = _write_evidence(tmp_path)

    data = read_evidence(evidence)

    assert data["run_id"] == "immutable-test-run"
    assert len(data["dynamics"]) == 1
    assert len(data["epochs"]) == 1


def test_read_evidence_rejects_changed_dynamics(tmp_path):
    evidence = _write_evidence(tmp_path)
    with (evidence / "model" / "training_dynamics.jsonl").open("a") as stream:
        stream.write("{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_evidence(evidence)


def test_wis_primary_history_preserves_joint_loss_and_metric_identity(tmp_path):
    evidence = _write_evidence(tmp_path)
    model = evidence / "model"
    path = model / "training_dynamics_manifest.json"
    payload = json.loads(path.read_text())
    payload["summary"].update(
        {
            "val_loss_metric": "mean_target_wis",
            "val_loss_trace": [0.3],
            "val_joint_loss_trace": [0.25],
            "validation_epoch_trace": [
                {
                    "epoch": 1,
                    "step": 100,
                    "joint_loss": 0.25,
                    "selection_metric": "mean_target_wis",
                    "selection_score": 0.3,
                    "wis_cases": 0.3,
                    "wis_dispersion_cases": 0.1,
                }
            ],
        }
    )
    path.write_text(json.dumps(payload))
    receipt_path = model / "training_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["training_dynamics"]["manifest_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    log = evidence / "logs/train_model.log"
    log.write_text(
        "native epoch 1/1 mean_train_loss=0.500000 val_loss=0.300000 "
        "selection=mean_target_wis=0.300000 val_joint_loss=0.250000\n"
    )
    history = build_history(read_evidence(evidence))[100]
    assert history["val/val_wis"] == 0.3
    assert history["val/selection_score"] == 0.3
    assert history["val/training_objective_diagnostic"] == 0.25
    assert history["val/wis_dispersion_cases"] == 0.1
    assert "val/mean_target_crps" not in history
    log.write_text(
        log.read_text().replace("val_joint_loss=0.250000", "val_joint_loss=0.600000")
    )
    with pytest.raises(ValueError, match="disagree on joint loss"):
        read_evidence(evidence)


def test_read_evidence_rejects_epoch_disagreement(tmp_path):
    evidence = _write_evidence(tmp_path)
    (evidence / "logs" / "train_model.log").write_text(
        "native epoch 1/1 mean_train_loss=0.500000 "
        "val_loss=0.400000 selection=mean_target_crps=0.300000\n"
    )

    with pytest.raises(ValueError, match="disagree on validation loss"):
        read_evidence(evidence)


def test_epoch_head_traces_validate_and_map_to_wandb_history(tmp_path):
    evidence = _write_evidence(tmp_path)
    model = evidence / "model"
    dynamics_manifest = model / "training_dynamics_manifest.json"
    payload = json.loads(dynamics_manifest.read_text())
    definition = {
        "metric": "wis",
        "aggregation": "equal_head_mean",
        "direction": "minimize",
    }
    base = {
        "epoch": 1,
        "step": 100,
        "selection_metric_definition": definition,
        "selection_score": 0.3,
        "nll_ww": 1.2,
        "crps_ww": 0.4,
        "wis_ww": 0.3,
        "observed_count_ww": 3,
        "censored_nll_ww": 0.8,
        "censored_count_ww": 2,
    }
    payload["summary"] = {
        "total_epochs": 1,
        "global_step": 100,
        "val_wis_trace": [0.3],
        "val_wis": 0.3,
        "selection_score": 0.3,
        "selection_metric_definition": definition,
        "selected_epoch": 1,
        "selected_step": 100,
        "train_epoch_trace": [{"epoch": 1, "step": 100, "train_objective": 0.5}],
        "validation_epoch_trace": [{**base, "val_wis": 0.3}],
        "test_epoch_trace": [
            {
                **{
                    key: value
                    for key, value in base.items()
                    if key not in {"selection_score", "selection_metric_definition"}
                },
                "test_wis": 0.6,
            }
        ],
    }
    dynamics_path = model / "training_dynamics.jsonl"
    dynamics = json.loads(dynamics_path.read_text())
    dynamics["train_objective"] = dynamics.pop("train_loss")
    dynamics_path.write_text(json.dumps(dynamics) + "\n")
    # Canonical epoch records are sufficient; new imports do not score logs.
    (evidence / "logs/train_model.log").unlink()
    dynamics_manifest.write_text(json.dumps(payload))
    manifest = model / "training_manifest.json"
    receipt = json.loads(manifest.read_text())
    receipt["training_dynamics"]["artifact_sha256"] = hashlib.sha256(
        dynamics_path.read_bytes()
    ).hexdigest()
    receipt["training_dynamics"]["manifest_sha256"] = hashlib.sha256(
        dynamics_manifest.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(receipt))

    history = build_history(read_evidence(evidence))

    assert history[100]["val/nll_ww"] == 1.2
    assert history[100]["val/crps_ww"] == 0.4
    assert history[100]["train/train_objective"] == 1.0
    assert history[100]["train/epoch_train_objective"] == 0.5
    assert history[100]["val/val_wis"] == 0.3
    assert history[100]["val/selection_score"] == 0.3
    assert history[100]["test/test_wis"] == 0.6
    assert "test/selection_score" not in history[100]
    assert "val/joint_loss" not in history[100]
    assert history[100]["test/censored_count_ww"] == 2
    assert history[100]["val/censored_nll_ww"] == 0.8

    payload["summary"]["test_epoch_trace"][0]["step"] = 99
    dynamics_manifest.write_text(json.dumps(payload))
    receipt["training_dynamics"]["manifest_sha256"] = hashlib.sha256(
        dynamics_manifest.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Invalid test trace step"):
        read_evidence(evidence)
