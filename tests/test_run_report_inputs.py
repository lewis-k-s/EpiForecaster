"""Small readiness and operational-statistics contracts for single-run reports."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.analysis.render_run_report import sha256
from scripts.analysis.run_report_inputs import epoch_tables, gpu_tables, inspect_run


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "run"
    model = run / "outputs/training/experiment/model"
    evaluation = run / "outputs/evaluation"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"id": "example"},
                "training": {"experiment_name": "experiment", "model_id": "model"},
            }
        )
    )
    (run / "manifest.yaml").write_text("schema_version: 1\n")
    write_json(model / "config.json", {"training": {"epochs": 10}})
    write_json(model / "training_manifest.json", {"checkpoint_sha256": "checkpoint"})
    samples = model / "training_dynamics.jsonl"
    samples.write_text('{"step":1,"train_loss":0.5}\n')
    write_json(
        model / "training_dynamics_manifest.json",
        {
            "status": "complete",
            "artifact": samples.name,
            "artifact_sha256": sha256(samples),
            "summary": {"train_epoch_trace": [{"epoch": 1}], "global_step": 1},
        },
    )
    for split in ("val", "test"):
        write_json(evaluation / f"{split}_summary.json", {"split": split})
    write_json(
        evaluation / "evaluation_manifest.json",
        {
            "checkpoint_sha256": "checkpoint",
            "artifacts": [
                {"path": path.name, "sha256": sha256(path), "required": True}
                for path in sorted(evaluation.glob("*_summary.json"))
            ],
        },
    )
    return run, model, evaluation


def test_completed_early_stop_keeps_budget_counts_without_error(completed_run):
    run, _, _ = completed_run
    evidence = inspect_run(run)
    assert evidence["errors"] == []
    assert evidence["facts"]["completed_train_epochs"] == 1
    assert evidence["facts"]["expected_epochs"] == 10
    assert evidence["facts"]["epoch_budget_reached"] is False


def test_missing_and_changed_inputs_have_recorded_error_codes(completed_run):
    run, model, evaluation = completed_run
    (evaluation / "val_summary.json").unlink()
    (evaluation / "test_summary.json").write_text('{"changed": true}')
    (model / "training_dynamics.jsonl").write_text("changed")
    write_json(model / "training_manifest.json", {"checkpoint_sha256": "other"})
    evidence = inspect_run(run)
    assert {item["code"] for item in evidence["errors"]} == {
        "MISSING_ARTIFACT",
        "HASH_MISMATCH",
        "CHECKPOINT_MISMATCH",
    }


def test_recorded_incomplete_training_is_a_status_error(completed_run):
    run, model, _ = completed_run
    path = model / "training_dynamics_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "failed"
    write_json(path, manifest)
    errors = inspect_run(run)["errors"]
    assert errors[0]["code"] == "TRAINING_NOT_COMPLETE"
    assert errors[0]["recorded_value"] == "failed"


def test_gpu_statistics_keep_missing_values_and_device_identity():
    frame = pd.DataFrame(
        {
            "gpu_uuid": ["GPU-1"] * 5 + ["GPU-2"],
            "gpu_utilization_percent": [0, "N/A", 10, 30, 5, "N/A"],
            **{
                key: [1] * 6
                for key in (
                    "memory_controller_utilization_percent",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_watts",
                    "temperature_celsius",
                )
            },
        }
    )
    summary, bands = gpu_tables(frame, {"execution_attempt_id": "attempt"})
    activity = summary[summary.field == "gpu_utilization_percent"].set_index("gpu_uuid")
    assert activity.loc["GPU-1", "sample_mean"] == pytest.approx(11.25)
    assert activity.loc["GPU-1", "missing_samples"] == 1
    assert activity.loc["GPU-1", "valid_samples"] == 4
    assert pd.isna(activity.loc["GPU-2", "sample_mean"])
    assert bands[bands.gpu_uuid == "GPU-1"].sample_percent.tolist() == [25] * 4
    assert bands[bands.gpu_uuid == "GPU-2"].sample_percent.isna().all()
    assert set(summary.execution_attempt_id) == {"attempt"}


def test_epoch_tables_use_explicit_objective_and_evaluation_names():
    traces = {
        "train_epoch_trace": [{"epoch": 1, "train_objective": 2}],
        "validation_epoch_trace": [{"epoch": 1, "val_wis": 3, "selection_score": 4}],
        "test_epoch_trace": [],
    }
    frames = epoch_tables(traces)
    assert frames["train"].iloc[0].train_objective == 2
    assert frames["val"].iloc[0].val_wis == 3
    assert frames["val"].iloc[0].selection_score == 4
    assert frames["test"].empty
    historical = epoch_tables(
        {
            "train_epoch_trace": [{"epoch": 1, "joint_loss": 2}],
            "test_epoch_trace": [],
            "validation_epoch_trace": [
                {
                    "epoch": 1,
                    "joint_loss": 3,
                    "selection_score": 4,
                    "selection_metric": "mean_target_wis",
                }
            ],
        }
    )
    assert historical["train"].iloc[0].train_objective == 2
    assert historical["val"].iloc[0].val_wis == 4
    assert "joint_loss" not in historical["val"].columns
