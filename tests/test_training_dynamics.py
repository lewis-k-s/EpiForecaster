from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from scripts.analysis.analyze_training_dynamics import (
    analyze_training_dynamics,
    write_analysis_bundle,
)
from scripts.analysis.query_run_provenance import (
    collect_training_provenance,
    provenance_table,
)
from training.training_dynamics import (
    DYNAMICS_ARTIFACT_NAME,
    DYNAMICS_MANIFEST_NAME,
    TrainingDynamicsWriter,
    load_training_dynamics_identity,
    sha256_file,
)


def _writer(tmp_path, *, resume: bool = False) -> TrainingDynamicsWriter:
    return TrainingDynamicsWriter(
        tmp_path,
        progress_log_frequency=10,
        grad_norm_log_frequency=20,
        mobility_enabled=True,
        resume=resume,
    )


def test_writer_normalizes_nonfinite_values_and_finalizes(tmp_path) -> None:
    writer = _writer(tmp_path)
    row = writer.append(
        step=10,
        epoch=1,
        values={
            "loss_train_step": 1.5,
            "gradnorm_total_preclip": math.inf,
        },
    )
    manifest = writer.finalize("complete")

    assert row["gradnorm_total_preclip"] is None
    assert row["nonfinite_fields"] == ["gradnorm_total_preclip"]
    assert manifest["row_count"] == 1
    assert manifest["first_step"] == 10
    assert manifest["last_step"] == 10
    assert manifest["status"] == "complete"
    artifact = tmp_path / DYNAMICS_ARTIFACT_NAME
    assert manifest["artifact_sha256"] == sha256_file(artifact)
    assert json.loads((tmp_path / DYNAMICS_MANIFEST_NAME).read_text()) == manifest
    identity = load_training_dynamics_identity(tmp_path)
    assert identity["status"] == "complete"
    assert identity["row_count"] == 1


def test_writer_materializes_and_hashes_empty_terminal_artifact(tmp_path) -> None:
    writer = _writer(tmp_path)

    manifest = writer.finalize("complete")

    artifact = tmp_path / DYNAMICS_ARTIFACT_NAME
    assert artifact.is_file()
    assert artifact.read_bytes() == b""
    assert manifest["row_count"] == 0
    assert manifest["artifact_sha256"] == sha256_file(artifact)
    assert load_training_dynamics_identity(tmp_path)["row_count"] == 0
    analysis = analyze_training_dynamics(tmp_path)
    assert analysis["analysis_status"] == "insufficient_data"


def test_terminal_manifest_persists_json_safe_epoch_summary(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(step=10, epoch=0, values={"loss_train_step": 1.0})

    manifest = writer.finalize(
        "complete",
        summary={
            "training_history": {
                "train_loss": [1.0],
                "val_loss": [1.2],
                "val_selection_score": [0.4],
                "val_selection_metric": ["mean_target_crps"],
                "learning_rate": [0.001],
                "epoch_times": [3.0],
            },
            "best_val_selection_score": 0.4,
            "best_val_selection_metric": "mean_target_crps",
            "best_val_joint_loss": 1.2,
            "unavailable": math.inf,
        },
    )

    assert manifest["summary"]["best_val_joint_loss"] == pytest.approx(1.2)
    assert manifest["summary"]["unavailable"] is None
    analysis = analyze_training_dynamics(tmp_path)
    assert analysis["validation"]["status"] == "analyzed"
    assert analysis["validation"]["best_epoch"] == 0
    assert analysis["validation"]["best_val_joint_loss"] == pytest.approx(1.2)
    assert analysis["validation"]["series"][0]["val_selection_score"] == pytest.approx(
        0.4
    )
    assert (
        analysis["validation"]["series"][0]["val_selection_metric"]
        == "mean_target_crps"
    )


def test_writer_refuses_overwrite_and_out_of_order_resume(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(step=10, epoch=0, values={"loss_train_step": 1.0})

    with pytest.raises(FileExistsError):
        _writer(tmp_path)

    resumed = _writer(tmp_path, resume=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        resumed.append(step=10, epoch=0, values={"loss_train_step": 0.9})
    resumed.append(step=20, epoch=0, values={"loss_train_step": 0.8})
    manifest = resumed.finalize("pruned")
    assert manifest["status"] == "pruned"
    assert manifest["row_count"] == 2


def test_writer_refuses_resume_identity_mismatch(tmp_path) -> None:
    _writer(tmp_path)

    with pytest.raises(ValueError, match="identity does not match"):
        TrainingDynamicsWriter(
            tmp_path,
            progress_log_frequency=11,
            grad_norm_log_frequency=20,
            mobility_enabled=True,
            resume=True,
        )


def test_writer_refuses_terminal_resume(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(step=10, epoch=0, values={"loss_train_step": 1.0})
    writer.finalize("failed", error_type="RuntimeError", error_message="boom")

    with pytest.raises(ValueError, match="terminal"):
        _writer(tmp_path, resume=True)


def test_analyzer_reports_insufficient_data_and_disabled_mobility(tmp_path) -> None:
    writer = TrainingDynamicsWriter(
        tmp_path,
        progress_log_frequency=10,
        grad_norm_log_frequency=10,
        mobility_enabled=False,
    )
    for step in (10, 20, 30):
        writer.append(
            step=step,
            epoch=0,
            values={
                "loss_train_step": 1.0,
                "gradnorm_total_preclip": 2.0,
                "gradient_clip_threshold": 1.0,
            },
        )
    writer.finalize("complete")

    analysis = analyze_training_dynamics(tmp_path)
    assert analysis["analysis_status"] == "insufficient_data"
    assert analysis["clipping"]["clipped_fraction"] == 1.0
    assert analysis["components"]["gradnorm_mobility_gnn"]["status"] == "not_applicable"


def test_analyzer_rejects_non_monotonic_steps(tmp_path) -> None:
    artifact = tmp_path / DYNAMICS_ARTIFACT_NAME
    artifact.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "step": step,
                    "epoch": 0,
                    "nonfinite_fields": [],
                }
            )
            for step in (2, 1)
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_training_dynamics(artifact)


def test_analyzer_rejects_tampered_terminal_artifact(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(step=10, epoch=0, values={"loss_train_step": 1.0})
    writer.finalize("complete")
    (tmp_path / DYNAMICS_ARTIFACT_NAME).write_text('{"tampered": true}\n')

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analyze_training_dynamics(tmp_path)


def test_provenance_query_reconciles_dvc_and_survives_portable_copy(
    tmp_path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    model_root = run_root / "outputs/training/single/model"
    writer = _writer(model_root)
    writer.append(step=10, epoch=0, values={"loss_train_step": 1.0})
    writer.finalize("complete")
    (model_root / "training_manifest.json").write_text(
        json.dumps({"schema_version": 1}) + "\n"
    )

    manifest_path = run_root / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "run": {"root": str(run_root)},
                "identity": {
                    "git_sha": "a" * 40,
                    "source_snapshot_sha256": "b" * 64,
                    "snakemake_config_sha256": "c" * 64,
                    "dispatch_request_sha256": "d" * 64,
                    "container_digest": "sha256:" + "e" * 64,
                    "workflow_tooling_key": "tooling-key",
                    "workflow_kind": "train",
                    "workflow_target": "train",
                },
                "source": {"dirty": True},
                "workflow": {"execution_adapter": "interactive"},
                "dvc": {
                    "targets": {
                        "data/processed/data.zarr.dvc": {
                            "md5": "data.dir",
                            "output": "data/processed/data.zarr",
                            "role": "materialized",
                            "size": 123,
                            "nfiles": 2,
                        },
                        "data/raw/source.zarr.dvc": {
                            "md5": "source.dir",
                            "output": "data/raw/source.zarr",
                            "role": "provenance",
                        },
                    }
                },
            }
        )
    )
    state = run_root / ".state"
    state.mkdir()
    (state / "data_manifest.json").write_text(
        json.dumps({"targets": {"data/processed/data.zarr.dvc": {"md5": "data.dir"}}})
    )
    dependencies = state / "dependencies"
    dependencies.mkdir()
    (dependencies / "training.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "fingerprint": "f" * 64,
            }
        )
    )

    provenance = collect_training_provenance(
        model_root,
        run_manifest_path=manifest_path,
    )

    assert provenance["status"] == "resolved"
    assert provenance["run"]["run_id"] == "run-1"
    assert provenance["dvc"]["reconciliation"] == "matched"
    assert provenance["dvc"]["targets"][0]["materialization"] == "confirmed"
    assert provenance["dvc"]["targets"][1]["materialization"] == "not_applicable"
    assert provenance["training_evidence"]["retention_class"] == ("run_scoped_hashed")
    table = provenance_table(provenance)
    assert any(
        row["category"] == "dvc_input" and row["identity"] == "data.dir"
        for row in table
    )

    portable_path = model_root / "training_provenance.json"
    portable_path.write_text(json.dumps(provenance) + "\n")
    portable = collect_training_provenance(model_root)
    assert portable == provenance
    retained = run_root / "evidence/training/single/model"
    retained.mkdir(parents=True)
    (retained / "artifacts.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": "evidence.dir", "path": "artifacts"}]})
    )
    (retained / "retention.json").write_text(
        json.dumps(
            {
                "dvc_pointer": "artifacts.dvc",
                "dvc_md5": "evidence.dir",
                "artifacts": {
                    artifact["name"]: {
                        "sha256": artifact["sha256"],
                        "bytes": artifact["bytes"],
                    }
                    for artifact in provenance["training_evidence"]["artifacts"]
                },
            }
        )
    )
    dvc_retained = collect_training_provenance(
        model_root,
        run_manifest_path=manifest_path,
    )
    assert dvc_retained["training_evidence"]["retention_class"] == "dvc_retained"
    assert dvc_retained["training_evidence"]["dvc_md5"] == "evidence.dir"
    assert any(
        row["category"] == "training_evidence_bundle"
        for row in provenance_table(dvc_retained)
    )
    analysis = analyze_training_dynamics(model_root)
    assert analysis["provenance"]["run"]["run_id"] == "run-1"
    assert analysis["provenance"]["training_evidence"]["dvc_tracking"] == ("recorded")
    output_dir = tmp_path / "portable-report"
    bundle = write_analysis_bundle(analysis, output_dir)
    assert Path(bundle["tables"]["provenance"]).is_file()
