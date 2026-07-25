from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.analysis.analyze_training_dynamics import (
    _analyze_trace,
    analyze_training_dynamics,
    write_analysis_bundle,
)
from scripts.analysis.query_run_provenance import (
    collect_training_provenance,
    provenance_table,
)
from training.epiforecaster_trainer import EpiForecasterTrainer
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


def test_artifact_contract_import_does_not_load_training_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import training.training_dynamics; "
                "assert 'torch' not in sys.modules; "
                "assert 'training.epiforecaster_trainer' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
    assert json.loads(
        (tmp_path / DYNAMICS_MANIFEST_NAME).read_text()
    ) == manifest
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
                "val_mae": [0.4],
                "learning_rate": [0.001],
                "epoch_times": [3.0],
            },
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
    assert (
        analysis["components"]["gradnorm_mobility_gnn"]["status"]
        == "not_applicable"
    )


def test_analyzer_uses_exact_rows_for_loss_gradient_association(tmp_path) -> None:
    writer = _writer(tmp_path)
    for index in range(20):
        writer.append(
            step=index * 10,
            epoch=index // 5,
            values={
                "loss_train_step": float(20 - index),
                "gradnorm_total_preclip": float(index + 1),
                "gradient_clip_threshold": 100.0,
                "gradnorm_backbone_encoder": float(index + 1) / 2,
                "time_batch_s": 0.2 + index / 100,
            },
        )
    writer.finalize("complete")

    analysis = analyze_training_dynamics(tmp_path / DYNAMICS_MANIFEST_NAME)
    assert analysis["analysis_status"] == "analyzed"
    assert analysis["loss"]["count"] == 20
    assert analysis["loss_gradient_association"]["count"] == 20
    assert (
        analysis["components"]["gradnorm_backbone_encoder"]["status"]
        == "analyzed"
    )


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


def test_analyzer_reports_partial_lifecycle_and_exact_spike_steps(tmp_path) -> None:
    writer = _writer(tmp_path)
    for index in range(21):
        writer.append(
            step=index * 10,
            epoch=0,
            values={"loss_train_step": 100.0 if index == 7 else 1.0},
        )

    analysis = analyze_training_dynamics(tmp_path)

    assert analysis["identity"]["manifest_status"] == "running"
    assert analysis["loss"]["robust_spike_steps"] == [70]
    assert analysis["findings"][0]["code"] == "partial_lifecycle"
    assert analysis["findings"][0]["evidence"]["manifest_status"] == "running"
    assert analysis["findings"][1]["code"] == "robust_loss_spikes"
    assert analysis["findings"][1]["evidence"]["steps"] == [70]
    assert all("hypothesis" not in finding for finding in analysis["findings"])


def test_nonfinite_helper_persists_triggering_observation(tmp_path) -> None:
    writer = _writer(tmp_path)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer = SimpleNamespace(
        training_dynamics_writer=writer,
        config=SimpleNamespace(
            training=SimpleNamespace(gradient_clip_value=5.0)
        ),
        optimizer=torch.optim.SGD([parameter], lr=0.1),
        current_epoch=2,
        global_step=10,
        curriculum_sampler=None,
    )

    EpiForecasterTrainer._append_nonfinite_dynamics_record(
        trainer,
        loss=torch.tensor(float("nan")),
        grad_norm=torch.tensor(2.0),
        batch_start_time=time.time(),
        data_time_s=0.1,
        model_step_time_s=0.2,
        component_gradnorm_log_data={},
        gradnorm_step_log_data={},
        gradient_snapshot_log_data={},
        model_diagnostics_log_data={},
    )

    row = json.loads((tmp_path / DYNAMICS_ARTIFACT_NAME).read_text())
    assert row["loss_train_step"] is None
    assert row["nonfinite_fields"] == ["loss_train_step"]
    assert row["step"] == 10


def test_trainer_run_finalizes_complete_dynamics() -> None:
    finalized: list[tuple[str, dict[str, object]]] = []
    trainer = SimpleNamespace(
        training_dynamics_writer=SimpleNamespace(
            finalize=lambda status, **kwargs: finalized.append((status, kwargs))
        ),
        nan_loss_triggered=False,
        _run_training=lambda: {"ok": True},
    )

    result = EpiForecasterTrainer.run(trainer)

    assert result == {"ok": True}
    assert finalized[0][0] == "complete"
    assert finalized[0][1]["summary"]["training_history"] == {}


def test_trainer_run_finalizes_pruned_and_failed_dynamics() -> None:
    class TrialPruned(Exception):
        pass

    for error, expected in [
        (TrialPruned("pruned"), "pruned"),
        (RuntimeError("failed"), "failed"),
    ]:
        finalized: list[tuple[str, dict[str, object]]] = []

        def fail() -> None:
            raise error

        trainer = SimpleNamespace(
            training_dynamics_writer=SimpleNamespace(
                finalize=lambda status, **kwargs: finalized.append((status, kwargs))
            ),
            _run_training=fail,
        )
        with pytest.raises(type(error)):
            EpiForecasterTrainer.run(trainer)
        assert finalized[0][0] == expected
        assert finalized[0][1]["error_type"] == type(error).__name__
        assert finalized[0][1]["summary"]["training_history"] == {}


def test_analyzer_exposes_curve_clipping_and_component_series(tmp_path) -> None:
    writer = _writer(tmp_path)
    for index in range(25):
        loss = 5.0 if index == 12 else 1.0 - index / 100.0
        total = 4.0 if index in {12, 13} else 0.5
        writer.append(
            step=(index + 1) * 10,
            epoch=index // 5,
            values={
                "loss_train_step": loss,
                "gradnorm_total_preclip": total,
                "gradient_clip_threshold": 1.0,
                "gradient_clip_factor": min(1.0, 1.0 / total),
                "gradnorm_backbone_encoder": total / 2.0,
                "time_batch_s": 0.1,
                "time_dataload_s": 0.02,
                "time_model_step_s": 0.08,
            },
        )
    writer.finalize("complete")

    analysis = analyze_training_dynamics(tmp_path)

    assert len(analysis["loss"]["series"]) == 25
    assert analysis["loss"]["best_sampled_step"] == 250
    assert analysis["loss"]["epoch_summaries"][0]["count"] == 5
    assert analysis["clipping"]["clipped_steps"] == [130, 140]
    assert analysis["clipping"]["strongest_clip_factor"] == pytest.approx(0.25)
    backbone = analysis["components"]["gradnorm_backbone_encoder"]
    assert len(backbone["series"]) == 25
    assert backbone["median_squared_norm_share_pct"] == pytest.approx(25.0)
    assert analysis["evidence_coverage"]["validation_history"] == "not_available"


def test_profiler_trace_analysis_reports_hotspots_and_duration_semantics(
    tmp_path,
) -> None:
    trace_path = tmp_path / "worker.pt.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "ProfilerStep#1",
                        "cat": "user_annotation",
                        "ph": "X",
                        "ts": 0,
                        "dur": 1000,
                    },
                    {
                        "name": "aten::matmul",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 100,
                        "dur": 400,
                    },
                    {
                        "name": "gemm_kernel",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 200,
                        "dur": 300,
                    },
                    {
                        "name": "cudaDeviceSynchronize",
                        "cat": "cuda_runtime",
                        "ph": "X",
                        "ts": 600,
                        "dur": 100,
                    },
                    {
                        "name": "CompiledFxGraph",
                        "cat": "user_annotation",
                        "ph": "X",
                        "ts": 700,
                        "dur": 200,
                    },
                ]
            }
        )
    )

    analysis = _analyze_trace(trace_path)

    assert analysis["trace_duration_ms"] == pytest.approx(1.0)
    assert analysis["top_cpu_operations"][0]["name"] == "aten::matmul"
    assert analysis["top_cuda_kernels"][0]["name"] == "gemm_kernel"
    assert analysis["synchronization"]["count"] == 1
    assert analysis["compilation"]["summed_duration_ms"] == pytest.approx(0.2)
    assert analysis["duration_semantics"] == "inclusive_nested"


def test_analysis_bundle_writes_tables_plots_and_refuses_overwrite(tmp_path) -> None:
    model_root = tmp_path / "model"
    writer = _writer(model_root)
    for index in range(20):
        writer.append(
            step=(index + 1) * 10,
            epoch=index // 5,
            values={
                "loss_train_step": 1.0 - index / 100.0,
                "gradnorm_total_preclip": 0.5,
                "gradient_clip_threshold": 1.0,
                "time_batch_s": 0.1,
            },
        )
    writer.finalize("complete")
    analysis = analyze_training_dynamics(model_root)
    output_dir = tmp_path / "report"

    bundle = write_analysis_bundle(analysis, output_dir)

    assert (output_dir / "analysis.json").is_file()
    assert (output_dir / "provenance.json").is_file()
    assert (output_dir / "bundle_manifest.json").is_file()
    assert not (output_dir / "report.md").exists()
    assert Path(bundle["tables"]["loss_series"]).is_file()
    assert Path(bundle["tables"]["timing_summary"]).is_file()
    assert Path(bundle["figures"]["loss_curve"]).is_file()
    manifest = json.loads((output_dir / "bundle_manifest.json").read_text())
    assert manifest["tables"]["loss_series"] == "tables/loss_series.csv"
    assert manifest["figures"]["loss_curve"] == "figures/loss_curve.png"
    assert manifest["provenance"] == "provenance.json"
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        write_analysis_bundle(analysis, output_dir)


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
        json.dumps(
            {
                "targets": {
                    "data/processed/data.zarr.dvc": {"md5": "data.dir"}
                }
            }
        )
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
    assert provenance["training_evidence"]["retention_class"] == (
        "run_scoped_hashed"
    )
    table = provenance_table(provenance)
    assert any(
        row["category"] == "dvc_input"
        and row["identity"] == "data.dir"
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
    assert analysis["provenance"]["training_evidence"]["dvc_tracking"] == (
        "recorded"
    )
    output_dir = tmp_path / "portable-report"
    bundle = write_analysis_bundle(analysis, output_dir)
    assert Path(bundle["tables"]["provenance"]).is_file()


def test_analyzer_reports_adaptive_gradnorm_controller_telemetry(tmp_path) -> None:
    writer = _writer(tmp_path)
    for index in range(20):
        values = {
            "loss_train_step": 1.0,
            "gradnorm_total_preclip": 1.0,
            "gradient_clip_threshold": 5.0,
        }
        for task_index, task in enumerate(("ww", "hosp", "cases", "deaths")):
            values[f"gradnorm_w_{task}"] = 0.2 + task_index / 100
            values[f"gradnorm_active_{task}"] = 1.0
            values[f"gradnorm_task_loss_{task}"] = 0.5
            values[f"gradnorm_task_norm_{task}"] = 0.1
            values[f"gradnorm_target_norm_{task}"] = 0.11
            values[f"gradnorm_rate_{task}"] = 1.0
        values["gradnorm_sidecar_ran"] = float(index % 2 == 0)
        values["gradnorm_L_grad"] = 0.04
        if index % 5 == 0:
            values["grad_snapshot_global_norm"] = 0.5
            values["grad_snapshot_non_finite_layers"] = 0
            values["grad_snapshot_vanishing_layers"] = 2
            values["grad_snapshot_exploding_layers"] = 0
            values["grad_snapshot_head_ww_unexpected_zero"] = float(index == 10)
        writer.append(
            step=(index + 1) * 10,
            epoch=index // 5,
            values=values,
        )
    writer.finalize("complete")

    analysis = analyze_training_dynamics(tmp_path)

    controller = analysis["adaptive_gradnorm"]
    assert controller["status"] == "analyzed"
    assert controller["sidecar_update_count"] == 10
    assert controller["tasks"]["ww"]["median_weight"] == pytest.approx(0.2)
    assert analysis["evidence_coverage"]["adaptive_gradnorm_controller"] == "analyzed"
    assert analysis["gradient_snapshots"]["count"] == 4
    assert analysis["gradient_snapshots"]["unexpected_zero_steps"]["ww"] == [110]
