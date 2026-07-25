from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.analysis.compare_synthetic_pretraining import generate_comparison


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, document: dict) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    *,
    stage: str,
    mae: float,
    residual_fraction: float,
) -> dict[str, str]:
    run_dir = root / stage
    run_dir.mkdir()
    checkpoint = f"/workspace/{stage}/best_model.pt"
    evaluation_run_manifest = run_dir / "manifest.yaml"
    evaluation_data_manifest = run_dir / "data_manifest.json"
    summary = run_dir / "test_summary.json"
    dynamics = run_dir / "training_dynamics.jsonl"
    dynamics_manifest = run_dir / "training_dynamics_manifest.json"
    training_manifest = run_dir / "training_manifest.json"
    provenance = run_dir / "training_provenance.json"
    evaluation_manifest = run_dir / "evaluation_manifest.json"

    dataset_pointer = "data/processed/real_with_id.zarr.dvc"
    dataset_output = "data/processed/real_with_id.zarr"
    dvc_md5 = "c" * 32 + ".dir"
    _write_yaml(
        evaluation_run_manifest,
        {
            "schema_version": 1,
            "run_id": f"run-{stage}",
            "identity": {
                "git_sha": "a" * 40,
                "source_snapshot_sha256": "b" * 64,
                "dvc_hashes": {dataset_pointer: dvc_md5},
            },
            "dvc": {
                "targets": {
                    dataset_pointer: {
                        "md5": dvc_md5,
                        "output": dataset_output,
                        "role": "materialized",
                    }
                }
            },
        },
    )
    _write_json(
        evaluation_data_manifest,
        {
            "schema_version": 1,
            "targets": {
                dataset_pointer: {
                    "md5": dvc_md5,
                    "output": dataset_output,
                }
            },
        },
    )
    _write_json(
        summary,
        {
            "schema_version": 1,
            "split": "test",
            "checkpoint": checkpoint,
            "eval_loss": mae,
            "metrics": {
                "mae_hosp_log1p_per_100k": mae,
                "crps_cases_log1p_per_100k": mae / 2.0,
                "coverage_80_hosp_per_h": [0.5, 0.6],
                "pathway_hosp_residual_abs_fraction": residual_fraction,
                "pathway_hosp_reconstruction_max_abs_error": 0.0,
                "latent_pearson_i": 0.25,
                "latent_mae_i_fraction": 0.1,
                "latent_mass_conservation_mae": 1.0e-8,
                "loss_latent_i": 0.2,
            },
        },
    )
    dynamics.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 0,
                        "epoch": 0,
                        "pathgrad_hosp_residual_grad_activation_fraction": (
                            residual_fraction
                        ),
                        "pathgrad_sidecar_ran": 1.0,
                    }
                ),
                json.dumps({"step": 1, "epoch": 0, "loss_train_step": mae}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        dynamics_manifest,
        {
            "schema_version": 1,
            "artifact": dynamics.name,
            "artifact_sha256": _sha256(dynamics),
            "status": "complete",
            "row_count": 2,
            "first_step": 0,
            "last_step": 1,
        },
    )
    _write_json(
        training_manifest,
        {
            "schema_version": 1,
            "checkpoint": checkpoint,
            "checkpoint_sha256": "d" * 64,
            "training_dynamics": {
                "artifact": str(dynamics),
                "artifact_sha256": _sha256(dynamics),
                "manifest": str(dynamics_manifest),
                "manifest_sha256": _sha256(dynamics_manifest),
                "row_count": 2,
                "status": "complete",
            },
        },
    )
    _write_json(
        provenance,
        {
            "schema_version": 1,
            "status": "resolved",
            "run": {
                "run_id": f"run-{stage}",
                "git_sha": "a" * 40,
                "source_snapshot_sha256": "b" * 64,
            },
            "dvc": {
                "status": "available",
                "reconciliation": "matched",
                "targets": [
                    {
                        "pointer": dataset_pointer,
                        "dvc_md5": dvc_md5,
                        "role": "materialized",
                        "materialization": "confirmed",
                    }
                ],
            },
            "training_evidence": {
                "artifacts": [
                    {
                        "name": dynamics.name,
                        "sha256": _sha256(dynamics),
                        "bytes": dynamics.stat().st_size,
                    },
                    {
                        "name": dynamics_manifest.name,
                        "sha256": _sha256(dynamics_manifest),
                        "bytes": dynamics_manifest.stat().st_size,
                    },
                    {
                        "name": training_manifest.name,
                        "sha256": _sha256(training_manifest),
                        "bytes": training_manifest.stat().st_size,
                    },
                ]
            },
        },
    )
    _write_json(
        evaluation_manifest,
        {
            "schema_version": 1,
            "checkpoint": checkpoint,
            "checkpoint_sha256": "d" * 64,
            "artifacts": [
                {
                    "path": summary.name,
                    "sha256": _sha256(summary),
                    "required": True,
                }
            ],
        },
    )
    return {
        "stage": stage,
        "dataset_role": "real",
        "evaluation_run_manifest": str(evaluation_run_manifest.relative_to(root)),
        "evaluation_data_manifest": str(
            evaluation_data_manifest.relative_to(root)
        ),
        "evaluation_summary": str(summary.relative_to(root)),
        "training_dynamics": str(dynamics.relative_to(root)),
        "training_dynamics_manifest": str(dynamics_manifest.relative_to(root)),
        "training_manifest": str(training_manifest.relative_to(root)),
        "training_provenance": str(provenance.relative_to(root)),
        "evaluation_manifest": str(evaluation_manifest.relative_to(root)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_comparison_emits_provenance_checked_tables(tmp_path: Path) -> None:
    control = _write_run(
        tmp_path,
        stage="control",
        mae=2.0,
        residual_fraction=0.4,
    )
    fine_tune = _write_run(
        tmp_path,
        stage="fine_tune",
        mae=1.5,
        residual_fraction=0.25,
    )
    spec = tmp_path / "comparison.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "runs": [control, fine_tune],
            "comparisons": [
                {
                    "name": "pretraining_effect",
                    "baseline": "control",
                    "candidate": "fine_tune",
                }
            ],
        },
    )

    output_dir = tmp_path / "comparison"
    manifest = generate_comparison(spec, output_dir)

    assert manifest["stages"] == ["control", "fine_tune"]
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {
        "comparison_deltas.csv",
        "evaluation_metrics.csv",
        "latent_metrics.csv",
        "latent_structure_metrics.csv",
        "pathway_gradients.csv",
        "pathway_metrics.csv",
        "run_inventory.csv",
    }

    evaluation_rows = _read_csv(output_dir / "evaluation_metrics.csv")
    assert not any(row["metric"].endswith("_per_h") for row in evaluation_rows)
    hosp_mae = next(
        row
        for row in evaluation_rows
        if row["stage"] == "control"
        and row["target"] == "hospitalizations"
        and row["metric"] == "mae_log1p_per_100k"
    )
    assert float(hosp_mae["value"]) == pytest.approx(2.0)

    latent_rows = _read_csv(output_dir / "latent_metrics.csv")
    assert {
        (row["latent"], row["metric"])
        for row in latent_rows
        if row["stage"] == "control"
    } == {
        ("I", "loss"),
        ("I", "mae_fraction"),
        ("I", "pearson"),
    }
    structural_rows = _read_csv(output_dir / "latent_structure_metrics.csv")
    assert structural_rows[0]["metric"] == "latent_mass_conservation_mae"

    gradient_rows = _read_csv(output_dir / "pathway_gradients.csv")
    assert any(
        row["target"] == "hospitalizations"
        and row["metric"] == "residual_grad_activation_fraction"
        and row["step"] == "0"
        for row in gradient_rows
    )

    delta_rows = _read_csv(output_dir / "comparison_deltas.csv")
    mae_delta = next(
        row
        for row in delta_rows
        if row["table"] == "evaluation"
        and row["target"] == "hospitalizations"
        and row["metric"] == "mae_log1p_per_100k"
    )
    assert float(mae_delta["candidate_minus_baseline"]) == pytest.approx(-0.5)
    assert float(mae_delta["candidate_over_baseline"]) == pytest.approx(0.75)
    residual_delta = next(
        row
        for row in delta_rows
        if row["table"] == "pathway"
        and row["metric"] == "residual_abs_fraction"
    )
    assert float(residual_delta["candidate_minus_baseline"]) == pytest.approx(-0.15)

    inventory = _read_csv(output_dir / "run_inventory.csv")
    assert inventory[0]["checkpoint_sha256"] == "d" * 64
    assert inventory[0]["training_run_id"] == "run-control"
    assert inventory[0]["evaluation_run_id"] == "run-control"
    assert (
        "data/processed/real_with_id.zarr.dvc"
        in inventory[0]["evaluation_dvc_targets_json"]
    )


def test_generate_comparison_rejects_mismatched_dynamics_hash(
    tmp_path: Path,
) -> None:
    control = _write_run(
        tmp_path,
        stage="control",
        mae=2.0,
        residual_fraction=0.4,
    )
    dynamics = tmp_path / control["training_dynamics"]
    dynamics.write_text(
        dynamics.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    spec = tmp_path / "comparison.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "runs": [control],
            "comparisons": [],
        },
    )

    with pytest.raises(ValueError, match="Dynamics manifest hash mismatch"):
        generate_comparison(spec, tmp_path / "comparison")


def test_generate_comparison_rejects_wrong_training_checkpoint(
    tmp_path: Path,
) -> None:
    control = _write_run(
        tmp_path,
        stage="control",
        mae=2.0,
        residual_fraction=0.4,
    )
    training_manifest_path = tmp_path / control["training_manifest"]
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_manifest["checkpoint_sha256"] = "e" * 64
    _write_json(training_manifest_path, training_manifest)
    provenance_path = tmp_path / control["training_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    training_evidence = provenance["training_evidence"]["artifacts"]
    recorded_manifest = next(
        artifact
        for artifact in training_evidence
        if artifact["name"] == training_manifest_path.name
    )
    recorded_manifest["sha256"] = _sha256(training_manifest_path)
    recorded_manifest["bytes"] = training_manifest_path.stat().st_size
    _write_json(provenance_path, provenance)
    spec = tmp_path / "comparison.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "runs": [control],
            "comparisons": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="Training and evaluation checkpoint hashes differ",
    ):
        generate_comparison(spec, tmp_path / "comparison")


def test_generate_comparison_rejects_wrong_dataset_role(tmp_path: Path) -> None:
    control = _write_run(
        tmp_path,
        stage="control",
        mae=2.0,
        residual_fraction=0.4,
    )
    control["dataset_role"] = "synthetic"
    spec = tmp_path / "comparison.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "runs": [control],
            "comparisons": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="did not stage data/processed/synthetic_full_holt_damped.zarr.dvc",
    ):
        generate_comparison(spec, tmp_path / "comparison")
