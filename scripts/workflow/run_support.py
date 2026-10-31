"""Runtime helpers for the MN5 Snakemake workflow."""

from __future__ import annotations

import argparse
import copy
import csv
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

from scripts.workflow.attempts import complete_attempt, new_attempt, restore_attempt
from scripts.hpo.trial_artifacts import (
    BEST_MODEL_PATH_ATTR,
    BEST_MODEL_SHA256_ATTR,
    TRAINING_DYNAMICS_MANIFEST_ATTR,
    TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR,
    TRAINING_DYNAMICS_STATUS_ATTR,
    TRIAL_DIR_ATTR,
    best_model_path,
    trial_directory,
)
from scripts.workflow.run_config import (
    preprocessing_dataset_name,
    validate_workflow_contract,
)
from training.training_dynamics import (
    DYNAMICS_ARTIFACT_NAME,
    DYNAMICS_MANIFEST_NAME,
    load_training_dynamics_identity,
)

# Packed sharded Zarr store (data.preprocess.zarr_layout.ZARR_ZIP_SUFFIX).
# Kept local so host-side commands do not import the scientific stack.
PROCESSED_DATASET_SUFFIX = ".zarr.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return document


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")


def write_yaml(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False))


def write_dependency_contract(
    config_path: Path,
    source_root: Path,
    output_path: Path,
    stage: str,
    source_paths: list[Path],
) -> Path:
    """Write the content-addressed source/config contract for one DAG stage."""
    config = load_yaml(config_path)
    stage_sections = {
        "data": ("dvc",),
        "training": ("training",),
        "evaluation": ("evaluation",),
        "hpo": ("optuna",),
        "ablation": ("ablation",),
        "crossval": ("crossval", "evaluation"),
        "selection": ("optuna",),
        "promotion": ("promotion", "dvc"),
        "preprocessing": ("preprocessing", "dvc"),
    }
    if stage == "ablation" and config["workflow"]["kind"] == "ablation_suite":
        stage_sections["ablation"] = ("ablation_suite",)
    if stage not in stage_sections:
        raise ValueError(f"Unknown dependency stage: {stage}")

    files = []
    for path in sorted(set(source_paths)):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(source_root.resolve(strict=True))
        if not resolved.is_file():
            raise ValueError(f"Dependency source must be a regular file: {path}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(resolved),
            }
        )
    if not files:
        raise ValueError(f"Dependency stage {stage!r} has no source files")

    config_fragments = {
        section: config[section]
        for section in stage_sections[stage]
        if section in config
    }
    payload = {
        "schema_version": 1,
        "stage": stage,
        "sources": files,
        "config": config_fragments,
        "container_digest": _container_digest(config["container"]),
    }
    fingerprint_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    payload["fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    write_json(output_path, payload)
    return output_path


def dvc_output(pointer_path: Path) -> dict[str, Any]:
    document = load_yaml(pointer_path)
    outputs = document.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError(f"Expected one DVC output in {pointer_path}")
    output = outputs[0]
    if not isinstance(output, dict) or "md5" not in output or "path" not in output:
        raise ValueError(f"Malformed DVC output in {pointer_path}")
    return output


def is_dvc_import(pointer_path: Path) -> bool:
    """Return whether a DVC pointer has a Git-backed import dependency."""
    document = load_yaml(pointer_path)
    dependencies = document.get("deps", [])
    return any(
        isinstance(dependency, dict) and isinstance(dependency.get("repo"), dict)
        for dependency in dependencies
    )


def md5_file(path: Path) -> str:
    """Return the content MD5 for a DVC-tracked regular file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_location(name: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    return str(distribution.locate_file(""))


def _container_digest(container: dict[str, Any]) -> str:
    configured = container.get("digest")
    if configured:
        return str(configured)
    digest_file = Path(str(container["digest_file"]))
    if not digest_file.is_file():
        raise FileNotFoundError(
            f"Container digest sidecar is missing: {digest_file}. "
            "Runs must pin an immutable SIF identity."
        )
    digest = digest_file.read_text().split()[0]
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def prepare_run(config_path: Path, source_root: Path, run_root: Path) -> Path:
    config = load_yaml(config_path)
    validate_workflow_contract(config)
    configured_run = Path(str(config["run"]["root"]))
    configured_source = Path(str(config["run"]["source_root"]))
    if configured_run != run_root or configured_source != source_root:
        raise ValueError("Runtime paths do not match the pinned workflow config")
    if source_root.stat().st_mode & 0o222:
        raise PermissionError(
            f"Source snapshot is writable: {source_root}. Prepare must make it read-only."
        )
    if config_path.stat().st_mode & 0o222:
        raise PermissionError(
            f"Workflow config is writable: {config_path}. Prepared runs are immutable."
        )

    for relative in (
        ".state",
        ".state/dependencies",
        ".state/workers",
        "logs",
        "outputs",
        "outputs/training",
        "outputs/evaluation",
        "outputs/preprocessing",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)

    dvc_identities: dict[str, dict[str, Any]] = {}
    materialized_targets = [str(target) for target in config["dvc"]["targets"]]
    provenance_targets = [
        str(target) for target in config["dvc"].get("provenance_targets", [])
    ]
    for target in [*materialized_targets, *provenance_targets]:
        pointer = source_root / str(target)
        output = dvc_output(pointer)
        output_path = pointer.parent / str(output["path"])
        dvc_identities[str(target)] = {
            "md5": output["md5"],
            "size": output.get("size"),
            "nfiles": output.get("nfiles"),
            "output": str(output_path.relative_to(source_root)),
            "role": (
                "materialized" if str(target) in materialized_targets else "provenance"
            ),
        }

    config_sha256 = sha256_file(config_path)
    dispatch_path = run_root / ".state/dispatch.tsv"
    dispatch_digest_path = run_root / ".state/dispatch.tsv.sha256"
    publication_path = run_root / ".state/published.done"
    for required in (dispatch_path, dispatch_digest_path, publication_path):
        if not required.is_file() or required.is_symlink():
            raise FileNotFoundError(f"Published run contract is missing: {required}")
    dispatch_sha256 = sha256_file(dispatch_path)
    recorded_dispatch_sha256 = dispatch_digest_path.read_text().split()[0]
    if dispatch_sha256 != recorded_dispatch_sha256:
        raise ValueError("Immutable dispatch request failed SHA-256 verification")

    publication = dict(
        line.split("=", 1)
        for line in publication_path.read_text().splitlines()
        if "=" in line
    )
    expected_publication = {
        "schema_version": "1",
        "run_id": str(config["run"]["id"]),
        "source_snapshot_sha256": str(config["source"]["source_snapshot_sha256"]),
        "config_sha256": config_sha256,
        "tooling_key": str(config["workflow"]["tooling_key"]),
    }
    if publication != expected_publication:
        raise ValueError("Publication receipt does not match the immutable run config")

    container_digest = _container_digest(config["container"])
    expected_snakemake_version = str(config["workflow"]["snakemake_version"])
    installed_snakemake_version = _package_version("snakemake")
    if installed_snakemake_version != expected_snakemake_version:
        raise RuntimeError(
            "Configured Snakemake does not match the coordinator environment: "
            f"expected {expected_snakemake_version}, installed "
            f"{installed_snakemake_version}"
        )
    installed_snakemake_location = _package_location("snakemake")
    if installed_snakemake_location is None:
        raise RuntimeError("Snakemake is not installed in the coordinator environment")
    workflow_kind = str(config["workflow"]["kind"])
    workflow_target = str(config["workflow"].get("target", workflow_kind))
    study_config = config.get("optuna") or config.get("ablation")
    crossval_config = config.get("crossval")
    identity = {
        "git_sha": config["source"]["git_sha"],
        "source_snapshot_sha256": config["source"]["source_snapshot_sha256"],
        "dvc_hashes": {
            target: identity["md5"] for target, identity in dvc_identities.items()
        },
        "container_digest": container_digest,
        "workflow_tooling_key": config["workflow"]["tooling_key"],
        "snakemake_config_sha256": config_sha256,
        "dispatch_request_sha256": dispatch_sha256,
        "workflow_kind": workflow_kind,
        "workflow_target": workflow_target,
    }
    if isinstance(study_config, dict):
        identity["study_name"] = study_config["study_name"]
    if "ablation_suite" in config:
        identity["study_names"] = [
            study["study_name"] for study in config["ablation_suite"]["studies"]
        ]
    if isinstance(crossval_config, dict):
        identity["crossval_campaign_id"] = crossval_config["campaign_id"]

    def pin_checkpoint(
        configured_checkpoint: Path,
        pinned_checkpoint: Path,
        *,
        label: str,
    ) -> dict[str, str]:
        if configured_checkpoint.is_symlink():
            raise ValueError(
                f"{label} input checkpoint must not be a symlink: "
                f"{configured_checkpoint}"
            )
        resolved_checkpoint = configured_checkpoint.resolve(strict=True)
        if not resolved_checkpoint.is_file():
            raise ValueError(
                f"{label} input checkpoint must resolve to a regular file: "
                f"{configured_checkpoint}"
            )
        checkpoint_sha256 = sha256_file(resolved_checkpoint)
        pinned_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if pinned_checkpoint.exists():
            if pinned_checkpoint.is_symlink() or not pinned_checkpoint.is_file():
                raise ValueError(
                    f"Pinned {label.lower()} checkpoint must be a regular "
                    f"non-symlink file: {pinned_checkpoint}"
                )
            if sha256_file(pinned_checkpoint) != checkpoint_sha256:
                raise ValueError(
                    f"Pinned {label.lower()} checkpoint conflicts with configured "
                    f"input: {pinned_checkpoint}"
                )
        else:
            temporary_checkpoint = pinned_checkpoint.with_suffix(".pt.partial")
            shutil.copyfile(resolved_checkpoint, temporary_checkpoint)
            if sha256_file(temporary_checkpoint) != checkpoint_sha256:
                temporary_checkpoint.unlink(missing_ok=True)
                raise ValueError(f"{label} checkpoint changed while being pinned")
            temporary_checkpoint.chmod(0o444)
            temporary_checkpoint.replace(pinned_checkpoint)
        return {
            "configured_checkpoint": str(configured_checkpoint),
            "resolved_checkpoint": str(resolved_checkpoint),
            "pinned_checkpoint": str(pinned_checkpoint),
            "pinned_checkpoint_relative": str(pinned_checkpoint.relative_to(run_root)),
            "checkpoint_sha256": checkpoint_sha256,
        }

    training_input = None
    if workflow_kind == "train":
        configured_init_checkpoint = config["training"].get("init_checkpoint")
        if configured_init_checkpoint is not None:
            training_input = pin_checkpoint(
                Path(str(configured_init_checkpoint)),
                run_root / "inputs/training_init_checkpoint.pt",
                label="Training initialization",
            )
            identity["init_checkpoint_sha256"] = training_input["checkpoint_sha256"]

    evaluation_input = None
    if workflow_kind == "evaluate":
        evaluation_input = pin_checkpoint(
            Path(str(config["evaluation"]["checkpoint"])),
            run_root / "inputs/evaluation_checkpoint.pt",
            label="Evaluation",
        )
        identity["input_checkpoint_sha256"] = evaluation_input["checkpoint_sha256"]

    manifest = {
        "schema_version": 1,
        "run_id": config["run"]["id"],
        "created_at": datetime.now(UTC).isoformat(),
        "identity": identity,
        "source": config["source"],
        "dvc": {
            "remote": config["dvc"]["remote"],
            "cache": config["dvc"]["cache"],
            "targets": dvc_identities,
        },
        "container": {
            "sif": config["container"]["sif"],
            "digest": container_digest,
        },
        "workflow": {
            "config_sha256": config_sha256,
            "dispatch_request_sha256": dispatch_sha256,
            "tooling_key": config["workflow"]["tooling_key"],
            "snakemake_bin": config["workflow"]["snakemake_bin"],
            "version": installed_snakemake_version,
            "installed_location": installed_snakemake_location,
            "execution_mode": config["workflow"]["execution_mode"],
            "execution_adapter": os.environ.get(
                "EPIFORECASTER_EXECUTION_MODE", "batch"
            ),
            "execution_attempt_id": os.environ.get(
                "EPIFORECASTER_EXECUTION_ATTEMPT_ID"
            ),
            "kind": workflow_kind,
            "target": workflow_target,
            "python_version": platform.python_version(),
            "initial_coordinator_job_id": os.environ.get("SLURM_JOB_ID"),
            "initial_coordinator_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "resources": config["resources"],
    }
    for section in (
        "training",
        "evaluation",
        "preprocessing",
        "crossval",
        "optuna",
        "ablation",
        "ablation_suite",
        "promotion",
        "provenance",
    ):
        if section in config:
            manifest[section] = config[section]
    if evaluation_input is not None:
        manifest["evaluation_input"] = evaluation_input
    if training_input is not None:
        manifest["training_input"] = training_input
    manifest_path = run_root / "manifest.yaml"
    write_yaml(manifest_path, manifest)
    prepared_path = run_root / ".state/prepared.done"
    write_json(
        prepared_path,
        {
            "manifest": str(manifest_path),
            "identity": manifest["identity"],
        },
    )
    manifest_path.chmod(0o444)
    prepared_path.chmod(0o444)
    return manifest_path


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _safe_component(value: object, *, field: str) -> str:
    component = str(value)
    if not component or component in {".", ".."} or Path(component).name != component:
        raise ValueError(f"{field} must be one safe path component: {component!r}")
    return component


def training_artifact_paths(
    config: dict[str, Any], run_root: Path
) -> tuple[Path, Path]:
    training = config["training"]
    experiment_name = _safe_component(
        training["experiment_name"], field="training.experiment_name"
    )
    model_id = _safe_component(training["model_id"], field="training.model_id")
    model_root = run_root / "outputs/training" / experiment_name / model_id
    return model_root / "checkpoints/best_model.pt", model_root


def train_model(config_path: Path, source_root: Path, run_root: Path) -> Path:
    """Run one deterministic training entity and record its exact checkpoint."""
    config = load_yaml(config_path)
    training = config["training"]
    checkpoint, model_root = training_artifact_paths(config, run_root)
    base_config = source_root / str(training["base_config"])
    if not base_config.is_file():
        raise FileNotFoundError(f"Training config is missing: {base_config}")

    command = [
        sys.executable,
        "-m",
        "cli",
        "train",
        "epiforecaster",
        "--config",
        str(base_config),
    ]
    max_batches = training.get("max_batches")
    if max_batches is not None:
        command.extend(["--max-batches", str(int(max_batches))])
    overrides = [
        *[str(value) for value in training.get("overrides", [])],
        f"output.log_dir={run_root / 'outputs/training'}",
        f"output.experiment_name={training['experiment_name']}",
        "output.save_checkpoints=true",
        "training.run_final_evaluation=false",
    ]
    if training.get("init_checkpoint") is not None:
        pinned_init_checkpoint = run_root / "inputs/training_init_checkpoint.pt"
        manifest = load_yaml(run_root / "manifest.yaml")
        expected_sha256 = str(manifest["training_input"]["checkpoint_sha256"])
        if (
            not pinned_init_checkpoint.is_file()
            or pinned_init_checkpoint.is_symlink()
            or sha256_file(pinned_init_checkpoint) != expected_sha256
        ):
            raise ValueError(
                "Pinned training initialization checkpoint failed validation"
            )
        overrides.append(f"training.init_checkpoint_path={pinned_init_checkpoint}")
    for override in overrides:
        command.extend(["--override", override])

    environment = os.environ.copy()
    environment["EPIFORECASTER_MODEL_ID"] = str(training["model_id"])
    _run(command, cwd=source_root, env=environment)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(
            f"Training completed without the canonical checkpoint: {checkpoint}"
        )
    training_dynamics = load_training_dynamics_identity(model_root)

    training_manifest = model_root / "training_manifest.json"
    training_record = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_root": str(model_root),
        "base_config": str(base_config),
        "overrides": overrides,
        "training_dynamics": training_dynamics,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(training_manifest, training_record)
    marker = run_root / ".state/training.done"
    write_json(
        marker,
        {
            "training_manifest": str(training_manifest),
            **training_record,
        },
    )
    return marker


def _crossval_fold_root(config: dict[str, Any], run_root: Path, fold: int) -> Path:
    campaign_id = _safe_component(
        config["crossval"]["campaign_id"], field="crossval.campaign_id"
    )
    return run_root / "outputs/crossval" / campaign_id / f"fold-{fold:03d}"


def _crossval_fold_config(config: dict[str, Any], fold: int) -> dict[str, Any]:
    options = config["crossval"]
    folds = int(options["folds"])
    if not 0 <= fold < folds:
        raise ValueError(f"Cross-validation fold {fold} is outside [0, {folds})")
    fold_config = copy.deepcopy(config)
    fold_config["workflow"] = {"kind": "train", "target": "train_evaluate"}
    fold_config["training"] = {
        "base_config": str(options["base_config"]),
        "experiment_name": "training",
        "model_id": "model",
        "max_batches": options.get("max_batches"),
        "overrides": [
            *[str(value) for value in options.get("overrides", [])],
            "training.crossval_enabled=true",
            "training.node_split_strategy=crossval",
            f"training.crossval_num_folds={folds}",
            f"training.crossval_fold_index={fold}",
            f"training.crossval_holdout_role={options.get('holdout_role', 'test')}",
            f"training.crossval_manifest_path={options['split_manifest']}",
            f"training.seed={int(options['model_seed'])}",
        ],
    }
    fold_config["evaluation"] = copy.deepcopy(config["evaluation"])
    return fold_config


def train_crossval_fold(
    config_path: Path,
    source_root: Path,
    run_root: Path,
    fold: int,
) -> Path:
    """Train one deterministic cross-validation fold."""
    config = load_yaml(config_path)
    options = config["crossval"]
    split_manifest = source_root / str(options["split_manifest"])
    if not split_manifest.is_file():
        raise FileNotFoundError(
            f"Cross-validation split manifest is missing: {split_manifest}"
        )
    fold_config = _crossval_fold_config(config, fold)
    fold_root = _crossval_fold_root(config, run_root, fold)
    attempts = run_root / ".state/attempts/crossval" / f"fold-{fold:03d}/training"
    identity = {
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_manifest),
        "fold": fold,
    }
    fold_training_manifest = fold_root / "fold_training_manifest.json"
    if restore_attempt(attempts, identity, run_root):
        return fold_training_manifest
    attempt = new_attempt(attempts)
    attempt_config = attempt / "config.yaml"
    write_yaml(attempt_config, fold_config)
    train_model(attempt_config, source_root, attempt)
    attempt_checkpoint, attempt_model_root = training_artifact_paths(
        fold_config, attempt
    )
    attempt_training_manifest = attempt_model_root / "training_manifest.json"
    training_record = json.loads(attempt_training_manifest.read_text())
    fold_config_path = fold_root / "config.yaml"
    checkpoint, model_root = training_artifact_paths(fold_config, fold_root)
    training_manifest = model_root / "training_manifest.json"
    attempt_manifest = attempt / "fold_training_manifest.json"
    write_json(
        attempt_manifest,
        {
            "schema_version": 1,
            "campaign_id": str(options["campaign_id"]),
            "fold": fold,
            "folds": int(options["folds"]),
            "model_seed": int(options["model_seed"]),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
            "config": str(fold_config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(attempt_checkpoint),
            "training_manifest": str(training_manifest),
            "training_manifest_sha256": sha256_file(attempt_training_manifest),
            "training_dynamics": training_record["training_dynamics"],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    complete_attempt(
        attempt,
        identity,
        {
            checkpoint: attempt_checkpoint,
            training_manifest: attempt_training_manifest,
            fold_config_path: attempt_config,
            fold_training_manifest: attempt_manifest,
        },
        run_root,
    )
    return fold_training_manifest


def evaluate_crossval_fold(
    config_path: Path,
    run_root: Path,
    fold: int,
) -> Path:
    """Evaluate the exact checkpoint emitted by one cross-validation fold."""
    config = load_yaml(config_path)
    options = config["crossval"]
    _crossval_fold_config(config, fold)
    fold_root = _crossval_fold_root(config, run_root, fold)
    fold_config_path = fold_root / "config.yaml"
    fold_config = load_yaml(fold_config_path)
    checkpoint, _ = training_artifact_paths(fold_config, fold_root)
    training_fold_manifest_path = fold_root / "fold_training_manifest.json"
    training_fold_manifest = json.loads(training_fold_manifest_path.read_text())
    if int(training_fold_manifest["fold"]) != fold:
        raise ValueError(f"Cross-validation fold identity mismatch: {fold_root}")
    if Path(str(training_fold_manifest["checkpoint"])) != checkpoint:
        raise ValueError(f"Cross-validation checkpoint path mismatch: {fold_root}")
    if sha256_file(checkpoint) != training_fold_manifest["checkpoint_sha256"]:
        raise ValueError(f"Cross-validation checkpoint failed SHA-256: {checkpoint}")
    split_manifest = Path(str(training_fold_manifest["split_manifest"])).resolve(
        strict=True
    )
    if sha256_file(split_manifest) != training_fold_manifest["split_manifest_sha256"]:
        raise ValueError(
            f"Cross-validation split manifest failed SHA-256: {split_manifest}"
        )
    training_manifest = Path(str(training_fold_manifest["training_manifest"])).resolve(
        strict=True
    )
    if (
        sha256_file(training_manifest)
        != training_fold_manifest["training_manifest_sha256"]
    ):
        raise ValueError(
            f"Cross-validation training manifest failed SHA-256: {training_manifest}"
        )

    attempts = run_root / ".state/attempts/crossval" / f"fold-{fold:03d}/evaluation"
    identity = {
        "config_sha256": sha256_file(config_path),
        "training_sha256": sha256_file(training_fold_manifest_path),
        "fold": fold,
    }
    fold_manifest = fold_root / "fold_manifest.json"
    if restore_attempt(attempts, identity, run_root):
        return fold_manifest
    attempt = new_attempt(attempts)
    attempt_evaluation_manifest = evaluate_configured_model(
        fold_config_path,
        attempt,
        checkpoint,
    )
    evaluation_manifest = fold_root / "outputs/evaluation/evaluation_manifest.json"
    attempt_manifest = attempt / "fold_manifest.json"
    write_json(
        attempt_manifest,
        {
            "schema_version": 1,
            "campaign_id": str(options["campaign_id"]),
            "fold": fold,
            "folds": int(options["folds"]),
            "model_seed": int(options["model_seed"]),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
            "config": str(fold_config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_fold_manifest": str(training_fold_manifest_path),
            "training_fold_manifest_sha256": sha256_file(training_fold_manifest_path),
            "evaluation_manifest": str(evaluation_manifest),
            "evaluation_manifest_sha256": sha256_file(attempt_evaluation_manifest),
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    artifacts = {
        fold_root / path.relative_to(attempt): path
        for path in (attempt / "outputs/evaluation").rglob("*")
        if path.is_file()
    }
    artifacts[fold_manifest] = attempt_manifest
    complete_attempt(attempt, identity, artifacts, run_root)
    return fold_manifest


def _scalar_evaluation_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"eval_loss": summary.get("eval_loss")}
    for name, value in summary.get("metrics", {}).items():
        if value is None or isinstance(value, (int, float)):
            metrics[str(name)] = None if value is None else float(value)
    return metrics


def finalize_crossval(config_path: Path, run_root: Path) -> Path:
    """Validate every fold and aggregate canonical evaluation summaries."""
    config = load_yaml(config_path)
    options = config["crossval"]
    folds = int(options["folds"])
    split = str(config["evaluation"]["split"])
    output_dir = run_root / "outputs/crossval" / str(options["campaign_id"])
    fold_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    for fold in range(folds):
        fold_root = _crossval_fold_root(config, run_root, fold)
        fold_manifest_path = fold_root / "fold_manifest.json"
        fold_manifest = json.loads(fold_manifest_path.read_text())
        if int(fold_manifest["fold"]) != fold:
            raise ValueError(f"Cross-validation fold identity mismatch: {fold_root}")
        for field in (
            "checkpoint",
            "training_fold_manifest",
            "evaluation_manifest",
        ):
            artifact = Path(str(fold_manifest[field])).resolve(strict=True)
            expected_hash = str(fold_manifest[f"{field}_sha256"])
            if sha256_file(artifact) != expected_hash:
                raise ValueError(f"Cross-validation {field} failed SHA-256: {artifact}")

        summary_path = fold_root / "outputs/evaluation" / f"{split}_summary.json"
        summary = json.loads(summary_path.read_text())
        for metric, value in sorted(_scalar_evaluation_metrics(summary).items()):
            fold_rows.append(
                {
                    "fold": fold,
                    "model_seed": int(options["model_seed"]),
                    "metric": metric,
                    "value": value,
                }
            )
        fold_records.append(
            {
                "fold": fold,
                "split_manifest": str(fold_manifest["split_manifest"]),
                "split_manifest_sha256": str(fold_manifest["split_manifest_sha256"]),
                "fold_manifest": str(fold_manifest_path),
                "fold_manifest_sha256": sha256_file(fold_manifest_path),
                "evaluation_summary": str(summary_path),
                "evaluation_summary_sha256": sha256_file(summary_path),
            }
        )

    fold_metrics = output_dir / f"{split}_fold_metrics.csv"
    with fold_metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "model_seed", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for metric in sorted({str(row["metric"]) for row in fold_rows}):
        values = [
            float(row["value"])
            for row in fold_rows
            if row["metric"] == metric and row["value"] is not None
        ]
        aggregate_rows.append(
            {
                "metric": metric,
                "folds": len(values),
                "mean": fmean(values) if values else None,
                "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
            }
        )
    aggregate_metrics = output_dir / f"{split}_aggregate_metrics.csv"
    with aggregate_metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "folds", "mean", "std"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    split_manifest_identities = {
        (
            str(record["split_manifest"]),
            str(record["split_manifest_sha256"]),
        )
        for record in fold_records
    }
    if len(split_manifest_identities) != 1:
        raise ValueError(
            "Cross-validation folds do not share one split manifest identity"
        )
    split_manifest_path, split_manifest_sha256 = split_manifest_identities.pop()

    manifest = output_dir / "crossval_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "campaign_id": str(options["campaign_id"]),
            "folds": fold_records,
            "model_seed": int(options["model_seed"]),
            "split_manifest": split_manifest_path,
            "split_manifest_sha256": split_manifest_sha256,
            "split": split,
            "artifacts": [
                {"path": str(fold_metrics), "sha256": sha256_file(fold_metrics)},
                {
                    "path": str(aggregate_metrics),
                    "sha256": sha256_file(aggregate_metrics),
                },
            ],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    write_json(
        run_root / ".state/crossval.done",
        {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest)},
    )
    return manifest


def _with_cache_lock(cache_path: Path):
    cache_path.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path / ".dvc-workflow.lock"
    lock_path.touch(mode=0o664, exist_ok=True)
    lock_handle = lock_path.open("r+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle


def stage_dvc(config_path: Path, source_root: Path, run_root: Path) -> list[Path]:
    config = load_yaml(config_path)
    workspace = run_root / ".state/dvc-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".dvc").is_dir():
        _run(["dvc", "init", "--no-scm"], cwd=workspace)

    targets: list[str] = []
    imported_targets: list[tuple[str, Path, dict[str, Any]]] = []
    output_paths: list[Path] = []
    for target_value in config["dvc"]["targets"]:
        target = Path(str(target_value))
        source_pointer = source_root / target
        workspace_pointer = workspace / target
        workspace_pointer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pointer, workspace_pointer)
        output = dvc_output(source_pointer)
        output_relative = target.parent / str(output["path"])
        output_paths.append(output_relative)
        target_name = target.as_posix()
        targets.append(target_name)
        if is_dvc_import(source_pointer):
            imported_targets.append((target_name, output_relative, output))

    remote = Path(str(config["dvc"]["remote"]))
    cache = Path(str(config["dvc"]["cache"]))
    _run(["dvc", "remote", "add", "--force", "mn5-nfs", str(remote)], cwd=workspace)
    _run(["dvc", "config", "core.remote", "mn5-nfs"], cwd=workspace)
    # Keep DVC's SQLite/index state off the quota-constrained cluster home.
    # A relative path is resolved inside this run's isolated DVC workspace.
    _run(["dvc", "config", "core.site_cache_dir", ".dvc/tmp"], cwd=workspace)
    _run(["dvc", "config", "cache.dir", str(cache)], cwd=workspace)
    _run(["dvc", "config", "cache.shared", "group"], cwd=workspace)
    _run(["dvc", "config", "cache.type", "symlink"], cwd=workspace)

    lock_handle = _with_cache_lock(cache)
    try:
        _run(["dvc", "pull", *targets], cwd=workspace)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    status_targets = [
        target
        for target in targets
        if target not in {item[0] for item in imported_targets}
    ]
    if status_targets:
        _run(["dvc", "status", *status_targets], cwd=workspace)

    for output_path in output_paths:
        if not (workspace / output_path).exists():
            raise FileNotFoundError(f"DVC output was not materialized: {output_path}")

    # DVC content-addresses the cache, so the imported object is trusted by its
    # path. Rehashing a 173 GB corpus over GPFS cost ~8 min per run; a size
    # check still catches a truncated or wrong object.
    for target, output_path, output in imported_targets:
        expected_md5 = str(output["md5"])
        materialized = workspace / output_path
        if expected_md5.endswith(".dir") or not materialized.is_file():
            raise ValueError(
                "Git-backed DVC imports must materialize one regular file for "
                f"offline validation: {target}"
            )
        expected_size = output.get("size")
        actual_size = materialized.stat().st_size
        if expected_size is not None and actual_size != int(expected_size):
            raise ValueError(
                f"Imported DVC output size mismatch for {target}: "
                f"{actual_size} != {expected_size}"
            )

    bind_roots = sorted({path.parent for path in output_paths})
    for bind_root in bind_roots:
        if not (source_root / bind_root).is_dir():
            raise FileNotFoundError(
                f"DVC bind root must exist in the source snapshot: {bind_root}"
            )
    (run_root / ".state/dvc_paths.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in output_paths)
    )
    (run_root / ".state/dvc_bind_roots.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in bind_roots)
    )
    data_manifest = run_root / ".state/data_manifest.json"
    data_record = {
        "schema_version": 1,
        "workspace": str(workspace),
        "targets": {
            target: {
                "md5": dvc_output(source_root / target)["md5"],
                "output": output_path.as_posix(),
            }
            for target, output_path in zip(targets, output_paths, strict=True)
        },
        "outputs": [path.as_posix() for path in output_paths],
        "bind_roots": [path.as_posix() for path in bind_roots],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(data_manifest, data_record)
    write_json(
        run_root / ".state/data_staged.done",
        {"data_manifest": str(data_manifest), **data_record},
    )
    return output_paths


def _load_preprocessing_document(
    config_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Load a source-owned preprocessing config and its local base configs."""
    resolved = config_path.resolve(strict=True)
    source = source_root.resolve(strict=True)
    resolved.relative_to(source)
    document = load_yaml(resolved)
    base_config = document.pop("base_config", None)
    if base_config is None:
        return document
    base_path = Path(str(base_config))
    if base_path.is_absolute() or ".." in base_path.parts:
        raise ValueError(
            "Preprocessing base_config must stay inside the source snapshot"
        )
    parent = _load_preprocessing_document(resolved.parent / base_path, source_root)
    return _deep_merge(parent, document)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _processed_dataset_qa(
    dataset: Any,
    workflow_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the declared processed-data contract with labeled arrays."""
    import numpy as np

    preprocessing = workflow_config["preprocessing"]
    if not dataset.data_vars:
        raise ValueError("Processed dataset contains no data variables")
    empty_dimensions = {
        name: int(size) for name, size in dataset.sizes.items() if int(size) <= 0
    }
    if empty_dimensions:
        raise ValueError(f"Processed dataset has empty dimensions: {empty_dimensions}")

    expected_run_count = preprocessing.get("expected_run_count")
    if expected_run_count is not None:
        actual_run_count = int(dataset.sizes.get("run_id", 0))
        if actual_run_count != int(expected_run_count):
            raise ValueError(
                "Processed dataset run count differs from the workflow contract: "
                f"{actual_run_count} != {expected_run_count}"
            )

    required_variables = [
        str(value) for value in preprocessing.get("required_variables", [])
    ]
    missing = sorted(set(required_variables) - set(dataset.data_vars))
    if missing:
        raise ValueError(f"Processed dataset is missing required variables: {missing}")

    float32_variables = [
        str(value) for value in preprocessing.get("float32_variables", [])
    ]
    wrong_dtypes = {
        name: str(dataset[name].dtype)
        for name in float32_variables
        if name in dataset and dataset[name].dtype != np.dtype("float32")
    }
    if wrong_dtypes:
        raise ValueError(
            f"Processed continuous variables are not float32: {wrong_dtypes}"
        )

    nonnegative_variables = [
        str(value) for value in preprocessing.get("nonnegative_variables", [])
    ]
    variable_qa: dict[str, dict[str, Any]] = {}
    for name in sorted(set(float32_variables) | set(nonnegative_variables)):
        if name not in dataset:
            continue
        values = np.asarray(dataset[name].values)
        finite = np.isfinite(values)
        finite_count = int(finite.sum())
        nonfinite_count = int(values.size - finite_count)
        negative_count = int((values[finite] < 0).sum())
        if name in nonnegative_variables and (nonfinite_count or negative_count):
            raise ValueError(
                f"Processed variable {name!r} has {nonfinite_count} non-finite "
                f"and {negative_count} negative values"
            )
        variable_qa[name] = {
            "dtype": str(values.dtype),
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "negative_count": negative_count,
            "positive_count": int((values[finite] > 0).sum()),
        }

    flow_alignment: dict[str, Any] | None = None
    if {"latent_D_true", "latent_death_flow_true"} <= set(dataset.data_vars):
        cumulative_difference = dataset["latent_D_true"].diff("date")
        destination_flow = dataset["latent_death_flow_true"].sel(
            date=cumulative_difference["date"]
        )
        difference_values = np.asarray(cumulative_difference.values)
        flow_values = np.asarray(destination_flow.values)
        comparable = np.isfinite(difference_values) & np.isfinite(flow_values)
        absolute_error = np.abs(difference_values[comparable] - flow_values[comparable])
        max_absolute_error = float(absolute_error.max()) if absolute_error.size else 0.0
        if not np.allclose(
            difference_values[comparable],
            flow_values[comparable],
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError(
                "Processed latent death flow is not destination-aligned with "
                f"latent_D_true.diff(date); max error={max_absolute_error:.8g}"
            )
        flow_alignment = {
            "comparison_count": int(comparable.sum()),
            "max_absolute_error": max_absolute_error,
            "rtol": 1e-5,
            "atol": 1e-8,
        }

    preservation_json = dataset.attrs.get("positive_clinical_preservation_json")
    positive_preservation = (
        json.loads(str(preservation_json)) if preservation_json is not None else None
    )
    if positive_preservation is not None:
        for name, record in positive_preservation.items():
            if int(record["zeroed_raw_positive_count"]) != 0:
                raise ValueError(
                    f"Processed {name!r} lost positive raw observations according "
                    "to persisted preprocessing evidence"
                )
            if int(record["processed_positive_count"]) < int(
                record["raw_positive_count"]
            ):
                raise ValueError(
                    f"Processed {name!r} has fewer positive values than the raw "
                    "positive count"
                )

    return {
        "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "data_variables": sorted(dataset.data_vars),
        "variable_qa": variable_qa,
        "latent_death_flow_alignment": flow_alignment,
        "positive_clinical_preservation": positive_preservation,
    }


def _synthetic_corpus_provenance(dataset: Any) -> dict[str, Any] | None:
    """Return the processed synthetic corpus identity and mobility layout QA."""
    from data.preprocess.pipeline import SYNTHETIC_CORPUS_PROVENANCE_ATTR

    value = dataset.attrs.get(SYNTHETIC_CORPUS_PROVENANCE_ATTR)
    return json.loads(str(value)) if value is not None else None


def _handoff_qa_acceptance(
    source_root: Path,
    workflow_config: dict[str, Any],
    handoff: dict[str, Any],
    handoff_sha256: str,
) -> dict[str, Any]:
    """Accept a failed generator handoff only under an exact owner waiver.

    The waiver must name this handoff manifest by SHA-256 and list exactly the
    gates that the handoff reports as failed, with the same observed values.
    """
    acceptance_value = workflow_config["preprocessing"].get("handoff_qa_acceptance")
    if acceptance_value is None:
        raise ValueError("Synthetic generator handoff QA did not pass")
    relative = Path(str(acceptance_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("preprocessing.handoff_qa_acceptance must be source-relative")
    acceptance_path = (source_root / relative).resolve(strict=True)
    acceptance_path.relative_to(source_root.resolve(strict=True))
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("handoff_manifest_sha256") != handoff_sha256:
        raise ValueError(
            "Handoff QA acceptance names a different handoff manifest SHA-256"
        )
    if acceptance.get("generated_qa_passed") is not False:
        raise ValueError("Handoff QA acceptance must record generated_qa_passed=false")
    failed = {
        str(gate["gate"]): gate["observed"]
        for gate in handoff.get("qa_gates", [])
        if gate.get("passed") is not True
    }
    accepted = {
        str(gate["gate"]): gate["observed"]
        for gate in acceptance.get("accepted_failed_gates", [])
    }
    if not failed:
        raise ValueError(
            "Synthetic generator handoff QA did not pass and lists no failed gates"
        )
    if accepted != failed:
        raise ValueError(
            "Handoff QA acceptance must list exactly the failed gates with their "
            f"observed values: failed={failed}, accepted={accepted}"
        )
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(acceptance_path),
        "decision": acceptance.get("decision"),
        "date": acceptance.get("date"),
        "accepted_failed_gates": sorted(accepted),
    }


def _handoff_provenance_qa(
    source_root: Path,
    workflow_config: dict[str, Any],
    effective_config: dict[str, Any],
    dataset: Any,
) -> dict[str, Any] | None:
    """Validate and retain the generator handoff identity and seed record."""
    handoff_value = workflow_config["preprocessing"].get("handoff_manifest")
    if handoff_value is None:
        return None
    relative = Path(str(handoff_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("preprocessing.handoff_manifest must be source-relative")
    handoff_path = (source_root / relative).resolve(strict=True)
    handoff_path.relative_to(source_root.resolve(strict=True))
    handoff = json.loads(handoff_path.read_text())
    handoff_sha256 = sha256_file(handoff_path)
    expected_sha256 = effective_config.get("source_provenance", {}).get(
        "handoff_manifest_sha256"
    )
    if expected_sha256 and handoff_sha256 != str(expected_sha256):
        raise ValueError(
            "Synthetic generator handoff manifest SHA-256 does not match the "
            "preprocessing source contract"
        )
    acceptance = None
    if handoff.get("qa_passed") is not True:
        acceptance = _handoff_qa_acceptance(
            source_root,
            workflow_config,
            handoff,
            handoff_sha256,
        )
    handoff_run_ids = [str(value) for value in handoff.get("run_ids", [])]
    # Config-declared exclusions are the only runs allowed to be missing.
    exclusions = effective_config.get("synthetic_run_exclusions") or {}
    unknown_exclusions = sorted(set(exclusions) - set(handoff_run_ids))
    if unknown_exclusions:
        raise ValueError(
            "synthetic_run_exclusions names runs absent from the generator "
            f"handoff: {unknown_exclusions}"
        )
    expected_run_ids = [
        run_id for run_id in handoff_run_ids if run_id not in exclusions
    ]
    observed_run_ids = [str(value) for value in dataset["run_id"].values]
    if sorted(expected_run_ids) != sorted(observed_run_ids):
        raise ValueError(
            "Processed run IDs differ from the synthetic generator handoff "
            "minus the declared exclusions"
        )
    seeds = handoff.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        raise ValueError("Synthetic generator handoff has no seed record")
    for name, values in seeds.items():
        if not isinstance(values, list) or len(values) != len(handoff_run_ids):
            raise ValueError(
                f"Synthetic generator seed vector {name!r} does not match run count"
            )
    seed_by_run = {
        name: {
            run_id: seed
            for run_id, seed in zip(handoff_run_ids, values, strict=True)
            if run_id not in exclusions
        }
        for name, values in seeds.items()
    }
    return {
        "path": relative.as_posix(),
        "sha256": handoff_sha256,
        "qa_passed": acceptance is None,
        "qa_acceptance": acceptance,
        "run_ids": expected_run_ids,
        "handoff_run_ids": handoff_run_ids,
        "excluded_runs": [
            {"run_id": run_id, "reason": str(exclusions[run_id])}
            for run_id in sorted(exclusions)
        ],
        "seeds": seeds,
        "seed_by_run": seed_by_run,
        "corpus_full_tree_sha256": handoff.get("corpus_full_tree_sha256"),
        "corpus_metadata_tree_sha256": handoff.get("corpus_metadata_tree_sha256"),
        "reporting_config_sha256": handoff.get("reporting_config_sha256"),
    }


def run_preprocessing(config_path: Path, source_root: Path, run_root: Path) -> Path:
    """Build and QA a processed dataset inside one immutable run root."""
    config = load_yaml(config_path)
    validate_workflow_contract(config)
    preprocessing = config["preprocessing"]
    source_config = source_root / Path(str(preprocessing["base_config"]))
    effective = _load_preprocessing_document(source_config, source_root)
    effective["write_workers"] = int(preprocessing.get("write_workers", 1))
    output_marker = run_root / "outputs/preprocessing/.preprocessed.done"
    attempts = run_root / ".state/attempts/preprocessing"
    identity = {"config_sha256": sha256_file(config_path), "effective": effective}
    # Normalize YAML dates for the JSON receipt comparison on a later attempt.
    identity = json.loads(json.dumps(identity, default=str))
    if restore_attempt(attempts, identity, run_root):
        return output_marker
    attempt = new_attempt(attempts)
    # Node-local output keeps the write, the post-write read-back, QA and the
    # SHA-256 off shared GPFS; one sequential copy then moves the result into
    # the attempt. The container /tmp is the step's node-local scratch.
    node_local_output = bool(preprocessing.get("node_local_output", False))
    local_root = (
        Path(tempfile.mkdtemp(prefix="epiforecaster-preprocess-"))
        if node_local_output
        else None
    )
    try:
        return _run_preprocessing_attempt(
            config,
            effective,
            attempt,
            identity,
            run_root,
            source_root,
            source_config,
            local_root,
        )
    finally:
        if local_root is not None:
            shutil.rmtree(local_root, ignore_errors=True)


def _run_preprocessing_attempt(
    config: dict[str, Any],
    effective: dict[str, Any],
    attempt: Path,
    identity: dict[str, Any],
    run_root: Path,
    source_root: Path,
    source_config: Path,
    local_root: Path | None,
) -> Path:
    output_root = run_root / "outputs/preprocessing"
    effective_config = output_root / "effective_config.yaml"
    output_marker = output_root / ".preprocessed.done"
    qa_path = output_root / "qa.json"
    build_root = attempt if local_root is None else local_root
    effective["output_path"] = str(build_root / "dataset")
    attempt_config = attempt / "effective_config.yaml"
    write_yaml(attempt_config, effective)

    from data.preprocess import OfflinePreprocessingPipeline, PreprocessingConfig

    pipeline_config = PreprocessingConfig.from_file(attempt_config)
    built_dataset = Path(OfflinePreprocessingPipeline(pipeline_config).run()).resolve(
        strict=True
    )
    built_dataset.relative_to(build_root.resolve(strict=True))
    if built_dataset.is_symlink() or not built_dataset.is_file():
        raise ValueError(
            f"Preprocessing output must be a regular packed Zarr file: {built_dataset}"
        )
    dataset_path = (
        output_root / f"{preprocessing_dataset_name(config)}{PROCESSED_DATASET_SUFFIX}"
    )

    from data.epi_dataset import open_processed_dataset

    dataset = open_processed_dataset(built_dataset)
    try:
        qa_record = {
            "schema_version": 2,
            "dataset": str(dataset_path),
            "dataset_relative": str(dataset_path.relative_to(run_root)),
            **_processed_dataset_qa(dataset, config),
            "generator_handoff": _handoff_provenance_qa(
                source_root,
                config,
                effective,
                dataset,
            ),
            "synthetic_corpus_provenance": _synthetic_corpus_provenance(dataset),
            "effective_config": str(effective_config),
            "effective_config_sha256": sha256_file(attempt_config),
            "source_config": str(source_config.relative_to(source_root)),
            "source_config_sha256": sha256_file(source_config),
            "completed_at": datetime.now(UTC).isoformat(),
        }
    finally:
        dataset.close()
    dataset_sha256 = sha256_file(built_dataset)
    if local_root is None:
        attempt_dataset = built_dataset
    else:
        attempt_dataset = attempt / "dataset" / built_dataset.name
        attempt_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built_dataset, attempt_dataset)
        if attempt_dataset.stat().st_size != built_dataset.stat().st_size:
            raise ValueError(
                f"Copy of the node-local dataset is incomplete: {attempt_dataset}"
            )
    attempt_qa = attempt / "qa.json"
    attempt_marker = attempt / ".preprocessed.done"
    write_json(attempt_qa, qa_record)
    write_json(
        attempt_marker,
        {
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "qa": str(qa_path),
            "qa_sha256": sha256_file(attempt_qa),
        },
    )
    complete_attempt(
        attempt,
        identity,
        {
            dataset_path: attempt_dataset,
            effective_config: attempt_config,
            qa_path: attempt_qa,
            output_marker: attempt_marker,
        },
        run_root,
    )
    return output_marker


def publish_preprocessed_dataset(config_path: Path, run_root: Path) -> Path:
    """Publish a QA-passed dataset as a run-owned DVC pointer."""
    config = load_yaml(config_path)
    dataset_name = preprocessing_dataset_name(config)
    dataset_pointer = f"{dataset_name}{PROCESSED_DATASET_SUFFIX}.dvc"
    publication_payload = f"{dataset_name}{PROCESSED_DATASET_SUFFIX}"
    output_root = run_root / "outputs/preprocessing"
    marker_path = output_root / ".preprocessed.done"
    marker = json.loads(marker_path.read_text())
    dataset = Path(str(marker["dataset"])).resolve(strict=True)
    dataset.relative_to(output_root.resolve(strict=True))
    if dataset.is_symlink() or not dataset.is_file():
        raise ValueError("Preprocessed dataset must be a non-symlink packed Zarr file")
    if sha256_file(dataset) != str(marker["dataset_sha256"]):
        raise ValueError("Preprocessed dataset changed after QA")
    qa_path = Path(str(marker["qa"])).resolve(strict=True)
    if sha256_file(qa_path) != str(marker["qa_sha256"]):
        raise ValueError("Preprocessing QA record changed before publication")

    destination = output_root / "publication"
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise FileExistsError(
                "Preprocessing publication already exists for this run"
            )
        destination.rmdir()
    # Stage next to the dataset on the shared filesystem and hard-link it.
    # Node-local /tmp may be smaller than the dataset (118 GB for b354b27 on
    # Nord4), and DVC moves the payload into the shared cache anyway.
    scratch_staging = Path(
        tempfile.mkdtemp(prefix=f".publication.staging-{os.getpid()}-", dir=output_root)
    )
    payload = scratch_staging / publication_payload
    commit_staging = output_root / f".publication.commit-{os.getpid()}"
    try:
        try:
            os.link(dataset, payload)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(dataset, payload)
            if sha256_file(payload) != str(marker["dataset_sha256"]):
                raise ValueError(
                    "Publication staging changed the processed dataset"
                ) from None
        remote = Path(str(config["dvc"]["remote"]))
        cache = Path(str(config["dvc"]["cache"]))
        _run(["dvc", "init", "--no-scm"], cwd=scratch_staging)
        _run(
            ["dvc", "remote", "add", "--force", "mn5-nfs", str(remote)],
            cwd=scratch_staging,
        )
        _run(["dvc", "config", "core.remote", "mn5-nfs"], cwd=scratch_staging)
        _run(["dvc", "config", "cache.dir", str(cache)], cwd=scratch_staging)
        _run(["dvc", "config", "cache.shared", "group"], cwd=scratch_staging)
        _run(["dvc", "config", "cache.type", "symlink"], cwd=scratch_staging)
        lock_handle = _with_cache_lock(cache)
        try:
            _run(["dvc", "add", publication_payload], cwd=scratch_staging)
            _run(["dvc", "push", dataset_pointer], cwd=scratch_staging)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        pointer = scratch_staging / dataset_pointer
        pointer_identity = dvc_output(pointer)
        receipt = {
            "schema_version": 1,
            "run_id": str(config["run"]["id"]),
            "source_git_sha": str(config["source"]["git_sha"]),
            "source_snapshot_sha256": str(config["source"]["source_snapshot_sha256"]),
            "config_sha256": sha256_file(config_path),
            "container_sif": str(config["container"]["sif"]),
            "container_digest": _container_digest(config["container"]),
            "input_dvc_hashes": {
                target: str(record["md5"])
                for target, record in json.loads(
                    (run_root / ".state/data_manifest.json").read_text()
                )["targets"].items()
            },
            "dvc_pointer": dataset_pointer,
            "dvc_md5": str(pointer_identity["md5"]),
            "dataset_sha256": str(marker["dataset_sha256"]),
            "qa": str(qa_path),
            "qa_sha256": str(marker["qa_sha256"]),
            "published_at": datetime.now(UTC).isoformat(),
        }
        commit_staging.mkdir()
        shutil.copy2(pointer, commit_staging / dataset_pointer)
        write_json(commit_staging / "receipt.json", receipt)
        commit_staging.rename(destination)
    except Exception:
        shutil.rmtree(commit_staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch_staging, ignore_errors=True)
    receipt_path = destination / "receipt.json"
    write_json(
        run_root / ".state/preprocessed_published.done",
        {"receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)},
    )
    return receipt_path


def resolve_best_trial_artifacts(
    config_path: Path,
    run_root: Path,
    journal_path: Path,
    summary_path: Path,
    best_config_path: Path,
) -> dict[str, Any]:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    config = load_yaml(config_path)
    study_name = str(config["optuna"]["study_name"])
    study = optuna.load_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(str(journal_path))),
    )
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError("Optuna study has no completed trials")
    best = study.best_trial
    state_counts = Counter(trial.state.name for trial in study.trials)

    trials_root = run_root / "optuna/trials"
    expected_trial_dir = trial_directory(trials_root, study_name, best.number)
    recorded_trial_dir = best.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(recorded_trial_dir, str) or not recorded_trial_dir:
        raise ValueError(
            f"Best trial {best.number} has no {TRIAL_DIR_ATTR!r} user attribute"
        )
    if Path(recorded_trial_dir).resolve() != expected_trial_dir:
        raise ValueError(
            f"Best trial {best.number} recorded unexpected directory: "
            f"{recorded_trial_dir}; expected {expected_trial_dir}"
        )
    if not expected_trial_dir.is_dir() or expected_trial_dir.is_symlink():
        raise FileNotFoundError(
            f"Best trial directory is missing or unsafe: {expected_trial_dir}"
        )
    training_dynamics = load_training_dynamics_identity(expected_trial_dir)
    if (
        best.user_attrs.get(TRAINING_DYNAMICS_MANIFEST_ATTR)
        != training_dynamics["manifest"]
        or best.user_attrs.get(TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR)
        != training_dynamics["manifest_sha256"]
        or best.user_attrs.get(TRAINING_DYNAMICS_STATUS_ATTR)
        != training_dynamics["status"]
    ):
        raise ValueError(
            f"Best trial {best.number} training dynamics identity mismatch"
        )

    expected_checkpoint = best_model_path(expected_trial_dir)
    recorded_checkpoint = best.user_attrs.get(BEST_MODEL_PATH_ATTR)
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint:
        raise ValueError(
            f"Best trial {best.number} has no {BEST_MODEL_PATH_ATTR!r} user attribute"
        )
    if expected_checkpoint.is_symlink():
        raise ValueError(
            f"Best trial checkpoint must not be a symlink: {expected_checkpoint}"
        )
    resolved_checkpoint = expected_checkpoint.resolve(strict=True)
    if Path(recorded_checkpoint).resolve(strict=True) != resolved_checkpoint:
        raise ValueError(
            f"Best trial {best.number} recorded unexpected checkpoint: "
            f"{recorded_checkpoint}; expected {expected_checkpoint}"
        )

    recorded_sha256 = best.user_attrs.get(BEST_MODEL_SHA256_ATTR)
    observed_sha256 = sha256_file(resolved_checkpoint)
    if recorded_sha256 != observed_sha256:
        raise ValueError(
            f"Best trial {best.number} checkpoint SHA-256 mismatch: "
            f"recorded={recorded_sha256!r}, observed={observed_sha256}"
        )

    trial_config_path = expected_trial_dir / "config.yaml"
    if not trial_config_path.is_file() or trial_config_path.is_symlink():
        raise FileNotFoundError(
            f"Best trial effective config is missing or unsafe: {trial_config_path}"
        )
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trial_config_path, best_config_path)

    summary = {
        "study_name": study.study_name,
        "journal": str(journal_path),
        "trial_count": len(study.trials),
        "state_counts": dict(sorted(state_counts.items())),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
        "selected_trial_dir": str(expected_trial_dir),
        "selected_checkpoint": str(resolved_checkpoint),
        "selected_checkpoint_sha256": observed_sha256,
        "training_dynamics": training_dynamics,
        "best_config": str(best_config_path),
    }
    write_json(summary_path, summary)
    return {
        "study": study,
        "best_trial": best,
        "trial_dir": expected_trial_dir,
        "checkpoint": resolved_checkpoint,
        "checkpoint_sha256": observed_sha256,
        "training_dynamics": training_dynamics,
        "trial_config": trial_config_path,
        "summary": summary,
    }


def evaluate_best(
    checkpoint_path: Path,
    output_dir: Path,
    split: str,
    overrides: list[str],
    granular: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{split}_summary.json"
    node_metrics_path = output_dir / f"{split}_node_metrics.csv"
    per_head_metrics_path = output_dir / f"{split}_node_metrics_per_head.csv"
    granular_path = output_dir / f"{split}_granular.parquet"
    command = [
        sys.executable,
        "-m",
        "cli",
        "eval",
        "epiforecaster",
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split,
        "--output",
        str(output_dir),
        "--node-metrics-csv",
        str(node_metrics_path),
        "--summary-json",
        str(summary_path),
    ]
    for override in overrides:
        command.extend(["--override", override])
    if granular:
        command.append("--granular")
    _run(command)
    required_artifacts = [summary_path, node_metrics_path, per_head_metrics_path]
    if granular:
        required_artifacts.append(granular_path)
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Evaluation completed without required artifacts: {rendered}"
        )

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {".done", "evaluation_manifest.json"}
    )
    evaluation_manifest = output_dir / "evaluation_manifest.json"
    write_json(
        evaluation_manifest,
        {
            "schema_version": 1,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "split": split,
            "granular": granular,
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                    "required": path in required_artifacts,
                }
                for path in artifact_paths
            ],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    marker = output_dir / ".done"
    write_json(
        marker,
        {
            "evaluation_manifest": str(evaluation_manifest),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return evaluation_manifest


def evaluate_research_model(
    config_path: Path,
    run_root: Path,
    checkpoint_path: Path,
) -> Path:
    """Evaluate development roles and publish one complete research attempt."""
    import torch
    from data.research_roles import load_role_manifest
    from evaluation.research_pipeline import write_research_outputs
    from models.configs import EpiForecasterConfig
    from scripts.workflow.run_config import RESEARCH_OUTPUT_FILES

    workflow_config = load_yaml(config_path)
    validate_workflow_contract(workflow_config)
    evaluation = workflow_config["evaluation"]
    overrides = [str(value) for value in evaluation.get("overrides", [])]
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = EpiForecasterConfig.from_dict(saved["config"])
    original_joint = config.training.loss.joint
    config = EpiForecasterConfig.apply_overrides(config, overrides)
    if config.training.loss.joint != original_joint:
        raise ValueError(
            "Research evaluation cannot change the trained observation objective"
        )
    if (
        config.training.split_strategy != "research"
        or config.training.research_evaluation_role != "calibration"
        or config.training.research_confirmation_enabled
    ):
        raise ValueError(
            "Development research requires protected selection/calibration roles"
        )
    role_path = config.training.research_role_manifest_path
    if role_path is None:
        raise ValueError("Research evaluation requires a role manifest")
    manifest, role_hash = load_role_manifest(role_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    identity = {
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_hash,
        "role_manifest_sha256": role_hash,
    }
    attempts = run_root / ".state/attempts/research_evaluation"
    destination = run_root / "outputs/evaluation"
    if restore_attempt(attempts, identity, run_root):
        return destination / "evaluation_manifest.json"
    attempt = new_attempt(attempts)
    output = attempt / "outputs/evaluation"
    selection_manifest = evaluate_best(checkpoint_path, output, "val", overrides, True)
    shutil.copy2(selection_manifest, output / "selection_evaluation_manifest.json")
    evaluate_best(checkpoint_path, output / "calibration", "test", overrides, True)
    write_research_outputs(
        config=config,
        manifest=manifest,
        checkpoint_sha256=checkpoint_hash,
        selection_path=output / "val_granular.parquet",
        calibration_path=output / "calibration/test_granular.parquet",
        output_dir=output,
    )
    required = {
        "val_summary.json",
        "val_node_metrics.csv",
        "val_node_metrics_per_head.csv",
        "val_granular.parquet",
        *RESEARCH_OUTPUT_FILES,
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise FileNotFoundError(f"Research evaluation is incomplete: {missing}")
    artifacts = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path not in {selection_manifest, output / ".done"}
    )
    write_json(
        selection_manifest,
        {
            "schema_version": 1,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "split": "val",
            "granular": True,
            "research_outputs": True,
            "role_manifest_sha256": role_hash,
            "data_roles": {"val": "selection", "test": "calibration"},
            "artifacts": [
                {
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                    "required": str(path.relative_to(output)) in required,
                }
                for path in artifacts
            ],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    write_json(
        output / ".done",
        {
            "evaluation_manifest": str(destination / "evaluation_manifest.json"),
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    complete_attempt(
        attempt,
        identity,
        {
            destination / path.relative_to(output): path
            for path in output.rglob("*")
            if path.is_file()
        },
        run_root,
    )
    return destination / "evaluation_manifest.json"


def evaluate_configured_model(
    config_path: Path,
    run_root: Path,
    checkpoint_path: Path,
) -> Path:
    """Evaluate one explicitly selected checkpoint as a standalone DAG entity."""
    config = load_yaml(config_path)
    evaluation = config["evaluation"]
    if checkpoint_path.is_symlink():
        raise ValueError(
            f"Evaluation checkpoint must be a regular non-symlink file: {checkpoint_path}"
        )
    resolved_checkpoint = checkpoint_path.resolve(strict=True)
    if not resolved_checkpoint.is_file():
        raise ValueError(
            f"Evaluation checkpoint must be a regular file: {checkpoint_path}"
        )
    if str(config["workflow"]["kind"]) == "evaluate":
        manifest = load_yaml(run_root / "manifest.yaml")
        evaluation_input = manifest.get("evaluation_input")
        if not isinstance(evaluation_input, dict):
            raise ValueError("Run manifest is missing the pinned evaluation input")
        expected_relative = Path(str(evaluation_input["pinned_checkpoint_relative"]))
        if expected_relative.is_absolute() or ".." in expected_relative.parts:
            raise ValueError(
                "Pinned evaluation checkpoint must be run-relative: "
                f"{expected_relative}"
            )
        expected_path = run_root / expected_relative
        if expected_path.is_symlink():
            raise ValueError(
                "Pinned evaluation checkpoint must be a regular non-symlink file: "
                f"{expected_path}"
            )
        expected_resolved = expected_path.resolve(strict=True)
        if not expected_resolved.is_file():
            raise ValueError(
                f"Pinned evaluation checkpoint must be a regular file: {expected_path}"
            )
        if resolved_checkpoint != expected_resolved:
            raise ValueError(
                "Evaluation-only workflow must consume its pinned checkpoint: "
                f"expected {expected_path}, got {resolved_checkpoint}"
            )
        expected_sha256 = str(evaluation_input["checkpoint_sha256"])
        if sha256_file(resolved_checkpoint) != expected_sha256:
            raise ValueError("Pinned evaluation checkpoint failed SHA-256 validation")
    if evaluation.get("research_outputs", False):
        return evaluate_research_model(config_path, run_root, resolved_checkpoint)
    return evaluate_best(
        resolved_checkpoint,
        run_root / "outputs/evaluation",
        str(evaluation["split"]),
        [str(value) for value in evaluation.get("overrides", [])],
        bool(evaluation.get("granular", False)),
    )


def finalize_ablation(
    config_path: Path,
    run_root: Path,
    journal_path: Path,
    study_name: str | None = None,
) -> Path:
    """Validate a completed ablation queue and generate its aggregate reports."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    from scripts.hpo.ablation_worker import is_study_complete, load_campaign
    from evaluation.ablation_probabilistic_analysis import campaign_analysis_spec

    config = load_yaml(config_path)
    if study_name is None:
        ablation = config["ablation"]
        output_dir = run_root / "outputs/ablation_analysis"
        stale_marker = run_root / ".state/workers_complete.done"
        marker = run_root / ".state/ablation.done"
    else:
        studies = config["ablation_suite"]["studies"]
        ablation = next(
            (study for study in studies if study["study_name"] == study_name), None
        )
        if ablation is None:
            raise ValueError(f"Unknown ablation suite study: {study_name}")
        output_dir = run_root / "outputs/ablation_analysis" / study_name
        stale_marker = run_root / ".state/ablation_suite" / f"{study_name}.workers.done"
        marker = run_root / ".state/ablation_suite" / f"{study_name}.done"
    smoketest = bool(ablation.get("smoketest", False))
    study = optuna.load_study(
        study_name=str(ablation["study_name"]),
        storage=JournalStorage(JournalFileBackend(str(journal_path))),
    )
    campaign = load_campaign(Path(str(ablation["campaign"])))
    if not is_study_complete(study, campaign, smoketest=smoketest):
        state_counts = Counter(trial.state.name for trial in study.trials)
        # The worker rule's completion marker is meaningless while the identity
        # matrix is incomplete; drop it so the next attach re-runs workers and
        # queue initialization re-enqueues the missing identities.
        if stale_marker.exists():
            stale_marker.unlink()
            print(
                f"Removed stale workers completion marker: {stale_marker}",
                file=sys.stderr,
            )
        raise RuntimeError(
            "Ablation workers exited before the complete campaign identity matrix was "
            f"terminal: {dict(sorted(state_counts.items()))}"
        )

    campaign_id = campaign.campaign_id
    training_dir = run_root / "outputs/training" / campaign_id
    command = [
        sys.executable,
        "-m",
        "scripts.analysis.analyze_ablations",
        "--training-dir",
        str(training_dir),
        "--output-dir",
        str(output_dir),
        "--campaign",
        str(campaign.path),
        "--journal",
        str(journal_path),
        "--study-name",
        study.study_name,
    ]
    reuse_evidence_dir = ablation.get("reuse_evidence_dir")
    if reuse_evidence_dir is not None:
        command.extend(["--reuse-evidence-dir", str(reuse_evidence_dir)])
    if smoketest:
        command.append("--smoketest")
    _run(command)

    if campaign_analysis_spec(campaign) is not None:
        required_artifacts = [
            output_dir / "campaign_completeness_manifest.csv",
            output_dir / "condition_scorecard.csv",
        ]
    else:
        required_artifacts = [
            output_dir / "seed_coverage_report.csv",
            output_dir / "ablation_run_context.csv",
            output_dir / "ablation_metrics_aggregated.csv",
        ]
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Ablation analysis completed without required artifacts: {rendered}"
        )
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "ablation_manifest.json"
    )
    ablation_manifest = output_dir / "ablation_manifest.json"
    write_json(
        ablation_manifest,
        {
            "schema_version": 1,
            "study_name": study.study_name,
            "campaign_id": campaign_id,
            "campaign_sha256": campaign.sha256,
            "journal": str(journal_path),
            "trial_count": len(study.trials),
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                    "required": path in required_artifacts,
                }
                for path in artifact_paths
            ],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    write_json(
        marker,
        {
            "ablation_manifest": str(ablation_manifest),
            "study_name": study.study_name,
            "campaign_sha256": campaign.sha256,
            "trial_count": len(study.trials),
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return marker


def finalize_ablation_suite(config_path: Path, run_root: Path) -> Path:
    """Bind all completed campaign analyses to one suite manifest."""
    config = load_yaml(config_path)
    studies = config["ablation_suite"]["studies"]
    entries = []
    for study in studies:
        name = str(study["study_name"])
        path = run_root / "outputs/ablation_analysis" / name / "ablation_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Ablation suite study is incomplete: {path}")
        manifest = json.loads(path.read_text())
        if manifest["study_name"] != name:
            raise ValueError(f"Ablation suite study identity mismatch: {path}")
        entries.append(
            {
                "study_name": name,
                "campaign_id": manifest["campaign_id"],
                "campaign_sha256": manifest["campaign_sha256"],
                "manifest": str(path.relative_to(run_root)),
                "manifest_sha256": sha256_file(path),
            }
        )
    output = run_root / "outputs/ablation_analysis/suite_manifest.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "studies": entries,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return output


def complete_workflow(
    config_path: Path,
    run_root: Path,
    terminal_artifact: Path,
) -> Path:
    """Write the common terminal marker after the selected DAG branch succeeds."""
    config = load_yaml(config_path)
    if not terminal_artifact.exists():
        raise FileNotFoundError(
            f"Terminal workflow artifact is missing: {terminal_artifact}"
        )
    marker = run_root / ".state/workflow.done"
    write_json(
        marker,
        {
            "workflow_kind": config["workflow"]["kind"],
            "workflow_target": config["workflow"].get(
                "target", config["workflow"]["kind"]
            ),
            "terminal_artifact": str(terminal_artifact),
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return marker


def delete_losing_trial_checkpoints(
    trials_root: Path,
    selected_trial_dir: Path,
) -> list[str]:
    """Delete checkpoint directories for every deterministic non-winning trial."""
    resolved_root = trials_root.resolve(strict=True)
    resolved_selected = selected_trial_dir.resolve(strict=True)
    resolved_selected.relative_to(resolved_root)

    deleted: list[str] = []
    for trial_dir in sorted(resolved_root.iterdir()):
        if not trial_dir.is_dir() or not re.fullmatch(r"trial-\d{6,}", trial_dir.name):
            continue
        resolved_trial_dir = trial_dir.resolve(strict=True)
        resolved_trial_dir.relative_to(resolved_root)
        if resolved_trial_dir == resolved_selected:
            continue
        checkpoint_dir = resolved_trial_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise ValueError(f"Unsafe trial checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        deleted.append(str(checkpoint_dir))
    return deleted


def select_best_trial(
    config_path: Path,
    run_root: Path,
    journal_path: Path,
) -> Path:
    """Resolve and copy the exact best-trial checkpoint without evaluating it."""
    config = load_yaml(config_path)
    summary_path = run_root / "optuna/study_summary.json"
    best_config_path = run_root / "optuna/best_config.yaml"
    selected_checkpoint_path = run_root / "outputs/best_model.pt"
    selection_path = run_root / "outputs/selection.json"

    selected = resolve_best_trial_artifacts(
        config_path,
        run_root,
        journal_path,
        summary_path,
        best_config_path,
    )
    source_checkpoint = selected["checkpoint"]
    source_sha256 = selected["checkpoint_sha256"]
    selected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, selected_checkpoint_path)
    selected_sha256 = sha256_file(selected_checkpoint_path)
    if selected_sha256 != source_sha256:
        raise ValueError(
            "Selected checkpoint changed while being copied: "
            f"source={source_sha256}, selected={selected_sha256}"
        )

    best_trial = selected["best_trial"]
    write_json(
        selection_path,
        {
            "study_name": config["optuna"]["study_name"],
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            },
            "source_trial_dir": selected["trial_dir"],
            "source_checkpoint": source_checkpoint,
            "source_checkpoint_sha256": source_sha256,
            "selected_checkpoint": selected_checkpoint_path,
            "selected_checkpoint_sha256": selected_sha256,
            "best_config": best_config_path,
            "selected_at": datetime.now(UTC).isoformat(),
        },
    )
    return selection_path


def cleanup_study_checkpoints(config_path: Path, run_root: Path) -> Path:
    """Delete non-winning HPO checkpoints after evaluation and promotion."""
    config = load_yaml(config_path)
    promoted = run_root / ".state/promoted.done"
    if not promoted.is_file():
        raise FileNotFoundError(
            f"HPO checkpoint cleanup requires successful promotion: {promoted}"
        )
    selection_path = run_root / "outputs/selection.json"
    selection = json.loads(selection_path.read_text())
    selected_trial_dir = Path(str(selection["source_trial_dir"]))
    selected_checkpoint = Path(str(selection["selected_checkpoint"]))
    expected_sha256 = str(selection["selected_checkpoint_sha256"])
    if sha256_file(selected_checkpoint) != expected_sha256:
        raise ValueError("Selected checkpoint changed before HPO cleanup")
    trials_root = run_root / "optuna/trials" / str(config["optuna"]["study_name"])
    deleted = delete_losing_trial_checkpoints(trials_root, selected_trial_dir)
    marker = run_root / ".state/checkpoints_cleaned.done"
    write_json(
        marker,
        {
            "selection": str(selection_path),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": expected_sha256,
            "deleted_losing_checkpoint_dirs": deleted,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return marker


def _validated_regular_file(path: Path, root: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return resolved


def _promotion_integrity(run_root: Path) -> dict[str, str]:
    """Validate selected-model and evaluation evidence before publication."""
    selection_path = _validated_regular_file(
        run_root / "outputs/selection.json", run_root, label="Selection record"
    )
    selection = json.loads(selection_path.read_text())
    selected_checkpoint = _validated_regular_file(
        Path(str(selection["selected_checkpoint"])),
        run_root,
        label="Selected checkpoint",
    )
    expected_checkpoint = (run_root / "outputs/best_model.pt").resolve(strict=True)
    if selected_checkpoint != expected_checkpoint:
        raise ValueError(
            "Selection record does not identify the canonical checkpoint: "
            f"{selected_checkpoint}"
        )
    selected_sha256 = sha256_file(selected_checkpoint)
    source_checkpoint = _validated_regular_file(
        Path(str(selection["source_checkpoint"])),
        run_root,
        label="Source trial checkpoint",
    )
    source_sha256 = sha256_file(source_checkpoint)
    recorded_hashes = {
        str(selection["selected_checkpoint_sha256"]),
        str(selection["source_checkpoint_sha256"]),
    }
    if recorded_hashes != {selected_sha256} or source_sha256 != selected_sha256:
        raise ValueError("Selected checkpoint failed promotion SHA-256 validation")

    evaluation_dir = (run_root / "outputs/evaluation").resolve(strict=True)
    evaluation_manifest_path = _validated_regular_file(
        evaluation_dir / "evaluation_manifest.json",
        evaluation_dir,
        label="Evaluation manifest",
    )
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text())
    if str(evaluation_manifest["checkpoint_sha256"]) != selected_sha256:
        raise ValueError("Evaluation manifest does not match the selected checkpoint")
    evaluated_checkpoint = Path(str(evaluation_manifest["checkpoint"]))
    if evaluated_checkpoint.resolve(strict=True) != selected_checkpoint:
        raise ValueError("Evaluation manifest names a different checkpoint")
    artifacts = evaluation_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Evaluation manifest contains no artifacts")

    hashes = {"outputs/best_model.pt": selected_sha256}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Malformed evaluation artifact record")
        relative = Path(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe evaluation artifact path: {relative}")
        artifact_path = _validated_regular_file(
            evaluation_dir / relative,
            evaluation_dir,
            label="Evaluation artifact",
        )
        observed = sha256_file(artifact_path)
        if observed != str(artifact["sha256"]):
            raise ValueError(
                f"Evaluation artifact failed SHA-256 validation: {relative}"
            )
        hashes[f"outputs/evaluation/{relative.as_posix()}"] = observed
    return hashes


def _tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Promotion source tree contains a symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def retain_training_evidence(
    config_path: Path,
    run_root: Path,
    model_root: Path,
) -> Path:
    """Publish one compact, explicitly retained DVC training-evidence pack."""
    config = load_yaml(config_path)
    resolved_run = run_root.resolve(strict=True)
    resolved_model = model_root.resolve(strict=True)
    resolved_model.relative_to(resolved_run)
    training = config["training"]
    experiment_name = _safe_component(
        training["experiment_name"], field="training.experiment_name"
    )
    model_id = _safe_component(training["model_id"], field="training.model_id")
    destination = run_root / "evidence/training" / experiment_name / model_id
    if destination.exists():
        raise FileExistsError(
            "Training evidence destination already exists and will not be "
            f"overwritten: {destination}"
        )

    required = (
        DYNAMICS_ARTIFACT_NAME,
        DYNAMICS_MANIFEST_NAME,
        "training_manifest.json",
        "training_provenance.json",
    )
    selected = [
        _validated_regular_file(
            model_root / name,
            run_root,
            label="Training evidence",
        )
        for name in required
    ]
    selected.extend(
        _validated_regular_file(path, run_root, label="Profiler trace")
        for path in sorted(model_root.glob("*.pt.trace.json"))
    )
    artifact_hashes = {
        path.name: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in selected
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{model_id}.staging-{os.getpid()}"
    artifacts = staging / "artifacts"
    artifacts.mkdir(parents=True)
    try:
        for source in selected:
            shutil.copy2(source, artifacts / source.name)
        write_json(
            artifacts / "evidence_manifest.json",
            {
                "schema_version": 1,
                "run_id": config["run"]["id"],
                "experiment_name": experiment_name,
                "model_id": model_id,
                "artifacts": artifact_hashes,
            },
        )

        remote = Path(str(config["dvc"]["remote"]))
        cache = Path(str(config["dvc"]["cache"]))
        _run(["dvc", "init", "--no-scm"], cwd=staging)
        _run(
            ["dvc", "remote", "add", "--force", "mn5-nfs", str(remote)],
            cwd=staging,
        )
        _run(["dvc", "config", "core.remote", "mn5-nfs"], cwd=staging)
        _run(["dvc", "config", "cache.dir", str(cache)], cwd=staging)
        _run(["dvc", "config", "cache.shared", "group"], cwd=staging)
        _run(["dvc", "config", "cache.type", "symlink"], cwd=staging)
        lock_handle = _with_cache_lock(cache)
        try:
            _run(["dvc", "add", "artifacts"], cwd=staging)
            _run(["dvc", "push", "artifacts.dvc"], cwd=staging)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

        pointer = staging / "artifacts.dvc"
        pointer_identity = dvc_output(pointer)
        if artifacts.is_symlink():
            artifacts.unlink()
        else:
            shutil.rmtree(artifacts)
        retention = {
            "schema_version": 1,
            "run_id": config["run"]["id"],
            "retention_class": "dvc_retained",
            "dvc_pointer": "artifacts.dvc",
            "dvc_md5": pointer_identity["md5"],
            "artifacts": artifact_hashes,
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_json(staging / "retention.json", retention)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    retention_path = destination / "retention.json"
    write_json(
        run_root / ".state/training_evidence_retained.done",
        {
            "retention": str(retention_path),
            "retention_sha256": sha256_file(retention_path),
        },
    )
    return retention_path


def promote_artifacts(config_path: Path, run_root: Path) -> Path:
    config = load_yaml(config_path)
    run_id = str(config["run"]["id"])
    promotion_root = Path(str(config["promotion"]["root"]))
    destination = promotion_root / run_id
    if destination.exists():
        raise FileExistsError(
            f"Promotion destination already exists and will not be overwritten: {destination}"
        )
    integrity_hashes = _promotion_integrity(run_root)
    promotion_root.mkdir(parents=True, exist_ok=True)
    staging = promotion_root / f".{run_id}.staging-{os.getpid()}"
    artifacts = staging / "artifacts"
    artifacts.mkdir(parents=True)

    selected_files = {
        run_root / "manifest.yaml": artifacts / "manifest.yaml",
        run_root / "optuna/study_summary.json": artifacts / "study_summary.json",
        run_root / "optuna/best_config.yaml": artifacts / "best_config.yaml",
        run_root / "outputs/best_model.pt": artifacts / "best_model.pt",
        run_root / "outputs/selection.json": artifacts / "selection.json",
    }
    for source, target in selected_files.items():
        _validated_regular_file(source, run_root, label="Promotion artifact")
        shutil.copy2(source, target)
    data_manifest_source = _validated_regular_file(
        run_root / ".state/data_manifest.json", run_root, label="Data manifest"
    )
    shutil.copy2(data_manifest_source, artifacts / "data_manifest.json")
    dependencies_source = run_root / ".state/dependencies"
    dependencies_hashes = _tree_hashes(dependencies_source)
    evaluation_source = run_root / "outputs/evaluation"
    evaluation_hashes = _tree_hashes(evaluation_source)
    shutil.copytree(dependencies_source, artifacts / "dependencies")
    shutil.copytree(evaluation_source, artifacts / "evaluation")

    for source, target in selected_files.items():
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f"Promotion staging copy changed artifact: {source}")
    if sha256_file(data_manifest_source) != sha256_file(
        artifacts / "data_manifest.json"
    ):
        raise ValueError("Promotion staging changed the data manifest")
    if _tree_hashes(artifacts / "dependencies") != dependencies_hashes:
        raise ValueError("Promotion staging changed dependency contracts")
    if _tree_hashes(artifacts / "evaluation") != evaluation_hashes:
        raise ValueError("Promotion staging changed evaluation artifacts")
    if (
        sha256_file(artifacts / "best_model.pt")
        != integrity_hashes["outputs/best_model.pt"]
    ):
        raise ValueError("Promotion staging changed the selected checkpoint")

    remote = Path(str(config["dvc"]["remote"]))
    cache = Path(str(config["dvc"]["cache"]))
    try:
        _run(["dvc", "init", "--no-scm"], cwd=staging)
        _run(
            ["dvc", "remote", "add", "--force", "mn5-nfs", str(remote)],
            cwd=staging,
        )
        _run(["dvc", "config", "core.remote", "mn5-nfs"], cwd=staging)
        _run(["dvc", "config", "cache.dir", str(cache)], cwd=staging)
        _run(["dvc", "config", "cache.shared", "group"], cwd=staging)
        _run(["dvc", "config", "cache.type", "symlink"], cwd=staging)
        lock_handle = _with_cache_lock(cache)
        try:
            _run(["dvc", "add", "artifacts"], cwd=staging)
            _run(["dvc", "push", "artifacts.dvc"], cwd=staging)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

        promotion_record = {
            "run_id": run_id,
            "destination": str(destination),
            "dvc_pointer": str(destination / "artifacts.dvc"),
            "manifest_identity": load_yaml(run_root / "manifest.yaml")["identity"],
        }
        write_json(staging / "promotion.json", promotion_record)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    marker = run_root / ".state/promoted.done"
    write_json(marker, promotion_record)
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def runtime_paths(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--source-root", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    runtime_paths(prepare)

    stage = subparsers.add_parser("stage-data")
    runtime_paths(stage)

    train = subparsers.add_parser("train")
    runtime_paths(train)

    preprocess = subparsers.add_parser("preprocess")
    runtime_paths(preprocess)

    publish_preprocessed = subparsers.add_parser("publish-preprocessed")
    publish_preprocessed.add_argument("--config", type=Path, required=True)
    publish_preprocessed.add_argument("--run-root", type=Path, required=True)

    crossval_train_fold = subparsers.add_parser("crossval-train-fold")
    runtime_paths(crossval_train_fold)
    crossval_train_fold.add_argument("--fold", type=int, required=True)

    crossval_evaluate_fold = subparsers.add_parser("crossval-evaluate-fold")
    crossval_evaluate_fold.add_argument("--config", type=Path, required=True)
    crossval_evaluate_fold.add_argument("--run-root", type=Path, required=True)
    crossval_evaluate_fold.add_argument("--fold", type=int, required=True)

    finalize_crossval_parser = subparsers.add_parser("finalize-crossval")
    finalize_crossval_parser.add_argument("--config", type=Path, required=True)
    finalize_crossval_parser.add_argument("--run-root", type=Path, required=True)

    dependencies = subparsers.add_parser("dependencies")
    runtime_paths(dependencies)
    dependencies.add_argument("--stage", required=True)
    dependencies.add_argument("--output", type=Path, required=True)
    dependencies.add_argument("--source", type=Path, action="append", default=[])

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--run-root", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)

    select = subparsers.add_parser("select-best")
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--run-root", type=Path, required=True)
    select.add_argument("--journal", type=Path, required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--config", type=Path, required=True)
    promote.add_argument("--run-root", type=Path, required=True)

    retain_evidence = subparsers.add_parser("retain-training-evidence")
    retain_evidence.add_argument("--config", type=Path, required=True)
    retain_evidence.add_argument("--run-root", type=Path, required=True)
    retain_evidence.add_argument("--model-root", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup-study")
    cleanup.add_argument("--config", type=Path, required=True)
    cleanup.add_argument("--run-root", type=Path, required=True)

    finalize_ablation_parser = subparsers.add_parser("finalize-ablation")
    finalize_ablation_parser.add_argument("--config", type=Path, required=True)
    finalize_ablation_parser.add_argument("--run-root", type=Path, required=True)
    finalize_ablation_parser.add_argument("--journal", type=Path, required=True)
    finalize_ablation_parser.add_argument("--study-name")
    finalize_suite_parser = subparsers.add_parser("finalize-ablation-suite")
    finalize_suite_parser.add_argument("--config", type=Path, required=True)
    finalize_suite_parser.add_argument("--run-root", type=Path, required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--config", type=Path, required=True)
    complete.add_argument("--run-root", type=Path, required=True)
    complete.add_argument("--terminal-artifact", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_run(args.config, args.source_root, args.run_root)
    elif args.command == "stage-data":
        stage_dvc(args.config, args.source_root, args.run_root)
    elif args.command == "train":
        train_model(args.config, args.source_root, args.run_root)
    elif args.command == "preprocess":
        run_preprocessing(args.config, args.source_root, args.run_root)
    elif args.command == "publish-preprocessed":
        publish_preprocessed_dataset(args.config, args.run_root)
    elif args.command == "crossval-train-fold":
        train_crossval_fold(args.config, args.source_root, args.run_root, args.fold)
    elif args.command == "crossval-evaluate-fold":
        evaluate_crossval_fold(args.config, args.run_root, args.fold)
    elif args.command == "finalize-crossval":
        finalize_crossval(args.config, args.run_root)
    elif args.command == "dependencies":
        write_dependency_contract(
            args.config,
            args.source_root,
            args.output,
            args.stage,
            args.source,
        )
    elif args.command == "evaluate":
        evaluate_configured_model(args.config, args.run_root, args.checkpoint)
    elif args.command == "select-best":
        select_best_trial(args.config, args.run_root, args.journal)
    elif args.command == "promote":
        promote_artifacts(args.config, args.run_root)
    elif args.command == "retain-training-evidence":
        retain_training_evidence(
            args.config,
            args.run_root,
            args.model_root,
        )
    elif args.command == "cleanup-study":
        cleanup_study_checkpoints(args.config, args.run_root)
    elif args.command == "finalize-ablation":
        finalize_ablation(args.config, args.run_root, args.journal, args.study_name)
    elif args.command == "finalize-ablation-suite":
        finalize_ablation_suite(args.config, args.run_root)
    elif args.command == "complete":
        complete_workflow(
            args.config,
            args.run_root,
            args.terminal_artifact,
        )


if __name__ == "__main__":
    main()
