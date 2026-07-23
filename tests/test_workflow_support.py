from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import optuna
import pytest
import yaml

from scripts.hpo.trial_artifacts import (
    BEST_MODEL_PATH_ATTR,
    BEST_MODEL_SHA256_ATTR,
    TRIAL_DIR_ATTR,
    best_model_path,
    trial_directory,
)
from scripts.workflow import run_config, run_support, tooling_identity


REPO_ROOT = Path(__file__).resolve().parents[1]


def _workflow_shell_source(tmp_path: Path) -> str:
    """Return Bash that sources the dispatcher with an isolated test config."""
    config = tmp_path / "mn5.conf"
    config.write_text(
        "\n".join(
            [
                "COMPUTE_HOST=fake-compute",
                "TRANSFER_HOST=fake-transfer",
                "CONTAINER_NETWORK_HOST=fake-network",
                f"REMOTE_WORKFLOW_RUN_ROOT={tmp_path / 'remote-runs'}",
                f"REMOTE_WORKFLOW_TOOLING_ROOT={tmp_path / 'tooling'}",
                f"REMOTE_CONTAINER_SIF={tmp_path / 'runtime.sif'}",
                f"DVC_REMOTE_PATH={tmp_path / 'dvc-remote'}",
                f"DVC_CACHE_PATH={tmp_path / 'dvc-cache'}",
            ]
        )
        + "\n"
    )
    dispatcher = REPO_ROOT / "scripts/workflow/mn5_workflow.sh"
    return (
        f"export CLUSTER_CONF={shlex.quote(str(config))}\n"
        f"source {shlex.quote(str(dispatcher))}\n"
    )


def _write_pointer(
    source_root: Path,
    relative: str,
    *,
    md5: str,
    output_name: str,
) -> None:
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": md5,
                        "size": 10,
                        "nfiles": 2,
                        "hash": "md5",
                        "path": output_name,
                    }
                ]
            },
            sort_keys=False,
        )
    )


def test_mn5_smoke_uses_short_single_gpu_debug_workload() -> None:
    production = yaml.safe_load((REPO_ROOT / "workflow/config/mn5.yaml").read_text())
    smoke = yaml.safe_load((REPO_ROOT / "workflow/config/mn5_smoke.yaml").read_text())

    assert smoke["workflow"] == production["workflow"]
    assert smoke["dvc"] == production["dvc"]
    assert smoke["promotion"] == production["promotion"]
    assert smoke["optuna"]["base_config"] == production["optuna"]["base_config"]
    assert smoke["optuna"]["workers"] == 1
    assert smoke["optuna"]["trials_per_worker"] == 1
    assert smoke["optuna"]["timeout_s"] == 300
    assert smoke["optuna"]["epochs"] == 1
    assert smoke["optuna"]["max_batches"] == 1
    assert "training.compile=false" in smoke["optuna"]["overrides"]
    assert smoke["evaluation"]["overrides"] == [
        "training.max_batches=1",
        "training.compile=false",
        "training.val_workers=1",
        "training.test_workers=1",
    ]
    assert smoke["resources"]["allocation"] == {
        "account": "bsc08",
        "partition": "acc",
        "qos": "acc_debug",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 20,
        "gres": "gpu:1",
        "runtime": 20,
    }


def test_probabilistic_smoke_reuses_short_debug_shape_with_normal_nll() -> None:
    smoke = yaml.safe_load((REPO_ROOT / "workflow/config/mn5_smoke.yaml").read_text())
    probabilistic = yaml.safe_load(
        (REPO_ROOT / "workflow/config/mn5_probabilistic_smoke.yaml").read_text()
    )

    assert probabilistic["workflow"] == smoke["workflow"]
    assert probabilistic["dvc"] == smoke["dvc"]
    assert probabilistic["evaluation"] == smoke["evaluation"]
    assert probabilistic["resources"] == smoke["resources"]
    for field in (
        "workers",
        "trials_per_worker",
        "timeout_s",
        "epochs",
        "max_batches",
    ):
        assert probabilistic["optuna"][field] == smoke["optuna"][field]
    assert (
        "training.loss.joint.observation_loss=nll"
        in probabilistic["optuna"]["overrides"]
    )
    assert (
        "training.loss.joint.distribution_family=normal"
        in probabilistic["optuna"]["overrides"]
    )


def test_probabilistic_batch_smoke_only_changes_qos() -> None:
    debug = yaml.safe_load(
        (REPO_ROOT / "workflow/config/mn5_probabilistic_smoke.yaml").read_text()
    )
    batch = yaml.safe_load(
        (REPO_ROOT / "workflow/config/mn5_probabilistic_smoke_batch.yaml").read_text()
    )

    debug["resources"]["allocation"]["qos"] = "acc_bscls"
    assert batch == debug


def test_single_training_smoke_composes_training_and_evaluation() -> None:
    smoke = yaml.safe_load(
        (REPO_ROOT / "workflow/config/mn5_train_smoke.yaml").read_text()
    )

    assert smoke["workflow"] == {"kind": "train", "target": "train_evaluate"}
    assert smoke["training"]["max_batches"] == 10
    assert "training.num_workers=1" in smoke["training"]["overrides"]
    assert "training.val_workers=1" in smoke["training"]["overrides"]
    assert "training.test_workers=1" in smoke["training"]["overrides"]
    assert "training.run_final_evaluation=false" in smoke["training"]["overrides"]
    assert "training.compile=false" in smoke["training"]["overrides"]
    assert smoke["evaluation"]["split"] == "test"
    assert smoke["resources"]["allocation"] == {
        "account": "bsc08",
        "partition": "acc",
        "qos": "acc_bscls",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 20,
        "gres": "gpu:1",
        "runtime": 60,
    }


def _workflow_config(run_root: Path, source_root: Path) -> dict:
    resources = {
        "allocation": {
            "account": "bsc08",
            "partition": "acc",
            "qos": "acc_bscls",
            "nodes": 1,
            "tasks": 4,
            "cpus_per_task": 20,
            "gres": "gpu:4",
            "runtime": 60,
        }
    }
    return {
        "run": {
            "id": "test-run",
            "root": str(run_root),
            "source_root": str(source_root),
        },
        "source": {
            "git_sha": "a" * 40,
            "git_branch": "test",
            "dirty": False,
            "git_status": [],
            "dirty_patch_sha256": "b" * 64,
            "source_snapshot_sha256": "c" * 64,
            "source_file_count": 3,
        },
        "container": {
            "sif": "/containers/runtime.sif",
            "digest": f"sha256:{'d' * 64}",
        },
        "workflow": {
            "kind": "hpo",
            "tooling_key": "workflow-key",
            "snakemake_bin": "/tooling/bin/snakemake",
            "snakemake_version": version("snakemake"),
            "execution_mode": "single-node-preallocated",
        },
        "dvc": {
            "remote": str(run_root / "dvc-project/storage"),
            "cache": str(run_root / "dvc-project/cache"),
            "targets": [
                "data/files/raw.zarr.dvc",
                "data/processed/processed.zarr.dvc",
            ],
        },
        "optuna": {
            "study_name": "test-study",
            "base_config": "configs/production_only/train_epifor_mn5_full.yaml",
            "workers": 4,
            "trials_per_worker": 1,
            "timeout_s": 60,
            "epochs": 1,
            "seed": 42,
            "sampler": "random",
            "pruning_start_epoch": 1,
            "search_space": "default",
            "overrides": [],
        },
        "evaluation": {"split": "test", "granular": False, "overrides": []},
        "promotion": {"root": str(run_root / "promoted")},
        "resources": resources,
    }


def _write_published_run_contract(
    run_root: Path,
    config_path: Path,
    config: dict,
) -> str:
    state = run_root / ".state"
    state.mkdir(parents=True, exist_ok=True)
    dispatch = state / "dispatch.tsv"
    dispatch.write_text("schema_version\t1\n")
    dispatch_sha256 = run_support.sha256_file(dispatch)
    (state / "dispatch.tsv.sha256").write_text(f"{dispatch_sha256}  dispatch.tsv\n")
    (state / "published.done").write_text(
        "\n".join(
            [
                "schema_version=1",
                f"run_id={config['run']['id']}",
                f"source_snapshot_sha256={config['source']['source_snapshot_sha256']}",
                f"config_sha256={run_support.sha256_file(config_path)}",
                f"tooling_key={config['workflow']['tooling_key']}",
                "",
            ]
        )
    )
    return dispatch_sha256


def test_source_snapshot_tracks_nonignored_worktree_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("untracked\n")
    (repo / "ignored.txt").write_text("ignored\n")

    identity = run_config.git_identity(repo)

    assert identity["dirty"] is True
    assert identity["files"] == [
        Path(".gitignore"),
        Path("tracked.txt"),
        Path("untracked.txt"),
    ]
    original_digest = identity["source_snapshot_sha256"]
    (repo / "untracked.txt").write_text("changed\n")
    assert run_config.git_identity(repo)["source_snapshot_sha256"] != original_digest


def test_relocatable_bundle_keeps_dvc_pointer_but_excludes_payload(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("data/input.bin\n")
    (repo / "workflow").mkdir()
    (repo / "workflow/Snakefile").write_text("rule all:\n    input: []\n")
    _write_pointer(
        repo,
        "data/input.bin.dvc",
        md5="input-md5",
        output_name="input.bin",
    )
    (repo / "data/input.bin").write_bytes(b"vendored data")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    config = {"dvc": {"targets": ["data/input.bin.dvc"]}}

    paths = run_config.source_files(repo)
    payloads = run_config.validate_code_data_separation(repo, paths, config)

    assert Path("workflow/Snakefile") in paths
    assert Path("data/input.bin.dvc") in paths
    assert Path("data/input.bin") not in paths
    assert payloads == {Path("data/input.bin.dvc"): Path("data/input.bin")}


def test_relocatable_bundle_includes_provenance_only_dvc_pointer(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("data/raw.bin\ndata/processed.bin\n")
    _write_pointer(
        repo,
        "data/raw.bin.dvc",
        md5="raw-md5",
        output_name="raw.bin",
    )
    _write_pointer(
        repo,
        "data/processed.bin.dvc",
        md5="processed-md5",
        output_name="processed.bin",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    config = {
        "dvc": {
            "targets": ["data/processed.bin.dvc"],
            "provenance_targets": ["data/raw.bin.dvc"],
        }
    }

    paths = run_config.source_files(repo)
    payloads = run_config.validate_code_data_separation(repo, paths, config)

    assert payloads == {
        Path("data/processed.bin.dvc"): Path("data/processed.bin"),
        Path("data/raw.bin.dvc"): Path("data/raw.bin"),
    }


def test_relocatable_bundle_rejects_tracked_dvc_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _write_pointer(
        repo,
        "data/input.bin.dvc",
        md5="input-md5",
        output_name="input.bin",
    )
    (repo / "data/input.bin").write_bytes(b"vendored data")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    config = {"dvc": {"targets": ["data/input.bin.dvc"]}}

    with pytest.raises(ValueError, match="DVC payloads must not be included"):
        run_config.validate_code_data_separation(
            repo,
            run_config.source_files(repo),
            config,
        )


def test_workflow_tooling_key_tracks_lockfile_and_bootstrap(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    bootstrap = tmp_path / "remote_tooling.sh"
    lock.write_text("version = 1\n")
    bootstrap.write_text("#!/bin/bash\n")

    initial = tooling_identity.workflow_tooling_key(lock, bootstrap)

    assert initial == tooling_identity.workflow_tooling_key(lock, bootstrap)
    lock.write_text("version = 2\n")
    changed_lock = tooling_identity.workflow_tooling_key(lock, bootstrap)
    assert initial != changed_lock
    bootstrap.write_text("#!/bin/bash\nset -e\n")
    assert changed_lock != tooling_identity.workflow_tooling_key(lock, bootstrap)


def test_dispatch_request_pins_submission_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/test-run"
    source_root = run_root / "source"
    config = _workflow_config(run_root, source_root)
    config["container"] = {
        "sif": "/containers/runtime.sif",
        "digest_file": "/containers/runtime.sif.sha256",
    }
    config_path = tmp_path / "config.yaml"
    dispatch_path = tmp_path / "dispatch.tsv"
    config_path.write_text(yaml.safe_dump(config))

    request = run_config.write_dispatch_request(config_path, dispatch_path)

    assert request["run_id"] == "test-run"
    assert request["run_root"] == str(run_root)
    assert request["source_root"] == str(source_root)
    assert request["workflow_kind"] == "hpo"
    assert request["workflow_target"] == "hpo"
    assert request["tasks"] == "4"
    assert request["cpus_per_task"] == "20"
    assert request["gres"] == "gpu:4"
    assert request["config_sha256"] == run_support.sha256_file(config_path)
    assert dispatch_path.with_suffix(".tsv.sha256").is_file()
    assert "snakemake_arg" not in request


def test_copied_source_snapshot_verifies_after_worktree_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo)
    source = repo / "workflow.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "snapshot"], cwd=repo, check=True)

    snapshot = tmp_path / "snapshot"
    paths = run_config.copy_source_snapshot(repo, snapshot)
    identity = run_config.git_identity(repo, snapshot_root=snapshot, paths=paths)
    source.write_text("value = 2\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"source": {"source_snapshot_sha256": identity["source_snapshot_sha256"]}}
        )
    )
    file_list = tmp_path / "source-files.bin"
    file_list.write_bytes(
        b"".join(os.fsencode(path.as_posix()) + b"\0" for path in paths)
    )

    observed = run_config.verify_snapshot(config_path, snapshot, file_list)

    assert observed == identity["source_snapshot_sha256"]
    assert (snapshot / "workflow.py").read_text() == "value = 1\n"


def test_prepare_run_writes_provenance_identity(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/test-run"
    source_root = run_root / "source"
    _write_pointer(
        source_root,
        "data/files/raw.zarr.dvc",
        md5="raw.dir",
        output_name="raw.zarr",
    )
    _write_pointer(
        source_root,
        "data/processed/processed.zarr.dvc",
        md5="processed.dir",
        output_name="processed.zarr",
    )
    config = _workflow_config(run_root, source_root)
    _write_pointer(
        source_root,
        "data/files/upstream.zarr.dvc",
        md5="upstream.dir",
        output_name="upstream.zarr",
    )
    config["dvc"]["provenance_targets"] = ["data/files/upstream.zarr.dvc"]
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    dispatch_sha256 = _write_published_run_contract(run_root, config_path, config)
    config_path.chmod(0o444)
    source_root.chmod(0o555)

    try:
        manifest_path = run_support.prepare_run(config_path, source_root, run_root)
    finally:
        source_root.chmod(0o755)

    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["identity"] == {
        "git_sha": "a" * 40,
        "source_snapshot_sha256": "c" * 64,
        "dvc_hashes": {
            "data/files/raw.zarr.dvc": "raw.dir",
            "data/files/upstream.zarr.dvc": "upstream.dir",
            "data/processed/processed.zarr.dvc": "processed.dir",
        },
        "container_digest": f"sha256:{'d' * 64}",
        "workflow_tooling_key": "workflow-key",
        "snakemake_config_sha256": run_support.sha256_file(config_path),
        "dispatch_request_sha256": dispatch_sha256,
        "workflow_kind": "hpo",
        "workflow_target": "hpo",
        "study_name": "test-study",
    }
    assert manifest["workflow"]["version"] == version("snakemake")
    assert manifest["workflow"]["installed_location"]
    assert manifest["workflow"]["execution_mode"] == "single-node-preallocated"
    assert manifest["dvc"]["targets"]["data/files/raw.zarr.dvc"]["role"] == (
        "materialized"
    )
    assert manifest["dvc"]["targets"]["data/files/upstream.zarr.dvc"]["role"] == (
        "provenance"
    )
    assert (run_root / ".state/prepared.done").is_file()
    assert not manifest_path.stat().st_mode & 0o222
    assert not (run_root / ".state/prepared.done").stat().st_mode & 0o222


def test_prepare_run_pins_evaluation_checkpoint_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/evaluate"
    source_root = run_root / "source"
    _write_pointer(
        source_root,
        "data/files/raw.zarr.dvc",
        md5="raw.dir",
        output_name="raw.zarr",
    )
    _write_pointer(
        source_root,
        "data/processed/processed.zarr.dvc",
        md5="processed.dir",
        output_name="processed.zarr",
    )
    external_checkpoint = tmp_path / "external/best_model.pt"
    external_checkpoint.parent.mkdir()
    external_checkpoint.write_bytes(b"original checkpoint")
    config = _workflow_config(run_root, source_root)
    config["workflow"].update({"kind": "evaluate", "target": "evaluate"})
    config["evaluation"]["checkpoint"] = str(external_checkpoint)
    config.pop("optuna")
    config.pop("promotion")
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_published_run_contract(run_root, config_path, config)
    config_path.chmod(0o444)
    source_root.chmod(0o555)

    try:
        manifest_path = run_support.prepare_run(config_path, source_root, run_root)
    finally:
        source_root.chmod(0o755)

    pinned = run_root / "inputs/evaluation_checkpoint.pt"
    expected_hash = run_support.sha256_file(pinned)
    manifest = yaml.safe_load(manifest_path.read_text())
    assert pinned.read_bytes() == b"original checkpoint"
    assert not pinned.stat().st_mode & 0o222
    assert manifest["evaluation_input"]["pinned_checkpoint"] == str(pinned)
    assert manifest["evaluation_input"]["checkpoint_sha256"] == expected_hash

    external_checkpoint.write_bytes(b"later mutation")
    captured: dict[str, Path] = {}

    def fake_evaluate(checkpoint_path, output_dir, split, overrides, granular):
        captured["checkpoint"] = checkpoint_path
        return output_dir / "evaluation_manifest.json"

    monkeypatch.setattr(run_support, "evaluate_best", fake_evaluate)
    run_support.evaluate_configured_model(config_path, run_root, pinned)
    assert captured["checkpoint"] == pinned
    assert run_support.sha256_file(pinned) == expected_hash


def test_production_workflow_tracks_required_region_embedding() -> None:
    workflow = yaml.safe_load((REPO_ROOT / "workflow/config/mn5.yaml").read_text())
    training = yaml.safe_load(
        (REPO_ROOT / "configs/production_only/train_epifor_mn5_full.yaml").read_text()
    )
    pointer_relative = Path("outputs/region_embeddings/region_embeddings.pt.dvc")
    pointer = REPO_ROOT / pointer_relative

    assert pointer_relative.as_posix() in workflow["dvc"]["targets"]
    output = run_support.dvc_output(pointer)
    assert pointer_relative.parent / output["path"] == Path(
        training["data"]["region2vec_path"]
    )
    assert output["md5"] == "5205439ed1bd8a079aa57aed37507654"


def test_production_source_bundle_contains_pipeline_without_dvc_payloads() -> None:
    workflow = yaml.safe_load((REPO_ROOT / "workflow/config/mn5.yaml").read_text())
    paths = run_config.source_files(REPO_ROOT)

    payloads = run_config.validate_code_data_separation(REPO_ROOT, paths, workflow)

    assert Path("workflow/Snakefile") in paths
    assert Path("scripts/workflow/mn5_container_exec.sh") in paths
    assert Path("scripts/workflow/remote_run.sh") in paths
    assert Path("scripts/workflow/remote_tooling.sh") in paths
    assert set(payloads) == {
        Path("data/files/mobility.zarr.dvc"),
        Path("data/processed/real_with_id.zarr.dvc"),
        Path("outputs/region_embeddings/region_embeddings.pt.dvc"),
    }
    assert not any(
        path == payload or payload in path.parents
        for path in paths
        for payload in payloads.values()
    )


def test_mn5_profile_uses_one_preallocated_acc_node() -> None:
    profile = yaml.safe_load(
        (REPO_ROOT / "workflow/profiles/mn5/config.yaml").read_text()
    )
    workflow = yaml.safe_load((REPO_ROOT / "workflow/config/mn5.yaml").read_text())
    coordinator = (
        REPO_ROOT / "scripts/workflow/snakemake_coordinator.sbatch"
    ).read_text()
    launcher = (REPO_ROOT / "scripts/workflow/launch_parallel_workers.sh").read_text()

    assert profile["executor"] == "local"
    assert profile["jobs"] == 1
    assert profile["cores"] == 80
    assert not any(key.startswith("slurm-") for key in profile)
    allocation = workflow["resources"]["allocation"]
    assert allocation == {
        "account": "bsc08",
        "partition": "acc",
        "qos": "acc_bscls",
        "nodes": 1,
        "tasks": 4,
        "cpus_per_task": 20,
        "gres": "gpu:4",
        "runtime": 420,
    }
    assert "#SBATCH --partition=acc" in coordinator
    assert "#SBATCH --ntasks=4" in coordinator
    assert "#SBATCH --cpus-per-task=20" in coordinator
    assert "#SBATCH --gres=gpu:4" in coordinator
    assert '--ntasks="$WORKERS"' in launcher
    assert '--cpus-per-task="$CPUS_PER_TASK"' in launcher
    assert "--gpus-per-task=1" in launcher
    assert "--kill-on-bad-exit=1" in launcher
    assert (
        "for optional_host_path in /opt/.snoopy/lib /opt/EAR/current"
        in coordinator
    )
    assert 'if [ -e "$optional_host_path" ]' in coordinator
    assert "--mem" not in coordinator
    assert "--mem" not in launcher


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nodes", 2, "nodes=2"),
        ("tasks", 5, "tasks=5"),
        ("cpus_per_task", 19, "cpus_per_task=19"),
        ("gres", "gpu:3", "gres='gpu:3'"),
    ],
)
def test_mn5_allocation_rejects_noncanonical_topology(
    tmp_path: Path,
    field: str,
    value: int | str,
    message: str,
) -> None:
    config = _workflow_config(tmp_path / "run", tmp_path / "run/source")
    config["resources"]["allocation"][field] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        run_config.validate_mn5_allocation(config)


def test_mn5_allocation_scales_with_study_worker_count(tmp_path: Path) -> None:
    config = _workflow_config(tmp_path / "run", tmp_path / "run/source")
    config["optuna"]["workers"] = 2
    config["resources"]["allocation"].update(tasks=2, gres="gpu:2")

    allocation = run_config.validate_mn5_allocation(config)

    assert allocation["tasks"] == 2
    assert allocation["gres"] == "gpu:2"


def test_single_training_requires_one_task_and_one_gpu(tmp_path: Path) -> None:
    config = _workflow_mode_config("train", tmp_path / "run")
    config["resources"]["allocation"].update(tasks=4, gres="gpu:4")

    with pytest.raises(ValueError, match="tasks=4"):
        run_config.validate_mn5_allocation(config)


def test_stage_dvc_records_bind_roots(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "runs/test-run"
    source_root = run_root / "source"
    _write_pointer(
        source_root,
        "data/files/raw.zarr.dvc",
        md5="raw.dir",
        output_name="raw.zarr",
    )
    _write_pointer(
        source_root,
        "data/processed/processed.zarr.dvc",
        md5="processed.dir",
        output_name="processed.zarr",
    )
    config = _workflow_config(run_root, source_root)
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def fake_run(command: list[str], *, cwd: Path | None = None) -> None:
        assert cwd is not None
        if command[1:3] == ["init", "--no-scm"]:
            (cwd / ".dvc").mkdir()
        elif command[1] == "pull":
            (cwd / "data/files/raw.zarr").mkdir(parents=True)
            (cwd / "data/processed/processed.zarr").mkdir(parents=True)

    monkeypatch.setattr(run_support, "_run", fake_run)

    outputs = run_support.stage_dvc(config_path, source_root, run_root)

    assert outputs == [
        Path("data/files/raw.zarr"),
        Path("data/processed/processed.zarr"),
    ]
    assert (run_root / ".state/dvc_bind_roots.txt").read_text().splitlines() == [
        "data/files",
        "data/processed",
    ]
    data_manifest = json.loads((run_root / ".state/data_manifest.json").read_text())
    assert data_manifest["targets"] == {
        "data/files/raw.zarr.dvc": {
            "md5": "raw.dir",
            "output": "data/files/raw.zarr",
        },
        "data/processed/processed.zarr.dvc": {
            "md5": "processed.dir",
            "output": "data/processed/processed.zarr",
        },
    }


def _complete_trial(
    study: optuna.Study,
    run_root: Path,
    *,
    value: float,
    checkpoint_bytes: bytes,
) -> tuple[Path, Path]:
    trial = study.ask()
    trial_dir = trial_directory(
        run_root / "optuna/trials",
        study.study_name,
        trial.number,
    )
    checkpoint = best_model_path(trial_dir)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
    (trial_dir / "config.yaml").write_text(
        yaml.safe_dump({"trial_number": trial.number, "value": value})
    )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(BEST_MODEL_PATH_ATTR, str(checkpoint))
    trial.set_user_attr(
        BEST_MODEL_SHA256_ATTR,
        run_support.sha256_file(checkpoint),
    )
    study.tell(trial, value)
    return trial_dir, checkpoint


@pytest.mark.parametrize("study_name", ["", ".", "..", "nested/study"])
def test_trial_directory_rejects_unsafe_study_names(
    tmp_path: Path,
    study_name: str,
) -> None:
    with pytest.raises(ValueError, match="one path component"):
        trial_directory(tmp_path, study_name, 0)


def test_resolve_best_trial_uses_recorded_checkpoint_and_config(tmp_path: Path) -> None:
    journal = tmp_path / "study.journal"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))
    )
    study = optuna.create_study(
        study_name="test-study", storage=storage, direction="minimize"
    )
    run_root = tmp_path / "run"
    best_trial_dir, best_checkpoint = _complete_trial(
        study,
        run_root,
        value=0.5,
        checkpoint_bytes=b"best",
    )
    _complete_trial(study, run_root, value=1.0, checkpoint_bytes=b"loser")
    config = _workflow_config(run_root, REPO_ROOT)
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    summary_path = run_root / "optuna/study_summary.json"
    best_config = run_root / "optuna/best_config.yaml"

    selected = run_support.resolve_best_trial_artifacts(
        config_path,
        run_root,
        journal,
        summary_path,
        best_config,
    )

    effective = yaml.safe_load(best_config.read_text())
    summary = selected["summary"]
    assert summary["best_trial"]["number"] == 0
    assert selected["trial_dir"] == best_trial_dir
    assert selected["checkpoint"] == best_checkpoint
    assert summary["selected_checkpoint_sha256"] == run_support.sha256_file(
        best_checkpoint
    )
    assert effective == {"trial_number": 0, "value": 0.5}


def test_resolve_best_trial_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    journal = tmp_path / "study.journal"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))
    )
    study = optuna.create_study(
        study_name="test-study", storage=storage, direction="minimize"
    )
    run_root = tmp_path / "run"
    _, checkpoint = _complete_trial(
        study,
        run_root,
        value=0.5,
        checkpoint_bytes=b"original",
    )
    checkpoint.write_bytes(b"tampered")
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(_workflow_config(run_root, REPO_ROOT)))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_support.resolve_best_trial_artifacts(
            config_path,
            run_root,
            journal,
            run_root / "optuna/study_summary.json",
            run_root / "optuna/best_config.yaml",
        )


def test_hpo_selection_and_post_promotion_cleanup_are_separate(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "study.journal"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))
    )
    study = optuna.create_study(
        study_name="test-study", storage=storage, direction="minimize"
    )
    run_root = tmp_path / "run"
    best_trial_dir, best_checkpoint = _complete_trial(
        study,
        run_root,
        value=0.25,
        checkpoint_bytes=b"selected model",
    )
    losing_trial_dir, _ = _complete_trial(
        study,
        run_root,
        value=0.75,
        checkpoint_bytes=b"losing model",
    )
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(_workflow_config(run_root, REPO_ROOT)))
    selection_path = run_support.select_best_trial(config_path, run_root, journal)

    assert selection_path == run_root / "outputs/selection.json"
    assert best_model_path(best_trial_dir).is_file()
    assert (losing_trial_dir / "checkpoints").is_dir()
    selected_checkpoint = run_root / "outputs/best_model.pt"
    assert selected_checkpoint.read_bytes() == best_checkpoint.read_bytes()
    selection = json.loads(selection_path.read_text())
    assert selection["source_checkpoint"] == str(best_checkpoint)
    assert selection["selected_checkpoint_sha256"] == run_support.sha256_file(
        best_checkpoint
    )

    promoted = run_root / ".state/promoted.done"
    run_support.write_json(promoted, {"promoted": True})
    cleanup = run_support.cleanup_study_checkpoints(config_path, run_root)

    assert best_model_path(best_trial_dir).is_file()
    assert not (losing_trial_dir / "checkpoints").exists()
    cleanup_record = json.loads(cleanup.read_text())
    assert cleanup_record["deleted_losing_checkpoint_dirs"] == [
        str(losing_trial_dir / "checkpoints")
    ]


def test_finalize_ablation_runs_analysis_only_after_complete_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "ablation.journal"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))
    )
    study = optuna.create_study(
        study_name="ablation-study",
        storage=storage,
        direction="minimize",
    )
    ablations = ["baseline", "mobility:off"]
    distributions = {
        "ablation": optuna.distributions.CategoricalDistribution(ablations),
        "seed": optuna.distributions.CategoricalDistribution([42]),
    }
    for ablation in ablations:
        study.add_trial(
            optuna.trial.create_trial(
                params={"ablation": ablation, "seed": 42},
                distributions=distributions,
                value=1.0,
            )
        )

    run_root = tmp_path / "run"
    config = {
        "ablation": {
            "study_name": "ablation-study",
            "campaign_id": "campaign-a",
            "seeds": [42],
            "ablations": "baseline mobility:off",
        }
    }
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    captured: dict[str, list[str]] = {}

    def fake_run(command, *, cwd=None, env=None):
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "seed_coverage_report.csv").write_text("seed\n")
        (output_dir / "ablation_run_context.csv").write_text("run\n")
        (output_dir / "ablation_metrics_aggregated.csv").write_text("metric\n")

    monkeypatch.setattr(run_support, "_run", fake_run)

    marker = run_support.finalize_ablation(config_path, run_root, journal)

    metadata = json.loads(marker.read_text())
    assert metadata["trial_count"] == 2
    manifest = json.loads(
        (run_root / "outputs/ablation_analysis/ablation_manifest.json").read_text()
    )
    assert [artifact["path"] for artifact in manifest["artifacts"]] == [
        "ablation_metrics_aggregated.csv",
        "ablation_run_context.csv",
        "seed_coverage_report.csv",
    ]
    assert captured["command"][:3] == [
        os.sys.executable,
        "-m",
        "scripts.analysis.analyze_ablations",
    ]


def test_finalize_crossval_validates_folds_and_aggregates_scalar_metrics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = {
        "crossval": {"campaign_id": "campaign-a", "folds": 2, "seed": 42},
        "evaluation": {"split": "test"},
    }
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))

    for fold, mae in enumerate((0.2, 0.4)):
        fold_root = run_root / "outputs/crossval/campaign-a" / f"fold-{fold:03d}"
        checkpoint = (
            fold_root / "outputs/training/training/model/checkpoints/best_model.pt"
        )
        training_manifest = fold_root / "fold_training_manifest.json"
        evaluation_manifest = fold_root / "outputs/evaluation/evaluation_manifest.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{fold}".encode())
        run_support.write_json(training_manifest, {"fold": fold})
        run_support.write_json(evaluation_manifest, {"fold": fold})
        run_support.write_json(
            fold_root / "outputs/evaluation/test_summary.json",
            {
                "eval_loss": mae + 0.1,
                "metrics": {"mae": mae, "mae_by_horizon": [mae]},
            },
        )
        run_support.write_json(
            fold_root / "fold_manifest.json",
            {
                "fold": fold,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run_support.sha256_file(checkpoint),
                "training_fold_manifest": str(training_manifest),
                "training_fold_manifest_sha256": run_support.sha256_file(
                    training_manifest
                ),
                "evaluation_manifest": str(evaluation_manifest),
                "evaluation_manifest_sha256": run_support.sha256_file(
                    evaluation_manifest
                ),
            },
        )

    manifest_path = run_support.finalize_crossval(config_path, run_root)

    aggregate = (
        run_root / "outputs/crossval/campaign-a/test_aggregate_metrics.csv"
    ).read_text()
    assert "mae,2,0.30000000000000004" in aggregate
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["folds"]) == 2
    assert (run_root / ".state/crossval.done").is_file()


def test_crossval_fold_builds_isolated_train_evaluate_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = {
        "crossval": {
            "campaign_id": "campaign-a",
            "base_config": "configs/production_only/train.yaml",
            "folds": 3,
            "seed": 17,
            "holdout_role": "test",
            "max_batches": 2,
            "overrides": ["training.compile=false"],
        },
        "evaluation": {"split": "test", "granular": False, "overrides": []},
    }
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    captured: dict[str, object] = {}

    def fake_train(path: Path, source: Path, fold_root: Path) -> Path:
        fold_config = run_support.load_yaml(path)
        captured["config"] = fold_config
        captured["source"] = source
        checkpoint, model_root = run_support.training_artifact_paths(
            fold_config, fold_root
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fold checkpoint")
        run_support.write_json(model_root / "training_manifest.json", {"ok": True})
        return fold_root / ".state/training.done"

    def fake_evaluate(path: Path, fold_root: Path, checkpoint: Path) -> Path:
        captured["checkpoint"] = checkpoint
        manifest = fold_root / "outputs/evaluation/evaluation_manifest.json"
        run_support.write_json(manifest, {"ok": True})
        return manifest

    monkeypatch.setattr(run_support, "train_model", fake_train)
    monkeypatch.setattr(run_support, "evaluate_configured_model", fake_evaluate)

    training_manifest = run_support.train_crossval_fold(
        config_path, source_root, run_root, fold=1
    )
    manifest = run_support.evaluate_crossval_fold(config_path, run_root, fold=1)

    fold_config = captured["config"]
    assert isinstance(fold_config, dict)
    assert fold_config["workflow"] == {"kind": "train", "target": "train_evaluate"}
    assert fold_config["training"]["model_id"] == "model"
    assert "training.crossval_fold_index=1" in fold_config["training"]["overrides"]
    assert "training.crossval_num_folds=3" in fold_config["training"]["overrides"]
    assert captured["source"] == source_root
    assert captured["checkpoint"] == (
        run_root
        / "outputs/crossval/campaign-a/fold-001"
        / "outputs/training/training/model/checkpoints/best_model.pt"
    )
    assert json.loads(training_manifest.read_text())["fold"] == 1
    assert json.loads(manifest.read_text())["fold"] == 1


def _run_workflow_dry_run(
    config_path: Path,
    run_root: Path,
    home: Path,
    *targets: str,
) -> subprocess.CompletedProcess[str]:
    snakemake = shutil.which("snakemake")
    assert snakemake is not None
    env = os.environ.copy()
    env["HOME"] = str(home)
    return subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(REPO_ROOT / "workflow/Snakefile"),
            "--configfile",
            str(config_path),
            "--workflow-profile",
            str(REPO_ROOT / "workflow/profiles/mn5"),
            "--directory",
            str(run_root),
            "--runtime-source-cache-path",
            str(run_root / ".snakemake/source-cache"),
            "--dry-run",
            *targets,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_source_inventory(run_root: Path) -> None:
    inventory = run_root / ".state/source-files.bin"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_bytes(
        b"".join(
            os.fsencode(path.as_posix()) + b"\0"
            for path in run_config.source_files(REPO_ROOT)
        )
    )
    (run_root / ".state/dispatch.tsv").write_text("test\tdispatch\n")
    (run_root / ".state/published.done").write_text("run_id=test-run\n")


def _use_repository_sources(config: dict) -> None:
    production = yaml.safe_load((REPO_ROOT / "workflow/config/mn5.yaml").read_text())
    config["run"]["source_root"] = str(REPO_ROOT)
    config["dvc"]["targets"] = production["dvc"]["targets"]


def _workflow_mode_config(kind: str, run_root: Path) -> dict:
    config = _workflow_config(run_root, REPO_ROOT)
    _use_repository_sources(config)
    config["workflow"]["kind"] = kind
    if kind == "train":
        template = yaml.safe_load(
            (REPO_ROOT / "workflow/config/mn5_train.yaml").read_text()
        )
        config["training"] = template["training"]
        config["evaluation"] = template["evaluation"]
        config["resources"] = template["resources"]
        config.pop("optuna")
        config.pop("promotion")
    elif kind == "evaluate":
        template = yaml.safe_load(
            (REPO_ROOT / "workflow/config/mn5_train.yaml").read_text()
        )
        config["evaluation"] = {
            **template["evaluation"],
            "checkpoint": str(run_root / "inputs/best_model.pt"),
        }
        config["resources"] = template["resources"]
        config.pop("optuna")
        config.pop("promotion")
    elif kind == "ablation":
        template = yaml.safe_load(
            (REPO_ROOT / "workflow/config/mn5_ablation.yaml").read_text()
        )
        config["ablation"] = template["ablation"]
        config["resources"] = template["resources"]
        config.pop("optuna")
        config.pop("evaluation")
        config.pop("promotion")
    elif kind == "crossval":
        template = yaml.safe_load(
            (REPO_ROOT / "workflow/config/mn5_crossval.yaml").read_text()
        )
        config["crossval"] = template["crossval"]
        config["evaluation"] = template["evaluation"]
        config["resources"] = template["resources"]
        config.pop("optuna")
        config.pop("promotion")
    return config


def test_snakefile_builds_expected_dag(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = _workflow_config(run_root, REPO_ROOT)
    _use_repository_sources(config)
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)
    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode == 0, result.stderr
    assert "launch_optuna_workers" in result.stdout
    assert "select_best_trial" in result.stdout
    assert "evaluate_model" in result.stdout
    assert "promote_best_model" in result.stdout
    assert "cleanup_losing_checkpoints" in result.stdout
    assert "--workers 4" in result.stdout
    assert "--cpus-per-task 20" in result.stdout
    assert "--gpus-per-task=1" in result.stdout
    assert "training.run_final_evaluation=false" in result.stdout
    assert "complete_workflow" in result.stdout
    assert re.search(r"^total\s+14$", result.stdout, re.MULTILINE)


def test_single_training_dag_has_distinct_train_and_evaluate_entities(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _workflow_mode_config("train", run_root)
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    train_result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert train_result.returncode == 0, train_result.stdout + train_result.stderr
    assert "train_model" in train_result.stdout
    assert "evaluate_model" not in train_result.stdout
    assert not re.search(
        r"^launch_optuna_workers\s+\d", train_result.stdout, re.MULTILINE
    )
    assert "--ntasks=1" in train_result.stdout
    assert "--gpus-per-task=1" in train_result.stdout
    assert "training.run_final_evaluation=false" in config["training"]["overrides"]
    assert str(REPO_ROOT / "training/epiforecaster_trainer.py") in train_result.stdout
    assert str(REPO_ROOT / "configs/production_only/train_epifor_mn5_full.yaml") in (
        train_result.stdout
    )
    assert re.search(r"^total\s+7$", train_result.stdout, re.MULTILINE)

    composed_result = _run_workflow_dry_run(
        config_path,
        run_root,
        tmp_path / "home-composed",
        "train_evaluate",
    )
    assert composed_result.returncode == 0, (
        composed_result.stdout + composed_result.stderr
    )
    assert "train_model" in composed_result.stdout
    assert "evaluate_model" in composed_result.stdout
    assert "test_summary.json" in composed_result.stdout
    assert "evaluation_manifest.json" in composed_result.stdout


def test_single_training_smoke_default_dag_reaches_evaluation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _workflow_mode_config("train", run_root)
    smoke = yaml.safe_load(
        (REPO_ROOT / "workflow/config/mn5_train_smoke.yaml").read_text()
    )
    config["workflow"]["target"] = smoke["workflow"]["target"]
    config["training"] = smoke["training"]
    config["evaluation"] = smoke["evaluation"]
    config["resources"] = smoke["resources"]
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_model" in result.stdout
    assert "evaluate_model" in result.stdout
    assert "test_summary.json" in result.stdout
    assert "evaluation_manifest.json" in result.stdout
    assert "complete_workflow" in result.stdout
    assert re.search(r"^total\s+9$", result.stdout, re.MULTILINE)


def test_ablation_dag_uses_shared_worker_step_and_post_worker_finalizer(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _workflow_mode_config("ablation", run_root)
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "launch_ablation_workers" in result.stdout
    assert "scripts.hpo.ablation_worker" in result.stdout
    assert "finalize_ablation" in result.stdout
    assert "ablation_manifest.json" in result.stdout
    assert "seed_coverage_report.csv" in result.stdout
    assert "--workers 4" in result.stdout
    assert re.search(r"^total\s+8$", result.stdout, re.MULTILINE)


def test_crossval_dag_uses_one_rank_per_fold_and_one_finalizer(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    config = _workflow_mode_config("crossval", run_root)
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_crossval_folds" in result.stdout
    assert "evaluate_crossval_folds" in result.stdout
    assert "finalize_crossval" in result.stdout
    assert "--worker-group crossval-training" in result.stdout
    assert "--worker-group crossval-evaluation" in result.stdout
    assert "--workers 4" in result.stdout
    assert (
        "--fold '{worker_id}'" in result.stdout or "--fold {worker_id}" in result.stdout
    )
    assert "test_fold_metrics.csv" in result.stdout
    assert "test_aggregate_metrics.csv" in result.stdout
    assert re.search(r"^total\s+10$", result.stdout, re.MULTILINE)


def test_evaluation_only_dag_pins_checkpoint_inside_the_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = _workflow_mode_config("evaluate", run_root)
    checkpoint = Path(config["evaluation"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "evaluate_model" in result.stdout
    assert "train_model" not in result.stdout
    pinned_checkpoint = run_root / "inputs/evaluation_checkpoint.pt"
    assert str(pinned_checkpoint) in result.stdout
    assert f"--input-path {checkpoint}" not in result.stdout
    assert "test_summary.json" in result.stdout
    assert "evaluation_manifest.json" in result.stdout
    assert re.search(r"^total\s+7$", result.stdout, re.MULTILINE)


def test_scientific_source_inventory_excludes_docs_and_tests() -> None:
    paths = [
        Path("cli.py"),
        Path("constants.py"),
        Path("dataviz/eval_head_plots.py"),
        Path("models/forecaster.py"),
        Path("training/trainer.py"),
        Path("workflow/Snakefile"),
        Path("workflow/config/mn5.yaml"),
        Path("workflow/rules/training.smk"),
        Path("data/processed/data.zarr"),
        Path("docs/architecture.md"),
        Path("tests/test_model.py"),
        Path("tex/paper.py"),
    ]

    assert run_config.scientific_source_files(paths) == [
        Path("cli.py"),
        Path("constants.py"),
        Path("dataviz/eval_head_plots.py"),
        Path("models/forecaster.py"),
        Path("training/trainer.py"),
        Path("workflow/Snakefile"),
        Path("workflow/config/mn5.yaml"),
        Path("workflow/rules/training.smk"),
    ]


def test_every_stage_depends_on_the_complete_scientific_source_bundle() -> None:
    paths = [
        Path("cli.py"),
        Path("models/epiforecaster.py"),
        Path("training/epiforecaster_trainer.py"),
        Path("evaluation/eval_loop.py"),
        Path("plotting/forecast_plots.py"),
        Path("scripts/hpo/hpsearch_worker.py"),
        Path("configs/production_only/train.yaml"),
        Path("workflow/rules/training.smk"),
        Path("workflow/rules/evaluation.smk"),
        Path("workflow/rules/studies.smk"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    config = {
        "training": {"base_config": "configs/production_only/train.yaml"},
        "evaluation": {"split": "test"},
        "optuna": {"base_config": "configs/production_only/train.yaml"},
    }

    expected = run_config.scientific_source_files(paths)
    for stage in run_config.DEPENDENCY_STAGES:
        assert run_config.stage_source_files(paths, config, stage) == expected


def test_evaluation_writes_and_hashes_required_artifact_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "evaluation"

    def fake_run(command, *, cwd=None, env=None):
        summary = Path(command[command.index("--summary-json") + 1])
        node_metrics = Path(command[command.index("--node-metrics-csv") + 1])
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text('{"eval_loss": 0.5}\n')
        node_metrics.write_text("node_id,mae\n0,0.1\n")
        node_metrics.with_name("test_node_metrics_per_head.csv").write_text(
            "target,node_id,mae\ncases,0,0.1\n"
        )
        (summary.parent / "test_granular.parquet").write_bytes(b"parquet")

    monkeypatch.setattr(run_support, "_run", fake_run)

    manifest_path = run_support.evaluate_best(
        checkpoint,
        output_dir,
        "test",
        [],
        True,
    )

    assert manifest_path == output_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoint_sha256"] == run_support.sha256_file(checkpoint)
    assert [artifact["path"] for artifact in manifest["artifacts"]] == [
        "test_granular.parquet",
        "test_node_metrics.csv",
        "test_node_metrics_per_head.csv",
        "test_summary.json",
    ]
    assert all(artifact["required"] for artifact in manifest["artifacts"])


def test_train_model_records_deterministic_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"
    base_config = source_root / "configs/train.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("env: test\n")
    config = {
        "training": {
            "base_config": "configs/train.yaml",
            "experiment_name": "single",
            "model_id": "model",
            "max_batches": 2,
            "overrides": ["training.epochs=1"],
        }
    }
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd=None, env=None):
        captured.update(command=command, cwd=cwd, env=env)
        checkpoint, _ = run_support.training_artifact_paths(config, run_root)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"trained")

    monkeypatch.setattr(run_support, "_run", fake_run)

    marker = run_support.train_model(config_path, source_root, run_root)

    checkpoint, model_root = run_support.training_artifact_paths(config, run_root)
    metadata = json.loads(marker.read_text())
    training_manifest = model_root / "training_manifest.json"
    assert checkpoint == model_root / "checkpoints/best_model.pt"
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["training_manifest"] == str(training_manifest)
    assert json.loads(training_manifest.read_text())["checkpoint"] == str(checkpoint)
    assert metadata["checkpoint_sha256"] == run_support.sha256_file(checkpoint)
    assert captured["cwd"] == source_root
    assert captured["env"]["EPIFORECASTER_MODEL_ID"] == "model"
    assert captured["command"][:5] == [
        os.sys.executable,
        "-m",
        "cli",
        "train",
        "epiforecaster",
    ]
    assert captured["command"][captured["command"].index("--max-batches") :][:2] == [
        "--max-batches",
        "2",
    ]


@pytest.mark.parametrize(
    ("resource_name", "resource_value"),
    [
        ("mem_mb", 1024),
        ("mem_mb_per_cpu", 64),
        ("slurm_extra", "--constraint=foo --mem=1024"),
    ],
)
def test_snakefile_rejects_explicit_memory_during_dag_construction(
    tmp_path: Path,
    resource_name: str,
    resource_value: int | str,
) -> None:
    run_root = tmp_path / "run"
    config = _workflow_config(run_root, REPO_ROOT)
    _use_repository_sources(config)
    config["resources"]["allocation"][resource_name] = resource_value
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    _write_source_inventory(run_root)

    result = _run_workflow_dry_run(config_path, run_root, tmp_path / "home")

    assert result.returncode != 0
    assert "must not request explicit memory" in result.stdout + result.stderr


def test_promotion_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = _workflow_config(run_root, run_root / "source")
    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    destination = Path(config["promotion"]["root"]) / config["run"]["id"]
    destination.mkdir(parents=True)

    try:
        run_support.promote_artifacts(config_path, run_root)
    except FileExistsError as exc:
        assert str(destination) in str(exc)
    else:
        raise AssertionError("Promotion unexpectedly overwrote an existing run")


def _write_promotion_evidence(run_root: Path) -> tuple[Path, Path]:
    source_checkpoint = (
        run_root / "optuna/trials/test-study/trial-000000/checkpoints/best_model.pt"
    )
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"selected checkpoint")
    checkpoint = run_root / "outputs/best_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"selected checkpoint")
    checkpoint_hash = run_support.sha256_file(checkpoint)
    run_support.write_json(
        run_root / "outputs/selection.json",
        {
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": checkpoint_hash,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": checkpoint_hash,
        },
    )
    evaluation_dir = run_root / "outputs/evaluation"
    summary = evaluation_dir / "test_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text('{"loss": 0.5}\n')
    run_support.write_json(
        evaluation_dir / "evaluation_manifest.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "artifacts": [
                {
                    "path": summary.name,
                    "sha256": run_support.sha256_file(summary),
                    "required": True,
                }
            ],
        },
    )
    return checkpoint, summary


def test_promotion_integrity_rejects_tampered_selected_checkpoint(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    checkpoint, _ = _write_promotion_evidence(run_root)
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="promotion SHA-256"):
        run_support._promotion_integrity(run_root)


def test_promotion_integrity_rejects_tampered_evaluation_artifact(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _, summary = _write_promotion_evidence(run_root)
    summary.write_text('{"loss": 99}\n')

    with pytest.raises(ValueError, match="Evaluation artifact failed"):
        run_support._promotion_integrity(run_root)


def test_promotion_integrity_accepts_selected_and_evaluated_hashes(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    checkpoint, summary = _write_promotion_evidence(run_root)

    assert run_support._promotion_integrity(run_root) == {
        "outputs/best_model.pt": run_support.sha256_file(checkpoint),
        "outputs/evaluation/test_summary.json": run_support.sha256_file(summary),
    }


def test_selection_metadata_is_json_serializable(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    run_support.write_json(metadata, {"path": Path("checkpoint.pt")})
    assert json.loads(metadata.read_text()) == {"path": "checkpoint.pt"}


def test_source_snapshot_permission_check_uses_mode_bits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source.chmod(source.stat().st_mode | stat.S_IWUSR)
    assert source.stat().st_mode & 0o222


def test_container_build_dispatch_dry_run_is_offline() -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "containers/runtime/build_sif_on_mn5.sh"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "no transfer, VPN, SSH, dependency sync, or Slurm command ran" in result.stdout
    )
    assert "network fetch host: wifi-acc" in result.stdout
    assert "dependency key:" in result.stdout


def test_container_build_dry_run_accepts_existing_allocation() -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "containers/runtime/build_sif_on_mn5.sh"),
            "--dry-run",
            "--job-id",
            "43720217",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "existing allocation: 43720217" in result.stdout


def test_dependency_groups_keep_runtime_and_workflow_contracts_explicit() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    groups = project["dependency-groups"]

    assert groups["runtime"] == ["dvc==3.67.1"]
    assert "snakemake==9.23.1" in groups["workflow"]
    assert not any("executor-plugin" in dependency for dependency in groups["workflow"])
    assert not any("snakemake" in dependency for dependency in groups["runtime"])


def test_container_build_is_source_independent_and_batch_validated() -> None:
    batch = (REPO_ROOT / "containers/runtime/build_sif.sbatch").read_text()
    dispatcher = (REPO_ROOT / "containers/runtime/build_sif_on_mn5.sh").read_text()
    fetch = (REPO_ROOT / "containers/runtime/fetch_runtime_deps_on_wifi.sh").read_text()

    assert "#SBATCH --partition=acc" in batch
    assert 'singularity build --sandbox "$sandbox" "$BASE_SIF"' in batch
    assert 'cp -a "$PREFIX_ROOT/."' in batch
    assert 'chmod -R u+w "$sandbox/opt/epiforecaster"' in batch
    assert 'chmod -R a+rX "$sandbox/opt/epiforecaster"' in batch
    assert 'exec /opt/epiforecaster/.venv/bin/python "$@"' in batch
    assert 'for console_script in "$PREFIX_ROOT/.venv/bin/"*' in batch
    assert '"exec /opt/epiforecaster/.venv/bin/$script_name' in batch
    assert 'exec /opt/epiforecaster/.venv/bin/python -m dvc "$@"' in batch
    assert 'singularity build --force "$scratch_sif" "$sandbox"' in batch
    assert 'test "$(command -v python)" = /usr/local/bin/python' in batch
    assert 'assert sys.prefix == "/opt/epiforecaster/.venv"' in batch
    assert '"/scratch/tmp/${USER:-unknown}"' in batch
    assert 'scratch_leaf="epiforecaster-sif-' in batch
    assert 'scratch_root="${SLURM_TMPDIR:-}"' not in batch
    assert 'chmod -R u+w "$scratch_root"' in batch
    assert 'test -f "$metadata_file"' in batch
    assert 'test -f "$inspect_file"' in batch
    assert 'test -f "$success_marker"' in batch
    assert '-L "$PREFIX_ROOT/.venv/bin/python"' in batch
    assert 'test -x "$PREFIX_ROOT/.venv/bin/python"' not in batch
    assert "import dvc, geopandas, netCDF4" in batch
    assert "torch_geometric, xarray, zarr" in batch
    assert '"$scratch_sif" dvc --version' in batch
    assert '"$scratch_sif" snakemake --version' in batch
    assert "singularity exec --cleanenv --nv" in batch
    assert "torch.cuda.is_available()" in batch
    assert "uv sync" in fetch
    assert "--frozen" in fetch
    assert "--python /usr/local/bin/python3.11" in fetch
    assert "--group workflow" in fetch
    assert "--group runtime" not in fetch
    assert "UV_CONCURRENT_INSTALLS=1" in fetch
    assert "UV_LINK_MODE=copy" in fetch
    assert '"$prefix_root/.venv/bin/python"' in fetch
    assert "sha256sum -c" in dispatcher
    assert "remote_bash" not in dispatcher
    assert 'declare -f "$function_name"' in dispatcher
    assert "remote_container_state()" in dispatcher
    assert "remote_submit_build()" in dispatcher
    assert "remote_build_in_allocation()" in dispatcher
    assert '--jobid="$allocation_id"' in dispatcher
    assert 'UV_CACHE_ROOT="/scratch/tmp/${USER:-unknown}/' in dispatcher
    key_material = dispatcher.split('dependency_key="$({', 1)[1].split(
        '} | sha256_stream)"', 1
    )[0]
    assert '"$REPO_ROOT/pyproject.toml"' not in key_material
    assert '"$REPO_ROOT/uv.lock"' in key_material
    assert 'grep -Fxq "dependency_key=$dependency_key"' in dispatcher
    assert 'mkdir "$request_root/dispatching"' in dispatcher
    assert 'exec 9>"$request_root/submit.lock"' in dispatcher


def _make_remote_run_staging(
    tmp_path: Path,
    *,
    token: str,
) -> tuple[Path, Path, dict[str, str]]:
    """Create a faithful local staging bundle for the transferred runner."""
    run_parent = tmp_path / "runs"
    run_root = run_parent / "test-run"
    staging = run_parent / f".staging-test-run.{token}"
    source_root = staging / "source"
    run_parent.mkdir(exist_ok=True)

    for relative in (
        "scripts/__init__.py",
        "scripts/workflow/__init__.py",
        "scripts/workflow/run_config.py",
        "scripts/workflow/remote_run.sh",
        "scripts/workflow/snakemake_coordinator.sbatch",
    ):
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)

    source_paths = sorted(
        (
            path.relative_to(source_root)
            for path in source_root.rglob("*")
            if path.is_file() or path.is_symlink()
        ),
        key=lambda path: path.as_posix(),
    )
    source_digest = run_config.source_snapshot_sha256(source_root, source_paths)
    state = staging / ".state"
    state.mkdir()
    (state / "source-files.bin").write_bytes(
        b"".join(os.fsencode(path.as_posix()) + b"\0" for path in source_paths)
    )

    sif = tmp_path / "containers/runtime.sif"
    sif.parent.mkdir(exist_ok=True)
    sif.write_bytes(b"immutable runtime")
    sif_digest = run_support.sha256_file(sif)
    Path(f"{sif}.sha256").write_text(f"{sif_digest}  {sif.name}\n")

    snakemake_bin = Path("/usr/local/bin/snakemake")
    tooling_python = Path("/usr/local/bin/python")
    config = _workflow_config(run_root, run_root / "source")
    config["source"]["source_snapshot_sha256"] = source_digest
    config["source"]["source_file_count"] = len(source_paths)
    config["container"] = {
        "sif": str(sif),
        "digest_file": f"{sif}.sha256",
    }
    config["workflow"]["snakemake_bin"] = str(snakemake_bin)
    config["workflow"]["snakemake_version"] = version("snakemake")
    config_path = staging / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_config.write_dispatch_request(config_path, state / "dispatch.tsv")

    request = (state / "dispatch.tsv").read_text()
    expected = f"tooling_python\t{tooling_python}\n"
    assert expected in request

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    singularity = fake_bin / "singularity"
    singularity.write_text(
        f"""#!/bin/bash
set -euo pipefail
test "$1" = exec
shift
working_directory=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cleanenv|--containall)
      shift
      ;;
    --bind)
      shift 2
      ;;
    --env)
      export "$2"
      shift 2
      ;;
    --pwd)
      working_directory="$2"
      shift 2
      ;;
    *)
      shift
      break
      ;;
  esac
done
cd "$working_directory"
case "$1" in
  /usr/local/bin/python)
    shift
    exec {sys.executable} "$@"
    ;;
  /usr/local/bin/snakemake)
    shift
    exec {Path(sys.executable).parent / "snakemake"} "$@"
    ;;
esac
exec "$@"
"""
    )
    singularity.chmod(0o755)
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    return staging, run_root, env


def test_transferred_runner_publishes_once_and_reserves_run_id(
    tmp_path: Path,
) -> None:
    staging, run_root, env = _make_remote_run_staging(tmp_path, token="owner")
    runner = staging / "source/scripts/workflow/remote_run.sh"

    result = subprocess.run(
        [str(runner), "publish"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == f"published:{run_root}"
    assert not staging.exists()
    assert (run_root / ".state/published.done").is_file()
    assert (run_root / ".state/dispatch.tsv").is_file()
    assert not (run_root / "config.yaml").stat().st_mode & 0o222
    assert not (run_root / "source").stat().st_mode & 0o222
    reservation = run_root.parent / ".reservations/test-run"
    assert reservation.is_dir()
    assert (reservation / "owner").read_text() == ".staging-test-run.owner\n"

    competing, _, env = _make_remote_run_staging(tmp_path, token="competitor")
    duplicate = subprocess.run(
        [str(competing / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode == 73
    assert "already reserved" in duplicate.stderr
    assert run_root.is_dir()


def test_transferred_runner_rejects_tampered_config_before_reservation(
    tmp_path: Path,
) -> None:
    staging, run_root, env = _make_remote_run_staging(tmp_path, token="tampered")
    config_path = staging / "config.yaml"
    config_path.write_text(config_path.read_text() + "tampered: true\n")

    result = subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "config failed SHA-256" in result.stderr
    assert not run_root.exists()
    assert not (run_root.parent / ".reservations/test-run").exists()


def test_transferred_runner_submission_is_idempotent_and_pinned(
    tmp_path: Path,
) -> None:
    staging, run_root, env = _make_remote_run_staging(tmp_path, token="submit")
    subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=True,
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    calls = tmp_path / "sbatch.calls"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$*" >>"$CALLS_FILE"\n'
        "printf '12345;cluster\\n'\n"
    )
    sbatch.chmod(0o755)
    squeue = fake_bin / "squeue"
    squeue.write_text("#!/bin/bash\nprintf 'RUNNING\\n'\n")
    squeue.chmod(0o755)
    env = env | {
        "PATH": f"{fake_bin}:{env['PATH']}",
        "CALLS_FILE": str(calls),
    }
    runner = run_root / "source/scripts/workflow/remote_run.sh"

    first = subprocess.run(
        [str(runner), "submit"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [str(runner), "submit"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert first.stdout == "submitted:12345"
    assert second.stdout == "active:12345:RUNNING"
    call_lines = calls.read_text().splitlines()
    assert len(call_lines) == 1
    assert "--nodes=1" in call_lines[0]
    assert "--ntasks=4" in call_lines[0]
    assert "--cpus-per-task=20" in call_lines[0]
    assert "--gres=gpu:4" in call_lines[0]
    assert "--mem" not in call_lines[0]
    assert "--cores" not in call_lines[0]
    assert (run_root / ".state/coordinator.job_id").read_text() == "12345\n"
    assert (run_root.parent / "by-slurm/12345_0").resolve() == run_root


@pytest.mark.parametrize("command", ["run", "dry-run"])
def test_dispatcher_rejects_submission_time_snakemake_arguments(
    tmp_path: Path,
    command: str,
) -> None:
    program = _workflow_shell_source(tmp_path) + f"main {command} test-run --cores 4\n"

    result = subprocess.run(
        ["bash", "-c", program],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_container_wrapper_preserves_scientific_exit_status(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"
    fake_bin = tmp_path / "bin"
    scratch = tmp_path / "scratch"
    sif = tmp_path / "runtime.sif"
    run_root.mkdir()
    source_root.mkdir()
    fake_bin.mkdir()
    scratch.mkdir()
    sif.touch()

    singularity = fake_bin / "singularity"
    singularity.write_text("#!/bin/bash\nexit 7\n")
    singularity.chmod(0o755)
    wrapper = REPO_ROOT / "scripts/workflow/mn5_container_exec.sh"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TMPDIR": str(scratch),
    }

    result = subprocess.run(
        [
            "bash",
            str(wrapper),
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--sif",
            str(sif),
            "--",
            "scientific-command",
            "argument with spaces",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert not (scratch / "epiforecaster" / run_root.name / "local_batch_0").exists()


def test_workflow_dispatch_enforces_isolation_and_single_coordinator() -> None:
    dispatcher = (REPO_ROOT / "scripts/workflow/mn5_workflow.sh").read_text()
    remote_runner = (REPO_ROOT / "scripts/workflow/remote_run.sh").read_text()
    container_exec = (REPO_ROOT / "scripts/workflow/mn5_container_exec.sh").read_text()
    coordinator = (
        REPO_ROOT / "scripts/workflow/snakemake_coordinator.sbatch"
    ).read_text()
    launcher = (REPO_ROOT / "scripts/workflow/launch_parallel_workers.sh").read_text()
    worker = (REPO_ROOT / "scripts/workflow/run_parallel_worker.sh").read_text()

    assert "remote_bash" not in dispatcher
    assert "declare -f" not in dispatcher
    assert "remote_function" not in dispatcher
    assert "remote_run.sh" in dispatcher
    assert "remote_tooling.sh" not in dispatcher
    assert 'staging_root="$REMOTE_WORKFLOW_RUN_ROOT/.staging-' in dispatcher
    assert "--dispatch-file" in dispatcher
    assert "uv sync" not in dispatcher
    assert "uv sync" not in remote_runner
    assert "scripts.workflow.tooling_identity" not in dispatcher
    assert 'REMOTE_SNAKEMAKE_BIN="/usr/local/bin/snakemake"' in dispatcher
    assert 'TOOLING_KEY="sif-sha256:$observed_sha"' in dispatcher
    assert "executor_commit=" not in dispatcher
    assert "EXECUTOR_SUBMODULE" not in dispatcher
    assert 'exec 9>"$RUN_ROOT/.state/coordinator.launch.lock"' in remote_runner
    assert "workflow/config/mn5_train.yaml" in dispatcher
    assert "$RUN_ROOT/.state/workflow.done" in remote_runner
    assert "sbatch --parsable" in remote_runner
    assert '--ntasks="$TASKS"' in remote_runner
    assert '--cpus-per-task="$CPUS_PER_TASK"' in remote_runner
    assert '--gres="$GRES"' in remote_runner
    assert '"$TASKS"' in remote_runner
    assert '"$CPUS_PER_TASK"' in remote_runner
    assert '"$CONTAINER_SIF"' in remote_runner
    assert '"$SNAKEMAKE_BIN"' not in remote_runner.split(
        "sbatch --parsable", 1
    )[1].split(')"', 1)[0]
    assert (
        '"$@"'
        not in remote_runner.split("submit_run()", 1)[1].split("show_status()", 1)[0]
    )
    assert "--mem" not in coordinator
    squeue_state_probes = [
        line for line in remote_runner.splitlines() if 'state="$(squeue' in line
    ]
    assert len(squeue_state_probes) == 3
    assert all("head -n 1 || true" in line for line in squeue_state_probes)
    assert "--dvc-project-root" not in container_exec
    assert "--containall" in container_exec
    assert '"$DVC_REMOTE:$DVC_REMOTE:ro"' in container_exec
    assert '"$DVC_REMOTE:$DVC_REMOTE:rw"' in container_exec
    assert (
        'execution_id="${slurm_job_id}_${slurm_step_id}_${slurm_proc_id}"'
        in container_exec
    )
    assert (
        'allocated_cpus="${REQUESTED_CPUS:-${SLURM_CPUS_PER_TASK:-1}}"'
        in container_exec
    )
    assert "trap - EXIT" in container_exec
    assert 'chmod -R u+w "$SCRATCH_ROOT"' in container_exec
    assert 'exit "$status"' in container_exec
    assert 'exec 9>"$RUN_ROOT/.state/coordinator.run.lock"' in coordinator
    assert "exec singularity" in coordinator
    assert "SLURM_STEP_NUM_NODES" in coordinator
    assert '--env "SLURM_JOB_NUM_NODES=$ALLOCATED_NODES"' in coordinator
    assert "compgen -A variable SLURM_" not in coordinator
    assert "/usr/bin/srun:/usr/bin/srun:ro" in coordinator
    assert "/run/munge:/run/munge:ro" in coordinator
    assert 'host_step: bool = False' in (
        REPO_ROOT / "workflow/common.py"
    ).read_text()
    assert "#SBATCH --ntasks=4" in coordinator
    assert "#SBATCH --cpus-per-task=20" in coordinator
    assert "#SBATCH --gres=gpu:4" in coordinator
    assert '--ntasks="$WORKERS"' in launcher
    assert "--gpus-per-task=1" in launcher
    assert "--kill-on-bad-exit=1" in launcher
    assert 'WORKER_INDEX="${SLURM_PROCID:' in worker
    assert "EPIFORECASTER_WORKER_ID=$WORKER_INDEX" in worker
