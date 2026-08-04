"""Render one immutable workflow run configuration from a template."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import yaml


ALLOCATION_FIELDS = (
    "account",
    "partition",
    "qos",
    "nodes",
    "tasks",
    "cpus_per_task",
    "gres",
    "runtime",
)
DISPATCH_REQUEST_SCHEMA_VERSION = 1
WORKFLOW_KINDS = {
    "train",
    "evaluate",
    "hpo",
    "ablation",
    "crossval",
    "preprocess",
}
WORKFLOW_TARGETS = {
    "train": {"train", "train_evaluate"},
    "evaluate": {"evaluate"},
    "hpo": {"hpo"},
    "ablation": {"ablation"},
    "crossval": {"crossval"},
    "preprocess": {"preprocess"},
}
SCIENTIFIC_SOURCE_EXCLUDED_ROOTS = {
    ".agents",
    ".codex",
    ".github",
    "docs",
    "outputs",
    "reports",
    "tests",
    "tex",
    "vendor",
}
SCIENTIFIC_SOURCE_FILES = {"cli.py", "pyproject.toml", "uv.lock"}
SCIENTIFIC_SOURCE_SUFFIXES = {".py", ".sh", ".smk", ".toml", ".yaml", ".yml"}
DEPENDENCY_STAGES = {
    "data",
    "training",
    "evaluation",
    "hpo",
    "ablation",
    "crossval",
    "selection",
    "promotion",
    "preprocessing",
}


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo_root)


def source_files(repo_root: Path) -> list[Path]:
    """Return tracked and non-ignored untracked files in a worktree.

    Gitlinks are directories in the worktree and are intentionally omitted.
    """

    output = _git(
        repo_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    paths = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        absolute = repo_root / relative
        if absolute.is_file() or absolute.is_symlink():
            paths.append(relative)
    return sorted(set(paths), key=lambda path: path.as_posix())


def _safe_relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value!r}")
    return path


def dvc_payload_paths(
    source_root: Path,
    workflow_config: dict[str, Any],
) -> dict[Path, Path]:
    """Map DVC pointer paths to payload paths without materializing the data."""
    dvc_config = workflow_config.get("dvc")
    if not isinstance(dvc_config, dict):
        raise ValueError("Workflow config must define a dvc mapping")
    targets = dvc_config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Workflow config must define at least one dvc target")
    provenance_targets = dvc_config.get("provenance_targets", [])
    if not isinstance(provenance_targets, list):
        raise ValueError("dvc.provenance_targets must be a list when provided")

    payloads: dict[Path, Path] = {}
    for target_value in [*targets, *provenance_targets]:
        target = _safe_relative_path(str(target_value), field="DVC target")
        pointer = source_root / target
        if not pointer.is_file() or pointer.is_symlink():
            raise FileNotFoundError(
                f"DVC pointer must be a regular file in the source bundle: {target}"
            )
        pointer_document = yaml.safe_load(pointer.read_text())
        outputs = (
            pointer_document.get("outs") if isinstance(pointer_document, dict) else None
        )
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError(f"Expected one DVC output in {target}")
        output = outputs[0]
        if not isinstance(output, dict) or "md5" not in output or "path" not in output:
            raise ValueError(f"Malformed DVC output in {target}")
        output_name = _safe_relative_path(
            str(output["path"]),
            field=f"DVC output in {target}",
        )
        payload = target.parent / output_name
        if payload in payloads.values():
            raise ValueError(
                f"Multiple DVC pointers select the same payload: {payload}"
            )
        payloads[target] = payload
    return payloads


def validate_code_data_separation(
    source_root: Path,
    paths: list[Path],
    workflow_config: dict[str, Any],
) -> dict[Path, Path]:
    """Require DVC pointers in the code bundle and DVC payloads outside it."""
    payloads = dvc_payload_paths(source_root, workflow_config)
    selected = set(paths)
    missing_pointers = sorted(
        set(payloads) - selected, key=lambda path: path.as_posix()
    )
    if missing_pointers:
        missing = ", ".join(path.as_posix() for path in missing_pointers)
        raise ValueError(f"DVC pointers are absent from the source bundle: {missing}")

    included_payloads = sorted(
        (
            path
            for path in selected
            if any(
                path == payload or payload in path.parents
                for payload in payloads.values()
            )
        ),
        key=lambda path: path.as_posix(),
    )
    if included_payloads:
        included = ", ".join(path.as_posix() for path in included_payloads)
        raise ValueError(
            "DVC payloads must not be included in the relocatable source bundle: "
            f"{included}"
        )
    return payloads


def workflow_kind(workflow_config: dict[str, Any]) -> str:
    workflow = workflow_config.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("Workflow config must define a workflow mapping")
    kind = workflow.get("kind")
    if kind not in WORKFLOW_KINDS:
        supported = ", ".join(sorted(WORKFLOW_KINDS))
        raise ValueError(f"workflow.kind must be one of {supported}: {kind!r}")
    return str(kind)


def workflow_target(workflow_config: dict[str, Any]) -> str:
    """Return the pinned public target for an immutable workflow run."""
    kind = workflow_kind(workflow_config)
    target = workflow_config["workflow"].get("target", kind)
    if target not in WORKFLOW_TARGETS[kind]:
        supported = ", ".join(sorted(WORKFLOW_TARGETS[kind]))
        raise ValueError(
            f"workflow.target for kind={kind!r} must be one of {supported}: {target!r}"
        )
    return str(target)


def preprocessing_dataset_name(workflow_config: dict[str, Any]) -> str:
    """Return the safe output dataset stem for a preprocessing run."""
    if workflow_kind(workflow_config) != "preprocess":
        raise ValueError("preprocessing.dataset_name is valid only for preprocess runs")
    preprocessing = workflow_config["preprocessing"]
    value = str(preprocessing.get("dataset_name", ""))
    path = Path(value)
    if not value or path.name != value or value in {".", ".."} or path.suffix:
        raise ValueError(
            "preprocessing.dataset_name must be one safe filename stem without a suffix"
        )
    return value


def scientific_source_files(paths: list[Path]) -> list[Path]:
    """Select source files whose changes invalidate scientific DAG rules."""
    selected: list[Path] = []
    for path in paths:
        path_string = path.as_posix()
        excluded = (
            bool(path.parts) and path.parts[0] in SCIENTIFIC_SOURCE_EXCLUDED_ROOTS
        )
        if not excluded and (
            path_string in SCIENTIFIC_SOURCE_FILES
            or path.suffix in SCIENTIFIC_SOURCE_SUFFIXES
            or path.name == "Snakefile"
        ):
            selected.append(path)
    return sorted(set(selected), key=lambda path: path.as_posix())


def stage_source_files(
    paths: list[Path],
    workflow_config: dict[str, Any],
    stage: str,
) -> list[Path]:
    """Return the complete scientific source contract for one DAG stage."""
    if stage not in DEPENDENCY_STAGES:
        raise ValueError(f"Unknown dependency stage: {stage}")
    selected = set(scientific_source_files(paths))
    if stage in {"crossval", "training", "evaluation"} and isinstance(
        workflow_config.get("crossval"),
        dict,
    ):
        manifest = _safe_relative_path(
            str(workflow_config["crossval"]["split_manifest"]),
            field="crossval.split_manifest",
        )
        if manifest not in paths:
            raise FileNotFoundError(
                "Cross-validation split manifest is absent from the source "
                f"bundle: {manifest}"
            )
        selected.add(manifest)
    return sorted(selected, key=lambda path: path.as_posix())


def validate_workflow_contract(workflow_config: dict[str, Any]) -> str:
    """Validate mode-specific configuration before snapshot transfer."""
    kind = workflow_kind(workflow_config)
    workflow_target(workflow_config)
    required_sections = {
        "train": ("training", "evaluation"),
        "evaluate": ("evaluation",),
        "hpo": ("optuna", "evaluation", "promotion"),
        "ablation": ("ablation",),
        "crossval": ("crossval", "evaluation"),
        "preprocess": ("preprocessing",),
    }[kind]
    for section in required_sections:
        if not isinstance(workflow_config.get(section), dict):
            raise ValueError(f"workflow.kind={kind!r} requires a {section} mapping")

    if kind == "preprocess":
        preprocessing = workflow_config["preprocessing"]
        base_config = Path(str(preprocessing.get("base_config", "")))
        if (
            not base_config.parts
            or base_config.is_absolute()
            or ".." in base_config.parts
        ):
            raise ValueError("preprocessing.base_config must be a source-relative path")
        preprocessing_dataset_name(workflow_config)
        handoff_manifest = preprocessing.get("handoff_manifest")
        if handoff_manifest is not None:
            handoff_path = Path(str(handoff_manifest))
            if handoff_path.is_absolute() or ".." in handoff_path.parts:
                raise ValueError(
                    "preprocessing.handoff_manifest must be a source-relative path"
                )

    provenance = workflow_config.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping when provided")
    evidence_retention = provenance.get("training_evidence_retention", "run")
    if evidence_retention not in {"run", "dvc"}:
        raise ValueError(
            "provenance.training_evidence_retention must be 'run' or 'dvc'"
        )
    if kind != "train" and evidence_retention == "dvc":
        raise ValueError(
            "DVC training-evidence retention is currently supported only for "
            "workflow.kind='train'"
        )

    if kind == "train":
        training = workflow_config["training"]
        for field in ("base_config", "experiment_name", "model_id"):
            if not training.get(field):
                raise ValueError(f"training.{field} is required")
        for field in ("experiment_name", "model_id"):
            value = str(training[field])
            if Path(value).name != value or value in {".", ".."}:
                raise ValueError(f"training.{field} must be one safe path component")
        init_checkpoint = training.get("init_checkpoint")
        if init_checkpoint is not None and not Path(str(init_checkpoint)).is_absolute():
            raise ValueError(
                "training.init_checkpoint must be an absolute GPFS path when provided"
            )
        require_init_checkpoint = training.get("require_init_checkpoint", False)
        if not isinstance(require_init_checkpoint, bool):
            raise ValueError("training.require_init_checkpoint must be a boolean")
        if require_init_checkpoint and init_checkpoint is None:
            raise ValueError(
                "training.init_checkpoint is required by this workflow template"
            )
    elif kind == "evaluate":
        checkpoint = workflow_config["evaluation"].get("checkpoint")
        if checkpoint is None or not Path(str(checkpoint)).is_absolute():
            raise ValueError(
                "evaluation.checkpoint must be an absolute GPFS path for an "
                "evaluation-only workflow"
            )
    elif kind == "crossval":
        crossval = workflow_config["crossval"]
        for field in (
            "base_config",
            "campaign_id",
            "split_manifest",
            "folds",
            "model_seed",
        ):
            if crossval.get(field) is None:
                raise ValueError(f"crossval.{field} is required")
        campaign_id = str(crossval["campaign_id"])
        if Path(campaign_id).name != campaign_id or campaign_id in {".", ".."}:
            raise ValueError("crossval.campaign_id must be one safe path component")
        split_manifest = Path(str(crossval["split_manifest"]))
        if split_manifest.is_absolute() or ".." in split_manifest.parts:
            raise ValueError("crossval.split_manifest must be a source-relative path")
        model_seed = crossval["model_seed"]
        if (
            not isinstance(model_seed, int)
            or isinstance(model_seed, bool)
            or model_seed < 0
        ):
            raise ValueError("crossval.model_seed must be a non-negative integer")
        folds = crossval["folds"]
        if (
            not isinstance(folds, int)
            or isinstance(folds, bool)
            or not 2 <= folds <= 100
        ):
            raise ValueError("crossval.folds must be an integer between 2 and 100")
    return kind


def validate_mn5_allocation(workflow_config: dict[str, Any]) -> dict[str, Any]:
    """Validate the one-node ACC allocation for the selected workflow kind."""
    resources = workflow_config.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("Workflow config must define a resources mapping")
    allocation = resources.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("Workflow config must define resources.allocation")

    explicit_memory = sorted(
        key for key in ("mem_mb", "mem_mb_per_cpu") if key in allocation
    )
    slurm_extra = str(allocation.get("slurm_extra", ""))
    if re.search(r"--mem(?:-per-cpu)?(?:=|\s)", slurm_extra):
        explicit_memory.append("slurm_extra")
    if explicit_memory:
        raise ValueError(
            "resources.allocation must not request explicit memory: "
            f"{', '.join(explicit_memory)}"
        )

    kind = validate_workflow_contract(workflow_config)
    if kind == "hpo":
        workload = workflow_config.get("optuna")
        workload_name = "optuna"
    elif kind == "ablation":
        workload = workflow_config.get("ablation")
        workload_name = "ablation"
    elif kind == "crossval":
        folds = workflow_config["crossval"].get("folds")
        workload = {"workers": min(folds, 4) if isinstance(folds, int) else folds}
        workload_name = "crossval"
    else:
        workload = None
        workload_name = kind

    if workload is None and kind in {"hpo", "ablation"}:
        raise ValueError(f"Workflow config must define a {workload_name} mapping")
    expected_tasks = 1 if workload is None else workload.get("workers")
    if (
        not isinstance(expected_tasks, int)
        or isinstance(expected_tasks, bool)
        or not 1 <= expected_tasks <= 4
    ):
        raise ValueError(f"{workload_name}.workers must be an integer between 1 and 4")

    expected = {
        "account": "bsc08",
        "partition": "acc",
        "nodes": 1,
        "tasks": expected_tasks,
        "cpus_per_task": 20,
        "gres": f"gpu:{expected_tasks}",
    }
    mismatches = {
        key: (allocation.get(key), value)
        for key, value in expected.items()
        if allocation.get(key) != value
    }
    qos = allocation.get("qos")
    if qos not in {"acc_bscls", "acc_debug"}:
        mismatches["qos"] = (qos, "acc_bscls or acc_debug")
    runtime = allocation.get("runtime")
    if not isinstance(runtime, int) or isinstance(runtime, bool) or runtime <= 0:
        mismatches["runtime"] = (runtime, "positive integer minutes")
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Invalid MN5 single-node ACC allocation: {details}")
    if "gpu" in allocation or "gpus" in allocation:
        raise ValueError(
            "resources.allocation must use gres='gpu:<tasks>', not gpu/gpus"
        )
    return {field: allocation[field] for field in ALLOCATION_FIELDS}


def source_snapshot_sha256(source_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        absolute = source_root / relative
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        executable_bits = stat.S_IMODE(absolute.lstat().st_mode) & 0o111
        digest.update(f"exec:{executable_bits:o}\0".encode())
        if absolute.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(absolute).encode())
        else:
            digest.update(b"file\0")
            with absolute.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def copy_source_snapshot(
    repo_root: Path, snapshot_root: Path, paths: list[Path] | None = None
) -> list[Path]:
    """Copy exactly the runnable worktree files into an isolated local snapshot."""

    selected = source_files(repo_root) if paths is None else paths
    snapshot_root.mkdir(parents=True, exist_ok=True)
    if any(snapshot_root.iterdir()):
        raise FileExistsError(f"Snapshot directory is not empty: {snapshot_root}")
    for relative in selected:
        source = repo_root / relative
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, destination)
    return selected


def git_identity(
    repo_root: Path,
    *,
    snapshot_root: Path | None = None,
    paths: list[Path] | None = None,
) -> dict[str, Any]:
    selected = source_files(repo_root) if paths is None else paths
    identity_root = repo_root if snapshot_root is None else snapshot_root
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=normal")
    patch = _git(repo_root, "diff", "--binary", "HEAD")
    branch = _git(repo_root, "branch", "--show-current").decode().strip()
    return {
        "git_sha": _git(repo_root, "rev-parse", "HEAD").decode().strip(),
        "git_branch": branch or "detached",
        "dirty": bool(status.strip()),
        "git_status": status.decode().splitlines(),
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "source_snapshot_sha256": source_snapshot_sha256(identity_root, selected),
        "source_file_count": len(selected),
        "files": selected,
    }


def render_config(
    *,
    template_path: Path,
    output_path: Path,
    dispatch_file_path: Path,
    file_list_path: Path,
    snapshot_root: Path,
    repo_root: Path,
    run_id: str,
    run_root: Path,
    container_sif: Path,
    dvc_remote: Path,
    dvc_cache: Path,
    workflow_tooling_key: str,
    snakemake_version: str,
    snakemake_bin: Path,
) -> dict[str, Any]:
    paths = copy_source_snapshot(repo_root, snapshot_root)
    try:
        relative_template = template_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        snapshot_template = template_path
    else:
        snapshot_template = snapshot_root / relative_template

    document = yaml.safe_load(snapshot_template.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Workflow template must be a mapping: {template_path}")

    validate_code_data_separation(snapshot_root, paths, document)
    validate_workflow_contract(document)
    validate_mn5_allocation(document)

    identity = git_identity(repo_root, snapshot_root=snapshot_root, paths=paths)
    source_root = run_root / "source"
    document["run"] = {
        "id": run_id,
        "root": str(run_root),
        "source_root": str(source_root),
    }
    document["source"] = {
        key: value for key, value in identity.items() if key != "files"
    }
    document.setdefault("container", {})
    document["container"]["sif"] = str(container_sif)
    document["container"]["digest_file"] = f"{container_sif}.sha256"
    document.setdefault("workflow", {})
    document["workflow"].update(
        {
            "tooling_key": workflow_tooling_key,
            "snakemake_bin": str(snakemake_bin),
            "snakemake_version": snakemake_version,
            "execution_mode": "single-node-preallocated",
        }
    )
    document.setdefault("dvc", {})
    document["dvc"].update(
        {
            "remote": str(dvc_remote),
            "cache": str(dvc_cache),
        }
    )
    document.setdefault("provenance", {})
    document["provenance"].setdefault("training_evidence_retention", "run")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    )
    write_dispatch_request(output_path, dispatch_file_path)
    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    file_list_path.write_bytes(
        b"".join(os.fsencode(path.as_posix()) + b"\0" for path in identity["files"])
    )
    return document


def _dispatch_field(value: object, *, field: str) -> str:
    """Render one unambiguous TSV value for the remote Bash runner."""
    rendered = str(value)
    if not rendered or "\t" in rendered or "\n" in rendered or "\r" in rendered:
        raise ValueError(f"Dispatch field {field} must be one non-empty TSV value")
    return rendered


def write_dispatch_request(config_path: Path, output_path: Path) -> dict[str, str]:
    """Write the immutable, shell-safe bindings needed before Snakemake starts.

    The effective YAML remains the scientific workflow request.  This compact
    TSV record lets the transferred runner publish and submit that request
    without parsing YAML in Bash or receiving config-derived values over SSH.
    """
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Workflow config must be a mapping: {config_path}")

    allocation = validate_mn5_allocation(document)
    run = document["run"]
    source = document["source"]
    container = document["container"]
    workflow = document["workflow"]
    run_id = str(run["id"])
    run_root = Path(str(run["root"]))
    source_root = Path(str(run["source_root"]))
    snakemake_bin = Path(str(workflow["snakemake_bin"]))

    if not run_root.is_absolute() or run_root.name != run_id:
        raise ValueError("run.root must be absolute and end with run.id")
    if source_root != run_root / "source":
        raise ValueError("run.source_root must be the run-local source directory")
    if not snakemake_bin.is_absolute():
        raise ValueError("workflow.snakemake_bin must be absolute")

    fields: list[tuple[str, object]] = [
        ("schema_version", DISPATCH_REQUEST_SCHEMA_VERSION),
        ("run_id", run_id),
        ("run_root", run_root),
        ("run_parent", run_root.parent),
        ("source_root", source_root),
        ("source_snapshot_sha256", source["source_snapshot_sha256"]),
        ("config_sha256", hashlib.sha256(config_path.read_bytes()).hexdigest()),
        ("container_sif", container["sif"]),
        ("container_digest_file", container["digest_file"]),
        ("tooling_key", workflow["tooling_key"]),
        ("tooling_python", snakemake_bin.with_name("python")),
        ("snakemake_bin", snakemake_bin),
        ("snakemake_version", workflow["snakemake_version"]),
        ("workflow_kind", workflow_kind(document)),
        ("workflow_target", workflow_target(document)),
        *[(field, allocation[field]) for field in ALLOCATION_FIELDS],
    ]
    request = {field: _dispatch_field(value, field=field) for field, value in fields}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{field}\t{value}\n" for field, value in request.items())
    )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n"
    )
    return request


def verify_snapshot(config_path: Path, source_root: Path, file_list_path: Path) -> str:
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Workflow config must be a mapping: {config_path}")
    paths = [
        Path(os.fsdecode(value))
        for value in file_list_path.read_bytes().split(b"\0")
        if value
    ]
    actual_paths = sorted(
        (
            path.relative_to(source_root)
            for path in source_root.rglob("*")
            if path.is_file() or path.is_symlink()
        ),
        key=lambda path: path.as_posix(),
    )
    if actual_paths != paths:
        raise ValueError(
            "Transferred source file inventory does not match the snapshot"
        )
    actual = source_snapshot_sha256(source_root, paths)
    expected = str(document["source"]["source_snapshot_sha256"])
    if actual != expected:
        raise ValueError(
            f"Transferred source hash mismatch: expected {expected}, observed {actual}"
        )
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--template", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--dispatch-file", type=Path, required=True)
    render.add_argument("--file-list", type=Path, required=True)
    render.add_argument("--snapshot-root", type=Path, required=True)
    render.add_argument("--repo-root", type=Path, required=True)
    render.add_argument("--run-id", required=True)
    render.add_argument("--run-root", type=Path, required=True)
    render.add_argument("--container-sif", type=Path, required=True)
    render.add_argument("--dvc-remote", type=Path, required=True)
    render.add_argument("--dvc-cache", type=Path, required=True)
    render.add_argument("--workflow-tooling-key", required=True)
    render.add_argument("--snakemake-version", required=True)
    render.add_argument("--snakemake-bin", type=Path, required=True)

    allocation = subparsers.add_parser("allocation")
    allocation.add_argument("--config", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--file-list", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "render":
        render_config(
            template_path=args.template,
            output_path=args.output,
            dispatch_file_path=args.dispatch_file,
            file_list_path=args.file_list,
            snapshot_root=args.snapshot_root,
            repo_root=args.repo_root.resolve(),
            run_id=args.run_id,
            run_root=args.run_root,
            container_sif=args.container_sif,
            dvc_remote=args.dvc_remote,
            dvc_cache=args.dvc_cache,
            workflow_tooling_key=args.workflow_tooling_key,
            snakemake_version=args.snakemake_version,
            snakemake_bin=args.snakemake_bin,
        )
    elif args.command == "allocation":
        document = yaml.safe_load(args.config.read_text())
        if not isinstance(document, dict):
            raise ValueError(f"Workflow config must be a mapping: {args.config}")
        allocation = validate_mn5_allocation(document)
        print("\t".join(str(allocation[field]) for field in ALLOCATION_FIELDS))
    elif args.command == "verify":
        print(verify_snapshot(args.config, args.source_root, args.file_list))


if __name__ == "__main__":
    main()
