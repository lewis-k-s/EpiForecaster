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
# CPU-only preprocessing may run on one MN5 GPP node (112 cores) or on one
# Nord4 main node (48 cores). Both clusters share the same GPFS.
GPP_QOS = {"gp_bscls", "gp_debug"}
GPP_MAX_CPUS_PER_TASK = 112
GPP_GRES = "none"
NORD4_QOS = {"bsc_ls", "debug"}
NORD4_MAX_CPUS_PER_TASK = 48
CPU_PREPROCESS_PARTITIONS = {
    "gpp": ("GPP", "MN5 single-node GPP", GPP_QOS, GPP_MAX_CPUS_PER_TASK),
    "main": ("Nord4", "Nord4 single-node main", NORD4_QOS, NORD4_MAX_CPUS_PER_TASK),
}
DISPATCH_REQUEST_SCHEMA_VERSION = 1
RESEARCH_OUTPUT_FILES = (
    "research_fit_cases.parquet",
    "research_calibration.json",
    "research_scored.parquet",
    "research_summary.json",
    "research_events.parquet",
    "research_events.json",
    "calibration/test_summary.json",
    "calibration/test_node_metrics.csv",
    "calibration/test_node_metrics_per_head.csv",
    "calibration/test_granular.parquet",
)
WORKFLOW_KINDS = {
    "train",
    "evaluate",
    "hpo",
    "ablation",
    "ablation_suite",
    "crossval",
    "preprocess",
}
WORKFLOW_TARGETS = {
    "train": {"train", "train_evaluate"},
    "evaluate": {"evaluate"},
    "hpo": {"hpo"},
    "ablation": {"ablation"},
    "ablation_suite": {"ablation_suite"},
    "crossval": {"crossval"},
    # preprocess_qa stops after dataset QA and publishes nothing.
    "preprocess": {"preprocess", "preprocess_qa"},
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
SCIENTIFIC_SOURCE_SUFFIXES = {".py", ".sh", ".smk", ".toml", ".yaml", ".yml", ".json"}
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
    missing_bind_roots = sorted(
        {
            payload.parent
            for payload in payloads.values()
            if payload.parent != Path(".")
            and not any(payload.parent in path.parents for path in selected)
        },
        key=lambda path: path.as_posix(),
    )
    if missing_bind_roots:
        raise ValueError(
            "DVC mount parents must exist in the source inventory; place the "
            "pointer beside its payload: "
            + ", ".join(path.as_posix() for path in missing_bind_roots)
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


def _require_stale_recovery(options: dict[str, Any], *, field: str) -> None:
    """Rolling allocations must recover trials killed with their allocation.

    A wall-time kill leaves its trial RUNNING in the journal. Without recovery
    the identity is never claimed again and the campaign exits pending forever.
    """
    if options.get("recover_stale_running") is not True:
        raise ValueError(
            f"{field}.recover_stale_running must be true: a trial killed at the "
            "allocation wall time otherwise stays RUNNING and blocks completion"
        )


def _validate_container_pin(workflow_config: dict[str, Any]) -> None:
    container = workflow_config.get("container") or {}
    if not isinstance(container, dict):
        raise ValueError("container must be a mapping")
    pin = container.get("expected_sha256")
    if pin is not None and (
        not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None
    ):
        raise ValueError("container.expected_sha256 must be 64 lowercase hex digits")


def validate_workflow_contract(workflow_config: dict[str, Any]) -> str:
    """Validate mode-specific configuration before snapshot transfer."""
    kind = workflow_kind(workflow_config)
    workflow_target(workflow_config)
    _validate_container_pin(workflow_config)
    required_sections = {
        "train": ("training", "evaluation"),
        "evaluate": ("evaluation",),
        "hpo": ("optuna", "evaluation", "promotion"),
        "ablation": ("ablation",),
        "ablation_suite": ("ablation_suite",),
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
        if not isinstance(preprocessing.get("node_local_output", False), bool):
            raise ValueError("preprocessing.node_local_output must be Boolean")
        write_workers = preprocessing.get("write_workers", 1)
        cpus = (
            workflow_config.get("resources", {})
            .get("allocation", {})
            .get("cpus_per_task")
        )
        if (
            not isinstance(write_workers, int)
            or isinstance(write_workers, bool)
            or write_workers < 1
            or (isinstance(cpus, int) and write_workers > cpus)
        ):
            raise ValueError(
                "preprocessing.write_workers must be an integer between 1 and "
                "resources.allocation.cpus_per_task"
            )

    evaluation = workflow_config.get("evaluation", {})
    research_outputs = evaluation.get("research_outputs", False)
    if not isinstance(research_outputs, bool):
        raise ValueError("evaluation.research_outputs must be Boolean")
    if research_outputs and (
        kind not in {"train", "evaluate", "crossval"}
        or evaluation.get("split") != "val"
        or evaluation.get("granular") is not True
    ):
        raise ValueError(
            "Research outputs require train/evaluate/crossval with split=val and granular=true"
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
    elif kind == "ablation":
        ablation = workflow_config["ablation"]
        closeout_signal_seconds = ablation.get("closeout_signal_seconds")
        if (
            not isinstance(closeout_signal_seconds, int)
            or isinstance(closeout_signal_seconds, bool)
            or closeout_signal_seconds < 60
        ):
            raise ValueError(
                "ablation.closeout_signal_seconds must be an integer of at least 60"
            )
        _require_stale_recovery(ablation, field="ablation")
        timeout_s = ablation.get("timeout_s")
        if (
            not isinstance(timeout_s, int)
            or isinstance(timeout_s, bool)
            or timeout_s <= 0
        ):
            raise ValueError("ablation.timeout_s must be a positive integer")
    elif kind == "ablation_suite":
        suite = workflow_config["ablation_suite"]
        studies = suite.get("studies")
        if not isinstance(studies, list) or len(studies) < 2:
            raise ValueError("ablation_suite.studies must contain at least two studies")
        names: set[str] = set()
        campaigns: set[str] = set()
        for study in studies:
            if not isinstance(study, dict):
                raise ValueError("Each ablation suite study must be a mapping")
            name = study.get("study_name")
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
                raise ValueError(
                    "Ablation suite study_name must be a safe path component"
                )
            if name in names:
                raise ValueError(f"Duplicate ablation suite study_name: {name}")
            names.add(name)
            campaign = _safe_relative_path(
                str(study.get("campaign", "")), field="ablation suite campaign"
            ).as_posix()
            if campaign in campaigns:
                raise ValueError(f"Duplicate ablation suite campaign: {campaign}")
            campaigns.add(campaign)
            _require_stale_recovery(study, field=f"ablation_suite study {name}")
            workers = study.get("workers")
            if (
                not isinstance(workers, int)
                or isinstance(workers, bool)
                or workers != 4
            ):
                raise ValueError("Each ablation suite study must use four workers")
        closeout = suite.get("closeout_signal_seconds")
        if not isinstance(closeout, int) or isinstance(closeout, bool) or closeout < 60:
            raise ValueError(
                "ablation_suite.closeout_signal_seconds must be at least 60"
            )
        timeout_s = suite.get("timeout_s")
        if (
            not isinstance(timeout_s, int)
            or isinstance(timeout_s, bool)
            or timeout_s <= 0
        ):
            raise ValueError("ablation_suite.timeout_s must be a positive integer")
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


def _validate_cpu_allocation(allocation: dict[str, Any], kind: str) -> dict[str, Any]:
    """Validate a one-task CPU allocation (MN5 GPP or Nord4) for preprocessing.

    GPP grants memory per allocated CPU, so cpus_per_task sizes both the core
    count and the memory of the step. A full 48-CPU Nord4 node is exclusive.
    The dispatcher passes ``--gres=none``.
    """
    short, label, allowed_qos, max_cpus = CPU_PREPROCESS_PARTITIONS[
        allocation["partition"]
    ]
    if kind != "preprocess":
        raise ValueError(
            f"{short} allocations are supported only for workflow.kind='preprocess'"
        )
    if "gres" in allocation or "gpu" in allocation or "gpus" in allocation:
        raise ValueError(f"A {short} allocation must not request GPUs or gres")
    expected = {"account": "bsc08", "nodes": 1, "tasks": 1}
    mismatches = {
        key: (allocation.get(key), value)
        for key, value in expected.items()
        if allocation.get(key) != value
    }
    qos = allocation.get("qos")
    if qos not in allowed_qos:
        mismatches["qos"] = (qos, " or ".join(sorted(allowed_qos)))
    cpus = allocation.get("cpus_per_task")
    if not isinstance(cpus, int) or isinstance(cpus, bool) or not 1 <= cpus <= max_cpus:
        mismatches["cpus_per_task"] = (cpus, f"integer 1..{max_cpus}")
    runtime = allocation.get("runtime")
    if not isinstance(runtime, int) or isinstance(runtime, bool) or runtime <= 0:
        mismatches["runtime"] = (runtime, "positive integer minutes")
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Invalid {label} allocation: {details}")
    return {
        field: (GPP_GRES if field == "gres" else allocation[field])
        for field in ALLOCATION_FIELDS
    }


def validate_mn5_allocation(workflow_config: dict[str, Any]) -> dict[str, Any]:
    """Validate the one-node allocation for the selected workflow kind.

    ACC is the default for every kind. Preprocessing may instead use one GPP
    or Nord4 main task, because it runs no GPU step.
    """
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
    if allocation.get("partition") in CPU_PREPROCESS_PARTITIONS:
        return _validate_cpu_allocation(allocation, kind)
    if kind == "hpo":
        workload = workflow_config.get("optuna")
        workload_name = "optuna"
    elif kind == "ablation":
        workload = workflow_config.get("ablation")
        workload_name = "ablation"
    elif kind == "ablation_suite":
        workload = {"workers": 4}
        workload_name = "ablation_suite"
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
    runtime = allocation.get("runtime")
    qos = allocation.get("qos")
    if qos not in {"acc_bscls", "acc_debug", "acc_interactive"}:
        mismatches["qos"] = (qos, "acc_bscls, acc_debug, or acc_interactive")
    if qos == "acc_interactive" and expected_tasks > 2:
        mismatches["tasks"] = (
            expected_tasks,
            "at most 2 for acc_interactive's 40-CPU ceiling",
        )
    if qos == "acc_interactive" and isinstance(runtime, int) and runtime > 120:
        mismatches["runtime"] = (
            runtime,
            "at most 120 minutes for acc_interactive",
        )
    if not isinstance(runtime, int) or isinstance(runtime, bool) or runtime <= 0:
        mismatches["runtime"] = (runtime, "positive integer minutes")
    if kind == "ablation":
        ablation = workflow_config["ablation"]
        expected_timeout_s = runtime * 60 - ablation["closeout_signal_seconds"]
        if ablation["timeout_s"] != expected_timeout_s:
            raise ValueError(
                "ablation.timeout_s must equal resources.allocation.runtime "
                "minus ablation.closeout_signal_seconds"
            )
    if kind == "ablation_suite":
        suite = workflow_config["ablation_suite"]
        if suite["timeout_s"] != runtime * 60 - suite["closeout_signal_seconds"]:
            raise ValueError(
                "ablation_suite.timeout_s must equal allocation runtime minus closeout"
            )
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


MAX_TEMPLATE_EXTENDS_DEPTH = 4


def _merge_template(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings; the child wins and lists or scalars are replaced."""
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_template(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_template(template_path: Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one template and resolve its `extends` chain into one mapping.

    `extends` is a path relative to the extending file. The resolved mapping
    never contains the `extends` key, so rendered configs stay standalone.
    """
    resolved_path = template_path.resolve()
    if resolved_path in _chain:
        raise ValueError(f"Workflow template extends cycle at {template_path}")
    if len(_chain) >= MAX_TEMPLATE_EXTENDS_DEPTH:
        raise ValueError(
            f"Workflow template extends depth exceeds "
            f"{MAX_TEMPLATE_EXTENDS_DEPTH}: {template_path}"
        )
    document = yaml.safe_load(template_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Workflow template must be a mapping: {template_path}")
    parent_ref = document.pop("extends", None)
    if parent_ref is None:
        return document
    if not isinstance(parent_ref, str):
        raise ValueError(f"Workflow template extends must be a path: {template_path}")
    parent_path = template_path.parent / parent_ref
    if not parent_path.is_file():
        raise ValueError(f"Workflow template parent not found: {parent_path}")
    parent = load_template(parent_path, (*_chain, resolved_path))
    return _merge_template(parent, document)


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

    document = load_template(snapshot_template)

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
        (
            "closeout_signal_seconds",
            document["ablation"]["closeout_signal_seconds"]
            if workflow_kind(document) == "ablation"
            else document["ablation_suite"]["closeout_signal_seconds"]
            if workflow_kind(document) == "ablation_suite"
            else 0,
        ),
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


def parse_build_metadata(text: str) -> dict[str, str]:
    """Parse the ``key=value`` build sidecar written by the SIF assembler."""
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip():
            metadata[key.strip()] = value.strip()
    return metadata


def verify_container_identity(
    *,
    resolved_sif: str,
    metadata_text: str,
    observed_sha256: str,
    uv_lock_sha256: str,
    expected_sha256: str | None = None,
) -> None:
    """Require the runtime image to match the bundle's dependency lock.

    The image alias can point at an image built from an older ``uv.lock``. The
    assembler records the lock hash in the build sidecar, so compare it with the
    lock that travels in the source bundle.
    """
    metadata = parse_build_metadata(metadata_text)
    recorded_sif = metadata.get("sif_sha256")
    if recorded_sif is not None and recorded_sif != observed_sha256:
        raise ValueError(
            f"Container build metadata sif_sha256 {recorded_sif} does not match "
            f"the observed image digest {observed_sha256}: {resolved_sif}"
        )
    if expected_sha256 is not None and expected_sha256 != observed_sha256:
        raise ValueError(
            f"Container {resolved_sif} has digest {observed_sha256}, but the "
            f"template pins container.expected_sha256={expected_sha256}"
        )
    image_lock = metadata.get("uv_lock_sha256")
    if image_lock != uv_lock_sha256:
        raise ValueError(
            f"Container {resolved_sif} was built from uv.lock "
            f"{image_lock or '(unrecorded)'}, but the source bundle uv.lock is "
            f"{uv_lock_sha256}. Set REMOTE_CONTAINER_SIF to the digest-named "
            "epiforecaster-runtime-<build_key>.sif image whose .build.txt lists "
            f"uv_lock_sha256={uv_lock_sha256}."
        )


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

    container = subparsers.add_parser("verify-container")
    container.add_argument("--config", type=Path, required=True)
    container.add_argument("--resolved-sif", required=True)
    container.add_argument("--metadata-file", type=Path, required=True)
    container.add_argument("--observed-sha256", required=True)
    container.add_argument("--uv-lock", type=Path, required=True)
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
    elif args.command == "verify-container":
        document = yaml.safe_load(args.config.read_text())
        pin = (document.get("container") or {}).get("expected_sha256")
        try:
            verify_container_identity(
                resolved_sif=args.resolved_sif,
                metadata_text=args.metadata_file.read_text(),
                observed_sha256=args.observed_sha256,
                uv_lock_sha256=hashlib.sha256(args.uv_lock.read_bytes()).hexdigest(),
                expected_sha256=pin,
            )
        except ValueError as error:
            raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
