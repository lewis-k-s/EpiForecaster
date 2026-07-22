"""Shared path and command helpers for the modular Snakemake DAG."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
from typing import Any

from scripts.workflow.run_config import stage_source_files


CONTAINER_RUN = Path("/workspace/run")
CONTAINER_SOURCE = Path("/workspace/source")


def read_source_inventory(run_root: Path) -> list[Path]:
    inventory = run_root / ".state/source-files.bin"
    if not inventory.is_file():
        raise FileNotFoundError(f"Prepared source inventory is missing: {inventory}")
    return [
        Path(os.fsdecode(value))
        for value in inventory.read_bytes().split(b"\0")
        if value
    ]


def dependency_sources(
    source_root: Path,
    inventory: list[Path],
    config: dict[str, Any],
    stage: str,
) -> list[str]:
    selected = stage_source_files(inventory, config, stage)
    if not selected:
        raise ValueError(f"Prepared run has no {stage!r} source dependencies")
    return [str(source_root / relative) for relative in selected]


def source_fingerprint(source_root: Path, source_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for source_value in sorted(source_paths):
        source = Path(source_value)
        relative = source.relative_to(source_root)
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def container_command(
    arguments: list[object],
    *,
    run_root: Path,
    source_root: Path,
    sif: str,
    dvc_remote: str,
    dvc_cache: str,
    stage_data: bool = False,
    nv: bool = False,
    dvc_mode: str | None = None,
    cpus: int | None = None,
    promotion_root: str | None = None,
    input_paths: tuple[Path, ...] = (),
) -> str:
    command = [
        str(source_root / "scripts/workflow/mn5_container_exec.sh"),
        "--run-root",
        str(run_root),
        "--source-root",
        str(source_root),
        "--sif",
        sif,
    ]
    if dvc_mode is not None:
        command.extend(
            [
                "--dvc-mode",
                dvc_mode,
                "--dvc-remote",
                dvc_remote,
                "--dvc-cache",
                dvc_cache,
            ]
        )
        if dvc_mode == "push":
            if promotion_root is None:
                raise ValueError("DVC push requires a promotion root")
            command.extend(["--promotion-root", promotion_root])
    for input_path in input_paths:
        command.extend(["--input-path", str(input_path)])
    if stage_data:
        command.append("--stage-data")
    if nv:
        command.append("--nv")
    if cpus is not None:
        command.extend(["--cpus", str(cpus)])
    command.extend(["--", *[str(value) for value in arguments]])
    return shlex.join(command)


def slurm_step(command: str, cpus_per_task: int, *, gpu: bool = True) -> str:
    step = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus_per_task}",
        "--exact",
        "--kill-on-bad-exit=1",
    ]
    if gpu:
        step.append("--gpus-per-task=1")
    return f"{shlex.join(step)} {command}"


def launch_workers_command(
    arguments: list[object],
    *,
    run_root: Path,
    source_root: Path,
    sif: str,
    workers: int,
    cpus_per_task: int,
) -> str:
    return shlex.join(
        [
            str(source_root / "scripts/workflow/launch_optuna_workers.sh"),
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--sif",
            sif,
            "--workers",
            str(workers),
            "--cpus-per-task",
            str(cpus_per_task),
            "--",
            *[str(value) for value in arguments],
        ]
    )


def hpo_worker_arguments(config: dict[str, Any]) -> list[object]:
    options = config["optuna"]
    study_name = str(options["study_name"])
    arguments: list[object] = [
        "python",
        "-m",
        "scripts.hpo.hpsearch_worker",
        "--config",
        CONTAINER_SOURCE / str(options["base_config"]),
        "--study-name",
        study_name,
        "--journal-file",
        CONTAINER_RUN / "optuna" / f"{study_name}.journal",
        "--run-root",
        CONTAINER_RUN / "optuna/trials",
        "--n-trials",
        int(options["trials_per_worker"]),
        "--timeout-s",
        int(options["timeout_s"]),
        "--epochs",
        int(options["epochs"]),
        "--seed",
        int(options["seed"]),
        "--sampler",
        options["sampler"],
        "--pruning-start-epoch",
        int(options["pruning_start_epoch"]),
        "--search-space",
        options["search_space"],
    ]
    if options.get("max_batches") is not None:
        arguments.extend(["--max-batches", int(options["max_batches"])])
    for override in options.get("overrides", []):
        arguments.extend(["--override", str(override)])
    arguments.extend(["--override", "training.run_final_evaluation=false"])
    if options.get("sample_training_loss_objective", False):
        arguments.append("--sample-training-loss-objective")
    return arguments


def ablation_worker_arguments(config: dict[str, Any]) -> list[object]:
    options = config["ablation"]
    study_name = str(options["study_name"])
    arguments: list[object] = [
        "python",
        "-m",
        "scripts.hpo.ablation_worker",
        "--config",
        CONTAINER_SOURCE / str(options["base_config"]),
        "--study-name",
        study_name,
        "--journal-file",
        CONTAINER_RUN / "optuna" / f"{study_name}.journal",
        "--campaign-id",
        options["campaign_id"],
        "--seeds",
        " ".join(str(seed) for seed in options["seeds"]),
        "--run-root",
        CONTAINER_RUN / "outputs/training",
        "--timeout-s",
        int(options["timeout_s"]),
        "--seed",
        int(options["sampler_seed"]),
    ]
    if options.get("ablations") is not None:
        arguments.extend(["--ablations", str(options["ablations"])])
    if options.get("overrides"):
        arguments.extend(
            ["--overrides", " ".join(str(value) for value in options["overrides"])]
        )
    if options.get("trials_per_worker") is not None:
        arguments.extend(["--n-trials", int(options["trials_per_worker"])])
    if options.get("smoketest", False):
        arguments.append("--smoketest")
    if options.get("requeue_running", False):
        arguments.append("--requeue-running")
    return arguments
