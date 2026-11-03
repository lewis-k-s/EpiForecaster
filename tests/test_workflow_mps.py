"""CUDA MPS packing: allocation contract, rank-to-GPU mapping, and daemon lifetime."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from scripts.workflow import run_config
from scripts.workflow import run_support
from tests.test_workflow_support import _make_remote_run_staging, _workflow_config

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "scripts/workflow"
GPU_UUIDS = ("GPU-aaaaaaaa-0000", "GPU-bbbbbbbb-1111")

SRUN = """#!/bin/bash
trap '' USR1
echo "$*" >> "$EVENTS_DIR/srun.log"
args=("$@")
index=-1
position=0
for argument in "$@"; do
  case "$argument" in *run_parallel_worker.sh) index=$position ;; esac
  position=$((position + 1))
done
if [ "$index" -lt 0 ]; then exit 0; fi
tasks=1
for argument in "$@"; do
  case "$argument" in --ntasks=*) tasks="${argument#--ntasks=}" ;; esac
done
pids=()
for ((rank = 0; rank < tasks; rank++)); do
  SLURM_PROCID=$rank SLURM_STEP_ID=7 "${args[@]:$index}" </dev/null &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
"""

NVIDIA_SMI = """#!/bin/bash
for uuid in $GPU_LIST; do echo "$uuid"; done
"""

MPS_CONTROL = """#!/bin/bash
if [ "${1:-}" = "-d" ]; then
  echo "start pipe=$CUDA_MPS_PIPE_DIRECTORY log=$CUDA_MPS_LOG_DIRECTORY visible=$CUDA_VISIBLE_DEVICES" >> "$EVENTS_DIR/events.log"
  : > "$CUDA_MPS_PIPE_DIRECTORY/control"
  echo started > "$CUDA_MPS_LOG_DIRECTORY/control.log"
else
  read -r command
  echo "$command pipe=$CUDA_MPS_PIPE_DIRECTORY" >> "$EVENTS_DIR/events.log"
fi
"""

# Rank 0 of every GPU exits first so the leader must wait for its peers.
SINGULARITY = """#!/bin/bash
echo "$*" > "$EVENTS_DIR/run.$SLURM_PROCID"
echo "run rank=$SLURM_PROCID" >> "$EVENTS_DIR/events.log"
if [ "${SING_MODE:-}" = "usr1" ]; then
  # An ignored signal stays ignored across exec; Python installs its handler.
  exec "$PYTHON" -c '
import os, signal, sys, time
rank = os.environ["SLURM_PROCID"]
events = os.environ["EVENTS_DIR"]
done = []
signal.signal(signal.SIGUSR1, lambda *_: done.append(1))
open(events + "/ready." + rank, "w").close()
deadline = time.time() + 20
while not done and time.time() < deadline:
    time.sleep(0.02)
with open(events + "/events.log", "a") as log:
    log.write(("usr1 rank=" + rank if done else "timeout rank=" + rank) + "\\n")
sys.exit(0 if done else 3)
'
fi
if [ $((SLURM_PROCID % ${SING_GROUP:-4})) -ne 0 ]; then sleep 1; fi
echo "exit rank=$SLURM_PROCID" >> "$EVENTS_DIR/events.log"
exit "${SING_STATUS:-0}"
"""


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    source_root = tmp_path / "source"
    scripts = source_root / "scripts/workflow"
    scripts.mkdir(parents=True)
    for name in (
        "launch_parallel_workers.sh",
        "run_parallel_worker.sh",
        "mn5_container_exec.sh",
        "node_input_cache.sh",
    ):
        shutil.copy2(WORKFLOW / name, scripts / name)
    run_root = tmp_path / "run"
    (run_root / ".state/dvc-workspace").mkdir(parents=True)
    (run_root / ".state/dvc_bind_roots.txt").write_text("")
    (run_root / ".state/data_manifest.json").write_text("{}")
    (run_root / "config.yaml").write_text("{}\n")
    events = tmp_path / "events"
    events.mkdir()
    jobtmp = tmp_path / "jobtmp"
    jobtmp.mkdir()
    (tmp_path / "runtime.sif").touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in (
        ("srun", SRUN),
        ("nvidia-smi", NVIDIA_SMI),
        ("nvidia-cuda-mps-control", MPS_CONTROL),
        ("singularity", SINGULARITY),
    ):
        (fake_bin / name).write_text(body)
        (fake_bin / name).chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EVENTS_DIR": str(events),
        "GPU_LIST": " ".join(GPU_UUIDS),
        "PYTHON": sys.executable,
        "TMPDIR": str(jobtmp),
        "EPIFORECASTER_NODE_CACHE_ROOT": str(tmp_path / "node-cache"),
        "SLURM_JOB_ID": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "8",
        "SLURM_CPUS_PER_TASK": "5",
    }
    return source_root, run_root, events, env


def _launch(
    source_root: Path,
    run_root: Path,
    env: dict[str, str],
    *,
    workers: int,
    cpus: int,
    mps: int | None,
    **kwargs,
):
    command = [
        str(source_root / "scripts/workflow/launch_parallel_workers.sh"),
        "--run-root",
        str(run_root),
        "--source-root",
        str(source_root),
        "--sif",
        str(run_root.parent / "runtime.sif"),
        "--workers",
        str(workers),
        "--cpus-per-task",
        str(cpus),
        "--worker-group",
        "hpo",
    ]
    if mps is not None:
        command += ["--mps-clients-per-gpu", str(mps)]
    command += ["--", "true"]
    return subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )


def _events(events: Path) -> list[str]:
    path = events / "events.log"
    return path.read_text().splitlines() if path.exists() else []


def _run(source_root, run_root, env, **kwargs):
    process = _launch(source_root, run_root, env, **kwargs)
    output, _ = process.communicate(timeout=120)
    return process.returncode, output


def test_mps_ranks_map_to_gpus_and_daemons_outlive_every_client(tmp_path: Path) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)

    status, output = _run(source_root, run_root, env, workers=8, cpus=5, mps=4)

    assert status == 0, output
    lines = _events(events)
    srun_line = (events / "srun.log").read_text().splitlines()[0]
    assert "--gres=gpu:2" in srun_line
    assert "--gpu-bind=none" in srun_line
    assert "--gpus-per-task" not in srun_line
    assert "--ntasks=8" in srun_line
    assert "--mps-clients-per-gpu 4" in srun_line

    starts = [line for line in lines if line.startswith("start ")]
    assert len(starts) == 2
    jobtmp = str(tmp_path / "jobtmp")
    pipes = set()
    for start, uuid in zip(
        sorted(starts, key=lambda s: s.split("visible=")[1]), GPU_UUIDS
    ):
        assert f"visible={uuid}" in start
        pipe = start.split("pipe=")[1].split()[0]
        assert pipe.startswith(jobtmp) and not pipe.startswith("/tmp/")
        assert start.split("log=")[1].split()[0].startswith(jobtmp)
        pipes.add(pipe)
    assert len(pipes) == 2

    runs = {rank: (events / f"run.{rank}").read_text().strip() for rank in range(8)}
    assert sorted(
        int(line.split("=")[1]) for line in lines if line.startswith("run ")
    ) == list(range(8))
    for rank, line in runs.items():
        uuid = GPU_UUIDS[rank // 4]
        assert " --contain " in line and "--containall" not in line
        assert f"CUDA_VISIBLE_DEVICES={uuid}" in line
        pipe = next(p for p in pipes if f"CUDA_MPS_PIPE_DIRECTORY={p} " in line + " ")
        assert f"{pipe}:{pipe}:rw" in line
        assert f"EPIFORECASTER_WORKER_ID={rank} " in line + " "
        assert (run_root / f".state/workers/hpo/{rank:03d}.done").is_file()

    # Each daemon quits only after all four clients on its GPU exited.
    quits = [(i, line) for i, line in enumerate(lines) if line.startswith("quit ")]
    assert len(quits) == 2
    for index, line in quits:
        pipe = line.split("pipe=")[1]
        owner = next(
            rank
            for rank in range(8)
            if f"CUDA_MPS_PIPE_DIRECTORY={pipe} " in runs[rank] + " "
        )
        group = range(owner // 4 * 4, owner // 4 * 4 + 4)
        exits = [
            i
            for i, text in enumerate(lines)
            if text.startswith("exit ") and int(text.split("=")[1]) in group
        ]
        assert len(exits) == 4 and max(exits) < index
    assert (run_root / "logs/mps").is_dir()


def test_single_client_per_gpu_is_unchanged(tmp_path: Path) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)
    env = env | {"SLURM_NTASKS": "2", "SLURM_CPUS_PER_TASK": "20", "SING_GROUP": "1"}

    status, output = _run(source_root, run_root, env, workers=2, cpus=20, mps=None)

    assert status == 0, output
    srun_line = (events / "srun.log").read_text().splitlines()[0]
    assert "--gpus-per-task=1" in srun_line
    assert "--gpu-bind" not in srun_line and "--gres" not in srun_line
    lines = _events(events)
    assert not any(line.startswith(("start ", "quit ")) for line in lines)
    runs = [(events / f"run.{rank}").read_text() for rank in range(2)]
    for line in runs:
        assert "--containall" in line
        assert " --contain " not in line
        assert "CUDA_MPS_PIPE_DIRECTORY" not in line
    assert not (tmp_path / "jobtmp/mps").exists()


def test_failed_client_still_stops_daemons(tmp_path: Path) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)
    env = env | {
        "SLURM_NTASKS": "2",
        "SLURM_CPUS_PER_TASK": "10",
        "SING_STATUS": "9",
        "SING_GROUP": "1",
    }

    status, _ = _run(source_root, run_root, env, workers=2, cpus=10, mps=2)

    assert status == 9
    lines = _events(events)
    assert len([line for line in lines if line.startswith("start ")]) == 1
    assert len([line for line in lines if line.startswith("quit ")]) == 1


def test_closeout_signal_reaches_clients_and_daemons_stop(tmp_path: Path) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)
    env = env | {
        "SLURM_NTASKS": "2",
        "SLURM_CPUS_PER_TASK": "10",
        "SING_MODE": "usr1",
    }
    process = _launch(
        source_root, run_root, env, workers=2, cpus=10, mps=2, start_new_session=True
    )
    try:
        for _ in range(500):
            if (events / "ready.0").exists() and (events / "ready.1").exists():
                break
            time.sleep(0.02)
        assert (events / "ready.0").exists() and (events / "ready.1").exists()
        # Slurm signals every process of the step (not the launcher's caller);
        # the wrappers must ignore it and only the client handles it.
        pattern = (
            f"{tmp_path}/(source/scripts/workflow/"
            r"(run_parallel_worker|mn5_container_exec)\.sh|bin/singularity)"
            r"|signal\.SIGUSR1"
        )
        subprocess.run(["pkill", "-USR1", "-f", pattern], check=True)
        output, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
    assert process.returncode == 0, output
    lines = _events(events)
    assert sorted(line for line in lines if line.startswith("usr1 ")) == [
        "usr1 rank=0",
        "usr1 rank=1",
    ]
    assert len([line for line in lines if line.startswith("quit ")]) == 1


@pytest.mark.parametrize(
    ("workers", "cpus", "mps", "ntasks"),
    [
        (8, 20, 4, "8"),  # cpus do not fit the 20-per-GPU contract
        (6, 5, 4, "6"),  # workers are not a multiple of K
        (20, 4, 4, "20"),  # more than four GPUs
        (8, 5, 4, "4"),  # SLURM_NTASKS differs from the single MPS wave
    ],
)
def test_launcher_rejects_inconsistent_mps_requests(
    tmp_path: Path, workers: int, cpus: int, mps: int, ntasks: str
) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)
    env = env | {"SLURM_NTASKS": ntasks, "SLURM_CPUS_PER_TASK": str(cpus)}

    status, _ = _run(source_root, run_root, env, workers=workers, cpus=cpus, mps=mps)

    assert status != 0
    assert not (events / "srun.log").exists()


def test_missing_gpu_uuid_fails_before_any_daemon(tmp_path: Path) -> None:
    source_root, run_root, events, env = _fixture(tmp_path)
    env = env | {"GPU_LIST": GPU_UUIDS[0]}

    status, output = _run(source_root, run_root, env, workers=8, cpus=5, mps=4)

    assert status != 0
    assert "No GPU UUID" in output
    # Ranks 4-7 have no GPU, so their clients never start.
    assert not any((events / f"run.{rank}").exists() for rank in range(4, 8))


def _allocation_config(tmp_path: Path, **allocation) -> dict:
    run_root = tmp_path / "runs/test-run"
    config = _workflow_config(run_root, run_root / "source")
    config["optuna"]["workers"] = 8
    config["resources"]["allocation"] |= {
        "qos": "acc_debug",
        "tasks": 8,
        "cpus_per_task": 5,
        "gres": "gpu:2",
        "mps_clients_per_gpu": 4,
    } | allocation
    return config


def test_allocation_accepts_mps_packing(tmp_path: Path) -> None:
    allocation = run_config.validate_mn5_allocation(_allocation_config(tmp_path))

    assert allocation["mps_clients_per_gpu"] == 4
    assert allocation["tasks"] == 8
    assert allocation["cpus_per_task"] == 5
    assert allocation["gres"] == "gpu:2"


def test_allocation_defaults_to_one_client_per_gpu(tmp_path: Path) -> None:
    config = _workflow_config(tmp_path / "run", tmp_path / "run/source")

    allocation = run_config.validate_mn5_allocation(config)

    assert allocation["mps_clients_per_gpu"] == 1
    assert allocation["cpus_per_task"] == 20


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tasks": 4}, "tasks=4"),
        ({"cpus_per_task": 10}, "cpus_per_task=10"),
        ({"gres": "gpu:4"}, "gres='gpu:4'"),
        ({"mps_clients_per_gpu": 3}, "multiple of"),
        ({"mps_clients_per_gpu": 0}, "at least 1"),
        ({"mps_clients_per_gpu": True}, "at least 1"),
        ({"mps_clients_per_gpu": 1}, "between 1 and 4"),
    ],
)
def test_allocation_rejects_inconsistent_mps_packing(
    tmp_path: Path, override: dict, message: str
) -> None:
    with pytest.raises(ValueError) as error:
        run_config.validate_mn5_allocation(_allocation_config(tmp_path, **override))

    assert message in str(error.value)


def test_allocation_rejects_mps_for_other_workflow_kinds(tmp_path: Path) -> None:
    config = _allocation_config(tmp_path, tasks=1, cpus_per_task=20, gres="gpu:1")
    config["workflow"]["kind"] = "train"
    config["training"] = {
        "base_config": "configs/x.yaml",
        "experiment_name": "e",
        "model_id": "m",
    }
    config["evaluation"] = {"split": "test"}
    config["resources"]["allocation"]["mps_clients_per_gpu"] = 2

    with pytest.raises(ValueError, match="only for"):
        run_config.validate_mn5_allocation(config)


def test_dispatch_request_carries_mps_clients(tmp_path: Path) -> None:
    config = _allocation_config(tmp_path)
    config["container"] = {
        "sif": "/containers/runtime.sif",
        "digest_file": "/containers/runtime.sif.sha256",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    request = run_config.write_dispatch_request(path, tmp_path / "dispatch.tsv")

    assert request["mps_clients_per_gpu"] == "4"
    assert request["tasks"] == "8"
    assert request["gres"] == "gpu:2"


def test_core_mps_debug_template_is_valid() -> None:
    template = run_config.load_template(
        REPO_ROOT / "workflow/config/mn5_hpo_core_mps_debug.yaml"
    )

    allocation = run_config.validate_mn5_allocation(template)

    assert allocation["mps_clients_per_gpu"] == 4
    assert template["optuna"]["search_space"] == "core"
    assert template["optuna"]["workers"] == 8
    assert template["evaluation"]["split"] == "val"
    assert template["optuna"]["timeout_s"] == allocation["runtime"] * 60 - 900


def _replicate_template() -> dict:
    return run_config.load_template(
        REPO_ROOT / "workflow/config/mn5_hpo_core_replicates.yaml"
    )


def test_replicate_template_queues_every_trial_it_runs() -> None:
    template = _replicate_template()

    allocation = run_config.validate_mn5_allocation(template)

    optuna = template["optuna"]
    queue = optuna["enqueue_trials"]
    assert allocation["mps_clients_per_gpu"] == 4
    assert optuna["workers"] * optuna["trials_per_worker"] == len(
        queue["params"]
    ) * len(queue["seeds"])
    assert optuna["timeout_s"] <= allocation["runtime"] * 60 - 900


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trials_per_worker", 5, "must equal the 32 enqueued trials"),
        ("pruning_start_epoch", 15, "pruning_start_epoch >= epochs"),
    ],
)
def test_replicate_queue_rejects_sampling_or_pruning(field, value, message) -> None:
    template = _replicate_template()
    template["optuna"][field] = value

    with pytest.raises(ValueError, match=message):
        run_config.validate_workflow_contract(template)


def _mps_allocation(config: dict) -> None:
    config["optuna"]["workers"] = 8
    config["resources"]["allocation"] |= {
        "qos": "acc_debug",
        "tasks": 8,
        "cpus_per_task": 5,
        "gres": "gpu:2",
        "mps_clients_per_gpu": 4,
    }


def test_transferred_runner_accepts_mps_dispatch(tmp_path: Path) -> None:
    staging, run_root, env = _make_remote_run_staging(
        tmp_path, token="mps", configure=_mps_allocation
    )

    result = subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    dispatch = (run_root / ".state/dispatch.tsv").read_text()
    assert "mps_clients_per_gpu\t4\n" in dispatch


def test_transferred_runner_rejects_inconsistent_mps_dispatch(tmp_path: Path) -> None:
    staging, run_root, env = _make_remote_run_staging(tmp_path, token="mpsbad")
    dispatch = staging / ".state/dispatch.tsv"
    if not dispatch.exists():
        dispatch = next(staging.rglob("dispatch.tsv"))
    text = dispatch.read_text().replace(
        "mps_clients_per_gpu\t1", "mps_clients_per_gpu\t3"
    )
    assert text != dispatch.read_text()
    dispatch.write_text(text)
    digest = run_support.sha256_file(dispatch)
    dispatch.with_suffix(".tsv.sha256").write_text(f"{digest}  dispatch.tsv\n")

    result = subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not run_root.exists()
