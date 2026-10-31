"""Close-out contracts: signal survival, stale markers, and rank receipts."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "scripts/workflow"

CHILD = """
import signal, sys, time
from pathlib import Path
marker = Path(sys.argv[1])
handled = []
signal.signal(signal.SIGUSR1, lambda *_: handled.append(1))
(marker.parent / "ready").write_text("ready")
deadline = time.time() + 20
while not handled and time.time() < deadline:
    time.sleep(0.02)
marker.write_text("handled" if handled else "timeout")
sys.exit(0 if handled else 3)
"""


def test_closeout_signal_reaches_python_and_wrappers_survive(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    scripts = source_root / "scripts/workflow"
    scripts.mkdir(parents=True)
    for name in (
        "run_parallel_worker.sh",
        "mn5_container_exec.sh",
        "node_input_cache.sh",
    ):
        shutil.copy2(WORKFLOW / name, scripts / name)
    run_root = tmp_path / "run"
    (run_root / ".state/dvc-workspace").mkdir(parents=True)
    (run_root / ".state/workers/ablation").mkdir(parents=True)
    (run_root / ".state/dvc_bind_roots.txt").write_text("")
    (run_root / ".state/data_manifest.json").write_text("{}")
    (run_root / "config.yaml").write_text("{}\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sif = tmp_path / "runtime.sif"
    sif.touch()
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # The stub replaces Singularity and runs the Python child directly, as the
    # container would deliver the step signal to the scientific process.
    singularity = fake_bin / "singularity"
    singularity.write_text(f'#!/bin/bash\nexec {sys.executable} {child} "$MARKER"\n')
    singularity.chmod(0o755)
    marker = tmp_path / "out/handled.txt"
    marker.parent.mkdir()
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TMPDIR": str(scratch),
        "MARKER": str(marker),
        "EPIFORECASTER_NODE_CACHE_ROOT": str(tmp_path / "node-cache"),
        "SLURM_JOB_ID": "1",
        "SLURM_STEP_ID": "0",
        "SLURM_PROCID": "0",
    }
    process = subprocess.Popen(
        [
            str(scripts / "run_parallel_worker.sh"),
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--sif",
            str(sif),
            "--cpus",
            "20",
            "--worker-group",
            "ablation",
            "--",
            "unused",
        ],
        env=env,
        start_new_session=True,
    )
    try:
        for _ in range(500):
            if (marker.parent / "ready").exists():
                break
            time.sleep(0.02)
        assert (marker.parent / "ready").exists()
        # Slurm signals every process of the step, not only the Python child.
        os.killpg(process.pid, signal.SIGUSR1)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
    assert marker.read_text() == "handled"
    done = run_root / ".state/workers/ablation/000.done"
    assert "outcome=complete" in done.read_text()


def _launcher_fixture(tmp_path: Path, worker_body: str) -> tuple[Path, Path, dict]:
    source_root = tmp_path / "source"
    scripts = source_root / "scripts/workflow"
    scripts.mkdir(parents=True)
    shutil.copy2(WORKFLOW / "launch_parallel_workers.sh", scripts)
    (scripts / "mn5_container_exec.sh").write_text("#!/bin/bash\nexit 75\n")
    (scripts / "mn5_container_exec.sh").chmod(0o755)
    (scripts / "run_parallel_worker.sh").write_text(worker_body)
    (scripts / "run_parallel_worker.sh").chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # srun runs the worker script (last positional pieces) or the check step.
    (fake_bin / "srun").write_text(
        "#!/bin/bash\n"
        'if [[ " $* " == *" --cpus-per-task=1 "* ]]; then exit 75; fi\n'
        'for a in "$@"; do\n'
        '  case "$a" in *run_parallel_worker.sh) exec "$a" --fake;; esac\n'
        "done\nexit 0\n"
    )
    (fake_bin / "srun").chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "20",
        "EPIFORECASTER_EXECUTION_ATTEMPT_ID": "attempt-new",
    }
    return source_root, tmp_path / "run", env


def _run_launcher(source_root: Path, run_root: Path, env: dict, command: list[str]):
    return subprocess.run(
        [
            str(source_root / "scripts/workflow/launch_parallel_workers.sh"),
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--sif",
            str(run_root.parent / "runtime.sif"),
            "--workers",
            "1",
            "--cpus-per-task",
            "20",
            "--worker-group",
            "ablation_test",
            "--",
            *command,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launcher_removes_stale_marker_before_a_failing_step(tmp_path: Path) -> None:
    source_root, run_root, env = _launcher_fixture(tmp_path, "#!/bin/bash\nexit 9\n")
    state = run_root / ".state"
    state.mkdir(parents=True)
    (state / "continuation.required").write_text("attempt_id\told\n")
    # The srun stub only forwards to the worker script when it is an argument;
    # force a failing step instead.
    (tmp_path / "bin/srun").write_text("#!/bin/bash\nexit 9\n")

    result = _run_launcher(source_root, run_root, env, ["echo", "x"])

    assert result.returncode == 9
    assert not (state / "continuation.required").exists()


def test_launcher_marker_carries_attempt_id_and_clears_stale_receipts(
    tmp_path: Path,
) -> None:
    source_root, run_root, env = _launcher_fixture(tmp_path, "#!/bin/bash\nexit 0\n")
    markers = run_root / ".state/workers/ablation_test"
    markers.mkdir(parents=True)
    (markers / "000.done").write_text("outcome=complete\n")
    (markers / "000.partial").write_text("outcome=continuation\n")
    # A fresh continuation receipt is published by the worker in this run.
    (source_root / "scripts/workflow/run_parallel_worker.sh").write_text(
        f"#!/bin/bash\necho outcome=continuation >{markers}/000.partial\n"
    )

    result = _run_launcher(
        source_root, run_root, env, ["python", "-m", "scripts.hpo.ablation_worker"]
    )

    assert result.returncode == 75, result.stderr
    assert not (markers / "000.done").exists()
    marker = (run_root / ".state/continuation.required").read_text()
    assert "attempt_id\tattempt-new\n" in marker


def _coordinator_run(
    tmp_path: Path, marker_text: str | None, workflow_exit: int
) -> tuple[subprocess.CompletedProcess[str], Path]:
    run_root = tmp_path / "run"
    attempt_root = run_root / ".state/executions/attempt-new"
    attempt_root.mkdir(parents=True)
    (run_root / "source").mkdir()
    (run_root / "config.yaml").write_text("{}\n")
    if marker_text is not None:
        (run_root / ".state/continuation.required").write_text(marker_text)
    sif = tmp_path / "runtime.sif"
    sif.touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    singularity = fake_bin / "singularity"
    singularity.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = buildcfg ]; then printf "SESSIONDIR=%s\\n" '
        '"${TMPDIR:-/tmp}/singularity/mnt/session"; exit 0; fi\n'
        f"exit {workflow_exit}\n"
    )
    singularity.chmod(0o755)
    result = subprocess.run(
        [
            str(WORKFLOW / "snakemake_coordinator.sh"),
            str(run_root),
            str(sif),
            "1",
            "20",
            "batch",
            str(attempt_root),
        ],
        env=os.environ
        | {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "5",
            "SLURM_STEP_ID": "batch",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "20",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, attempt_root


def test_coordinator_ignores_continuation_marker_of_another_attempt(
    tmp_path: Path,
) -> None:
    result, attempt_root = _coordinator_run(
        tmp_path, "schema_version\t1\nattempt_id\told-attempt\n", 1
    )

    assert result.returncode == 1
    assert "exit_code\t1\n" in (attempt_root / "completed.tsv").read_text()


def test_coordinator_records_distinct_closeout_code_for_current_attempt(
    tmp_path: Path,
) -> None:
    result, attempt_root = _coordinator_run(
        tmp_path, "schema_version\t1\nattempt_id\tattempt-new\n", 1
    )

    assert result.returncode == 0
    receipt = (attempt_root / "completed.tsv").read_text()
    assert "exit_code\t75\n" in receipt


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [("75", "coordinator=closeout exit_code=75"), ("3", "coordinator=terminal")],
)
def test_status_labels_closeout_receipt(
    tmp_path: Path, exit_code: str, expected: str
) -> None:
    from tests.test_workflow_support import _make_remote_run_staging

    staging, run_root, env = _make_remote_run_staging(tmp_path, token="closeout")
    subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=True,
    )
    attempt_root = run_root / ".state/executions/attempt-1"
    attempt_root.mkdir(parents=True)
    (attempt_root / "binding.tsv").write_text(
        "mode\texisting-allocation\nallocation_job_id\t7\n"
    )
    (attempt_root / "completed.tsv").write_text(f"exit_code\t{exit_code}\n")
    (run_root / ".state/active_execution").write_text("attempt-1\n")

    result = subprocess.run(
        [str(run_root / "source/scripts/workflow/remote_run.sh"), "status"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected in result.stdout
    assert "workflow=incomplete" in result.stdout


def test_attach_runs_coordinator_detached_and_records_its_pid(tmp_path: Path) -> None:
    from tests.test_workflow_support import _make_remote_run_staging

    staging, run_root, env = _make_remote_run_staging(tmp_path, token="detach")
    subprocess.run(
        [str(staging / "source/scripts/workflow/remote_run.sh"), "publish"],
        env=env,
        check=True,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "sbatch").write_text("#!/bin/bash\nexit 99\n")
    (fake_bin / "scontrol").write_text(
        "#!/bin/bash\n"
        'printf "JobId=98765 JobState=RUNNING UserId=%s(1) Account=bsc08 '
        "QOS=acc_interactive Partition=accinteractive NumNodes=1 NumTasks=4 "
        "NumCPUs=80 AllocTRES=cpu=80,gres/gpu=4 "
        'EndTime=2099-01-01T00:00:00\\n" "$USER"\n'
    )
    # The probe step has --cpus-per-task=1 too; only the coordinator step has
    # --overlap, and it records its own PID and whether stdin is a terminal.
    (fake_bin / "srun").write_text(
        "#!/bin/bash\n"
        'case " $* " in *" --overlap "*)\n'
        '  printf "%s\\n" "$$" >"$SRUN_PID"\n'
        '  if [ -t 0 ]; then echo tty >"$SRUN_PID.stdin"; else echo none '
        '>"$SRUN_PID.stdin"; fi;;\n'
        "esac\nexit 0\n"
    )
    (fake_bin / "date").write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-d" ]; then printf "4070908800\\n"; '
        'else exec /bin/date "$@"; fi\n'
    )
    for name in ("sbatch", "scontrol", "srun", "date"):
        (fake_bin / name).chmod(0o755)
    env = env | {
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SRUN_PID": str(tmp_path / "srun.pid"),
    }

    result = subprocess.run(
        [str(run_root / "source/scripts/workflow/remote_run.sh"), "attach", "98765"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    attempt_id = result.stdout.strip().split(":", 2)[2]
    attempt_root = run_root / ".state/executions" / attempt_id
    assert (attempt_root / "launcher.pid").read_text() == (
        tmp_path / "srun.pid"
    ).read_text()
    assert (tmp_path / "srun.pid.stdin").read_text().strip() == "none"
