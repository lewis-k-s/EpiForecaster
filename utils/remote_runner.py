"""Remote runner for submitting experiments to MN5 cluster via SSH.

This module provides utilities for:
1. Syncing code to remote cluster
2. Generating and submitting sbatch scripts dynamically
3. Polling job status
4. Syncing results back

The dynamic script generation approach allows flexible configuration without
needing static sbatch files on the remote host.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class JobStatus(Enum):
    """SLURM job status states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class SlurmJobSpec:
    """Specification for a SLURM batch job.

    This is a simple dataclass that mirrors common SBATCH directives.
    """

    job_name: str
    account: str = "bsc08"
    qos: str = "acc_bscls"
    partition: str | None = None  # e.g., "gp_bscls" for CPU-only jobs
    gres: str = "gpu:1"
    cpus_per_task: int = 20
    time: str = "04:00:00"
    output: str = "slurm-%x-%j.out"
    array: str | None = None  # e.g., "0-3" for array jobs

    def to_sbatch_script(self, commands: str) -> str:
        """Generate a complete sbatch script.

        Args:
            commands: The shell commands to execute (after setup)

        Returns:
            Complete sbatch script content
        """
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --qos={self.qos}",
        ]

        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")

        lines.extend(
            [
                f"#SBATCH --gres={self.gres}",
                f"#SBATCH --cpus-per-task={self.cpus_per_task}",
                f"#SBATCH --time={self.time}",
                f"#SBATCH --output={self.output}",
            ]
        )

        if self.array:
            lines.append(f"#SBATCH --array={self.array}")
            # Update output pattern for array jobs
            if "%a" not in self.output:
                lines[-2] = "#SBATCH --output=slurm-%x-%A_%a.out"

        lines.extend(
            [
                "",
                "# Fail fast on errors; print commands for easier debugging.",
                "set -euo pipefail",
                "",
                commands,
            ]
        )

        return "\n".join(lines)


@dataclass
class ExperimentResult:
    """Result of a remote experiment execution."""

    job_id: str
    experiment_name: str
    status: JobStatus
    local_output_path: Path | None = None
    error_message: str | None = None


def sync_to_remote(
    experiment_name: str,
    remote_host: str = "dt",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster",
) -> bool:
    """Sync code to remote cluster using syncto script.

    Args:
        experiment_name: Name of the experiment (for logging)
        remote_host: SSH host alias (default: dt for BSC transfer node)
        remote_path: Remote destination path

    Returns:
        True if sync succeeded, False otherwise
    """
    repo_root = Path(__file__).parent.parent
    sync_script = repo_root / "syncto_mn5.sh"

    if not sync_script.exists():
        raise FileNotFoundError(f"Sync script not found: {sync_script}")

    try:
        subprocess.run(
            [str(sync_script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Synced to {remote_host}:{remote_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Sync failed: {e}")
        print(f"stderr: {e.stderr}")
        return False


def generate_training_script(
    config_path: str,
    overrides: Sequence[str] | None = None,
    venv_path: str = "$PROJECT_ROOT/.venv",
    cuda_module: str = "EB/apps EB/install CUDA/12.1.1",
) -> str:
    """Generate a training job script.

    Args:
        config_path: Path to training config YAML (on remote)
        overrides: Optional config overrides (e.g., ["training.batch_size=24"])
        venv_path: Path to virtualenv on remote
        cuda_module: CUDA modules to load

    Returns:
        Shell commands for the training job
    """
    override_str = ""
    if overrides:
        override_args = " ".join([f"--override {o}" for o in overrides])
        override_str = f" {override_args}"

    commands = f'''# Setup environment
export PROJECT_ROOT="${{PROJECT_ROOT:-$PWD}}"
cd "$PROJECT_ROOT"

# Source CUDA environment (bypasses module system issues)
if [ -f "$PROJECT_ROOT/env/mn5_cuda.env" ]; then
    source "$PROJECT_ROOT/env/mn5_cuda.env"
else
    # Fallback: try module load
    module load {cuda_module} 2>/dev/null || echo "Warning: CUDA setup may be incomplete"
fi

# Activate virtualenv if present
if [ -d "{venv_path}" ]; then
    source "{venv_path}/bin/activate"
fi

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Run training
if command -v uv >/dev/null 2>&1; then
    uv run main train epiforecaster --config "{config_path}"{override_str}
else
    python -m cli train epiforecaster --config "{config_path}"{override_str}
fi'''

    return commands


def generate_preprocessing_script(
    config_path: str,
    venv_path: str = "$PROJECT_ROOT/.venv",
) -> str:
    """Generate a preprocessing job script.

    Args:
        config_path: Path to preprocessing config YAML (on remote)
        venv_path: Path to virtualenv on remote

    Returns:
        Shell commands for the preprocessing job
    """
    commands = f'''# Setup environment
export PROJECT_ROOT="${{PROJECT_ROOT:-$PWD}}"
cd "$PROJECT_ROOT"

# Activate virtualenv if present
if [ -d "{venv_path}" ]; then
    source "{venv_path}/bin/activate"
fi

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Run preprocessing
if command -v uv >/dev/null 2>&1; then
    uv run main preprocess epiforecaster --config "{config_path}"
else
    python -m cli preprocess epiforecaster --config "{config_path}"
fi'''

    return commands


def generate_optuna_script(
    study_name: str,
    config_path: str,
    epochs: int = 2,
    timeout_secs: int = 14200,
    venv_path: str = "$PROJECT_ROOT/.venv",
    cuda_module: str = "EB/apps EB/install CUDA/12.1.1",
) -> str:
    """Generate an Optuna HPO worker script.

    Args:
        study_name: Name of the Optuna study
        config_path: Path to training config YAML (on remote)
        epochs: Number of epochs per trial
        timeout_secs: Worker timeout in seconds
        venv_path: Path to virtualenv on remote
        cuda_module: CUDA modules to load

    Returns:
        Shell commands for the Optuna worker
    """
    commands = f'''# Setup environment
# Source CUDA environment (bypasses module system issues)
if [ -f "$PROJECT_ROOT/env/mn5_cuda.env" ]; then
    source "$PROJECT_ROOT/env/mn5_cuda.env"
else
    # Fallback: try module load
    module load {cuda_module} 2>/dev/null || echo "Warning: CUDA setup may be incomplete"
fi

export PROJECT_ROOT="${{PROJECT_ROOT:-$PWD}}"
cd "$PROJECT_ROOT"

# Activate virtualenv if present
if [ -d "{venv_path}" ]; then
    source "{venv_path}/bin/activate"
fi

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Optuna paths
JOURNAL_FILE="$PROJECT_ROOT/outputs/optuna/{study_name}.journal"
RUN_ROOT="$PROJECT_ROOT/outputs/optuna"

# Run Optuna worker
ARGS=(
    --config "{config_path}"
    --study-name "{study_name}"
    --journal-file "$JOURNAL_FILE"
    --run-root "$RUN_ROOT"
    --epochs {epochs}
    --timeout-s {timeout_secs}
)

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/optuna_epiforecaster_worker.py "${{ARGS[@]}}"
else
    python scripts/optuna_epiforecaster_worker.py "${{ARGS[@]}}"
fi'''

    return commands


def submit_job_with_script(
    job_spec: SlurmJobSpec,
    commands: str,
    remote_host: str = "mn5",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster",
) -> str | None:
    """Submit a job by generating and piping a script over SSH.

    This generates the sbatch script locally and pipes it to the remote
    host via SSH, avoiding the need for static script files on MN5.

    Args:
        job_spec: SLURM job specification (resources, name, etc.)
        commands: Shell commands to execute in the job
        remote_host: SSH host alias
        remote_path: Remote working directory

    Returns:
        Job ID if submission succeeded, None otherwise

    Example:
        spec = SlurmJobSpec(
            job_name="my_training",
            time="02:00:00",
            gres="gpu:1",
        )
        commands = generate_training_script(
            "configs/train.yaml",
            overrides=["training.batch_size=32"],
        )
        job_id = submit_job_with_script(spec, commands)
    """
    script_content = job_spec.to_sbatch_script(commands)

    # Create a temp file path on remote
    import uuid

    temp_script = f"/tmp/sbatch_{uuid.uuid4().hex[:8]}.sh"

    try:
        # Step 1: Copy script content to remote temp file
        subprocess.run(
            ["ssh", remote_host, f"cat > {temp_script}"],
            input=script_content,
            capture_output=True,
            text=True,
            check=True,
        )

        # Step 2: Submit the script and clean up
        submit_cmd = f"cd {remote_path} && sbatch {temp_script} && rm {temp_script}"
        result = subprocess.run(
            ["ssh", remote_host, submit_cmd],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse job ID from sbatch output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Submitted job {job_id} for '{job_spec.job_name}'")
            return job_id
        else:
            print(f"Could not parse job ID from output: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Job submission failed: {e}")
        print(f"stderr: {e.stderr}")
        # Try to clean up temp file on failure
        try:
            subprocess.run(
                ["ssh", remote_host, f"rm -f {temp_script}"],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass
        return None


def submit_job(
    experiment_name: str,
    script_path: str = "scripts/train_single_gpu.sbatch",
    remote_host: str = "mn5",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster",
    slurm_args: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str | None:
    """Submit a job using existing static sbatch script (legacy method).

    Works with existing sbatch scripts that use environment variables for
    configuration. See scripts/train_single_gpu.sbatch for example.

    Args:
        experiment_name: Name of the experiment (for logging/tracking)
        script_path: Path to the batch script on remote (relative to remote_path)
        remote_host: SSH host alias
        remote_path: Remote working directory
        slurm_args: Additional SLURM arguments
        env_vars: Environment variables to export before sbatch

    Returns:
        Job ID if submission succeeded, None otherwise
    """
    cmd_parts = [f"cd {remote_path}"]

    if env_vars:
        for key, value in env_vars.items():
            cmd_parts.append(f'export {key}="{value}"')

    if slurm_args:
        sbatch_cmd = f"sbatch {' '.join(slurm_args)} {script_path}"
    else:
        sbatch_cmd = f"sbatch {script_path}"

    cmd_parts.append(sbatch_cmd)
    remote_cmd = " && ".join(cmd_parts)

    try:
        result = subprocess.run(
            ["ssh", remote_host, remote_cmd],
            capture_output=True,
            text=True,
            check=True,
        )

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Submitted job {job_id} for experiment '{experiment_name}'")
            return job_id
        else:
            print(f"Could not parse job ID from output: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Job submission failed: {e}")
        print(f"stderr: {e.stderr}")
        return None


def get_job_status(job_id: str, remote_host: str = "mn5") -> JobStatus:
    """Query SLURM for job status.

    Args:
        job_id: SLURM job ID
        remote_host: SSH host alias

    Returns:
        Current job status
    """
    try:
        result = subprocess.run(
            ["ssh", remote_host, f"scontrol show job {job_id}"],
            capture_output=True,
            text=True,
            check=True,
        )

        match = re.search(r"JobState=(\w+)", result.stdout)
        if match:
            state = match.group(1)
            status_map = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED,
                "TIMEOUT": JobStatus.TIMEOUT,
            }
            return status_map.get(state, JobStatus.UNKNOWN)

        return _get_job_status_from_sacct(job_id, remote_host)

    except subprocess.CalledProcessError:
        return _get_job_status_from_sacct(job_id, remote_host)


def _get_job_status_from_sacct(job_id: str, remote_host: str) -> JobStatus:
    """Fallback to sacct for job status (for completed jobs)."""
    try:
        result = subprocess.run(
            [
                "ssh",
                remote_host,
                f"sacct -j {job_id} --format=State --noheader --parsable2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        states = [s.strip() for s in result.stdout.strip().split("|") if s.strip()]
        if states:
            state = states[0]
            status_map = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED,
                "TIMEOUT": JobStatus.TIMEOUT,
            }
            return status_map.get(state, JobStatus.UNKNOWN)

        return JobStatus.UNKNOWN

    except subprocess.CalledProcessError:
        return JobStatus.UNKNOWN


def wait_until_running(
    job_id: str,
    remote_host: str = "mn5",
    extra_secs: int = 30,
    poll_interval_secs: int = 10,
) -> tuple[JobStatus, str]:
    """Poll until job is RUNNING, then wait extra_secs and return logs.

    This handles queue delays gracefully - waits until the job actually starts,
    gets initial logs, then returns. Unlike poll_job(), this does not wait for
    job completion.

    Args:
        job_id: SLURM job ID to poll
        remote_host: SSH host alias
        extra_secs: Additional seconds to wait after RUNNING state
        poll_interval_secs: Seconds between polls

    Returns:
        Tuple of (final_status, logs) where logs is a string containing
        initial log output after the job starts running
    """
    start_time = time.time()
    last_status = JobStatus.UNKNOWN

    while True:
        status = get_job_status(job_id, remote_host)
        elapsed = time.time() - start_time

        if status != last_status:
            print(f"Job {job_id} status: {status.value} (elapsed: {elapsed:.0f}s)")
            last_status = status

        if status == JobStatus.RUNNING:
            # Job is running, wait extra_secs for initial logs
            print(f"Job {job_id} is RUNNING, waiting {extra_secs}s for initial logs...")
            time.sleep(extra_secs)

            # Fetch initial logs
            logs = _fetch_job_logs(job_id, remote_host, tail_lines=50)
            return (JobStatus.RUNNING, logs)

        if status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        ):
            # Job finished before we could see it running
            logs = _fetch_job_logs(job_id, remote_host, tail_lines=50)
            return (status, logs)

        if status == JobStatus.UNKNOWN:
            # Job may have disappeared from queue
            logs = _fetch_job_logs(job_id, remote_host, tail_lines=50)
            return (JobStatus.UNKNOWN, logs)

        time.sleep(poll_interval_secs)


def _fetch_job_logs(job_id: str, remote_host: str, tail_lines: int = 50) -> str:
    """Fetch job logs from the default output location.

    Args:
        job_id: SLURM job ID
        remote_host: SSH host alias
        tail_lines: Number of lines to fetch

    Returns:
        Log output as string, or empty string if logs not found
    """
    # Try common SLURM output locations
    log_patterns = [
        f"slurm-{job_id}.out",
        f"slurm-{job_id}.out",  # Try in PWD
    ]

    for pattern in log_patterns:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    remote_host,
                    f"tail -n {tail_lines} {pattern} 2>/dev/null || echo 'LOG_NOT_FOUND'",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if "LOG_NOT_FOUND" not in result.stdout:
                return result.stdout
        except Exception:
            pass

    return ""


def poll_job(
    job_id: str,
    remote_host: str = "mn5",
    poll_interval_secs: int = 30,
    timeout_secs: int = 86400,
    callback: callable | None = None,
) -> JobStatus:
    """Poll a job until completion or timeout.

    Args:
        job_id: SLURM job ID to poll
        remote_host: SSH host alias
        poll_interval_secs: Seconds between polls
        timeout_secs: Maximum time to wait (0 for no timeout)
        callback: Optional callback(status, elapsed_secs) for progress updates

    Returns:
        Final job status
    """
    start_time = time.time()
    last_status = JobStatus.UNKNOWN

    while True:
        status = get_job_status(job_id, remote_host)
        elapsed = time.time() - start_time

        if status != last_status:
            print(f"Job {job_id} status: {status.value} (elapsed: {elapsed:.0f}s)")
            last_status = status

        if callback:
            callback(status, elapsed)

        if status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        ):
            return status

        if timeout_secs > 0 and elapsed > timeout_secs:
            print(f"Polling timeout reached after {elapsed:.0f}s")
            return JobStatus.UNKNOWN

        time.sleep(poll_interval_secs)


def sync_from_remote(
    experiment_name: str,
    remote_host: str = "dt",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster/outputs/training",
    local_output_dir: str = "./outputs/training",
) -> Path | None:
    """Sync results from remote cluster.

    Args:
        experiment_name: Name of the experiment to sync
        remote_host: SSH host alias
        remote_path: Remote path to outputs
        local_output_dir: Local directory for outputs

    Returns:
        Path to local output directory if successful, None otherwise
    """
    repo_root = Path(__file__).parent.parent
    sync_script = repo_root / "syncback_from_mn5.sh"

    if not sync_script.exists():
        raise FileNotFoundError(f"Sync script not found: {sync_script}")

    try:
        subprocess.run(
            [str(sync_script), experiment_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        local_path = repo_root / local_output_dir / experiment_name
        if local_path.exists():
            print(f"Synced results to {local_path}")
            return local_path
        else:
            print(f"Sync completed but output not found at {local_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Sync from remote failed: {e}")
        print(f"stderr: {e.stderr}")
        return None


def run_experiment(
    experiment_name: str,
    script_path: str = "scripts/train_single_gpu.sbatch",
    remote_hosts: dict[str, str] | None = None,
    slurm_args: Sequence[str] | None = None,
    env_vars: dict[str, str] | None = None,
    poll_interval_secs: int = 30,
    timeout_secs: int = 86400,
    skip_sync_to: bool = False,
    skip_sync_from: bool = False,
) -> ExperimentResult:
    """Run a full experiment using static sbatch script: sync, submit, poll, sync back.

    Args:
        experiment_name: Name of the experiment
        script_path: Path to batch script on remote
        remote_hosts: Dict with 'transfer' and 'login' keys for host aliases
        slurm_args: Additional SLURM arguments
        env_vars: Environment variables to export before sbatch
        poll_interval_secs: Seconds between status polls
        timeout_secs: Maximum wait time in seconds
        skip_sync_to: Skip syncing code to remote
        skip_sync_from: Skip syncing results back

    Returns:
        ExperimentResult with job info and status
    """
    hosts = remote_hosts or {
        "transfer": "dt",
        "login": "mn5",
    }

    if not skip_sync_to:
        print("Syncing code to remote...")
        if not sync_to_remote(experiment_name, hosts["transfer"]):
            return ExperimentResult(
                job_id="",
                experiment_name=experiment_name,
                status=JobStatus.FAILED,
                error_message="Failed to sync to remote",
            )

    print("Submitting job...")
    job_id = submit_job(
        experiment_name=experiment_name,
        script_path=script_path,
        remote_host=hosts["login"],
        slurm_args=slurm_args,
        env_vars=env_vars,
    )

    if job_id is None:
        return ExperimentResult(
            job_id="",
            experiment_name=experiment_name,
            status=JobStatus.FAILED,
            error_message="Failed to submit job",
        )

    print(f"Polling job {job_id}...")
    final_status = poll_job(
        job_id=job_id,
        remote_host=hosts["login"],
        poll_interval_secs=poll_interval_secs,
        timeout_secs=timeout_secs,
    )

    local_path = None
    if not skip_sync_from and final_status == JobStatus.COMPLETED:
        print("Syncing results back...")
        local_path = sync_from_remote(experiment_name, hosts["transfer"])

    return ExperimentResult(
        job_id=job_id,
        experiment_name=experiment_name,
        status=final_status,
        local_output_path=local_path,
    )


def run_experiment_dynamic(
    experiment_name: str,
    job_spec: SlurmJobSpec,
    commands: str,
    remote_hosts: dict[str, str] | None = None,
    poll_interval_secs: int = 30,
    timeout_secs: int = 86400,
    skip_sync_to: bool = False,
    skip_sync_from: bool = False,
) -> ExperimentResult:
    """Run a full experiment with dynamically generated script: sync, submit, poll, sync back.

    This uses the dynamic script generation approach - no static sbatch files needed.

    Args:
        experiment_name: Name of the experiment
        job_spec: SLURM job resource specification
        commands: Shell commands to execute (use generate_training_script() or generate_optuna_script())
        remote_hosts: Dict with 'transfer' and 'login' keys for host aliases
        poll_interval_secs: Seconds between status polls
        timeout_secs: Maximum wait time in seconds
        skip_sync_to: Skip syncing code to remote
        skip_sync_from: Skip syncing results back

    Returns:
        ExperimentResult with job info and status

    Example:
        from utils.remote_runner import (
            run_experiment_dynamic, SlurmJobSpec, generate_training_script
        )

        # Run training with dynamic script
        job_spec = SlurmJobSpec(
            job_name="dynamic_test",
            time="02:00:00",
            gres="gpu:1",
        )
        commands = generate_training_script(
            "configs/train.yaml",
            overrides=["training.batch_size=48", "training.epochs=10"],
        )

        result = run_experiment_dynamic(
            experiment_name="dynamic_test",
            job_spec=job_spec,
            commands=commands,
            timeout_secs=7200,
        )
    """
    hosts = remote_hosts or {
        "transfer": "dt",
        "login": "mn5",
    }

    if not skip_sync_to:
        print("Syncing code to remote...")
        if not sync_to_remote(experiment_name, hosts["transfer"]):
            return ExperimentResult(
                job_id="",
                experiment_name=experiment_name,
                status=JobStatus.FAILED,
                error_message="Failed to sync to remote",
            )

    print("Submitting job with dynamic script...")
    job_id = submit_job_with_script(
        job_spec=job_spec,
        commands=commands,
        remote_host=hosts["login"],
    )

    if job_id is None:
        return ExperimentResult(
            job_id="",
            experiment_name=experiment_name,
            status=JobStatus.FAILED,
            error_message="Failed to submit job",
        )

    print(f"Polling job {job_id}...")
    final_status = poll_job(
        job_id=job_id,
        remote_host=hosts["login"],
        poll_interval_secs=poll_interval_secs,
        timeout_secs=timeout_secs,
    )

    local_path = None
    if not skip_sync_from and final_status == JobStatus.COMPLETED:
        print("Syncing results back...")
        local_path = sync_from_remote(experiment_name, hosts["transfer"])

    return ExperimentResult(
        job_id=job_id,
        experiment_name=experiment_name,
        status=final_status,
        local_output_path=local_path,
    )


def run_training(
    experiment_name: str,
    config_path: str,
    overrides: Sequence[str] | None = None,
    time: str = "04:00:00",
    gres: str = "gpu:1",
    dynamic: bool = True,
    **kwargs,
) -> ExperimentResult:
    """Convenience function for running training jobs.

    By default uses dynamic script generation. Set dynamic=False to use
    the static scripts/train_single_gpu.sbatch file.

    Args:
        experiment_name: Name for tracking
        config_path: Path to training config YAML on remote
        overrides: Optional list of config overrides (e.g., ["training.batch_size=24"])
        time: SLURM time limit (default "04:00:00")
        gres: SLURM gres specification (default "gpu:1")
        dynamic: Use dynamic script generation (default True)
        **kwargs: Passed to run_experiment() or run_experiment_dynamic()

    Returns:
        ExperimentResult

    Example:
        # Dynamic generation with overrides (recommended)
        result = run_training(
            experiment_name="batch_test",
            config_path="configs/train.yaml",
            overrides=["training.batch_size=48", "training.epochs=5"],
            time="02:00:00",
        )

        # Static script (legacy)
        result = run_training(
            experiment_name="static_test",
            config_path="configs/train.yaml",
            dynamic=False,
            env_vars={"OVERRIDES": "training.batch_size=24"},
        )
    """
    if dynamic:
        job_spec = SlurmJobSpec(
            job_name=experiment_name,
            time=time,
            gres=gres,
        )
        commands = generate_training_script(config_path, overrides)
        return run_experiment_dynamic(
            experiment_name=experiment_name,
            job_spec=job_spec,
            commands=commands,
            **kwargs,
        )
    else:
        # Legacy static script approach
        env_vars: dict[str, str] = {"CONFIG": config_path}
        if overrides:
            env_vars["OVERRIDES"] = " ".join(overrides)
        return run_experiment(
            experiment_name=experiment_name,
            script_path="scripts/train_single_gpu.sbatch",
            env_vars=env_vars,
            **kwargs,
        )


def run_preprocessing(
    experiment_name: str,
    config_path: str,
    time: str = "01:00:00",
    cpus_per_task: int = 1,
    dynamic: bool = True,
    **kwargs,
) -> ExperimentResult:
    """Convenience function for running preprocessing jobs.

    Preprocessing is CPU-only and benefits from more cores for parallel
    data processing. Results are saved to data/processed/ on the remote
    and can be synced back with sync_from_remote() if needed.

    Args:
        experiment_name: Name for tracking the preprocessing job
        config_path: Path to preprocessing config YAML on remote
        time: SLURM time limit (default "04:00:00")
        cpus_per_task: Number of CPUs for parallel processing (default 40)
        dynamic: Use dynamic script generation (default True)
        **kwargs: Passed to run_experiment_dynamic()

    Returns:
        ExperimentResult

    Example:
        # Run preprocessing with default settings
        result = run_preprocessing(
            experiment_name="synth_preprocess_v2",
            config_path="configs/preprocess_mn5_synth.yaml",
            time="02:00:00",
        )

        This command is mostly single-threaded so does not benefit from more CPU resources
    """
    if dynamic:
        job_spec = SlurmJobSpec(
            job_name=experiment_name,
            time=time,
            qos="gp_bscls",  # GPP queue for CPU-only jobs
            partition="gp_bscls",  # GPP partition for CPU-only jobs
            gres="gpu:0",  # CPU-only, no GPU needed
            cpus_per_task=cpus_per_task,
        )
        commands = generate_preprocessing_script(config_path)
        return run_experiment_dynamic(
            experiment_name=experiment_name,
            job_spec=job_spec,
            commands=commands,
            skip_sync_from=True,  # Preprocessing outputs to data/processed/, not outputs/training/
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Static script approach not supported for preprocessing"
        )


def run_optuna_study(
    study_name: str,
    config_path: str,
    epochs: int = 2,
    array_size: int = 4,
    time: str = "04:00:00",
    dynamic: bool = True,
    **kwargs,
) -> ExperimentResult:
    """Convenience function for running Optuna HPO jobs.

    By default uses dynamic script generation with proper array support.

    Args:
        study_name: Name for the Optuna study
        config_path: Path to training config YAML on remote
        epochs: Number of epochs per trial (default 2)
        array_size: Number of parallel array tasks (default 4)
        time: SLURM time limit (default "04:00:00")
        dynamic: Use dynamic script generation (default True)
        **kwargs: Passed to run_experiment() or run_experiment_dynamic()

    Returns:
        ExperimentResult

    Example:
        # Dynamic array job
        result = run_optuna_study(
            study_name="hpo_v3",
            config_path="configs/train.yaml",
            epochs=3,
            array_size=8,
            time="04:00:00",
        )

        # Static script (legacy)
        result = run_optuna_study(
            study_name="hpo_v2",
            config_path="configs/train.yaml",
            dynamic=False,
            slurm_args=["--array=0-7"],
        )
    """
    if dynamic:
        job_spec = SlurmJobSpec(
            job_name=study_name,
            time=time,
            gres="gpu:1",
            array=f"0-{array_size - 1}",
        )
        commands = generate_optuna_script(study_name, config_path, epochs)
        return run_experiment_dynamic(
            experiment_name=study_name,
            job_spec=job_spec,
            commands=commands,
            **kwargs,
        )
    else:
        # Legacy static script approach
        return run_experiment(
            experiment_name=study_name,
            script_path="scripts/optuna_epiforecaster.sbatch",
            env_vars={
                "STUDY_NAME": study_name,
                "CONFIG": config_path,
                "EPOCHS": str(epochs),
            },
            slurm_args=[f"--array=0-{array_size - 1}"],
            **kwargs,
        )


def list_remote_experiments(
    remote_host: str = "dt",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster/outputs/training",
) -> list[dict]:
    """List experiments on remote cluster.

    Args:
        remote_host: SSH host alias for transfer node
        remote_path: Remote path to training outputs

    Returns:
        List of experiment info dicts with name, size, modified time
    """
    try:
        result = subprocess.run(
            ["ssh", remote_host, f"ls -la {remote_path}"],
            capture_output=True,
            text=True,
            check=True,
        )

        experiments = []
        for line in result.stdout.strip().split("\n")[2:]:
            parts = line.split()
            if len(parts) >= 9:
                name = parts[-1]
                if name not in (".", ".."):
                    experiments.append(
                        {
                            "name": name,
                            "size": parts[4],
                            "modified": " ".join(parts[5:8]),
                        }
                    )

        return experiments

    except subprocess.CalledProcessError as e:
        print(f"Failed to list remote experiments: {e}")
        return []


def cancel_job(job_id: str, remote_host: str = "mn5") -> bool:
    """Cancel a running SLURM job.

    Args:
        job_id: SLURM job ID
        remote_host: SSH host alias

    Returns:
        True if cancellation succeeded
    """
    try:
        subprocess.run(
            ["ssh", remote_host, f"scancel {job_id}"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Cancelled job {job_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to cancel job {job_id}: {e}")
        return False


def get_job_logs(
    job_id: str,
    experiment_name: str,
    remote_host: str = "mn5",
    remote_path: str = "/home/bsc/bsc008913/EpiForecaster/outputs/training",
    tail_lines: int = 50,
) -> dict[str, str]:
    """Get tail of job log files from remote.

    Args:
        job_id: SLURM job ID
        experiment_name: Name of the experiment
        remote_host: SSH host alias
        remote_path: Remote path to outputs
        tail_lines: Number of lines to tail

    Returns:
        Dict with 'stdout' and 'stderr' keys
    """
    exp_path = f"{remote_path}/{experiment_name}"

    log_patterns = [
        f"{exp_path}/slurm-{job_id}.out",
        f"{exp_path}/slurm-{experiment_name}-{job_id}.out",
        f"{exp_path}/logs/train.log",
    ]

    logs = {"stdout": "", "stderr": ""}

    for pattern in log_patterns[:2]:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    remote_host,
                    f"tail -n {tail_lines} {pattern} 2>/dev/null || echo 'LOG_NOT_FOUND'",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if "LOG_NOT_FOUND" not in result.stdout:
                logs["stdout"] = result.stdout
                break
        except Exception:
            pass

    return logs


if __name__ == "__main__":
    print("Remote runner module - import and use functions:")
    print("  from utils.remote_runner import run_training, poll_job, sync_from_remote")
    print()
    print("Examples:")
    print("  # Dynamic training with overrides:")
    print("  result = run_training(")
    print('      "my_experiment",')
    print('      "configs/train.yaml",')
    print('      overrides=["training.batch_size=48", "training.epochs=10"],')
    print('      time="02:00:00",')
    print("  )")
    print()
    print("  # Dynamic optuna HPO:")
    print("  result = run_optuna_study(")
    print('      study_name="hpo_v3",')
    print('      config_path="configs/train.yaml",')
    print("      epochs=3,")
    print("      array_size=8,")
    print("  )")
    print()
    print("Quick test: list remote experiments")
    exps = list_remote_experiments()
    for exp in exps[:5]:
        print(f"  - {exp['name']} ({exp['size']})")
