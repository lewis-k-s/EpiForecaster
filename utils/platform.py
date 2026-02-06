"""Platform-specific helpers for cluster detection and resource management."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)
MpContext = Literal["fork", "spawn"]


def is_slurm_cluster() -> bool:
    """Detect if running on a SLURM-managed cluster.

    Returns:
        True if SLURM_JOB_ID environment variable is set.
    """
    return bool(os.getenv("SLURM_JOB_ID"))


def is_interactive_slurm_session() -> bool:
    """Detect if running in an interactive SLURM session.

    Returns:
        True if SLURM session is interactive (not a batch job).
    """
    if not is_slurm_cluster():
        return False
    job_name = os.getenv("SLURM_JOB_NAME", "")
    job_qos = os.getenv("SLURM_JOB_QOS", "")
    return job_name == "interactive" or "_interactive" in job_qos


def get_nvme_path() -> Path:
    """Get path to node-local NVMe storage.

    Uses TMPDIR if set (standard on SLURM clusters), otherwise /tmp.
    Creates a job-specific subdirectory.

    Returns:
        Path to NVMe staging directory for this job.
    """
    base_path = Path(os.getenv("TMPDIR", "/tmp"))
    job_id = os.getenv("SLURM_JOB_ID", str(os.getpid()))
    nvme_path = base_path / f"epiforecaster_{job_id}"
    nvme_path.mkdir(parents=True, exist_ok=True)
    return nvme_path


def stage_dataset_to_nvme(
    source_path: Path,
    nvme_path: Path | None = None,
    enable_staging: bool = True,
) -> Path:
    """Stage dataset from NFS to node-local NVMe storage.

    Copies dataset directory to NVMe for improved I/O performance.
    Uses rsync if available, falls back to shutil.copytree.

    Args:
        source_path: Path to dataset on NFS/shared storage.
        nvme_path: Target directory on NVMe (defaults to job-specific path).
        enable_staging: If False, returns source_path unchanged.

    Returns:
        Path to staged dataset (on NVMe if staging enabled, else original path).
    """
    if not enable_staging:
        return source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    if nvme_path is None:
        nvme_path = get_nvme_path()

    # Already on NVMe or local storage?
    source_resolved = source_path.resolve()
    if str(source_resolved).startswith(str(nvme_path.resolve())):
        logger.debug(f"Dataset already on NVMe: {source_path}")
        return source_path

    dest_path = nvme_path / source_path.name

    if dest_path.exists():
        logger.info(f"Dataset already staged: {dest_path}")
        return dest_path

    logger.info(f"Staging dataset to NVMe: {source_path} -> {dest_path}")
    start_time = time.time()

    try:
        # Use rsync if available (preserves permissions, faster for large files)
        if shutil.which("rsync"):
            import subprocess

            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--info=progress2",
                    str(source_path) + "/",
                    str(dest_path) + "/",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # Fallback to shutil
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

        elapsed = time.time() - start_time
        logger.info(f"Staging complete: {elapsed:.1f}s -> {dest_path}")
        return dest_path

    except Exception as e:
        logger.warning(f"Staging failed: {e}. Using original path.")
        return source_path


def cleanup_nvme_staging(nvme_path: Path | None = None) -> None:
    """Clean up staged data from NVMe.

    Args:
        nvme_path: Path to clean up (defaults to job-specific path).
    """
    if nvme_path is None:
        nvme_path = get_nvme_path()

    if nvme_path.exists():
        logger.info(f"Cleaning up NVMe staging: {nvme_path}")
        shutil.rmtree(nvme_path, ignore_errors=True)


def select_multiprocessing_context(
    device: torch.device | str,
    *,
    all_num_workers_zero: bool,
) -> MpContext | None:
    """Select DataLoader multiprocessing context based on platform and device.

    Args:
        device: Torch device or device string.
        all_num_workers_zero: True if all loader worker counts are zero.

    Returns:
        Multiprocessing context name or None to use default.
    """
    if all_num_workers_zero:
        return None

    device_type = device.type if isinstance(device, torch.device) else str(device)
    is_linux = platform.system() == "Linux"

    if device_type == "cuda" and not is_linux:
        return "spawn"

    return "fork" if is_linux else None
