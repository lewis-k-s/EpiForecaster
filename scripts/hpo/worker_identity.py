"""Deterministic worker identity helpers for parallel HPO processes."""

from __future__ import annotations

import os


def compute_worker_seed(base_seed: int) -> int:
    """Offset ``base_seed`` by the logical worker or Slurm array task ID.

    Snakemake supplies ``EPIFORECASTER_WORKER_ID`` independently of Slurm's
    scheduling identity.  Prefer that stable logical ID so resubmissions keep
    the same sampler seed.  Direct array launches fall back to the Slurm task
    ID.  Empty values are treated as absent because the container wrapper
    deliberately exports unset Slurm variables as empty strings.
    """
    raw_worker_id = os.getenv("EPIFORECASTER_WORKER_ID") or os.getenv(
        "SLURM_ARRAY_TASK_ID"
    )
    if not raw_worker_id:
        return base_seed

    try:
        worker_id = int(raw_worker_id)
    except ValueError as exc:
        raise ValueError(
            f"HPO worker identity must be an integer, got {raw_worker_id!r}"
        ) from exc
    if worker_id < 0:
        raise ValueError(f"HPO worker identity must be non-negative, got {worker_id}")
    return base_seed + worker_id
