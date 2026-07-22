"""Deterministic worker identity helpers for parallel HPO processes."""

from __future__ import annotations

import os


def logical_worker_id() -> int:
    """Return the explicit logical rank assigned by the workflow launcher."""
    raw_worker_id = os.getenv("EPIFORECASTER_WORKER_ID")
    if not raw_worker_id:
        return 0

    try:
        worker_id = int(raw_worker_id)
    except ValueError as exc:
        raise ValueError(
            f"HPO worker identity must be an integer, got {raw_worker_id!r}"
        ) from exc
    if worker_id < 0:
        raise ValueError(f"HPO worker identity must be non-negative, got {worker_id}")
    return worker_id


def compute_worker_seed(base_seed: int) -> int:
    """Offset ``base_seed`` by the stable logical worker identity."""
    return base_seed + logical_worker_id()
