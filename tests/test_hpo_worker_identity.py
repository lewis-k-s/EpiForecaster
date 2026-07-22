from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from scripts.hpo.worker_identity import compute_worker_seed


def test_returns_base_seed_without_worker_identity() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert compute_worker_seed(42) == 42


def test_ignores_empty_slurm_array_task_id() -> None:
    with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": ""}, clear=True):
        assert compute_worker_seed(42) == 42


def test_prefers_stable_logical_worker_id() -> None:
    with patch.dict(
        os.environ,
        {"EPIFORECASTER_WORKER_ID": "3", "SLURM_ARRAY_TASK_ID": "9"},
        clear=True,
    ):
        assert compute_worker_seed(42) == 45


def test_falls_back_to_slurm_array_task_id() -> None:
    with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "4"}, clear=True):
        assert compute_worker_seed(42) == 46


def test_blank_logical_worker_id_falls_back_to_slurm() -> None:
    with patch.dict(
        os.environ,
        {"EPIFORECASTER_WORKER_ID": "", "SLURM_ARRAY_TASK_ID": "2"},
        clear=True,
    ):
        assert compute_worker_seed(42) == 44


@pytest.mark.parametrize("worker_id", ["invalid", "-1"])
def test_rejects_invalid_worker_identity(worker_id: str) -> None:
    with patch.dict(os.environ, {"EPIFORECASTER_WORKER_ID": worker_id}, clear=True):
        with pytest.raises(ValueError, match="HPO worker identity"):
            compute_worker_seed(42)
