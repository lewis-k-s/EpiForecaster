"""Stable filesystem and Optuna attribute names for HPO trial artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


TRIAL_DIR_ATTR = "trial_dir"
BEST_MODEL_PATH_ATTR = "best_model_path"
BEST_MODEL_SHA256_ATTR = "best_model_sha256"
TRAINING_DYNAMICS_MANIFEST_ATTR = "training_dynamics_manifest"
TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR = "training_dynamics_manifest_sha256"
TRAINING_DYNAMICS_STATUS_ATTR = "training_dynamics_status"


def trial_model_id(trial_number: int) -> str:
    """Return the deterministic model directory name for an Optuna trial."""
    if trial_number < 0:
        raise ValueError(f"Optuna trial number must be non-negative: {trial_number}")
    return f"trial-{trial_number:06d}"


def trial_directory(
    trials_root: Path,
    study_name: str,
    trial_number: int,
) -> Path:
    """Return the absolute directory reserved for one trial's trainer output."""
    if (
        not study_name
        or study_name in {".", ".."}
        or Path(study_name).name != study_name
    ):
        raise ValueError(
            f"Optuna study name must be one path component: {study_name!r}"
        )
    return (trials_root / study_name / trial_model_id(trial_number)).resolve()


def best_model_path(trial_dir: Path) -> Path:
    """Return the canonical trainer checkpoint path for a trial."""
    return trial_dir / "checkpoints/best_model.pt"


def sha256_file(path: Path) -> str:
    """Hash a file without loading the checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
