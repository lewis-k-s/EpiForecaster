"""Utilities for discovering and resolving runs in the outputs directory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT = Path("outputs")


class RunInfo(NamedTuple):
    """Information about a single run."""

    experiment_name: str
    run_id: str
    run_path: Path
    has_checkpoint: bool
    has_eval_metrics: bool


def discover_runs(
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    experiment_name: str | None = None,
) -> list[RunInfo]:
    """Scan outputs directory and return all discovered runs.

    Args:
        outputs_root: Root directory containing outputs (default: outputs/)
        experiment_name: Optional filter to specific experiment

    Returns:
        List of RunInfo objects, sorted by run_id (most recent last)
    """
    runs: list[RunInfo] = []

    training_root = outputs_root / "training"
    eval_root = outputs_root / "eval"

    # Determine which experiments to scan
    if experiment_name:
        experiment_dirs = [d for d in [training_root / experiment_name] if d.is_dir()]
    else:
        experiment_dirs = [d for d in training_root.iterdir() if d.is_dir()]

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

        for run_dir in run_dirs:
            run_id = run_dir.name

            # Check for checkpoint
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            has_checkpoint = checkpoint_path.is_file()

            # Check for eval metrics
            eval_dir = eval_root / exp_name / run_id
            val_metrics = eval_dir / "val_metrics.csv"
            has_eval_metrics = val_metrics.is_file()

            runs.append(
                RunInfo(
                    experiment_name=exp_name,
                    run_id=run_id,
                    run_path=run_dir,
                    has_checkpoint=has_checkpoint,
                    has_eval_metrics=has_eval_metrics,
                )
            )

    # Sort by run_id (timestamp-based, so string sort works)
    runs.sort(key=lambda r: r.run_id)
    return runs


def resolve_checkpoint_path(
    *,
    experiment_name: str,
    run_id: str,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    """Resolve checkpoint path from experiment/run.

    Args:
        experiment_name: Name of the experiment
        run_id: Run ID (e.g., 'run_1767364191170741000')
        outputs_root: Root outputs directory

    Returns:
        Path to the checkpoint file

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    checkpoint_path = (
        outputs_root / "training" / experiment_name / run_id / "checkpoints" / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{experiment_name}', run '{run_id}'"
        )
    return checkpoint_path


def resolve_eval_csv_path(
    *,
    experiment_name: str,
    run_id: str,
    split: str,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    """Resolve eval CSV path from experiment/run.

    Note: This resolves the AGGREGATE metrics CSV (val_metrics.csv), not the
    per-node metrics CSV. The per-node CSV is written by eval_checkpoint.

    Args:
        experiment_name: Name of the experiment
        run_id: Run ID (e.g., 'run_1767364191170741000')
        split: Which split ('val' or 'test')
        outputs_root: Root outputs directory

    Returns:
        Path to the eval metrics CSV file

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    csv_path = outputs_root / "eval" / experiment_name / run_id / f"{split}_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No {split} metrics CSV found for experiment '{experiment_name}', run '{run_id}'"
        )
    return csv_path


def resolve_node_metrics_csv_path(
    *,
    experiment_name: str,
    run_id: str,
    split: str,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    """Resolve per-node metrics CSV path from experiment/run.

    This resolves the CSV written by eval_checkpoint with columns
    node_id, mae, num_samples, which is used by plot_forecasts_from_csv.

    Args:
        experiment_name: Name of the experiment
        run_id: Run ID (e.g., 'run_1767364191170741000')
        split: Which split ('val' or 'test')
        outputs_root: Root outputs directory

    Returns:
        Path to the per-node metrics CSV file

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    csv_path = (
        outputs_root / "eval" / experiment_name / run_id / f"{split}_node_metrics.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No {split} node metrics CSV found for experiment '{experiment_name}', run '{run_id}'"
        )
    return csv_path


def get_eval_output_dir(
    *,
    experiment_name: str,
    run_id: str,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    """Get the eval output directory for a given experiment/run.

    Creates the directory if it doesn't exist. Used for auto-resolving
    output paths like --output-csv and --output.

    Args:
        experiment_name: Name of the experiment
        run_id: Run ID (e.g., 'run_1767364191170741000')
        outputs_root: Root outputs directory

    Returns:
        Path to eval output directory (e.g., outputs/eval/{experiment}/{run}/)
    """
    eval_dir = outputs_root / "eval" / experiment_name / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def list_available_runs(
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    experiment_name: str | None = None,
) -> str:
    """Return formatted string listing available runs for error messages.

    Args:
        outputs_root: Root outputs directory
        experiment_name: Optional filter to specific experiment

    Returns:
        Formatted string listing available runs
    """
    runs = discover_runs(outputs_root=outputs_root, experiment_name=experiment_name)

    if not runs:
        if experiment_name:
            return f"No runs found for experiment '{experiment_name}'"
        return "No runs found in outputs directory"

    lines: list[str] = []
    if experiment_name:
        lines.append(f"Available runs for experiment '{experiment_name}':")
    else:
        lines.append("Available runs:")

    for run in runs:
        flags = []
        if run.has_checkpoint:
            flags.append("checkpoint")
        if run.has_eval_metrics:
            flags.append("eval_metrics")
        flags_str = ", ".join(flags) if flags else "no artifacts"
        lines.append(f"  {run.run_id}  (has: {flags_str})")

    return "\n".join(lines)
