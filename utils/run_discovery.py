"""Utilities for discovering and resolving runs in the outputs directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import click

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
        run_dirs = [
            d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
        ]

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


def resolve_trace_paths(
    *,
    experiment_name: str,
    run_id: str,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> list[Path]:
    """Resolve profiler trace paths from experiment/run.

    Args:
        experiment_name: Name of the experiment
        run_id: Run ID (e.g., 'run_1767364191170741000')
        outputs_root: Root outputs directory

    Returns:
        List of trace JSON paths.

    Raises:
        FileNotFoundError: If run or traces do not exist
    """
    run_dir = outputs_root / "training" / experiment_name / run_id
    if not run_dir.is_dir():
        available = list_available_runs(
            outputs_root=outputs_root, experiment_name=experiment_name
        )
        raise FileNotFoundError(f"Run directory not found: {run_dir}\n{available}")

    trace_paths = list(run_dir.glob("**/*.pt.trace.json"))
    if not trace_paths:
        trace_paths = list(run_dir.glob("**/*trace*.json"))

    if not trace_paths:
        raise FileNotFoundError(f"No trace JSON files found under {run_dir}")

    return sorted(trace_paths)


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
        outputs_root
        / "training"
        / experiment_name
        / run_id
        / "checkpoints"
        / "best_model.pt"
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


def extract_run_from_checkpoint_path(
    checkpoint_path: Path,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> tuple[str, str] | None:
    """Extract experiment name and run ID from a checkpoint path.

    Supports both patterns:
    - outputs_root/training/{experiment}/{run_id}/checkpoints/best_model.pt
    - outputs_root/optuna/{experiment}/{run_id}/checkpoints/best_model.pt

    Args:
        checkpoint_path: Path to checkpoint file
        outputs_root: Root outputs directory (default: outputs/)

    Returns:
        (experiment_name, run_id) tuple if path matches expected pattern,
        None if path doesn't match or run_id doesn't match expected patterns.
    """
    # Convert both to absolute paths for consistent matching
    # Use absolute() instead of resolve() to handle relative paths correctly
    # (resolve() follows symlinks which can cause path mismatches in tests)
    if checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path
    else:
        checkpoint_path = Path.cwd() / checkpoint_path

    if outputs_root.is_absolute():
        outputs_root = outputs_root
    else:
        outputs_root = Path.cwd() / outputs_root

    # Check if path contains outputs_root/training/ or outputs_root/optuna/
    checkpoint_str = str(checkpoint_path)
    outputs_str = str(outputs_root)

    # Pattern 1: outputs_root/training/{experiment}/{run_id}/checkpoints/best_model.pt
    # Pattern 2: outputs_root/optuna/{experiment}/{run_id}/checkpoints/best_model.pt
    training_pattern = (
        rf"{re.escape(outputs_str)}/(training|optuna)/([^/]+)/([^/]+)/checkpoints/"
    )
    match = re.search(training_pattern, checkpoint_str)

    if not match:
        return None

    run_id = match.group(3)
    experiment_name = match.group(2)

    # Validate run_id matches expected patterns:
    # - run_* (standard training runs)
    # - *trial*_* (optuna trials, e.g., local_trial29_1768952246597587000)
    # The pattern matches:
    #   1. run_* at the start
    #   2. Anything containing "trial" followed by at least one underscore-separated part
    #     (e.g., trial1_123, local_trial29_1768952246597587000)
    run_id_pattern = r"^(run_|.*trial.+_.+)"
    if not re.match(run_id_pattern, run_id):
        return None

    return experiment_name, run_id


def prompt_to_save_eval(experiment: str, run: str, default: bool = True) -> bool:
    """Prompt user whether to save eval results to outputs/eval/{experiment}/{run}/.

    Args:
        experiment: Experiment name
        run: Run ID
        default: Default value (True means Enter confirms Y)

    Returns:
        True if user wants to save, False otherwise (including on abort/interrupt)
    """
    eval_dir = DEFAULT_OUTPUTS_ROOT / "eval" / experiment / run
    prompt_text = (
        f"Save eval results to {eval_dir}/? [Y/n] "
        if default
        else f"Save eval results to {eval_dir}/? [y/N] "
    )
    try:
        return click.confirm(prompt_text, default=default)
    except click.exceptions.Abort:
        # User pressed Ctrl+C or similar - treat as "no"
        return False
