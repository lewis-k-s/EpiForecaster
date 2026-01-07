"""Optuna worker for EpiForecaster hyperparameter search.

Designed for SLURM task arrays + Optuna JournalStorage coordination.

This script intentionally does NOT modify existing config dataclasses or CLI wiring.
It:
- Loads a base YAML config via `EpiForecasterConfig.from_file()`.
- Samples a trial's hyperparameters with Optuna.
- Applies overrides directly to the in-memory config object.
- Runs a single training job via `EpiForecasterTrainer`.
- Reports `best_val_loss` back to Optuna.

Typical SLURM usage pattern:
- Use a shared journal file on a shared filesystem.
- Run one (or a few) trials per task-array worker.

Example:
  uv run python scripts/optuna_epiforecaster_worker.py \
    --config configs/train_epiforecaster.yaml \
    --study-name epiforecaster_hpo_v1 \
    --journal-file outputs/optuna/epiforecaster_hpo_v1.journal \
    --n-trials 1

Notes:
- The current training loop does not expose intermediate metrics to Optuna, so
  pruning is not wired up yet.
- We force `training.plot_forecasts=False` by default to avoid workers clobbering
  shared forecast images (trainer writes `{split}_forecasts.png` at the
  experiment root).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any


import click

from models.configs import EpiForecasterConfig
from training.epiforecaster_trainer import EpiForecasterTrainer
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Global flag for signal handling
_shutdown_requested = False


def _handle_shutdown(signum: int, frame: Any) -> None:
    """Handle SLURM termination signals gracefully."""
    global _shutdown_requested
    logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
    _shutdown_requested = True


def _categorical_choices(choices: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    """Normalize choices so Optuna can persist them in storage.

    Optuna expects categorical choices to be None/bool/int/float/str. When a
    choice is a list/tuple/dict (e.g., a clip range), encode it as JSON.
    """
    normalized: list[Any] = []
    for choice in choices:
        if isinstance(choice, (list, tuple, dict)):
            normalized.append(json.dumps(choice))
        else:
            normalized.append(choice)
    return tuple(normalized)


def _slurm_identity() -> dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "HOSTNAME",
    ]
    return {k: os.getenv(k, "") for k in keys if os.getenv(k)}


def _overrides_to_dotlist(overrides: dict[str, Any]) -> list[str]:
    """Convert dict of overrides to dotlist format for OmegaConf.

    Args:
        overrides: Dict like {"training.learning_rate": 0.001, "model.gnn_depth": 2}

    Returns:
        List of strings like ["training.learning_rate=0.001", "model.gnn_depth=2"]
    """
    dotlist = []
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value_str = str(value)
        else:
            value_str = str(value)
        dotlist.append(f"{key}={value_str}")
    return dotlist


def suggest_epiforecaster_params(
    *, trial: Any, base_cfg: EpiForecasterConfig
) -> dict[str, Any]:
    """Define the search space and return dotted-key overrides.

    This is intentionally conservative and only targets params that are:
    - Exposed in current YAML/dataclass config surface, and
    - Actually used by dataset/trainer/model as implemented today.

    If you later move to OmegaConf or expose more model-head params, add them here.
    """

    overrides: dict[str, Any] = {}

    # --- training knobs (high leverage) ---
    overrides["training.learning_rate"] = trial.suggest_float(
        "training.learning_rate", 1e-5, 3e-3, log=True
    )
    overrides["training.weight_decay"] = trial.suggest_float(
        "training.weight_decay", 1e-8, 1e-3, log=True
    )
    overrides["training.batch_size"] = trial.suggest_categorical(
        "training.batch_size", _categorical_choices((16, 32, 64, 128))
    )
    # # Early stopping affects compute/overfit tradeoff; keep it moderate.
    # overrides["training.early_stopping_patience"] = trial.suggest_int(
    #     "training.early_stopping_patience", 5, 20
    # )

    # --- data knobs (high leverage; affect effective signal/noise) ---
    # overrides["data.log_scale"] = trial.suggest_categorical(
    #     "data.log_scale", _categorical_choices((False, True))
    # )

    # Neighborhood mask uses RAW mobility (not normalized). Using a categorical grid is
    # safer than continuous until we confirm the mobility unit scale in your dataset.
    overrides["data.mobility_threshold"] = trial.suggest_categorical(
        "data.mobility_threshold",
        _categorical_choices((0.0, 1.0, 5.0, 10.0, 20.0, 50.0)),
    )

    # Missingness filters dataset windows; can help robustness.
    overrides["data.missing_permit"] = trial.suggest_int("data.missing_permit", 0, 3)

    # --- model knobs (conditional) ---
    if base_cfg.model.type.mobility:
        overrides["model.gnn_depth"] = trial.suggest_int("model.gnn_depth", 1, 4)
        overrides["model.gnn_module"] = trial.suggest_categorical(
            "model.gnn_module", _categorical_choices(("gcn", "gat"))
        )
        overrides["model.mobility_embedding_dim"] = trial.suggest_categorical(
            "model.mobility_embedding_dim", _categorical_choices((16, 32, 64, 128))
        )

    if base_cfg.model.type.regions:
        # Region embedding dim must match the precomputed region2vec weights.
        # Keep it fixed in HPO; tune it in the region2vec trainer instead.
        pass

    return overrides


def objective(
    trial: Any,
    *,
    base_config_path: Path,
    study_name: str,
    run_root: Path | None,
    fixed_epochs: int | None,
    fixed_max_batches: int | None,
    seed: int | None,
) -> float:
    start_time = time.time()
    logger.info("=== Trial %d started ===", trial.number)
    logger.info("Trial %d suggested params: %s", trial.number, trial.params)

    cfg = EpiForecasterConfig.from_file(str(base_config_path))

    # Sample trial-specific overrides.
    overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
    override_list = _overrides_to_dotlist(overrides)

    # Add runtime-specific overrides
    override_list.append("training.plot_forecasts=false")
    override_list.append("training.profiler.enabled=false")
    if fixed_epochs is not None:
        override_list.append(f"training.epochs={fixed_epochs}")
    if fixed_max_batches is not None:
        override_list.append(f"training.max_batches={fixed_max_batches}")

    # Reload config with all overrides via OmegaConf
    cfg = EpiForecasterConfig.load(
        str(base_config_path),
        overrides=override_list,
    )

    # Make each run's artifacts unique.
    slurm = _slurm_identity()
    worker_tag = "-".join(
        [
            slurm.get("SLURM_ARRAY_JOB_ID", "") or slurm.get("SLURM_JOB_ID", ""),
            slurm.get("SLURM_ARRAY_TASK_ID", ""),
            slurm.get("SLURM_PROCID", ""),
        ]
    ).strip("-")
    if not worker_tag:
        worker_tag = "local"

    cfg.output.experiment_name = study_name
    if run_root is not None:
        cfg.output.log_dir = str(run_root)

    cfg.training.model_id = f"{worker_tag}_trial{trial.number}_{time.time_ns()}"

    # Basic reproducibility hook.
    if seed is not None:
        import numpy as np
        import torch

        s = int(seed) + int(trial.number)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    trial.set_user_attr("slurm", slurm)
    trial.set_user_attr("overrides", overrides)

    # Run training and return objective.
    trainer = EpiForecasterTrainer(cfg)
    results = trainer.run()
    best_val = float(results.get("best_val_loss", float("inf")))

    duration_s = time.time() - start_time
    logger.info(
        "Trial %d complete: loss=%.6f, duration=%.1fs",
        trial.number,
        best_val,
        duration_s,
    )
    logger.info("=== Trial %d finished ===", trial.number)

    # Safely get best value (returns inf if no trials completed yet)
    try:
        best_so_far = trial.study.best_value
    except ValueError:
        best_so_far = float("inf")

    if best_val == best_so_far:
        logger.info(
            "Trial %d is the new best trial! Best loss: %.6f",
            trial.number,
            best_val,
        )

    # Persist a small JSON summary next to the run logs if desired.
    try:
        log_dir = (
            Path(cfg.output.log_dir)
            / cfg.output.experiment_name
            / cfg.training.model_id
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "optuna_trial.json").write_text(
            json.dumps(
                {
                    "study": study_name,
                    "trial_number": trial.number,
                    "value": best_val,
                    "params": trial.params,
                    "overrides": overrides,
                    "slurm": slurm,
                    "config_effective": {
                        "output.log_dir": cfg.output.log_dir,
                        "output.experiment_name": cfg.output.experiment_name,
                        "training.model_id": cfg.training.model_id,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except Exception:
        # Best-effort only; objective value is what matters.
        logger.error("Failed to persist JSON summary")
        pass

    return best_val


@click.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option(
    "--journal-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Shared Optuna journal file (for SLURM array coordination).",
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Number of trials to run in this worker process (unset to run until timeout).",
)
@click.option(
    "--timeout-s",
    type=int,
    default=None,
    help="Optional wall-clock timeout for this worker (seconds).",
)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path),
    default=Path("outputs/optuna"),
    show_default=True,
    help="Root directory for trial outputs (log_dir override).",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Optional fixed epochs override for HPO speed.",
)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional fixed max_batches override for HPO speed.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Optional base RNG seed; trial.number is added.",
)
def main(
    *,
    config_path: Path,
    study_name: str,
    journal_file: Path,
    n_trials: int | None,
    timeout_s: int | None,
    run_root: Path,
    epochs: int | None,
    max_batches: int | None,
    seed: int | None,
) -> None:
    """Run one Optuna worker process."""
    setup_logging()

    try:
        optuna = importlib.import_module("optuna")
        storages = importlib.import_module("optuna.storages")
        JournalStorage = getattr(storages, "JournalStorage")
        journal_module = importlib.import_module("optuna.storages.journal")
        JournalFileBackend = getattr(journal_module, "JournalFileBackend")
    except Exception as exc:  # pragma: no cover
        raise click.ClickException(
            f"optuna is not available in this environment. Import error: {exc}"
        ) from exc

    run_root.mkdir(parents=True, exist_ok=True)
    journal_file.parent.mkdir(parents=True, exist_ok=True)

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    logger.info("Starting Optuna worker for study '%s'", study_name)
    logger.info("Config: %s", config_path)
    logger.info("Journal file: %s", journal_file)
    logger.info("Run root: %s", run_root)
    slurm = _slurm_identity()
    if any(slurm.values()):
        logger.info("SLURM identity: %s", slurm)
    logger.info("Starting trials: n_trials=%s, timeout_s=%s", n_trials, timeout_s)

    def _log_trial_complete(study: Any, trial: Any) -> None:
        logger.info(
            "Trial %d recorded by Optuna (value=%.6f). Completed %d/%s in study '%s'.",
            trial.number,
            float(trial.value) if trial.value is not None else float("inf"),
            len(study.trials),
            n_trials if n_trials is not None else "∞",
            study.study_name,
        )

    study.optimize(
        lambda t: objective(
            t,
            base_config_path=config_path,
            study_name=study_name,
            run_root=run_root,
            fixed_epochs=epochs,
            fixed_max_batches=max_batches,
            seed=seed,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        callbacks=[_log_trial_complete],
    )

    logger.info(
        "Worker complete: %d trials completed in study '%s'",
        len(study.trials),
        study_name,
    )
    if len(study.best_trials) > 0:
        logger.info(
            "Best result: %.6f (trial %d)",
            study.best_value,
            study.best_trial.number,
        )
        logger.info("Best params: %s", study.best_params)


if __name__ == "__main__":
    main()
