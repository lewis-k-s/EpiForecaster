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
- Uses CMA-ES sampler by default for better convergence on high-dimensional spaces.
- Implements early pruning to kill unpromising trials and save compute.
- Default: 50 epochs per trial with pruning after epoch 10.
- We force `training.plot_forecasts=False` by default to avoid workers clobbering
  shared forecast images (trainer writes `{split}_forecasts.png` at the
  experiment root).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is in path before other imports (fixes scripts/utils conflict)
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import click  # noqa: E402

from models.configs import EpiForecasterConfig  # noqa: E402
from training.epiforecaster_trainer import EpiForecasterTrainer  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

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


def _decode_categorical_value(value: Any) -> Any:
    """Decode JSON-encoded categorical values produced by _categorical_choices."""
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("[") or candidate.startswith("{"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON categorical value: {value!r}") from exc
    return value


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


def _compute_worker_seed(base_seed: int) -> int:
    """Compute sampler seed offset by SLURM task ID for parallel worker diversity.

    When multiple workers start simultaneously with the same seed, they sample
    identical hyperparameters until TPE has completed trials to learn from.
    Offsetting by SLURM_ARRAY_TASK_ID ensures each worker has a unique seed.

    Args:
        base_seed: The base seed from CLI --seed argument.

    Returns:
        Worker-specific seed (base_seed + SLURM_ARRAY_TASK_ID if set, else base_seed).
    """
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is not None:
        return base_seed + int(slurm_task_id)
    return base_seed


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
    *,
    trial: Any,
    base_cfg: EpiForecasterConfig,
) -> dict[str, Any]:
    """Define the search space and return dotted-key overrides.

    This is an intentionally reduced first-pass search space. The current Optuna
    journal is failure-heavy and completion-poor, so we focus on a small set of
    first-order knobs before revisiting startup/init and controller tuning.
    """

    overrides: dict[str, Any] = {}

    # --- optimizer + data knobs ---
    overrides["training.learning_rate"] = trial.suggest_float(
        "training.learning_rate", 1e-5, 3e-3, log=True
    )
    overrides["training.weight_decay"] = trial.suggest_float(
        "training.weight_decay", 1e-8, 1e-3, log=True
    )
    # NOTE: batch_size is NOT swept - fixed at base config value (32 works well).
    # Previously we swept batch/grad_accum combos, but gradient_accumulation_steps
    # is now hardcoded to 1 in the trainer and batch_size 32 has proven optimal.
    # Leaving commented for reference:
    # valid_batch_grad_combos: list[tuple[int, int]] = [
    #     (8, 1),  # effective 8
    #     (8, 2),  # effective 16
    #     (8, 4),  # effective 32
    #     (16, 1),  # effective 16
    #     (16, 2),  # effective 32
    #     (32, 1),  # effective 32
    # ]
    # combo = trial.suggest_categorical(
    #     "training.batch_grad_combo",
    #     _categorical_choices(valid_batch_grad_combos),
    # )
    # batch_size, grad_accum = _parse_batch_grad_combo(combo)
    # overrides["training.batch_size"] = batch_size
    # overrides["training.gradient_accumulation_steps"] = grad_accum
    # # Early stopping affects compute/overfit tradeoff; keep it moderate.
    # overrides["training.early_stopping_patience"] = trial.suggest_int(
    #     "training.early_stopping_patience", 5, 20
    # )

    # --- data knobs (high leverage; affect effective signal/noise) ---
    # overrides["data.log_scale"] = trial.suggest_categorical(
    #     "data.log_scale", _categorical_choices((False, True))
    # )

    # Node vs time iteration order; independent of split_strategy.
    overrides["data.sample_ordering"] = trial.suggest_categorical(
        "data.sample_ordering",
        _categorical_choices(("node", "time")),
    )

    # Neighborhood mask uses RAW mobility (not normalized). Using a categorical grid is
    # safer than continuous until we confirm the mobility unit scale in your dataset.
    overrides["data.mobility_threshold"] = trial.suggest_categorical(
        "data.mobility_threshold",
        _categorical_choices((0.0, 1.0, 5.0, 10.0, 20.0, 50.0)),
    )

    # Pin fixed values for the simplified pass rather than inheriting config-specific
    # defaults, so each trial uses the same reduced baseline.
    overrides["model.head_positional_encoding"] = "sinusoidal"
    overrides["model.observation_heads.residual_mode"] = "additive"
    overrides["model.observation_heads.obs_context_dim"] = 96
    overrides["model.observation_heads.residual_layers"] = 2
    overrides["model.observation_heads.residual_dropout"] = 0.1
    overrides["model.observation_heads.learnable_kernel_ww"] = False
    overrides["model.observation_heads.learnable_kernel_hosp"] = False

    # --- mobility graph knobs (conditional) ---
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

    if base_cfg.model.temporal_covariates_dim > 0:
        overrides["model.include_day_of_week"] = base_cfg.model.include_day_of_week
        overrides["model.include_holidays"] = base_cfg.model.include_holidays

    # --- observation-head capacity ---
    overrides["model.observation_heads.residual_hidden_dim"] = (
        trial.suggest_categorical(
            "model.observation_heads.residual_hidden_dim",
            _categorical_choices((16, 32, 64, 128)),
        )
    )
    overrides["training.loss.joint.w_continuity"] = trial.suggest_categorical(
        "training.loss.joint.w_continuity",
        _categorical_choices((0.0, 0.01, 0.05, 0.1)),
    )

    # --- deferred phase-2 tuning: init and startup dynamics ---
    # overrides["model.init_weights.rezero_init"] = trial.suggest_categorical(
    #     "init_weights.rezero_init",
    #     _categorical_choices((1.0e-3, 3.0e-3, 1.0e-2)),
    # )
    # overrides["model.init_weights.rate_head_final_gain"] = trial.suggest_categorical(
    #     "init_weights.rate_head_final_gain",
    #     _categorical_choices((5.0e-3, 1.0e-2, 2.0e-2)),
    # )
    # overrides["model.init_weights.initial_state_final_gain"] = (
    #     trial.suggest_categorical(
    #         "init_weights.initial_state_final_gain",
    #         _categorical_choices((5.0e-3, 1.0e-2, 2.0e-2)),
    #     )
    # )
    # overrides["model.init_weights.obs_context_final_gain"] = trial.suggest_categorical(
    #     "init_weights.obs_context_final_gain",
    #     _categorical_choices((0.25, 0.5, 1.0)),
    # )
    # overrides["model.observation_heads.residual_scale"] = trial.suggest_float(
    #     "init_weights.observation_residual_scale", 0.03, 0.2
    # )

    # --- deferred phase-2 tuning: GradNorm controller ---
    # adaptive_scheme = base_cfg.training.loss.joint.adaptive_scheme
    # if adaptive_scheme.lower() == "gradnorm":
    #     overrides["training.loss.joint.gradnorm_alpha"] = trial.suggest_float(
    #         "training.loss.joint.gradnorm_alpha", 0.5, 2.5
    #     )
    #     overrides["training.loss.joint.gradnorm_weight_lr"] = trial.suggest_float(
    #         "training.loss.joint.gradnorm_weight_lr",
    #         1.0e-4,
    #         5.0e-3,
    #         log=True,
    #     )
    #     overrides["training.loss.joint.gradnorm_warmup_steps"] = trial.suggest_int(
    #         "training.loss.joint.gradnorm_warmup_steps",
    #         0,
    #         100,
    #     )
    #     overrides["training.loss.joint.gradnorm_update_every"] = (
    #         trial.suggest_categorical(
    #             "training.loss.joint.gradnorm_update_every",
    #             _categorical_choices((8, 16, 32)),
    #         )
    #     )
    #     overrides["training.loss.joint.gradnorm_ema_decay"] = trial.suggest_float(
    #         "training.loss.joint.gradnorm_ema_decay",
    #         0.85,
    #         0.99,
    #     )
    #     overrides["training.loss.joint.gradnorm_min_weight"] = (
    #         trial.suggest_categorical(
    #             "training.loss.joint.gradnorm_min_weight",
    #             _categorical_choices((5.0e-4, 1.0e-3, 2.0e-3)),
    #         )
    #     )

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
    cli_overrides: list[str],
    pruning_start_epoch: int = 10,
) -> float:
    start_time = time.time()
    logger.info("=== Trial %d started ===", trial.number)
    logger.info("Trial %d suggested params: %s", trial.number, trial.params)

    cfg = EpiForecasterConfig.from_file(str(base_config_path))

    # Sample trial-specific overrides.
    # Note: effective batch size constraint (batch_size * grad_accum >= 32) is
    # enforced in suggest_epiforecaster_params via conditional sampling.
    overrides = suggest_epiforecaster_params(
        trial=trial,
        base_cfg=cfg,
    )

    override_list = _overrides_to_dotlist(overrides)

    # Add runtime-specific overrides
    override_list.append("training.plot_forecasts=false")
    override_list.append("training.profiler.enabled=false")
    # Disable WandB by default for Optuna sweeps (Optuna tracks metrics/parameters)
    override_list.append("output.wandb_mode=disabled")
    # Disable early stopping for HPO - rely on Optuna pruning instead
    override_list.append("training.early_stopping_patience=null")
    if fixed_epochs is not None:
        override_list.append(f"training.epochs={fixed_epochs}")
    if fixed_max_batches is not None:
        override_list.append(f"training.max_batches={fixed_max_batches}")

    # Add CLI-provided overrides (dotted keys, same as train CLI)
    for override in cli_overrides:
        override_list.append(override)

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
    trial.set_user_attr("base_config_path", str(base_config_path))

    # Run training and return objective.
    # Pass trial to trainer for intermediate pruning support.
    trainer = EpiForecasterTrainer(
        cfg, trial=trial, pruning_start_epoch=pruning_start_epoch
    )

    try:
        results = trainer.run()
        best_val = float(results.get("best_val_loss", float("inf")))
    except Exception as exc:
        # Check if this is an Optuna TrialPruned exception
        if hasattr(exc, "__class__") and exc.__class__.__name__ == "TrialPruned":
            logger.info("Trial %d pruned by Optuna", trial.number)
            raise
        # Re-raise other exceptions
        raise

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
            Path(cfg.output.log_dir) / cfg.output.experiment_name / trainer.model_id
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
                        "trainer.model_id": trainer.model_id,
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
    default=50,
    show_default=True,
    help="Fixed epochs per trial for HPO (default: 50).",
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
    default=42,
    show_default=True,
    help="Base RNG seed for reproducibility (default: 42).",
)
@click.option(
    "--sampler",
    type=click.Choice(["cmaes", "tpe", "random"], case_sensitive=False),
    default="cmaes",
    show_default=True,
    help="Optuna sampler algorithm to use.",
)
@click.option(
    "--pruning-start-epoch",
    type=int,
    default=10,
    show_default=True,
    help="Epoch to start checking for pruning (default: 10).",
)
@click.option(
    "--override",
    "cli_overrides",
    type=str,
    multiple=True,
    help="Override config values using dotted keys (e.g., env=mn5, training.device=cuda). Can be repeated.",
)
def main(
    *,
    config_path: Path,
    study_name: str,
    journal_file: Path,
    n_trials: int | None,
    timeout_s: int | None,
    run_root: Path,
    epochs: int,
    max_batches: int | None,
    seed: int,
    sampler: str,
    pruning_start_epoch: int,
    cli_overrides: tuple[str, ...],
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

    # Offset seed by SLURM task ID to ensure diversity across parallel workers
    worker_seed = _compute_worker_seed(seed)
    if worker_seed != seed:
        logger.info(
            "Offsetting seed by SLURM_ARRAY_TASK_ID: %d -> %d",
            seed,
            worker_seed,
        )

    # Select sampler based on CLI choice
    # Note: CMA-ES doesn't support categorical parameters well, so we default to TPE
    if sampler == "cmaes":
        try:
            from optuna.samplers import CmaEsSampler

            # CMA-ES doesn't support categorical parameters; warn user
            logger.warning(
                "CMA-ES selected but search space contains categorical parameters. "
                "CMA-ES will fall back to RandomSampler for categoricals. "
                "Consider using 'tpe' sampler instead for mixed spaces."
            )
            selected_sampler = CmaEsSampler(
                seed=worker_seed, warn_independent_sampling=False
            )
            logger.info("Using CMA-ES sampler (seed=%d)", worker_seed)
        except ImportError:
            logger.warning("CMA-ES not available, falling back to TPE sampler")
            selected_sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=worker_seed
            )
    elif sampler == "tpe":
        # Use multivariate=True for better handling of parameter correlations
        selected_sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=worker_seed
        )
        logger.info("Using TPE sampler with multivariate=True (seed=%d)", worker_seed)
    else:  # random
        selected_sampler = optuna.samplers.RandomSampler(seed=worker_seed)
        logger.info("Using Random sampler (seed=%d)", worker_seed)

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    # Use PercentilePruner to enable early pruning even with few workers
    # Prunes trials in the bottom 25th percentile after just 2 trials complete
    selected_pruner = optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=2,
        n_warmup_steps=pruning_start_epoch,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=selected_sampler,
        pruner=selected_pruner,
    )

    logger.info("Starting Optuna worker for study '%s'", study_name)
    logger.info("Config: %s", config_path)
    logger.info("Journal file: %s", journal_file)
    logger.info("Run root: %s", run_root)
    logger.info(
        "Settings: epochs=%d, pruning_start_epoch=%d, seed=%d, "
        "pruner=PercentilePruner(25%%)",
        epochs,
        pruning_start_epoch,
        seed,
    )
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

    # Build override list from CLI (dotted keys, same as train CLI)
    base_overrides = list(cli_overrides)

    study.optimize(
        lambda t: objective(
            t,
            base_config_path=config_path,
            study_name=study_name,
            run_root=run_root,
            fixed_epochs=epochs,
            fixed_max_batches=max_batches,
            seed=seed,
            cli_overrides=base_overrides,
            pruning_start_epoch=pruning_start_epoch,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        callbacks=[_log_trial_complete],
        catch=(Exception,),
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
