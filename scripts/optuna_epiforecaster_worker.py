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
- Joint loss weights are fixed by default to avoid objective drift across trials.
- We force `training.plot_forecasts=False` by default to avoid workers clobbering
  shared forecast images (trainer writes `{split}_forecasts.png` at the
  experiment root).
"""

from __future__ import annotations

import importlib
import json
import logging
import math
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
_JOINT_WEIGHT_KEYS = ("w_ww", "w_hosp", "w_cases", "w_deaths", "w_sir")


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


def _parse_batch_grad_combo(value: Any) -> tuple[int, int]:
    """Validate and parse (batch_size, grad_accum) from categorical value."""
    decoded = _decode_categorical_value(value)
    if not isinstance(decoded, (list, tuple)):
        raise ValueError(
            "training.batch_grad_combo must be a 2-item list/tuple like [32, 4]"
        )
    if len(decoded) != 2:
        raise ValueError(
            "training.batch_grad_combo must have exactly 2 values: "
            "[batch_size, gradient_accumulation_steps]"
        )

    batch_size, grad_accum = decoded
    if not isinstance(batch_size, int) or not isinstance(grad_accum, int):
        raise ValueError(
            "training.batch_grad_combo values must be positive integers: "
            "[batch_size, gradient_accumulation_steps]"
        )
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError(
            "training.batch_grad_combo values must be > 0: "
            "[batch_size, gradient_accumulation_steps]"
        )
    return batch_size, grad_accum


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


def _bounded_joint_loss_weight_overrides(
    *,
    trial: Any,
    base_cfg: EpiForecasterConfig,
    max_ratio: float,
) -> dict[str, float]:
    """Tune joint loss weights in a bounded simplex around base config values.

    Keeps the total joint weight sum fixed to the base config and limits each
    component's relative movement by `max_ratio`.
    """
    if max_ratio < 1.0:
        raise ValueError("loss_weight_max_ratio must be >= 1.0")

    base_joint = base_cfg.training.loss.joint
    base_weights = {key: float(getattr(base_joint, key)) for key in _JOINT_WEIGHT_KEYS}
    total = sum(base_weights.values())
    if total <= 0:
        raise ValueError("Sum of joint loss weights must be > 0 for bounded tuning")

    lower = 1.0 / max_ratio
    upper = max_ratio
    scaled = {}
    for key, base in base_weights.items():
        mult = trial.suggest_float(
            f"training.loss.joint.mult_{key}",
            lower,
            upper,
            log=True,
        )
        scaled[key] = max(base, 1e-12) * mult

    scaled_sum = sum(scaled.values())
    normalized = {key: total * (value / scaled_sum) for key, value in scaled.items()}
    return {f"training.loss.joint.{key}": value for key, value in normalized.items()}


def suggest_epiforecaster_params(
    *,
    trial: Any,
    base_cfg: EpiForecasterConfig,
    loss_weight_mode: str = "fixed",
    loss_weight_max_ratio: float = 2.0,
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
    # Jointly sample batch_size and grad_accum to guarantee constraints:
    # - batch_size * grad_accum >= 8 (minimum effective batch)
    # - batch_size * grad_accum <= 32 (maximum effective batch, reduced for OOM safety)
    valid_batch_grad_combos: list[tuple[int, int]] = [
        (8, 1),  # effective 8
        (8, 2),  # effective 16
        (8, 4),  # effective 32
        (16, 1),  # effective 16
        (16, 2),  # effective 32
        (32, 1),  # effective 32
    ]
    combo = trial.suggest_categorical(
        "training.batch_grad_combo",
        _categorical_choices(valid_batch_grad_combos),
    )
    batch_size, grad_accum = _parse_batch_grad_combo(combo)
    overrides["training.batch_size"] = batch_size
    overrides["training.gradient_accumulation_steps"] = grad_accum
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

    # Missingness filters dataset windows; computed from window sizes.
    # Daily series (cases, deaths): allow up to 50% missing days
    # Weekly series (hospitalizations, biomarkers_joint): range based on expected measurements
    input_len = base_cfg.model.input_window_length
    horizon_len = base_cfg.model.forecast_horizon

    # Daily series - input window: 0-50% of input_window_length
    max_missing_input_daily = int(input_len * 0.5)
    missing_input_daily = trial.suggest_int(
        "data.missing_permit_input_daily", 0, max_missing_input_daily
    )
    overrides["data.missing_permit.input.cases"] = missing_input_daily
    overrides["data.missing_permit.input.deaths"] = missing_input_daily

    # Daily series - horizon window: 0-50% of forecast_horizon
    max_missing_horizon_daily = int(horizon_len * 0.5)
    missing_horizon_daily = trial.suggest_int(
        "data.missing_permit_horizon_daily", 0, max_missing_horizon_daily
    )
    overrides["data.missing_permit.horizon.cases"] = missing_horizon_daily
    overrides["data.missing_permit.horizon.deaths"] = missing_horizon_daily

    # Weekly series - input window: range from "all present" to "at least 1 measurement"
    # min_missing = window_length - ceil(window_length / 7)  (all weekly measurements present)
    # max_missing = window_length - 1  (only 1 day has data)
    expected_input_weekly = math.ceil(input_len / 7)
    min_missing_input_weekly = input_len - expected_input_weekly
    max_missing_input_weekly = input_len - 1
    missing_input_weekly = trial.suggest_int(
        "data.missing_permit_input_weekly",
        min_missing_input_weekly,
        max_missing_input_weekly,
    )
    overrides["data.missing_permit.input.hospitalizations"] = missing_input_weekly
    overrides["data.missing_permit.input.biomarkers_joint"] = missing_input_weekly

    # Weekly series - horizon window: same logic
    expected_horizon_weekly = math.ceil(horizon_len / 7)
    min_missing_horizon_weekly = horizon_len - expected_horizon_weekly
    max_missing_horizon_weekly = horizon_len - 1
    missing_horizon_weekly = trial.suggest_int(
        "data.missing_permit_horizon_weekly",
        min_missing_horizon_weekly,
        max_missing_horizon_weekly,
    )
    overrides["data.missing_permit.horizon.hospitalizations"] = missing_horizon_weekly
    overrides["data.missing_permit.horizon.biomarkers_joint"] = missing_horizon_weekly

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

    if base_cfg.model.temporal_covariates_dim > 0:
        use_temporal_covariates = trial.suggest_categorical(
            "model.use_temporal_covariates",
            _categorical_choices((True, False)),
        )
        overrides["model.include_day_of_week"] = (
            base_cfg.model.include_day_of_week if use_temporal_covariates else False
        )
        overrides["model.include_holidays"] = (
            base_cfg.model.include_holidays if use_temporal_covariates else False
        )

    # Backbone positional encoding type (already implemented in TransformerBackbone).
    overrides["model.head_positional_encoding"] = trial.suggest_categorical(
        "model.head_positional_encoding",
        _categorical_choices(("sinusoidal", "learned")),
    )

    # --- SIR joint inference knobs (high leverage for observation heads) ---
    # Only tune if using joint_inference loss
    if base_cfg.training.loss.name == "joint_inference":
        # Loss-weight policy:
        # - fixed: keep base config objective unchanged across trials.
        # - bounded: small relative re-balancing around base config proportions.
        if loss_weight_mode == "bounded":
            overrides.update(
                _bounded_joint_loss_weight_overrides(
                    trial=trial,
                    base_cfg=base_cfg,
                    max_ratio=loss_weight_max_ratio,
                )
            )
        elif loss_weight_mode != "fixed":
            raise ValueError(
                f"Unknown loss_weight_mode={loss_weight_mode!r}; expected fixed|bounded"
            )

        # Residual connection params - affects model capacity for observation heads
        overrides["model.observation_heads.residual_scale"] = trial.suggest_float(
            "model.observation_heads.residual_scale", 0.05, 0.5
        )
        overrides["model.observation_heads.residual_hidden_dim"] = (
            trial.suggest_categorical(
                "model.observation_heads.residual_hidden_dim",
                _categorical_choices((16, 32, 64, 128)),
            )
        )
        overrides["model.observation_heads.residual_layers"] = trial.suggest_int(
            "model.observation_heads.residual_layers", 1, 4
        )
        overrides["model.observation_heads.residual_dropout"] = trial.suggest_float(
            "model.observation_heads.residual_dropout", 0.0, 0.3
        )

        # Observation context dimension - affects representational power
        overrides["model.observation_heads.obs_context_dim"] = (
            trial.suggest_categorical(
                "model.observation_heads.obs_context_dim",
                _categorical_choices((32, 64, 96, 128, 192)),
            )
        )

        # Residual mode - additive vs modulation
        overrides["model.observation_heads.residual_mode"] = trial.suggest_categorical(
            "model.observation_heads.residual_mode",
            _categorical_choices(("additive", "modulation")),
        )

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
    loss_weight_mode: str = "fixed",
    loss_weight_max_ratio: float = 2.0,
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
        loss_weight_mode=loss_weight_mode,
        loss_weight_max_ratio=loss_weight_max_ratio,
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
    "--loss-weight-mode",
    type=click.Choice(["fixed", "bounded"], case_sensitive=False),
    default="fixed",
    show_default=True,
    help=(
        "Joint loss weight search mode: fixed (recommended), "
        "bounded (re-balance around config)"
    ),
)
@click.option(
    "--loss-weight-max-ratio",
    type=float,
    default=2.0,
    show_default=True,
    help=(
        "Max multiplicative deviation for --loss-weight-mode=bounded. "
        "Example: 2.0 means each weight multiplier is in [0.5, 2.0]."
    ),
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
    loss_weight_mode: str,
    loss_weight_max_ratio: float,
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
            selected_sampler = CmaEsSampler(seed=seed, warn_independent_sampling=False)
            logger.info("Using CMA-ES sampler (seed=%d)", seed)
        except ImportError:
            logger.warning("CMA-ES not available, falling back to TPE sampler")
            selected_sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    elif sampler == "tpe":
        # Use multivariate=True for better handling of parameter correlations
        selected_sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
        logger.info("Using TPE sampler with multivariate=True (seed=%d)", seed)
    else:  # random
        selected_sampler = optuna.samplers.RandomSampler(seed=seed)
        logger.info("Using Random sampler (seed=%d)", seed)

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
        "loss_weight_mode=%s, loss_weight_max_ratio=%.3f, "
        "pruner=PercentilePruner(25%%)",
        epochs,
        pruning_start_epoch,
        seed,
        loss_weight_mode,
        loss_weight_max_ratio,
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
            loss_weight_mode=loss_weight_mode,
            loss_weight_max_ratio=loss_weight_max_ratio,
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
