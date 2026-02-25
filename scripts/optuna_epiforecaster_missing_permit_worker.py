"""Optuna worker for missing-permit-only multi-objective EpiForecaster search.

This worker runs a dedicated sweep over data.missing_permit.* parameters while
freezing all non-missing-permit hyperparameters from a prior full-HPO study.

Objectives:
1) Minimize validation loss (best_val_loss)
2) Maximize validation coverage ratio (val_samples / val_samples_reference)
"""

from __future__ import annotations

import importlib
import json
import logging
import math
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
from scripts.optuna_epiforecaster_worker import (  # noqa: E402
    _compute_worker_seed,
    _overrides_to_dotlist,
    _slurm_identity,
)
from training.epiforecaster_trainer import EpiForecasterTrainer  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def suggest_missing_permit_params(
    *, trial: Any, base_cfg: EpiForecasterConfig
) -> dict[str, int]:
    """Define missing-permit-only search space and return dotted-key overrides."""
    overrides: dict[str, int] = {}

    input_len = int(base_cfg.model.input_window_length)
    horizon_len = int(base_cfg.model.forecast_horizon)

    # Daily series - input and horizon windows: allow up to 50% missing.
    max_missing_input_daily = int(input_len * 0.5)
    missing_input_daily = trial.suggest_int(
        "data.missing_permit_input_daily",
        0,
        max_missing_input_daily,
    )
    overrides["data.missing_permit.input.cases"] = missing_input_daily
    overrides["data.missing_permit.input.deaths"] = missing_input_daily

    max_missing_horizon_daily = int(horizon_len * 0.5)
    missing_horizon_daily = trial.suggest_int(
        "data.missing_permit_horizon_daily",
        0,
        max_missing_horizon_daily,
    )
    overrides["data.missing_permit.horizon.cases"] = missing_horizon_daily
    overrides["data.missing_permit.horizon.deaths"] = missing_horizon_daily

    # Weekly series - range from "all expected weekly measurements present"
    # to "at least one day with data in the window".
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

    return overrides


def max_permissive_missing_permit_overrides(
    base_cfg: EpiForecasterConfig,
) -> dict[str, int]:
    """Build most permissive missing-permit settings within this sweep's bounds."""
    input_len = int(base_cfg.model.input_window_length)
    horizon_len = int(base_cfg.model.forecast_horizon)

    max_input_daily = int(input_len * 0.5)
    max_horizon_daily = int(horizon_len * 0.5)
    max_input_weekly = input_len - 1
    max_horizon_weekly = horizon_len - 1

    return {
        "data.missing_permit.input.cases": max_input_daily,
        "data.missing_permit.input.deaths": max_input_daily,
        "data.missing_permit.horizon.cases": max_horizon_daily,
        "data.missing_permit.horizon.deaths": max_horizon_daily,
        "data.missing_permit.input.hospitalizations": max_input_weekly,
        "data.missing_permit.input.biomarkers_joint": max_input_weekly,
        "data.missing_permit.horizon.hospitalizations": max_horizon_weekly,
        "data.missing_permit.horizon.biomarkers_joint": max_horizon_weekly,
    }


def _is_missing_permit_override(key: str) -> bool:
    return key.startswith("data.missing_permit.")


def _non_missing_permit_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Remove missing_permit keys from override mapping."""
    return {k: v for k, v in overrides.items() if not _is_missing_permit_override(k)}


def _compute_val_coverage_ratio(val_samples: int, val_samples_reference: int) -> float:
    """Compute val sample retention ratio with zero-reference guard."""
    if val_samples_reference <= 0:
        logger.warning(
            "val_samples_reference=%d is non-positive; coverage ratio set to 0.0",
            val_samples_reference,
        )
        return 0.0
    return float(val_samples) / float(val_samples_reference)


def _load_freeze_trial(
    *,
    storage: Any,
    study_name: str,
    freeze_trial_number: int | None,
) -> Any:
    """Load source full-HPO trial used to freeze non-permit parameters."""
    optuna = importlib.import_module("optuna")
    study = optuna.load_study(study_name=study_name, storage=storage)

    if freeze_trial_number is None:
        if not study.best_trials:
            raise ValueError(
                f"Study '{study_name}' has no completed trials; cannot freeze parameters."
            )
        return study.best_trial

    for trial in study.trials:
        if trial.number == freeze_trial_number:
            return trial

    raise ValueError(
        f"Trial number {freeze_trial_number} not found in study '{study_name}'."
    )


def load_frozen_non_permit_overrides(
    *,
    freeze_journal_file: Path,
    freeze_study_name: str,
    freeze_trial_number: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen non-missing-permit overrides from an existing full-HPO study."""
    storages = importlib.import_module("optuna.storages")
    JournalStorage = getattr(storages, "JournalStorage")
    journal_module = importlib.import_module("optuna.storages.journal")
    JournalFileBackend = getattr(journal_module, "JournalFileBackend")

    freeze_storage = JournalStorage(JournalFileBackend(str(freeze_journal_file)))
    source_trial = _load_freeze_trial(
        storage=freeze_storage,
        study_name=freeze_study_name,
        freeze_trial_number=freeze_trial_number,
    )

    raw_overrides = source_trial.user_attrs.get("overrides")
    if not isinstance(raw_overrides, dict):
        raise ValueError(
            "Selected freeze trial is missing user_attrs['overrides']; "
            "cannot freeze non-missing-permit hyperparameters."
        )

    frozen = _non_missing_permit_overrides(raw_overrides)
    source_meta = {
        "freeze_journal_file": str(freeze_journal_file),
        "freeze_study_name": freeze_study_name,
        "freeze_trial_number": int(source_trial.number),
        "freeze_trial_value": source_trial.value,
    }
    return frozen, source_meta


def create_missing_permit_study(
    *,
    study_name: str,
    storage: Any,
    sampler: Any,
    pruner: Any,
) -> Any:
    """Create/load missing-permit multi-objective study."""
    optuna = importlib.import_module("optuna")
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )


def _build_effective_override_list(
    *,
    frozen_non_permit_overrides: dict[str, Any],
    missing_permit_overrides: dict[str, Any],
    fixed_epochs: int | None,
    fixed_max_batches: int | None,
    cli_overrides: list[str],
) -> list[str]:
    """Build complete dotted-key overrides for a trial config."""
    merged: dict[str, Any] = {}
    merged.update(frozen_non_permit_overrides)
    merged.update(missing_permit_overrides)

    override_list = _overrides_to_dotlist(merged)

    # Runtime overrides for clean HPO workers.
    override_list.append("training.plot_forecasts=false")
    override_list.append("training.profiler.enabled=false")
    override_list.append("output.wandb_mode=disabled")
    override_list.append("training.early_stopping_patience=null")
    if fixed_epochs is not None:
        override_list.append(f"training.epochs={fixed_epochs}")
    if fixed_max_batches is not None:
        override_list.append(f"training.max_batches={fixed_max_batches}")

    override_list.extend(cli_overrides)
    return override_list


def _compute_reference_val_samples(
    *,
    base_config_path: Path,
    frozen_non_permit_overrides: dict[str, Any],
    cli_overrides: list[str],
    fixed_epochs: int | None,
    fixed_max_batches: int | None,
    base_cfg_for_bounds: EpiForecasterConfig,
) -> int:
    """Compute val sample count for maximally permissive missing-permit settings."""
    reference_permit = max_permissive_missing_permit_overrides(base_cfg_for_bounds)
    override_list = _build_effective_override_list(
        frozen_non_permit_overrides=frozen_non_permit_overrides,
        missing_permit_overrides=reference_permit,
        fixed_epochs=fixed_epochs,
        fixed_max_batches=fixed_max_batches,
        cli_overrides=cli_overrides,
    )
    cfg = EpiForecasterConfig.load(str(base_config_path), overrides=override_list)

    trainer = EpiForecasterTrainer(cfg)
    try:
        return int(len(trainer.val_dataset))
    finally:
        trainer.cleanup_dataloaders()


def objective(
    trial: Any,
    *,
    base_config_path: Path,
    study_name: str,
    run_root: Path | None,
    fixed_epochs: int | None,
    fixed_max_batches: int | None,
    seed: int | None,
    pruning_start_epoch: int,
    cli_overrides: list[str],
    frozen_non_permit_overrides: dict[str, Any],
    frozen_overrides_source: dict[str, Any],
    base_cfg_for_bounds: EpiForecasterConfig,
    val_samples_reference: int,
) -> tuple[float, float]:
    """Run one trial and return (best_val_loss, val_coverage_ratio)."""
    start_time = time.time()
    logger.info("=== Trial %d started ===", trial.number)

    missing_overrides = suggest_missing_permit_params(
        trial=trial,
        base_cfg=base_cfg_for_bounds,
    )
    override_list = _build_effective_override_list(
        frozen_non_permit_overrides=frozen_non_permit_overrides,
        missing_permit_overrides=missing_overrides,
        fixed_epochs=fixed_epochs,
        fixed_max_batches=fixed_max_batches,
        cli_overrides=cli_overrides,
    )

    cfg = EpiForecasterConfig.load(str(base_config_path), overrides=override_list)

    slurm = _slurm_identity()
    cfg.output.experiment_name = study_name
    if run_root is not None:
        cfg.output.log_dir = str(run_root)

    if seed is not None:
        import numpy as np
        import torch

        s = int(seed) + int(trial.number)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    trial.set_user_attr("slurm", slurm)
    trial.set_user_attr("missing_permit_overrides", missing_overrides)
    trial.set_user_attr("frozen_overrides_source", frozen_overrides_source)

    trainer = EpiForecasterTrainer(
        cfg,
        trial=trial,
        pruning_start_epoch=pruning_start_epoch,
    )

    try:
        results = trainer.run()
        best_val_loss = float(results.get("best_val_loss", float("inf")))
        val_samples = int(len(trainer.val_dataset))
        val_coverage_ratio = _compute_val_coverage_ratio(
            val_samples=val_samples,
            val_samples_reference=val_samples_reference,
        )
    except Exception as exc:
        if hasattr(exc, "__class__") and exc.__class__.__name__ == "TrialPruned":
            logger.info("Trial %d pruned by Optuna", trial.number)
            raise
        raise

    trial.set_user_attr("val_samples", val_samples)
    trial.set_user_attr("val_samples_reference", val_samples_reference)
    trial.set_user_attr("val_coverage_ratio", val_coverage_ratio)

    duration_s = time.time() - start_time
    logger.info(
        "Trial %d complete: best_val_loss=%.6f, val_coverage_ratio=%.6f, duration=%.1fs",
        trial.number,
        best_val_loss,
        val_coverage_ratio,
        duration_s,
    )
    logger.info("=== Trial %d finished ===", trial.number)

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
                    "values": [best_val_loss, val_coverage_ratio],
                    "missing_permit_overrides": missing_overrides,
                    "frozen_overrides_source": frozen_overrides_source,
                    "val_samples": val_samples,
                    "val_samples_reference": val_samples_reference,
                    "slurm": slurm,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except Exception:
        logger.error("Failed to persist trial JSON summary")

    return best_val_loss, val_coverage_ratio


@click.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option(
    "--journal-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Shared Optuna journal file for this missing-permit sweep.",
)
@click.option(
    "--freeze-journal-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Journal file for the full-HPO source study used to freeze non-permit params.",
)
@click.option(
    "--freeze-study-name",
    type=str,
    required=True,
    help="Study name in --freeze-journal-file used as freeze source.",
)
@click.option(
    "--freeze-trial-number",
    type=int,
    default=None,
    help="Optional explicit source trial number from freeze study (default: best trial).",
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Number of trials for this worker (unset to run until timeout).",
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
    help="Fixed epochs per trial for this sweep.",
)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional fixed max_batches override for faster trials.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Base RNG seed.",
)
@click.option(
    "--sampler",
    type=click.Choice(["cmaes", "tpe", "random"], case_sensitive=False),
    default="tpe",
    show_default=True,
    help="Optuna sampler algorithm.",
)
@click.option(
    "--pruning-start-epoch",
    type=int,
    default=10,
    show_default=True,
    help="Epoch to start pruning checks.",
)
@click.option(
    "--override",
    "cli_overrides",
    type=str,
    multiple=True,
    help="Override config values using dotted keys; can be repeated.",
)
def main(
    *,
    config_path: Path,
    study_name: str,
    journal_file: Path,
    freeze_journal_file: Path,
    freeze_study_name: str,
    freeze_trial_number: int | None,
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
    """Run one Optuna worker for missing-permit Pareto sweep."""
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

    worker_seed = _compute_worker_seed(seed)
    if worker_seed != seed:
        logger.info(
            "Offsetting seed by SLURM_ARRAY_TASK_ID: %d -> %d",
            seed,
            worker_seed,
        )

    if sampler == "cmaes":
        try:
            from optuna.samplers import CmaEsSampler

            logger.warning(
                "CMA-ES selected with integer-only search space; "
                "consider TPE for robust mixed-worker behavior."
            )
            selected_sampler = CmaEsSampler(
                seed=worker_seed,
                warn_independent_sampling=False,
            )
            logger.info("Using CMA-ES sampler (seed=%d)", worker_seed)
        except ImportError:
            logger.warning("CMA-ES not available, falling back to TPE sampler")
            selected_sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=worker_seed,
            )
    elif sampler == "tpe":
        selected_sampler = optuna.samplers.TPESampler(
            multivariate=True,
            seed=worker_seed,
        )
        logger.info("Using TPE sampler with multivariate=True (seed=%d)", worker_seed)
    else:
        selected_sampler = optuna.samplers.RandomSampler(seed=worker_seed)
        logger.info("Using Random sampler (seed=%d)", worker_seed)

    selected_pruner = optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=2,
        n_warmup_steps=pruning_start_epoch,
    )

    cli_overrides_list = list(cli_overrides)
    frozen_non_permit_overrides, frozen_source = load_frozen_non_permit_overrides(
        freeze_journal_file=freeze_journal_file,
        freeze_study_name=freeze_study_name,
        freeze_trial_number=freeze_trial_number,
    )
    logger.info(
        "Loaded %d frozen non-permit overrides from %s",
        len(frozen_non_permit_overrides),
        frozen_source,
    )

    # Build base config for bound calculations after fixed overrides are applied.
    bound_cfg = EpiForecasterConfig.load(
        str(config_path),
        overrides=_overrides_to_dotlist(frozen_non_permit_overrides)
        + cli_overrides_list,
    )
    val_samples_reference = _compute_reference_val_samples(
        base_config_path=config_path,
        frozen_non_permit_overrides=frozen_non_permit_overrides,
        cli_overrides=cli_overrides_list,
        fixed_epochs=epochs,
        fixed_max_batches=max_batches,
        base_cfg_for_bounds=bound_cfg,
    )
    logger.info("Reference validation samples: %d", val_samples_reference)

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = create_missing_permit_study(
        study_name=study_name,
        storage=storage,
        sampler=selected_sampler,
        pruner=selected_pruner,
    )

    logger.info("Starting missing-permit Optuna worker for study '%s'", study_name)
    logger.info("Config: %s", config_path)
    logger.info("Journal file: %s", journal_file)
    logger.info("Run root: %s", run_root)
    logger.info("Freeze source: %s", frozen_source)
    logger.info(
        "Settings: epochs=%d, pruning_start_epoch=%d, seed=%d, "
        "pruner=PercentilePruner(25%%), objectives=(best_val_loss, val_coverage_ratio)",
        epochs,
        pruning_start_epoch,
        seed,
    )
    slurm = _slurm_identity()
    if any(slurm.values()):
        logger.info("SLURM identity: %s", slurm)
    logger.info("Starting trials: n_trials=%s, timeout_s=%s", n_trials, timeout_s)

    def _log_trial_complete(study: Any, trial: Any) -> None:
        values = trial.values if trial.values is not None else [float("inf"), 0.0]
        logger.info(
            "Trial %d recorded: best_val_loss=%.6f, val_coverage_ratio=%.6f. "
            "Completed %d/%s in study '%s'.",
            trial.number,
            float(values[0]),
            float(values[1]),
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
            pruning_start_epoch=pruning_start_epoch,
            cli_overrides=cli_overrides_list,
            frozen_non_permit_overrides=frozen_non_permit_overrides,
            frozen_overrides_source=frozen_source,
            base_cfg_for_bounds=bound_cfg,
            val_samples_reference=val_samples_reference,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        callbacks=[_log_trial_complete],
        catch=(Exception,),
    )

    logger.info(
        "Worker complete: %d trials in study '%s' (%d Pareto-optimal)",
        len(study.trials),
        study.study_name,
        len(study.best_trials),
    )
    if study.best_trials:
        for idx, pareto_trial in enumerate(study.best_trials[:5], start=1):
            values = pareto_trial.values if pareto_trial.values is not None else []
            logger.info(
                "Pareto #%d: trial=%d values=%s",
                idx,
                pareto_trial.number,
                values,
            )


if __name__ == "__main__":
    main()
