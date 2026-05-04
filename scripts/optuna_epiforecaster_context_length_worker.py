"""Optuna worker for controlled input-window length studies.

This is intentionally a fixed-design study, not a hyperparameter sweep:
workers enqueue the full ``context_length x seed`` matrix and only vary
``model.input_window_length`` plus ``training.seed``. Optuna JournalStorage is
used for coordination and resumability across SLURM array workers.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import click  # noqa: E402
import optuna  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import JournalFileBackend  # noqa: E402

from models.configs import EpiForecasterConfig  # noqa: E402
from training.epiforecaster_trainer import EpiForecasterTrainer  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

QUEUE_INITIALIZED_ATTR = "context_length_queue_initialized"
TERMINAL_TRIAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED,
}


def parse_int_list(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in raw.replace(",", " ").split():
        try:
            value = int(item)
        except ValueError as exc:
            raise click.BadParameter(
                f"{name} entries must be integers: {item}"
            ) from exc
        if value <= 0:
            raise click.BadParameter(f"{name} entries must be positive: {value}")
        values.append(value)

    if not values:
        raise click.BadParameter(f"{name} must contain at least one value")
    return values


def _compute_worker_seed(base_seed: int) -> int:
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is None:
        return base_seed
    return base_seed + int(slurm_task_id)


@contextmanager
def _file_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_trial_queue(
    *, context_lengths: list[int], seeds: list[int]
) -> list[dict[str, int]]:
    queue: list[dict[str, int]] = []
    for seed in seeds:
        for context_length in context_lengths:
            queue.append(
                {
                    "seed": int(seed),
                    "context_length": int(context_length),
                }
            )
    return queue


def _trial_identity(params: dict[str, object]) -> tuple[int, int] | None:
    seed = params.get("seed")
    context_length = params.get("context_length")
    if not isinstance(seed, int) or not isinstance(context_length, int):
        return None
    return seed, context_length


def trial_identity(trial: optuna.trial.FrozenTrial) -> tuple[int, int] | None:
    identity = _trial_identity(trial.params)
    if identity is not None:
        return identity

    fixed_params = trial.system_attrs.get("fixed_params", {})
    if isinstance(fixed_params, dict):
        return _trial_identity(fixed_params)
    return None


def initialize_study_queue(
    *,
    study: optuna.Study,
    context_lengths: list[int],
    seeds: list[int],
    lock_path: Path,
) -> None:
    expected = build_trial_queue(context_lengths=context_lengths, seeds=seeds)
    with _file_lock(lock_path):
        existing = {
            identity
            for trial in study.trials
            if (identity := trial_identity(trial)) is not None
        }
        missing = [
            params
            for params in expected
            if (params["seed"], params["context_length"]) not in existing
        ]
        for params in missing:
            study.enqueue_trial(params)
        if missing or not study.user_attrs.get(QUEUE_INITIALIZED_ATTR, False):
            study.set_user_attr(QUEUE_INITIALIZED_ATTR, True)
        logger.info(
            "Context-length queue initialized: expected=%d existing=%d enqueued=%d",
            len(expected),
            len(existing),
            len(missing),
        )


def is_study_complete(
    *, study: optuna.Study, context_lengths: list[int], seeds: list[int]
) -> bool:
    expected = {
        (seed, context_length) for seed in seeds for context_length in context_lengths
    }
    terminal = {
        identity
        for trial in study.trials
        if trial.state in TERMINAL_TRIAL_STATES
        if (identity := trial_identity(trial)) is not None
    }
    return terminal == expected


def _slurm_identity() -> dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "HOSTNAME",
    ]
    return {key: os.getenv(key, "") for key in keys if os.getenv(key)}


def _runtime_overrides(
    *,
    context_length: int,
    seed: int,
    campaign_id: str,
    epochs: int | None,
    max_batches: int | None,
    cli_overrides: list[str],
    smoketest: bool,
) -> list[str]:
    overrides = list(cli_overrides)
    overrides.extend(
        [
            f"model.input_window_length={context_length}",
            f"training.seed={seed}",
            "training.plot_forecasts=false",
            "training.profiler.enabled=false",
            "output.write_granular_eval=true",
            f"output.experiment_name=mn5_context_length__{campaign_id}__L{context_length}",
            f"output.wandb_group=context_length_{campaign_id}",
            f"output.wandb_tags=[mn5,context_length,campaign_{campaign_id},L{context_length},seed_{seed}]",
        ]
    )
    if epochs is not None:
        overrides.append(f"training.epochs={epochs}")
    if max_batches is not None:
        overrides.append(f"training.max_batches={max_batches}")
    if smoketest:
        overrides.append("training.epochs=1")
        overrides.append("training.max_batches=2")
    return overrides


def objective(
    trial: optuna.Trial,
    *,
    base_config_path: Path,
    context_lengths: list[int],
    seeds: list[int],
    run_root: Path,
    campaign_id: str,
    epochs: int | None,
    max_batches: int | None,
    cli_overrides: list[str],
    smoketest: bool = False,
) -> float:
    context_length = int(trial.suggest_categorical("context_length", context_lengths))
    seed = int(trial.suggest_categorical("seed", seeds))
    logger.info(
        "=== Trial %d started: context_length=%d seed=%d ===",
        trial.number,
        context_length,
        seed,
    )

    overrides = _runtime_overrides(
        context_length=context_length,
        seed=seed,
        campaign_id=campaign_id,
        epochs=epochs,
        max_batches=max_batches,
        cli_overrides=cli_overrides,
        smoketest=smoketest,
    )
    cfg = EpiForecasterConfig.load(str(base_config_path), overrides=overrides)
    cfg.output.log_dir = str(run_root)

    trial.set_user_attr("context_length", context_length)
    trial.set_user_attr("seed", seed)
    trial.set_user_attr("overrides", overrides)
    trial.set_user_attr("base_config_path", str(base_config_path))
    trial.set_user_attr("slurm", _slurm_identity())

    trainer = EpiForecasterTrainer(cfg)
    trainer.model_id = f"optuna_t{trial.number}_s{seed}_L{context_length}"

    results = trainer.run()
    best_val = float(results.get("best_val_loss", float("inf")))
    trial.set_user_attr("best_val_loss", best_val)
    if "total_epochs" in results:
        trial.set_user_attr("total_epochs", int(results["total_epochs"]))

    try:
        log_dir = (
            Path(cfg.output.log_dir) / cfg.output.experiment_name / trainer.model_id
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "context_length_trial.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "trial_number": trial.number,
                    "context_length": context_length,
                    "seed": seed,
                    "value": best_val,
                    "params": trial.params,
                    "overrides": overrides,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to persist context-length trial summary")

    logger.info(
        "=== Trial %d finished: context_length=%d seed=%d loss=%.6f ===",
        trial.number,
        context_length,
        seed,
        best_val,
    )
    return best_val


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option("--journal-file", type=click.Path(path_type=Path), required=True)
@click.option("--campaign-id", type=str, required=True)
@click.option(
    "--context-lengths",
    type=str,
    default="14 28 42 60 84 112",
    show_default=True,
    help="Space- or comma-separated model.input_window_length values.",
)
@click.option("--seeds", type=str, default="42 43 44", show_default=True)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path),
    default=Path("outputs/optuna"),
    show_default=True,
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
    "--epochs",
    type=int,
    default=20,
    show_default=True,
    help="Fixed epochs per context-length run.",
)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional fixed training.max_batches override.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Base sampler seed for reproducible worker ordering.",
)
@click.option(
    "--override",
    "cli_overrides",
    type=str,
    multiple=True,
    help="Common dotted-key config override applied before study-specific overrides.",
)
@click.option(
    "--smoketest",
    is_flag=True,
    default=False,
    help="Run max_batches=2 and epochs=1 for quick validation.",
)
def main(
    *,
    config: Path,
    study_name: str,
    journal_file: Path,
    campaign_id: str,
    context_lengths: str,
    seeds: str,
    run_root: Path,
    n_trials: int | None,
    timeout_s: int | None,
    epochs: int,
    max_batches: int | None,
    seed: int,
    cli_overrides: tuple[str, ...],
    smoketest: bool,
) -> None:
    setup_logging()

    context_length_list = parse_int_list(context_lengths, name="context-lengths")
    seed_list = parse_int_list(seeds, name="seeds")
    run_root.mkdir(parents=True, exist_ok=True)
    journal_file.parent.mkdir(parents=True, exist_ok=True)

    worker_seed = _compute_worker_seed(seed)
    sampler = optuna.samplers.RandomSampler(seed=worker_seed)
    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    queue_lock_path = journal_file.with_suffix(f"{journal_file.suffix}.queue.lock")
    initialize_study_queue(
        study=study,
        context_lengths=context_length_list,
        seeds=seed_list,
        lock_path=queue_lock_path,
    )

    logger.info("Context-length worker started for study '%s'", study_name)
    logger.info("Campaign ID: %s", campaign_id)
    logger.info("Context lengths: %s", context_length_list)
    logger.info("Seeds: %s", seed_list)
    logger.info("Total combinations: %d", len(context_length_list) * len(seed_list))
    logger.info("Starting trials: n_trials=%s, timeout_s=%s", n_trials, timeout_s)
    if smoketest:
        logger.info("SMOKETEST MODE: max_batches=2 and epochs=1")

    def _log_trial_complete(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        logger.info(
            "Trial %d recorded (state=%s, value=%s)",
            trial.number,
            trial.state.name,
            trial.value,
        )
        if is_study_complete(
            study=study,
            context_lengths=context_length_list,
            seeds=seed_list,
        ):
            logger.info("Context-length study matrix is complete")

    study.optimize(
        lambda current_trial: objective(
            current_trial,
            base_config_path=config,
            context_lengths=context_length_list,
            seeds=seed_list,
            run_root=run_root,
            campaign_id=campaign_id,
            epochs=epochs,
            max_batches=max_batches,
            cli_overrides=list(cli_overrides),
            smoketest=smoketest,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        callbacks=[_log_trial_complete],
        catch=(Exception,),
    )


if __name__ == "__main__":
    main()
