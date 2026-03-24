"""Optuna worker for paired-seed ablation study coordination.

Uses Optuna JournalStorage to distribute ablation x seed trials across a
smaller pool of long-running workers. Trials are enqueued in seed-major order
so interrupted campaigns are biased toward complete k-seed paired studies.
"""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

# Ensure repo root is in path
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

QUEUE_INITIALIZED_ATTR = "ablation_queue_initialized"
ANALYSIS_LAUNCHED_ATTR = "ablation_analysis_launched"

# Keep baseline first so each seed block starts with a matched baseline run.
ABLATIONS: dict[str, str] = {
    "baseline": "",
    "sir:off": "training.loss.joint.w_sir=0.0",
    "sig:ww:aux": "training.loss.joint.disable_ww=true",
    "sig:ww:proxy": "training.loss.joint.mask_input_ww=true",
    "sig:ww:off": "training.loss.joint.disable_ww=true training.loss.joint.mask_input_ww=true",
    "sig:cases:aux": "training.loss.joint.disable_cases=true",
    "sig:cases:proxy": "training.loss.joint.mask_input_cases=true",
    "sig:cases:off": "training.loss.joint.disable_cases=true training.loss.joint.mask_input_cases=true",
    "sig:hosp:aux": "training.loss.joint.disable_hosp=true",
    "sig:hosp:proxy": "training.loss.joint.mask_input_hosp=true",
    "sig:hosp:off": "training.loss.joint.disable_hosp=true training.loss.joint.mask_input_hosp=true",
    "sig:deaths:aux": "training.loss.joint.disable_deaths=true",
    "sig:deaths:proxy": "training.loss.joint.mask_input_deaths=true",
    "sig:deaths:off": "training.loss.joint.disable_deaths=true training.loss.joint.mask_input_deaths=true",
    "residual:off": "model.observation_heads.residual_scale=0.0",
    "mobility:off": "model.type.mobility=false",
    "regions:off": "model.type.regions=false",
    "context:off": "model.type.mobility=false model.type.regions=false",
    "kernel:fixed": "model.observation_heads.learnable_kernel_hosp=false model.observation_heads.learnable_kernel_cases=false model.observation_heads.learnable_kernel_deaths=false",
    "gradnorm:off": "training.loss.joint.adaptive_scheme=none",
}

TERMINAL_TRIAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED,
}

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


def build_trial_queue(seeds: list[int]) -> list[dict[str, int | str]]:
    queue: list[dict[str, int | str]] = []
    ordered_ablations = list(ABLATIONS.keys())
    for seed in seeds:
        for ablation_name in ordered_ablations:
            queue.append({"seed": int(seed), "ablation": ablation_name})
    return queue


def _trial_identity(params: dict[str, object]) -> tuple[int, str] | None:
    ablation = params.get("ablation")
    seed = params.get("seed")
    if not isinstance(ablation, str) or not isinstance(seed, int):
        return None
    return seed, ablation


def trial_identity(trial: optuna.trial.FrozenTrial) -> tuple[int, str] | None:
    identity = _trial_identity(trial.params)
    if identity is not None:
        return identity

    fixed_params = trial.system_attrs.get("fixed_params", {})
    if isinstance(fixed_params, dict):
        return _trial_identity(fixed_params)
    return None


def _expected_trial_identities(seeds: list[int]) -> set[tuple[int, str]]:
    return {(seed, ablation) for seed in seeds for ablation in ABLATIONS}


def initialize_study_queue(
    *,
    study: optuna.Study,
    seeds: list[int],
    lock_path: Path,
) -> None:
    expected = build_trial_queue(seeds)
    with _file_lock(lock_path):
        existing = {
            identity
            for trial in study.trials
            if (identity := trial_identity(trial)) is not None
        }
        missing = [params for params in expected if (params["seed"], params["ablation"]) not in existing]
        for params in missing:
            study.enqueue_trial(params)
        if missing or not study.user_attrs.get(QUEUE_INITIALIZED_ATTR, False):
            study.set_user_attr(QUEUE_INITIALIZED_ATTR, True)
        logger.info(
            "Study queue initialized: expected=%d existing=%d enqueued=%d",
            len(expected),
            len(existing),
            len(missing),
        )


def is_study_complete(study: optuna.Study, seeds: list[int]) -> bool:
    expected = _expected_trial_identities(seeds)
    terminal: set[tuple[int, str]] = set()
    baseline_by_seed: set[int] = set()
    candidates_by_seed: dict[int, set[str]] = {seed: set() for seed in seeds}

    for trial in study.trials:
        identity = _trial_identity(trial.params)
        if identity is None or identity not in expected:
            continue
        if trial.state not in TERMINAL_TRIAL_STATES:
            continue

        seed, ablation = identity
        terminal.add(identity)
        if ablation == "baseline":
            baseline_by_seed.add(seed)
        else:
            candidates_by_seed.setdefault(seed, set()).add(ablation)

    if terminal != expected:
        return False

    candidate_names = {name for name in ABLATIONS if name != "baseline"}
    for seed in seeds:
        if seed not in baseline_by_seed:
            return False
        if candidates_by_seed.get(seed, set()) != candidate_names:
            return False

    return True


def maybe_run_analysis(
    *,
    study: optuna.Study,
    campaign_id: str,
    seeds: list[int],
    coordination_lock_path: Path,
    runner: Callable[[str], None] | None = None,
) -> bool:
    with _file_lock(coordination_lock_path):
        if study.user_attrs.get(ANALYSIS_LAUNCHED_ATTR, False):
            return False
        if not is_study_complete(study, seeds):
            return False

        analysis_runner = runner or _run_campaign_analysis
        analysis_runner(campaign_id)
        study.set_user_attr(ANALYSIS_LAUNCHED_ATTR, True)
        logger.info("Marked analysis launched for campaign '%s'", campaign_id)
        return True


def _run_campaign_analysis(campaign_id: str) -> None:
    logger.info("Running ablation analysis for completed campaign '%s'", campaign_id)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/analyze_ablations.py",
            "--campaign-id",
            campaign_id,
        ],
        check=True,
    )


def objective(
    trial: optuna.Trial,
    *,
    base_config_path: Path,
    seeds: list[int],
    run_root: Path,
    campaign_id: str,
    smoketest: bool = False,
) -> float:
    ablation_name = trial.suggest_categorical("ablation", list(ABLATIONS.keys()))
    seed = int(trial.suggest_categorical("seed", seeds))
    logger.info(
        "=== Trial %d started: ablation=%s seed=%d ===",
        trial.number,
        ablation_name,
        seed,
    )

    overrides: list[str] = []
    ablation_overrides = ABLATIONS[ablation_name]
    if ablation_overrides:
        overrides.extend(ablation_overrides.split())

    overrides.append(f"training.seed={seed}")
    overrides.append("output.write_granular_eval=true")

    experiment_name = f"mn5_ablation__{campaign_id}__{ablation_name}"
    overrides.append(f"output.experiment_name={experiment_name}")
    overrides.append(f"output.wandb_group=ablation_cv_{campaign_id}")
    overrides.append(
        f"output.wandb_tags=[mn5,ablation,cv,{ablation_name},campaign_{campaign_id},seed_{seed}]"
    )

    if smoketest:
        overrides.append("training.max_batches=2")
        overrides.append("training.epochs=1")

    cfg = EpiForecasterConfig.load(str(base_config_path), overrides=overrides)
    cfg.output.log_dir = str(run_root)

    trainer = EpiForecasterTrainer(cfg)
    trainer.model_id = f"optuna_t{trial.number}_s{seed}"

    results = trainer.run()
    val_loss = float(results.get("best_val_loss", float("inf")))
    logger.info("=== Trial %d finished: loss=%.6f ===", trial.number, val_loss)
    return val_loss


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option("--journal-file", type=click.Path(path_type=Path), required=True)
@click.option("--campaign-id", type=str, required=True)
@click.option("--seeds", type=str, default="42 43 44 45 46", show_default=True)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path),
    default=Path("outputs/training"),
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
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Base sampler seed for reproducibility.",
)
@click.option(
    "--smoketest",
    is_flag=True,
    default=False,
    help="Run smoketest mode with max_batches=2 and epochs=1 for quick validation.",
)
def main(
    config: Path,
    study_name: str,
    journal_file: Path,
    campaign_id: str,
    seeds: str,
    run_root: Path,
    n_trials: int | None,
    timeout_s: int | None,
    seed: int,
    smoketest: bool,
) -> None:
    setup_logging()

    seed_list = [int(item) for item in seeds.split()]
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
    analysis_lock_path = journal_file.with_suffix(f"{journal_file.suffix}.analysis.lock")
    initialize_study_queue(study=study, seeds=seed_list, lock_path=queue_lock_path)

    logger.info("Ablation worker started for study '%s'", study_name)
    logger.info("Campaign ID: %s", campaign_id)
    logger.info("Ablations: %d, Seeds: %d", len(ABLATIONS), len(seed_list))
    logger.info("Total combinations: %d", len(ABLATIONS) * len(seed_list))
    logger.info("Starting trials: n_trials=%s, timeout_s=%s", n_trials, timeout_s)
    if smoketest:
        logger.info("SMOKETEST MODE: max_batches=2, epochs=1")

    def _log_trial_complete(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info(
            "Trial %d recorded (state=%s, value=%s)",
            trial.number,
            trial.state.name,
            trial.value,
        )
        if trial.state in TERMINAL_TRIAL_STATES:
            try:
                maybe_run_analysis(
                    study=study,
                    campaign_id=campaign_id,
                    seeds=seed_list,
                    coordination_lock_path=analysis_lock_path,
                )
            except Exception:
                logger.exception("Post-trial analysis check failed")

    study.optimize(
        lambda current_trial: objective(
            current_trial,
            base_config_path=config,
            seeds=seed_list,
            run_root=run_root,
            campaign_id=campaign_id,
            smoketest=smoketest,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        callbacks=[_log_trial_complete],
        catch=(Exception,),
    )

    try:
        maybe_run_analysis(
            study=study,
            campaign_id=campaign_id,
            seeds=seed_list,
            coordination_lock_path=analysis_lock_path,
        )
    except Exception:
        logger.exception("Final analysis check failed")


if __name__ == "__main__":
    main()
