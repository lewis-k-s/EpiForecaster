"""Ablation worker for paired-seed ablation study coordination.

Uses Optuna JournalStorage to distribute ablation x seed trials across a
smaller pool of long-running workers. Trials are enqueued in seed-major order
so interrupted campaigns are biased toward complete k-seed paired studies.
"""

from __future__ import annotations

import fcntl
import logging
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import click
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from models.configs import EpiForecasterConfig
from scripts.hpo.worker_identity import compute_worker_seed as _compute_worker_seed
from training.epiforecaster_trainer import EpiForecasterTrainer
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

QUEUE_INITIALIZED_ATTR = "ablation_queue_initialized"

# Keep baseline first so each seed block starts with a matched baseline run.
# Order: baseline → spatial/context → physics → loss → signal heads → kernels.
ABLATIONS: dict[str, str] = {
    "baseline": "",
    "mobility:off": "model.type.mobility=false",
    "mobility:spatial_queen": "model.graph_adjacency_source=spatial_queen",
    # "mobility:spatial_knn": "model.graph_adjacency_source=spatial_knn",
    "regions:off": "model.type.regions=false",
    "context:off": "model.type.mobility=false model.type.regions=false",
    "context:spatial_queen": "model.type.regions=false model.graph_adjacency_source=spatial_queen",
    # "context:spatial_knn": "model.type.regions=false model.graph_adjacency_source=spatial_knn",
    "residual:off": "model.observation_heads.residual_scale=0.0",
    "loss:crps": "training.loss.joint.observation_loss=crps",
    "gradnorm:crps": "training.loss.joint.observation_loss=crps training.loss.joint.adaptive_scheme=gradnorm",
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
    "kernel:fixed": "model.observation_heads.learnable_kernel_hosp=false model.observation_heads.learnable_kernel_cases=false model.observation_heads.learnable_kernel_deaths=false",
    "kernel:ww:free": "model.observation_heads.learnable_kernel_ww=true model.observation_heads.kernel_parameterization_ww=free",
    "kernel:hosp:free": "model.observation_heads.learnable_kernel_hosp=true model.observation_heads.kernel_parameterization_hosp=free",
    "kernel:cases:free": "model.observation_heads.learnable_kernel_cases=true model.observation_heads.kernel_parameterization_cases=free",
    "kernel:deaths:free": "model.observation_heads.learnable_kernel_deaths=true model.observation_heads.kernel_parameterization_deaths=free",
    "kernel:all:free": "model.observation_heads.learnable_kernel_ww=true model.observation_heads.learnable_kernel_hosp=true model.observation_heads.learnable_kernel_cases=true model.observation_heads.learnable_kernel_deaths=true model.observation_heads.kernel_parameterization_ww=free model.observation_heads.kernel_parameterization_hosp=free model.observation_heads.kernel_parameterization_cases=free model.observation_heads.kernel_parameterization_deaths=free",
    "kernel:all:mlp": "model.observation_heads.kernel_mlp_ww=true model.observation_heads.kernel_mlp_hosp=true model.observation_heads.kernel_mlp_cases=true model.observation_heads.kernel_mlp_deaths=true",
}

TERMINAL_TRIAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED,
}


def resolve_ablation_plan(requested: str | None) -> dict[str, str]:
    """Return the ordered ablation subset requested for this campaign."""
    if requested is None or not requested.strip():
        return dict(ABLATIONS)

    names = [name.strip() for name in requested.replace(",", " ").split()]
    unknown = [name for name in names if name not in ABLATIONS]
    if unknown:
        raise click.ClickException(
            "Unknown ablation(s): "
            f"{', '.join(unknown)}. Available: {', '.join(ABLATIONS)}"
        )
    return {name: ABLATIONS[name] for name in names}


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
    seeds: list[int], ablations: dict[str, str] | None = None
) -> list[dict[str, int | str]]:
    queue: list[dict[str, int | str]] = []
    ordered_ablations = list((ablations or ABLATIONS).keys())
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


def _expected_trial_identities(
    seeds: list[int], ablations: dict[str, str] | None = None
) -> set[tuple[int, str]]:
    return {(seed, ablation) for seed in seeds for ablation in (ablations or ABLATIONS)}


def initialize_study_queue(
    *,
    study: optuna.Study,
    seeds: list[int],
    ablations: dict[str, str] | None = None,
    lock_path: Path,
    requeue_running: bool = False,
) -> None:
    ablations = ablations or ABLATIONS
    expected = build_trial_queue(seeds, ablations)
    with _file_lock(lock_path):
        existing = set()
        for trial in study.trials:
            identity = trial_identity(trial)
            if identity is None:
                continue
            if requeue_running and trial.state == optuna.trial.TrialState.RUNNING:
                continue
            existing.add(identity)
        missing = [
            params
            for params in expected
            if (params["seed"], params["ablation"]) not in existing
        ]
        for params in missing:
            study.enqueue_trial(params)
        if missing or not study.user_attrs.get(QUEUE_INITIALIZED_ATTR, False):
            study.set_user_attr(QUEUE_INITIALIZED_ATTR, True)
        logger.info(
            "Study queue initialized: expected=%d existing=%d enqueued=%d requeue_running=%s",
            len(expected),
            len(existing),
            len(missing),
            requeue_running,
        )


def is_study_complete(
    study: optuna.Study, seeds: list[int], ablations: dict[str, str] | None = None
) -> bool:
    ablations = ablations or ABLATIONS
    expected = _expected_trial_identities(seeds, ablations)
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

    candidate_names = {name for name in ablations if name != "baseline"}
    for seed in seeds:
        if seed not in baseline_by_seed:
            return False
        if candidates_by_seed.get(seed, set()) != candidate_names:
            return False

    return True


def objective(
    trial: optuna.Trial,
    *,
    base_config_path: Path,
    seeds: list[int],
    run_root: Path,
    campaign_id: str,
    ablations: dict[str, str],
    common_overrides: list[str] | None = None,
    smoketest: bool = False,
) -> float:
    ablation_name = trial.suggest_categorical("ablation", list(ablations.keys()))
    seed = int(trial.suggest_categorical("seed", seeds))
    logger.info(
        "=== Trial %d started: ablation=%s seed=%d ===",
        trial.number,
        ablation_name,
        seed,
    )

    overrides: list[str] = list(common_overrides or [])
    ablation_overrides = ablations[ablation_name]
    if ablation_overrides:
        overrides.extend(ablation_overrides.split())

    overrides.append(f"training.seed={seed}")
    overrides.append("training.plot_forecasts=false")
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
    cfg.output.log_dir = str(run_root / campaign_id)

    trainer = EpiForecasterTrainer(
        cfg,
        model_id=f"ablation_t{trial.number}_s{seed}",
    )

    results = trainer.run()
    selection_score = float(results["best_val_selection_score"])
    selection_metric = str(results["best_val_selection_metric"])
    trial.set_user_attr("best_val_selection_score", selection_score)
    trial.set_user_attr("best_val_selection_metric", selection_metric)
    logger.info(
        "=== Trial %d finished: %s=%.6f ===",
        trial.number,
        selection_metric,
        selection_score,
    )
    return selection_score


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--study-name", type=str, required=True)
@click.option(
    "--journal-file", type=click.Path(path_type=Path), required=False, default=None
)
@click.option("--campaign-id", type=str, required=True)
@click.option("--seeds", type=str, default="42 43 44 45 46", show_default=True)
@click.option(
    "--ablations",
    type=str,
    default=None,
    help=(
        "Whitespace- or comma-separated ablation names to run. "
        "Defaults to the full registry."
    ),
)
@click.option(
    "--overrides",
    type=str,
    default="",
    help="Whitespace-separated config overrides applied to every ablation trial.",
)
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
@click.option(
    "--requeue-running",
    is_flag=True,
    default=False,
    help=(
        "Treat existing RUNNING trials as stale and enqueue their identities again. "
        "Use only when recovering a campaign after all previous workers have exited."
    ),
)
def main(
    config: Path,
    study_name: str,
    journal_file: Path,
    campaign_id: str,
    seeds: str,
    ablations: str | None,
    overrides: str,
    run_root: Path,
    n_trials: int | None,
    timeout_s: int | None,
    seed: int,
    smoketest: bool,
    requeue_running: bool,
) -> None:
    setup_logging()

    seed_list = [int(item) for item in seeds.split()]
    ablation_plan = resolve_ablation_plan(ablations)
    common_overrides = overrides.split() if overrides.strip() else []
    run_root.mkdir(parents=True, exist_ok=True)

    if journal_file is None:
        journal_file = Path("outputs/ablation_studies") / f"{study_name}.journal"
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
        seeds=seed_list,
        ablations=ablation_plan,
        lock_path=queue_lock_path,
        requeue_running=requeue_running,
    )

    logger.info("Ablation worker started for study '%s'", study_name)
    logger.info("Campaign ID: %s", campaign_id)
    logger.info("Ablations: %d, Seeds: %d", len(ablation_plan), len(seed_list))
    logger.info("Ablation names: %s", ", ".join(ablation_plan))
    if common_overrides:
        logger.info("Common overrides: %s", " ".join(common_overrides))
    logger.info("Total combinations: %d", len(ablation_plan) * len(seed_list))
    logger.info("Starting trials: n_trials=%s, timeout_s=%s", n_trials, timeout_s)
    if smoketest:
        logger.info("SMOKETEST MODE: max_batches=2, epochs=1")
    if requeue_running:
        logger.info("Requeueing stale RUNNING trial identities")

    study.optimize(
        lambda current_trial: objective(
            current_trial,
            base_config_path=config,
            seeds=seed_list,
            run_root=run_root,
            campaign_id=campaign_id,
            ablations=ablation_plan,
            common_overrides=common_overrides,
            smoketest=smoketest,
        ),
        n_trials=n_trials,
        timeout=timeout_s,
        catch=(Exception,),
    )


if __name__ == "__main__":
    main()
