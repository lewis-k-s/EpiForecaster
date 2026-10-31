"""Stale-RUNNING recovery is mandatory and wall-time kills spare the retry budget."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import yaml

from scripts.hpo import ablation_worker as worker
from scripts.workflow.run_config import load_template, validate_workflow_contract
from tests.test_ablation_worker import _write_campaign

TEMPLATES = sorted(Path("workflow/config").glob("*.yaml"))


def _kind(path: Path) -> str:
    return load_template(path)["workflow"]["kind"]


def test_every_ablation_template_enables_stale_recovery() -> None:
    ablation_templates = [path for path in TEMPLATES if _kind(path) == "ablation"]
    assert ablation_templates
    for path in ablation_templates:
        document = load_template(path)
        assert document["ablation"]["recover_stale_running"] is True, path.name


def test_ablation_contract_rejects_missing_stale_recovery() -> None:
    document = yaml.safe_load(
        Path("workflow/config/mn5_ablation.yaml").read_text(encoding="utf-8")
    )
    assert validate_workflow_contract(document) == "ablation"
    del document["ablation"]["recover_stale_running"]
    with pytest.raises(ValueError, match="recover_stale_running must be true"):
        validate_workflow_contract(document)
    document["ablation"]["recover_stale_running"] = False
    with pytest.raises(ValueError, match="recover_stale_running must be true"):
        validate_workflow_contract(document)


def _campaign_and_study(tmp_path: Path) -> tuple[worker.AblationCampaign, optuna.Study]:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    study = optuna.create_study()
    return campaign, study


def _start_in_allocation(study: optuna.Study, allocation: str) -> optuna.Trial:
    trial = study.ask()
    trial.set_user_attr("slurm_allocation_id", allocation)
    return trial


def _claim_identity(
    study: optuna.Study, identity: tuple[str, int, int | None], allocation: str
) -> None:
    """Claim queued attempts until the requested identity is RUNNING."""
    while True:
        trial = _start_in_allocation(study, allocation)
        if worker.trial_identity(study.trials[trial.number]) == identity:
            return


def test_killed_running_trial_is_recovered_and_requeued(tmp_path: Path) -> None:
    campaign, study = _campaign_and_study(tmp_path)
    lock = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock)
    killed = _start_in_allocation(study, "100")
    identity = worker.trial_identity(study.trials[killed.number])

    # Next allocation: the old RUNNING attempt is stale and must not block.
    worker.initialize_study_queue(
        study=study,
        campaign=campaign,
        lock_path=lock,
        recover_stale_running=True,
        active_allocation_id="101",
    )

    attempts = [t for t in study.trials if worker.trial_identity(t) == identity]
    assert [t.state for t in attempts] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,
    ]
    assert study.user_attrs[worker.STALE_RECOVERED_ATTR] == [killed.number]
    assert worker.study_completion_status(study, campaign) == "pending"


def test_wall_time_kills_do_not_exhaust_the_failure_budget(tmp_path: Path) -> None:
    campaign, study = _campaign_and_study(tmp_path)
    lock = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock)
    identity = worker.trial_identity(study.trials[0])

    # More kills than FAILED_ATTEMPT_BUDGET, all below the stale-recovery bound.
    for allocation in range(worker.STALE_RECOVERY_BUDGET):
        _claim_identity(study, identity, str(allocation))
        worker.initialize_study_queue(
            study=study,
            campaign=campaign,
            lock_path=lock,
            recover_stale_running=True,
            active_allocation_id="next",
        )
    attempts = [t for t in study.trials if worker.trial_identity(t) == identity]
    assert worker._budgeted_failures(study, attempts) == 0
    assert attempts[-1].state == optuna.trial.TrialState.WAITING
    assert worker.study_completion_status(study, campaign) == "pending"


def test_repeated_kills_beyond_the_bound_exhaust_the_budget(tmp_path: Path) -> None:
    campaign, study = _campaign_and_study(tmp_path)
    lock = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock)
    identity = worker.trial_identity(study.trials[0])
    for allocation in range(
        worker.STALE_RECOVERY_BUDGET + worker.FAILED_ATTEMPT_BUDGET
    ):
        _claim_identity(study, identity, str(allocation))
        worker.initialize_study_queue(
            study=study,
            campaign=campaign,
            lock_path=lock,
            recover_stale_running=True,
            active_allocation_id="next",
        )
    attempts = [t for t in study.trials if worker.trial_identity(t) == identity]
    assert worker._budgeted_failures(study, attempts) == worker.FAILED_ATTEMPT_BUDGET
    with pytest.raises(ValueError, match="exhausted its retry budget"):
        worker.study_completion_status(study, campaign)


def test_model_failures_still_consume_the_budget(tmp_path: Path) -> None:
    campaign, study = _campaign_and_study(tmp_path)
    lock = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock)
    trial = study.ask()
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    attempts = [study.trials[trial.number]]
    assert worker._budgeted_failures(study, attempts) == 1
