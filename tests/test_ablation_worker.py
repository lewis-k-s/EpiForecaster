from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import optuna
import pytest
import yaml

from scripts.hpo import ablation_worker as worker


@dataclass
class _QueuedTrial:
    params: dict[str, int | str]
    state: optuna.trial.TrialState = optuna.trial.TrialState.WAITING


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_QueuedTrial] = []
        self.user_attrs: dict[str, object] = {}

    def enqueue_trial(self, params: dict[str, int | str]) -> None:
        self.trials.append(_QueuedTrial(dict(params)))

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


def _write_campaign(
    tmp_path: Path,
    *,
    folds: list[int] | None = None,
    smoke_folds: list[int] | None = None,
    allowed_paths: list[str] | None = None,
    require_full_manifest_folds: bool = False,
    stage: str = "ablation_smoke",
    extra: dict[str, object] | None = None,
) -> Path:
    base = Path("configs/train_epifor_smoke.yaml").resolve()
    campaign: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "campaign-test",
        "stage": stage,
        "hypothesis": "Structured conditions are paired.",
        "base_config": str(base),
        "baseline_condition": "baseline",
        "smoke_overrides": {"training": {"compile": False}},
        "fixed_overrides": {"training": {"epochs": 1}},
        "conditions": [
            {
                "name": "baseline",
                "hypothesis": "reference",
                "overrides": {},
                "allowed_difference_paths": [],
            },
            {
                "name": "mobility:off",
                "hypothesis": "model switch",
                "overrides": {"model.type.mobility": False},
                "allowed_difference_paths": (
                    ["model.type.mobility"] if allowed_paths is None else allowed_paths
                ),
            },
        ],
        "paired_model_seeds": [42, 43],
        "metrics": {
            "primary": {"name": "mae_median", "direction": "minimize"},
            "secondary": ["rmse_median"],
            "safety": ["coverage_ratio"],
            "diagnostic": ["best_val_selection_value"],
        },
        "equal_budget_policy": {"paths": ["training.epochs", "training.max_batches"]},
        "completeness_requirements": {
            "require_exact_identity_matrix": True,
            "require_terminal_trials": True,
            "require_finite_primary_metric": True,
            "require_full_manifest_folds": require_full_manifest_folds,
        },
        "provenance_requirements": {
            "require_campaign_sha256": True,
            "require_output_identity": True,
        },
    }
    campaign.update(extra or {})
    if folds is not None:
        manifest = tmp_path / "splits.json"
        manifest.write_text(
            json.dumps({"num_folds": max(folds) + 1, "folds": []}) + "\n",
            encoding="utf-8",
        )
        cross_validation: dict[str, object] = {
            "split_manifest": str(manifest),
            "fold_indices": folds,
        }
        if smoke_folds is not None:
            cross_validation["smoke_fold_indices"] = smoke_folds
        campaign["cross_validation"] = cross_validation
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
    return path


def test_structured_campaign_compiles_condition_fold_seed_matrix(
    tmp_path: Path,
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path, folds=[0, 1]))
    assert campaign.crossval_num_folds == 2

    queue = worker.build_trial_queue(campaign)

    assert queue == [
        {"condition": "baseline", "seed": 42, "fold": 0},
        {"condition": "baseline", "seed": 43, "fold": 0},
        {"condition": "baseline", "seed": 42, "fold": 1},
        {"condition": "baseline", "seed": 43, "fold": 1},
        {"condition": "mobility:off", "seed": 42, "fold": 0},
        {"condition": "mobility:off", "seed": 43, "fold": 0},
        {"condition": "mobility:off", "seed": 42, "fold": 1},
        {"condition": "mobility:off", "seed": 43, "fold": 1},
    ]
    assert len(worker.expected_trial_identities(campaign)) == 8


def test_smoketest_queue_uses_declaration_owned_fold_subset(tmp_path: Path) -> None:
    campaign = worker.load_campaign(
        _write_campaign(tmp_path, folds=[0, 1, 2], smoke_folds=[0, 2])
    )

    assert campaign.selected_fold_indices(smoketest=False) == (0, 1, 2)
    assert campaign.selected_fold_indices(smoketest=True) == (0, 2)
    assert len(worker.expected_trial_identities(campaign)) == 12
    assert len(worker.expected_trial_identities(campaign, smoketest=True)) == 8


def test_folded_objective_enables_manifest_backed_cross_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path, folds=[0, 1]))
    captured: dict[str, list[str]] = {}
    attrs: dict[str, object] = {}

    class Trial:
        number = 0

        def suggest_categorical(self, name, choices):  # noqa: ANN001
            return {"condition": "baseline", "seed": 42, "fold": 1}[name]

        def set_user_attr(self, key, value):  # noqa: ANN001
            attrs[key] = value

    class Trainer:
        experiment_dir = None

        def __init__(self, config, model_id):  # noqa: ANN001
            return None

        def run(self):
            return {
                "best_val_selection_score": 1.0,
                "best_val_selection_metric": "mae_median",
            }

    def load_config(path, overrides):  # noqa: ANN001
        captured["overrides"] = overrides
        return SimpleNamespace(output=SimpleNamespace(log_dir=""))

    monkeypatch.setattr(worker.EpiForecasterConfig, "load", staticmethod(load_config))
    monkeypatch.setattr(worker, "EpiForecasterTrainer", Trainer)

    worker.objective(Trial(), campaign=campaign, run_root=tmp_path, smoketest=True)

    assert "training.crossval_enabled=true" in captured["overrides"]
    assert "training.crossval_num_folds=2" in captured["overrides"]
    assert "training.crossval_fold_index=1" in captured["overrides"]
    assert "training.compile=false" in captured["overrides"]
    wandb_tags_override = next(
        value
        for value in captured["overrides"]
        if value.startswith("output.wandb_tags=")
    )
    wandb_tags = json.loads(wandb_tags_override.split("=", 1)[1])
    assert f"campaign_sha256_{campaign.sha256[:48]}" in wandb_tags
    assert all(1 <= len(tag) <= 64 for tag in wandb_tags)
    assert attrs["best_val_selection_metric"] == "mae_median"
    assert attrs["best_val_selection_value"] == 1.0
    assert attrs["optuna_value_purpose"] == "validation_selection_coordination"
    assert "campaign_primary_metric" not in attrs
    assert "campaign_primary_direction" not in attrs


def test_campaign_rejects_fold_outside_manifest_bounds(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path, folds=[0, 1])
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["cross_validation"]["fold_indices"] = [2]
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="exceed manifest num_folds"):
        worker.load_campaign(path)


def test_campaign_full_coverage_requires_every_manifest_fold(tmp_path: Path) -> None:
    path = _write_campaign(
        tmp_path,
        folds=[0, 1, 2],
        require_full_manifest_folds=True,
    )
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["cross_validation"]["fold_indices"] = [1, 2]
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="full manifest fold coverage"):
        worker.load_campaign(path)


def test_campaign_rejects_undeclared_config_drift(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path, allowed_paths=[])

    with pytest.raises(ValueError, match="undeclared config drift"):
        worker.load_campaign(path)


def test_campaign_rejects_missing_split_manifest_before_execution(
    tmp_path: Path,
) -> None:
    path = _write_campaign(tmp_path)
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["cross_validation"] = {"fold_indices": [0, 1]}
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="split_manifest is required"):
        worker.load_campaign(path)


def test_initialize_study_queue_is_idempotent(tmp_path: Path) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path, folds=[0, 1]))
    study = _FakeStudy()

    worker.initialize_study_queue(
        study=study, campaign=campaign, lock_path=tmp_path / "queue.lock"
    )
    worker.initialize_study_queue(
        study=study, campaign=campaign, lock_path=tmp_path / "queue.lock"
    )

    identities = [worker._trial_identity(trial.params) for trial in study.trials]
    assert len(identities) == 8
    assert len(set(identities)) == 8
    assert study.user_attrs["campaign_sha256"] == campaign.sha256


def test_initialize_study_queue_rejects_campaign_hash_mismatch(tmp_path: Path) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    study = _FakeStudy()
    study.user_attrs["campaign_sha256"] = "different-declaration"

    with pytest.raises(ValueError, match="campaign_sha256"):
        worker.initialize_study_queue(
            study=study,
            campaign=campaign,
            lock_path=tmp_path / "queue.lock",
        )


def test_campaign_rejects_hydrated_duplicate_model_params_identity(
    tmp_path: Path,
) -> None:
    path = _write_campaign(tmp_path)
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["conditions"].append(
        {
            "name": "params-equivalent",
            "hypothesis": "raw YAML aliases must not create a second condition",
            "overrides": {"model": {"params": {"mobility_embedding_dim": 16}}},
            "allowed_difference_paths": [],
        }
    )
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="same semantic identity"):
        worker.load_campaign(path)


def test_campaign_rejects_declared_budget_drift(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path)
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["conditions"][1]["overrides"] = {"training": {"epochs": 2}}
    campaign["conditions"][1]["allowed_difference_paths"] = ["training.epochs"]
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="equal budget path training.epochs"):
        worker.load_campaign(path)


def test_study_completion_rejects_running_attempt_after_complete(
    tmp_path: Path,
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    complete_trials = [
        SimpleNamespace(
            params=params,
            system_attrs={},
            state=optuna.trial.TrialState.COMPLETE,
        )
        for params in worker.build_trial_queue(campaign)
    ]
    study = SimpleNamespace(trials=complete_trials, user_attrs={})

    assert worker.is_study_complete(study, campaign)

    study.trials.append(
        SimpleNamespace(
            params=worker.build_trial_queue(campaign)[0],
            system_attrs={},
            state=optuna.trial.TrialState.RUNNING,
        )
    )
    assert not worker.is_study_complete(study, campaign)


def test_parallel_workers_do_not_requeue_running_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    journal = tmp_path / "ablation.journal"
    lock_path = tmp_path / "queue.lock"
    studies = [
        optuna.create_study(
            study_name="concurrent-ablation",
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(journal))
            ),
            direction="minimize",
            load_if_exists=True,
        )
        for _ in range(4)
    ]
    worker.initialize_study_queue(
        study=studies[0], campaign=campaign, lock_path=lock_path
    )
    all_claimed = Barrier(4)

    def finish_after_all_claims(trial, **kwargs):  # noqa: ANN001, ANN003, ANN202
        del kwargs
        all_claimed.wait(timeout=10)
        if trial.number == 0:
            time.sleep(0.05)
        else:
            time.sleep(0.1)
        return 1.0

    monkeypatch.setattr(worker, "objective", finish_after_all_claims)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda study: worker.run_queued_trials(
                    study=study,
                    campaign=campaign,
                    lock_path=lock_path,
                    run_root=tmp_path,
                    n_trials=None,
                    timeout_s=None,
                    smoketest=False,
                ),
                studies,
            )
        )

    final_study = optuna.load_study(
        study_name="concurrent-ablation",
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(journal))
        ),
    )
    assert results == [False] * 4
    assert len(final_study.trials) == 4
    assert worker.study_completion_status(final_study, campaign) == "complete"


def test_failed_identity_retries_without_leaving_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    study = optuna.create_study()
    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock_path)
    attempts = 0

    def fail_once(trial, **kwargs):  # noqa: ANN001, ANN003, ANN202
        nonlocal attempts
        del kwargs
        if trial.number == 0 and attempts == 0:
            attempts += 1
            raise RuntimeError("transient failure")
        return 1.0

    monkeypatch.setattr(worker, "objective", fail_once)
    continuation_required = worker.run_queued_trials(
        study=study,
        campaign=campaign,
        lock_path=lock_path,
        run_root=tmp_path,
        n_trials=None,
        timeout_s=None,
        smoketest=False,
    )

    assert not continuation_required
    assert len(study.trials) == 5
    assert worker.study_completion_status(study, campaign) == "complete"


def test_journal_completion_status_rejects_duplicate_complete_trial(
    tmp_path: Path,
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    trials = [
        SimpleNamespace(
            params=params, system_attrs={}, state=optuna.trial.TrialState.COMPLETE
        )
        for params in worker.build_trial_queue(campaign)
    ]
    study = SimpleNamespace(trials=trials, user_attrs={})
    assert worker.study_completion_status(study, campaign) == "complete"

    trials.append(
        SimpleNamespace(
            params=trials[0].params,
            system_attrs={},
            state=optuna.trial.TrialState.COMPLETE,
        )
    )
    with pytest.raises(ValueError, match="duplicate attempts"):
        worker.study_completion_status(study, campaign)

    trials.pop()
    trials[0].state = optuna.trial.TrialState.WAITING
    assert worker.study_completion_status(study, campaign) == "pending"


def test_requeue_running_terminalizes_and_replaces_stale_attempt(
    tmp_path: Path,
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))

    class RecoveryStudy(_FakeStudy):
        def enqueue_trial(self, params):  # noqa: ANN001
            trial = _QueuedTrial(dict(params))
            trial.number = len(self.trials)
            self.trials.append(trial)

        def tell(self, trial_number, *, state):  # noqa: ANN001
            self.trials[trial_number].state = state

    study = RecoveryStudy()
    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock_path)
    stale = study.trials[0]
    stale.state = optuna.trial.TrialState.RUNNING

    worker.initialize_study_queue(
        study=study,
        campaign=campaign,
        lock_path=lock_path,
        requeue_running=True,
    )

    matching = [
        trial
        for trial in study.trials
        if worker._trial_identity(trial.params) == worker._trial_identity(stale.params)
    ]
    assert stale.state == optuna.trial.TrialState.FAIL
    assert [trial.state for trial in matching] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,
    ]


def test_failed_only_identity_is_requeued_on_queue_recovery(tmp_path: Path) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    study = _FakeStudy()
    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock_path)
    failed = study.trials[0]
    failed.state = optuna.trial.TrialState.FAIL

    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock_path)

    matching = [
        trial
        for trial in study.trials
        if worker._trial_identity(trial.params) == worker._trial_identity(failed.params)
    ]
    assert [trial.state for trial in matching] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,
    ]


def test_stale_attempt_terminalization_uses_study_tell() -> None:
    calls: list[tuple[int, optuna.trial.TrialState]] = []

    class Study:
        def tell(self, trial_number, *, state):  # noqa: ANN001
            calls.append((trial_number, state))

    worker._mark_running_attempt_failed(Study(), 17)

    assert calls == [(17, optuna.trial.TrialState.FAIL)]


def test_closeout_finishes_claimed_trial_without_claiming_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))

    class QueueStudy(_FakeStudy):
        def ask(self):  # noqa: ANN201
            trial = next(
                trial
                for trial in self.trials
                if trial.state == optuna.trial.TrialState.WAITING
            )
            trial.number = self.trials.index(trial)
            trial.system_attrs = {"fixed_params": trial.params}
            trial.state = optuna.trial.TrialState.RUNNING
            return trial

        def tell(self, trial, value=None, state=None):  # noqa: ANN001
            del value
            trial.state = state or optuna.trial.TrialState.COMPLETE

    study = QueueStudy()
    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, campaign=campaign, lock_path=lock_path)
    closeout_requested = Event()

    def finish_current_trial(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        closeout_requested.set()
        return 1.0

    monkeypatch.setattr(worker, "objective", finish_current_trial)

    continuation_required = worker.run_queued_trials(
        study=study,
        campaign=campaign,
        lock_path=lock_path,
        run_root=tmp_path,
        n_trials=None,
        timeout_s=None,
        smoketest=False,
        closeout_requested=closeout_requested,
    )

    assert continuation_required
    states = [trial.state for trial in study.trials]
    assert states.count(optuna.trial.TrialState.COMPLETE) == 1
    assert states.count(optuna.trial.TrialState.WAITING) == len(states) - 1


def test_failed_trial_requeue_receives_study_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = worker.load_campaign(_write_campaign(tmp_path))
    study = optuna.create_study()
    worker.initialize_study_queue(
        study=study,
        campaign=campaign,
        lock_path=tmp_path / "queue.lock",
    )
    calls = []

    def fail_trial(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("test failure")

    def record_requeue(
        received_study: optuna.Study,
        identity: tuple[str, int, int | None],
        *,
        attempts_budget: int,
    ) -> None:
        calls.append((received_study, identity, attempts_budget))

    monkeypatch.setattr(worker, "objective", fail_trial)
    monkeypatch.setattr(worker, "_reenqueue_failed_identity", record_requeue)
    worker.run_queued_trials(
        study=study,
        campaign=campaign,
        lock_path=tmp_path / "queue.lock",
        run_root=tmp_path,
        n_trials=1,
        timeout_s=None,
        smoketest=False,
    )

    assert len(calls) == 1
    assert calls[0][0] is study
    assert calls[0][1] == worker.trial_identity(study.trials[0])
    assert calls[0][2] == worker.FAILED_ATTEMPT_BUDGET


def test_smoke_declared_full_validation_is_not_clamped_to_two_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(
        Path("configs/campaigns/ablation_smoke_v1.yaml").read_text()
    )
    document["fixed_overrides"]["training"]["max_batches"] = None
    document["smoke_overrides"] = {"training": {"epochs": 1, "max_batches": None}}
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(document))
    campaign = worker.load_campaign(campaign_path)
    captured: list[str] = []

    def load_config(_path: str, *, overrides: list[str]):
        captured.extend(overrides)
        return SimpleNamespace(output=SimpleNamespace(log_dir=""))

    class Trainer:
        experiment_dir = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self) -> dict[str, object]:
            return {
                "best_val_selection_score": 0.5,
                "best_val_selection_metric": "mean_target_wis",
            }

    monkeypatch.setattr(worker.EpiForecasterConfig, "load", load_config)
    monkeypatch.setattr(worker, "EpiForecasterTrainer", Trainer)
    trial = optuna.create_study().ask()

    worker.objective(trial, campaign=campaign, run_root=tmp_path, smoketest=True)

    assert "training.max_batches=null" in captured
    assert "training.max_batches=2" not in captured


def test_campaign_stage_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        worker.load_campaign(_write_campaign(tmp_path, stage="exploratory"))


def test_ablation_stage_with_folds_requires_full_manifest_coverage(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="require_full_manifest_folds"):
        worker.load_campaign(_write_campaign(tmp_path, folds=[0, 1], stage="ablation"))

    campaign = worker.load_campaign(
        _write_campaign(
            tmp_path, folds=[0, 1], stage="ablation", require_full_manifest_folds=True
        )
    )
    assert campaign.stage == "ablation"


def test_ablation_stage_requires_two_conditions_and_two_seeds(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path, stage="ablation")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["paired_model_seeds"] = [42]
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="two paired seeds"):
        worker.load_campaign(path)
    document["stage"] = "ablation_smoke"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert worker.load_campaign(path).stage == "ablation_smoke"


# --- campaign-declared init checkpoints -------------------------------------


def _init_campaign_extra(
    tmp_path: Path,
    *,
    checkpoints: dict[str, str] | None = None,
    finetune_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "init_checkpoints": (
            {"synth-a": str(tmp_path / "a.pt"), "synth-b": str(tmp_path / "b.pt")}
            if checkpoints is None
            else checkpoints
        ),
        "conditions": [
            {
                "name": "real-only",
                "hypothesis": "reference",
                "overrides": {},
                "allowed_difference_paths": [],
            },
            {
                "name": "ft-a",
                "hypothesis": "fine-tune from a",
                "overrides": finetune_overrides or {},
                "init_checkpoint": "synth-a",
                "allowed_difference_paths": ["training.init_checkpoint_path"],
            },
            {
                "name": "ft-b",
                "hypothesis": "fine-tune from b",
                "overrides": finetune_overrides or {},
                "init_checkpoint": "synth-b",
                "allowed_difference_paths": ["training.init_checkpoint_path"],
            },
        ],
        "baseline_condition": "real-only",
    }


def _init_campaign(tmp_path: Path, **kwargs: object) -> worker.AblationCampaign:
    return worker.load_campaign(
        _write_campaign(tmp_path, extra=_init_campaign_extra(tmp_path, **kwargs))
    )


def _write_run_manifest(
    run_root: Path, campaign: worker.AblationCampaign, payloads: dict[str, bytes]
) -> Path:
    """Write pinned checkpoints and the manifest section prepare would record."""
    recorded: dict[str, object] = {}
    for name, payload in payloads.items():
        relative = Path("inputs/campaign_init") / campaign.campaign_id / f"{name}.pt"
        pinned = run_root / relative
        pinned.parent.mkdir(parents=True, exist_ok=True)
        pinned.write_bytes(payload)
        recorded[name] = {
            "configured_checkpoint": str(campaign.init_checkpoints[name]),
            "pinned_checkpoint_relative": str(relative),
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = run_root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"campaign_init_inputs": {campaign.campaign_id: recorded}}),
        encoding="utf-8",
    )
    return manifest


def test_init_campaign_loads_and_requires_the_path_in_declared_drift(
    tmp_path: Path,
) -> None:
    campaign = _init_campaign(tmp_path)

    assert set(campaign.init_checkpoints) == {"synth-a", "synth-b"}
    assert campaign.conditions_by_name["ft-a"].init_checkpoint == "synth-a"
    assert campaign.conditions_by_name["real-only"].init_checkpoint is None

    extra = _init_campaign_extra(tmp_path)
    extra["conditions"][1]["allowed_difference_paths"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="undeclared config drift"):
        worker.load_campaign(_write_campaign(tmp_path, extra=extra))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda extra: extra["conditions"][1].update(init_checkpoint="missing"),
            "undeclared init_checkpoint",
        ),
        (
            lambda extra: extra["init_checkpoints"].update({"unused": "/gpfs/u.pt"}),
            "used by no condition",
        ),
        (
            lambda extra: extra["init_checkpoints"].update({"synth-a": "rel/a.pt"}),
            "absolute path",
        ),
        (
            lambda extra: extra["init_checkpoints"].update({"bad name": "/gpfs/u.pt"}),
            "must match",
        ),
        (
            lambda extra: extra.update(init_checkpoints=["/gpfs/a.pt"]),
            "must be a mapping",
        ),
        (
            lambda extra: extra["conditions"][1]["overrides"].update(
                {"training": {"init_checkpoint_path": "/gpfs/raw.pt"}}
            ),
            "must not set training.init_checkpoint_path",
        ),
    ],
)
def test_init_campaign_rejects_invalid_declarations(
    tmp_path: Path,
    mutate,
    message: str,  # noqa: ANN001
) -> None:
    extra = _init_campaign_extra(tmp_path)
    mutate(extra)

    with pytest.raises(ValueError, match=message):
        worker.load_campaign(_write_campaign(tmp_path, extra=extra))


def test_init_checkpoint_path_in_fixed_overrides_is_rejected(tmp_path: Path) -> None:
    extra = _init_campaign_extra(tmp_path)
    extra["fixed_overrides"] = {
        "training": {"epochs": 1, "init_checkpoint_path": "/gpfs/raw.pt"}
    }

    with pytest.raises(ValueError, match="fixed_overrides must not set"):
        worker.load_campaign(_write_campaign(tmp_path, extra=extra))


def test_campaign_hash_ignores_checkpoint_location(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first = _init_campaign(tmp_path / "one")
    second = _init_campaign(
        tmp_path / "two",
        checkpoints={"synth-a": "/elsewhere/a.pt", "synth-b": "/elsewhere/b.pt"},
    )

    assert first.sha256 == second.sha256


def test_pinned_checkpoints_are_verified_against_the_manifest(tmp_path: Path) -> None:
    campaign = _init_campaign(tmp_path)
    run_root = tmp_path / "run"
    manifest = _write_run_manifest(
        run_root, campaign, {"synth-a": b"checkpoint a", "synth-b": b"checkpoint b"}
    )

    pins = worker.load_pinned_init_checkpoints(manifest, campaign)

    assert pins["synth-a"].path == (
        run_root / "inputs/campaign_init" / campaign.campaign_id / "synth-a.pt"
    )
    assert pins["synth-a"].sha256 == hashlib.sha256(b"checkpoint a").hexdigest()

    pins["synth-b"].path.chmod(0o644)
    pins["synth-b"].path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed validation"):
        worker.load_pinned_init_checkpoints(manifest, campaign)


def test_pinned_checkpoints_must_match_declared_names(tmp_path: Path) -> None:
    campaign = _init_campaign(tmp_path)
    manifest = _write_run_manifest(
        tmp_path / "run", campaign, {"synth-a": b"checkpoint a"}
    )

    with pytest.raises(ValueError, match="run manifest pins init checkpoints"):
        worker.load_pinned_init_checkpoints(manifest, campaign)


def _capture_objective(
    monkeypatch: pytest.MonkeyPatch,
    campaign: worker.AblationCampaign,
    tmp_path: Path,
    condition: str,
) -> tuple[list[str], dict[str, object], list[Path]]:
    captured: list[str] = []
    attrs: dict[str, object] = {}
    experiment_dirs: list[Path] = []

    class Trial:
        number = 0

        def suggest_categorical(self, name, choices):  # noqa: ANN001
            return {"condition": condition, "seed": 42}[name]

        def set_user_attr(self, key, value):  # noqa: ANN001
            attrs[key] = value

    class Trainer:
        experiment_dir = tmp_path / "experiment"

        def __init__(self, config, model_id):  # noqa: ANN001
            self.experiment_dir.mkdir(exist_ok=True)
            experiment_dirs.append(self.experiment_dir)

        def run(self):
            return {
                "best_val_selection_score": 1.0,
                "best_val_selection_metric": "mae_median",
            }

    def load_config(path, overrides):  # noqa: ANN001
        captured.extend(overrides)
        return SimpleNamespace(output=SimpleNamespace(log_dir=""))

    monkeypatch.setattr(worker.EpiForecasterConfig, "load", staticmethod(load_config))
    monkeypatch.setattr(worker, "EpiForecasterTrainer", Trainer)
    worker.objective(Trial(), campaign=campaign, run_root=tmp_path)
    return captured, attrs, experiment_dirs


def test_objective_passes_the_pinned_path_and_records_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _init_campaign(tmp_path)
    run_root = tmp_path / "run"
    manifest = _write_run_manifest(
        run_root, campaign, {"synth-a": b"checkpoint a", "synth-b": b"checkpoint b"}
    )
    bound = worker.bind_pinned_init_checkpoints(campaign, manifest)
    pinned = run_root / "inputs/campaign_init" / campaign.campaign_id / "synth-a.pt"

    overrides, attrs, experiment_dirs = _capture_objective(
        monkeypatch, bound, tmp_path, "ft-a"
    )

    assert f"training.init_checkpoint_path={json.dumps(str(pinned))}" in overrides
    assert str(tmp_path / "a.pt") not in " ".join(overrides)
    digest = hashlib.sha256(b"checkpoint a").hexdigest()
    assert attrs["campaign_init_checkpoint_sha256"] == digest
    identity = json.loads(
        (experiment_dirs[0] / "campaign_identity.json").read_text(encoding="utf-8")
    )
    assert identity["init_checkpoint"] == "synth-a"
    assert identity["init_checkpoint_sha256"] == digest

    baseline_overrides, _, _ = _capture_objective(
        monkeypatch, bound, tmp_path, "real-only"
    )
    assert not any(
        o.startswith("training.init_checkpoint_path") for o in baseline_overrides
    )


def test_objective_rejects_a_tampered_pinned_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _init_campaign(tmp_path)
    manifest = _write_run_manifest(
        tmp_path / "run",
        campaign,
        {"synth-a": b"checkpoint a", "synth-b": b"checkpoint b"},
    )
    bound = worker.bind_pinned_init_checkpoints(campaign, manifest)
    bound.pinned_init_checkpoints["synth-a"].path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="failed validation"):
        _capture_objective(monkeypatch, bound, tmp_path, "ft-a")


def test_objective_requires_a_bound_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="not bound to run-pinned copies"):
        _capture_objective(monkeypatch, _init_campaign(tmp_path), tmp_path, "ft-a")


def test_hydrated_identity_follows_checkpoint_hash_not_path(tmp_path: Path) -> None:
    campaign = _init_campaign(tmp_path)
    manifest_one = _write_run_manifest(
        tmp_path / "run1", campaign, {"synth-a": b"one", "synth-b": b"b"}
    )
    manifest_same = _write_run_manifest(
        tmp_path / "run2", campaign, {"synth-a": b"one", "synth-b": b"b"}
    )
    manifest_other = _write_run_manifest(
        tmp_path / "run3", campaign, {"synth-a": b"two", "synth-b": b"b"}
    )

    def digest(manifest: Path, condition: str) -> str:
        return worker.hydrated_training_config_sha256(
            worker.bind_pinned_init_checkpoints(campaign, manifest),
            condition=condition,
            fold=0,
            seed=42,
        )

    # Different run roots give different pinned paths but the same identity.
    assert digest(manifest_one, "ft-a") == digest(manifest_same, "ft-a")
    assert digest(manifest_one, "ft-a") != digest(manifest_other, "ft-a")
    assert digest(manifest_one, "ft-a") != digest(manifest_one, "ft-b")
    assert digest(manifest_one, "real-only") == digest(manifest_other, "real-only")


def test_reuse_acceptance_distinguishes_checkpoint_hashes(tmp_path: Path) -> None:
    campaign = _init_campaign(tmp_path)
    manifest = _write_run_manifest(
        tmp_path / "run", campaign, {"synth-a": b"one", "synth-b": b"b"}
    )
    bound = worker.bind_pinned_init_checkpoints(campaign, manifest)
    hydrated = worker.hydrated_training_config(bound, condition="ft-a", fold=0, seed=42)
    execution = worker.ExecutionIdentity(
        source_bundle_sha256="a" * 64,
        container_digest="c" * 64,
        dvc_md5_by_output={
            Path(str(hydrated["data"]["dataset_path"])).as_posix(): "d" * 32,
            Path(str(hydrated["data"]["region2vec_path"])).as_posix(): "e" * 32,
        },
    )
    sha = hashlib.sha256(b"one").hexdigest()
    record = {
        "condition": "ft-a",
        "seed": 42,
        "fold": 0,
        "status": "reusable",
        "source_bundle_sha256": "a" * 64,
        "container_digest": f"sha256:{'c' * 64}",
        "dataset_dvc_md5": "d" * 32,
        "region2vec_dvc_md5": "e" * 32,
        "hydrated_training_config_sha256": worker.hydrated_training_config_sha256(
            bound, condition="ft-a", fold=0, seed=42
        ),
        "init_checkpoint_sha256": sha,
    }

    assert (
        worker.reuse_rejection_reasons(record, campaign=bound, execution=execution)
        == []
    )
    stale = {**record, "init_checkpoint_sha256": hashlib.sha256(b"two").hexdigest()}
    assert worker.reuse_rejection_reasons(
        stale, campaign=bound, execution=execution
    ) == ["init_checkpoint_sha256_mismatch"]
    missing = {
        key: value for key, value in record.items() if key != "init_checkpoint_sha256"
    }
    assert "init_checkpoint_sha256_mismatch" in worker.reuse_rejection_reasons(
        missing, campaign=bound, execution=execution
    )

    other_manifest = _write_run_manifest(
        tmp_path / "run-other", campaign, {"synth-a": b"two", "synth-b": b"b"}
    )
    rebound = worker.bind_pinned_init_checkpoints(campaign, other_manifest)
    reasons = worker.reuse_rejection_reasons(
        record, campaign=rebound, execution=execution
    )
    assert "hydrated_training_config_mismatch" in reasons
    assert "init_checkpoint_sha256_mismatch" in reasons


def test_study_queue_records_and_checks_init_hashes(tmp_path: Path) -> None:
    campaign = _init_campaign(tmp_path)
    manifest = _write_run_manifest(
        tmp_path / "run", campaign, {"synth-a": b"one", "synth-b": b"b"}
    )
    bound = worker.bind_pinned_init_checkpoints(campaign, manifest)
    study = _FakeStudy()
    lock = tmp_path / "queue.lock"

    worker.initialize_study_queue(study=study, campaign=bound, lock_path=lock)  # type: ignore[arg-type]

    assert study.user_attrs[worker.INIT_CHECKPOINT_SHA256_ATTR] == {
        "synth-a": hashlib.sha256(b"one").hexdigest(),
        "synth-b": hashlib.sha256(b"b").hexdigest(),
    }
    changed = _write_run_manifest(
        tmp_path / "run-changed", campaign, {"synth-a": b"two", "synth-b": b"b"}
    )
    with pytest.raises(ValueError, match="init checkpoint SHA-256 identities"):
        worker.initialize_study_queue(
            study=study,  # type: ignore[arg-type]
            campaign=worker.bind_pinned_init_checkpoints(campaign, changed),
            lock_path=lock,
        )
    with pytest.raises(ValueError, match="not bound to run-pinned copies"):
        worker.initialize_study_queue(
            study=_FakeStudy(),  # type: ignore[arg-type]
            campaign=campaign,
            lock_path=lock,
        )
