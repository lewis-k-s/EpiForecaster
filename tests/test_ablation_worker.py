from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
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
) -> Path:
    base = Path("configs/train_epifor_legacy_municipal_smoke.yaml").resolve()
    campaign: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "campaign-test",
        "stage": "ablation",
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
        },
        "provenance_requirements": {
            "require_campaign_sha256": True,
            "require_output_identity": True,
        },
    }
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
    study = SimpleNamespace(trials=complete_trials)

    assert worker.is_study_complete(study, campaign)

    study.trials.append(
        SimpleNamespace(
            params=worker.build_trial_queue(campaign)[0],
            system_attrs={},
            state=optuna.trial.TrialState.RUNNING,
        )
    )
    assert not worker.is_study_complete(study, campaign)


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
