from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import optuna
import pytest
import yaml

from scripts.hpo import ablation_worker as worker
from tests.config_fixtures import config_fixture


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
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(config_fixture().to_dict()))
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
                "overrides": {"native.mobility_enabled": False},
                "allowed_difference_paths": (
                    ["native.mobility_enabled"]
                    if allowed_paths is None
                    else allowed_paths
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
        cross_validation: dict[str, object] = {
            "num_folds": max(3, max(folds) + 1),
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
    assert campaign.crossval_num_folds == 3

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


def test_native_decoder_mobility_campaign_end_to_end(tmp_path: Path) -> None:
    """Run the real four-arm queue through selected-checkpoint test reduction."""
    import json

    import pandas as pd
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    from scripts.analysis.analyze_ablations import (
        aggregate_ablation_metrics,
        collect_ablation_runs,
        compute_seed_matched_baseline_deltas,
        validate_campaign_outputs,
        validate_campaign_trials,
    )
    from scripts.workflow.run_support import finalize_ablation, sha256_file
    from tests.native_fixtures import pipeline_config

    declaration = Path("configs/campaigns/native_decoder_mobility_v1.yaml")
    production = worker.load_campaign(declaration)
    assert len(worker.build_trial_queue(production)) == 4
    assert production.fixed_overrides["training"]["max_batches"] is None
    assert production.fixed_overrides["training"]["epochs"] == 10
    data_root = tmp_path / "inputs"
    data_root.mkdir()
    config = pipeline_config(data_root, strict=True, mobility=True)
    base = tmp_path / "acceptance.yaml"
    base.write_text(yaml.safe_dump(config.to_dict()))
    raw = yaml.safe_load(declaration.read_text())
    raw["base_config"] = str(base)
    raw["cross_validation"]["num_folds"] = 3
    raw["fixed_overrides"]["native"]["wastewater_filename"] = "obs_wastewater.zarr"
    raw["fixed_overrides"]["model"] = {
        "input_window_length": 28,
        "forecast_horizon": 14,
    }
    raw["fixed_overrides"]["training"].update(epochs=1, max_batches=2)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw))
    campaign = worker.load_campaign(path)
    journal = tmp_path / "study.log"
    study = optuna.create_study(
        study_name="native-acceptance",
        direction="minimize",
        storage=JournalStorage(JournalFileBackend(str(journal))),
    )
    worker.initialize_study_queue(
        study=study,
        campaign=campaign,
        lock_path=tmp_path / "queue.lock",
        smoketest=True,
    )
    study.optimize(
        lambda trial: worker.objective(
            trial,
            campaign=campaign,
            run_root=tmp_path / "outputs/training",
            smoketest=True,
        ),
        n_trials=4,
    )
    assert worker.is_study_complete(study, campaign, smoketest=True)
    runs, _ = collect_ablation_runs(
        tmp_path / "outputs/training" / campaign.campaign_id,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id=campaign.campaign_id,
    )
    validate_campaign_trials(
        campaign, journal_path=journal, study_name=study.study_name, smoketest=True
    )
    validate_campaign_outputs(campaign, runs, smoketest=True)
    assert len(runs) == 4
    assignments = []
    for condition, outputs in runs.items():
        assert len(outputs) == 1 and outputs[0].fold == 0
        run = outputs[0].run_dir
        saved = json.loads((run / "config.json").read_text())
        assert saved["native"]["include_vaccination"] is True
        assert saved["native"]["mobility_enabled"] == condition.endswith("_on")
        assert saved["native"]["decoder"] == condition.split("_")[0]
        summary = json.loads((run / "training_dynamics_manifest.json").read_text())[
            "summary"
        ]
        test = summary["test_epoch_trace"]
        assert len(test) == 1 and test[0]["epoch"] == summary["selected_epoch"]
        assert "selection_score" not in test[0]
        assignments.append(
            pd.read_parquet(run / "inputs/native_source_sets/assignments.parquet")
        )
    for assignment in assignments[1:]:
        pd.testing.assert_frame_equal(assignments[0], assignment)
    aggregated = aggregate_ablation_metrics(runs, assert_consistent_config=True)
    assert set(aggregated.model) == set(runs)
    joint = aggregated[aggregated.target == "equal_head_mean"]
    assert len(joint) == 4 and joint.wis_mean.notna().all()
    paired, _ = compute_seed_matched_baseline_deltas(
        runs, baseline_name=campaign.baseline_condition
    )
    assert set(paired.model) == set(runs) - {campaign.baseline_condition}
    assert "wis_delta_pct" in paired.columns
    workflow_config = tmp_path / "workflow.yaml"
    workflow_config.write_text(
        yaml.safe_dump(
            {
                "ablation": {
                    "campaign": str(path),
                    "study_name": study.study_name,
                    "smoketest": True,
                }
            }
        )
    )
    marker = finalize_ablation(workflow_config, tmp_path, journal)
    assert json.loads(marker.read_text())["trial_count"] == 4
    output = tmp_path / "outputs/ablation_analysis"
    reduced = pd.read_csv(output / "ablation_metrics_aggregated.csv")
    pd.testing.assert_frame_equal(reduced, aggregated)
    companion = pd.read_csv(output / "ablation_censored_wastewater.csv")
    assert len(companion) == 4
    for _, row in companion.iterrows():
        source = runs[row.model][0].run_dir / "training_dynamics_manifest.json"
        test = json.loads(source.read_text())["summary"]["test_epoch_trace"][0]
        for key in ("censored_count_ww", "observed_count_ww"):
            assert row[key] == test[key]
        assert row.selected_step == test["step"]
        if test["censored_count_ww"]:
            assert row.censored_nll_ww == pytest.approx(test["censored_nll_ww"])
    manifest = json.loads((output / "ablation_manifest.json").read_text())
    assert manifest["campaign_sha256"] == campaign.sha256
    for artifact in manifest["artifacts"]:
        assert sha256_file(output / artifact["path"]) == artifact["sha256"]


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
                "best_val_selection_metric": "wis",
                "selection_score": 1.0,
                "selection_metric_definition": {
                    "metric": "wis",
                    "aggregation": "equal_head_mean",
                    "direction": "minimize",
                },
            }

    def load_config(path, overrides):  # noqa: ANN001
        captured["overrides"] = overrides
        return SimpleNamespace(
            output=SimpleNamespace(log_dir=""), native=SimpleNamespace(fold_manifest="")
        )

    monkeypatch.setattr(worker.EpiForecasterConfig, "load", staticmethod(load_config))
    monkeypatch.setattr(worker, "NativeTrainer", Trainer)
    monkeypatch.setattr(worker, "build_for_config", lambda *args: None)

    worker.objective(Trial(), campaign=campaign, run_root=tmp_path, smoketest=True)

    assert "native.fold_index=1" in captured["overrides"]
    assert "native.num_folds=3" in captured["overrides"]
    assert "native.fold_index=1" in captured["overrides"]
    assert "training.compile=false" in captured["overrides"]
    assert "output.wandb_mode=disabled" in captured["overrides"]


def test_campaign_rejects_fold_outside_manifest_bounds(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path, folds=[0, 1])
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["cross_validation"]["fold_indices"] = [3]
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="exceed num_folds"):
        worker.load_campaign(path)


def test_campaign_rejects_undeclared_config_drift(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path, allowed_paths=[])

    with pytest.raises(ValueError, match="undeclared config drift"):
        worker.load_campaign(path)


def test_campaign_rejects_invalid_native_fold_count(tmp_path: Path) -> None:
    path = _write_campaign(tmp_path)
    campaign = yaml.safe_load(path.read_text())
    campaign["cross_validation"] = {"num_folds": 2, "fold_indices": [0, 1]}
    path.write_text(yaml.safe_dump(campaign))
    with pytest.raises(ValueError, match="at least three"):
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
