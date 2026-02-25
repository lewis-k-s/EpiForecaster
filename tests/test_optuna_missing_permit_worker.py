from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import optuna
import pytest

from scripts.optuna_epiforecaster_missing_permit_worker import (
    _compute_val_coverage_ratio,
    create_missing_permit_study,
    load_frozen_non_permit_overrides,
    objective,
    suggest_missing_permit_params,
)


class _StubTrial:
    def __init__(self, int_values: dict[str, int] | None = None):
        self._int_values = int_values or {}
        self.suggest_calls: list[tuple[str, int, int]] = []
        self.user_attrs: dict[str, Any] = {}
        self.number = 0

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.suggest_calls.append((name, low, high))
        return self._int_values.get(name, low)

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


def _journal_storage(path):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(path)))


def test_missing_permit_bounds_and_mapping() -> None:
    trial = _StubTrial(
        int_values={
            "data.missing_permit_input_daily": 5,
            "data.missing_permit_horizon_daily": 3,
            "data.missing_permit_input_weekly": 50,
            "data.missing_permit_horizon_weekly": 20,
        }
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(input_window_length=60, forecast_horizon=28)
    )
    overrides = suggest_missing_permit_params(trial=trial, base_cfg=cfg)

    assert ("data.missing_permit_input_daily", 0, 30) in trial.suggest_calls
    assert ("data.missing_permit_horizon_daily", 0, 14) in trial.suggest_calls

    input_weekly = next(
        call
        for call in trial.suggest_calls
        if call[0] == "data.missing_permit_input_weekly"
    )
    assert input_weekly[1] == 51  # 60 - ceil(60/7)
    assert input_weekly[2] == 59

    horizon_weekly = next(
        call
        for call in trial.suggest_calls
        if call[0] == "data.missing_permit_horizon_weekly"
    )
    assert horizon_weekly[1] == 24  # 28 - ceil(28/7)
    assert horizon_weekly[2] == 27

    assert overrides["data.missing_permit.input.cases"] == 5
    assert overrides["data.missing_permit.input.deaths"] == 5
    assert overrides["data.missing_permit.horizon.cases"] == 3
    assert overrides["data.missing_permit.horizon.deaths"] == 3
    assert overrides["data.missing_permit.input.hospitalizations"] == 50
    assert overrides["data.missing_permit.input.biomarkers_joint"] == 50
    assert overrides["data.missing_permit.horizon.hospitalizations"] == 20
    assert overrides["data.missing_permit.horizon.biomarkers_joint"] == 20


def test_load_frozen_overrides_from_journal(tmp_path) -> None:
    journal = tmp_path / "freeze_source.journal"
    storage = _journal_storage(journal)
    study = optuna.create_study(
        study_name="full_hpo",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    trial = optuna.create_trial(
        params={"x": 0.1},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        value=0.2,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={
            "overrides": {
                "training.learning_rate": 1e-3,
                "model.gnn_depth": 3,
                "data.missing_permit.input.cases": 1,
            }
        },
    )
    study.add_trial(trial)

    frozen, source = load_frozen_non_permit_overrides(
        freeze_journal_file=journal,
        freeze_study_name="full_hpo",
        freeze_trial_number=None,
    )

    assert frozen == {"training.learning_rate": 1e-3, "model.gnn_depth": 3}
    assert source["freeze_trial_number"] == 0
    assert source["freeze_study_name"] == "full_hpo"


def test_load_frozen_overrides_fails_without_overrides_attr(tmp_path) -> None:
    journal = tmp_path / "freeze_source_missing.journal"
    storage = _journal_storage(journal)
    study = optuna.create_study(
        study_name="full_hpo_missing",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    trial = optuna.create_trial(
        params={"x": 0.1},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        value=0.1,
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)

    with pytest.raises(ValueError, match="missing user_attrs\\['overrides'\\]"):
        load_frozen_non_permit_overrides(
            freeze_journal_file=journal,
            freeze_study_name="full_hpo_missing",
            freeze_trial_number=None,
        )


def test_multi_objective_study_directions(tmp_path) -> None:
    journal = tmp_path / "pareto.journal"
    storage = _journal_storage(journal)
    study = create_missing_permit_study(
        study_name="missing_permit_pareto",
        storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=123),
        pruner=optuna.pruners.NopPruner(),
    )

    assert len(study.directions) == 2
    assert study.directions[0].name == "MINIMIZE"
    assert study.directions[1].name == "MAXIMIZE"


def test_compute_val_coverage_ratio_guard() -> None:
    assert _compute_val_coverage_ratio(val_samples=40, val_samples_reference=100) == 0.4
    assert _compute_val_coverage_ratio(val_samples=10, val_samples_reference=0) == 0.0


def test_objective_returns_multiobjective_tuple(monkeypatch, tmp_path) -> None:
    import scripts.optuna_epiforecaster_missing_permit_worker as module

    trial = _StubTrial()

    class _DummyCfg:
        def __init__(self):
            self.output = SimpleNamespace(experiment_name="", log_dir=str(tmp_path))

    def _fake_load(_path: str, overrides: list[str] | None = None):
        _ = overrides
        return _DummyCfg()

    class _FakeTrainer:
        def __init__(self, cfg, trial=None, pruning_start_epoch=10):
            _ = (cfg, trial, pruning_start_epoch)
            self.val_dataset = [1] * 8
            self.model_id = "fake-model-id"

        def run(self):
            return {"best_val_loss": 0.5}

    monkeypatch.setattr(module.EpiForecasterConfig, "load", staticmethod(_fake_load))
    monkeypatch.setattr(module, "EpiForecasterTrainer", _FakeTrainer)

    values = objective(
        trial=trial,
        base_config_path=tmp_path / "dummy.yaml",
        study_name="pareto_study",
        run_root=tmp_path / "runs",
        fixed_epochs=1,
        fixed_max_batches=1,
        seed=None,
        pruning_start_epoch=1,
        cli_overrides=[],
        frozen_non_permit_overrides={"training.learning_rate": 1e-3},
        frozen_overrides_source={"freeze_trial_number": 0},
        base_cfg_for_bounds=SimpleNamespace(
            model=SimpleNamespace(input_window_length=60, forecast_horizon=28)
        ),
        val_samples_reference=10,
    )

    assert isinstance(values, tuple)
    assert len(values) == 2
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.8)
    assert trial.user_attrs["val_samples"] == 8
    assert trial.user_attrs["val_samples_reference"] == 10
