from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import optuna
import pytest
from click.testing import CliRunner
from optuna.distributions import CategoricalDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import create_trial

from scripts.hpo import hpsearch_context_length_worker as worker


CONTEXT_LENGTH_CHOICES = [14, 28, 60]
SEED_CHOICES = [42, 43]
PARAM_DISTS = {
    "context_length": CategoricalDistribution(CONTEXT_LENGTH_CHOICES),
    "seed": CategoricalDistribution(SEED_CHOICES),
}


@dataclass
class _QueuedTrial:
    params: dict[str, int]
    state: optuna.trial.TrialState = optuna.trial.TrialState.WAITING


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_QueuedTrial] = []
        self.user_attrs: dict[str, object] = {}
        self.optimize_kwargs: dict[str, object] | None = None

    def enqueue_trial(self, params: dict[str, int]) -> None:
        self.trials.append(_QueuedTrial(params=dict(params)))

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def optimize(self, func, **kwargs) -> None:  # noqa: ANN001
        self.optimize_kwargs = kwargs


def _add_terminal_trial(
    study: optuna.Study,
    *,
    seed: int,
    context_length: int,
    state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE,
) -> None:
    kwargs = {
        "params": {"context_length": context_length, "seed": seed},
        "distributions": PARAM_DISTS,
        "state": state,
    }
    if state == optuna.trial.TrialState.COMPLETE:
        kwargs["value"] = 1.0
    study.add_trial(create_trial(**kwargs))


def test_parse_int_list_accepts_spaces_and_commas() -> None:
    assert worker.parse_int_list("14 28,60", name="values") == [14, 28, 60]


def test_parse_int_list_rejects_empty_values() -> None:
    with pytest.raises(Exception, match="at least one value"):
        worker.parse_int_list(" ", name="values")


def test_build_trial_queue_is_seed_major() -> None:
    queue = worker.build_trial_queue(
        context_lengths=[14, 28, 60],
        seeds=[42, 43],
    )

    assert queue == [
        {"seed": 42, "context_length": 14},
        {"seed": 42, "context_length": 28},
        {"seed": 42, "context_length": 60},
        {"seed": 43, "context_length": 14},
        {"seed": 43, "context_length": 28},
        {"seed": 43, "context_length": 60},
    ]


def test_initialize_study_queue_is_idempotent(tmp_path: Path) -> None:
    journal_file = tmp_path / "context-length.journal"
    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(
        study_name="context-length-queue",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )

    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(
        study=study,
        context_lengths=[14, 28, 60],
        seeds=[42, 43],
        lock_path=lock_path,
    )
    worker.initialize_study_queue(
        study=study,
        context_lengths=[14, 28, 60],
        seeds=[42, 43],
        lock_path=lock_path,
    )

    identities = [
        identity
        for trial in study.trials
        if (identity := worker.trial_identity(trial)) is not None
    ]
    assert len(identities) == 6
    assert len(set(identities)) == len(identities)
    assert study.user_attrs[worker.QUEUE_INITIALIZED_ATTR] is True


def test_is_study_complete_requires_full_matrix() -> None:
    study = optuna.create_study(direction="minimize")
    _add_terminal_trial(study, seed=42, context_length=14)
    _add_terminal_trial(study, seed=42, context_length=28)
    _add_terminal_trial(study, seed=43, context_length=14)

    assert (
        worker.is_study_complete(
            study=study,
            context_lengths=[14, 28],
            seeds=[42, 43],
        )
        is False
    )

    _add_terminal_trial(study, seed=43, context_length=28)
    assert (
        worker.is_study_complete(
            study=study,
            context_lengths=[14, 28],
            seeds=[42, 43],
        )
        is True
    )


def test_runtime_overrides_keep_context_length_study_specific() -> None:
    overrides = worker._runtime_overrides(
        context_length=28,
        seed=43,
        campaign_id="camp-a",
        epochs=20,
        max_batches=None,
        cli_overrides=["data.log_scale=false"],
        smoketest=False,
    )

    assert overrides[0] == "data.log_scale=false"
    assert "model.input_window_length=28" in overrides
    assert "training.seed=43" in overrides
    assert "training.epochs=20" in overrides
    assert "output.write_granular_eval=true" in overrides
    assert any(item.startswith("output.experiment_name=") for item in overrides)


def test_main_passes_timeout_and_catch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_study = _FakeStudy()

    monkeypatch.setattr(worker, "JournalStorage", lambda backend: object())
    monkeypatch.setattr(worker, "JournalFileBackend", lambda path: object())
    monkeypatch.setattr(worker.optuna, "create_study", lambda **kwargs: fake_study)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  seed: 42\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        worker.main,
        [
            "--config",
            str(config_path),
            "--study-name",
            "study-a",
            "--journal-file",
            str(tmp_path / "study.journal"),
            "--campaign-id",
            "camp-a",
            "--context-lengths",
            "14 28",
            "--seeds",
            "42 43",
            "--timeout-s",
            "123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_study.optimize_kwargs is not None
    assert fake_study.optimize_kwargs["timeout"] == 123
    assert fake_study.optimize_kwargs["n_trials"] is None
    assert fake_study.optimize_kwargs["catch"] == (Exception,)
