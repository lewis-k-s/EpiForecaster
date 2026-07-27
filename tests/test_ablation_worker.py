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

from scripts.hpo import ablation_worker as worker


SEED_CHOICES = [42, 43]
PARAM_DISTS = {
    "ablation": CategoricalDistribution(list(worker.ABLATIONS.keys())),
    "seed": CategoricalDistribution(SEED_CHOICES),
}


@dataclass
class _QueuedTrial:
    params: dict[str, int | str]
    state: optuna.trial.TrialState = optuna.trial.TrialState.WAITING


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_QueuedTrial] = []
        self.user_attrs: dict[str, object] = {}
        self.optimize_kwargs: dict[str, object] | None = None

    def enqueue_trial(self, params: dict[str, int | str]) -> None:
        self.trials.append(_QueuedTrial(params=dict(params)))

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def optimize(self, func, **kwargs) -> None:  # noqa: ANN001
        self.optimize_kwargs = kwargs


def _add_terminal_trial(
    study: optuna.Study,
    *,
    seed: int,
    ablation: str,
    state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE,
) -> None:
    kwargs = {
        "params": {"ablation": ablation, "seed": seed},
        "distributions": PARAM_DISTS,
        "state": state,
    }
    if state == optuna.trial.TrialState.COMPLETE:
        kwargs["value"] = 1.0
    study.add_trial(create_trial(**kwargs))


def test_ablation_names_are_valid() -> None:
    assert "regions:off" in worker.ABLATIONS
    assert worker.ABLATIONS["regions:off"] == "model.type.regions=false"
    assert "kernel:all:mlp" in worker.ABLATIONS
    assert (
        "model.observation_heads.kernel_mlp_cases=true"
        in worker.ABLATIONS["kernel:all:mlp"]
    )
    assert worker.ABLATIONS["loss:crps"] == (
        "training.loss.joint.observation_loss=crps"
    )
    assert worker.ABLATIONS["gradnorm:crps"] == (
        "training.loss.joint.observation_loss=crps "
        "training.loss.joint.adaptive_scheme=gradnorm"
    )
    assert worker.ABLATIONS["mobility:spatial_queen"] == (
        "model.graph_adjacency_source=spatial_queen"
    )
    assert worker.ABLATIONS["context:spatial_queen"] == (
        "model.type.regions=false model.graph_adjacency_source=spatial_queen"
    )
    assert "mobility:spatial_knn" not in worker.ABLATIONS
    assert "context:spatial_knn" not in worker.ABLATIONS
    assert "sir:off" not in worker.ABLATIONS
    assert "gradnorm:on" not in worker.ABLATIONS
    assert "model.type.regions=false" not in worker.ABLATIONS


def test_build_trial_queue_is_seed_major() -> None:
    queue = worker.build_trial_queue([42, 43])
    expected_block_size = len(worker.ABLATIONS)

    assert len(queue) == expected_block_size * 2
    assert queue[0] == {"seed": 42, "ablation": "baseline"}
    assert queue[expected_block_size] == {"seed": 43, "ablation": "baseline"}
    assert {entry["ablation"] for entry in queue[:expected_block_size]} == set(
        worker.ABLATIONS
    )
    assert all(entry["seed"] == 42 for entry in queue[:expected_block_size])
    assert all(entry["seed"] == 43 for entry in queue[expected_block_size:])


def test_resolve_ablation_plan_supports_focused_subset() -> None:
    plan = worker.resolve_ablation_plan("baseline,mobility:off context:off")

    assert list(plan) == ["baseline", "mobility:off", "context:off"]
    assert plan["mobility:off"] == "model.type.mobility=false"


def test_build_trial_queue_accepts_ablation_subset() -> None:
    plan = worker.resolve_ablation_plan("baseline mobility:off")
    queue = worker.build_trial_queue([42, 43], plan)

    assert queue == [
        {"seed": 42, "ablation": "baseline"},
        {"seed": 42, "ablation": "mobility:off"},
        {"seed": 43, "ablation": "baseline"},
        {"seed": 43, "ablation": "mobility:off"},
    ]


def test_initialize_study_queue_is_idempotent(tmp_path: Path) -> None:
    journal_file = tmp_path / "ablation.journal"
    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(
        study_name="ablation-queue",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )

    lock_path = tmp_path / "queue.lock"
    worker.initialize_study_queue(study=study, seeds=[42, 43], lock_path=lock_path)
    worker.initialize_study_queue(study=study, seeds=[42, 43], lock_path=lock_path)

    identities = [
        identity
        for trial in study.trials
        if (identity := worker.trial_identity(trial)) is not None
    ]
    assert len(identities) == len(worker.ABLATIONS) * 2
    assert len(set(identities)) == len(identities)
    assert study.user_attrs[worker.QUEUE_INITIALIZED_ATTR] is True


def test_initialize_study_queue_can_requeue_running_trial(tmp_path: Path) -> None:
    journal_file = tmp_path / "ablation.journal"
    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(
        study_name="ablation-stale-running",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    plan = worker.resolve_ablation_plan("baseline")
    _add_terminal_trial(
        study,
        seed=42,
        ablation="baseline",
        state=optuna.trial.TrialState.RUNNING,
    )

    worker.initialize_study_queue(
        study=study,
        seeds=[42],
        ablations=plan,
        lock_path=tmp_path / "queue.lock",
        requeue_running=True,
    )

    identities = [
        identity
        for trial in study.trials
        if (identity := worker.trial_identity(trial)) is not None
    ]
    assert identities.count((42, "baseline")) == 2


def test_is_study_complete_requires_full_matrix() -> None:
    study = optuna.create_study(direction="minimize")
    for ablation in worker.ABLATIONS:
        _add_terminal_trial(study, seed=42, ablation=ablation)
    _add_terminal_trial(study, seed=43, ablation="baseline")

    assert worker.is_study_complete(study, [42, 43]) is False


def test_is_study_complete_accepts_ablation_subset() -> None:
    study = optuna.create_study(direction="minimize")
    plan = worker.resolve_ablation_plan("baseline mobility:off")
    for seed in [42, 43]:
        for ablation in plan:
            _add_terminal_trial(study, seed=seed, ablation=ablation)

    assert worker.is_study_complete(study, [42, 43], plan) is True


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
