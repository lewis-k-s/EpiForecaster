from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
import yaml

from scripts.analysis.analyze_hpsearch_journal import analyze_hpsearch
from scripts.hpo.trial_artifacts import (
    TRAINING_DYNAMICS_MANIFEST_ATTR,
    TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR,
    TRAINING_DYNAMICS_STATUS_ATTR,
)
from training.training_dynamics import (
    TrainingDynamicsWriter,
    load_training_dynamics_identity,
)


def _study(journal: Path, name: str = "study") -> optuna.Study:
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))
    )
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
    )


def _complete_trial(
    study: optuna.Study,
    *,
    value: float,
    x: float | None = None,
) -> int:
    trial = study.ask()
    if x is not None:
        trial.suggest_float("x", 0.0, 100.0)
    study.tell(trial, value)
    return trial.number


def test_analyzer_preserves_trial_numbers_and_all_states(tmp_path: Path) -> None:
    journal = tmp_path / "study.journal"
    study = _study(journal)
    failed = study.ask()
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    pruned = study.ask()
    study.tell(pruned, state=optuna.trial.TrialState.PRUNED)
    complete_number = _complete_trial(study, value=0.25, x=5.0)
    running = study.ask()

    data, warnings = analyze_hpsearch(
        run_root=None,
        journal=journal,
        study_name="study",
        top_n=5,
        importance=False,
    )

    assert warnings == []
    assert data["study"]["state_counts"] == {
        "COMPLETE": 1,
        "FAIL": 1,
        "PRUNED": 1,
        "RUNNING": 1,
    }
    assert complete_number == 2
    assert data["top_trials"][0]["number"] == 2
    assert running.number == 3
    assert [(trial["number"], trial["state"]) for trial in data["trials"]] == [
        (0, "FAIL"),
        (1, "PRUNED"),
        (2, "COMPLETE"),
        (3, "RUNNING"),
    ]


def test_importance_requires_twenty_completed_trials(tmp_path: Path) -> None:
    journal = tmp_path / "study.journal"
    study = _study(journal)
    for index in range(19):
        _complete_trial(study, value=float(index), x=float(index))

    data, warnings = analyze_hpsearch(
        run_root=None,
        journal=journal,
        study_name="study",
        top_n=3,
        importance=True,
    )

    assert data["importance"] is None
    assert warnings == [
        "Importance requires at least 20 completed trials; got 19"
    ]


def test_run_root_reconciles_summary_and_dynamics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    journal = run_root / "optuna/study.journal"
    journal.parent.mkdir(parents=True)
    study = _study(journal)
    trial = study.ask()
    trial_dir = run_root / "optuna/trials/study/trial-000000"
    writer = TrainingDynamicsWriter(
        trial_dir,
        progress_log_frequency=10,
        grad_norm_log_frequency=10,
        mobility_enabled=True,
    )
    writer.append(step=0, epoch=0, values={"loss_train_step": 1.0})
    writer.finalize("complete")
    identity = load_training_dynamics_identity(trial_dir)
    trial.set_user_attr(
        TRAINING_DYNAMICS_MANIFEST_ATTR,
        (
            "/workspace/run/optuna/trials/study/trial-000000/"
            "training_dynamics_manifest.json"
        ),
    )
    trial.set_user_attr(
        TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR,
        identity["manifest_sha256"],
    )
    trial.set_user_attr(TRAINING_DYNAMICS_STATUS_ATTR, identity["status"])
    study.tell(trial, 0.2)

    (run_root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {"kind": "hpo", "target": "hpo"},
                "optuna": {"study_name": "study"},
            }
        )
    )
    summary = {
        "study_name": "study",
        "trial_count": 1,
        "state_counts": {"COMPLETE": 1},
        "best_trial": {"number": 0},
    }
    (run_root / "optuna/study_summary.json").write_text(
        json.dumps(summary)
    )

    data, warnings = analyze_hpsearch(
        run_root=run_root,
        journal=None,
        study_name=None,
        top_n=5,
        importance=False,
    )

    assert warnings == []
    assert data["run"]["status"] == "partial"
    assert data["run"]["issues"] == []
    assert data["training_dynamics"]["available_count"] == 1
    assert data["training_dynamics"]["issues"] == []


def test_dynamics_integrity_detects_tampered_jsonl(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    journal = run_root / "optuna/study.journal"
    journal.parent.mkdir(parents=True)
    study = _study(journal)
    trial = study.ask()
    trial_dir = run_root / "optuna/trials/study/trial-000000"
    writer = TrainingDynamicsWriter(
        trial_dir,
        progress_log_frequency=10,
        grad_norm_log_frequency=10,
        mobility_enabled=True,
    )
    writer.append(step=0, epoch=0, values={"loss_train_step": 1.0})
    writer.finalize("complete")
    identity = load_training_dynamics_identity(trial_dir)
    trial.set_user_attr(TRAINING_DYNAMICS_MANIFEST_ATTR, identity["manifest"])
    trial.set_user_attr(
        TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR,
        identity["manifest_sha256"],
    )
    trial.set_user_attr(TRAINING_DYNAMICS_STATUS_ATTR, "complete")
    study.tell(trial, 0.2)
    (trial_dir / "training_dynamics.jsonl").write_text('{"tampered": true}\n')
    (run_root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {"kind": "hpo", "target": "hpo"},
                "optuna": {"study_name": "study"},
            }
        )
    )

    data, warnings = analyze_hpsearch(
        run_root=run_root,
        journal=None,
        study_name=None,
        top_n=5,
        importance=False,
    )

    assert data["training_dynamics"]["issues"][0]["trial_number"] == 0
    assert "SHA-256 mismatch" in data["training_dynamics"]["issues"][0]["detail"]
    assert warnings and "training dynamics" in warnings[0]


def test_invalid_workflow_receipt_cannot_mark_run_complete(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    journal = run_root / "optuna/study.journal"
    journal.parent.mkdir(parents=True)
    study = _study(journal)
    _complete_trial(study, value=0.1)
    (run_root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {"kind": "hpo", "target": "hpo"},
                "optuna": {"study_name": "study"},
            }
        )
    )
    marker = run_root / ".state/workflow.done"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "workflow_kind": "train",
                "workflow_target": "train",
                "terminal_artifact": "/workspace/run/missing.json",
            }
        )
    )

    data, _ = analyze_hpsearch(
        run_root=run_root,
        journal=None,
        study_name=None,
        top_n=5,
        importance=False,
    )

    assert data["run"]["status"] == "partial"
    assert "Workflow completion kind does not match config" in data["run"]["issues"]


def test_direct_journal_requires_explicit_study_name(tmp_path: Path) -> None:
    journal = tmp_path / "study.journal"
    _study(journal)

    with pytest.raises(ValueError, match="study-name"):
        analyze_hpsearch(
            run_root=None,
            journal=journal,
            study_name=None,
            top_n=5,
            importance=False,
        )
