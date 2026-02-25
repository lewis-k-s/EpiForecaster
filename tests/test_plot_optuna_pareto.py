from __future__ import annotations

import csv

import optuna

from scripts.plot_optuna_pareto import export_pareto_artifacts


def _journal_storage(path):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(path)))


def test_export_pareto_artifacts_smoke(tmp_path) -> None:
    journal = tmp_path / "pareto_study.journal"
    storage = _journal_storage(journal)
    study = optuna.create_study(
        study_name="missing_permit_pareto",
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )

    distribution = {"x": optuna.distributions.FloatDistribution(0.0, 1.0)}
    trial0 = optuna.create_trial(
        params={"x": 0.1},
        distributions=distribution,
        values=[0.5, 0.8],
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={
            "val_samples": 80,
            "val_samples_reference": 100,
            "val_coverage_ratio": 0.8,
            "missing_permit_overrides": {"data.missing_permit.input.cases": 10},
            "frozen_overrides_source": {"freeze_trial_number": 0},
        },
    )
    trial1 = optuna.create_trial(
        params={"x": 0.3},
        distributions=distribution,
        values=[0.4, 0.6],
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={
            "val_samples": 60,
            "val_samples_reference": 100,
            "val_coverage_ratio": 0.6,
            "missing_permit_overrides": {"data.missing_permit.input.cases": 5},
            "frozen_overrides_source": {"freeze_trial_number": 0},
        },
    )
    study.add_trial(trial0)
    study.add_trial(trial1)

    out_html = tmp_path / "pareto.html"
    out_csv = tmp_path / "pareto.csv"
    html_path, csv_path = export_pareto_artifacts(
        journal_file=journal,
        study_name="missing_permit_pareto",
        out_html=out_html,
        out_csv=out_csv,
    )

    assert html_path == out_html
    assert csv_path == out_csv
    assert out_html.exists()
    assert out_csv.exists()

    html_text = out_html.read_text()
    assert "best_val_loss" in html_text
    assert "val_coverage_ratio" in html_text

    with out_csv.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert "best_val_loss" in rows[0]
    assert "val_coverage_ratio" in rows[0]
    assert "val_samples" in rows[0]
    assert "param.x" in rows[0]
