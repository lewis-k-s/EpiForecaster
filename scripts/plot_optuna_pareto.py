"""Plot and export Pareto results for missing-permit multi-objective Optuna studies."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any

PARETO_TARGET_NAMES = ["best_val_loss", "val_coverage_ratio"]
USER_ATTR_COLUMNS = [
    "val_samples",
    "val_samples_reference",
    "val_coverage_ratio",
    "missing_permit_overrides",
    "frozen_overrides_source",
]


def _json_or_scalar(value: Any) -> str:
    """Encode nested values for CSV fields while preserving scalar readability."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def load_study_from_journal(*, journal_file: Path, study_name: str) -> Any:
    """Load an Optuna study backed by journal storage."""
    storages = importlib.import_module("optuna.storages")
    JournalStorage = getattr(storages, "JournalStorage")
    journal_module = importlib.import_module("optuna.storages.journal")
    JournalFileBackend = getattr(journal_module, "JournalFileBackend")
    optuna = importlib.import_module("optuna")

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    return optuna.load_study(study_name=study_name, storage=storage)


def export_trials_csv(*, study: Any, out_csv: Path) -> Path:
    """Export completed trials to CSV with objectives, params, and key user attrs."""
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]

    param_names: set[str] = set()
    for trial in completed:
        param_names.update(trial.params.keys())
    ordered_params = sorted(param_names)

    fieldnames = [
        "trial_number",
        "state",
        PARETO_TARGET_NAMES[0],
        PARETO_TARGET_NAMES[1],
        *USER_ATTR_COLUMNS,
        *(f"param.{name}" for name in ordered_params),
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in completed:
            values = trial.values or [None, None]
            row = {
                "trial_number": trial.number,
                "state": trial.state.name,
                PARETO_TARGET_NAMES[0]: values[0] if len(values) > 0 else None,
                PARETO_TARGET_NAMES[1]: values[1] if len(values) > 1 else None,
            }
            for attr_key in USER_ATTR_COLUMNS:
                row[attr_key] = _json_or_scalar(trial.user_attrs.get(attr_key))
            for param_key in ordered_params:
                row[f"param.{param_key}"] = _json_or_scalar(trial.params.get(param_key))
            writer.writerow(row)

    return out_csv


def export_pareto_html(*, study: Any, out_html: Path) -> Path:
    """Export interactive Pareto front plot as HTML."""
    optuna = importlib.import_module("optuna")
    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=PARETO_TARGET_NAMES,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return out_html


def export_pareto_artifacts(
    *,
    journal_file: Path,
    study_name: str,
    out_html: Path,
    out_csv: Path,
) -> tuple[Path, Path]:
    """Load study and export HTML pareto plot + CSV table."""
    study = load_study_from_journal(journal_file=journal_file, study_name=study_name)
    html_path = export_pareto_html(study=study, out_html=out_html)
    csv_path = export_trials_csv(study=study, out_csv=out_csv)
    return html_path, csv_path


def _default_out_path(*, journal_file: Path, study_name: str, suffix: str) -> Path:
    return journal_file.parent / f"{study_name}_pareto.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and export Pareto front from Optuna journal study.",
    )
    parser.add_argument(
        "--journal-file",
        type=Path,
        required=True,
        help="Path to Optuna journal file.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Optuna study name within the journal.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Optional output path for Pareto HTML plot.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output path for completed trials CSV.",
    )

    args = parser.parse_args()

    out_html = args.out_html or _default_out_path(
        journal_file=args.journal_file,
        study_name=args.study_name,
        suffix="html",
    )
    out_csv = args.out_csv or _default_out_path(
        journal_file=args.journal_file,
        study_name=args.study_name,
        suffix="csv",
    )

    html_path, csv_path = export_pareto_artifacts(
        journal_file=args.journal_file,
        study_name=args.study_name,
        out_html=out_html,
        out_csv=out_csv,
    )
    print(f"Pareto HTML: {html_path}")
    print(f"Trials CSV:  {csv_path}")


if __name__ == "__main__":
    main()
