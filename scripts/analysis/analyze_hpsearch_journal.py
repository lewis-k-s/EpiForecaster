"""Analyze Optuna studies through the supported JournalStorage API."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import optuna
import yaml
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FrozenTrial, TrialState

from scripts.analysis.skill_output import SkillOutputBuilder, print_output
from scripts.hpo.trial_artifacts import (
    TRAINING_DYNAMICS_MANIFEST_ATTR,
    TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR,
    TRAINING_DYNAMICS_STATUS_ATTR,
)
from training.training_dynamics import (
    load_training_dynamics_identity,
    sha256_file,
)

MIN_IMPORTANCE_COMPLETE_TRIALS = 20
DEFAULT_TOP_TRIALS = 5


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return document


def _resolve_study_input(
    *,
    run_root: Path | None,
    journal: Path | None,
    study_name: str | None,
) -> tuple[Path, str, Path | None]:
    if (run_root is None) == (journal is None):
        raise ValueError("Provide exactly one of --run-root or --journal")

    if run_root is not None:
        root = run_root.resolve()
        config_path = root / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Immutable run config is missing: {config_path}")
        config = _load_yaml(config_path)
        if config.get("workflow", {}).get("kind") != "hpo":
            raise ValueError(f"Run root is not an HPO workflow: {root}")
        configured_name = str(config["optuna"]["study_name"])
        if study_name is not None and study_name != configured_name:
            raise ValueError(
                "Explicit study name disagrees with the immutable run config: "
                f"{study_name!r} != {configured_name!r}"
            )
        resolved_journal = root / "optuna" / f"{configured_name}.journal"
        resolved_name = configured_name
        resolved_root = root
    else:
        if not study_name:
            raise ValueError("--study-name is required with --journal")
        resolved_journal = journal.resolve()  # type: ignore[union-attr]
        resolved_name = study_name
        resolved_root = None

    if not resolved_journal.is_file():
        raise FileNotFoundError(f"Optuna journal is missing: {resolved_journal}")
    return resolved_journal, resolved_name, resolved_root


def _load_study(journal: Path, study_name: str) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(str(journal)))
    available = {summary.study_name for summary in optuna.get_all_study_summaries(storage)}
    if study_name not in available:
        raise ValueError(
            f"Study {study_name!r} is not in {journal}; available={sorted(available)}"
        )
    return optuna.load_study(study_name=study_name, storage=storage)


def _trial_values(trial: FrozenTrial) -> list[float] | None:
    if trial.values is not None:
        return [float(value) for value in trial.values]
    if trial.value is not None:
        return [float(trial.value)]
    return None


def _serialize_trial(trial: FrozenTrial) -> dict[str, Any]:
    duration_s = (
        (trial.datetime_complete - trial.datetime_start).total_seconds()
        if trial.datetime_start is not None and trial.datetime_complete is not None
        else None
    )
    return {
        "number": trial.number,
        "state": trial.state.name,
        "values": _trial_values(trial),
        "params": dict(sorted(trial.params.items())),
        "distributions": {
            name: repr(distribution)
            for name, distribution in sorted(trial.distributions.items())
        },
        "user_attrs": dict(sorted(trial.user_attrs.items())),
        "datetime_start": (
            trial.datetime_start.isoformat()
            if trial.datetime_start is not None
            else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat()
            if trial.datetime_complete is not None
            else None
        ),
        "duration_s": duration_s,
    }


def _ranked_trials(study: optuna.Study, top_n: int) -> list[dict[str, Any]]:
    if len(study.directions) == 1:
        completed = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE and trial.value is not None
        ]
        reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
        ranked = sorted(
            completed,
            key=lambda trial: float(trial.value),
            reverse=reverse,
        )
    else:
        ranked = sorted(study.best_trials, key=lambda trial: trial.number)
    return [_serialize_trial(trial) for trial in ranked[:top_n]]


def _objective_summary(study: optuna.Study) -> dict[str, Any]:
    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.values is not None
    ]
    objectives: list[dict[str, Any]] = []
    for index, direction in enumerate(study.directions):
        values = [
            float(trial.values[index])
            for trial in completed
            if trial.values is not None
            and index < len(trial.values)
            and math.isfinite(float(trial.values[index]))
        ]
        objectives.append(
            {
                "index": index,
                "direction": direction.name,
                "count": len(values),
                "best": (
                    min(values)
                    if direction == optuna.study.StudyDirection.MINIMIZE
                    else max(values)
                )
                if values
                else None,
                "median": median(values) if values else None,
                "mean": fmean(values) if values else None,
                "worst": (
                    max(values)
                    if direction == optuna.study.StudyDirection.MINIMIZE
                    else min(values)
                )
                if values
                else None,
            }
        )
    return {"objectives": objectives}


def _parameter_summary(study: optuna.Study) -> dict[str, Any]:
    completed = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    values: dict[str, list[Any]] = defaultdict(list)
    for trial in completed:
        for name, value in trial.params.items():
            values[name].append(value)

    result: dict[str, Any] = {}
    for name in sorted(values):
        observed = values[name]
        coverage = len(observed)
        summary: dict[str, Any] = {
            "completed_trial_coverage": coverage,
            "completed_trial_fraction": (
                coverage / len(completed) if completed else 0.0
            ),
        }
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in observed
        ):
            numeric = [float(value) for value in observed]
            summary.update(
                {
                    "kind": "numeric",
                    "min": min(numeric),
                    "max": max(numeric),
                    "median": median(numeric),
                }
            )
        else:
            counts = Counter(str(value) for value in observed)
            summary.update(
                {
                    "kind": "categorical",
                    "counts": dict(sorted(counts.items())),
                }
            )
        result[name] = summary
    return result


def _relocate_run_path(path_value: str, run_root: Path | None) -> Path:
    path = Path(path_value)
    if run_root is not None and path.is_absolute():
        try:
            return run_root / path.relative_to("/workspace/run")
        except ValueError:
            pass
    return path


def _dynamics_integrity(
    trials: list[FrozenTrial],
    *,
    run_root: Path | None,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    available = 0
    issues: list[dict[str, Any]] = []
    for trial in trials:
        manifest_value = trial.user_attrs.get(TRAINING_DYNAMICS_MANIFEST_ATTR)
        digest_value = trial.user_attrs.get(
            TRAINING_DYNAMICS_MANIFEST_SHA256_ATTR
        )
        status = trial.user_attrs.get(TRAINING_DYNAMICS_STATUS_ATTR)
        if status is not None:
            status_counts[str(status)] += 1
        if not isinstance(manifest_value, str) or not manifest_value:
            continue
        available += 1
        manifest = _relocate_run_path(manifest_value, run_root)
        if not manifest.is_file():
            issues.append(
                {"trial_number": trial.number, "issue": "manifest_missing"}
            )
            continue
        try:
            identity = load_training_dynamics_identity(manifest.parent)
            if digest_value != identity["manifest_sha256"]:
                raise ValueError("manifest_sha256_mismatch")
            if status is not None and str(status) != identity["status"]:
                raise ValueError("status_mismatch")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "trial_number": trial.number,
                    "issue": "invalid_training_dynamics",
                    "detail": str(exc),
                }
            )
    return {
        "available_count": available,
        "status_counts": dict(sorted(status_counts.items())),
        "issues": issues,
    }


def _importance(
    study: optuna.Study,
    *,
    enabled: bool,
) -> tuple[dict[str, float] | None, list[str]]:
    if not enabled:
        return None, []
    complete = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if len(study.directions) != 1:
        return None, ["Importance is unavailable for multi-objective studies"]
    if len(complete) < MIN_IMPORTANCE_COMPLETE_TRIALS:
        return None, [
            "Importance requires at least "
            f"{MIN_IMPORTANCE_COMPLETE_TRIALS} completed trials; got {len(complete)}"
        ]
    common = set(complete[0].params)
    for trial in complete[1:]:
        common.intersection_update(trial.params)
    if not common:
        return None, ["No parameters are common to every completed trial"]
    evaluator = FanovaImportanceEvaluator(seed=0)
    importances = get_param_importances(
        study,
        evaluator=evaluator,
        params=sorted(common),
    )
    return dict(sorted(importances.items(), key=lambda item: (-item[1], item[0]))), []


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _reconcile_run_root(
    run_root: Path,
    study: optuna.Study,
    state_counts: dict[str, int],
) -> dict[str, Any]:
    summary_path = run_root / "optuna/study_summary.json"
    selection_path = run_root / "outputs/selection.json"
    evaluation_path = run_root / "outputs/evaluation/evaluation_manifest.json"
    receipt_paths = {
        "study_summary": summary_path,
        "selection": selection_path,
        "evaluation": evaluation_path,
        "promotion": run_root / ".state/promoted.done",
        "cleanup": run_root / ".state/checkpoints_cleaned.done",
        "workflow_done": run_root / ".state/workflow.done",
    }
    receipts = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in receipt_paths.items()
    }

    issues: list[str] = []
    summary = _load_json_if_present(summary_path)
    selection = _load_json_if_present(selection_path)
    evaluation = _load_json_if_present(evaluation_path)
    promotion = _load_json_if_present(receipt_paths["promotion"])
    cleanup = _load_json_if_present(receipt_paths["cleanup"])
    workflow_done = _load_json_if_present(receipt_paths["workflow_done"])
    config = _load_yaml(run_root / "config.yaml")
    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    best_number = study.best_trial.number if completed else None
    if summary is not None:
        if summary.get("study_name") != study.study_name:
            issues.append("Study summary name does not match JournalStorage")
        if int(summary.get("trial_count", -1)) != len(study.trials):
            issues.append("Study summary trial count does not match JournalStorage")
        if summary.get("state_counts") != state_counts:
            issues.append("Study summary state counts do not match JournalStorage")
        if summary.get("best_trial", {}).get("number") != best_number:
            issues.append("Study summary best trial does not match JournalStorage")
    if selection is not None and selection.get("best_trial", {}).get("number") != best_number:
        issues.append("Selection best trial does not match JournalStorage")
    if selection is not None:
        checkpoint = _relocate_run_path(
            str(selection.get("selected_checkpoint", "")),
            run_root,
        )
        expected = selection.get("selected_checkpoint_sha256")
        if not checkpoint.is_file() or expected != sha256_file(checkpoint):
            issues.append("Selected checkpoint is missing or failed SHA-256")
        if evaluation is not None:
            evaluated_checkpoint = _relocate_run_path(
                str(evaluation.get("checkpoint", "")),
                run_root,
            )
            if evaluation.get("checkpoint_sha256") != expected:
                issues.append("Evaluation checkpoint hash does not match selection")
            if evaluated_checkpoint.resolve() != checkpoint.resolve():
                issues.append("Evaluation checkpoint path does not match selection")
    if evaluation is not None and selection is None:
        issues.append("Evaluation exists without a selection receipt")
    if cleanup is not None:
        if selection is None:
            issues.append("Cleanup exists without a selection receipt")
        else:
            cleanup_checkpoint = _relocate_run_path(
                str(cleanup.get("selected_checkpoint", "")),
                run_root,
            )
            selected_checkpoint = _relocate_run_path(
                str(selection.get("selected_checkpoint", "")),
                run_root,
            )
            if (
                cleanup_checkpoint.resolve() != selected_checkpoint.resolve()
                or cleanup.get("selected_checkpoint_sha256")
                != selection.get("selected_checkpoint_sha256")
            ):
                issues.append("Cleanup receipt does not match selection")
        if promotion is None:
            issues.append("Cleanup exists without a promotion receipt")
    if promotion is not None:
        configured_run_id = config.get("run", {}).get("id")
        if configured_run_id is not None and promotion.get("run_id") != configured_run_id:
            issues.append("Promotion receipt run ID does not match config")
        for required in ("destination", "dvc_pointer", "manifest_identity"):
            if required not in promotion:
                issues.append(f"Promotion receipt is missing {required}")
    workflow_valid = False
    if workflow_done is not None:
        expected_kind = config.get("workflow", {}).get("kind")
        expected_target = config.get("workflow", {}).get("target", expected_kind)
        if workflow_done.get("workflow_kind") != expected_kind:
            issues.append("Workflow completion kind does not match config")
        elif workflow_done.get("workflow_target") != expected_target:
            issues.append("Workflow completion target does not match config")
        else:
            terminal = _relocate_run_path(
                str(workflow_done.get("terminal_artifact", "")),
                run_root,
            )
            if not terminal.is_file():
                issues.append("Workflow completion terminal artifact is missing")
            else:
                workflow_valid = True

    if workflow_valid and not issues:
        status = "complete"
    elif any(
        receipt["exists"] for receipt in receipts.values()
    ):
        status = "partial"
    else:
        status = "live"
    return {"status": status, "receipts": receipts, "issues": issues}


def analyze_hpsearch(
    *,
    run_root: Path | None,
    journal: Path | None,
    study_name: str | None,
    top_n: int,
    importance: bool,
) -> tuple[dict[str, Any], list[str]]:
    if top_n < 1:
        raise ValueError("--top must be at least 1")
    journal_path, resolved_name, resolved_root = _resolve_study_input(
        run_root=run_root,
        journal=journal,
        study_name=study_name,
    )
    study = _load_study(journal_path, resolved_name)
    state_counts = dict(
        sorted(Counter(trial.state.name for trial in study.trials).items())
    )
    importances, warnings = _importance(study, enabled=importance)
    run = (
        _reconcile_run_root(resolved_root, study, state_counts)
        if resolved_root is not None
        else {"status": "standalone", "receipts": {}, "issues": []}
    )
    data = {
        "study": {
            "name": study.study_name,
            "journal": str(journal_path),
            "directions": [direction.name for direction in study.directions],
            "trial_count": len(study.trials),
            "state_counts": state_counts,
        },
        "run": run,
        "objectives": _objective_summary(study),
        "trials": [_serialize_trial(trial) for trial in study.trials],
        "top_trials": _ranked_trials(study, top_n),
        "parameters": _parameter_summary(study),
        "training_dynamics": _dynamics_integrity(
            study.trials,
            run_root=resolved_root,
        ),
        "importance": importances,
    }
    warnings.extend(
        "Trial "
        f"{issue['trial_number']} training dynamics: {issue['issue']}"
        + (f" ({issue['detail']})" if issue.get("detail") else "")
        for issue in data["training_dynamics"]["issues"]
    )
    return data, warnings


def _print_text(data: dict[str, Any], warnings: list[str]) -> None:
    study = data["study"]
    print(f"HPO STUDY: {study['name']}")
    print(f"Run status: {data['run']['status']}")
    print(f"Trials: {study['trial_count']} {study['state_counts']}")
    for trial in data["top_trials"]:
        print(f"Trial {trial['number']}: values={trial['values']}")
    for warning in warnings:
        print(f"Warning: {warning}")
    for issue in data["run"]["issues"]:
        print(f"Integrity issue: {issue}")
    for issue in data["training_dynamics"]["issues"]:
        print(
            "Dynamics integrity issue: "
            f"trial={issue['trial_number']} {issue['issue']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an Optuna HPO study through JournalStorage"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--journal", type=Path)
    parser.add_argument("--study-name")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_TRIALS)
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    input_path = args.run_root or args.journal
    builder = SkillOutputBuilder("hpsearch-analyze", input_path=str(input_path))
    try:
        data, warnings = analyze_hpsearch(
            run_root=args.run_root,
            journal=args.journal,
            study_name=args.study_name,
            top_n=args.top,
            importance=args.importance,
        )
        for warning in warnings:
            builder.add_warning(warning)
        for issue in data["run"]["issues"]:
            builder.add_warning(issue)
        if args.text:
            _print_text(data, [*warnings, *data["run"]["issues"]])
        else:
            print_output(
                builder.success(data),
                indent=None if args.compact else 2,
            )
    except Exception as exc:
        print_output(
            builder.error(type(exc).__name__, str(exc)),
            indent=None if args.compact else 2,
        )


if __name__ == "__main__":
    main()
