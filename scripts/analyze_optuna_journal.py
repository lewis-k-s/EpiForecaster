"""Analyze Optuna journal hyperparameter search results.

This script parses Optuna journal files and produces:
- Summary statistics of completed trials
- Best trials and their parameters
- Parameter importance analysis
- Parameter distribution plots (if matplotlib available)
- Recommendations for search space refinement

Usage:
    uv run python scripts/analyze_optuna_journal.py outputs/optuna/epiforecaster_hpo_v1.journal
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.skill_output import SkillOutputBuilder, print_output


@dataclass
class Trial:
    """Represents a single Optuna trial."""

    number: int
    params: dict[str, Any]
    value: float | None
    state: str  # "COMPLETE", "FAILED", "RUNNING", etc.
    datetime_start: str | None = None
    datetime_complete: str | None = None
    user_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudySummary:
    """Summary of an Optuna study."""

    study_name: str
    trials: list[Trial] = field(default_factory=list)

    @property
    def completed_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.state == "COMPLETE"]

    @property
    def best_trial(self) -> Trial | None:
        completed = self.completed_trials
        if not completed:
            return None
        return min(
            completed, key=lambda t: t.value if t.value is not None else float("inf")
        )

    @property
    def best_value(self) -> float | None:
        if self.best_trial and self.best_trial.value is not None:
            return self.best_trial.value
        return None


def parse_journal(path: Path) -> StudySummary:
    """Parse an Optuna journal file.

    The journal format is line-delimited JSON with op_codes:
    - 0: REGISTER_STUDY
    - 4: START_TRIAL
    - 5: SET_UP_TRIAL_PARAM
    - 6: FINISH_TRIAL
    - 8: SET_TRIAL_USER_ATTR
    """
    study_name = "unknown"
    trial_params: dict[int, dict[str, Any]] = defaultdict(dict)
    trial_attrs: dict[int, dict[str, Any]] = defaultdict(dict)
    trial_start_times: dict[int, str] = {}
    trial_states: dict[int, int] = {}
    trial_values: dict[int, float | None] = {}
    trial_end_times: dict[int, str] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            op_code = entry.get("op_code")

            if op_code == 0:  # REGISTER_STUDY
                study_name = entry.get("study_name", "unknown")

            elif op_code == 4:  # START_TRIAL
                trial_id = entry.get("trial_id")
                if trial_id is None:
                    continue
                trial_start_times[trial_id] = entry.get("datetime_start")
                trial_states[trial_id] = 0  # RUNNING

            elif op_code == 5:  # SET_UP_TRIAL_PARAM
                trial_id = entry.get("trial_id")
                if trial_id is None:
                    continue
                param_name = entry["param_name"]
                distribution = json.loads(entry["distribution"])

                # Decode categorical values that were JSON-encoded
                param_value_internal = entry["param_value_internal"]
                if distribution["name"] == "CategoricalDistribution":
                    choices = distribution["attributes"]["choices"]
                    # param_value_internal is an integer index into choices
                    selected = choices[int(param_value_internal)]
                    if isinstance(selected, str):
                        # Check if it's a JSON-encoded string
                        try:
                            param_value = json.loads(selected)
                        except (json.JSONDecodeError, TypeError):
                            param_value = selected
                    else:
                        param_value = selected
                else:
                    param_value = param_value_internal

                trial_params[trial_id][param_name] = param_value

            elif op_code == 6:  # FINISH_TRIAL
                trial_id = entry.get("trial_id")
                if trial_id is None:
                    continue
                trial_states[trial_id] = entry["state"]
                trial_values[trial_id] = (
                    entry["values"][0] if entry.get("values") else None
                )
                trial_end_times[trial_id] = entry.get("datetime_complete")

            elif op_code == 8:  # SET_TRIAL_USER_ATTR
                trial_id = entry.get("trial_id")
                if trial_id is None:
                    continue
                for key, value in entry.get("user_attr", {}).items():
                    trial_attrs[trial_id][key] = value

    # Convert to Trial objects
    # Determine trial numbers by sorting by trial_id
    sorted_trial_ids = sorted(trial_params.keys())
    trials = []
    for i, trial_id in enumerate(sorted_trial_ids):
        state_code = trial_states.get(trial_id, 0)
        state_map = {0: "RUNNING", 1: "COMPLETE", 2: "PRUNED", 3: "FAIL"}
        state = state_map.get(state_code, "UNKNOWN")

        trial = Trial(
            number=i,
            params=trial_params.get(trial_id, {}),
            value=trial_values.get(trial_id),
            state=state,
            datetime_start=trial_start_times.get(trial_id),
            datetime_complete=trial_end_times.get(trial_id),
            user_attrs=trial_attrs.get(trial_id, {}),
        )
        trials.append(trial)

    return StudySummary(study_name=study_name, trials=trials)


def analyze_param_importance(summary: StudySummary) -> dict[str, float]:
    """Compute simple parameter importance based on correlation with objective.

    This is a simplified version using Spearman correlation.
    For proper importance, use optuna.importance.get_param_importances().
    """
    completed = summary.completed_trials
    if len(completed) < 2:
        return {}

    # Get all param names
    param_names = set()
    for t in completed:
        param_names.update(t.params.keys())

    importance = {}
    for param in param_names:
        # Get numeric values for this param
        values = []
        scores = []
        for t in completed:
            if param not in t.params or t.value is None:
                continue
            val = t.params[param]

            # Convert to numeric if possible
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif isinstance(val, str):
                # Use string hash for categorical
                values.append(hash(val))
            else:
                continue

            scores.append(t.value)

        if len(values) < 2:
            continue

        # Compute Spearman correlation
        values_arr = np.array(values)
        scores_arr = np.array(scores)

        # Rank-based correlation
        rank_x = np.argsort(np.argsort(values_arr))
        rank_y = np.argsort(np.argsort(scores_arr))

        corr = np.corrcoef(rank_x, rank_y)[0, 1]
        if not np.isnan(corr):
            importance[param] = abs(corr)

    # Sort by importance
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def format_summary(summary: StudySummary) -> dict[str, Any]:
    """Format summary statistics."""
    completed = summary.completed_trials
    failed = [t for t in summary.trials if t.state == "FAIL"]
    running = [t for t in summary.trials if t.state == "RUNNING"]

    result = {
        "study_name": summary.study_name,
        "total_trials": len(summary.trials),
        "completed": len(completed),
        "failed": len(failed),
        "running": len(running),
    }

    if completed:
        values = [t.value for t in completed if t.value is not None]
        if values:
            result["objective_stats"] = {
                "best": float(min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "worst": float(max(values)),
            }

    return result


def print_summary(summary: StudySummary) -> None:
    """Print a formatted summary of study."""
    result = format_summary(summary)

    print(f"\n{'=' * 60}")
    print(f"Study: {result['study_name']}")
    print(f"{'=' * 60}")
    print(f"Total trials:     {result['total_trials']}")
    print(f"Completed:        {result['completed']}")
    print(f"Failed:           {result['failed']}")
    print(f"Running:          {result['running']}")

    if "objective_stats" in result:
        stats = result["objective_stats"]
        print("\nObjective (best_val_loss):")
        print(f"  Best:            {stats['best']:.6f}")
        print(f"  Median:          {stats['median']:.6f}")
        print(f"  Mean:            {stats['mean']:.6f}")
        print(f"  Std:             {stats['std']:.6f}")
        print(f"  Worst:           {stats['worst']:.6f}")

    print(f"\n{'=' * 60}\n")


def format_best_trials(summary: StudySummary, n: int = 5) -> list[dict[str, Any]]:
    """Format N best trials."""
    completed = [t for t in summary.completed_trials if t.value is not None]
    if not completed:
        return []

    # Sort by value (lower is better for loss)
    def get_value(t: Trial) -> float:
        assert t.value is not None
        return t.value

    sorted_trials = sorted(completed, key=get_value)[:n]

    return [
        {
            "rank": i,
            "trial_number": trial.number,
            "value": float(trial.value) if trial.value is not None else None,
            "params": trial.params,
        }
        for i, trial in enumerate(sorted_trials, 1)
    ]


def print_best_trials(summary: StudySummary, n: int = 5) -> None:
    """Print N best trials."""
    trials = format_best_trials(summary, n)

    if not trials:
        print("No completed trials with values.")
        return

    print(f"\nTop {len(trials)} trials:")
    print("-" * 60)
    for trial in trials:
        print(
            f"\n#{trial['rank']} Trial {trial['trial_number']} (value: {trial['value']:.6f})"
        )
        for param, value in sorted(trial["params"].items()):
            print(f"  {param}: {value}")


def format_param_distributions(summary: StudySummary) -> dict[str, dict[str, Any]]:
    """Format distribution statistics for each parameter."""
    completed = summary.completed_trials
    if not completed:
        return {}

    # Group params by name
    param_values: dict[str, list] = defaultdict(list)
    for t in completed:
        for param, value in t.params.items():
            param_values[param].append(value)

    distributions = {}

    for param, values in sorted(param_values.items()):
        dist: dict[str, Any] = {
            "param": param,
            "type": "categorical" if isinstance(values[0], str) else "numeric",
        }

        # Count categorical values
        if isinstance(values[0], str):
            counts = Counter(values)
            total = len(values)
            dist["values"] = [
                {"value": val, "count": count, "percentage": 100 * count / total}
                for val, count in sorted(
                    counts.items(), key=lambda x: x[1], reverse=True
                )
            ]
        else:
            # Numeric values
            numeric_vals = [float(v) for v in values]
            dist["values"] = {
                "min": float(min(numeric_vals)),
                "max": float(max(numeric_vals)),
                "mean": float(np.mean(numeric_vals)),
                "median": float(np.median(numeric_vals)),
                "p10": float(np.percentile(numeric_vals, 10)),
                "p90": float(np.percentile(numeric_vals, 90)),
            }

        distributions[param] = dist

    return distributions


def print_param_distributions(summary: StudySummary) -> None:
    """Print distribution statistics for each parameter."""
    distributions = format_param_distributions(summary)

    if not distributions:
        return

    print(f"\n{'=' * 60}")
    print("Parameter Distributions")
    print(f"{'=' * 60}")

    for param, dist in sorted(distributions.items()):
        print(f"\n{param}:")

        if dist["type"] == "categorical":
            for val_info in dist["values"]:
                print(
                    f"  {val_info['value']}: {val_info['count']} ({val_info['percentage']:.1f}%)"
                )
        else:
            vals = dist["values"]
            print(f"  min:     {vals['min']:g}")
            print(f"  max:     {vals['max']:g}")
            print(f"  mean:    {vals['mean']:g}")
            print(f"  median:  {vals['median']:g}")


def format_recommendations(summary: StudySummary) -> list[dict[str, Any]]:
    """Format recommendations for search space refinement."""
    completed = summary.completed_trials
    if len(completed) < 10:
        return []

    recommendations = []

    # Analyze each parameter
    param_values: dict[str, list] = defaultdict(list)
    for t in completed:
        for param, value in t.params.items():
            param_values[param].append(value)

    # Separate numeric and categorical
    for param, values in sorted(param_values.items()):
        if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            vals = [float(v) for v in values]
            p10, p90 = np.percentile(vals, [10, 90])

            rec = {
                "param": param,
                "type": "numeric",
                "current_range": [min(vals), max(vals)],
                "p80_interval": [p10, p90],
            }

            # Suggest narrowed range
            if p90 - p10 < max(vals) - min(vals):
                rec["suggestion"] = "narrow_range"
                rec["suggested_range"] = [p10, p90]

            recommendations.append(rec)

        # For categorical, check if any values are unused
        elif isinstance(values[0], str):
            counts = Counter(values)
            total = len(values)
            unused = [k for k, v in counts.items() if v / total < 0.05]

            rec = {
                "param": param,
                "type": "categorical",
                "value_counts": {str(k): v for k, v in counts.items()},
            }

            if unused:
                rec["suggestion"] = "remove_unused"
                rec["unused_values"] = unused

            recommendations.append(rec)

    return recommendations


def print_recommendations(summary: StudySummary) -> None:
    """Print recommendations for search space refinement."""
    recommendations = format_recommendations(summary)

    if not recommendations:
        print("\nRecommendations: run more trials for analysis")
        return

    print(f"\n{'=' * 60}")
    print("Recommendations for Search Space Refinement")
    print(f"{'=' * 60}")

    for rec in recommendations:
        if rec["type"] == "numeric":
            print(f"\n{rec['param']}:")
            print(
                f"  Current range: [{rec['current_range'][0]:g}, {rec['current_range'][1]:g}]"
            )
            print(
                f"  80% interval:  [{rec['p80_interval'][0]:g}, {rec['p80_interval'][1]:g}]"
            )

            if rec.get("suggestion") == "narrow_range":
                print(
                    f"  -> Suggest narrowing to: [{rec['suggested_range'][0]:g}, {rec['suggested_range'][1]:g}]"
                )

        elif rec["type"] == "categorical":
            if rec.get("suggestion") == "remove_unused":
                print(f"\n{rec['param']}:")
                print(f"  Rarely used values: {rec['unused_values']}")


def print_importance(summary: StudySummary) -> None:
    """Print parameter importance analysis."""
    importance = analyze_param_importance(summary)

    if not importance:
        print("\nParameter importance: insufficient data")
        return

    print(f"\n{'=' * 60}")
    print("Parameter Importance (abs(Spearman correlation))")
    print(f"{'=' * 60}")
    for param, imp in importance.items():
        print(f"  {param}: {imp:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Optuna journal hyperparameter search results"
    )
    parser.add_argument(
        "journal",
        type=Path,
        help="Path to Optuna journal file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top trials to show (default: 5)",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Compute and show parameter importance",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (no indentation)"
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="optuna-analyze",
        input_path=str(args.journal),
    )

    try:
        summary = parse_journal(args.journal)

        data = {
            "summary": format_summary(summary),
            "best_trials": format_best_trials(summary, n=args.top),
            "param_distributions": format_param_distributions(summary),
            "recommendations": format_recommendations(summary),
        }

        if args.importance:
            importance = analyze_param_importance(summary)
            data["importance"] = importance

        output = builder.success(data)

        if args.text:
            print_summary(summary)
            print_best_trials(summary, n=args.top)
            print_param_distributions(summary)

            if args.importance:
                print_importance(summary)

            print_recommendations(summary)
        else:
            indent = 0 if args.compact else 2
            print_output(output, indent=indent)

    except FileNotFoundError:
        print_output(
            builder.error(
                "FileNotFoundError", f"Journal file not found: {args.journal}"
            )
        )
    except json.JSONDecodeError as e:
        print_output(builder.error("JSONDecodeError", f"Invalid journal format: {e}"))
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    main()
