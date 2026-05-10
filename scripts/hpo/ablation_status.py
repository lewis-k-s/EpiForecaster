"""Lightweight ablation journal status — works without optuna import.

Parses an Optuna JournalStorage JSONL file and prints a human-readable
summary of ablation campaign progress.  Designed to run locally or on
MN5 with minimal dependencies (only stdlib + ``click`` for the CLI entry).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click

# Optuna trial state codes (op_code 6 "state" field)
_STATE_RUNNING = 0
_STATE_COMPLETE = 1
_STATE_FAIL = 2
_STATE_PRUNED = 3
_STATE_WAITING = 4

_STATE_NAMES = {
    _STATE_RUNNING: "RUNNING",
    _STATE_COMPLETE: "COMPLETE",
    _STATE_FAIL: "FAIL",
    _STATE_PRUNED: "PRUNED",
    _STATE_WAITING: "WAITING",
}

# op_code constants used in JournalStorage JSONL
_OP_STUDY_INFO = 0
_OP_CREATE_TRIAL = 4
_OP_SET_TRIAL_PARAM = 5
_OP_SET_TRIAL_STATE = 6


@dataclass
class TrialInfo:
    """Parsed info for a single trial."""

    trial_id: int
    seed: int | None = None
    ablation: str | None = None
    state: int = _STATE_WAITING
    value: float | None = None


@dataclass
class AblationSummary:
    """Aggregate status for an entire ablation journal."""

    study_name: str = ""
    total_trials: int = 0
    by_state: dict[int, int] = field(default_factory=dict)
    # (seed, ablation) -> TrialInfo  (last write wins)
    trials: dict[tuple[int, str], TrialInfo] = field(default_factory=dict)


def parse_journal(journal_path: str | Path) -> AblationSummary:
    """Parse an Optuna JournalStorage JSONL file into a summary."""
    journal_path = Path(journal_path)
    lines = journal_path.read_text().strip().splitlines()

    summary = AblationSummary()

    # trial_id -> TrialInfo  (accumulate then collapse to unique identity)
    trial_map: dict[int, TrialInfo] = {}

    for raw in lines:
        if not raw.strip():
            continue
        entry = json.loads(raw)
        op = entry.get("op_code")

        if op == _OP_STUDY_INFO:
            study_name = entry.get("study_name", "")
            if study_name and not summary.study_name:
                summary.study_name = study_name

        elif op == _OP_SET_TRIAL_PARAM:
            tid = entry["trial_id"]
            param_name = entry.get("param_name")
            info = trial_map.setdefault(tid, TrialInfo(trial_id=tid))
            # Decode categorical index from the distribution choices
            dist_raw = entry.get("distribution", "")
            if isinstance(dist_raw, str):
                try:
                    dist = json.loads(dist_raw)
                    choices = dist.get("attributes", {}).get("choices", [])
                except (json.JSONDecodeError, KeyError):
                    choices = []
            else:
                choices = []
            idx = entry.get("param_value_internal", 0)
            if param_name == "ablation" and choices:
                info.ablation = str(choices[idx]) if idx < len(choices) else None
            elif param_name == "seed" and choices:
                info.seed = int(choices[idx]) if idx < len(choices) else None

        elif op == _OP_SET_TRIAL_STATE:
            tid = entry["trial_id"]
            state = entry.get("state", _STATE_WAITING)
            info = trial_map.get(tid)
            if info is None:
                info = TrialInfo(trial_id=tid)
                trial_map[tid] = info
            info.state = state
            vals = entry.get("values")
            if vals is not None:
                info.value = vals[0] if isinstance(vals, list) else vals

    # Collapse to unique (seed, ablation) — keep the latest terminal state
    seen: dict[tuple[int, str], TrialInfo] = {}
    for info in trial_map.values():
        if info.seed is None or info.ablation is None:
            continue
        key = (info.seed, info.ablation)
        existing = seen.get(key)
        if existing is None:
            seen[key] = info
        else:
            # Prefer terminal states; within terminal, prefer later entry
            if info.state in (_STATE_COMPLETE, _STATE_FAIL):
                if existing.state not in (_STATE_COMPLETE, _STATE_FAIL):
                    seen[key] = info
                # If both terminal, keep the one with a value (completed)
                elif info.value is not None and existing.value is None:
                    seen[key] = info

    summary.trials = seen
    summary.total_trials = len(trial_map)
    for info in seen.values():
        summary.by_state[info.state] = summary.by_state.get(info.state, 0) + 1

    return summary


def format_status(summary: AblationSummary) -> str:
    """Format a summary into a printable status table."""
    lines: list[str] = []

    lines.append(f"Study: {summary.study_name}")
    total_unique = len(summary.trials)
    completed = summary.by_state.get(_STATE_COMPLETE, 0)
    running = summary.by_state.get(_STATE_RUNNING, 0)
    failed = summary.by_state.get(_STATE_FAIL, 0)
    pruned = summary.by_state.get(_STATE_PRUNED, 0)
    waiting = summary.by_state.get(_STATE_WAITING, 0)

    lines.append(
        f"Trials: {completed}/{total_unique} complete, "
        f"{running} running, {failed} failed, {pruned} pruned, {waiting} waiting"
    )

    # Group by ablation, show per-ablation seed coverage
    ablation_seeds: dict[str, dict[int, tuple[str, float | None]]] = {}
    for (seed, ablation), info in sorted(summary.trials.items()):
        state_name = _STATE_NAMES.get(info.state, f"?{info.state}")
        ablation_seeds.setdefault(ablation, {})[seed] = (state_name, info.value)

    if not ablation_seeds:
        lines.append("(no trials found)")
        return "\n".join(lines)

    # Determine column widths
    abl_width = max(len(a) for a in ablation_seeds)
    seed_col = 5  # "seed" header

    header = f"  {'ablation':<{abl_width}}  {'seed':>{seed_col}}  state      value"
    sep = f"  {'-' * abl_width}  {'-' * seed_col}  ----------  ----------"
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for abl in sorted(ablation_seeds):
        seeds_dict = ablation_seeds[abl]
        first = True
        for seed in sorted(seeds_dict):
            state_name, val = seeds_dict[seed]
            val_str = f"{val:.6f}" if val is not None else "-"
            abl_label = abl if first else ""
            lines.append(
                f"  {abl_label:<{abl_width}}  {seed:>{seed_col}}  {state_name:<10}  {val_str}"
            )
            first = False

    return "\n".join(lines)


def ablation_journal_status(journal_path: str | Path) -> str:
    """One-shot: parse journal and return formatted status string."""
    return format_status(parse_journal(journal_path))


@click.command()
@click.argument("journal_file", type=click.Path(exists=True, path_type=Path))
def main(journal_file: Path) -> None:
    """Print ablation campaign status from an Optuna journal file."""
    click.echo(ablation_journal_status(journal_file))


if __name__ == "__main__":
    main()
