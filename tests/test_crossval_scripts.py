from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_submit_crossval_derives_array_spec_from_seed_count() -> None:
    result = _run_bash(
        'source scripts/submit_crossval.sh >/dev/null; '
        'derive_array_spec_from_seeds "42 43 44"'
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "0-2"


def test_submit_crossval_parses_explicit_array_arg() -> None:
    result = _run_bash(
        "source scripts/submit_crossval.sh >/dev/null; "
        'parse_explicit_array_arg --qos debug --array=3-7'
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "3-7"


def test_run_crossval_resolves_relative_output_log_dir() -> None:
    result = _run_bash(
        "source scripts/run_crossval.sbatch >/dev/null; "
        'resolve_output_log_root "/repo" "outputs/training"'
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "/repo/outputs/training"


def test_run_crossval_preserves_absolute_output_log_dir() -> None:
    result = _run_bash(
        "source scripts/run_crossval.sbatch >/dev/null; "
        'resolve_output_log_root "/repo" "/scratch/training"'
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "/scratch/training"
