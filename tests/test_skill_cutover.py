from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETAINED_SKILLS = {
    "hpsearch-analyze",
    "model-training",
    "training-dynamics-analyze",
}
REMOVED_COMMANDS = {
    "dead-code-analyze",
    "gradnorm-analyze",
    "loss-curve-critic",
    "perf-analyze",
    "step-breakdown",
}


def test_project_exposes_exact_retained_skill_set() -> None:
    skills_root = REPO_ROOT / ".agents/skills"
    observed = {
        path.parent.name for path in skills_root.glob("*/SKILL.md")
    }
    assert observed == RETAINED_SKILLS
    assert not (REPO_ROOT / "skills/training-loss-analysis").exists()


def test_removed_analysis_commands_are_not_registered() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]

    assert REMOVED_COMMANDS.isdisjoint(scripts)
    assert scripts["training-dynamics-analyze"].endswith(
        "analyze_training_dynamics:main"
    )
    assert scripts["hpsearch-analyze"].endswith(
        "analyze_hpsearch_journal:main"
    )
    assert {"train", "eval"}.issubset(scripts)


def test_core_profiler_and_ruff_capabilities_remain() -> None:
    trainer = (REPO_ROOT / "training/epiforecaster_trainer.py").read_text()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert "tensorboard_trace_handler" in trainer
    assert any(
        requirement.startswith("ruff")
        for requirement in pyproject["project"]["optional-dependencies"]["dev"]
    )
