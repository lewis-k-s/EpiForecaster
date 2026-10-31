"""Audit helpers for historical checkpoint reuse."""

from __future__ import annotations

from scripts.research.audit_loss_checkpoint_reuse import (
    IDENTITY_PATTERN,
    _dirty_training_paths,
)


def test_identity_pattern_accepts_names_with_and_without_retry_suffix() -> None:
    for name in ("normal-crps-m0-f1-s52-r1", "student_t-nll-m1-f4-s53"):
        assert IDENTITY_PATTERN.fullmatch(name) is not None
    assert IDENTITY_PATTERN.fullmatch("normal-crps-m0-f1-s52-rx") is None


def test_dirty_filter_allows_only_docs_and_tex_directories() -> None:
    harmless = [" M docs/notes.md", "?? tex/EpiForecaster/a.tex", "A  docs/new.md"]
    assert _dirty_training_paths(harmless) == []
    assert _dirty_training_paths([" M texture/model.py"]) == ["texture/model.py"]
    assert _dirty_training_paths(["M  training/trainer.py"]) == ["training/trainer.py"]
    assert _dirty_training_paths(["?? models/new.py"]) == ["models/new.py"]
    assert _dirty_training_paths(["R  docs/a.md -> models/a.py"]) == ["models/a.py"]
