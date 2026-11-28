"""Campaign-declared initialization checkpoints.

A campaign declares named fine-tuning sources once (``init_checkpoints``) and
conditions reference them by name (``init_checkpoint``). This module holds only
the declaration contract and the pinned-input layout, using nothing beyond the
standard library and PyYAML, so render, prepare and the Snakefile can share it
with the campaign loader without importing the training stack.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_INIT_DIR = Path("inputs/campaign_init")
INIT_CHECKPOINT_PATH_KEY = "training.init_checkpoint_path"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_init_checkpoints(raw: object) -> dict[str, Path]:
    """Validate the campaign's ``init_checkpoints`` mapping (absent means none)."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("init_checkpoints must be a mapping of name to absolute path")
    declared: dict[str, Path] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError(
                f"init_checkpoints name {name!r} must match [A-Za-z0-9][A-Za-z0-9_-]*"
            )
        if not isinstance(value, str) or not value:
            raise ValueError(f"init_checkpoints.{name} must be a non-empty path string")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(
                f"init_checkpoints.{name} must be an absolute path: {value}"
            )
        declared[name] = path
    return declared


def condition_init_checkpoint(raw: object, *, field: str) -> str | None:
    """Validate one condition's optional ``init_checkpoint`` name."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a declared init_checkpoints name")
    return raw


def validate_init_checkpoint_references(
    declared: Mapping[str, Path], references: Mapping[str, str | None]
) -> None:
    """Require every reference to be declared and every declaration to be used."""
    for condition, name in references.items():
        if name is not None and name not in declared:
            raise ValueError(
                f"condition {condition!r} references undeclared init_checkpoint "
                f"{name!r}; declared: {sorted(declared)}"
            )
    unused = sorted(set(declared) - {name for name in references.values() if name})
    if unused:
        raise ValueError(
            f"init_checkpoints declared but used by no condition: {', '.join(unused)}"
        )


def read_campaign_init_checkpoints(campaign_path: Path) -> tuple[str, dict[str, Path]]:
    """Return ``(campaign_id, declared checkpoints)`` from a campaign file.

    Applies the same declaration checks as the campaign loader without
    hydrating any configuration.
    """
    try:
        root: Any = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read campaign {campaign_path}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ValueError(f"campaign must be a mapping: {campaign_path}")
    campaign_id = root.get("campaign_id")
    if (
        not isinstance(campaign_id, str)
        or _SAFE_COMPONENT.fullmatch(campaign_id) is None
    ):
        raise ValueError("campaign_id must be one safe path component")
    declared = parse_init_checkpoints(root.get("init_checkpoints"))
    conditions = root.get("conditions")
    if not isinstance(conditions, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str)
        for item in conditions
    ):
        raise ValueError("conditions must be a list of named mappings")
    validate_init_checkpoint_references(
        declared,
        {
            item["name"]: condition_init_checkpoint(
                item.get("init_checkpoint"),
                field=f"condition {item['name']!r} init_checkpoint",
            )
            for item in conditions
        },
    )
    return campaign_id, declared


def pinned_init_checkpoint_relative(campaign_id: str, name: str) -> Path:
    """Run-relative location of a pinned campaign checkpoint."""
    return CAMPAIGN_INIT_DIR / campaign_id / f"{name}.pt"


def init_checkpoint_identity_value(sha256: str) -> str:
    """Path-independent stand-in for ``training.init_checkpoint_path`` in identities."""
    return f"sha256:{sha256}"
