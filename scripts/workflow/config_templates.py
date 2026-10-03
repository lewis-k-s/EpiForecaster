"""Resolve source-owned YAML bases and named variants before run preparation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def merge_templates(base: dict, update: dict) -> dict:
    """Merge mappings; replace lists except dotted-key override lists."""
    result = deepcopy(base)
    for key, value in update.items():
        previous = result.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            result[key] = merge_templates(previous, value)
        elif (
            key == "overrides"
            and isinstance(previous, list)
            and isinstance(value, list)
        ):
            options = {}
            for option in [*previous, *value]:
                if not isinstance(option, str) or "=" not in option:
                    raise ValueError(
                        "Template overrides must be dotted-key assignments"
                    )
                options[option.split("=", 1)[0]] = option
            result[key] = list(options.values())
        else:
            result[key] = deepcopy(value)
    return result


def load_config_template(
    path: str | Path,
    *,
    root: Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Load a template with source-root-relative bases and an explicit variant.

    Base paths must stay inside the source root. Other lists replace inherited
    lists. An empty overrides list retains inherited options. Variant selection
    is required for templates with variants; it is never a runtime default.
    """
    filename = Path(path).resolve()
    if root is None:
        root = next(
            (
                parent
                for parent in filename.parents
                if (parent / "pyproject.toml").is_file()
            ),
            Path(__file__).resolve().parents[2],
        )
    source_root = root.resolve()

    def read(filename: Path, stack: tuple[Path, ...]) -> dict:
        filename = filename.resolve()
        if filename in stack:
            raise ValueError(f"Cyclic template bases: {filename}")
        document = yaml.safe_load(filename.read_text())
        if not isinstance(document, dict):
            raise ValueError(f"Configuration template must be a mapping: {filename}")
        return resolve(document, (*stack, filename))

    def resolve(document: dict, stack: tuple[Path, ...]) -> dict:
        document = deepcopy(document)
        bases = document.pop("extends", [])
        if isinstance(bases, str):
            bases = [bases]
        if not isinstance(bases, list) or any(not isinstance(p, str) for p in bases):
            raise ValueError("Template extends must contain source-relative paths")
        result = {}
        for reference in bases:
            relative = Path(reference)
            filename = (source_root / relative).resolve()
            if relative.is_absolute() or not filename.is_relative_to(source_root):
                raise ValueError(f"Template base escapes the source root: {reference}")
            base = read(filename, stack)
            if "variants" in base:
                raise ValueError("A template base cannot declare variants")
            result = merge_templates(result, base)
        return merge_templates(result, document)

    result = read(Path(path), ())
    variants = result.pop("variants", None)
    if variants is not None:
        if not isinstance(variants, dict) or not variants:
            raise ValueError("Template variants must be a nonempty mapping")
        if variant not in variants:
            raise ValueError(f"Select a template variant from {sorted(variants)}")
        selected = variants[variant]
        if not isinstance(selected, dict):
            raise ValueError("Template variant must be a mapping")
        result = merge_templates(result, resolve(selected, (Path(path).resolve(),)))
        if "variants" in result:
            raise ValueError("Nested template variants are not supported")
    elif variant is not None:
        raise ValueError(f"Template does not declare variant {variant!r}")
    return result


def apply_template_overrides(document: dict, overrides: list[str]) -> dict:
    """Apply preparation-only dotted-key assignments to a resolved workflow."""
    result = deepcopy(document)
    for option in overrides:
        if "=" not in option:
            raise ValueError("Workflow overrides must be dotted-key assignments")
        name, value = option.split("=", 1)
        parts = name.split(".")
        if any(not part or not part.replace("_", "").isalnum() for part in parts):
            raise ValueError(f"Invalid workflow override path: {name!r}")
        if parts[0] in {"run", "source", "container", "extends", "variants"}:
            raise ValueError(f"Workflow override cannot set {parts[0]!r}")
        parent = result
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
            if not isinstance(parent, dict):
                raise ValueError(f"Workflow override cannot traverse {name!r}")
        parent[parts[-1]] = yaml.safe_load(value)
    return result
