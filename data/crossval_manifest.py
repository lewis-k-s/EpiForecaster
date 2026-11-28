"""Validation and loading for persisted cross-validation region folds."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrossvalFoldManifest:
    """Validated, exactly-once region membership for cross-validation folds.

    ``region_sets`` are named, static subsets of the assigned regions. They
    stratify held-out evaluation and can limit the training population.
    """

    path: Path
    folds: tuple[tuple[str, ...], ...]
    region_id_hash: str
    run_id: str
    split_seed: int
    region_sets: dict[str, frozenset[str]]


_REGION_SET_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


def parse_region_sets(
    payload: dict, *, assigned_region_ids: set[str]
) -> dict[str, frozenset[str]]:
    """Validate a manifest's optional ``region_sets`` mapping."""
    raw = payload.get("region_sets", {})
    if not isinstance(raw, dict):
        raise ValueError("Cross-validation manifest 'region_sets' must be a mapping")
    region_sets: dict[str, frozenset[str]] = {}
    for name, members in raw.items():
        if not isinstance(name, str) or _REGION_SET_NAME.fullmatch(name) is None:
            raise ValueError(f"Invalid region set name: {name!r}")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Region set {name!r} must be a non-empty list")
        member_ids = [str(region_id) for region_id in members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError(f"Region set {name!r} repeats a region")
        unknown = sorted(set(member_ids) - assigned_region_ids)
        if unknown:
            raise ValueError(
                f"Region set {name!r} has regions outside the folds: {unknown[:5]}"
            )
        region_sets[name] = frozenset(member_ids)
    return region_sets


def load_manifest_region_sets(path: Path) -> dict[str, frozenset[str]]:
    """Read the validated region sets of a persisted manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assigned = {
        str(region_id)
        for entry in payload.get("folds", [])
        for region_id in entry["region_ids"]
    }
    return parse_region_sets(payload, assigned_region_ids=assigned)


def region_sets_for_split(
    split_name: str, manifest_path: str | None
) -> dict[str, frozenset[str]] | None:
    """Region sets to score a split on: only the test split, only with a manifest."""
    if split_name.lower() != "test" or not manifest_path:
        return None
    return load_manifest_region_sets(Path(manifest_path))


def hash_region_ids(region_ids: list[str]) -> str:
    """Hash ordered region IDs using the manifest's stable newline contract."""
    return hashlib.sha256("\n".join(region_ids).encode("utf-8")).hexdigest()


def load_crossval_fold_manifest(
    path: Path,
    *,
    valid_region_ids: list[str],
    expected_num_folds: int,
    expected_run_id: str,
) -> CrossvalFoldManifest:
    """Load a versioned manifest and enforce exact once-only target coverage."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported cross-validation manifest version in {path}: "
            f"{payload.get('version')!r}"
        )

    num_folds = int(payload.get("num_folds", -1))
    if num_folds != expected_num_folds:
        raise ValueError(
            "Cross-validation manifest fold count does not match configuration: "
            f"{num_folds} != {expected_num_folds}"
        )

    run_id = str(payload.get("run_id", ""))
    if run_id != expected_run_id:
        raise ValueError(
            "Cross-validation manifest run_id does not match configuration: "
            f"{run_id!r} != {expected_run_id!r}"
        )

    entries = payload.get("folds")
    if not isinstance(entries, list):
        raise ValueError("Cross-validation manifest 'folds' must be a list")
    entries = sorted(entries, key=lambda entry: int(entry["fold_index"]))
    fold_indices = [int(entry["fold_index"]) for entry in entries]
    expected_indices = list(range(expected_num_folds))
    if fold_indices != expected_indices:
        raise ValueError(
            "Cross-validation manifest fold indices must be contiguous: "
            f"{fold_indices} != {expected_indices}"
        )

    folds = tuple(
        tuple(str(region_id) for region_id in entry["region_ids"]) for entry in entries
    )
    assigned_region_ids = [
        region_id for fold_region_ids in folds for region_id in fold_region_ids
    ]
    assignment_counts = Counter(assigned_region_ids)
    duplicate_ids = sorted(
        region_id for region_id, count in assignment_counts.items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "Cross-validation manifest assigns regions more than once: "
            f"{duplicate_ids[:5]}"
        )

    valid_set = set(valid_region_ids)
    assigned_set = set(assigned_region_ids)
    missing_ids = sorted(valid_set - assigned_set)
    unknown_ids = sorted(assigned_set - valid_set)
    if missing_ids or unknown_ids:
        raise ValueError(
            "Cross-validation manifest does not exactly cover valid targets: "
            f"missing={missing_ids[:5]}, unknown={unknown_ids[:5]}"
        )

    expected_small_size, larger_fold_count = divmod(
        len(valid_region_ids),
        expected_num_folds,
    )
    expected_sizes = sorted(
        [expected_small_size + 1] * larger_fold_count
        + [expected_small_size] * (expected_num_folds - larger_fold_count)
    )
    actual_sizes = sorted(len(fold) for fold in folds)
    if actual_sizes != expected_sizes:
        raise ValueError(
            "Cross-validation manifest fold sizes are not balanced: "
            f"{actual_sizes} != {expected_sizes}"
        )

    expected_hash = hash_region_ids(valid_region_ids)
    manifest_hash = str(payload.get("region_id_hash", ""))
    if manifest_hash != expected_hash:
        raise ValueError(
            "Cross-validation manifest region_id_hash does not match the dataset: "
            f"{manifest_hash!r} != {expected_hash!r}"
        )

    return CrossvalFoldManifest(
        path=path,
        folds=folds,
        region_id_hash=manifest_hash,
        run_id=run_id,
        split_seed=int(payload["split_seed"]),
        region_sets=parse_region_sets(payload, assigned_region_ids=assigned_set),
    )
