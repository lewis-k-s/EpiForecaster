"""Validation and loading for persisted cross-validation region folds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrossvalFoldManifest:
    """Validated, exactly-once region membership for cross-validation folds."""

    path: Path
    folds: tuple[tuple[str, ...], ...]
    region_id_hash: str
    run_id: str
    split_seed: int


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
    )
