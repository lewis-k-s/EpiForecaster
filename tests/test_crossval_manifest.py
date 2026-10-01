from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.crossval_manifest import (
    hash_region_ids,
    load_crossval_fold_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(
    path: Path,
    *,
    region_ids: list[str],
    folds: list[list[str]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_folds": len(folds),
                "run_id": "real",
                "split_seed": 42,
                "region_id_hash": hash_region_ids(region_ids),
                "folds": [
                    {"fold_index": index, "region_ids": fold}
                    for index, fold in enumerate(folds)
                ],
            }
        )
    )


def test_load_crossval_manifest_validates_exact_once_coverage(
    tmp_path: Path,
) -> None:
    region_ids = ["a", "b", "c", "d"]
    path = tmp_path / "manifest.json"
    _write_manifest(
        path,
        region_ids=region_ids,
        folds=[["a", "c"], ["b", "d"]],
    )

    manifest = load_crossval_fold_manifest(
        path,
        valid_region_ids=region_ids,
        expected_num_folds=2,
        expected_run_id="real",
    )

    assert manifest.folds == (("a", "c"), ("b", "d"))
    assert manifest.split_seed == 42


def test_load_crossval_manifest_rejects_duplicate_assignment(
    tmp_path: Path,
) -> None:
    region_ids = ["a", "b", "c", "d"]
    path = tmp_path / "manifest.json"
    _write_manifest(
        path,
        region_ids=region_ids,
        folds=[["a", "b"], ["b", "d"]],
    )

    with pytest.raises(ValueError, match="more than once"):
        load_crossval_fold_manifest(
            path,
            valid_region_ids=region_ids,
            expected_num_folds=2,
            expected_run_id="real",
        )


def test_load_crossval_manifest_rejects_dataset_identity_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(
        path,
        region_ids=["a", "b", "c", "different"],
        folds=[["a", "b"], ["c", "d"]],
    )

    with pytest.raises(ValueError, match="region_id_hash"):
        load_crossval_fold_manifest(
            path,
            valid_region_ids=["a", "b", "c", "d"],
            expected_num_folds=2,
            expected_run_id="real",
        )
