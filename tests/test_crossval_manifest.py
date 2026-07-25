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


def test_checked_representative_manifest_has_acceptance_receipt() -> None:
    path = REPO_ROOT / "configs/crossval/representative_10fold_mobility_refined.json"
    payload = json.loads(path.read_text())
    valid_region_ids = sorted(
        region_id for fold in payload["folds"] for region_id in fold["region_ids"]
    )

    manifest = load_crossval_fold_manifest(
        path,
        valid_region_ids=valid_region_ids,
        expected_num_folds=10,
        expected_run_id="real",
    )
    diagnostics = payload["diagnostics"]["refinement"]

    assert len(manifest.folds) == 10
    assert sorted(map(len, manifest.folds)) == [16] * 9 + [17]
    assert diagnostics["swap_count"] == 15
    assert diagnostics["final_minimum_target_retained_flow"] >= 0.75
    assert (
        diagnostics["final_worst_fold_internal_flow_fraction"]
        <= diagnostics["outlier_internal_flow_ceiling"]
    )
    assert (
        diagnostics["final_balance_squared_count_deviation"]
        <= diagnostics["baseline_balance_squared_count_deviation"]
    )
