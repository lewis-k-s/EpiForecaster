from __future__ import annotations

import json
from pathlib import Path

from data.crossval_manifest import load_crossval_fold_manifest
from scripts.analysis.extend_crossval_manifest import (
    RegionAttributes,
    build_extended_manifest,
)


def _attributes() -> RegionAttributes:
    regions = [f"b{i}" for i in range(4)] + [f"a{i}" for i in range(4)]
    return RegionAttributes(
        population={
            **{f"b{i}": 50_000.0 for i in range(4)},
            **{f"a{i}": 4_000.0 for i in range(4)},
        },
        # a0 shares the plant of b3, so its wastewater series is b3's twin.
        plants={"b3": frozenset({"P"}), "a0": frozenset({"P"})},
        comarca={region: "01" for region in regions},
    )


def test_extension_keeps_base_folds_and_colocates_wastewater_twins(
    tmp_path: Path,
) -> None:
    base_payload = {
        "run_id": "real",
        "split_seed": 42,
        "folds": [
            {"fold_index": 0, "region_ids": ["b0", "b1"]},
            {"fold_index": 1, "region_ids": ["b2", "b3"]},
        ],
    }
    targets = ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "b3"]

    payload = build_extended_manifest(
        base_payload=base_payload,
        target_ids=targets,
        run_id="real",
        attributes=_attributes(),
        provenance={"dataset_path": "data/processed/test.zarr.zip"},
    )

    folds = {entry["fold_index"]: entry["region_ids"] for entry in payload["folds"]}
    assert folds[0][:2] == ["b0", "b1"]
    assert folds[1][:2] == ["b2", "b3"]
    assert "a0" in folds[1]
    assert payload["diagnostics"]["extension"]["cross_fold_pairs"]["ww_base_added"] == 0
    assert payload["region_sets"]["base"] == ["b0", "b1", "b2", "b3"]
    assert payload["region_sets"]["pop_lt_5k"] == ["a0", "a1", "a2", "a3"]

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    manifest = load_crossval_fold_manifest(
        path, valid_region_ids=targets, expected_num_folds=2, expected_run_id="real"
    )
    assert sorted(map(len, manifest.folds)) == [4, 4]
    assert manifest.region_sets["added"] == frozenset({"a0", "a1", "a2", "a3"})
