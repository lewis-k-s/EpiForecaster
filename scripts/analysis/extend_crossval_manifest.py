"""Extend a persisted cross-validation manifest to a larger target-region set.

Every region of the base manifest keeps its fold, so the held-out regions of
the base set are identical in both manifests and models trained on either
target set can be compared on that subset. Added regions are placed with their
source twins: a region sharing a wastewater plant with a base region joins that
region's fold, so its (near-identical) wastewater series never trains a model
that is tested on the twin. Comarca mates, which share the comarca-allocated
death series, are the secondary affinity. Fold sizes stay balanced to within one
region, as ``load_crossval_fold_manifest`` requires.

The output manifest carries ``region_sets`` for evaluation strata and for
``training.crossval_region_set``: ``base`` (the original targets), ``added``,
and static population bins.

Usage:
    uv run python scripts/analysis/extend_crossval_manifest.py \
        --base-manifest configs/crossval/representative_10fold_mobility_refined.json \
        --dataset data/processed/real_with_id_v2.zarr.zip \
        --edar-map data/files/edar_muni_edges.nc \
        --municipality-reference data/files/municipality_id_reference.csv \
        --output configs/crossval/representative_10fold_pop3000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

from data.crossval_manifest import hash_region_ids
from data.preprocess.zarr_layout import open_zarr_store

# Weights of the placement objective for one added region in one fold.
WW_BASE_TWIN_WEIGHT = 1000.0
COMARCA_BASE_MATE_WEIGHT = 10.0
WW_ADDED_TWIN_WEIGHT = 1.0
COMARCA_ADDED_MATE_WEIGHT = 0.1
# Static population strata (lower bounds inclusive) used as region sets.
POPULATION_BINS = (
    ("pop_lt_5k", 0.0, 5_000.0),
    ("pop_5k_20k", 5_000.0, 20_000.0),
    ("pop_20k_100k", 20_000.0, 100_000.0),
    ("pop_ge_100k", 100_000.0, float("inf")),
)


@dataclass(frozen=True)
class RegionAttributes:
    """Static attributes that drive placement and strata."""

    population: dict[str, float]
    plants: dict[str, frozenset[str]]
    comarca: dict[str, str]


def _shares(a: frozenset[str], b: frozenset[str]) -> bool:
    return bool(a & b)


def _affinity(
    region: str,
    fold_members: list[str],
    attributes: RegionAttributes,
    *,
    twin_weight: float,
    mate_weight: float,
) -> float:
    plants = attributes.plants.get(region, frozenset())
    comarca = attributes.comarca.get(region)
    score = 0.0
    for other in fold_members:
        if plants and _shares(plants, attributes.plants.get(other, frozenset())):
            score += twin_weight
        if comarca is not None and attributes.comarca.get(other) == comarca:
            score += mate_weight
    return score


def _objective(
    assignment: dict[str, int],
    base_folds: list[list[str]],
    attributes: RegionAttributes,
) -> float:
    added_by_fold: dict[int, list[str]] = {}
    for region, fold in assignment.items():
        added_by_fold.setdefault(fold, []).append(region)
    total = 0.0
    for region, fold in assignment.items():
        total += _affinity(
            region,
            base_folds[fold],
            attributes,
            twin_weight=WW_BASE_TWIN_WEIGHT,
            mate_weight=COMARCA_BASE_MATE_WEIGHT,
        )
        total += _affinity(
            region,
            [other for other in added_by_fold[fold] if other != region],
            attributes,
            twin_weight=WW_ADDED_TWIN_WEIGHT,
            mate_weight=COMARCA_ADDED_MATE_WEIGHT,
        )
    return total


def assign_added_regions(
    base_folds: list[list[str]],
    added: list[str],
    attributes: RegionAttributes,
) -> dict[str, int]:
    """Place added regions in balanced folds, maximizing source-twin affinity.

    Base-region affinity is solved exactly as a capacity-constrained
    assignment; pairwise swaps then co-locate added regions that are twins of
    each other without changing fold sizes. The result is deterministic.
    """
    num_folds = len(base_folds)
    total = sum(len(fold) for fold in base_folds) + len(added)
    small, large_count = divmod(total, num_folds)
    # Base folds already differ by at most one; the larger base folds take the
    # smaller final sizes first so every fold ends balanced.
    order = sorted(range(num_folds), key=lambda fold: (-len(base_folds[fold]), fold))
    final_size = {fold: small for fold in range(num_folds)}
    for fold in order[num_folds - large_count :]:
        final_size[fold] = small + 1
    capacity = {fold: final_size[fold] - len(base_folds[fold]) for fold in order}
    if any(value < 0 for value in capacity.values()):
        raise ValueError("Base folds are too unbalanced to extend")
    slots = [fold for fold in range(num_folds) for _ in range(capacity[fold])]
    if len(slots) != len(added):
        raise ValueError("Slot count does not match the added regions")
    affinity = np.array(
        [
            [
                _affinity(
                    region,
                    base_folds[fold],
                    attributes,
                    twin_weight=WW_BASE_TWIN_WEIGHT,
                    mate_weight=COMARCA_BASE_MATE_WEIGHT,
                )
                for fold in slots
            ]
            for region in added
        ]
    )
    rows, columns = linear_sum_assignment(-affinity)
    assignment = {added[row]: slots[column] for row, column in zip(rows, columns)}

    current = _objective(assignment, base_folds, attributes)
    improved = True
    while improved:
        improved = False
        for i, first in enumerate(added):
            for second in added[i + 1 :]:
                if assignment[first] == assignment[second]:
                    continue
                assignment[first], assignment[second] = (
                    assignment[second],
                    assignment[first],
                )
                candidate = _objective(assignment, base_folds, attributes)
                if candidate > current + 1e-9:
                    current = candidate
                    improved = True
                else:
                    assignment[first], assignment[second] = (
                        assignment[second],
                        assignment[first],
                    )
    return assignment


def _cross_fold_pairs(
    folds: list[list[str]],
    left: set[str],
    right: set[str],
    related,
) -> int:
    """Count related (left, right) pairs that sit in different folds."""
    fold_of = {region: index for index, fold in enumerate(folds) for region in fold}
    return sum(
        1
        for a in left
        for b in right
        if a != b and related(a, b) and fold_of[a] != fold_of[b]
    )


def build_extended_manifest(
    *,
    base_payload: dict,
    target_ids: list[str],
    run_id: str,
    attributes: RegionAttributes,
    provenance: dict,
) -> dict:
    """Return the extended manifest payload."""
    base_folds = [
        [str(region) for region in entry["region_ids"]]
        for entry in sorted(base_payload["folds"], key=lambda e: int(e["fold_index"]))
    ]
    base_ids = {region for fold in base_folds for region in fold}
    target_set = set(target_ids)
    missing = sorted(base_ids - target_set)
    if missing:
        raise ValueError(f"Base targets absent from the new target set: {missing[:5]}")
    added = [region for region in target_ids if region not in base_ids]
    assignment = assign_added_regions(base_folds, added, attributes)
    folds = [
        list(fold) + [region for region in added if assignment[region] == index]
        for index, fold in enumerate(base_folds)
    ]

    def ww_twin(a: str, b: str) -> bool:
        return _shares(
            attributes.plants.get(a, frozenset()), attributes.plants.get(b, frozenset())
        )

    def comarca_mate(a: str, b: str) -> bool:
        return attributes.comarca.get(a) is not None and attributes.comarca.get(
            a
        ) == attributes.comarca.get(b)

    added_set = set(added)
    region_sets = {"base": sorted(base_ids), "added": sorted(added_set)}
    for name, low, high in POPULATION_BINS:
        members = sorted(
            region
            for region in target_ids
            if low <= attributes.population[region] < high
        )
        if members:
            region_sets[name] = members
    return {
        "version": 1,
        "strategy": "base-preserving-extension",
        "dataset_path": provenance["dataset_path"],
        "run_id": run_id,
        "split_seed": int(base_payload["split_seed"]),
        "num_folds": len(folds),
        "region_id_hash": hash_region_ids(target_ids),
        "diagnostics": {
            "extension": {
                **provenance,
                "base_region_count": len(base_ids),
                "added_region_count": len(added),
                "fold_sizes": [len(fold) for fold in folds],
                "weights": {
                    "ww_base_twin": WW_BASE_TWIN_WEIGHT,
                    "comarca_base_mate": COMARCA_BASE_MATE_WEIGHT,
                    "ww_added_twin": WW_ADDED_TWIN_WEIGHT,
                    "comarca_added_mate": COMARCA_ADDED_MATE_WEIGHT,
                },
                # Pairs split across folds: one region can train while its twin
                # is held out. Base-base pairs are inherited from the base split.
                "cross_fold_pairs": {
                    "ww_base_added": _cross_fold_pairs(
                        folds, base_ids, added_set, ww_twin
                    ),
                    "ww_added_added": _cross_fold_pairs(
                        folds, added_set, added_set, ww_twin
                    )
                    // 2,
                    "ww_base_base": _cross_fold_pairs(
                        folds, base_ids, base_ids, ww_twin
                    )
                    // 2,
                    "comarca_base_added": _cross_fold_pairs(
                        folds, base_ids, added_set, comarca_mate
                    ),
                    "comarca_base_base": _cross_fold_pairs(
                        folds, base_ids, base_ids, comarca_mate
                    )
                    // 2,
                },
                "population_bins": {
                    name: [low, None if np.isinf(high) else high]
                    for name, low, high in POPULATION_BINS
                },
            }
        },
        "region_sets": region_sets,
        "folds": [
            {"fold_index": index, "region_ids": fold}
            for index, fold in enumerate(folds)
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_attributes(
    dataset: xr.Dataset, edar_map: Path, municipality_reference: Path
) -> RegionAttributes:
    """Read population, wastewater plants, and comarca for every region."""
    region_ids = [str(region) for region in dataset["region_id"].values]
    population = dataset["population"]
    if "run_id" in population.dims:
        population = population.isel(run_id=0)
    contributions = xr.open_dataarray(edar_map).fillna(0.0)
    homes = [str(home) for home in contributions["home"].values]
    edars = [str(edar) for edar in contributions["edar_id"].values]
    values = contributions.transpose("home", "edar_id").values
    plants = {
        home: frozenset(edar for edar, value in zip(edars, row) if value > 0)
        for home, row in zip(homes, values)
    }
    reference = pd.read_csv(
        municipality_reference, dtype={"muni_code_5digit": str, "comarca_code": str}
    )
    comarca = dict(
        zip(
            reference["muni_code_5digit"].str.zfill(5),
            reference["comarca_code"],
            strict=True,
        )
    )
    return RegionAttributes(
        population=dict(zip(region_ids, population.values.astype(float), strict=True)),
        plants={region: plants[region] for region in region_ids if region in plants},
        comarca={region: comarca[region] for region in region_ids if region in comarca},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--edar-map", type=Path, required=True)
    parser.add_argument("--municipality-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_payload = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    dataset = open_zarr_store(args.dataset)
    valid = dataset["valid_targets"]
    if "run_id" in valid.dims:
        if valid.sizes["run_id"] != 1:
            raise ValueError("Expected a single-run real dataset")
        run_id = str(dataset["run_id"].values[0])
        valid = valid.isel(run_id=0)
    else:
        run_id = str(base_payload["run_id"])
    region_ids = [str(region) for region in dataset["region_id"].values]
    target_ids = [
        region for region, flag in zip(region_ids, valid.values.astype(bool)) if flag
    ]
    attributes = load_attributes(dataset, args.edar_map, args.municipality_reference)
    missing_comarca = sorted(set(target_ids) - set(attributes.comarca))
    if missing_comarca:
        raise ValueError(f"Targets without a comarca: {missing_comarca[:5]}")
    payload = build_extended_manifest(
        base_payload=base_payload,
        target_ids=target_ids,
        run_id=run_id,
        attributes=attributes,
        provenance={
            "dataset_path": args.dataset.as_posix(),
            "base_manifest": args.base_manifest.as_posix(),
            "base_manifest_sha256": _sha256(args.base_manifest),
            "edar_map": args.edar_map.as_posix(),
            "edar_map_sha256": _sha256(args.edar_map),
            "municipality_reference_sha256": _sha256(args.municipality_reference),
        },
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["diagnostics"]["extension"], indent=2))


if __name__ == "__main__":
    main()
