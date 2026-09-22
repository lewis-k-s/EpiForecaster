"""Source-aware splits for native sources (slice SPL-1, plan §11.7).

Derives ``native_source_folds.json`` from the representative municipal
cross-validation manifest: an observed ABS/comarca node belongs to the fold
owning the majority of its population overlap into folded municipalities
(``overlap_pop_est`` summed over ``edges/muni_abs`` / ``edges/muni_comarca``
rows whose municipality endpoint is in that fold). Exact population ties
resolve deterministically by hashing the source id over the tied fold indices
sorted ascending. Observed sources with no folded overlap mass either raise
or are excluded from all folds and recorded in the audit trail; they are
outside the representative valid-target universe and never held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.crossval_manifest import (
    CrossvalFoldManifest,
    hash_region_ids,
    load_crossval_fold_manifest,
)
from data.native_registry import SOURCE_ABS, SOURCE_COMARCA, SOURCE_MUNICIPALITY
from data.preprocess.native_sources.build_native_graph import open_native_graph

NATIVE_FOLDS_SCHEMA = "native_source_folds.v1"
NATIVE_FOLDS_FILENAME = "native_source_folds.json"
MUNICIPAL_MANIFEST_FILENAME = "representative_10fold_mobility_refined.json"
DEFAULT_MUNICIPAL_MANIFEST = Path("configs/crossval") / MUNICIPAL_MANIFEST_FILENAME
DEFAULT_GRAPH = Path("data/processed/native_support/native_graph.zarr")
DEFAULT_OUTPUT = Path("data/processed/native_support") / NATIVE_FOLDS_FILENAME

DERIVATION = "majority_overlap_pop"
GENERALIZATION_CLAIM = (
    "held-out source, visible context history (temporal-forecasting mode)"
)
NUM_FOLDS = 10
TIE_BREAK_RULE = "sha256(source_id) mod len(tied_folds ascending)"
UNASSIGNED_POLICIES = ("raise", "exclude")
_TIE_REL_TOL = 1e-12
_TIE_ABS_TOL = 1e-9


def hash_tie_break(source_id: str, tied_folds: list[int]) -> int:
    """Resolve an exact population tie from the source identity alone.

    ``tied_folds`` must be sorted ascending; the winner is
    ``tied_folds[sha256(source_id) mod len(tied_folds)]``, a pure function of
    the source id so assignments never depend on iteration order.
    """
    digest = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest(), 16)
    return tied_folds[digest % len(tied_folds)]


@dataclass(frozen=True)
class NodeTable:
    """Node attribute columns of ``native_graph.zarr``, index-aligned."""

    source_id: list[str]
    node_type: list[str]
    status: list[str]


@dataclass(frozen=True)
class RelationEdges:
    """Long-layout overlap edges (``src_idx``/``dst_idx`` into a node table)."""

    src_idx: np.ndarray
    dst_idx: np.ndarray
    overlap_pop_est: np.ndarray


@dataclass(frozen=True)
class SourceFoldDerivation:
    """Majority-overlap fold assignment for one graph."""

    fold_of_abs: dict[str, int]
    fold_of_comarca: dict[str, int]
    ties_abs: tuple[str, ...]
    ties_comarca: tuple[str, ...]
    unassigned_abs: tuple[str, ...]
    unassigned_comarca: tuple[str, ...]
    #: Overlap population owned by each fold (its assigned sources' winning
    #: fold mass), i.e. the held-out population share per fold.
    fold_pop_abs: dict[int, float]
    fold_pop_comarca: dict[int, float]


@dataclass(frozen=True)
class NativeFold:
    """Validated fold contents, ids sorted ascending per source type."""

    fold_index: int
    municipalities: tuple[str, ...]
    abs: tuple[str, ...]
    comarcas: tuple[str, ...]


@dataclass(frozen=True)
class NativeFoldManifest:
    """Validated ``native_source_folds.v1`` payload."""

    path: Path
    num_folds: int
    folds: tuple[NativeFold, ...]
    abs_id_hash: str
    comarca_id_hash: str
    excluded_abs: tuple[str, ...]
    excluded_comarcas: tuple[str, ...]
    municipal_ref: dict[str, str]
    audit: dict


def load_graph_tables(
    graph_path: Path,
) -> tuple[NodeTable, RelationEdges, RelationEdges]:
    """Open ``native_graph.zarr`` and return node attributes + clinical edges."""
    root, relations = open_native_graph(graph_path)
    columns = {
        name: [str(v) for v in root[name].values.tolist()]
        for name in ("source_id", "node_type", "status")
    }
    nodes = NodeTable(**columns)

    def edges(name: str) -> RelationEdges:
        rel = relations[name]
        return RelationEdges(
            src_idx=np.asarray(rel["src_idx"].values, dtype=np.int64),
            dst_idx=np.asarray(rel["dst_idx"].values, dtype=np.int64),
            overlap_pop_est=np.asarray(rel["overlap_pop_est"].values, dtype=np.float64),
        )

    return nodes, edges("muni_abs"), edges("muni_comarca")


def _fold_pop_sums(
    nodes: NodeTable,
    edges: RelationEdges,
    source_type: str,
    fold_of_municipality: dict[str, int],
    num_folds: int,
) -> dict[str, dict[int, float]]:
    """Per-source, per-fold summed overlap population for one relation."""
    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for src, dst, pop in zip(
        edges.src_idx.tolist(), edges.dst_idx.tolist(), edges.overlap_pop_est.tolist()
    ):
        src_type = nodes.node_type[src]
        dst_type = nodes.node_type[dst]
        if src_type == SOURCE_MUNICIPALITY and dst_type == source_type:
            source_id, municipality_id = nodes.source_id[dst], nodes.source_id[src]
        elif src_type == source_type and dst_type == SOURCE_MUNICIPALITY:
            # The graph stores reciprocal directed rows with the same overlap
            # mass. Consume municipality->source only so fold mass is counted
            # exactly once (review ruling R8).
            continue
        else:
            raise ValueError(
                f"edge does not join {source_type} with municipality: "
                f"{src_type}:{nodes.source_id[src]} - {dst_type}:{nodes.source_id[dst]}"
            )
        fold = fold_of_municipality.get(municipality_id)
        if fold is None:
            continue
        if not 0 <= fold < num_folds:
            raise ValueError(f"fold index {fold} outside 0..{num_folds - 1}")
        sums[source_id][fold] += pop
    return {source_id: dict(folds) for source_id, folds in sums.items()}


def _assign_majority(
    sums: dict[str, dict[int, float]],
    observed_ids: list[str],
    source_type: str,
    num_folds: int,
    tie_break,
    on_unassigned_observed: str,
) -> tuple[dict[str, int], list[str], list[str], dict[int, float]]:
    """Pick the majority fold per source; enforce the unassigned policy."""
    if on_unassigned_observed not in UNASSIGNED_POLICIES:
        raise ValueError(f"on_unassigned_observed must be one of {UNASSIGNED_POLICIES}")
    fold_of: dict[str, int] = {}
    ties: list[str] = []
    for source_id in observed_ids:
        folds = sums.get(source_id, {})
        if not folds:
            continue
        max_pop = max(folds.values())
        tied = sorted(
            fold
            for fold, pop in folds.items()
            if math.isclose(pop, max_pop, rel_tol=_TIE_REL_TOL, abs_tol=_TIE_ABS_TOL)
        )
        fold_of[source_id] = tie_break(source_id, tied) if len(tied) > 1 else tied[0]
        if len(tied) > 1:
            ties.append(source_id)
    unassigned = sorted(set(observed_ids) - set(fold_of))
    if unassigned and on_unassigned_observed == "raise":
        raise ValueError(
            f"observed {source_type} sources without folded overlap mass: {unassigned}"
        )
    fold_pop = defaultdict(float)
    for source_id, fold in fold_of.items():
        fold_pop[fold] += sums[source_id][fold]
    return fold_of, ties, unassigned, dict(fold_pop)


def _observed_ids(nodes: NodeTable, source_type: str) -> list[str]:
    """Supervised (registry-observed) source ids of one type, node order."""
    return [
        source_id
        for source_id, node_type, status in zip(
            nodes.source_id, nodes.node_type, nodes.status
        )
        if node_type == source_type and status == "observed"
    ]


def derive_source_folds(
    nodes: NodeTable,
    abs_edges: RelationEdges,
    comarca_edges: RelationEdges,
    fold_of_municipality: dict[str, int],
    *,
    num_folds: int = NUM_FOLDS,
    tie_break=hash_tie_break,
    on_unassigned_observed: str = "raise",
) -> SourceFoldDerivation:
    """Assign observed ABS/comarca nodes to municipal folds (plan §11.7).

    Only ``status == 'observed'`` sources (the supervised sources) receive
    folds. Sources with no overlap mass into folded municipalities follow
    ``on_unassigned_observed``: ``'raise'`` fails the derivation, ``'exclude'``
    leaves them out of every fold and reports them.
    """
    abs_sums = _fold_pop_sums(
        nodes, abs_edges, SOURCE_ABS, fold_of_municipality, num_folds
    )
    comarca_sums = _fold_pop_sums(
        nodes, comarca_edges, SOURCE_COMARCA, fold_of_municipality, num_folds
    )
    observed_abs = _observed_ids(nodes, SOURCE_ABS)
    observed_comarcas = _observed_ids(nodes, SOURCE_COMARCA)
    fold_abs, ties_abs, unassigned_abs, pop_abs = _assign_majority(
        abs_sums,
        observed_abs,
        SOURCE_ABS,
        num_folds,
        tie_break,
        on_unassigned_observed,
    )
    fold_comarca, ties_comarca, unassigned_comarca, pop_comarca = _assign_majority(
        comarca_sums,
        observed_comarcas,
        SOURCE_COMARCA,
        num_folds,
        tie_break,
        on_unassigned_observed,
    )
    return SourceFoldDerivation(
        fold_of_abs=fold_abs,
        fold_of_comarca=fold_comarca,
        ties_abs=tuple(ties_abs),
        ties_comarca=tuple(ties_comarca),
        unassigned_abs=tuple(unassigned_abs),
        unassigned_comarca=tuple(unassigned_comarca),
        fold_pop_abs=pop_abs,
        fold_pop_comarca=pop_comarca,
    )


@dataclass(frozen=True)
class MunicipalFolds:
    """Loaded representative municipal manifest plus provenance."""

    filename: str
    manifest: CrossvalFoldManifest
    fold_of_municipality: dict[str, int]
    file_sha256: str


def load_municipal_folds(
    municipal_manifest_path: Path, *, num_folds: int = NUM_FOLDS
) -> MunicipalFolds:
    """Load and validate the municipal manifest via the v1 loader."""
    payload = json.loads(municipal_manifest_path.read_text(encoding="utf-8"))
    valid_region_ids = sorted(
        region_id for fold in payload["folds"] for region_id in fold["region_ids"]
    )
    manifest = load_crossval_fold_manifest(
        municipal_manifest_path,
        valid_region_ids=valid_region_ids,
        expected_num_folds=num_folds,
        expected_run_id=str(payload["run_id"]),
    )
    fold_of_municipality = {
        region_id: fold_index
        for fold_index, fold in enumerate(manifest.folds)
        for region_id in fold
    }
    file_sha256 = hashlib.sha256(municipal_manifest_path.read_bytes()).hexdigest()
    return MunicipalFolds(
        filename=municipal_manifest_path.name,
        manifest=manifest,
        fold_of_municipality=fold_of_municipality,
        file_sha256=file_sha256,
    )


def _fold_members(fold_of: dict[str, int], fold_index: int) -> list[str]:
    """Ids assigned to one fold, sorted ascending (leading zeros preserved)."""
    return sorted(s for s, f in fold_of.items() if f == fold_index)


def assemble_native_fold_manifest(
    municipal: MunicipalFolds,
    derivation: SourceFoldDerivation,
    *,
    on_unassigned_observed: str,
    num_folds: int = NUM_FOLDS,
) -> dict:
    """Assemble the versioned payload from a derivation and its manifest."""
    folds = [
        {
            "fold_index": fold_index,
            "municipalities": sorted(municipal.manifest.folds[fold_index]),
            "abs": _fold_members(derivation.fold_of_abs, fold_index),
            "comarcas": _fold_members(derivation.fold_of_comarca, fold_index),
        }
        for fold_index in range(num_folds)
    ]
    municipal_sizes = [len(fold) for fold in municipal.manifest.folds]
    abs_sizes = [
        len(_fold_members(derivation.fold_of_abs, i)) for i in range(num_folds)
    ]
    comarca_sizes = [
        len(_fold_members(derivation.fold_of_comarca, i)) for i in range(num_folds)
    ]
    abs_observed = len(derivation.fold_of_abs) + len(derivation.unassigned_abs)
    comarca_observed = len(derivation.fold_of_comarca) + len(
        derivation.unassigned_comarca
    )
    payload = {
        "schema": NATIVE_FOLDS_SCHEMA,
        "num_folds": num_folds,
        "municipal_manifest_ref": {
            "filename": municipal.filename,
            "sha256": municipal.file_sha256,
            "region_id_hash": municipal.manifest.region_id_hash,
            "run_id": municipal.manifest.run_id,
        },
        "derivation": DERIVATION,
        "abs_id_hash": hash_region_ids(
            sorted(set(derivation.fold_of_abs) | set(derivation.unassigned_abs))
        ),
        "comarca_id_hash": hash_region_ids(
            sorted(
                set(derivation.fold_of_comarca)
                | set(derivation.unassigned_comarca)
            )
        ),
        "folds": folds,
        "audit": {
            "counts": {
                "municipalities": sum(municipal_sizes),
                "abs_observed": abs_observed,
                "abs_assigned": len(derivation.fold_of_abs),
                "abs_unassigned_observed": len(derivation.unassigned_abs),
                "comarca_observed": comarca_observed,
                "comarca_assigned": len(derivation.fold_of_comarca),
                "comarca_unassigned_observed": len(derivation.unassigned_comarca),
            },
            "fold_sizes": {
                "municipalities": municipal_sizes,
                "abs": abs_sizes,
                "comarcas": comarca_sizes,
            },
            "fold_size_spread": {
                "municipalities": max(municipal_sizes) - min(municipal_sizes),
                "abs": max(abs_sizes) - min(abs_sizes),
                "comarcas": max(comarca_sizes) - min(comarca_sizes),
            },
            "fold_owned_overlap_pop": {
                "abs": [
                    derivation.fold_pop_abs.get(fold_index, 0.0)
                    for fold_index in range(num_folds)
                ],
                "comarcas": [
                    derivation.fold_pop_comarca.get(fold_index, 0.0)
                    for fold_index in range(num_folds)
                ],
            },
            "ties_broken": {
                "abs": list(derivation.ties_abs),
                "comarcas": list(derivation.ties_comarca),
            },
            "unassigned_observed": {
                "abs": list(derivation.unassigned_abs),
                "comarcas": list(derivation.unassigned_comarca),
            },
            "policy": {
                "overlap_weight": "overlap_pop_est (GEOSTAT grid, plan §11.7)",
                "tie_break": TIE_BREAK_RULE,
                "unassigned_observed": on_unassigned_observed,
            },
            "generalization_claim": GENERALIZATION_CLAIM,
        },
    }
    return payload


def build_native_fold_manifest(
    municipal_manifest_path: Path,
    graph_path: Path,
    *,
    on_unassigned_observed: str,
    num_folds: int = NUM_FOLDS,
) -> dict:
    """Load both inputs, derive source folds, and assemble the payload."""
    municipal = load_municipal_folds(municipal_manifest_path, num_folds=num_folds)
    nodes, abs_edges, comarca_edges = load_graph_tables(graph_path)
    derivation = derive_source_folds(
        nodes,
        abs_edges,
        comarca_edges,
        municipal.fold_of_municipality,
        num_folds=num_folds,
        on_unassigned_observed=on_unassigned_observed,
    )
    return assemble_native_fold_manifest(
        municipal,
        derivation,
        on_unassigned_observed=on_unassigned_observed,
        num_folds=num_folds,
    )


def write_native_fold_manifest(payload: dict, path: Path) -> None:
    """Write the payload deterministically (sorted keys inside audit dicts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_native_fold_manifest(
    path: Path,
    *,
    municipal_manifest_path: Path | None = None,
    expected_num_folds: int = NUM_FOLDS,
) -> NativeFoldManifest:
    """Validate a persisted native fold manifest (exactly-once per type)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != NATIVE_FOLDS_SCHEMA:
        raise ValueError(
            f"Unsupported native fold manifest schema in {path}: "
            f"{payload.get('schema')!r}"
        )
    num_folds = int(payload.get("num_folds", -1))
    if num_folds != expected_num_folds:
        raise ValueError(
            f"Native fold manifest fold count does not match: "
            f"{num_folds} != {expected_num_folds}"
        )
    entries = payload.get("folds")
    if not isinstance(entries, list):
        raise ValueError("Native fold manifest 'folds' must be a list")
    entries = sorted(entries, key=lambda entry: int(entry["fold_index"]))
    if [int(entry["fold_index"]) for entry in entries] != list(range(num_folds)):
        raise ValueError(
            "Native fold manifest fold indices must be exactly 0..num_folds-1"
        )
    folds = tuple(
        NativeFold(
            fold_index=int(entry["fold_index"]),
            municipalities=tuple(str(x) for x in entry["municipalities"]),
            abs=tuple(str(x) for x in entry["abs"]),
            comarcas=tuple(str(x) for x in entry["comarcas"]),
        )
        for entry in entries
    )
    _assert_exactly_once(folds, "municipalities")
    _assert_exactly_once(folds, "abs")
    _assert_exactly_once(folds, "comarcas")
    audit = dict(payload.get("audit", {}))
    unassigned = audit.get("unassigned_observed", {})
    if not isinstance(unassigned, dict):
        raise ValueError("Native fold manifest audit.unassigned_observed must be a map")
    excluded_by_attr = {
        "abs": tuple(str(x) for x in unassigned.get("abs", [])),
        "comarcas": tuple(str(x) for x in unassigned.get("comarcas", [])),
    }
    for key, attr in (("abs_id_hash", "abs"), ("comarca_id_hash", "comarcas")):
        assigned = sorted(
            source_id for fold in folds for source_id in getattr(fold, attr)
        )
        excluded = excluded_by_attr[attr]
        if len(excluded) != len(set(excluded)):
            raise ValueError(f"Native fold manifest {attr} exclusions contain duplicates")
        overlap = sorted(set(assigned) & set(excluded))
        if overlap:
            raise ValueError(
                f"Native fold manifest {attr} sources are both assigned and excluded: "
                f"{overlap}"
            )
        universe = sorted(set(assigned) | set(excluded))
        expected = hash_region_ids(universe)
        if str(payload.get(key, "")) != expected:
            raise ValueError(
                f"Native fold manifest {key} mismatch for assigned plus excluded universe"
            )
    counts = audit.get("counts", {})
    expected_counts = {
        "abs_observed": sum(len(fold.abs) for fold in folds)
        + len(excluded_by_attr["abs"]),
        "comarca_observed": sum(len(fold.comarcas) for fold in folds)
        + len(excluded_by_attr["comarcas"]),
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(
                f"Native fold manifest {key} does not match assigned plus excluded universe"
            )
    municipal_ref = dict(payload.get("municipal_manifest_ref", {}))
    if municipal_manifest_path is not None:
        municipal = load_municipal_folds(municipal_manifest_path, num_folds=num_folds)
        if municipal_ref.get("sha256") != municipal.file_sha256:
            raise ValueError(
                "municipal_manifest_ref sha256 does not match the municipal manifest"
            )
        for fold in folds:
            expected_municipal = tuple(
                sorted(municipal.manifest.folds[fold.fold_index])
            )
            if fold.municipalities != expected_municipal:
                raise ValueError(
                    f"fold {fold.fold_index} municipalities differ from the "
                    "municipal manifest"
                )
    return NativeFoldManifest(
        path=path,
        num_folds=num_folds,
        folds=folds,
        abs_id_hash=str(payload["abs_id_hash"]),
        comarca_id_hash=str(payload["comarca_id_hash"]),
        excluded_abs=excluded_by_attr["abs"],
        excluded_comarcas=excluded_by_attr["comarcas"],
        municipal_ref=municipal_ref,
        audit=audit,
    )


def _assert_exactly_once(folds: tuple[NativeFold, ...], attr: str) -> None:
    """Leakage audit: no source of one type may appear in two folds."""
    counts: dict[str, int] = defaultdict(int)
    for fold in folds:
        for source_id in getattr(fold, attr):
            counts[source_id] += 1
    duplicated = sorted(source_id for source_id, n in counts.items() if n > 1)
    if duplicated:
        raise ValueError(f"{attr} sources assigned to more than one fold: {duplicated}")


def main(argv: list[str] | None = None) -> int:
    """CLI: derive and write ``native_source_folds.json``; print fold sizes."""
    parser = argparse.ArgumentParser(
        description="Derive native ABS/comarca folds from the municipal manifest."
    )
    parser.add_argument(
        "--municipal-manifest", type=Path, default=DEFAULT_MUNICIPAL_MANIFEST
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--on-unassigned-observed",
        choices=UNASSIGNED_POLICIES,
        default="exclude",
        help="raise on observed sources with no folded overlap mass, or exclude "
        "and audit them (recorded in the manifest audit)",
    )
    args = parser.parse_args(argv)
    payload = build_native_fold_manifest(
        args.municipal_manifest,
        args.graph,
        on_unassigned_observed=args.on_unassigned_observed,
    )
    write_native_fold_manifest(payload, args.output)
    load_native_fold_manifest(
        args.output,
        municipal_manifest_path=args.municipal_manifest,
        expected_num_folds=payload["num_folds"],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": payload["audit"]["counts"],
                "fold_sizes": payload["audit"]["fold_sizes"],
                "ties_broken": payload["audit"]["ties_broken"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
