"""Tests for native source-aware splits (slice SPL-1, plan §12.2).

Synthetic node/edge fixtures cover majority-overlap assignment, the
deterministic source-id tie-break, exactly-once validation, unassigned-policy
behavior, and manifest write/read round-trips. A skipif-guarded integration
class re-derives folds from the real representative manifest and
``native_graph.zarr`` and cross-checks the persisted artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.crossval_manifest import hash_region_ids
from data.native_crossval import (
    DEFAULT_MUNICIPAL_MANIFEST,
    NUM_FOLDS,
    assemble_native_fold_manifest,
    derive_source_folds,
    hash_tie_break,
    load_graph_tables,
    load_municipal_folds,
    load_native_fold_manifest,
    write_native_fold_manifest,
    NodeTable,
    RelationEdges,
    SourceFoldDerivation,
)
from data.native_registry import (
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_MUNICIPALITY,
)

pytestmark = pytest.mark.epiforecaster

MUNICIPAL_MANIFEST_REGION_IDS = ["08001", "08002", "08003", "08004"]
MUNICIPAL_FOLD_IDS = {
    "08001": 0,
    "08002": 0,
    "08003": 1,
    "08004": 2,
}


def _synthetic_nodes() -> NodeTable:
    """5 municipalities (08005 unfolded), 4 ABS, 2 comarcas."""
    return NodeTable(
        source_id=[
            "08001",
            "08002",
            "08003",
            "08004",
            "08005",
            "001",
            "002",
            "003",
            "004",
            "01",
            "02",
        ],
        node_type=[
            SOURCE_MUNICIPALITY,
            SOURCE_MUNICIPALITY,
            SOURCE_MUNICIPALITY,
            SOURCE_MUNICIPALITY,
            SOURCE_MUNICIPALITY,
            SOURCE_ABS,
            SOURCE_ABS,
            SOURCE_ABS,
            SOURCE_ABS,
            SOURCE_COMARCA,
            SOURCE_COMARCA,
        ],
        status=[
            "unobserved",
            "unobserved",
            "unobserved",
            "unobserved",
            "unobserved",
            "observed",
            "observed",
            "observed",
            "unmapped",
            "observed",
            "observed",
        ],
    )


def _synthetic_edges() -> tuple[RelationEdges, RelationEdges]:
    """ABS 001 -> fold 0 majority; ABS 002 exact tie folds 0/1; ABS 003 and
    comarca 02 only touch the unfolded municipality; comarca 01 -> fold 1.
    Every geometric pair is present in both graph directions; derivation must
    consume the canonical municipality->source direction exactly once (R8).
    """
    abs_edges = RelationEdges(
        src_idx=np.array([0, 5, 1, 5, 2, 5, 0, 6, 2, 6, 4, 7, 0, 8]),
        dst_idx=np.array([5, 0, 5, 1, 5, 2, 6, 0, 6, 2, 7, 4, 8, 0]),
        overlap_pop_est=np.array(
            [10.0, 10.0, 5.0, 5.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 9.0, 9.0, 50.0, 50.0]
        ),
    )
    comarca_edges = RelationEdges(
        src_idx=np.array([2, 9, 0, 9, 4, 10]),
        dst_idx=np.array([9, 2, 9, 0, 10, 4]),
        overlap_pop_est=np.array([9.0, 9.0, 3.0, 3.0, 6.0, 6.0]),
    )
    return abs_edges, comarca_edges


def _derive(**kwargs) -> SourceFoldDerivation:
    abs_edges, comarca_edges = _synthetic_edges()
    return derive_source_folds(
        _synthetic_nodes(),
        abs_edges,
        comarca_edges,
        MUNICIPAL_FOLD_IDS,
        num_folds=3,
        **kwargs,
    )


def _write_municipal_manifest(path: Path) -> None:
    payload = {
        "version": 1,
        "strategy": "synthetic",
        "dataset_path": "synthetic.zarr",
        "run_id": "real",
        "split_seed": 42,
        "num_folds": 3,
        "region_id_hash": hash_region_ids(sorted(MUNICIPAL_MANIFEST_REGION_IDS)),
        "folds": [
            {"fold_index": 0, "region_ids": ["08001", "08002"]},
            {"fold_index": 1, "region_ids": ["08003"]},
            {"fold_index": 2, "region_ids": ["08004"]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestMajorityDerivation:
    def test_majority_overlap_assigns_dominant_fold(self) -> None:
        derivation = _derive(on_unassigned_observed="exclude")
        tie_fold = hash_tie_break("002", [0, 1])
        # ABS 001: fold 0 = 10 + 5 = 15 vs fold 1 = 2.
        assert derivation.fold_of_abs["001"] == 0
        assert derivation.fold_pop_abs[0] == 15.0 + (4.0 if tie_fold == 0 else 0.0)
        # Only owned (winning) fold mass is reported per fold.
        if tie_fold == 1:
            assert derivation.fold_pop_abs[1] == 4.0
        else:
            assert 1 not in derivation.fold_pop_abs
        # Comarca 01: fold 1 = 9 vs fold 0 = 3.
        assert derivation.fold_of_comarca["01"] == 1
        assert derivation.fold_pop_comarca[1] == 9.0
        # Unmapped ABS 004 never receives a fold despite folded edges.
        assert "004" not in derivation.fold_of_abs

    def test_exact_tie_breaks_deterministically_by_source_id(self) -> None:
        derivation = _derive(on_unassigned_observed="exclude")
        # ABS 002 ties at 4.0 between folds 0 and 1; the winner is a pure
        # function of the source id over the ascending tied fold list.
        assert derivation.ties_abs == ("002",)
        assert derivation.fold_of_abs["002"] == hash_tie_break("002", [0, 1])
        redone = _derive(on_unassigned_observed="exclude")
        assert redone.fold_of_abs == derivation.fold_of_abs
        assert redone.fold_of_comarca == derivation.fold_of_comarca

    def test_tie_break_policy_is_injectable(self) -> None:
        high = _derive(
            on_unassigned_observed="exclude",
            tie_break=lambda source_id, tied: tied[-1],
        )
        low = _derive(
            on_unassigned_observed="exclude",
            tie_break=lambda source_id, tied: tied[0],
        )
        assert high.fold_of_abs["002"] == 1
        assert low.fold_of_abs["002"] == 0


class TestUnassignedPolicy:
    def test_unassigned_observed_source_raises(self) -> None:
        with pytest.raises(ValueError, match="003"):
            _derive(on_unassigned_observed="raise")

    def test_unassigned_observed_excluded_and_reported(self) -> None:
        derivation = _derive(on_unassigned_observed="exclude")
        assert set(derivation.fold_of_abs) == {"001", "002"}
        assert derivation.unassigned_abs == ("003",)
        assert derivation.unassigned_comarca == ("02",)

    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_unassigned_observed"):
            _derive(on_unassigned_observed="hash")

    def test_edge_must_join_source_and_municipality(self) -> None:
        abs_edges, comarca_edges = _synthetic_edges()
        bad = RelationEdges(
            src_idx=np.append(abs_edges.src_idx, 5),
            dst_idx=np.append(abs_edges.dst_idx, 6),
            overlap_pop_est=np.append(abs_edges.overlap_pop_est, 1.0),
        )
        with pytest.raises(ValueError, match="does not join"):
            derive_source_folds(
                _synthetic_nodes(),
                bad,
                comarca_edges,
                MUNICIPAL_FOLD_IDS,
                num_folds=3,
                on_unassigned_observed="exclude",
            )


class TestManifestRoundTrip:
    def test_assemble_write_load_round_trip(self, tmp_path: Path) -> None:
        municipal_path = tmp_path / "municipal.json"
        _write_municipal_manifest(municipal_path)
        municipal = load_municipal_folds(municipal_path, num_folds=3)
        derivation = _derive(on_unassigned_observed="exclude")
        payload = assemble_native_fold_manifest(
            municipal,
            derivation,
            on_unassigned_observed="exclude",
            num_folds=3,
        )
        assert payload["schema"] == "native_source_folds.v1"
        assert payload["derivation"] == "majority_overlap_pop"
        assert payload["municipal_manifest_ref"]["filename"] == "municipal.json"
        assert payload["abs_id_hash"] == hash_region_ids(["001", "002", "003"])
        assert payload["comarca_id_hash"] == hash_region_ids(["01", "02"])
        audit = payload["audit"]
        assert audit["counts"]["abs_assigned"] == 2
        assert audit["counts"]["abs_unassigned_observed"] == 1
        tie_fold = hash_tie_break("002", [0, 1])
        assert audit["fold_sizes"]["abs"] == [
            1 + (tie_fold == 0),
            1 if tie_fold == 1 else 0,
            0,
        ]
        assert audit["policy"]["tie_break"].startswith("sha256(source_id)")
        assert audit["generalization_claim"] == (
            "held-out source, visible context history (temporal-forecasting mode)"
        )

        out_path = tmp_path / "native_source_folds.json"
        write_native_fold_manifest(payload, out_path)
        loaded = load_native_fold_manifest(
            out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
        )
        assert [fold.fold_index for fold in loaded.folds] == [0, 1, 2]
        assert loaded.folds[0].municipalities == ("08001", "08002")
        assert loaded.abs_id_hash == payload["abs_id_hash"]
        assert loaded.excluded_abs == ("003",)
        assert loaded.excluded_comarcas == ("02",)

    def _write_and_break(self, tmp_path: Path, mutate) -> Path:
        municipal_path = tmp_path / "municipal.json"
        _write_municipal_manifest(municipal_path)
        municipal = load_municipal_folds(municipal_path, num_folds=3)
        payload = assemble_native_fold_manifest(
            municipal,
            _derive(on_unassigned_observed="exclude"),
            on_unassigned_observed="exclude",
            num_folds=3,
        )
        mutate(payload)
        out_path = tmp_path / "broken.json"
        write_native_fold_manifest(payload, out_path)
        return out_path, municipal_path

    def test_duplicate_source_across_folds_raises(self, tmp_path: Path) -> None:
        def duplicate_abs(payload: dict) -> None:
            payload["folds"][1]["abs"] = list(payload["folds"][0]["abs"])

        out_path, municipal_path = self._write_and_break(tmp_path, duplicate_abs)
        with pytest.raises(ValueError, match="more than one fold"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )

    def test_id_hash_mismatch_raises(self, tmp_path: Path) -> None:
        def bad_hash(payload: dict) -> None:
            payload["abs_id_hash"] = "0" * 64

        out_path, municipal_path = self._write_and_break(tmp_path, bad_hash)
        with pytest.raises(ValueError, match="abs_id_hash"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )

    def test_excluded_source_omission_breaks_universe_hash(self, tmp_path: Path) -> None:
        def omit_excluded(payload: dict) -> None:
            payload["audit"]["unassigned_observed"]["abs"] = []

        out_path, municipal_path = self._write_and_break(tmp_path, omit_excluded)
        with pytest.raises(ValueError, match="assigned plus excluded universe"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )

    def test_source_cannot_be_assigned_and_excluded(self, tmp_path: Path) -> None:
        def duplicate_universe_role(payload: dict) -> None:
            payload["audit"]["unassigned_observed"]["abs"].append("001")

        out_path, municipal_path = self._write_and_break(
            tmp_path, duplicate_universe_role
        )
        with pytest.raises(ValueError, match="both assigned and excluded"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )

    def test_bad_schema_raises(self, tmp_path: Path) -> None:
        def bad_schema(payload: dict) -> None:
            payload["schema"] = "native_source_folds.v2"

        out_path, municipal_path = self._write_and_break(tmp_path, bad_schema)
        with pytest.raises(ValueError, match="schema"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )

    def test_fold_count_mismatch_raises(self, tmp_path: Path) -> None:
        out_path, municipal_path = self._write_and_break(tmp_path, lambda p: None)
        with pytest.raises(ValueError, match="fold count"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=5
            )

    def test_municipal_ref_hash_mismatch_raises(self, tmp_path: Path) -> None:
        def bad_ref(payload: dict) -> None:
            payload["municipal_manifest_ref"]["sha256"] = "0" * 64

        out_path, municipal_path = self._write_and_break(tmp_path, bad_ref)
        with pytest.raises(ValueError, match="municipal_manifest_ref"):
            load_native_fold_manifest(
                out_path, municipal_manifest_path=municipal_path, expected_num_folds=3
            )


REAL_GRAPH = Path("data/processed/native_support/native_graph.zarr")
REAL_ARTIFACT = Path("data/processed/native_support/native_source_folds.json")
REAL_PRESENT = REAL_GRAPH.exists() and DEFAULT_MUNICIPAL_MANIFEST.exists()


@pytest.mark.skipif(
    not REAL_PRESENT, reason="real native-support artifacts not present"
)
class TestRealDerivation:
    def test_real_derivation_matches_persisted_artifact(self, tmp_path: Path) -> None:
        municipal = load_municipal_folds(DEFAULT_MUNICIPAL_MANIFEST)
        assert municipal.manifest.run_id == "real"
        assert len(municipal.manifest.folds) == NUM_FOLDS
        assert len(municipal.fold_of_municipality) == 161

        nodes, abs_edges, comarca_edges = load_graph_tables(REAL_GRAPH)
        derivation = derive_source_folds(
            nodes,
            abs_edges,
            comarca_edges,
            municipal.fold_of_municipality,
            num_folds=NUM_FOLDS,
            on_unassigned_observed="exclude",
        )
        # Registry observed status: every observed ABS/comarca is either
        # assigned or reported as unassigned (zero folded overlap mass).
        abs_observed = sum(
            1
            for typ, status in zip(nodes.node_type, nodes.status)
            if typ == SOURCE_ABS and status == "observed"
        )
        comarca_observed = sum(
            1
            for typ, status in zip(nodes.node_type, nodes.status)
            if typ == SOURCE_COMARCA and status == "observed"
        )
        assert abs_observed == 376
        assert comarca_observed == 42
        assert (
            len(derivation.fold_of_abs) + len(derivation.unassigned_abs) == abs_observed
        )
        assert (
            len(derivation.fold_of_comarca) + len(derivation.unassigned_comarca)
            == comarca_observed
        )

        payload = assemble_native_fold_manifest(
            municipal,
            derivation,
            on_unassigned_observed="exclude",
            num_folds=NUM_FOLDS,
        )
        out_path = tmp_path / "fresh_native_source_folds.json"
        write_native_fold_manifest(payload, out_path)
        loaded = load_native_fold_manifest(
            out_path,
            municipal_manifest_path=DEFAULT_MUNICIPAL_MANIFEST,
            expected_num_folds=NUM_FOLDS,
        )
        # Exactly-once per source type across the 10 folds.
        for attr in ("municipalities", "abs", "comarcas"):
            assigned = [
                source_id for fold in loaded.folds for source_id in getattr(fold, attr)
            ]
            assert len(assigned) == len(set(assigned))
        sizes = payload["audit"]["fold_sizes"]
        assert sum(sizes["abs"]) == len(derivation.fold_of_abs)
        assert sum(sizes["comarcas"]) == len(derivation.fold_of_comarca)
        assert set(payload["audit"]["fold_size_spread"]) == {
            "municipalities",
            "abs",
            "comarcas",
        }

        if REAL_ARTIFACT.exists():
            persisted = json.loads(REAL_ARTIFACT.read_text(encoding="utf-8"))
            assert persisted == payload
