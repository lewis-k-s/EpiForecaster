"""Tests for the native graph store assembly (slice GRF-1, plan §12.2).

Synthetic-registry fixtures (4 municipalities, 2 ABS, 2 comarcas, 3 mitma
zones) cover node counts, status passthrough, ABS population aggregation,
edge layout/index bounds, validation failures, write/read round-trip, and
determinism. A skipif-guarded integration class assembles and writes from
the real artifacts, computing expected counts from the inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.native_registry import (
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_MUNICIPALITY,
    STATUS_OBSERVED,
    STATUS_UNMAPPED,
    STATUS_UNOBSERVED,
    SourceRecord,
    SourceRegistry,
)
from data.preprocess.native_sources.build_native_graph import (
    SCHEMA,
    _load_registries,
    assert_same_assembly,
    assemble_native_graph,
    assign_relations,
    build,
    open_native_graph,
    validate_native_graph,
    write_native_graph,
)

pytestmark = pytest.mark.epiforecaster

MITMA = "mitma"
REAL_ROOT = Path("data/processed/native_support")
REAL_OVERLAP = REAL_ROOT / "overlap_edges.parquet"
REAL_OD = REAL_ROOT / "od_mitma.zarr"
REAL_CSVS = (
    Path("data/files/COVID-19__Persones_hospitalitzades.csv"),
    Path(
        "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    ),
    Path("data/files/mpiscatalunya.csv"),
    Path("data/files/fl_population_por_municipis.csv"),
    Path("data/files/municipality_id_reference.csv"),
    Path("data/files/geo/cartografia/ABS.shp"),
)
REAL_PRESENT = (
    REAL_OVERLAP.exists()
    and REAL_OD.exists()
    and all(path.exists() for path in REAL_CSVS)
)

OD_ZONE_IDS = ["17001", "17002", "17003"]
OD_ZONE_POPS = np.array([10.0, np.nan, 5.0], dtype=np.float32)
EXPECTED_UIDS = [
    "abs:001",
    "abs:002",
    "comarca:01",
    "comarca:02",
    "mitma:17001",
    "mitma:17002",
    "mitma:17003",
    "municipality:08001",
    "municipality:08002",
    "municipality:08003",
    "municipality:08004",
]


def _row(
    src_type: str,
    src_id: str,
    dst_type: str,
    dst_id: str,
    pop: float,
    frac_dst: float,
) -> dict:
    return {
        "src_type": src_type,
        "src_id": src_id,
        "dst_type": dst_type,
        "dst_id": dst_id,
        "overlap_area_m2": 1.0,
        "overlap_pop_est": pop,
        "frac_of_src_pop": 0.5,
        "frac_of_dst_pop": frac_dst,
        "coverage_src": 1.0,
        "coverage_dst": 1.0,
        "provenance": "test",
    }


def _synthetic_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.extend(
        [
            SourceRecord(
                SOURCE_MUNICIPALITY, "08001", STATUS_OBSERVED, 100.0, None, None
            ),
            SourceRecord(
                SOURCE_MUNICIPALITY, "08002", STATUS_UNOBSERVED, 200.0, None, None
            ),
            SourceRecord(
                SOURCE_MUNICIPALITY, "08003", STATUS_OBSERVED, 50.0, None, None
            ),
            SourceRecord(SOURCE_ABS, "001", STATUS_OBSERVED, None, None, None),
            SourceRecord(SOURCE_ABS, "002", STATUS_OBSERVED, None, None, None),
            SourceRecord(SOURCE_COMARCA, "01", STATUS_OBSERVED, 300.0, None, None),
            SourceRecord(SOURCE_COMARCA, "02", STATUS_UNOBSERVED, 150.0, None, None),
        ]
    )
    return registry


def _synthetic_overlap() -> pd.DataFrame:
    rows = [
        # exact memberships, both directions
        _row(SOURCE_MUNICIPALITY, "08001", SOURCE_COMARCA, "01", 100.0, 1 / 3),
        _row(SOURCE_MUNICIPALITY, "08002", SOURCE_COMARCA, "01", 200.0, 2 / 3),
        _row(SOURCE_COMARCA, "01", SOURCE_MUNICIPALITY, "08001", 100.0, 1.0),
        _row(SOURCE_COMARCA, "01", SOURCE_MUNICIPALITY, "08002", 200.0, 1.0),
        _row(SOURCE_MUNICIPALITY, "08003", SOURCE_COMARCA, "02", 50.0, 1 / 3),
        _row(SOURCE_MUNICIPALITY, "08004", SOURCE_COMARCA, "02", 100.0, 2 / 3),
        _row(SOURCE_COMARCA, "02", SOURCE_MUNICIPALITY, "08003", 50.0, 1.0),
        _row(SOURCE_COMARCA, "02", SOURCE_MUNICIPALITY, "08004", 100.0, 1.0),
        # municipality->abs; ABS 001 receives municipal contributions
        _row(SOURCE_MUNICIPALITY, "08001", SOURCE_ABS, "001", 60.0, 0.6),
        _row(SOURCE_MUNICIPALITY, "08002", SOURCE_ABS, "001", 40.0, 0.4),
        _row(SOURCE_ABS, "001", SOURCE_MUNICIPALITY, "08001", 60.0, 0.6),
        _row(SOURCE_ABS, "001", SOURCE_MUNICIPALITY, "08002", 40.0, 0.2),
        # mitma relations; ABS 002 is only touched through mitma
        _row(MITMA, "17001", SOURCE_MUNICIPALITY, "08001", 10.0, 0.1),
        _row(SOURCE_MUNICIPALITY, "08001", MITMA, "17001", 10.0, 0.6),
        _row(MITMA, "17002", SOURCE_ABS, "002", 7.0, 1.0),
        _row(SOURCE_ABS, "002", MITMA, "17002", 7.0, 0.5),
    ]
    return pd.DataFrame(rows)


def _assemble(
    overlap: pd.DataFrame | None = None,
    registry: SourceRegistry | None = None,
    od_ids: list[str] | None = None,
    od_pops: np.ndarray | None = None,
):
    return assemble_native_graph(
        registry if registry is not None else _synthetic_registry(),
        overlap if overlap is not None else _synthetic_overlap(),
        od_ids if od_ids is not None else OD_ZONE_IDS,
        od_pops if od_pops is not None else OD_ZONE_POPS,
    )


class TestSyntheticAssembly:
    def test_node_counts_order_and_ids(self):
        root, _relations, report = _assemble()

        assert root.sizes["node"] == 11
        assert root.coords["node"].values.tolist() == EXPECTED_UIDS
        assert report.node_counts == {
            SOURCE_ABS: 2,
            SOURCE_COMARCA: 2,
            MITMA: 3,
            SOURCE_MUNICIPALITY: 4,
        }
        assert dict(zip(root["source_id"].values, root["node_type"].values)) == {
            uid.split(":", 1)[1]: uid.split(":", 1)[0] for uid in EXPECTED_UIDS
        }

    def test_status_passthrough_with_unmapped_fallback(self):
        root, _relations, report = _assemble()
        status = dict(zip(root.coords["node"].values, root["status"].values))

        assert status["municipality:08001"] == STATUS_OBSERVED
        assert status["municipality:08002"] == STATUS_UNOBSERVED
        # 08004 participates in overlaps but is absent from the registry
        assert status["municipality:08004"] == STATUS_UNMAPPED
        assert status["abs:001"] == STATUS_OBSERVED
        assert status["comarca:02"] == STATUS_UNOBSERVED
        # every mitma context node is observed (mobility seen all window)
        assert status["mitma:17003"] == STATUS_OBSERVED
        assert report.status_counts[SOURCE_MUNICIPALITY] == {
            STATUS_OBSERVED: 2,
            STATUS_UNMAPPED: 1,
            STATUS_UNOBSERVED: 1,
        }

    def test_population_semantics(self):
        root, _relations, report = _assemble()
        pops = dict(zip(root.coords["node"].values, root["population"].values))
        overlap_pops = dict(
            zip(
                root.coords["node"].values,
                root["overlap_population_est"].values,
            )
        )

        # R3: audited ABS denominator uses municipal audited populations and
        # municipality-normalized fractions: 100*0.5 + 200*0.5 = 150.
        # The distinct GEOSTAT overlap quantity remains 60+40 = 100.
        assert pops["abs:001"] == pytest.approx(150.0, abs=1e-3)
        assert overlap_pops["abs:001"] == pytest.approx(100.0, abs=1e-3)
        assert np.isnan(pops["abs:002"])  # no municipality->abs rows
        assert np.isnan(overlap_pops["abs:002"])
        assert pops["municipality:08001"] == 100.0
        assert np.isnan(pops["municipality:08004"])  # registry-absent
        assert pops["comarca:01"] == 300.0
        assert pops["mitma:17001"] == 10.0
        assert np.isnan(pops["mitma:17002"])  # NaN zone population passthrough
        assert pops["mitma:17003"] == 5.0
        assert report.abs_population == {
            "min": pytest.approx(150.0),
            "median": pytest.approx(150.0),
            "max": pytest.approx(150.0),
        }

    def test_edge_layout_counts_and_order(self):
        root, relations, _report = _assemble()

        assert root["od_zone_ref"].values.tolist() == OD_ZONE_IDS
        assert root.sizes["od_zone"] == 3
        assert {n: int(ds.sizes["edge"]) for n, ds in relations.items()} == {
            "muni_abs": 4,
            "muni_comarca": 8,
            "mitma_abs": 2,
            "mitma_comarca": 0,
            "mitma_muni": 2,
        }
        muni_abs = relations["muni_abs"]
        assert muni_abs["src_idx"].dtype == np.int32
        assert muni_abs["weight"].dtype == np.float32
        # node order: abs:001=0 .. municipality:08001=7, 08002=8; (src, dst) sorted
        assert muni_abs["src_idx"].values.tolist() == [0, 0, 7, 8]
        assert muni_abs["dst_idx"].values.tolist() == [7, 8, 0, 0]
        assert muni_abs["weight"].values.tolist() == pytest.approx([0.6, 0.2, 0.6, 0.4])

    def test_edge_indices_in_bounds(self):
        _root, relations, _report = _assemble()
        for ds in relations.values():
            for var in ("src_idx", "dst_idx"):
                assert (ds[var].values >= 0).all()
                assert (ds[var].values < 11).all()

    def test_warnings_reported(self):
        _root, _relations, report = _assemble()

        assert any("NaN population" in w for w in report.warnings)
        assert len(report.warnings) == 1  # registry fully covered otherwise

    def test_deterministic_reassembly(self):
        assert_same_assembly(_assemble()[:2], _assemble()[:2])


class TestValidationFailures:
    def test_missing_relation_group_raises(self):
        root, relations, _report = _assemble()
        del relations["muni_abs"]

        with pytest.raises(ValueError, match="relation groups"):
            validate_native_graph(root, relations, _synthetic_registry(), OD_ZONE_IDS)

    def test_observed_node_without_edge_raises(self):
        root, relations, _report = _assemble()
        # keep the group (variables intact) but with zero rows
        relations["muni_abs"] = relations["muni_abs"].isel(edge=slice(0, 0))

        with pytest.raises(ValueError, match="abs:001"):
            validate_native_graph(root, relations, _synthetic_registry(), OD_ZONE_IDS)

    def test_observed_municipality_without_membership_raises(self):
        overlap = _synthetic_overlap()
        with_m1 = ((overlap["src_id"] == "08001") & (overlap["dst_id"] == "01")) | (
            (overlap["src_id"] == "01") & (overlap["dst_id"] == "08001")
        )
        # 08001 stays covered via muni_abs/mitma_muni but loses its membership
        # row; assemble_native_graph validates at the end, so it raises here.
        with pytest.raises(ValueError, match="membership"):
            _assemble(overlap=overlap[~with_m1])

    def test_out_of_bounds_index_raises(self):
        root, relations, _report = _assemble()
        relations["mitma_muni"]["src_idx"].values[0] = 11

        with pytest.raises(ValueError, match="out of bounds"):
            validate_native_graph(root, relations, _synthetic_registry(), OD_ZONE_IDS)

    def test_weight_out_of_bounds_raises(self):
        root, relations, _report = _assemble()
        relations["mitma_muni"]["weight"].values[0] = 1.5

        with pytest.raises(ValueError, match="weight"):
            validate_native_graph(root, relations, _synthetic_registry(), OD_ZONE_IDS)

    def test_mitma_od_misalignment_raises(self):
        root, relations, _report = _assemble()

        with pytest.raises(ValueError, match="od registry disagree"):
            validate_native_graph(
                root, relations, _synthetic_registry(), ["17001", "17002"]
            )

    def test_abs_population_out_of_range_raises(self):
        registry = _synthetic_registry()
        registry.add(
            SourceRecord(
                SOURCE_MUNICIPALITY,
                "08001",
                STATUS_OBSERVED,
                5.0e6,
                None,
                None,
            )
        )

        with pytest.raises(ValueError, match="ABS audited population"):
            _assemble(registry=registry)

    def test_unknown_relation_pair_raises(self):
        overlap = pd.concat(
            [
                _synthetic_overlap(),
                pd.DataFrame([_row(SOURCE_ABS, "001", SOURCE_COMARCA, "01", 1.0, 1.0)]),
            ]
        )
        with pytest.raises(ValueError, match="relation pairs"):
            _assemble(overlap=overlap)

    def test_duplicate_directed_row_raises(self):
        overlap = pd.concat([_synthetic_overlap(), _synthetic_overlap().iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate"):
            _assemble(overlap=overlap)


class TestWriteRoundTrip:
    def test_write_reopen_and_revalidate(self, tmp_path):
        root, relations, report = _assemble()
        out = write_native_graph(root, relations, tmp_path / "native_graph.zarr")

        reopened_root, reopened = open_native_graph(out)
        assert reopened_root.attrs["schema"] == SCHEMA
        assert reopened_root["od_zone_ref"].values.tolist() == OD_ZONE_IDS
        assert reopened["muni_abs"]["src_idx"].dtype == np.int32
        assert reopened["mitma_comarca"].sizes["edge"] == 0
        revalidated = validate_native_graph(
            reopened_root, reopened, _synthetic_registry(), OD_ZONE_IDS
        )
        assert revalidated.to_attr() == report.to_attr()

    def test_rebuild_is_byte_identical_on_coords_and_indices(self, tmp_path):
        first = tmp_path / "a.zarr"
        second = tmp_path / "b.zarr"
        for out in (first, second):
            root, relations, _report = _assemble()
            write_native_graph(root, relations, out)

        root_a, rel_a = open_native_graph(first)
        root_b, rel_b = open_native_graph(second)
        assert root_a.attrs == root_b.attrs
        assert_same_assembly((root_a, rel_a), (root_b, rel_b))


@pytest.mark.skipif(
    not REAL_PRESENT, reason="real native-support artifacts not present"
)
class TestRealArtifacts:
    def _inputs(self):
        registry = _load_registries(*REAL_CSVS[:5], REAL_CSVS[5])
        overlap = pd.read_parquet(REAL_OVERLAP)
        od = xr.open_zarr(REAL_OD)
        return registry, overlap, od

    def test_real_assembly_matches_inputs(self):
        registry, overlap, od = self._inputs()
        zone_ids = [str(z) for z in od["zone_id"].values]
        zone_pops = np.asarray(od["zone_population"].values, dtype=np.float32)

        root, _relations, report = assemble_native_graph(
            registry, overlap, zone_ids, zone_pops
        )

        endpoints = pd.concat(
            [
                overlap[["src_type", "src_id"]].rename(
                    columns={"src_type": "t", "src_id": "i"}
                ),
                overlap[["dst_type", "dst_id"]].rename(
                    columns={"dst_type": "t", "dst_id": "i"}
                ),
            ]
        )
        expected_ids = {t: set(g["i"]) for t, g in endpoints.groupby("t")}
        expected_ids.setdefault(MITMA, set()).update(zone_ids)
        for node_type, ids in expected_ids.items():
            sel = root["node_type"].values == node_type
            assert set(root["source_id"].values[sel]) == ids
        assert report.node_counts == {
            SOURCE_MUNICIPALITY: 963,
            SOURCE_ABS: 379,
            SOURCE_COMARCA: 43,
            MITMA: 456,
        }
        assert (
            report.edge_counts
            == assign_relations(overlap)["relation"].value_counts().to_dict()
        )
        assert root["od_zone_ref"].values.tolist() == zone_ids

        # Every observed clinical node has >=1 overlap edge. R3 uses audited
        # municipal population times the municipality-normalized overlap
        # fraction, not the raw GEOSTAT overlap sum.
        municipal_pop = {
            rec.source_id: float(rec.population)
            for rec in registry.records_for(SOURCE_MUNICIPALITY)
            if rec.population is not None
        }
        muni_abs = overlap[
            (overlap["src_type"] == SOURCE_MUNICIPALITY)
            & (overlap["dst_type"] == SOURCE_ABS)
        ].copy()
        muni_abs["allocated"] = (
            muni_abs["src_id"].map(municipal_pop) * muni_abs["frac_of_src_pop"]
        )
        abs_pops = muni_abs.groupby("dst_id")["allocated"].sum(min_count=1)
        assert report.abs_population["max"] == pytest.approx(abs_pops.max(), rel=1e-5)
        assert 0 < report.abs_population["min"] <= report.abs_population["max"] <= 2e6

        # status passthrough wherever the registry knows the code
        status = dict(zip(root.coords["node"].values, root["status"].values))
        for node_type, sid in zip(endpoints["t"], endpoints["i"]):
            record = registry.get(str(node_type), str(sid))
            if record is not None:
                assert status[f"{node_type}:{sid}"] == record.status

    def test_real_build_roundtrip_and_determinism(self, tmp_path):
        report = build(output=tmp_path / "native_graph.zarr")
        root_a, rel_a = open_native_graph(tmp_path / "native_graph.zarr")

        assert root_a.attrs["schema"] == SCHEMA
        assert (
            root_a.attrs["provenance"]["overlap_edges"]["sha256"]
            == hashlib.sha256(REAL_OVERLAP.read_bytes()).hexdigest()
        )
        assert report["node_counts"] == {
            SOURCE_MUNICIPALITY: 963,
            SOURCE_ABS: 379,
            SOURCE_COMARCA: 43,
            MITMA: 456,
        }

        build(output=tmp_path / "again.zarr")
        root_b, rel_b = open_native_graph(tmp_path / "again.zarr")
        assert root_a.attrs == root_b.attrs
        assert_same_assembly((root_a, rel_a), (root_b, rel_b))
