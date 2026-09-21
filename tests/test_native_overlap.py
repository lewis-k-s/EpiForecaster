"""Tests for the overlap engine and edge tables (OVL-1, plan §12.2).

Toy fixtures: a synthetic 2×2 GEOSTAT-style grid of 1-km cells and
rectangular municipal/ABS/comarca polygons on the same lattice, so every
population allocation has a hand-computable expectation. A real-file
integration class (skipped when any input is absent) runs one full
in-memory build and asserts coverage of every observed ABS and every
MITMA zone in the od registry.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from data.native_registry import build_abs_registry
from data.preprocess.native_sources import overlap as ov

pytestmark = [
    pytest.mark.epiforecaster,
    pytest.mark.filterwarnings("ignore::UserWarning:geopandas"),
]

ORIGIN_X = 950000
ORIGIN_Y = 4650000
KM = 1000
EPS = 1e-9

REAL_MUNICIPAL = Path("data/files/geo/fl_municipios_catalonia.geojson")
REAL_ABS = Path("data/files/geo/cartografia/ABS.shp")
REAL_COMARCA = Path("data/files/geo/generated/comarques_igcg_v2r1.geojson")
REAL_MAPPING = Path("data/files/mpiscatalunya.csv")
REAL_ABS_CSV = Path("data/files/COVID-19__Persones_hospitalitzades.csv")
REAL_DEATHS = Path(
    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
)
REAL_OD_ZARR = Path("data/processed/native_support/od_mitma.zarr")

requires_real_inputs = pytest.mark.skipif(
    not all(
        p.exists()
        for p in (
            REAL_MUNICIPAL,
            REAL_ABS,
            REAL_COMARCA,
            REAL_MAPPING,
            REAL_ABS_CSV,
            REAL_DEATHS,
            REAL_OD_ZARR,
        )
    ),
    reason="real overlap inputs not available",
)


def _km_box(col: int, row: int, ncols: int = 1, nrows: int = 1):
    """Rectangle spanning ncols×nrows cells of the toy 1-km lattice."""
    return box(
        ORIGIN_X + col * KM,
        ORIGIN_Y + row * KM,
        ORIGIN_X + (col + ncols) * KM,
        ORIGIN_Y + (row + nrows) * KM,
    )


def _gdf(ids: list[str], geoms, id_col: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({id_col: ids, "geometry": list(geoms)}, crs=ov.TARGET_CRS)


@pytest.fixture
def grid_surface() -> ov.PopulationSurface:
    """2×2 grid; cell (col, row) populations 100/200/300/400."""
    cells = {(0, 0): 100.0, (1, 0): 200.0, (0, 1): 300.0, (1, 1): 400.0}
    pieces = gpd.GeoDataFrame(
        {
            "piece_id": [f"1kmN{r}E{c}" for c, r in cells],
            "population": list(cells.values()),
            "piece_area": [float(KM * KM)] * len(cells),
            "geometry": [_km_box(c, r) for c, r in cells],
        },
        crs=ov.TARGET_CRS,
    )
    return ov.PopulationSurface(pieces=pieces, label="toy_grid")


@pytest.fixture
def munis() -> gpd.GeoDataFrame:
    """Left and right 1×2 km columns, partitioning the 2×2 block."""
    return _gdf(
        ["08001", "08002"],
        [_km_box(0, 0, 1, 2), _km_box(1, 0, 1, 2)],
        "municipality_id",
    )


@pytest.fixture
def abs_rows() -> gpd.GeoDataFrame:
    """Bottom and top 2×1 km rows, partitioning the 2×2 block."""
    return _gdf(["001", "002"], [_km_box(0, 0, 2, 1), _km_box(0, 1, 2, 1)], "abs_id")


@pytest.fixture
def mapping() -> pd.DataFrame:
    """Both municipalities belong to comarca 01; 08003 has no geometry."""
    return pd.DataFrame(
        {
            "municipality_id": ["08001", "08002", "08003"],
            "comarca_id": ["01", "01", "01"],
        }
    )


def _assemble(surface, munis, abs_rows, mapping=None, **kwargs):
    frames = {"municipality": munis, "abs": abs_rows}
    if mapping is not None:
        frames["comarca"] = _gdf(["01"], [_km_box(0, 0, 2, 2)], "comarca_id")
    return ov.assemble_edge_tables(
        surface,
        frames,
        membership=mapping,
        provenance="toy|policy=test",
        **kwargs,
    )


class TestFractionalCover:
    def test_partial_cover_fraction(self, grid_surface):
        half = _gdf(
            ["half"],
            [box(ORIGIN_X, ORIGIN_Y, ORIGIN_X + KM, ORIGIN_Y + KM / 2)],
            "abs_id",
        )
        frac = ov._fractional_cover(grid_surface, half, "abs_id")
        assert frac["frac"].sum() == pytest.approx(0.5)

    def test_pop_est_and_coverage(self, grid_surface, munis, abs_rows):
        cache: dict[str, pd.DataFrame] = {}
        stats_m = ov.node_stats(grid_surface, munis, "municipality_id", cache)
        stats_a = ov.node_stats(grid_surface, abs_rows, "abs_id", cache)
        # Left column: cells (0,0)+(0,1) → 100+300; right: 200+400.
        assert stats_m["pop_est"]["08001"] == pytest.approx(400.0)
        assert stats_m["pop_est"]["08002"] == pytest.approx(600.0)
        assert stats_m["coverage"]["08001"] == pytest.approx(1.0)
        # Bottom row: cells (0,0)+(1,0) → 100+200; top: 300+400.
        assert stats_a["pop_est"]["001"] == pytest.approx(300.0)
        assert stats_a["pop_est"]["002"] == pytest.approx(700.0)
        # The same frac cache serves both node sets.
        assert set(cache) == {"municipality_id", "abs_id"}

    def test_coverage_below_one_outside_grid(self, grid_surface):
        # 1 km inside the grid + 1 km beyond it: coverage and pop halved.
        shifted = _gdf(
            ["far"],
            [box(ORIGIN_X - KM, ORIGIN_Y, ORIGIN_X + KM, ORIGIN_Y + KM)],
            "abs_id",
        )
        stats = ov.node_stats(grid_surface, shifted, "abs_id")
        assert stats["coverage"]["far"] == pytest.approx(0.5)
        # Cell (0,0) fully covered (the polygon's outer half is off-grid).
        assert stats["pop_est"]["far"] == pytest.approx(100.0)

    def test_unpopulated_target(self, grid_surface):
        empty = _gdf(["003"], [_km_box(5, 5)], "abs_id")
        stats = ov.node_stats(grid_surface, empty, "abs_id")
        assert stats["pop_est"]["003"] == 0.0
        assert stats["coverage"]["003"] == 0.0


class TestBilinearOverlap:
    def test_shared_population_2x2(self, grid_surface, munis, abs_rows):
        fa = ov._fractional_cover(grid_surface, munis, "municipality_id")
        fb = ov._fractional_cover(grid_surface, abs_rows, "abs_id")
        table = ov.overlap_pop_table(grid_surface, fa, fb)
        got = {(r.a_id, r.b_id): r.overlap_pop_est for r in table.itertuples()}
        # Each cell shared by exactly one (muni, ABS) pair.
        assert got[("08001", "001")] == pytest.approx(100.0)
        assert got[("08002", "001")] == pytest.approx(200.0)
        assert got[("08001", "002")] == pytest.approx(300.0)
        assert got[("08002", "002")] == pytest.approx(400.0)

    def test_symmetry_under_swap(self, grid_surface, munis, abs_rows):
        fa = ov._fractional_cover(grid_surface, munis, "municipality_id")
        fb = ov._fractional_cover(grid_surface, abs_rows, "abs_id")
        forward = ov.overlap_pop_table(grid_surface, fa, fb)
        backward = ov.overlap_pop_table(grid_surface, fb, fa)
        fwd = {(r.a_id, r.b_id): r.overlap_pop_est for r in forward.itertuples()}
        bwd = {(r.b_id, r.a_id): r.overlap_pop_est for r in backward.itertuples()}
        assert fwd == pytest.approx(bwd)

    def test_polygon_overlap_area(self):
        a = _gdf(["a"], [_km_box(0, 0, 2, 1)], "abs_id")
        b = _gdf(["b"], [_km_box(1, 0, 2, 1)], "abs_id")
        area = ov.polygon_overlap_area(a, "abs_id", b, "abs_id")
        assert area["overlap_area_m2"].iloc[0] == pytest.approx(float(KM * KM))
        far = _gdf(["c"], [_km_box(9, 9)], "abs_id")
        assert ov.polygon_overlap_area(a, "abs_id", far, "abs_id").empty


class TestAssembleEdges:
    @pytest.fixture
    def built(self, grid_surface, munis, abs_rows, mapping):
        return _assemble(grid_surface, munis, abs_rows, mapping)

    def test_schema_and_provenance(self, built):
        table, _ = built
        assert list(table.columns) == list(ov.EDGE_COLUMNS)
        assert (table["provenance"] == "toy|policy=test").all()

    def test_both_directions_emitted(self, built):
        table, _ = built
        muni_abs = table[
            (table.src_type == "municipality") & (table.dst_type == "abs")
        ]
        abs_muni = table[(table.src_type == "abs") & (table.dst_type == "municipality")]
        assert len(muni_abs) == 4
        assert len(abs_muni) == 4

    def test_reciprocity(self, built):
        table, _ = built
        muni_abs = table[
            (table.src_type == "municipality") & (table.dst_type == "abs")
        ].set_index(["src_id", "dst_id"]).sort_index()
        abs_muni = table[
            (table.src_type == "abs") & (table.dst_type == "municipality")
        ].set_index(["dst_id", "src_id"]).sort_index()
        assert muni_abs.index.tolist() == abs_muni.index.tolist()
        area_f = muni_abs["overlap_area_m2"].to_numpy()
        area_b = abs_muni["overlap_area_m2"].to_numpy()
        pop_f = muni_abs["overlap_pop_est"].to_numpy()
        pop_b = abs_muni["overlap_pop_est"].to_numpy()
        assert area_f == pytest.approx(area_b)
        assert pop_f == pytest.approx(pop_b)
        # Same shared population, different endpoint fractions (§3).
        frac_f = muni_abs["frac_of_src_pop"].to_numpy()
        frac_b = abs_muni["frac_of_src_pop"].to_numpy()
        assert not (frac_f == pytest.approx(frac_b))

    def test_fractions_bounded(self, built):
        table, _ = built
        for col in ("frac_of_src_pop", "frac_of_dst_pop"):
            assert (table[col] >= -EPS).all()
            assert (table[col] <= 1.0 + EPS).all()

    def test_membership_fractions(self, built):
        table, report = built
        to_comarca = table[
            (table.src_type == "municipality") & (table.dst_type == "comarca")
        ]
        assert len(to_comarca) == 2
        assert to_comarca["frac_of_src_pop"].to_numpy() == pytest.approx(1.0)
        # Comarca destination share: Σ_muni frac_of_dst == 1 over mapped members.
        assert to_comarca["frac_of_dst_pop"].sum() == pytest.approx(1.0)
        # Overlap is the full municipal polygon area and grid-implied pop.
        row = to_comarca[to_comarca.src_id == "08001"].iloc[0]
        assert row["overlap_area_m2"] == pytest.approx(2.0 * KM * KM)
        assert row["overlap_pop_est"] == pytest.approx(400.0)
        from_comarca = table[
            (table.src_type == "comarca") & (table.dst_type == "municipality")
        ]
        # Share of comarca: 400/1000 and 600/1000.
        assert sorted(from_comarca["frac_of_src_pop"]) == pytest.approx(
            [0.4, 0.6]
        )
        assert from_comarca["frac_of_dst_pop"].to_numpy() == pytest.approx(1.0)
        relation = report["rows_per_relation"]["municipality-comarca"]
        assert relation["pairs"] == 2
        assert relation["unmapped_member_ids"] == ["08003"]

    def test_destination_deficit_recorded(self, grid_surface):
        # One municipal source under a destination extending beyond it:
        # pop_est(dst) = 1000, shared = 100 → column sum 0.1, deficit 0.9.
        munis = _gdf(["08001"], [_km_box(0, 0)], "municipality_id")
        big_abs = _gdf(["001"], [_km_box(0, 0, 2, 2)], "abs_id")
        table, report = _assemble(grid_surface, munis, big_abs)
        sel = table[table.src_type == "municipality"]
        assert sel["frac_of_dst_pop"].sum() == pytest.approx(0.1)
        deficit = report["destination_deficit"]["municipality-abs"]
        assert deficit["max_deficit"] == pytest.approx(0.9)
        assert deficit["nodes_with_deficit"] == 1

    def test_full_partition_deficit_zero(self, built):
        _, report = built
        deficit = report["destination_deficit"]["municipality-abs"]
        assert deficit["max_deficit"] == pytest.approx(0.0, abs=1e-12)

    def test_zero_overlap_nodes_reported(self, grid_surface, munis, abs_rows, mapping):
        far = _gdf(["003"], [_km_box(9, 9)], "abs_id")
        abs_with_far = pd.concat([abs_rows, far], ignore_index=True)
        _, report = _assemble(grid_surface, munis, abs_with_far, mapping)
        assert report["zero_overlap_nodes"]["abs"] == ["003"]

    def test_zero_pop_node_gets_no_edge(self, grid_surface, munis, abs_rows):
        # A municipality over unpopulated cells: pop_est 0 → no fractions,
        # no fabricated edges; surfaced as zero-overlap/zero-pop instead.
        ghost = _gdf(["08009"], [_km_box(5, 5)], "municipality_id")
        munis_with_ghost = pd.concat([munis, ghost], ignore_index=True)
        table, report = _assemble(grid_surface, munis_with_ghost, abs_rows)
        assert table[table.src_id == "08009"].empty
        assert "08009" in report["zero_overlap_nodes"]["municipality"]


class TestUniformDensityPolicy:
    @pytest.fixture
    def uniform(self, munis):
        surface, missing = ov.uniform_density_surface(
            munis, {"08001": 400.0, "08002": 600.0}
        )
        assert missing == []
        return surface

    def test_uniform_allocation(self, uniform, abs_rows):
        stats = ov.node_stats(uniform, abs_rows, "abs_id")
        # Each ABS row takes half of every municipal piece: 200+300 = 500.
        assert stats["pop_est"]["001"] == pytest.approx(500.0)
        assert stats["pop_est"]["002"] == pytest.approx(500.0)

    def test_uniform_overlap_and_fractions(self, uniform, munis, abs_rows, mapping):
        table, _ = _assemble(uniform, munis, abs_rows, mapping)
        sel = table[
            (table.src_type == "municipality") & (table.dst_type == "abs")
        ]
        # muni 08001 (audited 400, area 2 km²) ∩ ABS 001 (1 km² of it):
        # uniform density → 200 shared, 0.5 of the municipality.
        row = sel[(sel.src_id == "08001") & (sel.dst_id == "001")].iloc[0]
        assert row["overlap_pop_est"] == pytest.approx(200.0)
        assert row["frac_of_src_pop"] == pytest.approx(0.5)
        # Membership rows now use audited populations: 08001 → comarca pop 1000.
        member = table[
            (table.src_type == "municipality") & (table.dst_type == "comarca")
        ]
        assert sorted(member["frac_of_dst_pop"]) == pytest.approx([0.4, 0.6])

    def test_missing_audited_population_excluded(self, munis):
        surface, missing = ov.uniform_density_surface(munis, {"08001": 400.0})
        assert missing == ["08002"]
        assert len(surface.pieces) == 1


class TestParquetRoundTrip:
    def test_write_read_schema(self, grid_surface, munis, abs_rows, mapping, tmp_path):
        table, _ = _assemble(grid_surface, munis, abs_rows, mapping)
        out = tmp_path / "overlap_edges.parquet"
        table.to_parquet(out, index=False)
        back = pd.read_parquet(out)
        assert list(back.columns) == list(ov.EDGE_COLUMNS)
        assert len(back) == len(table)
        assert back["overlap_pop_est"].dtype == "float64"


@requires_real_inputs
class TestRealBuild:
    """One full in-memory build shared by all real-data assertions."""

    @pytest.fixture(scope="class")
    def real(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("real_overlap") / "overlap_edges.parquet"
        report = ov.build(write=True, out_path=out, report_path=None)
        return pd.read_parquet(out), report

    def test_schema_and_provenance(self, real):
        table, report = real
        assert list(table.columns) == list(ov.EDGE_COLUMNS)
        prov = report["provenance"]
        assert prov["policy"] == "geostat_grid"
        assert prov["grid_vintage"] == "GEOSTAT 2011"
        assert "GEOSTAT 2011" in prov["provenance_string"]
        assert len(prov["input_sha256"]) == 7
        assert prov["od_mitma_zone_count"] == 456

    def test_relations_present(self, real):
        _, report = real
        relations = report["rows_per_relation"]
        for name in (
            "municipality-abs",
            "municipality-comarca",
            "mitma-municipality",
            "mitma-abs",
            "mitma-comarca",
        ):
            assert relations[name]["pairs"] > 0

    def test_every_observed_abs_has_municipal_overlap(self, real):
        table, _ = real
        abs_ids = ov.load_geometry(REAL_ABS, "CODABSa", "abs_id", 3)["abs_id"]
        registry, _ = build_abs_registry(REAL_ABS_CSV, geometry_ids=abs_ids)
        observed = {
            r.source_id for r in registry.records_for("abs") if r.status == "observed"
        }
        muni_abs = table[
            (
                (table.src_type == "municipality") & (table.dst_type == "abs")
            ) | (
                (table.src_type == "abs") & (table.dst_type == "municipality")
            )
        ]
        covered = set(
            muni_abs.loc[
                (muni_abs.src_type == "abs") & (muni_abs.overlap_pop_est > 0), "src_id"
            ]
        ) | set(
            muni_abs.loc[
                (muni_abs.dst_type == "abs") & (muni_abs.overlap_pop_est > 0), "dst_id"
            ]
        )
        assert observed <= covered

    def test_mitma_coverage_recorded_exceptions(self, real):
        """MITMA zero-overlap zones are recorded, never forced (ruling)."""
        table, report = real
        zone_ids = set(ov.load_od_mitma_zone_ids(REAL_OD_ZARR))
        assert len(zone_ids) == 456
        sel = table[
            ((table.src_type == "mitma") & (table.dst_type == "municipality"))
            | ((table.src_type == "municipality") & (table.dst_type == "mitma"))
        ]
        covered = set(sel.loc[sel.overlap_pop_est > 0, "src_id"]) | set(
            sel.loc[sel.overlap_pop_est > 0, "dst_id"]
        )
        recorded = report["mitma_municipal_overlap"]
        # ≥90% of the registry must have nonzero municipal overlap...
        assert len(covered & zone_ids) >= 0.9 * len(zone_ids)
        # ...and every uncovered zone is explicitly listed in the report.
        assert set(recorded["zero_overlap_zone_ids"]) == zone_ids - covered
        assert recorded["covered_nonzero"] == len(covered & zone_ids)
        # All 442 prefix zones are covered or explicitly listed.
        prefix = set(ov.load_od_mitma_prefix_zone_ids(REAL_OD_ZARR))
        assert prefix <= covered | set(recorded["zero_overlap_zone_ids"])
        assert set(recorded["prefix_zero_overlap_zone_ids"]) == (
            prefix - covered
        )
        assert len(recorded["zero_overlap_zone_ids"]) < 10

    def test_every_observed_comarca_has_membership_overlap(self, real):
        """Hard clinical assertion: observed comarcas have nonzero rows."""
        table, _ = real
        codes = pd.read_csv(
            REAL_DEATHS, usecols=["Codi Comarca"], dtype={"Codi Comarca": str}
        )["Codi Comarca"]
        observed = {
            str(c).strip().zfill(2) for c in codes.dropna().unique()
        } - {"00"}
        member = table[
            ((table.src_type == "municipality") & (table.dst_type == "comarca"))
            | ((table.src_type == "comarca") & (table.dst_type == "municipality"))
        ]
        covered = set(member.loc[member.overlap_pop_est > 0, "src_id"]) | set(
            member.loc[member.overlap_pop_est > 0, "dst_id"]
        )
        assert observed <= covered

    def test_fractions_bounded_and_membership_complete(self, real):
        table, _ = real
        for col in ("frac_of_src_pop", "frac_of_dst_pop"):
            assert (table[col] >= -EPS).all()
            assert (table[col] <= 1.0 + EPS).all()
            assert table[col].notna().all()
        to_comarca = table[
            (table.src_type == "municipality") & (table.dst_type == "comarca")
        ]
        assert to_comarca["frac_of_src_pop"].to_numpy() == pytest.approx(1.0)
        per_muni = to_comarca.groupby("src_id").size()
        assert (per_muni == 1).all()
