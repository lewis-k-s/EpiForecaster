"""Tests for the comarca geometry builder (GEO-1, plan §12.2).

Toy 4-municipality / 2-comarca EPSG:3042 fixtures verify the dissolve
partition, ID propagation, area closure and reconciliation statuses.
A real-file integration class (skipped when any input is absent)
asserts the code set matches the deaths CSV, flags codes without
geometry as ``unmapped`` and records (never fails on) vintage
differences between the 2025 IGCG boundaries and the study-period
municipal dissolve.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from data.preprocess.native_sources import comarca_geometry as cg

pytestmark = pytest.mark.epiforecaster

REAL_IGCG = cg.DEFAULT_IGCG_SHAPEFILE
REAL_MUNICIPAL = Path("data/files/geo/fl_municipios_catalonia.geojson")
REAL_MAPPING = Path("data/files/mpiscatalunya.csv")
REAL_DEATHS = Path(
    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
)

requires_real_inputs = pytest.mark.skipif(
    not all(p.exists() for p in (REAL_IGCG, REAL_MUNICIPAL, REAL_MAPPING, REAL_DEATHS)),
    reason="real comarca geometry inputs not available",
)

ORIGIN_X = 950000
ORIGIN_Y = 4650000
KM = 1000


def _km_box(col: int, row: int):
    """1-km square on a grid anchored at (ORIGIN_X, ORIGIN_Y)."""
    return box(
        ORIGIN_X + col * KM,
        ORIGIN_Y + row * KM,
        ORIGIN_X + (col + 1) * KM,
        ORIGIN_Y + (row + 1) * KM,
    )


class TestDissolveMunicipalComarques:
    @pytest.fixture
    def municipalities(self):
        records = [
            ("08001", "Muni One", _km_box(0, 0)),
            ("08002", "Muni Two", _km_box(1, 0)),
            ("08003", "Muni Three", _km_box(0, 1)),
            ("08004", "Muni Four", _km_box(1, 1)),
        ]
        return gpd.GeoDataFrame(
            {
                "municipality_id": [r[0] for r in records],
                "municipality_name": [r[1] for r in records],
                "geometry": [r[2] for r in records],
            },
            crs=cg.TARGET_CRS,
        )

    @pytest.fixture
    def mapping(self):
        return pd.DataFrame(
            {
                "municipality_id": ["08001", "08002", "08003", "08004", "08005"],
                "municipality_name": ["Muni One", "Muni Two", "Muni Three", "Muni Four", "Ghost"],
                "comarca_id": ["01", "01", "02", "02", "03"],
                "comarca_name": ["Comarca Un", "Comarca Un", "Comarca Dos", "Comarca Dos", "Comarca Tres"],
            }
        )

    @pytest.fixture
    def dissolved(self, municipalities, mapping):
        return cg.dissolve_municipal_comarques(municipalities, mapping)

    def test_partition_and_membership(self, dissolved):
        geo = dissolved.geometries
        assert sorted(geo["comarca_id"]) == ["01", "02"]
        assert dict(zip(geo["comarca_id"], geo["membership_count"])) == {"01": 2, "02": 2}

    def test_each_mapped_municipality_counted_once(self, dissolved):
        assert sum(dissolved.geometries["membership_count"]) == 4

    def test_id_propagation_matches_mapped_comarca_set(self, dissolved, mapping):
        mapped_set = set(mapping[mapping["municipality_id"] != "08005"]["comarca_id"])
        assert set(dissolved.geometries["comarca_id"]) == mapped_set

    def test_area_closure_within_tolerance(self, dissolved):
        assert dissolved.area_closure_max_rel_error < 1e-9
        assert dissolved.municipal_part_area_m2 == {"01": 2_000_000.0, "02": 2_000_000.0}
        areas = dict(
            zip(dissolved.geometries["comarca_id"], dissolved.geometries["dissolved_area_m2"])
        )
        assert areas == {"01": 2_000_000.0, "02": 2_000_000.0}

    def test_unmapped_and_missing_geometry_recorded(self, dissolved):
        assert dissolved.unmapped_municipality_ids == []
        assert dissolved.mapping_ids_without_geometry == ["08005"]

    def test_unmapped_polygons_excluded_from_dissolve(self, municipalities, mapping):
        extra = gpd.GeoDataFrame(
            {
                "municipality_id": ["22001"],
                "municipality_name": ["Border Muni"],
                "geometry": [_km_box(2, 0)],
            },
            crs=cg.TARGET_CRS,
        )
        result = cg.dissolve_municipal_comarques(
            pd.concat([municipalities, extra], ignore_index=True), mapping
        )
        assert result.unmapped_municipality_ids == ["22001"]
        assert set(result.geometries["comarca_id"]) == {"01", "02"}

    def test_missing_municipality_id_column_raises(self, mapping):
        bare = gpd.GeoDataFrame({"geometry": [_km_box(0, 0)]}, crs=cg.TARGET_CRS)
        with pytest.raises(ValueError, match="municipality_id"):
            cg.dissolve_municipal_comarques(bare, mapping)


class TestReconcile:
    @pytest.fixture
    def municipalities(self):
        return gpd.GeoDataFrame(
            {
                "municipality_id": ["08001", "08002", "08003", "08004"],
                "geometry": [_km_box(0, 0), _km_box(1, 0), _km_box(0, 1), _km_box(1, 1)],
            },
            crs=cg.TARGET_CRS,
        )

    @pytest.fixture
    def mapping(self):
        return pd.DataFrame(
            {
                "municipality_id": ["08001", "08002", "08003", "08004"],
                "comarca_id": ["01", "01", "02", "02"],
                "comarca_name": ["Comarca Un", "Comarca Un", "Comarca Dos", "Comarca Dos"],
            }
        )

    @pytest.fixture
    def dissolved(self, municipalities, mapping):
        return cg.dissolve_municipal_comarques(municipalities, mapping)

    @pytest.fixture
    def igcg(self):
        """IGCG-like frame: 01 matched (slightly small -> positive diff), 03 igcg_only."""
        records = [
            ("01", "Comarca Un", 950000, 4650000, 951900, 4651000),
            ("03", "Comarca Tres", 952000, 4650000, 953000, 4651000),
        ]
        gdf = gpd.GeoDataFrame(
            {
                "comarca_id": [r[0] for r in records],
                "comarca_name": [r[1] for r in records],
                "igcg_reference_area_km2": [
                    (r[4] - r[2]) * (r[5] - r[3]) / 1e6 for r in records
                ],
                "geometry": [box(r[2], r[3], r[4], r[5]) for r in records],
            },
            crs=cg.TARGET_CRS,
        )
        gdf["igcg_area_m2"] = gdf.geometry.area
        return gdf

    def test_statuses(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved)
        statuses = {r["comarca_id"]: r["status"] for r in report["comarcas"]}
        assert statuses == {"01": "matched", "02": "dissolve_only", "03": "igcg_only"}

    def test_relative_area_difference_sign_and_none(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved)
        by_id = {r["comarca_id"]: r for r in report["comarcas"]}
        # Dissolved 01 is 2e6 m2, IGCG 01 is 1.9e6 m2 -> positive relative diff.
        assert by_id["01"]["relative_area_difference"] == pytest.approx(0.1 / 1.9)
        assert by_id["01"]["igcg_area_km2"] == pytest.approx(1.9)
        assert by_id["01"]["dissolved_area_km2"] == pytest.approx(2.0)
        # Ids present on only one side have no relative difference.
        assert by_id["02"]["relative_area_difference"] is None
        assert by_id["02"]["igcg_area_km2"] is None
        assert by_id["02"]["membership_count"] == 2
        assert by_id["03"]["relative_area_difference"] is None
        assert by_id["03"]["dissolved_area_km2"] is None
        assert by_id["03"]["membership_count"] is None

    def test_id_diffs(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved)
        assert report["id_diffs"] == {
            "igcg_only": ["03"],
            "dissolve_only": ["02"],
            "both_count": 1,
        }

    def test_dissolve_section(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved)
        section = report["dissolve"]
        assert section["membership_counts"] == {"01": 2, "02": 2}
        assert section["mapped_municipality_count"] == 4
        assert section["area_closure_max_rel_error"] < 1e-9

    def test_observed_codes_section(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved, observed_codes=["01", "00"])
        section = report["observed_codes"]
        assert section["codes_without_geometry"] == ["00"]
        assert section["codes_without_geometry_status"] == "unmapped"
        assert section["geometry_without_observed_codes"] == ["02", "03"]
        assert section["matched_codes"] == ["01"]

    def test_vintage_note_present(self, igcg, dissolved):
        report = cg.reconcile(igcg, dissolved)
        assert "2025-07-30" in report["vintage_note"]

    def test_report_is_json_serializable(self, igcg, dissolved):
        import json

        report = cg.reconcile(igcg, dissolved, observed_codes=["01"])
        assert json.loads(json.dumps(report)) == report


class TestLoadMappingCsv:
    def test_parses_bom_header_and_trailing_comma(self, tmp_path):
        content = (
            "Títol\n"
            "Segona línia\n"
            "Tercera línia\n"
            "Codi,Nom,Codi comarca,Nom comarca,\n"
            "080193,Barcelona,11,Barcelonès,\n"
            "250170,Alins,26,Pallars Sobirà,\n"
        )
        path = tmp_path / "mpi.csv"
        path.write_text(content, encoding="utf-8-sig")
        mapping = cg.load_municipality_comarca_mapping(path)
        assert len(mapping) == 2
        assert mapping["municipality_id"].tolist() == ["08019", "25017"]
        assert mapping["comarca_id"].tolist() == ["11", "26"]
        assert mapping["comarca_name"].tolist() == ["Barcelonès", "Pallars Sobirà"]

    def test_rejects_missing_codes(self, tmp_path):
        content = (
            "Títol\n"
            "Segona línia\n"
            "Tercera línia\n"
            "Codi,Nom,Codi comarca,Nom comarca,\n"
            "080193,Barcelona,,Barcelonès,\n"
        )
        path = tmp_path / "mpi.csv"
        path.write_text(content, encoding="utf-8-sig")
        with pytest.raises(ValueError, match="comarca codes"):
            cg.load_municipality_comarca_mapping(path)


@pytest.mark.integration
@requires_real_inputs
class TestRealComarcaGeometry:
    @pytest.fixture
    def igcg(self):
        return cg.load_igcg_comarques(REAL_IGCG)

    @pytest.fixture
    def dissolved(self):
        mapping = cg.load_municipality_comarca_mapping(REAL_MAPPING)
        municipalities = gpd.read_file(REAL_MUNICIPAL)
        municipalities["municipality_id"] = (
            municipalities["id"].astype(str).str.zfill(5)
        )
        return cg.dissolve_municipal_comarques(municipalities, mapping)

    @pytest.fixture
    def report(self, igcg, dissolved):
        codes = pd.read_csv(  # type: ignore[call-arg]
            REAL_DEATHS,
            usecols=["Codi Comarca"],
            dtype={"Codi Comarca": str},
        )["Codi Comarca"]
        return cg.reconcile(igcg, dissolved, observed_codes=codes.dropna().unique().tolist())

    def test_igcg_normalized(self, igcg):
        assert len(igcg) == 43
        assert list(igcg["comarca_id"]) == [f"{i:02d}" for i in range(1, 44)]
        assert igcg.crs.to_epsg() == 3042
        assert (igcg["igcg_area_m2"] > 0).all()
        assert igcg["comarca_id"].str.fullmatch(r"\d{2}").all()

    def test_dissolve_partition_and_closure(self, dissolved):
        assert len(dissolved.geometries) == 43
        assert dissolved.geometries["membership_count"].sum() == 945
        assert dissolved.area_closure_max_rel_error < 1e-9

    def test_mapping_ids_without_geometry(self, dissolved):
        assert dissolved.mapping_ids_without_geometry == ["25017", "25123"]

    def test_unmapped_municipalities_recorded(self, dissolved):
        assert "53055" in dissolved.unmapped_municipality_ids
        assert len(dissolved.unmapped_municipality_ids) == 18

    def test_deaths_code_set(self):
        codes = pd.read_csv(  # type: ignore[call-arg]
            REAL_DEATHS,
            usecols=["Codi Comarca"],
            dtype={"Codi Comarca": str},
        )["Codi Comarca"].dropna().unique().tolist()
        assert set(codes) == {"00"} | {f"{i:02d}" for i in range(1, 43)}

    def test_observed_codes_reconciliation(self, report):
        section = report["observed_codes"]
        assert section["codes_without_geometry"] == ["00"]
        assert section["codes_without_geometry_status"] == "unmapped"
        assert section["geometry_without_observed_codes"] == ["43"]
        assert section["matched_codes"] == [f"{i:02d}" for i in range(1, 43)]

    def test_id_diffs_empty(self, report):
        assert report["id_diffs"]["igcg_only"] == []
        assert report["id_diffs"]["dissolve_only"] == []
        assert report["id_diffs"]["both_count"] == 43

    def test_pallars_sobira_gap_recorded_not_fatal(self, report):
        row = next(r for r in report["comarcas"] if r["comarca_id"] == "26")
        assert "allars" in row["comarca_name"]
        assert row["status"] == "matched"
        # Alins + Lladorre (~330 km2) are outside the study-period dissolve.
        assert row["relative_area_difference"] == pytest.approx(-0.2407, abs=0.01)
        assert row["dissolved_area_km2"] == pytest.approx(1049.2, abs=10.0)
        assert row["igcg_area_km2"] == pytest.approx(1381.8, abs=10.0)

    def test_all_other_matches_within_vintage_drift(self, report):
        for row in report["comarcas"]:
            if row["comarca_id"] != "26":
                assert abs(row["relative_area_difference"]) < 0.02
