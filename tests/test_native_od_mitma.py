"""Tests for native MITMA OD harmonization (slice MOB-2; plan §12.2).

Unit tests run on synthetic zone sets, polygons, and daily CSV fixtures; the
real-file tests (geometry derivation, --limit 3 build) are skipif-guarded on
the read-only dasymetric_mob source root and stay well under 20s.
"""

import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import box

from data.preprocess.native_sources.od_mitma import (
    DEFAULT_FOOTPRINT,
    EXPECTED_FIRST_DATE,
    EXPECTED_LAST_DATE,
    REFERENCE_ZONE_COUNTS,
    STRATA_NOTE,
    DailyMatrix,
    ZoneSet,
    build_od_mitma,
    derive_zone_set,
    expected_dates,
    find_daily_files,
    harmonize_daily_csv,
    load_zone_populations,
    resolve_source_layout,
    zone_set_from_geodataframes,
)

pytestmark = pytest.mark.epiforecaster

SOURCE_ROOT = Path("/Volumes/HUBSSD/code/meprecisa/dasymetric_mob")
REAL_SHP = SOURCE_ROOT / "data" / "mitma_v1" / "zonificacion-distritos" / "distritos_mitma.shp"
DAILY_HEADER = (
    "date\torigin\tdestination\tactivity_origin\tactivity_destination\tdistance\t"
    "residency\tage\tstudy_origin_possible\tstudy_destination_possible\tsex\tincome\ttrips\ttrips_km"
)


def _write_daily(path: Path, date: str, rows: list[tuple[str, str, float]]) -> Path:
    lines = [DAILY_HEADER]
    for origin, destination, trips in rows:
        lines.append(
            "\t".join(
                [date, origin, destination, "home", "work", "002-005", "01", "", "", "", "", "", f"{trips}", f"{trips * 4.0}"]
            )
        )
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


class TestZoneSetUnion:
    def _gdf(self, records: list[tuple[str, tuple[float, float, float, float]]]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"ID": [rid for rid, _ in records]},
            geometry=[box(*bounds) for _, bounds in records],
            crs="EPSG:3042",
        )

    def test_union_membership_and_flags(self):
        zones = self._gdf(
            [
                ("08019", (0.0, 0.0, 10.0, 10.0)),  # prefix, inside footprint
                ("08999", (100.0, 100.0, 110.0, 110.0)),  # prefix, far outside
                ("99999", (5.0, 5.0, 15.0, 15.0)),  # non-prefix, overlaps footprint
                ("88888", (200.0, 200.0, 210.0, 210.0)),  # non-prefix, far away
            ]
        )
        footprint = self._gdf([("00000", (0.0, 0.0, 10.0, 10.0))])

        zone_set = zone_set_from_geodataframes(zones, footprint)

        assert zone_set.zone_ids == ("08019", "08999", "99999")
        assert zone_set.from_prefix == (True, True, False)
        assert zone_set.in_footprint == (True, False, True)
        assert zone_set.n_shp_zones == 4


class TestHarmonizeDaily:
    @pytest.fixture
    def zone_index(self) -> dict[str, int]:
        return {"01001_AM": 0, "08019": 1}

    @pytest.fixture
    def daily_file(self, tmp_path: Path) -> Path:
        return _write_daily(
            tmp_path / "2020-02-14.daily_mobility.v1.recurrent.csv.gz",
            "2020-02-14",
            rows=[
                ("01001_AM", "08019", 1.5),  # strata rows of the same pair
                ("01001_AM", "08019", 2.25),
                ("01001_AM", "08019", 0.25),
                ("08019", "01001_AM", 7.0),  # asymmetric reverse direction
                ("01001_AM", "01001_AM", 2.0),  # self flow retained in the matrix
                ("01001_AM", "99999", 100.0),  # outsider endpoints dropped
                ("99999", "08019", 50.0),
                ("99999", "99999", 3.0),
            ],
        )

    def test_strata_summation_filtering_and_dense_fill(self, daily_file: Path, zone_index: dict[str, int]):
        daily = harmonize_daily_csv(daily_file, zone_index)

        assert isinstance(daily, DailyMatrix)
        assert daily.matrix.dtype == np.float32
        assert daily.matrix.shape == (2, 2)
        assert daily.matrix[0, 1] == np.float32(4.0)  # strata summed
        assert daily.matrix[1, 0] == np.float32(7.0)  # direction preserved, not symmetrized
        assert daily.matrix[0, 0] == np.float32(2.0)  # self flow kept
        assert daily.matrix[1, 1] == np.float32(0.0)  # absent pair zero-filled
        assert daily.mass_total == pytest.approx(166.0)  # ALL raw rows, pre-filter
        assert daily.mass_zone_set == pytest.approx(13.0)  # after endpoint filter
        assert daily.max_strata_rel_err < 1e-6

    def test_date_column_must_match_filename(self, tmp_path: Path, zone_index: dict[str, int]):
        mismatched = _write_daily(
            tmp_path / "2020-02-15.daily_mobility.v1.recurrent.csv.gz",
            "2020-02-14",
            rows=[("01001_AM", "08019", 1.0)],
        )
        with pytest.raises(ValueError, match="does not match filename"):
            harmonize_daily_csv(mismatched, zone_index)

        unrecognized = tmp_path / "notes.csv.gz"
        unrecognized.write_text("nothing")
        with pytest.raises(ValueError, match="unrecognized daily mobility file name"):
            harmonize_daily_csv(unrecognized, zone_index)


class TestBuildStore:
    @pytest.fixture
    def zone_set(self) -> ZoneSet:
        return ZoneSet(
            zone_ids=("08019", "170092", "44001"),
            from_prefix=(True, True, False),
            in_footprint=(True, True, True),
        )

    @pytest.fixture
    def source_dir(self, tmp_path: Path) -> Path:
        mitma_root = tmp_path / "mitma_v1"
        mitma_root.mkdir()
        _write_daily(
            mitma_root / "2020-02-14.daily_mobility.v1.recurrent.csv.gz",
            "2020-02-14",
            rows=[
                ("08019", "170092", 1.0),
                ("08019", "170092", 2.0),
                ("170092", "08019", 5.0),
                ("44001", "08019", 2.0),
                ("08019", "08019", 4.0),
                ("99999", "08019", 100.0),  # outside zone set
            ],
        )
        _write_daily(
            mitma_root / "2020-02-16.daily_mobility.v1.recurrent.csv.gz",  # 02-15 missing on purpose
            "2020-02-16",
            rows=[
                ("08019", "170092", 7.0),
                ("08019", "170092", 12.5),
                ("44001", "44001", 1.0),
            ],
        )
        (tmp_path / "poblacion.csv").write_text(
            "distrito|municipio|provincia|poblacion\n08019|08019|08|100.0\n170092|170092|17|250.5\n"
        )
        return tmp_path

    @pytest.fixture
    def store(self, tmp_path: Path, source_dir: Path, zone_set: ZoneSet) -> xr.Dataset:
        out_path = tmp_path / "od_mitma.zarr"
        build_od_mitma(
            source_root=source_dir,
            out_path=out_path,
            zones=zone_set,
            population_path=source_dir / "poblacion.csv",
            start="2020-02-14",
            end="2020-02-16",
        )
        return xr.open_zarr(out_path)

    def test_dims_coords_and_dtypes(self, store: xr.Dataset, zone_set: ZoneSet):
        assert dict(store.sizes) == {"date": 2, "origin": 3, "target": 3, "zone": 3}
        assert store.mobility.dtype == np.float32
        assert store.mobility.chunks[0] == (1, 1)
        assert store.mobility.chunks[1] == (3,)
        np.testing.assert_array_equal(
            store.date.values,
            np.array(["2020-02-14", "2020-02-16"], dtype="datetime64[ns]"),
        )
        for coord in ("origin", "target", "zone_id"):
            assert list(store[coord].values) == list(zone_set.zone_ids)
        assert store.zone_population.dtype == np.float32

    def test_mobility_values_match_manual_groupby(self, store: xr.Dataset):
        expected_d1 = np.zeros((3, 3), dtype=np.float32)
        expected_d1[0, 1] = 3.0
        expected_d1[1, 0] = 5.0
        expected_d1[2, 0] = 2.0
        expected_d1[0, 0] = 4.0
        expected_d3 = np.zeros((3, 3), dtype=np.float32)
        expected_d3[0, 1] = 19.5
        expected_d3[2, 2] = 1.0

        np.testing.assert_array_equal(store.mobility.values[0], expected_d1)
        np.testing.assert_array_equal(store.mobility.values[1], expected_d3)

    def test_registry_roundtrip(self, store: xr.Dataset):
        np.testing.assert_array_equal(store.zone_population.values[:2], np.array([100.0, 250.5], dtype=np.float32))
        assert np.isnan(store.zone_population.values[2])  # absent from poblacion.csv
        assert store.in_footprint.values.tolist() == [True, True, True]
        assert store.from_prefix.values.tolist() == [True, True, False]

    def test_audit_attrs(self, store: xr.Dataset):
        attrs = store.attrs
        assert attrs["schema"] == "od_mitma.v1"
        assert attrs["zone_set_derivation"] == "injected"
        assert attrs["crs"] == "EPSG:3042"
        assert attrs["reference_zone_counts"] == REFERENCE_ZONE_COUNTS
        assert attrs["n_dates_expected_full"] == 451
        assert attrs["n_dates_expected"] == 3
        assert attrs["n_dates_processed"] == 2
        assert attrs["dates_missing"] == ["2020-02-15"]
        assert attrs["limit"] is None
        assert attrs["zone_prefix_count"] == 2
        assert attrs["zone_population_available"] == 2
        assert attrs["diag_mass_fraction"] == pytest.approx(5.0 / 34.5)
        assert attrs["mass_in_zone_set_fraction"] == pytest.approx(34.5 / 134.5)
        assert attrs["strata_summation_assertion"] == "passed"
        assert attrs["strata_summation_note"] == STRATA_NOTE

        assert attrs["retained_mass_retain_minflow0"] == pytest.approx(34.5)
        assert attrs["retained_mass_fraction_retain_minflow0"] == pytest.approx(1.0)
        assert attrs["edge_count_retain_minflow0"] == 6
        assert attrs["retained_mass_retain_minflow10"] == pytest.approx(19.5)
        assert attrs["edge_count_retain_minflow10"] == 1
        assert attrs["retained_mass_drop_minflow0"] == pytest.approx(29.5)  # minus diag 5.0
        assert attrs["edge_count_drop_minflow0"] == 4
        assert attrs["retained_mass_drop_minflow10"] == pytest.approx(19.5)
        assert attrs["edge_count_drop_minflow10"] == 1


class TestDeriveZoneSetReal:
    @pytest.fixture(autouse=True)
    def require_real_geometry(self):
        if not REAL_SHP.is_file() or not DEFAULT_FOOTPRINT.is_file():
            pytest.skip("real MITMA shapefile or municipal footprint not present")

    def test_prefix_count_and_determinism(self):
        first = derive_zone_set(REAL_SHP, DEFAULT_FOOTPRINT)
        second = derive_zone_set(REAL_SHP, DEFAULT_FOOTPRINT)

        assert first.n_shp_zones == 2850
        assert sum(first.from_prefix) == REFERENCE_ZONE_COUNTS["prefix_verified"] == 442
        assert all(first.from_prefix[i] == (z[:2] in ("08", "17", "25", "43")) for i, z in enumerate(first.zone_ids))
        assert list(first.zone_ids) == sorted(first.zone_ids)
        assert first.zone_ids == second.zone_ids
        assert first.from_prefix == second.from_prefix
        assert first.in_footprint == second.in_footprint


class TestIntegrationReal:
    @pytest.fixture(autouse=True)
    def require_real_source(self):
        if not (SOURCE_ROOT / "data" / "mitma_v1").is_dir():
            pytest.skip("dasymetric_mob source root not present")

    def test_limit3_build_roundtrips_and_cross_checks(self, tmp_path: Path):
        out_path = tmp_path / "od_mitma.zarr"
        audit = build_od_mitma(SOURCE_ROOT, out_path, limit=3)

        assert audit["zone_count"] > 0
        assert audit["zone_set_derivation"] == "prefix_union_footprint_sjoin"
        assert audit["zone_count_shp_total"] == 2850
        assert audit["zone_prefix_count"] == 442
        assert audit["reference_zone_counts"] == REFERENCE_ZONE_COUNTS
        assert audit["limit"] == 3
        assert audit["n_dates_processed"] == 3
        assert audit["n_dates_expected_full"] == 451
        assert audit["n_dates_expected"] == 451
        assert audit["dates_missing"] == []
        assert 0.0 < audit["mass_in_zone_set_fraction"] <= 1.0
        assert 0.0 <= audit["diag_mass_fraction"] < 1.0
        assert audit["population_path"].endswith("poblacion.csv")
        assert audit["zone_population_available"] > 0
        assert audit["strata_summation_assertion"] == "passed"

        store = xr.open_zarr(out_path)
        zone_ids = [str(z) for z in store.origin.values]
        index = {z: i for i, z in enumerate(zone_ids)}
        assert len(zone_ids) == audit["zone_count"]
        for ts in store.date.values:
            date = str(ts)[:10]
            raw = pd.read_csv(
                SOURCE_ROOT / "data" / "mitma_v1" / f"{date}.daily_mobility.v1.recurrent.csv.gz",
                sep="\t",
                usecols=["origin", "destination", "trips"],
                dtype={"origin": str, "destination": str, "trips": np.float64},
            )
            sub = raw[raw["origin"].isin(index) & raw["destination"].isin(index)]
            grouped = sub.groupby(["origin", "destination"])["trips"].sum()
            expected = np.zeros((len(zone_ids), len(zone_ids)), dtype=np.float32)
            for (origin, destination), trips in grouped.items():
                expected[index[origin], index[destination]] = np.float32(trips)
            np.testing.assert_allclose(
                store.mobility.sel(date=date).values, expected, rtol=1e-6, atol=1e-4
            )


class TestHelpers:
    def test_expected_dates_clips_window_and_audits_missing(self):
        available = [EXPECTED_FIRST_DATE, "2020-02-16"]
        missing = expected_dates(available, start=EXPECTED_FIRST_DATE, end="2020-02-16")
        assert missing == ["2020-02-15"]

        full_missing = expected_dates([EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE])
        assert len(full_missing) == 449
        assert full_missing[0] == "2020-02-15"
        assert full_missing[-1] == "2021-05-08"

    def test_find_daily_files_and_layout(self, tmp_path: Path):
        mitma_root = tmp_path / "data" / "mitma_v1"
        mitma_root.mkdir(parents=True)
        _write_daily(mitma_root / "2020-02-15.daily_mobility.v1.recurrent.csv.gz", "2020-02-15", [])
        _write_daily(mitma_root / "2020-02-14.daily_mobility.v1.recurrent.csv.gz", "2020-02-14", [])
        (mitma_root / "readme.txt").write_text("not a daily file")

        files = find_daily_files(tmp_path / "data" / "mitma_v1")
        assert [date for date, _ in files] == ["2020-02-14", "2020-02-15"]

        (tmp_path / "data" / "mitma_v2" / "mitma_v2" / "zonificacion").mkdir(parents=True, exist_ok=True)
        layout = resolve_source_layout(tmp_path)
        assert layout["mitma_root"] == mitma_root
        assert layout["shp_path"] == mitma_root / "zonificacion-distritos" / "distritos_mitma.shp"
        assert (
            layout["poblacion_path"]
            == tmp_path / "data" / "mitma_v2" / "mitma_v2" / "zonificacion" / "poblacion.csv"
        )

    def test_load_zone_populations_aligns_and_nans(self, tmp_path: Path):
        pop_file = tmp_path / "poblacion.csv"
        pop_file.write_text("distrito|municipio|provincia|poblacion\n01001|01001|01|2925.0\n08019|08019|08|100.0\n")
        populations = load_zone_populations(pop_file, ["08019", "99999", "01001"])
        assert populations.dtype == np.float32
        assert populations.tolist()[0] == pytest.approx(100.0)
        assert np.isnan(populations.tolist()[1])
        assert populations.tolist()[2] == pytest.approx(2925.0)
