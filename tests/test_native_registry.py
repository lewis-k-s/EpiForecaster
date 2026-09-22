"""Tests for native source registries (slice REG-1, plan section 12.2).

Fixture-driven classes cover sentinel filtering (the canonical structural
conjunction of :func:`data.native_registry.abs_sentinel_mask`, ruling R11),
sex/week aggregation, zero-observed-stays-observed, unmapped statuses in both
directions, population provenance, and string-ID preservation (leading zeros
never int-ified). A skipif-guarded integration class builds all three
registries from the real input files when present and asserts the audited
reconciliation numbers, including the exact-311 sentinel-set equivalence
between the canonical conjunction and the former OR filters.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.native_registry import (
    ABS_POPULATION_NOTE,
    POP_SOURCE_COMARCA_MEMBER_SUM,
    POP_SOURCE_MUNICIPAL_AUDITED,
    SENTINEL_LABEL,
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_MUNICIPALITY,
    STATUS_OBSERVED,
    STATUS_UNMAPPED,
    STATUS_UNOBSERVED,
    STUDY_WINDOW_START,
    SourceRecord,
    SourceRegistry,
    abs_sentinel_mask,
    build_abs_registry,
    build_comarca_registry,
    build_municipal_registry,
)

pytestmark = pytest.mark.epiforecaster

FIXTURES = Path(__file__).parent / "fixtures" / "native_support"
ABS_CSV = FIXTURES / "abs_hospitalizations.csv"
DEATHS_CSV = FIXTURES / "comarca_deaths.csv"
MPIS_CSV = FIXTURES / "mpis_mini.csv"
FL_CSV = FIXTURES / "fl_population_mini.csv"
REF_CSV = FIXTURES / "muni_reference_mini.csv"

REAL_ABS = Path("data/files/COVID-19__Persones_hospitalitzades.csv")
REAL_DEATHS = Path(
    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
)
REAL_MPIS = Path("data/files/mpiscatalunya.csv")
REAL_FL = Path("data/files/fl_population_por_municipis.csv")
REAL_REF = Path("data/files/municipality_id_reference.csv")
REAL_ABS_SHP = Path("data/files/geo/cartografia/ABS.shp")

REAL_FILES_PRESENT = all(
    path.exists() for path in (REAL_ABS, REAL_DEATHS, REAL_MPIS, REAL_FL, REAL_REF)
)


def _fixture_abs_registry(geometry_ids=None):
    return build_abs_registry(ABS_CSV, geometry_ids=geometry_ids)


def _fixture_comarca_registry():
    return build_comarca_registry(DEATHS_CSV, MPIS_CSV, FL_CSV, REF_CSV)


def _fixture_municipal_registry():
    return build_municipal_registry(FL_CSV, REF_CSV)


class TestSourceRegistry:
    def test_add_get_roundtrip_and_missing(self):
        registry = SourceRegistry()
        assert registry.get(SOURCE_ABS, "088") is None
        assert registry.source_status(SOURCE_ABS, "088") is None

        record = SourceRecord(SOURCE_ABS, "088", STATUS_OBSERVED, None, None, None)
        registry.add(record)

        assert registry.get(SOURCE_ABS, "088") is record
        assert registry.source_status(SOURCE_ABS, "088") == STATUS_OBSERVED
        assert len(registry) == 1

    def test_add_replaces_same_key(self):
        registry = SourceRegistry()
        registry.add(
            SourceRecord(SOURCE_ABS, "088", STATUS_UNOBSERVED, None, None, None)
        )
        registry.add(SourceRecord(SOURCE_ABS, "088", STATUS_OBSERVED, None, None, None))

        assert len(registry) == 1
        assert registry.get(SOURCE_ABS, "088").status == STATUS_OBSERVED

    def test_extend_iter_and_sorted_records_for(self):
        registry = SourceRegistry()
        registry.extend(
            [
                SourceRecord(SOURCE_ABS, "183", STATUS_OBSERVED, None, None, None),
                SourceRecord(SOURCE_ABS, "088", STATUS_OBSERVED, None, None, None),
                SourceRecord(SOURCE_COMARCA, "02", STATUS_OBSERVED, None, None, None),
            ]
        )

        assert [r.source_id for r in registry.records_for(SOURCE_ABS)] == [
            "088",
            "183",
        ]
        assert [r.source_id for r in registry.records_for(SOURCE_COMARCA)] == ["02"]
        assert {r.source_id for r in registry} == {"088", "183", "02"}
        assert len(registry) == 3

    def test_counts_by_status(self):
        registry = SourceRegistry()
        registry.extend(
            [
                SourceRecord(SOURCE_ABS, "088", STATUS_OBSERVED, None, None, None),
                SourceRecord(SOURCE_ABS, "243", STATUS_UNOBSERVED, None, None, None),
                SourceRecord(SOURCE_ABS, "999", STATUS_UNMAPPED, None, None, None),
                SourceRecord(SOURCE_ABS, "998", STATUS_UNMAPPED, None, None, None),
            ]
        )

        assert registry.counts_by_status(SOURCE_ABS) == {
            "observed": 1,
            "unobserved": 1,
            "unmapped": 2,
        }
        assert registry.counts_by_status(SOURCE_COMARCA) == {}


ABS_CSV_COLUMNS = [
    "setmana_epidemiologica",
    "any",
    "data_inici",
    "data_final",
    "codi_regio",
    "nom_regio",
    "codi_ambit",
    "nom_ambit",
    "codi_abs",
    "nom_abs",
    "sexe",
    "grup_edat",
    "index_socioeconomic",
    "casos",
    "poblacio",
]


def _abs_row(**overrides: object) -> dict[str, object]:
    """A real-fields ABS CSV row; overrides produce sentinel-like variants."""
    row: dict[str, object] = {
        "setmana_epidemiologica": 202018,
        "any": 2020,
        "data_inici": "27/04/2020",
        "data_final": "03/05/2020",
        "codi_regio": "08",
        "nom_regio": "Catalunya Central",
        "codi_ambit": "11",
        "nom_ambit": "Bages",
        "codi_abs": "088",
        "nom_abs": "Cardona",
        "sexe": "Home",
        "grup_edat": "65 a 69",
        "index_socioeconomic": 97,
        "casos": 2,
        "poblacio": 210,
    }
    row.update(overrides)
    return row


def _sentinel_row(**overrides: object) -> dict[str, object]:
    """A Catalonia-total row: every verified conjunction field matches."""
    row = {
        "codi_regio": "00",
        "nom_regio": SENTINEL_LABEL,
        "codi_ambit": "00",
        "nom_ambit": SENTINEL_LABEL,
        "codi_abs": "000",
        "nom_abs": SENTINEL_LABEL,
        "index_socioeconomic": -1,
        "casos": 73,
        "poblacio": 0,
    }
    row.update(overrides)
    return _abs_row(**row)


class TestAbsSentinelMask:
    def test_full_conjunction_flagged_in_both_stratum_shapes(self):
        # Real file: 297 sentinel rows carry "No disponible" strata, 14 (the
        # Home / 50 a 54 stratum) carry real stratum values; both shapes are
        # Catalonia-total sentinels because sexe/grup_edat are not conjuncts.
        nd_strata = pd.DataFrame(
            [_sentinel_row(sexe=SENTINEL_LABEL, grup_edat=SENTINEL_LABEL)]
        )
        real_strata = pd.DataFrame([_sentinel_row()])
        assert bool(abs_sentinel_mask(nd_strata).iloc[0]) is True
        assert bool(abs_sentinel_mask(real_strata).iloc[0]) is True

    def test_isolated_codi_abs_000_with_real_fields_is_retained(self):
        frame = pd.DataFrame([_abs_row(codi_abs="000")])
        assert not abs_sentinel_mask(frame).any()

    def test_isolated_index_socioeconomic_minus_one_is_retained(self):
        frame = pd.DataFrame([_abs_row(index_socioeconomic=-1)])
        assert not abs_sentinel_mask(frame).any()

    def test_isolated_poblacio_zero_is_retained(self):
        frame = pd.DataFrame([_abs_row(poblacio=0)])
        assert not abs_sentinel_mask(frame).any()

    def test_every_proper_subset_of_the_conjunction_is_retained(self):
        variants = [
            _sentinel_row(codi_regio="08"),
            _sentinel_row(nom_regio="Girona"),
            _sentinel_row(nom_ambit="Selva"),
            _sentinel_row(nom_abs="Amposta"),
            _sentinel_row(index_socioeconomic=97),
            _sentinel_row(poblacio=12),
        ]
        assert not abs_sentinel_mask(pd.DataFrame(variants)).any()

    def test_string_and_numeric_dtypes_agree(self):
        as_str = pd.DataFrame([_sentinel_row(), _abs_row()], dtype=str)
        as_native = pd.DataFrame([_sentinel_row(), _abs_row()])
        assert abs_sentinel_mask(as_str).equals(abs_sentinel_mask(as_native))
        assert abs_sentinel_mask(as_str).tolist() == [True, False]

    def test_missing_numeric_field_is_retained(self):
        row = _sentinel_row()
        row["poblacio"] = None
        assert not abs_sentinel_mask(pd.DataFrame([row])).any()


class TestAbsRegistry:
    def test_sentinel_rows_dropped_and_excluded(self):
        registry, report = _fixture_abs_registry()

        assert report.total_rows == 8
        assert report.sentinel_rows_dropped == 1
        assert report.real_codes == 3
        assert registry.get(SOURCE_ABS, "000") is None

    def test_index_socioeconomic_minus_one_rows_are_kept(self):
        _registry, report = _fixture_abs_registry()

        # index_socioeconomic == -1 is a conjunction member, but alone it
        # never marks a sentinel: the fixture's real ABS 088 row carries -1
        # with genuine admissions and must survive the filter.
        assert report.sentinel_rows_dropped == 1
        assert report.abs_week_observations_in_window == 4
        assert report.casos_in_window == 8

    def test_window_statuses(self):
        registry, report = _fixture_abs_registry()

        assert registry.source_status(SOURCE_ABS, "088") == STATUS_OBSERVED
        assert registry.source_status(SOURCE_ABS, "183") == STATUS_OBSERVED
        assert registry.source_status(SOURCE_ABS, "243") == STATUS_UNOBSERVED
        assert report.observed_codes == 2
        assert report.unobserved_codes == 1

    def test_dates_span_all_observations(self):
        registry, _report = _fixture_abs_registry()

        record = registry.get(SOURCE_ABS, "088")
        assert record.first_date == date(2020, 4, 27)
        assert record.last_date == date(2021, 5, 3)

        out_of_window = registry.get(SOURCE_ABS, "243")
        assert out_of_window.first_date == date(2021, 5, 10)
        assert out_of_window.last_date == date(2021, 5, 17)

    def test_zero_casos_row_keeps_source_observed(self):
        # ABS 088 has a reported casos == 0 row on 04/05/2020; presence of the
        # reported row is the criterion, so 088 stays observed.
        registry, _report = _fixture_abs_registry()

        assert registry.source_status(SOURCE_ABS, "088") == STATUS_OBSERVED

    def test_population_is_none_with_note(self):
        registry, report = _fixture_abs_registry()

        assert all(r.population is None for r in registry)
        assert all(r.population_source is None for r in registry)
        assert report.population_note == ABS_POPULATION_NOTE
        assert ABS_POPULATION_NOTE  # non-empty documentation

    def test_window_bounds_are_inclusive(self, tmp_path):
        csv = tmp_path / "abs_bounds.csv"
        csv.write_text(
            "setmana_epidemiologica,any,data_inici,data_final,codi_regio,"
            "nom_regio,codi_ambit,nom_ambit,codi_abs,nom_abs,sexe,grup_edat,"
            "index_socioeconomic,casos,poblacio\n"
            "202009,2020,01/03/2020,07/03/2020,08,Central,11,Bages,001,A,"
            "Home,65 a 69,97,1,100\n"
            "202118,2021,09/05/2021,15/05/2021,08,Central,11,Bages,001,A,"
            "Home,65 a 69,97,1,100\n"
            "202119,2021,10/05/2021,16/05/2021,08,Central,11,Bages,001,A,"
            "Home,65 a 69,97,1,100\n",
            encoding="utf-8",
        )
        registry, report = build_abs_registry(csv)

        assert report.abs_week_observations_in_window == 2
        record = registry.get(SOURCE_ABS, "001")
        assert record.status == STATUS_OBSERVED
        assert record.first_date == STUDY_WINDOW_START
        assert record.last_date == date(2021, 5, 10)

    def test_geometry_reconciliation_surfaces_discrepancies(self):
        registry, report = _fixture_abs_registry(geometry_ids={"001", "088", "379"})

        assert report.geometry_ids_provided is True
        assert report.geometry_only_codes == ("001", "379")
        assert report.csv_codes_missing_geometry == ("183", "243")

        for code in ("001", "379"):
            record = registry.get(SOURCE_ABS, code)
            assert record.status == STATUS_UNMAPPED
            assert record.population is None
            assert record.first_date is None and record.last_date is None

        # Observed overrides nothing: CSV codes without geometry become
        # unmapped but keep their observation dates.
        assert registry.source_status(SOURCE_ABS, "183") == STATUS_UNMAPPED
        assert registry.get(SOURCE_ABS, "183").first_date == date(2020, 3, 2)
        assert registry.source_status(SOURCE_ABS, "088") == STATUS_OBSERVED
        assert registry.counts_by_status(SOURCE_ABS) == {
            "observed": 1,
            "unmapped": 4,
        }

    def test_geometry_ids_normalized_to_three_digits(self):
        _registry, report = _fixture_abs_registry(geometry_ids={1, "88 "})

        assert report.geometry_only_codes == ("001",)
        assert report.csv_codes_missing_geometry == ("183", "243")

    def test_weekly_window_override(self):
        late_window = (date(2021, 5, 10), date(2021, 12, 31))
        registry, report = build_abs_registry(ABS_CSV, window=late_window)

        assert registry.source_status(SOURCE_ABS, "243") == STATUS_OBSERVED
        assert report.observed_codes == 1
        assert report.unobserved_codes == 2


class TestComarcaRegistry:
    def test_reconciliation_counts(self):
        _registry, report = _fixture_comarca_registry()

        assert report.deaths_rows == 6
        assert report.deaths_codes == 4
        assert report.membership_codes == 4
        assert report.shared_codes == 3
        assert report.deaths_only_codes == ("00",)
        assert report.membership_only_codes == ("43",)
        assert report.observed_codes == 2
        assert report.unobserved_codes == 1

    def test_deaths_only_code_is_unmapped_with_dates(self):
        registry = _fixture_comarca_registry()[0]

        record = registry.get(SOURCE_COMARCA, "00")
        assert record.status == STATUS_UNMAPPED
        assert record.population is None
        assert record.population_source is None
        assert record.first_date == date(2020, 3, 29)
        assert record.last_date == date(2020, 3, 29)

    def test_membership_only_code_is_unmapped_without_dates(self):
        registry = _fixture_comarca_registry()[0]

        record = registry.get(SOURCE_COMARCA, "43")
        assert record.status == STATUS_UNMAPPED
        assert record.population == pytest.approx(10.5)
        assert record.population_source == POP_SOURCE_COMARCA_MEMBER_SUM
        assert record.first_date is None
        assert record.last_date is None

    def test_shared_codes_observed_with_member_sum_population(self):
        registry = _fixture_comarca_registry()[0]

        record = registry.get(SOURCE_COMARCA, "01")
        assert record.status == STATUS_OBSERVED
        assert record.population == pytest.approx(150.75)
        assert record.population_source == POP_SOURCE_COMARCA_MEMBER_SUM
        # Two sex rows on the same date collapse to one observation date.
        assert record.first_date == record.last_date == date(2020, 3, 26)

    def test_reported_zero_deaths_stay_observed(self):
        registry = _fixture_comarca_registry()[0]

        record = registry.get(SOURCE_COMARCA, "02")
        assert record.status == STATUS_OBSERVED
        assert record.first_date == date(2020, 4, 2)
        assert record.last_date == date(2020, 4, 15)

    def test_out_of_window_only_comarca_is_unobserved(self):
        registry = _fixture_comarca_registry()[0]

        record = registry.get(SOURCE_COMARCA, "03")
        assert record.status == STATUS_UNOBSERVED
        assert record.population == pytest.approx(400.75)

    def test_death_sum_includes_all_sexes_and_window(self):
        _registry, report = _fixture_comarca_registry()

        # 8 (comarca 01, two sex rows) + 2 (comarca 02, incl. the zero row)
        # + 7 (unmapped comarca 00) = 17; comarca 03's 20/06/2021 is outside.
        assert report.deaths_in_window == 17

    def test_total_population_closes_over_members(self):
        _registry, report = _fixture_comarca_registry()

        assert report.total_population == pytest.approx(762.0)


class TestMunicipalRegistry:
    def test_all_matched_municipalities_unobserved_with_population(self):
        registry, report = _fixture_municipal_registry()

        assert len(registry) == 5
        assert registry.counts_by_status(SOURCE_MUNICIPALITY) == {
            "unobserved": 5,
        }
        assert report.matched_codes == 5
        assert report.population_only_codes == ()
        assert report.reference_only_codes == ()
        assert report.total_population == pytest.approx(762.0)
        assert all(
            r.population_source == POP_SOURCE_MUNICIPAL_AUDITED for r in registry
        )
        assert all(r.first_date is None and r.last_date is None for r in registry)

    def test_either_side_only_codes_become_unmapped(self, tmp_path):
        fl = tmp_path / "fl.csv"
        fl.write_text(
            "id,evend,c.cod_mun,c.name,d.area_m2,d.population,d.density_pop_m2\n"
            "01001,1,010019,MuniA,105.0,100.5,9.5e-06\n"
            "99999,1,999999,Phantom,10.0,1.0,1e-07\n",
            encoding="utf-8",
        )
        ref = tmp_path / "ref.csv"
        ref.write_text(
            "muni_code_6digit,muni_name,comarca_code,comarca_name,"
            "muni_code_5digit,has_edar\n"
            "010019,MuniA,01,Alt Camp,01001,no\n"
            "899999,Ghost,89,Fantasia,89999,no\n",
            encoding="utf-8",
        )
        registry, report = build_municipal_registry(fl, ref)

        assert report.matched_codes == 1
        assert report.population_only_codes == ("99999",)
        assert report.reference_only_codes == ("89999",)
        assert registry.source_status(SOURCE_MUNICIPALITY, "99999") == STATUS_UNMAPPED
        assert registry.source_status(SOURCE_MUNICIPALITY, "89999") == STATUS_UNMAPPED
        assert registry.get(SOURCE_MUNICIPALITY, "99999").population == 1.0
        assert registry.get(SOURCE_MUNICIPALITY, "89999").population is None

    def test_population_rows_and_reference_rows_counts(self):
        _registry, report = _fixture_municipal_registry()

        assert report.population_rows == 5
        assert report.reference_rows == 5


class TestIdPreservation:
    def test_abs_codes_keep_leading_zeros(self):
        registry, _report = _fixture_abs_registry()

        assert {r.source_id for r in registry.records_for(SOURCE_ABS)} == {
            "088",
            "183",
            "243",
        }
        assert registry.get(SOURCE_ABS, "88") is None
        assert all(isinstance(r.source_id, str) for r in registry)

    def test_comarca_codes_keep_two_digit_strings(self):
        registry, _report = _fixture_comarca_registry()

        for code in ("00", "01", "02", "03", "43"):
            assert isinstance(registry.get(SOURCE_COMARCA, code), SourceRecord)
        assert registry.get(SOURCE_COMARCA, "0") is None
        assert registry.get(SOURCE_COMARCA, "4") is None

    def test_municipal_codes_keep_five_digit_leading_zero(self):
        registry, _report = _fixture_municipal_registry()

        record = registry.get(SOURCE_MUNICIPALITY, "01001")
        assert record is not None
        assert record.population == pytest.approx(100.5)
        assert registry.get(SOURCE_MUNICIPALITY, "1001") is None

    def test_statuses_are_the_frozen_vocabulary(self):
        for build in (_fixture_abs_registry, _fixture_comarca_registry):
            registry = build()[0]
            assert {r.status for r in registry} <= {
                STATUS_OBSERVED,
                STATUS_UNOBSERVED,
                STATUS_UNMAPPED,
            }

    def test_source_status_protocol_shape(self):
        registry, _report = _fixture_comarca_registry()

        assert registry.source_status(SOURCE_COMARCA, "01") == STATUS_OBSERVED
        assert registry.source_status(SOURCE_COMARCA, "ZZ") is None


@pytest.mark.skipif(
    not REAL_FILES_PRESENT, reason="real native input files not present"
)
class TestRealFileIntegration:
    def test_abs_registry_matches_audited_counts(self):
        registry, report = build_abs_registry(REAL_ABS)

        assert report.total_rows == 163_237
        assert report.sentinel_rows_dropped == 311
        assert report.real_codes == 377
        assert report.observed_codes == 377
        assert report.unobserved_codes == 0
        assert registry.get(SOURCE_ABS, "000") is None
        assert all(r.population is None for r in registry)
        assert report.population_note == ABS_POPULATION_NOTE

    def test_abs_sentinel_conjunction_drops_exactly_311(self):
        frame = pd.read_csv(REAL_ABS, dtype=str)
        conjunction = abs_sentinel_mask(frame)

        assert int(conjunction.sum()) == 311
        # Same set as the former OR filters (both the registry and the
        # processor variants — verified identical on the real file).
        or_filter = (
            frame["codi_abs"].eq("000")
            | frame["codi_regio"].eq("00")
            | frame["nom_regio"].eq(SENTINEL_LABEL)
            | frame["nom_ambit"].eq(SENTINEL_LABEL)
            | frame["nom_abs"].eq(SENTINEL_LABEL)
        )
        assert frame.index[conjunction].equals(frame.index[or_filter])

        # The six real newer ABS carrying index_socioeconomic == -1 with
        # genuine admissions survive the conjunction.
        minus_one = pd.to_numeric(
            frame["index_socioeconomic"], errors="coerce"
        ).eq(-1)
        newer_abs = set(frame.loc[minus_one & ~conjunction, "codi_abs"].unique())
        assert newer_abs == {"399", "404", "405", "406", "408", "409"}

    def test_abs_geometry_reconciliation(self):
        geopandas = pytest.importorskip("geopandas")
        geometry_ids = set(
            geopandas.read_file(REAL_ABS_SHP, ignore_geometry=True)["CODABSa"]
        )
        registry, report = build_abs_registry(REAL_ABS, geometry_ids=geometry_ids)

        assert report.geometry_ids_provided is True
        assert len(geometry_ids) == 379
        # The 379-vs-377 discrepancy must surface as statuses, never raise:
        # at least one direction of the reconciliation is non-empty.
        assert report.geometry_only_codes or report.csv_codes_missing_geometry
        assert len(registry) == 377 + len(report.geometry_only_codes)
        counts = registry.counts_by_status(SOURCE_ABS)
        assert sum(counts.values()) == len(registry)
        assert all(
            registry.get(SOURCE_ABS, code).status == STATUS_UNMAPPED
            for code in report.geometry_only_codes
        )
        assert registry.get(SOURCE_ABS, "000") is None

    def test_comarca_registry_matches_audited_counts(self):
        registry, report = build_comarca_registry(
            REAL_DEATHS, REAL_MPIS, REAL_FL, REAL_REF
        )

        assert report.deaths_codes == 43
        assert report.membership_codes == 43
        assert report.shared_codes == 42
        assert report.deaths_only_codes == ("00",)
        assert report.membership_only_codes == ("43",)
        assert report.observed_codes == 42
        assert report.unobserved_codes == 0

        unclassified = registry.get(SOURCE_COMARCA, "00")
        assert unclassified.status == STATUS_UNMAPPED
        assert unclassified.population is None
        aran = registry.get(SOURCE_COMARCA, "43")
        assert aran.status == STATUS_UNMAPPED
        assert aran.first_date is None

    def test_municipal_registry_matches_audited_counts(self):
        registry, report = build_municipal_registry(REAL_FL, REAL_REF)

        assert len(registry) == 947
        assert registry.counts_by_status(SOURCE_MUNICIPALITY) == {"unobserved": 947}
        assert report.matched_codes == 947
        assert report.population_only_codes == ()
        assert report.reference_only_codes == ()
        assert report.total_population == pytest.approx(7_675_217.0, rel=1e-9)

    def test_comarca_population_closure_over_real_files(self):
        _registry, report = build_comarca_registry(
            REAL_DEATHS, REAL_MPIS, REAL_FL, REAL_REF
        )
        _municipal, municipal_report = build_municipal_registry(REAL_FL, REAL_REF)

        assert report.total_population == pytest.approx(
            municipal_report.total_population, rel=1e-9
        )
