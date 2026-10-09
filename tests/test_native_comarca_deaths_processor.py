"""Tests for the native comarca deaths processor (plan §12.2 OBS-2).

Covers sex aggregation, absent-row-is-missing semantics, genuine zeros,
string comarca-id preservation, the age channel convention, window
cropping, summed-municipal population, the obs_deaths schema, and the
zarr v3 roundtrip. A real-file integration test runs when the Catalonia
registry CSVs are present under data/files/.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.comarca_deaths_processor import (
    ComarcaDeathsProcessor,
    to_obs_store,
)
from data.preprocess.processors.municipality_mapping_processor import (
    MunicipalityMappingProcessor,
)

pytestmark = pytest.mark.epiforecaster

WINDOW_START = datetime(2021, 1, 1)
WINDOW_END = datetime(2021, 1, 20)

MAPPING_ROWS = [
    ("080019", "Mun A", "06", "Anoia"),
    ("080029", "Mun B", "06", "Anoia"),
    ("080039", "Mun C", "13", "Barcelonès"),
]
POPULATION_ROWS = [("08001", 100), ("08002", 50), ("08003", 200)]

# (date DD/MM/YYYY, comarca, comarca name, sex code, sex name, count)
BASE_DEATHS_ROWS = [
    ("01/01/2021", "06", "Anoia", "0", "Home", 0),
    ("01/01/2021", "06", "Anoia", "1", "Dona", 2),
    ("01/01/2021", "13", "Barcelonès", "0", "Home", 3),
    ("02/01/2021", "06", "Anoia", "1", "Dona", 1),
    ("03/01/2021", "06", "Anoia", "1", "Dona", 0),
    ("05/01/2021", "13", "Barcelonès", "0", "Home", 5),
    ("19/01/2021", "13", "Barcelonès", "0", "Home", 2),
    ("25/01/2021", "06", "Anoia", "0", "Home", 9),  # outside the window
]


def make_config(tmp_path: Path, start: datetime, end: datetime) -> PreprocessingConfig:
    """Build a validated config with dummy required input paths."""
    files = tmp_path / "files"
    files.mkdir(parents=True, exist_ok=True)
    for name in (
        "cases.csv",
        "mobility.zarr",
        "wastewater.csv",
        "population.csv",
        "metadata.csv",
    ):
        (files / name).touch()
    return PreprocessingConfig(
        data_dir=str(files),
        cases_file=str(files / "cases.csv"),
        mobility_path=str(files / "mobility.zarr"),
        population_file=str(files / "population.csv"),
        start_date=start,
        end_date=end,
        output_path=str(tmp_path / "out"),
        dataset_name="test",
    )


def write_deaths_csv(path: Path, rows: list[tuple]) -> None:
    """Write a deaths registry CSV with the real column names."""
    lines = ["Data defunció,Codi Comarca,Comarca,Codi Sexe,Sexe,Nombre defuncions"]
    lines += [
        f"{date},{code},{name},{sex_code},{sex_name},{count}"
        for date, code, name, sex_code, sex_name, count in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def write_mapping_csv(path: Path, rows: list[tuple]) -> None:
    """Write the municipality->comarca mapping CSV (4 junk header lines)."""
    lines = [
        "junk line 1",
        "junk line 2",
        "junk line 3",
        "Codi,Nom,Codi comarca,Nom comarca,",
    ]
    lines += [
        f"{muni},{muni_name},{comarca},{comarca_name},"
        for muni, muni_name, comarca, comarca_name in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def write_population_csv(path: Path, rows: list[tuple]) -> None:
    """Write the municipal population CSV."""
    lines = ["id,evend,c.cod_mun,c.name,d.area_m2,d.population,d.density_pop_m2"]
    lines += [
        f"{muni_id},2020-12-30T23:00:00.000Z,{muni_id},Muni,1.0,{pop},1.0"
        for muni_id, pop in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def write_inputs(
    files_dir: Path,
    deaths_rows: list[tuple] | None = None,
    mapping_rows: list[tuple] | None = None,
    population_rows: list[tuple] | None = None,
) -> None:
    """Write all three input CSVs (defaults = base fixture)."""
    write_deaths_csv(
        files_dir / ComarcaDeathsProcessor.DEATHS_FILE,
        deaths_rows if deaths_rows is not None else BASE_DEATHS_ROWS,
    )
    write_mapping_csv(
        files_dir / ComarcaDeathsProcessor.MAPPING_FILE,
        mapping_rows if mapping_rows is not None else MAPPING_ROWS,
    )
    write_population_csv(
        files_dir / ComarcaDeathsProcessor.POPULATION_FILE,
        population_rows if population_rows is not None else POPULATION_ROWS,
    )


@pytest.fixture
def processed(tmp_path) -> xr.Dataset:
    """Processor output for the base fixture over 2021-01-01..2021-01-20."""
    config = make_config(tmp_path, WINDOW_START, WINDOW_END)
    write_inputs(Path(config.data_dir))
    return ComarcaDeathsProcessor(config).process(config.data_dir)


class TestParsingAndAggregation:
    def test_sexes_aggregated_per_comarca_day(self, processed):
        assert float(processed["value"].sel(date=WINDOW_START, comarca_id="06")) == 2.0
        assert float(processed["value"].sel(date=WINDOW_START, comarca_id="13")) == 3.0

    def test_auxiliary_comarca_name(self, processed):
        assert processed["comarca_name"].values.tolist() == ["Anoia", "Barcelonès"]


class TestAbsentIsMissing:
    def test_absent_row_is_missing_not_zero(self, processed):
        cell = processed.sel(date=datetime(2021, 1, 2), comarca_id="13")
        assert np.isnan(float(cell["value"]))
        assert bool(cell["mask"]) is False

    def test_absent_day_missing_for_every_comarca(self, processed):
        day = processed.sel(date=datetime(2021, 1, 4))
        assert bool(day["mask"].values.any()) is False
        assert bool(np.isnan(day["value"].values).all())

    def test_mask_equals_finite_values(self, processed):
        mask = processed["mask"].values
        value = processed["value"].values
        assert np.array_equal(mask, ~np.isnan(value))

    def test_observed_cell_count_matches_reported_rows(self, processed):
        # 6 distinct (date, comarca) rows inside the window
        assert int(processed["mask"].values.sum()) == 6


class TestGenuineZero:
    def test_reported_zero_is_observed(self, processed):
        cell = processed.sel(date=datetime(2021, 1, 3), comarca_id="06")
        assert float(cell["value"]) == 0.0
        assert bool(cell["mask"]) is True


class TestIdPreservation:
    def test_comarca_ids_are_strings_with_leading_zeros(self, processed):
        ids = processed["comarca_id"].values
        assert ids.dtype == object
        assert ids.tolist() == ["06", "13"]

    def test_date_coord_is_exact_daily_window(self, processed):
        expected = pd.date_range(start=WINDOW_START, end=WINDOW_END, freq="D")
        assert np.array_equal(processed["date"].values, expected.values)


class TestAgeChannel:
    def test_age_vectors(self, processed):
        # 06: observations at idx 0, 1, 2 -> age grows 2, 3, ... capped at 14
        assert processed["age"].sel(comarca_id="06").values.tolist() == [
            1,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            14,
            14,
            14,
            14,
        ]
        # 13: observations at idx 0, 4, 18 -> idx 17 hits the cap exactly
        assert processed["age"].sel(comarca_id="13").values.tolist() == [
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            1,
            2,
        ]

    def test_age_before_first_observation_is_capped(self, tmp_path):
        config = make_config(tmp_path, WINDOW_START, WINDOW_END)
        write_inputs(
            Path(config.data_dir),
            deaths_rows=[("05/01/2021", "07", "Bages", "0", "Home", 4)],
        )
        ds = ComarcaDeathsProcessor(config).process(config.data_dir)
        assert ds["comarca_id"].values.tolist() == ["07"]
        assert ds["age"].values[:, 0].tolist() == [
            14,
            14,
            14,
            14,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            14,
            14,
        ]


class TestWindowCropping:
    def test_out_of_window_rows_excluded(self, processed):
        assert processed.sizes["date"] == 20
        assert processed["date"].values[-1] == np.datetime64(datetime(2021, 1, 20))
        # in-window observed total only (the 25/01 count of 9 is cropped)
        assert float(np.nansum(processed["value"].values)) == 13.0

    def test_comarca_with_only_out_of_window_rows_absent(self, tmp_path):
        config = make_config(tmp_path, WINDOW_START, WINDOW_END)
        write_inputs(
            Path(config.data_dir),
            deaths_rows=BASE_DEATHS_ROWS
            + [("25/01/2021", "42", "Moianès", "0", "Home", 7)],
        )
        ds = ComarcaDeathsProcessor(config).process(config.data_dir)
        assert ds["comarca_id"].values.tolist() == ["06", "13"]


class TestPopulation:
    def test_comarca_population_is_summed_municipal(self, processed):
        population = processed["population"]
        assert population.dtype == np.float32
        assert float(population.sel(comarca_id="06")) == 150.0
        assert float(population.sel(comarca_id="13")) == 200.0

    def test_unmapped_sentinel_comarca_gets_nan_population(self, tmp_path):
        config = make_config(tmp_path, WINDOW_START, WINDOW_END)
        write_inputs(
            Path(config.data_dir),
            deaths_rows=BASE_DEATHS_ROWS
            + [("02/01/2021", "00", "No classificat", "", "", 4)],
        )
        ds = ComarcaDeathsProcessor(config).process(config.data_dir)
        assert ds["comarca_id"].values.tolist() == ["00", "06", "13"]
        cell = ds.sel(date=datetime(2021, 1, 2), comarca_id="00")
        assert float(cell["value"]) == 4.0
        assert bool(cell["mask"]) is True
        assert np.isnan(float(ds["population"].sel(comarca_id="00")))
        assert ds.attrs["unmapped_comarca_ids"] == "00"


class TestSchema:
    def test_dims(self, processed):
        for var in ("value", "mask", "age"):
            assert processed[var].dims == ("date", "comarca_id")
        assert processed["population"].dims == ("comarca_id",)

    def test_dtypes(self, processed):
        assert processed["value"].dtype == np.float32
        assert processed["mask"].dtype == bool
        assert processed["age"].dtype == np.uint8
        assert processed["population"].dtype == np.float32

    def test_obs_deaths_attrs(self, processed):
        attrs = processed.attrs
        assert attrs["schema"] == "obs_deaths"
        assert attrs["schema_version"] == 1
        assert attrs["source_type"] == "comarca"
        assert attrs["head"] == "deaths"
        assert attrs["reporting_cadence_days"] == 1
        assert "mask 0" in attrs["absent_row_policy"]
        assert attrs["transforms"].startswith("none")
        # base fixture: every file code is mapped and observed
        assert attrs["unmapped_comarca_ids"] == ""
        assert attrs["unobserved_mapped_comarca_ids"] == ""


class TestToObsStore:
    def test_zarr_v3_roundtrip(self, processed, tmp_path):
        path = to_obs_store(processed, tmp_path / "native_support" / "obs_deaths.zarr")
        assert path.exists()

        restored = xr.open_zarr(path)
        assert restored["comarca_id"].values.tolist() == ["06", "13"]
        assert restored["value"].dtype == np.float32
        assert restored["mask"].dtype == bool
        assert restored["age"].dtype == np.uint8
        assert restored["population"].dtype == np.float32
        np.testing.assert_allclose(restored["value"].values, processed["value"].values)
        np.testing.assert_array_equal(restored["mask"].values, processed["mask"].values)
        np.testing.assert_array_equal(restored["age"].values, processed["age"].values)
        np.testing.assert_allclose(
            restored["population"].values, processed["population"].values
        )
        assert restored.attrs["schema"] == "obs_deaths"


REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "files"
REAL_INPUTS = [
    REAL_DATA_DIR / ComarcaDeathsProcessor.DEATHS_FILE,
    REAL_DATA_DIR / ComarcaDeathsProcessor.MAPPING_FILE,
    REAL_DATA_DIR / ComarcaDeathsProcessor.POPULATION_FILE,
]


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_INPUTS),
    reason="real Catalonia input files not present under data/files/",
)
class TestRealFileIntegration:
    def test_real_deaths_build(self, tmp_path):
        start = datetime(2020, 3, 1)
        end = datetime(2021, 5, 9)
        config = make_config(tmp_path, start, end)
        ds = ComarcaDeathsProcessor(config).process(REAL_DATA_DIR)

        raw = pd.read_csv(
            REAL_DATA_DIR / ComarcaDeathsProcessor.DEATHS_FILE,
            dtype={"Codi Comarca": str},
        )
        raw["date"] = pd.to_datetime(raw["Data defunció"], format="%d/%m/%Y")
        in_window = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        expected_codes = set(in_window["Codi Comarca"].dropna().unique())

        assert set(ds["comarca_id"].values.tolist()) == expected_codes
        assert ds.sizes["comarca_id"] == len(expected_codes)
        assert ds["date"].values[0] == np.datetime64(start)
        assert ds["date"].values[-1] == np.datetime64(end)

        observed = ds["value"].where(ds["mask"])
        assert float(observed.min()) >= 0.0
        assert float(observed.sum()) == float(in_window["Nombre defuncions"].sum())

        mapped = set(
            MunicipalityMappingProcessor(REAL_DATA_DIR)
            .load_mapping()["comarca_code"]
            .unique()
        )
        for code, population in zip(
            ds["comarca_id"].values.tolist(), ds["population"].values.tolist()
        ):
            assert np.isnan(population) == (code not in mapped)

        path = to_obs_store(ds, tmp_path / "native_support" / "obs_deaths.zarr")
        restored = xr.open_zarr(path)
        np.testing.assert_array_equal(restored["mask"].values, ds["mask"].values)
        assert float(restored["value"].where(restored["mask"]).sum()) == float(
            observed.sum()
        )

        print(
            f"\n  real comarca deaths: {ds.sizes['comarca_id']} comarcas, "
            f"{ds.sizes['date']} dates, observed total {float(observed.sum()):.0f}"
        )
