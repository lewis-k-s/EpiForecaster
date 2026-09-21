"""Tests for the native ABS admissions processor (slice OBS-1).

Unit tests run on a hand-computed synthetic fixture CSV (2 real ABS x 3 epi
weeks plus sentinel rows) exercising weekly placement, structural-triple
sentinel filtering, sex x age-group aggregation, stratum population sums and
the deviation warning, zero-vs-missing semantics, age-channel behavior,
window cropping, and the zarr v3 roundtrip. The integration test
cross-checks the real input files and is skipped when they are absent.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.abs_admissions_processor import (
    ABS_FILE,
    ABSAdmissionsProcessor,
    to_obs_store,
)
from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

pytestmark = pytest.mark.epiforecaster

CSV_COLUMNS = [
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

# Three Monday-start epi weeks with their Sunday ends
WEEK_STARTS = ("07/06/2021", "14/06/2021", "21/06/2021")
WEEK_ENDS = ("13/06/2021", "20/06/2021", "27/06/2021")

FULL_WINDOW = (datetime(2021, 6, 5), datetime(2021, 6, 27))


def _row(
    week: int,
    abs_id: str,
    casos: int,
    poblacio: int,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "setmana_epidemiologica": 23 + week,
        "any": 2021,
        "data_inici": WEEK_STARTS[week],
        "data_final": WEEK_ENDS[week],
        "codi_regio": "08",
        "nom_regio": "Catalunya Central",
        "codi_ambit": "01",
        "nom_ambit": "Ambit 1",
        "codi_abs": abs_id,
        "nom_abs": f"ABS {abs_id}",
        "sexe": "Home",
        "grup_edat": "50 a 54",
        "index_socioeconomic": 5,
        "casos": casos,
        "poblacio": poblacio,
    }
    row.update(overrides)
    return row


def _fixture_rows() -> list[dict[str, object]]:
    return [
        # ABS 001 week 0: two strata -> value 3, stratum_pop_sum 259 + 301
        _row(0, "001", 1, 259, sexe="Home", grup_edat="50 a 54"),
        _row(0, "001", 2, 301, sexe="Dona", grup_edat="60 a 64"),
        # ABS 001 week 1: reported zero-case stratum -> value 0, mask True
        _row(1, "001", 0, 560),
        # Sentinel rows, each caught by exactly one structural-triple
        # condition (§10.3(4)): the OR filter must drop all three
        _row(0, "000", 9, 5_000),  # codi_abs == "000"
        _row(0, "099", 7, 4_000, codi_regio="00"),  # codi_regio == "00"
        _row(0, "098", 6, 3_000, sexe="No disponible"),  # "No disponible"
        # Real newer ABS 002 carries index_socioeconomic == -1 with genuine
        # admissions -> must be retained, never treated as a sentinel
        _row(1, "002", 4, 100, index_socioeconomic=-1, grup_edat="30 a 34"),
        _row(2, "002", 1, 100, sexe="Dona", grup_edat="70 a 74"),
    ]


def _write_csv(data_dir: Path, rows: list[dict[str, object]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ABS_FILE
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)
    return path


def _make_config(
    tmp_path: Path, start: datetime, end: datetime
) -> PreprocessingConfig:
    # PreprocessingConfig.__post_init__ validates that data_dir, cases_file
    # and mobility_path exist and creates output_path.
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cases_file = tmp_path / "cases.csv"
    cases_file.touch()
    mobility_path = tmp_path / "mobility"
    mobility_path.mkdir(exist_ok=True)
    return PreprocessingConfig(
        data_dir=str(data_dir),
        cases_file=str(cases_file),
        mobility_path=str(mobility_path),
        wastewater_file="",
        population_file="",
        region_metadata_file="",
        start_date=start,
        end_date=end,
        output_path=str(tmp_path / "out"),
        dataset_name="native_abs_test",
    )


def _process(
    tmp_path: Path,
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
) -> xr.Dataset:
    data_dir = tmp_path / "data"
    _write_csv(data_dir, rows)
    config = _make_config(tmp_path, start, end)
    return ABSAdmissionsProcessor(config).process(data_dir)


@pytest.fixture
def processed(tmp_path: Path) -> xr.Dataset:
    return _process(tmp_path, _fixture_rows(), *FULL_WINDOW)


class TestABSAdmissionsProcessor:
    def test_weekly_placement_and_coords(self, processed: xr.Dataset) -> None:
        dates = pd.to_datetime(processed["date"].values)
        assert list(dates) == [pd.Timestamp("2021-06-07"), pd.Timestamp("2021-06-14"), pd.Timestamp("2021-06-21")]
        # week_end is the aligned Sunday coordinate
        week_ends = pd.to_datetime(processed["week_end"].values)
        np.testing.assert_array_equal((week_ends - dates).days, [6, 6, 6])
        # String IDs keep their leading zeros and are sorted
        assert list(processed["abs_id"].values) == ["001", "002"]

    def test_sex_age_aggregation(self, processed: xr.Dataset) -> None:
        # Week 0 of ABS 001: two strata (1 + 2 admissions)
        assert processed["value"].sel(date="2021-06-07", abs_id="001").item() == 3.0
        assert processed["value"].sel(date="2021-06-21", abs_id="002").item() == 1.0

    def test_mask_only_on_reported_weeks(self, processed: xr.Dataset) -> None:
        np.testing.assert_array_equal(
            processed["mask"].sel(abs_id="001").values, [True, True, False]
        )
        np.testing.assert_array_equal(
            processed["mask"].sel(abs_id="002").values, [False, True, True]
        )

    def test_zero_reported_is_observed_missing_is_not(self, processed: xr.Dataset) -> None:
        # ABS 001 week 1 reports a zero-case stratum: value 0, mask True
        zero = processed.sel(date="2021-06-14", abs_id="001")
        assert zero["value"].item() == 0.0
        assert bool(zero["mask"].item()) is True
        # ABS 002 week 0 has no rows at all: value 0, mask False
        missing = processed.sel(date="2021-06-07", abs_id="002")
        assert missing["value"].item() == 0.0
        assert bool(missing["mask"].item()) is False

    def test_stratum_pop_sum(self, processed: xr.Dataset) -> None:
        np.testing.assert_allclose(
            processed["stratum_pop_sum"].sel(abs_id="001").values, [560.0, 560.0, 0.0]
        )
        np.testing.assert_allclose(
            processed["stratum_pop_sum"].sel(abs_id="002").values, [0.0, 100.0, 100.0]
        )

    def test_age_channel(self, processed: xr.Dataset) -> None:
        # ABS 001: reported, reported (zero-case), then unreported -> 1, 1, 2
        np.testing.assert_array_equal(
            processed["age"].sel(abs_id="001").values, [1, 1, 2]
        )
        # ABS 002: leading unreported week -> 14, then reported -> 1, 1
        np.testing.assert_array_equal(
            processed["age"].sel(abs_id="002").values, [14, 1, 1]
        )

    def test_compute_age_cap_and_leading_gap(self, tmp_path: Path) -> None:
        processor = ABSAdmissionsProcessor(_make_config(tmp_path, *FULL_WINDOW))
        mask = np.zeros((16, 3), dtype=bool)
        mask[0, 0] = True  # report then a long unreported tail
        mask[1, 1] = True  # leading unreported week, then report
        age = processor._compute_age(mask)
        assert age[0, 0] == 1
        assert age[13, 0] == 14
        assert age[14, 0] == 14  # capped at 14
        assert age[15, 0] == 14
        assert age[0, 1] == 14  # weeks before first report
        assert age[1, 1] == 1
        assert age[2, 1] == 2
        assert age[15, 1] == 14  # tail also capped
        assert (age[:, 2] == 14).all()  # never reported

    def test_sentinel_filter_and_retention(self, processed: xr.Dataset) -> None:
        # All three single-condition sentinel rows are dropped...
        assert set(processed["abs_id"].values) == {"001", "002"}
        # ...while the index_socioeconomic == -1 real row is retained with
        # its genuine admissions (4 on ABS 002 week 1)
        assert processed["value"].sel(date="2021-06-14", abs_id="002").item() == 4.0

    def test_no_deviation_warning_on_consistent_weeks(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _process(tmp_path, _fixture_rows(), *FULL_WINDOW)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]

    def test_population_deviation_warning(self, tmp_path: Path) -> None:
        rows = [
            _row(0, "003", 1, 100),
            _row(1, "003", 1, 1_000),  # >1% off the window median (550)
        ]
        with pytest.warns(UserWarning, match="stratum_pop_sum"):
            _process(tmp_path, rows, *FULL_WINDOW)

    def test_window_cropping_partial_weeks(self, tmp_path: Path) -> None:
        # Thursday 2021-06-10 start skips the first week; Wednesday
        # 2021-06-23 end drops the third week (its Sunday is the 27th)
        ds = _process(
            tmp_path, _fixture_rows(), datetime(2021, 6, 10), datetime(2021, 6, 23)
        )
        dates = pd.to_datetime(ds["date"].values)
        assert list(dates) == [pd.Timestamp("2021-06-14")]
        assert ds["value"].sel(date="2021-06-14", abs_id="001").item() == 0.0
        assert ds["value"].sel(date="2021-06-14", abs_id="002").item() == 4.0

    def test_schema_dtypes_and_no_population_variable(
        self, processed: xr.Dataset
    ) -> None:
        assert processed["value"].dtype == np.float32
        assert processed["mask"].dtype == bool
        assert processed["age"].dtype == np.uint8
        assert processed["stratum_pop_sum"].dtype == np.float32
        # Ruling 1: no legacy "population" variable in obs_admissions; the
        # audited denominator arrives at native_graph assembly
        assert "population" not in processed.data_vars
        assert processed.attrs["schema"] == "obs_admissions v1 (plan §11.3)"
        assert "lower bound" in processed.attrs["population_caveat"]
        assert "index_socioeconomic" in processed.attrs["sentinel_filter"]

    def test_empty_dataset_when_only_sentinels(self, tmp_path: Path) -> None:
        rows = [
            _row(
                0,
                "000",
                3,
                0,
                codi_regio="00",
                sexe="No disponible",
                grup_edat="No disponible",
            )
        ]
        ds = _process(tmp_path, rows, *FULL_WINDOW)
        assert ds.sizes["abs_id"] == 0
        assert ds.sizes["date"] == 3  # complete window grid
        assert ds["value"].dtype == np.float32
        assert not ds["mask"].any()
        assert (ds["age"].values == 14).all()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, *FULL_WINDOW)
        with pytest.raises(FileNotFoundError):
            ABSAdmissionsProcessor(config).process(tmp_path / "data")

    def test_to_obs_store_roundtrip(self, tmp_path: Path, processed: xr.Dataset) -> None:
        store = tmp_path / "stores" / "obs_admissions.zarr"
        to_obs_store(processed, store)
        assert store.exists()

        loaded = xr.open_zarr(store)
        assert loaded["value"].dtype == np.float32
        assert loaded["mask"].dtype == bool
        assert loaded["age"].dtype == np.uint8
        assert loaded["stratum_pop_sum"].dtype == np.float32
        np.testing.assert_array_equal(
            loaded["abs_id"].values, processed["abs_id"].values
        )
        np.testing.assert_array_equal(
            pd.to_datetime(loaded["date"].values), pd.to_datetime(processed["date"].values)
        )
        np.testing.assert_array_equal(
            pd.to_datetime(loaded["week_end"].values),
            pd.to_datetime(processed["week_end"].values),
        )
        np.testing.assert_array_equal(loaded["value"].values, processed["value"].values)
        np.testing.assert_array_equal(loaded["mask"].values, processed["mask"].values)
        np.testing.assert_array_equal(loaded["age"].values, processed["age"].values)
        np.testing.assert_array_equal(
            loaded["stratum_pop_sum"].values, processed["stratum_pop_sum"].values
        )


ABS_CSV = Path("data/files/COVID-19__Persones_hospitalitzades.csv")
MUNI_CSV = Path("data/files/hospitalizations_municipality.csv")


@pytest.mark.skipif(
    not (ABS_CSV.exists() and MUNI_CSV.exists()),
    reason="real input files not present",
)
def test_integration_real_files_within_vintage_drift(tmp_path: Path) -> None:
    # data/files is read-only input territory; only output_path (tmp) is written
    config = PreprocessingConfig(
        data_dir="data/files",
        cases_file=str(ABS_CSV),
        mobility_path="data/files",
        wastewater_file="",
        population_file="",
        region_metadata_file="",
        start_date=datetime(2020, 3, 1),
        end_date=datetime(2021, 5, 9),
        output_path=str(tmp_path / "out"),
        dataset_name="native_abs_integration",
    )
    ds = ABSAdmissionsProcessor(config).process("data/files")

    # 62 whole epi weeks, Monday-start with Sunday-aligned week_end
    dates = pd.to_datetime(ds["date"].values)
    assert len(dates) == 62
    assert dates[0] == pd.Timestamp("2020-03-02")
    assert dates[-1] == pd.Timestamp("2021-05-03")
    assert (dates.weekday == 0).all()
    week_ends = pd.to_datetime(ds["week_end"].values)
    np.testing.assert_array_equal((week_ends - dates).days, np.full(62, 6))

    assert ds["value"].dtype == np.float32
    assert ds["mask"].dtype == bool
    assert ds["age"].dtype == np.uint8
    assert ds["stratum_pop_sum"].dtype == np.float32
    assert "population" not in ds.data_vars

    # Observed ABS count inside the orchestration band (real: 377)
    observed = int(ds["mask"].any(dim="abs_id").sum())
    assert 50 <= observed <= 500

    # Raw counts, non-negative where observed
    assert bool((ds["value"].values[ds["mask"].values] >= 0).all())

    # In-window total near the measured 72,666 (tolerance for revisions)
    total = float(ds["value"].sum())
    assert abs(total - 72_666) <= 2_000

    # Vintage-drift cross-check vs the projected municipal file (§10.3(4)):
    # the projection was built from an earlier ABS CSV vintage, so exact
    # equality is unattainable; native weekly sums must still track closely.
    muni = pd.read_csv(MUNI_CSV, dtype={"municipality_code": str})
    muni_week_start = pd.to_datetime(muni["data_inici"], format="%d/%m/%Y")
    muni_weekly = muni.groupby(muni_week_start)["casos_muni"].sum()
    native_weekly = ds["value"].sum(dim="abs_id").to_series()
    common = native_weekly.index.intersection(muni_weekly.index)
    diffs = (native_weekly.loc[common] - muni_weekly.loc[common]).abs()
    median_diff = float(diffs.median())
    max_diff = float(diffs.max())
    total_diff = float(diffs.sum())
    print(
        f"  vintage-drift cross-check over {len(common)} common weeks: "
        f"median |diff|={median_diff:.1f}, max={max_diff:.0f}, "
        f"total={total_diff:.0f} (native - projected municipal)"
    )
    assert median_diff < 20

    ds.attrs["vintage_drift_cross_check"] = (
        f"native weekly sums vs {MUNI_CSV} over {len(common)} common weeks: "
        f"median |diff|={median_diff:.1f}, max={max_diff:.0f}, "
        f"total={total_diff:.0f}; the projected file was built from an "
        "earlier ABS CSV vintage (plan §10.3(4))"
    )
