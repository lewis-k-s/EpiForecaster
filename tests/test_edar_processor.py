"""Tests for the EDAR (wastewater) processor."""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from data.preprocess.processors.edar_processor import EDARProcessor, _TobitKalman
from data.preprocess.config import PreprocessingConfig, REGION_COORD


@pytest.fixture
def mock_config(tmp_path):
    # Create dummy files to pass config validation
    (tmp_path / "cases.csv").touch()
    (tmp_path / "mob.nc").touch()
    (tmp_path / "ww.csv").touch()
    (tmp_path / "pop.csv").write_text("id,d.population\n")
    (tmp_path / "meta.nc").touch()
    (tmp_path / "hosp.csv").touch()
    (tmp_path / "deaths.csv").touch()

    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(tmp_path / "cases.csv"),
        mobility_path=str(tmp_path / "mob.nc"),
        wastewater_file=str(tmp_path / "ww.csv"),
        population_file=str(tmp_path / "pop.csv"),
        region_metadata_file=str(tmp_path / "meta.nc"),
        hospitalizations_file=str(tmp_path / "hosp.csv"),
        deaths_file=str(tmp_path / "deaths.csv"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 31),
        output_path=str(tmp_path / "out"),
        dataset_name="test",
        forecast_horizon=1,
        sequence_length=1,
    )


def test_tobit_kalman_censor_flags(mock_config):
    """Verify Tobit-Kalman censor flag semantics (0=uncensored, 1=censored, 2=missing)."""
    # Test _TobitKalman directly
    tk = _TobitKalman(process_var=0.1, measurement_var=0.1)

    # 0: Uncensored (value > limit)
    # 1: Censored (value <= limit)
    # 2: Missing (NaN)

    values = np.array([10.0, 5.0, np.nan, 12.0])
    limits = np.array([1.0, 7.0, 1.0, 1.0])

    filtered, flags = tk.filter_series(values, limits)

    assert flags[0] == 0  # 10.0 > 1.0
    assert flags[1] == 1  # 5.0 <= 7.0
    assert flags[2] == 2  # NaN
    assert flags[3] == 0  # 12.0 > 1.0


def test_edar_daily_resampling_gaps(mock_config):
    """Verify daily resampling inserts date gaps without fabricating positive measurements."""
    proc = EDARProcessor(mock_config)

    # Create dummy flow data with gaps
    # 2022-01-01 and 2022-01-04 measured
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-04")],
            "edar_id": ["E1", "E1"],
            "variant": ["N2", "N2"],
            "total_covid_flow": [100.0, 200.0],
            "limit_flow": [10.0, 10.0],
        }
    )

    resampled = proc._resample_to_daily(df)

    # Check 2022-01-02 and 2022-01-03 exist and are NaN (min_count=1 in sum preserves NaNs)
    resampled = resampled.set_index("date")
    assert np.isnan(resampled.loc["2022-01-02", "total_covid_flow"])
    assert np.isnan(resampled.loc["2022-01-03", "total_covid_flow"])
    assert resampled.loc["2022-01-01", "total_covid_flow"] == 100.0
    assert resampled.loc["2022-01-04", "total_covid_flow"] == 200.0


def test_edar_age_channel_normalization(mock_config):
    """Verify age channel stores raw days (0-14) as uint8, clipped at max_age."""
    proc = EDARProcessor(mock_config)

    # Create a mask DataArray (run_id, date, region_id)
    dates = pd.date_range("2022-01-01", periods=20)
    regions = ["R1"]

    mask_data = np.zeros((1, 20, 1))
    # Observation at index 5 (2022-01-06)
    mask_data[0, 5, 0] = 1.0
    # Observation at index 10 (2022-01-11)
    mask_data[0, 10, 0] = 1.0

    mask = xr.DataArray(
        mask_data,
        coords={"run_id": ["real"], "date": dates, REGION_COORD: regions},
        dims=["run_id", "date", REGION_COORD],
    )

    max_age = 14
    age = proc._compute_age_channel(mask, max_age=max_age)

    # Age is now stored as uint8 (0-14 raw days), not normalized
    # Before first observation: should be 14 (max_age, no prior observation)
    assert age.isel(date=0, run_id=0, region_id=0) == 14
    assert age.isel(date=4, run_id=0, region_id=0) == 14

    # At first observation: age 0 (current day)
    assert age.isel(date=5, run_id=0, region_id=0) == 0

    # At index 6: last_seen is 5, current is 6, age is 1 (raw day)
    assert age.isel(date=6, run_id=0, region_id=0) == 1

    # Long gap: index 10 is next observation.
    # index 9: last_seen 5, current 9, age 4 (raw days)
    assert age.isel(date=9, run_id=0, region_id=0) == 4

    # At index 10: last_seen is 10, current is 10, age 0.
    assert age.isel(date=10, run_id=0, region_id=0) == 0

    # Very long gap (beyond 14 days)
    # Let's test clipping explicitly
    mask_data_short = np.zeros((1, 30, 1))
    mask_data_short[0, 0, 0] = 1.0  # First day measured
    mask_short = xr.DataArray(
        mask_data_short,
        coords={
            "run_id": ["real"],
            "date": pd.date_range("2022-01-01", periods=30),
            REGION_COORD: regions,
        },
        dims=["run_id", "date", REGION_COORD],
    )
    age_long = proc._compute_age_channel(mask_short, max_age=14)
    # Index 20: 20 days since last observation. Clipped at 14.
    assert age_long.isel(date=20, run_id=0, region_id=0) == 14
