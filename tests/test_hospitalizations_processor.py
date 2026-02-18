"""Tests for the hospitalizations processor."""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from data.preprocess.processors.hospitalizations_processor import HospitalizationsProcessor
from data.preprocess.config import PreprocessingConfig

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
        validation_options={
            "process_var": 0.1,
            "measurement_var": 0.2
        }
    )

def test_weekly_to_daily_preserves_totals(mock_config):
    """Verify weekly-to-daily interpolation preserves weekly totals per municipality."""
    proc = HospitalizationsProcessor(mock_config)
    
    # Create dummy weekly data
    # 2 weeks, 1 municipality
    weekly_data = pd.DataFrame({
        "week_start": [pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-10")],
        "municipality_code": ["08019", "08019"],
        "hospitalizations": [7.0, 14.0]
    })
    
    daily_df = proc._resample_weekly_to_daily(weekly_data)
    
    # Check total hospitalizations
    assert daily_df["hospitalizations"].sum() == 21.0
    
    # Check per week totals
    week1 = daily_df[(daily_df["date"] >= "2022-01-03") & (daily_df["date"] < "2022-01-10")]
    assert len(week1) == 7
    assert np.allclose(week1["hospitalizations"].sum(), 7.0)
    
    week2 = daily_df[(daily_df["date"] >= "2022-01-10") & (daily_df["date"] < "2022-01-17")]
    assert len(week2) == 7
    assert np.allclose(week2["hospitalizations"].sum(), 14.0)

def test_age_channel_behavior(mock_config):
    """Verify age channel resets on observation days and caps at 14."""
    proc = HospitalizationsProcessor(mock_config)
    
    # Create daily df with some missing observations
    dates = pd.date_range("2022-01-01", periods=30)
    daily_df = pd.DataFrame({
        "date": dates,
        "municipality_code": "08019",
        "hospitalizations": 1.0,
        "age": [1, 2, 3, 4, 5, 6, 7] * 4 + [1, 2], # Initial resampling age
        "missing_flag": 0
    })
    
    # Mock mask creation: only age=1 are actual observations
    # Let's make one observation missing (missing_flag=2)
    daily_df.loc[7, "missing_flag"] = 2 # Second week start is missing
    
    result = proc._create_mask_and_age_channels(daily_df)
    
    # First week start (index 2: 2022-01-03 is first Monday if 01-01 is Sat)
    # Wait, my dummy data above doesn't align with week starts strictly.
    # In _create_mask_and_age_channels:
    # mask = is_week_start & is_not_missing
    # is_week_start = (daily_df["age"] == 1)
    
    # 2022-01-01 (index 0): age=1, flag=0 -> mask=1, age=1
    assert result.iloc[0]["hospitalizations_mask"] == 1.0
    assert result.iloc[0]["hospitalizations_age"] == 1.0
    
    # 2022-01-02 (index 1): age=2, flag=0 -> mask=0, age=2
    assert result.iloc[1]["hospitalizations_mask"] == 0.0
    assert result.iloc[1]["hospitalizations_age"] == 2.0
    
    # 2022-01-08 (index 7): age=1, flag=2 -> mask=0, age=8 (7 days since last + 1)
    assert result.iloc[7]["hospitalizations_mask"] == 0.0
    assert result.iloc[7]["hospitalizations_age"] == 8.0
    
    # 2022-01-15 (index 14): age=1, flag=0 -> mask=1, age=1
    assert result.iloc[14]["hospitalizations_mask"] == 1.0
    assert result.iloc[14]["hospitalizations_age"] == 1.0

    # Test cap at 14
    # Make a long gap
    daily_df.loc[14:, "missing_flag"] = 2
    daily_df.loc[14:, "age"] = 1 # Force week start but mark as missing
    result_long_gap = proc._create_mask_and_age_channels(daily_df)
    assert result_long_gap.iloc[29]["hospitalizations_age"] == 14.0

def test_kalman_fallback_path(mock_config):
    """Test Kalman fallback path when parameter fitting fails."""
    proc = HospitalizationsProcessor(mock_config)
    
    # Create daily df with all zeros (fitting will fail)
    dates = pd.date_range("2022-01-01", periods=10)
    daily_df = pd.DataFrame({
        "date": dates,
        "municipality_code": "08019",
        "hospitalizations": 0.0
    })
    
    # This should currently FAIL with ValueError because fallback is not implemented in the loop
    # We want it to SUCCEED using fallbacks.
    try:
        smoothed = proc._apply_kalman_smoothing(daily_df)
        assert not smoothed.empty
        assert "hospitalizations" in smoothed
    except ValueError as e:
        pytest.fail(f"Kalman smoothing failed instead of using fallback: {e}")
