"""Tests for the Catalonia cases processor."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from data.preprocess.processors.catalonia_cases_processor import CataloniaCasesProcessor
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
    )

def test_cases_smoothing_non_negative_finite(mock_config):
    """Verify that smoothing does not emit inf/-inf and preserves non-negativity."""
    proc = CataloniaCasesProcessor(mock_config)
    
    dates = pd.date_range("2022-01-01", periods=10)
    # Noisy data with some zeros
    daily_df = pd.DataFrame({
        "date": dates,
        "municipality_code": "08019",
        "cases": [10.0, 0.0, 15.0, 0.0, 20.0, 5.0, 10.0, 0.0, 5.0, 10.0]
    })
    
    smoothed = proc._apply_kalman_smoothing(daily_df)
    
    assert np.all(np.isfinite(smoothed["cases"]))
    assert np.all(smoothed["cases"] >= 0)
    # Check that we have a range of values (not all the same or zero)
    assert smoothed["cases"].std() > 0

def test_cases_mask_age_semantics(mock_config):
    """Verify mask/age semantics for missing days and leading missing windows."""
    proc = CataloniaCasesProcessor(mock_config)
    
    dates = pd.date_range("2022-01-01", periods=10)
    daily_df = pd.DataFrame({
        "date": dates,
        "municipality_code": "08019",
        "cases": [np.nan, np.nan, 10.0, 11.0, np.nan, 12.0, 13.0, np.nan, np.nan, np.nan],
        "missing_flag": [2, 2, 0, 0, 2, 0, 0, 2, 2, 2]
    })
    
    result = proc._create_mask_and_age_channels(daily_df)
    
    # Leading missing (indices 0, 1) -> age = 14, mask = 0
    assert result.iloc[0]["cases_mask"] == 0.0
    assert result.iloc[0]["cases_age"] == 14.0
    assert result.iloc[1]["cases_mask"] == 0.0
    assert result.iloc[1]["cases_age"] == 14.0
    
    # First observation (index 2) -> age = 1, mask = 1
    assert result.iloc[2]["cases_mask"] == 1.0
    assert result.iloc[2]["cases_age"] == 1.0
    
    # Missing day in middle (index 4) -> age = 2 (1 day since index 3), mask = 0
    assert result.iloc[4]["cases_mask"] == 0.0
    assert result.iloc[4]["cases_age"] == 2.0
    
    # Observation after gap (index 5) -> age = 1, mask = 1
    assert result.iloc[5]["cases_mask"] == 1.0
    assert result.iloc[5]["cases_age"] == 1.0
    
    # Trailing missing (index 9) -> 3 days since index 6 (13.0)
    # 7: age 2, 8: age 3, 9: age 4
    assert result.iloc[9]["cases_mask"] == 0.0
    assert result.iloc[9]["cases_age"] == 4.0
