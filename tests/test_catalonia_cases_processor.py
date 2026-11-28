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
    """Verify gap filling stays finite/non-negative and preserves observed values."""
    proc = CataloniaCasesProcessor(mock_config)
    
    dates = pd.date_range("2022-01-01", periods=10)
    # Noisy data with some zeros
    daily_df = pd.DataFrame({
        "date": dates,
        "municipality_code": "08019",
        "cases": [10.0, 0.0, 15.0, 0.0, 20.0, 5.0, 10.0, 0.0, 5.0, 10.0]
    })
    
    smoothed = proc._fill_gaps(daily_df)
    
    assert np.all(np.isfinite(smoothed["cases"]))
    assert np.all(smoothed["cases"] >= 0)
    assert smoothed.iloc[0]["cases"] == pytest.approx(10.0)
    assert smoothed.iloc[2]["cases"] == pytest.approx(15.0)
    assert smoothed.iloc[4]["cases"] == pytest.approx(20.0)
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
        "cases_observed": [0, 0, 1, 1, 0, 1, 1, 0, 0, 0]
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


def test_cases_holt_damped_preserves_mask_age_semantics(mock_config):
    """Verify holt_damped still preserves observation-driven mask/age semantics."""
    mock_config.smoothing.clinical_method = "holt_damped"
    proc = CataloniaCasesProcessor(mock_config)

    dates = pd.date_range("2022-01-01", periods=6)
    daily_df = pd.DataFrame(
        {
            "date": dates,
            "municipality_code": "08019",
            "cases": [10.0, np.nan, np.nan, 11.0, np.nan, 12.0],
        }
    )

    smoothed = proc._fill_gaps(daily_df)
    result = proc._create_mask_and_age_channels(smoothed)

    expected_mask = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert smoothed.iloc[0]["cases"] == pytest.approx(10.0)
    assert smoothed.iloc[3]["cases"] == pytest.approx(11.0)
    assert smoothed.iloc[5]["cases"] == pytest.approx(12.0)
    assert result["cases_mask"].tolist() == expected_mask
    assert result.iloc[0]["cases_age"] == 1.0
    assert result.iloc[1]["cases_age"] == 2.0
    assert result.iloc[2]["cases_age"] == 3.0
    assert result.iloc[3]["cases_age"] == 1.0


@pytest.mark.parametrize("method", ["locf", "kalman_v2", "holt_damped"])
def test_cases_zero_counts_are_observed(mock_config, method):
    mock_config.smoothing.clinical_method = method
    proc = CataloniaCasesProcessor(mock_config)
    dates = pd.date_range("2022-01-01", periods=6)
    daily_df = pd.DataFrame(
        {
            "date": dates,
            "municipality_code": "08019",
            "cases": [0.0, 0.0, 10.0, np.nan, 12.0, 11.0],
        }
    )
    filled = proc._fill_gaps(daily_df)
    assert filled["cases"].iloc[:3].tolist() == [0.0, 0.0, 10.0]
    assert filled["cases"].iloc[4:].tolist() == [12.0, 11.0]
    assert np.isfinite(filled["cases"].iloc[3])
    result = proc._create_mask_and_age_channels(filled)
    assert result["cases_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert result["cases_age"].tolist() == [1.0, 1.0, 1.0, 2.0, 1.0, 1.0]


def test_cases_locf_carries_the_last_observation_including_zero(mock_config):
    mock_config.smoothing.clinical_method = "locf"
    proc = CataloniaCasesProcessor(mock_config)
    daily_df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=7),
            "municipality_code": "08019",
            "cases": [np.nan, 5.0, 0.0, np.nan, np.nan, 3.0, np.nan],
        }
    )
    filled = proc._fill_gaps(daily_df)
    assert np.isnan(filled["cases"].iloc[0])
    assert filled["cases"].iloc[1:].tolist() == [5.0, 0.0, 0.0, 0.0, 3.0, 3.0]
    assert filled["cases_observed"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_cases_dataset_marks_days_without_rows_inside_the_span_as_observed_zero(
    mock_config,
):
    proc = CataloniaCasesProcessor(mock_config)
    dates = pd.date_range("2022-01-01", periods=5)
    aggregated = pd.DataFrame(
        {
            "date": [dates[1], dates[3], dates[2]],
            "municipality_code": ["A", "A", "B"],
            "cases": [4.0, 6.0, 2.0],
        }
    )
    ds = proc._build_cases_dataset(aggregated, dates)
    cases = ds["cases"].sel(run_id="real", region_id="A").values
    mask = ds["cases_mask"].sel(run_id="real", region_id="A").values
    # The registry spans days 1-3: day 2 is an observed zero, days 0 and 4 lie
    # outside the span.
    assert np.isnan(cases[0])
    assert cases[1:4].tolist() == [4.0, 0.0, 6.0]
    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert ds["cases"].sel(run_id="real", region_id="B").values[1:4].tolist() == [
        0.0,
        2.0,
        0.0,
    ]


def test_cases_dataset_keeps_leading_nan(mock_config):
    mock_config.start_date = datetime(2022, 1, 1)
    proc = CataloniaCasesProcessor(mock_config)
    dates = pd.date_range("2022-01-01", periods=4)
    aggregated = pd.DataFrame(
        {
            "date": dates[2:],
            "municipality_code": "08019",
            "cases": [10.0, 12.0],
        }
    )
    ds = proc._build_cases_dataset(aggregated, dates)
    cases = ds["cases"].isel(run_id=0, region_id=0).values
    assert np.isnan(cases[:2]).all()
    assert np.isfinite(cases[2:]).all()
    assert ds["cases_mask"].isel(run_id=0, region_id=0).values.tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
