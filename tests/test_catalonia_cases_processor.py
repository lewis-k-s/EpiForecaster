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
        population_file=str(tmp_path / "pop.csv"),
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
    daily_df = pd.DataFrame(
        {
            "date": dates,
            "municipality_code": "08019",
            "cases": [10.0, 0.0, 15.0, 0.0, 20.0, 5.0, 10.0, 0.0, 5.0, 10.0],
        }
    )

    smoothed = proc._apply_kalman_smoothing(daily_df)

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
    daily_df = pd.DataFrame(
        {
            "date": dates,
            "municipality_code": "08019",
            "cases": [
                np.nan,
                np.nan,
                10.0,
                11.0,
                np.nan,
                12.0,
                13.0,
                np.nan,
                np.nan,
                np.nan,
            ],
            "missing_flag": [2, 2, 0, 0, 2, 0, 0, 2, 2, 2],
        }
    )

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

    smoothed = proc._apply_kalman_smoothing(daily_df)
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


@pytest.mark.parametrize("method", ["kalman_v2", "holt_damped"])
def test_cases_zero_missing_and_future_changes(mock_config, method):
    mock_config.smoothing.clinical_method = method
    proc = CataloniaCasesProcessor(mock_config)
    dates = pd.date_range("2022-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "municipality_code": "08019",
            "cases": [np.nan, 0.0, 10.0, np.nan, 12.0, np.nan],
        }
    )
    expected = proc._create_mask_and_age_channels(proc._apply_kalman_smoothing(frame))
    assert expected.cases.iloc[0] == 0.0
    assert expected.cases.iloc[1] == 0.0
    assert expected.cases_mask.tolist() == [0, 1, 1, 0, 1, 0]
    future = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-07", periods=20),
            "municipality_code": "08019",
            "cases": np.arange(20) * 1000.0,
        }
    )
    actual = proc._create_mask_and_age_channels(
        proc._apply_kalman_smoothing(pd.concat([frame, future], ignore_index=True))
    )
    pd.testing.assert_frame_equal(actual.iloc[:6], expected)


def test_cases_grid_keeps_explicit_zero_separate_from_absent_row(mock_config):
    dates = pd.date_range("2022-01-01", periods=4)
    aggregated = pd.DataFrame(
        {"date": dates[[1, 3]], "municipality_code": "08019", "cases": [0.0, 10.0]}
    )
    dataset = (
        CataloniaCasesProcessor(mock_config)
        ._build_cases_dataset(aggregated, dates)
        .sel(run_id="real", region_id="08019")
    )
    np.testing.assert_array_equal(dataset.cases_mask, [0, 1, 0, 1])
    np.testing.assert_array_equal(dataset.cases_age, [14, 1, 2, 1])
    np.testing.assert_array_equal(dataset.cases, [0, 0, 0, 10])
