"""Tests for the hospitalizations processor."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from data.preprocess.processors.hospitalizations_processor import (
    HospitalizationsProcessor,
)
from data.preprocess.config import PreprocessingConfig


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config for testing."""
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
            "measurement_var": 0.2,
        },
    )


@pytest.fixture
def mock_hospitalizations_file(tmp_path: Path) -> Path:
    """Create a mock hospitalizations CSV file."""
    data_dir = tmp_path / "hospitalizations_data"
    data_dir.mkdir()

    # Weekly data for 2 municipalities, 4 weeks
    hosp_file = data_dir / "hospitalizations_municipality.csv"
    hosp_file.write_text(
        "setmana_epidemiologica,any,data_inici,data_final,municipality_code,municipality_name,casos_muni\n"
        "1,2022,03/01/2022,09/01/2022,08019,Barcelona,14.0\n"
        "1,2022,03/01/2022,09/01/2022,08021,Abrera,7.0\n"
        "2,2022,10/01/2022,16/01/2022,08019,Barcelona,21.0\n"
        "2,2022,10/01/2022,16/01/2022,08021,Abrera,14.0\n"
        "3,2022,17/01/2022,23/01/2022,08019,Barcelona,28.0\n"
        "3,2022,17/01/2022,23/01/2022,08021,Abrera,21.0\n"
    )

    return data_dir


class TestResampleWeeklyToDaily:
    """Tests for the _resample_weekly_to_daily method."""

    def test_resample_creates_sparse_observations(self, mock_config):
        """Test that weekly data becomes sparse daily observations."""
        proc = HospitalizationsProcessor(mock_config)

        # Create dummy weekly data: 2 weeks, 1 municipality
        weekly_data = pd.DataFrame(
            {
                "week_start": [
                    pd.Timestamp("2022-01-03"),
                    pd.Timestamp("2022-01-10"),
                ],
                "municipality_code": ["08019", "08019"],
                "hospitalizations": [7.0, 14.0],
            }
        )

        daily_df = proc._resample_weekly_to_daily(weekly_data)

        # Should have 8 days (Jan 3-10 inclusive)
        assert len(daily_df) == 8

        # Only 2 days should have values (week starts)
        assert daily_df["hospitalizations"].notna().sum() == 2

        # Check observations are on correct dates
        obs_dates = daily_df[daily_df["hospitalizations"].notna()]["date"].tolist()
        assert pd.Timestamp("2022-01-03") in obs_dates
        assert pd.Timestamp("2022-01-10") in obs_dates

        # Check values are preserved (not distributed)
        week1_val = daily_df[daily_df["date"] == "2022-01-03"]["hospitalizations"].iloc[
            0
        ]
        week2_val = daily_df[daily_df["date"] == "2022-01-10"]["hospitalizations"].iloc[
            0
        ]
        assert week1_val == 7.0
        assert week2_val == 14.0

    def test_resample_multiple_municipalities(self, mock_config):
        """Test resampling with multiple municipalities."""
        proc = HospitalizationsProcessor(mock_config)

        weekly_data = pd.DataFrame(
            {
                "week_start": [
                    pd.Timestamp("2022-01-03"),
                    pd.Timestamp("2022-01-03"),
                    pd.Timestamp("2022-01-10"),
                    pd.Timestamp("2022-01-10"),
                ],
                "municipality_code": ["08019", "08021", "08019", "08021"],
                "hospitalizations": [7.0, 5.0, 14.0, 10.0],
            }
        )

        daily_df = proc._resample_weekly_to_daily(weekly_data)

        # Should have 16 days (8 days × 2 municipalities)
        assert len(daily_df) == 16

        # Should have 4 observations (2 weeks × 2 municipalities)
        assert daily_df["hospitalizations"].notna().sum() == 4

        # Check both municipalities are present
        assert set(daily_df["municipality_code"].unique()) == {"08019", "08021"}

    def test_resample_gap_between_weeks(self, mock_config):
        """Test resampling handles gaps between weeks."""
        proc = HospitalizationsProcessor(mock_config)

        # Non-consecutive weeks
        weekly_data = pd.DataFrame(
            {
                "week_start": [
                    pd.Timestamp("2022-01-03"),
                    pd.Timestamp("2022-01-24"),  # 2 week gap
                ],
                "municipality_code": ["08019", "08019"],
                "hospitalizations": [7.0, 14.0],
            }
        )

        daily_df = proc._resample_weekly_to_daily(weekly_data)

        # Should have 22 days (Jan 3-23 and Jan 24-30)
        assert len(daily_df) == 22

        # Only 2 observations
        assert daily_df["hospitalizations"].notna().sum() == 2

        # Gap days should be NaN
        gap_days = daily_df[
            (daily_df["date"] >= "2022-01-10") & (daily_df["date"] <= "2022-01-23")
        ]
        assert gap_days["hospitalizations"].isna().all()


class TestKalmanSmoothing:
    """Tests for the _apply_kalman_smoothing method."""

    def test_kalman_interpolates_gaps(self, mock_config):
        """Test that Kalman smoothing interpolates missing values."""
        proc = HospitalizationsProcessor(mock_config)

        # Create sparse daily data with gaps
        dates = pd.date_range("2022-01-01", periods=14)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [
                    10.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,  # Week 1 gap
                    15.0,  # Week 2 start
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,  # Week 2 gap
                ],
            }
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)

        # All values should be finite after smoothing
        assert smoothed["hospitalizations"].notna().all()

        # Original observations should be tracked
        assert "hospitalizations_observed" in smoothed.columns
        assert smoothed.iloc[0]["hospitalizations_observed"] == 1.0  # Observed
        assert smoothed.iloc[1]["hospitalizations_observed"] == 0.0  # Interpolated

    def test_kalman_preserves_trend(self, mock_config):
        """Test that Kalman smoothing preserves trends across gaps."""
        mock_config.smoothing.missing_policy = "momentum"
        proc = HospitalizationsProcessor(mock_config)

        # Increasing trend with gap
        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [
                    10.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    20.0,  # Clear upward trend
                    np.nan,
                    np.nan,
                    np.nan,
                    30.0,
                ],
            }
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)

        # Smoothed values should follow the trend
        # Gap between 10 and 20 should have intermediate values
        gap_values = smoothed.iloc[1:5]["hospitalizations"].values
        assert gap_values[0] > 10.0  # Trending up
        assert gap_values[-1] < 20.0  # Approaching next observation

    def test_configurable_holt_damped_method(self, mock_config):
        """Test holt_damped smoothing and unchanged mask/age semantics."""
        mock_config.smoothing.clinical_method = "holt_damped"
        proc = HospitalizationsProcessor(mock_config)

        dates = pd.date_range("2022-01-01", periods=8)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [10.0, np.nan, np.nan, 12.0, np.nan, 14.0, 15.0, np.nan],
            }
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)
        enriched = proc._create_mask_and_age_channels(smoothed)
        assert smoothed["hospitalizations"].notna().all()
        assert "missing_flag" in smoothed.columns
        assert enriched["hospitalizations_mask"].tolist() == [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
        ]

    def test_kalman_fallback_on_empty(self, mock_config):
        """Test Kalman uses fallback when parameter fitting fails."""
        proc = HospitalizationsProcessor(mock_config)

        # All zeros (can't fit parameters)
        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": 0.0,
            }
        )

        # Should not raise an error
        smoothed = proc._apply_kalman_smoothing(daily_df)

        assert not smoothed.empty
        assert "hospitalizations" in smoothed.columns

    def test_kalman_multiple_municipalities(self, mock_config):
        """Test Kalman smoothing with multiple municipalities."""
        proc = HospitalizationsProcessor(mock_config)

        dates = pd.date_range("2022-01-01", periods=7)
        daily_df = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "municipality_code": ["08019"] * 7 + ["08021"] * 7,
                "hospitalizations": [10.0] + [np.nan] * 6 + [5.0] + [np.nan] * 6,
            }
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)

        # Should process both municipalities
        assert smoothed["municipality_code"].nunique() == 2
        assert len(smoothed) == 14

        # All values should be finite
        assert smoothed["hospitalizations"].notna().all()


class TestCreateMaskAndAgeChannels:
    """Tests for the _create_mask_and_age_channels method."""

    def test_mask_tracks_observations(self, mock_config):
        """Test that mask correctly tracks original observations."""
        proc = HospitalizationsProcessor(mock_config)

        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [1.0] * 10,
                "hospitalizations_observed": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

        result = proc._create_mask_and_age_channels(daily_df)

        # Mask should be 1.0 only on observed days
        assert result.iloc[0]["hospitalizations_mask"] == 1.0
        assert result.iloc[1]["hospitalizations_mask"] == 0.0
        assert result.iloc[5]["hospitalizations_mask"] == 1.0

    def test_age_increases_between_observations(self, mock_config):
        """Test that age increases between observations."""
        proc = HospitalizationsProcessor(mock_config)

        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [1.0] * 10,
                "hospitalizations_observed": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

        result = proc._create_mask_and_age_channels(daily_df)

        # Age should increase between observations
        assert result.iloc[0]["hospitalizations_age"] == 1.0  # Day of observation
        assert result.iloc[1]["hospitalizations_age"] == 2.0  # 1 day since obs
        assert result.iloc[2]["hospitalizations_age"] == 3.0
        assert result.iloc[5]["hospitalizations_age"] == 1.0  # Reset on new obs
        assert result.iloc[6]["hospitalizations_age"] == 2.0

    def test_age_caps_at_14(self, mock_config):
        """Test that age channel caps at 14."""
        proc = HospitalizationsProcessor(mock_config)

        dates = pd.date_range("2022-01-01", periods=20)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "08019",
                "hospitalizations": [1.0] * 20,
                "hospitalizations_observed": [1.0] + [0.0] * 19,  # Long gap
            }
        )

        result = proc._create_mask_and_age_channels(daily_df)

        # Age should cap at 14
        assert result.iloc[0]["hospitalizations_age"] == 1.0
        assert result.iloc[13]["hospitalizations_age"] == 14.0
        assert result.iloc[19]["hospitalizations_age"] == 14.0


class TestFullPipeline:
    """End-to-end tests for the HospitalizationsProcessor."""

    def test_process_weekly_data(self, mock_config, mock_hospitalizations_file):
        """Test full processing pipeline with weekly data."""
        proc = HospitalizationsProcessor(mock_config)

        result = proc.process(mock_hospitalizations_file, apply_smoothing=True)

        # Check output structure
        assert "hospitalizations" in result
        assert "hospitalizations_mask" in result
        assert "hospitalizations_age" in result

        # Should have daily resolution covering full config range
        assert result["hospitalizations"].sizes["date"] == 31  # Jan 1-31

        # Should have 2 municipalities
        assert result["hospitalizations"].sizes["region_id"] == 2

        # Values between observations should be interpolated by Kalman
        # Gap between Jan 3 and Jan 10 should be filled
        gap_period = result["hospitalizations"].sel(
            date=slice("2022-01-03", "2022-01-09")
        )
        assert gap_period.notnull().all()

        # Mask should track original weekly observations
        # 3 weeks × 2 municipalities = 6 observations
        # Note: mask is only 1.0 on actual observation days (week starts)
        assert float(result["hospitalizations_mask"].sum()) == 6.0

    def test_process_without_smoothing(self, mock_config, mock_hospitalizations_file):
        """Test processing without Kalman smoothing."""
        proc = HospitalizationsProcessor(mock_config)

        result = proc.process(mock_hospitalizations_file, apply_smoothing=False)

        # Check that mask correctly identifies observation days
        # 3 weeks × 2 municipalities = 6 observations
        assert float(result["hospitalizations_mask"].sum()) == 6.0

        # Non-observation days should be marked in mask
        assert (result["hospitalizations_mask"] == 0).any()

        # Data should exist (not all NaN) since observations are present
        # Note: pivot_table with aggfunc="sum" converts NaN to 0 for the output,
        # but the mask tells us which are real observations vs filled
        assert result["hospitalizations"].notnull().any()

    def test_process_empty_data(self, mock_config, tmp_path: Path):
        """Test processing with empty data."""
        # Create empty hospitalizations file
        data_dir = tmp_path / "empty_hosp"
        data_dir.mkdir()
        hosp_file = data_dir / "hospitalizations_municipality.csv"
        hosp_file.write_text(
            "setmana_epidemiologica,any,data_inici,data_final,municipality_code,municipality_name,casos_muni\n"
        )

        proc = HospitalizationsProcessor(mock_config)
        result = proc.process(data_dir, apply_smoothing=True)

        # Should return empty but valid dataset
        assert "hospitalizations" in result
        assert result["hospitalizations"].sizes["region_id"] == 0
