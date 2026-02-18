"""Tests for preprocessing edge cases and robust failure modes."""

import pytest
from unittest.mock import Mock
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr

from data.preprocess.config import PreprocessingConfig, REGION_COORD, TEMPORAL_COORD
from data.preprocess.pipeline import OfflinePreprocessingPipeline
from data.preprocess.processors.hospitalizations_processor import (
    HospitalizationsProcessor,
)
from data.preprocess.processors.catalonia_cases_processor import CataloniaCasesProcessor
from data.preprocess.processors.edar_processor import EDARProcessor


@pytest.fixture
def minimal_config(tmp_path):
    # Create dummy files to pass config validation
    (tmp_path / "cases.csv").touch()
    (tmp_path / "mob.nc").touch()
    (tmp_path / "ww.csv").touch()
    (tmp_path / "pop.csv").write_text("id,d.population\n")
    (tmp_path / "meta.json").touch()
    (tmp_path / "hosp.csv").touch()
    (tmp_path / "deaths.csv").touch()

    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(tmp_path / "cases.csv"),
        mobility_path=str(tmp_path / "mob.nc"),
        wastewater_file=str(tmp_path / "ww.csv"),
        population_file=str(tmp_path / "pop.csv"),
        region_metadata_file=str(tmp_path / "meta.json"),
        hospitalizations_file=str(tmp_path / "hosp.csv"),
        deaths_file=str(tmp_path / "deaths.csv"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        output_path=str(tmp_path / "out"),
        dataset_name="test",
        forecast_horizon=1,
        sequence_length=1,
    )


class TestPipelineRobustness:
    def test_missing_mandatory_sources_handled_gracefully(self, minimal_config):
        """Test that pipeline handles missing mandatory sources gracefully with warnings."""
        # Setup config with valid paths but the files don't exist
        # The pipeline gracefully handles processor failures with warnings

        # Here we mock the processors to simulate failure
        pipeline = OfflinePreprocessingPipeline(minimal_config)

        # Mock the processors dictionary directly
        pipeline.processors["cases"] = Mock()
        pipeline.processors["cases"].process.return_value = "cases_data"

        pipeline.processors["mobility"] = Mock()
        pipeline.processors["mobility"].process.return_value = "mobility_data"

        pipeline.processors["edar"] = Mock()
        pipeline.processors["edar"].process.return_value = "edar_data"

        pipeline.processors["hospitalizations"] = Mock()
        pipeline.processors["hospitalizations"].process.side_effect = Exception(
            "Hosp failure"
        )

        pipeline.processors["deaths"] = Mock()
        pipeline.processors["deaths"].process.return_value = "deaths_data"

        # Mock population loading
        pipeline._load_population_data = Mock(return_value="pop_data")

        # Run _load_raw_sources - should not raise, handles gracefully
        raw_data = pipeline._load_raw_sources()

        # Verify hospitalizations is None due to failure
        assert raw_data["hospitalizations"] is None
        # Verify other sources loaded successfully
        assert raw_data["cases"] == "cases_data"
        assert raw_data["mobility"] == "mobility_data"

    def test_missing_config_source_skipped(self, minimal_config):
        """Test that pipeline skips sources not configured (None)."""
        minimal_config.hospitalizations_file = None
        pipeline = OfflinePreprocessingPipeline(minimal_config)

        # Mock the other processors
        pipeline.processors["cases"] = Mock()
        pipeline.processors["mobility"] = Mock()
        pipeline.processors["edar"] = Mock()
        pipeline.processors["cases"].process.return_value = "ok"
        pipeline.processors["mobility"].process.return_value = "ok"
        pipeline.processors["edar"].process.return_value = "ok"
        pipeline._load_population_data = Mock(return_value="pop_data")

        # Should not raise - hospitalizations is skipped when not configured
        raw_data = pipeline._load_raw_sources()

        # Verify hospitalizations is None when not configured
        assert raw_data["hospitalizations"] is None
        # Verify other sources loaded
        assert raw_data["cases"] == "ok"


class TestProcessorsKalmanFallback:
    def test_hospitalization_kalman_graceful_fallback(self, minimal_config):
        """Test that hospitalization processor uses fallback when Kalman fitting fails."""
        proc = HospitalizationsProcessor(minimal_config)

        # Create dummy daily df with invalid data for fitting
        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "municipality_code": "001",
                "hospitalizations": np.zeros(
                    10
                ),  # All zeros should fail log transform fitting
            }
        )

        # Should not raise - uses fallback variance values
        result = proc._apply_kalman_smoothing(daily_df)

        # Verify result is returned and has expected columns
        assert not result.empty
        assert "hospitalizations" in result.columns
        # Result should have the expected number of rows
        assert len(result) == len(daily_df)

    def test_cases_kalman_graceful_fallback(self, minimal_config):
        """Test that cases processor uses fallback when Kalman fitting fails."""
        proc = CataloniaCasesProcessor(minimal_config)

        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {"date": dates, "municipality_code": "001", "cases": np.zeros(10)}
        )

        # Should not raise - uses fallback variance values
        result = proc._apply_kalman_smoothing(daily_df)

        # Verify result is returned and has expected columns
        assert not result.empty
        assert "cases" in result.columns
        # Result should have the expected number of rows
        assert len(result) == len(daily_df)

    def test_edar_kalman_no_fallback(self, minimal_config):
        """Test that EDAR processor handles fitting failure by marking as missing."""
        proc = EDARProcessor(minimal_config)

        dates = pd.date_range("2022-01-01", periods=10)
        daily_df = pd.DataFrame(
            {
                "date": dates,
                "edar_id": "E1",
                "variant": "N1",
                "total_covid_flow": np.zeros(10),
                "limit_flow": np.ones(10),
            }
        )

        # Should not raise, but mark as missing (flag 2)
        result = proc._apply_tobit_kalman(daily_df)

        assert not result.empty
        assert (result["censor_flag"] == 2).all(), "Should mark failed fits as missing"
        assert result["total_covid_flow"].isna().all(), (
            "Flow should be NaN for failed fits"
        )


class TestSmoothedDataValidity:
    def test_catalonia_cases_smoothed_validity(self, minimal_config):
        """Verify smoothed cases are finite and non-negative."""
        proc = CataloniaCasesProcessor(minimal_config)

        # Create noisy but valid data
        dates = pd.date_range("2022-01-01", periods=20)
        rng = np.random.default_rng(42)
        cases = rng.lognormal(mean=2, sigma=0.5, size=20)

        daily_df = pd.DataFrame(
            {"date": dates, "municipality_code": "001", "cases": cases}
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)

        assert np.all(np.isfinite(smoothed["cases"])), "Smoothed cases must be finite"
        assert np.all(smoothed["cases"] >= 0), "Smoothed cases must be non-negative"
        # Check that we actually did something (values changed)
        assert not np.allclose(smoothed["cases"], cases), (
            "Smoothing should modify values"
        )

    def test_hospitalization_smoothed_validity(self, minimal_config):
        """Verify smoothed hospitalizations are finite and non-negative."""
        proc = HospitalizationsProcessor(minimal_config)

        dates = pd.date_range("2022-01-01", periods=20)
        rng = np.random.default_rng(42)
        hosp = rng.lognormal(mean=1, sigma=0.5, size=20)

        daily_df = pd.DataFrame(
            {"date": dates, "municipality_code": "001", "hospitalizations": hosp}
        )

        smoothed = proc._apply_kalman_smoothing(daily_df)

        assert np.all(np.isfinite(smoothed["hospitalizations"])), (
            "Smoothed hosp must be finite"
        )
        assert np.all(smoothed["hospitalizations"] >= 0), (
            "Smoothed hosp must be non-negative"
        )

    def test_edar_censor_flags_aggregation(self, minimal_config):
        """Test max-severity aggregation of censor flags."""
        proc = EDARProcessor(minimal_config)

        # Mock data:
        # Region R1 has contribution from E1 (flag 0) and E2 (flag 1) -> Expect 1
        # Region R2 has contribution from E3 (flag 2) -> Expect 2

        coords = {
            "run_id": ["real"],
            "date": [np.datetime64("2022-01-01")],
            "edar_id": ["E1", "E2", "E3"],
            "variant": ["N1"],
        }

        censor_data = np.array([[[[0.0], [1.0], [2.0]]]])  # (run, date, edar, var)
        censor_xr = xr.DataArray(
            censor_data, coords=coords, dims=["run_id", "date", "edar_id", "variant"]
        )

        emap_coords = {"edar_id": ["E1", "E2", "E3"], "region_id": ["R1", "R2"]}
        # E1->R1, E2->R1, E3->R2
        emap_data = np.array(
            [
                [1.0, 0.0],  # E1
                [1.0, 0.0],  # E2
                [0.0, 1.0],  # E3
            ]
        )
        emap = xr.DataArray(
            emap_data, coords=emap_coords, dims=["edar_id", "region_id"]
        )

        result = proc._aggregate_censor_flags(censor_xr, emap)

        # Check R1: max(0, 1) = 1
        r1_flag = result.sel(region_id="R1", variant="N1").values.item()
        assert r1_flag == 1.0, f"R1 should be censored (1), got {r1_flag}"


class TestStrictnessRefactor:
    def test_deaths_not_filled_with_zero(self, minimal_config):
        """Verify that missing deaths data remains NaN and is not filled with 0.0."""
        from data.preprocess.processors.alignment_processor import AlignmentProcessor

        proc = AlignmentProcessor(minimal_config)

        # Create dummy datasets
        dates = pd.date_range(minimal_config.start_date, minimal_config.end_date)
        regions = ["R1", "R2"]

        # Cases (complete)
        cases = xr.DataArray(
            np.ones((len(dates), len(regions))),
            coords={TEMPORAL_COORD: dates, REGION_COORD: regions},
            dims=[TEMPORAL_COORD, REGION_COORD],
            name="cases",
        )

        # Deaths (missing R2)
        deaths = xr.DataArray(
            np.ones((len(dates), 1)),
            coords={TEMPORAL_COORD: dates, REGION_COORD: ["R1"]},
            dims=[TEMPORAL_COORD, REGION_COORD],
            name="deaths",
        ).to_dataset()
        deaths["deaths_mask"] = deaths["deaths"].copy()
        deaths["deaths_age"] = deaths["deaths"].copy()

        # Mobility (complete)
        mobility = xr.Dataset(
            coords={TEMPORAL_COORD: dates, "origin": regions, "destination": regions}
        )

        # EDAR (dummy with one variable)
        edar = xr.Dataset(coords={TEMPORAL_COORD: dates, REGION_COORD: regions})
        edar["edar_biomarker_N1"] = xr.DataArray(
            np.zeros((len(dates), len(regions))),
            coords={TEMPORAL_COORD: dates, REGION_COORD: regions},
            dims=[TEMPORAL_COORD, REGION_COORD],
        )
        # Add run_id to match processor expectations
        edar["edar_biomarker_N1"] = edar["edar_biomarker_N1"].expand_dims(run_id=[1])

        # Population (dummy)
        pop = xr.DataArray(
            np.ones(len(regions)),
            coords={REGION_COORD: regions},
            dims=[REGION_COORD],
            name="population",
        )

        # Run alignment
        aligned = proc.align_datasets(
            cases_data=cases,
            mobility_data=mobility,
            edar_data=edar,
            population_data=pop,
            deaths_data=deaths,
        )

        # Check R2 deaths - should be NaN, NOT 0.0
        r2_deaths = aligned["deaths"].sel(region_id="R2").values
        assert np.all(np.isnan(r2_deaths)), "Missing deaths data should be NaN, not 0.0"

    def test_edar_missing_site_is_missing_flag(self, minimal_config):
        """Verify that an EDAR site with NO data results in Missing (2) flag, not Uncensored (0)."""
        proc = EDARProcessor(minimal_config)

        # Setup: Region R1 maps to E1. E1 has NO data in the censor array (NaN).

        coords = {
            "run_id": ["real"],
            "date": [np.datetime64("2022-01-01")],
            "edar_id": ["E1"],
            "variant": ["N1"],
        }

        # Censor data is all NaN (missing site)
        censor_data = np.full((1, 1, 1, 1), np.nan)
        censor_xr = xr.DataArray(
            censor_data, coords=coords, dims=["run_id", "date", "edar_id", "variant"]
        )

        emap_coords = {"edar_id": ["E1"], "region_id": ["R1"]}
        emap_data = np.array([[1.0]])
        emap = xr.DataArray(
            emap_data, coords=emap_coords, dims=["edar_id", "region_id"]
        )

        result = proc._aggregate_censor_flags(censor_xr, emap)

        # Check R1 flag
        r1_flag = result.sel(region_id="R1", variant="N1").values.item()

        # EXPECTED FAILURE: Currently implementation fills with 0 (Uncensored)
        assert r1_flag == 2.0, (
            f"Missing site should be flagged as Missing (2.0), got {r1_flag}"
        )
