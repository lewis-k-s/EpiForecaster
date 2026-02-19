"""Tests for the main preprocessing pipeline."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd

from data.preprocess.config import PreprocessingConfig, REGION_COORD
from data.preprocess.pipeline import OfflinePreprocessingPipeline

@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing."""
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
        end_date=datetime(2022, 1, 10),
        output_path=str(tmp_path / "out"),
        dataset_name="test_dataset",
        forecast_horizon=1,
        sequence_length=1,
    )

class TestPipelineSourceSelection:
    @patch("data.preprocess.pipeline.SyntheticProcessor")
    @patch("data.preprocess.pipeline.EDARProcessor")
    def test_synthetic_source_dispatch(self, MockEDAR, MockSynthetic, mock_config):
        """Verify synthetic data source selection and processor dispatch."""
        mock_config.synthetic_path = "synthetic_data.zarr"
        
        # Mock synthetic processor output
        mock_processed = {
            "cases": xr.DataArray([1], dims=["x"]),
            "mobility": xr.DataArray([1], dims=["x"]),
            "population": xr.DataArray([1], dims=["x"]),
            "edar_flow": xr.DataArray([1], dims=["x"]),
            "edar_censor": xr.DataArray([1], dims=["x"]),
        }
        MockSynthetic.return_value.process.return_value = mock_processed
        
        pipeline = OfflinePreprocessingPipeline(mock_config)
        
        # Mock EDAR processing from xarray
        MockEDAR.return_value.process_from_xarray.return_value = xr.Dataset()
        
        # We need to mock _load_raw_sources because it's called by run(), 
        # but we want to test its internal logic.
        processed = pipeline._load_raw_sources()
        
        assert MockSynthetic.called
        assert MockSynthetic.return_value.process.called_with("synthetic_data.zarr")
        assert "edar" in processed
        assert processed["hospitalizations"] is None
        assert processed["deaths"] is None

    @patch("data.preprocess.pipeline.CataloniaCasesProcessor")
    def test_cases_source_dispatch(self, MockCases, mock_config):
        """Verify real cases data source selection and processor dispatch."""
        # Setup mocks to avoid actual file loading
        pipeline = OfflinePreprocessingPipeline(mock_config)
        pipeline.processors["cases"] = Mock()
        pipeline.processors["mobility"] = Mock()
        pipeline.processors["edar"] = Mock()
        pipeline.processors["hospitalizations"] = Mock()
        pipeline.processors["deaths"] = Mock()
        pipeline._load_population_data = Mock()
        
        pipeline._load_raw_sources()
        
        assert pipeline.processors["cases"].process.called
        assert pipeline.processors["mobility"].process.called
        assert pipeline.processors["edar"].process.called

class TestPipelineOptionalSources:
    def test_hospitalizations_deaths_optional_behavior(self, mock_config):
        """Verify that hospitalizations/deaths can be optional if not configured."""
        # Note: Current implementation treats them as required for real data.
        # This test might fail if we don't update the code to match the plan.
        mock_config.hospitalizations_file = None
        mock_config.deaths_file = None
        
        pipeline = OfflinePreprocessingPipeline(mock_config)
        
        # Mock mandatory sources
        pipeline.processors["cases"] = Mock()
        pipeline.processors["mobility"] = Mock()
        pipeline.processors["edar"] = Mock()
        pipeline._load_population_data = Mock()
        
        # According to Phase 3 plan, it should continue and return None for these.
        # If it raises RuntimeError, the code needs adjustment.
        try:
            processed = pipeline._load_raw_sources()
            assert processed["hospitalizations"] is None
            assert processed["deaths"] is None
        except RuntimeError as e:
            pytest.fail(f"Pipeline failed with RuntimeError for optional sources: {e}")

class TestPipelineMasksAndMetadata:
    def test_compute_valid_targets_mask(self, mock_config):
        """Test valid_targets density threshold behavior."""
        pipeline = OfflinePreprocessingPipeline(mock_config)
        
        # Create a dataset with varying density
        # 10 days, 3 regions
        dates = pd.date_range("2022-01-01", periods=10)
        regions = ["R1", "R2", "R3"]
        run_ids = [0]
        
        cases_data = np.ones((1, 10, 3))
        # R1: 100% density
        # R2: 50% density (5 NaNs)
        cases_data[0, :5, 1] = np.nan
        # R3: 0% density (all NaNs)
        cases_data[0, :, 2] = np.nan
        
        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], cases_data)
            },
            coords={
                "run_id": run_ids,
                "date": dates,
                REGION_COORD: regions
            }
        )
        
        # Set threshold to 0.6
        mock_config.min_density_threshold = 0.6
        
        mask = pipeline._compute_valid_targets_mask(ds)
        
        assert mask.sel(run_id=0, region_id="R1") == 1
        assert mask.sel(run_id=0, region_id="R2") == 0
        assert mask.sel(run_id=0, region_id="R3") == 0
        
        # Boundary case: threshold == 0.5
        mock_config.min_density_threshold = 0.5
        mask = pipeline._compute_valid_targets_mask(ds)
        assert mask.sel(run_id=0, region_id="R2") == 1

    @patch("xarray.open_dataarray")
    def test_compute_edar_region_mask(self, mock_open_da, mock_config):
        """Test edar_has_source mask alignment/reindex."""
        pipeline = OfflinePreprocessingPipeline(mock_config)
        
        # Mock region metadata (emap)
        # edar_id, home (region_id)
        # E1 covers R1
        # E2 covers R2
        # R3 has no EDAR
        regions = ["R1", "R2", "R3"]
        edar_ids = ["E1", "E2"]
        
        emap_data = np.zeros((2, 3))
        emap_data[0, 0] = 1 # E1 -> R1
        emap_data[1, 1] = 1 # E2 -> R2
        
        mock_emap = xr.DataArray(
            emap_data,
            coords={"edar_id": edar_ids, "home": ["R1", "R2", "R3"]},
            dims=["edar_id", "home"],
            name="region_metadata"
        )
        mock_open_da.return_value = mock_emap
        
        mask = pipeline._compute_edar_region_mask("dummy_path", np.array(regions))
        
        assert mask.sel(region_id="R1") == 1
        assert mask.sel(region_id="R2") == 1
        assert mask.sel(region_id="R3") == 0
        assert mask.name == "edar_has_source"

class TestPipelineStorage:
    def test_saved_zarr_chunk_schema(self, mock_config, tmp_path):
        """Test saved zarr chunk schema for run_id, date, and spatial dims."""
        pipeline = OfflinePreprocessingPipeline(mock_config)
        
        # Create a dummy aligned dataset
        dates = pd.date_range("2022-01-01", periods=20)
        regions = [f"R{i}" for i in range(10)]
        run_ids = np.arange(5)
        
        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], np.random.rand(5, 20, 10)),
                "mobility": (["run_id", "date", "origin", "destination"], np.random.rand(5, 20, 10, 10)),
            },
            coords={
                "run_id": run_ids,
                "date": dates,
                REGION_COORD: regions,
                "origin": regions,
                "destination": regions,
            }
        )
        
        mock_config.run_id_chunk_size = 2
        mock_config.date_chunk_size = 10
        mock_config.mobility_chunk_size = 5
        mock_config.output_path = str(tmp_path)
        mock_config.dataset_name = "test_chunk_schema"
        
        # Mock _log_postwrite_summary to avoid opening the saved file again
        pipeline._log_postwrite_summary = Mock()
        
        saved_path = pipeline._save_aligned_dataset(ds)
        
        # Re-open and check chunks
        saved_ds = xr.open_zarr(saved_path)
        
        # Check 'cases' chunks
        # run_id: 2, date: 10, region_id: 10 (not explicitly chunked if < mobility_chunk_size?)
        # Actually _save_aligned_dataset chunks all spatial dims by mobility_chunk_size
        
        cases_chunks = saved_ds.cases.chunks
        # cases_chunks is a tuple of tuples: ((2, 2, 1), (10, 10), (5, 5))
        assert cases_chunks[0] == (2, 2, 1)
        assert cases_chunks[1] == (10, 10)
        assert cases_chunks[2] == (5, 5)
        
        mobility_chunks = saved_ds.mobility.chunks
        assert mobility_chunks[0] == (2, 2, 1) # run_id
        assert mobility_chunks[1] == (10, 10)  # date
        assert mobility_chunks[2] == (5, 5)    # origin
        assert mobility_chunks[3] == (5, 5)    # destination

        # Check smoothing metadata attrs are persisted in output dataset.
        assert saved_ds.attrs["log_transformed"] is True
        assert saved_ds.attrs["population_norm"] is True
        assert saved_ds.attrs["smoothing_clinical_method"] == "kalman_v2"
        assert saved_ds.attrs["smoothing_wastewater_method"] == "tobit_kalman_v2"
        assert saved_ds.attrs["smoothing_missing_policy"] == "predict"
        assert "preprocessing_config_yaml" in saved_ds.attrs
        config_yaml = saved_ds.attrs["preprocessing_config_yaml"]
        assert isinstance(config_yaml, str)
        assert "dataset_name: test_chunk_schema" in config_yaml
        assert "smoothing:" in config_yaml
