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

    @patch("data.preprocess.pipeline.SyntheticProcessor")
    @patch("data.preprocess.pipeline.EDARProcessor")
    def test_synthetic_source_dispatch_preserves_temporal_covariates(
        self,
        MockEDAR,
        MockSynthetic,
        mock_config,
    ):
        """Synthetic source loading should pass temporal covariates through."""
        mock_config.synthetic_path = "synthetic_data.zarr"
        temporal_covariates = xr.DataArray(
            np.ones((3, 3), dtype=np.float16),
            dims=["date", "covariate"],
            coords={
                "date": pd.date_range("2022-01-01", periods=3, freq="D"),
                "covariate": ["dow_sin", "dow_cos", "is_holiday"],
            },
        )
        mock_processed = {
            "cases": xr.DataArray([1], dims=["x"]),
            "mobility": xr.DataArray([1], dims=["x"]),
            "population": xr.DataArray([1], dims=["x"]),
            "edar_flow": xr.DataArray([1], dims=["x"]),
            "edar_censor": xr.DataArray([1], dims=["x"]),
            "temporal_covariates": temporal_covariates,
        }
        MockSynthetic.return_value.process.return_value = mock_processed

        pipeline = OfflinePreprocessingPipeline(mock_config)
        MockEDAR.return_value.process_from_xarray.return_value = xr.Dataset()
        processed = pipeline._load_raw_sources()

        assert "temporal_covariates" in processed
        assert processed["temporal_covariates"].equals(temporal_covariates)

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
            {"cases": (["run_id", "date", REGION_COORD], cases_data)},
            coords={"run_id": run_ids, "date": dates, REGION_COORD: regions},
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

    def test_compute_valid_targets_mask_uses_cases_mask(self, mock_config):
        """Imputed/non-null case values should not count as observed density."""
        pipeline = OfflinePreprocessingPipeline(mock_config)

        dates = pd.date_range("2022-01-01", periods=10)
        regions = ["R1", "R2"]
        cases_data = np.ones((1, 10, 2), dtype=np.float32)
        cases_mask = np.ones((1, 10, 2), dtype=bool)
        cases_mask[0, 3:, 1] = False

        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], cases_data),
                "cases_mask": (["run_id", "date", REGION_COORD], cases_mask),
            },
            coords={"run_id": [0], "date": dates, REGION_COORD: regions},
        )

        mock_config.min_density_threshold = 0.5
        mask = pipeline._compute_valid_targets_mask(ds)

        assert bool(mask.sel(run_id=0, region_id="R1"))
        assert not bool(mask.sel(run_id=0, region_id="R2"))

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
        emap_data[0, 0] = 1  # E1 -> R1
        emap_data[1, 1] = 1  # E2 -> R2

        mock_emap = xr.DataArray(
            emap_data,
            coords={"edar_id": edar_ids, "home": ["R1", "R2", "R3"]},
            dims=["edar_id", "home"],
            name="region_metadata",
        )
        mock_open_da.return_value = mock_emap

        mask = pipeline._compute_edar_region_mask("dummy_path", np.array(regions))

        assert mask.sel(region_id="R1") == 1
        assert mask.sel(region_id="R2") == 1
        assert mask.sel(region_id="R3") == 0
        assert mask.name == "edar_has_source"


class TestPipelineStorage:
    def test_saved_netcdf_chunk_schema(self, mock_config, tmp_path):
        """Test saved NetCDF chunk schema for run_id, date, and spatial dims."""
        pipeline = OfflinePreprocessingPipeline(mock_config)

        # Create a dummy aligned dataset
        dates = pd.date_range("2022-01-01", periods=20)
        regions = [f"R{i}" for i in range(10)]
        run_ids = np.arange(5)

        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], np.random.rand(5, 20, 10)),
                "mobility": (
                    ["run_id", "date", "origin", "destination"],
                    np.random.rand(5, 20, 10, 10),
                ),
            },
            coords={
                "run_id": run_ids,
                "date": dates,
                REGION_COORD: regions,
                "origin": regions,
                "destination": regions,
            },
        )

        mock_config.run_id_chunk_size = 2
        mock_config.date_chunk_size = 10
        mock_config.mobility_chunk_size = 5
        mock_config.output_path = str(tmp_path)
        mock_config.dataset_name = "test_chunk_schema"
        mock_config.source_provenance = {
            "generator_source_commit": "abc123",
            "raw_corpus_full_tree_sha256": "deadbeef",
        }

        # Mock _log_postwrite_summary to avoid opening the saved file again
        pipeline._log_postwrite_summary = Mock()

        saved_path = pipeline._save_aligned_dataset(ds)

        # Re-open and check on-disk chunksizes from encoding
        saved_ds = xr.open_dataset(saved_path, engine="netcdf4")

        # Check 'cases' chunksizes (on-disk HDF5 chunk layout)
        cases_chunks = saved_ds.cases.encoding["chunksizes"]
        assert cases_chunks[0] == 2  # run_id
        assert cases_chunks[1] == 10  # date
        assert cases_chunks[2] == 5  # region_id (<= mobility_chunk_size)
        assert saved_ds.cases.dtype == np.float32  # upcast from float16 for NetCDF

        mobility_chunks = saved_ds.mobility.encoding["chunksizes"]
        assert mobility_chunks[0] == 2  # run_id
        assert mobility_chunks[1] == 10  # date
        assert mobility_chunks[2] == 5  # origin
        assert mobility_chunks[3] == 5  # destination

        # Check smoothing metadata attrs are persisted in output dataset.
        assert saved_ds.attrs["log_transformed"] == 1
        assert saved_ds.attrs["population_norm"] == 1
        assert saved_ds.attrs["smoothing_clinical_method"] == "kalman_v2"
        assert saved_ds.attrs["smoothing_wastewater_method"] == "tobit_kalman_v2"
        assert saved_ds.attrs["smoothing_missing_policy"] == "predict"
        assert "preprocessing_config_yaml" in saved_ds.attrs
        config_yaml = saved_ds.attrs["preprocessing_config_yaml"]
        assert isinstance(config_yaml, str)
        assert "dataset_name: test_chunk_schema" in config_yaml
        assert "smoothing:" in config_yaml
        source_provenance_yaml = saved_ds.attrs["source_provenance_yaml"]
        assert "generator_source_commit: abc123" in source_provenance_yaml
        assert "raw_corpus_full_tree_sha256: deadbeef" in source_provenance_yaml

    def test_float32_storage_preserves_small_positive_supervision(
        self, mock_config, tmp_path
    ):
        mock_config.continuous_storage_dtype = "float32"
        mock_config.output_path = str(tmp_path)
        mock_config.dataset_name = "float32_supervision"

        dataset = xr.Dataset(
            {
                "cases": (
                    ("run_id", "date", "region_id"),
                    np.array([[[1e-12], [2e-12]]], dtype=np.float64),
                ),
                "cases_mask": (
                    ("run_id", "date", "region_id"),
                    np.ones((1, 2, 1), dtype=bool),
                ),
                "latent_I_true": (
                    ("run_id", "date", "region_id"),
                    np.array([[[1e-12], [2e-12]]], dtype=np.float32),
                ),
                "population": (
                    ("run_id", "region_id"),
                    np.array([[1000]], dtype=np.int32),
                ),
            },
            coords={
                "run_id": ["synthetic"],
                "date": np.arange(2),
                "region_id": ["001"],
            },
        )
        pipeline = OfflinePreprocessingPipeline(mock_config)
        pipeline._log_postwrite_summary = Mock()

        saved_path = pipeline._save_aligned_dataset(dataset)

        saved = xr.open_dataset(saved_path, engine="netcdf4")
        try:
            assert saved.attrs["continuous_storage_dtype"] == "float32"
            assert saved["cases"].dtype == np.float32
            assert saved["latent_I_true"].dtype == np.float32
            assert bool((saved["cases"] > 0).all())
            assert bool((saved["latent_I_true"] > 0).all())
        finally:
            saved.close()

    def test_postwrite_summary_streams_saved_mobility(
        self, mock_config, tmp_path, capsys
    ):
        """The read-back counts must not load the full mobility tensor."""
        regions = ["r1", "r2", "r3"]
        mobility = np.ones((2, 4, 3, 3), dtype=np.float32)
        mobility[1, 2] = np.nan
        path = tmp_path / "saved.nc"
        xr.Dataset(
            {
                "mobility": (("run_id", "date", "origin", "destination"), mobility),
                "cases": (("run_id", "date", REGION_COORD), np.ones((2, 4, 3))),
            },
            coords={
                "run_id": ["a", "b"],
                "date": pd.date_range("2022-01-01", periods=4),
                "origin": regions,
                "destination": regions,
                REGION_COORD: regions,
            },
        ).to_netcdf(path, engine="netcdf4")
        opened: list[xr.Dataset] = []
        real_open = xr.open_dataset

        def recording_open(*args, **kwargs):
            ds = real_open(*args, **kwargs)
            opened.append(ds)
            return ds

        with patch("data.preprocess.pipeline.xr.open_dataset", recording_open):
            OfflinePreprocessingPipeline(mock_config)._log_postwrite_summary(path)

        assert opened[0]["mobility"].chunks is not None
        assert "Verified mobility data: 63 non-null values" in capsys.readouterr().out


class TestPreprocessingConfigValidation:
    def test_invalid_continuous_storage_dtype_raises(self, mock_config):
        mock_config.continuous_storage_dtype = "float64"

        with pytest.raises(
            ValueError,
            match="continuous_storage_dtype must be one of",
        ):
            mock_config._validate_processing_options()

    def test_missing_holiday_calendar_file_raises(self, tmp_path):
        """Holiday calendar must exist when holidays are enabled."""
        (tmp_path / "cases.csv").touch()
        (tmp_path / "mob.nc").touch()
        (tmp_path / "ww.csv").touch()
        (tmp_path / "pop.csv").write_text("id,d.population\n")
        (tmp_path / "meta.nc").touch()
        missing_holiday_file = tmp_path / "does_not_exist.csv"

        with pytest.raises(ValueError, match="Holiday calendar file does not exist"):
            PreprocessingConfig(
                data_dir=str(tmp_path),
                cases_file=str(tmp_path / "cases.csv"),
                mobility_path=str(tmp_path / "mob.nc"),
                wastewater_file=str(tmp_path / "ww.csv"),
                population_file=str(tmp_path / "pop.csv"),
                region_metadata_file=str(tmp_path / "meta.nc"),
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 1, 10),
                output_path=str(tmp_path / "out"),
                dataset_name="test_dataset",
                forecast_horizon=1,
                sequence_length=1,
                temporal_covariates={
                    "include_day_of_week": True,
                    "include_holidays": True,
                    "holiday_calendar_file": str(missing_holiday_file),
                },
            )
