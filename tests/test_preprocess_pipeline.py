"""Tests for the main preprocessing pipeline."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from data.preprocess.config import PreprocessingConfig, REGION_COORD
from data.epi_dataset import open_processed_dataset
from data.preprocess.pipeline import OfflinePreprocessingPipeline
from data.preprocess.zarr_layout import RunWriteResult, write_dataset


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
    def test_rolling_target_selection_uses_observed_mask_and_end_date(
        self, mock_config
    ):
        dates = pd.date_range("2020-01-01", periods=12)
        observed = np.zeros((1, 12, 3), dtype=bool)
        observed[0, 2:5, 0] = True
        observed[0, 9:12, 1] = True
        observed[0, 1, 2] = True
        observed[0, 4, 2] = True
        observed[0, 7, 2] = True
        dataset = xr.Dataset(
            {
                "cases": (("run_id", "date", REGION_COORD), np.ones((1, 12, 3))),
                "cases_mask": (("run_id", "date", REGION_COORD), observed),
            },
            coords={"run_id": ["real"], "date": dates, REGION_COORD: ["A", "B", "C"]},
        )
        mock_config.valid_target_rolling_days = 5
        mock_config.valid_target_min_observed_days = 3
        mock_config.valid_target_qualifying_end_date = "2020-01-08"

        selected = OfflinePreprocessingPipeline(
            mock_config
        )._compute_valid_targets_mask(dataset)

        assert selected.sel(run_id="real").values.tolist() == [1, 0, 0]

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

    def test_compute_valid_targets_mask_positive_basis(self, mock_config):
        """With observed zeros, density counts the days with a positive count."""
        dates = pd.date_range("2022-01-01", periods=10)
        cases_data = np.zeros((1, 10, 2), dtype=np.float32)
        cases_data[0, :6, 0] = 3.0
        cases_data[0, :2, 1] = 3.0
        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], cases_data),
                "cases_mask": (
                    ["run_id", "date", REGION_COORD],
                    np.ones((1, 10, 2), dtype=bool),
                ),
            },
            coords={"run_id": [0], "date": dates, REGION_COORD: ["R1", "R2"]},
        )
        mock_config.min_density_threshold = 0.5

        observed = OfflinePreprocessingPipeline(
            mock_config
        )._compute_valid_targets_mask(ds)
        assert observed.sel(run_id=0).values.tolist() == [1, 1]

        mock_config.valid_target_density_basis = "positive"
        positive = OfflinePreprocessingPipeline(
            mock_config
        )._compute_valid_targets_mask(ds)
        assert positive.sel(run_id=0).values.tolist() == [1, 0]

    def test_compute_valid_targets_mask_population_floor(self, mock_config):
        """A population floor selects targets without reading case counts."""
        dates = pd.date_range("2022-01-01", periods=4)
        cases_data = np.zeros((1, 4, 3), dtype=np.float32)
        cases_data[0, :, 2] = 9.0  # the active region is the smallest one
        ds = xr.Dataset(
            {
                "cases": (["run_id", "date", REGION_COORD], cases_data),
                "population": ([REGION_COORD], [3000.0, np.nan, 2999.0]),
            },
            coords={"run_id": [0], "date": dates, REGION_COORD: ["R1", "R2", "R3"]},
        )
        mock_config.valid_target_min_population = 3000.0

        mask = OfflinePreprocessingPipeline(mock_config)._compute_valid_targets_mask(ds)

        assert mask.dims == ("run_id", REGION_COORD)
        assert mask.sel(run_id=0).values.tolist() == [1, 0, 0]

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


def _save(pipeline: OfflinePreprocessingPipeline, aligned: xr.Dataset) -> Path:
    """Prepare and write an aligned dataset as the real-data path does."""
    path = Path(pipeline.config.output_path) / (
        pipeline.config.dataset_name + ".zarr.zip"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(
        pipeline._prepare_dataset(aligned),
        path,
        bitround_keepbits=pipeline.config.bitround_keepbits,
    )
    return path


class TestPipelineStorage:
    def test_saved_zarr_shard_schema(self, mock_config, tmp_path):
        """Each per-run variable is stored as one shard per run."""
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

        saved_path = _save(pipeline, ds)
        assert saved_path.name == "test_chunk_schema.zarr.zip"
        saved_ds = open_processed_dataset(saved_path)

        # The in-memory Dask chunks do not set the stored layout.
        assert saved_ds.cases.encoding["shards"] == (1, 20, 10)
        assert saved_ds.cases.encoding["chunks"] == (1, 20, 10)
        assert saved_ds.cases.dtype == np.float32  # float16 is upcast
        assert saved_ds.mobility.encoding["shards"] == (1, 20, 10, 10)
        assert saved_ds.mobility.encoding["chunks"] == (1, 20, 10, 10)
        assert saved_ds.attrs["zarr_layout_version"] == "synthetic-zarr-v1"

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

        saved_path = _save(pipeline, dataset)

        saved = open_processed_dataset(saved_path)
        try:
            assert saved.attrs["continuous_storage_dtype"] == "float32"
            assert saved["cases"].dtype == np.float32
            assert saved["latent_I_true"].dtype == np.float32
            assert bool((saved["cases"] > 0).all())
            assert bool((saved["latent_I_true"] > 0).all())
        finally:
            saved.close()

    def test_postwrite_summary_uses_writer_counts(self, capsys):
        """The summary adds the per-run counts and does not reopen the store."""
        results = [
            RunWriteResult(0, "a", {"mobility": 36, "cases": 12}),
            RunWriteResult(1, "b", {"mobility": 27, "cases": 12}),
        ]

        OfflinePreprocessingPipeline._log_postwrite_summary(results)

        out = capsys.readouterr().out
        assert "Verified mobility data: 63 non-null values" in out
        assert "Verified cases data: 24 non-null values" in out

    def test_postwrite_summary_rejects_all_nan_mobility(self):
        with pytest.raises(ValueError, match="all NaN"):
            OfflinePreprocessingPipeline._log_postwrite_summary(
                [RunWriteResult(0, "a", {"mobility": 0})]
            )


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
