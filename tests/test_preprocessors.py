import pytest
import torch
import numpy as np
import xarray as xr
from unittest.mock import MagicMock
from data.clinical_series_preprocessor import (
    ClinicalSeriesPreprocessor,
    ClinicalSeriesPreprocessorConfig,
)
from data.cases_preprocessor import CasesPreprocessor, CasesPreprocessorConfig


class TestClinicalSeriesPreprocessor:
    """Tests for ClinicalSeriesPreprocessor."""

    @pytest.fixture
    def mock_dataset(self):
        # Create a dummy xarray Dataset
        times = pd.date_range("2020-01-01", periods=10)
        regions = [0, 1]

        # (Time, Region)
        values = np.random.rand(10, 2).astype(np.float32)
        mask = (values > 0.5).astype(np.float32)
        age = np.zeros_like(values)  # Simplified

        ds = xr.Dataset(
            {
                "hosp": (("date", "region_id"), values),
                "hosp_mask": (("date", "region_id"), mask),
                "hosp_age": (("date", "region_id"), age),
            },
            coords={"date": times, "region_id": regions},
        )

        population = xr.DataArray(
            np.array([1000, 2000]), dims="region_id", coords={"region_id": regions}
        )

        return ds, population

    def test_preprocess_shapes(self, mock_dataset):
        ds, pop = mock_dataset
        config = ClinicalSeriesPreprocessorConfig(per_100k=False, log_transform=False)
        processor = ClinicalSeriesPreprocessor(config, var_name="hosp")

        output = processor.preprocess_dataset(ds, population=pop)

        # Output should be (T, N, 3)
        assert output.shape == (10, 2, 3)
        # Channels: value, mask, age
        assert torch.allclose(
            output[..., 1], torch.from_numpy(ds.hosp_mask.values).float()
        )

    def test_transforms(self, mock_dataset):
        ds, pop = mock_dataset
        # Enable transforms
        config = ClinicalSeriesPreprocessorConfig(per_100k=True, log_transform=True)
        processor = ClinicalSeriesPreprocessor(config, var_name="hosp")

        output = processor.preprocess_dataset(ds, population=pop)

        # Check logic manually for one point
        val_raw = ds.hosp.values[0, 0]
        pop_val = pop.values[0]
        expected = np.log1p(val_raw * 100000 / pop_val)

        assert np.isclose(output[0, 0, 0].item(), expected, atol=1e-5)

    def test_missing_variable_error(self, mock_dataset):
        ds, pop = mock_dataset
        config = ClinicalSeriesPreprocessorConfig()
        processor = ClinicalSeriesPreprocessor(config, var_name="missing_var")

        with pytest.raises(ValueError, match="Variable 'missing_var' not found"):
            processor.preprocess_dataset(ds, population=pop)


class TestCasesPreprocessor:
    """Tests for CasesPreprocessor (legacy but used for internal logic)."""

    @pytest.fixture
    def mock_cases_dataset(self):
        times = pd.date_range("2020-01-01", periods=20)
        regions = [0, 1]
        values = np.random.rand(20, 2).astype(np.float32) * 100

        ds = xr.Dataset(
            {
                "cases": (("date", "region_id"), values),
                "population": (("region_id"), np.array([10000, 20000])),
            },
            coords={"date": times, "region_id": regions},
        )

        return ds

    def test_preprocess_dataset(self, mock_cases_dataset):
        config = CasesPreprocessorConfig(
            history_length=7, per_100k=True, log_scale=True
        )
        processor = CasesPreprocessor(config)

        processed, mean, std = processor.preprocess_dataset(mock_cases_dataset)

        # Check shapes
        # processed: (T, N, 3)
        # mean/std: (T, N, 1)
        T, N = 20, 2
        assert processed.shape == (T, N, 3)
        assert mean.shape == (T, N, 1)
        assert std.shape == (T, N, 1)

        # Check channels
        # Channel 1 is mask (should be all 1s since we generated finite data)
        assert torch.all(processed[..., 1] == 1.0)

    def test_make_normalized_window(self, mock_cases_dataset):
        config = CasesPreprocessorConfig(history_length=5)
        processor = CasesPreprocessor(config)
        processor.preprocess_dataset(mock_cases_dataset)

        norm_window, mean, std = processor.make_normalized_window(
            range_start=0, history_length=5, forecast_horizon=3
        )

        # Window size = L + H = 5 + 3 = 8
        assert norm_window.shape == (8, 2, 3)
        assert mean.shape == (2, 1)
        assert std.shape == (2, 1)


import pandas as pd
