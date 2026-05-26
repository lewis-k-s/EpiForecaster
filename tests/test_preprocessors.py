import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from data.clinical_series_preprocessor import (
    ClinicalSeriesPreprocessor,
    ClinicalSeriesPreprocessorConfig,
)


class TestClinicalSeriesPreprocessor:
    """Tests for ClinicalSeriesPreprocessor.

    Note: Clinical data is already log1p(per-100k) transformed from preprocessing pipeline.
    The preprocessor only normalizes age to [0,1] and stacks channels.
    """

    @pytest.fixture
    def mock_dataset(self):
        # Create a dummy xarray Dataset with already log1p(per-100k) transformed values
        times = pd.date_range("2020-01-01", periods=10)
        regions = [0, 1]

        # Values are already log1p(per-100k) transformed
        # e.g., typical range might be [0, 10] after log1p transform
        values = np.random.rand(10, 2).astype(np.float32) * 5.0
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

        # Population is no longer used for transforms (data is already transformed)
        population = xr.DataArray(
            np.array([1000, 2000]), dims="region_id", coords={"region_id": regions}
        )

        return ds, population

    def test_preprocess_shapes(self, mock_dataset):
        ds, pop = mock_dataset
        config = ClinicalSeriesPreprocessorConfig()
        processor = ClinicalSeriesPreprocessor(config, var_name="hosp")

        output = processor.preprocess_dataset(ds, population=pop)

        # Output should be (T, N, 3)
        assert output.shape == (10, 2, 3)
        # Channels: value, mask, age
        # Output is float16, convert to float32 for comparison
        output_float = output.float()
        assert torch.allclose(
            output_float[..., 1], torch.from_numpy(ds.hosp_mask.values).float()
        )

    def test_values_passthrough(self, mock_dataset):
        """Verify values pass through unchanged (already log1p per-100k transformed)."""
        ds, pop = mock_dataset
        config = ClinicalSeriesPreprocessorConfig()
        processor = ClinicalSeriesPreprocessor(config, var_name="hosp")

        output = processor.preprocess_dataset(ds, population=pop)

        # Values should pass through unchanged (already log1p transformed)
        val_raw = ds.hosp.values[0, 0]
        # float16 has ~3-4 decimal digits precision, use rtol=1e-2
        assert np.isclose(output[0, 0, 0].item(), val_raw, rtol=1e-2)

    def test_missing_variable_error(self, mock_dataset):
        ds, pop = mock_dataset
        config = ClinicalSeriesPreprocessorConfig()
        processor = ClinicalSeriesPreprocessor(config, var_name="missing_var")

        with pytest.raises(ValueError, match="Variable 'missing_var' not found"):
            processor.preprocess_dataset(ds, population=pop)
