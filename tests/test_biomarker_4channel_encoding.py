import numpy as np
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor


def _create_dataset(values, dims=("date", "region_id")):
    """Helper to create xarray dataset."""
    da = xr.DataArray(
        values,
        dims=dims,
        coords={"date": range(values.shape[0]), "region_id": [0]},
    )
    return xr.Dataset({"edar_biomarker": da})


def test_region_without_biomarker_data():
    """Verify regions without biomarker data get zero encoding."""
    preprocessor = BiomarkerPreprocessor(age_max=10)

    values = np.array([[np.nan], [np.nan], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    # No biomarker data: value=0, mask=0, age=max (normalized to 1.0)
    # Shape (T, N, 3) -> (3, 1, 3)
    expected = np.array(
        [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    np.testing.assert_array_almost_equal(encoded, expected)
