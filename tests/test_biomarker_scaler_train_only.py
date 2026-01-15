import numpy as np
import pytest
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor


def test_scaler_fitted_on_train_only():
    """Verify scalers are computed using only train nodes."""
    dataset = xr.Dataset(
        {
            "edar_biomarker": xr.DataArray(
                np.array([[1.0, 2.0], [np.nan, 10.0], [5.0, np.nan]]),
                dims=["time", "region_id"],
            )
        }
    )

    preprocessor = BiomarkerPreprocessor()

    train_nodes = [0]
    preprocessor.fit_scaler(dataset, train_nodes)

    expected_values = np.log1p(np.array([1.0, 5.0]))
    expected_center = np.median(expected_values)
    assert np.isclose(preprocessor.scaler_params.center, expected_center)


def test_scaler_reused_for_val_test():
    """Verify same scaler params are used for val/test."""
    dataset = xr.Dataset(
        {
            "edar_biomarker": xr.DataArray(
                np.array([[1.0, 2.0], [np.nan, 10.0]]),
                dims=["time", "region_id"],
            )
        }
    )

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    train_params = preprocessor.scaler_params

    preprocessor.set_scaler_params(train_params)

    assert preprocessor.scaler_params.center == train_params.center
    assert preprocessor.scaler_params.scale == train_params.scale


def test_near_constant_biomarker():
    """Verify scale=1.0 when IQR≈0."""
    dataset = xr.Dataset(
        {
            "edar_biomarker": xr.DataArray(
                np.array([[1.0], [1.0], [1.0]]),
                dims=["time", "region_id"],
            )
        }
    )

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    assert preprocessor.scaler_params.scale == 1.0


def test_scaler_no_finite_values_error():
    """Verify error raised when no finite values in train nodes."""
    dataset = xr.Dataset(
        {
            "edar_biomarker": xr.DataArray(
                np.array([[np.nan], [np.nan]]),
                dims=["time", "region_id"],
            )
        }
    )

    preprocessor = BiomarkerPreprocessor()

    with pytest.raises(ValueError, match="No finite biomarker values"):
        preprocessor.fit_scaler(dataset, [0])


def test_scaler_log_transform():
    """Verify scaler is computed on log1p transformed values, excluding zeros."""
    dataset = xr.Dataset(
        {
            "edar_biomarker": xr.DataArray(
                np.array([[1.0, 0.0]]),
                dims=["time", "region_id"],
            )
        }
    )

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0, 1])

    # Zeros (below detection limit) are excluded from scaler fitting
    expected_values = np.log1p(np.array([1.0]))  # 0.0 excluded
    expected_center = np.median(expected_values)
    # Single value -> IQR = 0, so scale = 1.0
    expected_scale = 1.0

    assert np.isclose(preprocessor.scaler_params.center, expected_center)
    assert np.isclose(preprocessor.scaler_params.scale, expected_scale)
