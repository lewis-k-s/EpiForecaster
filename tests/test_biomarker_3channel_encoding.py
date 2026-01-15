import numpy as np
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor, BiomarkerScalerParams


def _create_dataset(values, dims=("date", "region_id")):
    """Helper to create xarray dataset."""
    da = xr.DataArray(
        values,
        dims=dims,
        coords={"date": range(values.shape[0]), "region_id": [0]},
    )
    return xr.Dataset({"edar_biomarker": da})


def test_locf_forward_fill():
    """Verify last-observation-carried-forward works correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)
    # Shape: (5, 1, 3)

    expected_value = np.log1p(np.array([1.0, 1.0, 1.0, 4.0, 4.0]))
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


def test_mask_channel_correctness():
    """Verify mask is 1.0 only on observed days."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_mask = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 1], expected_mask)


def test_age_channel_computation():
    """Verify age increments daily and caps at age_max."""
    preprocessor = BiomarkerPreprocessor(age_max=5)

    values = np.array(
        [
            [1.0],
            [np.nan],
            [np.nan],
            [np.nan],
            [4.0],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
        ]
    )
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_age = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0]) / 5.0
    np.testing.assert_array_almost_equal(encoded[:, 0, 2], expected_age)


def test_age_normalization():
    """Verify age is normalized to [0, 1]."""
    preprocessor = BiomarkerPreprocessor(age_max=10)

    values = np.array([[1.0], [np.nan], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_age = np.array([0.0, 0.1, 0.2])
    np.testing.assert_array_almost_equal(encoded[:, 0, 2], expected_age)


def test_edge_case_never_observed():
    """Verify biomarker never observed: value=0, mask=0, age=age_max."""
    preprocessor = BiomarkerPreprocessor(age_max=10)

    values = np.array([[np.nan], [np.nan], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    np.testing.assert_array_equal(encoded[:, 0, 0], 0.0)
    np.testing.assert_array_equal(encoded[:, 0, 1], 0.0)
    np.testing.assert_array_equal(encoded[:, 0, 2], 1.0)


def test_log_transform_and_scaling():
    """Verify log1p, robust scaling, and clipping are applied correctly."""
    preprocessor = BiomarkerPreprocessor()

    scaler_params = BiomarkerScalerParams(center=2.0, scale=1.0, is_fitted=True)
    preprocessor.set_scaler_params(scaler_params)

    values = np.array([[np.e - 1], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    # log1p(e-1) = 1.0. (1.0 - 2.0) / 1.0 = -1.0
    expected_value = np.array([-1.0, -1.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


def test_age_capping():
    """Verify age caps at age_max and normalizes to 1.0."""
    preprocessor = BiomarkerPreprocessor(age_max=5)

    values = np.array([[1.0]] + [[np.nan]] * 10)
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    assert encoded[5, 0, 2] == 1.0
    assert encoded[10, 0, 2] == 1.0


def test_zeros_excluded_from_mask():
    """Verify zero values are not marked as measurements (below-detection-limit handling)."""
    preprocessor = BiomarkerPreprocessor()

    # Values with zeros (below-LD), NaN (missing), and positive (valid measurements)
    values = np.array([[5.0], [0.0], [0.0], [np.nan], [3.0], [np.nan]])
    ds = _create_dataset(values)

    encoded = preprocessor.preprocess_dataset(ds)

    # Mask should be 0 for zeros (below-LD)
    # Mask should be 0 for NaN (missing)
    # Mask should be 1 for positive values (valid measurements)
    expected_mask = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 1], expected_mask)

    # Age should increase through zero periods (treating zeros as non-measurements)
    # Age pattern: 0 (measured), 1 (zero), 2 (zero), 3 (NaN), 0 (measured), 1 (NaN)
    # But LOCF fills zeros with previous value, so age only increments at true gaps
    # With zeros marked as non-measurements, age should increment at zeros too
    expected_age = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0]) / preprocessor.age_max
    np.testing.assert_array_almost_equal(encoded[:, 0, 2], expected_age)
