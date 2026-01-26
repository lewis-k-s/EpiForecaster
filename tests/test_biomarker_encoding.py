import numpy as np
import pytest
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor, BiomarkerScalerParams


def _create_dataset_with_channels(
    values, mask=None, censor=None, age=None, dims=("date", "region_id")
):
    """Helper to create xarray dataset with all required channels.

    Channel layout: [value, mask, censor, age] - 4 channels per variant
    """
    T, N = values.shape
    da = xr.DataArray(
        values,
        dims=dims,
        coords={"date": range(T), "region_id": range(N)},
    )

    # Default mask: 1.0 if measured (finite and positive), 0.0 otherwise
    if mask is None:
        mask = (np.isfinite(values) & (values > 0)).astype(np.float32)

    # Default censor: 0.0 (no censoring by default)
    if censor is None:
        censor = np.zeros_like(values, dtype=np.float32)

    # Default age: compute days since last measurement
    if age is None:
        age = np.zeros_like(values)
        for i in range(N):
            last_seen = -1
            for t in range(T):
                if mask[t, i] > 0:
                    last_seen = t
                    age[t, i] = 0.0
                elif last_seen >= 0:
                    age[t, i] = min(t - last_seen, 14) / 14.0
                else:
                    age[t, i] = 1.0  # Never observed

    mask_da = xr.DataArray(
        mask,
        dims=dims,
        coords={"date": range(T), "region_id": range(N)},
    )
    censor_da = xr.DataArray(
        censor,
        dims=dims,
        coords={"date": range(T), "region_id": range(N)},
    )
    age_da = xr.DataArray(
        age,
        dims=dims,
        coords={"date": range(T), "region_id": range(N)},
    )

    return xr.Dataset(
        {
            "edar_biomarker_N1": da,
            "edar_biomarker_N1_mask": mask_da,
            "edar_biomarker_N1_censor": censor_da,
            "edar_biomarker_N1_age": age_da,
        }
    )


def test_value_channel_log_transform():
    """Verify value channel gets log1p transform without LOCF."""
    preprocessor = BiomarkerPreprocessor()

    # Values with NaN gaps - no LOCF should be applied
    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]])
    mask = np.array([[1.0], [0.0], [0.0], [1.0], [0.0]])
    censor = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    age = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)
    # Shape: (5, 1, 4) - 4 channels: [value, mask, censor, age]

    # Value channel: NaN becomes 0.0, valid values get log1p
    # No LOCF, so NaN values stay as 0.0
    expected_value = np.array([np.log1p(1.0), 0.0, 0.0, np.log1p(4.0), 0.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


def test_mask_channel_correctness():
    """Verify mask channel is passed through correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]])
    mask = np.array([[1.0], [0.0], [0.0], [1.0], [0.0]])
    censor = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    age = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_mask = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 1], expected_mask)


def test_censor_channel_passthrough():
    """Verify censor channel is passed through correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [0.0], [4.0], [np.nan]])
    mask = np.array([[1.0], [0.0], [0.0], [1.0], [0.0]])
    censor = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]])  # One censored value
    age = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_censor = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 2], expected_censor)


def test_age_channel_passthrough():
    """Verify age channel is passed through correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan]])
    # Pre-computed age: different from what online computation would produce
    age = np.array([[0.5], [0.3], [0.2]])
    mask = np.array([[1.0], [0.0], [0.0]])
    censor = np.array([[0.0], [0.0], [0.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Age channel should match the pre-computed age
    np.testing.assert_array_almost_equal(encoded[:, 0, 3], age.flatten())


def test_log_transform_and_scaling():
    """Verify log1p, robust scaling, and clipping are applied correctly."""
    preprocessor = BiomarkerPreprocessor()

    scaler_params = BiomarkerScalerParams(
        center={"edar_biomarker_N1": 2.0},
        scale={"edar_biomarker_N1": 1.0},
        is_fitted=True,
    )
    preprocessor.set_scaler_params(scaler_params)

    values = np.array([[np.e - 1], [np.nan]])
    mask = np.array([[1.0], [0.0]])
    censor = np.array([[0.0], [0.0]])
    age = np.array([[0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # log1p(e-1) = 1.0. (1.0 - 2.0) / 1.0 = -1.0
    # NaN value becomes 0.0, then (0.0 - 2.0) / 1.0 = -2.0
    expected_value = np.array([-1.0, -2.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


def test_zeros_below_detection_limit():
    """Verify zero values are handled correctly (below-LD)."""
    preprocessor = BiomarkerPreprocessor()

    # Values with zeros (below-LD), NaN (missing), and positive (valid measurements)
    values = np.array([[5.0], [0.0], [0.0], [np.nan], [3.0], [np.nan]])
    # Mask: 0 for zeros (below-LD), 0 for NaN (missing), 1 for positive (valid)
    mask = np.array([[1.0], [0.0], [0.0], [0.0], [1.0], [0.0]])
    # Censor: 1.0 for zeros (censored at LD), 0.0 for others
    censor = np.array([[0.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    # Age: increments through gaps
    age = np.array([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]]) / 14.0
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Value channel: zeros become 0.0 (log(0) would be -inf)
    expected_value = np.array([np.log1p(5.0), 0.0, 0.0, 0.0, np.log1p(3.0), 0.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)

    # Mask, censor, and age should pass through
    np.testing.assert_array_equal(encoded[:, 0, 1], mask.flatten())
    np.testing.assert_array_equal(encoded[:, 0, 2], censor.flatten())
    np.testing.assert_array_almost_equal(encoded[:, 0, 3], age.flatten())


def test_multiple_regions():
    """Verify preprocessing works correctly with multiple regions."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array(
        [
            [1.0, 2.0, np.nan],
            [np.nan, 3.0, 4.0],
            [5.0, np.nan, np.nan],
        ]
    )
    mask = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    censor = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    age = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Value channel: check each region
    expected_value = np.array(
        [
            [np.log1p(1.0), np.log1p(2.0), 0.0],
            [0.0, np.log1p(3.0), np.log1p(4.0)],
            [np.log1p(5.0), 0.0, 0.0],
        ]
    )
    np.testing.assert_array_almost_equal(encoded[:, :, 0], expected_value)
    np.testing.assert_array_equal(encoded[:, :, 1], mask)
    np.testing.assert_array_equal(encoded[:, :, 2], censor)
    np.testing.assert_array_equal(encoded[:, :, 3], age)


def test_required_channels_validation():
    """Verify that missing required channels raise ValueError."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan]])
    ds_incomplete = xr.Dataset(
        {
            "edar_biomarker_N1": xr.DataArray(
                values,
                dims=("date", "region_id"),
                coords={"date": range(2), "region_id": [0]},
            ),
            # Missing mask, censor, and age channels
        }
    )

    with pytest.raises(
        ValueError, match="Missing required channel: edar_biomarker_N1_mask"
    ):
        preprocessor.preprocess_dataset(ds_incomplete)


def test_censor_flag_validation():
    """Verify censor channel values are validated (0.0 or 1.0 flags)."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[5.0], [0.0], [3.0]])
    mask = np.array([[1.0], [0.0], [1.0]])
    # Censor flags should be 0.0 or 1.0
    censor = np.array([[0.0], [1.0], [0.0]])
    age = np.array([[0.0], [1.0], [0.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Verify censor flags are preserved correctly
    np.testing.assert_array_equal(encoded[:, 0, 2], [0.0, 1.0, 0.0])


def test_censor_alignment_with_mask():
    """Verify censored points align with mask=0 (unmeasured)."""
    preprocessor = BiomarkerPreprocessor()

    # When a point is censored, mask should be 0 (no valid measurement)
    values = np.array([[5.0], [0.0], [3.0], [0.0]])
    mask = np.array([[1.0], [0.0], [1.0], [0.0]])  # Mask 0 for censored values
    censor = np.array([[0.0], [1.0], [0.0], [1.0]])  # Censor flag for zeros
    age = np.array([[0.0], [1.0], [0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Verify mask=0 where censor=1
    for i in range(4):
        if encoded[i, 0, 2] == 1.0:  # If censored
            assert encoded[i, 0, 1] == 0.0, "Censored points should have mask=0"


def test_clip_range():
    """Verify values are clipped to the configured range."""
    preprocessor = BiomarkerPreprocessor(clip_range=(-2.0, 2.0))

    scaler_params = BiomarkerScalerParams(
        center={"edar_biomarker_N1": 0.0},
        scale={"edar_biomarker_N1": 1.0},
        is_fitted=True,
    )
    preprocessor.set_scaler_params(scaler_params)

    # Large positive and negative values after scaling
    values = np.array([[100.0], [0.0]])
    mask = np.array([[1.0], [0.0]])
    censor = np.array([[0.0], [0.0]])
    age = np.array([[0.0], [1.0]])
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # log1p(100) ≈ 4.6, should be clipped to 2.0
    # 0.0 stays at 0.0
    assert encoded[0, 0, 0] == 2.0
    assert encoded[1, 0, 0] == 0.0


def test_region_without_biomarker_data():
    """Verify regions without biomarker data get zero encoding."""
    preprocessor = BiomarkerPreprocessor(age_max=10)

    values = np.array([[np.nan], [np.nan], [np.nan]])
    mask = np.array([[0.0], [0.0], [0.0]])  # No measurements
    censor = np.array([[0.0], [0.0], [0.0]])  # No censoring
    age = np.array([[1.0], [1.0], [1.0]])  # Never observed (normalized to 1.0)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # No biomarker data: value=0, mask=0, censor=0, age=max (normalized to 1.0)
    # Shape (T, N, 4) -> (3, 1, 4)
    expected = np.array(
        [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(encoded, expected)
