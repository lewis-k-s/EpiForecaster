import numpy as np
import pytest
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor, BiomarkerScalerParams


def _create_dataset_with_channels(
    values, mask=None, censor=None, age=None, dims=("date", "region_id")
):
    """Helper to create xarray dataset with all required channels.

    Channel layout: [value, mask, censor, age] - 4 channels per variant

    Note:
    - Values should be already log1p-transformed (from preprocessing pipeline)
    - Age is stored as uint8 (0-14 raw days), normalized to [0,1] at load time.
    """
    T, N = values.shape
    da = xr.DataArray(
        values,
        dims=dims,
        coords={"date": range(T), "region_id": range(N)},
    )

    # Default mask: bool (True if measured, False otherwise)
    if mask is None:
        mask = np.isfinite(values) & (values > 0)

    # Default censor: uint8 (0=uncensored, 1=censored, 2=missing)
    if censor is None:
        censor = np.zeros_like(values, dtype=np.uint8)

    # Default age: uint8 (0-14 raw days since last measurement)
    if age is None:
        age = np.zeros_like(values, dtype=np.uint8)
        for i in range(N):
            last_seen = -1
            for t in range(T):
                if mask[t, i]:
                    last_seen = t
                    age[t, i] = 0
                elif last_seen >= 0:
                    age[t, i] = min(t - last_seen, 14)
                else:
                    age[t, i] = 14  # Never observed

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


@pytest.mark.region
def test_value_channel_passthrough():
    """Verify value channel passes through already-log1p-transformed values."""
    preprocessor = BiomarkerPreprocessor()

    # Values are already log1p-transformed from preprocessing pipeline
    # e.g., log1p(1.0) = 0.693, log1p(4.0) = 1.609
    values = np.array([[0.693], [np.nan], [np.nan], [1.609], [np.nan]])
    mask = np.array([[True], [False], [False], [True], [False]])
    censor = np.array([[0], [0], [0], [0], [0]], dtype=np.uint8)
    age = np.array([[0], [1], [2], [0], [1]], dtype=np.uint8)  # Raw days
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)
    # Shape: (5, 1, 4) - 4 channels: [value, mask, censor, age]

    # Value channel: NaN becomes 0.0, valid values pass through (already log1p)
    # No LOCF, so NaN values stay as 0.0
    expected_value = np.array([0.693, 0.0, 0.0, 1.609, 0.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


@pytest.mark.region
def test_mask_channel_correctness():
    """Verify mask channel is passed through correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]])
    mask = np.array([[True], [False], [False], [True], [False]])
    censor = np.array([[0], [0], [0], [0], [0]], dtype=np.uint8)
    age = np.array([[0], [1], [2], [0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_mask = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 1], expected_mask)


@pytest.mark.region
def test_censor_channel_passthrough():
    """Verify censor channel is passed through correctly."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [0.0], [4.0], [np.nan]])
    mask = np.array([[True], [False], [False], [True], [False]])
    censor = np.array([[0], [0], [1], [0], [0]], dtype=np.uint8)  # One censored value
    age = np.array([[0], [1], [2], [0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    expected_censor = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0, 2], expected_censor)


@pytest.mark.region
def test_age_channel_passthrough():
    """Verify age channel is normalized from uint8 (0-14) to float (0-1)."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[1.0], [np.nan], [np.nan]])
    # Age stored as raw days (0-14), not normalized
    age = np.array([[7], [4], [3]], dtype=np.uint8)  # 7, 4, 3 days
    mask = np.array([[True], [False], [False]])
    censor = np.array([[0], [0], [0]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Age channel should be normalized: 7/14=0.5, 4/14≈0.286, 3/14≈0.214
    expected_age = np.array([7 / 14, 4 / 14, 3 / 14])
    np.testing.assert_array_almost_equal(encoded[:, 0, 3], expected_age)


@pytest.mark.region
def test_robust_scaling_only():
    """Verify robust scaling is applied to already-log-transformed values."""
    preprocessor = BiomarkerPreprocessor()

    scaler_params = BiomarkerScalerParams(
        center={"edar_biomarker_N1": 2.0},
        scale={"edar_biomarker_N1": 1.0},
        is_fitted=True,
    )
    preprocessor.set_scaler_params(scaler_params)

    # Values are already log1p-transformed
    values = np.array([[1.0], [np.nan]])
    mask = np.array([[True], [False]])
    censor = np.array([[0], [0]], dtype=np.uint8)
    age = np.array([[0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # (1.0 - 2.0) / 1.0 = -1.0
    # NaN value becomes 0.0, then (0.0 - 2.0) / 1.0 = -2.0
    expected_value = np.array([-1.0, -2.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)


@pytest.mark.region
def test_zeros_below_detection_limit():
    """Verify zero values are handled correctly (below-LD)."""
    preprocessor = BiomarkerPreprocessor()

    # Values are already log1p-transformed
    # log1p(5.0) = 1.79, log1p(3.0) = 1.39
    # Zeros represent below-detection-limit values (already 0.0 after pipeline)
    values = np.array([[1.79], [0.0], [0.0], [np.nan], [1.39], [np.nan]])
    # Mask: False for zeros (below-LD), False for NaN (missing), True for positive
    mask = np.array([[True], [False], [False], [False], [True], [False]])
    # Censor: 1 for zeros (censored at LD), 0 for others
    censor = np.array([[0], [1], [1], [0], [0], [0]], dtype=np.uint8)
    # Age: raw days (0-14), increments through gaps
    age = np.array([[0], [1], [2], [3], [0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Value channel: zeros and NaN become 0.0 (no measurement)
    expected_value = np.array([1.79, 0.0, 0.0, 0.0, 1.39, 0.0])
    np.testing.assert_array_almost_equal(encoded[:, 0, 0], expected_value)

    # Mask and censor should be converted to float
    np.testing.assert_array_equal(encoded[:, 0, 1], mask.astype(float).flatten())
    np.testing.assert_array_equal(encoded[:, 0, 2], censor.astype(float).flatten())
    # Age should be normalized: age/14
    expected_age = age.flatten().astype(float) / 14.0
    np.testing.assert_array_almost_equal(encoded[:, 0, 3], expected_age)


@pytest.mark.region
def test_multiple_regions():
    """Verify preprocessing works correctly with multiple regions."""
    preprocessor = BiomarkerPreprocessor()

    # Values are already log1p-transformed
    # log1p(1.0) = 0.69, log1p(2.0) = 1.10, log1p(3.0) = 1.39, log1p(4.0) = 1.61, log1p(5.0) = 1.79
    values = np.array(
        [
            [0.69, 1.10, np.nan],
            [np.nan, 1.39, 1.61],
            [1.79, np.nan, np.nan],
        ]
    )
    mask = np.array(
        [
            [True, True, False],
            [False, True, True],
            [True, False, False],
        ]
    )
    censor = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    age = np.array(
        [
            [0, 0, 14],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Value channel: NaN becomes 0.0, valid values pass through
    expected_value = np.array(
        [
            [0.69, 1.10, 0.0],
            [0.0, 1.39, 1.61],
            [1.79, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_almost_equal(encoded[:, :, 0], expected_value)
    np.testing.assert_array_equal(encoded[:, :, 1], mask.astype(float))
    np.testing.assert_array_equal(encoded[:, :, 2], censor.astype(float))
    # Age normalized to [0, 1]
    expected_age = age.astype(float) / 14.0
    np.testing.assert_array_almost_equal(encoded[:, :, 3], expected_age)


@pytest.mark.region
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


@pytest.mark.region
def test_censor_flag_validation():
    """Verify censor channel values are validated (0, 1, or 2 flags)."""
    preprocessor = BiomarkerPreprocessor()

    values = np.array([[5.0], [0.0], [3.0]])
    mask = np.array([[True], [False], [True]])
    # Censor flags: 0=uncensored, 1=censored
    censor = np.array([[0], [1], [0]], dtype=np.uint8)
    age = np.array([[0], [1], [0]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Verify censor flags are preserved correctly (as float)
    np.testing.assert_array_equal(encoded[:, 0, 2], [0.0, 1.0, 0.0])


@pytest.mark.region
def test_censor_alignment_with_mask():
    """Verify censored points align with mask=0 (unmeasured)."""
    preprocessor = BiomarkerPreprocessor()

    # When a point is censored, mask should be False (no valid measurement)
    values = np.array([[5.0], [0.0], [3.0], [0.0]])
    mask = np.array(
        [[True], [False], [True], [False]]
    )  # Mask False for censored values
    censor = np.array([[0], [1], [0], [1]], dtype=np.uint8)  # Censor flag for zeros
    age = np.array([[0], [1], [0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # Verify mask=0 where censor=1
    for i in range(4):
        if encoded[i, 0, 2] == 1.0:  # If censored
            assert encoded[i, 0, 1] == 0.0, "Censored points should have mask=0"


@pytest.mark.region
def test_clip_range():
    """Verify values are clipped to the configured range after scaling."""
    preprocessor = BiomarkerPreprocessor(clip_range=(-2.0, 2.0))

    scaler_params = BiomarkerScalerParams(
        center={"edar_biomarker_N1": 0.0},
        scale={"edar_biomarker_N1": 1.0},
        is_fitted=True,
    )
    preprocessor.set_scaler_params(scaler_params)

    # Large positive value (already log1p-transformed)
    values = np.array([[100.0], [0.0]])
    mask = np.array([[True], [False]])
    censor = np.array([[0], [0]], dtype=np.uint8)
    age = np.array([[0], [1]], dtype=np.uint8)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # 100.0 should be clipped to 2.0 (upper bound)
    # 0.0 stays at 0.0
    assert encoded[0, 0, 0] == 2.0
    assert encoded[1, 0, 0] == 0.0


@pytest.mark.region
def test_region_without_biomarker_data():
    """Verify regions without biomarker data get zero encoding."""
    preprocessor = BiomarkerPreprocessor(age_max=14)

    values = np.array([[np.nan], [np.nan], [np.nan]])
    mask = np.array([[False], [False], [False]])  # No measurements
    censor = np.array([[0], [0], [0]], dtype=np.uint8)  # No censoring
    age = np.array([[14], [14], [14]], dtype=np.uint8)  # Never observed (max age)
    ds = _create_dataset_with_channels(values, mask=mask, censor=censor, age=age)

    encoded = preprocessor.preprocess_dataset(ds)

    # No biomarker data: value=0, mask=0, censor=0, age=14/14=1.0 (normalized)
    # Shape (T, N, 4) -> (3, 1, 4)
    expected = np.array(
        [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(encoded, expected)
