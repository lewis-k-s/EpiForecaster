import numpy as np
import pytest
import xarray as xr
from data.biomarker_preprocessor import BiomarkerPreprocessor


def _make_dataset_with_mask(values: np.ndarray, mask: np.ndarray | None = None):
    """Create a test dataset with biomarker values and mask channel.

    Args:
        values: (T, N) array of biomarker values
        mask: (T, N) array of mask values. If None, mask is derived from values
              (1.0 where finite and positive, else 0.0)
    """
    if mask is None:
        mask = (np.isfinite(values) & (values > 0)).astype(float)

    return xr.Dataset(
        {
            "edar_biomarker_N1": xr.DataArray(values, dims=["date", "region_id"]),
            "edar_biomarker_N1_mask": xr.DataArray(mask, dims=["date", "region_id"]),
        }
    )


@pytest.mark.region
def test_scaler_fitted_on_train_only():
    """Verify scalers are computed using only train nodes (data already log1p-transformed)."""
    # Values are already log1p-transformed from preprocessing pipeline
    # e.g., log1p(1.0)=0.69, log1p(2.0)=1.10, log1p(5.0)=1.79, log1p(10.0)=2.40
    # Mask indicates which values are truly observed (not interpolated/filled)
    values = np.array([[0.69, 1.10], [np.nan, 2.40], [1.79, np.nan]])
    # All finite positive values are observed
    mask = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()

    train_nodes = [0]
    preprocessor.fit_scaler(dataset, train_nodes)

    # Values from train node (region 0): 0.69, 1.79 (excluding nan)
    expected_values = np.array([0.69, 1.79])
    expected_center = np.median(expected_values)
    assert preprocessor.scaler_params is not None
    assert np.isclose(
        preprocessor.scaler_params.center["edar_biomarker_N1"], expected_center
    )


@pytest.mark.region
def test_scaler_reused_for_val_test():
    """Verify same scaler params are used for val/test."""
    values = np.array([[0.69, 1.10], [np.nan, 2.40]])
    mask = np.array([[1.0, 1.0], [0.0, 1.0]])
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    train_params = preprocessor.scaler_params
    assert train_params is not None

    preprocessor.set_scaler_params(train_params)

    assert preprocessor.scaler_params is not None
    assert preprocessor.scaler_params.center == train_params.center
    assert preprocessor.scaler_params.scale == train_params.scale


@pytest.mark.region
def test_near_constant_biomarker():
    """Verify scale=1.0 when IQR≈0."""
    values = np.array([[1.0], [1.0], [1.0]])
    mask = np.array([[1.0], [1.0], [1.0]])
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    assert preprocessor.scaler_params is not None
    assert preprocessor.scaler_params.scale["edar_biomarker_N1"] == 1.0


@pytest.mark.region
def test_scaler_no_observed_values_skips_variant():
    """Verify individual variants with no observed values (mask > 0) are skipped."""
    # Create dataset with two variants - one with observed values, one without
    values = np.array([[1.0], [2.0]])
    mask = np.array([[1.0], [0.0]])  # Only first value observed
    dataset = _make_dataset_with_mask(values, mask)

    # Add another variant with NO observed values
    dataset = dataset.assign(
        {
            "edar_biomarker_N2": xr.DataArray(
                np.array([[np.nan], [np.nan]]), dims=["date", "region_id"]
            ),
            "edar_biomarker_N2_mask": xr.DataArray(
                np.array([[0.0], [0.0]]), dims=["date", "region_id"]
            ),
        }
    )

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    # N1 should have scaler (has observed values), N2 should not (skipped)
    assert preprocessor.scaler_params is not None
    assert "edar_biomarker_N1" in preprocessor.scaler_params.center
    assert "edar_biomarker_N2" not in preprocessor.scaler_params.center


@pytest.mark.region
def test_scaler_all_variants_no_observed_uses_default():
    """Verify default scaler is used when ALL variants have no observed values."""
    values = np.array([[np.nan], [np.nan]])
    mask = np.array([[0.0], [0.0]])  # Nothing observed
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    # Should not raise, just use default scaler
    preprocessor.fit_scaler(dataset, [0])

    # Default scaler is fitted (center=0, scale=1)
    assert preprocessor.scaler_params is not None
    assert preprocessor.scaler_params.is_fitted
    assert "edar_biomarker_N1" in preprocessor.scaler_params.center
    assert preprocessor.scaler_params.center["edar_biomarker_N1"] == 0.0
    assert preprocessor.scaler_params.scale["edar_biomarker_N1"] == 1.0


@pytest.mark.region
def test_scaler_excludes_zeros():
    """Verify scaler excludes zeros (below detection limit) from fitting.

    Data is already log1p-transformed from preprocessing pipeline.
    Zeros represent below-detection-limit values and should be excluded.
    """
    values = np.array([[0.69, 0.0]])  # log1p(1.0)=0.69, 0 is below-LD
    # Mask 0 for the zero value - it's not truly observed
    mask = np.array([[1.0, 0.0]])
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0, 1])

    # Only 0.69 is observed (mask > 0, finite, positive)
    expected_center = 0.69
    expected_scale = 1.0

    assert preprocessor.scaler_params is not None
    assert np.isclose(
        preprocessor.scaler_params.center["edar_biomarker_N1"], expected_center
    )
    assert np.isclose(
        preprocessor.scaler_params.scale["edar_biomarker_N1"], expected_scale
    )


@pytest.mark.region
def test_scaler_uses_mask_not_just_finite():
    """Verify scaler uses mask channel, not just finite values.

    This is the key fix: values where mask=0 should be excluded even if finite,
    because mask=0 indicates interpolated/filled values that shouldn't affect stats.
    """
    # Region 0 has observed values [1.0, 2.0] and interpolated value 100.0
    # Without mask filtering, 100.0 would skew the median
    # With mask filtering, only [1.0, 2.0] are used
    values = np.array([[1.0], [2.0], [100.0]])  # Last value is interpolated
    mask = np.array([[1.0], [1.0], [0.0]])  # Last value has mask=0
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    # Only observed values [1.0, 2.0] should be used
    observed_values = np.array([1.0, 2.0])
    expected_center = np.median(observed_values)  # 1.5

    assert preprocessor.scaler_params is not None
    assert np.isclose(
        preprocessor.scaler_params.center["edar_biomarker_N1"], expected_center
    )
    # Without the fix, center would be np.median([1.0, 2.0, 100.0]) = 2.0


@pytest.mark.region
def test_scaler_handles_inf_values():
    """Verify scaler excludes inf values via isfinite in observed mask."""
    values = np.array([[1.0], [np.inf], [2.0]])
    mask = np.array([[1.0], [1.0], [1.0]])
    dataset = _make_dataset_with_mask(values, mask)

    preprocessor = BiomarkerPreprocessor()
    # isfinite in observed mask excludes inf automatically
    preprocessor.fit_scaler(dataset, [0])

    # Only finite values [1.0, 2.0] should be used
    expected_center = np.median(np.array([1.0, 2.0]))
    assert preprocessor.scaler_params is not None
    assert np.isclose(
        preprocessor.scaler_params.center["edar_biomarker_N1"], expected_center
    )


@pytest.mark.region
def test_scaler_all_inf_skips_variant():
    """Verify variants with all inf values are skipped (caught by isfinite in mask)."""
    # Create dataset with two variants - one valid, one all inf
    values = np.array([[1.0], [2.0]])
    mask = np.array([[1.0], [1.0]])
    dataset = _make_dataset_with_mask(values, mask)

    # Add another variant with all inf values
    dataset = dataset.assign(
        {
            "edar_biomarker_N2": xr.DataArray(
                np.array([[np.inf], [np.inf]]), dims=["date", "region_id"]
            ),
            "edar_biomarker_N2_mask": xr.DataArray(
                np.array([[1.0], [1.0]]), dims=["date", "region_id"]
            ),
        }
    )

    preprocessor = BiomarkerPreprocessor()
    preprocessor.fit_scaler(dataset, [0])

    # N1 should have scaler, N2 should be skipped (all inf -> no finite values)
    assert preprocessor.scaler_params is not None
    assert "edar_biomarker_N1" in preprocessor.scaler_params.center
    assert "edar_biomarker_N2" not in preprocessor.scaler_params.center


@pytest.mark.region
def test_missing_mask_channel_raises_error():
    """Verify error when mask channel is missing."""
    dataset = xr.Dataset(
        {
            "edar_biomarker_N1": xr.DataArray(
                np.array([[0.69, 1.10]]),
                dims=["date", "region_id"],
            )
            # Missing edar_biomarker_N1_mask
        }
    )

    preprocessor = BiomarkerPreprocessor()

    with pytest.raises(ValueError, match="Missing mask channel for scaler fitting"):
        preprocessor.fit_scaler(dataset, [0])
