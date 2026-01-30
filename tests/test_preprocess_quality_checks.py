import numpy as np
import pytest
import xarray as xr

from data.preprocess.processors.quality_checks import (
    DataQualityThresholds,
    validate_notna_and_std,
)


@pytest.mark.epiforecaster
def test_validate_notna_and_std_raises_on_low_notna():
    da = xr.DataArray([1.0, np.nan, np.nan], dims=["t"])
    with pytest.raises(ValueError, match="notna fraction"):
        validate_notna_and_std(
            da, name="x", thresholds=DataQualityThresholds(min_notna_fraction=0.9)
        )


@pytest.mark.epiforecaster
def test_validate_notna_and_std_raises_on_constant_std():
    da = xr.DataArray([2.0, 2.0, 2.0], dims=["t"])
    with pytest.raises(ValueError, match="std"):
        validate_notna_and_std(
            da, name="x", thresholds=DataQualityThresholds(min_std_epsilon=1e-6)
        )


@pytest.mark.epiforecaster
def test_validate_notna_and_std_passes_on_variable_data():
    da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"])
    validate_notna_and_std(
        da, name="x", thresholds=DataQualityThresholds(min_std_epsilon=1e-6)
    )


@pytest.mark.epiforecaster
def test_validate_notna_and_std_raises_on_empty_array():
    da = xr.DataArray([], dims=["t"])
    with pytest.raises(ValueError, match="empty array"):
        validate_notna_and_std(da, name="x")


@pytest.mark.epiforecaster
def test_validate_notna_and_std_raises_on_nan_std():
    da = xr.DataArray([np.nan, np.nan], dims=["t"])
    with pytest.raises(ValueError, match="std is NaN|notna fraction"):
        validate_notna_and_std(da, name="x")
