import numpy as np
import pandas as pd
import pytest
import xarray as xr


from data.cases_preprocessor import CasesPreprocessor, CasesPreprocessorConfig


@pytest.mark.epiforecaster
def test_cases_preprocessor_basic():
    # Create dummy dataset
    dates = pd.date_range("2020-01-01", periods=10)
    regions = ["A", "B"]

    cases = np.random.rand(10, 2) * 100
    # Add some NaNs
    cases[5, 0] = np.nan

    population = np.array([1000, 2000])  # A, B

    ds = xr.Dataset(
        data_vars={
            "cases": (("date", "region_id"), cases),
            "population": (("region_id"), population),
        },
        coords={"date": dates, "region_id": regions},
    )

    config = CasesPreprocessorConfig(history_length=3, log_scale=True, per_100k=True)

    processor = CasesPreprocessor(config)

    p_cases, p_mean, p_std = processor.preprocess_dataset(ds)

    # p_cases has 3 channels: [value, mask, age]
    assert p_cases.shape == (10, 2, 3)
    assert p_mean.shape == (10, 2, 1)
    assert p_std.shape == (10, 2, 1)

    # Check per 100k (channel 0 is value)
    raw_val = cases[0, 0]
    expected_100k = raw_val * (100000 / 1000)
    expected_log = np.log1p(expected_100k)

    # float16 has ~3-4 decimal digits precision, use rtol=1e-2
    assert np.isclose(p_cases[0, 0, 0].item(), expected_log, rtol=1e-2)

    # Check mask channel (channel 1) - should be 1.0 for finite values
    assert p_cases[0, 0, 1].item() == 1.0
    # NaN values should have mask 0.0
    assert p_cases[5, 0, 1].item() == 0.0

    # Check rolling stats (t=2, window=3 -> indices 0,1,2)
    # The returned mean/std at index t is computed over [t-L+1 : t+1].

    vals = p_cases[:3, 0, 0].float().numpy()  # Convert to float32 for comparison
    expected_mean = np.mean(vals)
    assert np.isclose(p_mean[2, 0, 0].item(), expected_mean, rtol=1e-2)

    # Check NaN handling (t=5 has NaN)
    # Window at 5 includes 3, 4, 5. 5 is NaN.
    # Should ignore NaN.
    vals_nan = p_cases[3:6, 0, 0].float().numpy()  # Convert to float32 for comparison
    # indices 3, 4 are valid. 5 is NaN.
    # Note: p_cases has NaN where input had NaN (after log transform)
    assert np.isnan(vals_nan[2])

    expected_mean_nan = np.nanmean(vals_nan)
    assert np.isclose(p_mean[5, 0, 0].item(), expected_mean_nan, rtol=1e-2)

    # Check std (float16 has poor precision for small values, use atol)
    expected_std_nan = np.nanstd(vals_nan)
    assert np.isclose(p_std[5, 0, 0].item(), expected_std_nan, rtol=1e-1, atol=1e-2)
