import pytest
import numpy as np
import pandas as pd
from datetime import datetime

@pytest.fixture(params=["sparse", "dense", "gap_heavy"])
def synthetic_time_series(request):
    """
    Parametrized fixture providing different types of synthetic time series.
    Returns a pandas Series with a date index.
    """
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    
    # Base signal (log-normal)
    values = rng.lognormal(mean=2, sigma=0.5, size=60)
    
    mode = request.param
    if mode == "sparse":
        # Only 20% of data present
        mask = rng.random(60) < 0.2
        values[~mask] = np.nan
    elif mode == "dense":
        # 95% of data present
        mask = rng.random(60) < 0.95
        values[~mask] = np.nan
    elif mode == "gap_heavy":
        # Large gaps in data
        values[10:25] = np.nan
        values[40:55] = np.nan
        
    return pd.Series(values, index=dates)
