import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from dataviz.catchment_cases_vs_biomarkers import aggregate_cases_to_catchments


def test_aggregate_cases_to_catchments_uses_canonical_weighted_average():
    dates = pd.date_range("2022-01-01", periods=2)
    regions = ["R1", "R2"]
    cases = np.array([[[2.0, 6.0], [4.0, 10.0]]], dtype=np.float32)
    cases_mask = np.array([[[True, True], [True, False]]])
    dataset = xr.Dataset(
        {
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD), cases),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
        },
        coords={"run_id": ["real"], TEMPORAL_COORD: dates, REGION_COORD: regions},
    )
    mapping = xr.DataArray(
        np.array([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32),
        dims=("edar_id", REGION_COORD),
        coords={"edar_id": ["E1", "E2"], REGION_COORD: regions},
    )

    result = aggregate_cases_to_catchments(dataset, mapping)

    assert result["cases"].sel(date=dates[0], edar_id="E1").item() == pytest.approx(
        5.0
    )
    assert result["cases"].sel(date=dates[1], edar_id="E1").item() == pytest.approx(
        4.0
    )
    assert result["cases"].sel(date=dates[0], edar_id="E2").item() == pytest.approx(
        2.0
    )
    assert result["cases"].sel(date=dates[1], edar_id="E2").item() == pytest.approx(
        4.0
    )
