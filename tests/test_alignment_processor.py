"""Tests for the alignment processor."""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from data.preprocess.processors.alignment_processor import AlignmentProcessor
from data.preprocess.config import PreprocessingConfig, REGION_COORD


@pytest.fixture
def mock_config(tmp_path):
    # Create dummy files to pass config validation
    (tmp_path / "cases.csv").touch()
    (tmp_path / "mob.nc").touch()
    (tmp_path / "pop.csv").write_text("id,d.population\n")
    (tmp_path / "hosp.csv").touch()
    (tmp_path / "deaths.csv").touch()

    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(tmp_path / "cases.csv"),
        mobility_path=str(tmp_path / "mob.nc"),
        population_file=str(tmp_path / "pop.csv"),
        hospitalizations_file=str(tmp_path / "hosp.csv"),
        deaths_file=str(tmp_path / "deaths.csv"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        output_path=str(tmp_path / "out"),
        dataset_name="test",
        forecast_horizon=1,
        sequence_length=1,
    )


def test_outer_alignment_preserves_full_range_and_mobility_mask(mock_config):
    """Outer alignment keeps the configured range and marks missing mobility."""
    mock_config.temporal_alignment_mode = "outer"
    proc = AlignmentProcessor(mock_config)

    target_dates = pd.date_range("2022-01-01", periods=10)
    short_dates = pd.date_range("2022-01-01", periods=5)
    regions = ["R1"]

    cases = xr.Dataset(
        {
            "cases": (
                ["run_id", "date", REGION_COORD],
                np.ones((1, 5, 1), dtype=np.float32),
            ),
            "cases_mask": (
                ["run_id", "date", REGION_COORD],
                np.ones((1, 5, 1), dtype=bool),
            ),
        },
        coords={"run_id": ["real"], "date": short_dates, REGION_COORD: regions},
    )
    mobility = xr.Dataset(
        {
            "mobility": (
                ["run_id", "date", "origin", "destination"],
                np.ones((1, 5, 1, 1), dtype=np.float32),
            )
        },
        coords={
            "run_id": ["real"],
            "date": short_dates,
            "origin": regions,
            "destination": regions,
        },
    )
    pop = xr.DataArray(
        np.ones(1),
        coords={REGION_COORD: regions},
        dims=[REGION_COORD],
        name="population",
    )

    aligned = proc.align_datasets(
        cases_data=cases,
        mobility_data=mobility,
        population_data=pop,
    )

    np.testing.assert_array_equal(aligned["date"].values, target_dates.values)
    np.testing.assert_array_equal(
        aligned["mobility_time_mask"].values,
        np.array([[True, True, True, True, True, False, False, False, False, False]]),
    )
    assert not bool(aligned["cases_mask"].sel(date="2022-01-06").values.item())
    assert not any(name.startswith("edar_") for name in aligned.variables)
