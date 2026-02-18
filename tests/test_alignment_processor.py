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
    (tmp_path / "ww.csv").touch()
    (tmp_path / "pop.csv").write_text("id,d.population\n")
    (tmp_path / "meta.nc").touch()
    (tmp_path / "hosp.csv").touch()
    (tmp_path / "deaths.csv").touch()

    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(tmp_path / "cases.csv"),
        mobility_path=str(tmp_path / "mob.nc"),
        wastewater_file=str(tmp_path / "ww.csv"),
        population_file=str(tmp_path / "pop.csv"),
        region_metadata_file=str(tmp_path / "meta.nc"),
        hospitalizations_file=str(tmp_path / "hosp.csv"),
        deaths_file=str(tmp_path / "deaths.csv"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        output_path=str(tmp_path / "out"),
        dataset_name="test",
        forecast_horizon=1,
        sequence_length=1,
    )


def test_edar_expansion_preserves_nan_gaps(mock_config):
    """Verify EDAR expansion preserves NaN gaps and fills only mask/censor/age defaults."""
    proc = AlignmentProcessor(mock_config)

    # Target dates: 2022-01-01 to 2022-01-10
    dates = pd.date_range("2022-01-01", periods=10)
    regions = ["R1"]

    # EDAR data only for middle dates, with a gap
    edar_dates = pd.date_range("2022-01-03", periods=5)
    # EDARProcessor output format for align_datasets is Dataset with per-variant variables (run, date, region)

    edar_ds = xr.Dataset(
        {
            "edar_biomarker_N1": (
                ["run_id", "date", REGION_COORD],
                np.array([[[10.0], [np.nan], [15.0], [20.0], [np.nan]]]),
            ),
            "edar_biomarker_N1_mask": (
                ["run_id", "date", REGION_COORD],
                np.array([[[True], [False], [True], [True], [False]]], dtype=bool),
            ),
            "edar_biomarker_N1_censor": (
                ["run_id", "date", REGION_COORD],
                np.array([[[0], [2], [0], [0], [2]]], dtype=np.uint8),
            ),
            "edar_biomarker_N1_age": (
                ["run_id", "date", REGION_COORD],
                np.array([[[0], [1], [0], [0], [1]]], dtype=np.uint8),
            ),
        },
        coords={"run_id": ["real"], "date": edar_dates, REGION_COORD: regions},
    )

    # Other mandatory datasets
    cases = xr.DataArray(
        np.ones((1, 10, 1)),
        coords={"run_id": ["real"], "date": dates, REGION_COORD: regions},
        dims=["run_id", "date", REGION_COORD],
        name="cases",
    )

    mobility = xr.Dataset(
        {
            "mobility": (
                ["run_id", "date", "origin", "destination"],
                np.ones((1, 10, 1, 1)),
            )
        },
        coords={
            "run_id": ["real"],
            "date": dates,
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
        cases_data=cases, mobility_data=mobility, edar_data=edar_ds, population_data=pop
    )

    # Check 2022-01-01 (expanded date): Biomarker should be NaN, but mask/censor/age should have defaults
    jan1 = aligned.sel(date="2022-01-01")
    assert np.isnan(jan1["edar_biomarker_N1"].values)
    assert not jan1["edar_biomarker_N1_mask"].values  # Default: no measurement (False)
    assert jan1["edar_biomarker_N1_censor"].values == 0  # Default: not censored
    assert jan1["edar_biomarker_N1_age"].values == 14  # Default: max_age (no data)

    # Check 2022-01-04 (gap in original EDAR data): Should remain NaN
    jan4 = aligned.sel(date="2022-01-04")
    assert np.isnan(jan4["edar_biomarker_N1"].values)
    assert not jan4["edar_biomarker_N1_mask"].values
    assert (
        jan4["edar_biomarker_N1_censor"].values == 2
    )  # Preserved from original (uint8)
    assert (
        jan4["edar_biomarker_N1_age"].values == 1
    )  # Preserved from original (uint8: 1 day)
