from datetime import datetime

import numpy as np
import pytest
import pandas as pd
import xarray as xr

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.alignment_processor import AlignmentProcessor
from data.preprocess.processors.synthetic_processor import SyntheticProcessor


def _make_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dummy = data_dir / "dummy.txt"
    dummy.write_text("ok")

    return PreprocessingConfig(
        data_dir=str(data_dir),
        synthetic_path=str(dummy),
        cases_file=str(dummy),
        mobility_path=str(dummy),
        wastewater_file=str(dummy),
        population_file=str(dummy),
        region_metadata_file=str(dummy),
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 4),
        forecast_horizon=1,
        sequence_length=1,
        output_path=str(tmp_path / "out"),
        dataset_name="synthetic_test",
        wastewater_flow_mode="concentration",
        min_density_threshold=0.0,
        validate_alignment=False,
        generate_alignment_report=False,
    )


def _make_synthetic_dataset(run_ids, dates, region_ids):
    rng = np.random.default_rng(42)

    cases = xr.DataArray(
        rng.random((len(run_ids), len(dates), len(region_ids))),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )

    mobility_base = xr.DataArray(
        rng.random((len(region_ids), len(region_ids))),
        dims=("origin", "target"),
        coords={"origin": region_ids, "target": region_ids},
    )
    mobility_kappa0 = xr.DataArray(
        rng.random((len(run_ids), len(dates))),
        dims=("run_id", "date"),
        coords={"run_id": run_ids, "date": dates},
    )

    population = xr.DataArray(
        rng.random((len(run_ids), len(region_ids))),
        dims=("run_id", "region_id"),
        coords={"run_id": run_ids, "region_id": region_ids},
    )

    edar_biomarker = xr.DataArray(
        rng.random((len(run_ids), len(dates), len(region_ids))),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
        name="edar_biomarker_N1",
    )

    return xr.Dataset(
        {
            "cases": cases,
            "mobility_base": mobility_base,
            "mobility_kappa0": mobility_kappa0,
            "population": population,
            "edar_biomarker_N1": edar_biomarker,
        }
    )


def _expected_age_from_mask(mask: np.ndarray, max_age: int = 14) -> np.ndarray:
    """Compute integer age channel from a (date, region) binary mask."""
    age = np.full(mask.shape, max_age, dtype=np.float32)
    for region_idx in range(mask.shape[1]):
        last_seen = None
        for t in range(mask.shape[0]):
            if mask[t, region_idx] > 0:
                last_seen = t
                age[t, region_idx] = 1.0
            elif last_seen is not None:
                age[t, region_idx] = float(min((t - last_seen) + 1, max_age))
    return age


@pytest.mark.epiforecaster
def test_alignment_preserves_mobility_coords(tmp_path):
    config = _make_config(tmp_path)
    processor = SyntheticProcessor(config)
    aligner = AlignmentProcessor(config)

    run_ids = np.array(["run_a", "run_b"], dtype="U10")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["001", "002", "003"], dtype="U10")

    ds = _make_synthetic_dataset(run_ids, dates, region_ids)

    cases_ds = processor._extract_cases(ds)
    mobility_ds = processor._extract_mobility(ds)
    population_da = processor._extract_population(ds)

    edar_ds = xr.Dataset({"edar_biomarker_N1": ds["edar_biomarker_N1"]})

    aligned = aligner.align_datasets(
        cases_data=cases_ds["cases"],
        mobility_data=mobility_ds,
        edar_data=edar_ds,
        population_data=population_da,
    )

    mobility = aligned["mobility"]
    assert list(mobility["origin"].values) == list(region_ids)
    assert list(mobility["destination"].values) == list(region_ids)

    sample = mobility.isel(
        run_id=0, date=0, origin=slice(0, 2), destination=slice(0, 2)
    )
    assert not bool(sample.isnull().all())


@pytest.mark.epiforecaster
def test_alignment_derives_deaths_mask_and_age_from_observations(tmp_path):
    config = _make_config(tmp_path)
    aligner = AlignmentProcessor(config)

    run_ids = np.array(["real"], dtype="U10")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["001", "002"], dtype="U10")

    ds = _make_synthetic_dataset(run_ids, dates, region_ids)

    cases_da = ds["cases"]
    mobility = xr.Dataset(
        {
            "mobility": xr.DataArray(
                np.ones((1, 4, 2, 2), dtype=np.float32),
                dims=("run_id", "date", "origin", "destination"),
                coords={
                    "run_id": run_ids,
                    "date": dates,
                    "origin": region_ids,
                    "destination": region_ids,
                },
            )
        }
    )
    population_da = xr.DataArray(
        np.array([1000.0, 2000.0], dtype=np.float32),
        dims=("region_id",),
        coords={"region_id": region_ids},
        name="population",
    )
    edar_ds = xr.Dataset({"edar_biomarker_N1": ds["edar_biomarker_N1"]})

    # Deaths values include true missing observations (NaN) that should remain mask=0.
    deaths_values = np.array(
        [
            [5.0, np.nan],
            [np.nan, np.nan],
            [np.nan, 3.0],
            [1.0, np.nan],
        ],
        dtype=np.float32,
    )
    deaths_ds = xr.Dataset(
        {
            "deaths": xr.DataArray(
                deaths_values,
                dims=("date", "region_id"),
                coords={"date": dates, "region_id": region_ids},
            )
        }
    )

    aligned = aligner.align_datasets(
        cases_data=cases_da,
        mobility_data=mobility,
        edar_data=edar_ds,
        population_data=population_da,
        deaths_data=deaths_ds,
    )

    deaths_mask = aligned["deaths_mask"].values
    expected_mask = np.isfinite(deaths_values).astype(np.float32)
    np.testing.assert_array_equal(deaths_mask, expected_mask)

    deaths_age = aligned["deaths_age"].values
    expected_age = _expected_age_from_mask(expected_mask)
    np.testing.assert_array_equal(deaths_age, expected_age)


@pytest.mark.epiforecaster
def test_alignment_recomputes_hospitalization_age_from_mask(tmp_path):
    config = _make_config(tmp_path)
    aligner = AlignmentProcessor(config)

    run_ids = np.array(["real"], dtype="U10")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["001", "002"], dtype="U10")

    ds = _make_synthetic_dataset(run_ids, dates, region_ids)
    cases_da = ds["cases"]
    mobility = xr.Dataset(
        {
            "mobility": xr.DataArray(
                np.ones((1, 4, 2, 2), dtype=np.float32),
                dims=("run_id", "date", "origin", "destination"),
                coords={
                    "run_id": run_ids,
                    "date": dates,
                    "origin": region_ids,
                    "destination": region_ids,
                },
            )
        }
    )
    population_da = xr.DataArray(
        np.array([1000.0, 2000.0], dtype=np.float32),
        dims=("region_id",),
        coords={"region_id": region_ids},
        name="population",
    )
    edar_ds = xr.Dataset({"edar_biomarker_N1": ds["edar_biomarker_N1"]})

    hosp_values = np.array(
        [[[10.0, 20.0], [9.0, 19.0], [8.0, 18.0], [7.0, 17.0]]], dtype=np.float32
    )
    hosp_mask = np.array(
        [[[1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=np.float32
    )
    # Broken upstream age channel that currently passes through unchanged.
    broken_age = np.ones((1, 4, 2), dtype=np.float32)
    hospitalizations_ds = xr.Dataset(
        {
            "hospitalizations": xr.DataArray(
                hosp_values,
                dims=("run_id", "date", "region_id"),
                coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
            ),
            "hospitalizations_mask": xr.DataArray(
                hosp_mask,
                dims=("run_id", "date", "region_id"),
                coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
            ),
            "hospitalizations_age": xr.DataArray(
                broken_age,
                dims=("run_id", "date", "region_id"),
                coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
            ),
        }
    )

    aligned = aligner.align_datasets(
        cases_data=cases_da,
        mobility_data=mobility,
        edar_data=edar_ds,
        population_data=population_da,
        hospitalizations_data=hospitalizations_ds,
    )

    out_age = aligned["hospitalizations_age"].values.squeeze(0)
    expected_age = _expected_age_from_mask(hosp_mask.squeeze(0))
    np.testing.assert_array_equal(out_age, expected_age)
