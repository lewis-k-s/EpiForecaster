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
