from datetime import datetime

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import warnings
import xarray as xr

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.synthetic_processor import SyntheticProcessor
from data.preprocess.pipeline import OfflinePreprocessingPipeline


def _write_minimal_synthetic_zarr(path, *, run_ids, dates, region_ids, edar_ids):
    rng = np.random.default_rng(123)

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

    edar_vars = {}
    variants = ["N1", "N2", "IP4"]
    for variant in variants:
        biomarker = xr.DataArray(
            rng.random((len(run_ids), len(dates), len(edar_ids))),
            dims=("run_id", "date", "edar_id"),
            coords={"run_id": run_ids, "date": dates, "edar_id": edar_ids},
        )
        lod = xr.DataArray(
            rng.random((len(run_ids), len(edar_ids))),
            dims=("run_id", "edar_id"),
            coords={"run_id": run_ids, "edar_id": edar_ids},
        )
        edar_vars[f"edar_biomarker_{variant}"] = biomarker
        edar_vars[f"edar_biomarker_{variant}_LoD"] = lod

    ds = xr.Dataset(
        {
            "cases": cases,
            "mobility_base": mobility_base,
            "mobility_kappa0": mobility_kappa0,
            "population": population,
            **edar_vars,
        }
    )

    ds.to_zarr(path, mode="w", zarr_format=2)


def _write_region_metadata(path, *, region_ids, edar_ids):
    data = np.ones((len(edar_ids), len(region_ids)), dtype=np.float32)
    emap = xr.DataArray(
        data,
        dims=("edar_id", "home"),
        coords={"edar_id": edar_ids, "home": region_ids},
    )
    emap.to_netcdf(path)


def _make_config(tmp_path, synthetic_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dummy = data_dir / "dummy.txt"
    dummy.write_text("ok")

    return PreprocessingConfig(
        data_dir=str(data_dir),
        synthetic_path=str(synthetic_path),
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
        run_id_chunk_size=1,
        chunk_size=10,
        min_density_threshold=0.0,
        validate_alignment=False,
        generate_alignment_report=False,
    )


@pytest.mark.epiforecaster
def test_synthetic_processor_preserves_run_id(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a", "synth_b"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2", "r3"], dtype="U10")
    edar_ids = np.array(["e1", "e2"], dtype="U10")
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=run_ids,
        dates=dates,
        region_ids=region_ids,
        edar_ids=edar_ids,
    )

    config = _make_config(tmp_path, synthetic_path)
    processor = SyntheticProcessor(config)

    result = processor.process(str(synthetic_path))

    cases = result["cases"]["cases"]
    mobility = result["mobility"]["mobility"]
    population = result["population"]

    assert cases.dims == ("run_id", "date", "region_id")
    assert mobility.dims == ("run_id", "date", "origin", "destination")
    assert population.dims == ("run_id", "region_id")
    assert len(cases.run_id) == 2


@pytest.mark.epiforecaster
def test_synthetic_processor_run_filter(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a", "synth_b", "synth_c"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")
    edar_ids = np.array(["e1"], dtype="U10")
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=run_ids,
        dates=dates,
        region_ids=region_ids,
        edar_ids=edar_ids,
    )

    config = _make_config(tmp_path, synthetic_path)
    processor = SyntheticProcessor(config)

    # Suppress chunking warning when filtering runs - this is expected behavior
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified chunks separate the stored chunks",
            category=UserWarning,
        )
        result = processor.process(str(synthetic_path), run_filter=["synth_b"])
    cases = result["cases"]["cases"]

    assert list(cases.run_id.values) == ["synth_b"]


@pytest.mark.epiforecaster
def test_mobility_reconstruction_matches_formula(tmp_path):
    run_ids = np.array(["synth_a", "synth_b"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")

    mobility_base = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("origin", "target"),
        coords={"origin": region_ids, "target": region_ids},
    )
    mobility_kappa0 = xr.DataArray(
        np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float32),
        dims=("run_id", "date"),
        coords={"run_id": run_ids, "date": dates},
    )

    ds = xr.Dataset(
        {
            "mobility_base": mobility_base,
            "mobility_kappa0": mobility_kappa0,
        }
    )

    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    processor = SyntheticProcessor(config)

    mobility_ds = processor._extract_mobility(ds)
    mobility = mobility_ds["mobility"]

    expected = 1.0 - mobility_kappa0.values
    for run_idx in range(len(run_ids)):
        for t in range(len(dates)):
            assert np.allclose(mobility.values[run_idx, t], expected[run_idx, t])


@pytest.mark.epiforecaster
def test_pipeline_save_rechunks_run_only(tmp_path):
    run_ids = np.array(["s1", "s2", "s3"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    region_ids = np.array(["r1", "r2", "r3", "r4"], dtype="U10")

    data = np.random.rand(len(run_ids), len(dates), len(region_ids))

    cases = xr.DataArray(
        da.from_array(data, chunks=(2, 2, 4)),  # type: ignore[arg-type]
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )
    biomarker_mask = xr.DataArray(
        da.from_array(data, chunks=(1, 3, 4)),  # type: ignore[arg-type]
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )

    aligned_dataset = xr.Dataset(
        {
            "cases": cases,
            "edar_biomarker_N1_mask": biomarker_mask,
        }
    )

    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    pipeline = OfflinePreprocessingPipeline(config)

    output_path = pipeline._save_aligned_dataset(aligned_dataset)
    saved = xr.open_zarr(output_path)

    chunks = saved["edar_biomarker_N1_mask"].encoding.get("chunks")
    assert chunks is not None
    assert chunks[0] == min(config.run_id_chunk_size, len(run_ids))
    assert chunks[1] == len(dates)
    assert chunks[2] == len(region_ids)

    saved.close()


@pytest.mark.epiforecaster
def test_pipeline_end_to_end_synthetic(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    region_metadata_path = tmp_path / "edar_map.nc"

    run_ids = np.array(["synth_a", "synth_b"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2", "r3"], dtype="U10")
    edar_ids = np.array(["e1", "e2"], dtype="U10")

    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=run_ids,
        dates=dates,
        region_ids=region_ids,
        edar_ids=edar_ids,
    )
    _write_region_metadata(
        region_metadata_path,
        region_ids=region_ids,
        edar_ids=edar_ids,
    )

    config = _make_config(tmp_path, synthetic_path)
    config.region_metadata_file = str(region_metadata_path)

    pipeline = OfflinePreprocessingPipeline(config)
    output_path = pipeline.run()

    ds = xr.open_zarr(output_path)
    assert "run_id" in ds.dims
    assert "mobility" in ds
    assert ds["mobility"].dims[0] == "run_id"
    assert len(ds["run_id"]) == len(run_ids)
    ds.close()


@pytest.mark.epiforecaster
def test_save_aligned_dataset_preserves_mobility(tmp_path):
    run_ids = np.array(["r1", "r2"], dtype="U10")
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    region_ids = np.array(["001", "002", "003", "004"], dtype="U10")

    mobility_data = np.ones((2, 3, 4, 4), dtype=np.float32)
    cases_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    mobility = xr.DataArray(
        da.from_array(mobility_data, chunks=(1, 3, 4, 4)),  # type: ignore[arg-type]
        dims=("run_id", "date", "origin", "destination"),
        coords={
            "run_id": run_ids,
            "date": dates,
            "origin": region_ids,
            "destination": region_ids,
        },
    )

    cases = xr.DataArray(
        da.from_array(cases_data, chunks=(1, 3, 4)),  # type: ignore[arg-type]
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )

    aligned_dataset = xr.Dataset({"mobility": mobility, "cases": cases})

    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    pipeline = OfflinePreprocessingPipeline(config)

    output_path = pipeline._save_aligned_dataset(aligned_dataset)
    saved = xr.open_zarr(output_path)

    saved_mob = saved["mobility"].isel(
        run_id=0, date=0, origin=slice(0, 2), destination=slice(0, 2)
    )
    saved_mob = saved_mob.compute()
    assert not bool(saved_mob.isnull().all())
    assert float(saved_mob.min()) >= 1.0
    assert float(saved_mob.max()) <= 1.0

    saved.close()
