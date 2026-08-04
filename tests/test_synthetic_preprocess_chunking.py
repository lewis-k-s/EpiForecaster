import json

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import warnings
import xarray as xr

from data.preprocess.processors.synthetic_processor import SyntheticProcessor
from data.preprocess.pipeline import OfflinePreprocessingPipeline
from tests.dataset_test_helpers import make_preprocessing_config


def _write_minimal_synthetic_zarr(
    path,
    *,
    run_ids,
    dates,
    region_ids,
    edar_ids,
    latent_h_name="latent_H_true",
    include_deaths_true=True,
    death_flow_offset=0.0,
):
    rng = np.random.default_rng(123)

    cases = xr.DataArray(
        rng.random((len(run_ids), len(dates), len(region_ids))),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )

    # Required mask channel for clinical series preprocessing
    # cases_age is computed from mask by SyntheticProcessor
    cases_mask = xr.DataArray(
        rng.random((len(run_ids), len(dates), len(region_ids))) > 0.1,  # ~90% observed
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
        rng.integers(500, 1500, size=(len(run_ids), len(region_ids))).astype(
            np.float32
        ),
        dims=("run_id", "region_id"),
        coords={"run_id": run_ids, "region_id": region_ids},
    )

    latent_s = xr.DataArray(
        rng.random((len(run_ids), len(region_ids), len(dates))).astype(np.float32)
        * population.values[:, :, None],
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
    )
    latent_i = xr.DataArray(
        rng.random((len(run_ids), len(region_ids), len(dates))).astype(np.float32)
        * 10.0,
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
    )
    latent_h = xr.DataArray(
        rng.random((len(run_ids), len(region_ids), len(dates))).astype(np.float32)
        * 5.0,
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
    )
    latent_r = xr.DataArray(
        rng.random((len(run_ids), len(region_ids), len(dates))).astype(np.float32)
        * population.values[:, :, None]
        * 0.2,
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
    )
    death_flow_values = rng.random((len(run_ids), len(region_ids), len(dates))).astype(
        np.float32
    )
    death_flow_values[:, :, 0] = 0.0
    latent_d_values = np.cumsum(death_flow_values, axis=2, dtype=np.float32)
    latent_d = xr.DataArray(
        latent_d_values,
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
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
            rng.random((len(run_ids), len(dates), len(edar_ids))),
            dims=("run_id", "date", "edar_id"),
            coords={"run_id": run_ids, "date": dates, "edar_id": edar_ids},
        )
        edar_vars[f"edar_biomarker_{variant}"] = biomarker
        edar_vars[f"limit_of_detection_{variant}"] = lod

    data_vars = {
        "cases": cases,
        "cases_mask": cases_mask,
        "mobility_base": mobility_base,
        "mobility_kappa0": mobility_kappa0,
        "population": population,
        "latent_S_true": latent_s,
        "latent_I_true": latent_i,
        latent_h_name: latent_h,
        "latent_R_true": latent_r,
        "latent_D_true": latent_d,
        **edar_vars,
    }
    if include_deaths_true:
        data_vars["deaths_true"] = xr.DataArray(
            death_flow_values + death_flow_offset,
            dims=("run_id", "region_id", "date"),
            coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
        )

    ds = xr.Dataset(data_vars)

    ds.to_zarr(path, mode="w", zarr_format=2)


def _write_region_metadata(path, *, region_ids, edar_ids):
    data = np.ones((len(edar_ids), len(region_ids)), dtype=np.float32)
    emap = xr.DataArray(
        data,
        dims=("edar_id", "home"),
        coords={"edar_id": edar_ids, "home": region_ids},
    )
    emap.to_netcdf(path)


def _make_config(
    tmp_path,
    synthetic_path,
    *,
    include_temporal_covariates: bool = True,
    holiday_calendar_path=None,
):
    return make_preprocessing_config(
        tmp_path,
        synthetic_path=synthetic_path,
        include_temporal_covariates=include_temporal_covariates,
        holiday_calendar_path=holiday_calendar_path,
        options={"run_id_chunk_size": 1, "chunk_size": 10},
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
    latent_sird = result["latent_sird"]

    assert cases.dims == ("run_id", "date", "region_id")
    assert mobility.dims == ("run_id", "date", "origin", "destination")
    assert population.dims == ("run_id", "region_id")
    assert latent_sird["latent_S_true"].dims == ("run_id", "date", "region_id")
    assert latent_sird["latent_H_true"].dims == ("run_id", "date", "region_id")
    assert latent_sird["latent_death_flow_true"].dims == (
        "run_id",
        "date",
        "region_id",
    )
    assert latent_sird["latent_death_flow_true"].dtype == np.float32
    assert "latent_S_true_mask" not in latent_sird
    assert "latent_H_true_mask" not in latent_sird
    assert len(cases.run_id) == 2


@pytest.mark.epiforecaster
def test_synthetic_processor_normalizes_continuous_death_flow(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a"], dtype="U50")
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

    result = SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
        str(synthetic_path)
    )
    raw = xr.open_zarr(synthetic_path)
    try:
        expected = (
            raw["deaths_true"].transpose("run_id", "date", "region_id")
            / raw["population"]
        ).astype(np.float32)
        actual = result["latent_sird"]["latent_death_flow_true"]
        np.testing.assert_allclose(actual.values, expected.values)
        assert np.any(actual.values > 0)
    finally:
        raw.close()


@pytest.mark.epiforecaster
def test_synthetic_processor_requires_death_flow_for_latent_bundle(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        include_deaths_true=False,
    )

    with pytest.raises(ValueError, match="latent S/I/R/D supervision bundle"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("death_flow_offset", "message"),
    [(-2.0, "nonnegative"), (np.nan, "finite")],
)
def test_synthetic_processor_rejects_invalid_death_flow(
    tmp_path, death_flow_offset, message
):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        death_flow_offset=death_flow_offset,
    )

    with pytest.raises(ValueError, match=message):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
def test_synthetic_processor_rejects_misaligned_death_flow(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        death_flow_offset=0.25,
    )

    with pytest.raises(ValueError, match="destination-dated"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
def test_synthetic_processor_aliases_latent_hospitalized_target(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")
    edar_ids = np.array(["e1"], dtype="U10")
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=run_ids,
        dates=dates,
        region_ids=region_ids,
        edar_ids=edar_ids,
        latent_h_name="latent_hospitalized_true",
    )

    config = _make_config(tmp_path, synthetic_path)
    processor = SyntheticProcessor(config)

    result = processor.process(str(synthetic_path))

    latent_sird = result["latent_sird"]
    assert "latent_H_true" in latent_sird
    assert "latent_hospitalized_true" not in latent_sird
    assert "latent_H_true_mask" not in latent_sird


@pytest.mark.epiforecaster
def test_synthetic_processor_resolves_nested_zarr_wrapper(tmp_path):
    wrapper_path = tmp_path / "raw_synth_wrapper.zarr"
    nested_path = wrapper_path / "reduced_raw_synthetic_observations.zarr"
    run_ids = np.array(["synth_a"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")
    edar_ids = np.array(["e1"], dtype="U10")
    nested_path.parent.mkdir(parents=True)
    _write_minimal_synthetic_zarr(
        nested_path,
        run_ids=run_ids,
        dates=dates,
        region_ids=region_ids,
        edar_ids=edar_ids,
    )

    config = _make_config(tmp_path, wrapper_path)
    processor = SyntheticProcessor(config)

    result = processor.process(str(wrapper_path))
    assert "latent_sird" in result
    assert result["cases"]["cases"].shape[0] == 1


@pytest.mark.epiforecaster
def test_synthetic_processor_includes_temporal_covariates(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a", "synth_b"], dtype="U50")
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
    result = processor.process(str(synthetic_path))

    temporal_covariates = result["temporal_covariates"]
    assert temporal_covariates.dims == ("date", "covariate")
    assert list(temporal_covariates["covariate"].values) == [
        "dow_sin",
        "dow_cos",
        "is_holiday",
    ]
    np.testing.assert_array_equal(temporal_covariates["date"].values, dates.values)


@pytest.mark.epiforecaster
def test_synthetic_kappa0_lockdown_buckets_include_zero_and_upper_bounds(tmp_path):
    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    config.temporal_covariates.include_lockdown_severity = True
    processor = SyntheticProcessor(config)

    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    config.end_date = dates[-1].to_pydatetime()
    base_covariates = xr.DataArray(
        np.zeros((6, 3), dtype=np.float32),
        dims=("date", "covariate"),
        coords={
            "date": dates,
            "covariate": ["dow_sin", "dow_cos", "is_holiday"],
        },
    )
    raw_ds = xr.Dataset(
        {
            "mobility_kappa0": xr.DataArray(
                np.array([[0.0, 0.1, 0.2, 0.3, 0.5, 0.6]], dtype=np.float32),
                dims=("run_id", "date"),
                coords={"run_id": ["synth_a"], "date": dates},
            )
        }
    )

    result = processor._inject_kappa0_lockdown(base_covariates, raw_ds)

    np.testing.assert_array_equal(
        result.sel(covariate="lockdown_severity").values,
        np.array([[0.0, 1.0, 1.0, 2.0, 2.0, 3.0]], dtype=np.float32),
    )


@pytest.mark.epiforecaster
def test_synthetic_processor_requires_temporal_covariates_config(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    run_ids = np.array(["synth_a"], dtype="U50")
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

    config = _make_config(
        tmp_path,
        synthetic_path,
        include_temporal_covariates=False,
    )
    processor = SyntheticProcessor(config)

    with pytest.raises(
        ValueError,
        match="Synthetic preprocessing requires temporal_covariates configuration",
    ):
        processor.process(str(synthetic_path))


@pytest.mark.epiforecaster
def test_synthetic_preprocess_missing_holiday_file_fails(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    missing_holiday_file = tmp_path / "missing_holidays.csv"
    with pytest.raises(ValueError, match="Holiday calendar file does not exist"):
        _make_config(
            tmp_path,
            synthetic_path,
            holiday_calendar_path=missing_holiday_file,
        )


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
def test_hospitalizations_prefers_observed_over_true(tmp_path):
    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    processor = SyntheticProcessor(config)

    run_ids = np.array(["synth_a"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")

    observed = xr.DataArray(
        np.full((1, 4, 2), 999, dtype=np.float32),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )
    truth = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(1, 2, 4),
        dims=("run_id", "region_id", "date"),
        coords={"run_id": run_ids, "region_id": region_ids, "date": dates},
    )
    ds = xr.Dataset({"hospitalizations": observed, "hospitalizations_true": truth})

    extracted = processor._extract_hospitalizations(ds)

    assert extracted is not None
    expected = observed.values
    np.testing.assert_array_equal(extracted["hospitalizations"].values, expected)


@pytest.mark.epiforecaster
def test_deaths_prefers_raw_observed_channel(tmp_path):
    config = _make_config(tmp_path, synthetic_path=tmp_path / "raw_synth.zarr")
    processor = SyntheticProcessor(config)

    run_ids = np.array(["synth_a"], dtype="U50")
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    region_ids = np.array(["r1", "r2"], dtype="U10")

    observed = xr.DataArray(
        np.full((1, 4, 2), 999, dtype=np.float32),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )
    observed_raw = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(1, 4, 2),
        dims=("run_id", "date", "region_id"),
        coords={"run_id": run_ids, "date": dates, "region_id": region_ids},
    )
    ds = xr.Dataset({"deaths": observed, "deaths_raw": observed_raw})

    extracted = processor._extract_deaths(ds)

    assert extracted is not None
    expected = observed_raw.values
    np.testing.assert_array_equal(extracted["deaths"].values, expected)


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
    config.continuous_storage_dtype = "float32"

    pipeline = OfflinePreprocessingPipeline(config)
    output_path = pipeline.run()

    ds = xr.open_zarr(output_path)
    assert "run_id" in ds.dims
    assert "mobility" in ds
    assert "temporal_covariates" in ds
    assert "latent_S_true" in ds
    assert "latent_I_true" in ds
    assert "latent_H_true" in ds
    assert "latent_R_true" in ds
    assert "latent_D_true" in ds
    assert "latent_S_true_mask" not in ds
    assert "latent_H_true_mask" not in ds
    assert ds["temporal_covariates"].dims == ("date", "covariate")
    assert list(ds["temporal_covariates"]["covariate"].values) == [
        "dow_sin",
        "dow_cos",
        "is_holiday",
    ]
    assert ds["mobility"].dims[0] == "run_id"
    assert len(ds["run_id"]) == len(run_ids)
    case_values = ds["cases"].values
    case_mask = ds["cases_mask"].values
    assert case_values.dtype == np.float32
    assert np.all(case_values[case_mask] > 0)
    preservation = json.loads(ds.attrs["positive_clinical_preservation_json"])
    assert preservation["cases"]["raw_positive_count"] > 0
    assert preservation["cases"]["zeroed_raw_positive_count"] == 0
    ds.close()


@pytest.mark.epiforecaster
def test_positive_clinical_preservation_rejects_zeroed_positive_value():
    raw = xr.Dataset({"cases": xr.DataArray([0.25, np.nan], dims=("date",))})
    processed = xr.Dataset({"cases": xr.DataArray([0.0, np.nan], dims=("date",))})

    with pytest.raises(ValueError, match="positive raw synthetic 'cases'"):
        OfflinePreprocessingPipeline._validate_positive_clinical_preservation(
            raw_aligned=raw,
            processed=processed,
        )


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
    # Values are log1p-transformed: log1p(1.0) = 0.693
    expected_log_value = np.log1p(1.0)
    assert float(saved_mob.min()) >= expected_log_value - 0.01
    assert float(saved_mob.max()) <= expected_log_value + 0.01

    saved.close()
