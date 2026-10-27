import hashlib
import json

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import warnings
import xarray as xr

from data.epi_dataset import open_processed_dataset
from data.preprocess.processors.synthetic_processor import (
    SyntheticProcessor,
    _split_attr_list,
)
from data.preprocess.pipeline import OfflinePreprocessingPipeline
from data.preprocess.zarr_layout import pack_store, with_variable_length_strings
from tests.dataset_test_helpers import make_preprocessing_config


STATE_SPACE_CONTRACT = {
    "contract_schema_version": 1,
    "state_space": "SIRHD",
    "state_space_schema_version": "epiforecaster_sirhd_v1",
    "units": "persons",
    "denominator": "population[run_id, region_id]",
    "primitive_stock_variables": {"I": "latent_I_true", "S": "latent_S_true"},
    "audit_only_stock_variables": {"CH": "latent_CH_true"},
    "aggregation_map": {
        "sirhd_D": ["D"],
        "sirhd_H": ["HR", "HD"],
        "sirhd_I": ["E", "A", "I", "PH", "PD"],
        "sirhd_R": ["R"],
        "sirhd_S": ["S"],
    },
    "applicability": {"required_zero_stocks": ["CH"], "on_violation": "fail"},
    "timestamp_convention": {"flow": "destination-dated daily interval"},
}
STATE_SPACE_CONTRACT_JSON = json.dumps(
    STATE_SPACE_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
STATE_SPACE_CONTRACT_SHA256 = hashlib.sha256(
    STATE_SPACE_CONTRACT_JSON.encode()
).hexdigest()


def _write_minimal_synthetic_zarr(path, **kwargs):
    _minimal_synthetic_dataset(**kwargs).to_zarr(path, mode="w", zarr_format=2)


def _minimal_synthetic_dataset(
    *,
    run_ids,
    dates,
    region_ids,
    edar_ids,
    include_sirhd=True,
    include_death_flow=True,
    include_primitive_latents=True,
    death_flow_offset=0.0,
    partition_offset=0.0,
    confined_stock=0.0,
    contract_json=STATE_SPACE_CONTRACT_JSON,
    contract_sha256=STATE_SPACE_CONTRACT_SHA256,
):
    """Build a corpus that starts one day before ``dates``.

    The extra leading corpus day gives every requested processing date a
    destination-dated death-flow interval, as in the generator contract.
    """
    rng = np.random.default_rng(123)
    corpus_dates = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)]).append(dates)
    n_runs, n_dates, n_regions = len(run_ids), len(corpus_dates), len(region_ids)
    clinical_coords = {
        "run_id": run_ids,
        "date": corpus_dates,
        "region_id": region_ids,
    }

    cases = xr.DataArray(
        rng.random((n_runs, n_dates, n_regions)),
        dims=("run_id", "date", "region_id"),
        coords=clinical_coords,
    )

    # Required mask channel for clinical series preprocessing
    # cases_age is computed from mask by SyntheticProcessor
    cases_mask = xr.DataArray(
        rng.random((n_runs, n_dates, n_regions)) > 0.1,  # ~90% observed
        dims=("run_id", "date", "region_id"),
        coords=clinical_coords,
    )

    mobility_base = xr.DataArray(
        rng.random((n_regions, n_regions)),
        dims=("origin", "target"),
        coords={"origin": region_ids, "target": region_ids},
    )

    mobility_kappa0 = xr.DataArray(
        rng.random((n_runs, n_dates)),
        dims=("run_id", "date"),
        coords={"run_id": run_ids, "date": corpus_dates},
    )

    population_values = rng.integers(500, 1500, size=(n_runs, n_regions))
    population = xr.DataArray(
        population_values.astype(np.float32),
        dims=("run_id", "region_id"),
        coords={"run_id": run_ids, "region_id": region_ids},
    )

    # Exhaustive SIRHD partition in persons: S + I + H + R + D = population.
    shape = (n_runs, n_dates, n_regions)
    pop = population_values[:, None, :].astype(np.float64)
    daily_deaths = rng.random(shape)
    daily_deaths[:, 0, :] = 0.0
    stock_d = np.cumsum(daily_deaths, axis=1).astype(np.float32)
    stock_i = (rng.random(shape) * 10.0).astype(np.float32)
    stock_h = (rng.random(shape) * 5.0).astype(np.float32)
    stock_r = (rng.random(shape) * pop * 0.2).astype(np.float32)
    stock_s = (pop - stock_i - stock_h - stock_r - stock_d + partition_offset).astype(
        np.float32
    )

    edar_vars = {}
    variants = ["N1", "N2", "IP4"]
    for variant in variants:
        biomarker = xr.DataArray(
            rng.random((n_runs, n_dates, len(edar_ids))),
            dims=("run_id", "date", "edar_id"),
            coords={"run_id": run_ids, "date": corpus_dates, "edar_id": edar_ids},
        )
        lod = xr.DataArray(
            rng.random((n_runs, n_dates, len(edar_ids))),
            dims=("run_id", "date", "edar_id"),
            coords={"run_id": run_ids, "date": corpus_dates, "edar_id": edar_ids},
        )
        edar_vars[f"edar_biomarker_{variant}"] = biomarker
        edar_vars[f"limit_of_detection_{variant}"] = lod

    def stock(values):
        return xr.DataArray(
            values, dims=("run_id", "date", "region_id"), coords=clinical_coords
        )

    data_vars = {
        "cases": cases,
        "cases_mask": cases_mask,
        "mobility_base": mobility_base,
        "mobility_kappa0": mobility_kappa0,
        "population": population,
        **edar_vars,
    }
    if include_primitive_latents:
        # Primitive symptomatic I is a strict subset of sirhd_I.
        data_vars["latent_S_true"] = stock(stock_s)
        data_vars["latent_I_true"] = stock((stock_i * 0.5).astype(np.float32))
        data_vars["latent_CH_true"] = stock(
            np.full(shape, confined_stock, dtype=np.float32)
        )
    if include_sirhd:
        data_vars.update(
            {
                "sirhd_S": stock(stock_s),
                "sirhd_I": stock(stock_i),
                "sirhd_H": stock(stock_h),
                "sirhd_R": stock(stock_r),
                "sirhd_D": stock(stock_d),
            }
        )
    if include_sirhd and include_death_flow:
        data_vars["sirhd_death_flow"] = xr.DataArray(
            np.diff(stock_d, axis=1) + np.float32(death_flow_offset),
            dims=("run_id", "death_interval_date", "region_id"),
            coords={
                "run_id": run_ids,
                "death_interval_date": corpus_dates[1:],
                "region_id": region_ids,
            },
        )

    ds = xr.Dataset(data_vars)
    if include_sirhd:
        ds.attrs.update(
            {
                "state_space_schema_version": "epiforecaster_sirhd_v1",
                "state_space_contract_json": contract_json,
                "state_space_contract_sha256": contract_sha256,
            }
        )
    return ds


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
    options=None,
):
    return make_preprocessing_config(
        tmp_path,
        synthetic_path=synthetic_path,
        include_temporal_covariates=include_temporal_covariates,
        holiday_calendar_path=holiday_calendar_path,
        options={
            "run_id_chunk_size": 1,
            "chunk_size": 10,
            "synthetic_state_space_contract_sha256": STATE_SPACE_CONTRACT_SHA256,
            **(options or {}),
        },
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
def test_synthetic_processor_reads_packed_zarr_corpus_like_zarr_directory(tmp_path):
    kwargs = dict(
        run_ids=np.array(["synth_a", "synth_b"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1", "r2", "r3"], dtype="U10"),
        edar_ids=np.array(["e1", "e2"], dtype="U10"),
    )
    directory = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(directory, **kwargs)
    # The EpiSim raw layout: Zarr v3, variable-length strings, packed ZIP64.
    staging = tmp_path / "raw_synth_v3.zarr"
    with_variable_length_strings(_minimal_synthetic_dataset(**kwargs)).to_zarr(
        staging, mode="w", zarr_format=3, consolidated=False
    )
    packed = tmp_path / "raw_synth.zarr.zip"
    pack_store(staging, packed)

    config = _make_config(tmp_path, directory)
    processor = SyntheticProcessor(config)
    assert processor._resolve_synthetic_store_path(str(packed)) == str(packed)
    expected = processor.process(str(directory))
    actual = SyntheticProcessor(_make_config(tmp_path, packed)).process(str(packed))

    assert list(actual["cases"].run_id.values) == ["synth_a", "synth_b"]
    for group in ("cases", "mobility", "latent_sird"):
        for name, var in expected[group].data_vars.items():
            assert np.array_equal(
                actual[group][name].values, var.values, equal_nan=True
            ), (group, name)
    assert np.array_equal(actual["population"].values, expected["population"].values)


@pytest.mark.epiforecaster
def test_synthetic_processor_opens_zarr_directory_with_nc_suffix(tmp_path):
    synthetic_path = tmp_path / "raw_synth.nc"
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

    dataset = SyntheticProcessor(
        _make_config(tmp_path, synthetic_path)
    )._open_synthetic_data(str(synthetic_path))
    try:
        assert dataset.sizes["run_id"] == 1
    finally:
        dataset.close()


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
            raw["sirhd_death_flow"]
            .rename({"death_interval_date": "date"})
            .sel(date=dates)
            / raw["population"]
        ).astype(np.float32)
        actual = result["latent_sird"]["latent_death_flow_true"]
        np.testing.assert_array_equal(actual["date"].values, dates.values)
        np.testing.assert_allclose(
            actual.values, expected.transpose("run_id", "date", "region_id").values
        )
        assert np.any(actual.values > 0)
    finally:
        raw.close()


@pytest.mark.epiforecaster
def test_synthetic_processor_maps_latents_to_sirhd_partition(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a", "synth_b"], dtype="U50"),
        dates=dates,
        region_ids=np.array(["r1", "r2", "r3"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
    )

    latent = SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
        str(synthetic_path)
    )["latent_sird"]
    raw = xr.open_zarr(synthetic_path)
    try:
        population = raw["population"].astype(np.float32)
        expected_i = (raw["sirhd_I"].sel(date=dates) / population).astype(np.float32)
        primitive_i = (raw["latent_I_true"].sel(date=dates) / population).astype(
            np.float32
        )
        np.testing.assert_allclose(latent["latent_I_true"].values, expected_i.values)
        assert not np.allclose(latent["latent_I_true"].values, primitive_i.values)
    finally:
        raw.close()

    total = sum(
        latent[name].values.astype(np.float64)
        for name in (
            "latent_S_true",
            "latent_I_true",
            "latent_H_true",
            "latent_R_true",
            "latent_D_true",
        )
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-5)
    assert set(latent.data_vars) == {
        "latent_S_true",
        "latent_I_true",
        "latent_H_true",
        "latent_R_true",
        "latent_D_true",
        "latent_death_flow_true",
    }


@pytest.mark.epiforecaster
def test_synthetic_processor_requires_death_flow_for_latent_bundle(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        include_death_flow=False,
    )

    with pytest.raises(ValueError, match="latent SIRHD supervision bundle"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
def test_synthetic_processor_rejects_primitive_only_latents(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        include_sirhd=False,
    )

    with pytest.raises(ValueError, match="Primitive latent stocks"):
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
def test_synthetic_processor_rejects_date_without_death_interval(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-02", periods=3, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
    )

    # The corpus starts on 2020-01-01, which has no incoming death interval.
    with pytest.raises(ValueError, match="no destination-dated interval"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
def test_synthetic_processor_rejects_broken_partition(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        partition_offset=-50.0,
    )

    with pytest.raises(ValueError, match="partition the population"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
def test_synthetic_processor_rejects_nonzero_confined_stock(tmp_path):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        confined_stock=1.0,
    )

    with pytest.raises(ValueError, match="requires zero 'latent_CH_true'"):
        SyntheticProcessor(_make_config(tmp_path, synthetic_path)).process(
            str(synthetic_path)
        )


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("writer_options", "config_options", "message"),
    [
        ({}, {"synthetic_state_space_contract_sha256": None}, "requires"),
        ({}, {"synthetic_state_space_contract_sha256": "0" * 64}, "differs from"),
        (
            {"contract_json": STATE_SPACE_CONTRACT_JSON.replace("SIRHD", "SIR")},
            {},
            "does not match its recorded",
        ),
        (
            {},
            {"synthetic_state_space_schema_version": "epiforecaster_sirhd_v2"},
            "schema versions",
        ),
    ],
)
def test_synthetic_processor_enforces_state_space_contract(
    tmp_path, writer_options, config_options, message
):
    synthetic_path = tmp_path / "raw_synth.zarr"
    _write_minimal_synthetic_zarr(
        synthetic_path,
        run_ids=np.array(["synth_a"], dtype="U50"),
        dates=pd.date_range("2020-01-01", periods=4, freq="D"),
        region_ids=np.array(["r1"], dtype="U10"),
        edar_ids=np.array(["e1"], dtype="U10"),
        **writer_options,
    )

    config = _make_config(tmp_path, synthetic_path, options=config_options)
    with pytest.raises(ValueError, match=message):
        SyntheticProcessor(config).process(str(synthetic_path))


@pytest.mark.epiforecaster
def test_preprocessing_config_rejects_malformed_contract_sha256(tmp_path):
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        _make_config(
            tmp_path,
            tmp_path / "raw_synth.zarr",
            options={"synthetic_state_space_contract_sha256": "not-a-digest"},
        )


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
    saved = open_processed_dataset(output_path)

    # One shard per run, whatever the in-memory run_id chunking.
    shards = saved["edar_biomarker_N1_mask"].encoding.get("shards")
    assert shards == (1, len(dates), len(region_ids))

    saved.close()


@pytest.mark.epiforecaster
@pytest.mark.integration
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

    ds = open_processed_dataset(output_path)
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
    saved = open_processed_dataset(output_path)

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


def test_split_attr_list_accepts_netcdf_strings_and_zarr_lists():
    assert _split_attr_list("cases, deaths") == ["cases", "deaths"]
    assert _split_attr_list(["cases", " deaths"]) == ["cases", "deaths"]
    assert _split_attr_list("S") == _split_attr_list(["S"]) == ["S"]
    assert _split_attr_list(None) == []
