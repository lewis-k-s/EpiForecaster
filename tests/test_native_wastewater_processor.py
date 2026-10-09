"""Tests for canonical EDAR-level wastewater observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.processors.native_wastewater_processor import (
    SCHEMA,
    build_native_wastewater_observations,
    write_native_wastewater_observations,
)

pytestmark = pytest.mark.epiforecaster


def _write_wastewater(path: Path) -> None:
    pd.DataFrame(
        {
            "id mostra": [
                "E001-2020-03-03",
                "E001-2020-03-05",
                "E002-2020-03-04",
            ],
            "depuradora": ["Plant 1", "Plant 1", "Plant 2"],
            "LD(CG/L)": [10.0, 10.0, 5.0],
            "N1(CG/L)": [10.0, 30.0, 7.0],
            "N2(CG/L)": [20.0, np.nan, 9.0],
            "IP4(CG/L)": [np.nan, 8.0, 11.0],
            "Cabal últimes 24h(m3)": [100.0, 120.0, 80.0],
            "Pluja(mm)": [0.0, 2.0, 1.0],
        }
    ).to_csv(path, index=False)


def test_site_level_store_retains_assays_censoring_and_flow(tmp_path: Path) -> None:
    path = tmp_path / "wastewater.csv"
    _write_wastewater(path)

    dataset = build_native_wastewater_observations(
        path, start_date="2020-03-01", end_date="2020-03-06"
    )

    assert dataset.attrs["schema"] == SCHEMA
    assert dataset["assay"].values.tolist() == ["N1", "N2", "IP4"]
    assert dataset["edar_id"].values.tolist() == ["E001", "E002"]
    assert (
        dataset["concentration_raw"]
        .sel(date="2020-03-03", edar_id="E001", assay="N1")
        .item()
        == 10.0
    )
    assert (
        dataset["censor_code"].sel(date="2020-03-03", edar_id="E001", assay="N1").item()
        == 1
    )
    assert (
        dataset["censor_code"].sel(date="2020-03-03", edar_id="E001", assay="N2").item()
        == 0
    )
    assert (
        dataset["censor_code"]
        .sel(date="2020-03-03", edar_id="E001", assay="IP4")
        .item()
        == 2
    )
    assert dataset["flow_m3_24h"].sel(date="2020-03-03", edar_id="E001").item() == 100.0
    assert dataset.attrs["availability_date_status"] == "not_provided_by_source"


def test_causal_history_is_absent_before_first_sample(tmp_path: Path) -> None:
    path = tmp_path / "wastewater.csv"
    _write_wastewater(path)
    dataset = build_native_wastewater_observations(
        path, start_date="2020-03-01", end_date="2020-03-06"
    )

    before = dataset["history_log1p"].sel(
        date=slice("2020-03-01", "2020-03-02"), edar_id="E001", assay="N1"
    )
    assert np.isnan(before.values).all()
    assert (
        not dataset["history_available_mask"]
        .sel(date=slice("2020-03-01", "2020-03-02"), edar_id="E001", assay="N1")
        .values.any()
    )
    assert (
        dataset["history_available_mask"]
        .sel(date="2020-03-03", edar_id="E001", assay="N1")
        .item()
    )
    assert np.isnan(
        dataset["history_age_days"]
        .sel(date="2020-03-02", edar_id="E001", assay="N1")
        .item()
    )
    assert (
        dataset["history_age_days"]
        .sel(date="2020-03-04", edar_id="E001", assay="N1")
        .item()
        == 1.0
    )


def test_all_censored_plant_samples_remain_observed(tmp_path: Path) -> None:
    path = tmp_path / "wastewater.csv"
    _write_wastewater(path)
    frame = pd.read_csv(path)
    frame[["N1(CG/L)", "N2(CG/L)", "IP4(CG/L)"]] = 0.0
    frame.to_csv(path, index=False)
    dataset = build_native_wastewater_observations(
        path, start_date="2020-03-01", end_date="2020-03-06"
    )
    observed = dataset["observed_mask"].values
    available = dataset["history_available_mask"].values
    assert observed.sum() == 9
    assert (dataset["censor_code"].values[observed] == 1).all()
    assert np.isfinite(dataset["history_log1p"].values[available]).all()
    assert not available[:2].any()


def test_zarr_round_trip_uses_xarray_contract(tmp_path: Path) -> None:
    path = tmp_path / "wastewater.csv"
    _write_wastewater(path)
    dataset = build_native_wastewater_observations(
        path, start_date="2020-03-01", end_date="2020-03-06"
    )
    output = tmp_path / "obs_wastewater.zarr"

    write_native_wastewater_observations(dataset, output)
    restored = xr.open_zarr(output)

    assert restored.attrs["schema"] == SCHEMA
    assert restored.sizes == dataset.sizes
    xr.testing.assert_equal(restored["observed_mask"], dataset["observed_mask"])
