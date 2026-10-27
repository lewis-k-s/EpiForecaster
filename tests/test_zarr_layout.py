"""Packed sharded Zarr layout for processed datasets."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import xarray as xr

from data.epi_dataset import open_processed_dataset
from data.preprocess.zarr_layout import (
    LAYOUT_VERSION,
    MOBILITY_DATE_CHUNK,
    build_encoding,
    open_zarr_zip,
    write_dataset,
)
from scripts.workflow.run_config import validate_workflow_contract


def _dataset(runs: int = 4, dates: int = 70, regions: int = 6) -> xr.Dataset:
    rng = np.random.default_rng(7)
    mobility = rng.random((runs, dates, regions, regions), dtype=np.float32)
    mobility[0, 0, 0, 0] = np.nan
    region_ids = np.array([f"{i:05d}" for i in range(regions)])
    return xr.Dataset(
        {
            "mobility": (("run_id", "date", "origin", "destination"), mobility),
            "cases": (
                ("run_id", "date", "region_id"),
                rng.random((runs, dates, regions), dtype=np.float32),
            ),
            "cases_mask": (
                ("run_id", "date", "region_id"),
                (rng.random((runs, dates, regions)) > 0.5).astype(np.int8),
            ),
            "population": (
                ("run_id", "region_id"),
                rng.integers(1, 1000, (runs, regions)).astype(np.int32),
            ),
            "edar_has_source": (("region_id",), np.ones(regions, np.int8)),
        },
        coords={
            "run_id": np.array([f"{i}_Baseline" for i in range(runs)]),
            "date": np.arange(dates).astype("datetime64[D]").astype("datetime64[ns]"),
            "region_id": region_ids,
            "origin": region_ids,
            "destination": region_ids,
        },
        attrs={"preprocessing_config_yaml": "dataset_name: tiny\n"},
    )


def _assert_same_values(actual: xr.Dataset, expected: xr.Dataset) -> None:
    for name, var in expected.variables.items():
        assert np.array_equal(
            actual[name].values,
            var.values,
            equal_nan=var.dtype.kind == "f",
        ), name
        if var.dtype.kind not in "UO":
            assert actual[name].dtype == var.dtype, name
    assert actual.attrs["preprocessing_config_yaml"] == "dataset_name: tiny\n"


@pytest.mark.parametrize("workers", [1, 3])
def test_write_dataset_round_trips_values_with_any_worker_count(tmp_path, workers):
    source = _dataset()
    destination = tmp_path / "tiny.zarr.zip"

    results = write_dataset(
        source.chunk({"run_id": 1, "date": 30}), destination, workers=workers
    )

    saved = open_zarr_zip(destination)
    _assert_same_values(saved, source)
    assert [result.run_id for result in results] == list(source.run_id.values)
    assert results[0].non_null["mobility"] == source.mobility[0].size - 1
    assert saved.attrs["zarr_layout_version"] == LAYOUT_VERSION
    # The staging directory is removed after packing.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tiny.zarr.zip"]


def test_parallel_and_serial_outputs_hold_the_same_members(tmp_path):
    source = _dataset().chunk({"run_id": 1})
    serial, parallel = tmp_path / "s.zarr.zip", tmp_path / "p.zarr.zip"
    write_dataset(source, serial, workers=1)
    write_dataset(source, parallel, workers=2)

    with zipfile.ZipFile(serial) as left, zipfile.ZipFile(parallel) as right:
        assert left.namelist()[0] == "zarr.json"
        assert sorted(left.namelist()) == sorted(right.namelist())
        assert all(info.compress_type == zipfile.ZIP_STORED for info in left.infolist())
        shards = [name for name in left.namelist() if name.startswith("mobility/c/")]
        for name in shards:
            assert left.read(name) == right.read(name), name
    # One shard per run for each per-run variable.
    assert len(shards) == 4


def test_encoding_uses_one_shard_per_run_and_whole_inner_chunks():
    dataset = _dataset(dates=70)
    encoding = build_encoding(dataset)

    assert encoding["mobility"]["chunks"] == (1, MOBILITY_DATE_CHUNK, 6, 6)
    # 70 dates need two 65-date inner chunks; the shard holds both.
    assert encoding["mobility"]["shards"] == (1, 2 * MOBILITY_DATE_CHUNK, 6, 6)
    assert encoding["cases"]["shards"] == (1, 70, 6)
    assert encoding["cases"]["chunks"] == (1, 70, 6)
    assert "shards" not in encoding["population"]
    assert encoding["population"]["chunks"] == (4, 6)


def test_fixed_width_strings_are_stored_as_variable_length(tmp_path):
    destination = tmp_path / "tiny.zarr.zip"
    write_dataset(_dataset(), destination, workers=1)

    saved = open_zarr_zip(destination)
    assert saved.run_id.dtype == object
    assert list(saved.region_id.values) == [f"{i:05d}" for i in range(6)]


def test_write_dataset_requires_packed_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.zarr\.zip"):
        write_dataset(_dataset(), tmp_path / "tiny.zarr", workers=1)


def test_open_processed_dataset_dispatches_packed_zarr(tmp_path):
    destination = tmp_path / "tiny.zarr.zip"
    write_dataset(_dataset(), destination, workers=1)

    saved = open_processed_dataset(destination)
    assert saved.mobility.encoding["shards"] == (1, 130, 6, 6)


def test_convert_netcdf_keeps_values_and_records_source(tmp_path):
    source = _dataset()
    netcdf = tmp_path / "tiny.nc"
    source.to_netcdf(netcdf, engine="netcdf4")
    destination = tmp_path / "tiny.zarr.zip"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "data.preprocess.convert_netcdf",
            str(netcdf),
            str(destination),
            "--workers",
            "2",
            "--source-dvc-md5",
            "abc123",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
    )

    saved = open_zarr_zip(destination)
    _assert_same_values(saved, xr.open_dataset(netcdf, engine="h5netcdf"))
    assert saved.attrs["converted_from"] == "tiny.nc"
    assert saved.attrs["converted_from_dvc_md5"] == "abc123"


def _preprocess_workflow(write_workers: object) -> dict:
    return {
        "workflow": {"kind": "preprocess"},
        "run": {},
        "source": {},
        "container": {},
        "dvc": {"targets": [], "provenance_targets": []},
        "preprocessing": {
            "base_config": "configs/preprocess.yaml",
            "dataset_name": "tiny",
            "write_workers": write_workers,
        },
        "resources": {
            "allocation": {
                "account": "bsc08",
                "partition": "gpp",
                "qos": "gp_bscls",
                "nodes": 1,
                "tasks": 1,
                "cpus_per_task": 16,
                "gres": "none",
                "runtime": 60,
            }
        },
    }


@pytest.mark.parametrize("value", [0, 17, True, "4"])
def test_workflow_rejects_invalid_write_workers(value):
    with pytest.raises(ValueError, match="write_workers"):
        validate_workflow_contract(_preprocess_workflow(value))
