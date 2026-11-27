"""Packed sharded Zarr layout for processed datasets."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import xarray as xr

from data.preprocess.zarr_layout import (
    LAYOUT_VERSION,
    MOBILITY_DATE_CHUNK,
    bitround,
    build_encoding,
    open_zarr_store,
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


def test_write_dataset_round_trips_values(tmp_path):
    source = _dataset()
    destination = tmp_path / "tiny.zarr.zip"

    results = write_dataset(source.chunk({"run_id": 1, "date": 30}), destination)

    saved = open_zarr_zip(destination)
    _assert_same_values(saved, source)
    assert [result.run_id for result in results] == list(source.run_id.values)
    assert results[0].non_null["mobility"] == source.mobility[0].size - 1
    assert saved.attrs["zarr_layout_version"] == LAYOUT_VERSION
    # The staging directory is removed after packing.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tiny.zarr.zip"]


def test_large_run_variables_are_verified_in_date_slabs(tmp_path, monkeypatch):
    import data.preprocess.zarr_layout as zarr_layout

    # Treat every per-run variable as large: computed into one array and
    # read back slab by slab across several inner mobility chunks.
    monkeypatch.setattr(zarr_layout, "LARGE_RUN_VARIABLE_BYTES", 0)
    source = _dataset(runs=2, dates=140)
    destination = tmp_path / "tiny.zarr.zip"

    results = write_dataset(source.chunk({"run_id": 1, "date": 30}), destination)

    _assert_same_values(open_zarr_zip(destination), source)
    assert results[0].non_null["mobility"] == source.mobility[0].size - 1


def test_write_run_rejects_a_dataset_with_several_runs(tmp_path):
    from data.preprocess.zarr_layout import create_store, prepare_for_store, write_run

    prepared = prepare_for_store(_dataset())
    store = create_store(prepared, tmp_path / "tiny.zarr.zip")

    with pytest.raises(ValueError, match="exactly one run"):
        write_run(store, 0, prepared.isel(run_id=slice(0, 2)))


def test_write_reads_a_packed_source_correctly(tmp_path):
    source = _dataset(runs=8, dates=130, regions=60)
    packed = tmp_path / "source.zarr.zip"
    write_dataset(source, packed)
    destination = tmp_path / "copy.zarr.zip"

    write_dataset(open_zarr_zip(packed, chunks={"run_id": 1}), destination)

    _assert_same_values(open_zarr_zip(destination), source)


def test_packed_store_holds_root_metadata_first_and_one_shard_per_run(tmp_path):
    destination = tmp_path / "s.zarr.zip"
    write_dataset(_dataset().chunk({"run_id": 1}), destination)

    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist()[0] == "zarr.json"
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert archive.testzip() is None
        shards = [name for name in archive.namelist() if name.startswith("mobility/c/")]
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
    write_dataset(_dataset(), destination)

    saved = open_zarr_zip(destination)
    assert saved.run_id.dtype == object
    assert list(saved.region_id.values) == [f"{i:05d}" for i in range(6)]


def test_directory_store_variable_length_strings_open_as_object(tmp_path):
    store = tmp_path / "ids.zarr"
    ids = np.array(["08001", "08002"], dtype=object)
    # A string data variable along the string index dimension, as in the EpiSim
    # corpus (run_id with per-run string metadata), must not revert the index.
    xr.Dataset(
        {
            "x": ("region_id", [1.0, 2.0]),
            "kind": ("region_id", np.array(["a", "b"], dtype=object)),
        },
        coords={"region_id": ids},
    ).to_zarr(store, zarr_format=3)

    opened = open_zarr_store(store, chunks={"region_id": 1})

    assert opened.region_id.dtype == object
    assert opened.kind.dtype == object
    assert opened.region_id.values.astype(str).tolist() == ["08001", "08002"]
    assert opened.region_id.values.astype("U").tolist() == ["08001", "08002"]


def test_write_dataset_requires_packed_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.zarr\.zip"):
        write_dataset(_dataset(), tmp_path / "tiny.zarr")


def test_open_zarr_store_dispatches_packed_zarr(tmp_path):
    destination = tmp_path / "tiny.zarr.zip"
    write_dataset(_dataset(), destination)

    saved = open_zarr_store(destination)
    assert saved.mobility.encoding["shards"] == (1, 130, 6, 6)


def test_bitround_keeps_the_requested_mantissa_bits():
    rng = np.random.default_rng(3)
    values = np.concatenate(
        [
            rng.random(10_000, dtype=np.float32),
            np.array([0.0, -0.0, 5.7e-20, 0.68, -1.5], np.float32),
            np.array([np.nan, np.inf, -np.inf], np.float32),
        ]
    )
    for keepbits in (7, 10, 12):
        rounded = bitround(values, keepbits)
        finite = np.isfinite(values)
        assert rounded.dtype == np.float32
        # Every dropped mantissa bit is zero.
        dropped = (1 << (23 - keepbits)) - 1
        assert not np.any(rounded[finite].view(np.uint32) & dropped)
        # Round to nearest: the error is at most half a quantum.
        nonzero = finite & (values != 0)
        relative = np.abs(rounded[nonzero] - values[nonzero]) / np.abs(values[nonzero])
        assert relative.max() <= 2.0 ** -(keepbits + 1)
        assert rounded[np.isnan(values)].size == 1
        assert np.isnan(rounded[np.isnan(values)]).all()
        assert np.array_equal(rounded[np.isinf(values)], values[np.isinf(values)])
        # The float32 exponent range is kept: small flows do not become zero.
        assert rounded[values == np.float32(5.7e-20)][0] > 0


def test_bitround_matches_float16_precision_at_ten_bits():
    # 10 kept bits equal the float16 mantissa, so a float16 cast after the
    # rounding gives the same values as a direct cast (normal float16 range).
    values = np.random.default_rng(5).uniform(1e-3, 0.7, 50_000).astype(np.float32)
    assert np.array_equal(
        bitround(values, 10).astype(np.float16), values.astype(np.float16)
    )


def test_bitround_rounds_half_to_even():
    one_plus_half_quantum = np.array([1.0 + 2.0**-11], np.float32)
    assert bitround(one_plus_half_quantum, 10)[0] == np.float32(1.0)
    # 1.5 quanta above 1.0: the tie goes to the even neighbour, 1 + 2 quanta.
    three_halves_up = np.array([1.0 + 3 * 2.0**-11], np.float32)
    assert bitround(three_halves_up, 10)[0] == np.float32(1.0 + 2.0**-9)


def test_write_dataset_applies_bitround_and_records_bits(tmp_path):
    source = _dataset()
    destination = tmp_path / "tiny.zarr.zip"

    write_dataset(
        source.chunk({"run_id": 1}),
        destination,
        bitround_keepbits={"mobility": 10},
    )

    saved = open_zarr_zip(destination)
    expected = bitround(source.mobility.values, 10)
    assert np.array_equal(saved.mobility.values, expected, equal_nan=True)
    assert np.isnan(saved.mobility.values[0, 0, 0, 0])
    assert saved.mobility.attrs["bitround_keepbits"] == 10
    assert json.loads(saved.attrs["zarr_bitround_keepbits_json"]) == {"mobility": 10}
    # Variables without rounding are unchanged.
    assert np.array_equal(saved.cases.values, source.cases.values)
    assert "bitround_keepbits" not in saved.cases.attrs


def test_write_dataset_rejects_bitround_on_non_float32(tmp_path):
    with pytest.raises(TypeError, match="population"):
        write_dataset(
            _dataset(),
            tmp_path / "tiny.zarr.zip",
            bitround_keepbits={"population": 10},
        )


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


@pytest.mark.parametrize("bits", [0, 23, True, "10"])
def test_preprocessing_config_rejects_invalid_bitround_bits(tmp_path, bits):
    from tests.dataset_test_helpers import make_preprocessing_config

    with pytest.raises(ValueError, match="bitround_keepbits"):
        make_preprocessing_config(
            tmp_path, options={"bitround_keepbits": {"mobility": bits}}
        )


def test_a97c812_preprocess_config_rounds_mobility_to_float16_precision():
    from data.preprocess.config import PreprocessingConfig

    loaded = PreprocessingConfig._load_config_dict(
        Path(__file__).resolve().parents[1]
        / "configs/production_only/preprocess_mn5_synth_sirhd_v1_a97c812_reactive_n100.yaml",
        seen_paths=set(),
    )
    assert loaded["bitround_keepbits"] == {"mobility": 10}
