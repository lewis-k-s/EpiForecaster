"""Packing a Zarr directory store into one ``*.zarr.zip`` file."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import zipfile

from click.testing import CliRunner
import numpy as np
import pytest
import xarray as xr

from data.preprocess.processors.mobility_processor import MobilityProcessor
from data.preprocess.zarr_layout import open_zarr_store, open_zarr_zip
from scripts.data.pack_zarr_store import (
    VerificationError,
    cli,
    pack_and_verify,
    verify_datasets,
)


def _mobility_dataset() -> xr.Dataset:
    rng = np.random.default_rng(3)
    dates = np.datetime64("2020-03-01", "ns") + np.arange(10) * np.timedelta64(1, "D")
    ids = np.array([f"{i:05d}" for i in range(4)])
    mobility = rng.random((10, 4, 4)).astype(np.float32)
    mobility[2, 1, 1] = np.nan
    return xr.Dataset(
        {
            "mobility": (("date", "origin", "target"), mobility),
            "population": (("origin",), np.arange(4, dtype=np.int32)),
        },
        coords={"date": dates, "origin": ids, "target": ids},
        attrs={"title": "tiny"},
    )


def _write_directory(path: Path, ds: xr.Dataset | None = None) -> xr.Dataset:
    ds = ds or _mobility_dataset()
    ds["mobility"].attrs["units"] = "trips"
    ds.to_zarr(
        path,
        mode="w",
        zarr_format=2,
        consolidated=True,
        encoding={"mobility": {"chunks": (3, 4, 2)}},
    )
    return ds


def test_pack_round_trip_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "mobility.zarr"
    _write_directory(source)
    destination = tmp_path / "mobility.zarr.zip"

    summary = pack_and_verify(source, destination)

    assert destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert len(str(summary["md5"])) == 32
    with zipfile.ZipFile(destination) as zf:
        assert {i.compress_type for i in zf.infolist()} == {zipfile.ZIP_STORED}
    with xr.open_zarr(source) as expected, open_zarr_zip(destination) as actual:
        assert actual["mobility"].encoding["chunks"] == (3, 4, 2)
        assert actual["mobility"].attrs["units"] == "trips"
        xr.testing.assert_identical(expected, actual)


def test_cli_refuses_overwrite_and_reports_md5(tmp_path: Path) -> None:
    source = tmp_path / "mobility.zarr"
    _write_directory(source)
    destination = tmp_path / "mobility.zarr.zip"
    runner = CliRunner()

    first = runner.invoke(cli, ["pack", str(source), str(destination)])
    assert first.exit_code == 0, first.output
    assert "md5=" in first.output
    before = destination.read_bytes()

    second = runner.invoke(cli, ["pack", str(source), str(destination)])
    assert second.exit_code != 0
    assert destination.read_bytes() == before


def test_verification_fails_on_mismatched_archive(tmp_path: Path) -> None:
    source = tmp_path / "mobility.zarr"
    ds = _write_directory(source)
    other = tmp_path / "other.zarr"
    changed = ds.copy(deep=True)
    changed["mobility"][0, 0, 0] = 99.0
    _write_directory(other, changed)
    archive = tmp_path / "other.zarr.zip"
    pack_and_verify(other, archive)

    with pytest.raises(VerificationError):
        verify_datasets(source, archive)


def test_pack_fails_when_source_changes_during_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.data.pack_zarr_store as module

    source = tmp_path / "mobility.zarr"
    _write_directory(source)
    real_pack = module.pack_store

    def corrupting_pack(store: Path, destination: Path) -> None:
        real_pack(store, destination)
        chunk = next(p for p in sorted(store.rglob("*")) if p.name.startswith("0."))
        chunk.write_bytes(chunk.read_bytes()[:-1] + b"\x00")

    monkeypatch.setattr(module, "pack_store", corrupting_pack)
    destination = tmp_path / "mobility.zarr.zip"
    with pytest.raises(VerificationError):
        pack_and_verify(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_mobility_processor_reads_zip_like_directory(tmp_path: Path) -> None:
    source = tmp_path / "mobility.zarr"
    _write_directory(source)
    archive = tmp_path / "mobility.zarr.zip"
    pack_and_verify(source, archive)
    config = SimpleNamespace(
        run_id_chunk_size=1, start_date="2020-01-01", end_date="2020-12-31"
    )
    processor = MobilityProcessor(config)  # type: ignore[arg-type]

    from_dir = processor.process(str(source))
    from_zip = processor.process(str(archive))

    assert np.array_equal(
        from_dir["mobility"].values, from_zip["mobility"].values, equal_nan=True
    )
    assert from_zip["mobility"].dims == from_dir["mobility"].dims


def test_open_zarr_store_dispatches_on_suffix(tmp_path: Path) -> None:
    source = tmp_path / "mobility.zarr"
    _write_directory(source)
    archive = tmp_path / "mobility.zarr.zip"
    pack_and_verify(source, archive)
    xr.testing.assert_identical(open_zarr_store(source), open_zarr_store(archive))
