"""Sharded Zarr v3 layout for processed datasets, packed into one ZIP file.

Each variable with ``run_id`` and ``date`` dimensions gets one shard per run,
so a worker that writes run ``i`` is the only writer of every shard it
touches. The directory store is packed into one uncompressed ZIP64 file, which
DVC tracks as one object. See docs/SYNTHETIC_ZARR_LAYOUT.md.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import shutil
from typing import Any
import zipfile

import dask
import numpy as np
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

LAYOUT_VERSION = "synthetic-zarr-v1"
ZARR_ZIP_SUFFIX = ".zarr.zip"
RUN_DIM = "run_id"
DATE_DIM = "date"
MOBILITY_DIMS = ("origin", "destination")
# Seven inner chunks per 455-date mobility shard; each stays below the 2 GB
# Blosc buffer limit and lets one run decode on several threads.
MOBILITY_DATE_CHUNK = 65
COMPRESSOR = {"cname": "zstd", "clevel": 5, "shuffle": "shuffle"}


def is_zarr_zip(path: Path | str) -> bool:
    return str(path).endswith(ZARR_ZIP_SUFFIX)


def open_zarr_zip(path: Path | str, **kwargs: Any) -> xr.Dataset:
    """Open a packed store read-only."""
    store = zarr.storage.ZipStore(Path(path), mode="r")
    # Consolidated metadata is not part of the Zarr v3 specification; the
    # per-array metadata in a local ZIP file is cheap to read.
    return xr.open_zarr(store, consolidated=False, **kwargs)


def is_per_run(var: xr.DataArray) -> bool:
    """Variables that workers write: one shard per run along ``run_id``."""
    return bool(var.dims) and var.dims[0] == RUN_DIM and DATE_DIM in var.dims


def with_variable_length_strings(dataset: xr.Dataset) -> xr.Dataset:
    """Store fixed-width numpy strings as Zarr v3 variable-length strings.

    Fixed-width UTF-32 has no Zarr v3 specification. NetCDF outputs also read
    strings back as variable-length object arrays.
    """
    converted = {
        name: var.astype(object)
        for name, var in dataset.variables.items()
        if var.dtype.kind == "U"
    }
    return dataset.assign_coords(
        {name: var for name, var in converted.items() if name in dataset.coords}
    ).assign(
        {name: var for name, var in converted.items() if name in dataset.data_vars}
    )


def build_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    compressor = BloscCodec(
        cname=COMPRESSOR["cname"],
        clevel=COMPRESSOR["clevel"],
        shuffle=BloscShuffle(COMPRESSOR["shuffle"]),
    )
    encoding: dict[str, dict[str, Any]] = {}
    for name, var in dataset.data_vars.items():
        shape = tuple(var.shape)
        if is_per_run(var):
            shard = [1, *shape[1:]]
            chunks = list(shard)
            if all(dim in var.dims for dim in MOBILITY_DIMS):
                date_axis = var.dims.index(DATE_DIM)
                length = shape[date_axis]
                chunk = min(MOBILITY_DATE_CHUNK, length)
                chunks[date_axis] = chunk
                # A shard must hold whole inner chunks; it may extend past the
                # end of the array.
                shard[date_axis] = -(-length // chunk) * chunk
            encoding[str(name)] = {
                "chunks": tuple(chunks),
                "shards": tuple(shard),
                "compressors": [compressor],
            }
        else:
            encoding[str(name)] = {
                "chunks": shape if shape else None,
                "compressors": [compressor],
            }
    return encoding


@dataclass(frozen=True)
class RunWriteResult:
    run_index: int
    run_id: str
    non_null: dict[str, int]


_WORKER_DATASET: xr.Dataset | None = None
_WORKER_STORE: Path | None = None
_WORKER_VARIABLES: tuple[str, ...] = ()
_WORKER_COUNT_VARIABLES: tuple[str, ...] = ()


def _close_cached_files() -> None:
    # Forked workers must not share open HDF5 handles with the parent.
    # xarray reopens a closed file on the next access.
    for key in list(FILE_CACHE.keys()):
        handle = FILE_CACHE.pop(key, None)
        if handle is not None:
            handle.close()


def _values_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _write_run(run_index: int) -> RunWriteResult:
    dataset, store = _WORKER_DATASET, _WORKER_STORE
    if dataset is None or store is None:
        raise RuntimeError("Zarr run writer is not initialised")
    region = {RUN_DIM: slice(run_index, run_index + 1)}
    written = xr.open_zarr(store, consolidated=False)
    non_null: dict[str, int] = {}
    with dask.config.set(scheduler="synchronous"):
        for name in _WORKER_VARIABLES:
            values = dataset[name].isel(region).compute()
            part = xr.Dataset({name: values}).drop_vars(list(values.coords))
            part.to_zarr(store, region=region, mode="r+", consolidated=False)
            readback = written[name].isel(region).values
            if not _values_equal(values.values, readback):
                raise ValueError(
                    f"Zarr read-back of {name!r} differs for run index {run_index}"
                )
            if name in _WORKER_COUNT_VARIABLES:
                non_null[name] = int(np.count_nonzero(~np.isnan(values.values)))
    run_id = str(dataset[RUN_DIM].values[run_index])
    return RunWriteResult(run_index=run_index, run_id=run_id, non_null=non_null)


def _write_whole_variable(store: Path, dataset: xr.Dataset, name: str) -> None:
    var = dataset[name]
    with dask.config.set(scheduler="synchronous"):
        values = var.compute()
    part = xr.Dataset({name: values}).drop_vars(list(values.coords))
    region = {str(dim): slice(None) for dim in var.dims}
    part.to_zarr(store, region=region, mode="r+", consolidated=False)


def pack_store(store: Path, destination: Path) -> None:
    """Pack a directory store into an uncompressed ZIP64 file."""
    files = sorted(p for p in store.rglob("*") if p.is_file())
    # Root metadata first, so a reader finds it at the start of the archive.
    files.sort(key=lambda p: p.relative_to(store).as_posix() != "zarr.json")
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(store).as_posix())


def write_dataset(
    dataset: xr.Dataset,
    destination: Path,
    *,
    workers: int,
    count_variables: tuple[str, ...] = ("mobility", "cases"),
    log: Callable[[str], None] = print,
) -> list[RunWriteResult]:
    """Write ``dataset`` to ``destination`` (``*.zarr.zip``) in parallel.

    Workers compute and write one run at a time and verify each written region
    by read-back before the store is packed. Returns the per-run results in
    run order.
    """
    global _WORKER_DATASET, _WORKER_STORE, _WORKER_VARIABLES, _WORKER_COUNT_VARIABLES
    destination = Path(destination)
    if not is_zarr_zip(destination):
        raise ValueError(f"Packed Zarr output must end in {ZARR_ZIP_SUFFIX}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    store = destination.with_name(destination.name.removesuffix(".zip") + ".staging")
    if store.exists():
        shutil.rmtree(store)
    destination.unlink(missing_ok=True)

    dataset = with_variable_length_strings(dataset).assign_attrs(
        zarr_layout_version=LAYOUT_VERSION,
        zarr_compressor=f"blosc-{COMPRESSOR['cname']}-{COMPRESSOR['clevel']}-"
        f"{COMPRESSOR['shuffle']}",
    )
    # Data held in memory is written with the template; dask-backed data is
    # written below, per run or as a whole variable. Those writes use computed
    # NumPy regions that cover whole shards, so the Dask chunk layout does not
    # need to match the shards.
    dataset.to_zarr(
        store,
        mode="w",
        compute=False,
        encoding=build_encoding(dataset),
        zarr_format=3,
        consolidated=False,
        safe_chunks=False,
    )
    lazy = [str(name) for name, var in dataset.data_vars.items() if var.chunks]
    per_run = tuple(name for name in lazy if is_per_run(dataset[name]))
    for name in lazy:
        if name not in per_run:
            _write_whole_variable(store, dataset, name)

    run_count = dataset.sizes.get(RUN_DIM, 0)
    _WORKER_DATASET, _WORKER_STORE = dataset, store
    _WORKER_VARIABLES = per_run
    _WORKER_COUNT_VARIABLES = tuple(n for n in count_variables if n in per_run)
    try:
        log(
            f"  Writing {len(per_run)} per-run variables for {run_count} runs "
            f"with {workers} worker(s)"
        )
        if workers == 1 or run_count <= 1:
            results = [_write_run(index) for index in range(run_count)]
        else:
            _close_cached_files()
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(workers, run_count), mp_context=context
            ) as pool:
                results = []
                for result in pool.map(_write_run, range(run_count)):
                    results.append(result)
                    log(f"    wrote run {result.run_index + 1}/{run_count}")
    finally:
        _WORKER_DATASET, _WORKER_STORE = None, None
        _WORKER_VARIABLES, _WORKER_COUNT_VARIABLES = (), ()

    pack_store(store, destination)
    shutil.rmtree(store)
    return results
