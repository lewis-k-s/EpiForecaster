"""Sharded Zarr v3 layout for processed datasets, packed into one ZIP file.

Each variable with ``run_id`` and ``date`` dimensions gets one shard per run,
so a worker that writes run ``i`` is the only writer of every shard it
touches. The directory store is packed into one uncompressed ZIP64 file, which
DVC tracks as one object. See docs/SYNTHETIC_ZARR_LAYOUT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Any
import weakref
import zipfile

import dask
import dask.array as da
import numpy as np
import xarray as xr
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
FLOAT32_MANTISSA_BITS = 23


def is_zarr_zip(path: Path | str) -> bool:
    return str(path).endswith(ZARR_ZIP_SUFFIX)


# Every open packed store, so that a forked child can reopen each one. Keyed by
# id() because ZipStore defines __eq__ and is therefore unhashable.
_OPEN_ZIP_STORES: weakref.WeakValueDictionary[int, zarr.storage.ZipStore] = (
    weakref.WeakValueDictionary()
)


def _reopen_zip_stores_in_child() -> None:
    # A forked child inherits the parent's open archive. All processes then
    # share one file offset, and ZipFile's seek-then-read races across them
    # (its lock is per process), so concurrent reads return wrong bytes. Give
    # the child its own file handle and lock.
    for store in list(_OPEN_ZIP_STORES.values()):
        if store._is_open:
            store._is_open = False
            store._sync_open()


os.register_at_fork(after_in_child=_reopen_zip_stores_in_child)


def restore_object_strings(dataset: xr.Dataset) -> xr.Dataset:
    """Return variable-length string variables as object arrays.

    xarray 2026 decodes Zarr variable-length strings as NumPy ``StringDType``,
    which has no unsized ``astype(str)`` or ``np.str_`` elements. Every consumer
    of these datasets was written for object arrays, so restore them at the
    single open boundary. Data variables stay lazy.
    """
    coords = {
        name: dataset[name].astype(object)
        for name in dataset.coords
        if dataset[name].dtype.kind == "T"
    }
    data_vars = {
        name: dataset[name].astype(object)
        for name in dataset.data_vars
        if dataset[name].dtype.kind == "T"
    }
    # Coordinates go last: assigning data variables afterwards re-aligns them
    # on the original index and reverts it to StringDType.
    if data_vars:
        dataset = dataset.assign(data_vars)
    if coords:
        dataset = dataset.assign_coords(coords)
    return dataset


def open_zarr_zip(path: Path | str, **kwargs: Any) -> xr.Dataset:
    """Open a packed store read-only; forked children reopen it."""
    store = zarr.storage.ZipStore(Path(path), mode="r")
    _OPEN_ZIP_STORES[id(store)] = store
    # Consolidated metadata is not part of the Zarr v3 specification; the
    # per-array metadata in a local ZIP file is cheap to read.
    return restore_object_strings(xr.open_zarr(store, consolidated=False, **kwargs))


def open_zarr_store(path: Path | str, **kwargs: Any) -> xr.Dataset:
    """Open a packed ``*.zarr.zip`` or a Zarr directory store read-only."""
    if is_zarr_zip(path):
        return open_zarr_zip(path, **kwargs)
    return restore_object_strings(xr.open_zarr(str(path), **kwargs))


def bitround(values: np.ndarray, keepbits: int) -> np.ndarray:
    """Round float32 values to ``keepbits`` mantissa bits (round half to even).

    The dropped bits become zero, so the compressor removes them. The float32
    exponent range does not change; NaN and infinite values stay as they are.
    """
    if values.dtype != np.float32:
        raise TypeError(f"BitRound needs float32 values, not {values.dtype}")
    if not 0 < keepbits < FLOAT32_MANTISSA_BITS:
        raise ValueError(f"keepbits must be in 1..{FLOAT32_MANTISSA_BITS - 1}")
    maskbits = FLOAT32_MANTISSA_BITS - keepbits
    bits = values.view(np.uint32)
    half_quantum = np.uint32((1 << (maskbits - 1)) - 1)
    lowest_kept = (bits >> np.uint32(maskbits)) & np.uint32(1)
    mask = np.uint32((0xFFFFFFFF >> maskbits) << maskbits)
    rounded = ((bits + lowest_kept + half_quantum) & mask).view(np.float32)
    return np.where(np.isfinite(values), rounded, values)


def apply_bitround(dataset: xr.Dataset, keepbits: dict[str, int]) -> xr.Dataset:
    """Round the named float32 variables and record the bits in their attrs."""
    rounded = {}
    for name, bits in keepbits.items():
        if name not in dataset.data_vars:
            raise KeyError(f"BitRound variable {name!r} is not in the dataset")
        var = dataset[name]
        if var.dtype != np.float32:
            raise TypeError(f"BitRound variable {name!r} is {var.dtype}, not float32")
        result = xr.apply_ufunc(
            bitround,
            var,
            kwargs={"keepbits": bits},
            dask="parallelized",
            output_dtypes=[np.float32],
            keep_attrs=True,
        )
        rounded[name] = result.assign_attrs(bitround_keepbits=bits)
    return dataset.assign(rounded)


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
    """What one ``write_run`` call wrote and verified.

    ``run_level`` holds the run's values of the variables that keep all runs in
    one chunk (``run_id`` but no ``date``); only one process may write those, so
    the caller collects them and passes them to ``finish_store``. ``static``
    holds the variables without ``run_id``.
    """

    run_index: int
    run_id: str
    non_null: dict[str, int]
    run_level: dict[str, np.ndarray] = field(default_factory=dict)
    static: dict[str, np.ndarray] = field(default_factory=dict)


# Per-run variables larger than this are computed into one preallocated array
# and verified in date slabs, so a worker holds a single copy of the run.
LARGE_RUN_VARIABLE_BYTES = 256 << 20
# Packing reads and writes in 16 MiB blocks; zipfile's 8 KiB default packs a
# large shard about half as fast.
PACK_COPY_BYTES = 16 << 20


def _values_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _non_null(values: np.ndarray) -> int:
    if np.issubdtype(values.dtype, np.floating):
        return int(np.count_nonzero(~np.isnan(values)))
    return int(values.size)


def prepare_for_store(
    dataset: xr.Dataset, bitround_keepbits: dict[str, int] | None = None
) -> xr.Dataset:
    """Apply BitRound, variable-length strings and the layout attributes."""
    keepbits = dict(sorted((bitround_keepbits or {}).items()))
    dataset = apply_bitround(dataset, keepbits)
    return with_variable_length_strings(dataset).assign_attrs(
        zarr_layout_version=LAYOUT_VERSION,
        zarr_compressor=f"blosc-{COMPRESSOR['cname']}-{COMPRESSOR['clevel']}-"
        f"{COMPRESSOR['shuffle']}",
        zarr_bitround_keepbits_json=json.dumps(keepbits),
    )


def staging_path(destination: Path) -> Path:
    return destination.with_name(destination.name.removesuffix(".zip") + ".staging")


def create_store(template: xr.Dataset, destination: Path) -> Path:
    """Create the directory store for ``destination`` (``*.zarr.zip``).

    ``template`` must come from ``prepare_for_store``. Variables held in memory
    are written now. Dask-backed variables only define their arrays: the
    template replaces them with empty arrays of the same shape, so creating the
    store does not build the task graph of the data.
    """
    destination = Path(destination)
    if not is_zarr_zip(destination):
        raise ValueError(f"Packed Zarr output must end in {ZARR_ZIP_SUFFIX}")
    store = staging_path(destination)
    if store.exists():
        shutil.rmtree(store)
    destination.unlink(missing_ok=True)
    skeleton = template.assign(
        {
            name: var.copy(data=da.empty(var.shape, dtype=var.dtype, chunks=-1))
            for name, var in template.data_vars.items()
            if var.chunks
        }
    )
    skeleton.to_zarr(
        store,
        mode="w",
        compute=False,
        encoding=build_encoding(template),
        zarr_format=3,
        consolidated=False,
        safe_chunks=False,
    )
    return store


def _compute_into(var: xr.DataArray) -> np.ndarray:
    if not isinstance(var.data, da.Array):
        return np.asarray(var.data)
    values = np.empty(var.shape, dtype=var.dtype)
    da.store(var.data, values, lock=False, scheduler="synchronous")
    return values


def write_run(
    store: Path,
    run_index: int,
    dataset: xr.Dataset,
    *,
    count_variables: tuple[str, ...] = ("mobility", "cases"),
) -> RunWriteResult:
    """Compute, write and verify run ``run_index`` of a store from ``create_store``.

    ``dataset`` holds this one run (``run_id`` of size 1) and must come from
    ``prepare_for_store``. Each per-run variable fills one shard, so concurrent
    calls for different runs never write the same file. Every written region
    is read back and compared before the call returns.
    """
    if dataset.sizes.get(RUN_DIM) != 1:
        raise ValueError("write_run needs a dataset that holds exactly one run")
    per_run, run_level, static, large = [], [], [], []
    for name, var in dataset.data_vars.items():
        if is_per_run(var):
            (large if var.nbytes > LARGE_RUN_VARIABLE_BYTES else per_run).append(name)
        elif RUN_DIM in var.dims:
            run_level.append(name)
        else:
            static.append(name)
    small = [*per_run, *run_level, *static]
    with dask.config.set(scheduler="synchronous"):
        computed = dict(
            zip(small, dask.compute(*(dataset[name].data for name in small)))
        )
    computed = {name: np.asarray(values) for name, values in computed.items()}

    region = {RUN_DIM: slice(run_index, run_index + 1)}
    group = zarr.open_group(store, mode="r", zarr_format=3)
    non_null: dict[str, int] = {}

    def write(names: list[str], values: dict[str, np.ndarray]) -> None:
        part = xr.Dataset({name: (dataset[name].dims, values[name]) for name in names})
        part.to_zarr(store, region=region, mode="r+", consolidated=False)

    if per_run:
        write(per_run, computed)
        for name in per_run:
            if not _values_equal(
                computed[name], group[name][run_index : run_index + 1]
            ):
                raise ValueError(
                    f"Zarr read-back of {name!r} differs for run index {run_index}"
                )
            if name in count_variables:
                non_null[name] = _non_null(computed[name])
    for name in large:
        values = _compute_into(dataset[name])
        write([name], {name: values})
        array = group[name]
        date_axis = dataset[name].dims.index(DATE_DIM)
        count = 0
        for start in range(0, values.shape[date_axis], MOBILITY_DATE_CHUNK):
            window = slice(start, start + MOBILITY_DATE_CHUNK)
            index = [slice(None)] * values.ndim
            index[date_axis] = window
            expected = values[tuple(index)]
            index[0] = slice(run_index, run_index + 1)
            if not _values_equal(expected, array[tuple(index)]):
                raise ValueError(
                    f"Zarr read-back of {name!r} differs for run index {run_index}"
                )
            count += _non_null(expected)
        if name in count_variables:
            non_null[name] = count
        del values
    return RunWriteResult(
        run_index=run_index,
        run_id=str(dataset[RUN_DIM].values[0]),
        non_null=non_null,
        run_level={name: computed[name] for name in run_level},
        static={name: computed[name] for name in static},
    )


def finish_store(
    store: Path,
    destination: Path,
    *,
    run_level: xr.Dataset | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write the run-level variables and attributes, then pack the store."""
    if run_level is not None and run_level.data_vars:
        run_level = run_level.drop_vars(list(run_level.coords))
        region = {str(dim): slice(None) for dim in run_level.dims}
        run_level.to_zarr(store, region=region, mode="r+", consolidated=False)
    if attrs:
        zarr.open_group(store, mode="r+", zarr_format=3).attrs.update(attrs)
    pack_store(store, Path(destination))
    shutil.rmtree(store)


def collect_run_level(
    template: xr.Dataset, results: list[RunWriteResult]
) -> xr.Dataset:
    """Stack the run-level values that ``write_run`` returned, in run order."""
    ordered = sorted(results, key=lambda result: result.run_index)
    names = sorted(ordered[0].run_level) if ordered else []
    return xr.Dataset(
        {
            name: (
                template[name].dims,
                np.concatenate([result.run_level[name] for result in ordered]),
            )
            for name in names
        }
    )


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
            info = zipfile.ZipInfo.from_file(path, path.relative_to(store).as_posix())
            info.compress_type = zipfile.ZIP_STORED
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, PACK_COPY_BYTES)


def write_dataset(
    dataset: xr.Dataset,
    destination: Path,
    *,
    bitround_keepbits: dict[str, int] | None = None,
    count_variables: tuple[str, ...] = ("mobility", "cases"),
) -> list[RunWriteResult]:
    """Write an in-process ``dataset`` to ``destination`` (``*.zarr.zip``).

    Runs are written one after another with ``write_run``, so every region is
    verified by read-back before the store is packed. ``bitround_keepbits``
    maps float32 variables to the mantissa bits to keep. Returns the per-run
    results in run order.
    """
    destination = Path(destination)
    prepared = prepare_for_store(dataset, bitround_keepbits)
    store = create_store(prepared, destination)
    for name, var in prepared.data_vars.items():
        if var.chunks and RUN_DIM not in var.dims:
            _write_whole_variable(store, prepared, str(name))
    run_vars = [name for name, var in prepared.data_vars.items() if RUN_DIM in var.dims]
    runs = prepared[run_vars]
    results = [
        write_run(
            store,
            index,
            runs.isel({RUN_DIM: slice(index, index + 1)}),
            count_variables=count_variables,
        )
        for index in range(prepared.sizes.get(RUN_DIM, 0))
    ]
    finish_store(store, destination, run_level=collect_run_level(prepared, results))
    return results
