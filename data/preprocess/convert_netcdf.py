"""Convert a processed NetCDF dataset to the packed sharded Zarr layout.

Usage:
    python -m data.preprocess.convert_netcdf SOURCE.nc DEST.zarr.zip \
        --workers 16 [--source-dvc-md5 MD5] [--bitround mobility=10] [--runs N]

Without ``--bitround``, values, dimensions, coordinates and attributes are
unchanged. ``--bitround NAME=BITS`` rounds a float32 variable to BITS mantissa
bits, as ``preprocessing.bitround_keepbits`` does. ``--runs N`` keeps the first
N runs, for bounded acceptance checks. Every written run is verified by
read-back before the store is packed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import xarray as xr

from .zarr_layout import RUN_DIM, write_dataset


def parse_bitround(items: list[str]) -> dict[str, int]:
    keepbits: dict[str, int] = {}
    for item in items:
        name, sep, bits = item.partition("=")
        if not sep or not name or not bits.isdigit():
            raise ValueError(f"--bitround needs NAME=BITS, got {item!r}")
        keepbits[name] = int(bits)
    return keepbits


def convert(
    source: Path,
    destination: Path,
    *,
    workers: int,
    source_dvc_md5: str | None,
    bitround_keepbits: dict[str, int] | None = None,
    runs: int | None = None,
) -> None:
    dataset = xr.open_dataset(source, engine="h5netcdf", chunks={RUN_DIM: 1})
    if runs is not None:
        dataset = dataset.isel({RUN_DIM: slice(0, runs)})
    attrs = {"converted_from": source.name}
    if runs is not None:
        attrs["converted_run_subset"] = f"first {runs} runs"
    if source_dvc_md5 is not None:
        attrs["converted_from_dvc_md5"] = source_dvc_md5
    dataset = dataset.assign_attrs(attrs)
    started = time.monotonic()
    results = write_dataset(
        dataset, destination, workers=workers, bitround_keepbits=bitround_keepbits
    )
    elapsed = time.monotonic() - started
    print(
        f"Converted {len(results)} runs to {destination} in {elapsed:.0f} s "
        f"({destination.stat().st_size / 1e9:.2f} GB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-dvc-md5", default=None)
    parser.add_argument("--bitround", action="append", default=[])
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()
    convert(
        args.source,
        args.destination,
        workers=args.workers,
        source_dvc_md5=args.source_dvc_md5,
        bitround_keepbits=parse_bitround(args.bitround),
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
