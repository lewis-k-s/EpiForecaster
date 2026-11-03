#!/usr/bin/env python3
"""Pack a Zarr directory store into one uncompressed ``*.zarr.zip`` file.

Usage:
    uv run python -m scripts.data.pack_zarr_store pack SOURCE_DIR DEST.zarr.zip

Chunk files are copied byte for byte (ZIP_STORED, no recompression), with the
same archive convention as ``data.preprocess.zarr_layout.pack_store``. The
archive is built in a temporary file next to the destination, verified against
the source, and then renamed. The command exits non-zero on any mismatch and
prints the md5 that a DVC pointer needs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import zipfile
import zlib

import click
import numpy as np
import xarray as xr

from data.preprocess.zarr_layout import ZARR_ZIP_SUFFIX, open_zarr_zip, pack_store

BLOCK_BYTES = 256 * 1024 * 1024
RAW = {"mask_and_scale": False, "decode_times": False, "decode_timedelta": False}
ENCODING_KEYS = ("chunks", "preferred_chunks", "dtype")


class VerificationError(Exception):
    """The archive differs from the source store."""


def _file_digests(path: Path) -> dict[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()  # noqa: S324
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            crc = zlib.crc32(block, crc)
    return crc


def verify_files(source: Path, archive: Path) -> tuple[int, int]:
    """Check that the archive holds every source file, unchanged and stored."""
    files = {
        p.relative_to(source).as_posix(): p for p in source.rglob("*") if p.is_file()
    }
    total = 0
    with zipfile.ZipFile(archive) as zf:
        entries = {info.filename: info for info in zf.infolist()}
        if set(entries) != set(files):
            missing = sorted(set(files) - set(entries))[:5]
            extra = sorted(set(entries) - set(files))[:5]
            raise VerificationError(
                f"File sets differ: missing {missing} extra {extra}"
            )
        bad_zip = zf.testzip()
        if bad_zip is not None:
            raise VerificationError(f"Corrupt archive member: {bad_zip}")
        for name, info in entries.items():
            path = files[name]
            if info.compress_type != zipfile.ZIP_STORED:
                raise VerificationError(f"{name} is compressed")
            if info.file_size != path.stat().st_size or info.CRC != _crc32(path):
                raise VerificationError(f"{name} differs from the source file")
            total += info.file_size
    return len(files), total


def _equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    if expected.dtype.kind in "fc":
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _attrs_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(_equal_value(left[k], right[k]) for k in left)


def _equal_value(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return _attrs_equal(left, right)
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def verify_datasets(source: Path, archive: Path) -> int:
    """Compare structure, metadata, and exact values. Returns variable count."""
    with xr.open_zarr(str(source)) as src, open_zarr_zip(archive) as zipped:
        if set(src.variables) != set(zipped.variables):
            raise VerificationError("Variable names differ")
        if set(src.coords) != set(zipped.coords):
            raise VerificationError("Coordinate names differ")
        if dict(src.sizes) != dict(zipped.sizes):
            raise VerificationError("Dimension sizes differ")
        if not _attrs_equal(src.attrs, zipped.attrs):
            raise VerificationError("Dataset attrs differ")
        for name, var in src.variables.items():
            other = zipped.variables[name]
            label = f"variable {name!r}"
            if var.dims != other.dims:
                raise VerificationError(f"{label} dims differ")
            if var.dtype != other.dtype:
                raise VerificationError(f"{label} dtype differs")
            if not _attrs_equal(var.attrs, other.attrs):
                raise VerificationError(f"{label} attrs differ")
            for key in ENCODING_KEYS:
                if var.encoding.get(key) != other.encoding.get(key):
                    raise VerificationError(f"{label} encoding {key!r} differs")
    with (
        xr.open_zarr(str(source), **RAW) as src,
        open_zarr_zip(archive, **RAW) as zipped,
    ):
        for name, var in src.variables.items():
            _compare_values(name, var, zipped.variables[name])
        return len(src.variables)


def _compare_values(name: str, left: xr.Variable, right: xr.Variable) -> None:
    if left.ndim == 0:
        pairs = [(left.values, right.values)]
    else:
        row_bytes = max(1, left.dtype.itemsize * int(np.prod(left.shape[1:])))
        step = max(1, BLOCK_BYTES // row_bytes)
        pairs = (
            (left[i : i + step].values, right[i : i + step].values)
            for i in range(0, left.shape[0], step)
        )
    for expected, actual in pairs:
        if not _equal(expected, actual):
            raise VerificationError(f"Values of variable {name!r} differ")


def pack_and_verify(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_dir():
        raise click.ClickException(f"Source is not a directory: {source}")
    if not destination.name.endswith(ZARR_ZIP_SUFFIX):
        raise click.ClickException(f"Destination must end in {ZARR_ZIP_SUFFIX}")
    if destination.exists():
        raise click.ClickException(f"Refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise click.ClickException(f"Stale temporary file: {temporary}")
    try:
        pack_store(source, temporary)
        file_count, byte_count = verify_files(source, temporary)
        variable_count = verify_datasets(source, temporary)
        digests = _file_digests(temporary)
        if destination.exists():
            raise click.ClickException(f"Refusing to overwrite: {destination}")
        temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "files": file_count,
        "source_bytes": byte_count,
        "zip_bytes": destination.stat().st_size,
        "variables": variable_count,
        **digests,
    }


@click.group()
def cli() -> None:
    """Pack Zarr directory stores into single-file archives."""


@cli.command()
@click.argument("source", type=click.Path(path_type=Path))
@click.argument("destination", type=click.Path(path_type=Path))
def pack(source: Path, destination: Path) -> None:
    """Pack SOURCE (Zarr directory) into DESTINATION (``*.zarr.zip``)."""
    try:
        summary = pack_and_verify(source, destination)
    except VerificationError as error:
        click.echo(f"VERIFICATION FAILED: {error}", err=True)
        sys.exit(1)
    click.echo(f"Packed {destination}")
    click.echo(
        f"  files={summary['files']} variables={summary['variables']} "
        f"source_bytes={summary['source_bytes']} zip_bytes={summary['zip_bytes']}"
    )
    click.echo(f"  md5={summary['md5']}")
    click.echo(f"  sha256={summary['sha256']}")


if __name__ == "__main__":
    cli()
