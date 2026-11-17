import errno
import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/workflow/parallel_copy.py"


def _load():
    spec = importlib.util.spec_from_file_location("parallel_copy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "a.bin").write_bytes(bytes(range(256)) * 3)
    (source / "nested/b.bin").write_bytes(b"payload-" * 40)
    return source


def _run(module, monkeypatch, tmp_path, source, destination):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            str(source),
            str(destination),
            "--min-bytes",
            "1",
            "--reserve-bytes",
            "0",
            "--lock",
            str(tmp_path / "staging.lock"),
            "--streams",
            "4",
        ],
    )
    module.main()


def test_reserved_destinations_receive_exact_contents(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "BLOCK_BYTES", 64)
    monkeypatch.setattr(
        os,
        "posix_fallocate",
        lambda fd, offset, length: os.ftruncate(fd, offset + length),
        raising=False,
    )
    source = _sources(tmp_path)
    destination = tmp_path / "destination"
    (destination / "nested").mkdir(parents=True)
    (destination / "nested/b.bin").write_bytes(b"stale" * 1000)

    _run(module, monkeypatch, tmp_path, source, destination)

    for name in ["a.bin", "nested/b.bin"]:
        assert (destination / name).read_bytes() == (source / name).read_bytes()


def test_losing_the_space_race_releases_reservations_before_copying(
    tmp_path, monkeypatch
):
    module = _load()
    calls = []

    def fallocate(fd, offset, length):
        calls.append(length)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        os.ftruncate(fd, offset + length)

    monkeypatch.setattr(os, "posix_fallocate", fallocate, raising=False)
    source = _sources(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(SystemExit) as failure:
        _run(module, monkeypatch, tmp_path, source, destination)

    assert "filled while reserving" in str(failure.value.code)
    assert len(calls) == 2
    assert not [path for path in destination.rglob("*") if path.is_file()]
