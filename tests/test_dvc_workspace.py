"""DVC workspace contracts against a real local remote and shared cache."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.workflow import dvc_workspace
from tests.dvc_test_helpers import publish_pointer, remote_object

RAW = "data/files/raw.zarr.dvc"
PROCESSED = "data/processed/dataset.zarr.zip.dvc"
RAW_MEMBERS = {"zarr.json": b"{}", "values/0": b"\x00\x01\x02"}
DATASET = b"packed-zarr-payload"


@pytest.fixture
def store(tmp_path: Path) -> dvc_workspace.DvcStore:
    return dvc_workspace.DvcStore(
        remote=tmp_path / "remote", cache=tmp_path / "publish-cache"
    )


def _node_store(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> dvc_workspace.DvcStore:
    """The same remote behind an empty cache, so a pull must fetch."""
    return dvc_workspace.DvcStore(remote=store.remote, cache=tmp_path / "node-cache")


def _source_with_pointers(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    publish_pointer(store, tmp_path / "scratch", source_root, RAW, RAW_MEMBERS)
    publish_pointer(store, tmp_path / "scratch", source_root, PROCESSED, DATASET)
    return run_root, source_root


def test_publish_pushes_the_payload_and_returns_its_pointer(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    workspace = tmp_path / "publication"
    (workspace / "artifacts").mkdir(parents=True)
    (workspace / "artifacts/summary.json").write_text("{}\n")

    pointer = dvc_workspace.publish(store, workspace, "artifacts")

    assert pointer == dvc_workspace.read_pointer(workspace / "artifacts.dvc")
    assert pointer.path == Path("artifacts")
    assert pointer.md5.endswith(".dir")
    assert pointer.nfiles == 1
    assert remote_object(store, pointer.md5).is_file()


def test_pull_materializes_pushed_outputs_and_records_bind_roots(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)

    record = dvc_workspace.pull(
        _node_store(store, tmp_path), source_root, [RAW, PROCESSED], run_root
    )

    workspace = run_root / dvc_workspace.WORKSPACE
    for member, content in RAW_MEMBERS.items():
        assert (workspace / "data/files/raw.zarr" / member).read_bytes() == content
    assert (workspace / "data/processed/dataset.zarr.zip").read_bytes() == DATASET
    outputs = ["data/files/raw.zarr", "data/processed/dataset.zarr.zip"]
    assert (run_root / dvc_workspace.OUTPUTS_FILE).read_text().splitlines() == outputs
    assert (run_root / dvc_workspace.BIND_ROOTS_FILE).read_text().splitlines() == [
        "data/files",
        "data/processed",
    ]
    assert json.loads((run_root / dvc_workspace.DATA_MANIFEST).read_text()) == record
    assert record["outputs"] == outputs
    assert record["targets"] == {
        RAW: {
            "md5": dvc_workspace.read_pointer(source_root / RAW).md5,
            "output": "data/files/raw.zarr",
        },
        PROCESSED: {
            "md5": dvc_workspace.read_pointer(source_root / PROCESSED).md5,
            "output": "data/processed/dataset.zarr.zip",
        },
    }


def test_pull_keeps_dvc_site_cache_on_node_local_temp(
    store: dvc_workspace.DvcStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    node_local = tmp_path / "node-local"
    node_local.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(node_local))

    dvc_workspace.pull(_node_store(store, tmp_path), source_root, [RAW], run_root)

    (site_cache,) = node_local.glob("dvc-site-cache-*")
    assert any(site_cache.iterdir())


def test_pull_resumes_over_a_read_only_pointer(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    (source_root / RAW).chmod(0o444)
    node_store = _node_store(store, tmp_path)

    dvc_workspace.pull(node_store, source_root, [RAW], run_root)
    record = dvc_workspace.pull(node_store, source_root, [RAW], run_root)

    assert record["outputs"] == ["data/files/raw.zarr"]


def test_pull_rejects_an_output_missing_from_remote_and_cache(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    remote_object(
        store, dvc_workspace.read_pointer(source_root / PROCESSED).md5
    ).unlink()

    with pytest.raises(ValueError, match="differ from their pointers"):
        dvc_workspace.pull(
            _node_store(store, tmp_path), source_root, [PROCESSED], run_root
        )


def _as_git_import(pointer_path: Path, *, size: int) -> None:
    document = yaml.safe_load(pointer_path.read_text())
    document["outs"][0]["size"] = size
    # Production imports are frozen at a locked upstream revision. The upstream
    # repository does not exist here: a status check would fail.
    document["frozen"] = True
    document["deps"] = [
        {
            "path": "data/datasets/raw.nc",
            "repo": {
                "url": str(pointer_path.parent / "upstream"),
                "rev_lock": "0" * 40,
            },
        }
    ]
    pointer_path.write_text(yaml.safe_dump(document, sort_keys=False))


# Production places imported objects directly in the shared cache; DVC does
# not fetch a frozen import from the remote.
def test_pull_trusts_a_cached_git_import_of_the_pinned_size(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    _as_git_import(source_root / PROCESSED, size=len(DATASET))

    record = dvc_workspace.pull(store, source_root, [PROCESSED], run_root)

    assert record["outputs"] == ["data/processed/dataset.zarr.zip"]
    assert dvc_workspace.read_pointer(source_root / PROCESSED).imported


def test_pull_rejects_a_git_import_with_the_wrong_size(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    _as_git_import(source_root / PROCESSED, size=1)

    with pytest.raises(ValueError, match="size mismatch"):
        dvc_workspace.pull(store, source_root, [PROCESSED], run_root)


def test_pull_rejects_a_git_import_missing_from_the_cache(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    _as_git_import(source_root / PROCESSED, size=len(DATASET))

    # DVC links the absent object and exits 0.
    with pytest.raises(FileNotFoundError, match="not materialized"):
        dvc_workspace.pull(
            _node_store(store, tmp_path), source_root, [PROCESSED], run_root
        )


def test_pull_requires_the_bind_root_in_the_source_snapshot(
    store: dvc_workspace.DvcStore, tmp_path: Path
) -> None:
    run_root, source_root = _source_with_pointers(store, tmp_path)
    pointer = source_root / "nested/raw.zarr.dvc"
    pointer.parent.mkdir()
    document = yaml.safe_load((source_root / RAW).read_text())
    document["outs"][0]["path"] = "payload/raw.zarr"
    pointer.write_text(yaml.safe_dump(document))

    with pytest.raises(FileNotFoundError, match="bind root"):
        dvc_workspace.pull(
            _node_store(store, tmp_path), source_root, ["nested/raw.zarr.dvc"], run_root
        )


@pytest.mark.parametrize(
    ("outs", "message"),
    [
        ([], "Expected one DVC output"),
        ([{"path": "a"}, {"path": "b"}], "Expected one DVC output"),
        ([{"path": "raw.zarr"}], "Malformed DVC output"),
        ([{"md5": "0" * 32, "path": "../raw.zarr"}], "safe relative path"),
        ([{"md5": "0" * 32, "path": "/data/raw.zarr"}], "safe relative path"),
    ],
)
def test_read_pointer_rejects_malformed_pointers(
    tmp_path: Path, outs: list[dict[str, str]], message: str
) -> None:
    pointer = tmp_path / "raw.zarr.dvc"
    pointer.write_text(yaml.safe_dump({"outs": outs}))

    with pytest.raises(ValueError, match=message):
        dvc_workspace.read_pointer(pointer)
