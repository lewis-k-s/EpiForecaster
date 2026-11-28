"""DVC workspace: pull pointer targets into a run and publish run payloads.

Every workflow DVC operation runs the pinned ``dvc`` command in a no-SCM
workspace that is configured for the shared MN5 cache and remote. The module
owns that configuration, the lock on the shared cache, pointer parsing, and the
run-state files that the shell launchers read to bind pulled inputs.

Pointer parsing imports neither DVC nor the workflow runtime, because the
login-node renderer reads pointers without the container environment.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REMOTE_NAME = "mn5-nfs"
# Run-relative state read by node_input_cache.sh and mn5_container_exec.sh.
WORKSPACE = Path(".state/dvc-workspace")
OUTPUTS_FILE = Path(".state/dvc_paths.txt")
BIND_ROOTS_FILE = Path(".state/dvc_bind_roots.txt")
DATA_MANIFEST = Path(".state/data_manifest.json")
CACHE_LOCK_NAME = ".dvc-workflow.lock"


@dataclass(frozen=True)
class Pointer:
    """The one output of a ``.dvc`` pointer file."""

    path: Path
    md5: str
    size: int | None
    nfiles: int | None
    imported: bool


@dataclass(frozen=True)
class DvcStore:
    """Shared DVC remote and cache of one workflow run."""

    remote: Path
    cache: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DvcStore:
        dvc = config["dvc"]
        return cls(remote=Path(str(dvc["remote"])), cache=Path(str(dvc["cache"])))


def read_pointer(pointer_path: Path, *, label: object | None = None) -> Pointer:
    """Parse a pointer with one output at a safe path beside the pointer."""
    name = pointer_path if label is None else label
    document = yaml.safe_load(pointer_path.read_text())
    outputs = document.get("outs") if isinstance(document, dict) else None
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError(f"Expected one DVC output in {name}")
    output = outputs[0]
    if not isinstance(output, dict) or "md5" not in output or "path" not in output:
        raise ValueError(f"Malformed DVC output in {name}")
    path = Path(str(output["path"]))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(
            f"DVC output in {name} must be a safe relative path: {output['path']!r}"
        )
    dependencies = document.get("deps", [])
    return Pointer(
        path=path,
        md5=str(output["md5"]),
        size=None if output.get("size") is None else int(output["size"]),
        nfiles=None if output.get("nfiles") is None else int(output["nfiles"]),
        imported=any(
            isinstance(dependency, dict) and isinstance(dependency.get("repo"), dict)
            for dependency in dependencies
        ),
    )


def pull(
    store: DvcStore,
    source_root: Path,
    targets: Sequence[str | Path],
    run_root: Path,
) -> dict[str, Any]:
    """Materialize the targets' outputs in the run workspace and record them.

    Writes the output list, the bind roots that the launchers mount over the
    source snapshot, and the data manifest, and returns the manifest record.
    """
    workspace = run_root / WORKSPACE
    pulled: list[tuple[str, Path, Pointer]] = []
    for target_value in targets:
        target = Path(target_value)
        source_pointer = source_root / target
        workspace_pointer = workspace / target
        workspace_pointer.parent.mkdir(parents=True, exist_ok=True)
        # A retried attempt finds the read-only pointer copied by the last one.
        workspace_pointer.unlink(missing_ok=True)
        shutil.copy2(source_pointer, workspace_pointer)
        pointer = read_pointer(source_pointer)
        pulled.append((target.as_posix(), target.parent / pointer.path, pointer))
    names = [name for name, _, _ in pulled]

    with _configured(store, workspace) as dvc:
        with _cache_lock(store.cache):
            dvc("pull", *names)
        # Imports are excluded: their status asks the upstream Git repository.
        local = [name for name, _, pointer in pulled if not pointer.imported]
        if local:
            try:
                dvc("status", "--quiet", *local)
            except subprocess.CalledProcessError as error:
                raise ValueError(
                    "DVC targets differ from their pointers after pull: "
                    + ", ".join(local)
                ) from error

    for _, output, _ in pulled:
        if not (workspace / output).exists():
            raise FileNotFoundError(f"DVC output was not materialized: {output}")
    # DVC content-addresses the cache, so the imported object is trusted by its
    # path. Rehashing a 173 GB corpus over GPFS cost ~8 min per run; a size
    # check still catches a truncated or wrong object.
    for name, output, pointer in pulled:
        if not pointer.imported:
            continue
        materialized = workspace / output
        if pointer.md5.endswith(".dir") or not materialized.is_file():
            raise ValueError(
                "Git-backed DVC imports must materialize one regular file for "
                f"offline validation: {name}"
            )
        actual_size = materialized.stat().st_size
        if pointer.size is not None and actual_size != pointer.size:
            raise ValueError(
                f"Imported DVC output size mismatch for {name}: "
                f"{actual_size} != {pointer.size}"
            )

    outputs = [output for _, output, _ in pulled]
    bind_roots = sorted({output.parent for output in outputs})
    for bind_root in bind_roots:
        if not (source_root / bind_root).is_dir():
            raise FileNotFoundError(
                f"DVC bind root must exist in the source snapshot: {bind_root}"
            )
    (run_root / OUTPUTS_FILE).write_text(
        "".join(f"{output.as_posix()}\n" for output in outputs)
    )
    (run_root / BIND_ROOTS_FILE).write_text(
        "".join(f"{bind_root.as_posix()}\n" for bind_root in bind_roots)
    )
    data_record = {
        "schema_version": 1,
        "workspace": str(workspace),
        "targets": {
            name: {"md5": pointer.md5, "output": output.as_posix()}
            for name, output, pointer in pulled
        },
        "outputs": [output.as_posix() for output in outputs],
        "bind_roots": [bind_root.as_posix() for bind_root in bind_roots],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (run_root / DATA_MANIFEST).write_text(
        json.dumps(data_record, indent=2, sort_keys=True) + "\n"
    )
    return data_record


def publish(store: DvcStore, workspace: Path, payload: str) -> Pointer:
    """Cache and push ``workspace/payload``; its pointer is ``payload.dvc``.

    DVC replaces the payload with links into the shared cache. The caller owns
    the workspace and places the pointer.
    """
    with _configured(store, workspace) as dvc, _cache_lock(store.cache):
        dvc("add", payload)
        dvc("push", f"{payload}.dvc")
    return read_pointer(workspace / f"{payload}.dvc")


@contextmanager
def _configured(store: DvcStore, workspace: Path) -> Iterator[Callable[..., None]]:
    """Yield a ``dvc`` runner for a workspace set up for the shared store."""
    # Keep DVC's SQLite/index state off the cluster home and off GPFS: one
    # SQLite update per file on GPFS made checkout take about 1 s per file.
    # TMPDIR is the execution's node-local scratch inside the container.
    environment = {
        **os.environ,
        "DVC_SITE_CACHE_DIR": tempfile.mkdtemp(prefix="dvc-site-cache-"),
    }

    def dvc(*arguments: str) -> None:
        subprocess.run(["dvc", *arguments], cwd=workspace, env=environment, check=True)

    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".dvc").is_dir():
        dvc("init", "--no-scm")
    (workspace / ".dvc/config").write_text(
        "[core]\n"
        "    no_scm = True\n"
        f"    remote = {REMOTE_NAME}\n"
        f"['remote \"{REMOTE_NAME}\"']\n"
        f"    url = {store.remote}\n"
        "[cache]\n"
        f"    dir = {store.cache}\n"
        "    shared = group\n"
        "    type = symlink\n"
    )
    yield dvc


@contextmanager
def _cache_lock(cache: Path) -> Iterator[None]:
    """Serialize writers of one shared cache across workspaces and nodes."""
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = cache / CACHE_LOCK_NAME
    lock_path.touch(mode=0o664, exist_ok=True)
    with lock_path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
