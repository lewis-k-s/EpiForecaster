"""Real DVC stores for workflow tests: a local remote and cache under tmp."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.workflow import dvc_workspace


def publish_pointer(
    store: dvc_workspace.DvcStore,
    scratch: Path,
    source_root: Path,
    pointer: str,
    payload: bytes | dict[str, bytes],
) -> dvc_workspace.Pointer:
    """Push a file or directory payload and place its pointer in a source tree."""
    relative = Path(pointer)
    name = relative.name.removesuffix(".dvc")
    workspace = scratch / relative.parent
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / name
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        for member, content in payload.items():
            (target / member).parent.mkdir(parents=True, exist_ok=True)
            (target / member).write_bytes(content)
    published = dvc_workspace.publish(store, workspace, name)
    destination = source_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace / f"{name}.dvc", destination)
    return published


def remote_object(store: dvc_workspace.DvcStore, md5: str) -> Path:
    """Path of one content-addressed object in a DVC 3 local remote."""
    return store.remote / "files/md5" / md5[:2] / md5[2:]
