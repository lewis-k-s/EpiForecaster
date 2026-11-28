"""Keep durable stage attempts separate from disposable Snakemake outputs."""

from __future__ import annotations

from collections.abc import Collection
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _identity(path: Path, record: dict) -> dict:
    # Datasets are recorded by size: DVC hashes them once at publication, and
    # hashing a 200 GB payload at every stage boundary costs minutes.
    if "bytes" in record:
        return {"bytes": path.stat().st_size}
    return {"sha256": file_hash(path)}


def new_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="attempt-", dir=root))


def _publish(
    attempt: Path, record: dict, run_root: Path, *, verify: bool = True
) -> None:
    # Validate the entire attempt before publishing any file.
    files = []
    for relative, artifact in record["artifacts"].items():
        source = attempt / artifact["path"]
        destination = run_root / relative
        source.resolve().relative_to(attempt.resolve())
        destination.resolve().relative_to(run_root.resolve())
        expected = {
            key: artifact[key] for key in ("sha256", "bytes") if key in artifact
        }
        if source.is_symlink() or (verify and _identity(source, artifact) != expected):
            raise ValueError(f"Completed attempt failed validation: {source}")
        if destination.is_symlink():
            raise ValueError(f"Attempt output must not be a symlink: {destination}")
        if (
            destination.exists()
            and not os.path.samefile(source, destination)
            and _identity(destination, artifact) != expected
        ):
            raise ValueError(
                f"Attempt output conflicts with an existing file: {destination}"
            )
        files.append((source, destination))
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}")
        try:
            # Both paths belong to the same run filesystem. A hard link avoids
            # copying large datasets/checkpoints and survives output cleanup.
            os.link(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def restore_attempt(root: Path, identity: dict, run_root: Path) -> bool:
    for receipt in sorted(root.glob("attempt-*/complete.json")):
        record = json.loads(receipt.read_text())
        if record["identity"] == identity:
            _publish(receipt.parent, record, run_root)
            return True
    return False


def complete_attempt(
    attempt: Path,
    identity: dict,
    artifacts: dict[Path, Path],
    run_root: Path,
    *,
    size_only: Collection[Path] = (),
) -> None:
    """Record and publish an attempt's artifacts.

    Artifacts are recorded by SHA-256, except the destinations in
    ``size_only``, which are recorded by byte size.
    """
    record = {
        "identity": identity,
        "artifacts": {
            str(destination.relative_to(run_root)): {
                "path": str(source.relative_to(attempt)),
                **(
                    {"bytes": source.stat().st_size}
                    if destination in size_only
                    else {"sha256": file_hash(source)}
                ),
            }
            for destination, source in artifacts.items()
        },
    }
    temporary = attempt / "complete.json.tmp"
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(attempt / "complete.json")
    # This process just recorded its private attempt. Reuse that record on
    # first publication; a later restoration always checks the files again.
    _publish(attempt, record, run_root, verify=False)
