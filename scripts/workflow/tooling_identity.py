"""Compute the content identity of the reusable Snakemake environment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def workflow_tooling_key(lock_path: Path, bootstrap_path: Path) -> str:
    """Hash the inputs that define the coordinator environment."""
    lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    bootstrap_digest = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    material = "\n".join(
        (
            "python=3.13",
            "dependency_group=workflow",
            f"uv.lock={lock_digest}",
            f"bootstrap={bootstrap_digest}",
            "",
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    args = parser.parse_args()
    print(workflow_tooling_key(args.lock_file, args.bootstrap))


if __name__ == "__main__":
    main()
