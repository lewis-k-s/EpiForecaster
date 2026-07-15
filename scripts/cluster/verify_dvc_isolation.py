"""Verify that two completed MN5 DVC smoke runs are mutually isolated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(project_root: Path) -> dict:
    smoke_root = project_root / "outputs" / "dvc-smoke"
    result_paths = list(smoke_root.glob("*/result.json"))
    if len(result_paths) != 1:
        raise RuntimeError(
            f"Expected one smoke result under {smoke_root}, found {result_paths}"
        )
    return json.loads(result_paths[0].read_text())


def verify_pair(
    left_root: Path,
    right_root: Path,
    expected_raw_md5: str,
    expected_processed_md5: str,
) -> dict[str, object]:
    left_root = left_root.resolve()
    right_root = right_root.resolve()
    if left_root == right_root:
        raise RuntimeError("Both runs resolve to the same remote project directory")

    left = load_result(left_root)
    right = load_result(right_root)
    if left["run_id"] == right["run_id"]:
        raise RuntimeError("Both results use the same run id")

    expected = {
        "raw": expected_raw_md5,
        "processed": expected_processed_md5,
    }
    for label, result, root, other in (
        ("left", left, left_root, right),
        ("right", right, right_root, left),
    ):
        if Path(result["project_root"]).resolve() != root:
            raise RuntimeError(f"{label} result reports the wrong project root")
        if result["raw"]["dvc"]["md5"] != expected["raw"]:
            raise RuntimeError(f"{label} raw DVC identity does not match the pilot")
        if result["processed"]["dvc"]["md5"] != expected["processed"]:
            raise RuntimeError(
                f"{label} processed DVC identity does not match the pilot"
            )
        if result["raw"]["cache_links"] <= 0 or result["processed"]["cache_links"] <= 0:
            raise RuntimeError(f"{label} did not use the shared DVC cache")

        own_canary = f".dvc-isolation-canary-{result['run_id']}"
        foreign_canary = f".dvc-isolation-canary-{other['run_id']}"
        canaries = result["processed"]["canaries"]
        if canaries != [own_canary] or foreign_canary in canaries:
            raise RuntimeError(f"{label} workspace contains cross-run canaries")

        foreign_output = root / "outputs" / "dvc-smoke" / other["run_id"]
        if foreign_output.exists():
            raise RuntimeError(f"{label} workspace contains the other run's output")

    if left["raw"]["zarr"] != right["raw"]["zarr"]:
        raise RuntimeError("Raw Zarr schemas differ between runs")
    if left["processed"]["zarr"] != right["processed"]["zarr"]:
        raise RuntimeError("Processed Zarr schemas differ between runs")

    summary = {
        "status": "passed",
        "run_ids": [left["run_id"], right["run_id"]],
        "project_roots": [str(left_root), str(right_root)],
        "raw_md5": expected_raw_md5,
        "processed_md5": expected_processed_md5,
        "cross_run_canaries": False,
        "cross_run_outputs": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--expected-raw-md5", required=True)
    parser.add_argument("--expected-processed-md5", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_pair(
        args.left_root,
        args.right_root,
        args.expected_raw_md5,
        args.expected_processed_md5,
    )


if __name__ == "__main__":
    main()
