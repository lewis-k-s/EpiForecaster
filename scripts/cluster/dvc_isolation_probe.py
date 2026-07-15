"""Verify one MN5 DVC workspace and write an isolation result manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xarray as xr
import yaml

RAW_PATH = Path("data/files/mobility.zarr")
PROCESSED_PATH = Path("data/processed/real_with_id.zarr")
RAW_DVC_PATH = RAW_PATH.with_suffix(".zarr.dvc")
PROCESSED_DVC_PATH = PROCESSED_PATH.with_suffix(".zarr.dvc")


def read_dvc_output(path: Path) -> dict[str, int | str]:
    document = yaml.safe_load(path.read_text())
    output = document["outs"][0]
    return {
        "md5": output["md5"],
        "size": int(output["size"]),
        "nfiles": int(output["nfiles"]),
    }


def inspect_zarr(path: Path) -> dict[str, object]:
    dataset = xr.open_zarr(path)
    try:
        return {
            "sizes": dict(dataset.sizes),
            "data_vars": sorted(dataset.data_vars),
        }
    finally:
        dataset.close()


def count_cache_links(path: Path) -> int:
    return sum(
        file_path.is_symlink()
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def run_probe(run_id: str, project_root: Path) -> Path:
    project_root = project_root.resolve()
    raw_path = project_root / RAW_PATH
    processed_path = project_root / PROCESSED_PATH

    if not raw_path.is_dir() or not processed_path.is_dir():
        raise FileNotFoundError("DVC did not materialize both pilot Zarr datasets")

    foreign_canaries = sorted(
        path.name
        for path in processed_path.glob(".dvc-isolation-canary-*")
        if path.name != f".dvc-isolation-canary-{run_id}"
    )
    if foreign_canaries:
        raise RuntimeError(f"Foreign workspace canaries found: {foreign_canaries}")

    canary = processed_path / f".dvc-isolation-canary-{run_id}"
    canary.write_text(f"{run_id}\n")

    raw_links = count_cache_links(raw_path)
    processed_links = count_cache_links(processed_path)
    if raw_links == 0 or processed_links == 0:
        raise RuntimeError("Pilot datasets are not linked to the shared DVC cache")

    result = {
        "run_id": run_id,
        "project_root": str(project_root),
        "raw": {
            "dvc": read_dvc_output(project_root / RAW_DVC_PATH),
            "zarr": inspect_zarr(raw_path),
            "cache_links": raw_links,
        },
        "processed": {
            "dvc": read_dvc_output(project_root / PROCESSED_DVC_PATH),
            "zarr": inspect_zarr(processed_path),
            "cache_links": processed_links,
            "canaries": sorted(
                path.name
                for path in processed_path.glob(".dvc-isolation-canary-*")
            ),
        },
    }

    output_dir = project_root / "outputs" / "dvc-smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_probe(args.run_id, args.project_root)


if __name__ == "__main__":
    main()
