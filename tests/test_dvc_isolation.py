from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from scripts.cluster import dvc_isolation_probe
from scripts.cluster.verify_dvc_isolation import verify_pair


RAW_MD5 = "raw.dir"
PROCESSED_MD5 = "processed.dir"


def _write_dvc_file(path: Path, *, md5: str, size: int, nfiles: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": md5,
                        "size": size,
                        "nfiles": nfiles,
                        "hash": "md5",
                        "path": path.name.removesuffix(".dvc"),
                    }
                ]
            },
            sort_keys=False,
        )
    )


def _write_pilot_data(project_root: Path) -> None:
    raw_path = project_root / dvc_isolation_probe.RAW_PATH
    processed_path = project_root / dvc_isolation_probe.PROCESSED_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    xr.Dataset(
        {"mobility": (("date", "origin", "target"), np.ones((1, 2, 2)))}
    ).to_zarr(raw_path, zarr_format=2)
    xr.Dataset(
        {"cases": (("run_id", "date", "region_id"), np.ones((1, 2, 2)))}
    ).to_zarr(processed_path, zarr_format=2)

    _write_dvc_file(
        project_root / dvc_isolation_probe.RAW_DVC_PATH,
        md5=RAW_MD5,
        size=10,
        nfiles=2,
    )
    _write_dvc_file(
        project_root / dvc_isolation_probe.PROCESSED_DVC_PATH,
        md5=PROCESSED_MD5,
        size=20,
        nfiles=3,
    )


def test_probe_records_dataset_identity_and_workspace_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    _write_pilot_data(project_root)
    monkeypatch.setattr(dvc_isolation_probe, "count_cache_links", lambda _: 4)

    result_path = dvc_isolation_probe.run_probe("run-a", project_root)

    result = json.loads(result_path.read_text())
    assert result["run_id"] == "run-a"
    assert result["raw"]["dvc"]["md5"] == RAW_MD5
    assert result["processed"]["dvc"]["md5"] == PROCESSED_MD5
    assert result["processed"]["canaries"] == [
        ".dvc-isolation-canary-run-a"
    ]
    assert result["raw"]["zarr"]["sizes"] == {
        "date": 1,
        "origin": 2,
        "target": 2,
    }


def _write_result(project_root: Path, run_id: str, canaries: list[str]) -> None:
    result = {
        "run_id": run_id,
        "project_root": str(project_root.resolve()),
        "raw": {
            "dvc": {"md5": RAW_MD5},
            "zarr": {"sizes": {"date": 1}, "data_vars": ["mobility"]},
            "cache_links": 2,
        },
        "processed": {
            "dvc": {"md5": PROCESSED_MD5},
            "zarr": {"sizes": {"date": 1}, "data_vars": ["cases"]},
            "cache_links": 3,
            "canaries": canaries,
        },
    }
    output_dir = project_root / "outputs" / "dvc-smoke" / run_id
    output_dir.mkdir(parents=True)
    (output_dir / "result.json").write_text(json.dumps(result))


def test_verify_pair_accepts_isolated_workspaces(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_result(left_root, "run-a", [".dvc-isolation-canary-run-a"])
    _write_result(right_root, "run-b", [".dvc-isolation-canary-run-b"])

    summary = verify_pair(left_root, right_root, RAW_MD5, PROCESSED_MD5)

    assert summary["status"] == "passed"
    assert summary["cross_run_canaries"] is False
    assert summary["cross_run_outputs"] is False


def test_verify_pair_rejects_cross_run_canary(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_result(
        left_root,
        "run-a",
        [".dvc-isolation-canary-run-a", ".dvc-isolation-canary-run-b"],
    )
    _write_result(right_root, "run-b", [".dvc-isolation-canary-run-b"])

    with pytest.raises(RuntimeError, match="cross-run canaries"):
        verify_pair(left_root, right_root, RAW_MD5, PROCESSED_MD5)


def test_mn5_run_rejects_unsafe_run_id() -> None:
    script = Path("scripts/cluster/mn5_run.sh")
    result = subprocess.run(
        ["bash", str(script), "path", "../unsafe"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Invalid run id" in result.stderr


def test_mn5_prepare_builds_isolated_sync_and_dvc_pull(
    tmp_path: Path,
) -> None:
    command_log = tmp_path / "commands.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command_name in ("ssh", "rsync"):
        command_path = bin_dir / command_name
        command_path.write_text(
            "#!/bin/bash\n"
            f"printf '{command_name} %s\\n' \"$*\" >> \"$MN5_TEST_LOG\"\n"
        )
        command_path.chmod(0o755)

    cluster_conf = tmp_path / "mn5.conf"
    cluster_conf.write_text(
        "COMPUTE_HOST=mn5-test\n"
        "TRANSFER_HOST=dt-test\n"
        "REMOTE_RUN_ROOT=/nfs/epiforecaster-runs\n"
        "DVC_REMOTE_PATH=/nfs/epiforecaster-dvc/storage\n"
        "DVC_CACHE_PATH=/nfs/epiforecaster-dvc/cache\n"
        "REMOTE_DVC_BIN=/nfs/env/bin/dvc\n"
        "REMOTE_PYTHON_BIN=/nfs/env/bin/python\n"
    )
    script = Path("scripts/cluster/mn5_run.sh").resolve()
    environment = {
        "PATH": f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}",
        "CLUSTER_CONF": str(cluster_conf),
        "MN5_TEST_LOG": str(command_log),
    }

    result = subprocess.run(
        ["bash", str(script), "prepare", "test-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text()
    assert "dt-test:/nfs/epiforecaster-runs/test-run/project/" in commands
    assert "--exclude=.git " in commands
    assert "--exclude=.git/" not in commands
    assert "git init -q" in commands
    assert "/nfs/env/bin/dvc" in commands
    assert "flock -x" in commands
    assert "/nfs/epiforecaster-dvc/cache/.dvc-pull.lock" in commands
    assert "pull data/files/mobility.zarr.dvc" in commands
    assert "data/processed/real_with_id.zarr.dvc" in commands
