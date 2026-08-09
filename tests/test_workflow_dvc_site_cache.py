from __future__ import annotations

from pathlib import Path

import yaml

from scripts.workflow import run_support


def _write_pointer(path: Path, *, md5: str, output_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": md5,
                        "size": 1,
                        "hash": "md5",
                        "path": output_name,
                    }
                ]
            }
        )
    )


def test_stage_dvc_keeps_site_cache_inside_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    pointer = source_root / "data/processed/synthetic.zarr.dvc"
    _write_pointer(pointer, md5="synthetic.dir", output_name="synthetic.zarr")

    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "dvc": {
                    "targets": ["data/processed/synthetic.zarr.dvc"],
                    "remote": str(tmp_path / "remote"),
                    "cache": str(tmp_path / "cache"),
                }
            }
        )
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path | None = None) -> None:
        assert cwd is not None
        commands.append(command)
        if command[1:3] == ["init", "--no-scm"]:
            (cwd / ".dvc").mkdir()
        elif command[1] == "pull":
            (cwd / "data/processed/synthetic.zarr").mkdir(parents=True)

    monkeypatch.setattr(run_support, "_run", fake_run)

    run_support.stage_dvc(config_path, source_root, run_root)

    assert [
        "dvc",
        "config",
        "core.site_cache_dir",
        ".dvc/tmp",
    ] in commands


def test_stage_dvc_validates_git_import_from_the_pinned_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    pointer = source_root / "data/files/imported.nc.dvc"
    payload = b"pinned-import-object"
    payload_path = tmp_path / "payload"
    payload_path.write_bytes(payload)
    _write_pointer(
        pointer,
        md5=run_support.md5_file(payload_path),
        output_name="imported.nc",
    )
    document = yaml.safe_load(pointer.read_text())
    document["deps"] = [
        {
            "path": "data/datasets/imported.nc",
            "repo": {"url": "../upstream", "rev_lock": "deadbeef"},
        }
    ]
    pointer.write_text(yaml.safe_dump(document, sort_keys=False))

    config_path = run_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "dvc": {
                    "targets": ["data/files/imported.nc.dvc"],
                    "remote": str(tmp_path / "remote"),
                    "cache": str(tmp_path / "cache"),
                }
            }
        )
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path | None = None) -> None:
        assert cwd is not None
        commands.append(command)
        if command[1:3] == ["init", "--no-scm"]:
            (cwd / ".dvc").mkdir()
        elif command[1] == "pull":
            output = cwd / "data/files/imported.nc"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)

    monkeypatch.setattr(run_support, "_run", fake_run)

    run_support.stage_dvc(config_path, source_root, run_root)

    assert not any(command[1] == "status" for command in commands)
