"""Config composition is resolved once before immutable preparation."""

from pathlib import Path

import pytest
import yaml

from scripts.workflow.config_templates import (
    apply_template_overrides,
    load_config_template,
)


def write_yaml(root: Path, name: str, value: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))
    return path


def test_bases_and_variant_merge_overrides_by_key(tmp_path):
    write_yaml(tmp_path, "inputs.yaml", {"dvc": {"targets": ["raw.dvc"]}})
    write_yaml(
        tmp_path,
        "base.yaml",
        {"training": {"overrides": ["native.decoder=sir", "training.epochs=3"]}},
    )
    template = write_yaml(
        tmp_path,
        "train.yaml",
        {
            "extends": ["inputs.yaml", "base.yaml"],
            "variants": {
                "direct": {
                    "dvc": {"targets": ["causal.dvc"]},
                    "training": {"overrides": ["native.decoder=direct"]},
                }
            },
        },
    )
    resolved = load_config_template(template, root=tmp_path, variant="direct")
    assert resolved == {
        "dvc": {"targets": ["causal.dvc"]},
        "training": {"overrides": ["native.decoder=direct", "training.epochs=3"]},
    }
    for variant in (None, "missing"):
        with pytest.raises(ValueError, match="Select a template variant"):
            load_config_template(template, root=tmp_path, variant=variant)


def test_template_bases_reject_cycles_and_escape(tmp_path):
    template = write_yaml(tmp_path, "base.yaml", {"extends": "base.yaml"})
    with pytest.raises(ValueError, match="Cyclic"):
        load_config_template(template, root=tmp_path)
    for reference in ("../outside.yaml", str(tmp_path.parent / "outside.yaml")):
        write_yaml(tmp_path, "base.yaml", {"extends": reference})
        with pytest.raises(ValueError, match="escapes"):
            load_config_template(template, root=tmp_path)


def test_preparation_overrides_do_not_change_input_or_identity_blocks():
    original = {"resources": {"allocation": {"runtime": 30}}}
    changed = apply_template_overrides(original, ["resources.allocation.runtime=8"])
    assert changed["resources"]["allocation"]["runtime"] == 8
    assert original["resources"]["allocation"]["runtime"] == 30
    for option in ("run.id=bad", "source.git_sha=bad", "container.sif=bad"):
        with pytest.raises(ValueError, match="cannot set"):
            apply_template_overrides(original, [option])


def test_model_base_resolves_from_the_snapshot(tmp_path):
    from models.configs import EpiForecasterConfig

    (tmp_path / "pyproject.toml").write_text("")
    scientific = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/native_base.yaml").read_text()
    )
    scientific["training"]["epochs"] = 7
    write_yaml(tmp_path, "configs/base.yaml", scientific)
    template = write_yaml(
        tmp_path, "configs/train.yaml", {"extends": "configs/base.yaml"}
    )
    assert EpiForecasterConfig.load(template).training.epochs == 7


def test_render_freezes_variant_and_preparation_overrides(tmp_path):
    import shutil
    import subprocess

    from scripts.workflow.run_config import render_config

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "pyproject.toml").write_text("")
    source_root = Path(__file__).resolve().parents[1]
    for name in (
        "configs/native_base.yaml",
        "configs/train_epifor_smoke.yaml",
        "data/processed/real_with_id.zarr.dvc",
    ):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, destination)
    template = write_yaml(
        repo,
        "workflow/train.yaml",
        {
            "workflow": {"kind": "train", "target": "train"},
            "dvc": {"targets": ["data/processed/real_with_id.zarr.dvc"]},
            "evaluation": {"split": "test", "granular": False},
            "training": {
                "base_config": "configs/train_epifor_smoke.yaml",
                "experiment_name": "test",
                "model_id": "model",
                "overrides": ["native.wastewater_enabled=false"],
            },
            "resources": {
                "allocation": {
                    "account": "bsc08",
                    "partition": "acc",
                    "qos": "acc_debug",
                    "nodes": 1,
                    "tasks": 1,
                    "cpus_per_task": 20,
                    "gres": "gpu:1",
                    "runtime": 30,
                }
            },
            "variants": {"bounded": {"training": {"max_batches": 2}}},
        },
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    run = tmp_path / "test"
    output = run / "config.yaml"
    frozen = render_config(
        template_path=template,
        variant="bounded",
        overrides=["resources.allocation.runtime=8"],
        output_path=output,
        dispatch_file_path=run / "dispatch.tsv",
        file_list_path=run / "files.bin",
        snapshot_root=run / "source",
        repo_root=repo,
        run_id="test",
        run_root=run,
        container_sif=Path("/gpfs/runtime.sif"),
        dvc_remote=tmp_path / "remote",
        dvc_cache=tmp_path / "cache",
        workflow_tooling_key="test",
        snakemake_version="9.23.1",
        snakemake_bin=Path("/usr/local/bin/snakemake"),
    )
    assert frozen["training"]["max_batches"] == 2
    assert frozen["resources"]["allocation"]["runtime"] == 8
    assert "extends" not in frozen and "variants" not in frozen
    assert frozen["provenance"]["template"]["variant"] == "bounded"
    template.write_text("training: {}\n")
    assert yaml.safe_load(output.read_text()) == frozen
