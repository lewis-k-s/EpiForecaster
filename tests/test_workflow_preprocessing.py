"""Focused contracts for immutable preprocessing workflow runs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import xarray as xr
import yaml

from scripts.workflow import run_config, run_support


def _preprocess_workflow() -> dict:
    return {
        "workflow": {"kind": "preprocess", "target": "preprocess"},
        "dvc": {"targets": ["data/files/raw.zarr.dvc"]},
        "preprocessing": {
            "base_config": "configs/production_only/preprocess.yaml",
            "dataset_name": "synthetic_smoke",
            "expected_run_count": 1,
            "required_variables": ["latent_D_true", "latent_death_flow_true"],
            "float32_variables": ["latent_D_true", "latent_death_flow_true"],
            "nonnegative_variables": ["latent_D_true", "latent_death_flow_true"],
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
    }


def test_preprocess_workflow_requires_safe_source_contract() -> None:
    config = _preprocess_workflow()

    assert run_config.validate_workflow_contract(config) == "preprocess"
    assert run_config.workflow_target(config) == "preprocess"
    assert run_config.preprocessing_dataset_name(config) == "synthetic_smoke"

    config["preprocessing"]["base_config"] = "../outside.yaml"
    try:
        run_config.validate_workflow_contract(config)
    except ValueError as error:
        assert "preprocessing.base_config" in str(error)
    else:
        raise AssertionError("Unsafe preprocessing base config was accepted")

    config = _preprocess_workflow()
    config["preprocessing"]["dataset_name"] = "synthetic_smoke.zarr"
    try:
        run_config.validate_workflow_contract(config)
    except ValueError as error:
        assert "preprocessing.dataset_name" in str(error)
    else:
        raise AssertionError("Suffixed preprocessing dataset name was accepted")


def test_processed_dataset_qa_validates_destination_dated_flow() -> None:
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    death_flow = np.array([[[0.0], [0.1], [0.2]]], dtype=np.float32)
    cumulative = np.cumsum(death_flow, axis=1, dtype=np.float32)
    dataset = xr.Dataset(
        {
            "latent_D_true": (("run_id", "date", "region_id"), cumulative),
            "latent_death_flow_true": (
                ("run_id", "date", "region_id"),
                death_flow,
            ),
        },
        coords={"run_id": [0], "date": dates, "region_id": [1]},
        attrs={
            "positive_clinical_preservation_json": json.dumps(
                {
                    "cases": {
                        "raw_positive_count": 3,
                        "processed_positive_count": 3,
                        "zeroed_raw_positive_count": 0,
                    }
                }
            )
        },
    )

    qa = run_support._processed_dataset_qa(dataset, _preprocess_workflow())

    assert qa["dimensions"]["run_id"] == 1
    assert qa["latent_death_flow_alignment"]["comparison_count"] == 2
    assert qa["latent_death_flow_alignment"]["max_absolute_error"] < 1e-7
    assert qa["positive_clinical_preservation"]["cases"] == {
        "raw_positive_count": 3,
        "processed_positive_count": 3,
        "zeroed_raw_positive_count": 0,
    }


def test_run_preprocessing_rejects_partial_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_config = source_root / "configs/production_only/preprocess.yaml"
    source_config.parent.mkdir(parents=True)
    source_config.write_text("dataset_name: synthetic_smoke\n")
    run_root = tmp_path / "run"
    output_root = run_root / "outputs/preprocessing"
    output_root.mkdir(parents=True)
    (output_root / "synthetic_smoke.zarr").mkdir()
    config = _preprocess_workflow()
    config.update(
        {
            "run": {
                "id": "run",
                "root": str(run_root),
                "source_root": str(source_root),
            }
        }
    )
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    try:
        run_support.run_preprocessing(config_path, source_root, run_root)
    except FileExistsError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("Partial preprocessing output was overwritten")


def test_handoff_provenance_qa_retains_run_ids_and_seeds(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    handoff_path = source_root / "data/files/qa/handoff_manifest.json"
    handoff_path.parent.mkdir(parents=True)
    handoff = {
        "qa_passed": True,
        "run_ids": ["run_b", "run_a"],
        "seeds": {"synthetic_reporting_seed": [22, 11]},
        "corpus_full_tree_sha256": "raw-tree",
        "corpus_metadata_tree_sha256": "metadata-tree",
        "reporting_config_sha256": "reporting-config",
    }
    handoff_path.write_text(json.dumps(handoff))
    config = _preprocess_workflow()
    config["preprocessing"]["handoff_manifest"] = "data/files/qa/handoff_manifest.json"
    dataset = xr.Dataset(coords={"run_id": ["run_b", "run_a"]})
    effective = {
        "source_provenance": {
            "handoff_manifest_sha256": run_support.sha256_file(handoff_path)
        }
    }

    qa = run_support._handoff_provenance_qa(
        source_root,
        config,
        effective,
        dataset,
    )

    assert qa is not None
    assert qa["run_ids"] == ["run_b", "run_a"]
    assert qa["seeds"] == {"synthetic_reporting_seed": [22, 11]}
    assert qa["seed_by_run"] == {"synthetic_reporting_seed": {"run_b": 22, "run_a": 11}}


def test_publish_preprocessed_dataset_writes_run_owned_pointer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    source_pointer = source_root / "data/files/raw.zarr.dvc"
    source_pointer.parent.mkdir(parents=True)
    source_pointer.write_text(
        yaml.safe_dump({"outs": [{"md5": "raw.dir", "path": "raw.zarr"}]})
    )
    dataset = run_root / "outputs/preprocessing/dataset.zarr"
    dataset.mkdir(parents=True)
    (dataset / ".zgroup").write_text('{"zarr_format": 2}\n')
    qa = run_root / "outputs/preprocessing/qa.json"
    qa.write_text("{}\n")
    tree_hash = run_support._tree_digest(dataset)
    marker = run_root / "outputs/preprocessing/.preprocessed.done"
    marker.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_tree_sha256": tree_hash,
                "qa": str(qa),
                "qa_sha256": run_support.sha256_file(qa),
            }
        )
    )
    digest_file = tmp_path / "runtime.sif.sha256"
    digest_file.write_text("abc123  runtime.sif\n")
    config = _preprocess_workflow()
    config.update(
        {
            "run": {
                "id": "run",
                "root": str(run_root),
                "source_root": str(source_root),
            },
            "source": {"git_sha": "abc", "source_snapshot_sha256": "def"},
            "container": {
                "sif": str(tmp_path / "runtime.sif"),
                "digest_file": str(digest_file),
            },
            "dvc": {
                "targets": ["data/files/raw.zarr.dvc"],
                "remote": str(tmp_path / "remote"),
                "cache": str(tmp_path / "cache"),
            },
        }
    )
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data_manifest = run_root / ".state/data_manifest.json"
    data_manifest.parent.mkdir()
    data_manifest.write_text(
        json.dumps(
            {
                "targets": {
                    "data/files/raw.zarr.dvc": {
                        "md5": "raw.dir",
                        "output": "data/files/raw.zarr",
                    }
                }
            }
        )
    )
    (run_root / "outputs/preprocessing/publication").mkdir()
    original_copytree = shutil.copytree
    observed_copy_function: list[object] = []

    def observe_copytree(*args: object, **kwargs: object) -> Path:
        observed_copy_function.append(kwargs.get("copy_function"))
        return original_copytree(*args, **kwargs)

    monkeypatch.setattr(run_support.shutil, "copytree", observe_copytree)

    def fake_run(command: list[str], *, cwd: Path | None = None, **_: object) -> None:
        assert cwd is not None
        if command[1:3] == ["init", "--no-scm"]:
            (cwd / ".dvc").mkdir()
        elif command[1:3] == ["add", "synthetic_smoke.zarr"]:
            (cwd / "synthetic_smoke.zarr.dvc").write_text(
                yaml.safe_dump(
                    {"outs": [{"md5": "published.dir", "path": "synthetic_smoke.zarr"}]}
                )
            )

    monkeypatch.setattr(run_support, "_run", fake_run)

    receipt_path = run_support.publish_preprocessed_dataset(config_path, run_root)

    receipt = json.loads(receipt_path.read_text())
    assert receipt["dvc_md5"] == "published.dir"
    assert receipt["input_dvc_hashes"] == {"data/files/raw.zarr.dvc": "raw.dir"}
    assert (receipt_path.parent / "synthetic_smoke.zarr.dvc").is_file()
    assert not (receipt_path.parent / "synthetic_smoke.zarr").exists()
    assert (run_root / ".state/preprocessed_published.done").is_file()
    assert observed_copy_function == [None]


def test_publish_preprocessed_dataset_rejects_nonempty_publication(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output_root = run_root / "outputs/preprocessing"
    dataset = output_root / "dataset.zarr"
    dataset.mkdir(parents=True)
    (dataset / ".zgroup").write_text('{"zarr_format": 2}\n')
    qa = output_root / "qa.json"
    qa.write_text("{}\n")
    marker = output_root / ".preprocessed.done"
    marker.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_tree_sha256": run_support._tree_digest(dataset),
                "qa": str(qa),
                "qa_sha256": run_support.sha256_file(qa),
            }
        )
    )
    publication = output_root / "publication"
    publication.mkdir()
    (publication / "existing").write_text("occupied\n")
    config = _preprocess_workflow()
    config.update(
        {
            "run": {"id": "run", "root": str(run_root)},
            "dvc": {
                "targets": ["data/files/raw.zarr.dvc"],
                "remote": str(tmp_path / "remote"),
                "cache": str(tmp_path / "cache"),
            },
        }
    )
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    try:
        run_support.publish_preprocessed_dataset(config_path, run_root)
    except FileExistsError as error:
        assert "publication already exists" in str(error)
    else:
        raise AssertionError("Nonempty publication directory was accepted")
