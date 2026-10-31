"""Workflow templates may extend one parent; the resolved mapping is standalone."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.workflow.run_config import (
    load_template,
    validate_mn5_allocation,
    validate_workflow_contract,
)

TEMPLATES = sorted(Path("workflow/config").glob("*.yaml"))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_extends_deep_merges_and_child_wins(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a:\n  x: 1\n  y: 2\nb: 3\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\na:\n  y: 9\n  z: 4\n")
    assert load_template(child) == {"a": {"x": 1, "y": 9, "z": 4}, "b": 3}


def test_extends_replaces_lists_and_scalars(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "items: [1, 2, 3]\nname: a\nmap: {k: 1}\n")
    child = _write(
        tmp_path / "child.yaml", "extends: base.yaml\nitems: [4]\nname: null\nmap: 5\n"
    )
    assert load_template(child) == {"items": [4], "name": None, "map": 5}


def test_extends_key_is_absent_from_output(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a: 1\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nb: 2\n")
    assert "extends" not in load_template(child)


def test_extends_chains_through_several_parents(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "v: 1\nk: a\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\nv: 2\n")
    c = _write(tmp_path / "c.yaml", "extends: b.yaml\nw: 3\n")
    assert load_template(c) == {"v": 2, "k": "a", "w": 3}


def test_extends_resolves_relative_to_the_extending_file(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "base.yaml", "a: 1\n")
    child = _write(tmp_path / "sub" / "child.yaml", "extends: ../base.yaml\n")
    assert load_template(child) == {"a": 1}


def test_extends_cycle_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_template(tmp_path / "a.yaml")


def test_extends_missing_parent_fails(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", "extends: nope.yaml\n")
    with pytest.raises(ValueError, match="parent not found"):
        load_template(child)


def test_extends_depth_is_limited(tmp_path: Path) -> None:
    _write(tmp_path / "t0.yaml", "v: 0\n")
    for index in range(1, 7):
        _write(tmp_path / f"t{index}.yaml", f"extends: t{index - 1}.yaml\n")
    with pytest.raises(ValueError, match="depth"):
        load_template(tmp_path / "t6.yaml")


def test_extends_must_be_a_path_string(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", "extends: [a.yaml]\n")
    with pytest.raises(ValueError, match="extends must be a path"):
        load_template(child)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_template_resolves_and_passes_contract(path: Path) -> None:
    document = load_template(path)
    assert "extends" not in document
    # Copy-and-fill templates are incomplete by design.
    if ".template." in path.name or (
        document["workflow"]["kind"] == "evaluate"
        and document["evaluation"].get("checkpoint") is None
    ):
        pytest.skip("copy-and-fill template")
    validate_workflow_contract(document)
    validate_mn5_allocation(document)


def test_ablation_smoke_overlay_resolves_to_frozen_mapping() -> None:
    assert load_template(Path("workflow/config/mn5_ablation_smoke.yaml")) == {
        "run": {},
        "source": {},
        "container": {},
        "workflow": {"kind": "ablation", "target": "ablation"},
        "dvc": {
            "targets": [
                "data/files/mobility.zarr.dvc",
                "data/processed/real_with_id.zarr.dvc",
                "outputs/region_embeddings/region_embeddings.pt.dvc",
            ]
        },
        "ablation": {
            "study_name": "epiforecaster_ablation_smoke_v1",
            "campaign": "configs/campaigns/ablation_smoke_v1.yaml",
            "workers": 2,
            "trials_per_worker": 4,
            "closeout_signal_seconds": 300,
            "timeout_s": 3300,
            "sampler_seed": 42,
            "smoketest": True,
            "recover_stale_running": True,
            "requeue_running": False,
        },
        "resources": {
            "allocation": {
                "account": "bsc08",
                "partition": "acc",
                "qos": "acc_bscls",
                "nodes": 1,
                "tasks": 2,
                "cpus_per_task": 20,
                "gres": "gpu:2",
                "runtime": 60,
            }
        },
    }
