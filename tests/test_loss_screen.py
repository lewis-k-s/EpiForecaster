import copy
from pathlib import Path

import pytest
import yaml

from scripts.research import loss_screen


@pytest.fixture
def protocol():
    return yaml.safe_load(Path("configs/research/loss_protocol_v1.yaml").read_text())


def test_screen_matrix_keeps_comparison_budget_and_roles_fixed(
    monkeypatch, tmp_path, protocol
):
    # Use the current tracked historical pointer only as a renderer fixture.
    # This test does not load, approve, or train on its data.
    pointer = "data/processed/real_with_id.zarr.dvc"
    protocol["data"]["research_dataset_dvc_md5"] = yaml.safe_load(
        Path(pointer).read_text()
    )["outs"][0]["md5"]
    templates = loss_screen.screen_templates(protocol, pointer)
    assert len(templates) == 48
    assert len({name for name, _ in templates}) == 48
    first = templates[0][1]["training"]["overrides"]
    variable = {
        "training.crossval_fold_index",
        "training.seed",
        "model.type.mobility",
        "training.loss.joint.distribution_family",
        "training.loss.joint.observation_loss",
    }
    fixed = [item for item in first if item.split("=")[0] not in variable]
    seen = set()
    for _, template in templates:
        overrides = template["training"]["overrides"]
        assert [
            item for item in overrides if item.split("=")[0] not in variable
        ] == fixed
        values = dict(item.split("=", 1) for item in overrides)
        seen.add(tuple(values[key] for key in sorted(variable)))
        assert values["training.split_strategy"] == "research"
        assert values["training.research_confirmation_enabled"] == "false"
        assert values["training.research_evaluation_role"] == "calibration"
        assert values["training.selection_metric"] == "mean_target_wis"
        assert values["training.epochs"] == "50"
        assert template["training"]["max_batches"] is None
        assert template["evaluation"]["research_outputs"] is True
    assert len(seen) == 48
    changed = copy.deepcopy(protocol)
    changed["budget"]["max_scientific_training_identities"] = 47
    with pytest.raises(ValueError, match="budget"):
        loss_screen.screen_templates(changed, pointer)


def test_screen_refuses_unpinned_dataset(protocol):
    protocol["data"]["research_dataset_dvc_md5"] = None
    with pytest.raises(ValueError, match="pass QA"):
        loss_screen.screen_templates(protocol, "data/processed/real_with_id.zarr.dvc")
    with pytest.raises(ValueError, match="source-relative"):
        loss_screen.screen_templates(protocol, "../data.dvc")


def test_acceptance_covers_family_loss_and_mobility_without_science_budget(protocol):
    pointer = "data/processed/real_with_id.zarr.dvc"
    protocol["data"]["research_dataset_dvc_md5"] = yaml.safe_load(
        Path(pointer).read_text()
    )["outs"][0]["md5"]
    templates = loss_screen.acceptance_templates(protocol, pointer)
    assert len(templates) == 8
    settings = set()
    for identity, template in templates:
        assert identity.startswith("loss-acceptance-")
        assert template["training"]["max_batches"] == 10
        assert template["training"]["experiment_name"].endswith("-acceptance")
        values = dict(item.split("=", 1) for item in template["training"]["overrides"])
        assert values["training.epochs"] == "2"
        assert values["training.early_stopping_patience"] == "1"
        assert values["training.compile"] == "false"
        assert values["training.compile_backward"] == "false"
        assert values["training.crossval_fold_index"] == "1"
        assert values["training.seed"] == "52"
        assert template["resources"]["allocation"]["runtime"] == 30
        settings.add(
            (
                values["training.loss.joint.distribution_family"],
                values["training.loss.joint.observation_loss"],
                values["model.type.mobility"],
            )
        )
    assert len(settings) == 8
    assert protocol["screen"]["max_epochs"] == 50
    assert protocol["screen"]["folds"] == [1, 4, 5]
