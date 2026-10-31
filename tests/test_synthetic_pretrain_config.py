from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from models.configs import EpiForecasterConfig
from scripts.workflow.run_config import validate_workflow_contract


CORRECTED_PROCESSED_PATH = (
    "data/processed/synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped_latent_flow_v2.nc"
)
CORRECTED_PROCESSED_POINTER = f"{CORRECTED_PROCESSED_PATH}.dvc"
CORRECTED_RAW_POINTER = (
    "data/files/raw_synthetic_observations_sirhd_reporting_v1_cb6b4e5.nc.dvc"
)


def test_synthetic_pretraining_keeps_fractional_mobility_edges() -> None:
    config = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_synth_pretrain.yaml")
    )

    assert config.data.mobility_min_flow_raw == 0.0
    assert config.model.max_neighbors == 20
    assert config.data.dataset_path == CORRECTED_PROCESSED_PATH
    assert config.model.observation_heads.relevant_source_mapping() == {
        "hosp": "hospitalization_flow",
        "ww": "I",
        "cases": "I",
        "deaths": "death_flow",
    }


@pytest.mark.parametrize(
    "workflow_path",
    [
        "workflow/config/mn5_synth_pretrain.yaml",
    ],
)
def test_live_synthetic_pretraining_workflows_use_corrected_corpus(
    workflow_path: str,
) -> None:
    workflow = yaml.safe_load(Path(workflow_path).read_text())

    assert CORRECTED_PROCESSED_POINTER in workflow["dvc"]["targets"]
    assert workflow["dvc"]["provenance_targets"] == [CORRECTED_RAW_POINTER]


def test_synthetic_pretrain_production_emits_evaluation_diagnostics() -> None:
    config = yaml.safe_load(Path("workflow/config/mn5_synth_pretrain.yaml").read_text())
    training = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_synth_pretrain.yaml")
    )

    assert config["workflow"]["target"] == "train_evaluate"
    assert config["evaluation"]["granular"] is True
    assert training.training.pathway_gradient_frequency == 500


def test_synthetic_pretrain_control_matches_real_training_contract() -> None:
    control = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_control.yaml").read_text()
    )
    canonical = yaml.safe_load(Path("workflow/config/mn5_train.yaml").read_text())

    assert control["workflow"]["target"] == "train_evaluate"
    assert control["training"]["base_config"] == canonical["training"]["base_config"]
    assert control["training"]["max_batches"] is None
    assert control["training"]["overrides"] == canonical["training"]["overrides"]
    assert control["dvc"] == canonical["dvc"]
    assert control["evaluation"] == canonical["evaluation"]
    assert control["resources"] == canonical["resources"]


def test_reporting_v1_finetune_matches_from_scratch_control() -> None:
    control = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_control.yaml").read_text()
    )
    fine_tune = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_finetune.template.yaml").read_text()
    )

    assert fine_tune["workflow"] == control["workflow"]
    assert fine_tune["dvc"] == control["dvc"]
    assert fine_tune["evaluation"] == control["evaluation"]
    assert fine_tune["resources"] == control["resources"]
    for field in ("base_config", "model_id", "max_batches", "overrides"):
        assert fine_tune["training"][field] == control["training"][field]
    assert fine_tune["training"]["require_init_checkpoint"] is True
    assert fine_tune["training"]["init_checkpoint"] is None

    with pytest.raises(
        ValueError,
        match="training.init_checkpoint is required",
    ):
        validate_workflow_contract(fine_tune)

    rendered = deepcopy(fine_tune)
    rendered["training"]["init_checkpoint"] = (
        "/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/"
        "EpiForecaster-DVC/runs/pretrain/outputs/training/pretrain/model/"
        "checkpoints/best_model.pt"
    )
    assert validate_workflow_contract(rendered) == "train"


def test_reporting_v1_finetune_smoke_matches_from_scratch_smoke() -> None:
    control = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_control_smoke.yaml").read_text()
    )
    fine_tune = yaml.safe_load(
        Path(
            "workflow/config/mn5_synth_pretrain_finetune_smoke.template.yaml"
        ).read_text()
    )

    assert validate_workflow_contract(control) == "train"
    assert fine_tune["workflow"] == control["workflow"]
    assert fine_tune["dvc"] == control["dvc"]
    assert fine_tune["evaluation"] == control["evaluation"]
    assert fine_tune["resources"] == control["resources"]
    for field in ("base_config", "model_id", "max_batches", "overrides"):
        assert fine_tune["training"][field] == control["training"][field]
    assert fine_tune["training"]["require_init_checkpoint"] is True
    assert fine_tune["training"]["init_checkpoint"] is None

    with pytest.raises(
        ValueError,
        match="training.init_checkpoint is required",
    ):
        validate_workflow_contract(fine_tune)


def test_reporting_v1_checkpoint_evaluation_template_fails_closed() -> None:
    template = yaml.safe_load(
        Path(
            "workflow/config/mn5_sirhd_reporting_v1_checkpoint_evaluate.template.yaml"
        ).read_text()
    )

    assert template["evaluation"]["checkpoint"] is None
    assert CORRECTED_PROCESSED_POINTER in template["dvc"]["targets"]
    assert template["dvc"]["provenance_targets"] == [CORRECTED_RAW_POINTER]
    with pytest.raises(
        ValueError,
        match="evaluation.checkpoint must be an absolute GPFS path",
    ):
        validate_workflow_contract(template)

    rendered = deepcopy(template)
    rendered["evaluation"]["checkpoint"] = (
        "/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/"
        "EpiForecaster-DVC/runs/model/outputs/training/model/checkpoints/"
        "best_model.pt"
    )
    assert validate_workflow_contract(rendered) == "evaluate"
