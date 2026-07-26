from pathlib import Path

import pytest
import yaml

from models.configs import EpiForecasterConfig
from scripts.workflow.run_config import validate_workflow_contract


CORRECTED_PROCESSED_PATH = (
    "data/processed/synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped.zarr"
)
CORRECTED_PROCESSED_POINTER = f"{CORRECTED_PROCESSED_PATH}.dvc"
CORRECTED_RAW_POINTER = (
    "data/files/"
    "raw_synthetic_observations_sirhd_reporting_v1_cb6b4e5.zarr.dvc"
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
        "workflow/config/mn5_synth_pretrain_interactive_smoke.yaml",
        "workflow/config/mn5_synth_pretrain_compiled_smoke.yaml",
    ],
)
def test_live_synthetic_pretraining_workflows_use_corrected_corpus(
    workflow_path: str,
) -> None:
    workflow = yaml.safe_load(Path(workflow_path).read_text())

    assert CORRECTED_PROCESSED_POINTER in workflow["dvc"]["targets"]
    assert workflow["dvc"]["provenance_targets"] == [CORRECTED_RAW_POINTER]


def test_synthetic_pretrain_interactive_smoke_disables_all_compilation() -> None:
    config = yaml.safe_load(
        Path(
            "workflow/config/mn5_synth_pretrain_interactive_smoke.yaml"
        ).read_text()
    )

    overrides = set(config["training"]["overrides"])
    assert "training.compile=false" in overrides
    assert "training.compile_backward=false" in overrides
    assert "training.pathway_gradient_frequency=1" in overrides
    assert config["workflow"]["target"] == "train_evaluate"
    assert config["evaluation"]["granular"] is True


def test_synthetic_pretrain_production_emits_evaluation_diagnostics() -> None:
    config = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain.yaml").read_text()
    )
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


def test_control_synthetic_cross_evaluation_uses_pretraining_data_contract() -> None:
    workflow = yaml.safe_load(
        Path(
            "workflow/config/mn5_sirhd_reporting_v1_control_evaluate.yaml"
        ).read_text()
    )
    overrides = workflow["evaluation"]["overrides"]
    evaluation = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_full.yaml"),
        overrides=overrides,
    )

    assert validate_workflow_contract(workflow) == "evaluate"
    assert Path(workflow["evaluation"]["checkpoint"]).is_absolute()
    assert workflow["evaluation"]["granular"] is True
    assert evaluation.data.dataset_path == CORRECTED_PROCESSED_PATH
    assert evaluation.data.mobility_min_flow_raw == 0.0
    assert evaluation.data.run_id_chunk_size == 1
    assert evaluation.training.curriculum.enabled is True
    assert evaluation.training.loss.joint.w_sird_supervision == 0.01


def test_old_batch_mixing_smoke_remains_historical() -> None:
    path = Path("workflow/config/mn5_synth_pretrain_batch_mixing_smoke.yaml")
    text = path.read_text()
    workflow = yaml.safe_load(text)

    assert "HISTORICAL SUPERSEDED-CORPUS DIAGNOSTIC" in text
    assert (
        "data/processed/synthetic_full_holt_damped.zarr.dvc"
        in workflow["dvc"]["targets"]
    )


def test_batch_mixing_smoke_shuffles_single_time_major_batches() -> None:
    workflow = yaml.safe_load(
        Path(
            "workflow/config/mn5_synth_pretrain_batch_mixing_smoke.yaml"
        ).read_text()
    )
    training = EpiForecasterConfig.load(
        Path("configs/production_only/train_epifor_mn5_synth_pretrain.yaml"),
        overrides=workflow["training"]["overrides"],
    )

    assert validate_workflow_contract(workflow) == "train"
    assert workflow["workflow"]["target"] == "train"
    assert workflow["training"]["max_batches"] == 1000
    assert training.training.batch_size == 32
    assert training.data.sample_ordering == "time"
    assert training.training.curriculum.chunk_size == 32
    assert training.training.progress_log_frequency == 20
