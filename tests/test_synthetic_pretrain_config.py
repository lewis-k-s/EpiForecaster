from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from data.preprocess.config import PreprocessingConfig
from models.configs import EpiForecasterConfig
from scripts.workflow.run_config import validate_workflow_contract


A97C812_RAW_POINTER = (
    "data/files/epiforecaster_sirhd_v1_zarr_a97c812_reactive_n100.zarr.zip.dvc"
)
A97C812_DATASET_NAME = "synthetic_sirhd_v1_a97c812_reactive_n100_holt_damped"
A97C812_PROCESSED_PATH = f"data/processed/{A97C812_DATASET_NAME}.zarr.zip"
SIRHD_V1_CONTRACT_SHA256 = (
    "d8837d63a35a36c1d095184d58c622880b78c5f4ee85d83370c9e9dac6ffe582"
)


def test_synthetic_pretraining_keeps_fractional_mobility_edges() -> None:
    config = EpiForecasterConfig.load(
        Path(
            "configs/production_only/train_epifor_mn5_synth_pretrain_reactive_n100.yaml"
        )
    )

    assert config.data.mobility_min_flow_raw == 0.0
    assert config.model.max_neighbors == 20
    assert config.data.dataset_path == A97C812_PROCESSED_PATH
    assert config.model.observation_heads.relevant_source_mapping() == {
        "hosp": "hospitalization_flow",
        "ww": "I",
        "cases": "I",
        "deaths": "death_flow",
    }


def test_synthetic_pretrain_production_emits_evaluation_diagnostics() -> None:
    config = yaml.safe_load(
        Path("workflow/config/mn5_synth_pretrain_reactive_n100.yaml").read_text()
    )
    training = EpiForecasterConfig.load(
        Path(
            "configs/production_only/train_epifor_mn5_synth_pretrain_reactive_n100.yaml"
        )
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


def test_reactive_finetune_matches_from_scratch_control() -> None:
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


def test_reactive_finetune_smoke_matches_from_scratch_smoke() -> None:
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


@pytest.mark.parametrize(
    "workflow_path",
    [
        "workflow/config/mn5_synth_pretrain_reactive_n100.yaml",
        "workflow/config/mn5_synth_pretrain_reactive_n100_interactive_debug.yaml",
    ],
)
def test_reactive_pretraining_consumes_reactive_corpus(workflow_path: str) -> None:
    workflow = yaml.safe_load(Path(workflow_path).read_text())
    training = EpiForecasterConfig.load(
        Path(workflow["training"]["base_config"]),
        overrides=workflow["training"]["overrides"],
    )

    assert validate_workflow_contract(workflow) == "train"
    assert f"{A97C812_PROCESSED_PATH}.dvc" in workflow["dvc"]["targets"]
    assert workflow["dvc"]["provenance_targets"] == [A97C812_RAW_POINTER]
    assert training.data.dataset_path == A97C812_PROCESSED_PATH
    assert training.data.mobility_min_flow_raw == 0.0
    assert training.training.loss.joint.latent.total_weight > 0.0
    (phase,) = training.training.curriculum.schedule
    assert phase.end_epoch == training.training.epochs


def test_reactive_interactive_debug_keeps_production_loader_settings() -> None:
    workflow = yaml.safe_load(
        Path(
            "workflow/config/mn5_synth_pretrain_reactive_n100_interactive_debug.yaml"
        ).read_text()
    )
    base = EpiForecasterConfig.load(Path(workflow["training"]["base_config"]))
    debug = EpiForecasterConfig.load(
        Path(workflow["training"]["base_config"]),
        overrides=workflow["training"]["overrides"],
    )

    assert workflow["workflow"]["target"] == "train"
    assert workflow["training"]["max_batches"] == 300
    assert debug.training.compile is False
    assert debug.training.profiler.enabled is False
    for field in ("batch_size", "num_workers", "prefetch_factor", "pin_memory"):
        assert getattr(debug.training, field) == getattr(base.training, field)


def test_a97c812_raw_pointer_pins_seed_capped_zarr_corpus() -> None:
    pointer = yaml.safe_load(Path(A97C812_RAW_POINTER).read_text())

    assert pointer["frozen"] is True
    (dependency,) = pointer["deps"]
    assert dependency["repo"]["url"] == "../EpiSim.jl"
    assert dependency["repo"]["rev_lock"] == "323d78a2c2064ad31d077a939d83e3c3a2837a6f"
    (output,) = pointer["outs"]
    assert output["md5"] == "cfdd2a02c7a81a8f9e008339b693b1a5"
    assert output["size"] == 46612607993
    assert not Path(A97C812_RAW_POINTER.removesuffix(".dvc")).exists()


def test_a97c812_preprocessing_pins_corpus_and_state_space_contract() -> None:
    workflow = yaml.safe_load(
        Path(
            "workflow/config/mn5_preprocess_sirhd_v1_a97c812_reactive_n100.yaml"
        ).read_text()
    )
    preprocessing = workflow["preprocessing"]
    # Parse without input-path validation: the corpus stays on GPFS.
    config = PreprocessingConfig._load_config_dict(
        Path(preprocessing["base_config"]), seen_paths=set()
    )
    provenance = config["source_provenance"]

    assert validate_workflow_contract(workflow) == "preprocess"
    assert "handoff_manifest" not in preprocessing
    assert A97C812_RAW_POINTER in workflow["dvc"]["targets"]
    assert config["synthetic_path"] == A97C812_RAW_POINTER.removesuffix(".dvc")
    assert preprocessing["dataset_name"] == config["dataset_name"]
    assert config["dataset_name"] == A97C812_DATASET_NAME
    assert "synthetic_run_exclusions" not in config
    assert preprocessing["expected_run_count"] == 200
    assert config["bitround_keepbits"] == {"mobility": 10}
    assert config["synthetic_state_space_contract_sha256"] == SIRHD_V1_CONTRACT_SHA256
    assert config["start_date"] == "2020-02-10"
    assert config["end_date"] == "2021-05-09"
    assert provenance["raw_corpus_dvc_md5"] == "cfdd2a02c7a81a8f9e008339b693b1a5"
