from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from data.preprocess.config import PreprocessingConfig
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


def test_latent_rescaling_study_arms_change_only_identity_and_scaling() -> None:
    paths = {
        mode: Path(f"workflow/config/mn5_latent_rescaling_{mode}_mse.yaml")
        for mode in ("unscaled", "std", "asinh")
    }
    documents = {
        mode: yaml.safe_load(path.read_text(encoding="utf-8"))
        for mode, path in paths.items()
    }

    expected_modes = {"unscaled": "none", "std": "std", "asinh": "asinh"}
    normalized = {}
    for arm, document in documents.items():
        assert document["training"]["max_batches"] is None
        assert "training.epochs=1" in document["training"]["overrides"]
        scaling_override = f"training.loss.joint.latent.scaling={expected_modes[arm]}"
        assert scaling_override in document["training"]["overrides"]
        copy = deepcopy(document)
        copy["training"]["experiment_name"] = "ARM"
        copy["training"]["overrides"].remove(scaling_override)
        normalized[arm] = copy

    assert normalized["unscaled"] == normalized["std"] == normalized["asinh"]


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
        Path("workflow/config/mn5_synth_pretrain_interactive_smoke.yaml").read_text()
    )

    overrides = set(config["training"]["overrides"])
    assert "training.compile=false" in overrides
    assert "training.compile_backward=false" in overrides
    assert "training.pathway_gradient_frequency=1" in overrides
    assert config["workflow"]["target"] == "train_evaluate"
    assert config["evaluation"]["granular"] is True


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


def test_control_synthetic_cross_evaluation_uses_pretraining_data_contract() -> None:
    workflow = yaml.safe_load(
        Path("workflow/config/mn5_sirhd_reporting_v1_control_evaluate.yaml").read_text()
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
    assert evaluation.training.loss.joint.latent.total_weight == 0.0


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
        Path("workflow/config/mn5_synth_pretrain_batch_mixing_smoke.yaml").read_text()
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


REACTIVE_RAW_POINTER = (
    "data/files/epiforecaster_sirhd_v1_netcdf_d9c5bae_reactive_n100.nc.dvc"
)
REACTIVE_DATASET_NAME = "synthetic_sirhd_v1_d9c5bae_reactive_n100_holt_damped"
REACTIVE_PROCESSED_PATH = f"data/processed/{REACTIVE_DATASET_NAME}.nc"
SIRHD_V1_CONTRACT_SHA256 = (
    "d8837d63a35a36c1d095184d58c622880b78c5f4ee85d83370c9e9dac6ffe582"
)


def test_reactive_raw_pointer_pins_upstream_corpus_without_payload() -> None:
    pointer = yaml.safe_load(Path(REACTIVE_RAW_POINTER).read_text())

    assert pointer["frozen"] is True
    (dependency,) = pointer["deps"]
    assert dependency["repo"]["url"] == "../EpiSim.jl"
    assert dependency["repo"]["rev_lock"] == "cd60c2524856ec69f4927b2cb71702d15809d50a"
    (output,) = pointer["outs"]
    assert output["md5"] == "a30e21844967255fa43a15157e49ccfa"
    assert output["size"] == 133926669691
    assert not Path(REACTIVE_RAW_POINTER.removesuffix(".dvc")).exists()


def test_reactive_preprocessing_pins_state_space_contract() -> None:
    workflow = yaml.safe_load(
        Path("workflow/config/mn5_preprocess_sirhd_v1_reactive_n100.yaml").read_text()
    )
    # Parse without input-path validation: the corpus stays on GPFS.
    config = PreprocessingConfig._load_config_dict(
        Path(workflow["preprocessing"]["base_config"]), seen_paths=set()
    )

    assert validate_workflow_contract(workflow) == "preprocess"
    assert REACTIVE_RAW_POINTER in workflow["dvc"]["targets"]
    assert workflow["preprocessing"]["expected_run_count"] == 200
    assert workflow["preprocessing"]["dataset_name"] == REACTIVE_DATASET_NAME
    assert config["dataset_name"] == REACTIVE_DATASET_NAME
    assert config["synthetic_path"] == REACTIVE_RAW_POINTER.removesuffix(".dvc")
    assert config["synthetic_state_space_contract_sha256"] == SIRHD_V1_CONTRACT_SHA256
    assert config["start_date"] == "2020-02-10"
    assert config["end_date"] == "2021-05-09"
    assert config["source_provenance"]["raw_corpus_dvc_md5"] == (
        "a30e21844967255fa43a15157e49ccfa"
    )


B354B27_RAW_POINTER = (
    "data/files/epiforecaster_sirhd_v1_netcdf_b354b27_reactive_n100.nc.dvc"
)
B354B27_PROCESSED_PATH = (
    "data/processed/synthetic_sirhd_v1_b354b27_reactive_n100_excl_p13_holt_damped.nc"
)


def test_b354b27_processed_pointer_pins_published_dataset() -> None:
    pointer = yaml.safe_load(Path(f"{B354B27_PROCESSED_PATH}.dvc").read_text())

    (output,) = pointer["outs"]
    assert output["md5"] == "946dae8597ffe4fce705b0ddedf2131d"
    assert output["size"] == 117834036904
    assert output["path"] == Path(B354B27_PROCESSED_PATH).name


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
    assert f"{B354B27_PROCESSED_PATH}.dvc" in workflow["dvc"]["targets"]
    assert workflow["dvc"]["provenance_targets"] == [B354B27_RAW_POINTER]
    assert training.data.dataset_path == B354B27_PROCESSED_PATH
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
