from pathlib import Path
import subprocess

import pytest

from data.preprocess.config import PreprocessingConfig
from models.configs import EpiForecasterConfig, ObservationHeadConfig
from training.region2vec_trainer import RegionTrainerConfig
from scripts.hpo.ablation_worker import load_campaign


# Config file lists by type
EPIFORECASTER_TRAINING_CONFIGS = [
    "configs/train_epifor_smoke.yaml",
    "configs/production_only/train_epifor_mn5_full.yaml",
    "configs/production_only/train_epifor_mn5_synth_pretrain.yaml",
    "configs/production_only/train_epifor_smoke_netcdf.yaml",
    "configs/experiment/sirhd_latent_rescaling_baseline.yaml",
    "configs/experiment/sirhd_latent_rescaling_std.yaml",
    "configs/experiment/sirhd_latent_rescaling_asinh.yaml",
    "configs/experiment/sirhd_latent_rescaling_unscaled.yaml",
]

REGION2VEC_CONFIGS = [
    "configs/train_regions.yaml",
]

PREPROCESSING_CONFIGS = [
    "configs/preprocess_full.yaml",
    "configs/production_only/preprocess_mn5_synth.yaml",
    "configs/production_only/preprocess_mn5_synth_holt_damped.yaml",
    "configs/production_only/preprocess_mn5_synth_sirhd_reporting_v1.yaml",
    "configs/production_only/preprocess_mn5_synth_netcdf_smoke.yaml",
]

CAMPAIGN_CONFIGS = [
    "configs/campaigns/ablation_default_v1.yaml",
    "configs/campaigns/ablation_smoke_v1.yaml",
]


@pytest.mark.parametrize("campaign_path", CAMPAIGN_CONFIGS)
def test_ablation_campaign_configs_parse(campaign_path):
    """Campaign YAML has its own schema; it is not a training configuration."""
    campaign = load_campaign(Path(campaign_path))
    assert campaign.conditions
    assert campaign.baseline_condition in campaign.conditions_by_name
    assert campaign.equal_budget_policy.paths == (
        "training.epochs",
        "training.max_batches",
    )


@pytest.mark.epiforecaster
@pytest.mark.parametrize("config_path", EPIFORECASTER_TRAINING_CONFIGS)
def test_epiforecaster_training_configs_parse(config_path):
    """Test that all EpiForecaster training configs can be parsed."""
    cfg = EpiForecasterConfig.load(config_path)
    # Basic validation that the config loaded
    assert cfg.model is not None
    assert cfg.data is not None
    assert cfg.training is not None
    assert cfg.output is not None
    # Verify that the config uses joint_inference loss (required by EpiForecasterTrainer)
    assert cfg.training.loss.name == "joint_inference"


@pytest.mark.epiforecaster
def test_epiforecaster_rejects_non_joint_loss_name():
    with pytest.raises(
        ValueError, match="only supports training.loss.name='joint_inference'"
    ):
        EpiForecasterConfig.load(
            "configs/train_epifor_smoke.yaml",
            overrides=["training.loss.name=mse"],
        )


@pytest.mark.region
@pytest.mark.parametrize("config_path", REGION2VEC_CONFIGS)
def test_region2vec_configs_parse(config_path):
    """Test that all Region2Vec configs can be parsed."""
    cfg = RegionTrainerConfig.from_file(config_path)
    assert cfg.data is not None
    assert cfg.encoder is not None
    assert cfg.training is not None
    assert cfg.sampling is not None
    assert cfg.loss is not None
    assert cfg.output is not None
    assert cfg.clustering is not None


@pytest.mark.parametrize("config_path", PREPROCESSING_CONFIGS)
def test_preprocessing_configs_parse(config_path):
    """Test that all preprocessing configs can be parsed."""
    config_path = Path(config_path)
    config_dict = PreprocessingConfig._load_config_dict(
        config_path,
        seen_paths=set(),
    )

    # Verify required top-level fields exist
    assert "data_dir" in config_dict
    assert "start_date" in config_dict
    assert "end_date" in config_dict

    # Verify date format is valid
    from datetime import datetime

    datetime.fromisoformat(config_dict["start_date"])
    datetime.fromisoformat(config_dict["end_date"])


def test_corrected_sirhd_preprocessing_has_versioned_artifact_identity():
    config = PreprocessingConfig._load_config_dict(
        Path("configs/production_only/preprocess_mn5_synth_sirhd_reporting_v1.yaml"),
        seen_paths=set(),
    )

    assert config["env"] == "mn5"
    assert config["synthetic_path"] == (
        "data/files/raw_synthetic_observations_sirhd_reporting_v1_cb6b4e5.nc"
    )
    assert config["dataset_name"] == (
        "synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped_latent_flow_v2"
    )
    assert config["smoothing"]["clinical_method"] == "holt_damped"
    assert config["continuous_storage_dtype"] == "float32"


def test_all_config_files_accounted_for():
    """Ensure test lists match tracked config files in configs/ directory."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    yaml_files = sorted(
        path for path in result.stdout.splitlines() if Path(path).is_file()
    )

    # Combine all test lists
    tested_files = (
        EPIFORECASTER_TRAINING_CONFIGS
        + REGION2VEC_CONFIGS
        + PREPROCESSING_CONFIGS
        + CAMPAIGN_CONFIGS
    )
    tested_files_sorted = sorted(tested_files)

    # Check for any untested files
    untested = set(yaml_files) - set(tested_files_sorted)
    extra_tests = set(tested_files_sorted) - set(yaml_files)

    assert not untested, f"Config files not tested: {untested}"
    assert not extra_tests, f"Test lists contain non-existent files: {extra_tests}"


@pytest.mark.epiforecaster
@pytest.mark.parametrize("value", [0.0, -1.5])
@pytest.mark.parametrize(
    "field",
    [
        "kernel_gamma_shape_hosp",
        "kernel_gamma_scale_hosp",
        "kernel_gamma_shape_cases",
        "kernel_gamma_scale_cases",
        "kernel_gamma_shape_deaths",
        "kernel_gamma_scale_deaths",
    ],
)
def test_observation_heads_reject_nonpositive_gamma_priors(field, value):
    """All clinical Gamma prior fields must be strictly positive."""
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        ObservationHeadConfig(**{field: value})


@pytest.mark.epiforecaster
def test_observation_heads_reject_wastewater_gamma_parameterization():
    """Wastewater routes through SheddingConvolution, which rejects Gamma."""
    with pytest.raises(ValueError, match="kernel_parameterization_ww"):
        ObservationHeadConfig(kernel_parameterization_ww="gamma")


@pytest.mark.epiforecaster
@pytest.mark.parametrize("head", ["hosp", "cases", "deaths"])
def test_observation_heads_accept_clinical_gamma_parameterization(head):
    """Clinical heads route through DelayKernel, which supports Gamma."""
    cfg = ObservationHeadConfig(**{f"kernel_parameterization_{head}": "gamma"})
    assert getattr(cfg, f"kernel_parameterization_{head}") == "gamma"
