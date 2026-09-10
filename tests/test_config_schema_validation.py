from pathlib import Path

import pytest

from data.preprocess.config import PreprocessingConfig
from models.configs import EpiForecasterConfig, ObservationHeadConfig
from training.region2vec_trainer import RegionTrainerConfig
from scripts.hpo.ablation_worker import load_campaign


CONFIG_ROOT = Path("configs")
EPIFORECASTER_TRAINING_CONFIGS = sorted(
    [
        *CONFIG_ROOT.glob("train_epifor*.yaml"),
        *CONFIG_ROOT.glob("production_only/train_epifor*.yaml"),
        *CONFIG_ROOT.glob("experiment/*.yaml"),
    ]
)
REGION2VEC_CONFIGS = sorted(CONFIG_ROOT.glob("train_regions*.yaml"))
PREPROCESSING_CONFIGS = sorted(
    [
        *CONFIG_ROOT.glob("preprocess*.yaml"),
        *CONFIG_ROOT.glob("production_only/preprocess*.yaml"),
    ]
)
CAMPAIGN_CONFIGS = sorted(CONFIG_ROOT.glob("campaigns/*.yaml"))


@pytest.mark.integration
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


def test_default_ablation_campaign_uses_gamma_for_clinical_delay_arms():
    """Delay-family ablations stay normalized and diagnosable."""
    campaign = load_campaign(Path("configs/campaigns/ablation_default_v1.yaml"))
    conditions = campaign.conditions_by_name

    assert {
        "kernel:ww:gamma",
        "kernel:hosp:gamma",
        "kernel:cases:gamma",
        "kernel:deaths:gamma",
        "kernel:clinical:gamma",
    }.issubset(conditions)
    assert not {
        "kernel:ww:free",
        "kernel:hosp:free",
        "kernel:cases:free",
        "kernel:deaths:free",
        "kernel:all:free",
    }.intersection(conditions)
    assert conditions["kernel:hosp:gamma"].overrides == {
        "model": {
            "observation_heads": {
                "learnable_kernel_hosp": True,
                "kernel_parameterization_hosp": "gamma",
            }
        }
    }
    assert conditions["kernel:clinical:gamma"].overrides == {
        "model": {
            "observation_heads": {
                "learnable_kernel_hosp": True,
                "kernel_parameterization_hosp": "gamma",
                "kernel_parameterization_cases": "gamma",
                "kernel_parameterization_deaths": "gamma",
            }
        }
    }
    assert conditions["kernel:ww:gamma"].overrides == {
        "model": {
            "observation_heads": {
                "learnable_kernel_ww": True,
                "kernel_parameterization_ww": "gamma",
                "kernel_length_ww": 35,
            }
        }
    }


@pytest.mark.epiforecaster
@pytest.mark.integration
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
@pytest.mark.integration
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


@pytest.mark.integration
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

@pytest.mark.epiforecaster
@pytest.mark.parametrize("value", [0.0, -1.5])
@pytest.mark.parametrize(
    "field",
    [
        "kernel_gamma_shape_ww",
        "kernel_gamma_scale_ww",
        "kernel_gamma_shape_hosp",
        "kernel_gamma_scale_hosp",
        "kernel_gamma_shape_cases",
        "kernel_gamma_scale_cases",
        "kernel_gamma_shape_deaths",
        "kernel_gamma_scale_deaths",
    ],
)
def test_observation_heads_reject_out_of_box_gamma_priors(field, value):
    """All Gamma priors must be finite and strictly inside their model boxes."""
    with pytest.raises(ValueError, match=f"{field} must be finite and strictly inside"):
        ObservationHeadConfig(**{field: value})


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kernel_gamma_shape_ww", 30.0),
        ("kernel_gamma_scale_ww", 0.2),
        ("kernel_gamma_scale_ww", 15.0),
        ("kernel_gamma_scale_ww", 16.0),
        ("kernel_gamma_scale_ww", float("nan")),
    ],
)
def test_observation_heads_reject_wastewater_gamma_priors_outside_bounds(field, value):
    """WW Gamma priors must fit the same open box used by its kernel."""
    with pytest.raises(ValueError, match=f"{field} must be finite and strictly inside"):
        ObservationHeadConfig(**{field: value})


@pytest.mark.epiforecaster
def test_observation_heads_accept_wastewater_gamma_parameterization():
    """Wastewater routes through SheddingConvolution, which supports Gamma."""
    cfg = ObservationHeadConfig(kernel_parameterization_ww="gamma")
    assert cfg.kernel_parameterization_ww == "gamma"
    assert cfg.kernel_gamma_shape_ww == pytest.approx(2.0)
    assert cfg.kernel_gamma_scale_ww == pytest.approx(6.0)


@pytest.mark.epiforecaster
@pytest.mark.parametrize("head", ["hosp", "cases", "deaths"])
def test_observation_heads_accept_clinical_gamma_parameterization(head):
    """Clinical heads route through DelayKernel, which supports Gamma."""
    cfg = ObservationHeadConfig(**{f"kernel_parameterization_{head}": "gamma"})
    assert getattr(cfg, f"kernel_parameterization_{head}") == "gamma"
