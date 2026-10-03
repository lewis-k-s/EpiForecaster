"""Validate templates alongside the pending workflow configuration changes."""

from pathlib import Path
import pytest
from models.configs import EpiForecasterConfig
from scripts.workflow.config_templates import load_config_template
import yaml
from scripts.workflow.run_config import (
    validate_mn5_allocation,
    validate_workflow_contract,
)
from scripts.hpo.ablation_worker import load_campaign

CONFIG_ROOT = Path("configs")
EPIFORECASTER_TRAINING_CONFIGS = sorted(
    [
        *CONFIG_ROOT.glob("train_epifor*.yaml"),
        *CONFIG_ROOT.glob("production_only/train_epifor*.yaml"),
    ]
)
CAMPAIGN_CONFIGS = sorted(CONFIG_ROOT.glob("campaigns/*.yaml"))
WORKFLOW_CONFIGS = []
for path in sorted(Path("workflow/config").glob("*.yaml")) + sorted(
    Path("workflow/config/reproductions").glob("*.yaml")
):
    variants = yaml.safe_load(path.read_text()).get("variants", {None: {}})
    WORKFLOW_CONFIGS.extend((path, name) for name in variants)


@pytest.mark.parametrize("path,variant", WORKFLOW_CONFIGS)
def test_workflow_templates_resolve_and_validate(path, variant):
    workflow = load_config_template(path, variant=variant)
    if workflow.get("evaluation", {}).get("checkpoint", False) is None:
        with pytest.raises(ValueError, match="checkpoint must be an absolute"):
            validate_workflow_contract(workflow)
        workflow["evaluation"]["checkpoint"] = "/gpfs/example/checkpoints/best_model.pt"
    validate_workflow_contract(workflow)
    validate_mn5_allocation(workflow)
    for section_name in ("training", "crossval", "optuna"):
        section = workflow.get(section_name, {})
        if "base_config" in section:
            config = EpiForecasterConfig.load(
                section["base_config"], overrides=section.get("overrides", [])
            )
            assert config.native.enabled
            assert config.native.spatial_encoder == "hetero"
            if config.native.wastewater_enabled:
                assert (
                    f"{config.native.root}/{config.native.wastewater_filename}.dvc"
                    in workflow["dvc"]["targets"]
                )
    for target in workflow["dvc"]["targets"] + workflow["dvc"].get(
        "provenance_targets", []
    ):
        assert Path(target).is_file()


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


def test_direct_throughput_config_is_a_paired_decoder_change():
    import copy
    from scripts.workflow.run_config import (
        validate_mn5_allocation,
        validate_workflow_contract,
    )

    template = Path("workflow/config/mn5_native_performance.yaml")
    baseline = load_config_template(template, variant="all_causal")
    direct = load_config_template(template, variant="direct")
    for workflow in (baseline, direct):
        assert validate_workflow_contract(workflow) == "train"
        validate_mn5_allocation(workflow)
    normalized = copy.deepcopy(direct)
    normalized["training"]["experiment_name"] = baseline["training"]["experiment_name"]
    normalized["training"]["overrides"] = [
        "native.compile_sir_rollout=true"
        if option == "native.compile_sir_rollout=false"
        else option
        for option in normalized["training"]["overrides"]
        if not option.startswith("native.decoder=")
    ]
    assert normalized == baseline
    configs = [
        EpiForecasterConfig.load(
            Path(workflow["training"]["base_config"]),
            overrides=workflow["training"]["overrides"],
        )
        for workflow in (baseline, direct)
    ]
    assert configs[1].native.decoder == "direct"
    assert configs[1].native.compile_sir_rollout is False
    assert configs[1].native.mobility_enabled and configs[1].native.include_vaccination
    configs[1].native.decoder = "sir"
    configs[1].native.compile_sir_rollout = True
    assert configs[1].to_dict() == configs[0].to_dict()
