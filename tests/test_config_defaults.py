import pytest

from models.configs import DataConfig, JointLossConfig, ObservationHeadConfig


@pytest.mark.epiforecaster
def test_observation_heads_default_weekly_kernels_frozen() -> None:
    cfg = ObservationHeadConfig()
    assert cfg.learnable_kernel_ww is False
    assert cfg.learnable_kernel_hosp is False
    assert cfg.anchor_mode == "last_valid_step"


@pytest.mark.epiforecaster
def test_data_config_resolves_min_observed_from_missing_permit() -> None:
    cfg = DataConfig(
        missing_permit={
            "input": {
                "cases": 20,
                "hospitalizations": 53,
                "deaths": 20,
                "biomarkers_joint": 53,
            },
            "horizon": {
                "cases": 10,
                "hospitalizations": 26,
                "deaths": 10,
                "biomarkers_joint": 26,
            },
        }
    )
    assert cfg.resolve_min_observed_map(forecast_horizon=28) == {
        "cases": 18,
        "hospitalizations": 2,
        "deaths": 18,
        "wastewater": 2,
    }


@pytest.mark.epiforecaster
def test_joint_loss_defaults_use_balanced_n_eff_scaling() -> None:
    cfg = JointLossConfig()
    assert cfg.obs_n_eff_power == 0.5
    assert cfg.obs_n_eff_reference == 28.0
    assert cfg.ww_n_eff_reference == 0.0
    assert cfg.hosp_n_eff_reference == 0.0
    assert cfg.cases_n_eff_reference == 0.0
    assert cfg.deaths_n_eff_reference == 0.0
