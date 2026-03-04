import pytest

from models.configs import DataConfig, JointLossConfig, ObservationHeadConfig


@pytest.mark.epiforecaster
def test_observation_heads_default_weekly_kernels_frozen() -> None:
    cfg = ObservationHeadConfig()
    assert cfg.learnable_kernel_ww is False
    assert cfg.learnable_kernel_hosp is False


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
    assert cfg.horizon_norm_enabled is True
    assert cfg.horizon_norm_ema_decay == 0.9
    assert cfg.horizon_norm_eps == 1.0e-6
    assert cfg.horizon_norm_scale_floor == 1.0e-3
    assert cfg.horizon_weight_mode == "exp_decay"
    assert cfg.horizon_weight_gamma == 0.85
    assert cfg.horizon_weight_power == 1.0


@pytest.mark.epiforecaster
def test_joint_loss_horizon_weight_mode_validation() -> None:
    cfg = JointLossConfig(horizon_weight_mode="exp_growth")
    assert cfg.horizon_weight_mode == "exp_growth"
    with pytest.raises(ValueError):
        JointLossConfig(horizon_weight_mode="bad_mode")
