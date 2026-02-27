import pytest

from models.configs import DataConfig, JointLossConfig, ObservationHeadConfig


@pytest.mark.epiforecaster
def test_joint_loss_imputed_weights_default_to_point_zero_one() -> None:
    cfg = JointLossConfig()
    assert cfg.resolve_imputed_weight_map() == {
        "wastewater": 0.01,
        "hospitalizations": 0.01,
        "cases": 0.01,
        "deaths": 0.01,
    }


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
