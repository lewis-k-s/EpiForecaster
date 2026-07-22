import pytest

from models.configs import (
    DataConfig,
    JointLossConfig,
    ModelConfig,
    ObservationHeadConfig,
)


@pytest.mark.epiforecaster
def test_observation_heads_default_weekly_kernels_frozen() -> None:
    cfg = ObservationHeadConfig()
    assert cfg.learnable_kernel_ww is False
    assert cfg.learnable_kernel_hosp is False
    assert cfg.kernel_parameterization_ww == "simplex"
    assert cfg.kernel_parameterization_hosp == "simplex"
    assert cfg.anchor_mode == "last_valid_step"
    assert cfg.distribution_scale_init == pytest.approx(0.3)
    assert cfg.distribution_scale_floor == pytest.approx(1.0e-3)


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
def test_joint_loss_defaults_use_static_raw_observation_loss() -> None:
    cfg = JointLossConfig()
    assert cfg.observation_loss == "mse"
    assert cfg.distribution_family == "normal"
    assert cfg.distribution_df == pytest.approx(5.0)
    assert cfg.distribution_scale_floor == pytest.approx(1.0e-3)
    assert cfg.interval_levels == [0.5, 0.8, 0.95]
    assert cfg.adaptive_scheme == "none"
    assert cfg.w_sird_supervision == 0.05


@pytest.mark.epiforecaster
def test_joint_loss_distributional_validation() -> None:
    cfg = JointLossConfig(observation_loss="nll", distribution_family="student_t")
    assert cfg.distribution_family == "student_t"

    with pytest.raises(ValueError, match="currently requires"):
        JointLossConfig(observation_loss="crps", distribution_family="student_t")


@pytest.mark.epiforecaster
def test_model_graph_adjacency_source_defaults_to_mobility() -> None:
    cfg = ModelConfig(
        type={"cases": True, "mobility": True},
        mobility_embedding_dim=4,
        region_embedding_dim=4,
        input_window_length=3,
        forecast_horizon=1,
        max_neighbors=2,
        gnn_module="gcn",
    )
    assert cfg.graph_adjacency_source == "mobility"


@pytest.mark.epiforecaster
def test_model_temporal_covariates_dim_includes_lockdown_severity() -> None:
    cfg = ModelConfig(
        type={"cases": True, "mobility": True},
        mobility_embedding_dim=4,
        region_embedding_dim=4,
        input_window_length=3,
        forecast_horizon=1,
        max_neighbors=2,
        gnn_module="gcn",
        include_day_of_week=True,
        include_holidays=True,
        include_lockdown_severity=True,
    )
    assert cfg.temporal_covariates_dim == 4


@pytest.mark.epiforecaster
def test_model_graph_adjacency_source_accepts_spatial_knn() -> None:
    cfg = ModelConfig(
        type={"cases": True, "mobility": True},
        mobility_embedding_dim=4,
        region_embedding_dim=4,
        input_window_length=3,
        forecast_horizon=1,
        max_neighbors=2,
        gnn_module="gcn",
        graph_adjacency_source="spatial_knn",
    )
    assert cfg.graph_adjacency_source == "spatial_knn"


@pytest.mark.epiforecaster
def test_model_graph_adjacency_source_accepts_spatial_queen() -> None:
    cfg = ModelConfig(
        type={"cases": True, "mobility": True},
        mobility_embedding_dim=4,
        region_embedding_dim=4,
        input_window_length=3,
        forecast_horizon=1,
        max_neighbors=2,
        gnn_module="gcn",
        graph_adjacency_source="spatial_queen",
    )
    assert cfg.graph_adjacency_source == "spatial_queen"


@pytest.mark.epiforecaster
def test_model_graph_adjacency_source_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="graph_adjacency_source"):
        ModelConfig(
            type={"cases": True, "mobility": True},
            mobility_embedding_dim=4,
            region_embedding_dim=4,
            input_window_length=3,
            forecast_horizon=1,
            max_neighbors=2,
            gnn_module="gcn",
            graph_adjacency_source="unknown",
        )
