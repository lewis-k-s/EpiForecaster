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


@pytest.mark.epiforecaster
def test_data_config_resolves_cadence_adjusted_observation_thresholds() -> None:
    cfg = DataConfig(
        min_history_observed_fraction=0.5,
        min_horizon_observed_fraction=0.5,
    )
    assert cfg.resolve_min_history_observed_map(input_window_length=60) == {
        "cases": 30,
        "hospitalizations": 5,
        "deaths": 30,
        "wastewater": 5,
    }
    assert cfg.resolve_min_horizon_observed_map(forecast_horizon=28) == {
        "cases": 14,
        "hospitalizations": 2,
        "deaths": 14,
        "wastewater": 2,
    }


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_history_observed_fraction", -0.01),
        ("min_history_observed_fraction", 1.01),
        ("min_horizon_observed_fraction", -0.01),
        ("min_horizon_observed_fraction", 1.01),
    ],
)
def test_data_config_rejects_invalid_observed_fraction(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field_name):
        DataConfig(**{field_name: value})


@pytest.mark.epiforecaster
def test_joint_loss_defaults_use_static_raw_observation_loss() -> None:
    cfg = JointLossConfig()
    assert cfg.observation_loss == "mse"
    assert cfg.adaptive_scheme == "none"
    assert cfg.w_sird_supervision == 0.05


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
