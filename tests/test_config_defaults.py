import pytest

from models.configs import (
    DataConfig,
    EpiForecasterConfig,
    JointLossConfig,
    LatentSupervisionConfig,
    ModelConfig,
    ObservationHeadConfig,
    TrainingParams,
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
    assert cfg.relevant_source_mapping() == {
        "hosp": "hospitalization_flow",
        "ww": "I",
        "cases": "I",
        "deaths": "death_flow",
    }


@pytest.mark.epiforecaster
def test_observation_heads_default_clinical_simplex_kernels() -> None:
    """Every clinical head uses a simplex kernel by default."""
    cfg = ObservationHeadConfig()
    assert cfg.learnable_kernel_hosp is False
    assert cfg.learnable_kernel_cases is True
    assert cfg.learnable_kernel_deaths is True
    assert cfg.kernel_parameterization_hosp == "simplex"
    assert cfg.kernel_parameterization_cases == "simplex"
    assert cfg.kernel_parameterization_deaths == "simplex"
    # Each head keeps a Gamma prior available for an explicit Gamma override.
    assert cfg.kernel_gamma_shape_deaths == pytest.approx(12.25)
    assert cfg.kernel_gamma_scale_deaths == pytest.approx(4.0 / 7.0)


@pytest.mark.epiforecaster
def test_observation_heads_default_clinical_gamma_priors() -> None:
    """Hospitalization/cases expose the implicit Gamma(5, 2) initializer."""
    cfg = ObservationHeadConfig()
    assert cfg.kernel_gamma_shape_hosp == pytest.approx(5.0)
    assert cfg.kernel_gamma_scale_hosp == pytest.approx(2.0)
    assert cfg.kernel_gamma_shape_cases == pytest.approx(5.0)
    assert cfg.kernel_gamma_scale_cases == pytest.approx(2.0)
    # Current cases/hospitalization kernel behavior is unchanged.
    assert cfg.learnable_kernel_hosp is False
    assert cfg.kernel_parameterization_hosp == "simplex"
    assert cfg.learnable_kernel_cases is True
    assert cfg.kernel_parameterization_cases == "simplex"
    # Prior implies mean delay 10 days, std sqrt(5) * 2 days.
    assert cfg.kernel_gamma_shape_hosp * cfg.kernel_gamma_scale_hosp == pytest.approx(
        10.0
    )
    assert (
        cfg.kernel_gamma_shape_hosp** 0.5 * cfg.kernel_gamma_scale_hosp
        == pytest.approx(4.47213595499958)
    )


@pytest.mark.epiforecaster
def test_observation_heads_allow_compartment_plot_sources() -> None:
    cfg = ObservationHeadConfig(
        relevant_source_hosp="H",
        relevant_source_deaths="D",
    )
    assert cfg.relevant_source_mapping() == {
        "hosp": "H",
        "ww": "I",
        "cases": "I",
        "deaths": "D",
    }


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("relevant_source_hosp", "D"),
        ("relevant_source_ww", "H"),
        ("relevant_source_cases", "hospitalization_flow"),
        ("relevant_source_deaths", "H"),
    ],
)
def test_observation_heads_reject_incompatible_plot_sources(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ObservationHeadConfig(**{field_name: value})


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
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("mobility_min_flow_raw", -0.01),
        ("mobility_min_flow_raw", float("nan")),
        ("mobility_weight_clip_max", 0.0),
        ("mobility_weight_clip_max", float("inf")),
        ("mobility_weight_clip_max", 65_505.0),
    ],
)
def test_data_config_rejects_invalid_mobility_processing_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        DataConfig(**{field_name: value})


@pytest.mark.epiforecaster
def test_training_config_rejects_negative_split_seed() -> None:
    with pytest.raises(ValueError, match="split_seed"):
        TrainingParams(split_seed=-1)


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    "obsolete_field",
    ["mobility_threshold", "mobility_clip_range", "mobility_scale_epsilon"],
)
def test_from_dict_rejects_obsolete_mobility_fields(
    obsolete_field: str,
) -> None:
    config = {
        "model": {},
        "data": {obsolete_field: 1.0},
        "training": {},
        "output": {},
    }

    with pytest.raises(ValueError, match="Obsolete mobility configuration"):
        EpiForecasterConfig.from_dict(config)


@pytest.mark.epiforecaster
def test_joint_loss_defaults_use_normal_nll() -> None:
    cfg = JointLossConfig()
    assert cfg.observation_loss == "nll"
    assert cfg.distribution_family == "normal"
    assert cfg.distribution_df == pytest.approx(5.0)
    assert cfg.interval_levels == [0.5, 0.8, 0.95]
    assert cfg.adaptive_scheme == "none"
    assert cfg.latent.total_weight == 0.0
    assert cfg.latent.scaling == "none"
    assert cfg.latent.component_weights == {
        "s": 0.2,
        "i": 0.2,
        "h": 0.2,
        "r": 0.2,
        "death_flow": 0.2,
    }


@pytest.mark.epiforecaster
def test_latent_supervision_config_validates_scaling_contract() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        LatentSupervisionConfig(
            component_weights={
                "s": 1.0,
                "i": 1.0,
                "h": 1.0,
                "r": 1.0,
                "death_flow": 1.0,
            }
        )
    with pytest.raises(ValueError, match="finite and positive"):
        LatentSupervisionConfig(
            scale_floors={
                "s": 0.0,
                "i": 1.0e-8,
                "h": 1.0e-8,
                "r": 1.0e-8,
                "death_flow": 1.0e-8,
            }
        )


@pytest.mark.epiforecaster
def test_joint_loss_distributional_validation() -> None:
    cfg = JointLossConfig(observation_loss="nll", distribution_family="student_t")
    assert cfg.distribution_family == "student_t"

    for objective in ("crps", "wis"):
        cfg = JointLossConfig(
            observation_loss=objective, distribution_family="student_t"
        )
        assert cfg.observation_loss == objective


@pytest.mark.epiforecaster
def test_joint_loss_rejects_gradnorm_with_signed_nll() -> None:
    with pytest.raises(ValueError, match="requires a non-negative observation loss"):
        JointLossConfig(observation_loss="nll", adaptive_scheme="gradnorm")

    cfg = JointLossConfig(observation_loss="crps", adaptive_scheme="gradnorm")
    assert cfg.adaptive_scheme == "gradnorm"


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
