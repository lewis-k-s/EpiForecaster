"""Defaults and validation that determine scientific model behavior."""

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
def test_observation_head_defaults_and_source_validation() -> None:
    config = ObservationHeadConfig()
    assert (
        config.kernel_parameterization_ww,
        config.kernel_parameterization_hosp,
        config.kernel_parameterization_cases,
        config.kernel_parameterization_deaths,
    ) == ("simplex",) * 4
    assert (
        config.learnable_kernel_ww,
        config.learnable_kernel_hosp,
        config.learnable_kernel_cases,
        config.learnable_kernel_deaths,
    ) == (False, False, True, True)
    assert config.anchor_mode == "last_valid_step"
    assert config.distribution_scale_init == pytest.approx(0.3)
    assert config.distribution_scale_floor == pytest.approx(1e-3)
    assert config.relevant_source_mapping() == {
        "hosp": "hospitalization_flow",
        "ww": "I",
        "cases": "I",
        "deaths": "death_flow",
    }
    assert (
        (config.kernel_gamma_shape_hosp, config.kernel_gamma_scale_hosp),
        (config.kernel_gamma_shape_cases, config.kernel_gamma_scale_cases),
        (config.kernel_gamma_shape_deaths, config.kernel_gamma_scale_deaths),
    ) == ((5.0, 2.0), (5.0, 2.0), (12.25, 4.0 / 7.0))

    alternative = ObservationHeadConfig(
        relevant_source_hosp="H", relevant_source_deaths="D"
    )
    assert alternative.relevant_source_mapping() == {
        "hosp": "H",
        "ww": "I",
        "cases": "I",
        "deaths": "D",
    }
    for field, value in (
        ("relevant_source_hosp", "D"),
        ("relevant_source_ww", "H"),
        ("relevant_source_cases", "hospitalization_flow"),
        ("relevant_source_deaths", "H"),
    ):
        with pytest.raises(ValueError, match=field):
            ObservationHeadConfig(**{field: value})


@pytest.mark.epiforecaster
def test_data_thresholds_use_observation_cadence() -> None:
    config = DataConfig(
        min_history_observed_fraction=0.5,
        min_horizon_observed_fraction=0.5,
    )
    assert config.resolve_min_history_observed_map(input_window_length=60) == {
        "cases": 30,
        "hospitalizations": 5,
        "deaths": 30,
        "wastewater": 5,
    }
    assert config.resolve_min_horizon_observed_map(forecast_horizon=28) == {
        "cases": 14,
        "hospitalizations": 2,
        "deaths": 14,
        "wastewater": 2,
    }

    invalid = (
        ("min_history_observed_fraction", -0.01),
        ("min_history_observed_fraction", 1.01),
        ("min_horizon_observed_fraction", -0.01),
        ("min_horizon_observed_fraction", 1.01),
        ("mobility_min_flow_raw", -0.01),
        ("mobility_min_flow_raw", float("nan")),
        ("mobility_weight_clip_max", 0.0),
        ("mobility_weight_clip_max", float("inf")),
        ("mobility_weight_clip_max", 65_505.0),
    )
    for field, value in invalid:
        with pytest.raises(ValueError, match=field):
            DataConfig(**{field: value})


@pytest.mark.epiforecaster
def test_model_graph_and_temporal_options_validate() -> None:
    options = {
        "type": {"cases": True, "mobility": True},
        "mobility_embedding_dim": 4,
        "region_embedding_dim": 4,
        "input_window_length": 3,
        "forecast_horizon": 1,
        "max_neighbors": 2,
        "gnn_module": "gcn",
    }
    default = ModelConfig(**options)
    assert default.graph_adjacency_source == "mobility"
    for source in ("spatial_knn", "spatial_queen"):
        assert (
            ModelConfig(**options, graph_adjacency_source=source).graph_adjacency_source
            == source
        )
    with pytest.raises(ValueError, match="graph_adjacency_source"):
        ModelConfig(**options, graph_adjacency_source="unknown")
    temporal = ModelConfig(
        **options,
        include_day_of_week=True,
        include_holidays=True,
        include_lockdown_severity=True,
    )
    assert temporal.temporal_covariates_dim == 4


@pytest.mark.epiforecaster
def test_joint_loss_defaults_and_distribution_validation() -> None:
    config = JointLossConfig()
    assert (config.observation_loss, config.distribution_family) == ("nll", "normal")
    assert config.distribution_df == pytest.approx(5.0)
    assert config.interval_levels == [0.5, 0.8, 0.95]
    assert config.adaptive_scheme == "none"
    assert (config.latent.total_weight, config.latent.scaling) == (0.0, "none")
    assert config.latent.component_weights == {
        "s": 0.2,
        "i": 0.2,
        "h": 0.2,
        "r": 0.2,
        "death_flow": 0.2,
    }
    for objective in ("nll", "crps"):
        assert (
            JointLossConfig(
                observation_loss=objective, distribution_family="student_t"
            ).observation_loss
            == objective
        )
    with pytest.raises(ValueError, match="observation_loss"):
        JointLossConfig(observation_loss="wis", distribution_family="student_t")
    with pytest.raises(ValueError, match="distribution_df=5.0"):
        JointLossConfig(
            observation_loss="crps",
            distribution_family="student_t",
            distribution_df=7.0,
        )
    with pytest.raises(ValueError, match="requires a non-negative observation loss"):
        JointLossConfig(observation_loss="nll", adaptive_scheme="gradnorm")
    assert (
        JointLossConfig(
            observation_loss="crps", adaptive_scheme="gradnorm"
        ).adaptive_scheme
        == "gradnorm"
    )


@pytest.mark.epiforecaster
def test_latent_scaling_and_obsolete_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        LatentSupervisionConfig(
            component_weights=dict.fromkeys(("s", "i", "h", "r", "death_flow"), 1.0)
        )
    with pytest.raises(ValueError, match="finite and positive"):
        LatentSupervisionConfig(
            scale_floors={"s": 0.0, "i": 1e-8, "h": 1e-8, "r": 1e-8, "death_flow": 1e-8}
        )
    for field in (
        "mobility_threshold",
        "mobility_clip_range",
        "mobility_scale_epsilon",
    ):
        with pytest.raises(ValueError, match="Obsolete mobility configuration"):
            EpiForecasterConfig.from_dict(
                {"model": {}, "data": {field: 1.0}, "training": {}, "output": {}}
            )
    with pytest.raises(ValueError, match="split_seed"):
        TrainingParams(split_seed=-1)
