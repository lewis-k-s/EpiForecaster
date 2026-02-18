"""Tests for OmegaConf-based config loading with overrides."""

import pytest
from omegaconf import errors

from models.configs import EpiForecasterConfig

BASE_LOCAL_CONFIG = "configs/train_epifor_real_local.yaml"


@pytest.mark.epiforecaster
def test_load_config_with_simple_overrides():
    """Test that simple dotted overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=["training.learning_rate=0.0005", "data.log_scale=false"],
    )

    assert cfg.training.learning_rate == 0.0005
    assert cfg.data.log_scale is False


@pytest.mark.epiforecaster
def test_load_config_with_nested_overrides():
    """Test that nested overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=[
            "model.type.cases=true",
            "model.type.regions=false",
            "model.type.biomarkers=true",
        ],
    )

    assert cfg.model.type.cases is True
    assert cfg.model.type.regions is False
    assert cfg.model.type.biomarkers is True


@pytest.mark.epiforecaster
def test_load_config_with_bool_override():
    """Test that boolean overrides are handled correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=["training.plot_forecasts=false", "data.log_scale=true"],
    )

    assert cfg.training.plot_forecasts is False
    assert cfg.data.log_scale is True


@pytest.mark.epiforecaster
def test_load_config_without_overrides():
    """Test that config loads correctly without overrides."""
    cfg = EpiForecasterConfig.load(BASE_LOCAL_CONFIG)

    assert cfg.training.learning_rate == 0.0001
    assert cfg.data.log_scale is True


@pytest.mark.epiforecaster
def test_load_config_strict_mode_rejects_unknown_keys():
    """Test that strict mode rejects unknown keys."""
    with pytest.raises((errors.ConfigKeyError, errors.ValidationError, KeyError)):
        EpiForecasterConfig.load(
            BASE_LOCAL_CONFIG,
            overrides=["training.unknown_field=123"],
            strict=True,
        )


@pytest.mark.epiforecaster
def test_load_config_model_type_overrides():
    """Test that model.type overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=[
            "model.type.cases=true",
            "model.type.regions=true",
            "model.type.biomarkers=true",
            "model.type.mobility=true",
        ],
    )

    assert cfg.model.type.cases is True
    assert cfg.model.type.regions is True
    assert cfg.model.type.biomarkers is True
    assert cfg.model.type.mobility is True


@pytest.mark.epiforecaster
def test_load_config_with_gnn_depth_override():
    """Test that GNN depth overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=["model.gnn_depth=3"],
    )

    assert cfg.model.gnn_depth == 3


@pytest.mark.epiforecaster
def test_load_config_with_multiple_overrides_same_section():
    """Test that multiple overrides in the same section work correctly."""
    cfg = EpiForecasterConfig.load(
        BASE_LOCAL_CONFIG,
        overrides=[
            "training.learning_rate=0.0001",
            "training.batch_size=64",
            "training.epochs=50",
            "training.early_stopping_patience=15",
        ],
    )

    assert cfg.training.learning_rate == 0.0001
    assert cfg.training.batch_size == 64
    assert cfg.training.epochs == 50
    assert cfg.training.early_stopping_patience == 15


@pytest.mark.epiforecaster
def test_backward_compatibility_from_file():
    """Test that from_file() method still works."""
    cfg = EpiForecasterConfig.from_file(BASE_LOCAL_CONFIG)

    assert cfg.training.learning_rate == 0.0001
    assert cfg.data.log_scale is True


@pytest.mark.epiforecaster
def test_from_dict_reconstructs_config():
    """Test that from_dict() correctly reconstructs a config from to_dict()."""
    # Load a config and convert to dict (as would be saved in checkpoint)
    original = EpiForecasterConfig.load(BASE_LOCAL_CONFIG)
    config_dict = original.to_dict()

    # Reconstruct from dict (as would be loaded from checkpoint)
    reconstructed = EpiForecasterConfig.from_dict(config_dict)

    # Verify all top-level fields match
    assert reconstructed.model.type == original.model.type
    assert reconstructed.model.gnn_depth == original.model.gnn_depth
    assert reconstructed.data.dataset_path == original.data.dataset_path
    assert reconstructed.data.log_scale == original.data.log_scale
    assert reconstructed.training.learning_rate == original.training.learning_rate
    assert reconstructed.training.batch_size == original.training.batch_size
    assert reconstructed.output.log_dir == original.output.log_dir
