"""Tests for OmegaConf-based config loading with overrides."""

import pytest
from omegaconf import errors

from models.configs import EpiForecasterConfig


def test_load_config_with_simple_overrides():
    """Test that simple dotted overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
        overrides=["training.learning_rate=0.0005", "data.smoothing.window=10"],
    )

    assert cfg.training.learning_rate == 0.0005
    assert cfg.data.smoothing.window == 10


def test_load_config_with_nested_overrides():
    """Test that nested overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
        overrides=[
            "data.smoothing.enabled=true",
            "data.smoothing.smoothing_type=rolling_mean",
            "data.smoothing.window=7",
        ],
    )

    assert cfg.data.smoothing.enabled is True
    assert cfg.data.smoothing.smoothing_type == "rolling_mean"
    assert cfg.data.smoothing.window == 7


def test_load_config_with_bool_override():
    """Test that boolean overrides are handled correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
        overrides=["training.plot_forecasts=false", "data.log_scale=true"],
    )

    assert cfg.training.plot_forecasts is False
    assert cfg.data.log_scale is True


def test_load_config_without_overrides():
    """Test that config loads correctly without overrides."""
    cfg = EpiForecasterConfig.load("configs/train_epifor_full.yaml")

    assert cfg.training.learning_rate == 0.001
    assert cfg.data.smoothing.window == 5


def test_load_config_strict_mode_rejects_unknown_keys():
    """Test that strict mode rejects unknown keys."""
    with pytest.raises((errors.ConfigKeyError, errors.ValidationError, KeyError)):
        EpiForecasterConfig.load(
            "configs/train_epifor_full.yaml",
            overrides=["training.unknown_field=123"],
            strict=True,
        )


def test_load_config_model_type_overrides():
    """Test that model.type overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
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


def test_load_config_with_gnn_depth_override():
    """Test that GNN depth overrides work correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
        overrides=["model.gnn_depth=3"],
    )

    assert cfg.model.gnn_depth == 3


def test_load_config_with_multiple_overrides_same_section():
    """Test that multiple overrides in the same section work correctly."""
    cfg = EpiForecasterConfig.load(
        "configs/train_epifor_full.yaml",
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


def test_backward_compatibility_from_file():
    """Test that from_file() method still works."""
    cfg = EpiForecasterConfig.from_file("configs/train_epifor_full.yaml")

    assert cfg.training.learning_rate == 0.001
    assert cfg.data.smoothing.window == 5
