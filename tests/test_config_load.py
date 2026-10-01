"""Tests for OmegaConf-based config loading with overrides."""

from operator import attrgetter
import yaml

import pytest
from omegaconf import errors

from models.configs import EpiForecasterConfig
from tests.config_fixtures import config_fixture


@pytest.fixture
def base_config(tmp_path):
    config = config_fixture()
    config.training.learning_rate = 0.0005
    config.training.max_batches = 8
    config.training.num_workers = config.training.val_workers = (
        config.training.test_workers
    ) = 0
    config.training.loss.joint.observation_loss = "crps"
    config.training.loss.joint.distribution_family = "student_t"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


OVERRIDE_CASES = [
    {"training.learning_rate": 0.0005, "data.mobility_min_flow_raw": 2.5},
    {
        "model.type.cases": True,
        "model.type.regions": False,
        "model.type.biomarkers": True,
    },
    {"training.plot_forecasts": False, "data.use_valid_targets": True},
    {
        "model.type.cases": True,
        "model.type.regions": True,
        "model.type.biomarkers": True,
        "native.mobility_enabled": True,
    },
    {"model.gnn_depth": 3},
    {
        "training.learning_rate": 0.0001,
        "training.batch_size": 64,
        "training.epochs": 50,
        "training.early_stopping_patience": 15,
    },
    {
        "training.node_split_strategy": "random",
        "training.node_split_population_bins": 4,
    },
    {
        "training.node_split_strategy": "crossval",
        "training.crossval_enabled": True,
        "training.crossval_num_folds": 4,
        "training.crossval_fold_index": 2,
        "training.split_seed": 7,
        "training.node_split_biomarker_bins": 2,
    },
]


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    "expected",
    OVERRIDE_CASES,
    ids=[
        "simple",
        "nested-model-type",
        "booleans",
        "all-model-types",
        "gnn-depth",
        "training-section",
        "node-split",
        "crossval-node-split",
    ],
)
def test_load_config_overrides(expected, base_config):
    """Dotted overrides should update their typed config fields."""
    overrides = [
        f"{path}={str(value).lower() if isinstance(value, bool) else value}"
        for path, value in expected.items()
    ]
    cfg = EpiForecasterConfig.load(base_config, overrides=overrides)

    for path, value in expected.items():
        assert attrgetter(path)(cfg) == value


@pytest.mark.epiforecaster
def test_load_config_without_overrides(base_config):
    """Test that config loads correctly without overrides."""
    cfg = EpiForecasterConfig.load(base_config)

    assert cfg.training.learning_rate == 0.0005
    assert cfg.output.write_granular_eval is False
    assert cfg.output.granular_eval_format == "parquet"
    assert (
        cfg.output.resolve_granular_eval_filename(split="test")
        == "test_granular.parquet"
    )
    assert cfg.training.node_split_strategy == "random"
    assert cfg.training.node_split_population_bins == 5
    assert cfg.training.max_batches == 8
    assert cfg.training.num_workers == 0
    assert cfg.training.val_workers == 0
    assert cfg.training.test_workers == 0
    assert cfg.training.loss.joint.observation_loss == "crps"
    assert cfg.training.loss.joint.distribution_family == "student_t"


@pytest.mark.epiforecaster
def test_load_config_strict_mode_rejects_unknown_keys(base_config):
    """Test that strict mode rejects unknown keys."""
    with pytest.raises((errors.ConfigKeyError, errors.ValidationError, KeyError)):
        EpiForecasterConfig.load(
            base_config,
            overrides=["training.unknown_field=123"],
            strict=True,
        )


@pytest.mark.epiforecaster
def test_load_config_rejects_invalid_node_split_strategy(base_config):
    """Test that invalid node split strategy is rejected."""
    with pytest.raises(ValueError, match="Invalid node_split_strategy"):
        EpiForecasterConfig.load(
            base_config,
            overrides=["training.node_split_strategy=not_a_strategy"],
        )


@pytest.mark.epiforecaster
def test_backward_compatibility_from_file(base_config):
    """Test that from_file() method still works."""
    cfg = EpiForecasterConfig.from_file(base_config)

    assert cfg.training.learning_rate == 0.0005


@pytest.mark.epiforecaster
def test_from_dict_reconstructs_config(base_config):
    """Test that from_dict() correctly reconstructs a config from to_dict()."""
    # Load a config and convert to dict (as would be saved in checkpoint)
    original = EpiForecasterConfig.load(base_config)
    config_dict = original.to_dict()

    # Reconstruct from dict (as would be loaded from checkpoint)
    reconstructed = EpiForecasterConfig.from_dict(config_dict)

    # Verify all top-level fields match
    assert reconstructed.model.type == original.model.type
    assert reconstructed.model.gnn_depth == original.model.gnn_depth
    assert reconstructed.data.dataset_path == original.data.dataset_path
    assert reconstructed.training.learning_rate == original.training.learning_rate
    assert reconstructed.training.batch_size == original.training.batch_size
    assert reconstructed.output.log_dir == original.output.log_dir
    assert (
        reconstructed.output.write_granular_eval == original.output.write_granular_eval
    )
