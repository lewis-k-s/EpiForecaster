"""Test that all YAML configs in configs/ can parse successfully."""

import pytest
from pathlib import Path

from models.configs import EpiForecasterConfig
from training.region2vec_trainer import RegionTrainerConfig


# Config file lists by type
EPIFORECASTER_TRAINING_CONFIGS = [
    "configs/train_epifor_curriculum.yaml",
    "configs/train_epifor_full.yaml",
    "configs/train_epifor_real_local.yaml",
    "configs/train_epifor_synth_local.yaml",
    "configs/train_epifor_temporal.yaml",
    "configs/production_only/train_epifor_mn5_full.yaml",
    "configs/production_only/train_epifor_mn5_synth.yaml",
    "configs/production_only/train_epifor_sparsity_curriculum.yaml",
]

REGION2VEC_CONFIGS = [
    "configs/train_regions.yaml",
]

PREPROCESSING_CONFIGS = [
    "configs/preprocess_filtered.yaml",
    "configs/preprocess_full.yaml",
    "configs/production_only/preprocess_mn5_synth.yaml",
]


@pytest.mark.epiforecaster
@pytest.mark.parametrize("config_path", EPIFORECASTER_TRAINING_CONFIGS)
def test_epiforecaster_training_configs_parse(config_path):
    """Test that all EpiForecaster training configs can be parsed."""
    cfg = EpiForecasterConfig.load(config_path)
    # Basic validation that the config loaded
    assert cfg.model is not None
    assert cfg.data is not None
    assert cfg.training is not None
    assert cfg.output is not None


@pytest.mark.region
@pytest.mark.parametrize("config_path", REGION2VEC_CONFIGS)
def test_region2vec_configs_parse(config_path):
    """Test that all Region2Vec configs can be parsed."""
    cfg = RegionTrainerConfig.from_file(config_path)
    assert cfg.data is not None
    assert cfg.encoder is not None
    assert cfg.training is not None
    assert cfg.sampling is not None
    assert cfg.loss is not None
    assert cfg.output is not None
    assert cfg.clustering is not None


@pytest.mark.parametrize("config_path", PREPROCESSING_CONFIGS)
def test_preprocessing_configs_parse(config_path):
    """Test that all preprocessing configs can be parsed."""
    # PreprocessingConfig validates paths exist, so we expect FileNotFoundError
    # but we can verify the YAML structure is valid
    import yaml

    config_path = Path(config_path)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Verify required top-level fields exist
    assert "data_dir" in config_dict
    assert "start_date" in config_dict
    assert "end_date" in config_dict

    # Verify date format is valid
    from datetime import datetime

    datetime.fromisoformat(config_dict["start_date"])
    datetime.fromisoformat(config_dict["end_date"])


def test_all_config_files_accounted_for():
    """Ensure test lists match actual files in configs/ directory."""
    configs_dir = Path("configs")

    # Find all YAML files recursively
    yaml_files = sorted(
        [str(f.relative_to(configs_dir.parent)) for f in configs_dir.rglob("*.yaml")]
    )

    # Combine all test lists
    tested_files = (
        EPIFORECASTER_TRAINING_CONFIGS + REGION2VEC_CONFIGS + PREPROCESSING_CONFIGS
    )
    tested_files_sorted = sorted(tested_files)

    # Check for any untested files
    untested = set(yaml_files) - set(tested_files_sorted)
    extra_tests = set(tested_files_sorted) - set(yaml_files)

    assert not untested, f"Config files not tested: {untested}"
    assert not extra_tests, f"Test lists contain non-existent files: {extra_tests}"
