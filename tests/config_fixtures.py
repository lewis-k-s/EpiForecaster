"""Canonical typed configs, independent of deployment templates."""

from pathlib import Path

from models.configs import (
    DataConfig,
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    OutputConfig,
    TrainingParams,
)


def config_fixture() -> EpiForecasterConfig:
    return EpiForecasterConfig(
        model=ModelConfig(
            type=ModelVariant(
                cases=False, regions=False, biomarkers=False, mobility=False
            ),
            mobility_embedding_dim=16,
            region_embedding_dim=8,
            input_window_length=28,
            forecast_horizon=14,
            max_neighbors=20,
        ),
        data=DataConfig(dataset_path="fixture.zarr"),
        training=TrainingParams(epochs=2),
        output=OutputConfig(),
    )


def write_config_fixture(root: Path) -> Path:
    """Write a complete YAML fixture for config and campaign parsing."""
    import yaml

    path = root / "base.yaml"
    path.write_text(yaml.safe_dump(config_fixture().to_dict()))
    return path
