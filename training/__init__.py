"""
Training modules for epidemiological forecasting models.

This module provides the unified training infrastructure for all model variants
in the EpiForecaster pipeline.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .epiforecaster_trainer import EpiForecasterConfig, EpiForecasterTrainer
    from .region2vec_trainer import Region2VecTrainer, RegionTrainerConfig

__all__ = [
    "EpiForecasterTrainer",
    "EpiForecasterConfig",
    "Region2VecTrainer",
    "RegionTrainerConfig",
]


def __getattr__(name: str) -> Any:
    """Load trainer APIs lazily so artifact helpers stay dependency-light."""
    if name in {"EpiForecasterConfig", "EpiForecasterTrainer"}:
        from .epiforecaster_trainer import EpiForecasterConfig, EpiForecasterTrainer

        return {
            "EpiForecasterConfig": EpiForecasterConfig,
            "EpiForecasterTrainer": EpiForecasterTrainer,
        }[name]
    if name in {"Region2VecTrainer", "RegionTrainerConfig"}:
        from .region2vec_trainer import Region2VecTrainer, RegionTrainerConfig

        return {
            "Region2VecTrainer": Region2VecTrainer,
            "RegionTrainerConfig": RegionTrainerConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
