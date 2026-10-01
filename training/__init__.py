"""
Training modules for epidemiological forecasting models.

This module provides the unified training infrastructure for all model variants
in the EpiForecaster pipeline.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.configs import EpiForecasterConfig
    from .native_trainer import NativeTrainer
    from .region2vec_trainer import Region2VecTrainer, RegionTrainerConfig

__all__ = [
    "NativeTrainer",
    "EpiForecasterConfig",
    "Region2VecTrainer",
    "RegionTrainerConfig",
]


def __getattr__(name: str) -> Any:
    """Load trainer APIs lazily so artifact helpers stay dependency-light."""
    if name in {"EpiForecasterConfig", "NativeTrainer"}:
        from models.configs import EpiForecasterConfig
        from .native_trainer import NativeTrainer

        return {
            "EpiForecasterConfig": EpiForecasterConfig,
            "NativeTrainer": NativeTrainer,
        }[name]
    if name in {"Region2VecTrainer", "RegionTrainerConfig"}:
        from .region2vec_trainer import Region2VecTrainer, RegionTrainerConfig

        return {
            "Region2VecTrainer": Region2VecTrainer,
            "RegionTrainerConfig": RegionTrainerConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
