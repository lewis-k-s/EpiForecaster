"""
Training modules for epidemiological forecasting models.

This module provides the unified training infrastructure for all model variants
in the EpiForecaster pipeline.
"""

from .epiforecaster_trainer import EpiForecasterConfig, EpiForecasterTrainer
from .region2vec_trainer import Region2VecTrainer, RegionTrainerConfig

__all__ = [
    "EpiForecasterTrainer",
    "EpiForecasterConfig",
    "Region2VecTrainer",
    "RegionTrainerConfig",
]
