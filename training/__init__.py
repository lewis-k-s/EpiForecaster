"""
Training modules for epidemiological forecasting models.

This module provides the unified training infrastructure for all model variants
in the EpiForecaster pipeline.
"""

from .epiforecaster_trainer import EpiForecasterTrainerConfig, EpiForecasterTrainer
from .region_embedder_trainer import RegionTrainerConfig, RegionEmbedderTrainer

__all__ = [
    "EpiForecasterTrainer",
    "EpiForecasterTrainerConfig",
    "RegionTrainerConfig",
    "RegionEmbedderTrainer",
]
