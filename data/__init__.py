"""
Data loading and preprocessing modules for epidemiological forecasting.

This module provides the canonical data structures and preprocessing pipeline
for epidemiological forecasting. The main components are:

- EpiDataset: PyTorch Dataset interface for loading preprocessed Zarr datasets
- EpiBatch: Universal batch representation for all model variants
- DatasetStorage: Zarr-based storage for canonical datasets
- OfflinePreprocessingPipeline: Configuration-driven preprocessing pipeline
"""

from .dataset_storage import DatasetStorage
from .epi_batch import EpiBatch
from .epi_dataset import EpiDataset
from .preprocess import OfflinePreprocessingPipeline, PreprocessingConfig

__all__ = [
    "EpiBatch",
    "EpiDataset",
    "DatasetStorage",
    "OfflinePreprocessingPipeline",
    "PreprocessingConfig",
]
