"""
Data loading and preprocessing modules.

This module provides the canonical data structures and preprocessing pipelines. The main components are:

- EpiDataset: PyTorch Dataset interface for loading preprocessed epiforecaster Zarr datasets
- OfflinePreprocessingPipeline: Configuration-driven preprocessing pipeline for epiforecaster datasets
- RegionGraphPreprocessor: Configuration-driven preprocessing pipeline for region graph datasets
- RegionGraphDataset: PyTorch Dataset interface for loading preprocessed region graph Zarr datasets
"""

from .cases_preprocessor import CasesPreprocessor, CasesPreprocessorConfig
from .epi_dataset import EpiDataset
from .preprocess import (
    OfflinePreprocessingPipeline,
    PreprocessingConfig,
    RegionGraphPreprocessor,
)

__all__ = [
    "CasesPreprocessor",
    "CasesPreprocessorConfig",
    "EpiDataset",
    "OfflinePreprocessingPipeline",
    "PreprocessingConfig",
    "RegionGraphPreprocessor",
]
