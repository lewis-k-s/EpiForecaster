"""
Data loading and preprocessing modules.

This module provides the canonical data structures and preprocessing pipelines. The main components are:

- EpiDataset: PyTorch Dataset interface for loading preprocessed epiforecaster Zarr datasets
- build_datasets: Factory function for creating train/val/test dataset splits
- OfflinePreprocessingPipeline: Configuration-driven preprocessing pipeline for epiforecaster datasets
- RegionGraphPreprocessor: Configuration-driven preprocessing pipeline for region graph datasets
- RegionGraphDataset: PyTorch Dataset interface for loading preprocessed region graph Zarr datasets
"""

from .cases_preprocessor import CasesPreprocessor, CasesPreprocessorConfig
from .curriculum_builder import (
    CurriculumBuildResult,
    FittedPreprocessors,
    build_curriculum_datasets,
    discover_runs,
    load_sparsity_mapping,
    select_runs_by_sparsity,
)
from .dataset_factory import DatasetSplits, build_datasets, split_nodes_by_ratio
from .epi_dataset import EpiDataset
from .preprocess import (
    OfflinePreprocessingPipeline,
    PreprocessingConfig,
    RegionGraphPreprocessor,
)

__all__ = [
    "CasesPreprocessor",
    "CasesPreprocessorConfig",
    "CurriculumBuildResult",
    "DatasetSplits",
    "EpiDataset",
    "FittedPreprocessors",
    "OfflinePreprocessingPipeline",
    "PreprocessingConfig",
    "RegionGraphPreprocessor",
    "build_curriculum_datasets",
    "build_datasets",
    "discover_runs",
    "load_sparsity_mapping",
    "select_runs_by_sparsity",
    "split_nodes_by_ratio",
]
