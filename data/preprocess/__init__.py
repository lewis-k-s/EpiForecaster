"""
Offline preprocessing pipeline for EpiForecaster datasets.

This package provides tools for converting raw epidemiological data into
canonical EpiBatch datasets. The preprocessing pipeline handles:

- Raw data loading (mobility NetCDF files, case CSV files, wastewater data)
- Temporal and spatial alignment of multiple datasets
- Graph construction and temporal windowing
- Validation and quality reporting
- Efficient storage in Zarr format for downstream training

The pipeline is designed to be run once per dataset configuration, producing
persistent canonical datasets that can be efficiently loaded during training.
"""

from .config import PreprocessingConfig
from .pipeline import OfflinePreprocessingPipeline

__all__ = [
    "PreprocessingConfig",
    "OfflinePreprocessingPipeline",
]
