"""
Centralized storage dtype constants and utilities for EpiForecaster format.

This module isolates raw data representations (e.g. preprocessing schemas)
from the model execution layer.
"""

import torch

# =============================================================================
# STORAGE DTYPES (used in preprocessing and dataset loading)
# =============================================================================

# Input data dtypes - matches preprocessing output schema from data/preprocess/README.md
STORAGE_DTYPES = {
    "continuous": torch.float16,  # All continuous series (clinical, biomarkers, mobility)
    "mask": torch.bool,  # Binary masks (observed vs missing)
    "age": torch.uint8,  # Days since last observation (0-14)
    "censor": torch.uint8,  # Censor flags (0=uncensored, 1=censored, 2=imputed)
    "index": torch.int16,  # Data start indices (-1 sentinel)
    "population": torch.int32,  # Population counts
}

# Numpy equivalents for preprocessing
NUMPY_STORAGE_DTYPES = {
    "continuous": "float16",
    "mask": "bool",
    "age": "uint8",
    "censor": "uint8",
    "index": "int16",
    "population": "int32",
}

def get_storage_dtype(dtype_key: str) -> torch.dtype:
    """
    Get the storage dtype for a given key.

    Args:
        dtype_key: One of 'continuous', 'mask', 'age', 'censor', 'index', 'population'

    Returns:
        torch dtype for storage
    """
    return STORAGE_DTYPES[dtype_key]

def is_storage_dtype(tensor: torch.Tensor, dtype_key: str) -> bool:
    """Check if tensor matches the expected storage dtype."""
    return tensor.dtype == STORAGE_DTYPES[dtype_key]
