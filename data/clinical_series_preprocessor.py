"""
Clinical Series Preprocessor for 3-channel [value, mask, age] format.

This module provides preprocessing for clinical observation series (hospitalizations,
deaths, reported cases) into a unified 3-channel format suitable for model input.

The 3-channel format:
- Channel 0: Value (already log1p(per-100k) transformed in preprocessing pipeline)
- Channel 1: Mask (1.0 if observed, 0.0 if missing/interpolated)
- Channel 2: Age (days since last observation, normalized to [0, 1])

Unlike the old CasesPreprocessor, this does NOT compute rolling statistics since
SIR-based forecasting uses population fractions rather than normalized signals.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr

from utils import dtypes as dtype_utils

logger = logging.getLogger(__name__)


@dataclass
class ClinicalSeriesPreprocessorConfig:
    """Configuration for ClinicalSeriesPreprocessor.

    Args:
        age_max: Maximum age in days for normalization (default 14)
    """

    age_max: int = 14


class ClinicalSeriesPreprocessor:
    """Preprocess clinical observation series into 3-channel [value, mask, age] format.

    This preprocessor takes preprocessed clinical data (already containing value,
    mask, and age channels from the data pipeline) and stacks them into a unified
    3-channel tensor for model input.

    Contract:
        - Input: xarray Dataset with {var_name}, {var_name}_mask, {var_name}_age
        - Output: (T, N, 3) tensor with [value, mask, age] channels
        - Value channel is already log1p(per-100k) transformed from preprocessing
        - Age channel is normalized to [0, 1] range (age / age_max)

    Unlike CasesPreprocessor, this does NOT compute rolling mean/std since SIR-based
    forecasting outputs population fractions rather than normalized signals.
    """

    def __init__(self, config: ClinicalSeriesPreprocessorConfig, var_name: str):
        """Initialize preprocessor.

        Args:
            config: Preprocessor configuration
            var_name: Base variable name (e.g., "hospitalizations", "deaths", "cases")
        """
        self.config = config
        self.var_name = var_name
        self.processed_data: torch.Tensor | None = None

    def preprocess_dataset(
        self, dataset: xr.Dataset, population: xr.DataArray | None = None
    ) -> torch.Tensor:
        """Precompute 3-channel tensor from dataset.

        Args:
            dataset: xarray Dataset containing {var_name}, {var_name}_mask, {var_name}_age
            population: Ignored (kept for API compatibility; values already normalized)

        Returns:
            (T, N, 3) tensor with [value, mask, age] channels

        Raises:
            ValueError: If required variables are not found in dataset
        """
        # Load all three required arrays - fail early if any are missing
        value_da = self._load_variable(dataset, self.var_name)
        mask_da = self._load_variable(dataset, f"{self.var_name}_mask")
        age_da = self._load_variable(dataset, f"{self.var_name}_age")

        # Ensure consistent shape (T, N)
        values = self._ensure_2d(value_da)
        mask = self._ensure_2d(mask_da)
        age = self._ensure_2d(age_da)

        # Values are already log1p(per-100k) transformed from preprocessing pipeline
        # Fill NaN with 0 to prevent propagation through model (mask indicates validity)
        values = np.nan_to_num(values, nan=0.0)

        # Normalize age to [0, 1]
        age_normalized = np.clip(age / self.config.age_max, 0.0, 1.0)

        # Stack into 3-channel tensor using centralized storage dtype
        storage_dtype = dtype_utils.STORAGE_DTYPES["continuous"]
        values_t = torch.from_numpy(values).to(storage_dtype)
        mask_t = torch.from_numpy(mask).to(storage_dtype)
        age_t = torch.from_numpy(age_normalized).to(storage_dtype)

        data_3ch = torch.stack([values_t, mask_t, age_t], dim=-1)

        self.processed_data = data_3ch
        return data_3ch

    def _load_variable(self, dataset: xr.Dataset, var_name: str) -> xr.DataArray:
        """Load a variable from the dataset.

        Args:
            dataset: xarray Dataset
            var_name: Variable name to load

        Returns:
            DataArray for the variable

        Raises:
            ValueError: If variable not found in dataset
        """
        if var_name not in dataset.data_vars:
            available = list(dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{var_name}' not found in dataset. "
                f"Available variables: {available}"
            )
        return dataset[var_name]

    def _ensure_2d(self, da: xr.DataArray) -> np.ndarray:
        """Ensure DataArray is 2D (time, region), squeezing extra dims if needed.

        Args:
            da: Input DataArray

        Returns:
            2D numpy array with shape (T, N)
        """
        # Handle case where data has extra dimensions (e.g., run_id)
        if da.ndim > 2:
            # Try to squeeze dimensions of size 1
            da = da.squeeze()

        if da.ndim == 3:
            # If still 3D, assume first dim is run_id and take first run
            da = da.isel(run_id=0)

        # Transpose to (time, region) if needed
        dims = list(da.dims)
        if "date" in dims and "region_id" in dims:
            da = da.transpose("date", "region_id")
        elif "time" in dims and "region" in dims:
            da = da.transpose("time", "region")

        return da.values.astype(np.float32)

    def get_processed_data(self) -> torch.Tensor:
        """Get the preprocessed 3-channel data.

        Returns:
            (T, N, 3) tensor with [value, mask, age] channels

        Raises:
            RuntimeError: If preprocess_dataset has not been called
        """
        if self.processed_data is None:
            raise RuntimeError(
                "ClinicalSeriesPreprocessor not initialized. "
                "Call preprocess_dataset() first."
            )
        return self.processed_data

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return (
            f"ClinicalSeriesPreprocessor("
            f"var_name='{self.var_name}', "
            f"age_max={self.config.age_max}"
            f")"
        )
