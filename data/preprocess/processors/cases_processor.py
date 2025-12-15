"""
Processor for COVID-19 case data from CSV files.

This module handles the conversion of COVID-19 case data from CSV format into
canonical xarray formats. It processes temporal case counts, applies normalization,
and handles missing values. The output is a simple (time, region) time series.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig


class CasesProcessor:
    """
    Converts COVID-19 case CSV data to xarray Dataset.

    This processor handles:
    - Loading and parsing case data from CSV files
    - Temporal cropping to specified date ranges
    - Normalization (log1p, standard, minmax)
    - Missing value handling and interpolation
    - Municipality/region mapping and validation

    Output is a simple (time, region) time series. The model data loader
    handles horizon and history windowing decisions.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the cases processor.

        Args:
            config: Preprocessing configuration with case processing options
        """
        self.config = config
        self.validation_options = config.validation_options

    def _load_cases_data(self, cases_file: str) -> pd.DataFrame:
        """Load and validate case data from CSV."""
        cases_df = pd.read_csv(
            cases_file,
            usecols=["id", "evend", "d.cases"],
            dtype={"id": str, "evend": str, "d.cases": int},
        )
        cases_df = cases_df.rename(
            columns={"id": REGION_COORD, "evend": "date", "d.cases": "cases"}
        )

        assert not cases_df.empty, "No data found in cases file"
        assert {"date", REGION_COORD, "cases"} == set(cases_df.columns), (
            "Cases columns mismatch"
        )

        # Convert date column and handle timezone information
        cases_df["date"] = pd.to_datetime(cases_df["date"]).dt.tz_localize(None)

        # Remove invalid rows
        cases_df = cases_df.dropna(subset=["date", REGION_COORD, "cases"])

        if (cases_df["cases"] < 0).any():
            print("Warning: Negative cases found in cases file")

        cases_df = cases_df[cases_df["cases"] >= 0]

        return cases_df

    def process(self, cases_file: str) -> xr.Dataset:
        """
        Process COVID-19 case data into canonical xarray Dataset.

        Args:
            cases_file: Path to CSV file with case data

        Returns:
            xarray Dataset containing:
            - cases: xarray DataArray with cases (time, region)
        """
        print(f"Processing case data from {cases_file}")

        # Load and prepare DataFrame
        cases_df = self._load_cases_data(cases_file)

        # Crop to config date range
        cases_df = cases_df[
            (cases_df["date"] >= self.config.start_date)
            & (cases_df["date"] <= self.config.end_date)
        ]
        assert not cases_df.empty, "No data found in temporal range"

        # Pivot to wide format: date as index, region_id as columns. Assume unique but sum if not
        cases_pivot = cases_df.pivot_table(
            index="date", columns=REGION_COORD, values="cases", aggfunc="sum"
        )

        # Reindex to complete date range to ensure all dates are present
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        cases_pivot = cases_pivot.reindex(date_range)

        # Rename columns and index to match coordinate names
        cases_pivot.columns.name = REGION_COORD
        cases_pivot.index.name = TEMPORAL_COORD

        # Convert to xarray DataArray with proper coordinates
        # DataFrame index becomes temporal dimension, columns become region dimension
        cases_ds = xr.DataArray(
            cases_pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: cases_pivot.index,
                REGION_COORD: cases_pivot.columns,
            },
        )
        cases_ds = cases_ds.to_dataset(name="cases")

        # Handle missing values
        cases_ds = self._interpolate_cases(cases_ds)

        # Apply normalization
        cases_ds = self._normalize(cases_ds, self.config.cases_normalization)

        # Validate data quality
        self._validate_cases_data_xr(cases_ds)

        # Create region metadata
        print(cases_ds)
        # region_info = self._create_region_info(cases_ds, region_mapping)

        return cases_ds

    def _interpolate_cases(self, cases_ds: xr.Dataset) -> xr.Dataset:
        """Handle missing values in case data using xarray operations."""
        # Check temporal coverage for each region
        temporal_coverage = (~cases_ds["cases"].isnull()).mean(dim=TEMPORAL_COORD)
        min_coverage = self.validation_options.get("min_data_coverage", 0.8)

        # Identify regions with insufficient coverage
        low_coverage_regions = temporal_coverage < min_coverage
        if low_coverage_regions.any():
            num_low = int(low_coverage_regions.sum())
            print(
                f"Warning: {num_low} regions have < {min_coverage * 100}% data coverage"
            )

        # Interpolate missing values: forward fill then backward fill
        # TODO: interpolate better
        cases_filled = (
            cases_ds["cases"].ffill(dim=TEMPORAL_COORD).bfill(dim=TEMPORAL_COORD)
        )

        # For remaining NaN values (at beginning/end), use small value
        cases_filled = cases_filled.fillna(1e-6)
        cases_filled = cases_filled.to_dataset(name="cases")

        return cases_filled

    def _normalize(self, cases_da: xr.DataArray, normalization: str) -> xr.DataArray:
        """Apply normalization to case data."""
        if normalization == "none":
            return cases_da
        elif normalization == "log1p":
            return xr.apply_ufunc(np.log1p, cases_da + 1e-6)
        elif normalization == "standard":
            mean = cases_da.mean()
            std = cases_da.std()
            return (cases_da - mean) / (std + 1e-8)
        elif normalization == "minmax":
            min_val = cases_da.min()
            max_val = cases_da.max()
            return (cases_da - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    def _validate_cases_data_xr(self, cases_xr: xr.Dataset):
        """Validate processed case data quality."""
        cases_da = cases_xr.cases

        # Check for NaN values
        if cases_da.isnull().any():
            raise ValueError("NaN values found in processed case data")

        # Check for negative values (should not exist after normalization)
        if (cases_da < 0).any():
            print("Warning: Negative values found in case data")

        # Check temporal consistency
        self._check_temporal_consistency_xr(cases_da)

        # Check for outliers
        if self.validation_options.get("outlier_detection", True):
            self._detect_outliers_xr(cases_da)

    def _create_region_info(
        self, cases_xr: xr.Dataset, region_mapping: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Create region information from xarray Dataset."""
        unique_regions = cases_xr[REGION_COORD].values.tolist()
        num_regions = len(unique_regions)

        # Create region ID to index mapping
        region_id_to_index = {
            region_id: idx for idx, region_id in enumerate(unique_regions)
        }

        # Validate region mapping if provided
        if region_mapping is not None:
            mapped_regions = set(region_mapping.keys())
            data_regions = set(unique_regions)
            unmapped_regions = data_regions - mapped_regions

            if unmapped_regions:
                print(
                    f"Warning: {len(unmapped_regions)} regions not found in mapping: {unmapped_regions}"
                )

        return {
            "unique_regions": unique_regions,
            "num_regions": num_regions,
            "region_id_to_index": region_id_to_index,
            "region_mapping": region_mapping,
        }

    def _check_temporal_consistency_xr(self, cases_da: xr.DataArray):
        """Check for temporal consistency in case data."""
        # Compute day-to-day changes
        case_changes = cases_da.diff(dim=TEMPORAL_COORD)

        # Check for extreme changes (potential data errors)
        extreme_threshold = self.validation_options.get("outlier_threshold", 3.0)
        mean_change = case_changes.mean()
        std_change = case_changes.std()

        extreme_changes = (
            xr.ufuncs.abs(case_changes - mean_change) > extreme_threshold * std_change
        )

        if extreme_changes.any():
            num_extreme = int(extreme_changes.sum())
            total_changes = extreme_changes.size
            print(
                f"Warning: {num_extreme}/{total_changes} extreme temporal changes detected"
            )

    def _detect_outliers_xr(self, cases_da: xr.DataArray):
        """Detect outliers in case data using z-score method."""
        mean_cases = cases_da.mean()
        std_cases = cases_da.std()

        z_scores = xr.ufuncs.abs((cases_da - mean_cases) / (std_cases + 1e-8))
        outlier_threshold = self.validation_options.get("outlier_threshold", 3.0)

        outliers = z_scores > outlier_threshold

        if outliers.any():
            num_outliers = int(outliers.sum())
            total_values = outliers.size
            print(
                f"Warning: {num_outliers}/{total_values} outliers detected (z-score > {outlier_threshold})"
            )

    def _compute_case_statistics(self, cases_tensor: torch.Tensor) -> dict[str, float]:
        """Compute statistics for case data."""
        return {
            "total_cases": float(cases_tensor.sum()),
            "mean_cases_per_day": float(cases_tensor.mean()),
            "std_cases_per_day": float(cases_tensor.std()),
            "max_cases_per_day": float(cases_tensor.max()),
            "min_cases_per_day": float(cases_tensor.min()),
            "zero_days_percentage": float((cases_tensor == 0).float().mean() * 100),
            "regions_with_cases": int((cases_tensor.sum(dim=0) > 0).sum().item()),
        }

    def _compute_quality_metrics(self, cases_tensor: torch.Tensor) -> dict[str, Any]:
        """Compute data quality metrics."""
        # Temporal coverage per region
        temporal_coverage = ((cases_tensor > 0).float().mean(dim=0) * 100).tolist()

        # Data continuity (longest consecutive non-zero sequence)
        continuity_scores = []
        for region_idx in range(cases_tensor.shape[1]):
            region_data = cases_tensor[:, region_idx] > 0
            max_consecutive = 0
            current_consecutive = 0

            for is_nonzero in region_data:
                if is_nonzero:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            continuity_scores.append(max_consecutive)

        return {
            "temporal_coverage_percentage": temporal_coverage,
            "mean_continuity_days": float(np.mean(continuity_scores)),
            "median_continuity_days": float(np.median(continuity_scores)),
            "data_completeness_score": float(np.mean(temporal_coverage) / 100.0),
        }
