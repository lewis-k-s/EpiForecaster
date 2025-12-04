"""
Processor for COVID-19 case data from CSV files.

This module handles the conversion of COVID-19 case data from CSV format into
canonical tensor formats. It processes temporal case counts, applies normalization,
handles missing values, and creates target sequences for forecasting models.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import PreprocessingConfig


class CasesProcessor:
    """
    Converts COVID-19 case CSV data to aligned tensors.

    This processor handles:
    - Loading and parsing case data from CSV files
    - Temporal alignment to specified date ranges
    - Normalization (log1p, standard, minmax)
    - Missing value handling and interpolation
    - Creation of target sequences for forecasting
    - Municipality/region mapping and validation

    The output includes case tensors aligned with other data sources
    and properly formatted for input to the EpiBatch creation process.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the cases processor.

        Args:
            config: Preprocessing configuration with case processing options
        """
        self.config = config
        self.validation_options = config.validation_options

    def process(
        self, cases_file: str, region_mapping: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process COVID-19 case data into canonical tensors.

        Args:
            cases_file: Path to CSV file with case data
            region_mapping: Optional mapping from regions to municipality IDs

        Returns:
            Dictionary containing processed case data:
            - cases_tensor: [time, num_regions] tensor with case counts
            - target_sequences: [time, num_regions, forecast_horizon] tensor
            - region_metadata: Dictionary with region information
            - metadata: Processing metadata and statistics
        """
        print(f"Processing case data from {cases_file}")

        # Load case data
        cases_df = self._load_cases_data(cases_file)

        # Process region mapping
        region_info = self._process_region_mapping(cases_df, region_mapping)

        # Create temporal tensor
        cases_tensor = self._create_cases_tensor(cases_df, region_info)

        # Handle missing values
        cases_tensor = self._handle_missing_values(cases_tensor)

        # Apply normalization
        cases_tensor = self._normalize_cases(
            cases_tensor, self.config.cases_normalization
        )

        # Create target sequences for forecasting
        target_sequences = self._create_target_sequences(cases_tensor)

        # Validate data quality
        self._validate_cases_data(cases_tensor, target_sequences)

        # Create metadata
        metadata = {
            "num_timepoints": cases_tensor.shape[0],
            "num_regions": cases_tensor.shape[1],
            "forecast_horizon": self.config.forecast_horizon,
            "normalization": self.config.cases_normalization,
            "date_range": {
                "start": self.config.start_date.isoformat(),
                "end": self.config.end_date.isoformat(),
            },
            "data_stats": self._compute_case_statistics(cases_tensor),
            "quality_metrics": self._compute_quality_metrics(cases_tensor),
        }

        return {
            "cases_tensor": cases_tensor,
            "target_sequences": target_sequences,
            "region_metadata": region_info,
            "metadata": metadata,
        }

    def _load_cases_data(self, cases_file: str) -> pd.DataFrame:
        """
        Load and validate case data from CSV.

        Args:
            cases_file: Path to CSV file

        Returns:
            DataFrame with case data
        """
        cases_df = pd.read_csv(cases_file)

        # Apply column mapping if provided
        if self.config.cases_column_mapping:
            cases_df = cases_df.rename(columns=self.config.cases_column_mapping)

        # Validate required columns
        required_columns = ["date", "region_id", "cases"]
        missing_columns = [
            col for col in required_columns if col not in cases_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in cases file: {missing_columns}"
            )

        # Convert date column and handle timezone information
        cases_df["date"] = pd.to_datetime(cases_df["date"]).dt.tz_localize(None)

        # Validate data types
        cases_df["cases"] = pd.to_numeric(cases_df["cases"], errors="coerce")
        cases_df["region_id"] = pd.to_numeric(cases_df["region_id"], errors="coerce")

        # Remove invalid rows
        cases_df = cases_df.dropna(subset=["date", "region_id", "cases"])
        cases_df = cases_df[cases_df["cases"] >= 0]  # Remove negative cases

        return cases_df

    def _process_region_mapping(
        self, cases_df: pd.DataFrame, region_mapping: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Process and validate region mapping.

        Args:
            cases_df: DataFrame with case data
            region_mapping: Optional region mapping information

        Returns:
            Dictionary with region information
        """
        # Get unique regions from data
        unique_regions = sorted(cases_df["region_id"].unique())
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

        region_info = {
            "unique_regions": unique_regions,
            "num_regions": num_regions,
            "region_id_to_index": region_id_to_index,
            "region_mapping": region_mapping,
        }

        return region_info

    def _create_cases_tensor(
        self, cases_df: pd.DataFrame, region_info: dict[str, Any]
    ) -> torch.Tensor:
        """
        Create temporal tensor from case data.

        Args:
            cases_df: DataFrame with case data
            region_info: Region mapping information

        Returns:
            [time, num_regions] tensor with case counts
        """
        # Create date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )

        num_timepoints = len(date_range)
        num_regions = region_info["num_regions"]
        region_id_to_index = region_info["region_id_to_index"]

        # Initialize tensor
        cases_tensor = torch.zeros(num_timepoints, num_regions)

        # Fill tensor with case data
        for _, row in cases_df.iterrows():
            date = row["date"]
            region_id = int(row["region_id"])
            cases = float(row["cases"])

            # Check if date is in our range
            if self.config.start_date <= date <= self.config.end_date:
                date_idx = (date - self.config.start_date).days
                if date_idx < num_timepoints and region_id in region_id_to_index:
                    region_idx = region_id_to_index[region_id]
                    cases_tensor[date_idx, region_idx] = cases

        return cases_tensor

    def _handle_missing_values(self, cases_tensor: torch.Tensor) -> torch.Tensor:
        """
        Handle missing values in case data.

        Args:
            cases_tensor: [time, num_regions] tensor with case counts

        Returns:
            Tensor with missing values handled
        """
        # Replace zeros with NaN for missing value detection
        cases_nan = cases_tensor.clone()
        cases_nan[cases_nan == 0] = float("nan")

        # Check temporal coverage for each region
        temporal_coverage = (~torch.isnan(cases_nan)).float().mean(dim=0)
        min_coverage = self.validation_options.get("min_data_coverage", 0.8)

        # Identify regions with insufficient coverage
        low_coverage_regions = temporal_coverage < min_coverage
        if low_coverage_regions.any():
            print(
                f"Warning: {low_coverage_regions.sum().item()} regions have < {min_coverage * 100}% data coverage"
            )

        # Interpolate missing values
        cases_processed = self._interpolate_missing_values(cases_tensor)

        return cases_processed

    def _interpolate_missing_values(self, cases_tensor: torch.Tensor) -> torch.Tensor:
        """
        Interpolate missing values using configured strategy.

        Args:
            cases_tensor: Tensor with potential missing values

        Returns:
            Tensor with interpolated values
        """
        # Convert to numpy for interpolation
        cases_np = cases_tensor.numpy()

        # Forward fill then backward fill
        cases_df = pd.DataFrame(cases_np)
        cases_filled = cases_df.fillna(method="ffill").fillna(method="bfill")

        # For remaining NaN values (at beginning/end), use small value
        cases_filled = cases_filled.fillna(1e-6)

        return torch.from_numpy(cases_filled.values).float()

    def _normalize_cases(
        self, cases_tensor: torch.Tensor, normalization: str
    ) -> torch.Tensor:
        """
        Apply normalization to case data.

        Args:
            cases_tensor: Input case tensor
            normalization: Normalization method

        Returns:
            Normalized case tensor
        """
        if normalization == "none":
            return cases_tensor
        elif normalization == "log1p":
            # Add small epsilon to handle zeros
            return torch.log1p(cases_tensor + 1e-6)
        elif normalization == "standard":
            mean = cases_tensor.mean()
            std = cases_tensor.std()
            return (cases_tensor - mean) / (std + 1e-8)
        elif normalization == "minmax":
            min_val = cases_tensor.min()
            max_val = cases_tensor.max()
            return (cases_tensor - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    def _create_target_sequences(self, cases_tensor: torch.Tensor) -> torch.Tensor:
        """
        Create target sequences for forecasting.

        Args:
            cases_tensor: [time, num_regions] tensor with normalized cases

        Returns:
            [time, num_regions, forecast_horizon] tensor with target sequences
        """
        num_timepoints, num_regions = cases_tensor.shape
        forecast_horizon = self.config.forecast_horizon

        # Initialize target tensor
        target_sequences = torch.zeros(num_timepoints, num_regions, forecast_horizon)

        # Create target sequences by shifting the case tensor
        for h in range(forecast_horizon):
            if h < num_timepoints:
                target_sequences[: num_timepoints - h, :, h] = cases_tensor[h:]

        return target_sequences

    def _validate_cases_data(
        self, cases_tensor: torch.Tensor, target_sequences: torch.Tensor
    ):
        """
        Validate processed case data quality.

        Args:
            cases_tensor: Processed case tensor
            target_sequences: Target sequence tensor
        """
        # Check for NaN values
        if torch.isnan(cases_tensor).any():
            raise ValueError("NaN values found in processed case tensor")

        if torch.isnan(target_sequences).any():
            raise ValueError("NaN values found in target sequences")

        # Check for negative values (should not exist after normalization)
        if (cases_tensor < 0).any():
            print("Warning: Negative values found in case tensor")

        # Check temporal consistency
        self._check_temporal_consistency(cases_tensor)

        # Check for outliers
        if self.validation_options.get("outlier_detection", True):
            self._detect_outliers(cases_tensor)

    def _check_temporal_consistency(self, cases_tensor: torch.Tensor):
        """Check for temporal consistency in case data."""
        # Compute day-to-day changes
        case_changes = torch.diff(cases_tensor, dim=0)

        # Check for extreme changes (potential data errors)
        extreme_threshold = self.validation_options.get("outlier_threshold", 3.0)
        mean_change = case_changes.mean()
        std_change = case_changes.std()

        extreme_changes = (
            torch.abs(case_changes - mean_change) > extreme_threshold * std_change
        )

        if extreme_changes.any():
            num_extreme = extreme_changes.sum().item()
            total_changes = extreme_changes.numel()
            print(
                f"Warning: {num_extreme}/{total_changes} extreme temporal changes detected"
            )

    def _detect_outliers(self, cases_tensor: torch.Tensor):
        """Detect outliers in case data using z-score method."""
        mean_cases = cases_tensor.mean()
        std_cases = cases_tensor.std()

        z_scores = torch.abs((cases_tensor - mean_cases) / (std_cases + 1e-8))
        outlier_threshold = self.validation_options.get("outlier_threshold", 3.0)

        outliers = z_scores > outlier_threshold

        if outliers.any():
            num_outliers = outliers.sum().item()
            total_values = outliers.numel()
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
