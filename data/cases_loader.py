"""
COVID-19 Cases Data Loader for Epidemiological Forecasting.

This module provides functionality to load and preprocess COVID-19 case data
from CSV files, align it with mobility data temporally and spatially, and
prepare it for use as forecasting targets in graph neural networks.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from data.dataset_alignment import DatasetAlignmentManager

logger = logging.getLogger(__name__)


class CasesDataLoader:
    """
    Loads and processes COVID-19 case data for epidemiological forecasting.

    This class handles loading case data from CSV files, preprocessing,
    temporal alignment, and conversion to tensor format suitable for GNN training.
    """

    def __init__(
        self,
        cases_file: str,
        normalization: str = "log1p",
        min_cases_threshold: int = 0,
        fill_missing: str = "forward_fill",
    ):
        """
        Initialize the cases data loader.

        Args:
            cases_file: Path to CSV file containing case data
            normalization: Method for normalizing case counts ("log1p", "standard", "none")
            min_cases_threshold: Minimum case count threshold for filtering
            fill_missing: Method for handling missing data ("forward_fill", "zero", "interpolate")
        """
        self.cases_file = Path(cases_file)
        self.normalization = normalization
        self.min_cases_threshold = min_cases_threshold
        self.fill_missing = fill_missing

        # Loaded data
        self.cases_df: Optional[pd.DataFrame] = None
        self.cases_tensor: Optional[Tensor] = None
        self.municipality_mapping: Optional[Dict[str, int]] = None
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.municipalities: Optional[list] = None

        # Statistics
        self.total_cases: int = 0
        self.active_municipalities: int = 0
        self.timepoints: int = 0

    def load_data(self) -> pd.DataFrame:
        """
        Load COVID-19 case data from CSV file.

        Returns:
            DataFrame with columns: id, evstart, evend, d.cases, c.cumun
        """
        logger.info(f"Loading COVID cases data from: {self.cases_file}")

        if not self.cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {self.cases_file}")

        # Load CSV
        self.cases_df = pd.read_csv(self.cases_file)

        # Validate required columns
        required_columns = ["evstart", "evend", "d.cases", "c.cumun"]
        missing_columns = [
            col for col in required_columns if col not in self.cases_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert timestamp strings to datetime
        self.cases_df["evstart"] = pd.to_datetime(self.cases_df["evstart"])
        self.cases_df["evend"] = pd.to_datetime(self.cases_df["evend"])

        # Filter out invalid municipality codes
        self.cases_df = self.cases_df[
            self.cases_df["c.cumun"].astype(str).str.isdigit()
        ]
        self.cases_df["c.cumun"] = self.cases_df["c.cumun"].astype(int)

        # Filter by minimum cases threshold
        if self.min_cases_threshold > 0:
            self.cases_df = self.cases_df[
                self.cases_df["d.cases"] >= self.min_cases_threshold
            ]

        logger.info(f"Loaded {len(self.cases_df)} case records")
        return self.cases_df

    def preprocess_data(self) -> None:
        """
        Preprocess the loaded case data.
        """
        if self.cases_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Preprocessing COVID cases data")

        # Sort by municipality and date
        self.cases_df = self.cases_df.sort_values(["c.cumun", "evstart"])

        # Handle missing data
        self._handle_missing_data()

        # Create municipality mapping
        self.municipalities = sorted(self.cases_df["c.cumun"].unique())
        self.municipality_mapping = {
            muni: i for i, muni in enumerate(self.municipalities)
        }
        self.active_municipalities = len(self.municipalities)

        # Get date range
        self.date_range = (self.cases_df["evstart"].min(), self.cases_df["evend"].max())

        logger.info(f"Found {self.active_municipalities} active municipalities")
        logger.info(f"Date range: {self.date_range[0]} to {self.date_range[1]}")

    def create_temporal_tensor(self) -> Tensor:
        """
        Convert preprocessed data to temporal tensor format.

        Returns:
            Tensor of shape [num_municipalities, num_timepoints] with case counts
        """
        if self.cases_df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Creating temporal cases tensor")

        # Create date range
        start_date = self.date_range[0].date()
        end_date = self.date_range[1].date()
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        self.timepoints = len(date_range)

        # Initialize tensor
        cases_tensor = torch.zeros((self.active_municipalities, self.timepoints))

        # Fill tensor with case counts
        for _, row in self.cases_df.iterrows():
            muni_id = row["c.cumun"]
            if muni_id not in self.municipality_mapping:
                continue

            date_idx = (row["evstart"].date() - start_date).days
            if 0 <= date_idx < self.timepoints:
                tensor_idx = self.municipality_mapping[muni_id]
                cases_tensor[tensor_idx, date_idx] = row["d.cases"]

        # Apply normalization
        cases_tensor = self._normalize_cases(cases_tensor)

        self.cases_tensor = cases_tensor
        self.total_cases = int(cases_tensor.sum().item())

        logger.info(f"Created cases tensor: {cases_tensor.shape}")
        logger.info(f"Total cases: {self.total_cases}")
        logger.info(
            f"Average cases per municipality per day: {cases_tensor.mean():.2f}"
        )

        return cases_tensor

    def _handle_missing_data(self) -> None:
        """
        Handle missing data in the case records.
        """
        if self.fill_missing == "forward_fill":
            # Forward fill missing dates for each municipality
            self.cases_df = (
                self.cases_df.groupby("c.cumun")
                .apply(
                    lambda x: x.set_index("evstart").resample("D").ffill().reset_index()
                )
                .reset_index(drop=True)
            )
        elif self.fill_missing == "interpolate":
            # Linear interpolation for missing dates
            self.cases_df = (
                self.cases_df.groupby("c.cumun")
                .apply(
                    lambda x: x.set_index("evstart")
                    .resample("D")
                    .interpolate()
                    .reset_index()
                )
                .reset_index(drop=True)
            )
        elif self.fill_missing == "zero":
            # Zero-fill missing dates (already handled by tensor initialization)
            pass
        else:
            raise ValueError(f"Unknown fill_missing method: {self.fill_missing}")

    def _normalize_cases(self, cases_tensor: Tensor) -> Tensor:
        """
        Apply normalization to case counts.

        Args:
            cases_tensor: Raw case counts tensor

        Returns:
            Normalized case counts tensor
        """
        if self.normalization == "log1p":
            # Log transform: log(1 + x)
            return torch.log1p(cases_tensor)
        elif self.normalization == "standard":
            # Standardization: (x - mean) / std
            mean = cases_tensor.mean()
            std = cases_tensor.std()
            return (cases_tensor - mean) / (std + 1e-8)
        elif self.normalization == "none":
            return cases_tensor
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def get_cases_for_municipalities(
        self, municipality_ids: list[int], date_indices: list[int]
    ) -> Tensor:
        """
        Extract cases for specific municipalities and time points.

        Args:
            municipality_ids: List of municipality IDs
            date_indices: List of date indices

        Returns:
            Tensor of shape [len(municipality_ids), len(date_indices)] with case counts
        """
        if self.cases_tensor is None:
            raise ValueError(
                "Cases tensor not created. Call create_temporal_tensor() first."
            )

        # Map municipality IDs to tensor indices
        tensor_indices = []
        for muni_id in municipality_ids:
            if muni_id in self.municipality_mapping:
                tensor_indices.append(self.municipality_mapping[muni_id])
            else:
                # Handle missing municipalities (fill with zeros)
                tensor_indices.append(-1)

        # Extract cases
        if tensor_indices:
            valid_indices = [i for i in tensor_indices if i >= 0]
            if valid_indices:
                cases = self.cases_tensor[valid_indices][:, date_indices]
            else:
                cases = torch.zeros((len(municipality_ids), len(date_indices)))
        else:
            cases = torch.zeros((len(municipality_ids), len(date_indices)))

        return cases

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded cases data.

        Returns:
            Dictionary with data statistics
        """
        if self.cases_tensor is None:
            raise ValueError(
                "Cases tensor not created. Call create_temporal_tensor() first."
            )

        stats = {
            "total_records": len(self.cases_df),
            "total_cases": int(self.total_cases),
            "active_municipalities": self.active_municipalities,
            "timepoints": self.timepoints,
            "date_range": [str(d) for d in self.date_range],
            "mean_cases_per_day": float(self.cases_tensor.mean().item()),
            "std_cases_per_day": float(self.cases_tensor.std().item()),
            "max_cases_per_day": int(self.cases_tensor.max().item()),
            "zero_days_percentage": float(
                (self.cases_tensor == 0).float().mean().item() * 100
            ),
            "normalization": self.normalization,
        }

        return stats

    def align_with_mobility_data(
        self, mobility_municipalities: list[int], mobility_dates: list[datetime]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Align cases data with mobility data structure.

        Args:
            mobility_municipalities: List of municipality IDs from mobility data
            mobility_dates: List of dates from mobility data

        Returns:
            Tuple of (aligned_cases_tensor, alignment_info)
        """
        if self.cases_tensor is None:
            raise ValueError(
                "Cases tensor not created. Call create_temporal_tensor() first."
            )

        logger.info("Aligning cases data with mobility data")

        # Create alignment mapping
        aligned_municipalities = []
        aligned_cases = []

        for mob_muni in mobility_municipalities:
            if mob_muni in self.municipality_mapping:
                aligned_municipalities.append(mob_muni)
                cases_idx = self.municipality_mapping[mob_muni]
                aligned_cases.append(self.cases_tensor[cases_idx].numpy())
            else:
                # Missing municipality - fill with zeros
                aligned_cases.append(np.zeros(len(mobility_dates)))

        # Convert to tensor - ensure all arrays have same length
        max_length = max(
            len(arr) if isinstance(arr, np.ndarray) else len(arr)
            for arr in aligned_cases
        )
        padded_cases = []
        for arr in aligned_cases:
            if len(arr) < max_length:
                # Pad with zeros
                padded_arr = np.pad(arr, (0, max_length - len(arr)), mode="constant")
                padded_cases.append(padded_arr)
            else:
                padded_cases.append(arr)

        aligned_cases_tensor = torch.tensor(np.array(padded_cases), dtype=torch.float32)

        # Create alignment info
        alignment_info = {
            "mobility_municipalities": len(mobility_municipalities),
            "aligned_municipalities": len(aligned_municipalities),
            "coverage_ratio": len(aligned_municipalities)
            / len(mobility_municipalities),
            "missing_municipalities": len(mobility_municipalities)
            - len(aligned_municipalities),
        }

        logger.info(
            f"Aligned {alignment_info['aligned_municipalities']}/{alignment_info['mobility_municipalities']} municipalities"
        )
        logger.info(f"Coverage ratio: {alignment_info['coverage_ratio']:.1%}")

        return aligned_cases_tensor, alignment_info

    def align_multiple_datasets(
        self,
        datasets: Dict[str, Tensor],
        dataset_dates: Dict[str, List[datetime]],
        dataset_entities: Dict[str, List[Union[int, str]]],
        alignment_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Align multiple datasets using the advanced alignment manager.

        Args:
            datasets: Dictionary of dataset names to tensor data
            dataset_dates: Dictionary of dataset names to date lists
            dataset_entities: Dictionary of dataset names to entity lists
            alignment_config: Configuration for alignment manager

        Returns:
            Dictionary containing aligned datasets and alignment statistics
        """
        logger.info("Starting multi-dataset alignment with advanced alignment manager")

        # Create alignment manager with configuration
        if alignment_config is None:
            alignment_config = {}

        alignment_manager = DatasetAlignmentManager(
            target_dataset_name=alignment_config.get("target_dataset", "cases"),
            padding_strategy=alignment_config.get("padding_strategy", "interpolate"),
            crop_datasets=alignment_config.get("crop_datasets", True),
            alignment_buffer_days=alignment_config.get("alignment_buffer_days", 0),
            interpolation_method=alignment_config.get("interpolation_method", "linear"),
            validate_alignment=alignment_config.get("validate_alignment", True),
        )

        # Perform alignment
        alignment_result = alignment_manager.align_datasets(
            datasets, dataset_dates, dataset_entities
        )

        # Store alignment manager for later use
        self.alignment_manager = alignment_manager

        logger.info("Multi-dataset alignment completed successfully")
        return alignment_result


def create_cases_loader(
    cases_file: str,
    normalization: str = "log1p",
    min_cases_threshold: int = 0,
    fill_missing: str = "forward_fill",
) -> CasesDataLoader:
    """
    Factory function to create a CasesDataLoader instance.

    Args:
        cases_file: Path to CSV file containing case data
        normalization: Method for normalizing case counts
        min_cases_threshold: Minimum case count threshold for filtering
        fill_missing: Method for handling missing data

    Returns:
        Configured CasesDataLoader instance
    """
    loader = CasesDataLoader(
        cases_file=cases_file,
        normalization=normalization,
        min_cases_threshold=min_cases_threshold,
        fill_missing=fill_missing,
    )

    # Load and preprocess data
    loader.load_data()
    loader.preprocess_data()
    loader.create_temporal_tensor()

    return loader
