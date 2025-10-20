"""
Dataset Alignment Manager for Epidemiological Forecasting.

This module provides comprehensive dataset alignment functionality to ensure
all datasets are properly aligned temporally and spatially for COVID-19 forecasting.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class DatasetAlignmentManager:
    """
    Manages alignment of multiple datasets for epidemiological forecasting.

    This class handles:
    - Target-based dataset cropping (all datasets end when target ends)
    - Padding with zeros for initial periods
    - Interpolation preprocessing for padded segments
    - Multi-dataset alignment with flexible strategies
    """

    def __init__(
        self,
        target_dataset_name: str = "cases",
        padding_strategy: str = "interpolate",
        crop_datasets: bool = True,
        alignment_buffer_days: int = 0,
        interpolation_method: str = "linear",
        validate_alignment: bool = True,
    ):
        """
        Initialize the dataset alignment manager.

        Args:
            target_dataset_name: Name of the dataset to use as alignment target
            padding_strategy: Strategy for handling initial periods ('zero', 'interpolate', 'forward_fill')
            crop_datasets: Whether to crop datasets to target end date
            alignment_buffer_days: Number of days to add as buffer before cropping
            interpolation_method: Method for interpolation ('linear', 'cubic', 'spline')
            validate_alignment: Whether to validate alignment results
        """
        self.target_dataset_name = target_dataset_name
        self.padding_strategy = padding_strategy
        self.crop_datasets = crop_datasets
        self.alignment_buffer_days = alignment_buffer_days
        self.interpolation_method = interpolation_method
        self.validate_alignment = validate_alignment

        # Aligned datasets storage
        self.aligned_datasets: Dict[str, Any] = {}
        self.alignment_stats: Dict[str, Any] = {}

    def align_datasets(
        self,
        datasets: Dict[str, Any],
        dataset_dates: Dict[str, List[datetime]],
        dataset_entities: Dict[str, List[Union[int, str]]],
    ) -> Dict[str, Any]:
        """
        Align multiple datasets to a common temporal and spatial structure.

        Args:
            datasets: Dictionary of dataset names to tensor data
            dataset_dates: Dictionary of dataset names to date lists
            dataset_entities: Dictionary of dataset names to entity lists

        Returns:
            Dictionary containing aligned datasets and alignment statistics
        """
        logger.info(
            f"Starting dataset alignment with target: {self.target_dataset_name}"
        )

        if self.target_dataset_name not in datasets:
            raise ValueError(f"Target dataset '{self.target_dataset_name}' not found")

        # Get target dataset information
        target_tensor = datasets[self.target_dataset_name]
        target_dates = dataset_dates[self.target_dataset_name]
        target_entities = dataset_entities[self.target_dataset_name]

        # Determine alignment boundaries
        alignment_start, alignment_end = self._determine_alignment_boundaries(
            dataset_dates, target_dates
        )

        logger.info(f"Alignment boundaries: {alignment_start} to {alignment_end}")

        # Align each dataset
        aligned_datasets = {}
        alignment_stats = {}

        for dataset_name, dataset_tensor in datasets.items():
            logger.debug(f"Aligning dataset: {dataset_name}")

            dates = dataset_dates[dataset_name]
            entities = dataset_entities[dataset_name]

            # Perform alignment
            aligned_tensor, stats = self._align_single_dataset(
                dataset_tensor,
                dates,
                entities,
                alignment_start,
                alignment_end,
                target_entities,
                is_target=(dataset_name == self.target_dataset_name),
            )

            aligned_datasets[dataset_name] = aligned_tensor
            alignment_stats[dataset_name] = stats

            logger.debug(f"  {dataset_name}: {stats}")

        # Validate alignment
        if self.validate_alignment:
            self._validate_alignment(aligned_datasets, alignment_stats)

        # Store results
        self.aligned_datasets = aligned_datasets
        self.alignment_stats = alignment_stats

        result = {
            "aligned_datasets": aligned_datasets,
            "alignment_stats": alignment_stats,
            "alignment_boundaries": (alignment_start, alignment_end),
            "target_entities": target_entities,
            "target_dates": self._create_date_range(alignment_start, alignment_end),
        }

        logger.info("Dataset alignment completed successfully")
        return result

    def _determine_alignment_boundaries(
        self, dataset_dates: Dict[str, List[datetime]], target_dates: List[datetime]
    ) -> Tuple[datetime, datetime]:
        """
        Determine the alignment boundaries based on target dataset and configuration.

        Args:
            dataset_dates: Dictionary of dataset dates
            target_dates: Target dataset dates

        Returns:
            Tuple of (start_date, end_date) for alignment
        """
        target_start = min(target_dates)
        target_end = max(target_dates)

        if self.crop_datasets:
            # Crop all datasets to target end date
            alignment_end = target_end + timedelta(days=self.alignment_buffer_days)

            # Find the earliest start date across all datasets
            all_starts = [min(dates) for dates in dataset_dates.values()]
            alignment_start = min(all_starts)
        else:
            # Use full range of all datasets
            all_starts = [min(dates) for dates in dataset_dates.values()]
            all_ends = [max(dates) for dates in dataset_dates.values()]
            alignment_start = min(all_starts)
            alignment_end = max(all_ends)

        return alignment_start, alignment_end

    def _align_single_dataset(
        self,
        dataset_tensor: Tensor,
        dataset_dates: List[datetime],
        dataset_entities: List[Union[int, str]],
        alignment_start: datetime,
        alignment_end: datetime,
        target_entities: List[Union[int, str]],
        is_target: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Align a single dataset to the target structure.

        Args:
            dataset_tensor: Original dataset tensor
            dataset_dates: Original dataset dates
            dataset_entities: Original dataset entities
            alignment_start: Alignment start date
            alignment_end: Alignment end date
            target_entities: Target entities for spatial alignment
            is_target: Whether this is the target dataset

        Returns:
            Tuple of (aligned_tensor, alignment_statistics)
        """
        # Create target date range
        target_dates = self._create_date_range(alignment_start, alignment_end)
        num_timepoints = len(target_dates)

        # Spatial alignment: align entities
        aligned_tensor, entity_stats = self._align_entities(
            dataset_tensor, dataset_entities, target_entities
        )

        # Temporal alignment: align dates
        aligned_tensor, temporal_stats = self._align_temporal(
            aligned_tensor, dataset_dates, target_dates, is_target
        )

        # Apply padding strategy if needed
        if temporal_stats["padded_timepoints"] > 0:
            aligned_tensor = self._apply_padding_strategy(
                aligned_tensor, temporal_stats["valid_mask"]
            )

        # Combine statistics
        stats = {
            **entity_stats,
            **temporal_stats,
            "original_shape": tuple(dataset_tensor.shape),
            "aligned_shape": tuple(aligned_tensor.shape),
            "is_target": is_target,
        }

        return aligned_tensor, stats

    def _align_entities(
        self,
        dataset_tensor: Tensor,
        dataset_entities: List[Union[int, str]],
        target_entities: List[Union[int, str]],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Align dataset entities to target entities.

        Args:
            dataset_tensor: Dataset tensor [num_entities, num_timepoints]
            dataset_entities: Original entity list
            target_entities: Target entity list

        Returns:
            Tuple of (aligned_tensor, entity_statistics)
        """
        # Create entity mapping
        entity_to_idx = {entity: i for i, entity in enumerate(dataset_entities)}

        # Initialize aligned tensor
        aligned_tensor = torch.zeros((len(target_entities), dataset_tensor.shape[1]))

        # Align entities
        aligned_count = 0
        missing_entities = []

        for i, target_entity in enumerate(target_entities):
            if target_entity in entity_to_idx:
                source_idx = entity_to_idx[target_entity]
                aligned_tensor[i] = dataset_tensor[source_idx]
                aligned_count += 1
            else:
                missing_entities.append(target_entity)

        stats = {
            "aligned_entities": aligned_count,
            "total_entities": len(target_entities),
            "missing_entities": len(missing_entities),
            "entity_coverage": aligned_count / len(target_entities),
        }

        return aligned_tensor, stats

    def _align_temporal(
        self,
        dataset_tensor: Tensor,
        dataset_dates: List[datetime],
        target_dates: List[datetime],
        is_target: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Align dataset temporally to target dates.

        Args:
            dataset_tensor: Spatially aligned tensor [num_entities, num_timepoints]
            dataset_dates: Original dates
            target_dates: Target dates
            is_target: Whether this is the target dataset

        Returns:
            Tuple of (aligned_tensor, temporal_statistics)
        """
        # Create date to index mapping
        date_to_idx = {date: i for i, date in enumerate(dataset_dates)}

        # Initialize aligned tensor
        aligned_tensor = torch.zeros((dataset_tensor.shape[0], len(target_dates)))
        valid_mask = torch.ones(len(target_dates), dtype=torch.bool)

        # Align temporal data
        aligned_timepoints = 0
        missing_dates = []

        for i, target_date in enumerate(target_dates):
            if target_date in date_to_idx:
                source_idx = date_to_idx[target_date]
                aligned_tensor[:, i] = dataset_tensor[:, source_idx]
                aligned_timepoints += 1
            else:
                valid_mask[i] = False
                missing_dates.append(target_date)

        stats = {
            "aligned_timepoints": aligned_timepoints,
            "total_timepoints": len(target_dates),
            "missing_dates": len(missing_dates),
            "temporal_coverage": aligned_timepoints / len(target_dates),
            "valid_mask": valid_mask,
            "padded_timepoints": len(missing_dates),
        }

        return aligned_tensor, stats

    def _apply_padding_strategy(
        self, dataset_tensor: Tensor, valid_mask: Tensor
    ) -> Tensor:
        """
        Apply padding strategy to handle missing timepoints.

        Args:
            dataset_tensor: Tensor with missing timepoints
            valid_mask: Boolean mask indicating valid timepoints

        Returns:
            Tensor with padding applied
        """
        if self.padding_strategy == "zero":
            # Zero padding (already done by initialization)
            return dataset_tensor

        elif self.padding_strategy == "forward_fill":
            # Forward fill missing values
            for i in range(1, len(valid_mask)):
                if not valid_mask[i]:
                    dataset_tensor[:, i] = dataset_tensor[:, i - 1]
            return dataset_tensor

        elif self.padding_strategy == "interpolate":
            # Interpolate missing values
            return self._interpolate_missing_values(dataset_tensor, valid_mask)

        else:
            raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")

    def _interpolate_missing_values(
        self, dataset_tensor: Tensor, valid_mask: Tensor
    ) -> Tensor:
        """
        Interpolate missing values using the specified method.

        Args:
            dataset_tensor: Tensor with missing values
            valid_mask: Boolean mask indicating valid timepoints

        Returns:
            Tensor with interpolated values
        """
        # Convert to numpy for interpolation
        data_np = dataset_tensor.numpy()
        valid_np = valid_mask.numpy()

        # Interpolate each entity's time series
        for entity_idx in range(data_np.shape[0]):
            series = data_np[entity_idx]
            valid_indices = np.where(valid_np)[0]

            if len(valid_indices) < 2:
                # Not enough data for interpolation
                continue

            if self.interpolation_method == "linear":
                series = self._linear_interpolate(series, valid_np)
            elif self.interpolation_method == "cubic":
                series = self._cubic_interpolate(series, valid_np)
            elif self.interpolation_method == "spline":
                series = self._spline_interpolate(series, valid_np)
            elif self.interpolation_method == "smart":
                series = self._smart_interpolate_single(series, valid_np)

        # Convert back to tensor
        return torch.from_numpy(data_np).float()

    def _linear_interpolate(
        self, series: np.ndarray, valid_np: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation for missing values."""
        for i in range(len(series)):
            if not valid_np[i]:
                # Find nearest valid indices
                left_idx = i - 1
                while left_idx >= 0 and not valid_np[left_idx]:
                    left_idx -= 1

                right_idx = i + 1
                while right_idx < len(series) and not valid_np[right_idx]:
                    right_idx += 1

                if left_idx >= 0 and right_idx < len(series):
                    # Linear interpolation
                    alpha = (i - left_idx) / (right_idx - left_idx)
                    series[i] = (1 - alpha) * series[left_idx] + alpha * series[
                        right_idx
                    ]
                elif left_idx >= 0:
                    # Forward fill
                    series[i] = series[left_idx]
                elif right_idx < len(series):
                    # Backward fill
                    series[i] = series[right_idx]

        return series

    def _cubic_interpolate(
        self, series: np.ndarray, valid_np: np.ndarray
    ) -> np.ndarray:
        """Cubic interpolation for missing values."""
        try:
            from scipy.interpolate import interp1d

            valid_indices = np.where(valid_np)[0]

            if len(valid_indices) >= 4:  # Need at least 4 points for cubic
                f = interp1d(
                    valid_indices,
                    series[valid_indices],
                    kind="cubic",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                invalid_indices = np.where(~valid_np)[0]
                series[invalid_indices] = f(invalid_indices)
            else:
                logger.debug(
                    "Not enough points for cubic interpolation, falling back to linear"
                )
                series = self._linear_interpolate(series, valid_np)

        except Exception as e:
            logger.warning(f"Cubic interpolation failed: {e}, falling back to linear")
            series = self._linear_interpolate(series, valid_np)

        return series

    def _spline_interpolate(
        self, series: np.ndarray, valid_np: np.ndarray
    ) -> np.ndarray:
        """Spline interpolation for missing values."""
        try:
            from scipy.interpolate import UnivariateSpline

            valid_indices = np.where(valid_np)[0]

            if len(valid_indices) >= 4:  # Need at least 4 points for spline
                # Create spline with smoothing factor
                spline = UnivariateSpline(
                    valid_indices,
                    series[valid_indices],
                    s=len(valid_indices) * 0.1,  # Smoothing factor
                    k=3,  # Cubic spline
                )
                invalid_indices = np.where(~valid_np)[0]
                series[invalid_indices] = spline(invalid_indices)
            else:
                logger.debug(
                    "Not enough points for spline interpolation, falling back to linear"
                )
                series = self._linear_interpolate(series, valid_np)

        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, falling back to linear")
            series = self._linear_interpolate(series, valid_np)

        return series

    def _validate_interpolated_values(
        self, original_tensor: Tensor, interpolated_tensor: Tensor, valid_mask: Tensor
    ) -> Dict[str, Any]:
        """
        Validate interpolated values and return statistics.

        Args:
            original_tensor: Original tensor with missing values
            interpolated_tensor: Tensor after interpolation
            valid_mask: Boolean mask indicating valid timepoints

        Returns:
            Dictionary with validation statistics
        """
        original_np = original_tensor.numpy()
        interpolated_np = interpolated_tensor.numpy()
        valid_np = valid_mask.numpy()

        # Calculate statistics for interpolated regions
        interpolated_regions = ~valid_np
        validation_stats = {
            "total_interpolated_points": int(np.sum(interpolated_regions)),
            "interpolated_value_range": {
                "min": float(np.min(interpolated_np[interpolated_regions])),
                "max": float(np.max(interpolated_np[interpolated_regions])),
                "mean": float(np.mean(interpolated_np[interpolated_regions])),
                "std": float(np.std(interpolated_np[interpolated_regions])),
            },
            "valid_value_range": {
                "min": float(np.min(original_np[valid_np])),
                "max": float(np.max(original_np[valid_np])),
                "mean": float(np.mean(original_np[valid_np])),
                "std": float(np.std(original_np[valid_np])),
            },
            "out_of_bounds_points": 0,
            "interpolation_quality_score": 0.0,
        }

        # Check for out-of-bounds values (values outside valid range)
        valid_min = validation_stats["valid_value_range"]["min"]
        valid_max = validation_stats["valid_value_range"]["max"]

        out_of_bounds = np.where(
            (interpolated_np[interpolated_regions] < valid_min)
            | (interpolated_np[interpolated_regions] > valid_max)
        )[0]

        validation_stats["out_of_bounds_points"] = len(out_of_bounds)

        # Calculate interpolation quality score (0-1)
        if validation_stats["total_interpolated_points"] > 0:
            in_bounds_ratio = 1.0 - (
                len(out_of_bounds) / validation_stats["total_interpolated_points"]
            )
            validation_stats["interpolation_quality_score"] = in_bounds_ratio

        return validation_stats

    def _smart_interpolation(
        self, dataset_tensor: Tensor, valid_mask: Tensor
    ) -> Tensor:
        """
        Smart interpolation that adapts strategy based on data characteristics.

        Args:
            dataset_tensor: Tensor with missing values
            valid_mask: Boolean mask indicating valid timepoints

        Returns:
            Tensor with interpolated values
        """
        data_np = dataset_tensor.numpy()
        valid_np = valid_mask.numpy()

        # Analyze data characteristics
        valid_indices = np.where(valid_np)[0]
        if len(valid_indices) < 2:
            return dataset_tensor

        # Calculate data characteristics
        valid_values = data_np[valid_np]
        data_variance = np.var(valid_values)
        data_range = np.max(valid_values) - np.min(valid_values)

        # Choose interpolation method based on data characteristics
        if len(valid_indices) < 4:
            # Use linear interpolation for small datasets
            chosen_method = "linear"
        elif data_variance < 0.01 and data_range < 1.0:
            # Use linear interpolation for low-variance data
            chosen_method = "linear"
        elif len(valid_indices) >= 8 and data_variance > 0.1:
            # Use spline interpolation for large, high-variance datasets
            chosen_method = "spline"
        else:
            # Use cubic interpolation as default
            chosen_method = "cubic"

        logger.debug(f"Smart interpolation chose method: {chosen_method}")

        # Temporarily change interpolation method
        original_method = self.interpolation_method
        self.interpolation_method = chosen_method

        # Perform interpolation
        result = self._interpolate_missing_values(dataset_tensor, valid_mask)

        # Restore original method
        self.interpolation_method = original_method

        return result

    def _smart_interpolate_single(
        self, series: np.ndarray, valid_np: np.ndarray
    ) -> np.ndarray:
        """Smart interpolation for a single time series."""
        valid_indices = np.where(valid_np)[0]
        if len(valid_indices) < 2:
            return series

        # Calculate data characteristics
        valid_values = series[valid_np]
        data_variance = np.var(valid_values)
        data_range = np.max(valid_values) - np.min(valid_values)

        # Choose interpolation method based on data characteristics
        if len(valid_indices) < 4:
            # Use linear interpolation for small datasets
            return self._linear_interpolate(series, valid_np)
        elif data_variance < 0.01 and data_range < 1.0:
            # Use linear interpolation for low-variance data
            return self._linear_interpolate(series, valid_np)
        elif len(valid_indices) >= 8 and data_variance > 0.1:
            # Use spline interpolation for large, high-variance datasets
            try:
                return self._spline_interpolate(series, valid_np)
            except:
                logger.debug("Spline interpolation failed, falling back to cubic")
                return self._cubic_interpolate(series, valid_np)
        else:
            # Use cubic interpolation as default
            try:
                return self._cubic_interpolate(series, valid_np)
            except:
                logger.debug("Cubic interpolation failed, falling back to linear")
                return self._linear_interpolate(series, valid_np)

    def _create_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """Create a list of dates from start to end (inclusive)."""
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def _validate_alignment(
        self, aligned_datasets: Dict[str, Tensor], alignment_stats: Dict[str, Any]
    ) -> None:
        """
        Validate that all aligned datasets have consistent shapes.

        Args:
            aligned_datasets: Dictionary of aligned dataset tensors
            alignment_stats: Dictionary of alignment statistics
        """
        # Check that all datasets have the same number of entities
        entity_counts = [tensor.shape[0] for tensor in aligned_datasets.values()]
        if len(set(entity_counts)) > 1:
            raise ValueError(f"Inconsistent entity counts: {entity_counts}")

        # Check that all datasets have the same number of timepoints
        timepoint_counts = [tensor.shape[1] for tensor in aligned_datasets.values()]
        if len(set(timepoint_counts)) > 1:
            raise ValueError(f"Inconsistent timepoint counts: {timepoint_counts}")

        # Check alignment quality
        for dataset_name, stats in alignment_stats.items():
            if stats["entity_coverage"] < 0.5:
                logger.warning(
                    f"Low entity coverage for {dataset_name}: {stats['entity_coverage']:.1%}"
                )
            if stats["temporal_coverage"] < 0.5:
                logger.warning(
                    f"Low temporal coverage for {dataset_name}: {stats['temporal_coverage']:.1%}"
                )

        logger.info("Alignment validation passed")

    def get_alignment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the alignment results.

        Returns:
            Dictionary containing alignment summary statistics
        """
        if not self.alignment_stats:
            return {"status": "no_alignment_performed"}

        summary = {
            "target_dataset": self.target_dataset_name,
            "padding_strategy": self.padding_strategy,
            "interpolation_method": self.interpolation_method,
            "datasets_aligned": len(self.aligned_datasets),
            "alignment_stats": self.alignment_stats,
        }

        # Add overall statistics
        entity_coverages = [
            stats["entity_coverage"] for stats in self.alignment_stats.values()
        ]
        temporal_coverages = [
            stats["temporal_coverage"] for stats in self.alignment_stats.values()
        ]

        summary["mean_entity_coverage"] = np.mean(entity_coverages)
        summary["mean_temporal_coverage"] = np.mean(temporal_coverages)
        summary["min_entity_coverage"] = np.min(entity_coverages)
        summary["min_temporal_coverage"] = np.min(temporal_coverages)

        return summary


def create_alignment_manager(
    target_dataset_name: str = "cases",
    padding_strategy: str = "interpolate",
    crop_datasets: bool = True,
    alignment_buffer_days: int = 0,
    interpolation_method: str = "linear",
    validate_alignment: bool = True,
) -> DatasetAlignmentManager:
    """
    Factory function to create a DatasetAlignmentManager instance.

    Args:
        target_dataset_name: Name of the dataset to use as alignment target
        padding_strategy: Strategy for handling initial periods
        crop_datasets: Whether to crop datasets to target end date
        alignment_buffer_days: Number of days to add as buffer
        interpolation_method: Method for interpolation
        validate_alignment: Whether to validate alignment results

    Returns:
        Configured DatasetAlignmentManager instance
    """
    return DatasetAlignmentManager(
        target_dataset_name=target_dataset_name,
        padding_strategy=padding_strategy,
        crop_datasets=crop_datasets,
        alignment_buffer_days=alignment_buffer_days,
        interpolation_method=interpolation_method,
        validate_alignment=validate_alignment,
    )
