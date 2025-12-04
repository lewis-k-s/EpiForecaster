"""
Multi-dataset alignment processor.

This module handles temporal and spatial alignment of multiple datasets
(cases, mobility, wastewater biomarkers, etc.) into a common framework.
It supports different alignment strategies, validation, and quality reporting.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import interpolate

from ..config import PreprocessingConfig


class AlignmentProcessor:
    """
    Handles multi-dataset temporal and spatial alignment.

    This processor aligns different data sources to common temporal and spatial
    dimensions using configurable strategies. It supports:

    - Temporal alignment (interpolation, nearest neighbor, spline)
    - Spatial alignment (region matching, coordinate mapping)
    - Validation of alignment quality
    - Generation of comprehensive alignment reports
    - Handling of missing data and gaps

    The output ensures all datasets share the same temporal indexing and
    spatial dimensions for downstream processing.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the alignment processor.

        Args:
            config: Preprocessing configuration with alignment options
        """
        self.config = config
        self.alignment_strategy = config.alignment_strategy
        self.target_dataset = config.target_dataset
        self.crop_datasets = config.crop_datasets
        self.validate_alignment = config.validate_alignment

    def align_datasets(self, datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Align all datasets to common temporal and spatial grid.

        Args:
            datasets: Dictionary of datasets from individual processors
                - 'cases': {cases_tensor, target_sequences, region_metadata}
                - 'mobility': {node_features, edge_index, edge_attr, node_coords}
                - 'edar': {edar_features, edar_attention_mask, edar_metadata}

        Returns:
            Dictionary with aligned datasets and alignment metadata:
            - aligned_datasets: Dictionary with temporally/spatially aligned data
            - alignment_metadata: Alignment quality information
            - alignment_report: Detailed alignment statistics
        """
        print(f"Aligning datasets using strategy: {self.alignment_strategy}")

        # Extract temporal ranges from all datasets
        temporal_ranges = self._extract_temporal_ranges(datasets)

        # Determine common temporal range
        common_time_range = self._determine_common_temporal_range(temporal_ranges)

        # Align temporal dimensions
        aligned_temporal = self._align_temporal_dimensions(datasets, common_time_range)

        # Align spatial dimensions
        aligned_spatial = self._align_spatial_dimensions(aligned_temporal)

        # Validate alignment quality
        alignment_metadata = {}
        if self.validate_alignment:
            alignment_metadata = self._validate_alignment(aligned_spatial)

        # Generate alignment report
        alignment_report = self._generate_alignment_report(
            datasets, aligned_spatial, alignment_metadata
        )

        return {
            "aligned_datasets": aligned_spatial,
            "alignment_metadata": alignment_metadata,
            "alignment_report": alignment_report,
        }

    def _extract_temporal_ranges(
        self, datasets: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, datetime]]:
        """
        Extract temporal ranges from all datasets.

        Args:
            datasets: Dictionary of input datasets

        Returns:
            Dictionary with temporal range information for each dataset
        """
        temporal_ranges = {}

        # Cases data
        if "cases" in datasets:
            cases_metadata = datasets["cases"]["metadata"]
            start_date = datetime.fromisoformat(cases_metadata["date_range"]["start"])
            end_date = datetime.fromisoformat(cases_metadata["date_range"]["end"])
            temporal_ranges["cases"] = {
                "start": start_date,
                "end": end_date,
                "num_timepoints": cases_metadata["num_timepoints"],
            }

        # Mobility data
        if "mobility" in datasets:
            mobility_metadata = datasets["mobility"]["metadata"]
            time_range = mobility_metadata.get("time_range", {})
            if "start" in time_range and "end" in time_range:
                start_date = datetime.fromisoformat(time_range["start"])
                end_date = datetime.fromisoformat(time_range["end"])
                temporal_ranges["mobility"] = {
                    "start": start_date,
                    "end": end_date,
                    "num_timepoints": mobility_metadata.get("time_steps", 0),
                }

        # EDAR data
        if "edar" in datasets:
            edar_metadata = datasets["edar"]["metadata"]
            start_date = datetime.fromisoformat(edar_metadata["date_range"]["start"])
            end_date = datetime.fromisoformat(edar_metadata["date_range"]["end"])
            temporal_ranges["edar"] = {
                "start": start_date,
                "end": end_date,
                "num_timepoints": edar_metadata["num_timepoints"],
            }

        return temporal_ranges

    def _determine_common_temporal_range(
        self, temporal_ranges: dict[str, dict[str, datetime]]
    ) -> dict[str, Any]:
        """
        Determine common temporal range across all datasets.

        Args:
            temporal_ranges: Dictionary with temporal ranges for each dataset

        Returns:
            Dictionary with common temporal range information
        """
        if not temporal_ranges:
            raise ValueError("No datasets with temporal information found")

        # Find intersection of all ranges
        common_start = max(r["start"] for r in temporal_ranges.values())
        common_end = min(r["end"] for r in temporal_ranges.values())

        # Override with config dates if cropping is enabled
        if self.crop_datasets:
            common_start = max(common_start, self.config.start_date)
            common_end = min(common_end, self.config.end_date)

        if common_start >= common_end:
            raise ValueError(
                f"No overlapping temporal range found: {common_start} to {common_end}"
            )

        # Create common date range
        common_dates = pd.date_range(start=common_start, end=common_end, freq="D")

        return {
            "start": common_start,
            "end": common_end,
            "dates": common_dates,
            "num_timepoints": len(common_dates),
            "original_ranges": temporal_ranges,
        }

    def _align_temporal_dimensions(
        self, datasets: dict[str, dict[str, Any]], common_time_range: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """
        Align temporal dimensions of all datasets.

        Args:
            datasets: Input datasets
            common_time_range: Common temporal range information

        Returns:
            Dictionary with temporally aligned datasets
        """
        aligned_datasets = datasets.copy()
        common_dates = common_time_range["dates"]

        # Align cases data (target dataset)
        if "cases" in datasets:
            aligned_datasets["cases"] = self._align_cases_temporal(
                datasets["cases"], common_dates
            )

        # Align mobility data
        if "mobility" in datasets:
            aligned_datasets["mobility"] = self._align_mobility_temporal(
                datasets["mobility"], common_dates
            )

        # Align EDAR data
        if "edar" in datasets:
            aligned_datasets["edar"] = self._align_edar_temporal(
                datasets["edar"], common_dates
            )

        return aligned_datasets

    def _align_cases_temporal(
        self, cases_data: dict[str, Any], common_dates: pd.DatetimeIndex
    ) -> dict[str, Any]:
        """
        Align cases data to common temporal range.

        Args:
            cases_data: Cases dataset
            common_dates: Common date range

        Returns:
            Temporally aligned cases data
        """
        cases_metadata = cases_data["metadata"]
        original_start = datetime.fromisoformat(cases_metadata["date_range"]["start"])

        # Calculate time indices for common range
        time_indices = []
        for date in common_dates:
            time_idx = (date - original_start).days
            if 0 <= time_idx < cases_metadata["num_timepoints"]:
                time_indices.append(time_idx)
            else:
                # Handle out-of-range dates by clamping
                time_idx = max(0, min(time_idx, cases_metadata["num_timepoints"] - 1))
                time_indices.append(time_idx)

        # Align cases tensor
        cases_tensor = cases_data["cases_tensor"]
        aligned_cases = cases_tensor[time_indices]

        # Align target sequences
        target_sequences = cases_data["target_sequences"]
        aligned_targets = target_sequences[time_indices]

        return {
            "cases_tensor": aligned_cases,
            "target_sequences": aligned_targets,
            "region_metadata": cases_data["region_metadata"],
            "metadata": {
                **cases_data["metadata"],
                "date_range": {
                    "start": common_dates[0].isoformat(),
                    "end": common_dates[-1].isoformat(),
                },
                "num_timepoints": len(common_dates),
                "alignment_indices": time_indices,
            },
        }

    def _align_mobility_temporal(
        self, mobility_data: dict[str, Any], common_dates: pd.DatetimeIndex
    ) -> dict[str, Any]:
        """
        Align mobility data to common temporal range.

        Args:
            mobility_data: Mobility dataset
            common_dates: Common date range

        Returns:
            Temporally aligned mobility data
        """
        # Get edge_attr if available (temporal edge features)
        edge_attr = mobility_data.get("edge_attr")
        if edge_attr is not None:
            # Interpolate edge features to common time range
            aligned_edge_attr = self._interpolate_temporal_data(
                edge_attr, len(common_dates), self.alignment_strategy
            )
        else:
            aligned_edge_attr = None

        return {
            "node_features": mobility_data[
                "node_features"
            ],  # Static, no temporal alignment needed
            "edge_index": mobility_data[
                "edge_index"
            ],  # Static, no temporal alignment needed
            "edge_attr": aligned_edge_attr,
            "node_coords": mobility_data[
                "node_coords"
            ],  # Static, no temporal alignment needed
            "metadata": {
                **mobility_data["metadata"],
                "aligned_timepoints": len(common_dates),
            },
        }

    def _align_edar_temporal(
        self, edar_data: dict[str, Any], common_dates: pd.DatetimeIndex
    ) -> dict[str, Any]:
        """
        Align EDAR data to common temporal range.

        Args:
            edar_data: EDAR dataset
            common_dates: Common date range

        Returns:
            Temporally aligned EDAR data
        """
        edar_features = edar_data["edar_features"]
        original_timepoints = edar_features.shape[0]

        if original_timepoints != len(common_dates):
            # Interpolate to common time range
            aligned_features = self._interpolate_temporal_data(
                edar_features, len(common_dates), self.alignment_strategy
            )
        else:
            aligned_features = edar_features

        return {
            "edar_features": aligned_features,
            "edar_attention_mask": edar_data[
                "edar_attention_mask"
            ],  # Static, no alignment needed
            "edar_metadata": edar_data["edar_metadata"],
            "metadata": {
                **edar_data["metadata"],
                "date_range": {
                    "start": common_dates[0].isoformat(),
                    "end": common_dates[-1].isoformat(),
                },
                "num_timepoints": len(common_dates),
            },
        }

    def _interpolate_temporal_data(
        self, data: torch.Tensor, target_length: int, strategy: str
    ) -> torch.Tensor:
        """
        Interpolate temporal data to target length.

        Args:
            data: Input temporal tensor [time, ...]
            target_length: Desired temporal length
            strategy: Interpolation strategy

        Returns:
            Interpolated tensor with target temporal length
        """
        original_length = data.shape[0]

        if original_length == target_length:
            return data

        # Convert to numpy for interpolation
        data_np = data.numpy()

        # Create original and target time indices
        original_indices = np.linspace(0, 1, original_length)
        target_indices = np.linspace(0, 1, target_length)

        # Reshape for interpolation
        original_shape = data_np.shape
        flattened_data = data_np.reshape(original_length, -1)

        # Perform interpolation
        if strategy == "interpolate":
            # Linear interpolation
            interpolator = interpolate.interp1d(
                original_indices,
                flattened_data,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_flat = interpolator(target_indices)
        elif strategy == "nearest":
            # Nearest neighbor interpolation
            interpolator = interpolate.interp1d(
                original_indices,
                flattened_data,
                axis=0,
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_flat = interpolator(target_indices)
        elif strategy == "spline":
            # Cubic spline interpolation (requires enough points)
            if original_length >= 4:
                interpolator = interpolate.interp1d(
                    original_indices,
                    flattened_data,
                    axis=0,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interpolated_flat = interpolator(target_indices)
            else:
                # Fall back to linear for small datasets
                interpolator = interpolate.interp1d(
                    original_indices,
                    flattened_data,
                    axis=0,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interpolated_flat = interpolator(target_indices)
        else:
            raise ValueError(f"Unknown interpolation strategy: {strategy}")

        # Reshape back to original feature dimensions
        interpolated_data = interpolated_flat.reshape(
            target_length, *original_shape[1:]
        )

        return torch.from_numpy(interpolated_data).float()

    def _align_spatial_dimensions(
        self, datasets: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Align spatial dimensions of all datasets.

        Args:
            datasets: Temporally aligned datasets

        Returns:
            Spatially aligned datasets
        """
        # For now, assume spatial dimensions are already aligned
        # In the future, this could handle coordinate transformations,
        # region matching, etc.

        return datasets

    def _validate_alignment(
        self, aligned_datasets: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate alignment quality across datasets.

        Args:
            aligned_datasets: Aligned datasets

        Returns:
            Alignment validation metadata
        """
        validation_results = {}

        # Check temporal consistency
        temporal_validation = self._validate_temporal_alignment(aligned_datasets)
        validation_results["temporal"] = temporal_validation

        # Check spatial consistency
        spatial_validation = self._validate_spatial_alignment(aligned_datasets)
        validation_results["spatial"] = spatial_validation

        # Check data quality
        quality_validation = self._validate_data_quality(aligned_datasets)
        validation_results["quality"] = quality_validation

        return validation_results

    def _validate_temporal_alignment(
        self, datasets: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate temporal alignment consistency."""
        timepoints = {}
        for name, data in datasets.items():
            if "metadata" in data:
                num_timepoints = data["metadata"].get("num_timepoints")
                if num_timepoints:
                    timepoints[name] = num_timepoints

        # Check if all datasets have same number of timepoints
        unique_timepoints = set(timepoints.values())
        is_consistent = len(unique_timepoints) <= 1

        return {
            "is_consistent": is_consistent,
            "timepoints_per_dataset": timepoints,
            "unique_timepoint_counts": list(unique_timepoints),
        }

    def _validate_spatial_alignment(
        self, datasets: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate spatial alignment consistency."""
        spatial_info = {}

        # Check cases regions
        if "cases" in datasets:
            cases_data = datasets["cases"]
            if "region_metadata" in cases_data:
                spatial_info["cases"] = {
                    "num_regions": cases_data["region_metadata"]["num_regions"]
                }

        # Check mobility nodes
        if "mobility" in datasets:
            mobility_data = datasets["mobility"]
            if "metadata" in mobility_data:
                spatial_info["mobility"] = {
                    "num_nodes": mobility_data["metadata"]["num_nodes"]
                }

        # Check EDAR sites
        if "edar" in datasets:
            edar_data = datasets["edar"]
            if "metadata" in edar_data:
                spatial_info["edar"] = {
                    "num_edar_sites": edar_data["metadata"]["num_edar_sites"]
                }

        return {"spatial_dimensions": spatial_info}

    def _validate_data_quality(
        self, datasets: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate data quality metrics."""
        quality_metrics = {}

        for name, data in datasets.items():
            dataset_quality = {}

            # Check for missing values
            if name == "cases" and "cases_tensor" in data:
                tensor = data["cases_tensor"]
                dataset_quality["missing_percentage"] = float(
                    (tensor == 0).float().mean() * 100
                )
                dataset_quality["has_nan"] = torch.isnan(tensor).any().item()

            elif name == "mobility" and "node_features" in data:
                tensor = data["node_features"]
                dataset_quality["has_nan"] = torch.isnan(tensor).any().item()

            elif name == "edar" and "edar_features" in data:
                tensor = data["edar_features"]
                dataset_quality["has_nan"] = torch.isnan(tensor).any().item()
                dataset_quality["zero_percentage"] = float(
                    (tensor == 0).float().mean() * 100
                )

            if dataset_quality:
                quality_metrics[name] = dataset_quality

        return quality_metrics

    def _generate_alignment_report(
        self,
        original_datasets: dict[str, dict[str, Any]],
        aligned_datasets: dict[str, dict[str, Any]],
        validation_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate comprehensive alignment report.

        Args:
            original_datasets: Original datasets before alignment
            aligned_datasets: Datasets after alignment
            validation_metadata: Validation results

        Returns:
            Comprehensive alignment report
        """
        report = {
            "alignment_strategy": self.alignment_strategy,
            "target_dataset": self.target_dataset,
            "crop_datasets": self.crop_datasets,
            "datasets_aligned": list(aligned_datasets.keys()),
            "temporal_alignment": self._summarize_temporal_alignment(
                original_datasets, aligned_datasets
            ),
            "spatial_alignment": self._summarize_spatial_alignment(aligned_datasets),
            "validation_results": validation_metadata,
            "data_loss": self._compute_data_loss(original_datasets, aligned_datasets),
        }

        return report

    def _summarize_temporal_alignment(
        self,
        original_datasets: dict[str, dict[str, Any]],
        aligned_datasets: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize temporal alignment results."""
        summary = {}

        for name in aligned_datasets.keys():
            if name in original_datasets:
                orig_metadata = original_datasets[name].get("metadata", {})
                aligned_metadata = aligned_datasets[name].get("metadata", {})

                orig_timepoints = orig_metadata.get("num_timepoints", 0)
                aligned_timepoints = aligned_metadata.get("num_timepoints", 0)

                summary[name] = {
                    "original_timepoints": orig_timepoints,
                    "aligned_timepoints": aligned_timepoints,
                    "timepoint_change": aligned_timepoints - orig_timepoints,
                }

        return summary

    def _summarize_spatial_alignment(
        self, aligned_datasets: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Summarize spatial alignment results."""
        summary = {}

        for name, data in aligned_datasets.items():
            if name == "cases":
                if "region_metadata" in data:
                    summary[name] = {
                        "type": "regions",
                        "count": data["region_metadata"]["num_regions"],
                    }
            elif name == "mobility":
                if "metadata" in data:
                    summary[name] = {
                        "type": "nodes",
                        "count": data["metadata"]["num_nodes"],
                    }
            elif name == "edar":
                if "metadata" in data:
                    summary[name] = {
                        "type": "edar_sites",
                        "count": data["metadata"]["num_edar_sites"],
                    }

        return summary

    def _compute_data_loss(
        self,
        original_datasets: dict[str, dict[str, Any]],
        aligned_datasets: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute data loss statistics from alignment."""
        loss_stats = {}

        for name in aligned_datasets.keys():
            if name in original_datasets:
                orig_metadata = original_datasets[name].get("metadata", {})
                aligned_metadata = aligned_datasets[name].get("metadata", {})

                # Compute temporal coverage change
                orig_timepoints = orig_metadata.get("num_timepoints", 0)
                aligned_timepoints = aligned_metadata.get("num_timepoints", 0)

                if orig_timepoints > 0:
                    coverage_ratio = aligned_timepoints / orig_timepoints
                    loss_percentage = (1 - coverage_ratio) * 100
                else:
                    coverage_ratio = 1.0
                    loss_percentage = 0.0

                loss_stats[name] = {
                    "coverage_ratio": coverage_ratio,
                    "data_loss_percentage": loss_percentage,
                }

        return loss_stats
