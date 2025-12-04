"""
Main offline preprocessing pipeline for EpiForecaster.

This module orchestrates the complete preprocessing workflow, from raw data
loading to canonical dataset creation. It coordinates individual processors,
handles validation, and provides comprehensive reporting throughout the
process.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..dataset_storage import DatasetStorage
from ..epi_batch import EpiBatch
from .config import PreprocessingConfig
from .processors.alignment_processor import AlignmentProcessor
from .processors.cases_processor import CasesProcessor
from .processors.edar_processor import EDARProcessor
from .processors.mobility_processor import MobilityProcessor


class OfflinePreprocessingPipeline:
    """
    Complete offline preprocessing pipeline with comprehensive validation.

    This pipeline orchestrates the conversion of raw epidemiological data into
    canonical EpiBatch datasets. It handles all preprocessing steps including:

    1. Loading raw data from various sources (NetCDF, CSV, Excel)
    2. Processing each data type with specialized processors
    3. Multi-dataset temporal and spatial alignment
    4. Graph construction and feature engineering
    5. Creation of temporal sequences for forecasting
    6. Comprehensive validation and quality reporting
    7. Efficient storage in Zarr format

    The pipeline is designed for one-time execution per dataset configuration,
    producing persistent canonical datasets for efficient training.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the preprocessing pipeline.

        Args:
            config: Comprehensive preprocessing configuration
        """
        self.config = config
        self.processors = self._init_processors()

        # Initialize state tracking
        self.pipeline_state = {
            "start_time": datetime.now(),
            "current_stage": "initialization",
            "completed_stages": [],
            "errors": [],
            "warnings": [],
        }

    def _init_processors(self) -> dict[str, Any]:
        """Initialize individual data processors."""
        return {
            "mobility": MobilityProcessor(self.config),
            "cases": CasesProcessor(self.config),
            "edar": EDARProcessor(self.config),
            "alignment": AlignmentProcessor(self.config),
        }

    def run(self) -> Path:
        """
        Execute complete preprocessing pipeline with validation at each step.

        Returns:
            Path to the generated Zarr dataset
        """
        print("=" * 60)
        print("EPIFORECASTER OFFLINE PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Dataset: {self.config.dataset_name}")
        print(
            f"Temporal range: {self.config.start_date.date()} to {self.config.end_date.date()}"
        )
        print(f"Forecast horizon: {self.config.forecast_horizon} days")
        print()

        try:
            # Stage 1: Load and process raw data sources
            raw_data = self._load_raw_sources()

            # Stage 2: Multi-dataset temporal and spatial alignment
            aligned_data = self._align_datasets(raw_data)

            # Stage 3: Build graph structures and create EpiBatch objects
            epi_batches = self._create_epibatches(aligned_data)

            # Stage 4: Final validation and persistence
            output_path = self._validate_and_save_dataset(epi_batches)

            # Stage 5: Generate comprehensive report
            self._generate_final_report(epi_batches, output_path)

            total_time = datetime.now() - self.pipeline_state["start_time"]
            print("=" * 60)
            print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time}")
            print(f"Dataset saved to: {output_path}")
            print("=" * 60)

            return output_path

        except Exception as e:
            self.pipeline_state["errors"].append(str(e))
            print(f"PIPELINE FAILED: {str(e)}")
            raise

    def _load_raw_sources(self) -> dict[str, dict[str, Any]]:
        """Load and process raw data sources."""
        self._update_stage("loading_raw_data")
        print("Stage 1: Loading and processing raw data sources")
        print("-" * 50)

        raw_data = {}

        # Process cases data (required)
        print("Processing cases data...")
        try:
            cases_data = self.processors["cases"].process(self.config.cases_file)
            raw_data["cases"] = cases_data
            print(
                f"  ✓ Processed {cases_data['metadata']['num_timepoints']} timepoints "
                f"for {cases_data['metadata']['num_regions']} regions"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process cases data: {str(e)}") from e

        # Prepare region IDs for mobility alignment
        case_region_ids: list[str] | None = None
        if "cases" in raw_data:
            regions = raw_data["cases"]["region_metadata"]["unique_regions"]
            canonical_ids: list[str] = []
            for region in regions:
                try:
                    region_int = int(region)
                except ValueError:
                    continue
                if region_int < 0:
                    continue
                canonical_ids.append(str(region_int).zfill(5))
            case_region_ids = canonical_ids

        # Process mobility data (optional)
        if self.config.mobility_path:
            print("Processing mobility data...")
            try:
                mobility_data = self.processors["mobility"].process(
                    self.config.mobility_path,
                    population_data=None,
                    region_ids=case_region_ids,
                )
                raw_data["mobility"] = mobility_data
                print(
                    f"  ✓ Processed mobility data for {mobility_data['metadata']['num_nodes']} nodes "
                    f"with {mobility_data['metadata']['num_edges']} edges"
                )
            except Exception as e:
                self.pipeline_state["warnings"].append(
                    f"Failed to process mobility data: {str(e)}"
                )
                print(
                    "  ⚠ Warning: Failed to process mobility data, continuing without it"
                )
        else:
            print("Skipping mobility data (not configured for base model)")

        # Process wastewater data (optional)
        if self.config.wastewater_file:
            print("Processing wastewater data...")
            try:
                edar_data = self.processors["edar"].process(self.config.wastewater_file)
                raw_data["edar"] = edar_data
                print(
                    f"  ✓ Processed {edar_data['metadata']['num_edar_sites']} EDAR sites"
                )
            except Exception as e:
                self.pipeline_state["warnings"].append(
                    f"Failed to process EDAR data: {str(e)}"
                )
                print("  ⚠ Warning: Failed to process EDAR data, continuing without it")

        print()
        return raw_data

    def _align_datasets(
        self, raw_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Align datasets temporally and spatially."""
        self._update_stage("aligning_datasets")
        print("Stage 2: Multi-dataset temporal and spatial alignment")
        print("-" * 50)

        try:
            alignment_result = self.processors["alignment"].align_datasets(raw_data)

            aligned_data = alignment_result["aligned_datasets"]
            alignment_metadata = alignment_result["alignment_metadata"]
            alignment_report = alignment_result["alignment_report"]

            # Store alignment report in pipeline state
            self.pipeline_state["alignment_report"] = alignment_report

            print("Alignment Summary:")
            print(f"  ✓ Temporal alignment strategy: {self.config.alignment_strategy}")
            print(f"  ✓ Target dataset: {self.config.target_dataset}")

            # Report temporal alignment results
            if "temporal_alignment" in alignment_report:
                temporal = alignment_report["temporal_alignment"]
                for dataset, info in temporal.items():
                    print(
                        f"  ✓ {dataset}: {info['original_timepoints']} → {info['aligned_timepoints']} timepoints"
                    )

            # Report data loss
            if "data_loss" in alignment_report:
                loss = alignment_report["data_loss"]
                for dataset, info in loss.items():
                    print(
                        f"  ✓ {dataset}: {info['coverage_ratio']:.2%} coverage retained "
                        f"({info['data_loss_percentage']:.1f}% loss)"
                    )

            print()
            return aligned_data

        except Exception as e:
            raise RuntimeError(f"Dataset alignment failed: {str(e)}") from e

    def _create_epibatches(
        self, aligned_data: dict[str, dict[str, Any]]
    ) -> list[EpiBatch]:
        """Create EpiBatch objects from aligned data."""
        self._update_stage("creating_epibatches")
        print("Stage 3: Creating canonical EpiBatch objects")
        print("-" * 50)

        epi_batches = []

        # Extract common temporal information
        cases_data = aligned_data["cases"]
        num_timepoints = cases_data["metadata"]["num_timepoints"]

        # Get temporal dates
        start_date = datetime.fromisoformat(
            cases_data["metadata"]["date_range"]["start"]
        )
        dates = [start_date + pd.Timedelta(days=i) for i in range(num_timepoints)]

        # Extract common spatial information
        num_nodes = cases_data["metadata"]["num_regions"]

        # Extract graph structure from mobility data if available, otherwise create minimal graph
        if "mobility" in aligned_data:
            mobility_data = aligned_data["mobility"]
            edge_index = mobility_data["edge_index"]
            edge_attr = mobility_data.get("edge_attr")
            mobility_node_features = mobility_data[
                "node_features"
            ]  # [num_nodes, mobility_features]
        else:
            # Create minimal graph structure for base model
            print("Creating minimal graph structure (no mobility data)")
            # Create a simple identity-based edge connectivity (no edges for base model)
            edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
            edge_attr = None
            # Create dummy mobility features (all zeros)
            mobility_node_features = torch.zeros(
                num_nodes, 1
            )  # [num_nodes, 1] dummy feature

        # Process EDAR data if available
        edar_features = None
        edar_attention_mask = None
        if "edar" in aligned_data:
            edar_data = aligned_data["edar"]
            edar_features = edar_data["edar_features"]
            edar_attention_mask = edar_data["edar_attention_mask"]

        print(f"Creating {num_timepoints} EpiBatch objects...")

        # Create EpiBatch for each timepoint
        for time_idx in range(num_timepoints):
            timestamp = dates[time_idx]

            # Extract case features (node features = cases + mobility stats)
            cases_tensor = cases_data["cases_tensor"][time_idx]  # [num_nodes]
            target_sequences = cases_data["target_sequences"][
                time_idx
            ]  # [num_nodes, horizon]

            # Use already extracted mobility node features (or dummy features if no mobility data)

            # Create combined node features
            node_features = torch.cat(
                [
                    cases_tensor.unsqueeze(-1),  # [num_nodes, 1]
                    mobility_node_features,  # [num_nodes, mobility_features]
                ],
                dim=-1,
            )  # [num_nodes, 1 + mobility_features]

            # Extract edge attributes for this timepoint
            time_edge_attr = None
            if edge_attr is not None:
                time_edge_attr = edge_attr[time_idx]  # [num_edges, edge_features]

            # Extract EDAR features for this timepoint
            time_edar_features = None
            if edar_features is not None:
                time_edar_features = edar_features[
                    time_idx
                ]  # [num_edar_sites, edar_features]

            # Create EpiBatch with proper batch dimensions (each region is a batch item)
            batch = EpiBatch(
                batch_id=f"{self.config.dataset_name}_{time_idx:04d}",
                timestamp=timestamp,
                num_nodes=num_nodes,
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=time_edge_attr,
                time_index=torch.arange(
                    num_nodes
                ),  # [num_nodes] region indices as time_index
                sequence_length=self.config.sequence_length,
                target_sequences=target_sequences,  # [num_nodes, forecast_horizon] already correct
                region_embeddings=None,  # Will be added in future
                edar_features=time_edar_features,
                edar_attention_mask=edar_attention_mask,
                metadata={
                    "dataset_name": self.config.dataset_name,
                    "preprocessing_config": self.config.__dict__.copy(),
                    "time_index": time_idx,
                    "data_sources": list(aligned_data.keys()),
                },
            )

            epi_batches.append(batch)

            # Progress indicator
            if (time_idx + 1) % 100 == 0 or time_idx == num_timepoints - 1:
                print(f"  Processed {time_idx + 1}/{num_timepoints} timepoints")

        print(f"  ✓ Created {len(epi_batches)} EpiBatch objects")
        print(f"  ✓ Node feature dimension: {epi_batches[0].feature_dim}")
        print(f"  ✓ Number of edges: {epi_batches[0].num_edges}")
        if edar_features is not None:
            print(
                f"  ✓ EDAR features: {edar_features.shape[1]} sites, {edar_features.shape[2]} features"
            )

        print()
        return epi_batches

    def _validate_and_save_dataset(self, epi_batches: list[EpiBatch]) -> Path:
        """Validate final dataset and save to Zarr format."""
        self._update_stage("validating_and_saving")
        print("Stage 4: Final validation and dataset persistence")
        print("-" * 50)

        # Validate dataset consistency
        print("Validating dataset consistency...")
        self._validate_dataset_consistency(epi_batches)

        # Save dataset
        print("Saving dataset to Zarr format...")
        output_path = self.config.get_output_dataset_path()

        try:
            DatasetStorage.save_dataset(
                dataset=epi_batches,
                path=output_path,
                dataset_name=self.config.dataset_name,
                compression=self.config.compression,
            )
            print(f"  ✓ Dataset saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {str(e)}") from e

        # Validate saved dataset
        print("Validating saved dataset...")
        validation_result = DatasetStorage.validate_dataset(output_path)
        if validation_result["valid"]:
            print("  ✓ Saved dataset validation passed")
        else:
            print("  ⚠ Saved dataset validation issues found:")
            for issue in validation_result["issues"]:
                print(f"    - {issue}")

        print()
        return output_path

    def _validate_dataset_consistency(self, epi_batches: list[EpiBatch]):
        """Validate consistency across all EpiBatch objects."""
        if not epi_batches:
            raise ValueError("No EpiBatch objects to validate")

        first_batch = epi_batches[0]
        num_nodes = first_batch.num_nodes
        feature_dim = first_batch.feature_dim
        num_edges = first_batch.num_edges
        forecast_horizon = first_batch.forecast_horizon

        inconsistencies = []

        for i, batch in enumerate(epi_batches):
            # Check node consistency
            if batch.num_nodes != num_nodes:
                inconsistencies.append(
                    f"Batch {i}: num_nodes mismatch ({batch.num_nodes} != {num_nodes})"
                )

            if batch.feature_dim != feature_dim:
                inconsistencies.append(
                    f"Batch {i}: feature_dim mismatch ({batch.feature_dim} != {feature_dim})"
                )

            # Check edge consistency
            if batch.num_edges != num_edges:
                inconsistencies.append(
                    f"Batch {i}: num_edges mismatch ({batch.num_edges} != {num_edges})"
                )

            # Check temporal consistency
            if batch.forecast_horizon != forecast_horizon:
                inconsistencies.append(
                    f"Batch {i}: forecast_horizon mismatch ({batch.forecast_horizon} != {forecast_horizon})"
                )

            # Check for NaN values
            if torch.isnan(batch.node_features).any():
                inconsistencies.append(f"Batch {i}: NaN values in node_features")

            if torch.isnan(batch.target_sequences).any():
                inconsistencies.append(f"Batch {i}: NaN values in target_sequences")

        if inconsistencies:
            error_msg = "Dataset consistency validation failed:\n" + "\n".join(
                inconsistencies
            )
            raise ValueError(error_msg)

        print(f"  ✓ Validated {len(epi_batches)} batches for consistency")

    def _generate_final_report(self, epi_batches: list[EpiBatch], output_path: Path):
        """Generate comprehensive final report."""
        self._update_stage("generating_report")
        print("Stage 5: Generating comprehensive report")
        print("-" * 50)

        # Compute dataset statistics
        dataset_stats = self._compute_dataset_statistics(epi_batches)

        # Generate report
        report = {
            "pipeline_info": {
                "dataset_name": self.config.dataset_name,
                "start_time": self.pipeline_state["start_time"].isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_minutes": (
                    datetime.now() - self.pipeline_state["start_time"]
                ).total_seconds()
                / 60,
                "completed_stages": self.pipeline_state["completed_stages"],
                "errors": self.pipeline_state["errors"],
                "warnings": self.pipeline_state["warnings"],
            },
            "configuration": self.config.summary(),
            "dataset_statistics": dataset_stats,
            "alignment_report": self.pipeline_state.get("alignment_report", {}),
            "output_info": {
                "dataset_path": str(output_path),
                "dataset_size_mb": self._get_dataset_size_mb(output_path),
                "num_timepoints": len(epi_batches),
                "compression": self.config.compression,
            },
        }

        # Save report
        report_path = output_path.parent / f"{self.config.dataset_name}_report.json"
        import json

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"  ✓ Final report saved to {report_path}")
        print()

        # Print summary statistics
        print("Dataset Summary:")
        print(f"  • Timepoints: {dataset_stats['num_timepoints']}")
        print(f"  • Nodes: {dataset_stats['num_nodes']}")
        print(f"  • Edges: {dataset_stats['num_edges']}")
        print(f"  • Feature dimension: {dataset_stats['feature_dim']}")
        print(f"  • Forecast horizon: {dataset_stats['forecast_horizon']}")
        print(f"  • Dataset size: {report['output_info']['dataset_size_mb']:.1f} MB")

        if dataset_stats.get("has_edar_data", False):
            print(f"  • EDAR sites: {dataset_stats.get('num_edar_sites', 0)}")

        if self.pipeline_state["warnings"]:
            print(f"\nWarnings ({len(self.pipeline_state['warnings'])}):")
            for warning in self.pipeline_state["warnings"]:
                print(f"  ⚠ {warning}")

        print()

    def _compute_dataset_statistics(
        self, epi_batches: list[EpiBatch]
    ) -> dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        if not epi_batches:
            return {}

        first_batch = epi_batches[0]

        # Basic statistics
        stats = {
            "num_timepoints": len(epi_batches),
            "num_nodes": first_batch.num_nodes,
            "num_edges": first_batch.num_edges,
            "feature_dim": first_batch.feature_dim,
            "forecast_horizon": first_batch.forecast_horizon,
            "has_edar_data": first_batch.has_edar_data,
            "has_region_embeddings": first_batch.has_region_embeddings,
        }

        if first_batch.has_edar_data:
            stats["num_edar_sites"] = (
                first_batch.edar_features.shape[0]
                if first_batch.edar_features is not None
                else 0
            )

        # Feature statistics
        all_node_features = torch.stack([batch.node_features for batch in epi_batches])
        all_targets = torch.stack([batch.target_sequences for batch in epi_batches])

        stats.update(
            {
                "node_feature_stats": {
                    "mean": float(all_node_features.mean()),
                    "std": float(all_node_features.std()),
                    "min": float(all_node_features.min()),
                    "max": float(all_node_features.max()),
                },
                "target_stats": {
                    "mean": float(all_targets.mean()),
                    "std": float(all_targets.std()),
                    "min": float(all_targets.min()),
                    "max": float(all_targets.max()),
                },
            }
        )

        # Temporal statistics
        timestamps = [batch.timestamp for batch in epi_batches]
        stats["temporal_range"] = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
            "duration_days": (max(timestamps) - min(timestamps)).days,
        }

        return stats

    def _get_dataset_size_mb(self, dataset_path: Path) -> float:
        """Calculate dataset size in megabytes."""
        total_size = 0
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)

    def _update_stage(self, stage_name: str):
        """Update pipeline stage tracking."""
        if self.pipeline_state["current_stage"]:
            self.pipeline_state["completed_stages"].append(
                self.pipeline_state["current_stage"]
            )
        self.pipeline_state["current_stage"] = stage_name
        print(f"Entering stage: {stage_name.replace('_', ' ').title()}")
