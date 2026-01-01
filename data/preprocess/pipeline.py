"""
Main offline preprocessing pipeline for EpiForecaster.

This module orchestrates complete preprocessing workflow, from raw data
loading to canonical dataset creation. It coordinates individual processors,
handles validation, and provides comprehensive reporting throughout
process.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .processors.alignment_processor import AlignmentProcessor
from .processors.cases_processor import CasesProcessor
from .processors.edar_processor import EDARProcessor
from .processors.mobility_processor import MobilityProcessor


class OfflinePreprocessingPipeline:
    """
    Complete offline preprocessing pipeline with comprehensive validation.

    This pipeline orchestrates the conversion of raw epidemiological data into
    canonical EpiBatch datasets. It handles all preprocessing steps including:

    1. Loading raw data from various sources (NetCDF, Zarr, CSV)
    2. Processing each data type with specialized processors
    3. Multi-dataset temporal and spatial alignment
    7. Saving the aligned dataset to Zarr format

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
        self.processors = {
            "mobility": MobilityProcessor(self.config),
            "cases": CasesProcessor(self.config),
            "edar": EDARProcessor(self.config),
            "alignment": AlignmentProcessor(self.config),
        }

        # Initialize state tracking
        self.pipeline_state = {
            "start_time": datetime.now(),
            "current_stage": "initialization",
            "completed_stages": [],
            "errors": [],
            "warnings": [],
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
        print()

        try:
            # Stage 1: Load and process raw data sources
            processed_data = self._load_raw_sources()

            alignment_result = self.processors["alignment"].align_datasets(
                cases_data=processed_data["cases"],
                mobility_data=processed_data["mobility"],
                edar_data=processed_data["edar"],
                population_data=processed_data["population"],
            )

            alignment_result = alignment_result.chunk(
                {
                    TEMPORAL_COORD: self.config.chunk_size,
                    REGION_COORD: -1,
                    "origin": -1,
                    "destination": -1,
                }
            )

            # Compute valid_targets mask based on data density
            valid_targets_mask = self._compute_valid_targets_mask(alignment_result)
            alignment_result["valid_targets"] = valid_targets_mask

            # # Store alignment report in pipeline state
            # self.pipeline_state["alignment_report"] = alignment_report

            dataset_path = self._save_aligned_dataset(alignment_result)

            total_time = datetime.now() - self.pipeline_state["start_time"]
            print("=" * 60)
            print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time}")
            print(f"Dataset saved to: {dataset_path}")
            print("=" * 60)

            return dataset_path

        except Exception as e:
            self.pipeline_state["errors"].append(str(e))
            print(f"PIPELINE FAILED: {str(e)}")
            raise

    def _load_raw_sources(self) -> dict[str, xr.DataArray | xr.Dataset]:
        """Load and process raw data sources. All data sources are required."""
        self._update_stage("loading_raw_data")
        print("Stage 1: Loading and processing raw data sources")
        print("-" * 50)

        raw_data = {}

        # Process cases data (required)
        print("Processing cases data...")
        try:
            cases_data = self.processors["cases"].process(self.config.cases_file)
            raw_data["cases"] = cases_data
        except Exception as e:
            raise RuntimeError(f"Failed to process cases data: {str(e)}") from e

        # Process mobility data (required)
        if not self.config.mobility_path:
            raise RuntimeError("Mobility data path is required but not configured")

        print("Processing mobility data...")
        try:
            mobility_data = self.processors["mobility"].process(
                self.config.mobility_path
            )
            raw_data["mobility"] = mobility_data
        except Exception as e:
            raise RuntimeError(f"Failed to process mobility data: {str(e)}") from e

        # Process wastewater/biomarker data (required)
        if not self.config.wastewater_file:
            raise RuntimeError("Wastewater data path is required but not configured")

        print("Processing wastewater data...")
        try:
            edar_data = self.processors["edar"].process(
                self.config.wastewater_file, self.config.region_metadata_file
            )
            raw_data["edar"] = edar_data
        except Exception as e:
            raise RuntimeError(f"Failed to process EDAR data: {str(e)}") from e

        # Process population data (required)
        if not self.config.population_file:
            raise RuntimeError("Population data path is required but not configured")

        print("Processing population data...")
        try:
            population_data = self._load_population_data(self.config.population_file)
            raw_data["population"] = population_data
        except Exception as e:
            raise RuntimeError(f"Failed to process population data: {str(e)}") from e

        print()
        return raw_data

    def _load_population_data(self, population_file: str) -> xr.DataArray:
        from .config import REGION_COORD

        # Load population data with proper dtypes to preserve leading zeros
        df = pd.read_csv(
            population_file,
            usecols=["id", "d.population"],  # type: ignore[arg-type]
            dtype={"id": str, "d.population": int},
        )
        df = df.rename(columns={"id": REGION_COORD, "d.population": "population"})
        return df.set_index(REGION_COORD)["population"].to_xarray()

    def _compute_valid_targets_mask(self, aligned_dataset: xr.Dataset) -> xr.DataArray:
        """Compute boolean mask for regions that meet minimum density threshold.

        Args:
            aligned_dataset: Aligned dataset with cases data

        Returns:
            DataArray of shape (num_regions,) with boolean values
        """
        print("Computing valid_targets mask...")

        cases_da = aligned_dataset.cases
        density_threshold = self.config.min_density_threshold

        # Compute data density per region (fraction of non-NaN values)
        missing_mask = cases_da.isnull().values
        density = 1 - (missing_mask.sum(axis=0) / missing_mask.shape[0])

        # Create boolean mask
        valid_mask = density >= density_threshold

        print(
            f"  Regions with density >= {density_threshold}: {valid_mask.sum()}/{valid_mask.size}"
        )
        print(f"  Average density: {density.mean():.3f}")

        valid_targets_da = xr.DataArray(
            valid_mask.astype(np.int32),
            dims=[REGION_COORD],
            coords={REGION_COORD: cases_da[REGION_COORD].values},
        )

        return valid_targets_da

    def _save_aligned_dataset(self, aligned_dataset: xr.Dataset) -> Path:
        """Save aligned dataset to processed dir."""
        print("Saving aligned dataset to processed dir...")
        aligned_dataset_path = Path(self.config.output_path) / (
            self.config.dataset_name + ".zarr"
        )
        aligned_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        # allow overwrite mode w
        aligned_dataset.to_zarr(aligned_dataset_path, mode="w")
        print(f"  ✓ Aligned dataset saved to {aligned_dataset_path}")
        return aligned_dataset_path

    def _update_stage(self, stage_name: str):
        """Update pipeline stage tracking."""
        if self.pipeline_state["current_stage"]:
            self.pipeline_state["completed_stages"].append(
                self.pipeline_state["current_stage"]
            )
        self.pipeline_state["current_stage"] = stage_name
        print(f"Entering stage: {stage_name.replace('_', ' ').title()}")
