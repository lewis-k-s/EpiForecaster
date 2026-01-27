import logging

import numpy as np
import pandas as pd
import xarray as xr

from constants import (
    EDAR_BIOMARKER_PREFIX,
    EDAR_BIOMARKER_VARIANTS,
)
from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig

logger = logging.getLogger(__name__)


class AlignmentProcessor:
    """
    Handles multi-dataset temporal and spatial alignment.

    This processor aligns different data sources to common temporal and spatial
    dimensions using configurable strategies. It supports:

    - Temporal alignment (reindexing/cropping without interpolation)
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

        # Graph options for mobility processing
        self.graph_options = config.graph_options or {}

    def align_datasets(
        self,
        cases_data: xr.DataArray,
        mobility_data: xr.Dataset,
        edar_data: xr.Dataset,
        population_data: xr.DataArray,
    ) -> xr.Dataset:
        """
        Align all datasets to common temporal and spatial grid using xarray.

        Args:
            cases_data: Processed cases dataset
            mobility_data: Processed mobility dataset (OD matrix)
            edar_data: Processed EDAR dataset (per-variant variables)
            population_data: Processed population dataset

        Returns:
            xr.Dataset of all aligned datasets
        """
        print(f"Aligning datasets using strategy: {self.alignment_strategy}")

        # STEP 1: Temporal alignment
        target_dates = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        print(f"Target date range: {len(target_dates)} days")

        # check if dates are already the same
        assert (cases_data[TEMPORAL_COORD].values == target_dates.values).all(), (
            "Cases dates are not the same"
        )
        print("Cases dates are already the same")
        cases_aligned = cases_data

        assert (mobility_data[TEMPORAL_COORD].values == target_dates.values).all(), (
            "Mobility dates are not the same"
        )
        print("Mobility dates are already the same")
        mobility_aligned = mobility_data

        # edar data does not cover the same range so we only expect that it is subset of target dates
        assert np.isin(edar_data[TEMPORAL_COORD].values, target_dates.values).all(), (
            "EDAR subset dates are not the same"
        )

        # Check if EDAR already matches target range (e.g., synthetic data)
        edar_matches_target = (
            edar_data[TEMPORAL_COORD].values[0] == target_dates.values[0]
            and edar_data[TEMPORAL_COORD].values[-1] == target_dates.values[-1]
        )

        if edar_matches_target:
            print("EDAR dates already match target range (no expansion needed)")
            edar_aligned = edar_data
        else:
            print("Expanding EDAR dates to target dates (preserving NaNs)")
            edar_aligned = edar_data.reindex({TEMPORAL_COORD: target_dates})

        # STEP 2: Spatial alignment - identify common regions
        # All datasets should use REGION_COORD for spatial dimension
        cases_regions = set(cases_aligned[REGION_COORD].values)

        # Mobility regions (from origin/destination coordinates)
        # We assume origin and destination cover the same set of regions for the study area
        mobility_origins = set(mobility_aligned["origin"].values)
        mobility_destinations = set(mobility_aligned["destination"].values)
        mobility_regions = mobility_origins.union(mobility_destinations)

        # Find intersection of all region sets
        common_regions = sorted(cases_regions.intersection(mobility_regions))

        assert np.isin(edar_aligned[REGION_COORD].values, common_regions).all(), (
            "EDAR regions are not subset of common regions"
        )

        if not common_regions:
            raise ValueError("No common regions found between datasets")

        print(
            f"Spatial alignment: {len(cases_regions)} cases regions, {len(mobility_regions)} mobility regions -> {len(common_regions)} common"
        )

        cases_final = cases_aligned.sel({REGION_COORD: common_regions})
        mobility_final = mobility_aligned.sel(
            origin=common_regions, destination=common_regions
        )
        population_final = population_data.sel({REGION_COORD: common_regions})
        edar_final = edar_aligned.reindex({REGION_COORD: common_regions})

        # Fill mask/censor/age channels with proper defaults for regions without EDAR data
        # Mask: 0.0 (no measurement), Censor: 0.0 (not censored), Age: 1.0 (max age)
        for var_name in edar_final.data_vars:
            if var_name.endswith("_mask"):
                edar_final[var_name] = edar_final[var_name].fillna(0.0)
            elif var_name.endswith("_censor"):
                edar_final[var_name] = edar_final[var_name].fillna(0.0)
            elif var_name.endswith("_age"):
                edar_final[var_name] = edar_final[var_name].fillna(1.0)

        # Compute biomarker data start offset for each (run_id, region) pair
        # For each pair, find the first time index where biomarker data > 0
        # Use -1 for regions with no biomarker data
        print("Computing biomarker data start offset per (run_id, region)...")
        # Get only true biomarker variables (exclude mask/censor/age channels)
        biomarker_vars = [
            f"{EDAR_BIOMARKER_PREFIX}{v}"
            for v in EDAR_BIOMARKER_VARIANTS
            if f"{EDAR_BIOMARKER_PREFIX}{v}" in edar_final.data_vars
        ]

        # run_id always exists on all data variables
        first_biomarker = edar_final[biomarker_vars[0]]
        run_ids = first_biomarker["run_id"].values

        # Create 2D array for (run_id, region) pairs
        biomarker_data_start = xr.DataArray(
            np.full((len(run_ids), len(common_regions)), -1, dtype=np.int32),
            dims=["run_id", REGION_COORD],
            coords={"run_id": run_ids, REGION_COORD: common_regions},
            name="biomarker_data_start",
        )

        # Vectorized approach using xarray operations
        # Stack all biomarkers: (n_variants, n_runs, n_dates, n_regions)
        all_biomarkers = edar_final[biomarker_vars].to_array(dim="variant")

        # Find where any biomarker has valid data: (n_runs, n_dates, n_regions)
        has_data = (all_biomarkers > 0) & all_biomarkers.notnull()
        has_data_any = has_data.any(dim="variant")

        # Use argmax along date dimension to find first True
        # argmax on boolean returns first True index (0 if all False, but we handle that)
        first_idx = has_data_any.argmax(
            dim="date"
        ).compute()  # Compute for use as indexer

        # Handle all-False case: check if there's actually data at the argmax position
        has_data_at_first = has_data_any.isel(date=first_idx)
        first_idx_corrected = xr.where(has_data_at_first, first_idx, -1)

        # Convert to numpy and assign to biomarker_data_start
        # Note: first_idx_corrected already has dims (run_id, region_id)
        biomarker_data_start.values = first_idx_corrected.astype(np.int32).values

        print(
            f"  (run_id, region) pairs with biomarker data: {(biomarker_data_start.values >= 0).sum()}/{biomarker_data_start.size}"
        )
        if (biomarker_data_start.values >= 0).sum() > 0:
            valid_starts = biomarker_data_start.values[biomarker_data_start.values >= 0]
            print(f"  First data start index: {valid_starts.min()}")
            print(f"  Last data start index: {valid_starts.max()}")

        # Generate report
        _report = {
            "common_regions": len(common_regions),
            "timepoints": len(target_dates),
        }
        # TODO: write report?

        aligned_dataset = xr.merge(
            [
                cases_final,
                mobility_final,
                population_final,
                edar_final,
                biomarker_data_start,
            ],
            join="exact",
        )
        print("-" * 50)
        print("Aligned Dataset")
        print("-" * 50)
        print(aligned_dataset)
        print("-" * 50)
        return aligned_dataset
