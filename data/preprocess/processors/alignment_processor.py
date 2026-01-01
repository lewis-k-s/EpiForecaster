import logging

import numpy as np
import pandas as pd
import xarray as xr

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
        edar_data: xr.DataArray,
        population_data: xr.DataArray,
    ) -> xr.Dataset:
        """
        Align all datasets to common temporal and spatial grid using xarray.

        Args:
            cases_data: Processed cases dataset
            mobility_data: Processed mobility dataset (OD matrix)
            edar_data: Processed EDAR dataset
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
        assert not edar_data[TEMPORAL_COORD].values[0] == target_dates.values[0], (
            "UNEXPECTED: EDAR start date has already been expanded to target dates"
        )
        assert not edar_data[TEMPORAL_COORD].values[-1] == target_dates.values[-1], (
            "UNEXPECTED: EDAR end date has already been expanded to target dates"
        )
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

        # Generate report
        _report = {
            "common_regions": len(common_regions),
            "timepoints": len(target_dates),
        }
        # TODO: write report?

        aligned_dataset = xr.merge(
            [cases_final, mobility_final, population_final, edar_final], join="exact"
        )
        print("-" * 50)
        print("Aligned Dataset")
        print("-" * 50)
        print(aligned_dataset)
        print("-" * 50)
        return aligned_dataset
