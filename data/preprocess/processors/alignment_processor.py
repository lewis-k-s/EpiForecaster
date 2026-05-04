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

    @staticmethod
    def _compute_age_from_mask(mask: xr.DataArray, max_age: int = 14) -> xr.DataArray:
        """Compute integer age channel from binary observation mask.

        Returns uint8 array with values 0-14 (days since last observation).
        """
        if TEMPORAL_COORD not in mask.dims:
            raise ValueError(f"Mask must include '{TEMPORAL_COORD}' dimension")

        mask_binary = xr.where(mask > 0, 1, 0)
        time_indices = xr.DataArray(
            np.arange(mask_binary.sizes[TEMPORAL_COORD], dtype=np.int32),
            dims=[TEMPORAL_COORD],
            coords={TEMPORAL_COORD: mask_binary[TEMPORAL_COORD]},
        )

        last_seen = xr.where(mask_binary > 0, time_indices, np.nan)
        last_seen_filled = last_seen.ffill(dim=TEMPORAL_COORD)
        valid_history = ~np.isnan(last_seen_filled)

        current_time, _ = xr.broadcast(time_indices, mask_binary)
        current_age = current_time - last_seen_filled + 1
        final_age = xr.where(valid_history, np.minimum(current_age, max_age), max_age)
        return final_age.transpose(*mask.dims).astype(np.uint8)

    def _align_temporal_dataset(
        self,
        data: xr.Dataset | xr.DataArray,
        target_dates: pd.DatetimeIndex,
        *,
        label: str,
        allow_subset_in_strict: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        source_dates = data[TEMPORAL_COORD].values
        target_values = target_dates.values

        if self.config.temporal_alignment_mode == "strict":
            if len(source_dates) != len(target_values) or not np.array_equal(
                source_dates, target_values
            ):
                if allow_subset_in_strict and np.isin(
                    source_dates, target_values
                ).all():
                    print(f"Expanding {label} dates to target dates (preserving NaNs)")
                    return data.reindex({TEMPORAL_COORD: target_dates})
                raise AssertionError(f"{label} dates are not the same")
            print(f"{label} dates are already the same")
            return data

        if not np.isin(source_dates, target_values).all():
            raise ValueError(
                f"{label} dates are not a subset of configured target dates"
            )
        if len(source_dates) == len(target_values) and np.array_equal(
            source_dates, target_values
        ):
            print(f"{label} dates already match target range")
            return data

        print(f"Reindexing {label} dates to target dates (preserving missingness)")
        return data.reindex({TEMPORAL_COORD: target_dates})

    def _align_clinical_dataset(
        self,
        data: xr.Dataset | xr.DataArray,
        target_dates: pd.DatetimeIndex,
        *,
        value_name: str,
        label: str,
    ) -> xr.Dataset | xr.DataArray:
        aligned = self._align_temporal_dataset(data, target_dates, label=label)
        if not isinstance(aligned, xr.Dataset):
            return aligned

        mask_name = f"{value_name}_mask"
        age_name = f"{value_name}_age"
        if mask_name in aligned.data_vars:
            mask = aligned[mask_name].fillna(False)
        else:
            mask = aligned[value_name].notnull()
        aligned[mask_name] = mask
        aligned[age_name] = self._compute_age_from_mask(mask)
        return aligned

    def _build_mobility_time_mask(
        self,
        mobility_data: xr.Dataset,
        target_dates: pd.DatetimeIndex,
        run_ids: np.ndarray,
    ) -> xr.DataArray:
        source_dates = mobility_data[TEMPORAL_COORD].values
        if "run_id" in mobility_data.dims:
            source_run_ids = mobility_data["run_id"].values
        else:
            source_run_ids = run_ids

        source_mask = xr.DataArray(
            np.ones((len(source_run_ids), len(source_dates)), dtype=bool),
            dims=["run_id", TEMPORAL_COORD],
            coords={"run_id": source_run_ids, TEMPORAL_COORD: source_dates},
            name="mobility_time_mask",
        )
        return source_mask.reindex(
            {"run_id": run_ids, TEMPORAL_COORD: target_dates},
            fill_value=False,
        )

    def align_datasets(
        self,
        cases_data: xr.DataArray,
        mobility_data: xr.Dataset,
        edar_data: xr.Dataset,
        population_data: xr.DataArray,
        hospitalizations_data: xr.Dataset | None = None,
        deaths_data: xr.Dataset | None = None,
        latent_sird_data: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """
        Align all datasets to common temporal and spatial grid using xarray.

        Args:
            cases_data: Processed cases dataset
            mobility_data: Processed mobility dataset (OD matrix)
            edar_data: Processed EDAR dataset (per-variant variables)
            population_data: Processed population dataset
            hospitalizations_data: Optional processed hospitalizations dataset
            deaths_data: Optional processed deaths dataset
            latent_sird_data: Optional latent SIRD targets from synthetic data

        Returns:
            xr.Dataset of all aligned datasets
        """
        print(f"Aligning datasets using strategy: {self.alignment_strategy}")

        # STEP 1: Temporal alignment
        target_dates = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        print(f"Target date range: {len(target_dates)} days")

        cases_aligned = self._align_clinical_dataset(
            cases_data,
            target_dates,
            value_name="cases",
            label="Cases",
        )
        mobility_aligned = self._align_temporal_dataset(
            mobility_data,
            target_dates,
            label="Mobility",
        )
        edar_aligned = self._align_temporal_dataset(
            edar_data,
            target_dates,
            label="EDAR",
            allow_subset_in_strict=True,
        )

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

        # Validate hospitalizations regions if provided
        if hospitalizations_data is not None:
            hosp_regions = set(hospitalizations_data[REGION_COORD].values)
            # Check that there's at least some overlap between hospitalizations and common regions
            # Hospitalizations data may have additional regions that will be filtered via reindex
            common_regions_set = set(common_regions)
            overlap = hosp_regions.intersection(common_regions_set)
            if not overlap:
                raise ValueError("No hospitalizations regions found in common regions")
            dropped = hosp_regions - common_regions_set
            print(
                f"  Hospitalizations data: {len(hosp_regions)} regions, {len(overlap)} overlap with common"
            )
            if dropped:
                print(
                    f"  Dropping {len(dropped)} hospitalizations regions not in common regions (not in cases): {sorted(list(dropped))[:10]}..."
                )

        # Validate deaths regions if provided
        if deaths_data is not None:
            deaths_regions = set(deaths_data[REGION_COORD].values)
            # Check that there's at least some overlap between deaths and common regions
            # Deaths data may have additional regions that will be filtered via reindex
            overlap = deaths_regions.intersection(common_regions)
            if not overlap:
                raise ValueError("No deaths regions found in common regions")
            print(
                f"  Deaths data: {len(deaths_regions)} regions, {len(overlap)} overlap with common"
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

        # Align hospitalizations if provided
        if hospitalizations_data is not None:
            hosp_temporal = self._align_clinical_dataset(
                hospitalizations_data,
                target_dates,
                value_name="hospitalizations",
                label="Hospitalizations",
            )
            hosp_final = hosp_temporal.reindex({REGION_COORD: common_regions})
            # Preserve missingness signal before zero-filling values.
            if "hospitalizations_mask" in hosp_final.data_vars:
                hosp_mask = hosp_final["hospitalizations_mask"].fillna(False)
            else:
                hosp_mask = hosp_final["hospitalizations"].notnull()
            hosp_final["hospitalizations_mask"] = hosp_mask
            hosp_final["hospitalizations_age"] = self._compute_age_from_mask(
                hosp_final["hospitalizations_mask"]
            )
        else:
            hosp_final = None

        # Align deaths if provided
        if deaths_data is not None:
            deaths_temporal = self._align_clinical_dataset(
                deaths_data,
                target_dates,
                value_name="deaths",
                label="Deaths",
            )
            deaths_final = deaths_temporal.reindex({REGION_COORD: common_regions})
            # Preserve missingness signal before zero-filling values.
            if "deaths_mask" in deaths_final.data_vars:
                deaths_mask = deaths_final["deaths_mask"].fillna(False)
            else:
                deaths_mask = deaths_final["deaths"].notnull()
            deaths_final["deaths_mask"] = deaths_mask
            deaths_final["deaths_age"] = self._compute_age_from_mask(
                deaths_final["deaths_mask"]
            )
        else:
            deaths_final = None

        # Align latent SIRD targets if provided (synthetic-only path)
        if latent_sird_data is not None:
            latent_sird_final = latent_sird_data.reindex({REGION_COORD: common_regions})
            for var_name in list(latent_sird_final.data_vars):
                if var_name.endswith("_mask"):
                    latent_sird_final[var_name] = latent_sird_final[var_name].fillna(
                        False
                    )
                else:
                    latent_sird_final[var_name] = latent_sird_final[var_name].where(
                        np.isfinite(latent_sird_final[var_name])
                    )
        else:
            latent_sird_final = None

        # Fill mask/censor/age channels with proper defaults for regions without EDAR data
        # Mask: False (no measurement), Censor: 0 (not censored), Age: 14 (max age)
        for var_name in edar_final.data_vars:
            if var_name.endswith("_mask"):
                edar_final[var_name] = edar_final[var_name].fillna(False)
            elif var_name.endswith("_censor"):
                edar_final[var_name] = edar_final[var_name].fillna(0)
            elif var_name.endswith("_age"):
                edar_final[var_name] = edar_final[var_name].fillna(14)

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
        mobility_time_mask = self._build_mobility_time_mask(
            mobility_data,
            target_dates,
            run_ids,
        )

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

        # Build merge list with optional hospitalizations and deaths
        datasets_to_merge = [
            cases_final,
            mobility_final,
            mobility_time_mask,
            population_final,
            edar_final,
            biomarker_data_start,
        ]
        if hosp_final is not None:
            datasets_to_merge.append(hosp_final)
        if deaths_final is not None:
            datasets_to_merge.append(deaths_final)
        if latent_sird_final is not None:
            datasets_to_merge.append(latent_sird_final)

        aligned_dataset = xr.merge(datasets_to_merge, join="exact")
        print("-" * 50)
        print("Aligned Dataset")
        print("-" * 50)
        print(aligned_dataset)
        print("-" * 50)
        return aligned_dataset
