"""
Main offline preprocessing pipeline for EpiForecaster.

This module orchestrates complete preprocessing workflow, from raw data
loading to canonical dataset creation. It coordinates individual processors,
handles validation, and provides comprehensive reporting throughout
process.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from utils import dtypes as dtype_utils
from .config import REGION_COORD, PreprocessingConfig
from .utils import load_csv_with_string_ids
from .processors.alignment_processor import AlignmentProcessor
from .processors.catalonia_cases_processor import CataloniaCasesProcessor
from .processors.deaths_processor import DeathsProcessor
from .processors.edar_processor import EDARProcessor
from .processors.hospitalizations_processor import HospitalizationsProcessor
from .processors.mobility_processor import MobilityProcessor
from .processors.synthetic_processor import SyntheticProcessor
from .processors.temporal_covariates_processor import TemporalCovariatesProcessor


class OfflinePreprocessingPipeline:
    """
    Complete offline preprocessing pipeline with comprehensive validation.

    This pipeline orchestrates the conversion of raw epidemiological data into
    canonical EpiBatch datasets. It handles all preprocessing steps including:

    1. Loading raw data from various sources (NetCDF, Zarr, CSV)
    2. Processing each data type with specialized processors
    3. Multi-dataset temporal and spatial alignment
    4. Saving the aligned dataset to Zarr format

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
            "edar": EDARProcessor(self.config),
            "alignment": AlignmentProcessor(self.config),
        }

        # Initialize cases processor if cases file is configured
        if self.config.cases_file:
            self.processors["cases"] = CataloniaCasesProcessor(self.config)

        # Initialize DeathsProcessor if configured
        if self.config.deaths_file:
            self.processors["deaths"] = DeathsProcessor(self.config)

        # Initialize hospitalizations processor if hospitalizations file is provided
        if self.config.hospitalizations_file:
            self.processors["hospitalizations"] = HospitalizationsProcessor(self.config)

        # Initialize temporal covariates processor if configured
        if self.config.temporal_covariates is not None:
            self.processors["temporal_covariates"] = TemporalCovariatesProcessor(
                self.config
            )

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

            self._log_sample_stats(
                processed_data["cases"]["cases"],
                label="raw cases",
                sample=self._default_sample_indexer(processed_data["cases"]["cases"]),
            )
            self._log_sample_stats(
                processed_data["mobility"]["mobility"],
                label="raw mobility",
                sample=self._default_sample_indexer(
                    processed_data["mobility"]["mobility"]
                ),
                check_coords={"origin": REGION_COORD, "destination": REGION_COORD},
                dataset=processed_data["mobility"]
                if isinstance(processed_data["mobility"], xr.Dataset)
                else None,
            )

            alignment_result = self.processors["alignment"].align_datasets(
                cases_data=processed_data["cases"],
                mobility_data=processed_data["mobility"],
                edar_data=processed_data["edar"],
                population_data=processed_data["population"],
                hospitalizations_data=processed_data.get("hospitalizations"),
                deaths_data=processed_data.get("deaths"),
            )

            self._log_sample_stats(
                alignment_result["mobility"],
                label="aligned mobility",
                sample=self._default_sample_indexer(alignment_result["mobility"]),
                check_coords={"origin": REGION_COORD, "destination": REGION_COORD},
                dataset=alignment_result,
            )

            if "synthetic_sparsity_level" in processed_data:
                sparsity = processed_data["synthetic_sparsity_level"]
                if "run_id" in sparsity.dims and "run_id" in alignment_result.coords:
                    sparsity = sparsity.reindex(
                        run_id=alignment_result["run_id"].values
                    )
                alignment_result["synthetic_sparsity_level"] = sparsity
                print("  ✓ Preserved synthetic_sparsity_level metadata")

            # Only chunk run_id dimension for memory efficiency
            # Other dimensions are kept unchunked to avoid performance warnings
            alignment_result = alignment_result.chunk(
                {"run_id": self.config.run_id_chunk_size}
            )

            # Compute valid_targets mask based on data density
            valid_targets_mask = self._compute_valid_targets_mask(alignment_result)
            alignment_result["valid_targets"] = valid_targets_mask

            # Add wastewater source availability per region
            edar_region_mask = self._compute_edar_region_mask(
                self.config.region_metadata_file,
                alignment_result[REGION_COORD].values,
            )
            alignment_result["edar_has_source"] = edar_region_mask

            # Add temporal covariates if configured
            if "temporal_covariates" in self.processors:
                temporal_covariates_da = self.processors["temporal_covariates"].process(
                    start_date=alignment_result["date"].values[0],
                    end_date=alignment_result["date"].values[-1],
                )
                alignment_result["temporal_covariates"] = temporal_covariates_da
                print("  ✓ Added temporal covariates to dataset")

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

        # Check if using synthetic data
        if self.config.synthetic_path:
            print("Using synthetic data processor...")
            synthetic_processor = SyntheticProcessor(self.config)
            processed = synthetic_processor.process(self.config.synthetic_path)

            # Process EDAR using the same code path as real data!
            print("Processing synthetic EDAR data through shared aggregation path...")
            edar_processor = self.processors["edar"]
            processed["edar"] = edar_processor.process_from_xarray(
                processed["edar_flow"],
                processed["edar_censor"],
                self.config.region_metadata_file,
            )
            # Remove temporary flow/censor keys
            del processed["edar_flow"]
            del processed["edar_censor"]

            # Ensure hospitalizations and deaths are set to None if not present
            # (the pipeline expects these keys to exist)
            if "hospitalizations" not in processed:
                processed["hospitalizations"] = None
            if "deaths" not in processed:
                processed["deaths"] = None

            return processed

        # Standard real data processing
        raw_data = {}

        # Process cases data (required)
        if self.config.cases_file and "cases" in self.processors:
            print("Processing cases data...")
            try:
                cases_data = self.processors["cases"].process(self.config.cases_file)
                raw_data["cases"] = cases_data
            except Exception as e:
                raise RuntimeError(f"Failed to process cases data: {str(e)}") from e
        else:
            raise RuntimeError("No case data source configured")

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

        # Process hospitalizations data (optional)
        if self.config.hospitalizations_file and "hospitalizations" in self.processors:
            print("Processing hospitalizations data...")
            try:
                # HospitalizationsProcessor expects data_dir, extracts directory from file path
                hospitalizations_dir = str(
                    Path(self.config.hospitalizations_file).parent
                )
                hospitalizations_data = self.processors["hospitalizations"].process(
                    hospitalizations_dir
                )
                raw_data["hospitalizations"] = hospitalizations_data
            except Exception as e:
                print(f"  ! Warning: Failed to process hospitalizations data: {str(e)}")
                raw_data["hospitalizations"] = None
        else:
            raw_data["hospitalizations"] = None

        # Process deaths data (optional)
        if self.config.deaths_file and "deaths" in self.processors:
            print("Processing deaths data...")
            try:
                # DeathsProcessor expects data_dir
                deaths_dir = str(Path(self.config.deaths_file).parent)
                deaths_data = self.processors["deaths"].process(deaths_dir)
                raw_data["deaths"] = deaths_data
            except Exception as e:
                print(f"  ! Warning: Failed to process deaths data: {str(e)}")
                raw_data["deaths"] = None
        else:
            raw_data["deaths"] = None

        print()
        return raw_data

    def _load_population_data(self, population_file: str) -> xr.DataArray:
        from .config import REGION_COORD

        # Load population data using canonical CSV loader to preserve leading zeros
        df = load_csv_with_string_ids(
            population_file,
            usecols=["id", "d.population"],
        )
        df = df.rename(columns={"id": REGION_COORD, "d.population": "population"})
        return df.set_index(REGION_COORD)["population"].to_xarray()

    def _compute_valid_targets_mask(self, aligned_dataset: xr.Dataset) -> xr.DataArray:
        """Compute boolean mask for regions that meet minimum density threshold.

        Args:
            aligned_dataset: Aligned dataset with cases data (run_id, date, region_id)

        Returns:
            DataArray of shape (run_id, num_regions) with boolean values
        """
        print("Computing valid_targets mask...")

        cases_da = aligned_dataset.cases
        density_threshold = self.config.min_density_threshold

        # Compute data density per (run_id, region) (fraction of non-NaN values)
        # cases_da has shape (run_id, date, region_id)
        valid_count = cases_da.notnull().sum(dim="date")
        total_per_region = cases_da["date"].size
        density = valid_count / total_per_region

        # Create boolean mask
        valid_mask = density >= density_threshold

        print(
            f"  (run_id, region) pairs with density >= {density_threshold}: {valid_mask.sum()}/{valid_mask.size}"
        )
        # Compute mean for printing (dask arrays need compute() before formatting)
        avg_density = density.mean().compute()
        print(f"  Average density: {avg_density:.3f}")

        valid_targets_da = xr.DataArray(
            valid_mask.astype(np.int32),
            dims=["run_id", REGION_COORD],
            coords={
                "run_id": cases_da["run_id"].values,
                REGION_COORD: cases_da[REGION_COORD].values,
            },
        )

        return valid_targets_da

    def _compute_edar_region_mask(
        self, region_metadata_file: str, region_ids: np.ndarray
    ) -> xr.DataArray:
        """Compute mask for regions with EDAR contributions."""
        print("Computing edar_has_source mask...")
        emap = xr.open_dataarray(region_metadata_file)
        emap = emap.fillna(0).rename({"home": REGION_COORD})
        emap = emap.assign_coords(
            edar_id=emap["edar_id"].astype(str),
            **{REGION_COORD: emap[REGION_COORD].astype(str)},
        )
        emap = emap.reindex({REGION_COORD: region_ids}, fill_value=0)
        has_source = (emap > 0).any("edar_id").astype(np.int32)
        has_source.name = "edar_has_source"

        print(f"  Regions with EDAR source: {int(has_source.sum())}/{has_source.size}")
        return has_source

    def _save_aligned_dataset(self, aligned_dataset: xr.Dataset) -> Path:
        """Save aligned dataset to processed dir."""
        print("Saving aligned dataset to processed dir...")
        aligned_dataset_path = Path(self.config.output_path) / (
            self.config.dataset_name + ".zarr"
        )
        aligned_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Rechunk to uniform chunks for Zarr compatibility
        # Chunk run_id, date, and spatial dims to avoid oversized chunks
        # that cause data corruption when written to Zarr
        rechunked_dict = {}
        for var_name, var in aligned_dataset.data_vars.items():
            chunks = {}
            for dim in var.dims:
                if dim == "run_id":
                    dim_size = var.sizes[dim]
                    chunks[dim] = min(self.config.run_id_chunk_size, dim_size)
                elif dim == "date":
                    # Use configured date chunk size for time series
                    chunks[dim] = min(self.config.date_chunk_size, var.sizes[dim])
                elif dim in ("origin", "destination", "region_id"):
                    # Chunk spatial dims to avoid huge chunks (945x945 creates ~7.6GB chunks)
                    chunks[dim] = min(self.config.mobility_chunk_size, var.sizes[dim])
                else:
                    chunks[dim] = -1
            rechunked_dict[var_name] = var.chunk(chunks)

        rechunked_dataset = xr.Dataset(rechunked_dict, coords=aligned_dataset.coords)

        # Apply static transforms before dtype conversion to prevent float16 overflow
        # All continuous series are log1p-transformed; clinical series also get per_100k
        print("Applying static transforms (log1p, per_100k)...")
        population = rechunked_dataset.get("population")

        for var_name in list(rechunked_dataset.data_vars):
            var = rechunked_dataset[var_name]

            # Clinical series: log1p(per_100k)
            if var_name in ("cases", "hospitalizations", "deaths"):
                if population is not None:
                    # Apply per_100k then log1p using xarray operations (preserves chunks)
                    pop_values = population.where(
                        (population > 0) & np.isfinite(population)
                    )
                    per_100k_factor = 100000.0 / pop_values
                    # Broadcast and multiply
                    values_per_100k = var * per_100k_factor
                    transformed = np.log1p(values_per_100k.clip(min=0))
                    rechunked_dataset[var_name] = transformed
                    print(f"  {var_name}: log1p(per_100k) applied")
                else:
                    # No population - just log1p
                    transformed = np.log1p(var.clip(min=0))
                    rechunked_dataset[var_name] = transformed
                    print(f"  {var_name}: log1p applied (no population for per_100k)")

            # Mobility: log1p only
            elif var_name == "mobility":
                transformed = np.log1p(var.clip(min=0))
                rechunked_dataset[var_name] = transformed
                print(f"  {var_name}: log1p applied")

            # Biomarker values (not _mask, _censor, _age): log1p only
            elif var_name.startswith("edar_biomarker_") and not var_name.endswith(
                ("_mask", "_censor", "_age")
            ):
                transformed = np.log1p(var.clip(min=0))
                rechunked_dataset[var_name] = transformed
                print(f"  {var_name}: log1p applied")

        # Metadata attrs are assigned after dtype conversion because we rebuild the
        # Dataset object there and would otherwise drop attrs.
        dataset_attrs = {
            "log_transformed": True,
            "population_norm": True,
            "smoothing_clinical_method": self.config.smoothing.clinical_method,
            "smoothing_wastewater_method": self.config.smoothing.wastewater_method,
            "smoothing_missing_policy": self.config.smoothing.missing_policy,
            "preprocessing_config_yaml": self._serialize_config_yaml(),
        }

        # Convert float64 to float16 to reduce storage and memory usage
        # Uses centralized dtype constants from utils/dtypes.py
        print("Optimizing dtypes for storage efficiency...")
        converted_dict = {}
        for var_name, var in rechunked_dataset.data_vars.items():
            # Default: keep original dtype
            new_var = var
            old_dtype = var.dtype

            # Suffix-based rules take precedence (check these FIRST)
            # Masks: binary 0/1 -> bool (1 byte)
            if var_name.endswith("_mask"):
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["mask"])
                print(f"  {var_name}: {old_dtype} -> bool")
            # Age channels: 0-14 -> uint8 (1 byte)
            elif var_name.endswith("_age"):
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["age"])
                print(f"  {var_name}: {old_dtype} -> uint8")
            # Censor flags: 0/1/2 -> uint8 (1 byte)
            elif var_name.endswith("_censor"):
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["censor"])
                print(f"  {var_name}: {old_dtype} -> uint8")
            # Named variables with specific dtypes
            elif var_name == "biomarker_data_start":
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["index"])
                print(f"  {var_name}: {old_dtype} -> int16")
            elif var_name in ("edar_has_source", "valid_targets"):
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["mask"])
                print(f"  {var_name}: {old_dtype} -> bool")
            elif var_name == "population":
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["population"])
                print(f"  {var_name}: {old_dtype} -> int32")
            # Temporal covariates: float32 -> float16 for consistency
            elif var_name == "temporal_covariates" and var.dtype == np.float32:
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["continuous"])
                print(f"  {var_name}: float32 -> float16")
            # Generic float64 -> float16 (continuous values) - check LAST
            elif var.dtype == np.float64:
                new_var = var.astype(dtype_utils.NUMPY_STORAGE_DTYPES["continuous"])
                print(f"  {var_name}: float64 -> float16")

            converted_dict[var_name] = new_var
        rechunked_dataset = xr.Dataset(
            converted_dict,
            coords=rechunked_dataset.coords,
            attrs=dataset_attrs,
        )

        # Clear conflicting encodings from data variables and coordinates.
        # Variables from source zarr files retain v3-specific encodings that
        # are incompatible with zarr v2 format used for output.
        v3_encoding_keys = {
            "chunks",  # old chunk sizes conflict with rechunking
            "preferred_chunks",
            "compressors",  # v3 uses tuple of codecs
            "compressor",  # clear both styles
            "filters",  # v3 uses tuple of filters
            "serializer",  # v3-specific
            "object_codec",  # v3-specific
            "shards",  # v3-specific
        }
        for var_name in rechunked_dataset.data_vars:
            var = rechunked_dataset.data_vars[var_name]
            for key in v3_encoding_keys:
                var.encoding.pop(key, None)

        for coord_name in rechunked_dataset.coords:
            coord = rechunked_dataset.coords[coord_name]
            for key in v3_encoding_keys:
                coord.encoding.pop(key, None)

        # Save with uniform chunking, using Zarr v2 for NFS stability
        rechunked_dataset.to_zarr(
            aligned_dataset_path,
            mode="w",
            zarr_format=2,  # Use v2 for better NFS compatibility
            align_chunks=False,  # False since we already manually rechunked
            safe_chunks=True,  # True to prevent data corruption from partial chunks
            consolidated=True,  # True for better metadata performance
        )
        self._log_postwrite_summary(aligned_dataset_path)
        print(f"  ✓ Aligned dataset saved to {aligned_dataset_path}")
        return aligned_dataset_path

    def _serialize_config_yaml(self) -> str:
        """Serialize full preprocessing config as YAML for dataset provenance."""
        config_dict = self.config.__dict__.copy()
        config_dict["start_date"] = self.config.start_date.isoformat()
        config_dict["end_date"] = self.config.end_date.isoformat()
        if self.config.temporal_covariates is not None:
            config_dict["temporal_covariates"] = asdict(self.config.temporal_covariates)
        config_dict["smoothing"] = asdict(self.config.smoothing)
        return yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)

    def _update_stage(self, stage_name: str):
        """Update pipeline stage tracking."""
        if self.pipeline_state["current_stage"]:
            self.pipeline_state["completed_stages"].append(
                self.pipeline_state["current_stage"]
            )
        self.pipeline_state["current_stage"] = stage_name
        print(f"Entering stage: {stage_name.replace('_', ' ').title()}")

    def _default_sample_indexer(self, data: xr.DataArray) -> dict[str, int | slice]:
        indexer: dict[str, int | slice] = {}
        for dim in data.dims:
            dim_name = str(dim)
            size = data.sizes[dim]
            if dim_name in {"origin", "destination", REGION_COORD}:
                indexer[dim_name] = slice(0, min(20, size))
            elif dim_name == "date":
                indexer[dim_name] = 0
            elif dim_name == "run_id":
                indexer[dim_name] = 0
            else:
                indexer[dim_name] = 0
        return indexer

    def _log_sample_stats(
        self,
        data: xr.DataArray,
        *,
        label: str,
        sample: dict[str, int | slice],
        check_coords: dict[str, str] | None = None,
        dataset: xr.Dataset | None = None,
    ) -> None:
        sample_da = data.isel(sample)
        if hasattr(sample_da, "compute"):
            sample_da = sample_da.compute()

        nan_count = int(sample_da.isnull().sum())
        nonzero_count = int((sample_da > 0).sum())
        min_val = float(sample_da.min())
        max_val = float(sample_da.max())
        all_nan = bool(sample_da.isnull().all())

        print(
            f"Sample stats [{label}]: shape={sample_da.shape}, "
            f"nan={nan_count}, nonzero={nonzero_count}, "
            f"min={min_val}, max={max_val}, all_nan={all_nan}"
        )

        if check_coords and dataset is not None:
            for dim, ref in check_coords.items():
                if dim in data.coords and ref in dataset.coords:
                    coords_match = np.array_equal(
                        data.coords[dim].values, dataset.coords[ref].values
                    )
                    print(f"Sample coords [{label}]: {dim} == {ref} -> {coords_match}")

    def _log_postwrite_summary(self, dataset_path: Path) -> None:
        """Verify the saved dataset is valid (e.g., mobility not all NaN)."""
        ds = xr.open_zarr(dataset_path)
        try:
            if "mobility" in ds:
                # Check a sample to ensure data was preserved (not all NaN)
                sample = ds["mobility"].isel(
                    self._default_sample_indexer(ds["mobility"])
                )
                if hasattr(sample, "compute"):
                    sample = sample.compute()
                if bool(sample.isnull().all()):
                    raise ValueError(
                        "Saved mobility sample is all NaN; preprocessing failed."
                    )
                non_null_count = ds["mobility"].notnull().sum().compute()
                print(
                    f"  Verified mobility data: {non_null_count.values} non-null values"
                )
            if "cases" in ds:
                non_null_count = ds["cases"].notnull().sum().compute()
                print(f"  Verified cases data: {non_null_count.values} non-null values")
        finally:
            ds.close()
