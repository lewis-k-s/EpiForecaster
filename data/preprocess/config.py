"""
Configuration for the offline preprocessing pipeline.

This module defines configuration classes for the EpiForecaster preprocessing
pipeline. It supports comprehensive validation, multiple alignment strategies,
and flexible data source specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REGION_COORD = "region_id"
TEMPORAL_COORD = "date"


@dataclass
class PreprocessingConfig:
    """
    Configuration for offline preprocessing with comprehensive validation enabled.

    This configuration defines all parameters needed to convert raw epidemiological
    data into canonical EpiBatch datasets. It includes data source paths,
    processing options, validation settings, and output configuration.

    Attributes:
        data_dir: Base directory for raw data files
        synthetic_path: Optional path to synthetic data zarr bundle. If provided,
            individual data paths (cases_file, mobility_path, etc.) are ignored
            and SyntheticProcessor is used instead.
        mobility_path: Path to NetCDF mobility data
        cases_file: Path to COVID case data CSV
        wastewater_file: Path to wastewater biomarker data
        population_file: Path to population data for normalization
        region_metadata_file: Path to regional metadata

        start_date: Start date for temporal processing
        end_date: End date for temporal processing
        forecast_horizon: Number of days to forecast into the future
        sequence_length: Length of input sequences for models

        min_flow_threshold: Minimum flow threshold for mobility pairs (i,j)
        wastewater_flow_mode: "total_flow" or "concentration" for EDAR

        alignment_strategy: Strategy for aligning multiple datasets
        target_dataset: Which dataset to align others to
        crop_datasets: Whether to crop datasets to common temporal range
        validate_alignment: Whether to run comprehensive alignment validation
        generate_alignment_report: Whether to generate detailed alignment reports

        output_path: Directory path for processed dataset
        dataset_name: Human-readable name for the dataset
        compression: Compression algorithm for Zarr storage
        chunk_sizes: Custom chunk sizes for Zarr arrays

        num_workers: Number of parallel workers for processing
        chunk_size: Chunk size for processing large files
        memory_limit_gb: Memory limit for processing (GB)

        graph_options: Graph construction options
        validation_options: Validation and quality control options
    """

    # Required fields (no defaults)
    data_dir: str
    cases_file: str
    mobility_path: str
    wastewater_file: str
    population_file: str
    region_metadata_file: str
    start_date: datetime
    end_date: datetime
    output_path: str
    dataset_name: str

    # Optional synthetic data path
    synthetic_path: str | None = None

    # Optional environment
    env: str | None = None

    # Feature processing options
    min_flow_threshold: int = 10
    wastewater_flow_mode: str = "total_flow"  # "total_flow" or "concentration"

    # Temporal processing parameters
    forecast_horizon: int = 7
    sequence_length: int = 1

    # Region filtering
    min_density_threshold: float = (
        0.5  # Minimum data density (non-NaN fraction) for a region to be valid
    )

    # Output configuration
    compression: str = "blosc"
    chunk_sizes: dict[str, list[int]] | None = None

    # Alignment and validation options
    alignment_strategy: str = "interpolate"  # "interpolate", "nearest", "spline"
    target_dataset: str = "cases"  # "cases", "mobility", "wastewater"
    crop_datasets: bool = True
    validate_alignment: bool = True
    generate_alignment_report: bool = True

    # Processing configuration
    num_workers: int = 4
    chunk_size: int = 1000
    run_id_chunk_size: int = 5  # Chunk size for run_id dimension
    date_chunk_size: int = 30  # Chunk size for date dimension in zarr output
    mobility_chunk_size: int = 100  # Chunk size for spatial dims (origin/destination/region_id)
    memory_limit_gb: float = 8.0

    # Graph construction options
    graph_options: dict[str, Any] = field(
        default_factory=lambda: {
            "edge_strategy": "knn",  # "knn", "distance_threshold", "delaunay"
            "k_neighbors": 8,  # Number of neighbors for k-NN graph
            "distance_threshold": 50.0,  # Distance threshold in km
            "include_self_loops": False,
        }
    )

    # Validation and quality control options
    validation_options: dict[str, Any] = field(
        default_factory=lambda: {
            "max_missing_percentage": 0.1,  # Maximum percentage of missing data
            "min_data_coverage": 0.8,  # Minimum temporal coverage required
            "outlier_detection": True,
            "outlier_threshold": 3.0,  # Standard deviations for outlier detection
            "temporal_consistency_check": True,
            "spatial_consistency_check": True,
            "generate_plots": True,
        }
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_temporal_parameters()
        self._validate_processing_options()
        self._validate_graph_options()

    def _validate_paths(self):
        """Validate that input paths exist and are correctly formatted."""
        data_dir_path = Path(self.data_dir)
        if not data_dir_path.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if self.mobility_path:
            mobility_path = Path(self.mobility_path)
            if not mobility_path.exists():
                raise ValueError(f"Mobility file does not exist: {self.mobility_path}")

        cases_path = Path(self.cases_file)
        if not cases_path.exists():
            raise ValueError(f"Cases file does not exist: {self.cases_file}")

        if self.wastewater_file:
            wastewater_path = Path(self.wastewater_file)
            if not wastewater_path.exists():
                raise ValueError(
                    f"Wastewater file does not exist: {self.wastewater_file}"
                )

        # Validate output path is writable
        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    def _validate_temporal_parameters(self):
        """Validate temporal parameters are sensible."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        temporal_range = self.end_date - self.start_date
        if temporal_range.days < self.forecast_horizon:
            raise ValueError(
                f"Temporal range ({temporal_range.days} days) must be greater than "
                f"forecast_horizon ({self.forecast_horizon} days)"
            )

        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

    def _validate_processing_options(self):
        """Validate processing options."""
        valid_alignments = ["interpolate", "nearest", "spline"]
        if self.alignment_strategy not in valid_alignments:
            raise ValueError(
                f"Invalid alignment_strategy: {self.alignment_strategy}. "
                f"Valid options: {valid_alignments}"
            )

        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.run_id_chunk_size <= 0:
            raise ValueError("run_id_chunk_size must be positive")

        if self.date_chunk_size <= 0:
            raise ValueError("date_chunk_size must be positive")

        if self.mobility_chunk_size <= 0:
            raise ValueError("mobility_chunk_size must be positive")

        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")

    def _validate_graph_options(self):
        """Validate graph construction options."""
        valid_strategies = ["knn", "distance_threshold", "delaunay"]
        if self.graph_options["edge_strategy"] not in valid_strategies:
            raise ValueError(
                f"Invalid edge_strategy: {self.graph_options['edge_strategy']}. "
                f"Valid options: {valid_strategies}"
            )

        if self.graph_options["k_neighbors"] <= 0:
            raise ValueError("k_neighbors must be positive")

        if self.graph_options["distance_threshold"] <= 0:
            raise ValueError("distance_threshold must be positive")

    @classmethod
    def from_file(cls, config_path: str | Path) -> "PreprocessingConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            PreprocessingConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Convert string dates to datetime objects
        if "start_date" in config_dict:
            config_dict["start_date"] = datetime.fromisoformat(
                config_dict["start_date"]
            )
        if "end_date" in config_dict:
            config_dict["end_date"] = datetime.fromisoformat(config_dict["end_date"])

        return cls(**config_dict)

    def to_file(self, config_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path where to save the YAML configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime objects to strings
        config_dict = self.__dict__.copy()
        config_dict["start_date"] = self.start_date.isoformat()
        config_dict["end_date"] = self.end_date.isoformat()

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

    def get_dataset_filename(self) -> str:
        """Generate standardized filename for the processed dataset."""
        return f"{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.zarr"

    def get_output_dataset_path(self) -> Path:
        """Get full path for the output dataset."""
        return Path(self.output_path) / self.get_dataset_filename()

    def summary(self) -> dict[str, Any]:
        """Get summary of configuration parameters."""
        temporal_range = self.end_date - self.start_date

        return {
            "dataset_name": self.dataset_name,
            "temporal_range": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "duration_days": temporal_range.days,
            },
            "data_sources": {
                "mobility": self.mobility_path,
                "cases": self.cases_file,
                "wastewater": self.wastewater_file,
                "population": self.population_file,
            },
            "processing_parameters": {
                "forecast_horizon": self.forecast_horizon,
                "sequence_length": self.sequence_length,
            },
            "alignment": {
                "strategy": self.alignment_strategy,
                "target_dataset": self.target_dataset,
                "crop_datasets": self.crop_datasets,
                "validate_alignment": self.validate_alignment,
            },
            "output": {
                "path": self.output_path,
                "dataset_filename": self.get_dataset_filename(),
                "compression": self.compression,
            },
            "graph_options": self.graph_options,
            "validation": self.validation_options,
        }
