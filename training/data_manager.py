"""
Data Manager for Graph Neural Network Epidemiological Forecasting.

This module centralizes all data loading, processing, and preparation logic,
extracting it from main.py to improve code organization and reusability.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import xarray as xr

from data.cases_loader import create_cases_loader
from data.dataset_alignment import create_alignment_manager
from data.edar_attention_loader import create_edar_attention_loader
from data.edar_biomarker_loader import create_edar_biomarker_loader
from data.feature_extractor import GeometricFeatureExtractor, example_custom_features
from data.mobility_loader import MobilityDataLoader, example_preprocessing_hooks
from training.windowing import get_window_stats, split_temporal_data

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages all data loading and processing for the dual graph forecasting system.

    This class encapsulates the complex data loading logic from main.py, providing
    a clean interface for setting up mobility data, EDAR data, and creating
    temporal graph sequences.
    """

    def __init__(self):
        self.mobility_loader: Optional[MobilityDataLoader] = None
        self.feature_extractor: Optional[GeometricFeatureExtractor] = None
        self.cases_loader = None
        self.edar_attention_loader = None
        self.edar_biomarker_loader = None
        self.alignment_manager = None

    def setup_loaders(self, args) -> None:
        """
        Initialize all data loaders (mobility, cases, EDAR attention, EDAR biomarker).

        Args:
            args: CLI arguments containing data configuration
        """
        logger.debug("Setting up dual graph data loading pipeline")

        # Create mobility data loader
        self.mobility_loader = MobilityDataLoader(
            min_flow_threshold=args.min_flow_threshold,
            normalize_flows=True,
            undirected=False,
            allow_self_loops=False,
            edge_selector="nonzero",
            node_stats=("sum", "mean", "count_nonzero"),
            engine="h5netcdf",
            chunks={"time": 1},
        )

        # Register preprocessing hooks if enabled
        if args.enable_preprocessing_hooks:
            logger.debug("Registering preprocessing hooks")
            hooks = example_preprocessing_hooks()

            # Register NetCDF preprocessing hooks
            self.mobility_loader.register_preprocessing_hook(
                "netcdf_preprocessing", hooks["cap_outlier_flows"]
            )
            self.mobility_loader.register_preprocessing_hook(
                "population_preprocessing", hooks["remove_low_population_zones"]
            )
            self.mobility_loader.register_preprocessing_hook(
                "post_merge_preprocessing", hooks["normalize_population_features"]
            )

        # Setup COVID cases data loader
        try:
            # Construct full path to cases file
            data_dir = Path(args.data_dir)
            cases_file_path = (
                data_dir / args.cases_file
                if not Path(args.cases_file).is_absolute()
                else Path(args.cases_file)
            )

            self.cases_loader = create_cases_loader(
                cases_file=str(cases_file_path),
                normalization=args.cases_normalization,
                min_cases_threshold=args.min_cases_threshold,
                fill_missing="forward_fill",
            )
            logger.debug("COVID cases loader initialized successfully")
        except Exception as e:
            logger.error(f"Could not load COVID cases data: {e}")
            logger.error("COVID cases data is required for epidemiological forecasting")
            raise

        # Setup geometric feature extractor
        self.feature_extractor = GeometricFeatureExtractor(
            normalize_features=True, k_nearest=10
        )

        # Register custom feature hooks if enabled
        if args.enable_preprocessing_hooks:
            custom_features = example_custom_features()
            self.feature_extractor.register_feature_hook(
                custom_features["urban_rural_indicator"]
            )
            self.feature_extractor.register_feature_hook(
                custom_features["border_distance_feature"]
            )

        # Setup EDAR data loaders if using EDAR data
        if args.use_edar_data:
            try:
                # Load EDAR attention mask
                self.edar_attention_loader = create_edar_attention_loader(
                    data_dir=args.data_dir, normalize=True, threshold=0.01
                )
                logger.debug("EDAR attention loader initialized successfully")

                # Load EDAR biomarker time series
                self.edar_biomarker_loader = create_edar_biomarker_loader(
                    data_dir=args.data_dir,
                    biomarker_features=getattr(args, "edar_biomarker_features", None),
                    date_range=(
                        getattr(args, "start_date", None),
                        getattr(args, "end_date", None),
                    ),
                    normalize=True,
                    biomarker_path=getattr(args, "wastewater_path", None),
                )
                logger.debug("EDAR biomarker loader initialized successfully")

            except Exception as e:
                logger.warning(f"Could not load EDAR data: {e}")
                logger.warning("Falling back to mobility-only mode")
                args.use_edar_data = False
                self.edar_attention_loader = None
                self.edar_biomarker_loader = None

        if not args.use_edar_data:
            logger.info(
                "Running in cases-based mode (mobility features -> COVID cases forecasts)"
            )
        else:
            logger.info(
                "Running in full EDAR mode (mobility + wastewater -> COVID cases forecasts)"
            )

    def find_mobility_files(self, args, data_dir: Path) -> list[Path]:
        """
        Find and filter NetCDF files based on mobility path and date range.

        Args:
            args: CLI arguments containing mobility path and date filters
            data_dir: Base data directory

        Returns:
            List of NetCDF file paths matching criteria
        """
        mobility_path = data_dir / args.mobility

        # Check if it's a file or directory
        if mobility_path.is_file():
            if mobility_path.suffix != ".nc":
                raise ValueError(
                    f"Mobility file must be a NetCDF (.nc) file: {mobility_path}"
                )
            return [mobility_path]

        elif mobility_path.is_dir():
            # Find all NetCDF files in directory
            netcdf_files = list(mobility_path.glob("*.nc"))
            if not netcdf_files:
                raise FileNotFoundError(
                    f"No NetCDF files found in directory: {mobility_path}"
                )

            # Filter by date range if provided
            if args.start_date or args.end_date:
                filtered_files = []
                start_date = (
                    datetime.strptime(args.start_date, "%Y-%m-%d")
                    if args.start_date
                    else None
                )
                end_date = (
                    datetime.strptime(args.end_date, "%Y-%m-%d")
                    if args.end_date
                    else None
                )

                for file_path in netcdf_files:
                    # Extract date from filename using regex
                    # Pattern: mitma_mov_cat.daily_personhours.YYYY-MM-DD_YYYY-MM-DD.nc
                    date_match = re.search(
                        r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", file_path.name
                    )
                    if date_match:
                        file_start = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                        file_end = datetime.strptime(date_match.group(2), "%Y-%m-%d")

                        # Check if file overlaps with requested date range
                        include_file = True
                        if start_date and file_end < start_date:
                            include_file = False
                        if end_date and file_start > end_date:
                            include_file = False

                        if include_file:
                            filtered_files.append(file_path)

                if not filtered_files:
                    raise FileNotFoundError(
                        f"No NetCDF files found matching date range {args.start_date} to {args.end_date}"
                    )

                return sorted(filtered_files)

            return sorted(netcdf_files)

        else:
            raise FileNotFoundError(f"Mobility path not found: {mobility_path}")

    def find_population_file(self, args, data_dir: Path) -> Optional[str]:
        """
        Find population data file from standard locations.

        Args:
            args: CLI arguments containing auxiliary data directory
            data_dir: Base data directory

        Returns:
            Path to population file if found, None otherwise
        """
        aux_data_dir = data_dir / args.auxiliary_data_dir
        population_files = [
            aux_data_dir / "population.csv",
            aux_data_dir / "population_data.csv",
            aux_data_dir / "zones_population.csv",
        ]

        for pop_file in population_files:
            if pop_file.exists():
                logger.debug(f"Using population data: {pop_file}")
                return str(pop_file)

        logger.warning("No population data file found, using mobility data only")
        return None

    def create_temporal_graphs(
        self, netcdf_filepath: str, population_filepath: Optional[str]
    ) -> list:
        """
        Create temporal graph sequence using MobilityDataLoader streaming.

        Args:
            netcdf_filepath: Path to NetCDF file containing mobility data
            population_filepath: Path to population file (optional)

        Returns:
            List of temporal graph Data objects. Each Data object has:
            - x: [num_nodes, node_feature_dim] node features
            - edge_index: [2, num_edges] edge connectivity
            - edge_attr: [num_edges, edge_feature_dim] edge attributes
        """
        logger.debug("Creating temporal sequence using MobilityDataLoader streaming")
        temporal_graphs = list(
            self.mobility_loader.stream_dataset(
                netcdf_filepath=netcdf_filepath,
                population_filepath=population_filepath,
                edge_vars=["person_hours"],
                time_slice=None,  # Use all available timesteps
            )
        )

        # Validate temporal graph structure
        if len(temporal_graphs) == 0:
            raise ValueError("No temporal graphs created from NetCDF file")

        # Check that all graphs have consistent structure
        first_graph = temporal_graphs[0]
        num_nodes = first_graph.num_nodes
        node_feature_dim = first_graph.x.shape[1]

        for i, graph in enumerate(temporal_graphs):
            if graph.num_nodes != num_nodes:
                raise ValueError(
                    f"Graph {i} has {graph.num_nodes} nodes, expected {num_nodes}"
                )
            if graph.x.shape[1] != node_feature_dim:
                raise ValueError(
                    f"Graph {i} has {graph.x.shape[1]} features, expected {node_feature_dim}"
                )

        logger.debug(
            f"Created {len(temporal_graphs)} temporal graphs, each with {num_nodes} nodes and {node_feature_dim} features"
        )
        return temporal_graphs

    def prepare_dataset(
        self, args, netcdf_filepath: str, population_filepath: Optional[str]
    ) -> dict[str, Any]:
        """
        Prepare complete dataset dictionary with all metadata.

        Args:
            args: CLI arguments containing configuration
            netcdf_filepath: Path to NetCDF file containing mobility data
            population_filepath: Path to population file (optional)

        Returns:
            Dictionary containing dataset information and loaders.
            Key tensor shapes documented:
            - sample_graph.x: [num_nodes, node_feature_dim]
            - sample_graph.edge_index: [2, num_edges]
            - sample_graph.edge_attr: [num_edges, edge_feature_dim]
        """
        logger.debug("Loading and processing data for dual graph system")

        # Create first sample to understand data structure
        first_sample = self.mobility_loader.create_dataset(
            netcdf_filepath=netcdf_filepath,
            population_filepath=population_filepath,
            edge_vars=["person_hours"],
            time_index=0,
        )

        # Validate sample graph structure
        assert hasattr(first_sample, "x"), "Sample graph missing node features"
        assert hasattr(first_sample, "edge_index"), "Sample graph missing edge index"
        assert first_sample.x.ndim == 2, (
            f"Expected 2D node features, got {first_sample.x.shape}"
        )
        assert first_sample.edge_index.ndim == 2, (
            f"Expected 2D edge index, got {first_sample.edge_index.shape}"
        )
        assert first_sample.edge_index.shape[0] == 2, (
            f"Edge index should have shape [2, num_edges], got {first_sample.edge_index.shape}"
        )

        num_nodes = first_sample.num_nodes
        node_feature_dim = first_sample.x.shape[1]
        num_edges = first_sample.edge_index.shape[1]
        edge_feature_dim = (
            first_sample.edge_attr.shape[1] if hasattr(first_sample, "edge_attr") else 0
        )

        logger.debug("Mobility graph structure:")
        logger.debug(f"  Nodes: {num_nodes}")
        logger.debug(f"  Edges: {num_edges}")
        logger.debug(
            f"  Node features: {first_sample.x.shape} (shape: [num_nodes, node_feature_dim])"
        )
        logger.debug(
            f"  Edge features: {first_sample.edge_attr.shape if hasattr(first_sample, 'edge_attr') else 'None'}"
        )

        # Log detailed node feature information
        logger.info("Node feature analysis:")
        logger.info(f"  Feature tensor shape: {first_sample.x.shape}")
        logger.info(
            f"  Feature dimension: {node_feature_dim} (this becomes mobility_feature_dim)"
        )
        logger.info(f"  Feature tensor dtype: {first_sample.x.dtype}")
        logger.info(
            f"  Sample node features (first 3 nodes, first 5 features): {first_sample.x[:3, :5]}"
        )

        # Validate feature tensor values
        assert not torch.isnan(first_sample.x).any(), "Node features contain NaN values"
        assert not torch.isinf(first_sample.x).any(), (
            "Node features contain infinite values"
        )

        # Extract region IDs by creating a temporary mapping from the NetCDF file
        with xr.open_dataset(netcdf_filepath, engine=self.mobility_loader.engine):
            # Get zone IDs from mobility loader's zone registry
            if self.mobility_loader.zone_ids is not None:
                region_ids = self.mobility_loader.zone_ids
            else:
                # Fallback for cases where zone registry wasn't built
                region_ids = [f"zone_{i}" for i in range(first_sample.num_nodes)]

        # Create dataset structure
        dataset = {
            "netcdf_filepath": netcdf_filepath,
            "population_filepath": population_filepath,
            "sample_graph": first_sample,
            "num_nodes": first_sample.num_nodes,
            "node_feature_dim": first_sample.x.shape[1],
            "edge_feature_dim": first_sample.edge_attr.shape[1],
            "region_ids": region_ids,
            "cases_loader": self.cases_loader,
            "edar_attention_loader": self.edar_attention_loader,
            "edar_biomarker_loader": self.edar_biomarker_loader,
            "use_edar_data": args.use_edar_data,
        }

        # Add EDAR information if available (full EDAR mode)
        if (
            self.edar_attention_loader is not None
            and self.edar_biomarker_loader is not None
        ):
            # Get zone registry from mobility loader for extension
            zone_registry = self.mobility_loader.get_zone_registry()

            if zone_registry is not None:
                # Extend EDAR attention mask to full zone set
                logger.info("Extending EDAR attention mask to full municipality set")
                self.edar_attention_loader.extend_to_full_zone_set(
                    zone_registry=zone_registry,
                    fill_value=0.0,
                    update_normalization=True,
                )

                # Log cross-component validation results
                extension_stats = self.edar_attention_loader.get_extension_statistics()
                logger.info("Cross-component zone validation:")
                logger.info(f"  Mobility zones: {zone_registry.num_zones}")
                logger.info(
                    f"  EDAR municipalities: {extension_stats['original_municipalities']}"
                )
                logger.info(f"  Zone coverage: {extension_stats['coverage_ratio']:.1%}")
                logger.info(
                    f"  Control group size: {extension_stats['missing_municipalities']}"
                )

                # Get treatment/control group info for analysis
                treatment_indices = (
                    self.edar_attention_loader.get_treatment_group_indices()
                )
                control_indices = self.edar_attention_loader.get_control_group_indices()
                logger.info(
                    f"  Treatment group (with EDAR): {len(treatment_indices)} municipalities"
                )
                logger.info(
                    f"  Control group (no EDAR): {len(control_indices)} municipalities"
                )

                # Store extended statistics
                extended_stats = self.edar_attention_loader.get_statistics()

            else:
                logger.warning(
                    "No zone registry available for EDAR extension, using original mask"
                )
                extended_stats = self.edar_attention_loader.get_statistics()
                extension_stats = {}

            edar_biomarker_stats = self.edar_biomarker_loader.get_statistics()

            dataset.update(
                {
                    "n_edars": extended_stats["n_edars"],
                    "n_municipalities": extended_stats["n_municipalities"],
                    "edar_mask_sparsity": extended_stats["sparsity"],
                    "edar_timepoints": edar_biomarker_stats["n_timepoints"],
                    "edar_features": edar_biomarker_stats["features"],
                    "edar_date_range": edar_biomarker_stats["date_range"],
                    "edar_extension_stats": extension_stats,
                }
            )
            logger.debug(
                f"Extended EDAR attention mask: {extended_stats['n_municipalities']} municipalities -> {extended_stats['n_edars']} EDARs"
            )
            logger.debug(f"Extended mask sparsity: {extended_stats['sparsity']:.2%}")
            logger.debug(
                f"EDAR biomarkers: {edar_biomarker_stats['n_timepoints']} time points, {len(edar_biomarker_stats['features'])} features"
            )
            logger.debug(
                f"EDAR date range: {edar_biomarker_stats['date_range'][0]} to {edar_biomarker_stats['date_range'][1]}"
            )

        # Add COVID cases loader to dataset (alignment will be done after temporal graphs are created)
        if self.cases_loader is not None:
            # Log cases data statistics
            cases_stats = self.cases_loader.get_statistics()
            logger.info("COVID cases data statistics:")
            logger.info(f"  Total cases: {cases_stats['total_cases']}")
            logger.info(
                f"  Active municipalities: {cases_stats['active_municipalities']}"
            )
            logger.info(f"  Timepoints: {cases_stats['timepoints']}")
            logger.info(
                f"  Date range: {cases_stats['date_range'][0]} to {cases_stats['date_range'][1]}"
            )
            logger.info(f"  Normalization: {cases_stats['normalization']}")

            # Store cases statistics (tensor alignment will be done later)
            dataset.update(
                {
                    "cases_stats": cases_stats,
                }
            )

        return dataset

    def get_train_val_test_splits(
        self, temporal_graphs: list, args
    ) -> tuple[list, list, list]:
        """
        Create train/validation/test splits from temporal graphs using well-tested windowing.

        Args:
            temporal_graphs: List of temporal graph Data objects. Each graph has:
                - x: [num_nodes, node_feature_dim] features
                - edge_index: [2, num_edges] connectivity
                - edge_attr: [num_edges, edge_feature_dim] attributes
            args: CLI arguments containing forecast_horizon and windowing parameters

        Returns:
            Tuple of (train_graphs, val_graphs, test_graphs)
            Each split maintains the same graph structure as input temporal_graphs
        """
        # Validate input temporal graphs
        assert len(temporal_graphs) > 0, "No temporal graphs provided for splitting"

        sequence_length = 1  # Using k=1 subgraph training
        min_temporal_required = sequence_length + args.forecast_horizon
        total_graphs = len(temporal_graphs)

        # Validate that we have enough temporal data
        if total_graphs < min_temporal_required:
            raise ValueError(
                f"Insufficient temporal data: {total_graphs} graphs available, "
                f"but need minimum {min_temporal_required} for sequence_length={sequence_length} + "
                f"forecast_horizon={args.forecast_horizon}"
            )

        # Get windowing statistics for memory planning and validation
        window_stats = get_window_stats(
            temporal_graphs,
            seq_len=sequence_length,
            horizon=args.forecast_horizon,
            stride=getattr(args, "windowing_stride", 1),
        )

        logger.info("Temporal windowing analysis:")
        logger.info(f"  Total graphs: {window_stats['total_graphs']}")
        logger.info(f"  Valid windows possible: {window_stats['num_windows']}")
        logger.info(f"  Window coverage: {window_stats['window_coverage']:.1%}")
        logger.info(f"  Memory estimate: {window_stats['memory_estimate_mb']:.1f} MB")

        if not window_stats["valid"]:
            suggested_horizon = max(1, (total_graphs - sequence_length) // 3)
            raise ValueError(
                f"Insufficient temporal data for windowing:\n"
                f"  Available graphs: {total_graphs}\n"
                f"  Required: {min_temporal_required} (seq_len={sequence_length} + horizon={args.forecast_horizon})\n"
                f"  Suggestion: Use --forecast_horizon {suggested_horizon} or provide more temporal data"
            )

        # Use well-tested windowing function for chronological splits
        try:
            train_graphs, val_graphs, test_graphs = split_temporal_data(
                temporal_graphs,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                min_graphs_per_split=min_temporal_required,
            )

            logger.info(
                f"Windowing-based data splits - Train: {len(train_graphs)}, "
                f"Val: {len(val_graphs)}, Test: {len(test_graphs)}"
            )

            # Validate each split can create valid windows
            for split_name, split_graphs in [
                ("Train", train_graphs),
                ("Val", val_graphs),
                ("Test", test_graphs),
            ]:
                split_stats = get_window_stats(
                    split_graphs, sequence_length, args.forecast_horizon
                )
                if not split_stats["valid"]:
                    logger.warning(
                        f"{split_name} split has insufficient data for windowing"
                    )
                else:
                    logger.debug(
                        f"{split_name} split can create {split_stats['num_windows']} windows"
                    )

            return train_graphs, val_graphs, test_graphs

        except ValueError as e:
            # Fallback for very limited data
            logger.warning(f"Standard windowing splits failed: {e}")
            logger.warning("Using minimal fallback splits for limited data")

            if total_graphs >= min_temporal_required:
                # Use a single train/test split when data is very limited
                split_point = total_graphs - min_temporal_required
                train_graphs = temporal_graphs[: split_point + sequence_length]
                val_graphs = (
                    temporal_graphs[split_point : split_point + min_temporal_required]
                    if split_point > 0
                    else train_graphs
                )
                test_graphs = temporal_graphs[-min_temporal_required:]

                logger.debug(
                    f"Fallback data splits - Train: {len(train_graphs)}, "
                    f"Val: {len(val_graphs)}, Test: {len(test_graphs)}"
                )
                return train_graphs, val_graphs, test_graphs
            else:
                raise ValueError(
                    f"Insufficient temporal data: {total_graphs} graphs available, "
                    f"but need minimum {min_temporal_required} for sequence_length={sequence_length} + forecast_horizon={args.forecast_horizon}"
                ) from e

    def load_and_process_data(self, args) -> tuple[dict[str, Any], list]:
        """
        Complete data loading and processing pipeline.

        Args:
            args: CLI arguments

        Returns:
            Tuple of (dataset_dict, temporal_graphs)
        """
        # Setup loaders
        self.setup_loaders(args)

        data_dir = Path(args.data_dir)

        # Find NetCDF files
        netcdf_files = self.find_mobility_files(args, data_dir)

        # For now, use the first file (can be extended to handle multiple files)
        netcdf_path = netcdf_files[0]

        if len(netcdf_files) > 1:
            logger.debug(
                f"Found {len(netcdf_files)} NetCDF files, using first: {netcdf_path}"
            )
            logger.debug(f"Available files: {[f.name for f in netcdf_files]}")
        else:
            logger.debug(f"Using NetCDF file: {netcdf_path}")

        # Find population data file
        population_filepath = self.find_population_file(args, data_dir)

        # Prepare dataset dictionary
        dataset = self.prepare_dataset(args, str(netcdf_path), population_filepath)

        # Create temporal graphs
        temporal_graphs = self.create_temporal_graphs(
            str(netcdf_path), population_filepath
        )

        # Align datasets using advanced alignment manager if enabled
        if hasattr(args, "crop_datasets") and args.crop_datasets:
            self._align_all_datasets(args, dataset, temporal_graphs)
        else:
            # Use traditional alignment for backward compatibility
            if self.cases_loader is not None and "cases_tensor" not in dataset:
                self._align_cases_with_temporal_graphs(dataset, temporal_graphs)

        # Optionally attach pretrained region embeddings to node features
        self._maybe_attach_region_embeddings(args, dataset, temporal_graphs)

        return dataset, temporal_graphs

    def _align_cases_with_temporal_graphs(
        self, dataset: dict, temporal_graphs: list
    ) -> None:
        """
        Align COVID cases data with temporal graphs structure.

        Args:
            dataset: Dataset dictionary containing cases loader and region IDs
            temporal_graphs: List of temporal graph objects
        """
        if self.cases_loader is None or "cases_tensor" in dataset:
            return

        logger.info("Aligning COVID cases data with temporal graphs")

        try:
            # Get municipality IDs from mobility data (assuming region_ids are municipality codes)
            region_ids = dataset["region_ids"]
            mobility_municipalities = [
                int(region_id) if region_id.isdigit() else -1
                for region_id in region_ids
            ]

            # Create date range for alignment (based on temporal graphs count)
            # This is a simplified approach - in practice, you'd extract actual dates from NetCDF
            import pandas as pd

            cases_stats = dataset["cases_stats"]
            start_date = pd.to_datetime(cases_stats["date_range"][0])
            mobility_dates = [
                start_date + pd.Timedelta(days=i) for i in range(len(temporal_graphs))
            ]

            # Align cases data with mobility structure
            aligned_cases_tensor, alignment_info = (
                self.cases_loader.align_with_mobility_data(
                    mobility_municipalities, mobility_dates
                )
            )

            # Add alignment information to dataset
            dataset.update(
                {
                    "cases_tensor": aligned_cases_tensor,
                    "cases_alignment_info": alignment_info,
                }
            )

            logger.info("COVID cases data alignment completed:")
            logger.info(
                f"  Aligned municipalities: {alignment_info['aligned_municipalities']}/{alignment_info['mobility_municipalities']}"
            )
            logger.info(f"  Coverage ratio: {alignment_info['coverage_ratio']:.1%}")

        except Exception as e:
            logger.error(f"Failed to align COVID cases data with mobility data: {e}")
            raise

    # ---- Region embeddings integration
    def _maybe_attach_region_embeddings(
        self,
        args,
        dataset: dict[str, Any],
        temporal_graphs: list,
    ) -> None:
        """
        If enabled, load pretrained region embeddings and append to node features.

        Expects `args.use_region_embeddings` (flag) and `args.region_embeddings_path`.
        Embeddings must have shape [num_nodes, embed_dim] with num_nodes matching the
        mobility graph's number of nodes and in the same zone order.

        Tensor shape transformations:
        - Input embeddings: [num_nodes, embed_dim]
        - Original node features: [num_nodes, feature_dim]
        - Output node features: [num_nodes, feature_dim + embed_dim]
        """
        use_embeds = getattr(args, "use_region_embeddings", False)
        embeds_path = getattr(args, "region_embeddings_path", None)

        if not use_embeds:
            return
        if not embeds_path:
            raise ValueError(
                "--use_region_embeddings is set but --region_embeddings_path was not provided."
            )

        path = Path(embeds_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Region embeddings file not found: {path}. Configure --region_embeddings_path."
            )

        logger.info(f"Loading pretrained region embeddings from: {path}")

        Z = None
        try:
            suffix = path.suffix.lower()
            if suffix in [".pt", ".pth"]:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, dict) and "final_embeddings" in loaded:
                    Z = loaded["final_embeddings"]
                elif torch.is_tensor(loaded):
                    Z = loaded
                else:
                    # Best-effort: look for common keys
                    for k in ["embeddings", "region_embeddings", "Z"]:
                        if isinstance(loaded, dict) and k in loaded:
                            Z = loaded[k]
                            break
            elif suffix == ".npy":
                Z = torch.from_numpy(np.load(str(path)))
            elif suffix == ".npz":
                npz = np.load(str(path))
                key = "embeddings" if "embeddings" in npz else list(npz.keys())[0]
                Z = torch.from_numpy(npz[key])
            else:
                raise ValueError(
                    f"Unsupported embeddings format '{suffix}'. Use .pt/.pth/.npy/.npz"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load region embeddings: {e}")

        if Z is None:
            raise ValueError(
                "Region embeddings could not be resolved from file contents"
            )

        # Validate and normalize embeddings tensor
        if Z.dim() == 1:
            Z = Z.unsqueeze(1)  # [num_nodes] -> [num_nodes, 1]
        if Z.dtype != torch.float32:
            Z = Z.float()

        # Validate shape compatibility
        num_nodes = dataset["num_nodes"]
        if Z.shape[0] != num_nodes:
            raise ValueError(
                f"Region embeddings row count ({Z.shape[0]}) does not match graph nodes ({num_nodes})."
            )

        # Check for NaN/inf values
        assert not torch.isnan(Z).any(), "Region embeddings contain NaN values"
        assert not torch.isinf(Z).any(), "Region embeddings contain infinite values"

        embed_dim = Z.shape[1]
        logger.info(
            f"Appending region embeddings to node features: shape {tuple(Z.shape)}"
        )

        # Attach to sample graph in dataset
        sample_graph = dataset.get("sample_graph")
        if sample_graph is not None and hasattr(sample_graph, "x"):
            # Validate concatenation compatibility
            assert sample_graph.x.shape[0] == Z.shape[0], (
                f"Sample graph nodes ({sample_graph.x.shape[0]}) don't match embeddings ({Z.shape[0]})"
            )
            sample_graph.x = torch.cat(
                [sample_graph.x, Z.to(sample_graph.x.device)], dim=1
            )
            logger.debug(
                f"Sample graph features: {sample_graph.x.shape} (after concatenation)"
            )

        # Attach to every temporal graph
        original_feature_dim = temporal_graphs[0].x.shape[1]
        for i, g in enumerate(temporal_graphs):
            assert g.x.shape[0] == Z.shape[0], (
                f"Temporal graph {i} nodes ({g.x.shape[0]}) don't match embeddings ({Z.shape[0]})"
            )
            g.x = torch.cat([g.x, Z.to(g.x.device)], dim=1)
            assert g.x.shape == (num_nodes, original_feature_dim + embed_dim), (
                f"Unexpected shape after concatenation: {g.x.shape}"
            )

        # Update dataset metadata
        original_dim = int(dataset["node_feature_dim"])
        dataset["node_feature_dim"] = original_dim + int(embed_dim)
        dataset["region_embedding_dim"] = int(embed_dim)
        logger.info(
            f"Node feature dim updated to {dataset['node_feature_dim']} (+= {embed_dim} from embeddings)"
        )

    def _align_all_datasets(self, args, dataset: dict, temporal_graphs: list) -> None:
        """
        Align all datasets using the advanced alignment manager.

        Args:
            args: CLI arguments containing alignment configuration
            dataset: Dataset dictionary containing all data loaders
            temporal_graphs: List of temporal graph objects
        """
        logger.info("Starting advanced multi-dataset alignment")

        # Create alignment manager
        self.alignment_manager = create_alignment_manager(
            target_dataset_name=getattr(args, "target_dataset", "cases"),
            padding_strategy=getattr(args, "padding_strategy", "interpolate"),
            crop_datasets=getattr(args, "crop_datasets", True),
            alignment_buffer_days=getattr(args, "alignment_buffer_days", 0),
            interpolation_method=getattr(args, "interpolation_method", "linear"),
            validate_alignment=getattr(args, "validate_alignment", True),
        )

        # Prepare datasets for alignment
        datasets_to_align = {}
        dataset_dates = {}
        dataset_entities = {}

        # Add cases data
        if self.cases_loader is not None and self.cases_loader.cases_tensor is not None:
            datasets_to_align["cases"] = self.cases_loader.cases_tensor
            dataset_dates["cases"] = [
                self.cases_loader.date_range[0] + timedelta(days=i)
                for i in range(self.cases_loader.timepoints)
            ]
            dataset_entities["cases"] = self.cases_loader.municipalities

        # Add mobility data from temporal graphs
        if temporal_graphs:
            # Extract mobility tensor from temporal graphs
            mobility_tensor = torch.stack(
                [graph.x for graph in temporal_graphs]
            ).transpose(0, 1)
            datasets_to_align["mobility"] = mobility_tensor

            # Extract dates from temporal graphs (simplified - assumes consecutive days)
            start_date = dataset.get("sample_graph", {}).get("date", datetime.now())
            dataset_dates["mobility"] = [
                start_date + timedelta(days=i) for i in range(len(temporal_graphs))
            ]
            dataset_entities["mobility"] = dataset.get(
                "region_ids", list(range(mobility_tensor.shape[0]))
            )

        # Add EDAR data if available
        if self.edar_biomarker_loader is not None and hasattr(
            self.edar_biomarker_loader, "biomarker_tensor"
        ):
            datasets_to_align["edar"] = self.edar_biomarker_loader.biomarker_tensor
            dataset_dates["edar"] = self.edar_biomarker_loader.time_index
            dataset_entities["edar"] = list(
                range(self.edar_biomarker_loader.biomarker_tensor.shape[0])
            )

        # Perform alignment if we have multiple datasets
        if len(datasets_to_align) > 1:
            try:
                alignment_result = self.alignment_manager.align_datasets(
                    datasets_to_align, dataset_dates, dataset_entities
                )

                # Update dataset with aligned data
                if "cases" in alignment_result["aligned_datasets"]:
                    dataset["cases_tensor"] = alignment_result["aligned_datasets"][
                        "cases"
                    ]
                    dataset["cases_alignment_info"] = alignment_result[
                        "alignment_stats"
                    ]["cases"]

                if "mobility" in alignment_result["aligned_datasets"]:
                    # Update temporal graphs with aligned mobility data
                    aligned_mobility = alignment_result["aligned_datasets"][
                        "mobility"
                    ].transpose(0, 1)
                    for i, graph in enumerate(temporal_graphs):
                        if i < aligned_mobility.shape[0]:
                            graph.x = aligned_mobility[i]

                if "edar" in alignment_result["aligned_datasets"]:
                    dataset["edar_tensor"] = alignment_result["aligned_datasets"][
                        "edar"
                    ]
                    dataset["edar_alignment_info"] = alignment_result[
                        "alignment_stats"
                    ]["edar"]

                # Add alignment summary to dataset
                dataset["alignment_summary"] = (
                    self.alignment_manager.get_alignment_summary()
                )

                logger.info("Multi-dataset alignment completed successfully")
                logger.info(f"Alignment summary: {dataset['alignment_summary']}")

            except Exception as e:
                logger.error(f"Multi-dataset alignment failed: {e}")
                logger.warning("Falling back to traditional alignment method")
                # Fall back to traditional alignment
                if self.cases_loader is not None and "cases_tensor" not in dataset:
                    self._align_cases_with_temporal_graphs(dataset, temporal_graphs)
        else:
            logger.info("Only one dataset available, using traditional alignment")
            if self.cases_loader is not None and "cases_tensor" not in dataset:
                self._align_cases_with_temporal_graphs(dataset, temporal_graphs)
