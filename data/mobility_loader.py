"""
Mobility data loader with NetCDF streaming and population data integration.
"""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import reduce
from torch_geometric.data import Data

from .zone_registry import ZoneRegistry, create_zone_registry_from_mobility_coords

logger = logging.getLogger(__name__)


class MobilityDataLoader:
    """
    Loads and processes origin-destination (O-D) mobility data from NetCDF files
    with population data integration and preprocessing hooks.

    This class handles NetCDF streaming of mobility networks, population data loading,
    and provides preprocessing hooks for data normalization, filtering, and feature engineering.
    """

    def __init__(
        self,
        min_flow_threshold: Union[int, float] = 10,
        normalize_flows: bool = True,
        undirected: bool = False,
        allow_self_loops: bool = False,
        edge_selector: Union[str, float, None] = "nonzero",
        node_stats: tuple[str, ...] = ("sum", "mean", "count_nonzero"),
        engine: str = "h5netcdf",
        chunks: Optional[dict] = None,
    ):
        """
        Initialize the mobility data loader.

        Args:
            min_flow_threshold: Minimum flow count to include in graph
            normalize_flows: Whether to normalize flow values
            undirected: Make graph undirected
            allow_self_loops: Include self-loops
            edge_selector: Edge filtering strategy ("nonzero", threshold value, or None)
            node_stats: Statistics to compute for node features from edges
            engine: NetCDF engine ('h5netcdf', 'netcdf4')
            chunks: Chunking configuration for dask
        """
        self.min_flow_threshold = min_flow_threshold
        self.normalize_flows = normalize_flows
        self.undirected = undirected
        self.allow_self_loops = allow_self_loops
        self.edge_selector = edge_selector
        self.node_stats = node_stats
        self.engine = engine
        self.chunks = chunks or {"time": 1}

        # Preprocessing hooks - functions that can be registered for custom preprocessing
        self.preprocessing_hooks = {
            "netcdf_preprocessing": [],
            "population_preprocessing": [],
            "post_merge_preprocessing": [],
        }

        # Zone registry for centralized zone management
        self._zone_registry: Optional[ZoneRegistry] = None

    def register_preprocessing_hook(self, hook_type: str, func: callable):
        """
        Register a preprocessing function to be called during data loading.

        Args:
            hook_type: Type of preprocessing ('netcdf_preprocessing', 'population_preprocessing',
                      'post_merge_preprocessing')
            func: Function to call, should take and return the relevant data
        """
        if hook_type not in self.preprocessing_hooks:
            raise ValueError(f"Unknown hook type: {hook_type}")

        self.preprocessing_hooks[hook_type].append(func)
        logger.info(f"Registered preprocessing hook: {func.__name__} for {hook_type}")

    def set_zone_registry(self, zone_registry: ZoneRegistry) -> "MobilityDataLoader":
        """
        Set a zone registry to use for zone management.

        Args:
            zone_registry: Pre-built ZoneRegistry instance

        Returns:
            Self for method chaining
        """
        self._zone_registry = zone_registry
        logger.info(
            f"Using provided zone registry with {zone_registry.num_zones} zones"
        )
        return self

    def get_zone_registry(self) -> Optional[ZoneRegistry]:
        """Get the current zone registry, if any."""
        return self._zone_registry

    @property
    def zone_ids(self) -> Optional[list[str]]:
        """Get zone IDs from registry, if available."""
        return self._zone_registry.zone_ids if self._zone_registry else None

    def _build_id_map(
        self, home_zones: np.ndarray, dest_zones: np.ndarray
    ) -> tuple[dict[str, int], list[str], np.ndarray, np.ndarray]:
        """Build optimized mapping from zone string IDs to tensor indices using ZoneRegistry."""

        # Create or use existing zone registry
        if self._zone_registry is None:
            # Create registry from mobility coordinates
            self._zone_registry = create_zone_registry_from_mobility_coords(
                home_coords=home_zones, dest_coords=dest_zones, filter_zones={"out_cat"}
            )
            logger.info("Created new zone registry from mobility coordinates")
        else:
            # Extend registry with zones from current dataset if needed
            home_zone_strs = [str(zone) for zone in home_zones]
            dest_zone_strs = [str(zone) for zone in dest_zones]
            all_zone_strs = list(set(home_zone_strs + dest_zone_strs))

            self._zone_registry.extend_with_zones(all_zone_strs, "mobility_data")
            logger.info("Extended existing zone registry with mobility zones")

        # Get zone mappings from registry
        id2idx = self._zone_registry.id_to_idx
        idx2id = self._zone_registry.idx_to_id

        # Create fast lookup arrays using registry
        home_zone_strs = np.array([str(zone) for zone in home_zones])
        dest_zone_strs = np.array([str(zone) for zone in dest_zones])

        home_zone_indices = self._zone_registry.get_zone_indices(
            home_zone_strs.tolist()
        )
        dest_zone_indices = self._zone_registry.get_zone_indices(
            dest_zone_strs.tolist()
        )

        logger.info(
            f"Built zone mapping using registry with {self._zone_registry.num_zones} zones"
        )
        return id2idx, idx2id, home_zone_indices, dest_zone_indices

    def _make_edge_index_and_attr(
        self,
        ds_t: xr.Dataset,
        edge_vars: list[str],
        id2idx: dict[str, int],
        home_zone_indices: Optional[np.ndarray] = None,
        dest_zone_indices: Optional[np.ndarray] = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Extract edge connectivity and attributes from NetCDF dataset using vectorized operations.

        Args:
            ds_t: NetCDF dataset containing mobility data
            edge_vars: List of edge variable names to extract
            id2idx: Mapping from zone IDs to tensor indices
            home_zone_indices: Pre-computed home zone indices (optional)
            dest_zone_indices: Pre-computed destination zone indices (optional)

        Returns:
            Tuple of (edge_index, edge_attr)
            - edge_index: [2, num_edges] tensor of edge connectivity
            - edge_attr: [num_edges, len(edge_vars)] tensor of edge attributes
        """
        home_zones = ds_t.coords.get("home", ds_t.coords.get("origin", [])).values
        dest_zones = ds_t.coords.get("destination", ds_t.coords.get("dest", [])).values

        # Use pre-computed zone indices if available, otherwise compute them
        if home_zone_indices is None or dest_zone_indices is None:
            home_zone_strs = np.array([str(zone) for zone in home_zones])
            dest_zone_strs = np.array([str(zone) for zone in dest_zones])

            home_zone_indices = np.array(
                [id2idx.get(zone_str, -1) for zone_str in home_zone_strs]
            )
            dest_zone_indices = np.array(
                [id2idx.get(zone_str, -1) for zone_str in dest_zone_strs]
            )

        # Create boolean masks for valid zones (exclude -1 which represents invalid zones)
        home_valid_mask = home_zone_indices >= 0
        dest_valid_mask = dest_zone_indices >= 0

        # Get valid indices and create index mappings
        valid_home_indices = np.where(home_valid_mask)[0]
        valid_dest_indices = np.where(dest_valid_mask)[0]

        if len(valid_home_indices) == 0 or len(valid_dest_indices) == 0:
            logger.warning("No valid zones found")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, len(edge_vars)), dtype=torch.float
            )

        # Get node indices for valid zones
        home_node_indices = home_zone_indices[valid_home_indices]
        dest_node_indices = dest_zone_indices[valid_dest_indices]

        all_edge_indices = []
        all_edge_attrs = []

        for var_name in edge_vars:
            if var_name not in ds_t:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue

            var_data = ds_t[var_name].values
            if var_data.ndim == 3:
                var_data = var_data[0]  # Take first time slice
            elif var_data.ndim != 2:
                logger.warning(
                    f"Unsupported dimensionality for {var_name}: {var_data.ndim}"
                )
                continue

            # Extract submatrix for valid zones only
            valid_data = var_data[np.ix_(valid_home_indices, valid_dest_indices)]

            # Create coordinate meshgrids for vectorized operations
            # Shape: [num_valid_homes, num_valid_dests]
            home_mesh, dest_mesh = np.meshgrid(
                home_node_indices, dest_node_indices, indexing="ij"
            )

            # Flatten for edge list representation using einops for clarity
            # home_flat: [num_valid_homes * num_valid_dests]
            # dest_flat: [num_valid_homes * num_valid_dests]
            # values_flat: [num_valid_homes * num_valid_dests]
            home_flat = home_mesh.flatten()
            dest_flat = dest_mesh.flatten()
            values_flat = valid_data.flatten()

            # Validate shapes before processing
            assert len(home_flat) == len(dest_flat) == len(values_flat), (
                f"Shape mismatch after flattening: home={len(home_flat)}, "
                f"dest={len(dest_flat)}, values={len(values_flat)}"
            )

            # Apply self-loop filter
            if not self.allow_self_loops:
                self_loop_mask = home_flat != dest_flat
                home_flat = home_flat[self_loop_mask]
                dest_flat = dest_flat[self_loop_mask]
                values_flat = values_flat[self_loop_mask]

            # Apply edge filtering
            if self.edge_selector == "nonzero":
                valid_mask = (~np.isnan(values_flat)) & (values_flat > 0)
            elif isinstance(self.edge_selector, (int, float)):
                valid_mask = (~np.isnan(values_flat)) & (
                    values_flat >= self.edge_selector
                )
            elif self.edge_selector is None:
                valid_mask = np.ones(len(values_flat), dtype=bool)
                values_flat = np.nan_to_num(values_flat, nan=0.0)
            else:
                valid_mask = np.ones(len(values_flat), dtype=bool)

            # Apply threshold filter
            threshold_mask = values_flat >= self.min_flow_threshold
            final_mask = valid_mask & threshold_mask

            if not np.any(final_mask):
                continue

            # Extract valid edges
            valid_homes = home_flat[final_mask]
            valid_dests = dest_flat[final_mask]
            valid_values = values_flat[final_mask]

            # Create edge indices and attributes
            edge_indices = np.column_stack([valid_homes, valid_dests])
            edge_attrs = valid_values.reshape(-1, 1)

            # Add reverse edges for undirected graphs
            if self.undirected:
                # Only add reverse edges where source != destination
                non_self_mask = valid_homes != valid_dests
                if np.any(non_self_mask):
                    reverse_indices = np.column_stack(
                        [valid_dests[non_self_mask], valid_homes[non_self_mask]]
                    )
                    reverse_attrs = valid_values[non_self_mask].reshape(-1, 1)

                    edge_indices = np.vstack([edge_indices, reverse_indices])
                    edge_attrs = np.vstack([edge_attrs, reverse_attrs])

            all_edge_indices.append(edge_indices)
            all_edge_attrs.append(edge_attrs)

        if not all_edge_indices:
            logger.warning("No valid edges found")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, len(edge_vars)), dtype=torch.float
            )

        # Concatenate all edges
        if all_edge_indices:
            final_edge_indices = np.vstack(all_edge_indices)  # [total_edges, 2]
            final_edge_attrs = (
                np.hstack(all_edge_attrs) if len(edge_vars) > 1 else all_edge_attrs[0]
            )  # [total_edges, num_edge_vars]
        else:
            # Handle empty case
            final_edge_indices = np.empty((0, 2), dtype=np.int64)
            final_edge_attrs = np.empty((0, len(edge_vars)), dtype=np.float32)

        # Convert to tensors with shape validation
        edge_index = torch.LongTensor(
            final_edge_indices.T
        ).contiguous()  # [2, num_edges]
        edge_attr = torch.FloatTensor(final_edge_attrs)  # [num_edges, num_edge_vars]

        # Validate final tensor shapes
        assert edge_index.shape[0] == 2, (
            f"Edge index should have shape [2, num_edges], got {edge_index.shape}"
        )
        assert edge_index.shape[1] == edge_attr.shape[0], (
            f"Edge index and attr mismatch: {edge_index.shape} vs {edge_attr.shape}"
        )
        assert edge_attr.shape[1] == len(edge_vars), (
            f"Edge attr should have {len(edge_vars)} columns, got {edge_attr.shape[1]}"
        )

        # Normalize flows if requested
        if self.normalize_flows and edge_attr.numel() > 0:
            max_val = edge_attr.max()
            if max_val > 0:
                edge_attr = edge_attr / max_val

        return edge_index, edge_attr

    def _make_node_features_from_edges(
        self,
        ds_t: xr.Dataset,
        edge_vars: list[str],
        id2idx: dict[str, int],
        home_zone_indices: Optional[np.ndarray] = None,
        dest_zone_indices: Optional[np.ndarray] = None,
    ) -> torch.FloatTensor:
        """Compute node features by aggregating edge statistics using vectorized operations.

        Args:
            ds_t: NetCDF dataset containing mobility data
            edge_vars: List of edge variable names to aggregate
            id2idx: Mapping from zone IDs to tensor indices
            home_zone_indices: Pre-computed home zone indices (optional)
            dest_zone_indices: Pre-computed destination zone indices (optional)

        Returns:
            Node features tensor of shape [num_nodes, len(node_stats) * len(edge_vars)]
        """
        num_nodes = len(id2idx)
        home_zones = ds_t.coords.get("home", ds_t.coords.get("origin", [])).values
        dest_zones = ds_t.coords.get("destination", ds_t.coords.get("dest", [])).values

        # Use pre-computed zone indices if available, otherwise compute them
        if home_zone_indices is None or dest_zone_indices is None:
            home_zone_strs = np.array([str(zone) for zone in home_zones])
            dest_zone_strs = np.array([str(zone) for zone in dest_zones])

            home_zone_indices = np.array(
                [id2idx.get(zone_str, -1) for zone_str in home_zone_strs]
            )
            dest_zone_indices = np.array(
                [id2idx.get(zone_str, -1) for zone_str in dest_zone_strs]
            )

        # Create masks for valid zones
        home_valid_mask = home_zone_indices >= 0
        dest_valid_mask = dest_zone_indices >= 0

        # Get valid indices
        valid_home_indices = np.where(home_valid_mask)[0]
        valid_dest_indices = np.where(dest_valid_mask)[0]

        if len(valid_home_indices) == 0 or len(valid_dest_indices) == 0:
            # Return zero tensor with correct shape
            expected_shape = (num_nodes, len(self.node_stats) * len(edge_vars))
            return torch.zeros(expected_shape, dtype=torch.float)

        # Use the pre-computed zone indices directly
        home_to_node_idx = home_zone_indices  # This is already the mapping we need

        features = []

        for var_name in edge_vars:
            if var_name not in ds_t:
                continue

            var_data = ds_t[var_name].values
            if var_data.ndim == 3:
                var_data = var_data[0]

            # Extract submatrix for valid zones
            valid_data = var_data[np.ix_(valid_home_indices, valid_dest_indices)]

            # Replace NaN with 0 for computation
            valid_data_clean = np.nan_to_num(valid_data, nan=0.0)

            # Initialize node statistics array
            node_stats = np.zeros((num_nodes, len(self.node_stats)))

            # Compute statistics using vectorized operations with einops where helpful
            for k, stat in enumerate(self.node_stats):
                if stat == "sum":
                    # Sum across destination dimension (axis=1)
                    # Shape: [num_valid_homes] (sum over destinations)
                    stat_values = reduce(
                        valid_data_clean, "homes destinations -> homes", "sum"
                    )
                elif stat == "mean":
                    # Mean of non-zero values
                    # Create mask for non-zero values
                    nonzero_mask = valid_data_clean != 0
                    # Compute mean only where there are non-zero values
                    row_sums = reduce(
                        valid_data_clean, "homes destinations -> homes", "sum"
                    )
                    row_counts = reduce(
                        nonzero_mask.astype(float), "homes destinations -> homes", "sum"
                    )
                    stat_values = np.divide(
                        row_sums,
                        row_counts,
                        out=np.zeros_like(row_sums),
                        where=row_counts != 0,
                    )
                elif stat == "count_nonzero":
                    # Count non-zero values across destinations
                    # Shape: [num_valid_homes]
                    stat_values = reduce(
                        (valid_data_clean != 0).astype(int),
                        "homes destinations -> homes",
                        "sum",
                    )
                elif stat == "max":
                    # Maximum value across destinations
                    # Shape: [num_valid_homes]
                    stat_values = reduce(
                        valid_data_clean, "homes destinations -> homes", "max"
                    )
                elif stat == "std":
                    # Standard deviation across destinations
                    # Only compute for rows with at least 2 non-zero values
                    nonzero_mask = valid_data_clean != 0
                    row_counts = reduce(
                        nonzero_mask.astype(int), "homes destinations -> homes", "sum"
                    )

                    stat_values = np.zeros(len(valid_home_indices))
                    for i, _home_idx_in_valid in enumerate(valid_home_indices):
                        if row_counts[i] >= 2:  # Need at least 2 values for std
                            row_values = valid_data_clean[i, nonzero_mask[i]]
                            stat_values[i] = np.std(row_values)
                else:
                    # Unknown statistic, skip
                    stat_values = np.zeros(len(valid_home_indices))
                    logger.warning(f"Unknown node statistic: {stat}")

                # Map computed statistics to correct node indices
                for i, home_idx_in_original in enumerate(valid_home_indices):
                    node_idx = home_to_node_idx[home_idx_in_original]
                    if node_idx >= 0:  # Valid node index (>= 0, not != -1)
                        node_stats[node_idx, k] = stat_values[i]

            features.append(node_stats)

        if features:
            all_features = np.concatenate(features, axis=1)
        else:
            all_features = np.zeros((num_nodes, len(self.node_stats)))

        return torch.FloatTensor(all_features)

    def load_population_data(self, filepath: str) -> pd.DataFrame:
        """
        Load population data for zones.

        Expected columns: ['id', 'd.population', 'd.density_pop_m2'] (optional)

        Args:
            filepath: Path to population CSV file

        Returns:
            DataFrame with population data
        """
        logger.info(f"Loading population data from {filepath}")

        population_data = pd.read_csv(filepath)

        # Apply preprocessing hooks for population data
        for hook in self.preprocessing_hooks["population_preprocessing"]:
            logger.info(f"Applying population preprocessing hook: {hook.__name__}")
            population_data = hook(population_data)

        logger.info(f"Loaded population data for {len(population_data)} zones")
        return population_data

    def create_dataset(
        self,
        netcdf_filepath: str,
        population_filepath: Optional[str] = None,
        edge_vars: list[str] = None,
        time_index: int = 0,
    ) -> Data:
        """
        Create a PyTorch Geometric Data object from NetCDF and population data.

        Args:
            netcdf_filepath: Path to NetCDF file containing mobility data
            population_filepath: Path to population CSV file (optional)
            edge_vars: List of edge variable names to extract
            time_index: Which time index to use from NetCDF

        Returns:
            PyTorch Geometric Data object with structure:
            - x: [num_nodes, node_feature_dim] node features
            - edge_index: [2, num_edges] edge connectivity
            - edge_attr: [num_edges, edge_feature_dim] edge attributes
        """
        if edge_vars is None:
            edge_vars = ["person_hours"]
        logger.info("Creating dataset from NetCDF and population data")

        # Load NetCDF data
        try:
            ds = xr.open_dataset(
                netcdf_filepath, engine=self.engine, chunks=self.chunks
            )
            logger.info(f"Opened NetCDF dataset: {netcdf_filepath}")
        except Exception as e:
            logger.error(f"Failed to open {netcdf_filepath}: {e}")
            raise

        # Apply NetCDF preprocessing hooks
        for hook in self.preprocessing_hooks["netcdf_preprocessing"]:
            logger.info(f"Applying NetCDF preprocessing hook: {hook.__name__}")
            ds = hook(ds)

        # Get time dimension
        if "time" in ds.dims:
            times = ds["time"].values
            if time_index >= len(times):
                raise ValueError(
                    f"Time index {time_index} >= {len(times)} available times"
                )
            selected_time = times[time_index]
            ds_t = ds[edge_vars].sel(time=selected_time, method="nearest")
            logger.info(f"Selected time index {time_index}: {selected_time}")
        else:
            ds_t = ds[edge_vars]
            logger.info("No time dimension found, using all data")

        # Build zone ID mapping with optimized arrays
        home_coords = ds_t.coords.get("home", ds_t.coords.get("origin", None))
        dest_coords = ds_t.coords.get("destination", ds_t.coords.get("dest", None))

        if home_coords is None or dest_coords is None:
            raise ValueError("Could not find home/origin and destination coordinates")

        id2idx, idx2id, home_zone_indices, dest_zone_indices = self._build_id_map(
            home_coords.values, dest_coords.values
        )

        # Extract graph structure using optimized zone indices
        # edge_index: [2, num_edges], edge_attr: [num_edges, len(edge_vars)]
        edge_index, edge_attr = self._make_edge_index_and_attr(
            ds_t, edge_vars, id2idx, home_zone_indices, dest_zone_indices
        )

        # Compute node features from edge aggregations using optimized zone indices
        # mobility_features: [num_nodes, len(node_stats) * len(edge_vars)]
        mobility_features = self._make_node_features_from_edges(
            ds_t, edge_vars, id2idx, home_zone_indices, dest_zone_indices
        )

        # Load population data if provided
        node_features = mobility_features
        if population_filepath:
            population_data = self.load_population_data(population_filepath)
            # population_features: [num_nodes, num_pop_features]
            population_features = self._prepare_population_features(
                population_data, idx2id
            )
            # Concatenate features: [num_nodes, total_feature_dim]
            node_features = torch.cat([mobility_features, population_features], dim=1)
            logger.info(
                f"Combined mobility and population features: {node_features.shape}"
            )

        # Apply post-merge preprocessing hooks
        data_dict = {
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": len(idx2id),
        }

        for hook in self.preprocessing_hooks["post_merge_preprocessing"]:
            logger.info(f"Applying post-merge preprocessing hook: {hook.__name__}")
            data_dict = hook(data_dict)

        # Create PyTorch Geometric Data object and validate structure
        data = Data(**data_dict)

        # Validate final data structure
        assert hasattr(data, "x"), "Created Data object missing node features"
        assert hasattr(data, "edge_index"), "Created Data object missing edge index"
        assert data.x.ndim == 2, f"Expected 2D node features, got {data.x.shape}"
        assert data.edge_index.ndim == 2, (
            f"Expected 2D edge index, got {data.edge_index.shape}"
        )
        assert data.edge_index.shape[0] == 2, (
            f"Edge index should have shape [2, num_edges], got {data.edge_index.shape}"
        )

        ds.close()
        logger.info(
            f"Dataset creation completed: {data.num_nodes} nodes, {data.num_edges} edges, "
            f"node features: {data.x.shape}"
        )
        return data

    def _prepare_population_features(
        self, population_data: pd.DataFrame, idx2id: list[str]
    ) -> torch.FloatTensor:
        """Prepare population node features."""
        pop_dict = dict(
            zip(population_data["id"].astype(str), population_data["d.population"])
        )
        density_dict = (
            dict(
                zip(
                    population_data["id"].astype(str),
                    population_data["d.density_pop_m2"],
                )
            )
            if "d.density_pop_m2" in population_data.columns
            else {}
        )

        populations = []
        densities = []

        for zone_id in idx2id:
            pop = pop_dict.get(zone_id, 1000.0)
            density = density_dict.get(zone_id, 100.0)
            populations.append(pop)
            densities.append(density)

        # Log-transform and normalize
        populations = np.array(populations)
        log_pop = np.log1p(populations)

        features = [log_pop]
        if density_dict:
            densities = np.array(densities)
            log_density = np.log1p(densities)
            features.append(log_density)

        return torch.FloatTensor(np.column_stack(features))

    def stream_dataset(
        self,
        netcdf_filepath: str,
        population_filepath: Optional[str] = None,
        edge_vars: list[str] = None,
        time_slice: Optional[slice] = None,
    ):
        """
        Stream PyTorch Geometric Data objects for each time step.

        Args:
            netcdf_filepath: Path to NetCDF file
            population_filepath: Path to population CSV file (optional)
            edge_vars: List of edge variable names to extract
            time_slice: Slice of time dimension to use

        Yields:
            PyTorch Geometric Data objects for each time step
        """
        if edge_vars is None:
            edge_vars = ["person_hours"]
        logger.info("Starting dataset streaming")

        ds = xr.open_dataset(netcdf_filepath, engine=self.engine, chunks=self.chunks)

        if "time" not in ds.dims:
            raise ValueError("Time dimension required for streaming dataset")

        times = ds["time"].values
        if time_slice is None:
            selected_times = times
        else:
            selected_times = times[time_slice]

        # Build zone mapping once with optimized arrays
        home_coords = ds.coords.get("home", ds.coords.get("origin", None))
        dest_coords = ds.coords.get("destination", ds.coords.get("dest", None))
        id2idx, idx2id, home_zone_indices, dest_zone_indices = self._build_id_map(
            home_coords.values, dest_coords.values
        )

        # Load population features once
        population_features = None
        if population_filepath:
            population_data = self.load_population_data(population_filepath)
            population_features = self._prepare_population_features(
                population_data, idx2id
            )

        logger.info(f"Streaming {len(selected_times)} time steps")

        for t in selected_times:
            try:
                ds_t = ds[edge_vars].sel(time=t, method="nearest")

                edge_index, edge_attr = self._make_edge_index_and_attr(
                    ds_t, edge_vars, id2idx, home_zone_indices, dest_zone_indices
                )
                mobility_features = self._make_node_features_from_edges(
                    ds_t, edge_vars, id2idx, home_zone_indices, dest_zone_indices
                )

                if population_features is not None:
                    node_features = torch.cat(
                        [mobility_features, population_features], dim=1
                    )
                else:
                    node_features = mobility_features

                # Create timestamp tensor
                if hasattr(t, "values"):
                    timestamp = torch.tensor(
                        [
                            np.datetime64(t.values)
                            .astype("datetime64[ns]")
                            .astype("int64")
                        ]
                    )
                else:
                    timestamp = torch.tensor([hash(str(t)) % (2**31)])

                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    t=timestamp,
                    num_nodes=len(idx2id),
                )

                yield data

            except Exception as e:
                logger.error(f"Error processing time step {t}: {e}")
                continue

        ds.close()


def example_preprocessing_hooks():
    """
    Example preprocessing functions that can be registered as hooks.
    """

    def cap_outlier_flows(ds: xr.Dataset, percentile: float = 99) -> xr.Dataset:
        """Example: Cap extreme flow values at specified percentile for NetCDF data."""
        for var_name in ds.data_vars:
            data = ds[var_name].values
            cap_value = np.percentile(data[~np.isnan(data)], percentile)
            ds[var_name] = ds[var_name].clip(max=cap_value)
        return ds

    def remove_low_population_zones(
        population_data: pd.DataFrame, min_population: int = 100
    ) -> pd.DataFrame:
        """Example: Remove zones with very low population."""
        return population_data[population_data["d.population"] >= min_population]

    def normalize_population_features(data_dict: dict[str, Any]) -> dict[str, Any]:
        """Example: Normalize node features after merging."""
        x = data_dict["x"]
        # Normalize each feature column to [0, 1]
        x_min = x.min(dim=0)[0]
        x_max = x.max(dim=0)[0]
        x_range = x_max - x_min
        x_range[x_range == 0] = 1  # Avoid division by zero
        data_dict["x"] = (x - x_min) / x_range
        return data_dict

    return {
        "cap_outlier_flows": cap_outlier_flows,
        "remove_low_population_zones": remove_low_population_zones,
        "normalize_population_features": normalize_population_features,
    }


if __name__ == "__main__":
    # Example usage
    loader = MobilityDataLoader(
        min_flow_threshold=10,
        normalize_flows=True,
        undirected=False,
        allow_self_loops=False,
    )

    # Register example preprocessing hooks
    hooks = example_preprocessing_hooks()
    loader.register_preprocessing_hook(
        "netcdf_preprocessing", hooks["cap_outlier_flows"]
    )
    loader.register_preprocessing_hook(
        "population_preprocessing", hooks["remove_low_population_zones"]
    )
    loader.register_preprocessing_hook(
        "post_merge_preprocessing", hooks["normalize_population_features"]
    )

    print(
        "MobilityDataLoader initialized with NetCDF streaming and population integration"
    )
    print("Available methods:")
    print(
        "  - create_dataset(netcdf_path, population_path): Create single PyG Data object"
    )
    print(
        "  - stream_dataset(netcdf_path, population_path): Stream PyG Data objects over time"
    )
    print("  - load_population_data(csv_path): Load population CSV data")
