"""
Processor for mobility data from Zarr, NetCDF, or CSV files.

This module handles the conversion of mobility data into canonical tensor
formats suitable for graph neural network training. It extracts origin-destination
matrices, applies normalization, computes graph connectivity, and generates
node and edge features for the epidemiological forecasting models.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ..config import PreprocessingConfig


class MobilityProcessor:
    """
    Converts mobility data to canonical tensor format.

    This processor handles:
    - Loading Zarr mobility data with efficient chunked streaming (preferred)
    - Loading NetCDF mobility data with chunked streaming
    - Computing graph connectivity from geographic data
    - Extracting node features from flow statistics
    - Normalization of mobility data
    - Edge weight computation based on geographic distance

    The output includes node features (flow statistics), edge connectivity,
    and edge attributes (temporal flow information) that can be assembled
    into EpiBatch objects.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the mobility processor.

        Args:
            config: Preprocessing configuration with mobility processing options
        """
        self.config = config
        self.graph_options = config.graph_options or {}

    def process(
        self,
        mobility_path: str,
        population_data: dict[str, Any] | None = None,
        region_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Process mobility data into canonical tensors.

        Args:
            mobility_path: Path to Zarr mobility file
            population_data: Optional population data for normalization
            region_ids: Optional ordered region IDs to subset/crop the mobility graph

        Returns:
            Dictionary containing processed mobility data:
            - node_features: [num_nodes, feature_dim] tensor with node attributes
            - edge_index: [2, num_edges] tensor with graph connectivity
            - edge_attr: [time, num_edges, edge_dim] tensor with temporal edge features
            - node_coords: [num_nodes, 2] tensor with (lat, lon) coordinates
            - metadata: Processing metadata and statistics
        """
        print(f"Processing mobility data from {mobility_path}")

        mobility_path = Path(mobility_path)

        if mobility_path.is_dir():
            # Process Zarr dataset (preferred) - zarr datasets are directories
            with xr.open_zarr(
                str(mobility_path), chunks={"date": self.config.chunk_size}
            ) as ds:
                mobility_data = self._extract_zarr_data(ds)
                if region_ids is not None:
                    mobility_data = self._subset_zarr_regions(mobility_data, region_ids)
            node_coords = self._extract_coordinates_from_zarr(mobility_data)
        else:
            raise ValueError(f"Unsupported mobility data path: {mobility_path}")

        num_nodes = len(node_coords)
        node_ids: list[str] | None = None
        if isinstance(mobility_data, xr.Dataset) and "origin" in mobility_data.coords:
            node_ids = [str(v).strip() for v in mobility_data["origin"].values]
        elif isinstance(mobility_data, dict):
            mapping = mobility_data.get("municipality_mapping")
            if mapping is not None:
                node_ids = [str(mapping[i]) for i in range(len(mapping))]

        # Extract full origin-destination tensor once
        od_matrix = self._get_od_matrix(mobility_data)

        # Build connectivity directly from observed flows
        edge_index = self._build_od_edge_index(od_matrix)

        # Extract node features from flow statistics
        node_features = self._extract_node_features(od_matrix, population_data)

        # Extract edge features (temporal flow information)
        edge_attr = self._extract_edge_features(od_matrix, edge_index)

        # Apply normalization
        node_features = self._normalize_features(
            node_features, self.config.mobility_normalization
        )
        if edge_attr is not None:
            edge_attr = self._normalize_features(
                edge_attr, self.config.mobility_normalization
            )

        # Create metadata
        time_steps = od_matrix.shape[0]
        time_coords = None
        if isinstance(mobility_data, xr.Dataset) and "time" in mobility_data:
            time_coords = mobility_data["time"].values
        elif isinstance(mobility_data, dict):
            time_coords = mobility_data.get("dates")

        metadata = {
            "num_nodes": num_nodes,
            "num_edges": edge_index.shape[1],
            "feature_dim": node_features.shape[1],
            "edge_dim": edge_attr.shape[2] if edge_attr is not None else 0,
            "time_steps": time_steps,
            "graph_strategy": "origin_destination",
            "normalization": self.config.mobility_normalization,
            "data_stats": self._compute_statistics(od_matrix, time_coords),
        }
        if node_ids is not None:
            metadata["node_ids"] = node_ids

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_coords": node_coords,
            "metadata": metadata,
        }

    def _load_csv_mobility_data(self, mobility_dir: Path) -> dict[str, Any]:
        """
        Load and concatenate daily CSV mobility files.

        Args:
            mobility_dir: Directory containing daily CSV mobility files

        Returns:
            Dictionary with processed mobility data structure
        """
        print(f"Loading daily CSV files from {mobility_dir}")

        # Find all CSV files matching the pattern
        csv_files = sorted(
            [f for f in mobility_dir.glob("*.csv") if "daily_mobility" in f.name]
        )

        if not csv_files:
            raise ValueError(f"No daily mobility CSV files found in {mobility_dir}")

        print(f"Found {len(csv_files)} daily mobility files")

        # Load all CSV files
        mobility_dataframes = []
        dates = []

        for csv_file in csv_files:
            # Extract date from filename
            date_str = csv_file.name.split(".")[0]  # Extract YYYY-MM-DD
            date = pd.to_datetime(date_str)

            # Check if date is within desired range
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)

            if start_date <= date <= end_date:
                df = pd.read_csv(csv_file)
                df["date"] = date  # Add date column
                mobility_dataframes.append(df)
                dates.append(date)

        if not mobility_dataframes:
            raise ValueError(
                f"No data found in temporal range {self.config.start_date} to {self.config.end_date}"
            )

        # Concatenate all dataframes
        mobility_df = pd.concat(mobility_dataframes, ignore_index=True)

        # Sort by date
        mobility_df = mobility_df.sort_values("date")

        print(
            f"Loaded {len(mobility_df)} records from {len(mobility_dataframes)} files"
        )
        print(f"Date range: {mobility_df['date'].min()} to {mobility_df['date'].max()}")

        return {
            "data": mobility_df,
            "time_steps": len(mobility_dataframes),
            "dates": dates,
            "origin_destination_matrix": None,  # Will be created on demand
        }

    def _extract_coordinates_from_csv(
        self, mobility_data: dict[str, Any]
    ) -> torch.Tensor:
        """
        Extract node coordinates from mobility data (create dummy coordinates).

        Args:
            mobility_data: Dictionary with CSV mobility data

        Returns:
            [num_nodes, 2] tensor with (lat, lon) coordinates
        """
        # For now, create dummy coordinates since we don't have real geographic data
        # In a real implementation, you would load actual municipality coordinates
        df = mobility_data["data"]

        # Get unique municipality codes (excluding 'origin_code' column)
        municipality_cols = [
            col for col in df.columns if col != "origin_code" and col != "date"
        ]

        # Create dummy coordinates (will be replaced with real coordinates if available)
        num_nodes = len(municipality_cols)

        # Create grid-based dummy coordinates (for visualization purposes)
        # In a real implementation, these would be actual lat/lon coordinates
        lat_coords = torch.linspace(
            40.0, 42.5, num_nodes
        )  # Approximate Catalonia latitude range
        lon_coords = torch.linspace(
            0.0, 3.0, num_nodes
        )  # Approximate Catalonia longitude range

        # Create mapping from column index to municipality code
        municipality_mapping = dict(enumerate(municipality_cols))

        node_coords = torch.column_stack([lat_coords, lon_coords])

        # Store municipality mapping for later use
        mobility_data["municipality_mapping"] = municipality_mapping
        mobility_data["municipality_coords"] = node_coords

        return node_coords

    def _extract_mobility_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extract and validate mobility data from NetCDF.

        Args:
            ds: Xarray Dataset from NetCDF file

        Returns:
            Processed mobility data subset
        """
        # Validate required variables
        required_vars = ["origin_destination_matrix", "time", "lat", "lon"]
        for var in required_vars:
            if var not in ds:
                raise ValueError(f"Required variable '{var}' not found in NetCDF file")

        # Filter by temporal range
        start_date = np.datetime64(self.config.start_date)
        end_date = np.datetime64(self.config.end_date)

        time_mask = (ds.time >= start_date) & (ds.time <= end_date)
        filtered_ds = ds.isel(time=time_mask)

        if len(filtered_ds.time) == 0:
            raise ValueError(
                f"No data found in temporal range {self.config.start_date} to {self.config.end_date}"
            )

        return filtered_ds

    def _extract_zarr_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extract and validate mobility data from Zarr format.

        Args:
            ds: Xarray Dataset from Zarr file

        Returns:
            Processed mobility data subset
        """
        data_var = None
        if "trips" in ds:
            data_var = "trips"
        elif "mobility" in ds:
            data_var = "mobility"
        else:
            raise ValueError("Zarr file must contain 'trips' or 'mobility' variable")

        rename_dims = {}
        if "date" in ds.dims:
            rename_dims["date"] = "time"
        if "source" in ds.dims:
            rename_dims["source"] = "origin"
        if "destination" in ds.dims:
            rename_dims["destination"] = "target"

        filtered_ds = ds.rename(rename_dims)

        if "origin" not in filtered_ds.dims or "target" not in filtered_ds.dims:
            raise ValueError(
                "Zarr dataset must contain 'origin' and 'target' dimensions"
            )

        # Filter by temporal range using time coordinate
        if "time" not in filtered_ds.dims:
            raise ValueError("Zarr dataset must contain a temporal dimension 'time'")

        start_date = np.datetime64(self.config.start_date)
        end_date = np.datetime64(self.config.end_date)

        time_mask = (filtered_ds.time >= start_date) & (filtered_ds.time <= end_date)
        filtered_ds = filtered_ds.isel(time=time_mask)

        if len(filtered_ds.time) == 0:
            raise ValueError(
                f"No data found in temporal range {self.config.start_date} to {self.config.end_date}"
            )

        # Rename data variable and coordinates for compatibility
        filtered_ds = filtered_ds.rename({data_var: "origin_destination_matrix"})

        # Create dummy lat/lon coordinates if not present (for compatibility)
        if "lat" not in filtered_ds.coords:
            num_origins = len(filtered_ds.origin)
            # Create dummy coordinates for Catalonia region
            filtered_ds = filtered_ds.assign_coords(
                {
                    "lat": (
                        "origin",
                        np.linspace(40.0, 42.5, num_origins),
                    ),  # Catalonia latitude range
                    "lon": (
                        "origin",
                        np.linspace(0.0, 3.0, num_origins),
                    ),  # Catalonia longitude range
                }
            )

        return filtered_ds

    def _subset_zarr_regions(
        self, mobility_data: xr.Dataset, region_ids: Sequence[str]
    ) -> xr.Dataset:
        """Subset mobility dataset to the provided region IDs (ordered)."""
        if not region_ids:
            return mobility_data

        region_ids_str = [str(r).strip().zfill(5) for r in region_ids]
        available_ids = {str(v).strip() for v in mobility_data.origin.values}
        missing = [rid for rid in region_ids_str if rid not in available_ids]
        if missing:
            raise ValueError(
                "Mobility dataset is missing regions required by cases data: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
            )

        subset = mobility_data.sel(origin=region_ids_str, target=region_ids_str)
        return subset

    def _extract_coordinates(self, mobility_data: xr.Dataset) -> torch.Tensor:
        """
        Extract node coordinates from mobility data.

        Args:
            mobility_data: Xarray Dataset with coordinate data

        Returns:
            [num_nodes, 2] tensor with (lat, lon) coordinates
        """
        lat = mobility_data["lat"].values
        lon = mobility_data["lon"].values

        # Convert to tensor
        node_coords = torch.from_numpy(np.column_stack([lat, lon])).float()

        return node_coords

    def _extract_coordinates_from_zarr(self, mobility_data: xr.Dataset) -> torch.Tensor:
        """
        Extract node coordinates from Zarr mobility data.

        Args:
            mobility_data: Xarray Dataset with coordinate data

        Returns:
            [num_nodes, 2] tensor with (lat, lon) coordinates
        """
        lat = mobility_data["lat"].values
        lon = mobility_data["lon"].values

        # Convert to tensor
        node_coords = torch.from_numpy(np.column_stack([lat, lon])).float()

        return node_coords

    def _get_od_matrix(self, mobility_data: Any) -> np.ndarray:
        """Return the origin-destination tensor with shape [time, N, N]."""
        if isinstance(mobility_data, xr.Dataset):
            return mobility_data["origin_destination_matrix"].values
        if isinstance(mobility_data, dict):
            return self._create_od_matrix_from_csv(mobility_data)
        raise TypeError("Unsupported mobility data container for OD extraction")

    def _build_od_edge_index(self, od_matrix: np.ndarray) -> torch.Tensor:
        """Create directed edges from aggregated origin-destination flows."""
        if od_matrix.ndim != 3:
            raise ValueError(
                "Expected origin-destination tensor with shape [time, N, N]"
            )

        aggregated_flows = np.sum(od_matrix, axis=0)
        src_idx, dst_idx = np.nonzero(aggregated_flows)

        if src_idx.size == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        edge_index = torch.tensor(
            np.stack([src_idx, dst_idx]), dtype=torch.long
        ).contiguous()

        include_self_loops = self.graph_options.get("include_self_loops", False)
        if not include_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]

        return edge_index

    def _extract_node_features(
        self, od_matrix: np.ndarray, population_data: dict[str, Any] | None
    ) -> torch.Tensor:
        """
        Extract node features from mobility flow statistics.

        Args:
            od_matrix: Tensor with shape [time, N, N]
            population_data: Optional population data for normalization

        Returns:
            [num_nodes, feature_dim] tensor with node features
        """
        # Compute flow statistics for each node
        outgoing_flow = np.sum(od_matrix, axis=2)  # [time, N]
        incoming_flow = np.sum(od_matrix, axis=1)  # [time, N]
        total_flow = outgoing_flow + incoming_flow  # [time, N]

        # Compute temporal statistics
        mean_flow = np.mean(total_flow, axis=0)  # [N]
        std_flow = np.std(total_flow, axis=0)  # [N]
        max_flow = np.max(total_flow, axis=0)  # [N]
        min_flow = np.min(total_flow, axis=0)  # [N]

        # Compute flow ratios
        outgoing_ratio = np.mean(outgoing_flow / (total_flow + 1e-8), axis=0)  # [N]
        incoming_ratio = np.mean(incoming_flow / (total_flow + 1e-8), axis=0)  # [N]

        # Assemble node features
        node_features = np.column_stack(
            [mean_flow, std_flow, max_flow, min_flow, outgoing_ratio, incoming_ratio]
        )  # [N, 6]

        # Add population features if available
        if population_data is not None:
            if "population" in population_data:
                population = np.array(population_data["population"])
                if len(population) == node_features.shape[0]:
                    # Normalize population and add as feature
                    pop_normalized = population / np.max(population)
                    node_features = np.column_stack([node_features, pop_normalized])

        return torch.from_numpy(node_features).float()

    def _create_od_matrix_from_csv(self, mobility_data: dict[str, Any]) -> np.ndarray:
        """
        Create origin-destination matrix from CSV mobility data.

        Args:
            mobility_data: Dictionary with CSV mobility data

        Returns:
            [time, N, N] numpy array with OD matrices
        """
        df = mobility_data["data"]
        dates = sorted(df["date"].unique())

        # Get municipality columns (excluding 'origin_code' and 'date')
        municipality_cols = [
            col for col in df.columns if col != "origin_code" and col != "date"
        ]
        num_nodes = len(municipality_cols)

        # Create OD matrix for each time step
        od_matrices = []

        for date in dates:
            daily_df = df[df["date"] == date]

            # Initialize empty OD matrix
            od_matrix = np.zeros((num_nodes, num_nodes))

            # Fill OD matrix
            for _, row in daily_df.iterrows():
                origin_code = row["origin_code"]

                # Find origin index
                if origin_code in municipality_cols:
                    origin_idx = municipality_cols.index(origin_code)

                    # Fill flows to all destinations
                    for j, dest_code in enumerate(municipality_cols):
                        if dest_code in row and not pd.isna(row[dest_code]):
                            od_matrix[origin_idx, j] = row[dest_code]

            od_matrices.append(od_matrix)

        return np.array(od_matrices)  # [time, N, N]

    def _extract_edge_features(
        self,
        od_matrix: np.ndarray,
        edge_index: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Extract edge features from temporal flow data.

        Args:
            od_matrix: Tensor with shape [time, N, N]
            edge_index: [2, num_edges] tensor with edge connectivity
            node_coords: Optional coordinates for distance-based features

        Returns:
            [time, num_edges, edge_dim] tensor with edge features, or None
        """
        time_steps = od_matrix.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return None

        # Extract flow values for each edge across time
        edge_flows = []
        for t in range(time_steps):
            t_flows = []
            for i in range(num_edges):
                src, dst = edge_index[:, i]
                flow = od_matrix[t, src, dst]
                t_flows.append(flow)
            edge_flows.append(t_flows)

        edge_flows = np.array(edge_flows)  # [time, num_edges]

        # Only flows as feature for now
        edge_features = edge_flows[..., np.newaxis]  # [time, num_edges, 1]

        return torch.from_numpy(edge_features).float()

    def _normalize_features(
        self, features: torch.Tensor, normalization: str
    ) -> torch.Tensor:
        """
        Apply normalization to features.

        Args:
            features: Input features tensor
            normalization: Normalization method

        Returns:
            Normalized features tensor
        """
        if normalization == "none":
            return features
        elif normalization == "log1p":
            return torch.log1p(torch.clamp(features, min=0))
        elif normalization == "standard":
            mean = features.mean()
            std = features.std()
            return (features - mean) / (std + 1e-8)
        elif normalization == "minmax":
            min_val = features.min()
            max_val = features.max()
            return (features - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    def _compute_statistics(
        self, od_matrix: np.ndarray, time_coords: Sequence[Any] | None
    ) -> dict[str, Any]:
        """Compute dataset statistics for metadata."""

        stats = {
            "total_flows": float(np.sum(od_matrix)),
            "mean_flow": float(np.mean(od_matrix)),
            "std_flow": float(np.std(od_matrix)),
            "max_flow": float(np.max(od_matrix)),
            "min_flow": float(np.min(od_matrix)),
            "num_time_steps": int(od_matrix.shape[0]),
        }

        if time_coords is not None and len(time_coords) > 0:
            stats["time_range"] = {
                "start": str(time_coords[0]),
                "end": str(time_coords[-1]),
            }

        return stats
