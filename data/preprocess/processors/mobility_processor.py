"""
Processor for mobility data from Zarr files.

This module handles the loading of mobility data from Zarr files, enforcing a strict
(time, origin, destination) structure. It prepares the dense origin-destination
matrix for downstream alignment and graph construction.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..config import TEMPORAL_COORD, PreprocessingConfig


class MobilityProcessor:
    """
    Loads and validates mobility data from Zarr files.

    This processor handles:
    - Loading Zarr mobility data with chunked streaming
    - Enforcing strict (time, origin, destination) dimensionality
    - Validating coordinate consistency
    - Computing basic statistics on the raw OD matrix

    The output is an xarray Dataset containing the dense origin-destination matrix,
    which is later aligned and converted to graph structures.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the mobility processor.

        Args:
            config: Preprocessing configuration with mobility processing options
        """
        self.config = config

    def _open_dataset(self, mobility_path: str) -> xr.Dataset:
        print("Opening OD dataset...")
        # Open zarr with chunking on temporal dimension
        ds = xr.open_zarr(str(mobility_path), chunks={"date": self.config.chunk_size})
        ds = ds.rename({"target": "destination"})

        assert (ds["origin"].values == ds["destination"].values).all(), (
            "Square OD matrix"
        )
        regions = ds["origin"].values

        print(ds)
        ds = ds.assign_coords(
            {
                TEMPORAL_COORD: ("date", ds["date"].values),
                "origin": ("origin", regions),
                "destination": ("destination", regions),
            }
        )
        print(ds)

        # Filter by temporal range
        start_date = np.datetime64(self.config.start_date)
        end_date = np.datetime64(self.config.end_date)

        time_mask = (ds[TEMPORAL_COORD] >= start_date) & (
            ds[TEMPORAL_COORD] <= end_date
        )
        filtered_ds = ds.isel({TEMPORAL_COORD: time_mask})

        assert not filtered_ds.sizes[TEMPORAL_COORD] == 0, (
            "No data found in temporal range"
        )

        return filtered_ds

    def process(
        self,
        mobility_path: str,
    ) -> xr.Dataset:
        """
        Load and validate mobility data.

        Args:
            mobility_path: Path to Zarr mobility file

        Returns:
            xarray Dataset containing:
            - origin_destination_matrix: [time, origin, destination]
            - attrs: Metadata and statistics
        """
        print(f"Processing mobility data from {mobility_path}")

        mobility_path = Path(mobility_path)

        if not mobility_path.is_dir():
            raise ValueError(f"Mobility path must be a Zarr directory: {mobility_path}")

        mobility_data = self._open_dataset(str(mobility_path))

        # stats = self._compute_statistics(od_matrix, time_coords)

        print(
            f"  ✓ Processed mobility matrix: {mobility_data.sizes['origin']} origins x "
            f"{mobility_data.sizes['destination']} destinations x "
            f"{mobility_data.sizes[TEMPORAL_COORD]} time steps"
        )

        return mobility_data

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
        }

        if time_coords is not None and len(time_coords) > 0:
            stats["time_range"] = {
                "start": str(time_coords[0]),
                "end": str(time_coords[-1]),
            }

        return stats
