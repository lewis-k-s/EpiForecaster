"""
Processor for mobility data from Zarr files.

This module handles the loading of mobility data from Zarr files, enforcing a strict
(time, origin, destination) structure. It prepares the dense origin-destination
matrix for downstream alignment and graph construction.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

from ..config import TEMPORAL_COORD, PreprocessingConfig  # noqa: E402


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
        # Open zarr with chunking on run_id dimension only
        ds = xr.open_zarr(  # type: ignore[arg-type]
            str(mobility_path),
            chunks={"run_id": self.config.run_id_chunk_size},
        )

        # Detect and reconstruct synthetic data format (factorized mobility)
        # Synthetic data stores: mobility_base + mobility_kappa0 instead of full tensor
        # to avoid OOM errors on large datasets. Reconstruct to standard format.
        if "mobility_base" in ds and "mobility_kappa0" in ds:
            print(
                "Detected factorized mobility format (synthetic data). Reconstructing..."
            )
            base = ds["mobility_base"].values  # (origin, target)
            kappa0 = ds[
                "mobility_kappa0"
            ]  # (run_id, date) - keep as DataArray for chunking

            # Reconstruct mobility tensor: mobility[run, date, origin, target]
            # Using the formula: mobility[run, date] = mobility_base * (1 - kappa0[run, date])
            # This is done lazily using xarray's broadcasting to avoid loading all data
            reduction_factor = 1.0 - kappa0  # (run_id, date)

            # Broadcast to (run_id, date, origin, target)
            # reduction_factor[:, :, None, None] adds two new dimensions at the end
            mobility_reconstructed = (
                base[None, None, :, :] * reduction_factor[:, :, None, None]
            )

            ds["mobility"] = xr.DataArray(
                mobility_reconstructed,
                dims=("run_id", "date", "origin", "target"),
                coords={
                    "run_id": kappa0["run_id"].values,
                    "date": kappa0["date"].values,
                    "origin": ds["origin"].values,
                    "target": ds["target"].values,
                },
            )

            print(f"Reconstructed mobility tensor: {ds['mobility'].shape}")

        ds = ds.rename({"target": "destination"})

        # For real (non-synthetic) data, add run_id dimension to match synthetic format
        if "run_id" not in ds.dims:
            ds["mobility"] = ds["mobility"].expand_dims(run_id=["real"])

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

        mobility_dir = Path(mobility_path)  # type: ignore[assignment]

        if not mobility_dir.is_dir():
            raise ValueError(f"Mobility path must be a Zarr directory: {mobility_dir}")

        mobility_data = self._open_dataset(str(mobility_dir))

        # Skip early data quality validation - we'll assess quality at aligned stage

        # stats = self._compute_statistics(od_matrix, time_coords)

        print(
            f"  ✓ Processed mobility matrix: {mobility_data.sizes['origin']} origins x "
            f"{mobility_data.sizes['destination']} destinations x "
            f"{mobility_data.sizes[TEMPORAL_COORD]} time steps"
        )

        return mobility_data
