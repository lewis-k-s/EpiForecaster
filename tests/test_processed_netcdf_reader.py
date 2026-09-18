"""Real-file coverage for NetCDF4 input decoding and repeated lazy opens."""

from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.epi_dataset import EpiDataset, open_processed_dataset


@pytest.fixture
def netcdf_dataset(tmp_path: Path) -> Path:
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    values[0, 1, 0] = np.nan
    dataset = xr.Dataset(
        {
            "cases": (("run_id", "date", "region_id"), values),
            "cases_mask": (
                ("run_id", "date", "region_id"),
                np.isfinite(values).astype(np.int8),
            ),
            "valid_targets": (
                ("run_id", "region_id"),
                np.array([[1, 0], [1, 1]], dtype=np.int8),
            ),
        },
        coords={
            "run_id": ["real", "other"],
            "date": pd.date_range("2020-05-01", periods=3),
            "region_id": ["01001", "01002"],
        },
        attrs={"preprocessing_config_yaml": "smoothing: {}\n", "log_transformed": 1},
    )
    path = tmp_path / "processed.nc"
    dataset.to_netcdf(
        path,
        engine="netcdf4",
        encoding={"cases": {"zlib": True, "chunksizes": (1, 2, 2)}},
    )
    return path


def test_netcdf_reader_preserves_values_coordinates_masks_and_attrs(netcdf_dataset):
    with xr.open_dataset(netcdf_dataset, engine="netcdf4") as reference:
        expected = reference.load()
    with open_processed_dataset(netcdf_dataset, chunks={"run_id": 1}) as actual:
        xr.testing.assert_identical(actual.compute(), expected)


def test_reopened_canonical_reader_survives_sibling_close(netcdf_dataset):
    with xr.open_dataset(netcdf_dataset, engine="netcdf4") as reference:
        expected = reference.sel(run_id="real", drop=True).load()
    with dask.config.set(num_workers=4):
        for _ in range(3):
            parent = EpiDataset.load_canonical_dataset(netcdf_dataset, run_id="real")
            sibling = EpiDataset.load_canonical_dataset(netcdf_dataset, run_id="real")
            try:
                np.testing.assert_array_equal(sibling.valid_targets.values, [1, 0])
                sibling.close()
                xr.testing.assert_identical(parent.compute(), expected)
            finally:
                sibling.close()
                parent.close()
