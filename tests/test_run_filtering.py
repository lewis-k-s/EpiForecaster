"""Test run_id filtering logic for EpiDataset."""

import numpy as np
import xarray as xr
from data.epi_dataset import EpiDataset


def create_mock_dataset(run_ids, time_steps=10, regions=5):
    """Create a mock dataset for testing."""
    time_coords = np.arange(time_steps)
    region_coords = np.arange(regions)

    # Cases: (time, region) or (run_id, time, region)
    cases_data = np.random.rand(len(run_ids), time_steps, regions)

    # Mobility: (run_id, time, origin, destination)
    mobility_data = np.random.rand(len(run_ids), time_steps, regions, regions)

    # Valid targets: (run_id, region)
    valid_targets_data = np.random.choice([True, False], size=(len(run_ids), regions))

    # Build dataset
    ds = xr.Dataset(
        {
            "cases": (["run_id", "time", "region"], cases_data),
            "mobility": (["run_id", "time", "origin", "destination"], mobility_data),
            "valid_targets": (["run_id", "region"], valid_targets_data),
            "population": (["region"], np.random.rand(regions)),
        },
        coords={
            "run_id": run_ids,
            "time": time_coords,
            "region": region_coords,
            "origin": region_coords,
            "destination": region_coords,
        },
    )

    return ds


def test_filter_dataset_by_runs():
    """Test run_id filtering works correctly."""
    run_ids = [
        "0_Baseline                                        ",
        "1_Global_Timed_s05                                ",
    ]
    ds = create_mock_dataset(run_ids)

    # Test filtering to first run
    filtered = EpiDataset._filter_dataset_by_runs(ds, "0_Baseline")

    # Check run_id dimension is squeezed
    assert "run_id" not in filtered.dims, "run_id should be squeezed after filtering"
    assert filtered.dims["time"] == 10, "time dimension should remain"
    assert filtered.dims["region"] == 5, "region dimension should remain"

    # Check valid_targets aggregation works
    valid_targets = filtered.valid_targets
    assert valid_targets.ndim == 1, (
        f"valid_targets should be 1D, got {valid_targets.dims}"
    )
    assert valid_targets.shape == (5,), (
        f"valid_targets shape should be (5,), got {valid_targets.shape}"
    )

    print("✓ _filter_dataset_by_runs test passed")


def test_filter_with_none():
    """Test that None returns original dataset."""
    run_ids = [
        "0_Baseline                                        ",
        "1_Global_Timed_s05                                ",
    ]
    ds = create_mock_dataset(run_ids)

    filtered = EpiDataset._filter_dataset_by_runs(ds, None)

    assert filtered is ds, "None should return the original dataset"
    print("✓ filter with None test passed")


def test_valid_targets_aggregation():
    """Test valid_targets aggregation across run_id dimension."""
    run_ids = ["run_0", "run_1"]
    time_steps = 5
    regions = 3

    # Create dataset where region 0 is valid in run_0 but not run_1
    # and region 1 is valid in run_1 but not run_0
    cases_data = np.random.rand(len(run_ids), time_steps, regions)
    mobility_data = np.random.rand(len(run_ids), time_steps, regions, regions)
    valid_targets_data = np.array(
        [
            [True, False, True],
            [False, True, True],
        ]
    )

    ds = xr.Dataset(
        {
            "cases": (["run_id", "time", "region"], cases_data),
            "mobility": (["run_id", "time", "origin", "destination"], mobility_data),
            "valid_targets": (["run_id", "region"], valid_targets_data),
        },
        coords={
            "run_id": run_ids,
            "time": np.arange(time_steps),
            "region": np.arange(regions),
            "origin": np.arange(regions),
            "destination": np.arange(regions),
        },
    )

    # Filter to run_0
    filtered = EpiDataset._filter_dataset_by_runs(ds, "run_0")

    # Check valid_targets is aggregated to 1D
    valid_mask = filtered.valid_targets.values.astype(bool)
    assert valid_mask.ndim == 1, f"valid_mask should be 1D, got {valid_mask.ndim}"
    assert valid_mask.shape == (regions,), (
        f"valid_mask shape should be ({regions},), got {valid_mask.shape}"
    )

    # Region 0 should be valid, region 1 should not
    assert valid_mask[0], "Region 0 should be valid"
    assert not valid_mask[1], "Region 1 should not be valid"

    print("✓ valid_targets aggregation test passed")


def test_whitespace_handling():
    """Test that whitespace-padded run_ids are matched correctly."""
    # Create run_ids with padding (like in real dataset)
    run_ids = [
        "0_Baseline                                        ",
        "1_Global_Timed_s05                                ",
    ]
    ds = create_mock_dataset(run_ids)

    # Filter with trimmed string
    filtered = EpiDataset._filter_dataset_by_runs(ds, "0_Baseline")

    # Should successfully filter
    assert "run_id" not in filtered.dims, "run_id should be squeezed"
    assert filtered.dims["time"] == 10, "time dimension should remain"

    print("✓ whitespace handling test passed")


if __name__ == "__main__":
    test_filter_dataset_by_runs()
    test_filter_with_none()
    test_valid_targets_aggregation()
    test_whitespace_handling()
    print("\n✓ All run filtering tests passed!")
