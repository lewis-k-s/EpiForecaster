import numpy as np
import pytest

from data.epi_dataset import EpiDataset
from tests.dataset_test_helpers import (
    make_epi_config,
    make_tiny_epi_dataset,
    write_zarr_dataset,
)


def _make_config(
    dataset_path,
    *,
    history_fraction: float = 1.0,
    horizon_fraction: float = 1.0,
):
    return make_epi_config(
        dataset_path,
        model_options={
            "use_population": True,
            "gnn_module": "",
            "forecaster_head": "transformer",
            "region2vec_path": "",
        },
        data_options={
            "mobility_min_flow_raw": 0.1,
            "window_stride": 1,
            "min_history_observed_fraction": history_fraction,
            "min_horizon_observed_fraction": horizon_fraction,
        },
    )


def _write_tiny_dataset(path) -> None:
    ds = make_tiny_epi_dataset(periods=6)
    ds["cases"].values[0, 4, 0, 0] = np.nan
    ds["cases"].values[0, 1, 1, 0] = np.nan
    ds["cases_mask"].values[0, 4, 0] = 0.0
    ds["cases_mask"].values[0, 1, 1] = 0.0
    ds["cases_age"][:] = 1.0
    ds["hospitalizations_mask"][:] = 0.0
    ds["hospitalizations_age"][:] = 1.0
    ds["deaths_mask"][:] = 0.0
    ds["deaths_age"][:] = 1.0
    ds["edar_biomarker_N1"][:] = 0.0
    ds["edar_biomarker_N1"].values[0, :, 0] = 1.0
    ds["edar_biomarker_N1_mask"][:] = 0.0
    ds["mobility"][:] = 0.0
    ds["population"][:] = [100.0, 200.0]
    write_zarr_dataset(ds, path)


def _write_mask_value_mismatch_dataset(path) -> None:
    ds = make_tiny_epi_dataset(periods=6)
    ds["cases"][:] = 5.0  # finite values emulate interpolation
    ds["cases_mask"].values[0, [1, 3], 0] = 0.0
    ds["hospitalizations_mask"][:] = 0.0
    ds["deaths_mask"][:] = 0.0
    ds["edar_biomarker_N1_mask"][:] = 0.0
    ds["mobility"][:] = 0.0
    ds["population"][:] = 100.0
    write_zarr_dataset(ds, path)


def _write_deaths_sparse_target_dataset(path) -> None:
    ds = make_tiny_epi_dataset(periods=6)
    ds["cases_mask"][:] = 0.0
    ds["hospitalizations_mask"][:] = 0.0
    ds["deaths_mask"].values[0, 3:, 0] = 0.0
    ds["deaths_age"][:] = 1.0
    ds["edar_biomarker_N1_mask"][:] = 0.0
    ds["mobility"][:] = 0.0
    ds["population"][:] = 100.0
    write_zarr_dataset(ds, path)


@pytest.mark.epiforecaster
def test_observed_fractions_allow_half_observed_history_and_horizon(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(zarr_path)

    config = _make_config(
        str(zarr_path),
        history_fraction=0.5,
        horizon_fraction=0.5,
    )
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    assert dataset._valid_window_starts_by_node[0] == [0, 1]
    assert dataset._valid_window_starts_by_node[1] == [0, 1]
    assert dataset.num_windows() == 2
    assert len(dataset) == 4


@pytest.mark.epiforecaster
def test_full_observation_requirement_filters_missing_history_or_horizon(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(zarr_path)

    config = _make_config(str(zarr_path))
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    assert dataset._valid_window_starts_by_node[0] == []
    assert dataset._valid_window_starts_by_node[1] == []
    assert dataset.num_windows() == 0
    assert len(dataset) == 0


@pytest.mark.epiforecaster
def test_window_filtering_uses_cases_mask_not_interpolated_values(tmp_path):
    zarr_path = tmp_path / "mask_value_mismatch.zarr"
    _write_mask_value_mismatch_dataset(zarr_path)

    config = _make_config(
        str(zarr_path),
        history_fraction=1.0,
        horizon_fraction=1.0,
    )
    dataset = EpiDataset(config=config, target_nodes=[0], context_nodes=[0])

    # Two possible starts (0,1). Both should be filtered:
    # - start=0 fails history (t=1 missing)
    # - start=1 fails target (t=3 missing)
    assert dataset._valid_window_starts_by_node[0] == []
    assert dataset.num_windows() == 0

    relaxed = _make_config(
        str(zarr_path),
        history_fraction=0.5,
        horizon_fraction=0.5,
    )
    relaxed_ds = EpiDataset(config=relaxed, target_nodes=[0], context_nodes=[0])
    assert relaxed_ds._valid_window_starts_by_node[0] == [0]
    assert relaxed_ds.num_windows() == 1


@pytest.mark.epiforecaster
def test_uniform_observed_fractions_filter_sparse_daily_target(tmp_path):
    zarr_path = tmp_path / "deaths_sparse_target.zarr"
    _write_deaths_sparse_target_dataset(zarr_path)

    base = _make_config(
        str(zarr_path),
        history_fraction=0.0,
        horizon_fraction=0.0,
    )
    base_ds = EpiDataset(config=base, target_nodes=[0], context_nodes=[0])
    # Zero fractions admit windows even when every target mask is empty.
    assert base_ds._valid_window_starts_by_node[0] == [0, 1]

    half_observed = _make_config(
        str(zarr_path),
        history_fraction=0.5,
        horizon_fraction=0.5,
    )
    half_observed_ds = EpiDataset(
        config=half_observed,
        target_nodes=[0],
        context_nodes=[0],
    )
    assert half_observed_ds._valid_window_starts_by_node[0] == []


def _write_ww_masked_mean_dataset(path) -> None:
    # N1 and N2 partially observed, IP4 fully missing.
    n1 = np.zeros((1, 4, 2), dtype=np.float32)
    n1_mask = np.ones((1, 4, 2), dtype=np.float32)
    n2 = np.zeros((1, 4, 2), dtype=np.float32)
    n2_mask = np.ones((1, 4, 2), dtype=np.float32)

    # Region 0: partial variant availability.
    # Values are already log1p-transformed (from preprocessing pipeline)
    # log1p(10)=2.30, log1p(14)=2.71, log1p(20)=3.04, log1p(22)=3.14, etc.
    n1[0, :, 0] = np.log1p(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    n1_mask[0, :, 0] = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    n2[0, :, 0] = np.log1p(np.array([14.0, 22.0, np.nan, 50.0], dtype=np.float32))
    n2_mask[0, :, 0] = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    # Region 1: fully observed (not used in assertion, keeps dims stable).
    n1[0, :, 1] = np.log1p(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    n2[0, :, 1] = np.log1p(np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32))
    ip4 = np.full((1, 4, 2), 3.0, dtype=np.float32)
    ip4_mask = np.zeros((1, 4, 2), dtype=np.float32)
    zero = np.zeros((1, 4, 2), dtype=np.float32)

    ds = make_tiny_epi_dataset(
        periods=4,
        cases_age=np.ones((1, 4, 2), dtype=np.float32),
        hospitalizations_age=np.ones((1, 4, 2), dtype=np.float32),
        deaths_age=np.ones((1, 4, 2), dtype=np.float32),
        biomarker_values={"N1": n1, "N2": n2, "IP4": ip4},
        biomarker_masks={"N1": n1_mask, "N2": n2_mask, "IP4": ip4_mask},
        biomarker_censors={"N1": zero, "N2": zero, "IP4": zero},
        biomarker_ages={"N1": zero, "N2": zero, "IP4": zero},
        mobility=np.zeros((1, 4, 2, 2), dtype=np.float32),
        population=np.full(2, 100.0, dtype=np.float32),
    )
    write_zarr_dataset(ds, path)


@pytest.mark.epiforecaster
def test_ww_target_uses_biomarker_masked_mean_and_any_variant_mask(tmp_path):
    zarr_path = tmp_path / "ww_masked_mean.zarr"
    _write_ww_masked_mean_dataset(zarr_path)

    config = _make_config(
        str(zarr_path),
        history_fraction=0.0,
        horizon_fraction=0.0,
    )
    dataset = EpiDataset(config=config, target_nodes=[0], context_nodes=[0])

    ww = dataset.precomputed_ww[:, 0].numpy()
    ww_mask = dataset.precomputed_ww_mask[:, 0].numpy()

    # Values are already log1p-transformed in zarr
    # Expected: masked mean of log-transformed values
    # t0: (log1p(10) + log1p(14)) / 2 = (2.30 + 2.64) / 2 = 2.47
    # t1: log1p(22) = 3.14 (only N2 valid)
    # t2: log1p(30) = 3.43 (only N1 valid)
    # t3: log1p(50) = 3.93 (only N2 valid)
    expected_log_mean = np.array(
        [
            (np.log1p(10.0) + np.log1p(14.0)) / 2,  # t0: both valid
            np.log1p(22.0),  # t1: only N2 valid
            np.log1p(30.0),  # t2: only N1 valid
            np.log1p(50.0),  # t3: only N2 valid
        ],
        dtype=np.float32,
    )
    expected_mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # float16 has ~3-4 decimal digits precision, use rtol=1e-3
    np.testing.assert_allclose(ww, expected_log_mean, rtol=1e-2)
    np.testing.assert_allclose(ww_mask, expected_mask, rtol=0.0, atol=0.0)
