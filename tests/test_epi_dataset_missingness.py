import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import DataConfig, EpiForecasterConfig, ModelConfig


def _make_config(
    dataset_path: str,
    missing_permit: dict[str, int] | None = None,
) -> EpiForecasterConfig:
    model = ModelConfig(
        type={"cases": True, "regions": False, "biomarkers": True, "mobility": False},
        # Note: cases_dim and biomarkers_dim now have defaults (2 and 4)
        # that match the dataset output dimensions
        mobility_embedding_dim=1,
        region_embedding_dim=1,
        history_length=3,
        forecast_horizon=2,
        max_neighbors=1,
        gnn_depth=1,
        use_population=True,
        population_dim=1,
        gnn_module="",
        forecaster_head="transformer",
        region2vec_path="",
    )
    data_cfg = DataConfig(
        dataset_path=str(dataset_path),
        mobility_threshold=0.1,
        missing_permit=missing_permit
        or {
            "biomarkers_joint": 0,
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 0,
        },
        window_stride=1,
    )
    return EpiForecasterConfig(model=model, data=data_cfg)


def _write_tiny_dataset(path) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    regions = np.array([0, 1], dtype=np.int64)
    # Use padded run_id to match production data format
    run_id = "real                                            "

    cases = np.ones((1, 6, 2, 1), dtype=np.float32)
    cases[0, 4, 0, 0] = np.nan  # target NaN for node 0
    cases[0, 1, 1, 0] = np.nan  # history NaN for node 1
    cases_mask = np.ones((1, 6, 2), dtype=np.float32)
    cases_mask[0, 4, 0] = 0.0  # mask=0 for NaN
    cases_mask[0, 1, 1] = 0.0  # mask=0 for NaN
    cases_age = np.ones((1, 6, 2), dtype=np.float32)

    # Hospitalizations data (required by ClinicalSeriesPreprocessor)
    hospitalizations = np.ones((1, 6, 2), dtype=np.float32)
    hosp_mask = np.zeros((1, 6, 2), dtype=np.float32)
    hosp_age = np.ones((1, 6, 2), dtype=np.float32)

    # Deaths data (required by ClinicalSeriesPreprocessor)
    deaths = np.ones((1, 6, 2), dtype=np.float32)
    deaths_mask = np.zeros((1, 6, 2), dtype=np.float32)
    deaths_age = np.ones((1, 6, 2), dtype=np.float32)

    # Non-zero biomarkers for at least one node (zeros excluded from scaler fitting)
    biomarkers = np.zeros((1, 6, 2), dtype=np.float32)
    biomarkers[0, :, 0] = 1.0  # Node 0 has non-zero biomarkers

    # Required mask, censor, and age channels for biomarkers
    biomarker_mask = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_censor = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_age = np.zeros((1, 6, 2), dtype=np.float32)

    mobility = np.zeros((1, 6, 2, 2), dtype=np.float32)
    population = np.array([100.0, 200.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
            "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_age),
            "hospitalizations": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hospitalizations,
            ),
            "hospitalizations_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_mask,
            ),
            "hospitalizations_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_age,
            ),
            "deaths": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths),
            "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_mask),
            "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_age),
            "edar_biomarker_N1": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarkers),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_mask,
            ),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_censor,
            ),
            "edar_biomarker_N1_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_age,
            ),
            "mobility": (
                ("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"),
                mobility,
            ),
            "population": ((REGION_COORD,), population),
        },
        coords={
            "run_id": [run_id],
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def _write_mask_value_mismatch_dataset(path) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    regions = np.array([0, 1], dtype=np.int64)
    run_id = "real                                            "

    # Values are fully finite/positive to emulate interpolation.
    cases = np.full((1, 6, 2, 1), 5.0, dtype=np.float32)
    # But the mask marks missing in history (t=1) and target (t=3).
    cases_mask = np.ones((1, 6, 2), dtype=np.float32)
    cases_mask[0, 1, 0] = 0.0
    cases_mask[0, 3, 0] = 0.0
    cases_age = np.zeros((1, 6, 2), dtype=np.float32)

    hospitalizations = np.ones((1, 6, 2), dtype=np.float32)
    hosp_mask = np.zeros((1, 6, 2), dtype=np.float32)
    hosp_age = np.zeros((1, 6, 2), dtype=np.float32)
    deaths = np.ones((1, 6, 2), dtype=np.float32)
    deaths_mask = np.zeros((1, 6, 2), dtype=np.float32)
    deaths_age = np.zeros((1, 6, 2), dtype=np.float32)

    biomarkers = np.ones((1, 6, 2), dtype=np.float32)
    biomarker_mask = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_censor = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_age = np.zeros((1, 6, 2), dtype=np.float32)

    mobility = np.zeros((1, 6, 2, 2), dtype=np.float32)
    population = np.array([100.0, 100.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
            "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_age),
            "hospitalizations": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hospitalizations,
            ),
            "hospitalizations_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_mask,
            ),
            "hospitalizations_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_age,
            ),
            "deaths": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths),
            "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_mask),
            "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_age),
            "edar_biomarker_N1": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarkers),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_mask,
            ),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_censor,
            ),
            "edar_biomarker_N1_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_age,
            ),
            "mobility": (
                ("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"),
                mobility,
            ),
            "population": ((REGION_COORD,), population),
        },
        coords={
            "run_id": [run_id],
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def _write_deaths_sparse_target_dataset(path) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    regions = np.array([0, 1], dtype=np.int64)
    run_id = "real                                            "

    cases = np.ones((1, 6, 2, 1), dtype=np.float32)
    cases_mask = np.zeros((1, 6, 2), dtype=np.float32)
    cases_age = np.zeros((1, 6, 2), dtype=np.float32)

    hospitalizations = np.ones((1, 6, 2), dtype=np.float32)
    hosp_mask = np.zeros((1, 6, 2), dtype=np.float32)
    hosp_age = np.zeros((1, 6, 2), dtype=np.float32)

    deaths = np.ones((1, 6, 2), dtype=np.float32)
    deaths_mask = np.ones((1, 6, 2), dtype=np.float32)
    deaths_mask[0, 3:, 0] = 0.0  # all targets missing for both starts
    deaths_age = np.ones((1, 6, 2), dtype=np.float32)

    biomarkers = np.ones((1, 6, 2), dtype=np.float32)
    biomarker_mask = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_censor = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_age = np.zeros((1, 6, 2), dtype=np.float32)

    mobility = np.zeros((1, 6, 2, 2), dtype=np.float32)
    population = np.array([100.0, 100.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
            "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_age),
            "hospitalizations": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hospitalizations,
            ),
            "hospitalizations_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_mask,
            ),
            "hospitalizations_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_age,
            ),
            "deaths": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths),
            "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_mask),
            "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_age),
            "edar_biomarker_N1": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarkers),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_mask,
            ),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_censor,
            ),
            "edar_biomarker_N1_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_age,
            ),
            "mobility": (
                ("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"),
                mobility,
            ),
            "population": ((REGION_COORD,), population),
        },
        coords={
            "run_id": [run_id],
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


@pytest.mark.epiforecaster
def test_missing_permit_allows_history_nan_but_excludes_target_nan(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(zarr_path)

    config = _make_config(
        str(zarr_path),
        missing_permit={
            "biomarkers_joint": 1,
            "cases": 1,
            "hospitalizations": 1,
            "deaths": 1,
        },
    )
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    assert dataset._valid_window_starts_by_node[0] == [0, 1]
    assert dataset._valid_window_starts_by_node[1] == [0, 1]
    assert dataset.num_windows() == 2
    assert len(dataset) == 4


@pytest.mark.epiforecaster
def test_missing_permit_zero_filters_history_nan(tmp_path):
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
        missing_permit={
            "biomarkers_joint": 0,
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 0,
        },
    )
    dataset = EpiDataset(config=config, target_nodes=[0], context_nodes=[0])

    # Two possible starts (0,1). Both should be filtered:
    # - start=0 fails history (t=1 missing)
    # - start=1 fails target (t=3 missing)
    assert dataset._valid_window_starts_by_node[0] == []
    assert dataset.num_windows() == 0

    relaxed = _make_config(
        str(zarr_path),
        missing_permit={
            "biomarkers_joint": 1,
            "cases": 1,
            "hospitalizations": 1,
            "deaths": 1,
        },
    )
    relaxed_ds = EpiDataset(config=relaxed, target_nodes=[0], context_nodes=[0])
    assert relaxed_ds._valid_window_starts_by_node[0] == [0]
    assert relaxed_ds.num_windows() == 1


@pytest.mark.epiforecaster
def test_per_target_missing_permit_can_filter_deaths(tmp_path):
    zarr_path = tmp_path / "deaths_sparse_target.zarr"
    _write_deaths_sparse_target_dataset(zarr_path)

    base = _make_config(
        str(zarr_path),
        missing_permit={
            "biomarkers_joint": 0,
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 2,
        },
    )
    base_ds = EpiDataset(config=base, target_nodes=[0], context_nodes=[0])
    # With relaxed deaths permit, both starts are available.
    assert base_ds._valid_window_starts_by_node[0] == [0, 1]

    strict_deaths = _make_config(
        str(zarr_path),
        missing_permit={
            "biomarkers_joint": 0,
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 0,
        },
    )
    strict_ds = EpiDataset(config=strict_deaths, target_nodes=[0], context_nodes=[0])
    assert strict_ds._valid_window_starts_by_node[0] == []


def _write_ww_masked_mean_dataset(path) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    regions = np.array([0, 1], dtype=np.int64)
    run_id = "real                                            "

    cases = np.ones((1, 4, 2, 1), dtype=np.float32)
    cases_mask = np.ones((1, 4, 2), dtype=np.float32)
    cases_age = np.ones((1, 4, 2), dtype=np.float32)

    hospitalizations = np.ones((1, 4, 2), dtype=np.float32)
    hosp_mask = np.ones((1, 4, 2), dtype=np.float32)
    hosp_age = np.ones((1, 4, 2), dtype=np.float32)

    deaths = np.ones((1, 4, 2), dtype=np.float32)
    deaths_mask = np.ones((1, 4, 2), dtype=np.float32)
    deaths_age = np.ones((1, 4, 2), dtype=np.float32)

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

    mobility = np.zeros((1, 4, 2, 2), dtype=np.float32)
    population = np.array([100.0, 100.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
            "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_age),
            "hospitalizations": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hospitalizations,
            ),
            "hospitalizations_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_mask,
            ),
            "hospitalizations_age": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                hosp_age,
            ),
            "deaths": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths),
            "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_mask),
            "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_age),
            "edar_biomarker_N1": (("run_id", TEMPORAL_COORD, REGION_COORD), n1),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                n1_mask,
            ),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                zero,
            ),
            "edar_biomarker_N1_age": (("run_id", TEMPORAL_COORD, REGION_COORD), zero),
            "edar_biomarker_N2": (("run_id", TEMPORAL_COORD, REGION_COORD), n2),
            "edar_biomarker_N2_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                n2_mask,
            ),
            "edar_biomarker_N2_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                zero,
            ),
            "edar_biomarker_N2_age": (("run_id", TEMPORAL_COORD, REGION_COORD), zero),
            "edar_biomarker_IP4": (("run_id", TEMPORAL_COORD, REGION_COORD), ip4),
            "edar_biomarker_IP4_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                ip4_mask,
            ),
            "edar_biomarker_IP4_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                zero,
            ),
            "edar_biomarker_IP4_age": (("run_id", TEMPORAL_COORD, REGION_COORD), zero),
            "mobility": (
                ("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"),
                mobility,
            ),
            "population": ((REGION_COORD,), population),
        },
        coords={
            "run_id": [run_id],
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


@pytest.mark.epiforecaster
def test_ww_target_uses_biomarker_masked_mean_and_any_variant_mask(tmp_path):
    zarr_path = tmp_path / "ww_masked_mean.zarr"
    _write_ww_masked_mean_dataset(zarr_path)

    config = _make_config(
        str(zarr_path),
        missing_permit={
            "biomarkers_joint": 4,
            "cases": 4,
            "hospitalizations": 4,
            "deaths": 4,
        },
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
