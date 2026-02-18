import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr


from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import DataConfig, EpiForecasterConfig, ModelConfig


def _make_config(
    dataset_path: str,
    log_scale: bool = False,
    sample_ordering: str = "node",
) -> EpiForecasterConfig:
    # Minimal config
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
        population_dim=1,
    )
    data_cfg = DataConfig(
        dataset_path=str(dataset_path),
        mobility_threshold=0.0,
        missing_permit={
            "biomarkers_joint": 0,
            "cases": 0,
            "hospitalizations": 0,
            "deaths": 0,
        },
        log_scale=log_scale,
        sample_ordering=sample_ordering,
    )
    return EpiForecasterConfig(model=model, data=data_cfg)


def _write_tiny_dataset(zarr_path: str, periods: int = 10) -> None:
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    regions = np.array([0, 1], dtype=np.int64)
    # Use padded run_id to match production data format
    run_id = "real                                            "

    # Use 3D cases to test squeeze - add run_id dimension
    # Both nodes need non-zero cases to be valid (all-zero sequences are filtered)
    cases = np.zeros((1, periods, 2, 1), dtype=np.float32)
    cases[0, :, 0, 0] = 100.0  # Node 0
    cases[0, :, 1, 0] = 50.0  # Node 1

    # Required mask and age channels for cases (now required by ClinicalSeriesPreprocessor)
    cases_mask = np.ones((1, periods, 2), dtype=np.float32)
    cases_age = np.ones(
        (1, periods, 2), dtype=np.float32
    )  # Age = 1 (fresh observations)

    # Hospitalizations data (required by ClinicalSeriesPreprocessor)
    hospitalizations = np.zeros((1, periods, 2), dtype=np.float32)
    hospitalizations[0, :, 0] = 10.0  # Node 0
    hospitalizations[0, :, 1] = 5.0  # Node 1
    hosp_mask = np.ones((1, periods, 2), dtype=np.float32)
    hosp_age = np.ones((1, periods, 2), dtype=np.float32)

    # Deaths data (required by ClinicalSeriesPreprocessor)
    deaths = np.zeros((1, periods, 2), dtype=np.float32)
    deaths[0, :, 0] = 1.0  # Node 0
    deaths[0, :, 1] = 0.5  # Node 1
    deaths_mask = np.ones((1, periods, 2), dtype=np.float32)
    deaths_age = np.ones((1, periods, 2), dtype=np.float32)

    # Non-zero biomarkers for at least one node (zeros excluded from scaler fitting)
    biomarkers = np.zeros((1, periods, 2), dtype=np.float32)
    biomarkers[0, :, 0] = 1.0  # Node 0 has non-zero biomarkers

    # Required mask, censor, and age channels for biomarkers
    biomarker_mask = np.ones((1, periods, 2), dtype=np.float32)
    biomarker_censor = np.zeros((1, periods, 2), dtype=np.float32)
    biomarker_age = np.zeros((1, periods, 2), dtype=np.float32)

    # Mobility: full connectivity - add run_id dimension
    mobility = np.ones((1, periods, 2, 2), dtype=np.float32)

    population = np.array([1000.0, 1000.0], dtype=np.float32)

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
    ds.to_zarr(zarr_path, mode="w", zarr_format=2)


@pytest.mark.epiforecaster
def test_getitem_preserves_target_history_when_self_edge_missing(tmp_path):
    """Test that target history is preserved even when self-mobility is below threshold.

    This exposes a bug where target inclusion is forced on neigh_mask, but masking
    is applied with mobility_mask_float instead, causing target history to be zeroed
    when self-edges are missing.
    """
    zarr_path = tmp_path / "self_edge_missing.zarr"
    _write_tiny_dataset(str(zarr_path), periods=8)

    config = _make_config(str(zarr_path), log_scale=False)
    config.data.mobility_threshold = 0.5  # require positive flow
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    # Remove all self-mobility so target self-edge is below threshold
    dataset.preloaded_mobility[:, 0, 0] = 0.0
    dataset.preloaded_mobility[:, 1, 1] = 0.0
    dataset.mobility_mask = dataset.preloaded_mobility >= config.data.mobility_threshold
    dataset.mobility_mask_float = dataset.mobility_mask.to(torch.float32)

    item = dataset[0]  # target node 0, first window
    cases_hist = item["cases_hist"]  # (L, 3), channels [value, mask, age]

    # If target inclusion is honored, target mask should remain observed
    assert torch.all(cases_hist[:, 1] > 0.5), (
        "target history mask got zeroed by mobility masking"
    )


@pytest.mark.epiforecaster
def test_getitem_values(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(str(zarr_path))

    # Constant cases for easy verification
    # Node 0: 100 cases per day. Pop 1000. -> 10,000 per 100k.
    # Node 1: 50 cases per day. Pop 1000. -> 5,000 per 100k.
    # Note: Log scaling is now done in offline preprocessor, so dataset receives raw values

    config = _make_config(str(zarr_path), log_scale=False)
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    # Check item 0 (window start 0)
    # History: 0, 1, 2. Future: 3, 4.

    item = dataset[0]  # Should be node 0 (based on sorted node iteration usually)

    # Check node label to be sure
    # EpiDataset sorts target_nodes?
    # __init__: self.target_nodes = target_nodes (list)
    # _build_index_map: iterates self.target_nodes.

    if item["node_label"] == 0:
        # Node 0: constant cases.
        cases_hist = item["cases_hist"]  # (L, 3) - value, mask, age

        # Channel 0 (value) should be 100.0 (raw cases value from dataset)
        expected_value = 100.0
        assert torch.allclose(
            cases_hist[..., 0],
            torch.full_like(cases_hist[..., 0], expected_value),
            atol=1e-3,
        )
        # Channel 1 (mask) should be 1 (finite data)
        assert torch.allclose(
            cases_hist[..., 1], torch.ones_like(cases_hist[..., 1]), atol=1e-3
        )
        # Channel 2 (age) should be 1/14 ~ 0.0714 (normalized by age_max=14)
        expected_age = 1.0 / 14.0
        assert torch.allclose(
            cases_hist[..., 2],
            torch.full_like(cases_hist[..., 2], expected_age),
            atol=1e-3,
        )

    elif item["node_label"] == 1:
        # Node 1: 50 cases.
        cases_hist = item["cases_hist"]
        # Channel 0 (value) should be 50.0 (raw cases value)
        expected_value = 50.0
        assert torch.allclose(
            cases_hist[..., 0],
            torch.full_like(cases_hist[..., 0], expected_value),
            atol=1e-3,
        )
        # Channel 1 (mask) should be 1
        assert torch.allclose(
            cases_hist[..., 1], torch.ones_like(cases_hist[..., 1]), atol=1e-3
        )
        # Channel 2 (age) should be 1/14 ~ 0.0714 (normalized by age_max=14)
        expected_age = 1.0 / 14.0
        assert torch.allclose(
            cases_hist[..., 2],
            torch.full_like(cases_hist[..., 2], expected_age),
            atol=1e-3,
        )


@pytest.mark.epiforecaster
def test_index_ordering_time_major(tmp_path):
    zarr_path = tmp_path / "ordering.zarr"
    _write_tiny_dataset(str(zarr_path), periods=8)

    node_config = _make_config(str(zarr_path), sample_ordering="node")
    node_dataset = EpiDataset(
        config=node_config, target_nodes=[0, 1], context_nodes=[0, 1]
    )

    time_config = _make_config(str(zarr_path), sample_ordering="time")
    time_dataset = EpiDataset(
        config=time_config, target_nodes=[0, 1], context_nodes=[0, 1]
    )

    assert node_dataset._index_map[:4] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    assert time_dataset._index_map[:4] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]

    assert node_dataset[1]["window_start"] == 1
    assert time_dataset[1]["window_start"] == 0


@pytest.mark.epiforecaster
def test_imported_risk_gating(tmp_path):
    """Test that use_imported_risk flag correctly gates lag features.

    Lag features are value-only (no mask/age channels) to reduce dimensionality.
    """
    zarr_path = tmp_path / "risk_test.zarr"
    # Need enough periods for max lag (7) + history (3) + horizon (2)
    _write_tiny_dataset(str(zarr_path), periods=15)

    # Test 1: use_imported_risk=False (default) -> cases_output_dim = 9 (3 series x 3 channels)
    config = _make_config(str(zarr_path))
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset.cases_output_dim == 9  # 3 series x 3 channels

    # Test 2: use_imported_risk=True with lags -> cases_output_dim = 9 + len(lags)
    from copy import deepcopy

    config2 = deepcopy(config)
    config2.data.use_imported_risk = True
    config2.data.mobility_lags = [1, 7]

    dataset2 = EpiDataset(config=config2, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset2.cases_output_dim == 11  # 9 + 2 lags

    # Test 3: Verify item shape matches expected dimension
    item = dataset2[0]
    # Only cases_hist has lag features concatenated
    # hosp_hist and deaths_hist remain 3 channels (value, mask, age)
    assert item["hosp_hist"].shape[-1] == 3  # 3 channels, no lags
    assert item["deaths_hist"].shape[-1] == 3  # 3 channels, no lags
    assert item["cases_hist"].shape[-1] == 5  # 3 case channels + 2 lag channels
