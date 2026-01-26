import numpy as np
import pandas as pd
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
        missing_permit=0,
        log_scale=log_scale,
        sample_ordering=sample_ordering,
    )
    return EpiForecasterConfig(model=model, data=data_cfg)


def _write_tiny_dataset(zarr_path: str, periods: int = 10) -> None:
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    regions = np.array([0, 1], dtype=np.int64)

    # Use 3D cases to test squeeze
    cases = np.zeros((periods, 2, 1), dtype=np.float32)
    cases[:, 0, 0] = 100.0

    # Non-zero biomarkers for at least one node (zeros excluded from scaler fitting)
    biomarkers = np.zeros((periods, 2), dtype=np.float32)
    biomarkers[:, 0] = 1.0  # Node 0 has non-zero biomarkers

    # Required mask, censor, and age channels for biomarkers
    biomarker_mask = np.ones((periods, 2), dtype=np.float32)
    biomarker_censor = np.zeros((periods, 2), dtype=np.float32)
    biomarker_age = np.zeros((periods, 2), dtype=np.float32)

    # Mobility: full connectivity
    mobility = np.ones((periods, 2, 2), dtype=np.float32)

    population = np.array([1000.0, 1000.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": ((TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "edar_biomarker_N1": ((TEMPORAL_COORD, REGION_COORD), biomarkers),
            "edar_biomarker_N1_mask": ((TEMPORAL_COORD, REGION_COORD), biomarker_mask),
            "edar_biomarker_N1_censor": (
                (TEMPORAL_COORD, REGION_COORD),
                biomarker_censor,
            ),
            "edar_biomarker_N1_age": ((TEMPORAL_COORD, REGION_COORD), biomarker_age),
            "mobility": ((TEMPORAL_COORD, REGION_COORD, "region_id_to"), mobility),
            "population": ((REGION_COORD,), population),
        },
        coords={
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(zarr_path, mode="w")


def test_getitem_values(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(str(zarr_path))

    # Constant cases for easy verification
    # Node 0: 100 cases per day. Pop 1000. -> 10,000 per 100k. Log1p(10000) ~ 9.21
    # Node 1: 0 cases. Pop 1000. -> 0. Log1p(0) = 0.

    config = _make_config(str(zarr_path), log_scale=True)
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
        # Mean should be approx log1p(10000). Std should be epsilon.
        # Normalized values should be approx 0.
        cases_hist = item["case_node"]  # (L, 2) - value, mask

        # Channel 0 (value) should be 0 (normalized constant)
        assert torch.allclose(
            cases_hist[..., 0], torch.zeros_like(cases_hist[..., 0]), atol=1e-3
        )
        # Channel 1 (mask) should be 1 (finite data)
        assert torch.allclose(
            cases_hist[..., 1], torch.ones_like(cases_hist[..., 1]), atol=1e-3
        )

        # Check target scale
        # Should be epsilon
        assert item["target_scale"].item() < 1e-4

        # Check target mean
        # Should be approx log1p(10000) ~ 9.21
        assert abs(item["target_mean"].item() - 9.21) < 0.1

        # Check case mean/std sequences
        assert item["case_mean"].shape == (3, 1)  # history_length
        assert abs(item["case_mean"][-1].item() - 9.21) < 0.1
        assert item["case_std"].shape == (3, 1)
        assert item["case_std"][-1].item() < 1e-4

    elif item["node_label"] == 1:
        # Node 1: 0 cases.
        cases_hist = item["case_node"]
        # Channel 0 (value) should be 0
        assert torch.allclose(
            cases_hist[..., 0], torch.zeros_like(cases_hist[..., 0]), atol=1e-3
        )
        # Channel 1 (mask) should be 1
        assert torch.allclose(
            cases_hist[..., 1], torch.ones_like(cases_hist[..., 1]), atol=1e-3
        )


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


def test_imported_risk_gating(tmp_path):
    """Test that use_imported_risk flag correctly gates lag features.

    Lag features are value-only (no mask/age channels) to reduce dimensionality.
    """
    zarr_path = tmp_path / "risk_test.zarr"
    # Need enough periods for max lag (7) + history (3) + horizon (2)
    _write_tiny_dataset(str(zarr_path), periods=15)

    # Test 1: use_imported_risk=False (default) -> cases_output_dim = 3
    config = _make_config(str(zarr_path))
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset.cases_output_dim == 3

    # Test 2: use_imported_risk=True with lags -> cases_output_dim = 3 + len(lags)
    from copy import deepcopy

    config2 = deepcopy(config)
    config2.data.use_imported_risk = True
    config2.data.mobility_lags = [1, 7]

    dataset2 = EpiDataset(config=config2, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset2.cases_output_dim == 5  # 3 + 2 lags

    # Test 3: Verify item shape matches expected dimension
    item = dataset2[0]
    assert item["case_node"].shape[-1] == 5  # 3 case channels + 2 lag channels
