import numpy as np
import pandas as pd
import xarray as xr

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import DataConfig, EpiForecasterConfig, ModelConfig


def _make_config(dataset_path: str, missing_permit: int = 0) -> EpiForecasterConfig:
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
        missing_permit=missing_permit,
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

    # Non-zero biomarkers for at least one node (zeros excluded from scaler fitting)
    biomarkers = np.zeros((1, 6, 2), dtype=np.float32)
    biomarkers[0, :, 0] = 1.0  # Node 0 has non-zero biomarkers

    # Required mask, censor, and age channels for biomarkers
    biomarker_mask = np.ones((1, 6, 2), dtype=np.float32)
    biomarker_censor = np.zeros((1, 6, 2), dtype=np.float32)
    biomarker_age = np.zeros((1, 6, 2), dtype=np.float32)

    mobility = np.zeros((1, 6, 2, 2), dtype=np.float32)
    population = np.array([100.0, 200.0], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
            "edar_biomarker_N1": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarkers),
            "edar_biomarker_N1_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarker_mask),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                biomarker_censor,
            ),
            "edar_biomarker_N1_age": (("run_id", TEMPORAL_COORD, REGION_COORD), biomarker_age),
            "mobility": (("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"), mobility),
            "population": ((REGION_COORD,), population),
        },
        coords={
            "run_id": [run_id],
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )
    ds.to_zarr(path, mode="w")


def test_missing_permit_allows_history_nan_but_excludes_target_nan(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(zarr_path)

    config = _make_config(str(zarr_path), missing_permit=1)
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    assert dataset._valid_window_starts_by_node[0] == []
    assert dataset._valid_window_starts_by_node[1] == [0, 1]
    assert dataset.num_windows() == 2
    assert len(dataset) == 2


def test_missing_permit_zero_filters_history_nan(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(zarr_path)

    config = _make_config(str(zarr_path), missing_permit=0)
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    assert dataset._valid_window_starts_by_node[0] == []
    assert dataset._valid_window_starts_by_node[1] == []
    assert dataset.num_windows() == 0
    assert len(dataset) == 0
