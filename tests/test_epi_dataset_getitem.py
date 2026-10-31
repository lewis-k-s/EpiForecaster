import numpy as np
import pytest
import torch
from types import SimpleNamespace

from data.epi_dataset import EpiDataset
from data.epi_batch import collate_epiforecaster_batch
from data.region_embedding_store import RegionEmbeddingStore
from tests.dataset_test_helpers import (
    make_epi_config,
    make_tiny_epi_dataset,
    write_zarr_dataset,
)
from utils.training_utils import inject_gpu_mobility


def _make_config(dataset_path, sample_ordering="node"):
    return make_epi_config(
        dataset_path,
        data_options={"sample_ordering": sample_ordering},
    )


def _write_tiny_dataset(
    zarr_path: str,
    periods: int = 10,
    mobility_time_mask: np.ndarray | None = None,
) -> None:
    if mobility_time_mask is None:
        mobility_time_mask = np.ones((1, periods), dtype=bool)
    ds = make_tiny_epi_dataset(
        periods=periods,
        cases=np.broadcast_to(
            np.array([100.0, 50.0], dtype=np.float32).reshape(1, 1, 2, 1),
            (1, periods, 2, 1),
        ).copy(),
        cases_age=np.ones((1, periods, 2), dtype=np.float32),
        hospitalizations=np.broadcast_to(
            np.array([10.0, 5.0], dtype=np.float32), (1, periods, 2)
        ).copy(),
        hospitalizations_age=np.ones((1, periods, 2), dtype=np.float32),
        deaths=np.broadcast_to(
            np.array([1.0, 0.5], dtype=np.float32), (1, periods, 2)
        ).copy(),
        deaths_age=np.ones((1, periods, 2), dtype=np.float32),
        biomarker_values={
            "N1": np.broadcast_to(
                np.array([1.0, 0.0], dtype=np.float32), (1, periods, 2)
            ).copy()
        },
        mobility_time_mask=mobility_time_mask,
    )
    write_zarr_dataset(ds, zarr_path)


@pytest.mark.epiforecaster
def test_getitem_preserves_target_history_when_self_edge_missing(tmp_path):
    """Test that target history is preserved even when self-mobility is below threshold.

    This exposes a bug where target inclusion is forced on neigh_mask, but masking
    is applied with mobility_mask_float instead, causing target history to be zeroed
    when self-edges are missing.
    """
    zarr_path = tmp_path / "self_edge_missing.zarr"
    _write_tiny_dataset(str(zarr_path), periods=8)

    config = _make_config(str(zarr_path))
    config.data.mobility_min_flow_raw = 0.5
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    # Remove all self-mobility so target self-edge is below threshold
    dataset.preloaded_mobility[:, 0, 0] = 0.0
    dataset.preloaded_mobility[:, 1, 1] = 0.0
    dataset.mobility_mask[:, 0, 0] = False
    dataset.mobility_mask[:, 1, 1] = False

    item = dataset[0]  # target node 0, first window
    cases_hist = item["cases_hist"]  # (L, 3), channels [value, mask, age]

    # If target inclusion is honored, target mask should remain observed
    assert torch.all(cases_hist[:, 1] > 0.5), (
        "target history mask got zeroed by mobility masking"
    )


@pytest.mark.epiforecaster
def test_mobility_time_mask_blocks_edges_when_threshold_zero(tmp_path):
    zarr_path = tmp_path / "mobility_time_mask.zarr"
    mobility_time_mask = np.array(
        [[True, True, True, True, False, False, False, False]],
        dtype=bool,
    )
    _write_tiny_dataset(
        str(zarr_path),
        periods=8,
        mobility_time_mask=mobility_time_mask,
    )

    config = _make_config(str(zarr_path))
    config.data.mobility_min_flow_raw = 0.0
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    off_diagonal = ~torch.eye(2, dtype=torch.bool)
    assert torch.all(dataset.mobility_mask[:4, off_diagonal])
    assert not torch.any(torch.diagonal(dataset.mobility_mask[:4], dim1=-2, dim2=-1))
    assert not torch.any(dataset.mobility_mask[4:])
    assert not torch.any(dataset.preloaded_mobility[4:])
    assert not torch.any(dataset._get_adjacency_at_time(5))


@pytest.mark.epiforecaster
def test_dynamic_mask_is_shared_by_clinical_topology_and_gpu_adjacency(
    tmp_path,
) -> None:
    periods = 8
    raw_flow = np.zeros((1, periods, 3, 3), dtype=np.float32)
    raw_flow[:, :, 0, 1] = 9.0
    raw_flow[:, :, 2, 1] = 10.0
    raw_flow[:, :, 1, 2] = 11.0
    cases = np.broadcast_to(
        np.array([100.0, 200.0, 300.0], dtype=np.float32).reshape(1, 1, 3, 1),
        (1, periods, 3, 1),
    ).copy()
    dataset_path = tmp_path / "known_od.zarr"
    canonical = make_tiny_epi_dataset(
        periods=periods,
        num_nodes=3,
        cases=cases,
        mobility=np.log1p(raw_flow),
    )
    write_zarr_dataset(canonical, dataset_path)
    config = make_epi_config(
        dataset_path,
        data_options={"mobility_min_flow_raw": 10.0},
    )
    dataset = EpiDataset(
        config=config,
        target_nodes=[1],
        context_nodes=[0, 1, 2],
    )

    expected_od_mask = dataset.mobility_mask[0]
    assert not expected_od_mask[0, 1]
    assert expected_od_mask[2, 1]
    assert expected_od_mask[1, 2]
    assert torch.equal(dataset._get_adjacency_at_time(0), expected_od_mask.T)

    clinical_mask = dataset._get_clinical_neighbor_mask(
        range_start=0,
        range_end=config.model.input_window_length,
        target_idx=1,
    )
    assert torch.equal(
        clinical_mask,
        dataset.mobility_mask[: config.model.input_window_length, :, 1],
    )

    mob_batch = SimpleNamespace(
        global_t=torch.tensor([0]),
        run_id="real",
        target_node=torch.tensor([0]),
    )
    batch = SimpleNamespace(mob_batch=mob_batch)
    inject_gpu_mobility(batch, dataset, torch.device("cpu"))

    node_ids = torch.where(dataset._get_graph_node_mask())[0]
    expected_weights = dataset.preloaded_mobility[
        0,
        node_ids.unsqueeze(-1),
        node_ids,
    ].T.float()
    expected_weights = torch.maximum(
        expected_weights,
        torch.eye(len(node_ids)),
    )
    adjacency = batch.mob_batch.adj_table[batch.mob_batch.adj_time_ids]
    assert torch.allclose(adjacency[0].float(), expected_weights)


@pytest.mark.epiforecaster
def test_getitem_values(tmp_path):
    zarr_path = tmp_path / "tiny.zarr"
    _write_tiny_dataset(str(zarr_path))

    # Constant cases for easy verification
    # Node 0: 100 cases per day. Pop 1000. -> 10,000 per 100k.
    # Node 1: 50 cases per day. Pop 1000. -> 5,000 per 100k.
    # Note: Log scaling is now done in offline preprocessor, so dataset receives raw values

    config = _make_config(str(zarr_path))
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
def test_getitem_aligns_death_flow_and_rebases_cumulative_d(tmp_path):
    zarr_path = tmp_path / "latent_death_flow.zarr"
    periods = 8
    dataset_source = make_tiny_epi_dataset(periods=periods)
    time_index = np.arange(periods, dtype=np.float32)
    cumulative_d = np.stack((time_index * 0.1, time_index * 0.2), axis=-1)[None, ...]
    death_flow = np.stack(
        (
            np.concatenate(([0.0], np.full(periods - 1, 0.1, dtype=np.float32))),
            np.concatenate(([0.0], np.full(periods - 1, 0.2, dtype=np.float32))),
        ),
        axis=-1,
    )[None, ...]
    latent_dims = ("run_id", "date", "region_id")
    dataset_source["latent_D_true"] = (latent_dims, cumulative_d)
    dataset_source["latent_death_flow_true"] = (latent_dims, death_flow)
    write_zarr_dataset(dataset_source, zarr_path)

    config = _make_config(str(zarr_path))
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])

    item = dataset[0]
    target_node = item["target_node"]
    expected_flow = 0.1 if target_node == 0 else 0.2

    assert item["D_target"] is not None
    assert item["death_flow_target"] is not None
    assert item["D_target"].shape == (config.model.forecast_horizon + 1,)
    assert item["death_flow_target"].shape == (config.model.forecast_horizon,)
    assert item["D_target"][0].item() == pytest.approx(0.0)
    assert torch.allclose(
        item["D_target"],
        torch.tensor([0.0, expected_flow, 2.0 * expected_flow]),
    )
    assert torch.allclose(
        item["death_flow_target"],
        torch.full((config.model.forecast_horizon,), expected_flow),
    )

    batch = collate_epiforecaster_batch(
        [dataset[0], dataset[1]], require_region_index=False
    )
    assert batch.D_target is not None
    assert batch.death_flow_target is not None
    assert batch.D_target.shape == (2, config.model.forecast_horizon + 1)
    assert batch.death_flow_target.shape == (2, config.model.forecast_horizon)


@pytest.mark.epiforecaster
def test_getitem_maps_target_and_context_nodes_to_canonical_region_indices(tmp_path):
    zarr_path = tmp_path / "region_mapping.zarr"
    _write_tiny_dataset(str(zarr_path), periods=8)

    config = _make_config(str(zarr_path))
    store = RegionEmbeddingStore(
        embeddings=torch.randn(2, 1),
        region_id_to_index={"0": 1, "1": 0},
    )
    dataset = EpiDataset(
        config=config,
        target_nodes=[0, 1],
        context_nodes=[0, 1],
        region_embedding_store=store,
    )

    item = dataset[0]
    if item["target_node"] == 0:
        assert item["target_region_index"] == 1
    else:
        assert item["target_region_index"] == 0

    assert torch.equal(item["mob_real_node_idx"], torch.tensor([1, 0]))


@pytest.mark.epiforecaster
def test_getitem_reuses_cached_graph_region_indices(tmp_path):
    zarr_path = tmp_path / "cached_region_mapping.zarr"
    _write_tiny_dataset(str(zarr_path), periods=8)

    config = _make_config(str(zarr_path))
    store = RegionEmbeddingStore(
        embeddings=torch.randn(2, 1),
        region_id_to_index={"0": 1, "1": 0},
    )
    dataset = EpiDataset(
        config=config,
        target_nodes=[0, 1],
        context_nodes=[0, 1],
        region_embedding_store=store,
    )

    item = dataset[0]
    assert item["mob_real_node_idx"] is dataset._graph_region_indices


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
    config2.model.cases_dim = 11

    dataset2 = EpiDataset(config=config2, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset2.cases_output_dim == 11  # 9 + 2 lags

    # Test 3: Verify item shape matches expected dimension
    item = dataset2[0]
    # Only cases_hist has lag features concatenated
    # hosp_hist and deaths_hist remain 3 channels (value, mask, age)
    assert item["hosp_hist"].shape[-1] == 3  # 3 channels, no lags
    assert item["deaths_hist"].shape[-1] == 3  # 3 channels, no lags
    assert item["cases_hist"].shape[-1] == 5  # 3 case channels + 2 lag channels
