"""Bitwise equivalence tests for batch-side context feature assembly.

Dense mobility features ``(L, N_ctx, F)`` are prebuilt into context tables
at init and assembled batch-side (``data/context_assembly.py``). A verbatim
per-sample CPU loop provides the oracle; these tests require ``torch.equal``
agreement, guaranteeing unchanged model inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.epi_dataset import EpiDataset
from models.configs import EpiForecasterConfig, ModelVariant
from tests.dataset_test_helpers import (
    context_features_for_item,
    make_epi_config,
    make_tiny_epi_dataset,
    write_zarr_dataset,
)


def _make_config(dataset_path: str, gnn_depth: int = 2) -> EpiForecasterConfig:
    return make_epi_config(
        dataset_path,
        model_options={
            "type": ModelVariant(
                cases=True, regions=False, biomarkers=True, mobility=True
            ),
            "forecast_horizon": 2,
            "gnn_depth": gnn_depth,
            "gnn_module": "gcn",
            "gnn_hidden_dim": 8,
        },
        data_options={
            "use_imported_risk": True,
            "mobility_lags": [1],
        },
    )


def _write_dataset(zarr_path: str, num_nodes: int = 5, periods: int = 12) -> None:
    rng = np.random.default_rng(7)
    mobility = (rng.random((1, periods, num_nodes, num_nodes)) > 0.5).astype(np.float32)
    cases = rng.random((1, periods, num_nodes, 1)).astype(np.float32) * 100.0
    hosp = rng.random((1, periods, num_nodes, 1)).astype(np.float32) * 10.0
    biomarker_values = {
        f"edar_biomarker_{name}": rng.random((1, periods, num_nodes)).astype(np.float32)
        for name in ("N1", "N2", "IP4")
    }
    biomarker_masks = {
        f"edar_biomarker_{name}": (rng.random((1, periods, num_nodes)) > 0.3).astype(
            bool
        )
        for name in ("N1", "N2", "IP4")
    }
    dataset = make_tiny_epi_dataset(
        periods=periods,
        num_nodes=num_nodes,
        cases=cases,
        hospitalizations=hosp,
        deaths=hosp.copy(),
        mobility=mobility,
        biomarker_values=biomarker_values,
        biomarker_masks=biomarker_masks,
        clinical_feature_dim=True,
    )
    write_zarr_dataset(dataset, zarr_path)


def _legacy_item_features(dataset: EpiDataset, idx: int) -> torch.Tensor:
    """Verbatim port of the legacy per-sample feature construction loop."""
    L = dataset.config.model.input_window_length
    target_idx, range_start = dataset._index_map[idx]
    range_end = range_start + L
    node_ids = dataset._graph_node_ids

    neigh_mask = dataset._get_clinical_neighbor_mask(
        range_start=range_start,
        range_end=range_end,
        target_idx=target_idx,
    )
    if dataset.context_mask is not None:
        neigh_mask = neigh_mask & dataset.context_mask[None, :]
    neigh_mask[:, target_idx] = True

    hosp_history = dataset.precomputed_hosp_hist[range_start:range_end]
    deaths_history = dataset.precomputed_deaths_hist[range_start:range_end]
    cases_history = dataset.precomputed_cases_hist[range_start:range_end]
    neigh_mask_t = neigh_mask.unsqueeze(-1).to(torch.float16)
    hosp_history = hosp_history * neigh_mask_t
    deaths_history = deaths_history * neigh_mask_t
    cases_history = cases_history * neigh_mask_t
    if dataset.use_imported_risk and dataset.lagged_risk is not None:
        risk_slice = dataset.lagged_risk[range_start:range_end] * neigh_mask_t
        cases_history = torch.cat([cases_history, risk_slice], dim=-1)

    bio_slice = dataset.precomputed_biomarkers[range_start:range_end]
    range_end_idx = range_end - 1
    has_data = (range_end_idx >= dataset.biomarker_data_start).to(torch.float16)
    has_data[dataset.biomarker_data_start < 0] = 0.0
    has_data_3d = has_data.view(1, dataset.num_nodes, 1).expand(L, -1, -1)
    biomarker_history = torch.cat([bio_slice, has_data_3d], dim=-1)

    global_to_local = dataset._get_global_to_local_at_time(range_start)
    local_target_idx = int(global_to_local[target_idx].item())

    outputs = []
    for t in range(L):
        global_t = range_start + t
        feat = []
        if dataset.config.model.type.cases:
            feat.append(hosp_history[t][node_ids])
            feat.append(deaths_history[t][node_ids])
            feat.append(cases_history[t][node_ids])
        if dataset.config.model.type.biomarkers:
            feat.append(biomarker_history[t][node_ids])
            feat.append(dataset.precomputed_ww[global_t, node_ids].unsqueeze(-1))
            feat.append(dataset.precomputed_ww_mask[global_t, node_ids].unsqueeze(-1))
        x = torch.cat(feat, dim=-1)
        if dataset._target_k_hop_reach is not None:
            target_slot = list(dataset.target_nodes).index(target_idx)
            k_hop_mask = dataset._target_k_hop_reach[global_t, target_slot]
            local_k_hop_mask = k_hop_mask[node_ids].clone()
            local_k_hop_mask[local_target_idx] = True
            x = torch.where(
                local_k_hop_mask.unsqueeze(-1).expand_as(x),
                x,
                torch.zeros_like(x),
            )
        outputs.append(x)
    return torch.stack(outputs)


@pytest.mark.epiforecaster
@pytest.mark.parametrize("gnn_depth", [0, 1, 2])
def test_assembly_matches_legacy_loop_bitwise(tmp_path, gnn_depth: int) -> None:
    zarr_path = str(tmp_path / "equivalence.zarr")
    _write_dataset(zarr_path)
    config = _make_config(zarr_path, gnn_depth=gnn_depth)
    dataset = EpiDataset(
        config=config, target_nodes=[0, 1, 2], context_nodes=[0, 1, 2, 3, 4]
    )

    checked = 0
    for idx in range(min(8, len(dataset))):
        item = dataset[idx]
        assembled = context_features_for_item(dataset, item)
        legacy = _legacy_item_features(dataset, idx)
        assert assembled.shape == legacy.shape
        assert assembled.dtype == legacy.dtype == torch.float32
        assert torch.equal(assembled, legacy)
        checked += 1
    assert checked > 0


@pytest.mark.epiforecaster
def test_batch_injection_matches_per_item_assembly(tmp_path) -> None:
    from data.epi_batch import collate_epiforecaster_batch
    from utils.training_utils import inject_context_features

    zarr_path = str(tmp_path / "batch_equivalence.zarr")
    _write_dataset(zarr_path)
    config = _make_config(zarr_path, gnn_depth=2)
    dataset = EpiDataset(
        config=config, target_nodes=[0, 1, 2], context_nodes=[0, 1, 2, 3, 4]
    )

    items = [dataset[i] for i in range(4)]
    batch = collate_epiforecaster_batch(items, require_region_index=False)
    assert not hasattr(batch.mob_batch, "x_dense")

    inject_context_features(batch, dataset, torch.device("cpu"))
    x_dense = batch.mob_batch.x_dense
    L = config.model.input_window_length
    assert x_dense.shape[0] == len(items) * L

    per_item = torch.cat(
        [context_features_for_item(dataset, item) for item in items], dim=0
    )
    assert torch.equal(x_dense, per_item)
