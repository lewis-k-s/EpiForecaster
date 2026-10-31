"""Tests for the lightweight mobility placement container."""

import pytest
import torch

from data.epi_dataset import EpiDatasetItem
from data.epi_batch import optimized_collate_graphs


SHARED_NODES = torch.tensor([4, 5, 6], dtype=torch.long)


def _make_item(
    window_start: int, target_pos: int = 0, run_id="real", L: int = 3
) -> EpiDatasetItem:
    return {
        "node_label": "n",
        "region_id": str(target_pos),
        "target_node": target_pos,
        "window_start": window_start,
        "bio_node": torch.zeros(L, 1),
        "hosp_hist": torch.zeros(L, 3),
        "deaths_hist": torch.zeros(L, 3),
        "cases_hist": torch.zeros(L, 3),
        "mob_target_node_idx": torch.tensor(target_pos, dtype=torch.long),
        "mob_real_node_idx": SHARED_NODES,
        "population": torch.tensor(1.0),
        "run_id": run_id,
        "target_region_index": None,
        "temporal_covariates": torch.zeros(L, 2),
        "vaccination_hist": torch.zeros(L, 3),
        "ww_hist": torch.zeros(L),
        "ww_hist_mask": torch.zeros(L),
        "hosp_target": torch.zeros(1),
        "ww_target": torch.zeros(1),
        "cases_target": torch.zeros(1),
        "deaths_target": torch.zeros(1),
        "hosp_target_mask": torch.zeros(1),
        "ww_target_mask": torch.zeros(1),
        "cases_target_mask": torch.zeros(1),
        "deaths_target_mask": torch.zeros(1),
        "S_target": None,
        "I_target": None,
        "H_target": None,
        "R_target": None,
        "D_target": None,
        "death_flow_target": None,
    }


@pytest.mark.epiforecaster
def test_placement_tensors_follow_batch_order() -> None:
    B, L = 3, 3
    items = [_make_item(window_start=10 * b, target_pos=b) for b in range(B)]
    mob_batch = optimized_collate_graphs(items)

    assert not hasattr(mob_batch, "x_dense")
    assert mob_batch.global_t.shape == (B * L,)
    expected_global_t = torch.cat([torch.arange(10 * b, 10 * b + L) for b in range(B)])
    assert torch.equal(mob_batch.global_t, expected_global_t)

    expected_targets = torch.arange(B).repeat_interleave(L)
    assert torch.equal(mob_batch.target_node, expected_targets)


@pytest.mark.epiforecaster
def test_homogeneous_run_id_is_attached() -> None:
    items = [_make_item(0), _make_item(5)]
    mob_batch = optimized_collate_graphs(items)
    assert getattr(mob_batch, "run_id", None) == "real"


@pytest.mark.epiforecaster
def test_heterogeneous_or_missing_run_id_stays_none() -> None:
    mixed = [_make_item(0, run_id="a"), _make_item(1, run_id="b")]
    assert getattr(optimized_collate_graphs(mixed), "run_id", None) is None

    missing = [
        {
            "window_start": 0,
            "temporal_covariates": torch.zeros(2, 0),
            "mob_target_node_idx": torch.tensor(0),
        },
        {
            "window_start": 1,
            "temporal_covariates": torch.zeros(2, 0),
            "mob_target_node_idx": torch.tensor(0),
        },
    ]
    assert getattr(optimized_collate_graphs(missing), "run_id", None) is None


@pytest.mark.epiforecaster
def test_shared_context_nodes_must_be_identical_objects() -> None:
    item_a = _make_item(0)
    item_b = _make_item(1)
    item_b["mob_real_node_idx"] = item_b["mob_real_node_idx"].clone()
    with pytest.raises(ValueError, match="same cached tensor"):
        optimized_collate_graphs([item_a, item_b])


@pytest.mark.epiforecaster
def test_shared_window_start_emits_one_time_row() -> None:
    B, L = 3, 3
    items = [_make_item(window_start=7, target_pos=b) for b in range(B)]
    mob_batch = optimized_collate_graphs(items)

    assert torch.equal(mob_batch.global_t, torch.arange(7, 7 + L))
    assert mob_batch.target_node.shape == (B * L,)
