from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from data.epi_batch import EpiBatch


def _make_batch() -> EpiBatch:
    mob_batch = Batch()
    mob_batch.x_dense = torch.ones(1, 1, 1, dtype=torch.float32)
    mob_batch.global_t = torch.zeros(1, dtype=torch.long)
    mob_batch.target_node = torch.zeros(1, dtype=torch.long)

    return EpiBatch(
        hosp_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        deaths_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        cases_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        bio_node=torch.ones(1, 2, 1, dtype=torch.float32),
        mob_batch=mob_batch,
        population=torch.ones(1, dtype=torch.float32),
        b=1,
        t=2,
        target_node=torch.zeros(1, dtype=torch.long),
        target_region_index=torch.zeros(1, dtype=torch.long),
        window_start=torch.zeros(1, dtype=torch.long),
        node_labels=["node-0"],
        temporal_covariates=torch.ones(1, 2, 1, dtype=torch.float32),
        vaccination_hist=torch.zeros(1, 2, 3, dtype=torch.float32),
        ww_hist=torch.ones(1, 2, dtype=torch.float32),
        ww_hist_mask=torch.ones(1, 2, dtype=torch.float32),
        hosp_target=torch.ones(1, 1, dtype=torch.float32),
        ww_target=torch.ones(1, 1, dtype=torch.float32),
        cases_target=torch.ones(1, 1, dtype=torch.float32),
        deaths_target=torch.ones(1, 1, dtype=torch.float32),
        hosp_target_mask=torch.ones(1, 1, dtype=torch.float32),
        ww_target_mask=torch.ones(1, 1, dtype=torch.float32),
        cases_target_mask=torch.ones(1, 1, dtype=torch.float32),
        deaths_target_mask=torch.ones(1, 1, dtype=torch.float32),
        S_target=None,
        I_target=None,
        R_target=None,
        D_target=None,
    )


@pytest.mark.device
def test_epibatch_to_preserves_integer_indices_on_accelerator(
    accelerator_device,
) -> None:
    batch = _make_batch()
    batch.target_node = torch.tensor([17], dtype=torch.long)
    batch.target_region_index = torch.tensor([23], dtype=torch.long)
    batch.window_start = torch.tensor([31], dtype=torch.long)
    batch.mob_batch.target_node = torch.tensor([5], dtype=torch.long)
    batch.mob_batch.global_t = torch.tensor([11], dtype=torch.long)

    moved = batch.to(accelerator_device, non_blocking=True)

    assert moved.target_node.cpu().tolist() == [17]
    assert moved.target_region_index is not None
    assert moved.target_region_index.cpu().tolist() == [23]
    assert moved.window_start.cpu().tolist() == [31]
    assert moved.mob_batch.target_node.cpu().tolist() == [5]
    assert moved.mob_batch.global_t.cpu().tolist() == [11]
