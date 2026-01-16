from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Batch  # type: ignore[import-not-found]

from data.epi_dataset import EpiDatasetItem


def collate_epidataset_batch(batch: list[EpiDatasetItem]) -> dict[str, Any]:
    """Custom collate for per-node samples with PyG mobility graphs."""

    B = len(batch)
    case_node = torch.stack([item["case_node"] for item in batch], dim=0)
    bio_node = torch.stack([item["bio_node"] for item in batch], dim=0)
    case_mean = torch.stack([item["case_mean"] for item in batch], dim=0)
    case_std = torch.stack([item["case_std"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    target_scales = torch.stack([item["target_scale"] for item in batch], dim=0)
    target_mean = torch.stack([item["target_mean"] for item in batch], dim=0)
    target_nodes = torch.tensor(
        [item["target_node"] for item in batch], dtype=torch.long
    )
    population = torch.stack([item["population"] for item in batch], dim=0)

    # Flatten mobility graphs and annotate batch_id/time_id for batching
    # Pre-allocate list to avoid dynamic resizing
    B, T = len(batch), len(batch[0]["mob"]) if batch else 0
    graph_list = [None] * (B * T)
    idx = 0
    for b, item in enumerate(batch):
        for t, g in enumerate(item["mob"]):
            # Use scalar assignment instead of tensor creation for performance
            g.batch_id = b
            g.time_id = t
            graph_list[idx] = g  # type: ignore[list-item]
            idx += 1

    mob_batch = Batch.from_data_list(graph_list)  # type: ignore[arg-type]
    T = len(batch[0]["mob"]) if B > 0 else 0
    # store B and T on the batch for downstream reshaping
    mob_batch.B = torch.tensor([B], dtype=torch.long)  # type: ignore[attr-defined]
    mob_batch.T = torch.tensor([T], dtype=torch.long)  # type: ignore[attr-defined]
    # Precompute a global target node index per ego-graph in the batched `x`.
    # This enables fully-vectorized target gathering in the model without CUDA `.item()` syncs.
    if hasattr(mob_batch, "ptr") and hasattr(mob_batch, "target_node"):
        mob_batch.target_index = mob_batch.ptr[:-1] + mob_batch.target_node.reshape(-1)  # type: ignore[attr-defined]

    return {
        "CaseNode": case_node,  # (B, L, C)
        "CaseMean": case_mean,  # (B, L, 1)
        "CaseStd": case_std,  # (B, L, 1)
        "BioNode": bio_node,  # (B, L, B)
        "MobBatch": mob_batch,  # Batched PyG graphs
        "Population": population,  # (B,)
        "B": B,
        "T": T,
        "Target": targets,  # (B, H)
        "TargetScale": target_scales,
        "TargetMean": target_mean,
        "TargetNode": target_nodes,  # (B,)
        "NodeLabels": [item["node_label"] for item in batch],
    }
