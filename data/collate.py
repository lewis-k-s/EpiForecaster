from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Batch

from data.epi_dataset import EpiDatasetItem


def collate_epidataset_batch(batch: list[EpiDatasetItem]) -> dict[str, Any]:
    """Custom collate for per-node samples with PyG mobility graphs."""

    B = len(batch)
    case_node = torch.stack([item["case_node"] for item in batch], dim=0)
    bio_node = torch.stack([item["bio_node"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    target_nodes = torch.tensor(
        [item["target_node"] for item in batch], dtype=torch.long
    )
    population = torch.stack([item["population"] for item in batch], dim=0)

    # Flatten mobility graphs and annotate batch_id/time_id for batching
    graph_list = []
    for b, item in enumerate(batch):
        for t, g in enumerate(item["mob"]):
            g.batch_id = torch.tensor([b], dtype=torch.long)
            g.time_id = torch.tensor([t], dtype=torch.long)
            graph_list.append(g)

    mob_batch = Batch.from_data_list(graph_list)
    T = len(batch[0]["mob"]) if B > 0 else 0
    # store B and T on the batch for downstream reshaping
    mob_batch.B = torch.tensor([B], dtype=torch.long)
    mob_batch.T = torch.tensor([T], dtype=torch.long)
    # Precompute a global target node index per ego-graph in the batched `x`.
    # This enables fully-vectorized target gathering in the model without CUDA `.item()` syncs.
    if hasattr(mob_batch, "ptr") and hasattr(mob_batch, "target_node"):
        mob_batch.target_index = mob_batch.ptr[:-1] + mob_batch.target_node.reshape(-1)

    return {
        "CaseNode": case_node,  # (B, L, C)
        "BioNode": bio_node,  # (B, L, B)
        "MobBatch": mob_batch,  # Batched PyG graphs
        "Population": population,  # (B,)
        "B": B,
        "T": T,
        "Target": targets,  # (B, H)
        "TargetNode": target_nodes,  # (B,)
        "NodeLabels": [item["node_label"] for item in batch],
    }
