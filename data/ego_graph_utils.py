"""
Utility functions for ego-graph dataset creation and processing.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def ego_graph_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for ego-graph datasets with variable-sized graphs.

    Args:
        batch: List of ego-graph samples

    Returns:
        Collated batch with padded tensors
    """
    # Extract scalar values
    target_region_ids = torch.tensor([item["target_region_id"] for item in batch])

    # Pad temporal sequences
    cases_hist = torch.stack([item["cases_hist"] for item in batch])  # [B, L]
    biomarkers_hist = torch.stack(
        [item["biomarkers_hist"] for item in batch]
    )  # [B, L, F]
    cases_future = torch.stack([item["cases_future"] for item in batch])  # [B, H]

    # Handle variable-sized ego-graphs
    node_features_t = []
    edge_index_t = []
    edge_weight_t = []
    target_local_idx = []

    for t in range(len(batch[0]["node_features_t"])):  # Assuming same L for all samples
        # Collect node features for this timestep
        node_features_t_t = [item["node_features_t"][t] for item in batch]

        # Pad node features to max size in batch
        max_nodes = max([nf.shape[0] for nf in node_features_t_t])
        padded_features = []
        for nf in node_features_t_t:
            if nf.shape[0] < max_nodes:
                padding = torch.zeros(max_nodes - nf.shape[0], nf.shape[1])
                padded = torch.cat([nf, padding], dim=0)
            else:
                padded = nf
            padded_features.append(padded)

        node_features_t.append(torch.stack(padded_features))  # [B, max_nodes, 1]

        # Handle edge indices and weights
        edge_indices_t = []
        edge_weights_t = []
        target_indices_t = []

        for i, item in enumerate(batch):
            edge_idx = item["edge_index_t"][t]  # [2, E] or [2, 0]
            edge_weight = item["edge_weight_t"][t]  # [E] or []
            target_idx = item["target_local_idx"][t]

            if edge_idx.shape[1] > 0:
                # Offset edge indices by batch node offset
                offset = i * max_nodes
                offset_edge_idx = edge_idx + offset
            else:
                # Create empty edge tensors
                offset_edge_idx = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty(0, dtype=torch.float)

            edge_indices_t.append(offset_edge_idx)
            edge_weights_t.append(edge_weight)

            # Offset target local index by batch offset
            if target_idx >= 0:
                offset_target_idx = target_idx + i * max_nodes
            else:
                offset_target_idx = -1
            target_indices_t.append(offset_target_idx)

        # Combine edge indices and weights across batch
        if any(ei.shape[1] > 0 for ei in edge_indices_t):
            combined_edge_index = torch.cat(edge_indices_t, dim=1)  # [2, sum(E)]
            combined_edge_weight = torch.cat(edge_weights_t, dim=0)  # [sum(E)]
        else:
            combined_edge_index = torch.empty((2, 0), dtype=torch.long)
            combined_edge_weight = torch.empty(0, dtype=torch.float)

        edge_index_t.append(combined_edge_index)
        edge_weight_t.append(combined_edge_weight)
        target_local_idx.append(torch.tensor(target_indices_t))

    return {
        "target_region_ids": target_region_ids,
        "cases_hist": cases_hist,
        "biomarkers_hist": biomarkers_hist,
        "cases_future": cases_future,
        "node_features_t": node_features_t,  # List of [B, max_nodes_t, 1]
        "edge_index_t": edge_index_t,  # List of [2, sum(E_t)]
        "edge_weight_t": edge_weight_t,  # List of [sum(E_t)]
        "target_local_idx": target_local_idx,  # List of [B]
        "max_nodes_per_timestep": [nf.shape[1] for nf in node_features_t],
    }


def test_ego_graph_dataset(zarr_path: str, num_samples: int = 5):
    """
    Test function for ego-graph dataset creation and sampling.

    Args:
        zarr_path: Path to Zarr dataset
        num_samples: Number of samples to test
    """
    from .ego_graph_dataset import create_ego_graph_dataset

    print(f"Testing ego-graph dataset creation from: {zarr_path}")

    try:
        dataset = create_ego_graph_dataset(
            zarr_path=zarr_path,
            L=14,
            H=7,
            min_flow_threshold=10.0,
            max_neighbors=20,
            use_mobility=False,  # Set to False for cases-only dataset
        )

        print("Dataset created successfully!")
        print(f"Dataset info: {dataset.get_dataset_info()}")

        # Test sample retrieval
        print(f"\nTesting {min(num_samples, len(dataset))} samples:")
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Target region: {sample['target_region_id']}")
            print(f"  Cases hist shape: {sample['cases_hist'].shape}")
            print(f"  Biomarkers hist shape: {sample['biomarkers_hist'].shape}")
            print(f"  Cases future shape: {sample['cases_future'].shape}")
            print(
                f"  Node features per timestep: {[nf.shape for nf in sample['node_features_t']]}"
            )
            print(
                f"  Edge indices per timestep: {[ei.shape[1] for ei in sample['edge_index_t']]}"
            )
            print(
                f"  Edge weights per timestep: {[ew.shape[0] for ew in sample['edge_weight_t']]}"
            )

        # Test collate function
        if len(dataset) >= 2:
            print("\nTesting collate function with batch of 2 samples:")
            batch = [dataset[0], dataset[1]]
            collated = ego_graph_collate_fn(batch)
            print(f"  Collated keys: {list(collated.keys())}")
            print(f"  Target region IDs shape: {collated['target_region_ids'].shape}")
            print(f"  Cases hist batch shape: {collated['cases_hist'].shape}")
            print(
                f"  Node features timesteps: {[nf.shape for nf in collated['node_features_t']]}"
            )

        return True

    except Exception as e:
        print(f"Error testing ego-graph dataset: {e}")
        import traceback

        traceback.print_exc()
        return False
