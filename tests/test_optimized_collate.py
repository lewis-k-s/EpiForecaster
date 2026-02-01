import torch
import pytest
from torch_geometric.data import Data, Batch
from data.epi_dataset import optimized_collate_graphs

# Mocking the structures for the test
# We want to simulate the BEFORE and AFTER states of the refactor
# to ensure they produce the same result.


def mock_getitem_old(L=3, num_nodes=5):
    """Simulate the old __getitem__ returning list[Data]."""
    mob_graphs = []
    for t in range(L):
        # Create random features and edges
        x = torch.randn(num_nodes, 2)
        edge_index = torch.randint(0, num_nodes, (2, 10))
        edge_weight = torch.rand(10)

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        data.num_nodes = num_nodes
        data.target_node = torch.tensor([0], dtype=torch.long)
        data.node_ids = torch.arange(num_nodes)
        data.time_id = torch.tensor([t], dtype=torch.long)
        mob_graphs.append(data)

    return {"mob": mob_graphs}


def mock_collate_old(batch):
    """Simulate the old collate function using Batch.from_data_list."""
    import itertools

    graph_list = list(itertools.chain.from_iterable(item["mob"] for item in batch))
    mob_batch = Batch.from_data_list(graph_list)
    return mob_batch


@pytest.mark.epiforecaster
def test_manual_batching_equivalence():
    """Verify that manual batching produces identical results to Batch.from_data_list."""

    # Setup parameters
    B = 4
    L = 3
    num_nodes = 5
    F = 2

    # Generate raw data for the batch
    raw_data = []
    for b in range(B):
        sample = []
        for t in range(L):
            x = torch.randn(num_nodes, F)
            # Random edges (self loops included for simplicity)
            edge_index = torch.randint(0, num_nodes, (2, 10))
            edge_weight = torch.rand(10)
            sample.append((x, edge_index, edge_weight))
        raw_data.append(sample)

    # 1. Create Old Format Batch
    old_batch_input = []
    for b in range(B):
        mob_graphs = []
        for t in range(L):
            x, ei, ew = raw_data[b][t]
            data = Data(x=x, edge_index=ei, edge_weight=ew)
            data.num_nodes = num_nodes
            data.target_node = torch.tensor([0], dtype=torch.long)  # Dummy
            data.node_ids = torch.arange(num_nodes)  # Dummy
            data.time_id = torch.tensor([t], dtype=torch.long)  # Dummy
            mob_graphs.append(data)
        old_batch_input.append({"mob": mob_graphs})

    old_result = mock_collate_old(old_batch_input)

    # 2. Create New Format Batch (Manual Batching Logic)

    new_batch_input = []
    for b in range(B):
        mob_x_list = []
        mob_ei_list = []
        mob_ew_list = []
        for t in range(L):
            x, ei, ew = raw_data[b][t]
            mob_x_list.append(x)
            mob_ei_list.append(ei)
            mob_ew_list.append(ew)

        item = {
            "mob_x": torch.stack(mob_x_list),  # (L, N, F)
            "mob_edge_index": mob_ei_list,
            "mob_edge_weight": mob_ew_list,
            "mob_target_node_idx": 0,  # Dummy
        }
        new_batch_input.append(item)

    # Use the ACTUAL function
    new_result = optimized_collate_graphs(new_batch_input)

    # 3. Compare Results
    assert torch.allclose(old_result.x, new_result.x)
    assert torch.allclose(old_result.edge_index, new_result.edge_index)
    assert torch.allclose(old_result.edge_weight, new_result.edge_weight)
    assert torch.equal(old_result.batch, new_result.batch)

    # Check target_node reconstruction
    # Old result has target_node from Batch.from_data_list (concatenated list of tensors)
    # New result has it constructed from mob_target_node_idx
    assert torch.equal(old_result.target_node, new_result.target_node)

    print("Verification successful!")


if __name__ == "__main__":
    test_manual_batching_equivalence()
