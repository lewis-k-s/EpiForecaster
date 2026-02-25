import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

from data.epi_dataset import optimized_collate_graphs
from utils.training_utils import ensure_mobility_adj_dense_ready, inject_gpu_mobility


class _MobBatch:
    def __init__(self, global_t: list[int], run_id: str | None = None):
        self.global_t = torch.tensor(global_t, dtype=torch.long)
        self.target_node = torch.zeros(len(global_t), dtype=torch.long)
        self.run_id = run_id


class _DummyMobilityDataset(Dataset):
    def __init__(self, *, run_id: str, preloaded_mobility: torch.Tensor):
        self.run_id = run_id
        self.preloaded_mobility = preloaded_mobility

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> int:
        return idx

    def _get_graph_node_mask(self) -> torch.Tensor:
        num_nodes = self.preloaded_mobility.shape[-1]
        return torch.ones(num_nodes, dtype=torch.bool)


@pytest.mark.epiforecaster
def test_inject_gpu_mobility_uses_correct_concat_subdataset_by_run_id() -> None:
    real_mobility = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    )
    synth_mobility = torch.tensor(
        [[[1.0, 9.0], [9.0, 1.0]]], dtype=torch.float32
    )

    real_ds = _DummyMobilityDataset(run_id="real", preloaded_mobility=real_mobility)
    synth_ds = _DummyMobilityDataset(
        run_id="synth_1", preloaded_mobility=synth_mobility
    )
    dataset = ConcatDataset([real_ds, synth_ds])

    batch_data = {"MobBatch": _MobBatch(global_t=[0], run_id="synth_1")}
    inject_gpu_mobility(batch_data, dataset, torch.device("cpu"))

    adj_dense = batch_data["MobBatch"].adj_dense
    assert adj_dense.shape == (1, 2, 2)
    assert float(adj_dense[0, 0, 1]) == pytest.approx(9.0, abs=1e-3)
    assert float(adj_dense[0, 0, 0]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.epiforecaster
def test_inject_gpu_mobility_reuses_cached_tensor_without_crashing() -> None:
    mobility = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 3.0], [3.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    dataset = _DummyMobilityDataset(run_id="real", preloaded_mobility=mobility)
    device = torch.device("cpu")

    first_batch = {"MobBatch": _MobBatch(global_t=[0], run_id="real")}
    inject_gpu_mobility(first_batch, dataset, device)
    first_adj = first_batch["MobBatch"].adj_dense

    second_batch = {"MobBatch": _MobBatch(global_t=[1], run_id="real")}
    inject_gpu_mobility(second_batch, dataset, device)
    second_adj = second_batch["MobBatch"].adj_dense

    assert float(first_adj[0, 0, 1]) == pytest.approx(1.0, abs=1e-3)
    assert float(second_adj[0, 0, 1]) == pytest.approx(3.0, abs=1e-3)
    assert hasattr(dataset, "_gpu_mobility_cache")
    assert len(dataset._gpu_mobility_cache) == 1


@pytest.mark.epiforecaster
def test_optimized_collate_graphs_attaches_homogeneous_run_id() -> None:
    item = {
        "mob_x": torch.ones(2, 2, 1, dtype=torch.float32),
        "mob_t": torch.tensor([0, 1], dtype=torch.long),
        "mob_target_node_idx": 0,
        "run_id": "synth_2",
    }
    mob_batch = optimized_collate_graphs([item, item])
    assert getattr(mob_batch, "run_id", None) == "synth_2"


@pytest.mark.epiforecaster
def test_ensure_mobility_adj_dense_ready_noop_when_not_required() -> None:
    batch_data = {"MobBatch": _MobBatch(global_t=[0], run_id="real")}
    ensure_mobility_adj_dense_ready(batch_data, required=False)


@pytest.mark.epiforecaster
def test_ensure_mobility_adj_dense_ready_raises_when_missing_adj_dense() -> None:
    batch_data = {"MobBatch": _MobBatch(global_t=[0], run_id="real")}
    with pytest.raises(ValueError, match="MobBatch.adj_dense"):
        ensure_mobility_adj_dense_ready(
            batch_data,
            required=True,
            context="compiled training",
        )


@pytest.mark.epiforecaster
def test_ensure_mobility_adj_dense_ready_passes_when_present() -> None:
    mob_batch = _MobBatch(global_t=[0], run_id="real")
    mob_batch.adj_dense = torch.ones(1, 2, 2, dtype=torch.float16)
    batch_data = {"MobBatch": mob_batch}
    ensure_mobility_adj_dense_ready(
        batch_data,
        required=True,
        context="compiled training",
    )
