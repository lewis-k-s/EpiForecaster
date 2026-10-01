"""Curriculum sampling contracts that affect run balance and batch ordering."""

from dataclasses import dataclass

import pytest
from torch.utils.data import ConcatDataset, Dataset

from data.samplers import EpidemicCurriculumSampler
from models.configs import CurriculumConfig, CurriculumPhaseConfig


@dataclass
class _Sample:
    target: str
    window_start: int


class _Run(Dataset):
    def __init__(self, run_id: str, samples: list[_Sample], sparsity: float = 0.0):
        self.run_id = run_id
        self.samples = samples
        self.sparsity_level = sparsity

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _Sample:
        return self.samples[index]


def _samples(
    nodes: list[str], starts: range, *, time_major: bool = False
) -> list[_Sample]:
    pairs = (
        ((node, start) for start in starts for node in nodes)
        if time_major
        else ((node, start) for node in nodes for start in starts)
    )
    return [_Sample(node, start) for node, start in pairs]


def _run_index(dataset: ConcatDataset, index: int) -> int:
    return next(i for i, upper in enumerate(dataset.cumulative_sizes) if index < upper)


def _config(
    *, active_runs: int = 2, chunk_size: int = 5, run_sampling: str = "round_robin"
) -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        active_runs=active_runs,
        chunk_size=chunk_size,
        run_sampling=run_sampling,
        schedule=[
            CurriculumPhaseConfig(
                start_epoch=0,
                end_epoch=10,
                synth_ratio=0.5,
                min_sparsity=0.0,
                max_sparsity=0.5,
            )
        ],
    )


def test_curriculum_filters_runs_and_keeps_awkward_chunks_homogeneous() -> None:
    dataset = ConcatDataset(
        [
            _Run("real", _samples(["r0"], range(11))),
            _Run("synth_1", _samples(["s1"], range(11)), sparsity=0.8),
            _Run("synth_2", _samples(["s2"], range(11)), sparsity=0.2),
        ]
    )
    config = _config(active_runs=2)
    sampler = EpidemicCurriculumSampler(
        dataset=dataset, batch_size=3, config=config, real_run_id="real", seed=11
    )
    peer = EpidemicCurriculumSampler(
        dataset=dataset, batch_size=3, config=config, real_run_id="real", seed=11
    )
    sampler.set_curriculum(2)
    peer.set_curriculum(2)

    assert sampler.real_dataset_indices == [0]
    assert sampler.synth_dataset_indices == [1, 2]
    assert sampler._filter_runs_by_sparsity([1, 2], 0.0, 0.5) == [2]
    batches = list(sampler)
    assert batches == list(peer)
    assert batches
    assert sampler.num_batches_for_epoch(2) == len(batches)
    assert len(sampler) == len(batches)
    assert all(
        len({_run_index(dataset, index) for index in batch}) == 1 for batch in batches
    )


@pytest.mark.parametrize("time_major", [False, True])
def test_curriculum_preserves_source_order_within_batches(time_major: bool) -> None:
    run = _Run("real", _samples(["n0", "n1"], range(4), time_major=time_major))
    sampler = EpidemicCurriculumSampler(
        dataset=ConcatDataset([run]),
        batch_size=2,
        config=_config(active_runs=1, chunk_size=4),
        real_run_id="real",
        seed=3,
    )
    for batch in sampler:
        assert [run.samples[index] for index in batch] == run.samples[
            batch[0] : batch[0] + len(batch)
        ]


def test_curriculum_accepts_a_single_real_dataset() -> None:
    run = _Run("real", _samples(["r0"], range(10)))
    sampler = EpidemicCurriculumSampler(
        dataset=run,  # type: ignore[arg-type]
        batch_size=4,
        config=CurriculumConfig(enabled=True),
        seed=29,
    )
    assert sampler.real_dataset_indices == [0]
    assert sampler.synth_dataset_indices == []


@pytest.mark.parametrize("run_sampling", ["round_robin", "random"])
def test_active_runs_minus_one_selects_every_synthetic_run(run_sampling: str) -> None:
    dataset = ConcatDataset(
        [_Run(f"synth_{i}", _samples([f"s{i}"], range(6))) for i in range(5)]
    )
    sampler = EpidemicCurriculumSampler(
        dataset=dataset,
        batch_size=3,
        config=_config(active_runs=-1, chunk_size=6, run_sampling=run_sampling),
        real_run_id="real",
        seed=37,
    )
    assert sampler.real_dataset_indices == []
    assert sorted(sampler._resolve_active_synth_indices(epoch=0)) == list(range(5))
