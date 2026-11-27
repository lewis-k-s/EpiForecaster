"""Curriculum sampling contracts that affect run balance and batch ordering."""

from collections import Counter
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

    def sample_window_starts(self) -> list[int]:
        return [sample.window_start for sample in self.samples]


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


def test_window_start_batches_are_full_single_start_and_cover_every_sample() -> None:
    dataset = ConcatDataset(
        [
            _Run("synth_1", _samples(list("abcde"), range(3), time_major=True)),
            _Run("synth_2", _samples(list("ab"), range(4), time_major=True)),
        ]
    )
    config = CurriculumConfig(
        enabled=True,
        active_runs=-1,
        window_start_batches=True,
        schedule=[CurriculumPhaseConfig(start_epoch=0, end_epoch=10, synth_ratio=1.0)],
    )
    sampler = EpidemicCurriculumSampler(
        dataset=dataset,
        batch_size=4,
        config=config,
        drop_last=True,
        real_run_id="",
        seed=3,
    )
    sampler.set_curriculum(0)

    batches = list(sampler)

    seen: set[int] = set()
    for batch in batches:
        assert len(batch) == 4
        assert len({_run_index(dataset, index) for index in batch}) == 1
        assert len({dataset[index].window_start for index in batch}) == 1
        seen.update(batch)
    assert seen == set(range(len(dataset)))
    # 3 starts of 5 samples -> 2 batches each; 4 starts of 2 samples -> 1 each.
    assert len(batches) == sampler.num_batches_for_epoch(0) == 10


def _blocked_dataset(*, with_real: bool) -> ConcatDataset:
    runs = [_Run(f"synth_{i}", _samples([f"s{i}"], range(7 + 3 * i))) for i in range(7)]
    if with_real:
        runs.insert(0, _Run("real", _samples(["r0"], range(40))))
    return ConcatDataset(runs)


def _blocked_sampler(
    dataset: ConcatDataset, *, run_block_size: int, synth_ratio: float, seed: int = 5
) -> EpidemicCurriculumSampler:
    config = CurriculumConfig(
        enabled=True,
        active_runs=-1,
        chunk_size=5,
        run_block_size=run_block_size,
        schedule=[
            CurriculumPhaseConfig(start_epoch=0, end_epoch=10, synth_ratio=synth_ratio)
        ],
    )
    return EpidemicCurriculumSampler(
        dataset=dataset,
        batch_size=2,
        config=config,
        real_run_id="real",
        seed=seed,
    )


def test_blocked_epoch_covers_every_sample_once_and_confines_runs_to_blocks() -> None:
    dataset = _blocked_dataset(with_real=False)
    unblocked = _blocked_sampler(dataset, run_block_size=100, synth_ratio=1.0)
    blocked = _blocked_sampler(dataset, run_block_size=3, synth_ratio=1.0)

    plan = blocked.plan_epoch_blocks(0)
    assert [len(block.synth_dataset_indices) for block in plan] == [3, 3, 1]
    assert sorted(i for block in plan for i in block.synth_dataset_indices) == list(
        range(7)
    )
    for block in plan:
        runs = {_run_index(dataset, i) for batch in block.batches for i in batch}
        assert runs == set(block.synth_dataset_indices)

    batches = list(blocked)
    assert batches == [batch for block in plan for batch in block.batches]
    assert Counter(i for b in batches for i in b) == Counter(
        i for b in unblocked.plan_epoch_blocks(0)[0].batches for i in b
    )
    assert blocked.num_batches_for_epoch(0) == unblocked.num_batches_for_epoch(0)


def test_blocked_epoch_is_deterministic_and_changes_blocks_per_epoch() -> None:
    dataset = _blocked_dataset(with_real=False)
    sampler = _blocked_sampler(dataset, run_block_size=3, synth_ratio=1.0)
    peer = _blocked_sampler(dataset, run_block_size=3, synth_ratio=1.0)

    first = sampler.plan_epoch_blocks(0)
    assert [b.batches for b in first] == [b.batches for b in peer.plan_epoch_blocks(0)]
    second = sampler.plan_epoch_blocks(1)
    assert [b.synth_dataset_indices for b in first] != [
        b.synth_dataset_indices for b in second
    ]


def test_blocked_epoch_spreads_real_budget_over_blocks() -> None:
    dataset = _blocked_dataset(with_real=True)
    unblocked = _blocked_sampler(dataset, run_block_size=100, synth_ratio=0.5)
    blocked = _blocked_sampler(dataset, run_block_size=3, synth_ratio=0.5)

    plan = blocked.plan_epoch_blocks(0)
    assert len(plan) == 3
    real_total = 0
    for block in plan:
        runs = [_run_index(dataset, i) for batch in block.batches for i in batch]
        real_runs = [run for run in runs if run == 0]
        assert real_runs, "every block carries real data"
        assert set(runs) - {0} <= set(block.synth_dataset_indices)
        real_total += len(real_runs)
    unblocked_runs = [
        _run_index(dataset, i)
        for batch in unblocked.plan_epoch_blocks(0)[0].batches
        for i in batch
    ]
    assert real_total == unblocked_runs.count(0)
    assert blocked.num_batches_for_epoch(0) == unblocked.num_batches_for_epoch(0)


def test_one_block_keeps_the_resolved_run_order() -> None:
    dataset = _blocked_dataset(with_real=True)
    sampler = _blocked_sampler(dataset, run_block_size=7, synth_ratio=0.5)
    plan = sampler.plan_epoch_blocks(3)
    assert len(plan) == 1
    assert plan[0].synth_dataset_indices == sampler._resolve_active_synth_indices()


def test_resident_dataset_limit_is_block_runs_plus_real() -> None:
    dataset = _blocked_dataset(with_real=True)
    assert (
        _blocked_sampler(
            dataset, run_block_size=3, synth_ratio=0.5
        ).resident_dataset_limit
        == 4
    )
    assert (
        _blocked_sampler(
            dataset, run_block_size=50, synth_ratio=0.5
        ).resident_dataset_limit
        == 8
    )
    assert (
        _blocked_sampler(
            _blocked_dataset(with_real=False), run_block_size=2, synth_ratio=1.0
        ).resident_dataset_limit
        == 2
    )


class _PlanRecorder:
    def __init__(self) -> None:
        self.plans: list[tuple[list[list[Dataset]], int]] = []

    def begin_epoch(self, blocks: list[list[Dataset]], epoch: int) -> None:
        self.plans.append((blocks, epoch))


def test_iteration_publishes_the_block_plan_to_the_pipeline() -> None:
    dataset = _blocked_dataset(with_real=True)
    sampler = _blocked_sampler(dataset, run_block_size=3, synth_ratio=0.5)
    recorder = _PlanRecorder()
    sampler.block_pipeline = recorder
    sampler.set_curriculum(0)
    iterator = iter(sampler)
    assert not recorder.plans  # published when the first batch is requested
    next(iterator)
    plan = sampler.plan_epoch_blocks(0)
    assert recorder.plans == [
        (
            [
                [dataset.datasets[i] for i in block.synth_dataset_indices]
                for block in plan
            ],
            0,
        )
    ]
