from __future__ import annotations

from dataclasses import dataclass

import pytest
from torch.utils.data import ConcatDataset, Dataset

from data.samplers import EpidemicCurriculumSampler
from models.configs import CurriculumConfig, CurriculumPhaseConfig


@dataclass
class _DummySample:
    target: str
    window_start: int


class _OrderedDataset(Dataset):
    def __init__(
        self,
        *,
        run_id: str,
        samples: list[_DummySample],
        sparsity_level: float | None = None,
    ) -> None:
        self.run_id = run_id
        self.samples = samples
        self.sparsity_level = sparsity_level

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _DummySample:
        return self.samples[index]


def _node_major_samples(nodes: list[str], starts: list[int]) -> list[_DummySample]:
    return [_DummySample(target=node, window_start=start) for node in nodes for start in starts]


def _time_major_samples(nodes: list[str], starts: list[int]) -> list[_DummySample]:
    return [_DummySample(target=node, window_start=start) for start in starts for node in nodes]


def _dataset_index_for_global_index(concat: ConcatDataset, global_index: int) -> int:
    for dataset_idx, upper in enumerate(concat.cumulative_sizes):
        lower = 0 if dataset_idx == 0 else concat.cumulative_sizes[dataset_idx - 1]
        if lower <= global_index < upper:
            return dataset_idx
    raise AssertionError(f"Global index {global_index} out of range")


class TestEpidemicCurriculumSampler:
    @pytest.fixture
    def curriculum_config(self) -> CurriculumConfig:
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=0.75,
            min_sparsity=0.0,
            max_sparsity=1.0,
        )
        return CurriculumConfig(
            enabled=True,
            active_runs=2,
            chunk_size=5,
            run_sampling="round_robin",
            schedule=[phase],
        )

    @pytest.fixture
    def concat_dataset(self) -> ConcatDataset:
        real_ds = _OrderedDataset(
            run_id="real",
            samples=_node_major_samples(["r0", "r1"], list(range(6))),
        )
        synth_ds1 = _OrderedDataset(
            run_id="synth_1",
            samples=_node_major_samples(["s1"], list(range(12))),
            sparsity_level=0.8,
        )
        synth_ds2 = _OrderedDataset(
            run_id="synth_2",
            samples=_node_major_samples(["s2"], list(range(12))),
            sparsity_level=0.2,
        )
        return ConcatDataset([real_ds, synth_ds1, synth_ds2])

    def test_initialization(self, concat_dataset: ConcatDataset) -> None:
        config = CurriculumConfig(
            enabled=True, active_runs=1, chunk_size=10, run_sampling="round_robin"
        )

        sampler = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=4,
            config=config,
            real_run_id="real",
            seed=7,
        )

        assert sampler.real_dataset_indices == [0]
        assert sampler.synth_dataset_indices == [1, 2]

    def test_sparsity_filtering(self, concat_dataset: ConcatDataset) -> None:
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=0.5,
            min_sparsity=0.0,
            max_sparsity=0.5,
        )
        config = CurriculumConfig(enabled=True, schedule=[phase])
        sampler = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=4,
            config=config,
            real_run_id="real",
            seed=7,
        )

        sampler.set_curriculum(0)
        filtered = sampler._filter_runs_by_sparsity(
            sampler.synth_dataset_indices,
            sampler.state.min_sparsity,
            sampler.state.max_sparsity,
        )

        assert filtered == [2]

    def test_iteration_yields_homogeneous_batches(
        self,
        concat_dataset: ConcatDataset,
        curriculum_config: CurriculumConfig,
    ) -> None:
        sampler = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=3,
            config=curriculum_config,
            real_run_id="real",
            seed=11,
        )

        batches = list(sampler)
        assert batches
        for batch in batches:
            dataset_ids = {
                _dataset_index_for_global_index(concat_dataset, global_index)
                for global_index in batch
            }
            assert len(dataset_ids) == 1

    def test_iteration_never_mixes_datasets_across_awkward_chunk_boundaries(self) -> None:
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=0.5,
        )
        config = CurriculumConfig(
            enabled=True,
            active_runs=1,
            chunk_size=5,
            run_sampling="round_robin",
            schedule=[phase],
        )
        concat = ConcatDataset(
            [
                _OrderedDataset(
                    run_id="real",
                    samples=_node_major_samples(["r0"], list(range(11))),
                ),
                _OrderedDataset(
                    run_id="synth_1",
                    samples=_node_major_samples(["s0"], list(range(11))),
                    sparsity_level=0.3,
                ),
            ]
        )

        sampler = EpidemicCurriculumSampler(
            dataset=concat,
            batch_size=3,
            config=config,
            real_run_id="real",
            seed=5,
        )

        for batch in sampler:
            dataset_ids = {
                _dataset_index_for_global_index(concat, global_index)
                for global_index in batch
            }
            assert len(dataset_ids) == 1

    def test_node_major_ordering_is_preserved_within_batches(self) -> None:
        node_dataset = _OrderedDataset(
            run_id="real",
            samples=_node_major_samples(["n0", "n1"], [0, 1, 2, 3]),
        )
        concat = ConcatDataset([node_dataset])
        config = CurriculumConfig(enabled=True, active_runs=1, chunk_size=4)
        sampler = EpidemicCurriculumSampler(
            dataset=concat,
            batch_size=2,
            config=config,
            real_run_id="real",
            seed=3,
        )

        for batch in sampler:
            local_start = batch[0]
            expected = node_dataset.samples[local_start : local_start + len(batch)]
            observed = [node_dataset.samples[idx] for idx in batch]
            assert observed == expected

    def test_time_major_ordering_is_preserved_within_batches(self) -> None:
        time_dataset = _OrderedDataset(
            run_id="real",
            samples=_time_major_samples(["n0", "n1"], [0, 1, 2, 3]),
        )
        concat = ConcatDataset([time_dataset])
        config = CurriculumConfig(enabled=True, active_runs=1, chunk_size=4)
        sampler = EpidemicCurriculumSampler(
            dataset=concat,
            batch_size=2,
            config=config,
            real_run_id="real",
            seed=3,
        )

        for batch in sampler:
            local_start = batch[0]
            expected = time_dataset.samples[local_start : local_start + len(batch)]
            observed = [time_dataset.samples[idx] for idx in batch]
            assert observed == expected

    def test_num_batches_for_epoch_matches_iteration(
        self,
        concat_dataset: ConcatDataset,
        curriculum_config: CurriculumConfig,
    ) -> None:
        sampler = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=3,
            config=curriculum_config,
            real_run_id="real",
            seed=13,
        )

        assert sampler.num_batches_for_epoch(0) == len(list(sampler))
        assert len(sampler) == len(list(sampler))

    def test_fixed_seed_and_epoch_are_deterministic(
        self,
        concat_dataset: ConcatDataset,
        curriculum_config: CurriculumConfig,
    ) -> None:
        sampler_a = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=3,
            config=curriculum_config,
            real_run_id="real",
            seed=19,
        )
        sampler_b = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=3,
            config=curriculum_config,
            real_run_id="real",
            seed=19,
        )

        sampler_a.set_curriculum(2)
        sampler_b.set_curriculum(2)
        assert list(sampler_a) == list(sampler_b)

    def test_num_batches_for_epoch_uses_active_runs_exactly(
        self,
        concat_dataset: ConcatDataset,
    ) -> None:
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=0.5,
            min_sparsity=0.0,
            max_sparsity=1.0,
        )
        config = CurriculumConfig(
            enabled=True,
            active_runs=2,
            chunk_size=10,
            run_sampling="round_robin",
            schedule=[phase],
        )
        sampler = EpidemicCurriculumSampler(
            dataset=concat_dataset,
            batch_size=10,
            config=config,
            real_run_id="real",
            seed=23,
        )

        assert sampler.num_batches_for_epoch(0) == len(list(sampler))
        assert len(sampler) == len(list(sampler))

    def test_single_dataset_fallback(self) -> None:
        ds = _OrderedDataset(
            run_id="real",
            samples=_node_major_samples(["r0"], list(range(10))),
        )
        sampler = EpidemicCurriculumSampler(
            dataset=ds,  # type: ignore[arg-type]
            batch_size=4,
            config=CurriculumConfig(enabled=True),
            seed=29,
        )

        assert sampler.real_dataset_indices == [0]
        assert sampler.synth_dataset_indices == []

    def test_active_runs_minus_one_uses_all_synth_runs(self) -> None:
        """active_runs=-1 should select all available synthetic runs."""
        synth_datasets = [
            _OrderedDataset(
                run_id=f"{i}_Baseline",
                samples=_node_major_samples([f"s{i}"], list(range(6))),
                sparsity_level=0.1 * i,
            )
            for i in range(5)
        ]
        concat = ConcatDataset(synth_datasets)

        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=1.0,
        )
        config = CurriculumConfig(
            enabled=True,
            active_runs=-1,
            chunk_size=6,
            run_sampling="round_robin",
            schedule=[phase],
        )
        sampler = EpidemicCurriculumSampler(
            dataset=concat,
            batch_size=3,
            config=config,
            real_run_id="real",
            seed=37,
        )

        # All datasets are synthetic (no real run_id match)
        assert sampler.real_dataset_indices == []
        assert len(sampler.synth_dataset_indices) == 5

        # _resolve_active_synth_indices should return ALL 5 synth indices
        active = sampler._resolve_active_synth_indices(epoch=0)
        assert len(active) == 5

    def test_active_runs_minus_one_random_sampling(self) -> None:
        """active_runs=-1 with random sampling also uses all runs."""
        synth_datasets = [
            _OrderedDataset(
                run_id=f"{i}_Baseline",
                samples=_node_major_samples([f"s{i}"], list(range(6))),
            )
            for i in range(4)
        ]
        concat = ConcatDataset(synth_datasets)

        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=1.0,
        )
        config = CurriculumConfig(
            enabled=True,
            active_runs=-1,
            chunk_size=6,
            run_sampling="random",
            schedule=[phase],
        )
        sampler = EpidemicCurriculumSampler(
            dataset=concat,
            batch_size=3,
            config=config,
            real_run_id="real",
            seed=41,
        )

        active = sampler._resolve_active_synth_indices(epoch=0)
        assert len(active) == 4
