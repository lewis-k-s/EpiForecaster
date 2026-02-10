import pytest
from unittest.mock import MagicMock
from torch.utils.data import ConcatDataset
from data.samplers import EpidemicCurriculumSampler
from models.configs import CurriculumConfig, CurriculumPhaseConfig


class TestEpidemicCurriculumSampler:
    @pytest.fixture
    def mock_concat_dataset(self):
        # Create mock sub-datasets without spec to avoid AttributeError
        real_ds = MagicMock()
        real_ds.run_id = "real"
        real_ds.__len__.return_value = 100
        # Set sparsity_level to None so it is ignored by sampler logic
        real_ds.sparsity_level = None

        synth_ds1 = MagicMock()
        synth_ds1.run_id = "synth_1"
        synth_ds1.sparsity_level = 0.8
        synth_ds1.__len__.return_value = 100

        synth_ds2 = MagicMock()
        synth_ds2.run_id = "synth_2"
        synth_ds2.sparsity_level = 0.2
        synth_ds2.__len__.return_value = 100

        # ConcatDataset mocks attributes
        concat = MagicMock(spec=ConcatDataset)
        concat.datasets = [real_ds, synth_ds1, synth_ds2]
        concat.cumulative_sizes = [100, 200, 300]
        concat.__len__.return_value = 300

        return concat

    def test_initialization(self, mock_concat_dataset):
        config = CurriculumConfig(
            enabled=True, active_runs=1, chunk_size=10, run_sampling="round_robin"
        )

        sampler = EpidemicCurriculumSampler(
            dataset=mock_concat_dataset,
            batch_size=32,
            config=config,
            real_run_id="real",
        )

        assert len(sampler.real_dataset_indices) == 1
        assert len(sampler.synth_dataset_indices) == 2
        assert sampler.real_dataset_indices[0] == 0

    def test_sparsity_filtering(self, mock_concat_dataset):
        # Phase with max_sparsity=0.5 -> should filter out synth_1 (0.8)
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=10,
            synth_ratio=0.5,
            min_sparsity=0.0,
            max_sparsity=0.5,
        )
        config = CurriculumConfig(enabled=True, schedule=[phase])

        sampler = EpidemicCurriculumSampler(
            dataset=mock_concat_dataset,
            batch_size=32,
            config=config,
            real_run_id="real",
        )

        # Determine active runs for epoch 0
        sampler.set_curriculum(0)

        # Check filtering logic
        filtered = sampler._filter_runs_by_sparsity(
            sampler.synth_dataset_indices,
            sampler.state.min_sparsity,
            sampler.state.max_sparsity,
        )

        # Should only have synth_2 available
        assert len(filtered) == 1
        assert filtered[0] == 2  # Index of synth_2

    def test_iteration(self, mock_concat_dataset):
        config = CurriculumConfig(enabled=True)
        sampler = EpidemicCurriculumSampler(
            dataset=mock_concat_dataset, batch_size=10, config=config
        )

        # Just iterate and check we get indices
        batches = list(sampler)
        assert len(batches) > 0
        assert isinstance(batches[0], list)

    def test_single_dataset_fallback(self):
        # Test fallback when not ConcatDataset
        ds = MagicMock()
        ds.__len__.return_value = 100
        ds.run_id = "real"

        config = CurriculumConfig(enabled=True)
        sampler = EpidemicCurriculumSampler(ds, 32, config)

        assert len(sampler.real_dataset_indices) == 1
        assert len(sampler.synth_dataset_indices) == 0
