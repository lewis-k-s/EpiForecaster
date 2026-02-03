"""Tests for curriculum sampler sparsity functionality."""

from unittest.mock import MagicMock

import pytest

from models.configs import CurriculumConfig, CurriculumPhaseConfig


@pytest.mark.epiforecaster
class TestCurriculumPhaseConfig:
    """Tests for CurriculumPhaseConfig with sparsity validation."""

    def test_valid_sparsity_range(self):
        """Test that valid sparsity ranges are accepted."""
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=5,
            synth_ratio=0.8,
            min_sparsity=0.0,
            max_sparsity=0.5,
        )
        assert phase.min_sparsity == 0.0
        assert phase.max_sparsity == 0.5

    def test_min_equals_max_sparsity(self):
        """Test that min_sparsity == max_sparsity is valid."""
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=5,
            synth_ratio=0.8,
            min_sparsity=0.2,
            max_sparsity=0.2,
        )
        assert phase.min_sparsity == 0.2
        assert phase.max_sparsity == 0.2

    def test_none_sparsity_bounds(self):
        """Test that None sparsity bounds are valid (no filtering)."""
        phase = CurriculumPhaseConfig(
            start_epoch=0,
            end_epoch=5,
            synth_ratio=0.8,
            min_sparsity=None,
            max_sparsity=None,
        )
        assert phase.min_sparsity is None
        assert phase.max_sparsity is None

    def test_invalid_min_greater_than_max(self):
        """Test that min_sparsity > max_sparsity raises ValueError."""
        with pytest.raises(ValueError, match="min_sparsity.*> max_sparsity"):
            CurriculumPhaseConfig(
                start_epoch=0,
                end_epoch=5,
                synth_ratio=0.8,
                min_sparsity=0.8,
                max_sparsity=0.2,
            )

    def test_invalid_sparsity_below_zero(self):
        """Test that sparsity < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Sparsity must be in"):
            CurriculumPhaseConfig(
                start_epoch=0,
                end_epoch=5,
                synth_ratio=0.8,
                min_sparsity=-0.1,
            )

    def test_invalid_sparsity_above_one(self):
        """Test that sparsity > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Sparsity must be in"):
            CurriculumPhaseConfig(
                start_epoch=0,
                end_epoch=5,
                synth_ratio=0.8,
                max_sparsity=1.5,
            )


@pytest.mark.epiforecaster
class TestSparsityFiltering:
    """Tests for sparsity filtering logic in EpidemicCurriculumSampler."""

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler with sparsity data for testing."""
        from data.samplers import EpidemicCurriculumSampler

        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.datasets = []
        mock_dataset.cumulative_sizes = []

        # Create config
        config = CurriculumConfig(
            enabled=True,
            active_runs=2,
            chunk_size=512,
        )

        sampler = EpidemicCurriculumSampler(
            dataset=mock_dataset,
            batch_size=16,
            config=config,
        )

        # Manually set up sparsity data
        sampler._dataset_sparsity = {
            0: 0.05,  # Low sparsity
            1: 0.20,  # Low-medium
            2: 0.40,  # Medium
            3: 0.60,  # Medium-high
            4: 0.80,  # High
            5: None,  # Unknown
        }

        return sampler

    def test_filter_no_bounds(self, mock_sampler):
        """Test that no filtering occurs when bounds are None."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, None, None)
        assert filtered == indices

    def test_filter_min_only(self, mock_sampler):
        """Test filtering with only minimum sparsity."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, 0.3, None)
        # Should include 0.40, 0.60, 0.80 (indices 2, 3, 4)
        # Exclude 0.05, 0.20, and None (indices 0, 1, 5)
        assert filtered == [2, 3, 4]

    def test_filter_max_only(self, mock_sampler):
        """Test filtering with only maximum sparsity."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, None, 0.3)
        # Should include 0.05, 0.20 (indices 0, 1)
        # Exclude None and higher values (indices 2, 3, 4, 5)
        assert filtered == [0, 1]

    def test_filter_both_bounds(self, mock_sampler):
        """Test filtering with both min and max sparsity."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, 0.15, 0.5)
        # Should include 0.20, 0.40 (indices 1, 2)
        # Exclude 0.05, 0.60, 0.80, and None
        assert filtered == [1, 2]

    def test_filter_excludes_unknown(self, mock_sampler):
        """Test that runs with unknown sparsity are excluded when filtering."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, 0.0, 1.0)
        # Should exclude index 5 (None sparsity)
        assert filtered == [0, 1, 2, 3, 4]

    def test_filter_empty_result(self, mock_sampler):
        """Test that filtering can result in empty list."""
        indices = [0, 1, 2, 3, 4, 5]
        filtered = mock_sampler._filter_runs_by_sparsity(indices, 0.9, 1.0)
        # No runs with sparsity in this range
        assert filtered == []

    def test_filter_inclusive_bounds(self, mock_sampler):
        """Test that bounds are inclusive."""
        indices = [0, 1, 2, 3, 4, 5]
        # Filter for exactly 0.20 and 0.40
        filtered = mock_sampler._filter_runs_by_sparsity(indices, 0.2, 0.4)
        # Should include 0.20 and 0.40 (inclusive)
        assert filtered == [1, 2]


@pytest.mark.epiforecaster
class TestCurriculumConfig:
    """Tests for CurriculumConfig with sparsity curriculum schedule."""

    def test_curriculum_config_with_schedule(self):
        """Test CurriculumConfig with schedule including sparsity bounds."""
        config = CurriculumConfig(
            enabled=True,
            active_runs=2,
            chunk_size=512,
            schedule=[
                CurriculumPhaseConfig(
                    start_epoch=0,
                    end_epoch=5,
                    synth_ratio=1.0,
                    min_sparsity=0.0,
                    max_sparsity=0.1,
                ),
                CurriculumPhaseConfig(
                    start_epoch=5,
                    end_epoch=10,
                    synth_ratio=0.5,
                    min_sparsity=0.1,
                    max_sparsity=0.5,
                ),
            ],
        )
        assert len(config.schedule) == 2
        assert config.schedule[0].max_sparsity == 0.1
        assert config.schedule[1].min_sparsity == 0.1
