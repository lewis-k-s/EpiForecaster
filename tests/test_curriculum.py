"""Unit tests for curriculum training sparsity mapping functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from models.configs import (
    CurriculumConfig,
    DataConfig,
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    OutputConfig,
    TrainingParams,
)


def create_mock_config(dataset_path: str) -> EpiForecasterConfig:
    """Create a minimal config for testing."""
    return EpiForecasterConfig(
        data=DataConfig(
            dataset_path=dataset_path,
            real_dataset_path="",
        ),
        model=ModelConfig(
            type=ModelVariant(),
            mobility_embedding_dim=0,
            region_embedding_dim=0,
            history_length=4,
            forecast_horizon=4,
            max_neighbors=0,
        ),
        training=TrainingParams(
            batch_size=32,
            learning_rate=0.001,
            epochs=1,
            curriculum=CurriculumConfig(),
        ),
        output=OutputConfig(
            log_dir="/tmp/test_logs",
            experiment_name="test_experiment",
        ),
    )


class TestLoadSparsityMapping:
    """Tests for _load_sparsity_mapping method."""

    def test_load_sparsity_mapping_from_processed_dataset(self, tmp_path: Path):
        """Test that sparsity is correctly loaded from the processed dataset."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        # Create a sample zarr dataset with sparsity data
        dataset_path = tmp_path / "test_dataset.zarr"
        run_ids = np.array(["real_run", "synth_0.05", "synth_0.20", "synth_0.80"])
        sparsity_levels = np.array([0.0, 0.05, 0.20, 0.80])

        ds = xr.Dataset(
            {
                "cases": (
                    ["run_id", "region", "time"],
                    np.random.rand(len(run_ids), 10, 100),
                ),
                "synthetic_sparsity_level": (["run_id"], sparsity_levels),
            },
            coords={"run_id": run_ids, "region": range(10), "time": range(100)},
        )
        ds.to_zarr(str(dataset_path), mode="w")

        mock_config = create_mock_config(str(dataset_path))

        # Mock the __init__ to skip dataset loading
        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        mapping = trainer._load_sparsity_mapping()

        assert mapping == {
            "real_run": 0.0,
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
            "synth_0.80": 0.80,
        }

    def test_load_sparsity_mapping_missing_variable(self, tmp_path: Path):
        """Test graceful handling when synthetic_sparsity_level is missing."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        dataset_path = tmp_path / "test_dataset_no_sparsity.zarr"
        run_ids = np.array(["real_run", "synth_1"])

        ds = xr.Dataset(
            {
                "cases": (
                    ["run_id", "region", "time"],
                    np.random.rand(len(run_ids), 10, 100),
                ),
            },
            coords={"run_id": run_ids, "region": range(10), "time": range(100)},
        )
        ds.to_zarr(str(dataset_path), mode="w")

        mock_config = create_mock_config(str(dataset_path))

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        mapping = trainer._load_sparsity_mapping()

        assert mapping == {}

    def test_load_sparsity_mapping_nonexistent_dataset(self):
        """Test handling of non-existent dataset paths."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = create_mock_config("/nonexistent/path/to/dataset.zarr")

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        mapping = trainer._load_sparsity_mapping()

        assert mapping == {}


class TestSelectRunsForCurriculum:
    """Tests for _select_runs_for_curriculum method."""

    def test_select_runs_for_curriculum_with_sparsity(self):
        """Test run selection for sparsity diversity."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()
        mock_config.training.curriculum.enabled = False

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = [
            "synth_0.05",
            "synth_0.20",
            "synth_0.40",
            "synth_0.60",
            "synth_0.80",
            "synth_0.81",
        ]
        sparsity_map = {
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
            "synth_0.40": 0.40,
            "synth_0.60": 0.60,
            "synth_0.80": 0.80,
            "synth_0.81": 0.81,
        }

        # Set seed for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)

        selected = trainer._select_runs_for_curriculum(synth_runs, sparsity_map)

        # Should select at least one from each bucket
        assert len(selected) <= 5
        assert all(run in synth_runs for run in selected)

    def test_select_runs_for_curriculum_without_sparsity(self):
        """Test fallback to random selection when no sparsity data."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = ["synth_1", "synth_2", "synth_3", "synth_4", "synth_5"]
        sparsity_map: dict[str, float] = {}

        # Set seed for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)

        selected = trainer._select_runs_for_curriculum(synth_runs, sparsity_map)

        # Should return first max_runs (default 5) when no sparsity data
        assert len(selected) == min(5, len(synth_runs))

    def test_select_runs_for_curriculum_partial_sparsity(self):
        """Test with some runs having sparsity data."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = ["synth_0.05", "synth_0.80", "no_sparsity_1", "no_sparsity_2"]
        sparsity_map = {
            "synth_0.05": 0.05,
            "synth_0.80": 0.80,
        }

        # Set seed for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)

        selected = trainer._select_runs_for_curriculum(
            synth_runs, sparsity_map, max_runs=3
        )

        assert len(selected) <= 3
        assert all(run in synth_runs for run in selected)

    def test_select_runs_for_curriculum_max_runs_limit(self):
        """Test that max_runs limit is respected."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = [
            "synth_0.05",
            "synth_0.20",
            "synth_0.40",
            "synth_0.60",
            "synth_0.80",
        ]
        sparsity_map = {
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
            "synth_0.40": 0.40,
            "synth_0.60": 0.60,
            "synth_0.80": 0.80,
        }

        selected = trainer._select_runs_for_curriculum(
            synth_runs, sparsity_map, max_runs=2
        )

        assert len(selected) <= 2


class TestSelectSyntheticScalerRun:
    """Tests for _select_synthetic_scaler_run method."""

    def test_select_scaler_run_with_sparsity(self):
        """Test selecting lowest sparsity run for scaler fitting."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = ["synth_0.05", "synth_0.20", "synth_0.80"]
        trainer._load_sparsity_mapping = lambda: {
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
            "synth_0.80": 0.80,
        }

        selected = trainer._select_synthetic_scaler_run(synth_runs)

        assert selected == "synth_0.05"

    def test_select_scaler_run_without_sparsity(self):
        """Test fallback when no sparsity metadata available."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        synth_runs = ["synth_1", "synth_2", "synth_3"]
        trainer._load_sparsity_mapping = lambda: {}

        selected = trainer._select_synthetic_scaler_run(synth_runs)

        # Should return the last run as fallback
        assert selected == "synth_3"

    def test_select_scaler_run_empty_list(self):
        """Test error handling when no synthetic runs available."""
        from training.epiforecaster_trainer import EpiForecasterTrainer

        mock_config = MagicMock()

        with patch.object(EpiForecasterTrainer, "__init__", return_value=None):
            trainer = EpiForecasterTrainer(mock_config)
            trainer.config = mock_config

        with pytest.raises(ValueError, match="No synthetic runs available"):
            trainer._select_synthetic_scaler_run([])
