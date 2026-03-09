"""Unit tests for curriculum training sparsity mapping functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from data.curriculum_builder import (
    _select_synthetic_scaler_run,
    load_sparsity_mapping,
    select_runs_by_sparsity,
)
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
            input_window_length=4,
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
    """Tests for load_sparsity_mapping function."""

    def test_load_sparsity_mapping_from_processed_dataset(self, tmp_path: Path):
        """Test that sparsity is correctly loaded from the processed dataset."""
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

        mapping = load_sparsity_mapping(dataset_path)

        assert mapping == {
            "real_run": 0.0,
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
            "synth_0.80": 0.80,
        }

    def test_load_sparsity_mapping_missing_variable(self, tmp_path: Path):
        """Test graceful handling when synthetic_sparsity_level is missing."""
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

        mapping = load_sparsity_mapping(dataset_path)

        assert mapping == {}

    def test_load_sparsity_mapping_nonexistent_dataset(self):
        """Test handling of non-existent dataset paths."""
        mapping = load_sparsity_mapping(Path("/nonexistent/path/to/dataset.zarr"))

        assert mapping == {}


class TestSelectRunsForCurriculum:
    """Tests for select_runs_by_sparsity function."""

    def test_select_runs_for_curriculum_with_sparsity(self):
        """Test run selection for sparsity diversity."""
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

        import random

        random.seed(42)
        np.random.seed(42)

        selected = select_runs_by_sparsity(synth_runs, sparsity_map, max_runs=5)

        assert len(selected) <= 5
        assert all(run in synth_runs for run in selected)

    def test_select_runs_for_curriculum_without_sparsity(self):
        """Test fallback to random selection when no sparsity data."""
        synth_runs = ["synth_1", "synth_2", "synth_3", "synth_4", "synth_5"]
        sparsity_map: dict[str, float] = {}

        import random

        random.seed(42)
        np.random.seed(42)

        selected = select_runs_by_sparsity(synth_runs, sparsity_map, max_runs=5)

        assert selected == synth_runs[:5]

    def test_select_runs_for_curriculum_max_runs(self):
        """Test that max_runs is respected."""
        synth_runs = ["synth_1", "synth_2", "synth_3", "synth_4", "synth_5", "synth_6"]
        sparsity_map: dict[str, float] = {}

        selected = select_runs_by_sparsity(synth_runs, sparsity_map, max_runs=3)

        assert len(selected) == 3
        assert selected == synth_runs[:3]


class TestSelectSyntheticScalerRun:
    """Tests for _select_synthetic_scaler_run function."""

    def test_select_scaler_run_with_sparsity(self):
        """Test that lowest sparsity run is selected for scaler fitting."""
        synth_runs = ["synth_0.80", "synth_0.05", "synth_0.20"]
        sparsity_map = {
            "synth_0.80": 0.80,
            "synth_0.05": 0.05,
            "synth_0.20": 0.20,
        }

        with patch(
            "data.curriculum_builder.load_sparsity_mapping", return_value=sparsity_map
        ):
            selected = _select_synthetic_scaler_run(synth_runs, Path("/fake/path"))

        assert selected == "synth_0.05"

    def test_select_scaler_run_without_sparsity(self):
        """Test fallback when no sparsity metadata is available."""
        synth_runs = ["synth_1", "synth_2", "synth_3"]

        with patch("data.curriculum_builder.load_sparsity_mapping", return_value={}):
            selected = _select_synthetic_scaler_run(synth_runs, Path("/fake/path"))

        assert selected == "synth_3"

    def test_select_scaler_run_empty_list(self):
        """Test error handling for empty run list."""
        with pytest.raises(ValueError, match="No synthetic runs available"):
            _select_synthetic_scaler_run([], Path("/fake/path"))
