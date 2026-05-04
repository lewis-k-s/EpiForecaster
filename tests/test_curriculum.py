"""Unit tests for curriculum training sparsity mapping functionality."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from data.curriculum_builder import (
    _select_synthetic_scaler_run,
    build_curriculum_datasets,
    discover_runs,
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


class TestDiscoverRunsSynthOnly:
    """Tests for discover_runs() synth-only detection."""

    def _write_zarr_v2(self, ds: xr.Dataset, path: Path) -> None:
        """Write dataset in zarr v2 format for compatibility with discover_available_runs."""
        ds.to_zarr(str(path), mode="w", zarr_format=2)

    def test_synth_only_returns_empty_real_run(self, tmp_path: Path):
        """When no 'real' run exists, discover_runs returns ('', all_synth_runs)."""
        dataset_path = tmp_path / "synth_only.zarr"
        synth_runs = [f"{i}_Baseline" for i in range(5)]
        run_ids = np.array(synth_runs)
        ds = xr.Dataset(
            {
                "cases": (
                    ["run_id", "region", "time"],
                    np.random.rand(len(run_ids), 3, 20),
                ),
            },
            coords={"run_id": run_ids, "region": range(3), "time": range(20)},
        )
        self._write_zarr_v2(ds, dataset_path)

        config = create_mock_config(str(dataset_path))
        real_run, discovered_synth = discover_runs(config)

        assert real_run == ""
        assert len(discovered_synth) == 5
        assert all(r in discovered_synth for r in synth_runs)

    def test_synth_only_no_run_limiting(self, tmp_path: Path):
        """Synth-only mode should not limit runs to MAX_SYNTH_RUNS."""
        dataset_path = tmp_path / "many_synth.zarr"
        synth_runs = [f"{i}_Baseline" for i in range(19)]
        run_ids = np.array(synth_runs)
        ds = xr.Dataset(
            {
                "cases": (
                    ["run_id", "region", "time"],
                    np.random.rand(len(run_ids), 3, 20),
                ),
            },
            coords={"run_id": run_ids, "region": range(3), "time": range(20)},
        )
        self._write_zarr_v2(ds, dataset_path)

        config = create_mock_config(str(dataset_path))
        real_run, discovered_synth = discover_runs(config)

        assert real_run == ""
        assert len(discovered_synth) == 19

    def test_real_and_synth_returns_real_run(self, tmp_path: Path):
        """When 'real' run exists, discover_runs returns normal tuple."""
        dataset_path = tmp_path / "real_and_synth.zarr"
        run_ids = np.array(["real", "0_Baseline", "1_Baseline"])
        ds = xr.Dataset(
            {
                "cases": (
                    ["run_id", "region", "time"],
                    np.random.rand(len(run_ids), 3, 20),
                ),
            },
            coords={"run_id": run_ids, "region": range(3), "time": range(20)},
        )
        self._write_zarr_v2(ds, dataset_path)

        config = create_mock_config(str(dataset_path))
        real_run, discovered_synth = discover_runs(config)

        assert real_run == "real"
        assert len(discovered_synth) == 2

    def test_synth_only_uses_scaler_run_as_reference(self, tmp_path: Path):
        """Synth-only eval splits should share tensors from the same run they load."""

        class FakeDataset:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.run_id = kwargs["run_id"]
                self.biomarker_preprocessor = f"bio-{self.run_id}"
                self.mobility_preprocessor = f"mob-{self.run_id}"
                self.preloaded_mobility = f"mobility-{self.run_id}"
                self.mobility_mask = f"mask-{self.run_id}"
                self.shared_sparse_topology = f"topology-{self.run_id}"
                self.cases_output_dim = 3
                self.biomarkers_output_dim = 4
                self.temporal_covariates_dim = 2

            def __len__(self):
                return 1

            def release_shared_sparse_topology(self):
                self.shared_sparse_topology = None

        created: list[FakeDataset] = []

        def make_dataset(**kwargs):
            dataset = FakeDataset(**kwargs)
            created.append(dataset)
            return dataset

        config = create_mock_config(str(tmp_path / "synthetic.zarr"))
        synth_runs = ["0_sparse", "1_dense", "2_mid"]

        with (
            patch(
                "data.curriculum_builder._select_synthetic_scaler_run",
                return_value="1_dense",
            ),
            patch(
                "data.curriculum_builder._load_region_ids",
                return_value=["r0", "r1", "r2", "r3"],
            ) as load_region_ids,
            patch(
                "data.curriculum_builder._map_region_ids_to_nodes",
                return_value=[0, 1],
            ),
            patch("data.curriculum_builder.EpiDataset", side_effect=make_dataset),
        ):
            result = build_curriculum_datasets(
                config=config,
                train_nodes=[0, 1],
                val_nodes=[2],
                test_nodes=[3],
                real_run="",
                synth_runs=synth_runs,
            )

        load_region_ids.assert_called_once_with(
            Path(config.data.dataset_path),
            "1_dense",
        )
        assert result.real_run_id == ""
        assert result.synth_run_ids == synth_runs

        val_dataset = created[-2]
        test_dataset = created[-1]
        assert val_dataset.run_id == "1_dense"
        assert test_dataset.run_id == "1_dense"
        assert val_dataset.kwargs["preloaded_mobility"] == "mobility-1_dense"
        assert test_dataset.kwargs["preloaded_mobility"] == "mobility-1_dense"


class TestCurriculumConfigActiveRuns:
    """Tests for CurriculumConfig.active_runs validation."""

    def test_active_runs_minus_one_is_valid(self):
        """active_runs=-1 means 'use all available synthetic runs'."""
        config = CurriculumConfig(active_runs=-1)
        assert config.active_runs == -1

    def test_active_runs_zero_is_invalid(self):
        """active_runs=0 should raise ValueError."""
        with pytest.raises(ValueError, match="active_runs must be >= -1 and != 0"):
            CurriculumConfig(active_runs=0)

    def test_active_runs_negative_two_is_invalid(self):
        """active_runs=-2 should raise ValueError."""
        with pytest.raises(ValueError, match="active_runs must be >= -1 and != 0"):
            CurriculumConfig(active_runs=-2)

    def test_active_runs_positive_is_valid(self):
        """Positive active_runs values should still be valid."""
        config = CurriculumConfig(active_runs=3)
        assert config.active_runs == 3
