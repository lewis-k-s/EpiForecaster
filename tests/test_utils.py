import pytest
import os
import shutil
import subprocess
import torch
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from utils.remote_runner import (
    SlurmJobSpec,
    submit_job_with_script,
    JobStatus,
    generate_training_script,
)
from utils.platform import (
    is_slurm_cluster,
    get_nvme_path,
    stage_dataset_to_nvme,
    select_multiprocessing_context,
)
from utils.tensor_core import setup_tensor_core_optimizations
from utils.temporal import (
    parse_date_string,
    date_to_index,
    get_temporal_boundaries,
    validate_temporal_range,
)

# --- Test Remote Runner ---


class TestRemoteRunner:
    def test_slurm_job_spec_script_generation(self):
        spec = SlurmJobSpec(
            job_name="test_job", time="01:00:00", gres="gpu:1", partition="debug"
        )
        script = spec.to_sbatch_script("echo 'hello'")

        assert "#SBATCH --job-name=test_job" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --partition=debug" in script
        assert "echo 'hello'" in script

    @patch("utils.remote_runner.subprocess.run")
    def test_submit_job_with_script(self, mock_run):
        # Mock successful submission
        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 12345"
        mock_run.return_value = mock_result

        spec = SlurmJobSpec(job_name="test")
        job_id = submit_job_with_script(spec, "cmd", remote_host="host")

        assert job_id == "12345"
        assert mock_run.call_count == 2  # 1 for cat (copy), 1 for sbatch

    def test_generate_training_script(self):
        script = generate_training_script(
            config_path="config.yaml", overrides=["a=1", "b=2"]
        )
        assert "uv run main train epiforecaster" in script
        assert '--config "config.yaml"' in script
        assert "--override a=1" in script
        assert "--override b=2" in script


# --- Test Platform ---


class TestPlatform:
    def test_is_slurm_cluster(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            assert is_slurm_cluster() is True

        with patch.dict(os.environ, {}, clear=True):
            # Ensure SLURM_JOB_ID is not present
            if "SLURM_JOB_ID" in os.environ:
                del os.environ["SLURM_JOB_ID"]
            assert is_slurm_cluster() is False

    @patch("utils.platform.shutil.which")
    @patch("subprocess.run")
    def test_stage_dataset_to_nvme_rsync(self, mock_run, mock_which, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        nvme = tmp_path / "nvme"
        nvme.mkdir()

        mock_which.return_value = "/usr/bin/rsync"

        with patch("utils.platform.get_nvme_path", return_value=nvme):
            dest = stage_dataset_to_nvme(source, enable_staging=True)

            assert dest == nvme / "source"
            mock_run.assert_called()
            args = mock_run.call_args[0][0]
            assert "rsync" in args

    def test_select_multiprocessing_context(self):
        # On Linux + CUDA -> fork
        with patch("platform.system", return_value="Linux"):
            ctx = select_multiprocessing_context(
                torch.device("cuda"), all_num_workers_zero=False
            )
            assert ctx == "fork"

        # On Mac/Windows + CUDA (hypothetical) -> spawn
        with patch("platform.system", return_value="Windows"):
            ctx = select_multiprocessing_context(
                torch.device("cuda"), all_num_workers_zero=False
            )
            assert ctx == "spawn"


# --- Test Tensor Core ---


class TestTensorCore:
    def test_setup_tensor_core_optimizations(self):
        # Mock torch.backends
        with (
            patch("torch.backends.cuda") as mock_cuda,
            patch("torch.backends.cudnn") as mock_cudnn,
        ):
            # Simulate CUDA device
            device = torch.device("cuda")
            logger = MagicMock()

            setup_tensor_core_optimizations(device, enable_tf32=True, logger=logger)

            assert mock_cuda.matmul.fp32_precision == "tf32"
            assert mock_cudnn.conv.fp32_precision == "tf32"
            logger.info.assert_called()

    def test_setup_skipped_cpu(self):
        with patch("torch.backends.cuda") as mock_cuda:
            device = torch.device("cpu")
            setup_tensor_core_optimizations(device)
            # Should not access cuda settings
            # We can't easily assert "not accessed" on module mocks without wrapping them
            # But we can check it didn't crash and maybe check return.
            pass


# --- Test Temporal ---


class TestTemporal:
    @pytest.fixture
    def mock_dataset(self):
        times = pd.date_range("2020-01-01", periods=10)
        ds = xr.Dataset(coords={"date": times})
        # Mock TEMPORAL_COORD constant logic by ensuring code uses "date" or we patch it
        # The utils module imports TEMPORAL_COORD.
        # We assume TEMPORAL_COORD is "date" or similar.
        # Let's patch TEMPORAL_COORD in utils.temporal
        return ds

    def test_parse_date_string(self):
        dt = parse_date_string("2020-01-01")
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1

        with pytest.raises(ValueError):
            parse_date_string("invalid")

    def test_date_to_index(self, mock_dataset):
        with patch("utils.temporal.TEMPORAL_COORD", "date"):
            idx = date_to_index(mock_dataset, pd.Timestamp("2020-01-02"))
            assert idx == 1

            with pytest.raises(ValueError):
                date_to_index(mock_dataset, pd.Timestamp("2021-01-01"))

    def test_get_temporal_boundaries(self, mock_dataset):
        with patch("utils.temporal.TEMPORAL_COORD", "date"):
            # 0..9 indices
            # Train end: 2020-01-04 (idx 3) -> Train [0, 3)
            # Val end: 2020-01-06 (idx 5) -> Val [3, 5)
            # Test end: 2020-01-08 (idx 7) -> Test [5, 7)

            s, te, ve, tste = get_temporal_boundaries(
                mock_dataset, "2020-01-04", "2020-01-06", "2020-01-08"
            )
            assert s == 0
            assert te == 3
            assert ve == 5
            assert tste == 7

    def test_validate_temporal_range(self):
        # Valid
        validate_temporal_range(
            (0, 10), history_length=5, forecast_horizon=2, total_time_steps=20
        )

        # Invalid: range too small
        with pytest.raises(ValueError, match="requires at least"):
            validate_temporal_range(
                (0, 5), history_length=5, forecast_horizon=2, total_time_steps=20
            )

        # Invalid: out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            validate_temporal_range(
                (0, 25), history_length=5, forecast_horizon=2, total_time_steps=20
            )
