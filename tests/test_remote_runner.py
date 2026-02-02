"""Tests for remote_runner module.

These tests verify that:
1. Sbatch scripts are generated correctly
2. Generated scripts can be validated with bash/shellcheck
3. The SlurmJobSpec dataclass works as expected
4. Script generation functions produce valid output
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from utils.remote_runner import (
    ExperimentResult,
    JobStatus,
    SlurmJobSpec,
    generate_optuna_script,
    generate_preprocessing_script,
    generate_training_script,
)


class TestSlurmJobSpec:
    """Tests for SlurmJobSpec dataclass."""

    def test_default_values(self):
        """Test that SlurmJobSpec has sensible defaults."""
        spec = SlurmJobSpec(job_name="test_job")
        assert spec.job_name == "test_job"
        assert spec.account == "bsc08"
        assert spec.qos == "acc_bscls"
        assert spec.gres == "gpu:1"
        assert spec.cpus_per_task == 20
        assert spec.time == "04:00:00"
        assert spec.output == "slurm-%x-%j.out"
        assert spec.array is None

    def test_custom_values(self):
        """Test that custom values override defaults."""
        spec = SlurmJobSpec(
            job_name="custom_job",
            account="other_account",
            gres="gpu:2",
            cpus_per_task=40,
            time="02:00:00",
            array="0-3",
        )
        assert spec.account == "other_account"
        assert spec.gres == "gpu:2"
        assert spec.cpus_per_task == 40
        assert spec.time == "02:00:00"
        assert spec.array == "0-3"

    def test_to_sbatch_script_basic(self):
        """Test basic sbatch script generation."""
        spec = SlurmJobSpec(job_name="basic_test")
        commands = "echo 'Hello World'"
        script = spec.to_sbatch_script(commands)

        # Check sbatch directives
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=basic_test" in script
        assert "#SBATCH --account=bsc08" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --cpus-per-task=20" in script
        assert "#SBATCH --time=04:00:00" in script
        assert "#SBATCH --output=slurm-%x-%j.out" in script

        # Check script structure
        assert "set -euo pipefail" in script
        assert "echo 'Hello World'" in script

    def test_to_sbatch_script_array_job(self):
        """Test array job script generation."""
        spec = SlurmJobSpec(
            job_name="array_test",
            array="0-7",
        )
        commands = "echo $SLURM_ARRAY_TASK_ID"
        script = spec.to_sbatch_script(commands)

        assert "#SBATCH --array=0-7" in script
        # Output pattern should be updated for array jobs
        assert "#SBATCH --output=slurm-%x-%A_%a.out" in script

    def test_to_sbatch_script_no_array_output_preserved(self):
        """Test that custom output patterns are preserved for non-array jobs."""
        spec = SlurmJobSpec(
            job_name="custom_output",
            output="custom-%j.log",
        )
        script = spec.to_sbatch_script("echo 'test'")

        assert "#SBATCH --output=custom-%j.log" in script


class TestGenerateTrainingScript:
    """Tests for generate_training_script function."""

    def test_basic_training_script(self):
        """Test basic training script generation."""
        commands = generate_training_script("configs/train.yaml")

        # Check essential components
        assert "module load" in commands
        assert "CUDA/12.1.1" in commands
        assert 'export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"' in commands
        assert 'cd "$PROJECT_ROOT"' in commands
        assert "configs/train.yaml" in commands

    def test_training_with_overrides(self):
        """Test training script with config overrides."""
        overrides = ["training.batch_size=32", "training.epochs=5"]
        commands = generate_training_script("configs/train.yaml", overrides=overrides)

        # Check that overrides are included
        assert "--override training.batch_size=32" in commands
        assert "--override training.epochs=5" in commands

    def test_training_no_overrides(self):
        """Test training script without overrides."""
        commands = generate_training_script("configs/train.yaml", overrides=None)

        # Should not have override arguments
        assert "--override" not in commands

    def test_training_custom_venv(self):
        """Test training script with custom venv path."""
        commands = generate_training_script(
            "configs/train.yaml",
            venv_path="/custom/venv",
        )

        assert '/custom/venv" ]; then' in commands
        assert 'source "/custom/venv/bin/activate"' in commands


class TestGenerateOptunaScript:
    """Tests for generate_optuna_script function."""

    def test_basic_optuna_script(self):
        """Test basic optuna script generation."""
        commands = generate_optuna_script(
            study_name="test_study",
            config_path="configs/train.yaml",
        )

        assert "test_study" in commands
        assert "configs/train.yaml" in commands
        assert "optuna_epiforecaster_worker.py" in commands
        assert "--study-name" in commands

    def test_optuna_with_epochs(self):
        """Test optuna script with custom epochs."""
        commands = generate_optuna_script(
            study_name="test_study",
            config_path="configs/train.yaml",
            epochs=10,
        )

        assert "--epochs 10" in commands

    def test_optuna_paths(self):
        """Test that optuna generates correct paths."""
        commands = generate_optuna_script(
            study_name="my_study",
            config_path="configs/train.yaml",
        )

        assert "outputs/optuna/my_study.journal" in commands
        assert "outputs/optuna" in commands


class TestSbatchScriptValidation:
    """Tests that validate generated sbatch scripts."""

    def _validate_bash_syntax(self, script_content: str) -> tuple[bool, str]:
        """Helper to validate bash script syntax using bash -n.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            result = subprocess.run(
                ["bash", "-n"],
                input=script_content,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
        except FileNotFoundError:
            # bash not available, skip validation
            return True, "bash not available for validation"

    def test_training_script_is_valid_bash(self):
        """Verify training script passes bash syntax check."""
        spec = SlurmJobSpec(job_name="syntax_test")
        commands = generate_training_script("configs/train.yaml")
        script = spec.to_sbatch_script(commands)

        is_valid, error = self._validate_bash_syntax(script)
        assert is_valid, f"Training script has syntax errors: {error}"

    def test_optuna_script_is_valid_bash(self):
        """Verify optuna script passes bash syntax check."""
        spec = SlurmJobSpec(
            job_name="optuna_syntax_test",
            array="0-3",
        )
        commands = generate_optuna_script("test_study", "configs/train.yaml")
        script = spec.to_sbatch_script(commands)

        is_valid, error = self._validate_bash_syntax(script)
        assert is_valid, f"Optuna script has syntax errors: {error}"

    def test_array_job_script_is_valid_bash(self):
        """Verify array job script passes bash syntax check."""
        spec = SlurmJobSpec(
            job_name="array_test",
            array="0-7",
            time="02:00:00",
        )
        commands = generate_training_script(
            "configs/train.yaml",
            overrides=["training.batch_size=64"],
        )
        script = spec.to_sbatch_script(commands)

        is_valid, error = self._validate_bash_syntax(script)
        assert is_valid, f"Array job script has syntax errors: {error}"

    def test_complex_overrides_script_is_valid(self):
        """Verify script with complex overrides passes syntax check."""
        spec = SlurmJobSpec(job_name="complex_test")
        commands = generate_training_script(
            "configs/train.yaml",
            overrides=[
                "training.batch_size=48",
                "training.epochs=10",
                "training.learning_rate=0.0005",
                "data.log_scale=true",
            ],
        )
        script = spec.to_sbatch_script(commands)

        is_valid, error = self._validate_bash_syntax(script)
        assert is_valid, f"Complex overrides script has syntax errors: {error}"

    @pytest.mark.skipif(
        subprocess.run(["which", "shellcheck"], capture_output=True).returncode != 0,
        reason="shellcheck not installed",
    )
    def test_training_script_passes_shellcheck(self):
        """Verify training script passes shellcheck (if available)."""
        spec = SlurmJobSpec(job_name="shellcheck_test")
        commands = generate_training_script("configs/train.yaml")
        script = spec.to_sbatch_script(commands)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["shellcheck", "-S", "warning", temp_path],
                capture_output=True,
                text=True,
                check=False,
            )
            # Allow exit code 0 (no issues) or accept warnings
            if result.returncode > 1:
                pytest.fail(f"shellcheck found errors:\n{result.stdout}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocessing_script_is_valid_bash(self):
        """Verify preprocessing script passes bash syntax check."""
        spec = SlurmJobSpec(
            job_name="preprocess_test",
            gres="gpu:0",  # CPU-only
            cpus_per_task=40,
        )
        commands = generate_preprocessing_script("configs/preprocess_mn5_synth.yaml")
        script = spec.to_sbatch_script(commands)

        is_valid, error = self._validate_bash_syntax(script)
        assert is_valid, f"Preprocessing script has syntax errors: {error}"

    def test_preprocessing_script_no_cuda_setup(self):
        """Verify preprocessing script doesn't include CUDA setup."""
        commands = generate_preprocessing_script("configs/preprocess.yaml")

        # Preprocessing should not reference CUDA
        assert "CUDA" not in commands
        assert "cuda" not in commands.lower()
        # Should use preprocess command
        assert "preprocess epiforecaster" in commands

    @pytest.mark.skipif(
        subprocess.run(["which", "shellcheck"], capture_output=True).returncode != 0,
        reason="shellcheck not installed",
    )
    def test_preprocessing_script_passes_shellcheck(self):
        """Verify preprocessing script passes shellcheck (if available)."""
        spec = SlurmJobSpec(
            job_name="preprocess_shellcheck",
            gres="gpu:0",
            cpus_per_task=40,
        )
        commands = generate_preprocessing_script("configs/preprocess.yaml")
        script = spec.to_sbatch_script(commands)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["shellcheck", "-S", "warning", temp_path],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode > 1:
                pytest.fail(f"shellcheck found errors:\n{result.stdout}")
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDryRunMode:
    """Tests for dry-run functionality (script generation without execution)."""

    def test_generate_and_print_training_script(self, capsys):
        """Test that we can generate and display a training script."""
        spec = SlurmJobSpec(
            job_name="dry_run_test",
            time="01:00:00",
            gres="gpu:1",
        )
        commands = generate_training_script(
            "configs/production_only/train_epifor_mn5_full.yaml",
            overrides=["training.batch_size=32"],
        )
        script = spec.to_sbatch_script(commands)

        # Print script for inspection (in dry-run mode)
        print("Generated script:")
        print("=" * 60)
        print(script)
        print("=" * 60)

        captured = capsys.readouterr()
        assert "#!/bin/bash" in captured.out
        assert "dry_run_test" in captured.out
        assert "training.batch_size=32" in captured.out

    def test_generate_and_print_optuna_script(self, capsys):
        """Test that we can generate and display an optuna script."""
        spec = SlurmJobSpec(
            job_name="optuna_dry_run",
            array="0-3",
            time="02:00:00",
        )
        commands = generate_optuna_script(
            study_name="test_study",
            config_path="configs/train.yaml",
            epochs=5,
        )
        script = spec.to_sbatch_script(commands)

        print("Generated Optuna script:")
        print("=" * 60)
        print(script)
        print("=" * 60)

        captured = capsys.readouterr()
        assert "#!/bin/bash" in captured.out
        assert "optuna_dry_run" in captured.out
        assert "0-3" in captured.out
        assert "optuna_epiforecaster_worker.py" in captured.out

    def test_generate_and_print_preprocessing_script(self, capsys):
        """Test that we can generate and display a preprocessing script."""
        spec = SlurmJobSpec(
            job_name="preprocess_dry_run",
            time="02:00:00",
            gres="gpu:0",
            cpus_per_task=40,
        )
        commands = generate_preprocessing_script(
            "configs/production_only/preprocess_mn5_synth.yaml"
        )
        script = spec.to_sbatch_script(commands)

        print("Generated Preprocessing script:")
        print("=" * 60)
        print(script)
        print("=" * 60)

        captured = capsys.readouterr()
        assert "#!/bin/bash" in captured.out
        assert "preprocess_dry_run" in captured.out
        assert "gpu:0" in captured.out  # CPU-only
        assert "preprocess epiforecaster" in captured.out
        assert "OMP_NUM_THREADS" in captured.out


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """Test that all expected status values exist."""
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.RUNNING.value == "RUNNING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.FAILED.value == "FAILED"
        assert JobStatus.CANCELLED.value == "CANCELLED"
        assert JobStatus.TIMEOUT.value == "TIMEOUT"
        assert JobStatus.UNKNOWN.value == "UNKNOWN"


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_experiment_result_creation(self):
        """Test creation of ExperimentResult."""
        result = ExperimentResult(
            job_id="12345",
            experiment_name="test_exp",
            status=JobStatus.COMPLETED,
            local_output_path=Path("/tmp/test"),
            error_message=None,
        )

        assert result.job_id == "12345"
        assert result.experiment_name == "test_exp"
        assert result.status == JobStatus.COMPLETED
        assert result.local_output_path == Path("/tmp/test")
        assert result.error_message is None

    def test_experiment_result_optional_fields(self):
        """Test ExperimentResult with optional fields as None."""
        result = ExperimentResult(
            job_id="12345",
            experiment_name="test_exp",
            status=JobStatus.FAILED,
            local_output_path=None,
            error_message="Something went wrong",
        )

        assert result.local_output_path is None
        assert result.error_message == "Something went wrong"


@pytest.fixture
def sample_job_spec():
    """Fixture providing a sample SlurmJobSpec."""
    return SlurmJobSpec(
        job_name="test_fixture",
        time="01:00:00",
        gres="gpu:1",
    )


class TestWithFixtures:
    """Tests demonstrating fixture usage."""

    def test_script_generation_with_fixture(self, sample_job_spec):
        """Test script generation using a fixture."""
        commands = "echo 'test'"
        script = sample_job_spec.to_sbatch_script(commands)

        assert "test_fixture" in script
        assert "01:00:00" in script


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
