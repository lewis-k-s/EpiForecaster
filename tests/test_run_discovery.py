"""Tests for run_discovery utilities."""

import pytest
from pathlib import Path

from utils.run_discovery import extract_run_from_checkpoint_path


@pytest.mark.epiforecaster
class TestExtractRunFromCheckpointPath:
    """Tests for extract_run_from_checkpoint_path function."""

    def test_extract_from_standard_training_path(self, tmp_path: Path):
        """Test extraction with standard training checkpoint path."""
        # Create a mock checkpoint path
        checkpoint = (
            tmp_path
            / "outputs"
            / "training"
            / "my_experiment"
            / "run_1234567890"
            / "checkpoints"
            / "best_model.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = extract_run_from_checkpoint_path(
            checkpoint, outputs_root=tmp_path / "outputs"
        )

        assert result == ("my_experiment", "run_1234567890")

    def test_extract_from_optuna_path(self, tmp_path: Path):
        """Test extraction with optuna checkpoint path."""
        checkpoint = (
            tmp_path
            / "outputs"
            / "optuna"
            / "hpo_experiment"
            / "local_trial29_1768952246597587000"
            / "checkpoints"
            / "best_model.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = extract_run_from_checkpoint_path(
            checkpoint, outputs_root=tmp_path / "outputs"
        )

        assert result == ("hpo_experiment", "local_trial29_1768952246597587000")

    def test_extract_from_relative_path(self, tmp_path: Path):
        """Test extraction with relative path from cwd."""
        # Create a mock outputs directory structure in tmp_path
        (
            tmp_path / "outputs" / "training" / "test_exp" / "run_999" / "checkpoints"
        ).mkdir(parents=True)
        checkpoint = (
            tmp_path
            / "outputs"
            / "training"
            / "test_exp"
            / "run_999"
            / "checkpoints"
            / "best_model.pt"
        )
        checkpoint.touch()

        # Change to tmp_path to simulate cwd being the project root
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Now use relative path (as a user would from project root)
            result = extract_run_from_checkpoint_path(
                Path("outputs/training/test_exp/run_999/checkpoints/best_model.pt"),
                outputs_root=Path("outputs"),
            )
            assert result == ("test_exp", "run_999")
        finally:
            os.chdir(original_cwd)

    def test_extract_fails_for_non_standard_path(self, tmp_path: Path):
        """Test extraction fails for paths not matching expected pattern."""
        checkpoint = tmp_path / "some" / "random" / "path" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = extract_run_from_checkpoint_path(checkpoint, outputs_root=tmp_path)

        assert result is None

    def test_extract_fails_for_invalid_run_id(self, tmp_path: Path):
        """Test extraction fails for run_id not matching expected patterns."""
        # Create a checkpoint with invalid run_id (doesn't start with run_ or contain trial_)
        checkpoint = (
            tmp_path
            / "outputs"
            / "training"
            / "my_experiment"
            / "invalid_run_name"
            / "checkpoints"
            / "best_model.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = extract_run_from_checkpoint_path(
            checkpoint, outputs_root=tmp_path / "outputs"
        )

        assert result is None

    def test_extract_fails_for_missing_checkpoints_dir(self, tmp_path: Path):
        """Test extraction fails when checkpoints directory is not in path."""
        # Path without /checkpoints/ in it
        checkpoint = (
            tmp_path
            / "outputs"
            / "training"
            / "my_experiment"
            / "run_12345"
            / "best_model.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = extract_run_from_checkpoint_path(
            checkpoint, outputs_root=tmp_path / "outputs"
        )

        assert result is None

    def test_extract_with_different_trial_patterns(self, tmp_path: Path):
        """Test extraction works with various optuna trial patterns."""
        test_cases = [
            "local_trial1_123",
            "remote_trial5_456",
            "my_trial_10_789",
        ]

        for run_id in test_cases:
            checkpoint = (
                tmp_path
                / "outputs"
                / "optuna"
                / "hpo_exp"
                / run_id
                / "checkpoints"
                / "best_model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            result = extract_run_from_checkpoint_path(
                checkpoint, outputs_root=tmp_path / "outputs"
            )

            assert result == ("hpo_exp", run_id)
