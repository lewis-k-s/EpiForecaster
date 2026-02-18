import pytest
from unittest.mock import MagicMock, patch
import torch

from training.epiforecaster_trainer import EpiForecasterTrainer
from models.configs import (
    EpiForecasterConfig,
    ModelConfig,
    DataConfig,
    TrainingParams,
    OutputConfig,
    ModelVariant,
    SIRPhysicsConfig,
    ObservationHeadConfig,
    LossConfig,
    CurriculumConfig,
)
from evaluation.epiforecaster_eval import JointInferenceLoss


class TestEpiForecasterTrainer:
    @pytest.fixture
    def minimal_config(self):
        return EpiForecasterConfig(
            model=ModelConfig(
                type=ModelVariant(cases=True),
                mobility_embedding_dim=8,
                region_embedding_dim=8,
                history_length=14,
                forecast_horizon=7,
                max_neighbors=5,
                sir_physics=SIRPhysicsConfig(),
                observation_heads=ObservationHeadConfig(),
            ),
            data=DataConfig(
                dataset_path="dummy_dataset.zarr",
                run_id="test_run",
            ),
            training=TrainingParams(
                epochs=1,
                batch_size=4,
                loss=LossConfig(name="joint_inference"),
                curriculum=CurriculumConfig(enabled=False),
                enable_mixed_precision=False,  # Disable for tests on non-CUDA devices
            ),
            output=OutputConfig(log_dir="test_outputs", experiment_name="test_exp"),
        )

    @patch("training.epiforecaster_trainer.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    def test_initialization(
        self, mock_wandb, mock_model_cls, mock_dataset_cls, minimal_config
    ):
        """Test trainer initialization and component creation."""
        # Mock model parameters for optimizer
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        # Trainer calls .to() for device/dtype conversion
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset behavior
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.__len__.return_value = 100
        mock_dataset_instance.target_nodes = list(range(10))
        mock_dataset_cls.return_value = mock_dataset_instance

        # Mock class method load_canonical_dataset to avoid disk IO
        mock_dataset_cls.load_canonical_dataset.return_value = MagicMock()
        mock_dataset_cls.load_canonical_dataset.return_value.__enter__.return_value = (
            MagicMock()
        )
        mock_dataset_cls.load_canonical_dataset.return_value.__getitem__.return_value = MagicMock()
        mock_dataset_cls.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Initialize trainer
        trainer = EpiForecasterTrainer(minimal_config)

        assert isinstance(trainer.model, MagicMock)
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, JointInferenceLoss)

        # Check if model was initialized with correct config params
        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args
        assert kwargs["forecast_horizon"] == 7
        assert kwargs["sequence_length"] == 14

    @patch("training.epiforecaster_trainer.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    def test_loss_criterion_creation(
        self, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test correct loss function is created."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Test joint inference loss
        minimal_config.training.loss.name = "joint_inference"
        trainer = EpiForecasterTrainer(minimal_config)
        assert isinstance(trainer.criterion, JointInferenceLoss)

        # Test validation for wrong loss type
        minimal_config.training.loss.name = "mse"
        with pytest.raises(ValueError, match="requires JointInferenceLoss"):
            EpiForecasterTrainer(minimal_config)

    @patch("training.epiforecaster_trainer.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    def test_scheduler_creation(
        self, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test scheduler creation options."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # 1. Cosine
        minimal_config.training.scheduler_type = "cosine"
        trainer = EpiForecasterTrainer(minimal_config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        # 2. Step
        minimal_config.training.scheduler_type = "step"
        trainer = EpiForecasterTrainer(minimal_config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

        # 3. None
        minimal_config.training.scheduler_type = "none"
        trainer = EpiForecasterTrainer(minimal_config)
        assert trainer.scheduler is None

    @patch("training.epiforecaster_trainer.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    def test_checkpoint_logic(
        self, mock_wandb, mock_model_cls, mock_dataset, minimal_config, tmp_path
    ):
        """Test checkpoint discovery/resume logic."""
        # Mock model
        mock_model_instance = MagicMock()
        # Use float32 dtype (only supported dtype)
        param = torch.nn.Parameter(torch.randn(1, dtype=torch.float32))
        mock_model_instance.parameters.return_value = [param]
        # Track the parameter through dtype conversion
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Setup fake checkpoint
        minimal_config.output.log_dir = str(tmp_path)
        minimal_config.output.experiment_name = "test_exp"
        model_id = "test_model_id"
        minimal_config.training.model_id = model_id

        checkpoint_dir = tmp_path / "test_exp" / model_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create a dummy checkpoint file
        ckpt_path = checkpoint_dir / "checkpoint_epoch_5.pt"
        torch.save({"epoch": 5, "model_state_dict": {}}, ckpt_path)

        # Enable resume
        minimal_config.training.resume = True

        trainer = EpiForecasterTrainer(minimal_config)

        # Should have found the checkpoint
        assert trainer.resume is True
        # Since we mocked everything, actual load might not happen or fail on key mismatch,
        # but initialization passed meaning it didn't crash on finding path.

    @patch("training.epiforecaster_trainer.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    def test_curriculum_config_validation(
        self, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test curriculum config validation."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.sparsity_level = 0.5  # Fix numeric comparison
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Enable curriculum
        minimal_config.training.curriculum.enabled = True
        minimal_config.data.real_dataset_path = "real.zarr"

        # Should initialize (mocks handles actual data loading calls)
        with patch.object(
            EpiForecasterTrainer, "_discover_runs", return_value=("real", ["synth1"])
        ):
            with patch.object(
                EpiForecasterTrainer,
                "_split_dataset_by_nodes",
                return_value=([], [], []),
            ):
                with patch.object(
                    EpiForecasterTrainer, "_load_region_ids", return_value=["r1"]
                ):
                    # Mock _map_region_ids_to_nodes to return list
                    with patch.object(
                        EpiForecasterTrainer,
                        "_map_region_ids_to_nodes",
                        return_value=[0],
                    ):
                        trainer = EpiForecasterTrainer(minimal_config)
                        assert trainer.config.training.curriculum.enabled is True
