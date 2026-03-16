import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
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
from evaluation.losses import JointInferenceLoss


class TestEpiForecasterTrainer:
    @pytest.fixture
    def minimal_config(self):
        return EpiForecasterConfig(
            model=ModelConfig(
                type=ModelVariant(cases=True),
                mobility_embedding_dim=8,
                region_embedding_dim=8,
                input_window_length=14,
                forecast_horizon=7,
                max_neighbors=5,
                sir_physics=SIRPhysicsConfig(),
                observation_heads=ObservationHeadConfig(),
            ),
            data=DataConfig(
                dataset_path="dummy_dataset.zarr",
                run_id="real",
                run_id_chunk_size=1,
            ),
            training=TrainingParams(
                epochs=1,
                batch_size=4,
                loss=LossConfig(name="joint_inference"),
                curriculum=CurriculumConfig(enabled=False),
                enable_mixed_precision=False,
            ),
            output=OutputConfig(log_dir="test_outputs", experiment_name="test_exp"),
        )

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_initialization(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset_cls,
        minimal_config,
    ):
        """Test trainer initialization and component creation."""
        # Mock model parameters for optimizer
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        # Trainer calls .to() for device/dtype conversion
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset behavior
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
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

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_loss_criterion_creation(
        self, mock_dataloader, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test correct loss function is created."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Test joint inference loss
        minimal_config.training.loss.name = "joint_inference"
        trainer = EpiForecasterTrainer(minimal_config)
        assert isinstance(trainer.criterion, JointInferenceLoss)

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_scheduler_creation(
        self, mock_dataloader, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test scheduler creation options."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
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

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_checkpoint_logic(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset,
        minimal_config,
        tmp_path,
    ):
        """Test checkpoint resume logic with explicit checkpoint path."""
        # Mock model
        mock_model_instance = MagicMock()
        # Use float32 dtype (only supported dtype)
        param = torch.nn.Parameter(torch.randn(1, dtype=torch.float32))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        # Track the parameter through dtype conversion
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        # Setup fake checkpoint
        minimal_config.output.log_dir = str(tmp_path)
        minimal_config.output.experiment_name = "test_exp"

        # Create a dummy checkpoint file
        ckpt_path = tmp_path / "checkpoint_epoch_5.pt"
        torch.save({"epoch": 5, "model_state_dict": {}}, ckpt_path)

        # Enable resume with explicit path (as string for YAML serialization)
        minimal_config.training.resume_checkpoint_path = str(ckpt_path)

        trainer = EpiForecasterTrainer(minimal_config)

        # Should have the checkpoint path set
        assert trainer.config.training.resume_checkpoint_path == str(ckpt_path)

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_curriculum_config_validation(
        self, mock_dataloader, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        """Test curriculum config validation."""
        # Mock model
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
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
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset_instance.target_nodes = [0]
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset_instance.region_embeddings = None
        mock_loader_bundle = SimpleNamespace(
            train=MagicMock(),
            val=MagicMock(),
            test=MagicMock(),
            curriculum_sampler=None,
            multiprocessing_context=None,
        )
        mock_splits = SimpleNamespace(
            train=mock_dataset_instance,
            val=mock_dataset_instance,
            test=mock_dataset_instance,
            real_run_id="real",
            synth_run_ids=["synth1"],
            region_embedding_store=None,
        )
        with patch(
            "training.epiforecaster_trainer.build_datasets", return_value=mock_splits
        ):
            with patch(
                "training.epiforecaster_trainer.build_dataloaders",
                return_value=mock_loader_bundle,
            ):
                trainer = EpiForecasterTrainer(minimal_config)
                assert trainer.config.training.curriculum.enabled is True

    def test_early_stopping_disabled_during_synth_only_curriculum(self):
        trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(early_stopping_patience=5),
        )
        trainer.curriculum_sampler = SimpleNamespace(
            state=SimpleNamespace(synth_ratio=1.0)
        )

        assert trainer._is_early_stopping_enabled() is False

    def test_early_stopping_enabled_once_curriculum_mixes_real_data(self):
        trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(early_stopping_patience=5),
        )
        trainer.curriculum_sampler = SimpleNamespace(
            state=SimpleNamespace(synth_ratio=0.8)
        )

        assert trainer._is_early_stopping_enabled() is True

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_standard_split_reuses_sparse_topology_and_releases(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset_cls,
        minimal_config,
    ):
        """Val/test should reuse train full sparse topology and release references."""
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        shared_mobility = object()
        shared_mobility_mask = object()
        shared_sparse_topology = object()

        train_ds = MagicMock()
        train_ds.cases_output_dim = 1
        train_ds.biomarkers_output_dim = 1
        train_ds.temporal_covariates_dim = 0
        train_ds.target_nodes = [0, 1]
        train_ds.__len__.return_value = 16
        train_ds.preloaded_mobility = shared_mobility
        train_ds.mobility_mask = shared_mobility_mask
        train_ds.shared_sparse_topology = shared_sparse_topology
        train_ds.biomarker_preprocessor = object()
        train_ds.mobility_preprocessor = object()
        train_ds.region_embeddings = None

        val_ds = MagicMock()
        val_ds.target_nodes = [2]
        val_ds.__len__.return_value = 8
        val_ds.cases_output_dim = 1
        val_ds.biomarkers_output_dim = 1

        test_ds = MagicMock()
        test_ds.target_nodes = [3]
        test_ds.__len__.return_value = 8
        test_ds.cases_output_dim = 1
        test_ds.biomarkers_output_dim = 1

        mock_dataset_cls.side_effect = [train_ds, val_ds, test_ds]

        with patch(
            "data.dataset_factory.split_nodes_by_ratio",
            return_value=([0, 1], [2], [3]),
        ):
            _ = EpiForecasterTrainer(minimal_config)

        val_call = mock_dataset_cls.call_args_list[1]
        test_call = mock_dataset_cls.call_args_list[2]
        assert val_call.kwargs["shared_sparse_topology"] is shared_sparse_topology
        assert test_call.kwargs["shared_sparse_topology"] is shared_sparse_topology
        assert val_call.kwargs["preloaded_mobility"] is shared_mobility
        assert test_call.kwargs["preloaded_mobility"] is shared_mobility
        train_ds.release_shared_sparse_topology.assert_called_once()
        val_ds.release_shared_sparse_topology.assert_called_once()
        test_ds.release_shared_sparse_topology.assert_called_once()

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_gradnorm_compiles_adaptive_backward_step(
        self, mock_dataloader, mock_wandb, mock_model_cls, mock_dataset, minimal_config
    ):
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset_instance.__len__.return_value = 16
        mock_dataset_instance.target_nodes = [0, 1]
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        minimal_config.training.compile_backward = True
        trainer = EpiForecasterTrainer(minimal_config)
        assert trainer._compiled_training_step is not None
        assert trainer.gradnorm_controller is not None
        assert trainer.gradnorm_optimizer is not None

    @patch("data.dataset_factory.EpiDataset")
    @patch("training.epiforecaster_trainer.EpiForecaster")
    @patch("training.epiforecaster_trainer.wandb")
    @patch("training.dataloader_factory.DataLoader")
    def test_checkpoint_includes_gradnorm_state(
        self,
        mock_dataloader,
        mock_wandb,
        mock_model_cls,
        mock_dataset,
        minimal_config,
        tmp_path,
    ):
        mock_model_instance = MagicMock()
        param = torch.nn.Parameter(torch.randn(1))
        mock_model_instance.parameters.return_value = [param]
        mock_model_instance.named_parameters.return_value = [("backbone.mock", param)]
        mock_model_instance.to.side_effect = lambda *args, **kwargs: mock_model_instance
        mock_model_instance.state_dict.return_value = {"backbone.mock": param.detach()}
        mock_model_instance.dtype = torch.float32
        mock_model_cls.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.cases_output_dim = 1
        mock_dataset_instance.biomarkers_output_dim = 1
        mock_dataset_instance.temporal_covariates_dim = 0
        mock_dataset_instance.__len__.return_value = 16
        mock_dataset_instance.target_nodes = [0, 1]
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset.load_canonical_dataset.return_value.__getitem__.return_value.size = 10

        minimal_config.output.log_dir = str(tmp_path)
        minimal_config.output.experiment_name = "gradnorm_ckpt"
        trainer = EpiForecasterTrainer(minimal_config)
        trainer._save_checkpoint(epoch=0, val_loss=1.0, is_best=False, is_final=False)

        ckpt_files = list((trainer.checkpoint_dir).glob("checkpoint_epoch_*.pt"))
        assert ckpt_files
        checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
        assert "gradnorm_controller_state_dict" in checkpoint
        assert "gradnorm_optimizer_state_dict" in checkpoint
