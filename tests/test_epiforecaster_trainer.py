import pytest
from types import SimpleNamespace
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
from evaluation.losses import JointInferenceLoss


class _HistogramTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.beta_projection = torch.nn.Linear(2, 2)
        self.backbone.obs_context_projection = torch.nn.Linear(2, 2)
        self.cases_head = torch.nn.Module()
        self.cases_head.delay_kernel = torch.nn.Module()
        self.cases_head.delay_kernel.kernel = torch.nn.Parameter(
            torch.tensor([1.0, -1.0], dtype=torch.float32)
        )
        self.other = torch.nn.Linear(2, 1)
        self.scalar_param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


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


@patch("training.epiforecaster_trainer.wandb.Histogram")
def test_build_wandb_gradient_histogram_payload_selects_and_filters(mock_histogram):
    mock_histogram.side_effect = lambda values: {"count": len(values)}

    model = _HistogramTestModel()
    model.backbone.beta_projection.weight.grad = torch.tensor(
        [[1.0, 2.0], [float("nan"), 4.0]], dtype=torch.float32
    )
    model.backbone.beta_projection.bias.grad = torch.tensor(
        [0.5, 1.5], dtype=torch.float32
    )
    model.backbone.obs_context_projection.weight.grad = torch.tensor(
        [[float("inf"), 1.0], [2.0, 3.0]], dtype=torch.float32
    )
    model.backbone.obs_context_projection.bias.grad = torch.tensor(
        [0.25, -0.25], dtype=torch.float32
    )
    model.cases_head.delay_kernel.kernel.grad = torch.tensor(
        [0.1, 0.2], dtype=torch.float32
    )
    model.other.weight.grad = torch.ones_like(model.other.weight)
    model.other.bias.grad = torch.ones_like(model.other.bias)
    model.scalar_param.grad = torch.tensor(1.0, dtype=torch.float32)

    payload = EpiForecasterTrainer._build_wandb_gradient_histogram_payload(
        model=model,
        step=10,
        frequency=5,
        patterns=["backbone.beta_projection", "backbone.obs_context_projection"],
        max_params=2,
    )

    assert payload == {
        "grad_hist/backbone_beta_projection_bias": {"count": 2},
        "grad_hist/backbone_beta_projection_weight": {"count": 3},
        "grad_histograms_logged": 2,
    }


def test_build_wandb_gradient_histogram_payload_returns_empty_when_disabled():
    payload = EpiForecasterTrainer._build_wandb_gradient_histogram_payload(
        model=_HistogramTestModel(),
        step=10,
        frequency=0,
        patterns=["backbone.beta_projection"],
        max_params=4,
    )

    assert payload == {}


def test_build_wandb_gradient_histogram_payload_respects_logging_cadence():
    model = _HistogramTestModel()
    model.backbone.beta_projection.weight.grad = torch.ones(
        (2, 2), dtype=torch.float32
    )

    payload = EpiForecasterTrainer._build_wandb_gradient_histogram_payload(
        model=model,
        step=6,
        frequency=5,
        patterns=["backbone.beta_projection"],
        max_params=4,
    )

    assert payload == {}


@patch("training.epiforecaster_trainer.wandb.Histogram")
def test_build_wandb_gradient_histogram_payload_skips_scalar_and_missing_grads(
    mock_histogram,
):
    mock_histogram.side_effect = lambda values: {"count": len(values)}

    model = _HistogramTestModel()
    model.scalar_param.grad = torch.tensor(1.0, dtype=torch.float32)

    payload = EpiForecasterTrainer._build_wandb_gradient_histogram_payload(
        model=model,
        step=10,
        frequency=5,
        patterns=["scalar_param", "other"],
        max_params=4,
    )

    assert payload == {}
    mock_histogram.assert_not_called()


@patch("training.epiforecaster_trainer.wandb.Histogram")
def test_build_wandb_payload_filters_grad_scalars_and_merges_histograms(mock_histogram):
    mock_histogram.side_effect = lambda values: {"count": len(values)}

    trainer = object.__new__(EpiForecasterTrainer)
    trainer.model = _HistogramTestModel()
    trainer.model.backbone.beta_projection.weight.grad = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )
    trainer.model.backbone.beta_projection.bias.grad = torch.tensor(
        [0.5, -0.5], dtype=torch.float32
    )
    trainer.global_step = 10
    trainer.wandb_run = object()
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            wandb_gradient_histogram_frequency=5,
            wandb_gradient_histogram_patterns=["backbone.beta_projection"],
            wandb_gradient_histogram_max_params=4,
        )
    )

    payload = trainer._build_wandb_payload(
        log_this_step=True,
        log_data={
            "loss_train_step": 1.0,
            "time_batch_s": 0.2,
            "gradnorm_clipped_total": 2.0,
            "grad_snapshot_max_layer_norm": 3.0,
            "gradnorm_sidecar_ran": 1.0,
        },
        component_gradnorm_log_data={"gradnorm_sird_physics": 4.0},
        gradient_snapshot_log_data={"grad_snapshot_global_norm": 5.0},
    )

    assert payload is not None
    assert payload["loss_train_step"] == 1.0
    assert payload["time_batch_s"] == 0.2
    assert "gradnorm_clipped_total" not in payload
    assert "grad_snapshot_max_layer_norm" not in payload
    assert "gradnorm_sidecar_ran" not in payload
    assert payload["grad_hist/backbone_beta_projection_bias"] == {"count": 2}
    assert payload["grad_hist/backbone_beta_projection_weight"] == {"count": 4}
    assert payload["grad_histograms_logged"] == 2


def test_build_wandb_payload_omits_histograms_when_frequency_is_zero():
    trainer = object.__new__(EpiForecasterTrainer)
    trainer.model = _HistogramTestModel()
    trainer.global_step = 10
    trainer.wandb_run = object()
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            wandb_gradient_histogram_frequency=0,
            wandb_gradient_histogram_patterns=["backbone.beta_projection"],
            wandb_gradient_histogram_max_params=4,
        )
    )

    payload = trainer._build_wandb_payload(
        log_this_step=False,
        log_data={"gradnorm_clipped_total": 2.0},
        component_gradnorm_log_data={"gradnorm_sird_physics": 4.0},
        gradient_snapshot_log_data={"grad_snapshot_global_norm": 5.0},
    )

    assert payload is None
