import pytest
from unittest.mock import MagicMock, patch
import torch
from pathlib import Path

from training.region2vec_trainer import Region2VecTrainer, RegionTrainerConfig
from models.region_losses import CommunityOrientedLoss


class TestRegion2VecTrainer:
    @pytest.fixture
    def trainer_config(self):
        # Create a minimal config using from_dict
        raw_config = {
            "data": {"zarr_path": "dummy_regions.zarr", "normalize_features": True},
            "encoder": {"hidden_dim": 16, "embedding_dim": 8},
            "training": {"epochs": 1, "learning_rate": 1e-3, "device": "cpu"},
            "loss": {"loss_type": "community", "temperature": 0.1},
            "sampling": {
                "positive_pairs": 10,
                "negative_pairs": 10,
                "hop_pairs": 10,
                "min_flow_threshold": 0.1,
            },
            "output": {"output_dir": "test_output", "experiment_name": "test_exp"},
            "clustering": {"enabled": False},
        }
        return RegionTrainerConfig.from_dict(raw_config, base_dir=Path("."))

    def test_initialization(self, trainer_config):
        """Test trainer initialization."""
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb") as mock_wandb,
        ):
            # Mock dataset
            mock_ds = MagicMock()
            mock_ds.get_all_features.return_value = torch.randn(10, 4)
            mock_ds.get_edge_index.return_value = torch.randint(0, 10, (2, 20))
            mock_ds.get_flow_matrix.return_value = torch.rand(10, 10)
            mock_ds.get_region_ids.return_value = list(range(10))
            mock_ds.flow_source = "mobility"
            mock_dataset_cls.return_value = mock_ds

            # Mock model parameters
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_cls.return_value = mock_model_instance

            # Initialize
            trainer = Region2VecTrainer(trainer_config)

            assert trainer.num_nodes == 10
            assert trainer.feature_dim == 4
            assert isinstance(trainer.optimizer, torch.optim.Adam)
            assert isinstance(trainer.primary_loss, CommunityOrientedLoss)

    def test_pair_sampling(self, trainer_config):
        """Test pair sampler initialization and sampling."""
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb") as mock_wandb,
        ):
            # Mock dataset
            mock_ds = MagicMock()
            mock_ds.get_all_features.return_value = torch.randn(10, 4)
            mock_ds.get_edge_index.return_value = torch.randint(0, 10, (2, 20))
            mock_ds.get_flow_matrix.return_value = torch.rand(10, 10)
            mock_ds.get_region_ids.return_value = list(range(10))
            mock_ds.flow_source = "mobility"
            mock_dataset_cls.return_value = mock_ds

            # Mock model parameters
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_cls.return_value = mock_model_instance

            trainer = Region2VecTrainer(trainer_config)

            # Test sample method
            samples = trainer.pair_sampler.sample()

            assert "positive_pairs" in samples
            assert "negative_pairs" in samples
            assert "hop_pairs" in samples

            assert samples["positive_pairs"].shape == (
                2,
                trainer_config.sampling.positive_pairs,
            )
            assert samples["negative_pairs"].shape == (
                2,
                trainer_config.sampling.negative_pairs,
            )

    def test_training_step(self, trainer_config):
        """Smoke test for training loop logic."""
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb") as mock_wandb,
        ):
            # Mock dataset
            mock_ds = MagicMock()
            mock_ds.get_all_features.return_value = torch.randn(10, 4)
            mock_ds.get_edge_index.return_value = torch.randint(0, 10, (2, 20))
            mock_ds.get_flow_matrix.return_value = torch.rand(10, 10)
            mock_ds.get_region_ids.return_value = list(range(10))
            mock_ds.flow_source = "mobility"
            mock_dataset_cls.return_value = mock_ds

            # Mock encoder forward pass
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]
            mock_model_instance.return_value = torch.randn(
                10, 8, requires_grad=True
            )  # Embeddings
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_cls.return_value = mock_model_instance

            trainer = Region2VecTrainer(trainer_config)

            # Run one epoch
            metrics = trainer._train_one_epoch(epoch=1)

            assert "total_loss" in metrics
            assert "base_loss" in metrics
            assert metrics["total_loss"] > 0

    def test_validation_logic(self, trainer_config):
        """Test validation/artifact saving logic."""
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb") as mock_wandb,
        ):
            # Mock dataset
            mock_ds = MagicMock()
            mock_ds.get_all_features.return_value = torch.randn(10, 4)
            mock_ds.get_edge_index.return_value = torch.randint(0, 10, (2, 20))
            mock_ds.get_flow_matrix.return_value = torch.rand(10, 10)
            mock_ds.get_region_ids.return_value = list(range(10))
            mock_ds.flow_source = "mobility"
            mock_dataset_cls.return_value = mock_ds

            # Mock encoder output
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [
                torch.nn.Parameter(torch.randn(1))
            ]

            # Return tensor WITH grad for training loop, then WITHOUT grad for validation/artifacts
            # Assuming 1 epoch, so 1 training call, then 1 validation call
            t_grad = torch.randn(10, 8, requires_grad=True)
            t_no_grad = torch.randn(10, 8, requires_grad=False)
            mock_model_instance.side_effect = [t_grad, t_no_grad]

            mock_model_instance.to.return_value = mock_model_instance
            mock_model_cls.return_value = mock_model_instance

            trainer = Region2VecTrainer(trainer_config)

            # Trigger run
            results = trainer.run()

            assert "best_loss" in results
            assert "artifacts" in results
