import pytest
from unittest.mock import MagicMock, patch
import torch
from pathlib import Path

from graph.node_encoder import Region2Vec
from training.region2vec_trainer import (
    EncoderConfig,
    Region2VecTrainer,
    RegionTrainerConfig,
)
from evaluation.region_losses import CommunityOrientedLoss


class TestRegion2VecTrainer:
    @pytest.fixture
    def trainer_config(self):
        # Create a minimal config using from_dict
        raw_config = {
            "data": {"zarr_path": "dummy_regions.zarr", "normalize_features": True},
            "encoder": {
                "hidden_dim": 16,
                "embedding_dim": 8,
                "activation": "gelu",
                "hidden_norm": "layer",
                "sage_normalize": False,
            },
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
            patch("training.region2vec_trainer.wandb"),
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
            mock_model_cls.assert_called_once_with(
                input_dim=4,
                hidden_dim=16,
                output_dim=8,
                num_layers=2,
                aggregation="mean",
                dropout=0.2,
                activation="gelu",
                hidden_norm="layer",
                sage_normalize=False,
                residual=False,
                normalize=True,
            )

    def test_encoder_config_validation(self):
        with pytest.raises(ValueError, match="encoder.activation"):
            EncoderConfig(activation="swish")

        with pytest.raises(ValueError, match="encoder.hidden_norm"):
            EncoderConfig(hidden_norm="instance")

    @pytest.mark.parametrize(
        ("hidden_norm", "expected_type"),
        [
            ("layer", torch.nn.LayerNorm),
            ("batch", torch.nn.BatchNorm1d),
            ("none", torch.nn.Identity),
        ],
    )
    def test_region2vec_hidden_norm_selection(self, hidden_norm, expected_type):
        encoder = Region2Vec(
            input_dim=4,
            hidden_dim=8,
            output_dim=4,
            num_layers=2,
            hidden_norm=hidden_norm,
        )

        assert isinstance(encoder.batch_norms[0], expected_type)

    def test_pair_sampling(self, trainer_config):
        """Test pair sampler initialization and sampling."""
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb"),
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
            patch("training.region2vec_trainer.wandb"),
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

    def test_validation_logic(self, trainer_config, tmp_path):
        """Test validation/artifact saving logic."""
        trainer_config.output.output_dir = tmp_path
        with (
            patch("data.region_graph_dataset.RegionGraphDataset") as mock_dataset_cls,
            patch("training.region2vec_trainer.Region2Vec") as mock_model_cls,
            patch("training.region2vec_trainer.wandb"),
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

            # Return tensor WITH grad for training loop, then WITHOUT grad for
            # partial and final artifacts.
            t_grad = torch.randn(10, 8, requires_grad=True)
            t_no_grad = torch.randn(10, 8, requires_grad=False)
            mock_model_instance.side_effect = [t_grad, t_no_grad, t_no_grad]

            mock_model_instance.to.return_value = mock_model_instance
            mock_model_cls.return_value = mock_model_instance

            trainer = Region2VecTrainer(trainer_config)

            # Trigger run
            results = trainer.run()

            assert "best_loss" in results
            assert "artifacts" in results
            assert (tmp_path / "region_embeddings.partial.pt").exists()
