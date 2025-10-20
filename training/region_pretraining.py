"""
Training pipeline for region embedding pretraining.

Implements unsupervised pretraining with community-oriented loss,
validation metrics, and model checkpointing for region2vec approach.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from data.region_data import RegionDataProcessor, create_region_data_processor
from models.region_embedding import RegionEmbedder, create_region_embedder

logger = logging.getLogger(__name__)


class RegionPretrainer:
    """
    Training pipeline for region embedding pretraining.

    Handles unsupervised training with community-oriented loss,
    validation, checkpointing, and embedding quality assessment.
    """

    def __init__(
        self,
        model: RegionEmbedder,
        data_processor: RegionDataProcessor,
        optimizer_config: Optional[dict[str, Any]] = None,
        scheduler_config: Optional[dict[str, Any]] = None,
        training_config: Optional[dict[str, Any]] = None,
        checkpoint_dir: str = "./checkpoints/region_embedding",
        device: str = "auto",
    ):
        """
        Initialize region pretrainer.

        Args:
            model: RegionEmbedder model
            data_processor: RegionDataProcessor for data handling
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            training_config: Training hyperparameters
            checkpoint_dir: Directory for model checkpoints
            device: Device for training ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.data_processor = data_processor
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Training configuration
        self.training_config = training_config or {}
        self.epochs = self.training_config.get("epochs", 200)
        self.validation_freq = self.training_config.get("validation_freq", 10)
        self.checkpoint_freq = self.training_config.get("checkpoint_freq", 25)
        self.early_stopping_patience = self.training_config.get(
            "early_stopping_patience", 50
        )
        self.num_negative_samples = self.training_config.get(
            "num_negative_samples", None
        )

        # Optimizer setup
        optimizer_config = optimizer_config or {}
        optimizer_type = optimizer_config.get("type", "adam")
        optimizer_params = optimizer_config.get("params", {})

        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_params.get("lr", 0.001),
                weight_decay=optimizer_params.get("weight_decay", 1e-4),
                **{
                    k: v
                    for k, v in optimizer_params.items()
                    if k not in ["lr", "weight_decay"]
                },
            )
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_params.get("lr", 0.001),
                weight_decay=optimizer_params.get("weight_decay", 1e-2),
                **{
                    k: v
                    for k, v in optimizer_params.items()
                    if k not in ["lr", "weight_decay"]
                },
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Scheduler setup
        scheduler_config = scheduler_config or {}
        scheduler_type = scheduler_config.get("type", "plateau")
        scheduler_params = scheduler_config.get("params", {})

        if scheduler_type.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_params.get("factor", 0.5),
                patience=scheduler_params.get("patience", 20),
                verbose=True,
            )
        elif scheduler_type.lower() == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", 50),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        else:
            self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.training_history = []
        self.early_stopping_counter = 0

    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        flow_matrix: torch.Tensor,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            x: Node features
            edge_index: Spatial adjacency edges
            flow_matrix: Mobility flow matrix

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        # Move data to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        flow_matrix = flow_matrix.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass and compute loss
        loss_dict = self.model.compute_loss(
            x, edge_index, flow_matrix, self.num_negative_samples
        )

        total_loss = loss_dict["total_loss"]

        # Backward pass
        total_loss.backward()

        # Gradient clipping (optional)
        if self.training_config.get("gradient_clipping"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config["gradient_clipping"]
            )

        # Optimizer step
        self.optimizer.step()

        # Convert loss dict to float for logging
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

        return metrics

    def validate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        flow_matrix: torch.Tensor,
    ) -> dict[str, float]:
        """
        Validate model (same as training but without gradients).

        Args:
            x: Node features
            edge_index: Spatial adjacency edges
            flow_matrix: Mobility flow matrix

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            flow_matrix = flow_matrix.to(self.device)

            # Compute loss
            loss_dict = self.model.compute_loss(
                x, edge_index, flow_matrix, self.num_negative_samples
            )

            # Convert to float
            metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()
            }

            # Add embedding quality metrics
            embeddings = self.model.get_static_embeddings(x, edge_index)
            quality_metrics = self._compute_embedding_quality(embeddings, flow_matrix)
            metrics.update(quality_metrics)

        return metrics

    def _compute_embedding_quality(
        self, embeddings: torch.Tensor, flow_matrix: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute embedding quality metrics.

        Args:
            embeddings: Node embeddings
            flow_matrix: Flow matrix

        Returns:
            Dictionary with quality metrics
        """
        with torch.no_grad():
            # Embedding statistics
            embed_mean = embeddings.mean().item()
            embed_std = embeddings.std().item()
            embed_norm = torch.norm(embeddings, dim=1).mean().item()

            # Pairwise similarities
            similarities = torch.mm(embeddings, embeddings.T)
            similarity_mean = similarities.mean().item()
            similarity_std = similarities.std().item()

            # Flow-similarity correlation (simplified)
            flow_flat = flow_matrix.flatten()
            sim_flat = similarities.flatten()

            # Remove self-similarities and zero flows
            mask = (
                torch.eye(flow_matrix.size(0), device=flow_matrix.device).flatten() == 0
            ) & (flow_flat > 0)
            if mask.sum() > 0:
                masked_flow = flow_flat[mask]
                masked_sim = sim_flat[mask]
                flow_sim_corr = torch.corrcoef(torch.stack([masked_flow, masked_sim]))[
                    0, 1
                ].item()
            else:
                flow_sim_corr = 0.0

        return {
            "embed_mean": embed_mean,
            "embed_std": embed_std,
            "embed_norm": embed_norm,
            "similarity_mean": similarity_mean,
            "similarity_std": similarity_std,
            "flow_sim_corr": flow_sim_corr,
        }

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far

        Returns:
            Checkpoint file path
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "metrics": metrics,
            "training_history": self.training_history,
            "config": {
                "training": self.training_config,
                "model": self.model.__dict__.copy()
                if hasattr(self.model, "__dict__")
                else {},
            },
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint

    def train(
        self,
        dataset: dict[str, torch.Tensor],
        validation_split: float = 0.0,
        resume_from: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run complete training loop.

        Args:
            dataset: Region dataset dictionary
            validation_split: Fraction of data for validation (0.0 = no validation)
            resume_from: Path to checkpoint to resume from

        Returns:
            Training results dictionary
        """
        logger.info("Starting region embedding pretraining")

        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Extract data
        x = dataset["node_features"]
        edge_index = dataset["edge_index"]
        flow_matrix = dataset["flow_matrix"]

        # Split data if validation requested
        if validation_split > 0:
            num_nodes = x.size(0)
            num_val = int(validation_split * num_nodes)
            indices = torch.randperm(num_nodes)

            train_idx = indices[num_val:]
            val_idx = indices[:num_val]

            x_train, x_val = x[train_idx], x[val_idx]
            flow_train = flow_matrix[torch.ix_(train_idx, train_idx)]
            flow_val = flow_matrix[torch.ix_(val_idx, val_idx)]
        else:
            x_train, x_val = x, None
            flow_train, flow_val = flow_matrix, None

        # Training loop
        pbar = tqdm(range(self.current_epoch, self.epochs), desc="Training")

        for epoch in pbar:
            self.current_epoch = epoch

            # Training step
            train_metrics = self.train_epoch(x_train, edge_index, flow_train)

            # Validation step
            val_metrics = {}
            if x_val is not None and epoch % self.validation_freq == 0:
                val_metrics = self.validate(x_val, edge_index, flow_val)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.training_history.append(epoch_metrics)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{train_metrics['total_loss']:.4f}",
                    "cont": f"{train_metrics['contrastive_loss']:.4f}",
                    "spatial": f"{train_metrics['spatial_penalty']:.4f}",
                }
            )

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(train_metrics["total_loss"])
                else:
                    self.scheduler.step()

            # Check for best model
            current_loss = val_metrics.get(
                "val_total_loss", train_metrics["total_loss"]
            )
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            if epoch % self.checkpoint_freq == 0 or is_best:
                self.save_checkpoint(epoch, epoch_metrics, is_best)

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final results
        results = {
            "final_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "training_history": self.training_history,
            "final_embeddings": self.model.get_static_embeddings(x, edge_index)
            .detach()
            .cpu(),
        }

        # Optional clustering
        if self.model.clustering_config:
            logger.info("Computing final clustering")
            clustering_results = self.model.precompute_embeddings_and_clusters(
                x, edge_index
            )
            results.update(clustering_results)

        # Save final results
        results_path = self.checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            # Convert tensors to lists for JSON serialization
            json_results = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in results.items()
                if k != "final_embeddings"  # Skip large tensor
            }
            json.dump(json_results, f, indent=2)

        logger.info(f"Training completed. Results saved to {results_path}")
        return results


def create_region_pretrainer(
    config: dict[str, Any],
    data_processor: Optional[RegionDataProcessor] = None,
) -> RegionPretrainer:
    """
    Factory function to create RegionPretrainer from configuration.

    Args:
        config: Complete training configuration
        data_processor: Optional data processor (created from config if None)

    Returns:
        Configured RegionPretrainer
    """
    # Create data processor if not provided
    if data_processor is None:
        data_processor = create_region_data_processor(config.get("data", {}))

    # Create model
    model = create_region_embedder(config.get("model", {}))

    # Create trainer
    return RegionPretrainer(
        model=model,
        data_processor=data_processor,
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {}),
        training_config=config.get("training", {}),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints/region_embedding"),
        device=config.get("device", "auto"),
    )


if __name__ == "__main__":
    # Example usage and testing
    from ..models.region_embedding import RegionEmbedder

    # Create dummy dataset
    num_nodes = 50
    feature_dim = 32

    dataset = {
        "node_features": torch.randn(num_nodes, feature_dim),
        "edge_index": torch.randint(0, num_nodes, (2, 200)),
        "edge_weights": torch.ones(200),
        "flow_matrix": torch.rand(num_nodes, num_nodes) * 100,
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
    }

    # Training configuration
    config = {
        "model": {
            "model": {"input_dim": feature_dim, "embed_dim": 16},
            "loss": {"temperature": 0.1},
            "clustering": {"n_clusters": 5},
        },
        "training": {"epochs": 10, "validation_freq": 5},
        "optimizer": {"type": "adam", "params": {"lr": 0.01}},
        "checkpoint_dir": "./test_checkpoints",
    }

    # Create and test trainer
    trainer = create_region_pretrainer(config)
    results = trainer.train(dataset)

    print(f"Training completed in {results['final_epoch']} epochs")
    print(f"Best loss: {results['best_loss']:.4f}")
    print(f"Final embeddings shape: {results['final_embeddings'].shape}")

    print("Region pretraining pipeline initialized successfully")
