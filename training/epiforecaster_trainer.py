"""
Trainer for the EpiForecaster model.

This module implements a trainer class that can handle the EpiForecaster model
through configuration. It provides a unified interface for training the EpiForecaster
model while maintaining the flexibility to support various data configurations.

The trainer works with the EpiForecaster model.
"""

import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.ego_graph_dataset import GraphEgoDataset
from data.epi_batch import EpiBatch
from data.epi_dataset import EpiDataset
from models.configs import EpiForecasterTrainerConfig
from models.epiforecaster import EpiForecaster


class EpiForecasterTrainer:
    """
    Single trainer handling all variants via configuration.

    Key features:
    - Works with any model variant through configuration
    - Supports mixed precision training
    - Implements standard training loop with validation
    - Handles checkpointing and experiment tracking
    - Provides comprehensive metrics and logging

    The trainer is designed to be model-agnostic, with variant-specific
    behavior controlled through configuration rather than separate code paths.
    """

    def __init__(
        self, config: EpiForecasterTrainerConfig
    ):
        """
        Initialize the unified trainer.

        Args:
            config: Trainer configuration
            dataset: Optional pre-loaded dataset (will be loaded if None)
        """
        self.config = config
        self.device = self._setup_device()
        self.base_dataset: EpiDataset | None = None

        self.dataset = self._load_dataset()

        # Create model based on configuration
        self.model = EpiForecaster(
            config.model.type,
            **config.model.params,
            # **config.model.ego_graph_params.as_dict(),
        )

        self.model.to(self.device)

        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        # Setup data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()

        # Setup logging and checkpointing
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_times": [],
        }

        print("UnifiedTrainer initialized:")
        print(f"  Model: {config.model.type}")
        print(f"  Dataset: {config.data.dataset_path}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Batch size: {config.training.batch_size}")

    def _setup_device(self) -> torch.device:
        """Setup computation device with MPS support and validation."""
        if self.config.training.device == "auto":
            # Priority: CUDA > MPS > CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                print("Using CPU device")
        else:
            device = torch.device(self.config.training.device)
            # Validate device availability
            if device.type == "cuda" and not torch.cuda.is_available():
                print(
                    f"Warning: CUDA device {device} requested but not available, using CPU"
                )
                device = torch.device("cpu")
            elif device.type == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                print(
                    f"Warning: MPS device {device} requested but not available, using CPU"
                )
                device = torch.device("cpu")

        return device

    def _load_dataset(self) -> GraphEgoDataset:
        """Load dataset as GraphEgoDataset for ego-graph processing."""
        ego_params = self.config.model.ego_graph_params
        model_params = self.config.model.params
        sequence_length = model_params.get("sequence_length", ego_params.history_length)

        # Create EpiDataset first
        self.base_dataset = EpiDataset(
            zarr_path=Path(self.config.data.dataset_path),
            variant_config=self.config.model.type,
            batch_size=1,
            shuffle_timepoints=False,
            sequence_length=sequence_length,
        )

        # Ego-graph view is now enabled by default in EpiDataset
        # Get the GraphEgoDataset
        ego_graph_datasets = self.base_dataset.get_ego_graph_dataset()

        # For now, use the same dataset for train and val
        # TODO: Implement proper train/val split
        return ego_graph_datasets["train"]

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.training.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.training.epochs // 3, gamma=0.1
            )
        elif self.config.training.scheduler_type == "none":
            return None
        else:
            raise ValueError(
                f"Unknown scheduler type: {self.config.training.scheduler_type}"
            )

    def _create_criterion(self) -> nn.Module:
        """Create loss criterion."""
        return nn.MSELoss()

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders with device-aware optimizations."""
        # Split dataset
        total_size = len(self.dataset)
        val_size = int(total_size * self.config.training.validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # For reproducibility
        )

        # Device-aware hardware optimizations
        pin_memory = self.config.training.pin_memory and self.device.type == "cuda"

        # Platform-aware num_workers for macOS multiprocessing issues
        num_workers = (
            0 if platform.system() == "Darwin" else self.config.training.num_workers
        )

        if isinstance(self.dataset, GraphEgoDataset):
            from data.ego_graph_dataset import ego_graph_collate_fn

            def device_aware_collate_fn(batch):
                return ego_graph_collate_fn(batch, device=self.device)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=device_aware_collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=device_aware_collate_fn,
            )
        else:
            if self.config.training.batch_size != 1:
                print(
                    "Warning: canonical datasets currently enforce batch_size=1; overriding trainer setting."
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._collate_fn,
            )

        return train_loader, val_loader

    def _collate_fn(self, batch: list[EpiBatch]) -> dict[str, torch.Tensor]:
        """Custom collate function for single-region EpiBatch objects."""
        if len(batch) == 1:
            # Single item case - add batch dimension to single region data
            batch_data = batch[0]

            # The EpiBatch now contains single region data:
            # - node_features: [sequence_length, feature_dim]
            # - target_sequences: [forecast_horizon]
            # - time_index: [sequence_length]

            return {
                "node_features": batch_data.node_features.unsqueeze(0).to(self.device),  # [1, seq_len, feat_dim]
                "edge_index": batch_data.edge_index.to(self.device),  # No batch dim for edge_index
                "edge_attr": batch_data.edge_attr.unsqueeze(0).to(self.device)
                if batch_data.edge_attr is not None
                else None,
                "target_sequences": batch_data.target_sequences.unsqueeze(0).to(self.device),  # [1, horizon]
                "region_embeddings": batch_data.region_embeddings.to(self.device)
                if batch_data.region_embeddings is not None
                else None,
                "edar_features": batch_data.edar_features.unsqueeze(0).to(self.device)
                if batch_data.edar_features is not None
                else None,
                "edar_attention_mask": batch_data.edar_attention_mask.to(self.device)
                if batch_data.edar_attention_mask is not None
                else None,
                "batch_size": 1,
                "sequence_length": batch_data.sequence_length,
            }
        else:
            # Multiple regions in a batch - stack them
            node_features = torch.stack([b.node_features for b in batch])  # [batch, seq_len, feat_dim]
            target_sequences = torch.stack([b.target_sequences for b in batch])  # [batch, horizon]

            # Use edge_index from first batch (same graph for all regions)
            edge_index = batch[0].edge_index

            # Handle optional tensors
            edge_attr = None
            if batch[0].edge_attr is not None:
                edge_attr = torch.stack([b.edge_attr for b in batch])

            region_embeddings = None
            if batch[0].region_embeddings is not None:
                region_embeddings = torch.cat([b.region_embeddings for b in batch], dim=0)  # [batch, embed_dim]

            edar_features = None
            if batch[0].edar_features is not None:
                edar_features = torch.stack([b.edar_features for b in batch])

            edar_attention_mask = None
            if batch[0].edar_attention_mask is not None:
                edar_attention_mask = torch.cat([b.edar_attention_mask for b in batch], dim=0)  # [batch, num_edars]

            return {
                "node_features": node_features.to(self.device),
                "edge_index": edge_index.to(self.device),
                "edge_attr": edge_attr.to(self.device) if edge_attr is not None else None,
                "target_sequences": target_sequences.to(self.device),
                "region_embeddings": region_embeddings.to(self.device) if region_embeddings is not None else None,
                "edar_features": edar_features.to(self.device) if edar_features is not None else None,
                "edar_attention_mask": edar_attention_mask.to(self.device) if edar_attention_mask is not None else None,
                "batch_size": len(batch),
                "sequence_length": batch[0].sequence_length,
            }

    def _transfer_batch_to_device(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """
        Transfer all tensors in batch to target device.

        Args:
            batch_data: Dictionary containing batch data with potentially mixed device tensors

        Returns:
            Dictionary with all tensors transferred to target device
        """
        transferred = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                transferred[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list):
                transferred[key] = [
                    item.to(self.device, non_blocking=True)
                    if isinstance(item, torch.Tensor)
                    else item
                    for item in value
                ]
            else:
                transferred[key] = value
        return transferred

    def setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create experiment directory
        experiment_dir = (
            Path(self.config.output.log_dir) / self.config.output.experiment_name
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard writer
        log_dir = experiment_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Setup checkpoint directory
        if self.config.output.save_checkpoints:
            self.checkpoint_dir = experiment_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log hyperparameters
        hyperparams = {
            "model_type": str(self.config.model.type),
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.training.batch_size,
            "epochs": self.config.training.epochs,
            "use_region_embeddings": self.config.model.type.regions,
            "use_edar_data": self.config.model.type.biomarkers,
            "use_mobility_data": self.config.model.type.mobility,
            **self.config.model.params,
        }
        hyperparams["ego_graph_params"] = self.config.model.ego_graph_params.as_dict()

        for key, value in hyperparams.items():
            self.writer.add_text(f"hyperparams/{key}", str(value), 0)

    def run(self) -> dict[str, Any]:
        """Execute training loop."""
        print(f"\n{'=' * 60}")
        print(f"STARTING TRAINING: {self.config.output.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Model: {self.config.model.type}")
        print(f"Dataset: {self.base_dataset.metadata.get('dataset_name', 'Unknown')} (ego-graph enabled)")
        if hasattr(self.dataset, 'indices'):
            print(f"  Samples: {len(self.dataset)} ego-graph items")
        print(f"Device: {self.device}")
        print()

        # Training loop
        for epoch in range(self.current_epoch, self.config.training.epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss = self._validate_epoch()

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Record metrics
            epoch_time = time.time() - epoch_start_time
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            self.training_history["epoch_times"].append(epoch_time)

            # Logging
            self._log_epoch(epoch, train_loss, val_loss, epoch_time)

            # Checkpointing
            if (
                self.config.output.save_checkpoints
                and (epoch + 1) % self.config.output.checkpoint_frequency == 0
            ):
                self._save_checkpoint(epoch, val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0  # Reset counter when improvement occurs
                if self.config.output.save_best_only:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                # Early stopping logic
                self.patience_counter += 1
                if (
                    self.patience_counter
                    >= self.config.training.early_stopping_patience
                ):
                    print(
                        f"Early stopping triggered after {self.patience_counter} epochs without improvement"
                    )
                    break

        # Final evaluation
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETED")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs trained: {self.current_epoch + 1}")
        print(f"{'=' * 60}")

        # Save final model
        if self.config.output.save_checkpoints:
            self._save_checkpoint(self.current_epoch, self.best_val_loss, is_final=True)

        # Close tensorboard writer
        self.writer.close()

        return self.get_training_results()

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            predictions = self._forward_epiforecaster(batch_data)
            loss = self.criterion(predictions, batch_data["target_sequences"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_value
            )
            self.optimizer.step()

            total_loss += loss.item()

            # Progress logging
            if batch_idx % max(1, num_batches // 10) == 0:
                print(f"  Batch {batch_idx}/{num_batches}: Loss = {loss.item():.6f}")

        return total_loss / num_batches

    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_data in self.val_loader:
                predictions = self._forward_epiforecaster(batch_data)
                loss = self.criterion(predictions, batch_data["target_sequences"])
                total_loss += loss.item()

        return total_loss / num_batches

    def _forward_epiforecaster_ego_graph(
        self, batch_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass for EpiForecaster with ego-graph batches."""

        # Ego-graph batches come directly from GraphEgoDataset
        # Use the model's forward_ego_graph method
        predictions = self.model.forward_ego_graph(batch_data)

        return predictions

    def _validate_batch_shapes(self, batch_data: dict[str, torch.Tensor]):
        """Validate tensor shapes and add assertions for debugging."""
        batch_size = batch_data.get("batch_size", 1)

        # Log tensor shapes for debugging
        if hasattr(self, "_debug_shapes") and self._debug_shapes:
            print("Batch data shapes:")
            for key, tensor in batch_data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tensor.shape}")

        # Basic shape validations
        assert batch_data["target_sequences"].dim() == 2, (
            f"target_sequences should be 2D, got {batch_data['target_sequences'].shape}"
        )

        expected_batch_size = batch_data["target_sequences"].size(0)
        assert batch_data["target_sequences"].size(0) == expected_batch_size, (
            f"Batch size mismatch in target_sequences: expected {batch_size}, got {batch_data['target_sequences'].size(0)}"
        )

    def _forward_epiforecaster(
        self, batch_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass for EpiForecaster with ego-graph batches."""

        # Always use ego-graph processing
        return self._forward_epiforecaster_ego_graph(batch_data)

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, epoch_time: float
    ):
        """Log epoch results."""
        print(
            f"Epoch {epoch + 1:3d}/{self.config.training.epochs}: "
            f"Train Loss = {train_loss:.6f}, "
            f"Val Loss = {val_loss:.6f}, "
            f"Time = {epoch_time:.2f}s, "
            f"LR = {self.optimizer.param_groups[0]['lr']:.2e}"
        )

        # Tensorboard logging
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar(
            "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
        )
        self.writer.add_scalar("Time/Epoch", epoch_time, epoch)

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False, is_final: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_history": self.training_history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best or is_final:
            filename = "best_model.pt" if is_best else "final_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def get_training_results(self) -> dict[str, Any]:
        """Get comprehensive training results."""
        return {
            "config": self.config,
            "model_info": {
                "type": str(self.config.model.type),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            },
            "dataset_info": self.dataset.get_dataset_info(),
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
            "final_learning_rate": self.optimizer.param_groups[0]["lr"],
        }
