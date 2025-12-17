"""
Trainer for the EpiForecaster model.

This module implements a trainer class that can handle the EpiForecaster model
through configuration. It provides a unified interface for training the EpiForecaster
model while maintaining the flexibility to support various data configurations.

The trainer works with the EpiForecaster model.
"""

import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from data.epi_dataset import EpiDataset, EpiDatasetItem
from data.preprocess.config import REGION_COORD
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster


class EpiForecasterTrainer:
    """
    Single trainer handling all variants via configuration.

    Key features:
    - Works with any model variant through EpiForecasterConfig
    - Handles checkpointing and experiment tracking
    - Provides comprehensive metrics and logging

    The trainer is designed to be model-agnostic, with variant-specific behavior controlled through model configuration.
    """

    def __init__(self, config: EpiForecasterConfig):
        """
        Initialize the unified trainer.

        Args:
            config: Trainer configuration
            dataset: Optional pre-loaded dataset (will be loaded if None)
        """
        self.config = config
        self.device = self._setup_device()
        self.use_tqdm = getattr(self.config.training, "use_tqdm", True)

        train_nodes, val_nodes, test_nodes = self._split_dataset()
        train_nodes = list(train_nodes)
        val_nodes = list(val_nodes)
        test_nodes = list(test_nodes)

        region2vec_path = (
            Path(self.config.data.region2vec_path)
            if self.config.model.type.regions
            else None
        )

        self.train_dataset = EpiDataset(
            aligned_data_path=Path(self.config.data.dataset_path),
            region2vec_path=region2vec_path,
            config=self.config,
            target_nodes=train_nodes,
            context_nodes=train_nodes,
        )

        self.val_dataset = EpiDataset(
            aligned_data_path=Path(self.config.data.dataset_path),
            region2vec_path=region2vec_path,
            config=self.config,
            target_nodes=val_nodes,
            context_nodes=train_nodes + val_nodes,
        )

        self.test_dataset = EpiDataset(
            aligned_data_path=Path(self.config.data.dataset_path),
            region2vec_path=region2vec_path,
            config=self.config,
            target_nodes=test_nodes,
            context_nodes=train_nodes + val_nodes,
        )

        # Optional static region embeddings from dataset
        self.region_embeddings = None
        if hasattr(self.train_dataset, "region_embeddings"):
            self.region_embeddings = self.train_dataset.region_embeddings.to(
                self.device
            )
        elif self.config.model.type.regions:
            raise ValueError(
                "Region embeddings requested by config but region2vec_path was not provided."
            )

        # Create model based on configuration
        self.model = EpiForecaster(
            variant_type=self.config.model.type,
            cases_dim=self.config.model.cases_dim,
            biomarkers_dim=self.config.model.biomarkers_dim,
            region_embedding_dim=self.config.model.region_embedding_dim,
            mobility_embedding_dim=self.config.model.mobility_embedding_dim,
            gnn_depth=self.config.model.gnn_depth,
            sequence_length=self.config.model.history_length,
            forecast_horizon=self.config.model.forecast_horizon,
            use_population=self.config.model.use_population,
            population_dim=self.config.model.population_dim,
            device=self.device,
            gnn_module=self.config.model.gnn_module,
        )

        self.model.to(self.device)

        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = (
            self._create_data_loaders()
        )

        # Setup logging and checkpointing
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_smape": [],
            "val_r2": [],
            "learning_rate": [],
            "epoch_times": [],
        }
        self._model_graph_logged = False

        print("EpiForecasterTrainer initialized:")
        print(f"  Model: {config.model.type}")
        print(f"  Dataset: {config.data.dataset_path}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Batch size: {config.training.batch_size}")

    def _split_dataset(self) -> tuple[list[int], list[int], list[int]]:
        """
        Split the dataset into train, val, and test sets.
        We use node holdouts for splitting so that we can evaluate the ability of the model to generalize to new regions.
        """
        train_split = (
            1 - self.config.training.val_split - self.config.training.test_split
        )

        aligned_dataset = EpiDataset.load_canonical_dataset(
            Path(self.config.data.dataset_path)
        )
        N = aligned_dataset[REGION_COORD].size
        all_nodes = np.arange(N)
        rng = np.random.default_rng(42)
        rng.shuffle(all_nodes)
        n_train = int(len(all_nodes) * train_split)
        n_val = int(len(all_nodes) * self.config.training.val_split)
        train_nodes = all_nodes[:n_train]
        val_nodes = all_nodes[n_train : n_train + n_val]
        test_nodes = all_nodes[n_train + n_val :]

        assert len(train_nodes) + len(val_nodes) + len(test_nodes) == len(all_nodes), (
            "Dataset split is not correct"
        )

        return list(train_nodes), list(val_nodes), list(test_nodes)

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

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
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

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create training and validation data loaders with device-aware optimizations."""
        # Device-aware hardware optimizations
        pin_memory = self.config.training.pin_memory and self.device.type == "cuda"

        # Platform-aware num_workers for macOS multiprocessing issues
        if platform.system() == "Darwin":
            num_workers = 0
        else:
            avail_cores = (os.cpu_count() or 1) - 1
            cfg_workers = self.config.training.num_workers
            if cfg_workers == -1:
                num_workers = avail_cores
            else:
                num_workers = min(avail_cores, cfg_workers)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )
        return train_loader, val_loader, test_loader

    def _collate_fn(self, batch: list[EpiDatasetItem]) -> dict[str, Any]:
        "Custom collate for per-node samples with PyG mobility graphs."

        B = len(batch)
        case_node = torch.stack([item["case_node"] for item in batch], dim=0)
        bio_node = torch.stack([item["bio_node"] for item in batch], dim=0)
        targets = torch.stack([item["target"] for item in batch], dim=0)
        target_nodes = torch.tensor(
            [item["target_node"] for item in batch], dtype=torch.long
        )
        population = torch.stack([item["population"] for item in batch], dim=0)

        # Flatten mobility graphs and annotate batch_id/time_id for batching
        graph_list = []
        for b, item in enumerate(batch):
            for t, g in enumerate(item["mob"]):
                g.batch_id = torch.tensor([b], dtype=torch.long)
                g.time_id = torch.tensor([t], dtype=torch.long)
                graph_list.append(g)

        mob_batch = Batch.from_data_list(graph_list)
        T = len(batch[0]["mob"]) if B > 0 else 0
        # store B and T on the batch for downstream reshaping
        mob_batch.B = torch.tensor([B], dtype=torch.long)
        mob_batch.T = torch.tensor([T], dtype=torch.long)
        # Precompute a global target node index per ego-graph in the batched `x`.
        # This enables fully-vectorized target gathering in the model without CUDA `.item()` syncs.
        if hasattr(mob_batch, "ptr") and hasattr(mob_batch, "target_node"):
            mob_batch.target_index = mob_batch.ptr[:-1] + mob_batch.target_node.reshape(
                -1
            )

        return {
            "CaseNode": case_node,  # (B, L, C)
            "BioNode": bio_node,  # (B, L, B)
            "MobBatch": mob_batch,  # Batched PyG graphs
            "Population": population,  # (B,)
            "B": B,
            "T": T,
            "Target": targets,  # (B, H)
            "TargetNode": target_nodes,  # (B,)
            "NodeLabels": [item["node_label"] for item in batch],
        }

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
            "use_biomarkers": self.config.model.type.biomarkers,
            "use_mobility": self.config.model.type.mobility,
            "history_length": self.config.model.history_length,
            "forecast_horizon": self.config.model.forecast_horizon,
            "cases_dim": self.config.model.cases_dim,
            "biomarkers_dim": self.config.model.biomarkers_dim,
            "mobility_embedding_dim": self.config.model.mobility_embedding_dim,
            "region_embedding_dim": self.config.model.region_embedding_dim,
            "use_population": self.config.model.use_population,
            "population_dim": self.config.model.population_dim,
        }

        for key, value in hyperparams.items():
            self.writer.add_text(f"hyperparams/{key}", str(value), 0)

        # Enable TensorBoard Projector by logging embeddings (if available)
        self._log_projector_embeddings()

    # def _log_model_graph(self):
    #     """
    #     Write the model graph to TensorBoard using a real minibatch.

    #     This runs once before training to make the module shapes discoverable in the
    #     TensorBoard Graph tab. Failures are non-fatal to avoid blocking training on
    #     tracing issues with complex inputs (e.g., PyG batches).
    #     """
    #     if self._model_graph_logged:
    #         return

    #     try:
    #         example_batch = next(iter(self.train_loader))
    #     except StopIteration:
    #         print(
    #             "Skipping TensorBoard graph logging: training dataset is empty."
    #         )
    #         self._model_graph_logged = True
    #         return

    #     was_training = self.model.training
    #     self.model.eval()

    #     try:
    #         mob_batch = example_batch["MobBatch"].to(self.device)
    #         example_inputs = (
    #             example_batch["CaseNode"].to(self.device),
    #             example_batch["BioNode"].to(self.device),
    #             mob_batch,
    #             example_batch["TargetNode"].to(self.device),
    #             self.region_embeddings
    #             if self.region_embeddings is not None
    #             else None,
    #             example_batch["Population"].to(self.device),
    #         )
    #         with torch.no_grad():
    #             self.writer.add_graph(self.model, example_inputs, verbose=False)
    #         self._model_graph_logged = True
    #     except Exception as exc:  # pragma: no cover - trace failures are non-fatal
    #         print(f"Skipping TensorBoard graph logging: {exc}")
    #         self._model_graph_logged = True
    #     finally:
    #         if was_training:
    #             self.model.train()

    def _log_projector_embeddings(self):
        """
        Export embeddings to TensorBoard Projector.

        TensorBoard's projector tab expects either a checkpoint or an
        embedding summary (tensors + metadata). When we only write scalar
        summaries, the projector UI shows "No checkpoint was found". By
        proactively writing embeddings here, the projector can render
        without TensorFlow checkpoints.
        """
        if self.region_embeddings is None:
            return

        # Use region labels from the dataset for metadata so points are readable
        region_labels = [
            str(label) for label in self.train_dataset.dataset[REGION_COORD].values
        ]

        # The projector expects CPU tensors; ensure a detached copy
        self.writer.add_embedding(
            mat=self.region_embeddings.detach().cpu(),
            metadata=region_labels,
            tag="region_embeddings/static",
            global_step=0,
        )

    def _setup_profiler(self):
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profile_log_dir = self._resolve_profiler_log_dir()
        profile_log_dir.mkdir(parents=True, exist_ok=True)

        return profile(
            activities=activities,
            schedule=schedule(
                wait=self.config.training.profiler.wait_steps,
                warmup=self.config.training.profiler.warmup_steps,
                active=self.config.training.profiler.active_steps,
                repeat=self.config.training.profiler.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(str(profile_log_dir)),
            record_shapes=True,
            profile_memory=self.config.training.profiler.record_memory,
            with_stack=self.config.training.profiler.with_stack,
        )

    def _resolve_profiler_log_dir(self) -> Path:
        configured = getattr(self.config.training.profiler, "log_dir", "auto")
        if configured == "auto":
            writer_log_dir = getattr(self.writer, "log_dir", None)
            if writer_log_dir is not None:
                return Path(writer_log_dir)

            return Path(self.config.output.log_dir) / self.config.output.experiment_name

        return Path(configured)

    def run(self) -> dict[str, Any]:
        """Execute training loop."""
        print(f"\n{'=' * 60}")
        print(f"STARTING TRAINING: {self.config.output.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Model: {self.config.model.type}")
        print(f"Dataset: {self.config.data.dataset_path} (ego-graph per-node)")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples:   {len(self.val_dataset)}")
        print(f"Device: {self.device}")
        writer_log_dir = getattr(self.writer, "log_dir", None)
        if writer_log_dir is not None:
            print(f"TensorBoard: {writer_log_dir}")
        if self.config.training.profiler.enabled:
            print(f"Profiler: {self._resolve_profiler_log_dir()}")
        print()

        # Training loop
        # self._log_model_graph()

        epoch_bar = None
        if self.use_tqdm:
            epoch_bar = tqdm(
                total=self.config.training.epochs,
                initial=self.current_epoch,
                desc="Epoch progress",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
            )
            epoch_bar.set_postfix(prev_val=float("inf"))

        try:
            prev_val_loss = float("inf")
            for epoch in range(self.current_epoch, self.config.training.epochs):
                self.current_epoch = epoch
                if epoch_bar is not None:
                    epoch_bar.set_postfix(
                        epoch=f"{epoch + 1}/{self.config.training.epochs}",
                        prev_val=prev_val_loss,
                        best=self.best_val_loss,
                    )

                _train_loss = self._train_epoch()

                # Validation phase
                val_loss, val_metrics = self._evaluate_split(
                    self.val_loader, split_name="Val"
                )
                self._log_epoch(
                    split_name="Val", loss=val_loss, metrics=val_metrics, epoch=epoch
                )

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step()

                # Checkpointing
                if (
                    self.config.output.save_checkpoints
                    and (epoch + 1) % self.config.output.checkpoint_frequency == 0
                ):
                    self._save_checkpoint(epoch, val_loss)

                # Early stopping
                should_stop = False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.config.output.save_best_only:
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.patience_counter += 1
                    if (
                        self.patience_counter
                        >= self.config.training.early_stopping_patience
                    ):
                        self._status(
                            "Early stopping triggered after "
                            f"{self.patience_counter} epochs without improvement"
                        )
                        should_stop = True

                if epoch_bar is not None:
                    epoch_bar.update(1)
                    epoch_bar.set_postfix(
                        epoch=f"{epoch + 1}/{self.config.training.epochs}",
                        val=f"{val_loss:.4f}",
                        best=f"{self.best_val_loss:.4f}",
                    )
                prev_val_loss = val_loss

                if should_stop:
                    break
        finally:
            if epoch_bar is not None:
                epoch_bar.close()

        # Final evaluation
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETED")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs trained: {self.current_epoch + 1}")
        print(f"{'=' * 60}")

        # Save final model
        if self.config.output.save_checkpoints:
            self._save_checkpoint(self.current_epoch, self.best_val_loss, is_final=True)

        # Test phase
        test_start_time = time.time()
        test_loss, test_metrics = self.test_epoch()
        test_time = time.time() - test_start_time
        print(f"{'=' * 60}")
        print("TESTING COMPLETED")
        print(
            f"Test loss: {test_loss:.6f} | "
            f"MAE: {test_metrics['mae']:.6f} | "
            f"RMSE: {test_metrics['rmse']:.6f} | "
            f"sMAPE: {test_metrics['smape']:.6f} | "
            f"R2: {test_metrics['r2']:.6f} | "
            f"Time: {test_time:.2f}s"
        )
        print(f"{'=' * 60}")

        # Close tensorboard writer
        self.writer.close()

        return self.get_training_results()

    def _status(self, message: str) -> None:
        if self.use_tqdm:
            tqdm.write(message)
        else:
            print(message)

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        counted_batches = 0

        train_iter = self.train_loader
        if self.use_tqdm:
            train_iter = tqdm(
                self.train_loader,
                desc="Train batch progress",
                leave=False,
                total=num_batches,
                position=1,
                dynamic_ncols=True,
            )

        profiler = None
        profiler_active = False
        profiler_complete_announced = False
        if self.config.training.profiler.enabled:
            profiler = self._setup_profiler()
            profiler.__enter__()
            profiler_active = True
            self._status("==== PROFILING ACTIVE ====")

        fetch_start_time = time.time()
        try:
            for batch_idx, batch_data in enumerate(train_iter):
                # print(f"Start train iteration {batch_idx + 1}/{num_batches}")
                self.optimizer.zero_grad()

                data_time_s = time.time() - fetch_start_time
                batch_start_time = time.time()
                predictions = self.model.forward(
                    cases_hist=batch_data["CaseNode"].to(self.device),
                    biomarkers_hist=batch_data["BioNode"].to(self.device),
                    mob_graphs=batch_data["MobBatch"],
                    target_nodes=batch_data["TargetNode"].to(self.device),
                    region_embeddings=self.region_embeddings
                    if self.region_embeddings is not None
                    else None,
                    population=batch_data["Population"].to(self.device),
                )

                loss = self.criterion(predictions, batch_data["Target"].to(self.device))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_value
                )
                self.optimizer.step()

                total_loss += loss.item()
                counted_batches += 1

                batch_time_s = time.time() - batch_start_time
                fetch_start_time = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                # Progress logging
                if self.use_tqdm:
                    bsz = int(batch_data["CaseNode"].shape[0])
                    samples_per_s = (
                        (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                    )
                    train_iter.set_postfix(
                        loss=loss.item(),
                        lr=f"{lr:.2e}",
                        grad=f"{float(grad_norm):.3f}",
                        sps=f"{samples_per_s:7.1f}",
                    )

                # Tensorboard: per-iteration metrics
                self.writer.add_scalar(
                    "Loss/Train_step", loss.item(), self.global_step
                )
                self.writer.add_scalar("Learning_Rate/step", lr, self.global_step)
                self.writer.add_scalar(
                    "GradNorm/step", float(grad_norm), self.global_step
                )
                self.writer.add_scalar("Time/Batch_s", batch_time_s, self.global_step)
                self.writer.add_scalar("Time/DataLoad_s", data_time_s, self.global_step)
                self.writer.add_scalar("Time/Step_s", batch_time_s, self.global_step)

                self.global_step += 1

                if profiler_active and profiler is not None:
                    profiler.step()
                    if (
                        self.config.training.profiler.profile_batches is not None
                        and (batch_idx + 1)
                        >= self.config.training.profiler.profile_batches
                    ):
                        profiler.__exit__(None, None, None)
                        profiler_active = False
                        profiler = None
                        if not profiler_complete_announced:
                            self._status("==== PROFILING COMPLETE ====")
                            profiler_complete_announced = True

                # print(f"End train iteration {batch_idx + 1}/{num_batches}")
        finally:
            if profiler_active and profiler is not None:
                profiler.__exit__(None, None, None)
                if not profiler_complete_announced:
                    self._status("==== PROFILING COMPLETE ====")
                    profiler_complete_announced = True

        effective_batches = max(1, counted_batches)
        return total_loss / effective_batches

    def _evaluate_split(
        self, loader: DataLoader, *, split_name: str = "Eval"
    ) -> tuple[float, dict[str, Any]]:
        """Shared evaluation for validation and test splits with extra metrics."""
        self.model.eval()

        total_loss = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        smape_sum = 0.0

        # Running statistics for stable R2 (Welford)
        total_count = 0
        target_mean = 0.0
        target_m2 = 0.0

        horizon = self.config.model.forecast_horizon
        per_h_mae_sum = torch.zeros(horizon)
        per_h_mse_sum = torch.zeros(horizon)

        num_batches = len(loader)
        eval_iter = loader
        if self.use_tqdm:
            eval_iter = tqdm(
                loader,
                desc=f"{split_name} batch progress",
                leave=False,
                total=num_batches,
                position=1,
                dynamic_ncols=True,
            )

        epsilon = 1e-6
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_iter):
                batch_start_time = time.time()

                targets = batch_data["Target"].to(self.device)
                predictions = self.model.forward(
                    cases_hist=batch_data["CaseNode"].to(self.device),
                    biomarkers_hist=batch_data["BioNode"].to(self.device),
                    mob_graphs=batch_data["MobBatch"],
                    target_nodes=batch_data["TargetNode"].to(self.device),
                    region_embeddings=self.region_embeddings.to(self.device)
                    if self.region_embeddings is not None
                    else None,
                    population=batch_data["Population"].to(self.device),
                )

                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

                diff = predictions - targets
                abs_diff = diff.abs()
                mae_sum += abs_diff.sum().item()
                mse_sum += (diff**2).sum().item()
                smape_sum += (
                    (2 * abs_diff / (predictions.abs() + targets.abs() + epsilon))
                    .sum()
                    .item()
                )
                # Stable running mean/variance for targets (float64 to avoid catastrophic cancellation)
                flat_targets = targets.detach().double().reshape(-1)
                batch_count = flat_targets.numel()
                batch_mean = flat_targets.mean().item()
                batch_m2 = ((flat_targets - batch_mean) ** 2).sum().item()

                delta = batch_mean - target_mean
                new_count = total_count + batch_count
                target_mean += delta * batch_count / new_count
                target_m2 += (
                    batch_m2 + (delta**2) * (total_count * batch_count) / new_count
                )
                total_count = new_count

                per_h_mae_sum += abs_diff.sum(dim=0).detach().cpu()
                per_h_mse_sum += (diff**2).sum(dim=0).detach().cpu()

                if self.use_tqdm:
                    bsz = int(batch_data["CaseNode"].shape[0])
                    batch_time_s = time.time() - batch_start_time
                    samples_per_s = (
                        (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                    )
                    eval_iter.set_postfix(
                        n=f"{batch_idx + 1}/{num_batches}",
                        sps=samples_per_s,
                    )

        mean_loss = total_loss / max(1, num_batches)
        mean_mae = mae_sum / max(1, total_count)
        mean_rmse = math.sqrt(mse_sum / max(1, total_count)) if total_count else 0.0
        mean_smape = smape_sum / max(1, total_count)

        ss_res = mse_sum
        # Welford accumulates the total sum of squares around the mean
        ss_tot = target_m2
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        per_h_count = total_count / max(1, horizon)
        per_h_mae = (per_h_mae_sum / max(1, per_h_count)).tolist()
        per_h_rmse = (
            (per_h_mse_sum / max(1, per_h_count)).sqrt().tolist() if per_h_count else []
        )

        metrics = {
            "mae": mean_mae,
            "rmse": mean_rmse,
            "smape": mean_smape,
            "r2": r2,
            "mae_per_h": per_h_mae,
            "rmse_per_h": per_h_rmse,
        }

        return mean_loss, metrics

    def test_epoch(self) -> tuple[float, dict[str, Any]]:
        """Public test evaluation entrypoint."""
        test_loss, test_metrics = self._evaluate_split(
            self.test_loader, split_name="Test"
        )
        self._log_epoch(
            split_name="Test",
            loss=test_loss,
            metrics=test_metrics,
            epoch=self.current_epoch,
        )
        return test_loss, test_metrics

    def _log_epoch(
        self,
        split_name: str,
        loss: float,
        metrics: dict[str, Any],
        epoch: int | None = None,
    ):
        """Log metrics for a specific split."""
        if epoch is None:
            return
        prefix = split_name.capitalize()
        self.writer.add_scalar(f"Loss/{prefix}", loss, epoch)
        self.writer.add_scalar(f"Metrics/{prefix}/MAE", metrics["mae"], epoch)
        self.writer.add_scalar(f"Metrics/{prefix}/RMSE", metrics["rmse"], epoch)
        self.writer.add_scalar(f"Metrics/{prefix}/sMAPE", metrics["smape"], epoch)
        self.writer.add_scalar(f"Metrics/{prefix}/R2", metrics["r2"], epoch)
        for idx, (mae_h, rmse_h) in enumerate(
            zip(metrics["mae_per_h"], metrics["rmse_per_h"], strict=False)
        ):
            self.writer.add_scalar(f"Metrics/{prefix}/MAE_h{idx + 1}", mae_h, epoch)
            self.writer.add_scalar(f"Metrics/{prefix}/RMSE_h{idx + 1}", rmse_h, epoch)

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
        self._status(f"  Checkpoint saved: {checkpoint_path}")

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
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
            "final_learning_rate": self.optimizer.param_groups[0]["lr"],
        }
