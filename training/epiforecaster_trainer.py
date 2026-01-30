"""
Trainer for the EpiForecaster model.

This module implements a trainer class that can handle the EpiForecaster model
through configuration. It provides a unified interface for training the EpiForecaster
model while maintaining the flexibility to support various data configurations.

The trainer works with the EpiForecaster model.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch  # type: ignore[import-not-found]

from data.epi_dataset import EpiDataset, EpiDatasetItem, curriculum_collate_fn
from data.preprocess.config import REGION_COORD
from data.samplers import EpidemicCurriculumSampler
from evaluation.epiforecaster_eval import evaluate_loader
from utils import setup_tensor_core_optimizations
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster

logger = logging.getLogger(__name__)


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
        self.model_id = self._resolve_model_id()
        self.resume = self.config.training.resume

        # Branch on split strategy
        if config.training.split_strategy == "time":
            # Temporal splits: all nodes, different time ranges
            self.train_dataset, self.val_dataset, self.test_dataset = (
                self._split_dataset_temporal()
            )
        else:
            # Node-based splits: different nodes, all time windows
            train_nodes, val_nodes, test_nodes = self._split_dataset_by_nodes()
            train_nodes = list(train_nodes)
            val_nodes = list(val_nodes)
            test_nodes = list(test_nodes)

            # --- Curriculum Training Setup ---
            if self.config.training.curriculum.enabled:
                real_run, synth_runs = self._discover_runs()
                self._status(
                    f"Curriculum enabled. Found runs: Real='{real_run}', Synth={synth_runs}"
                )

                # 1. Real Dataset (Run ID = real)
                real_train_ds = EpiDataset(
                    config=self.config,
                    target_nodes=train_nodes,
                    context_nodes=train_nodes,
                    biomarker_preprocessor=None,
                    mobility_preprocessor=None,
                    run_id=real_run,
                )

                # Reuse preprocessors from Real Train
                fitted_bio_preprocessor = real_train_ds.biomarker_preprocessor
                fitted_mobility_preprocessor = real_train_ds.mobility_preprocessor

                # 2. Synthetic Datasets (One per run_id)
                synth_datasets = []
                for s_run in synth_runs:
                    s_ds = EpiDataset(
                        config=self.config,
                        target_nodes=train_nodes,
                        context_nodes=train_nodes,
                        biomarker_preprocessor=fitted_bio_preprocessor,
                        mobility_preprocessor=fitted_mobility_preprocessor,
                        run_id=s_run,
                    )
                    synth_datasets.append(s_ds)

                # Combine into ConcatDataset
                # Important: Real dataset must be first for the sampler to identify it (index 0)
                # unless sampler inspects run_id (which it does).
                self.train_dataset = ConcatDataset([real_train_ds] + synth_datasets)

                # 3. Val/Test are ALWAYS Real Data
                self.val_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=val_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    run_id=real_run,
                )

                self.test_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=test_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    run_id=real_run,
                )

            else:
                # --- Standard Training Setup ---
                # Build train dataset with None so it fits scaler internally on train regions
                self.train_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=train_nodes,
                    context_nodes=train_nodes,
                    biomarker_preprocessor=None,
                    mobility_preprocessor=None,
                )

                # Reuse train dataset's fitted preprocessors for val/test
                fitted_bio_preprocessor = self.train_dataset.biomarker_preprocessor
                fitted_mobility_preprocessor = self.train_dataset.mobility_preprocessor

                self.val_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=val_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                )

                self.test_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=test_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                )

        # Access cases_dim/biomarkers_dim safely (handle ConcatDataset)
        if isinstance(self.train_dataset, ConcatDataset):
            # Access the first dataset (Real)
            train_example_ds = self.train_dataset.datasets[0]
        else:
            train_example_ds = self.train_dataset

        # Optional static region embeddings from dataset
        self.region_embeddings = None
        embeddings = getattr(train_example_ds, "region_embeddings", None)
        if embeddings is not None:
            self.region_embeddings = embeddings.to(self.device)
        elif self.config.model.type.regions:
            raise ValueError(
                "Region embeddings requested by config but region2vec_path was not provided."
            )

        self.model = EpiForecaster(
            variant_type=self.config.model.type,
            temporal_input_dim=train_example_ds.cases_output_dim,
            biomarkers_dim=train_example_ds.biomarkers_output_dim,
            region_embedding_dim=self.config.model.region_embedding_dim,
            mobility_embedding_dim=self.config.model.mobility_embedding_dim,
            gnn_depth=self.config.model.gnn_depth,
            sequence_length=self.config.model.history_length,
            forecast_horizon=self.config.model.forecast_horizon,
            use_population=self.config.model.use_population,
            population_dim=self.config.model.population_dim,
            device=self.device,
            gnn_module=self.config.model.gnn_module,
            gnn_hidden_dim=self.config.model.gnn_hidden_dim,
            head_d_model=self.config.model.head_d_model,
            head_n_heads=self.config.model.head_n_heads,
            head_num_layers=self.config.model.head_num_layers,
            head_dropout=self.config.model.head_dropout,
        )

        self.model.to(self.device)

        # Enable TF32 for better performance on Ampere+ GPUs
        self._setup_tensor_core_optimizations()

        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = (
            self._create_data_loaders()
        )

        # Setup training components (optimizer, scheduler, criterion)
        self.optimizer = self._create_optimizer()

        # Calculate total steps for scheduler if needed
        total_steps = self.config.training.epochs * len(self.train_loader)
        self.scheduler = self._create_scheduler(total_steps=total_steps)
        self.criterion = self._create_criterion()

        # Setup logging and checkpointing
        self.setup_logging()
        if self.resume:
            self._resume_from_checkpoint()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.nan_loss_counter = 0
        self.nan_loss_triggered = False
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
        self._last_node_mae: dict[int, float] = {}

        self._status("=" * 60)
        self._status("EpiForecasterTrainer initialized:")
        self._status(f"  Model ID: {self.model_id}")
        self._status(f"  Model type: {config.model.type}")
        self._status(f"  Dataset: {config.data.dataset_path}")
        self._status(f"  Device: {self.device}")
        self._status(
            f"  Train samples: {len(self.train_dataset)} ({len(self.train_dataset.target_nodes)} nodes)"
        )
        self._status(
            f"  Val samples:   {len(self.val_dataset)} ({len(self.val_dataset.target_nodes)} nodes)"
        )
        self._status(
            f"  Test samples:  {len(self.test_dataset)} ({len(self.test_dataset.target_nodes)} nodes)"
        )
        self._status(f"  Cases dim: {self.train_dataset.cases_output_dim}")
        self._status(f"  Biomarkers dim: {self.train_dataset.biomarkers_output_dim}")
        self._status(f"  Learning rate: {self.config.training.learning_rate}")
        self._status(f"  Batch size: {config.training.batch_size}")
        # Log run_id configuration
        self._status(f"  Run ID: {self.config.data.run_id}")
        # Check max_batches limit
        if self.config.training.max_batches is not None:
            self._status(
                f"max_batches={self.config.training.max_batches}: Limited to 1 epoch"
            )
        else:
            self._status(f"  Epochs: {config.training.epochs}")
            self._status(f"  {len(self.train_loader)} batches per epoch")
        self._status(
            f"  Optimizer: Adam (weight_decay={self.config.training.weight_decay})"
        )
        self._status(f"  Scheduler: {self.config.training.scheduler_type}")
        self._status(f"  Resume: {'enabled' if self.resume else 'disabled'}")
        self._status("=" * 60)

    def __del__(self) -> None:
        """Cleanup DataLoader workers when trainer is garbage collected.

        This is a safety net for cases where run() hangs or is interrupted.
        """
        try:
            self.cleanup_dataloaders()
        except Exception:
            # Ignore errors during garbage collection
            pass

    def _discover_runs(self) -> tuple[str, list[str]]:
        """Discover available runs in the dataset (real vs synthetic)."""
        # Load metadata only to check runs
        ds = EpiDataset.load_canonical_dataset(
            Path(self.config.data.dataset_path), run_id_chunk_size=1
        )

        has_run_id = "run_id" in ds.coords or "run_id" in ds.dims
        if not has_run_id:
            return "real", []

        if "run_id" in ds.coords:
            run_vals = ds.run_id.values
        else:
            run_vals = ds.coords["run_id"].values

        unique_runs = np.unique(run_vals)
        # Convert to strings and strip
        unique_runs = sorted([str(r).strip() for r in unique_runs])

        if "real" in unique_runs:
            real_run = "real"
            synth_runs = [r for r in unique_runs if r != real_run]
        elif len(unique_runs) > 0:
            logger.warning(
                f"'real' run not found in dataset. Using '{unique_runs[0]}' as real run."
            )
            real_run = unique_runs[0]
            synth_runs = unique_runs[1:]
        else:
            # Empty dataset?
            real_run = "real"
            synth_runs = []

        # Limit synthetic runs to avoid OOM during preloading (temporary for dev/testing)
        # In production, we might need a lazier EpiDataset or fewer runs.
        if len(synth_runs) > 5:
            logger.warning(
                f"Limiting synthetic runs from {len(synth_runs)} to 5 for memory safety."
            )
            synth_runs = synth_runs[:5]

        return real_run, synth_runs

    def _split_dataset_by_nodes(self) -> tuple[list[int], list[int], list[int]]:
        """
        Split dataset into train, val, and test sets using node holdouts.

        This is the default split strategy - we use different regions for train/val/test
        to evaluate ability of model to generalize to new regions.
        """
        train_split = (
            1 - self.config.training.val_split - self.config.training.test_split
        )

        aligned_dataset = EpiDataset.load_canonical_dataset(
            Path(self.config.data.dataset_path)
        )
        N = aligned_dataset[REGION_COORD].size
        all_nodes = np.arange(N)

        # Get valid nodes using EpiDataset class method
        valid_mask = None
        if self.config.data.use_valid_targets:
            valid_mask = EpiDataset.get_valid_nodes(
                dataset_path=Path(self.config.data.dataset_path),
                run_id=self.config.data.run_id,
            )
            self._status(
                f"Using valid_targets filter: {valid_mask.sum()} valid regions"
            )
        else:
            self._status(f"Total regions: {N}")

        # Filter by valid_targets mask
        if valid_mask is not None:
            all_nodes = all_nodes[valid_mask]
            N = len(all_nodes)

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

    def _split_dataset_temporal(
        self,
    ) -> tuple[EpiDataset, EpiDataset, EpiDataset]:
        """
        Split dataset into train, val, and test sets using temporal boundaries.

        All nodes are used as targets in each split, but data is divided by date ranges.
        This returns pre-created datasets instead of node lists.
        """
        # TrainingParams.__post_init__ guarantees these are not None when split_strategy == "time"
        train_end: str = self.config.training.train_end_date or ""
        val_end: str = self.config.training.val_end_date or ""
        test_end: str | None = self.config.training.test_end_date

        return EpiDataset.create_temporal_splits(
            config=self.config,
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end,
        )

    def _setup_device(self) -> torch.device:
        """Setup computation device with MPS support and validation."""
        if self.config.training.device == "auto":
            # Priority: CUDA > MPS > CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self._status(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                self._status("Using MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                self._status("Using CPU device")
        else:
            device = torch.device(self.config.training.device)
            # Validate device availability
            if device.type == "cuda" and not torch.cuda.is_available():
                self._status(
                    f"Warning: CUDA device {device} requested but not available, using CPU"
                )
                device = torch.device("cpu")
            elif device.type == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                self._status(
                    f"Warning: MPS device {device} requested but not available, using CPU"
                )
                device = torch.device("cpu")

        return device

    def _setup_tensor_core_optimizations(self):
        """Enable TF32 and configure precision settings for Tensor Core utilization."""
        setup_tensor_core_optimizations(
            device=self.device,
            enable_tf32=self.config.training.enable_tf32,
            enable_mixed_precision=self.config.training.enable_mixed_precision,
            mixed_precision_dtype=self.config.training.mixed_precision_dtype,
            logger=logger,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _create_scheduler(
        self, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.training.scheduler_type == "cosine":
            # T_max is set to total_steps for a smooth curve across all epochs
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.config.training.scheduler_type == "step":
            # StepLR remains per-epoch based for simplicity
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
        from evaluation.epiforecaster_eval import get_loss_from_config

        return get_loss_from_config(self.config.training.loss)

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create training and validation data loaders with device-aware optimizations."""
        # Set multiprocessing context to avoid CUDA fork deadlocks
        # 'spawn' is required when using CUDA with num_workers > 0
        all_num_workers_zero = (
            self.config.training.num_workers == 0
            and self.config.training.val_workers == 0
        )
        mp_context = (
            "spawn" if self.device.type == "cuda" and not all_num_workers_zero else None
        )

        # Device-aware hardware optimizations
        pin_memory = self.config.training.pin_memory and self.device.type == "cuda"

        avail_cores = (os.cpu_count() or 1) - 1
        cfg_workers = self.config.training.num_workers
        if cfg_workers == -1:
            num_workers = avail_cores
        else:
            num_workers = min(avail_cores, cfg_workers)

        # Compute val_workers similarly (capped to avoid OOM during validation)
        cfg_val_workers = self.config.training.val_workers
        if cfg_val_workers == -1:
            val_num_workers = max(0, avail_cores)
        else:
            val_num_workers = min(max(0, avail_cores), cfg_val_workers)

        persistent_workers = num_workers > 0
        train_loader_kwargs = {
            "dataset": self.train_dataset,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "multiprocessing_context": mp_context,
        }

        # Configure Sampler & Collate based on Curriculum
        if self.config.training.curriculum.enabled and isinstance(
            self.train_dataset, ConcatDataset
        ):
            self._status("Creating EpidemicCurriculumSampler...")
            self.curriculum_sampler = EpidemicCurriculumSampler(
                dataset=self.train_dataset,
                batch_size=self.config.training.batch_size,
                config=self.config.training.curriculum,
                drop_last=False,
            )
            # When using batch_sampler, batch_size and shuffle must be omitted
            train_loader_kwargs["batch_sampler"] = self.curriculum_sampler
            train_loader_kwargs["collate_fn"] = curriculum_collate_fn
        else:
            # Standard training
            self.curriculum_sampler = None
            train_loader_kwargs["batch_size"] = self.config.training.batch_size
            train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["collate_fn"] = self._collate_fn

        if persistent_workers:
            train_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = (
                self.config.training.prefetch_factor
            )
        train_loader = DataLoader(**train_loader_kwargs)

        val_persistent_workers = val_num_workers > 0
        val_loader_kwargs = {
            "dataset": self.val_dataset,
            "batch_size": self.config.training.batch_size,
            "shuffle": False,
            "num_workers": val_num_workers,
            "persistent_workers": val_persistent_workers,
            "pin_memory": pin_memory,
            "collate_fn": self._collate_fn,
            "multiprocessing_context": mp_context,
        }
        if val_persistent_workers:
            val_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = self.config.training.prefetch_factor
        val_loader = DataLoader(**val_loader_kwargs)

        test_loader_kwargs = {
            "dataset": self.test_dataset,
            "batch_size": self.config.training.batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": self._collate_fn,
            "multiprocessing_context": mp_context,
        }
        if persistent_workers:
            test_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and num_workers > 0:
            test_loader_kwargs["prefetch_factor"] = self.config.training.prefetch_factor
        test_loader = DataLoader(**test_loader_kwargs)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _collate_fn(batch: list[EpiDatasetItem]) -> dict[str, Any]:
        "Custom collate for per-node samples with PyG mobility graphs."
        import itertools

        B = len(batch)
        case_node = torch.stack([item["case_node"] for item in batch], dim=0)
        bio_node = torch.stack([item["bio_node"] for item in batch], dim=0)
        case_mean = torch.stack([item["case_mean"] for item in batch], dim=0)
        case_std = torch.stack([item["case_std"] for item in batch], dim=0)
        targets = torch.stack([item["target"] for item in batch], dim=0)
        target_scales = torch.stack([item["target_scale"] for item in batch], dim=0)
        target_mean = torch.stack([item["target_mean"] for item in batch], dim=0)
        target_nodes = torch.tensor(
            [item["target_node"] for item in batch], dtype=torch.long
        )
        window_starts = torch.tensor(
            [item["window_start"] for item in batch], dtype=torch.long
        )
        population = torch.stack([item["population"] for item in batch], dim=0)

        # Flatten mobility graphs efficiently
        graph_list = list(itertools.chain.from_iterable(item["mob"] for item in batch))

        mob_batch = Batch.from_data_list(graph_list)  # type: ignore[arg-type]
        T = len(batch[0]["mob"]) if B > 0 else 0
        # store B and T on the batch for downstream reshaping
        mob_batch.B = torch.tensor([B], dtype=torch.long)  # type: ignore[attr-defined]
        mob_batch.T = torch.tensor([T], dtype=torch.long)  # type: ignore[attr-defined]
        # Precompute a global target node index per ego-graph in the batched `x`.
        # This enables fully-vectorized target gathering in the model without CUDA `.item()` syncs.
        if hasattr(mob_batch, "ptr") and hasattr(mob_batch, "target_node"):
            # Use dict assignment to ensure it's in the store and moves with .to(device)
            mob_batch["target_index"] = mob_batch.ptr[
                :-1
            ] + mob_batch.target_node.reshape(-1)  # type: ignore[index]

        return {
            "CaseNode": case_node,  # (B, L, C)
            "CaseMean": case_mean,  # (B, L, 1)
            "CaseStd": case_std,  # (B, L, 1)
            "BioNode": bio_node,  # (B, L, B)
            "MobBatch": mob_batch,  # Batched PyG graphs
            "Population": population,  # (B,)
            "B": B,
            "T": T,
            "Target": targets,  # (B, H)
            "TargetScale": target_scales,  # (B, C)
            "TargetMean": target_mean,  # (B, 1)
            "TargetNode": target_nodes,  # (B,)
            "WindowStart": window_starts,  # (B,)
            "NodeLabels": [item["node_label"] for item in batch],
        }

    def setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create experiment directory
        experiment_dir = (
            Path(self.config.output.log_dir)
            / self.config.output.experiment_name
            / self.model_id
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self._persist_run_config(experiment_dir)

        # Setup tensorboard writer
        self.writer = SummaryWriter(log_dir=str(experiment_dir))

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

            return (
                Path(self.config.output.log_dir)
                / self.config.output.experiment_name
                / self.model_id
            )

        return Path(configured)

    def _resolve_model_id(self) -> str:
        configured = self.config.training.model_id
        if configured:
            return configured

        sjid = os.getenv("SLURM_JOB_ID", "")
        if sjid:
            # Detect interactive SLURM session - use datetime ID instead
            job_name = os.getenv("SLURM_JOB_NAME", "")
            job_qos = os.getenv("SLURM_JOB_QOS", "")
            if job_name == "interactive" or "_interactive" in job_qos:
                # Interactive session - use unique datetime ID
                return f"run_{time.time_ns()}"
            return sjid

        return f"run_{time.time_ns()}"

    def _find_checkpoint_for_model_id(self) -> Path:
        if not self.config.output.save_checkpoints:
            raise ValueError(
                "Resume requested but checkpointing is disabled in the output config."
            )

        checkpoint_dir = (
            Path(self.config.output.log_dir)
            / self.config.output.experiment_name
            / self.model_id
            / "checkpoints"
        )
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found for model_id '{self.model_id}': "
                f"{checkpoint_dir}"
            )

        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            return best_checkpoint

        final_checkpoint = checkpoint_dir / "final_model.pt"
        if final_checkpoint.exists():
            return final_checkpoint

        epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if epoch_checkpoints:
            return epoch_checkpoints[-1]

        raise FileNotFoundError(
            f"No checkpoints found for model_id '{self.model_id}' in {checkpoint_dir}"
        )

    def _resume_from_checkpoint(self) -> None:
        checkpoint_path = self._find_checkpoint_for_model_id()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.training_history = checkpoint.get(
            "training_history", self.training_history
        )
        self._status(f"Resumed from checkpoint: {checkpoint_path}")

    def run(self) -> dict[str, Any]:
        """Execute training loop."""
        self._status(f"\n{'=' * 60}")
        self._status(f"STARTING TRAINING: {self.config.output.experiment_name}")
        self._status(f"{'=' * 60}")
        writer_log_dir = getattr(self.writer, "log_dir", None)
        if writer_log_dir is not None:
            self._status(f"TensorBoard: {writer_log_dir}")
        if self.config.training.profiler.enabled:
            self._status(f"Profiler: {self._resolve_profiler_log_dir()}")

        # Training loop
        # self._log_model_graph()

        _prev_val_loss = float("inf")
        epochs_todo = (
            1 if self.config.training.max_batches else self.config.training.epochs
        )
        for epoch in range(self.current_epoch, epochs_todo):
            self.current_epoch = epoch

            _train_loss = self._train_epoch()
            if self.nan_loss_triggered:
                self._status("Stopping training due to persistent non-finite loss.")
                break

            # Validation phase
            val_loss, val_metrics, _val_node_mae = self._evaluate_split(
                self.val_loader, split_name="Val"
            )
            self._log_epoch(
                split_name="Val", loss=val_loss, metrics=val_metrics, epoch=epoch
            )

            # Learning rate scheduling (if per-epoch)
            if self.scheduler and self.config.training.scheduler_type == "step":
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

            _prev_val_loss = val_loss

            if should_stop:
                break

        # Final evaluation
        if self.nan_loss_triggered:
            self._status(f"\n{'=' * 60}")
            self._status("TRAINING HALTED")
            self._status("Reason: non-finite training loss exceeded patience.")
            self._status(f"Total epochs trained: {self.current_epoch}")
            self._status(f"{'=' * 60}")
            self.writer.close()
            self.cleanup_dataloaders()
            return self.get_training_results()

        self._status(f"\n{'=' * 60}")
        self._status("TRAINING COMPLETED")
        self._status(f"Best validation loss: {self.best_val_loss:.6f}")
        self._status(f"Total epochs trained: {self.current_epoch}")
        self._status(f"{'=' * 60}")

        # Save final model
        if self.config.output.save_checkpoints:
            self._save_checkpoint(self.current_epoch, self.best_val_loss, is_final=True)

        # Test phase
        test_start_time = time.time()
        test_loss, test_metrics, _test_node_mae = self.test_epoch()
        test_time = time.time() - test_start_time
        self._status(f"{'=' * 60}")
        self._status("TESTING COMPLETED")
        self._status(
            f"Test loss: {test_loss:.6f} | "
            f"MAE: {test_metrics['mae']:.6f} | "
            f"RMSE: {test_metrics['rmse']:.6f} | "
            f"sMAPE: {test_metrics['smape']:.6f} | "
            f"R2: {test_metrics['r2']:.6f} | "
            f"Time: {test_time:.2f}s"
        )
        self._status(f"{'=' * 60}")

        # Close tensorboard writer
        self.writer.close()

        # Cleanup dataloader workers to prevent orphaned processes
        self.cleanup_dataloaders()

        return self.get_training_results()

    def _status(self, message: str, level: int = logging.INFO) -> None:
        logging.log(level, message)

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        # Update Curriculum
        if self.curriculum_sampler is not None:
            self.curriculum_sampler.set_curriculum(self.current_epoch)

        total_loss = 0.0
        _num_batches = len(self.train_loader)
        counted_batches = 0

        train_iter = self.train_loader
        profiler = None
        profiler_active = False
        profiler_complete_announced = False
        # NOTE: Profiler initialization moved below, after first batch fetch
        # to avoid CUDA context deadlock with multiprocessing workers

        fetch_start_time = time.time()
        max_batches = getattr(self.config.training, "max_batches", None)
        first_iteration_done = False

        # Gradient accumulation setup
        accum_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        # Track last gradnorm for progress logging (so we always show the most recent value)
        last_gradnorm = torch.tensor(0.0)

        try:
            for batch_idx, batch_data in enumerate(train_iter):
                # Initialize profiler AFTER first batch is fetched successfully
                # This avoids CUDA context deadlock with multiprocessing workers
                if not first_iteration_done:
                    first_iteration_done = True
                    if self.config.training.profiler.enabled:
                        profiler = self._setup_profiler()
                        profiler.__enter__()
                        profiler_active = True
                        self._status("==== PROFILING ACTIVE ====")
                if max_batches is not None and batch_idx >= max_batches:
                    break

                self._status(f"Batch {batch_idx}", logging.DEBUG)

                data_time_s = time.time() - fetch_start_time
                batch_start_time = time.time()

                if self.config.training.enable_mixed_precision:
                    dtype = (
                        torch.bfloat16
                        if self.config.training.mixed_precision_dtype == "bfloat16"
                        else torch.float16
                    )
                    autocast_enabled = self.device.type == "cuda"
                else:
                    dtype = torch.float32
                    autocast_enabled = False

                with torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=autocast_enabled
                ):
                    predictions, targets, target_mean, target_scale = (
                        self.model.forward_batch(
                            batch_data=batch_data,
                            region_embeddings=self.region_embeddings,
                        )
                    )

                    loss = self.criterion(
                        predictions,
                        targets,
                        target_mean,
                        target_scale,
                    )

                # Scale loss for gradient accumulation
                scaled_loss = loss / accum_steps

                scaled_loss.backward()

                # Only step optimizer every N batches (gradient accumulation)
                should_step = (batch_idx + 1) % accum_steps == 0
                is_last_batch = batch_idx == len(self.train_loader) - 1

                grad_norm = torch.tensor(0.0)
                if should_step or is_last_batch:
                    self._log_gradient_norms(step=self.global_step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_value,
                    )
                    last_gradnorm = grad_norm  # Update for progress logging
                    self.optimizer.step()

                    # Per-step scheduler update (e.g., for CosineAnnealingLR)
                    if (
                        self.scheduler
                        and self.config.training.scheduler_type == "cosine"
                    ):
                        self.scheduler.step()

                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += loss.detach()
                counted_batches += 1

                batch_time_s = time.time() - batch_start_time
                fetch_start_time = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                bsz = int(batch_data["CaseNode"].shape[0])
                samples_per_s = (
                    (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                )
                # Progress logging - only sync to CPU periodically to reduce overhead
                log_frequency = getattr(
                    self.config.training, "progress_log_frequency", 1
                )
                log_this_step = self.global_step % log_frequency == 0
                if log_this_step:
                    loss_value = loss.item()
                    self._status(
                        f"Epoch {self.current_epoch} | Step {self.global_step} | Loss: {loss_value:.6f} | Lr: {lr:.2e} | Grad: {float(last_gradnorm):.3f} | SPS: {samples_per_s:7.1f}",
                    )
                    self.writer.add_scalar(
                        "Loss/Train_step", loss_value, self.global_step
                    )
                self.writer.add_scalar("Learning_Rate/step", lr, self.global_step)
                self.writer.add_scalar(
                    "GradNorm/Clipped_Total", float(grad_norm), self.global_step
                )
                window_start_mean = float(
                    batch_data["WindowStart"].float().mean().item()
                )
                self.writer.add_scalar(
                    "Time/WindowStart", window_start_mean, self.global_step
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
        result = total_loss / effective_batches
        return result.item() if isinstance(result, torch.Tensor) else float(result)

    def _generate_forecast_plots(self, loader: DataLoader, split: str):
        """Generate and save forecast plots using quartile strategy."""
        from evaluation.epiforecaster_eval import (
            generate_forecast_plots,
            select_nodes_by_loss,
        )

        # Select nodes using quartile strategy
        node_groups = select_nodes_by_loss(
            node_mae=self._last_node_mae,
            strategy="quartile",
            samples_per_group=4,
        )

        # Generate plots (generic - works with any node_groups)
        output_dir = (
            Path(self.config.output.log_dir) / self.config.output.experiment_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{split}_forecasts.png"
        generate_forecast_plots(
            model=self.model,
            loader=loader,
            node_groups=node_groups,
            window="last",
            output_path=plot_path,
            log_dir=getattr(self, "writer.log_dir", None)
            if hasattr(self, "writer")
            else None,
        )

    def _evaluate_split(
        self, loader: DataLoader, split_name: str
    ) -> tuple[float, dict[str, Any], dict[int, float]]:
        """Shared evaluation for validation and test splits with extra metrics.

        Returns:
            Tuple of (loss, metrics_dict, node_mae_dict)
        """
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=self.model,
            loader=loader,
            criterion=self.criterion,  # type: ignore[arg-type]
            horizon=self.config.model.forecast_horizon,
            device=self.device,
            region_embeddings=self.region_embeddings,
            split_name=split_name,
            max_batches=getattr(self.config.training, "max_batches", None),
        )

        # Store node MAE for forecast plotting
        self._last_node_mae = node_mae_dict

        return eval_loss, eval_metrics, node_mae_dict

    def test_epoch(self) -> tuple[float, dict[str, Any], dict[int, float]]:
        """Public test evaluation entrypoint."""
        test_loss, test_metrics, test_node_mae = self._evaluate_split(
            self.test_loader, split_name="Test"
        )
        self._log_epoch(
            split_name="Test",
            loss=test_loss,
            metrics=test_metrics,
            epoch=self.current_epoch,
        )
        return test_loss, test_metrics, test_node_mae

    def _log_gradient_norms(self, step: int):
        """Calculates and logs the gradient norms for model components."""
        frequency = self.config.training.grad_norm_log_frequency
        if frequency <= 0 or (step % frequency != 0 and step != 0):
            return

        if not any(p.requires_grad for p in self.model.parameters()):
            return

        # Vectorized calculation on GPU to avoid CPU-GPU sync bottleneck
        gnn_sq_sum = torch.tensor(0.0, device=self.device)
        head_sq_sum = torch.tensor(0.0, device=self.device)
        other_sq_sum = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_sq_sum = param.grad.detach().pow(2).sum()
                if "mobility_gnn" in name:
                    gnn_sq_sum += grad_sq_sum
                elif "forecaster_head" in name:
                    head_sq_sum += grad_sq_sum
                else:
                    other_sq_sum += grad_sq_sum

        # Single synchronization for all group results
        group_sq_sums = torch.stack([gnn_sq_sum, head_sq_sum, other_sq_sum])
        total_sq_sum = group_sq_sums.sum()

        # Move all squared sums to CPU at once
        all_metrics = torch.cat([group_sq_sums, total_sq_sum.unsqueeze(0)])
        all_norms = all_metrics.sqrt().cpu().numpy()

        gnn_norm, head_norm, other_norm, total_norm = all_norms

        # Log to TensorBoard
        self.writer.add_scalar("GradNorm/Total_PreClip", total_norm, step)
        self.writer.add_scalar("GradNorm/MobilityGNN", gnn_norm, step)
        self.writer.add_scalar("GradNorm/ForecasterHead", head_norm, step)
        self.writer.add_scalar("GradNorm/Other", other_norm, step)

        # Log to console/file
        self._status(
            f"Grad norms @ step {step}: Total={total_norm:.4f} | "
            f"GNN={gnn_norm:.4f} | "
            f"Head={head_norm:.4f} | "
            f"Other={other_norm:.4f}",
            logging.DEBUG,
        )

    def _persist_run_config(self, run_dir: Path) -> None:
        """Copy the input configuration to the run directory.
        Note that the config is saved in the model snapshots eg. best_model.pt
        So this is purely a convenience method for easier readability
        """
        config_dict = self.config.to_dict()
        config_path = run_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

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

        self._status(
            f"{prefix} loss: {loss:.6f} | MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | sMAPE: {metrics['smape']:.6f} | R2: {metrics['r2']:.6f}"
        )
        for idx, (mae_h, rmse_h) in enumerate(
            zip(metrics["mae_per_h"], metrics["rmse_per_h"], strict=False)
        ):
            self._status(
                f"{prefix} MAE_h{idx + 1}: {mae_h:.6f} | RMSE_h{idx + 1}: {rmse_h:.6f}"
            )

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
            "config": self.config.to_dict(),
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
        self._status(f"Saved checkpoint to: {checkpoint_path}")

    def cleanup_dataloaders(self) -> None:
        """Explicitly cleanup DataLoader workers to prevent orphaned processes.

        This is critical for HPC/SLURM environments where persistent workers
        can cause hangs and memory leaks if not properly terminated.
        """
        loaders = [
            ("train", self.train_loader) if hasattr(self, "train_loader") else None,
            ("val", self.val_loader) if hasattr(self, "val_loader") else None,
            ("test", self.test_loader) if hasattr(self, "test_loader") else None,
        ]
        for name, loader in loaders:
            if loader is not None:
                # For persistent_workers=True, we need to explicitly shutdown
                # The DataLoader's __del__ method should handle this, but in
                # HPC environments with signal handling issues, explicit cleanup
                # is more reliable
                try:
                    # Store reference to workers before cleanup
                    if hasattr(loader, "_workers"):
                        workers = loader._workers  # type: ignore[attr-defined]
                        if workers:
                            self._status(
                                f"Cleaning up {name} loader ({len(workers)} workers)...",
                                logging.DEBUG,
                            )
                            # Terminate workers explicitly
                            for worker in workers:
                                if worker.is_alive():
                                    worker.terminate()
                    # Clear the loader reference
                    setattr(self, f"{name}_loader", None)
                except Exception as e:
                    self._status(
                        f"Error cleaning up {name} loader: {e}", logging.WARNING
                    )

    def get_training_results(self) -> dict[str, Any]:
        """Return training results summary dictionary."""
        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch,
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            },
        }
