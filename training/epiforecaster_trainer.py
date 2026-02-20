"""
Trainer for the EpiForecaster model.

This module implements a trainer class that can handle the EpiForecaster model
through configuration. It provides a unified interface for training the EpiForecaster
model while maintaining the flexibility to support various data configurations.

The trainer works with the EpiForecaster model.
"""

import importlib
import logging
import os
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
import xarray as xr
import yaml
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import ConcatDataset, DataLoader

from data.epi_dataset import (
    EpiDataset,
    collate_epiforecaster_batch,
)
from data.preprocess.config import REGION_COORD
from data.samplers import EpidemicCurriculumSampler, ShuffledBatchSampler
from evaluation.epiforecaster_eval import JointInferenceLoss, evaluate_loader
from utils import setup_tensor_core_optimizations
from utils.gradient_debug import GradientDebugger
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.training_utils import get_effective_optimizer_step, should_log_step
from utils.platform import (
    cleanup_nvme_staging,
    get_nvme_path,
    is_slurm_cluster,
    select_multiprocessing_context,
    stage_dataset_to_nvme,
)
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

    def __init__(
        self,
        config: EpiForecasterConfig,
        trial: Any | None = None,
        pruning_start_epoch: int = 10,
    ):
        """
        Initialize the unified trainer.

        Args:
            config: Trainer configuration
            trial: Optional Optuna trial for intermediate pruning
            pruning_start_epoch: Epoch to start checking for pruning (default: 10)
        """
        self.config = config
        self._device_hint = self._resolve_device_hint()
        # Keep CPU until DataLoader workers are forked to avoid CUDA init before forking.
        self.device = torch.device("cpu")

        # Set random seeds for reproducibility
        if config.training.seed is not None:
            import random

            random.seed(config.training.seed)
            np.random.seed(config.training.seed)
            torch.manual_seed(config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.training.seed)
                # Enable deterministic behavior for reproducibility
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        self.model_id = self._resolve_model_id()
        self.resume = self.config.training.resume
        self.experiment_dir: Path | None = None
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        # Optuna pruning support
        self.trial = trial
        self.pruning_start_epoch = pruning_start_epoch

        # Stage data to NVMe if running on SLURM cluster
        self._nvme_staging_path: Path | None = None
        if is_slurm_cluster():
            self._stage_data_to_nvme()

        # Branch on split strategy
        if config.training.split_strategy == "time":
            # Temporal splits: all nodes, different time ranges
            self.train_dataset, self.val_dataset, self.test_dataset = (
                self._split_dataset_temporal()
            )
        else:
            train_nodes: list[int]
            val_nodes: list[int]
            test_nodes: list[int]
            real_run_for_split: str | None = None
            split_dataset_path: Path | None = None

            # --- Curriculum Training Setup ---
            if self.config.training.curriculum.enabled:
                real_run, synth_runs = self._discover_runs()
                self.real_run_id = real_run
                self._status(
                    f"Curriculum enabled. Found runs: Real='{real_run}', Synth={synth_runs}"
                )
                real_run_for_split = real_run
                split_dataset_path = (
                    Path(self.config.data.real_dataset_path)
                    if self.config.data.real_dataset_path
                    else Path(self.config.data.dataset_path)
                )

                train_nodes, val_nodes, test_nodes = self._split_dataset_by_nodes(
                    dataset_path=split_dataset_path,
                    run_id=real_run_for_split,
                )
                train_nodes = list(train_nodes)
                val_nodes = list(val_nodes)
                test_nodes = list(test_nodes)
            else:
                # Node-based splits: different nodes, all time windows
                # Use config run_id for splitting
                if not self.config.data.run_id:
                    raise ValueError(
                        "run_id must be specified in config for node-based splits"
                    )
                train_nodes, val_nodes, test_nodes = self._split_dataset_by_nodes(
                    run_id=self.config.data.run_id
                )
                train_nodes = list(train_nodes)
                val_nodes = list(val_nodes)
                test_nodes = list(test_nodes)

            if self.config.training.curriculum.enabled:
                # 1. Real Dataset (Run ID = real)
                # If using separate real dataset, create a config copy for it
                if self.config.data.real_dataset_path:
                    import copy

                    real_config = copy.deepcopy(self.config)
                    real_config.data.dataset_path = self.config.data.real_dataset_path
                else:
                    real_config = self.config

                # Resolve region IDs from real dataset for mapping into synthetic datasets
                assert real_run_for_split is not None, (
                    "real_run_for_split must be set in curriculum mode"
                )
                real_region_ids = self._load_region_ids(
                    dataset_path=split_dataset_path
                    or Path(self.config.data.dataset_path),
                    run_id=real_run_for_split,
                )
                train_region_ids = [real_region_ids[n] for n in train_nodes]
                region_id_index = {
                    region_id: idx for idx, region_id in enumerate(real_region_ids)
                }

                real_train_ds = EpiDataset(
                    config=real_config,
                    target_nodes=train_nodes,
                    context_nodes=train_nodes,
                    biomarker_preprocessor=None,
                    mobility_preprocessor=None,
                    run_id=real_run,
                    region_id_index=region_id_index,
                )

                # Reuse preprocessors from Real Train for real val/test only
                fitted_bio_preprocessor = real_train_ds.biomarker_preprocessor
                fitted_mobility_preprocessor = real_train_ds.mobility_preprocessor
                shared_real_mobility = real_train_ds.preloaded_mobility
                shared_real_mobility_mask = real_train_ds.mobility_mask
                shared_real_sparse_topology = real_train_ds.shared_sparse_topology

                # Fit synthetic preprocessors separately to avoid leakage
                synth_scaler_run = self._select_synthetic_scaler_run(synth_runs)
                synth_train_nodes = self._map_region_ids_to_nodes(
                    train_region_ids,
                    dataset_path=Path(self.config.data.dataset_path),
                    run_id=synth_scaler_run,
                )
                if not synth_train_nodes:
                    synth_train_nodes = self._fallback_all_nodes(
                        dataset_path=Path(self.config.data.dataset_path),
                        run_id=synth_scaler_run,
                    )
                    logger.warning(
                        "Synthetic scaler run '%s' has no overlap with real regions; "
                        "fitting synthetic scalers on all nodes instead.",
                        synth_scaler_run,
                    )

                synth_scaler_ds = EpiDataset(
                    config=self.config,
                    target_nodes=synth_train_nodes,
                    context_nodes=synth_train_nodes,
                    biomarker_preprocessor=None,
                    mobility_preprocessor=None,
                    run_id=synth_scaler_run,
                    region_id_index=region_id_index,
                )

                synth_bio_preprocessor = synth_scaler_ds.biomarker_preprocessor
                synth_mobility_preprocessor = synth_scaler_ds.mobility_preprocessor

                # 2. Synthetic Datasets (One per run_id)
                synth_datasets = []
                for s_run in synth_runs:
                    if s_run == synth_scaler_run:
                        synth_datasets.append(synth_scaler_ds)
                        continue

                    mapped_train_nodes = self._map_region_ids_to_nodes(
                        train_region_ids,
                        dataset_path=Path(self.config.data.dataset_path),
                        run_id=s_run,
                    )
                    if not mapped_train_nodes:
                        mapped_train_nodes = self._fallback_all_nodes(
                            dataset_path=Path(self.config.data.dataset_path),
                            run_id=s_run,
                        )
                        logger.warning(
                            "Synthetic run '%s' has no overlap with real regions; "
                            "using all nodes for training targets.",
                            s_run,
                        )

                    s_ds = EpiDataset(
                        config=self.config,
                        target_nodes=mapped_train_nodes,
                        context_nodes=mapped_train_nodes,
                        biomarker_preprocessor=synth_bio_preprocessor,
                        mobility_preprocessor=synth_mobility_preprocessor,
                        run_id=s_run,
                        region_id_index=region_id_index,
                    )
                    synth_datasets.append(s_ds)

                # Combine into ConcatDataset
                # Important: Real dataset must be first for the sampler to identify it (index 0)
                # unless sampler inspects run_id (which it does).
                self.train_dataset = ConcatDataset([real_train_ds] + synth_datasets)

                # 3. Val/Test are ALWAYS Real Data
                self.val_dataset = EpiDataset(
                    config=real_config,
                    target_nodes=val_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    preloaded_mobility=shared_real_mobility,
                    mobility_mask=shared_real_mobility_mask,
                    shared_sparse_topology=shared_real_sparse_topology,
                    run_id=real_run,
                    region_id_index=region_id_index,
                )

                self.test_dataset = EpiDataset(
                    config=real_config,
                    target_nodes=test_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    preloaded_mobility=shared_real_mobility,
                    mobility_mask=shared_real_mobility_mask,
                    shared_sparse_topology=shared_real_sparse_topology,
                    run_id=real_run,
                    region_id_index=region_id_index,
                )

                # Drop full shared sparse topology references after split caches are built.
                real_train_ds.release_shared_sparse_topology()
                self.val_dataset.release_shared_sparse_topology()
                self.test_dataset.release_shared_sparse_topology()
                for ds in synth_datasets:
                    ds.release_shared_sparse_topology()

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
                shared_mobility = self.train_dataset.preloaded_mobility
                shared_mobility_mask = self.train_dataset.mobility_mask
                shared_sparse_topology = self.train_dataset.shared_sparse_topology

                self.val_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=val_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    preloaded_mobility=shared_mobility,
                    mobility_mask=shared_mobility_mask,
                    shared_sparse_topology=shared_sparse_topology,
                )

                self.test_dataset = EpiDataset(
                    config=self.config,
                    target_nodes=test_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    preloaded_mobility=shared_mobility,
                    mobility_mask=shared_mobility_mask,
                    shared_sparse_topology=shared_sparse_topology,
                )

                # Drop full shared sparse topology references after split caches are built.
                self.train_dataset.release_shared_sparse_topology()
                self.val_dataset.release_shared_sparse_topology()
                self.test_dataset.release_shared_sparse_topology()

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
            self.region_embeddings = embeddings.clone()
        elif self.config.model.type.regions:
            raise ValueError(
                "Region embeddings requested by config but region2vec_path was not provided."
            )

        if self.config.model.temporal_covariates_dim > 0:
            temporal_covariates_dim = self.config.model.temporal_covariates_dim
        else:
            temporal_covariates_dim = train_example_ds.temporal_covariates_dim

        self.model = EpiForecaster(
            variant_type=self.config.model.type,
            sir_physics=self.config.model.sir_physics,
            observation_heads=self.config.model.observation_heads,
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
            head_positional_encoding=self.config.model.head_positional_encoding,
            temporal_covariates_dim=temporal_covariates_dim,
        )

        # Setup data loaders before CUDA initialization when using fork
        self.train_loader, self.val_loader, self.test_loader = (
            self._create_data_loaders()
        )
        if self._should_prestart_dataloader_workers():
            self._prestart_dataloader_workers(
                self.train_loader, self.val_loader, self.test_loader
            )

        # Resolve actual device after worker prestart to avoid CUDA fork issues
        self.device = self._setup_device()
        self.model.device = self.device

        # Setup precision policy (FP32 params + optional BF16 autocast)
        from utils.precision_policy import resolve_precision_policy

        self.precision_policy = resolve_precision_policy(
            self.config.training, self.device
        )

        # Log precision configuration
        if self.precision_policy.autocast_enabled:
            self._status(
                f"Using FP32 parameters with BF16 autocast on {self.device.type.upper()}",
                logging.INFO,
            )
        else:
            self._status(
                f"Using FP32 parameters on {self.device.type.upper()}",
                logging.INFO,
            )

        if self.region_embeddings is not None:
            self.region_embeddings = self.region_embeddings.to(self.device)

        self.model.to(self.device)

        # Ensure model is FP32 (precision policy enforces this)
        if self.model.dtype != torch.float32:
            self._status(
                f"Converting model from {self.model.dtype} to float32",
                logging.INFO,
            )
            self.model = self.model.to(torch.float32)

        # Enable TF32 for better performance on Ampere+ GPUs
        self._setup_tensor_core_optimizations()

        # Setup training components (optimizer, scheduler, criterion)
        self.optimizer = self._create_optimizer()

        from training.schedulers import compute_scheduler_steps

        total_steps, warmup_steps = compute_scheduler_steps(
            epochs=self.config.training.epochs,
            batches_per_epoch=len(self.train_loader),
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_batches=self.config.training.warmup_steps,
        )
        self.scheduler = self._create_scheduler(
            total_steps=total_steps, warmup_steps=warmup_steps
        )
        self.criterion = self._create_criterion()
        if not isinstance(self.criterion, JointInferenceLoss):
            raise ValueError(
                "EpiForecasterTrainer now requires JointInferenceLoss. "
                "Set training.loss.name=joint_inference in the config."
            )

        # Setup logging and checkpointing
        self.setup_logging()

        # Training state - initialize best_val_loss BEFORE resume to avoid AttributeError
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
        # Curriculum phase tracking for LR warmup at transitions
        self._last_curriculum_phase_idx: int | None = None
        self._lr_warmup_remaining: int = 0
        self._lr_warmup_target_lr: float = 0.0  # Target LR to restore after warmup

        # Initialize gradient debugger (zero overhead when disabled)
        grad_debug_dir = self.config.training.gradient_debug_log_dir
        if grad_debug_dir is None and self.config.training.enable_gradient_debug:
            # Auto-set to experiment directory if enabled but not specified
            grad_debug_dir = (
                self.experiment_dir / "gradient_debug" if self.experiment_dir else None
            )
        self.gradient_debugger = GradientDebugger(
            enabled=self.config.training.enable_gradient_debug,
            log_dir=grad_debug_dir,
            logger_instance=logger,
        )
        if self.gradient_debugger.enabled and self.gradient_debugger.log_dir:
            self._status(
                f"Gradient debugging enabled. Reports will be saved to: {self.gradient_debugger.log_dir}",
                logging.INFO,
            )

        # Resume from checkpoint after all state is initialized
        if self.resume:
            self._resume_from_checkpoint()

        self._status("=" * 60)
        self._status("EpiForecasterTrainer initialized:")
        self._status(f"  Model ID: {self.model_id}")
        self._status(f"  Model type: {config.model.type}")
        self._status(f"  Dataset: {config.data.dataset_path}")
        self._status(f"  Device: {self.device}")
        self._status(
            f"  Train samples: {len(self.train_dataset)} ({len(train_example_ds.target_nodes)} nodes)"
        )
        self._status(
            f"  Val samples:   {len(self.val_dataset)} ({len(self.val_dataset.target_nodes)} nodes)"
        )
        self._status(
            f"  Test samples:  {len(self.test_dataset)} ({len(self.test_dataset.target_nodes)} nodes)"
        )
        self._status(f"  Cases dim: {train_example_ds.cases_output_dim}")
        self._status(f"  Biomarkers dim: {train_example_ds.biomarkers_output_dim}")

        # Log Observation Heads configuration
        if self.config.model.observation_heads:
            self._status("  Observation Heads:")
            heads = self.config.model.observation_heads
            self._status(
                f"    - Wastewater: kernel={heads.kernel_length_ww}, learnable={heads.learnable_kernel_ww}"
            )
            self._status(
                f"    - Hospital:   kernel={heads.kernel_length_hosp}, learnable={heads.learnable_kernel_hosp}"
            )
            self._status(
                f"    - Cases:      kernel={heads.kernel_length_cases}, learnable={heads.learnable_kernel_cases}"
            )
            self._status(
                f"    - Deaths:     kernel={heads.kernel_length_deaths}, learnable={heads.learnable_kernel_deaths}"
            )
            self._status(
                f"    - Residual:   mode={heads.residual_mode}, scale={heads.residual_scale}"
            )

        # Log Physics configuration
        if self.config.model.sir_physics:
            self._status("  SIR Physics:")
            physics = self.config.model.sir_physics
            self._status(f"    - dt: {physics.dt}")
            self._status(f"    - Mass cons: {physics.enforce_mass_conservation}")
            self._status(f"    - Non-neg:   {physics.enforce_nonnegativity}")

        # Log Loss configuration
        if self.config.training.loss.name == "joint_inference":
            self._status("  Joint Inference Loss Weights:")
            weights = self.config.training.loss.joint
            self._status(f"    - WW:    {weights.w_ww}")
            self._status(f"    - Hosp:  {weights.w_hosp}")
            self._status(f"    - Cases: {weights.w_cases}")
            self._status(f"    - Deaths: {weights.w_deaths}")
            self._status(f"    - SIR:   {weights.w_sir}")

        self._status(f"  Learning rate: {self.config.training.learning_rate}")
        self._status(f"  Batch size: {config.training.batch_size}")
        if self.curriculum_sampler is not None:
            self._status("  Train shuffle: enabled (curriculum batch sampler)")
        else:
            shuffle_status = (
                "enabled (random batch order, contiguous samples within batch)"
                if self.config.training.shuffle_train_batches
                else "disabled (sequential batch order)"
            )
            self._status(f"  Train shuffle: {shuffle_status}")
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
            accum = self.config.training.gradient_accumulation_steps
            total_sched_steps = (
                config.training.epochs * len(self.train_loader)
            ) // accum
            self._status(f"  {total_sched_steps} scheduler steps (accum={accum})")
        self._status(
            "  Optimizer: "
            f"{self.config.training.optimizer} "
            f"(weight_decay={self.config.training.weight_decay})"
        )
        self._status(f"  Scheduler: {self.config.training.scheduler_type}")
        if self.config.training.warmup_steps > 0:
            accum = self.config.training.gradient_accumulation_steps
            effective_warmup = self.config.training.warmup_steps // accum
            self._status(
                f"  Warmup steps: {self.config.training.warmup_steps} batches "
                f"({effective_warmup} scheduler steps, accum={accum})"
            )
        self._status(f"  Gradient clip: {self.config.training.gradient_clip_value}")
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

        # Cleanup NVMe staging if used
        try:
            if self._nvme_staging_path is not None:
                cleanup_nvme_staging(self._nvme_staging_path)
        except Exception:
            pass

    def _stage_data_to_nvme(self) -> None:
        """Stage dataset(s) to node-local NVMe storage for improved I/O.

        Updates config paths to point to staged locations on NVMe.
        Only runs when on a SLURM cluster with NVMe available.
        """
        enable_staging = os.getenv("EPFORECASTER_STAGE_TO_NVME", "1") != "0"
        if not enable_staging:
            logger.info("NVMe staging disabled via EPFORECASTER_STAGE_TO_NVME=0")
            return

        logger.info("Detected SLURM cluster - staging data to NVMe")
        self._nvme_staging_path = get_nvme_path()

        # Stage main dataset
        main_path = Path(self.config.data.dataset_path)
        if main_path.exists():
            staged_main = stage_dataset_to_nvme(
                main_path, self._nvme_staging_path, enable_staging=True
            )
            if staged_main != main_path:
                self.config.data.dataset_path = str(staged_main)
                logger.info(f"Using staged dataset: {staged_main}")

        # Stage real dataset if different (curriculum mode)
        if self.config.data.real_dataset_path:
            real_path = Path(self.config.data.real_dataset_path)
            if real_path.exists() and real_path != main_path:
                staged_real = stage_dataset_to_nvme(
                    real_path, self._nvme_staging_path, enable_staging=True
                )
                if staged_real != real_path:
                    self.config.data.real_dataset_path = str(staged_real)
                    logger.info(f"Using staged real dataset: {staged_real}")

    def _load_sparsity_mapping(self) -> dict[str, float]:
        """Load run_id -> sparsity mapping from processed zarr dataset.

        Reads the synthetic_sparsity_level variable from the processed dataset.
        Returns dict mapping run_id string to sparsity float (0.0-1.0).

        Returns:
            Dictionary mapping run_id strings to sparsity values.
            Returns empty dict if dataset is unavailable or variable is missing.
        """
        dataset_path = Path(self.config.data.dataset_path)
        if not dataset_path.exists():
            logger.warning(
                f"Processed dataset not found at {dataset_path}. "
                "Sparsity-based run selection will be disabled."
            )
            return {}

        try:
            ds = xr.open_zarr(str(dataset_path), chunks=None)
            if "synthetic_sparsity_level" not in ds:
                logger.warning(
                    f"Variable 'synthetic_sparsity_level' not found in {dataset_path}. "
                    "Sparsity-based run selection will be disabled."
                )
                ds.close()
                return {}

            run_ids = ds["run_id"].values
            sparsity = ds["synthetic_sparsity_level"].values

            mapping = {
                str(run_id).strip(): float(spars)
                for run_id, spars in zip(run_ids, sparsity)
            }
            ds.close()
            logger.info(
                f"Loaded sparsity mapping for {len(mapping)} runs from {dataset_path}"
            )
            return mapping
        except Exception as e:
            logger.warning(
                f"Failed to load sparsity mapping from {dataset_path}: {e}. "
                "Sparsity-based run selection will be disabled."
            )
            return {}

    def _select_runs_for_curriculum(
        self,
        synth_runs: list[str],
        sparsity_map: dict[str, float],
        max_runs: int = 5,
    ) -> list[str]:
        """Select synthetic runs to maximize sparsity diversity for curriculum training.

        First selects one run from each sparsity bucket (0.05, 0.20, 0.40, 0.60, 0.80)
        to ensure coverage across all curriculum phases. Then fills remaining slots
        randomly from available runs. Falls back to random selection if sparsity data
        unavailable.

        Args:
            synth_runs: List of available synthetic run IDs
            sparsity_map: Mapping from run_id to sparsity value
            max_runs: Maximum number of runs to return (memory limit)

        Returns:
            Selected list of run IDs
        """
        if not sparsity_map:
            logger.info("No sparsity map available, using random selection")
            import random

            return synth_runs[:max_runs]

        target_sparsities = [0.05, 0.20, 0.40, 0.60, 0.80]
        selected = []

        # Phase 1: Select one run from each sparsity bucket
        for target in target_sparsities:
            if len(selected) >= max_runs:
                break

            candidates = [
                (run_id, sparsity_map[run_id])
                for run_id in synth_runs
                if run_id not in selected
                and abs(sparsity_map.get(run_id, -1.0) - target) < 0.01
            ]

            if candidates:
                # Pick the run closest to target sparsity
                best_match = min(candidates, key=lambda x: abs(x[1] - target))
                selected.append(best_match[0])
                logger.info(
                    f"Selected run '{best_match[0]}' with sparsity {best_match[1]:.2f} "
                    f"for target sparsity {target:.2f}"
                )

        # Phase 2: Fill remaining slots randomly from available runs
        if len(selected) < min(max_runs, len(synth_runs)):
            remaining_needed = min(max_runs, len(synth_runs)) - len(selected)
            available = [r for r in synth_runs if r not in selected]

            if available:
                logger.info(
                    f"Adding {remaining_needed} random runs to fill quota (have {len(selected)})"
                )
                import random

                remaining = random.sample(
                    available, min(remaining_needed, len(available))
                )
                selected.extend(remaining)

        return selected

    def _discover_runs(self) -> tuple[str, list[str]]:
        """Discover available runs in the dataset (real vs synthetic)."""
        real_run = "real"  # User mandate: always "real"
        synth_runs = []

        ds_path = Path(self.config.data.dataset_path)
        if not ds_path.exists():
            logger.warning(f"Dataset path not found: {ds_path}")
            return real_run, []

        try:
            all_runs = EpiDataset.discover_available_runs(ds_path)
            synth_runs = [r for r in all_runs if r != real_run]
        except Exception as e:
            logger.warning(f"Failed to discover runs from {ds_path}: {e}")
            return real_run, []

        max_runs = 5

        if self.config.training.curriculum.enabled and len(synth_runs) > max_runs:
            sparsity_map = self._load_sparsity_mapping()

            if sparsity_map:
                logger.info(
                    f"Curriculum mode: selecting {max_runs} runs for sparsity diversity "
                    f"from {len(synth_runs)} available runs"
                )
                synth_runs = self._select_runs_for_curriculum(
                    synth_runs, sparsity_map, max_runs
                )
            else:
                logger.warning(
                    f"Limiting synthetic runs from {len(synth_runs)} to {max_runs} for memory safety."
                )
                synth_runs = synth_runs[:max_runs]
        elif len(synth_runs) > max_runs:
            logger.warning(
                f"Limiting synthetic runs from {len(synth_runs)} to {max_runs} for memory safety."
            )
            synth_runs = synth_runs[:max_runs]

        return real_run, synth_runs

    def _split_dataset_by_nodes(
        self,
        dataset_path: Path | None = None,
        run_id: str | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Split dataset into train, val, and test sets using node holdouts.

        This is the default split strategy - we use different regions for train/val/test
        to evaluate ability of model to generalize to new regions.
        """
        train_split = (
            1 - self.config.training.val_split - self.config.training.test_split
        )

        target_path = dataset_path or Path(self.config.data.dataset_path)
        # Determine which run_id to use for loading
        effective_run_id = run_id or self.config.data.run_id
        if not effective_run_id:
            raise ValueError(
                "run_id must be provided either as argument or in config.data.run_id"
            )
        aligned_dataset = EpiDataset.load_canonical_dataset(
            target_path,
            run_id=effective_run_id,
            run_id_chunk_size=self.config.data.run_id_chunk_size,
        )
        N = aligned_dataset[REGION_COORD].size
        all_nodes = np.arange(N)

        # Get valid nodes using EpiDataset class method
        valid_mask = None
        if self.config.data.use_valid_targets:
            run_id_for_valid = run_id or self.config.data.run_id
            valid_mask = EpiDataset.get_valid_nodes(
                dataset_path=target_path,
                run_id=run_id_for_valid,
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

        rng = np.random.default_rng(self.config.training.seed)
        rng.shuffle(all_nodes)
        n_train = int(len(all_nodes) * train_split)
        n_val = int(len(all_nodes) * self.config.training.val_split)
        train_nodes = all_nodes[:n_train]
        val_nodes = all_nodes[n_train : n_train + n_val]
        test_nodes = all_nodes[n_train + n_val :]

        assert len(train_nodes) + len(val_nodes) + len(test_nodes) == len(all_nodes), (
            "Dataset split is not correct"
        )

        aligned_dataset.close()
        return list(train_nodes), list(val_nodes), list(test_nodes)

    def _load_region_ids(
        self,
        dataset_path: Path,
        run_id: str,
    ) -> list[str]:
        aligned_dataset = EpiDataset.load_canonical_dataset(
            dataset_path,
            run_id=run_id,
            run_id_chunk_size=1,
        )
        region_ids = [str(r) for r in aligned_dataset[REGION_COORD].values]
        aligned_dataset.close()
        return region_ids

    def _map_region_ids_to_nodes(
        self,
        region_ids: list[str],
        dataset_path: Path,
        run_id: str,
    ) -> list[int]:
        target_region_ids = self._load_region_ids(dataset_path, run_id)
        region_id_index = {rid: i for i, rid in enumerate(target_region_ids)}
        missing = [rid for rid in region_ids if rid not in region_id_index]
        if missing:
            logger.warning(
                "Region ID mapping missing %d/%d regions for run '%s' from %s.",
                len(missing),
                len(region_ids),
                run_id,
                dataset_path,
            )
        return [region_id_index[rid] for rid in region_ids if rid in region_id_index]

    def _fallback_all_nodes(self, dataset_path: Path, run_id: str) -> list[int]:
        aligned_dataset = EpiDataset.load_canonical_dataset(
            dataset_path,
            run_id_chunk_size=1,
            run_id=run_id,
        )
        num_nodes = aligned_dataset[REGION_COORD].size
        aligned_dataset.close()
        return list(range(num_nodes))

    def _select_synthetic_scaler_run(self, synth_runs: list[str]) -> str:
        if not synth_runs:
            raise ValueError("No synthetic runs available for scaler fitting.")

        mapping = self._load_sparsity_mapping()
        if mapping:

            def resolve_sparsity(run_id: str) -> float | None:
                if run_id in mapping:
                    return mapping[run_id]
                if "_" in run_id:
                    suffix = run_id.split("_", 1)[1]
                    for k, v in mapping.items():
                        if "_" in k and k.split("_", 1)[1] == suffix:
                            return v
                return None

            candidates = [(run_id, resolve_sparsity(run_id)) for run_id in synth_runs]
            candidates = [
                (run_id, sparsity)
                for run_id, sparsity in candidates
                if sparsity is not None
            ]
            if candidates:
                # FIX: Select LOWEST sparsity (cleanest data) for scaler fitting
                # Previously used max() which selected noisiest data, causing spikes
                # when curriculum progressed to cleaner sparsity levels
                selected_run, selected_sparsity = min(candidates, key=lambda x: x[1])  # type: ignore[arg-type]
                logger.info(
                    "Synthetic scalers fitted on run '%s' (sparsity=%.3f).",
                    selected_run,
                    selected_sparsity,
                )
                return selected_run

        fallback_run = synth_runs[-1]
        logger.info(
            "Synthetic scalers fitted on run '%s' (no sparsity metadata available).",
            fallback_run,
        )
        return fallback_run

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
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            normalized_name = name.lower()
            if (
                name.endswith("bias")
                or "norm" in normalized_name
                or "alpha_" in normalized_name
                or param.ndim < 2
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer_name = self.config.training.optimizer.lower()
        optimizer_cls: type[torch.optim.Optimizer]
        if optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(
                f"Unknown optimizer type: {self.config.training.optimizer}"
            )

        return optimizer_cls(
            param_groups,
            lr=self.config.training.learning_rate,
            eps=self.precision_policy.optimizer_eps,
        )

    def _create_scheduler(
        self, total_steps: int, warmup_steps: int = 0
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.training.scheduler_type == "cosine":
            if warmup_steps > 0:
                from training.schedulers import WarmupCosineScheduler

                return WarmupCosineScheduler(
                    self.optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                )
            else:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps
                )
        elif self.config.training.scheduler_type == "step":
            if warmup_steps > 0:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "warmup_steps is set but StepLR scheduler does not support warmup. "
                    "Warmup will be ignored. Use scheduler_type='cosine' for warmup support."
                )
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

        return get_loss_from_config(
            self.config.training.loss,
            data_config=self.config.data,
            forecast_horizon=self.config.model.forecast_horizon,
        )

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create training and validation data loaders with device-aware optimizations."""
        # Select multiprocessing context for DataLoader workers
        all_num_workers_zero = (
            self.config.training.num_workers == 0
            and self.config.training.val_workers == 0
        )
        mp_context = select_multiprocessing_context(
            self._device_hint, all_num_workers_zero=all_num_workers_zero
        )
        self._multiprocessing_context = mp_context

        # Device-aware hardware optimizations
        pin_memory = self.config.training.pin_memory and self._device_hint == "cuda"

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

        persistent_workers = self.config.training.persistent_workers and num_workers > 0
        train_loader_kwargs = {
            "dataset": self.train_dataset,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        # Only pass multiprocessing_context when using workers
        if num_workers > 0:
            train_loader_kwargs["multiprocessing_context"] = mp_context

        # Configure Sampler & Collate based on Curriculum
        shared_collate = partial(
            collate_epiforecaster_batch,
            require_region_index=bool(self.config.model.type.regions),
        )
        if self.config.training.curriculum.enabled and isinstance(
            self.train_dataset, ConcatDataset
        ):
            self._status("Creating EpidemicCurriculumSampler...")
            self.curriculum_sampler = EpidemicCurriculumSampler(
                dataset=self.train_dataset,
                batch_size=self.config.training.batch_size,
                config=self.config.training.curriculum,
                drop_last=False,
                real_run_id=getattr(self, "real_run_id", "real"),
            )
            # When using batch_sampler, batch_size and shuffle must be omitted
            train_loader_kwargs["batch_sampler"] = self.curriculum_sampler
            train_loader_kwargs["collate_fn"] = shared_collate
        else:
            # Standard training
            self.curriculum_sampler = None
            if self.config.training.shuffle_train_batches:
                train_loader_kwargs["batch_sampler"] = ShuffledBatchSampler(
                    dataset_size=len(self.train_dataset),
                    batch_size=self.config.training.batch_size,
                    drop_last=False,
                    seed=self.config.training.seed,
                )
            else:
                train_loader_kwargs["batch_size"] = self.config.training.batch_size
                train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["collate_fn"] = shared_collate

        if persistent_workers:
            train_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = (
                self.config.training.prefetch_factor
            )
        train_loader = DataLoader(**train_loader_kwargs)

        val_persistent_workers = (
            self.config.training.persistent_workers and val_num_workers > 0
        )
        val_loader_kwargs = {
            "dataset": self.val_dataset,
            "batch_size": self.config.training.batch_size,
            "shuffle": False,
            "num_workers": val_num_workers,
            "pin_memory": pin_memory,
            "collate_fn": shared_collate,
        }
        # Only pass multiprocessing_context when using workers
        if val_num_workers > 0:
            val_loader_kwargs["multiprocessing_context"] = mp_context
        if val_persistent_workers:
            val_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = self.config.training.prefetch_factor
        val_loader = DataLoader(**val_loader_kwargs)

        # Compute test_workers (default to 0 since test runs once at end)
        cfg_test_workers = getattr(self.config.training, "test_workers", 0)
        if cfg_test_workers == -1:
            test_num_workers = max(0, avail_cores)
        else:
            test_num_workers = min(max(0, avail_cores), cfg_test_workers)

        test_persistent_workers = (
            self.config.training.persistent_workers and test_num_workers > 0
        )
        test_loader_kwargs = {
            "dataset": self.test_dataset,
            "batch_size": self.config.training.batch_size,
            "shuffle": False,
            "num_workers": test_num_workers,
            "pin_memory": pin_memory,
            "collate_fn": shared_collate,
        }
        # Only pass multiprocessing_context when using workers
        if test_num_workers > 0:
            test_loader_kwargs["multiprocessing_context"] = mp_context
        if test_persistent_workers:
            test_loader_kwargs["persistent_workers"] = True
        if self.config.training.prefetch_factor is not None and test_num_workers > 0:
            test_loader_kwargs["prefetch_factor"] = self.config.training.prefetch_factor
        test_loader = DataLoader(**test_loader_kwargs)
        return train_loader, val_loader, test_loader

    def _should_prestart_dataloader_workers(self) -> bool:
        if self._multiprocessing_context != "fork":
            return False
        if self._device_hint != "cuda":
            return False
        return True

    def _resolve_device_hint(self) -> str:
        requested = str(self.config.training.device)
        if requested == "auto":
            return "cuda" if platform.system() == "Linux" else "cpu"
        try:
            return torch.device(requested).type
        except (TypeError, ValueError):
            return requested

    def _prestart_dataloader_workers(self, *loaders: DataLoader) -> None:
        """Start DataLoader workers before CUDA initialization.

        This allows forked workers to inherit preloaded mobility tensors without
        copying once CUDA is initialized in the parent process.
        """
        for loader in loaders:
            if loader is None or loader.num_workers == 0:
                continue
            _ = iter(loader)

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

        self.experiment_dir = experiment_dir

        if wandb.run is None:
            group = self.config.output.wandb_group or self.config.output.experiment_name
            self.wandb_run = wandb.init(
                project=self.config.output.wandb_project,
                entity=self.config.output.wandb_entity,
                group=group,
                name=self.model_id,
                dir=str(experiment_dir),
                config=self.config.to_dict(),
                tags=self.config.output.wandb_tags or None,
                job_type="train",
                mode=self.config.output.wandb_mode,
            )
        else:
            self.wandb_run = wandb.run

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

        if self.wandb_run is not None:
            wandb.config.update(hyperparams, allow_val_change=True)

    # def _log_model_graph(self):
    #     """
    #     Write the model graph using a real minibatch.

    #     This runs once before training to make the module shapes discoverable in the
    #     Graph viewer. Failures are non-fatal to avoid blocking training on
    #     tracing issues with complex inputs (e.g., PyG batches).
    #     """
    #     if self._model_graph_logged:
    #         return

    #     try:
    #         example_batch = next(iter(self.train_loader))
    #     except StopIteration:
    #         print(
    #             "Skipping graph logging: training dataset is empty."
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
    #         print(f"Skipping graph logging: {exc}")
    #         self._model_graph_logged = True
    #     finally:
    #         if was_training:
    #             self.model.train()

    def _setup_profiler(self):
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profile_log_dir = self._resolve_profiler_log_dir()
        profile_log_dir.mkdir(parents=True, exist_ok=True)

        # Traces saved locally, use perf-analyze to inspect
        tb_handler = tensorboard_trace_handler(str(profile_log_dir))

        return profile(
            activities=activities,
            schedule=schedule(
                wait=self.config.training.profiler.wait_steps,
                warmup=self.config.training.profiler.warmup_steps,
                active=self.config.training.profiler.active_steps,
                repeat=self.config.training.profiler.repeat,
            ),
            on_trace_ready=tb_handler,
            record_shapes=True,
            profile_memory=self.config.training.profiler.record_memory,
            with_stack=self.config.training.profiler.with_stack,
        )

    def _resolve_profiler_log_dir(self) -> Path:
        configured = getattr(self.config.training.profiler, "log_dir", "auto")
        if configured == "auto":
            if self.experiment_dir is not None:
                return self.experiment_dir

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
        from utils.precision_policy import validate_old_checkpoint_compatible

        checkpoint_path = self._find_checkpoint_for_model_id()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Validate checkpoint precision compatibility (rejects FP16 checkpoints)
        validate_old_checkpoint_compatible(checkpoint, self.precision_policy)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Assert resumed model is FP32
        actual_dtype = next(iter(self.model.parameters())).dtype
        if actual_dtype != torch.float32:
            raise ValueError(
                f"Checkpoint dtype mismatch: model has {actual_dtype}, "
                f"but only float32 parameters are supported. "
                "Please retrain with current precision settings."
            )

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
        if self.experiment_dir is not None:
            self._status(f"Run directory: {self.experiment_dir}")
        if self.wandb_run is not None:
            self._status(f"W&B run: {self.wandb_run.project}/{self.wandb_run.name}")
        if self.config.training.profiler.enabled:
            self._status(f"Profiler: {self._resolve_profiler_log_dir()}")
            profile_epochs = self.config.training.profiler.profile_epochs
            if profile_epochs:
                self._status(f"Profiler epochs: {profile_epochs}")

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

            # Early stopping (disabled if patience is None)
            should_stop = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.config.output.save_best_only:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if (
                    self.config.training.early_stopping_patience is not None
                    and self.patience_counter
                    >= self.config.training.early_stopping_patience
                ):
                    self._status(
                        "Early stopping triggered after "
                        f"{self.patience_counter} epochs without improvement"
                    )
                    should_stop = True

            _prev_val_loss = val_loss

            # Optuna pruning: report intermediate value and check if trial should be pruned
            if self.trial is not None and epoch >= self.pruning_start_epoch:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    self._status(
                        f"Trial pruned by Optuna at epoch {epoch} "
                        f"(val_loss={val_loss:.6f})"
                    )
                    # Lazy import to avoid hard dependency on optuna
                    optuna_module = importlib.import_module("optuna")
                    raise optuna_module.TrialPruned()

            if should_stop:
                break

        # Final evaluation
        if self.nan_loss_triggered:
            self._status(f"\n{'=' * 60}")
            self._status("TRAINING HALTED")
            self._status("Reason: non-finite training loss exceeded patience.")
            self._status(f"Total epochs trained: {self.current_epoch}")
            self._status(f"{'=' * 60}")
            if self.wandb_run is not None:
                self.wandb_run.finish()
            self.cleanup_dataloaders()
            return self.get_training_results()

        self._status(f"\n{'=' * 60}")
        self._status("TRAINING COMPLETED")
        self._status(f"Best validation loss: {self.best_val_loss:.4g}")
        self._status(f"Total epochs trained: {self.current_epoch}")
        self._status(f"{'=' * 60}")

        # Save final model
        if self.config.output.save_checkpoints:
            self._save_checkpoint(self.current_epoch, self.best_val_loss, is_final=True)

        # Test phase
        test_start_time = time.time()
        test_loss, test_metrics, _test_node_mae = self.test_epoch()
        # Shutdown test iterator after evaluation
        self._shutdown_loader_iterator(self.test_loader)
        test_time = time.time() - test_start_time
        self._status(f"{'=' * 60}")
        self._status("TESTING COMPLETED")
        self._status(
            f"Test loss: {test_loss:.4g} | "
            f"MAE: {test_metrics['mae']:.4g} | "
            f"RMSE: {test_metrics['rmse']:.4g} | "
            f"sMAPE: {test_metrics['smape']:.4g} | "
            f"R2: {test_metrics['r2']:.4g} | "
            f"Time: {test_time:.2f}s"
        )
        if self.config.training.plot_forecasts:
            try:
                self._generate_forecast_plots(self.test_loader, split="test")
            except Exception as exc:
                self._status(
                    f"[plot] Test forecast plotting failed: {exc}",
                    level=logging.WARNING,
                )
        self._status(f"{'=' * 60}")

        if self.wandb_run is not None:
            self.wandb_run.summary["loss_test"] = test_loss
            self.wandb_run.summary["mae_test"] = test_metrics["mae"]
            self.wandb_run.summary["rmse_test"] = test_metrics["rmse"]
            self.wandb_run.summary["smape_test"] = test_metrics["smape"]
            self.wandb_run.summary["r2_test"] = test_metrics["r2"]
            self.wandb_run.summary["best_val_loss"] = self.best_val_loss
            self.wandb_run.finish()

        # Cleanup dataloader workers to prevent orphaned processes
        self.cleanup_dataloaders()

        return self.get_training_results()

    def _status(self, message: str, level: int = logging.INFO) -> None:
        logging.log(level, message)

    def _detect_curriculum_transition(self) -> bool:
        """Detect if we just transitioned to a new curriculum phase.

        Returns True if the current curriculum phase index differs from the previous
        epoch's phase. Used to trigger LR warmup after sparsity/synth_ratio changes.
        """
        if self.curriculum_sampler is None:
            return False
        if not hasattr(self.curriculum_sampler, "config"):
            return False

        current_idx = None
        for i, phase in enumerate(self.curriculum_sampler.config.schedule):
            if phase.start_epoch <= self.current_epoch < phase.end_epoch:
                current_idx = i
                break

        if current_idx is None:
            return False

        transition = (
            self._last_curriculum_phase_idx is not None
            and current_idx != self._last_curriculum_phase_idx
        )
        self._last_curriculum_phase_idx = current_idx
        return transition

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        # Update Curriculum
        if self.curriculum_sampler is not None:
            self.curriculum_sampler.set_curriculum(self.current_epoch)
            # Apply LR warmup after curriculum phase transitions to reduce
            # gradient explosions from sudden sparsity/synth_ratio changes
            if self._detect_curriculum_transition():
                # Save current LR (which may have been decayed by scheduler)
                # before reducing it, so we can restore to the correct value
                current_lr = self.optimizer.param_groups[0]["lr"]
                self._lr_warmup_target_lr = current_lr

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= 0.5
                self._lr_warmup_remaining = 100
                self._status(
                    f"Curriculum phase transition - LR warmup: {current_lr:.2e} -> {current_lr * 0.5:.2e} (will restore to {current_lr:.2e})",
                    logging.INFO,
                )

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
        # NaN check frequency matches progress log frequency to reduce GPU-CPU syncs
        nan_check_frequency = self.config.training.progress_log_frequency

        # Track last gradnorm for progress logging (so we always show the most recent value)
        last_gradnorm = torch.tensor(0.0)

        try:
            for batch_idx, batch_data in enumerate(train_iter):
                # Initialize profiler AFTER first batch is fetched successfully
                # This avoids CUDA context deadlock with multiprocessing workers
                if not first_iteration_done:
                    first_iteration_done = True
                    # Check if this epoch should be profiled
                    profile_epochs = self.config.training.profiler.profile_epochs
                    should_profile = self.config.training.profiler.enabled and (
                        profile_epochs is None
                        or (self.current_epoch + 1) in profile_epochs
                    )
                    if should_profile:
                        profiler = self._setup_profiler()
                        profiler.__enter__()
                        profiler_active = True
                        self._status("==== PROFILING ACTIVE ====")
                if max_batches is not None and batch_idx >= max_batches:
                    break

                self._status(f"Batch {batch_idx}", logging.DEBUG)

                data_time_s = time.time() - fetch_start_time
                batch_start_time = time.time()

                # Use precision policy for autocast settings
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.precision_policy.autocast_dtype,
                    enabled=self.precision_policy.autocast_enabled,
                ):
                    model_outputs, targets_dict = self.model.forward_batch(
                        batch_data=batch_data,
                        region_embeddings=self.region_embeddings,
                    )

                    loss = self.criterion(model_outputs, targets_dict)

                # Guard against non-finite losses to prevent corrupt optimizer state.
                # Only check at progress_log_frequency intervals to reduce GPU-CPU syncs
                should_check_nan = self.global_step % nan_check_frequency == 0
                if should_check_nan and not torch.isfinite(loss):
                    self.nan_loss_counter += 1
                    self._status(
                        "Non-finite training loss detected at "
                        f"epoch={self.current_epoch}, step={self.global_step}, "
                        f"batch={batch_idx} (counter={self.nan_loss_counter}).",
                        logging.WARNING,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    patience = self.config.training.nan_loss_patience
                    if (
                        patience is not None
                        and patience > 0
                        and self.nan_loss_counter >= patience
                    ):
                        self.nan_loss_triggered = True
                        break
                    fetch_start_time = time.time()
                    self.global_step += 1
                    continue

                # Reset counter once we see a valid loss.
                self.nan_loss_counter = 0

                # Scale loss for gradient accumulation
                scaled_loss = loss / accum_steps

                scaled_loss.backward()

                # Only step optimizer every N batches (gradient accumulation)
                should_step = (batch_idx + 1) % accum_steps == 0
                is_last_batch = batch_idx == len(self.train_loader) - 1

                grad_norm = torch.tensor(0.0)
                if should_step or is_last_batch:
                    self._log_gradient_norms(step=self.global_step)
                    self._log_sparsity_loss_correlation(
                        batch_data=batch_data,
                        model_outputs=model_outputs,
                        targets_dict=targets_dict,
                        step=self.global_step,
                    )
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_value,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        self.nan_loss_counter += 1
                        self._status(
                            "Non-finite gradient norm detected during clipping at "
                            f"epoch={self.current_epoch}, step={self.global_step}, "
                            f"batch={batch_idx} (counter={self.nan_loss_counter}). "
                            f"Error: {exc}. Skipping optimizer step.",
                            logging.WARNING,
                        )

                        # Capture gradient diagnostics if debugging is enabled
                        if self.gradient_debugger.enabled:
                            snapshot = self.gradient_debugger.capture_snapshot(
                                self.model,
                                loss=loss,
                                step_info={
                                    "step": self.global_step,
                                    "epoch": self.current_epoch,
                                    "batch_idx": batch_idx,
                                },
                            )
                            self.gradient_debugger.log_summary(snapshot)
                            self.gradient_debugger.save_report(snapshot)

                        self.optimizer.zero_grad(set_to_none=True)
                        patience = self.config.training.nan_loss_patience
                        if (
                            patience is not None
                            and patience > 0
                            and self.nan_loss_counter >= patience
                        ):
                            self.nan_loss_triggered = True
                            break
                        fetch_start_time = time.time()
                        self.global_step += 1
                        continue
                    last_gradnorm = grad_norm  # Update for progress logging
                    self.optimizer.step()

                    # LR warmup restore: after curriculum transition, gradually
                    # restore learning rate over 100 steps
                    if self._lr_warmup_remaining > 0:
                        self._lr_warmup_remaining -= 1
                        if self._lr_warmup_remaining == 0:
                            # Restore to the LR we saved before reducing
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self._lr_warmup_target_lr
                            self._status(
                                f"LR warmup complete - restored to {self._lr_warmup_target_lr:.2e}",
                                logging.INFO,
                            )

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

                bsz = batch_data["B"]
                samples_per_s = (
                    (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                )
                log_frequency = self.config.training.progress_log_frequency
                accum_steps = self.config.training.gradient_accumulation_steps
                effective_step = get_effective_optimizer_step(
                    self.global_step, accum_steps
                )
                log_this_step = should_log_step(
                    self.global_step, accum_steps, log_frequency
                )

                log_data = {
                    "learning_rate_step": lr,
                    "gradnorm_clipped_total": grad_norm,
                    "time_batch_s": batch_time_s,
                    "time_dataload_s": data_time_s,
                    "time_step_s": batch_time_s,
                    "epoch": self.current_epoch,
                }

                if log_this_step:
                    # Use detach() instead of item() - wandb handles tensor conversion
                    loss_detached = loss.detach()
                    log_data["loss_train_step"] = loss_detached
                    # Convert to scalar only for console logging
                    loss_value = float(loss_detached)
                    self._status(
                        f"Epoch {self.current_epoch} | Step {effective_step} | Loss: {loss_value:.4g} | Lr: {lr:.2e} | Grad: {float(last_gradnorm):.3f} | SPS: {samples_per_s:7.1f}",
                    )

                    # Keep as tensor - wandb handles CPU tensor conversion
                    window_start_mean = batch_data["WindowStart"].float().mean()
                    log_data["time_window_start"] = window_start_mean

                    # Log curriculum metrics for loss-curve-critic analysis
                    if self.curriculum_sampler is not None and hasattr(
                        self.curriculum_sampler, "state"
                    ):
                        log_data["train_sparsity_step"] = (
                            self.curriculum_sampler.state.max_sparsity or 0.0
                        )

                    if self.wandb_run is not None:
                        wandb.log(log_data, step=effective_step)

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
        """Generate and save forecast plots for best/worst performing regions."""
        from evaluation.epiforecaster_eval import (
            generate_forecast_plots,
            select_nodes_by_loss,
        )

        sample_count = max(1, int(self.config.training.num_forecast_samples))
        worst_nodes = select_nodes_by_loss(
            node_mae=self._last_node_mae,
            strategy="worst",
            k=sample_count,
        ).get("Worst", [])
        best_nodes = select_nodes_by_loss(
            node_mae=self._last_node_mae,
            strategy="best",
            k=sample_count,
        ).get("Best", [])
        node_groups = {
            "Poorly-performing": worst_nodes,
            "Well-performing": best_nodes,
        }
        if not any(node_groups.values()):
            self._status(f"[plot] No nodes selected for {split} forecast plotting")
            return

        output_dir = self.experiment_dir
        if output_dir is None:
            raise RuntimeError("experiment_dir not set; call setup_logging() first")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{split}_forecasts_joint.png"
        generate_forecast_plots(
            model=self.model,
            loader=loader,
            node_groups=node_groups,
            window="last",
            output_path=plot_path,
            log_dir=self.experiment_dir,
            wandb_prefix=f"forecasts_{split}",
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
        """Calculates and logs the gradient norms for model components.

        Groups parameters into semantically meaningful categories:
        - SIRD physics: beta/gamma/mortality/initial_states projections
        - Observation heads: ww_head, hosp_head, cases_head, deaths_head (per-head + total)
        - Backbone encoder: remaining backbone params (transformer, embeddings)
        - Mobility GNN: graph neural network parameters
        - Other: catch-all for any remaining parameters
        """
        frequency = self.config.training.grad_norm_log_frequency
        accum_steps = self.config.training.gradient_accumulation_steps
        effective_step = get_effective_optimizer_step(step, accum_steps)

        if frequency <= 0 or (effective_step % frequency != 0 and effective_step != 0):
            return

        if not any(p.requires_grad for p in self.model.parameters()):
            return

        sird_sq_sum = torch.tensor(0.0, device=self.device)
        encoder_sq_sum = torch.tensor(0.0, device=self.device)
        gnn_sq_sum = torch.tensor(0.0, device=self.device)
        other_sq_sum = torch.tensor(0.0, device=self.device)

        ww_sq_sum = torch.tensor(0.0, device=self.device)
        hosp_sq_sum = torch.tensor(0.0, device=self.device)
        cases_sq_sum = torch.tensor(0.0, device=self.device)
        deaths_sq_sum = torch.tensor(0.0, device=self.device)

        sird_projections = {
            "beta_projection",
            "gamma_projection",
            "mortality_projection",
            "initial_states_projection",
        }

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_sq_sum = param.grad.detach().pow(2).sum()

                if "mobility_gnn" in name:
                    gnn_sq_sum += grad_sq_sum
                elif "ww_head" in name:
                    ww_sq_sum += grad_sq_sum
                elif "hosp_head" in name:
                    hosp_sq_sum += grad_sq_sum
                elif "cases_head" in name:
                    cases_sq_sum += grad_sq_sum
                elif "deaths_head" in name:
                    deaths_sq_sum += grad_sq_sum
                elif any(proj in name for proj in sird_projections):
                    sird_sq_sum += grad_sq_sum
                elif "backbone" in name:
                    encoder_sq_sum += grad_sq_sum
                else:
                    other_sq_sum += grad_sq_sum

        obs_heads_sq_sum = ww_sq_sum + hosp_sq_sum + cases_sq_sum + deaths_sq_sum

        component_sq_sums = torch.stack(
            [
                sird_sq_sum,
                encoder_sq_sum,
                gnn_sq_sum,
                obs_heads_sq_sum,
                other_sq_sum,
            ]
        )
        total_sq_sum = component_sq_sums.sum()

        per_head_sq_sums = torch.stack(
            [ww_sq_sum, hosp_sq_sum, cases_sq_sum, deaths_sq_sum]
        )

        all_sq_sums = torch.cat(
            [
                total_sq_sum.unsqueeze(0),
                component_sq_sums,
                per_head_sq_sums,
            ]
        )
        all_norms = all_sq_sums.sqrt().cpu().numpy()

        (
            total_norm,
            sird_norm,
            encoder_norm,
            gnn_norm,
            obs_heads_norm,
            other_norm,
            ww_norm,
            hosp_norm,
            cases_norm,
            deaths_norm,
        ) = all_norms

        if self.wandb_run is not None:
            wandb.log(
                {
                    "gradnorm_total_preclip": total_norm,
                    "gradnorm_sird_physics": sird_norm,
                    "gradnorm_backbone_encoder": encoder_norm,
                    "gradnorm_mobility_gnn": gnn_norm,
                    "gradnorm_observation_heads": obs_heads_norm,
                    "gradnorm_obs_ww": ww_norm,
                    "gradnorm_obs_hosp": hosp_norm,
                    "gradnorm_obs_cases": cases_norm,
                    "gradnorm_obs_deaths": deaths_norm,
                    "gradnorm_other": other_norm,
                    "gradnorm_backbone": encoder_norm + sird_norm,
                },
                step=effective_step,
            )

        self._status(
            f"Grad norms @ step {step}: Total={total_norm:.4f} | "
            f"SIRD={sird_norm:.4f} | Enc={encoder_norm:.4f} | "
            f"GNN={gnn_norm:.4f} | Obs={obs_heads_norm:.4f} | Other={other_norm:.4f}",
            logging.DEBUG,
        )

    def _log_sparsity_loss_correlation(
        self,
        batch_data: dict[str, Any],
        model_outputs: dict[str, torch.Tensor],
        targets_dict: dict[str, torch.Tensor | None],
        step: int,
    ) -> None:
        """Log per-sample sparsity and loss distributions for correlation analysis.

        Uses the same frequency as gradient norm logging (grad_norm_log_frequency).
        Logs W&B histograms for:
        - Input history sparsity: hosp_hist, cases_hist, deaths_hist, bio_hist, mob_hist
        - Target sparsity: hosp_target, ww_target, cases_target, deaths_target
        - Per-sample losses: loss_hosp, loss_ww, loss_cases, loss_deaths

        Args:
            batch_data: Collated batch from EpiDataset
            model_outputs: Dict from EpiForecaster.forward() with predictions
            targets_dict: Dict with observation targets and masks
            step: Global training step
        """
        frequency = self.config.training.grad_norm_log_frequency
        accum_steps = self.config.training.gradient_accumulation_steps
        effective_step = get_effective_optimizer_step(step, accum_steps)

        if frequency <= 0 or (effective_step % frequency != 0 and effective_step != 0):
            return

        log_sparsity_loss_correlation(
            batch=batch_data,
            model_outputs=model_outputs,
            targets=targets_dict,
            wandb_run=self.wandb_run,
            step=effective_step,
            epoch=self.current_epoch,
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
        accum_steps = self.config.training.gradient_accumulation_steps
        effective_step = get_effective_optimizer_step(self.global_step, accum_steps)
        # Initial standard metrics
        log_data = {
            "epoch": epoch,
            f"loss_{prefix.lower()}": loss,
            f"mae_{prefix.lower()}": metrics["mae"],
            f"rmse_{prefix.lower()}": metrics["rmse"],
            f"smape_{prefix.lower()}": metrics["smape"],
            f"r2_{prefix.lower()}": metrics["r2"],
        }

        # Add Joint Inference specific metrics
        # Raw losses
        if "loss_ww" in metrics:
            log_data[f"loss_{prefix.lower()}_ww"] = metrics["loss_ww"]
        if "loss_hosp" in metrics:
            log_data[f"loss_{prefix.lower()}_hosp"] = metrics["loss_hosp"]
        if "loss_cases" in metrics:
            log_data[f"loss_{prefix.lower()}_cases"] = metrics["loss_cases"]
        if "loss_deaths" in metrics:
            log_data[f"loss_{prefix.lower()}_deaths"] = metrics["loss_deaths"]
        if "loss_sir" in metrics:
            log_data[f"loss_{prefix.lower()}_sir"] = metrics["loss_sir"]

        # Weighted losses
        if "loss_ww_weighted" in metrics:
            log_data[f"loss_{prefix.lower()}_ww_weighted"] = metrics["loss_ww_weighted"]
        if "loss_hosp_weighted" in metrics:
            log_data[f"loss_{prefix.lower()}_hosp_weighted"] = metrics[
                "loss_hosp_weighted"
            ]
        if "loss_cases_weighted" in metrics:
            log_data[f"loss_{prefix.lower()}_cases_weighted"] = metrics[
                "loss_cases_weighted"
            ]
        if "loss_deaths_weighted" in metrics:
            log_data[f"loss_{prefix.lower()}_deaths_weighted"] = metrics[
                "loss_deaths_weighted"
            ]
        if "loss_sir_weighted" in metrics:
            log_data[f"loss_{prefix.lower()}_sir_weighted"] = metrics[
                "loss_sir_weighted"
            ]

        # Add per-horizon metrics
        for idx, (mae_h, rmse_h) in enumerate(
            zip(metrics["mae_per_h"], metrics["rmse_per_h"], strict=False)
        ):
            log_data[f"mae_{prefix.lower()}_h{idx + 1}"] = mae_h
            log_data[f"rmse_{prefix.lower()}_h{idx + 1}"] = rmse_h

        # Log to WandB
        if self.wandb_run is not None:
            wandb.log(log_data)

        # Log curriculum metrics for loss-curve-critic analysis
        if self.curriculum_sampler is not None and hasattr(
            self.curriculum_sampler, "state"
        ):
            log_data["train_sparsity_epoch"] = (
                self.curriculum_sampler.state.max_sparsity or 0.0
            )
            log_data["train_synth_ratio_epoch"] = (
                self.curriculum_sampler.state.synth_ratio
            )
        if self.wandb_run is not None:
            wandb.log(log_data, step=effective_step)

        self._status(
            f"{prefix} loss: {loss:.4g} | MAE: {metrics['mae']:.4g} | RMSE: {metrics['rmse']:.4g} | sMAPE: {metrics['smape']:.4g} | R2: {metrics['r2']:.4g}"
        )
        if (
            "loss_ww" in metrics
            and "loss_hosp" in metrics
            and "loss_sir" in metrics
            and "loss_ww_weighted" in metrics
            and "loss_hosp_weighted" in metrics
            and "loss_sir_weighted" in metrics
        ):
            # Build component strings, only including non-zero cases/deaths
            components_str = (
                f"WW={metrics['loss_ww']:.4g} (w={metrics['loss_ww_weighted']:.4g}) | "
                f"Hosp={metrics['loss_hosp']:.4g} (w={metrics['loss_hosp_weighted']:.4g}) | "
                f"SIR={metrics['loss_sir']:.4g} (w={metrics['loss_sir_weighted']:.4g})"
            )
            if "loss_cases" in metrics:
                components_str += f" | Cases={metrics['loss_cases']:.4g} (w={metrics.get('loss_cases_weighted', 0):.4g})"
            if "loss_deaths" in metrics:
                components_str += f" | Deaths={metrics['loss_deaths']:.4g} (w={metrics.get('loss_deaths_weighted', 0):.4g})"
            self._status(f"{prefix} loss components: {components_str}")
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
        from utils.precision_policy import create_precision_signature

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
            "training_history": self.training_history,
            "precision_signature": create_precision_signature(self.precision_policy),
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

    def _shutdown_loader_iterator(self, loader) -> None:
        """Shutdown DataLoader iterator to release workers and resources.

        Call this between training phases (e.g., after train epoch, before validation)
        to prevent semaphore leaks from overlapping worker lifetimes.

        The iterator holds worker processes that keep Zarr file handles open.
        Shutting it down between phases allows proper resource cleanup.

        Args:
            loader: DataLoader instance whose iterator should be shut down
        """
        if (
            loader is not None
            and hasattr(loader, "_iterator")
            and loader._iterator is not None
        ):
            loader._iterator._shutdown_workers()
            loader._iterator = None

    def cleanup_dataloaders(self) -> None:
        """Explicitly cleanup DataLoader workers to prevent orphaned processes.

        This is critical for HPC/SLURM environments where persistent workers
        can cause hangs and resource leaks if not properly terminated.

        The cleanup process:
        1. Gracefully shutdown iterator workers first (allows dataset cleanup)
        2. If workers persist, terminate and join them (ensures process cleanup)
        3. Clear loader references to prevent access to cleaned-up resources

        This prevents semaphore leaks caused by Zarr file handles remaining open
        when workers are forcefully terminated without proper cleanup.
        """
        loaders = [
            ("train", self.train_loader) if hasattr(self, "train_loader") else None,
            ("val", self.val_loader) if hasattr(self, "val_loader") else None,
            ("test", self.test_loader) if hasattr(self, "test_loader") else None,
        ]
        for name, loader in loaders:
            if loader is not None:
                try:
                    # First, shut down the iterator if it exists (graceful shutdown)
                    # This allows workers to close file handles and release resources
                    self._shutdown_loader_iterator(loader)

                    # Then cleanup workers directly if any remain
                    if hasattr(loader, "_workers"):
                        workers = loader._workers  # type: ignore[attr-defined]
                        if workers:
                            self._status(
                                f"Cleaning up {name} loader ({len(workers)} workers)...",
                                logging.DEBUG,
                            )
                            for worker in workers:
                                if worker.is_alive():
                                    # Graceful shutdown via iterator should have handled this
                                    # but terminate any remaining workers and wait for cleanup
                                    worker.terminate()
                                    worker.join(timeout=5.0)
                                    if worker.is_alive():
                                        self._status(
                                            f"Worker {worker} did not shut down gracefully",
                                            logging.WARNING,
                                        )

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
