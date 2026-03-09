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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset_factory import build_datasets
from data.epi_batch import EpiBatch
from evaluation.aggregate_export import write_main_model_aggregate_csvs
from evaluation.epiforecaster_eval import evaluate_loader
from evaluation.losses import JointInferenceLoss
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from training.dataloader_factory import (
    build_dataloaders,
    should_prestart_dataloader_workers,
)
from training.gradnorm import GradNormController
from utils.device import setup_tensor_core_optimizations
from utils.gradnorm_logging import (
    append_gradnorm_sidecar_metrics,
    did_gradnorm_sidecar_run,
    format_gradnorm_controller_status,
    init_gradnorm_sidecar_log_data,
    mark_gradnorm_sidecar_complete,
)
from utils.gradient_debug import GradientDebugger
from utils.platform import (
    cleanup_nvme_staging,
    get_nvme_path,
    is_slurm_cluster,
    stage_dataset_to_nvme,
)
from utils.train_logging import (
    add_curriculum_metrics,
    build_epoch_logging_bundle,
    build_train_step_log_data,
    compute_gradient_norms_and_clip,
    format_component_gradnorm_status,
    format_train_progress_status,
    get_wandb_step_payload,
    should_log_gradnorm_components,
)
from utils.training_utils import should_log_step

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
        self.experiment_dir: Path | None = None
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        # Optuna pruning support
        self.trial = trial
        self.pruning_start_epoch = pruning_start_epoch

        # Stage data to NVMe if running on SLURM cluster
        self._nvme_staging_path: Path | None = None
        if is_slurm_cluster():
            self._stage_data_to_nvme()

        # Build datasets using factory
        dataset_splits = build_datasets(config)
        self.train_dataset = dataset_splits.train
        self.val_dataset = dataset_splits.val
        self.test_dataset = dataset_splits.test
        self.real_run_id = dataset_splits.real_run_id
        self.synth_run_ids = dataset_splits.synth_run_ids

        # Access cases_dim/biomarkers_dim safely (handle ConcatDataset)
        if isinstance(self.train_dataset, ConcatDataset):
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
            init_weights=self.config.model.init_weights,
            observation_heads=self.config.model.observation_heads,
            temporal_input_dim=train_example_ds.cases_output_dim,
            biomarkers_dim=train_example_ds.biomarkers_output_dim,
            region_embedding_dim=self.config.model.region_embedding_dim,
            mobility_embedding_dim=self.config.model.mobility_embedding_dim,
            gnn_depth=self.config.model.gnn_depth,
            sequence_length=self.config.model.input_window_length,
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
            strict=self.config.model.strict,
        )

        # Setup data loaders before CUDA initialization when using fork
        loader_bundle = build_dataloaders(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            training_config=self.config.training,
            model_type_config=self.config.model.type,
            curriculum_config=self.config.training.curriculum
            if self.config.training.curriculum.enabled
            else None,
            real_run_id=self.real_run_id or "real",
            device_hint=self._device_hint,
            seed=self.config.training.seed,
        )
        self.train_loader = loader_bundle.train
        self.val_loader = loader_bundle.val
        self.test_loader = loader_bundle.test
        self.curriculum_sampler = loader_bundle.curriculum_sampler
        self._multiprocessing_context = loader_bundle.multiprocessing_context

        if should_prestart_dataloader_workers(
            self._multiprocessing_context, self._device_hint
        ):
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

        # Apply torch.compile if enabled
        if self.config.training.compile:
            compile_kwargs: dict[str, Any] = {}
            if self.config.training.compile_mode != "default":
                compile_kwargs["mode"] = self.config.training.compile_mode

            if self.config.training.compile_dynamic is not None:
                compile_kwargs["dynamic"] = self.config.training.compile_dynamic

            if self.config.training.compile_fullgraph:
                compile_kwargs["fullgraph"] = True

            compile_desc = (
                ", ".join(f"{k}={v}" for k, v in compile_kwargs.items())
                if compile_kwargs
                else "default settings"
            )
            self._status(
                f"Applying torch.compile to model for performance ({compile_desc})",
                logging.INFO,
            )
            self.model = torch.compile(self.model, **compile_kwargs)

        # Enable TF32 for better performance on Ampere+ GPUs
        self._setup_tensor_core_optimizations()

        # Pre-compute parameter groups for efficient gradient norm logging
        self._init_gradient_norm_groups()

        # Setup training components (optimizer, scheduler, criterion)
        self.optimizer = self._create_optimizer()

        from training.schedulers import compute_scheduler_steps

        total_steps, warmup_steps = compute_scheduler_steps(
            epochs=self.config.training.epochs,
            batches_per_epoch=len(self.train_loader),
            gradient_accumulation_steps=1,
            warmup_batches=self.config.training.warmup_steps,
        )
        self.scheduler = self._create_scheduler(
            total_steps=total_steps, warmup_steps=warmup_steps
        )
        self.criterion = self._create_criterion()

        joint_cfg = self.config.training.loss.joint
        self._adaptive_scheme = joint_cfg.adaptive_scheme
        self._gradnorm_enabled = self._adaptive_scheme == "gradnorm"
        self.gradnorm_controller: GradNormController | None = None
        self.gradnorm_optimizer: torch.optim.Optimizer | None = None
        self._gradnorm_probe = joint_cfg.gradnorm_probe
        self._gradnorm_cached_weights = torch.full(
            (len(GradNormController.task_names),),
            joint_cfg.gradnorm_obs_weight_sum / len(GradNormController.task_names),
            device=self.device,
            dtype=torch.float32,
        )
        self._gradnorm_last_active_mask = torch.ones(
            len(GradNormController.task_names),
            device=self.device,
            dtype=torch.bool,
        )

        if self._gradnorm_enabled:
            self.gradnorm_controller = GradNormController(
                alpha=joint_cfg.gradnorm_alpha,
                obs_weight_sum=joint_cfg.gradnorm_obs_weight_sum,
                min_weight=joint_cfg.gradnorm_min_weight,
                warmup_steps=joint_cfg.gradnorm_warmup_steps,
                update_every=joint_cfg.gradnorm_update_every,
                ema_decay=joint_cfg.gradnorm_ema_decay,
                probe=joint_cfg.gradnorm_probe,
            ).to(self.device)
            self.gradnorm_optimizer = torch.optim.Adam(
                self.gradnorm_controller.parameters(),
                lr=joint_cfg.gradnorm_weight_lr,
            )
            self._gradnorm_cached_weights = self.gradnorm_controller.weights().detach()

        # Compile training step (forward + backward) if enabled
        self._compiled_training_step = None
        if self.config.training.compile_backward:
            compile_target = self._training_step_impl
            if self._gradnorm_enabled:
                compile_target = self._training_step_impl_adaptive
                self._status(
                    "Compiling adaptive training step for GradNorm hybrid mode",
                    logging.INFO,
                )
            else:
                self._status(
                    "Compiling training step (forward+backward)",
                    logging.INFO,
                )

            compile_kwargs: dict[str, Any] = {}
            if self.config.training.compile_mode != "default":
                compile_kwargs["mode"] = self.config.training.compile_mode

            if self.config.training.compile_dynamic is not None:
                compile_kwargs["dynamic"] = self.config.training.compile_dynamic

            # Use fullgraph=False to allow data-dependent control flow if needed
            compile_kwargs["fullgraph"] = False

            self._status(
                f"Compile mode={self.config.training.compile_mode}",
                logging.INFO,
            )
            self._compiled_training_step = torch.compile(
                compile_target, **compile_kwargs
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
        self.metric_artifacts: dict[str, Path] = {}
        self._model_graph_logged = False
        self._last_node_mae: dict[int, float] = {}
        # Curriculum phase tracking for LR warmup at transitions
        self._last_curriculum_phase_idx: int | None = None
        self._lr_warmup_remaining: int = 0
        self._lr_warmup_target_lr: float = 0.0  # Target LR to restore after warmup

        # Initialize gradient debugger (zero overhead when disabled)
        grad_debug_dir = self.config.training.gradient_debug_log_dir
        if grad_debug_dir is None and (
            self.config.training.enable_gradient_debug
            or self.config.training.gradient_snapshot_frequency > 0
        ):
            # Auto-set to experiment directory if enabled but not specified
            grad_debug_dir = (
                self.experiment_dir / "gradient_debug" if self.experiment_dir else None
            )
        gradient_debug_enabled = self.config.training.enable_gradient_debug or (
            self.config.training.gradient_snapshot_frequency > 0
        )
        self.gradient_debugger = GradientDebugger(
            enabled=gradient_debug_enabled,
            log_dir=grad_debug_dir,
            vanishing_threshold=self.config.training.gradient_vanishing_threshold,
            exploding_threshold=self.config.training.gradient_exploding_threshold,
            snapshot_top_k=self.config.training.gradient_snapshot_top_k,
            logger_instance=logger,
        )
        self._gradient_snapshot_frequency = (
            self.config.training.gradient_snapshot_frequency
        )
        if self.gradient_debugger.enabled and self.gradient_debugger.log_dir:
            self._status(
                f"Gradient diagnostics enabled. Reports will be saved to: {self.gradient_debugger.log_dir}",
                logging.INFO,
            )
        if self._gradient_snapshot_frequency > 0:
            snapshot_dir = (
                self.gradient_debugger.log_dir
                if self.gradient_debugger.log_dir is not None
                else "disabled (set training.gradient_debug_log_dir to persist JSON)"
            )
            self._status(
                "Periodic gradient snapshots enabled: "
                f"every {self._gradient_snapshot_frequency} steps "
                f"(top_k={self.config.training.gradient_snapshot_top_k}, "
                f"vanishing<={self.config.training.gradient_vanishing_threshold:.1e}, "
                f"exploding>={self.config.training.gradient_exploding_threshold:.1e})",
                logging.INFO,
            )
            self._status(f"  Snapshot output: {snapshot_dir}", logging.INFO)

        # Resume from checkpoint after all state is initialized
        if self.config.training.resume_checkpoint_path:
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
        loss_cfg = self.config.training.loss.joint
        self._status("  Joint Inference Loss:")
        self._status(f"    - Adaptive scheme: {loss_cfg.adaptive_scheme}")
        self._status(
            f"    - Obs weight sum (fixed eval objective): {loss_cfg.gradnorm_obs_weight_sum}"
        )
        self._status(f"    - SIR weight: {loss_cfg.w_sir}")
        self._status(f"    - Continuity weight: {loss_cfg.w_continuity}")
        self._status(
            "    - n_eff scaling: "
            f"power={loss_cfg.obs_n_eff_power}, "
            f"reference={loss_cfg.obs_n_eff_reference}, "
            f"per_head=(ww:{loss_cfg.ww_n_eff_reference}, "
            f"hosp:{loss_cfg.hosp_n_eff_reference}, "
            f"cases:{loss_cfg.cases_n_eff_reference}, "
            f"deaths:{loss_cfg.deaths_n_eff_reference})"
        )
        if self._gradnorm_enabled:
            self._status(
                "    - GradNorm settings: "
                f"alpha={loss_cfg.gradnorm_alpha}, "
                f"weight_lr={loss_cfg.gradnorm_weight_lr}, "
                f"warmup_steps={loss_cfg.gradnorm_warmup_steps}, "
                f"update_every={loss_cfg.gradnorm_update_every}, "
                f"ema_decay={loss_cfg.gradnorm_ema_decay}, "
                f"probe={loss_cfg.gradnorm_probe}, "
                f"min_weight={loss_cfg.gradnorm_min_weight}"
            )

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
        resume_status = (
            f"enabled from {self.config.training.resume_checkpoint_path}"
            if self.config.training.resume_checkpoint_path
            else "disabled"
        )
        self._status(f"  Resume: {resume_status}")
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
        Stores original paths for config export/checkpointing.
        Only runs when on a SLURM cluster with NVMe available.
        """
        enable_staging = os.getenv("EPFORECASTER_STAGE_TO_NVME", "1") != "0"
        if not enable_staging:
            logger.info("NVMe staging disabled via EPFORECASTER_STAGE_TO_NVME=0")
            return

        logger.info("Detected SLURM cluster - staging data to NVMe")
        self._nvme_staging_path = get_nvme_path()

        # Store original paths before staging for config export
        self._original_dataset_path = self.config.data.dataset_path
        self._original_real_dataset_path = self.config.data.real_dataset_path

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

        optimizer_kwargs: dict[str, Any] = {
            "lr": self.config.training.learning_rate,
            "eps": self.precision_policy.optimizer_eps,
        }

        # CUDA fast path: fused AdamW reduces optimizer kernel launch overhead.
        if optimizer_name == "adamw" and self.device.type == "cuda":
            try:
                return optimizer_cls(
                    param_groups,
                    fused=True,
                    capturable=True,
                    **optimizer_kwargs,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                self._status(
                    f"Fused AdamW unavailable, falling back to standard AdamW ({exc})",
                    logging.WARNING,
                )

        return optimizer_cls(param_groups, **optimizer_kwargs)

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

    def _create_criterion(self) -> JointInferenceLoss:
        """Create loss criterion."""
        from evaluation.losses import get_loss_from_config

        return get_loss_from_config(
            self.config.training.loss,
            data_config=self.config.data,
            forecast_horizon=self.config.model.forecast_horizon,
        )

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
                config=self._get_config_for_save(),
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
            "input_window_length": self.config.model.input_window_length,
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
    #         mob_batch = example_batch.mob_batch.to(self.device)
    #         example_inputs = (
    #             example_batch["CaseNode"].to(self.device),
    #             example_batch.bio_node.to(self.device),
    #             mob_batch,
    #             example_batch.target_node.to(self.device),
    #             self.region_embeddings
    #             if self.region_embeddings is not None
    #             else None,
    #             example_batch.population.to(self.device),
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
            record_shapes=False,
            profile_memory=self.config.training.profiler.record_memory,
            with_stack=self.config.training.profiler.with_stack,
        )

    def _resolve_profiler_log_dir(self) -> Path:
        configured = self.config.training.profiler.log_dir
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
        model_id_override = os.getenv("EPIFORECASTER_MODEL_ID", "").strip()
        if model_id_override:
            return model_id_override

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

    def _resume_from_checkpoint(self) -> None:
        from utils.precision_policy import validate_old_checkpoint_compatible

        checkpoint_path = self.config.training.resume_checkpoint_path
        if checkpoint_path is None:
            raise ValueError("resume_checkpoint_path is not set")
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
        gradnorm_controller = getattr(self, "gradnorm_controller", None)
        gradnorm_optimizer = getattr(self, "gradnorm_optimizer", None)
        if (
            gradnorm_controller is not None
            and "gradnorm_controller_state_dict" in checkpoint
        ):
            gradnorm_controller.load_state_dict(
                checkpoint["gradnorm_controller_state_dict"]
            )
            self._gradnorm_cached_weights = gradnorm_controller.weights().detach()
        if (
            gradnorm_optimizer is not None
            and "gradnorm_optimizer_state_dict" in checkpoint
        ):
            gradnorm_optimizer.load_state_dict(
                checkpoint["gradnorm_optimizer_state_dict"]
            )
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

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            _train_loss = self._train_epoch()

            if self.device.type == "cuda" and self.wandb_run is not None:
                max_mem_mb = torch.cuda.max_memory_allocated(self.device) / (
                    1024 * 1024
                )
                wandb.log(
                    {"epoch/max_gpu_memory_mb": max_mem_mb}, step=self.global_step
                )

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
            self._write_main_model_aggregate_csvs(
                split_name="val",
                eval_metrics=val_metrics,
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
        self._write_main_model_aggregate_csvs(
            split_name="test",
            eval_metrics=test_metrics,
        )
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

    def _training_step_impl(self, batch_data: EpiBatch) -> torch.Tensor:
        """Pure training step: forward + backward only.

        This method computes the forward pass, loss, and backward pass.
        It is designed to be compiled with torch.compile for performance.
        It excludes graph-breaking operations like CPU syncs, logging, I/O, and
        optimizer.step() / zero_grad() which are handled in the main loop.

        Args:
            batch_data: Dictionary containing batch tensors (already on device)

        Returns:
            Detached loss tensor on device
        """
        # Forward with autocast
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.precision_policy.autocast_dtype,
            enabled=self.precision_policy.autocast_enabled,
        ):
            model_outputs, targets_dict = self.model.forward_batch(
                batch_data=batch_data,
                region_embeddings=self.region_embeddings,
                skip_device_transfer=True,  # Tensors already moved by _move_batch_to_device
                mask_cases=self.criterion.mask_input_cases,
                mask_ww=self.criterion.mask_input_ww,
                mask_hosp=self.criterion.mask_input_hosp,
                mask_deaths=self.criterion.mask_input_deaths,
            )
            loss = self.criterion(model_outputs, targets_dict, batch_data)

        # Backward
        loss.backward()
        return loss.detach()

    def _training_step_impl_adaptive(
        self,
        batch_data: EpiBatch,
        obs_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled-friendly adaptive training step using cached observation weights."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.precision_policy.autocast_dtype,
            enabled=self.precision_policy.autocast_enabled,
        ):
            model_outputs, targets_dict = self.model.forward_batch(
                batch_data=batch_data,
                region_embeddings=self.region_embeddings,
                skip_device_transfer=True,
                mask_cases=self.criterion.mask_input_cases,
                mask_ww=self.criterion.mask_input_ww,
                mask_hosp=self.criterion.mask_input_hosp,
                mask_deaths=self.criterion.mask_input_deaths,
            )
            components = self.criterion.compute_components_train(
                model_outputs, targets_dict, batch_data
            )
            obs_active_mask = components["obs_active_mask"].to(
                device=obs_weights.device, dtype=torch.bool
            )
            active_weights = torch.where(
                obs_active_mask,
                obs_weights.to(torch.float32),
                torch.zeros_like(obs_weights, dtype=torch.float32),
            )
            active_sum = active_weights.sum().clamp_min(1e-8)
            normalized_weights = torch.where(
                obs_active_mask,
                active_weights * (self.criterion.obs_weight_sum / active_sum),
                torch.zeros_like(active_weights),
            )
            totals = self.criterion.compose_total_loss(
                components=components,
                obs_active_mask=obs_active_mask,
                obs_weights=normalized_weights.detach(),
            )
            loss = totals["total"]

        loss.backward()
        return loss.detach()

    def _gradnorm_sidecar_update(self, batch_data: EpiBatch) -> dict[str, torch.Tensor]:
        """Periodic eager GradNorm controller update on obs_context probe."""
        if self.gradnorm_controller is None or self.gradnorm_optimizer is None:
            return {}

        log_data = init_gradnorm_sidecar_log_data(self.device)

        if self.global_step % self.gradnorm_controller.update_every != 0:
            return log_data

        sidecar_start_time = time.time()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.precision_policy.autocast_dtype,
            enabled=self.precision_policy.autocast_enabled,
        ):
            model_outputs, targets_dict = self.model.forward_batch(
                batch_data=batch_data,
                region_embeddings=self.region_embeddings,
                skip_device_transfer=True,
            )
            components = self.criterion.compute_components_train(
                model_outputs, targets_dict, batch_data
            )

        obs_losses = torch.stack(
            [
                components["ww"].float(),
                components["hosp"].float(),
                components["cases"].float(),
                components["deaths"].float(),
            ]
        )
        obs_active_mask = components["obs_active_mask"].to(
            device=obs_losses.device,
            dtype=torch.bool,
        )
        self._gradnorm_last_active_mask = obs_active_mask.detach()

        self.gradnorm_controller.maybe_init_l0(
            obs_losses,
            step=self.global_step,
            active_mask=obs_active_mask,
        )

        # Keep cached weights current even before warmup/GradNorm loss activation.
        self._gradnorm_cached_weights = self.gradnorm_controller.weights().detach()

        if not bool(obs_active_mask.any()):
            mark_gradnorm_sidecar_complete(
                log_data,
                started_at=sidecar_start_time,
                device=self.device,
            )
            return log_data

        if not bool(
            self.gradnorm_controller.l0_initialized[obs_active_mask].all().item()
        ):
            mark_gradnorm_sidecar_complete(
                log_data,
                started_at=sidecar_start_time,
                device=self.device,
            )
            return log_data

        if self._gradnorm_probe != "obs_context":
            raise ValueError(f"Unsupported gradnorm probe: {self._gradnorm_probe!r}")
        probe = model_outputs["obs_context"]

        self.gradnorm_optimizer.zero_grad(set_to_none=True)
        gradnorm_terms = self.gradnorm_controller.compute_gradnorm_terms(
            obs_losses,
            probe=probe,
            active_mask=obs_active_mask,
        )
        gradnorm_loss = gradnorm_terms["gradnorm_loss"]
        gradnorm_loss.backward()
        self.gradnorm_optimizer.step()

        self._gradnorm_cached_weights = self.gradnorm_controller.weights().detach()
        self.model.zero_grad(set_to_none=True)

        mark_gradnorm_sidecar_complete(
            log_data,
            started_at=sidecar_start_time,
            device=self.device,
            gradnorm_loss=gradnorm_loss,
        )
        append_gradnorm_sidecar_metrics(
            log_data,
            cached_weights=self._gradnorm_cached_weights,
            gradnorm_terms=gradnorm_terms,
            task_names=GradNormController.task_names,
        )
        return log_data

    def _format_gradnorm_controller_status(
        self, gradnorm_step_log_data: dict[str, torch.Tensor]
    ) -> tuple[str, dict[str, float]]:
        """Build a concise, interpretable GradNorm controller progress summary."""
        return format_gradnorm_controller_status(
            gradnorm_enabled=self._gradnorm_enabled,
            gradnorm_step_log_data=gradnorm_step_log_data,
            cached_weights=self._gradnorm_cached_weights,
            last_active_mask=self._gradnorm_last_active_mask,
            task_names=GradNormController.task_names,
        )

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
        max_batches = self.config.training.max_batches
        first_iteration_done = False

        # Initialize optimizer state
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

                # Reconstruct massive dense graph tensors on the GPU directly
                # bypassing dataloader worker IPC queue.
                from utils.training_utils import (
                    ensure_mobility_adj_dense_ready,
                    inject_gpu_mobility,
                )

                inject_gpu_mobility(batch_data, train_iter.dataset, self.device)

                # Execute training step: either compiled (fast) or eager (debuggable)
                # Move all tensors to device before training step to avoid DeviceCopy ops
                batch_data = batch_data.to(
                    device=self.device, dtype=self.precision_policy.param_dtype
                )

                model_step_start_time = time.time()
                if self._gradnorm_enabled:
                    if self._compiled_training_step is not None:
                        ensure_mobility_adj_dense_ready(
                            batch_data,
                            required=bool(self.config.model.type.mobility),
                            context="compiled adaptive training",
                        )
                        loss = self._compiled_training_step(
                            batch_data,
                            self._gradnorm_cached_weights,
                        )
                    else:
                        loss = self._training_step_impl_adaptive(
                            batch_data,
                            self._gradnorm_cached_weights,
                        )
                # Call either compiled or eager version of the same step implementation
                elif self._compiled_training_step is not None:
                    ensure_mobility_adj_dense_ready(
                        batch_data,
                        required=bool(self.config.model.type.mobility),
                        context="compiled training",
                    )
                    loss = self._compiled_training_step(batch_data)
                else:
                    loss = self._training_step_impl(batch_data)
                model_step_time_s = time.time() - model_step_start_time

                gradient_snapshot_log_data: dict[str, float | int] = {}
                should_capture_snapshot = (
                    self._gradient_snapshot_frequency > 0
                    and should_log_step(
                        self.global_step, 1, self._gradient_snapshot_frequency
                    )
                )
                if should_capture_snapshot:
                    snapshot = self.gradient_debugger.capture_snapshot(
                        self.model,
                        loss=loss,
                        step_info={
                            "step": self.global_step,
                            "epoch": self.current_epoch,
                            "batch_idx": batch_idx,
                        },
                    )
                    gradient_snapshot_log_data = (
                        self.gradient_debugger.build_snapshot_log_data(snapshot)
                    )
                    if self.gradient_debugger.log_dir is not None:
                        self.gradient_debugger.save_report(snapshot)
                    self._status(
                        self.gradient_debugger.format_snapshot_status(snapshot),
                        logging.INFO,
                    )

                frequency = self.config.training.grad_norm_log_frequency
                component_gradnorm_log_data: dict[str, float] = {}
                if should_log_gradnorm_components(self.global_step, frequency):
                    grad_norm, component_gradnorm_log_data = (
                        self._compute_gradient_norms_and_clip(step=self.global_step)
                    )
                    self._status(
                        format_component_gradnorm_status(
                            self.global_step,
                            component_gradnorm_log_data,
                        ),
                        logging.DEBUG,
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.training.gradient_clip_value,
                        foreach=True,
                    )

                # Optimizer step and gradient zeroing (common to both paths)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                gradnorm_step_log_data: dict[str, torch.Tensor] = {}
                if self._gradnorm_enabled:
                    gradnorm_step_log_data = self._gradnorm_sidecar_update(batch_data)
                    for idx, task in enumerate(GradNormController.task_names):
                        if f"gradnorm_w_{task}" not in gradnorm_step_log_data:
                            gradnorm_step_log_data[f"gradnorm_w_{task}"] = (
                                self._gradnorm_cached_weights[idx]
                            )

                # Guard against non-finite losses/gradients to prevent corrupt optimizer state.
                # Only check at progress_log_frequency intervals to reduce GPU-CPU syncs
                should_check_nan = self.global_step % nan_check_frequency == 0
                if should_check_nan:
                    # Detach and move to CPU before checking to avoid GPU sync stall
                    loss_cpu = loss.detach().cpu()
                    if not torch.isfinite(loss_cpu):
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

                    # Check for non-finite grad norm on CPU (same frequency as loss check)
                    grad_norm_cpu = grad_norm.detach().cpu()
                    if not torch.isfinite(grad_norm_cpu):
                        self.nan_loss_counter += 1
                        self._status(
                            "Non-finite gradient norm detected during clipping at "
                            f"epoch={self.current_epoch}, step={self.global_step}, "
                            f"batch={batch_idx} (counter={self.nan_loss_counter}). "
                            "Skipping optimizer step.",
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

                # Reset counter once we see valid loss and grad_norm.
                self.nan_loss_counter = 0

                last_gradnorm = grad_norm  # Update for progress logging

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
                if self.scheduler and self.config.training.scheduler_type == "cosine":
                    self.scheduler.step()

                total_loss += loss.detach()
                counted_batches += 1

                batch_time_s = time.time() - batch_start_time
                fetch_start_time = time.time()
                lr = self.optimizer.param_groups[0]["lr"]

                bsz = batch_data.b
                samples_per_s = (
                    (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                )
                log_frequency = self.config.training.progress_log_frequency
                log_this_step = should_log_step(self.global_step, 1, log_frequency)

                log_data = build_train_step_log_data(
                    lr=lr,
                    grad_norm=grad_norm,
                    batch_time_s=batch_time_s,
                    data_time_s=data_time_s,
                    model_step_time_s=model_step_time_s,
                    epoch=self.current_epoch,
                    component_gradnorm_log_data=component_gradnorm_log_data,
                    gradnorm_step_log_data=gradnorm_step_log_data,
                    gradient_snapshot_log_data=gradient_snapshot_log_data,
                )

                if log_this_step:
                    # Use detach() instead of item() - wandb handles tensor conversion
                    loss_detached = loss.detach()
                    log_data["loss_train_step"] = loss_detached
                    # Convert to scalar only for console logging
                    loss_value = float(loss_detached)
                    gradnorm_status_line = ""
                    if self._gradnorm_enabled:
                        if did_gradnorm_sidecar_run(gradnorm_step_log_data):
                            gradnorm_status_line, gradnorm_controller_metrics = (
                                self._format_gradnorm_controller_status(
                                    gradnorm_step_log_data
                                )
                            )
                            log_data.update(gradnorm_controller_metrics)
                    self._status(
                        format_train_progress_status(
                            epoch=self.current_epoch,
                            step=self.global_step,
                            loss_value=loss_value,
                            lr=lr,
                            grad_norm=last_gradnorm,
                            samples_per_s=samples_per_s,
                            gradnorm_status_suffix="",
                        ),
                    )
                    if gradnorm_status_line:
                        self._status(f"  {gradnorm_status_line}")

                    # Keep as tensor - wandb handles CPU tensor conversion
                    window_start_mean = batch_data.window_start.float().mean()
                    log_data["time_window_start"] = window_start_mean

                    # Log curriculum metrics for loss-curve-critic analysis
                    add_curriculum_metrics(
                        log_data=log_data,
                        curriculum_sampler=self.curriculum_sampler,
                        key_suffix="step",
                        include_synth_ratio=False,
                    )
                wandb_payload = get_wandb_step_payload(
                    log_this_step=log_this_step,
                    log_data=log_data,
                    component_gradnorm_log_data=component_gradnorm_log_data,
                    gradient_snapshot_log_data=gradient_snapshot_log_data,
                )
                if self.wandb_run is not None and wandb_payload is not None:
                    wandb.log(wandb_payload, step=self.global_step)

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
            max_batches=self.config.training.max_batches,
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

    def _init_gradient_norm_groups(self):
        """Pre-compute parameter groups for efficient gradient norm computation.

        Categorizes parameters by component once at init to avoid string matching
        in the hot training loop.
        """
        sird_projections = {
            "beta_projection",
            "gamma_projection",
            "mortality_projection",
            "initial_states_projection",
        }

        self._grad_norm_groups: dict[str, list[torch.nn.Parameter]] = {
            "mobility_gnn": [],
            "ww_head": [],
            "hosp_head": [],
            "cases_head": [],
            "deaths_head": [],
            "sird": [],
            "backbone": [],
            "other": [],
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "mobility_gnn" in name:
                self._grad_norm_groups["mobility_gnn"].append(param)
            elif "ww_head" in name:
                self._grad_norm_groups["ww_head"].append(param)
            elif "hosp_head" in name:
                self._grad_norm_groups["hosp_head"].append(param)
            elif "cases_head" in name:
                self._grad_norm_groups["cases_head"].append(param)
            elif "deaths_head" in name:
                self._grad_norm_groups["deaths_head"].append(param)
            elif any(proj in name for proj in sird_projections):
                self._grad_norm_groups["sird"].append(param)
            elif "backbone" in name:
                self._grad_norm_groups["backbone"].append(param)
            else:
                self._grad_norm_groups["other"].append(param)

    def _compute_gradient_norms_and_clip(
        self, step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute gradient norms by component and apply clipping in a single pass.

        Combines gradient norm computation and clipping to avoid iterating
        parameters twice.

        Returns:
            Tuple of (global_grad_norm, component_norms_dict)
        """
        return compute_gradient_norms_and_clip(
            grad_norm_groups=self._grad_norm_groups,
            model=self.model,
            device=self.device,
            step=step,
            frequency=self.config.training.grad_norm_log_frequency,
            clip_value=self.config.training.gradient_clip_value,
        )

    def _persist_run_config(self, run_dir: Path) -> None:
        """Copy the input configuration to the run directory.
        Note that the config is saved in the model snapshots eg. best_model.pt
        So this is purely a convenience method for easier readability
        """
        config_dict = self._get_config_for_save()
        config_path = run_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def _write_main_model_aggregate_csvs(
        self,
        split_name: str,
        eval_metrics: dict[str, Any],
    ) -> dict[str, Path]:
        if self.experiment_dir is None:
            raise RuntimeError("experiment_dir not set; call setup_logging() first")

        artifacts = write_main_model_aggregate_csvs(
            run_dir=self.experiment_dir,
            split=split_name.lower(),
            eval_metrics=eval_metrics,
            model_name="epiforecaster",
        )
        self.metric_artifacts.update(artifacts)
        for artifact_name, artifact_path in artifacts.items():
            self._status(f"Saved {artifact_name}: {artifact_path}", logging.DEBUG)
        return artifacts

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
        aggregation = getattr(
            self.config.training, "horizon_metric_aggregation", "weekly"
        )
        log_data, status_lines = build_epoch_logging_bundle(
            split_name=split_name,
            loss=loss,
            metrics=metrics,
            epoch=epoch,
            aggregation=aggregation,
            curriculum_sampler=self.curriculum_sampler,
        )

        if self.wandb_run is not None:
            wandb.log(log_data, step=self.global_step)

        for status_line in status_lines:
            self._status(status_line)

    def _get_config_for_save(self) -> dict:
        """Get config dict with original dataset paths (not staged NVMe paths).

        When running on SLURM, datasets are staged to NVMe and config paths are
        updated to point to the staged locations. This method returns a config
        dict with the original paths preserved for checkpoint/export.

        Returns:
            Config dict with original dataset paths.
        """
        config_dict = self.config.to_dict()

        # Restore original paths if they were stored during staging
        if hasattr(self, "_original_dataset_path"):
            config_dict["data"]["dataset_path"] = self._original_dataset_path
        if hasattr(self, "_original_real_dataset_path"):
            original_real = self._original_real_dataset_path
            config_dict["data"]["real_dataset_path"] = original_real

        return config_dict

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
            "config": self._get_config_for_save(),
            "training_history": self.training_history,
            "precision_signature": create_precision_signature(self.precision_policy),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        gradnorm_controller = getattr(self, "gradnorm_controller", None)
        gradnorm_optimizer = getattr(self, "gradnorm_optimizer", None)
        if gradnorm_controller is not None:
            checkpoint["gradnorm_controller_state_dict"] = (
                gradnorm_controller.state_dict()
            )
        if gradnorm_optimizer is not None:
            checkpoint["gradnorm_optimizer_state_dict"] = (
                gradnorm_optimizer.state_dict()
            )

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
            "metric_artifacts": dict(getattr(self, "metric_artifacts", {})),
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            },
        }
