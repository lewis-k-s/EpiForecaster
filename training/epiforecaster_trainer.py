"""
Trainer for the EpiForecaster model.

This module implements a trainer class that can handle the EpiForecaster model
through configuration. It provides a unified interface for training the EpiForecaster
model while maintaining the flexibility to support various data configurations.

The trainer works with the EpiForecaster model.
"""

import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import yaml
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.epi_dataset import (
    EpiDataset,
    EpiDatasetItem,
    curriculum_collate_fn,
    optimized_collate_graphs,
)
from data.preprocess.config import REGION_COORD
from data.samplers import EpidemicCurriculumSampler
from evaluation.epiforecaster_eval import evaluate_loader
from utils import setup_tensor_core_optimizations
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

    def __init__(self, config: EpiForecasterConfig):
        """
        Initialize the unified trainer.

        Args:
            config: Trainer configuration
            dataset: Optional pre-loaded dataset (will be loaded if None)
        """
        self.config = config
        self._device_hint = self._resolve_device_hint()
        # Keep CPU until DataLoader workers are forked to avoid CUDA init before forking.
        self.device = torch.device("cpu")
        self.model_id = self._resolve_model_id()
        self.resume = self.config.training.resume

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
                    run_id=real_run,
                    region_id_index=region_id_index,
                )

                self.test_dataset = EpiDataset(
                    config=real_config,
                    target_nodes=test_nodes,
                    context_nodes=train_nodes + val_nodes,
                    biomarker_preprocessor=fitted_bio_preprocessor,
                    mobility_preprocessor=fitted_mobility_preprocessor,
                    run_id=real_run,
                    region_id_index=region_id_index,
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
            self.region_embeddings = embeddings
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
        if self.region_embeddings is not None:
            self.region_embeddings = self.region_embeddings.to(self.device)

        self.model.to(self.device)

        # Enable TF32 for better performance on Ampere+ GPUs
        self._setup_tensor_core_optimizations()

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
        # Curriculum phase tracking for LR warmup at transitions
        self._last_curriculum_phase_idx: int | None = None
        self._lr_warmup_remaining: int = 0
        self._lr_warmup_target_lr: float = 0.0  # Target LR to restore after warmup

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
            candidates = [(run_id, sparsity) for run_id, sparsity in candidates if sparsity is not None]
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

        val_persistent_workers = (
            self.config.training.persistent_workers and val_num_workers > 0
        )
        val_loader_kwargs = {
            "dataset": self.val_dataset,
            "batch_size": self.config.training.batch_size,
            "shuffle": False,
            "num_workers": val_num_workers,
            "pin_memory": pin_memory,
            "collate_fn": self._collate_fn,
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
            "collate_fn": self._collate_fn,
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

    @staticmethod
    def _collate_fn(batch: list[EpiDatasetItem]) -> dict[str, Any]:
        "Custom collate for per-node samples with PyG mobility graphs."

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

        # Use optimized manual batching
        mob_batch = optimized_collate_graphs(batch)

        T = batch[0]["mob_x"].shape[0] if B > 0 else 0
        # store B and T on the batch for downstream reshaping
        mob_batch.B = torch.tensor([B], dtype=torch.long)  # type: ignore[attr-defined]
        mob_batch.T = torch.tensor([T], dtype=torch.long)  # type: ignore[attr-defined]

        # Target index is now precomputed inside optimized_collate_graphs

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
        self._status(f"{'=' * 60}")

        # Close tensorboard writer
        self.writer.close()

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
                        f"Epoch {self.current_epoch} | Step {self.global_step} | Loss: {loss_value:.4g} | Lr: {lr:.2e} | Grad: {float(last_gradnorm):.3f} | SPS: {samples_per_s:7.1f}",
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

                # Log curriculum metrics for loss-curve-critic analysis
                if (
                    self.curriculum_sampler is not None
                    and hasattr(self.curriculum_sampler, "state")
                ):
                    self.writer.add_scalar(
                        "Train/Sparsity",
                        self.curriculum_sampler.state.max_sparsity or 0.0,
                        self.global_step,
                    )

                self.writer.add_scalar("epoch", self.current_epoch, self.global_step)

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

        # Log curriculum metrics for loss-curve-critic analysis
        if (
            self.curriculum_sampler is not None
            and hasattr(self.curriculum_sampler, "state")
        ):
            self.writer.add_scalar(
                "Train/Sparsity",
                self.curriculum_sampler.state.max_sparsity or 0.0,
                self.current_epoch,
            )
            self.writer.add_scalar(
                "Train/SynthRatio",
                self.curriculum_sampler.state.synth_ratio,
                self.current_epoch,
            )

        self._status(
            f"{prefix} loss: {loss:.4g} | MAE: {metrics['mae']:.4g} | RMSE: {metrics['rmse']:.4g} | sMAPE: {metrics['smape']:.4g} | R2: {metrics['r2']:.4g}"
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
