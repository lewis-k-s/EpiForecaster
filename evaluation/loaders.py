"""Data loading utilities for evaluation.

This module provides functions for loading models from checkpoints and
building data loaders for evaluation.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import zarr.errors
from torch.utils.data import DataLoader

from data.dataset_factory import build_datasets
from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import EpiDataset
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from utils.device import prefetch_enabled, resolve_device

logger = logging.getLogger(__name__)


def _suppress_zarr_warnings(worker_id: int) -> None:
    """Suppress zarr/numcodecs warnings in DataLoader worker processes."""
    import warnings

    warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> tuple[EpiForecaster, EpiForecasterConfig, dict[str, Any]]:
    """Load an EpiForecaster model + config from a saved trainer checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on
        overrides: Optional list of dotted-key config overrides applied before model creation

    Returns:
        Tuple of (model, config, checkpoint_dict)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is missing required keys or has invalid config
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate required keys
    required_keys = ["model_state_dict", "config"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            f"This checkpoint may be from an incompatible version or corrupted."
        )

    config_raw = checkpoint["config"]

    # Handle backwards compatibility: old checkpoints have pickled EpiForecasterConfig,
    # new checkpoints have plain dicts (robust to config class changes)
    if isinstance(config_raw, dict):
        # New format: plain dict (YAML-compatible)
        config = EpiForecasterConfig.from_dict(config_raw)
    elif isinstance(config_raw, EpiForecasterConfig):
        # Old format: pickled EpiForecasterConfig (for backwards compatibility)
        config = config_raw
    else:
        raise ValueError(
            f"Checkpoint config has invalid type: {type(config_raw).__name__}. "
            f"Expected EpiForecasterConfig or dict. "
            f"Please check that the checkpoint was created with a compatible version."
        )

    # Apply overrides BEFORE model creation (for architecture-affecting params)
    if overrides:
        config = EpiForecasterConfig.apply_overrides(config, overrides)
        logger.info(f"Applied {len(overrides)} config overrides before model creation")

    model = EpiForecaster(
        variant_type=config.model.type,
        temporal_input_dim=config.model.cases_dim,
        biomarkers_dim=config.model.biomarkers_dim,
        region_embedding_dim=config.model.region_embedding_dim,
        mobility_embedding_dim=config.model.mobility_embedding_dim,
        gnn_depth=config.model.gnn_depth,
        sequence_length=config.model.input_window_length,
        forecast_horizon=config.model.forecast_horizon,
        use_population=config.model.use_population,
        population_dim=config.model.population_dim,
        device=resolve_device(device),
        gnn_module=config.model.gnn_module,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        head_d_model=config.model.head_d_model,
        head_n_heads=config.model.head_n_heads,
        head_num_layers=config.model.head_num_layers,
        head_dropout=config.model.head_dropout,
        sir_physics=config.model.sir_physics,
        observation_heads=config.model.observation_heads,
        temporal_covariates_dim=config.model.temporal_covariates_dim,
    )
    # Strip _orig_mod. prefix from compiled model checkpoints
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(resolve_device(device))
    return model, config, checkpoint


def build_loader_from_config(
    config: EpiForecasterConfig,
    *,
    split: str,
    batch_size: int | None = None,
    device: str = "auto",
) -> tuple[DataLoader[EpiDataset], torch.Tensor | None]:
    """Build a DataLoader for the given split from the checkpoint config.

    Uses the same factory as training to ensure fair evaluation:
    - Preprocessors are fitted on train split (same as training)
    - Applied consistently to val/test splits (same as training)
    - Train and unused eval splits are discarded after preprocessing

    Args:
        config: The model configuration containing data paths and settings
        split: Which split to load ("val" or "test")
        batch_size: Optional batch size override (default: use config value)
        device: Device to load tensors to ("auto", "cuda", "cpu")

    Returns:
        Tuple of (DataLoader, region_embeddings). Region embeddings are pre-loaded
        to the target device to avoid repeated transfers during evaluation.
    """
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    # Use factory to get all splits with fitted preprocessors (same as training)
    # This ensures fair evaluation - same preprocessing as training
    splits = build_datasets(config)

    # Select requested split, discard others
    dataset = splits.val if split_key == "val" else splits.test
    # splits.train and the other eval split are garbage collected here

    # Worker configuration mirrors training defaults:
    # - validation loader uses val_workers
    # - test loader uses test_workers
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = (
        config.training.val_workers
        if split_key == "val"
        else getattr(config.training, "test_workers", 0)
    )
    if cfg_workers == -1:
        num_workers = max(0, avail_cores)
    else:
        num_workers = min(max(0, avail_cores), cfg_workers)

    resolved_batch = batch_size or config.training.batch_size
    resolved_device = resolve_device(device)
    pin_memory = bool(config.training.pin_memory) and resolved_device.type == "cuda"

    # Pre-load region embeddings to device to avoid repeated transfers
    region_embeddings = getattr(dataset, "region_embeddings", None)
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(resolved_device)

    loader = DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=config.training.prefetch_factor
        if prefetch_enabled(config.training.prefetch_factor) and num_workers > 0
        else None,
        collate_fn=partial(
            collate_epiforecaster_batch,
            require_region_index=bool(config.model.type.regions),
        ),
        worker_init_fn=_suppress_zarr_warnings if num_workers > 0 else None,
    )
    return loader, region_embeddings
