"""Data loader construction for evaluation."""

from __future__ import annotations

import logging
import os
from functools import partial

import torch
import zarr.errors
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from data.dataset_factory import build_datasets
from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import EpiDataset
from models.configs import EpiForecasterConfig
from utils.device import prefetch_enabled, resolve_device

logger = logging.getLogger(__name__)


def _suppress_zarr_warnings(worker_id: int) -> None:
    """Suppress zarr/numcodecs warnings in DataLoader worker processes."""
    import warnings

    warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


def build_loader_from_config(
    config: EpiForecasterConfig,
    *,
    split: str,
    batch_size: int | None = None,
    device: str = "auto",
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader[EpiDataset], torch.Tensor | None]:
    """Build a DataLoader for the given split from the checkpoint config.

    Uses the same factory as training to ensure fair evaluation:
    - Preprocessors are fitted on train split (same as training)
    - Applied consistently to val/test splits (same as training)
    - Train and unused eval splits are discarded after preprocessing

    Args:
        config: The model configuration containing data paths and settings
        split: Which split to load ("val", "test", or "full")
        batch_size: Optional batch size override (default: use config value)
        device: Device to load tensors to ("auto", "cuda", "cpu")
        num_workers: Optional DataLoader worker override
        pin_memory: Optional pin-memory override
        prefetch_factor: Optional prefetch override. Use 0 or None to disable.

    Returns:
        Tuple of (DataLoader, region_embeddings). Region embeddings are pre-loaded
        to the target device to avoid repeated transfers during evaluation.
    """
    split_key = split.lower()
    if split_key not in {"val", "test", "full"}:
        raise ValueError("split must be 'val', 'test', or 'full'")

    # Use factory to get all splits with fitted preprocessors (same as training)
    # This ensures fair evaluation - same preprocessing as training
    splits = build_datasets(config)

    if split_key == "full":
        if isinstance(splits.train, ConcatDataset):
            raise ValueError(
                "split='full' is not supported for curriculum datasets yet."
            )

        train_dataset = splits.train
        val_dataset = splits.val
        test_dataset = splits.test
        all_target_nodes = sorted(
            set(train_dataset.target_nodes)
            | set(val_dataset.target_nodes)
            | set(test_dataset.target_nodes)
        )
        dataset = EpiDataset(
            config=config,
            target_nodes=all_target_nodes,
            context_nodes=all_target_nodes,
            biomarker_preprocessor=val_dataset.biomarker_preprocessor,
            mobility_preprocessor=val_dataset.mobility_preprocessor,
            preloaded_mobility=val_dataset.preloaded_mobility,
            mobility_mask=val_dataset.mobility_mask,
            run_id=splits.real_run_id,
            region_embedding_store=splits.region_embedding_store,
        )
    else:
        # Select requested split, discard others
        dataset = splits.val if split_key == "val" else splits.test
    # splits.train and the other eval split are garbage collected here

    # Worker configuration mirrors training defaults:
    # - validation loader uses val_workers
    # - test loader uses test_workers
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = (
        config.training.val_workers
        if split_key in {"val", "full"}
        else config.training.test_workers
    )
    if num_workers is None:
        resolved_worker_cfg = cfg_workers
    else:
        resolved_worker_cfg = num_workers
    if resolved_worker_cfg == -1:
        resolved_num_workers = max(0, avail_cores)
    else:
        resolved_num_workers = min(max(0, avail_cores), resolved_worker_cfg)

    resolved_batch = batch_size or config.training.eval_batch_size
    resolved_device = resolve_device(device)
    resolved_pin_memory = (
        bool(config.training.pin_memory) if pin_memory is None else bool(pin_memory)
    ) and resolved_device.type == "cuda"
    resolved_prefetch = (
        config.training.prefetch_factor if prefetch_factor is None else prefetch_factor
    )

    # Pre-load region embeddings to device to avoid repeated transfers
    region_embeddings = getattr(dataset, "region_embeddings", None)
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(resolved_device)

    loader = DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=resolved_pin_memory,
        prefetch_factor=resolved_prefetch
        if prefetch_enabled(resolved_prefetch) and resolved_num_workers > 0
        else None,
        collate_fn=partial(
            collate_epiforecaster_batch,
            require_region_index=bool(config.model.type.regions),
        ),
        worker_init_fn=_suppress_zarr_warnings if resolved_num_workers > 0 else None,
    )
    return loader, region_embeddings
