"""
DataLoader factory for creating train/val/test dataloaders with samplers.

This module handles:
- Worker count computation (auto, -1, explicit)
- Pin memory, persistent workers, prefetch configuration
- Sampler selection (curriculum vs shuffled vs sequential)
- Multiprocessing context selection
"""

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import EpiDataset
from data.samplers import EpidemicCurriculumSampler, ShuffledBatchSampler
from models.configs import CurriculumConfig, ModelVariant, TrainingParams
from utils.platform import select_multiprocessing_context

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderBundle:
    """Train/val/test dataloader triple with sampler metadata."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    curriculum_sampler: EpidemicCurriculumSampler | None = None
    multiprocessing_context: str | None = None


def _compute_num_workers(cfg_workers: int, avail_cores: int) -> int:
    """Compute actual number of workers from config value.

    Args:
        cfg_workers: Config value (-1 for auto, 0+ for explicit)
        avail_cores: Available CPU cores

    Returns:
        Actual worker count to use
    """
    if cfg_workers == -1:
        return avail_cores
    return min(avail_cores, cfg_workers)


def build_dataloaders(
    train_dataset: EpiDataset | ConcatDataset,
    val_dataset: EpiDataset,
    test_dataset: EpiDataset,
    training_config: TrainingParams,
    model_type_config: ModelVariant,
    curriculum_config: CurriculumConfig | None = None,
    real_run_id: str = "real",
    device_hint: str = "cpu",
    seed: int | None = None,
) -> DataLoaderBundle:
    """Create DataLoaders with hardware-aware worker configuration.

    Args:
        train_dataset: Training dataset (EpiDataset or ConcatDataset for curriculum)
        val_dataset: Validation dataset
        test_dataset: Test dataset
        training_config: Training configuration
        model_type_config: Model type configuration (for region index requirement)
        curriculum_config: Optional curriculum configuration
        real_run_id: Real run ID for curriculum sampler
        device_hint: Device hint for pin memory and multiprocessing decisions
        seed: Random seed for shuffling

    Returns:
        DataLoaderBundle with train/val/test loaders and optional curriculum sampler
    """
    all_num_workers_zero = (
        training_config.num_workers == 0 and training_config.val_workers == 0
    )
    mp_context = select_multiprocessing_context(
        device_hint, all_num_workers_zero=all_num_workers_zero
    )

    pin_memory = training_config.pin_memory and device_hint == "cuda"

    avail_cores = max(0, (os.cpu_count() or 1) - 1)
    num_workers = _compute_num_workers(training_config.num_workers, avail_cores)
    val_num_workers = _compute_num_workers(training_config.val_workers, avail_cores)
    test_num_workers = _compute_num_workers(
        getattr(training_config, "test_workers", 0), avail_cores
    )

    persistent_workers = training_config.persistent_workers and num_workers > 0

    shared_collate = partial(
        collate_epiforecaster_batch,
        require_region_index=bool(model_type_config.regions),
    )

    curriculum_sampler: EpidemicCurriculumSampler | None = None

    train_loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        train_loader_kwargs["multiprocessing_context"] = mp_context

    if (
        curriculum_config is not None
        and curriculum_config.enabled
        and isinstance(train_dataset, ConcatDataset)
    ):
        logger.info("Creating EpidemicCurriculumSampler...")
        curriculum_sampler = EpidemicCurriculumSampler(
            dataset=train_dataset,
            batch_size=training_config.batch_size,
            config=curriculum_config,
            drop_last=True,
            real_run_id=real_run_id,
        )
        train_loader_kwargs["batch_sampler"] = curriculum_sampler
        train_loader_kwargs["collate_fn"] = shared_collate
    else:
        if training_config.shuffle_train_batches:
            train_loader_kwargs["batch_sampler"] = ShuffledBatchSampler(
                dataset_size=len(train_dataset),
                batch_size=training_config.batch_size,
                drop_last=True,
                seed=seed,
            )
        else:
            train_loader_kwargs["batch_size"] = training_config.batch_size
            train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["drop_last"] = True
        train_loader_kwargs["collate_fn"] = shared_collate

    if persistent_workers:
        train_loader_kwargs["persistent_workers"] = True
    if training_config.prefetch_factor is not None and num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = training_config.prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)

    val_persistent_workers = training_config.persistent_workers and val_num_workers > 0
    val_loader_kwargs: dict[str, Any] = {
        "dataset": val_dataset,
        "batch_size": training_config.batch_size,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": pin_memory,
        "collate_fn": shared_collate,
    }
    if val_num_workers > 0:
        val_loader_kwargs["multiprocessing_context"] = mp_context
    if val_persistent_workers:
        val_loader_kwargs["persistent_workers"] = True
    if training_config.prefetch_factor is not None and val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = training_config.prefetch_factor
    val_loader = DataLoader(**val_loader_kwargs)

    test_persistent_workers = (
        training_config.persistent_workers and test_num_workers > 0
    )
    test_loader_kwargs: dict[str, Any] = {
        "dataset": test_dataset,
        "batch_size": training_config.batch_size,
        "shuffle": False,
        "num_workers": test_num_workers,
        "pin_memory": pin_memory,
        "collate_fn": shared_collate,
    }
    if test_num_workers > 0:
        test_loader_kwargs["multiprocessing_context"] = mp_context
    if test_persistent_workers:
        test_loader_kwargs["persistent_workers"] = True
    if training_config.prefetch_factor is not None and test_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = training_config.prefetch_factor
    test_loader = DataLoader(**test_loader_kwargs)

    return DataLoaderBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        curriculum_sampler=curriculum_sampler,
        multiprocessing_context=mp_context,
    )


def should_prestart_dataloader_workers(
    multiprocessing_context: str | None,
    device_hint: str,
) -> bool:
    """Check if dataloader workers should be prestarted before CUDA init.

    Args:
        multiprocessing_context: The multiprocessing context being used
        device_hint: The device hint (cuda, cpu, etc.)

    Returns:
        True if workers should be prestarted
    """
    if multiprocessing_context != "fork":
        return False
    if device_hint != "cuda":
        return False
    return True
