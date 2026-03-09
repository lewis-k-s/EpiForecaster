"""
Unified dataset factory for creating train/val/test splits.

This module provides a single entry point for dataset creation that dispatches
to the appropriate strategy (temporal splits, node splits, or curriculum training).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from data.curriculum_builder import (
    build_curriculum_datasets,
    discover_runs,
)
from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from data.region_embedding_store import RegionEmbeddingStore
from models.configs import EpiForecasterConfig
from utils.temporal import (
    format_date_range,
    get_temporal_boundaries,
    validate_temporal_range,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    """Train/val/test dataset triple with optional curriculum metadata."""

    train: EpiDataset | ConcatDataset
    val: EpiDataset
    test: EpiDataset
    real_run_id: str | None = None
    synth_run_ids: list[str] | None = None
    region_embedding_store: RegionEmbeddingStore | None = None


def _build_region_embedding_store(
    config: EpiForecasterConfig,
) -> RegionEmbeddingStore | None:
    if not config.data.region2vec_path:
        return None
    return RegionEmbeddingStore.from_weights(
        config.data.region2vec_path,
        expected_dim=config.model.region_embedding_dim,
    )


def split_nodes_by_ratio(
    config: EpiForecasterConfig,
    dataset_path: Path | None = None,
    run_id: str | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset nodes into train/val/test sets.

    Args:
        config: EpiForecasterConfig with split ratios and seed
        dataset_path: Optional path override (defaults to config.data.dataset_path)
        run_id: Optional run_id override (defaults to config.data.run_id)

    Returns:
        Tuple of (train_nodes, val_nodes, test_nodes)
    """
    train_split = 1 - config.training.val_split - config.training.test_split

    target_path = dataset_path or Path(config.data.dataset_path)
    effective_run_id = run_id or config.data.run_id
    if not effective_run_id:
        raise ValueError(
            "run_id must be provided either as argument or in config.data.run_id"
        )

    aligned_dataset = EpiDataset.load_canonical_dataset(
        target_path,
        run_id=effective_run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    N = aligned_dataset[REGION_COORD].size
    all_nodes = np.arange(N)

    valid_mask = None
    if config.data.use_valid_targets:
        run_id_for_valid = run_id or config.data.run_id
        valid_mask = EpiDataset.get_valid_nodes(
            dataset_path=target_path,
            run_id=run_id_for_valid,
        )
        logger.info(f"Using valid_targets filter: {valid_mask.sum()} valid regions")
    else:
        logger.info(f"Total regions: {N}")

    if valid_mask is not None:
        all_nodes = all_nodes[valid_mask]
        N = len(all_nodes)

    rng = np.random.default_rng(config.training.seed)
    rng.shuffle(all_nodes)
    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * config.training.val_split)
    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]

    assert len(train_nodes) + len(val_nodes) + len(test_nodes) == len(all_nodes), (
        "Dataset split is not correct"
    )

    aligned_dataset.close()
    return list(train_nodes), list(val_nodes), list(test_nodes)


def _build_standard_splits(
    config: EpiForecasterConfig,
    train_nodes: list[int],
    val_nodes: list[int],
    test_nodes: list[int],
    region_embedding_store: RegionEmbeddingStore | None,
) -> DatasetSplits:
    """Build standard (non-curriculum) train/val/test splits with shared preprocessors."""
    train_dataset = EpiDataset(
        config=config,
        target_nodes=train_nodes,
        context_nodes=train_nodes,
        biomarker_preprocessor=None,
        mobility_preprocessor=None,
        region_embedding_store=region_embedding_store,
    )

    fitted_bio_preprocessor = train_dataset.biomarker_preprocessor
    fitted_mobility_preprocessor = train_dataset.mobility_preprocessor
    shared_mobility = train_dataset.preloaded_mobility
    shared_mobility_mask = train_dataset.mobility_mask
    shared_sparse_topology = train_dataset.shared_sparse_topology

    val_dataset = EpiDataset(
        config=config,
        target_nodes=val_nodes,
        context_nodes=train_nodes + val_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_mobility,
        mobility_mask=shared_mobility_mask,
        shared_sparse_topology=shared_sparse_topology,
        region_embedding_store=region_embedding_store,
    )

    test_dataset = EpiDataset(
        config=config,
        target_nodes=test_nodes,
        context_nodes=train_nodes + val_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_mobility,
        mobility_mask=shared_mobility_mask,
        shared_sparse_topology=shared_sparse_topology,
        region_embedding_store=region_embedding_store,
    )

    train_dataset.release_shared_sparse_topology()
    val_dataset.release_shared_sparse_topology()
    test_dataset.release_shared_sparse_topology()

    return DatasetSplits(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        region_embedding_store=region_embedding_store,
    )


def _build_temporal_splits(
    config: EpiForecasterConfig,
    train_end_date: str,
    val_end_date: str,
    test_end_date: str | None,
    region_embedding_store: RegionEmbeddingStore | None,
) -> DatasetSplits:
    """Build temporal train/val/test splits with shared preprocessors."""
    if not config.data.run_id:
        raise ValueError("run_id must be specified in config for temporal splits")

    aligned_dataset = EpiDataset.load_canonical_dataset(
        Path(config.data.dataset_path),
        run_id=config.data.run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )

    num_nodes = aligned_dataset[REGION_COORD].size
    all_nodes = list(range(num_nodes))

    if config.data.use_valid_targets and "valid_targets" in aligned_dataset:
        valid_targets = aligned_dataset.valid_targets
        if "run_id" in valid_targets.dims:
            valid_targets = valid_targets.any(dim="run_id")

        valid_mask = valid_targets.values.astype(bool)
        all_nodes = [i for i in all_nodes if valid_mask[i]]
        logger.info(
            "Using valid_targets filter: %d/%d training regions",
            len(all_nodes),
            num_nodes,
        )

    train_start, train_end, val_end, test_end = get_temporal_boundaries(
        aligned_dataset,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        test_end_date=test_end_date,
    )

    input_window = config.model.input_window_length
    forecast_horizon = config.model.forecast_horizon
    total_time_steps = len(aligned_dataset[TEMPORAL_COORD])

    for name, time_range in [
        ("train", (train_start, train_end)),
        ("val", (train_end, val_end)),
        ("test", (val_end, test_end)),
    ]:
        try:
            validate_temporal_range(
                time_range, input_window, forecast_horizon, total_time_steps
            )
        except ValueError as exc:
            raise ValueError(
                f"{name.upper()} split temporal range invalid: {exc}"
            ) from exc

    logger.info("Temporal split boundaries:")
    logger.info(
        "  TRAIN: %s", format_date_range(aligned_dataset, (train_start, train_end))
    )
    logger.info(
        "  VAL:   %s", format_date_range(aligned_dataset, (train_end, val_end))
    )
    logger.info(
        "  TEST:  %s", format_date_range(aligned_dataset, (val_end, test_end))
    )

    train_dataset = EpiDataset(
        config=config,
        target_nodes=all_nodes,
        context_nodes=all_nodes,
        biomarker_preprocessor=None,
        mobility_preprocessor=None,
        time_range=(train_start, train_end),
        region_embedding_store=region_embedding_store,
    )

    fitted_bio_preprocessor = train_dataset.biomarker_preprocessor
    fitted_mobility_preprocessor = train_dataset.mobility_preprocessor
    shared_mobility = train_dataset.preloaded_mobility
    shared_mobility_mask = train_dataset.mobility_mask

    val_dataset = EpiDataset(
        config=config,
        target_nodes=all_nodes,
        context_nodes=all_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_mobility,
        mobility_mask=shared_mobility_mask,
        time_range=(train_end, val_end),
        region_embedding_store=region_embedding_store,
    )

    test_dataset = EpiDataset(
        config=config,
        target_nodes=all_nodes,
        context_nodes=all_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_mobility,
        mobility_mask=shared_mobility_mask,
        time_range=(val_end, test_end),
        region_embedding_store=region_embedding_store,
    )

    return DatasetSplits(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        region_embedding_store=region_embedding_store,
    )


def build_datasets(config: EpiForecasterConfig) -> DatasetSplits:
    """Build train/val/test datasets based on configuration.

    Dispatches to the appropriate strategy:
    - "time": Temporal splits (all nodes, different time ranges)
    - "nodes" + curriculum: Node splits with synthetic data augmentation
    - "nodes" + standard: Node splits with preprocessor sharing

    Args:
        config: EpiForecasterConfig with dataset and training settings

    Returns:
        DatasetSplits with train/val/test datasets and optional curriculum metadata
    """
    region_embedding_store = _build_region_embedding_store(config)

    if config.training.split_strategy == "time":
        train_end: str = config.training.train_end_date or ""
        val_end: str = config.training.val_end_date or ""
        test_end: str | None = config.training.test_end_date

        return _build_temporal_splits(
            config=config,
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end,
            region_embedding_store=region_embedding_store,
        )

    real_run_for_split: str | None = None
    split_dataset_path: Path | None = None

    if config.training.curriculum.enabled:
        real_run, synth_runs = discover_runs(config)
        logger.info(
            f"Curriculum enabled. Found runs: Real='{real_run}', Synth={synth_runs}"
        )
        real_run_for_split = real_run
        split_dataset_path = (
            Path(config.data.real_dataset_path)
            if config.data.real_dataset_path
            else Path(config.data.dataset_path)
        )

        train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(
            config=config,
            dataset_path=split_dataset_path,
            run_id=real_run_for_split,
        )

        result = build_curriculum_datasets(
            config=config,
            train_nodes=list(train_nodes),
            val_nodes=list(val_nodes),
            test_nodes=list(test_nodes),
            real_run=real_run,
            synth_runs=synth_runs,
            region_embedding_store=region_embedding_store,
        )

        return DatasetSplits(
            train=result.train_dataset,
            val=result.val_dataset,
            test=result.test_dataset,
            real_run_id=result.real_run_id,
            synth_run_ids=result.synth_run_ids,
            region_embedding_store=result.region_embedding_store,
        )
    else:
        if not config.data.run_id:
            raise ValueError("run_id must be specified in config for node-based splits")
        train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config=config)

        return _build_standard_splits(
            config=config,
            train_nodes=list(train_nodes),
            val_nodes=list(val_nodes),
            test_nodes=list(test_nodes),
            region_embedding_store=region_embedding_store,
        )
