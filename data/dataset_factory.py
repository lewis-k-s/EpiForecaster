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


@dataclass(frozen=True)
class NodeSplitMetadata:
    """Metadata used to build node-based train/val/test splits."""

    total_nodes: int
    valid_nodes: np.ndarray
    population_bins: np.ndarray | None = None
    wastewater_source: np.ndarray | None = None
    wastewater_source_name: str | None = None


def _build_region_embedding_store(
    config: EpiForecasterConfig,
) -> RegionEmbeddingStore | None:
    if not config.data.region2vec_path:
        return None
    return RegionEmbeddingStore.from_weights(
        config.data.region2vec_path,
        expected_dim=config.model.region_embedding_dim,
    )


def _resolve_split_sizes(
    total_nodes: int,
    val_split: float,
    test_split: float,
) -> tuple[int, int, int]:
    train_split = 1.0 - val_split - test_split
    n_train = int(total_nodes * train_split)
    n_val = int(total_nodes * val_split)
    n_test = total_nodes - n_train - n_val
    return n_train, n_val, n_test


def _load_valid_nodes(
    *,
    config: EpiForecasterConfig,
    target_path: Path,
    total_nodes: int,
    effective_run_id: str,
) -> np.ndarray:
    if not config.data.use_valid_targets:
        logger.info("Total regions: %d", total_nodes)
        return np.ones(total_nodes, dtype=bool)

    valid_mask = EpiDataset.get_valid_nodes(
        dataset_path=target_path,
        run_id=effective_run_id,
    )
    logger.info("Using valid_targets filter: %d valid regions", int(valid_mask.sum()))
    return valid_mask


def _compute_population_bins(
    population: np.ndarray,
    *,
    num_bins: int,
) -> np.ndarray:
    if population.size == 0 or num_bins <= 1:
        return np.zeros(population.shape[0], dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(population, quantiles)
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = np.nextafter(edges[idx - 1], np.inf)
    return np.digitize(population, edges[1:-1], right=True).astype(np.int64)


def _resolve_wastewater_source(
    aligned_dataset,
    total_nodes: int,
) -> tuple[np.ndarray, str]:
    if "edar_has_source" in aligned_dataset:
        wastewater_source = np.asarray(
            aligned_dataset["edar_has_source"].values
        ).reshape(-1)
        if wastewater_source.size != total_nodes:
            raise ValueError(
                "edar_has_source must match region dimension for stratified node splits"
            )
        return wastewater_source.astype(bool), "edar_has_source"

    if "biomarker_data_start" in aligned_dataset:
        biomarker_data_start = np.asarray(
            aligned_dataset["biomarker_data_start"].values
        ).reshape(-1)
        if biomarker_data_start.size != total_nodes:
            raise ValueError(
                "biomarker_data_start must match region dimension for stratified node splits"
            )
        return biomarker_data_start >= 0, "biomarker_data_start"

    logger.warning(
        "No wastewater availability metadata found; stratified node split will "
        "balance only population bins."
    )
    return np.zeros(total_nodes, dtype=bool), "none"


def _load_node_split_metadata(
    *,
    config: EpiForecasterConfig,
    dataset_path: Path,
    run_id: str,
) -> NodeSplitMetadata:
    aligned_dataset = EpiDataset.load_canonical_dataset(
        dataset_path,
        run_id=run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    try:
        total_nodes = int(aligned_dataset[REGION_COORD].size)
        valid_mask = _load_valid_nodes(
            config=config,
            target_path=dataset_path,
            total_nodes=total_nodes,
            effective_run_id=run_id,
        )
        valid_nodes = np.arange(total_nodes, dtype=np.int64)[valid_mask]

        if config.training.node_split_strategy == "stratified":
            if "population" not in aligned_dataset:
                raise ValueError(
                    "population is required for stratified node splits but was not found"
                )
            population = np.asarray(
                aligned_dataset["population"].values, dtype=np.float64
            )
            if population.shape != (total_nodes,):
                raise ValueError(
                    "population must be a 1D region-level array for stratified node splits"
                )

            population_bins = np.full(total_nodes, -1, dtype=np.int64)
            population_bins[valid_nodes] = _compute_population_bins(
                population[valid_nodes],
                num_bins=config.training.node_split_population_bins,
            )
            wastewater_source, wastewater_source_name = _resolve_wastewater_source(
                aligned_dataset,
                total_nodes,
            )
        else:
            population_bins = None
            wastewater_source = None
            wastewater_source_name = None

        return NodeSplitMetadata(
            total_nodes=total_nodes,
            valid_nodes=valid_nodes,
            population_bins=population_bins,
            wastewater_source=wastewater_source,
            wastewater_source_name=wastewater_source_name,
        )
    finally:
        aligned_dataset.close()


def _shuffle_nodes(
    nodes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.array(nodes, copy=True)
    rng.shuffle(shuffled)
    return shuffled


def _split_nodes_random(
    *,
    valid_nodes: np.ndarray,
    rng: np.random.Generator,
    val_split: float,
    test_split: float,
) -> tuple[list[int], list[int], list[int]]:
    all_nodes = _shuffle_nodes(valid_nodes, rng)
    n_train, n_val, _n_test = _resolve_split_sizes(
        len(all_nodes),
        val_split=val_split,
        test_split=test_split,
    )
    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]
    return list(train_nodes), list(val_nodes), list(test_nodes)


def _allocate_split_across_strata(
    *,
    stratum_sizes: dict[tuple[int, int], int],
    target_size: int,
) -> dict[tuple[int, int], int]:
    if target_size <= 0:
        return {key: 0 for key in stratum_sizes}

    total_size = sum(stratum_sizes.values())
    if target_size > total_size:
        raise ValueError("target split size exceeds available stratum capacity")

    quotas = {
        key: (size * target_size / total_size) if total_size > 0 else 0.0
        for key, size in stratum_sizes.items()
    }
    allocation = {
        key: min(stratum_sizes[key], int(np.floor(quota)))
        for key, quota in quotas.items()
    }
    remaining = target_size - sum(allocation.values())

    if remaining > 0:
        ranked = sorted(
            stratum_sizes,
            key=lambda key: (
                quotas[key] - allocation[key],
                stratum_sizes[key],
                -key[0],
                -key[1],
            ),
            reverse=True,
        )
        for key in ranked:
            if remaining == 0:
                break
            if allocation[key] >= stratum_sizes[key]:
                continue
            allocation[key] += 1
            remaining -= 1

    return allocation


def _split_nodes_stratified(
    *,
    metadata: NodeSplitMetadata,
    rng: np.random.Generator,
    val_split: float,
    test_split: float,
) -> tuple[list[int], list[int], list[int]]:
    if metadata.population_bins is None:
        raise ValueError(
            "population_bins is required for stratified splits but was not loaded"
        )
    if metadata.wastewater_source is None:
        raise ValueError(
            "wastewater_source is required for stratified splits but was not loaded"
        )

    valid_nodes = metadata.valid_nodes
    n_train, n_val, n_test = _resolve_split_sizes(
        len(valid_nodes),
        val_split=val_split,
        test_split=test_split,
    )

    shuffled_nodes = _shuffle_nodes(valid_nodes, rng)
    strata: dict[tuple[int, int], list[int]] = {}
    for node in shuffled_nodes:
        key = (
            int(metadata.population_bins[node]),
            int(metadata.wastewater_source[node]),
        )
        strata.setdefault(key, []).append(int(node))

    train_nodes: list[int] = []
    val_nodes: list[int] = []
    test_nodes: list[int] = []

    remaining = {key: list(nodes) for key, nodes in strata.items()}
    remaining_sizes = {key: len(nodes) for key, nodes in remaining.items()}

    train_alloc = _allocate_split_across_strata(
        stratum_sizes=remaining_sizes,
        target_size=n_train,
    )
    for key, count in train_alloc.items():
        train_nodes.extend(remaining[key][:count])
        remaining[key] = remaining[key][count:]

    remaining_sizes = {key: len(nodes) for key, nodes in remaining.items()}
    val_alloc = _allocate_split_across_strata(
        stratum_sizes=remaining_sizes,
        target_size=n_val,
    )
    for key, count in val_alloc.items():
        val_nodes.extend(remaining[key][:count])
        remaining[key] = remaining[key][count:]

    for nodes in remaining.values():
        test_nodes.extend(nodes)

    if (
        len(train_nodes) != n_train
        or len(val_nodes) != n_val
        or len(test_nodes) != n_test
    ):
        raise AssertionError("Stratified node split did not match expected split sizes")

    train_nodes = list(_shuffle_nodes(np.asarray(train_nodes, dtype=np.int64), rng))
    val_nodes = list(_shuffle_nodes(np.asarray(val_nodes, dtype=np.int64), rng))
    test_nodes = list(_shuffle_nodes(np.asarray(test_nodes, dtype=np.int64), rng))
    return train_nodes, val_nodes, test_nodes


def _log_split_summary(
    *,
    name: str,
    nodes: list[int],
    metadata: NodeSplitMetadata,
) -> None:
    if not nodes:
        logger.info("%s split: n=0", name)
        return

    if metadata.wastewater_source is None or metadata.population_bins is None:
        logger.info("%s split: n=%d", name, len(nodes))
        return

    split_nodes = np.asarray(nodes, dtype=np.int64)
    wastewater_count = int(metadata.wastewater_source[split_nodes].sum())
    population_bins = metadata.population_bins[split_nodes]
    population_counts = np.bincount(population_bins, minlength=1).tolist()
    logger.info(
        "%s split: n=%d | %s=%d (%.3f) | population_bins=%s",
        name,
        len(nodes),
        metadata.wastewater_source_name or "unknown",
        wastewater_count,
        wastewater_count / len(nodes),
        population_counts,
    )


def _log_node_split_diagnostics(
    *,
    strategy: str,
    metadata: NodeSplitMetadata,
    train_nodes: list[int],
    val_nodes: list[int],
    test_nodes: list[int],
) -> None:
    logger.info(
        "Node split strategy=%s | valid_nodes=%d/%d | wastewater_source=%s",
        strategy,
        len(metadata.valid_nodes),
        metadata.total_nodes,
        metadata.wastewater_source_name or "n/a",
    )
    _log_split_summary(name="Train", nodes=train_nodes, metadata=metadata)
    _log_split_summary(name="Val", nodes=val_nodes, metadata=metadata)
    _log_split_summary(name="Test", nodes=test_nodes, metadata=metadata)


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
    target_path = dataset_path or Path(config.data.dataset_path)
    effective_run_id = run_id or config.data.run_id
    if not effective_run_id:
        raise ValueError(
            "run_id must be provided either as argument or in config.data.run_id"
        )

    rng = np.random.default_rng(config.training.seed)
    metadata = _load_node_split_metadata(
        config=config,
        dataset_path=target_path,
        run_id=effective_run_id,
    )

    if config.training.node_split_strategy == "stratified":
        train_nodes, val_nodes, test_nodes = _split_nodes_stratified(
            metadata=metadata,
            rng=rng,
            val_split=config.training.val_split,
            test_split=config.training.test_split,
        )
    else:
        train_nodes, val_nodes, test_nodes = _split_nodes_random(
            valid_nodes=metadata.valid_nodes,
            rng=rng,
            val_split=config.training.val_split,
            test_split=config.training.test_split,
        )

    assert len(train_nodes) + len(val_nodes) + len(test_nodes) == len(
        metadata.valid_nodes
    ), "Dataset split is not correct"

    _log_node_split_diagnostics(
        strategy=config.training.node_split_strategy,
        metadata=metadata,
        train_nodes=train_nodes,
        val_nodes=val_nodes,
        test_nodes=test_nodes,
    )
    return train_nodes, val_nodes, test_nodes


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
    logger.info("  VAL:   %s", format_date_range(aligned_dataset, (train_end, val_end)))
    logger.info("  TEST:  %s", format_date_range(aligned_dataset, (val_end, test_end)))

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
