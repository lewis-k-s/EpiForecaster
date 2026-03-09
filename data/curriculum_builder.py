"""
Curriculum dataset builder for combining real and synthetic training data.

This module handles the assembly of curriculum training datasets, including:
- Discovery of real vs synthetic runs
- Sparsity-based run selection
- Region ID mapping between runs
- Preprocessor fitting (separate for real/synthetic to avoid leakage)
- ConcatDataset assembly
"""

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import ConcatDataset

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD
from data.region_embedding_store import RegionEmbeddingStore
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)

REAL_RUN_ID = "real"
MAX_SYNTH_RUNS = 5


@dataclass
class FittedPreprocessors:
    """Fitted preprocessors to share across dataset splits."""

    biomarker: object
    mobility: object
    preloaded_mobility: torch.Tensor
    mobility_mask: torch.Tensor
    shared_sparse_topology: object | None


@dataclass
class CurriculumBuildResult:
    """Result of curriculum dataset building."""

    train_dataset: ConcatDataset
    val_dataset: EpiDataset
    test_dataset: EpiDataset
    real_run_id: str
    synth_run_ids: list[str]
    region_embedding_store: RegionEmbeddingStore | None


def load_sparsity_mapping(dataset_path: Path) -> dict[str, float]:
    """Load run_id -> sparsity mapping from processed zarr dataset.

    Reads the synthetic_sparsity_level variable from the processed dataset.
    Returns dict mapping run_id string to sparsity float (0.0-1.0).

    Args:
        dataset_path: Path to the Zarr dataset

    Returns:
        Dictionary mapping run_id strings to sparsity values.
        Returns empty dict if dataset is unavailable or variable is missing.
    """
    if not dataset_path.exists():
        logger.warning(
            f"Processed dataset not found at {dataset_path}. "
            "Sparsity-based run selection will be disabled."
        )
        return {}

    try:
        ds = xr.open_zarr(str(dataset_path), chunks=None)  # type: ignore[arg-type]
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


def select_runs_by_sparsity(
    synth_runs: list[str],
    sparsity_map: dict[str, float],
    max_runs: int = MAX_SYNTH_RUNS,
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
        return synth_runs[:max_runs]

    target_sparsities = [0.05, 0.20, 0.40, 0.60, 0.80]
    selected = []

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
            best_match = min(candidates, key=lambda x: abs(x[1] - target))
            selected.append(best_match[0])
            logger.info(
                f"Selected run '{best_match[0]}' with sparsity {best_match[1]:.2f} "
                f"for target sparsity {target:.2f}"
            )

    if len(selected) < min(max_runs, len(synth_runs)):
        remaining_needed = min(max_runs, len(synth_runs)) - len(selected)
        available = [r for r in synth_runs if r not in selected]

        if available:
            logger.info(
                f"Adding {remaining_needed} random runs to fill quota (have {len(selected)})"
            )
            remaining = random.sample(available, min(remaining_needed, len(available)))
            selected.extend(remaining)

    return selected


def discover_runs(config: EpiForecasterConfig) -> tuple[str, list[str]]:
    """Discover available runs in the dataset (real vs synthetic).

    Args:
        config: EpiForecasterConfig with dataset path

    Returns:
        Tuple of (real_run_id, list_of_synthetic_run_ids)
    """
    real_run = REAL_RUN_ID
    synth_runs = []

    ds_path = Path(config.data.dataset_path)
    if not ds_path.exists():
        logger.warning(f"Dataset path not found: {ds_path}")
        return real_run, []

    try:
        all_runs = EpiDataset.discover_available_runs(ds_path)
        synth_runs = [r for r in all_runs if r != real_run]
    except Exception as e:
        logger.warning(f"Failed to discover runs from {ds_path}: {e}")
        return real_run, []

    if config.training.curriculum.enabled and len(synth_runs) > MAX_SYNTH_RUNS:
        sparsity_map = load_sparsity_mapping(ds_path)

        if sparsity_map:
            logger.info(
                f"Curriculum mode: selecting {MAX_SYNTH_RUNS} runs for sparsity diversity "
                f"from {len(synth_runs)} available runs"
            )
            synth_runs = select_runs_by_sparsity(
                synth_runs, sparsity_map, MAX_SYNTH_RUNS
            )
        else:
            logger.warning(
                f"Limiting synthetic runs from {len(synth_runs)} to {MAX_SYNTH_RUNS} for memory safety."
            )
            synth_runs = synth_runs[:MAX_SYNTH_RUNS]
    elif len(synth_runs) > MAX_SYNTH_RUNS:
        logger.warning(
            f"Limiting synthetic runs from {len(synth_runs)} to {MAX_SYNTH_RUNS} for memory safety."
        )
        synth_runs = synth_runs[:MAX_SYNTH_RUNS]

    return real_run, synth_runs


def _load_region_ids(dataset_path: Path, run_id: str) -> list[str]:
    """Load region IDs from a dataset for a specific run."""
    aligned_dataset = EpiDataset.load_canonical_dataset(
        dataset_path,
        run_id=run_id,
        run_id_chunk_size=1,
    )
    region_ids = [str(r) for r in aligned_dataset[REGION_COORD].values]
    aligned_dataset.close()
    return region_ids


def _map_region_ids_to_nodes(
    region_ids: list[str],
    dataset_path: Path,
    run_id: str,
) -> list[int]:
    """Map region IDs to node indices for a specific run."""
    target_region_ids = _load_region_ids(dataset_path, run_id)
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


def _fallback_all_nodes(dataset_path: Path, run_id: str) -> list[int]:
    """Get all node indices when region ID mapping fails."""
    aligned_dataset = EpiDataset.load_canonical_dataset(
        dataset_path,
        run_id_chunk_size=1,
        run_id=run_id,
    )
    num_nodes = aligned_dataset[REGION_COORD].size
    aligned_dataset.close()
    return list(range(num_nodes))


def _select_synthetic_scaler_run(
    synth_runs: list[str],
    dataset_path: Path,
) -> str:
    """Select the best synthetic run for fitting scalers.

    Selects the run with LOWEST sparsity (cleanest data) for scaler fitting.
    """
    if not synth_runs:
        raise ValueError("No synthetic runs available for scaler fitting.")

    mapping = load_sparsity_mapping(dataset_path)
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
            selected_run, selected_sparsity = min(candidates, key=lambda x: x[1])
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


def _validate_curriculum_dataset_dimensions(datasets: list[EpiDataset]) -> None:
    """Validate that all datasets in curriculum have consistent dimensions."""
    if len(datasets) < 2:
        return

    ref_ds = datasets[0]
    ref_dims = {
        "cases_output_dim": ref_ds.cases_output_dim,
        "biomarkers_output_dim": ref_ds.biomarkers_output_dim,
        "temporal_covariates_dim": ref_ds.temporal_covariates_dim,
    }
    ref_run_id = getattr(ref_ds, "run_id", "unknown")

    mismatches = []
    for i, ds in enumerate(datasets[1:], start=1):
        ds_run_id = getattr(ds, "run_id", f"dataset_{i}")
        ds_dims = {
            "cases_output_dim": ds.cases_output_dim,
            "biomarkers_output_dim": ds.biomarkers_output_dim,
            "temporal_covariates_dim": ds.temporal_covariates_dim,
        }

        for dim_name, ref_val in ref_dims.items():
            ds_val = ds_dims[dim_name]
            if ds_val != ref_val:
                mismatches.append(
                    f"  {dim_name}: {ref_run_id}={ref_val} vs {ds_run_id}={ds_val}"
                )

    if mismatches:
        raise ValueError(
            f"Curriculum training requires all datasets to have identical "
            f"feature dimensions. Found {len(mismatches)} mismatch(es):\n"
            + "\n".join(mismatches)
            + "\n\nThis usually happens when:"
            "\n  - Real and synthetic data have different biomarker variants"
            "\n  - Real and synthetic data have different temporal_covariates config"
            "\n  - Preprocessing configs are inconsistent between datasets"
            "\n\nFix: Ensure all preprocessing configs include the same "
            "temporal_covariates and biomarker settings."
        )

    logger.info(
        f"Validated {len(datasets)} datasets have consistent dimensions: "
        f"cases={ref_dims['cases_output_dim']}, "
        f"biomarkers={ref_dims['biomarkers_output_dim']}, "
        f"temporal_covariates={ref_dims['temporal_covariates_dim']}"
    )


def build_curriculum_datasets(
    config: EpiForecasterConfig,
    train_nodes: list[int],
    val_nodes: list[int],
    test_nodes: list[int],
    real_run: str,
    synth_runs: list[str],
    region_embedding_store: RegionEmbeddingStore | None = None,
) -> CurriculumBuildResult:
    """Build curriculum training datasets with real + synthetic runs.

    Args:
        config: EpiForecasterConfig
        train_nodes: Node indices for training targets
        val_nodes: Node indices for validation targets
        test_nodes: Node indices for test targets
        real_run: Real run ID (typically "real")
        synth_runs: List of synthetic run IDs

    Returns:
        CurriculumBuildResult with train/val/test datasets
    """
    dataset_path = Path(config.data.dataset_path)

    real_config = config
    if config.data.real_dataset_path:
        real_config = copy.deepcopy(config)
        real_config.data.dataset_path = config.data.real_dataset_path
        split_dataset_path = Path(config.data.real_dataset_path)
    else:
        split_dataset_path = dataset_path

    real_region_ids = _load_region_ids(split_dataset_path, real_run)
    train_region_ids = [real_region_ids[n] for n in train_nodes]
    real_train_ds = EpiDataset(
        config=real_config,
        target_nodes=train_nodes,
        context_nodes=train_nodes,
        biomarker_preprocessor=None,
        mobility_preprocessor=None,
        run_id=real_run,
        region_embedding_store=region_embedding_store,
    )

    fitted_bio_preprocessor = real_train_ds.biomarker_preprocessor
    fitted_mobility_preprocessor = real_train_ds.mobility_preprocessor
    shared_real_mobility = real_train_ds.preloaded_mobility
    shared_real_mobility_mask = real_train_ds.mobility_mask
    shared_real_sparse_topology = real_train_ds.shared_sparse_topology

    synth_scaler_run = _select_synthetic_scaler_run(synth_runs, dataset_path)
    synth_train_nodes = _map_region_ids_to_nodes(
        train_region_ids,
        dataset_path=dataset_path,
        run_id=synth_scaler_run,
    )
    if not synth_train_nodes:
        synth_train_nodes = _fallback_all_nodes(dataset_path, synth_scaler_run)
        logger.warning(
            "Synthetic scaler run '%s' has no overlap with real regions; "
            "fitting synthetic scalers on all nodes instead.",
            synth_scaler_run,
        )

    synth_scaler_ds = EpiDataset(
        config=config,
        target_nodes=synth_train_nodes,
        context_nodes=synth_train_nodes,
        biomarker_preprocessor=None,
        mobility_preprocessor=None,
        run_id=synth_scaler_run,
        region_embedding_store=region_embedding_store,
    )

    synth_bio_preprocessor = synth_scaler_ds.biomarker_preprocessor
    synth_mobility_preprocessor = synth_scaler_ds.mobility_preprocessor

    synth_datasets = []
    for s_run in synth_runs:
        if s_run == synth_scaler_run:
            synth_datasets.append(synth_scaler_ds)
            continue

        mapped_train_nodes = _map_region_ids_to_nodes(
            train_region_ids,
            dataset_path=dataset_path,
            run_id=s_run,
        )
        if not mapped_train_nodes:
            mapped_train_nodes = _fallback_all_nodes(dataset_path, s_run)
            logger.warning(
                "Synthetic run '%s' has no overlap with real regions; "
                "using all nodes for training targets.",
                s_run,
            )

        s_ds = EpiDataset(
            config=config,
            target_nodes=mapped_train_nodes,
            context_nodes=mapped_train_nodes,
            biomarker_preprocessor=synth_bio_preprocessor,
            mobility_preprocessor=synth_mobility_preprocessor,
            run_id=s_run,
            region_embedding_store=region_embedding_store,
        )
        synth_datasets.append(s_ds)

    train_dataset = ConcatDataset([real_train_ds] + synth_datasets)

    val_dataset = EpiDataset(
        config=real_config,
        target_nodes=val_nodes,
        context_nodes=train_nodes + val_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_real_mobility,
        mobility_mask=shared_real_mobility_mask,
        shared_sparse_topology=shared_real_sparse_topology,
        run_id=real_run,
        region_embedding_store=region_embedding_store,
    )

    test_dataset = EpiDataset(
        config=real_config,
        target_nodes=test_nodes,
        context_nodes=train_nodes + val_nodes,
        biomarker_preprocessor=fitted_bio_preprocessor,
        mobility_preprocessor=fitted_mobility_preprocessor,
        preloaded_mobility=shared_real_mobility,
        mobility_mask=shared_real_mobility_mask,
        shared_sparse_topology=shared_real_sparse_topology,
        run_id=real_run,
        region_embedding_store=region_embedding_store,
    )

    real_train_ds.release_shared_sparse_topology()
    val_dataset.release_shared_sparse_topology()
    test_dataset.release_shared_sparse_topology()
    for ds in synth_datasets:
        ds.release_shared_sparse_topology()

    _validate_curriculum_dataset_dimensions([real_train_ds] + synth_datasets)

    return CurriculumBuildResult(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        real_run_id=real_run,
        synth_run_ids=synth_runs,
        region_embedding_store=region_embedding_store,
    )
