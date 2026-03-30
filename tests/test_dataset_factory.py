from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data.dataset_factory import split_nodes_by_ratio
from models.configs import (
    CurriculumConfig,
    DataConfig,
    EpiForecasterConfig,
    LossConfig,
    ModelConfig,
    ModelVariant,
    ObservationHeadConfig,
    OutputConfig,
    SIRPhysicsConfig,
    TrainingParams,
)


def _build_config(
    dataset_path: Path,
    *,
    node_split_strategy: str,
    node_split_population_bins: int = 5,
    node_split_biomarker_bins: int = 3,
    use_valid_targets: bool = True,
    seed: int = 42,
    crossval_enabled: bool = False,
    crossval_num_folds: int = 5,
    crossval_fold_index: int = 0,
) -> EpiForecasterConfig:
    return EpiForecasterConfig(
        model=ModelConfig(
            type=ModelVariant(cases=True),
            mobility_embedding_dim=8,
            region_embedding_dim=8,
            input_window_length=14,
            forecast_horizon=7,
            max_neighbors=5,
            sir_physics=SIRPhysicsConfig(),
            observation_heads=ObservationHeadConfig(),
        ),
        data=DataConfig(
            dataset_path=str(dataset_path),
            run_id="real",
            run_id_chunk_size=1,
            use_valid_targets=use_valid_targets,
        ),
        training=TrainingParams(
            epochs=1,
            batch_size=4,
            seed=seed,
            val_split=0.2,
            test_split=0.1,
            split_strategy="node",
            node_split_strategy=node_split_strategy,
            node_split_population_bins=node_split_population_bins,
            node_split_biomarker_bins=node_split_biomarker_bins,
            crossval_enabled=crossval_enabled,
            crossval_num_folds=crossval_num_folds,
            crossval_fold_index=crossval_fold_index,
            loss=LossConfig(name="joint_inference"),
            curriculum=CurriculumConfig(enabled=False),
            enable_mixed_precision=False,
        ),
        output=OutputConfig(log_dir="test_outputs", experiment_name="test_exp"),
    )


def _write_node_split_dataset(
    tmp_path: Path,
    *,
    include_edar_has_source: bool = True,
    invalid_nodes: set[int] | None = None,
) -> tuple[Path, dict[int, tuple[int, int]], np.ndarray]:
    invalid_nodes = invalid_nodes or set()
    per_stratum = 10
    population_levels = 5
    wastewater_levels = 2
    total_nodes = per_stratum * population_levels * wastewater_levels
    date_count = 4

    population = np.zeros(total_nodes, dtype=np.int32)
    edar_has_source = np.zeros(total_nodes, dtype=np.int32)
    biomarker_data_start = np.full(total_nodes, -1, dtype=np.int16)
    valid_targets = np.ones(total_nodes, dtype=bool)
    strata_by_node: dict[int, tuple[int, int]] = {}
    biomarker_fraction_by_node: dict[int, int] = {}

    node_idx = 0
    for pop_bin in range(population_levels):
        for has_wastewater in range(wastewater_levels):
            for offset in range(per_stratum):
                population[node_idx] = pop_bin * 1000 + has_wastewater * 100 + offset
                edar_has_source[node_idx] = has_wastewater
                biomarker_data_start[node_idx] = 0 if has_wastewater else -1
                valid_targets[node_idx] = node_idx not in invalid_nodes
                strata_by_node[node_idx] = (pop_bin, has_wastewater)
                biomarker_fraction_by_node[node_idx] = offset % 5
                node_idx += 1

    cases = np.ones((1, date_count, total_nodes), dtype=np.float32)
    n1_mask = np.zeros((1, date_count, total_nodes), dtype=bool)
    n2_mask = np.zeros((1, date_count, total_nodes), dtype=bool)
    for node_idx in range(total_nodes):
        if biomarker_data_start[node_idx] < 0:
            continue
        observed_steps = biomarker_fraction_by_node[node_idx]
        if observed_steps > 0:
            n1_mask[0, :observed_steps, node_idx] = True
        if observed_steps > 2:
            n2_mask[0, : observed_steps - 2, node_idx] = True

    coords = {
        "run_id": np.array(["real"], dtype=object),
        "date": np.arange(date_count),
        "region_id": np.array([f"{idx:03d}" for idx in range(total_nodes)], dtype=object),
    }
    data_vars = {
        "cases": (("run_id", "date", "region_id"), cases),
        "population": (("region_id",), population),
        "valid_targets": (("run_id", "region_id"), valid_targets[np.newaxis, :]),
        "biomarker_data_start": (
            ("run_id", "region_id"),
            biomarker_data_start[np.newaxis, :],
        ),
        "edar_biomarker_N1_mask": (("run_id", "date", "region_id"), n1_mask),
        "edar_biomarker_N2_mask": (("run_id", "date", "region_id"), n2_mask),
    }
    if include_edar_has_source:
        data_vars["edar_has_source"] = (("region_id",), edar_has_source)

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset_path = tmp_path / "node_split_fixture.zarr"
    dataset.to_zarr(dataset_path, mode="w", zarr_format=2)
    dataset.close()
    return dataset_path, strata_by_node, valid_targets


def _count_strata(
    nodes: list[int],
    strata_by_node: dict[int, tuple[int, int]],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for node in nodes:
        key = strata_by_node[node]
        counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_random_is_reproducible(tmp_path: Path) -> None:
    dataset_path, _strata_by_node, _valid_targets = _write_node_split_dataset(tmp_path)
    config = _build_config(dataset_path, node_split_strategy="random", seed=7)

    split_a = split_nodes_by_ratio(config)
    split_b = split_nodes_by_ratio(config)

    assert split_a == split_b


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_stratified_balances_population_and_wastewater(
    tmp_path: Path,
) -> None:
    dataset_path, strata_by_node, _valid_targets = _write_node_split_dataset(tmp_path)
    config = _build_config(dataset_path, node_split_strategy="stratified", seed=13)

    train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)

    assert len(train_nodes) == 70
    assert len(val_nodes) == 20
    assert len(test_nodes) == 10

    expected_train = {(pop_bin, ww): 7 for pop_bin in range(5) for ww in range(2)}
    expected_val = {(pop_bin, ww): 2 for pop_bin in range(5) for ww in range(2)}
    expected_test = {(pop_bin, ww): 1 for pop_bin in range(5) for ww in range(2)}

    assert _count_strata(train_nodes, strata_by_node) == expected_train
    assert _count_strata(val_nodes, strata_by_node) == expected_val
    assert _count_strata(test_nodes, strata_by_node) == expected_test


def _coverage_bins_for_nodes(
    nodes: list[int], metadata_bins: np.ndarray
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        key = int(metadata_bins[node])
        counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_respects_valid_targets_filter(tmp_path: Path) -> None:
    invalid_nodes = {8, 9, 18, 19}
    dataset_path, _strata_by_node, valid_targets = _write_node_split_dataset(
        tmp_path,
        invalid_nodes=invalid_nodes,
    )
    config = _build_config(dataset_path, node_split_strategy="stratified")

    train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)
    all_split_nodes = set(train_nodes) | set(val_nodes) | set(test_nodes)

    assert all_split_nodes.isdisjoint(invalid_nodes)
    assert len(all_split_nodes) == int(valid_targets.sum())


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_stratified_falls_back_to_biomarker_metadata(
    tmp_path: Path,
) -> None:
    dataset_path, strata_by_node, _valid_targets = _write_node_split_dataset(
        tmp_path,
        include_edar_has_source=False,
    )
    config = _build_config(dataset_path, node_split_strategy="stratified", seed=21)

    train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)

    expected_train = {(pop_bin, ww): 7 for pop_bin in range(5) for ww in range(2)}
    expected_val = {(pop_bin, ww): 2 for pop_bin in range(5) for ww in range(2)}
    expected_test = {(pop_bin, ww): 1 for pop_bin in range(5) for ww in range(2)}

    assert _count_strata(train_nodes, strata_by_node) == expected_train
    assert _count_strata(val_nodes, strata_by_node) == expected_val
    assert _count_strata(test_nodes, strata_by_node) == expected_test


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_crossval_is_deterministic(tmp_path: Path) -> None:
    dataset_path, _strata_by_node, _valid_targets = _write_node_split_dataset(tmp_path)
    config = _build_config(
        dataset_path,
        node_split_strategy="crossval",
        seed=17,
        crossval_enabled=True,
        crossval_num_folds=5,
        crossval_fold_index=2,
    )

    split_a = split_nodes_by_ratio(config)
    split_b = split_nodes_by_ratio(config)

    assert split_a == split_b


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_crossval_distributes_all_nodes_across_folds(
    tmp_path: Path,
) -> None:
    dataset_path, _strata_by_node, valid_targets = _write_node_split_dataset(tmp_path)
    all_nodes: set[int] = set()
    held_out_nodes: set[int] = set()
    per_fold_test_sizes: list[int] = []

    for fold_idx in range(5):
        config = _build_config(
            dataset_path,
            node_split_strategy="crossval",
            seed=17,
            crossval_enabled=True,
            crossval_num_folds=5,
            crossval_fold_index=fold_idx,
        )
        train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)

        split_nodes = set(train_nodes) | set(val_nodes) | set(test_nodes)
        assert split_nodes == set(range(int(valid_targets.sum())))
        assert set(train_nodes).isdisjoint(val_nodes)
        assert set(train_nodes).isdisjoint(test_nodes)
        assert set(val_nodes).isdisjoint(test_nodes)

        all_nodes |= split_nodes
        held_out_nodes |= set(test_nodes)
        per_fold_test_sizes.append(len(test_nodes))

    assert all_nodes == set(range(int(valid_targets.sum())))
    assert held_out_nodes == set(range(int(valid_targets.sum())))
    assert min(per_fold_test_sizes) >= 19
    assert max(per_fold_test_sizes) <= 21


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_crossval_balances_biomarker_coverage_bins(
    tmp_path: Path,
) -> None:
    dataset_path, _strata_by_node, _valid_targets = _write_node_split_dataset(tmp_path)
    metadata_config = _build_config(
        dataset_path,
        node_split_strategy="crossval",
        crossval_enabled=True,
        crossval_num_folds=5,
    )
    from data.dataset_factory import _load_node_split_metadata

    metadata = _load_node_split_metadata(
        config=metadata_config,
        dataset_path=dataset_path,
        run_id="real",
    )
    assert metadata.biomarker_observed_fraction_bin is not None

    test_bin_counts: list[dict[int, int]] = []
    for fold_idx in range(5):
        config = _build_config(
            dataset_path,
            node_split_strategy="crossval",
            seed=17,
            crossval_enabled=True,
            crossval_num_folds=5,
            crossval_fold_index=fold_idx,
        )
        _train_nodes, _val_nodes, test_nodes = split_nodes_by_ratio(config)
        test_bin_counts.append(
            _coverage_bins_for_nodes(test_nodes, metadata.biomarker_observed_fraction_bin)
        )

    for coverage_bin in {0, 1, 2}:
        values = [counts.get(coverage_bin, 0) for counts in test_bin_counts]
        assert max(values) - min(values) <= 3


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_crossval_falls_back_without_edar_source(
    tmp_path: Path,
) -> None:
    dataset_path, _strata_by_node, _valid_targets = _write_node_split_dataset(
        tmp_path,
        include_edar_has_source=False,
    )
    config = _build_config(
        dataset_path,
        node_split_strategy="crossval",
        seed=17,
        crossval_enabled=True,
        crossval_num_folds=5,
        crossval_fold_index=1,
    )

    train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)

    assert len(train_nodes) + len(val_nodes) + len(test_nodes) == 100


@pytest.mark.epiforecaster
def test_split_nodes_by_ratio_crossval_handles_sparse_strata(tmp_path: Path) -> None:
    dataset_path, _strata_by_node, _valid_targets = _write_node_split_dataset(tmp_path)
    config = _build_config(
        dataset_path,
        node_split_strategy="crossval",
        seed=17,
        crossval_enabled=True,
        crossval_num_folds=9,
        crossval_fold_index=3,
        node_split_biomarker_bins=4,
    )

    train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)

    assert len(train_nodes) + len(val_nodes) + len(test_nodes) == 100
