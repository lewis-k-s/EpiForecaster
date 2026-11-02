"""Memory-compaction tests for synthetic curriculum datasets.

``EpiDataset.release_dense_mobility`` drops the dense ``(T, N, N)`` mobility
state after init. These tests prove that the curriculum sampling order and
every batch tensor the model receives are bitwise identical with and without
the release, for production-like and thresholded edge selection.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

from data.context_assembly import pack_mask_bits, unpack_mask_bits
from data.curriculum_builder import build_curriculum_datasets
from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import RUN_STATE_ATTRS, EpiDataset
from data.samplers import EpidemicCurriculumSampler
from models.configs import CurriculumConfig, CurriculumPhaseConfig, ModelVariant
from tests.dataset_test_helpers import make_epi_config, make_tiny_epi_dataset
from utils import training_utils
from utils.training_utils import inject_context_features, inject_gpu_mobility

NUM_NODES = 12
PERIODS = 40
RUN_IDS = ["0_Reactive", "1_Baseline", "2_Reactive"]


def _write_runs(path: Path) -> None:
    parts = []
    for run_index, run_id in enumerate(RUN_IDS):
        rng = np.random.default_rng(11 + run_index)
        shape = (1, PERIODS, NUM_NODES, NUM_NODES)
        flow = rng.random(shape, dtype=np.float32) * 6.0
        flow[rng.random(shape) < 0.35] = 0.0
        # Positive flows that underflow to zero in float16 log space.
        flow[:, :, 0, :] = np.where(flow[:, :, 0, :] > 0, 1e-9, 0.0)
        # Flows below the 0.5 raw threshold used by one parametrization.
        flow[:, :, 1, :] = np.where(flow[:, :, 1, :] > 0, 0.3, 0.0)
        cases = rng.random((1, PERIODS, NUM_NODES, 1), dtype=np.float32) * 40.0
        dataset = make_tiny_epi_dataset(
            periods=PERIODS,
            num_nodes=NUM_NODES,
            cases=cases,
            hospitalizations=cases[..., 0] * 0.2,
            deaths=cases[..., 0] * 0.05,
            mobility=np.log1p(flow),
        )
        parts.append(dataset.assign_coords(run_id=[run_id]))
    xr.concat(parts, dim="run_id", data_vars="all").to_zarr(
        path, mode="w", zarr_format=2
    )


def _config(path: Path, *, min_flow_raw: float, clip_max: float):
    config = make_epi_config(
        path,
        model_options={
            "type": ModelVariant(
                cases=True, regions=False, biomarkers=True, mobility=True
            ),
            "input_window_length": 8,
            "forecast_horizon": 4,
            "gnn_depth": 2,
            "gnn_module": "gcn",
            "max_neighbors": 3,
            # 3 series x 3 channels + one imported-risk column per mobility lag.
            "cases_dim": 10,
        },
        data_options={
            "mobility_min_flow_raw": min_flow_raw,
            "mobility_weight_clip_max": clip_max,
            "window_stride": 3,
            "min_history_observed_fraction": 0.0,
            "min_horizon_observed_fraction": 0.0,
            "use_imported_risk": True,
            "mobility_lags": [1],
        },
    )
    return config


CURRICULUM = CurriculumConfig(
    enabled=True,
    active_runs=-1,
    chunk_size=5,
    run_sampling="round_robin",
    schedule=[CurriculumPhaseConfig(start_epoch=0, end_epoch=2, synth_ratio=1.0)],
)


def _build(
    path: Path,
    config,
    *,
    release: bool,
    run_block_size: int = CURRICULUM.run_block_size,
):
    nodes = list(range(NUM_NODES))
    kwargs = dict(
        config=config,
        train_nodes=nodes[:8],
        val_nodes=nodes[8:10],
        test_nodes=nodes[10:],
        real_run="",
        synth_runs=RUN_IDS,
        run_block_size=run_block_size,
    )
    if release:
        return build_curriculum_datasets(**kwargs)
    with patch.object(EpiDataset, "release_dense_mobility", lambda self: None):
        return build_curriculum_datasets(**kwargs)


def _epoch_batches(
    result, epoch: int, curriculum: CurriculumConfig = CURRICULUM
) -> list[list[int]]:
    sampler = EpidemicCurriculumSampler(
        result.train_dataset,
        batch_size=3,
        config=curriculum,
        real_run_id="",
        seed=42,
    )
    sampler.set_curriculum(epoch)
    return list(sampler)


def _collated(dataset, indices: list[int]):
    batch = collate_epiforecaster_batch(
        [dataset[i] for i in indices], require_region_index=False
    )
    inject_gpu_mobility(batch, dataset, torch.device("cpu"))
    inject_context_features(batch, dataset, torch.device("cpu"))
    return batch


def _assert_same_value(left, right, name: str) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), name
        assert left.dtype == right.dtype, name
        assert torch.equal(left, right), name
    else:
        assert left == right, name


def _assert_batches_equal(left, right) -> None:
    for field in dataclasses.fields(left):
        if field.name == "mob_batch":
            continue
        _assert_same_value(
            getattr(left, field.name), getattr(right, field.name), field.name
        )
    _assert_same_value(
        left.mob_batch.adj_table[left.mob_batch.adj_time_ids],
        right.mob_batch.adj_table[right.mob_batch.adj_time_ids],
        "adj_table",
    )
    for name in (
        "x_dense",
        "global_t",
        "target_node",
        "mob_real_node_idx",
        "run_id",
    ):
        _assert_same_value(
            getattr(left.mob_batch, name), getattr(right.mob_batch, name), name
        )


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("min_flow_raw", "clip_max"),
    [(0.0, 8.0), (0.5, 1.5)],
    ids=["production_selection", "threshold_and_clip"],
)
def test_release_keeps_sampling_order_and_batches_identical(
    tmp_path: Path, min_flow_raw: float, clip_max: float
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    released = _build(
        path, _config(path, min_flow_raw=min_flow_raw, clip_max=clip_max), release=True
    )
    dense = _build(
        path, _config(path, min_flow_raw=min_flow_raw, clip_max=clip_max), release=False
    )

    for dataset in released.train_dataset.datasets:
        assert dataset.preloaded_mobility is None
        assert dataset.mobility_mask is None
        assert dataset.graph_mobility_weights is not None
        assert not dataset._adjacency_cache
    for dataset in dense.train_dataset.datasets:
        assert dataset.preloaded_mobility is not None
        assert dataset.graph_mobility_weights is None

    checked = 0
    for epoch in range(2):
        released_batches = _epoch_batches(released, epoch)
        dense_batches = _epoch_batches(dense, epoch)
        assert released_batches == dense_batches
        assert len(released_batches) > len(RUN_IDS)
        for indices in released_batches:
            _assert_batches_equal(
                _collated(released.train_dataset, indices),
                _collated(dense.train_dataset, indices),
            )
            checked += 1
    assert checked > 0

    for split in ("val_dataset", "test_dataset"):
        released_split = getattr(released, split)
        dense_split = getattr(dense, split)
        indices = list(range(min(4, len(released_split))))
        _assert_batches_equal(
            _collated(released_split, indices), _collated(dense_split, indices)
        )


@pytest.mark.epiforecaster
def test_release_keeps_shared_eval_tensors_alive(tmp_path: Path) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    result = _build(path, _config(path, min_flow_raw=0.0, clip_max=8.0), release=True)

    scaler = next(
        ds
        for ds in result.train_dataset.datasets
        if ds.run_id == result.val_dataset.run_id
    )
    assert scaler.preloaded_mobility is None
    assert result.val_dataset.preloaded_mobility is not None
    assert (
        result.val_dataset.preloaded_mobility is result.test_dataset.preloaded_mobility
    )


@pytest.mark.epiforecaster
@pytest.mark.parametrize("width", [1, 7, 8, 9, 23])
def test_mask_bit_packing_round_trips(width: int) -> None:
    mask = torch.from_numpy(np.random.default_rng(width).random((3, 4, width)) > 0.5)
    packed = pack_mask_bits(mask)
    assert packed.dtype == torch.uint8
    assert packed.shape == (3, 4, (width + 7) // 8)
    assert torch.equal(unpack_mask_bits(packed, width), mask)


class _CachedDataset:
    def __init__(self) -> None:
        self._gpu_mobility_cache = {"key": torch.zeros(1)}
        self._context_device_cache = {"key": {"table": torch.zeros(1)}}


@pytest.mark.epiforecaster
def test_device_caches_keep_only_recent_datasets(monkeypatch) -> None:
    monkeypatch.setattr(
        training_utils,
        "_device_cached_datasets",
        type(training_utils._device_cached_datasets)(),
    )
    limit = training_utils.MAX_DEVICE_CACHED_DATASETS
    monkeypatch.setattr(training_utils, "_device_cache_limit", limit)
    datasets = [_CachedDataset() for _ in range(limit + 2)]
    for dataset in datasets:
        training_utils._retain_device_caches(dataset)
    # Reusing the oldest resident dataset makes it most recent again.
    training_utils._retain_device_caches(datasets[2])
    training_utils._retain_device_caches(_CachedDataset())

    evicted = {0, 1, 3}
    for index, dataset in enumerate(datasets):
        resident = index not in evicted
        assert bool(dataset._gpu_mobility_cache) is resident, index
        assert bool(dataset._context_device_cache) is resident, index


def _fresh_device_registry(monkeypatch, limit: int) -> None:
    monkeypatch.setattr(training_utils, "_device_cached_datasets", OrderedDict())
    monkeypatch.setattr(training_utils, "_device_cache_limit", limit)


@pytest.mark.epiforecaster
def test_release_drops_k_hop_reachability_and_keeps_context_tables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    released = _build(path, _config(path, min_flow_raw=0.0, clip_max=8.0), release=True)
    dense = _build(path, _config(path, min_flow_raw=0.0, clip_max=8.0), release=False)

    for dataset in released.train_dataset.datasets:
        assert dataset._target_k_hop_reach is None
        assert dataset.context_khop_table is not None
    for dataset in dense.train_dataset.datasets:
        assert dataset._target_k_hop_reach is not None


@pytest.mark.epiforecaster
def test_spilled_run_state_is_restored_bitwise_on_every_visit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config_args = dict(min_flow_raw=0.0, clip_max=8.0)
    resident = _build(path, _config(path, **config_args), release=True)
    spilled = _build(path, _config(path, **config_args), release=True, run_block_size=1)

    for expected, actual in zip(
        resident.train_dataset.datasets, spilled.train_dataset.datasets
    ):
        assert not expected.run_state_spilled
        assert actual.run_state_spilled
        assert all(getattr(actual, name) is None for name in RUN_STATE_ATTRS)
        for _ in range(2):
            actual.load_run_state()
            for name in RUN_STATE_ATTRS:
                want, got = getattr(expected, name), getattr(actual, name)
                if want is None:
                    assert got is None, name
                    continue
                assert got.dtype == want.dtype, name
                assert torch.equal(got, want), name
            actual.unload_run_state()
            assert all(getattr(actual, name) is None for name in RUN_STATE_ATTRS)

    # DataLoader workers receive datasets without the spilled tables.
    restored = pickle.loads(pickle.dumps(spilled.train_dataset.datasets[0]))
    assert restored.run_state_spilled
    assert all(getattr(restored, name) is None for name in RUN_STATE_ATTRS)


@pytest.mark.epiforecaster
def test_spill_directory_is_removed_with_the_training_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    spilled = _build(
        path,
        _config(path, min_flow_raw=0.0, clip_max=8.0),
        release=True,
        run_block_size=1,
    )
    state_dirs = list(scratch.glob("epiforecaster_*/run_state_*"))
    assert len(state_dirs) == 1
    assert len(list(state_dirs[0].glob("run_*.pt"))) == len(RUN_IDS)

    del spilled
    import gc

    gc.collect()
    assert not state_dirs[0].exists()


@pytest.mark.epiforecaster
@pytest.mark.parametrize("run_block_size", [1, 2])
def test_blocked_rotation_keeps_only_block_runs_resident(
    tmp_path: Path, monkeypatch, run_block_size: int
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config_args = dict(min_flow_raw=0.0, clip_max=8.0)
    resident = _build(path, _config(path, **config_args), release=True)
    spilled = _build(
        path,
        _config(path, **config_args),
        release=True,
        run_block_size=run_block_size,
    )
    curriculum = dataclasses.replace(CURRICULUM, run_block_size=run_block_size)
    sampler = EpidemicCurriculumSampler(
        spilled.train_dataset,
        batch_size=3,
        config=curriculum,
        real_run_id="",
        seed=42,
    )
    blocks = sampler.plan_epoch_blocks(0)
    assert len(blocks) == -(-len(RUN_IDS) // run_block_size)
    assert sampler.resident_dataset_limit == run_block_size

    _fresh_device_registry(monkeypatch, limit=100)
    expected = [
        _collated(resident.train_dataset, indices)
        for block in blocks
        for indices in block.batches
    ]

    _fresh_device_registry(monkeypatch, limit=sampler.resident_dataset_limit)
    datasets = spilled.train_dataset.datasets
    block_of_batch = [
        block for block in blocks for _ in block.batches
    ]  # the block each consecutive batch belongs to
    sampler.set_curriculum(0)
    consumed = 0
    for indices, block, want in zip(sampler, block_of_batch, expected):
        _assert_batches_equal(_collated(spilled.train_dataset, indices), want)
        device_cached = {
            i
            for i, ds in enumerate(datasets)
            if getattr(ds, "_gpu_mobility_cache", None)
        }
        host_loaded = {
            i for i, ds in enumerate(datasets) if ds.context_neigh_table is not None
        }
        assert device_cached <= set(block.synth_dataset_indices)
        assert host_loaded <= set(block.synth_dataset_indices)
        consumed += 1
    assert consumed == len(expected) == len(block_of_batch)
