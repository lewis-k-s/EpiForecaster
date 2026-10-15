"""Memory-compaction tests for synthetic curriculum datasets.

``EpiDataset.release_dense_mobility`` drops the dense ``(T, N, N)`` mobility
state after init. These tests prove that the curriculum sampling order and
every batch tensor the model receives are bitwise identical with and without
the release, for production-like and thresholded edge selection.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

from data.context_assembly import pack_bool_last_dim, unpack_bool_last_dim
from data.curriculum_builder import build_curriculum_datasets
from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import EpiDataset
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


def _build(path: Path, config, *, release: bool):
    nodes = list(range(NUM_NODES))
    kwargs = dict(
        config=config,
        train_nodes=nodes[:8],
        val_nodes=nodes[8:10],
        test_nodes=nodes[10:],
        real_run="",
        synth_runs=RUN_IDS,
    )
    if release:
        return build_curriculum_datasets(**kwargs)
    with patch.object(EpiDataset, "release_dense_mobility", lambda self: None):
        return build_curriculum_datasets(**kwargs)


def _epoch_batches(result, epoch: int) -> list[list[int]]:
    sampler = EpidemicCurriculumSampler(
        result.train_dataset,
        batch_size=3,
        config=CURRICULUM,
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
    for name in (
        "adj_dense",
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
        assert not dataset._precomputed_k_hop_masks
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
def test_bool_packing_round_trips(width: int) -> None:
    mask = torch.from_numpy(np.random.default_rng(width).random((3, 4, width)) > 0.5)
    packed = pack_bool_last_dim(mask)
    assert packed.dtype == torch.uint8
    assert packed.shape == (3, 4, (width + 7) // 8)
    assert torch.equal(unpack_bool_last_dim(packed, width), mask)


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
