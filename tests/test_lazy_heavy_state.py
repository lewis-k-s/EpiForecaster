"""Lazy per-block heavy state of curriculum runs.

A synthetic run's *light* state (series, masks, index maps) is built for every
run at startup; its *heavy* state (dense mobility reduction, k-hop closure,
graph weights, context tables) is built per block by ``RunBlockPipeline``.
These tests prove the split changes nothing the model sees: index maps and
sample counts equal the eager build, heavy tensors equal the eager build, and
every batch of a blocked epoch is identical (values and order) to the eager
reference for every block size / prefetch / worker setting.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import ConcatDataset

from data.context_assembly import pack_mask_bits, unpack_mask_bits
from data.curriculum_builder import build_curriculum_datasets
from data.epi_batch import collate_epiforecaster_batch
from data.epi_dataset import HEAVY_STATE_ATTRS, SHARED_SERIES_ATTRS, EpiDataset
from data.samplers import EpidemicCurriculumSampler
from models.configs import CurriculumConfig, CurriculumPhaseConfig, ModelVariant
from tests.dataset_test_helpers import make_epi_config, make_tiny_epi_dataset
from utils import training_utils
from utils.device import prepare_batch_for_device

NUM_NODES = 12
PERIODS = 40
RUN_IDS = ["0_Reactive", "1_Baseline", "2_Reactive"]
NODES = list(range(NUM_NODES))


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
        # Run-dependent missingness: different eligible windows per run.
        cases_mask = (rng.random((1, PERIODS, NUM_NODES)) > 0.3).astype(np.float32)
        dataset = make_tiny_epi_dataset(
            periods=PERIODS,
            num_nodes=NUM_NODES,
            cases=cases,
            cases_mask=cases_mask,
            hospitalizations=cases[..., 0] * 0.2,
            deaths=cases[..., 0] * 0.05,
            mobility=np.log1p(flow),
        )
        parts.append(dataset.assign_coords(run_id=[run_id]))
    xr.concat(parts, dim="run_id", data_vars="all").to_zarr(
        path, mode="w", zarr_format=2
    )


def _config(
    path: Path,
    *,
    min_flow_raw: float = 0.0,
    clip_max: float = 8.0,
    imported_risk: bool = True,
):
    return make_epi_config(
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
            "cases_dim": 10 if imported_risk else 9,
        },
        data_options={
            "mobility_min_flow_raw": min_flow_raw,
            "mobility_weight_clip_max": clip_max,
            "window_stride": 3,
            "min_history_observed_fraction": 0.5,
            "min_horizon_observed_fraction": 0.5,
            "use_imported_risk": imported_risk,
            "mobility_lags": [1] if imported_risk else [],
        },
    )


def _curriculum(
    *, run_block_size: int = 8, prefetch_blocks: int = 1, block_builders: int = 2
) -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        active_runs=-1,
        chunk_size=5,
        run_sampling="round_robin",
        run_block_size=run_block_size,
        prefetch_blocks=prefetch_blocks,
        block_builders=block_builders,
        builder_threads=1,
        light_build_workers=3,
        schedule=[CurriculumPhaseConfig(start_epoch=0, end_epoch=2, synth_ratio=1.0)],
    )


def _build(config, curriculum: CurriculumConfig):
    return build_curriculum_datasets(
        config=config,
        train_nodes=NODES[:8],
        val_nodes=NODES[8:10],
        test_nodes=NODES[10:],
        real_run="",
        synth_runs=RUN_IDS,
        curriculum=curriculum,
    )


def _sampler(dataset, curriculum: CurriculumConfig, **kwargs):
    return EpidemicCurriculumSampler(
        dataset, batch_size=3, config=curriculum, real_run_id="", seed=42, **kwargs
    )


def _eager_runs(config, lazy_runs) -> list[EpiDataset]:
    """The pre-lazy build: every run constructed with its heavy state."""
    reference = lazy_runs[0]
    return [
        EpiDataset(
            config=config,
            target_nodes=list(run.target_nodes),
            context_nodes=list(run.target_nodes),
            biomarker_preprocessor=reference.biomarker_preprocessor,
            mobility_preprocessor=reference.mobility_preprocessor,
            run_id=run.run_id,
        )
        for run in lazy_runs
    ]


def _assert_same_value(left, right, name: str) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), name
        assert left.dtype == right.dtype, name
        assert torch.equal(left, right), name
    else:
        assert left == right, name


def _assert_items_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for name in left:
        if left[name] is None or right[name] is None:
            assert left[name] is right[name], name
        else:
            _assert_same_value(left[name], right[name], name)


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
    for name in ("x_dense", "global_t", "target_node", "mob_real_node_idx", "run_id"):
        _assert_same_value(
            getattr(left.mob_batch, name), getattr(right.mob_batch, name), name
        )


def _prepared(dataset, indices: list[int]):
    batch = collate_epiforecaster_batch(
        [dataset[i] for i in indices], require_region_index=False
    )
    return prepare_batch_for_device(batch, dataset=dataset, device=torch.device("cpu"))


@pytest.mark.epiforecaster
@pytest.mark.parametrize("imported_risk", [True, False])
@pytest.mark.parametrize(
    ("min_flow_raw", "clip_max"),
    [(0.0, 8.0), (0.5, 1.5)],
    ids=["production_selection", "threshold_and_clip"],
)
def test_light_build_matches_the_eager_build(
    tmp_path: Path, min_flow_raw: float, clip_max: float, imported_risk: bool
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config = _config(
        path, min_flow_raw=min_flow_raw, clip_max=clip_max, imported_risk=imported_risk
    )
    lazy_runs = _build(config, _curriculum()).train_dataset.datasets
    eager_runs = _eager_runs(config, lazy_runs)

    for lazy, eager in zip(lazy_runs, eager_runs, strict=True):
        assert lazy.lazy_heavy_state and not lazy.heavy_state_ready
        assert all(getattr(lazy, name) is None for name in HEAVY_STATE_ATTRS)
        # Index maps and sample counts: exactly the eager build.
        assert lazy._index_map == eager._index_map
        assert lazy._index_lookup == eager._index_lookup
        assert lazy.window_starts == eager.window_starts
        assert lazy._valid_window_starts_by_node == eager._valid_window_starts_by_node
        assert len(lazy) == len(eager) > 0
        assert lazy.sample_window_starts() == eager.sample_window_starts()
        for name in SHARED_SERIES_ATTRS:
            left, right = getattr(lazy, name), getattr(eager, name)
            if left is None:
                assert right is None, name
            else:
                _assert_same_value(left, right, name)
        if imported_risk:
            _assert_same_value(lazy.lagged_risk, eager.lagged_risk, "lagged_risk")
        # Everything a DataLoader worker reads.
        for index in range(len(eager)):
            _assert_items_equal(lazy[index], eager[index])


@pytest.mark.epiforecaster
def test_index_maps_do_not_depend_on_mobility_or_k_hop(tmp_path: Path) -> None:
    """Eligibility reads only the series masks, never mobility or the k-hop set."""
    path = tmp_path / "runs.zarr"
    _write_runs(path)

    def build(min_flow_raw: float) -> EpiDataset:
        return EpiDataset(
            config=_config(path, min_flow_raw=min_flow_raw),
            target_nodes=NODES[:4],
            context_nodes=NODES[:8],
            run_id=RUN_IDS[0],
        )

    with_edges = build(0.0)
    # Above every flow: no edge survives, so the k-hop closure is the targets.
    without_edges = build(1e6)
    assert without_edges._graph_node_ids.numel() == 4
    assert with_edges._graph_node_ids.numel() > 4
    assert with_edges._index_map == without_edges._index_map
    assert with_edges.window_starts == without_edges.window_starts


@pytest.mark.epiforecaster
@pytest.mark.parametrize("imported_risk", [True, False])
def test_lazy_heavy_state_matches_the_eager_build(
    tmp_path: Path, imported_risk: bool
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config = _config(path, imported_risk=imported_risk)
    lazy_runs = _build(config, _curriculum()).train_dataset.datasets
    eager_runs = _eager_runs(config, lazy_runs)

    for lazy, eager in zip(lazy_runs, eager_runs, strict=True):
        assert lazy.build_heavy_state() > 0
        assert lazy.heavy_state_ready
        assert lazy.build_heavy_state() == 0.0  # already resident
        eager.release_dense_mobility()  # the compact layout the lazy build keeps
        for name in HEAVY_STATE_ATTRS:
            want, got = getattr(eager, name), getattr(lazy, name)
            if want is None:
                assert got is None, name
            else:
                _assert_same_value(got, want, name)
        assert lazy.mobility_diagnostics == eager.mobility_diagnostics
        assert set(lazy.heavy_build_timings) >= {"dense_mobility_s", "k_hop_s"}
        if imported_risk:
            _assert_same_value(lazy.lagged_risk, eager.lagged_risk, "lagged_risk")

        lazy.release_heavy_state()
        assert not lazy.heavy_state_ready
        assert all(getattr(lazy, name) is None for name in HEAVY_STATE_ATTRS)
        assert lazy.mobility_diagnostics == eager.mobility_diagnostics
        with pytest.raises(RuntimeError, match="not resident"):
            _ = lazy.mob_real_node_idx


@pytest.mark.epiforecaster
def test_diagnostics_are_computed_once_per_run(tmp_path: Path) -> None:
    """Block builds after the first reuse the run's diagnostics."""
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    run = _build(_config(path), _curriculum()).train_dataset.datasets[0]

    assert run.heavy_build_request().compute_diagnostics
    run.build_heavy_state()
    diagnostics = run.mobility_diagnostics
    assert diagnostics is not None
    assert "diagnostics_s" in run.heavy_build_timings

    run.release_heavy_state()
    assert not run.heavy_build_request().compute_diagnostics
    run.build_heavy_state()
    assert run.mobility_diagnostics is diagnostics
    assert "diagnostics_s" not in run.heavy_build_timings


@pytest.mark.epiforecaster
def test_eager_datasets_build_heavy_state_in_init(tmp_path: Path) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    result = _build(_config(path), _curriculum())
    for split in (result.val_dataset, result.test_dataset):
        assert split.heavy_state_ready and not split.lazy_heavy_state
        assert split.block_pipeline is None
        with pytest.raises(RuntimeError, match="eagerly"):
            split.build_heavy_state()
    # Shared dense mobility of the validation split outlives the scaler run's
    # conversion to a lazy dataset.
    assert result.val_dataset.preloaded_mobility is not None
    assert (
        result.val_dataset.preloaded_mobility is result.test_dataset.preloaded_mobility
    )


@pytest.mark.epiforecaster
def test_pickled_datasets_never_carry_heavy_state(tmp_path: Path) -> None:
    """DataLoader workers (spawn) get light state only and can still sample."""
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    run = _build(_config(path), _curriculum()).train_dataset.datasets[0]
    run.build_heavy_state()
    restored = pickle.loads(pickle.dumps(run))
    assert not restored.heavy_state_ready
    assert restored.block_pipeline is None
    assert all(getattr(restored, name) is None for name in HEAVY_STATE_ATTRS)
    _assert_items_equal(restored[0], run[0])
    assert run.heavy_state_ready  # the original is untouched


@pytest.mark.epiforecaster
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("prefetch_blocks", [0, 1, 2])
@pytest.mark.parametrize("run_block_size", [1, 2])
def test_blocked_epochs_match_the_eager_reference(
    tmp_path: Path, run_block_size: int, prefetch_blocks: int, workers: int
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config = _config(path)
    curriculum = _curriculum(
        run_block_size=run_block_size,
        prefetch_blocks=prefetch_blocks,
        block_builders=workers,
    )
    lazy = _build(config, curriculum)
    pipeline = lazy.block_pipeline
    assert pipeline is not None

    # Pre-lazy behavior: every run resident, no pipeline.
    eager_runs = _eager_runs(config, lazy.train_dataset.datasets)
    for run in eager_runs:
        run.release_dense_mobility()
    eager_train = ConcatDataset(eager_runs)

    lazy_sampler = _sampler(lazy.train_dataset, curriculum, block_pipeline=pipeline)
    eager_sampler = _sampler(eager_train, curriculum)
    checked = 0
    for epoch in range(2):
        eager_order = [
            batch
            for block in eager_sampler.plan_epoch_blocks(epoch)
            for batch in block.batches
        ]
        assert len(eager_order) > len(RUN_IDS)

        lazy_order = [
            batch
            for block in lazy_sampler.plan_epoch_blocks(epoch)
            for batch in block.batches
        ]
        assert lazy_order == eager_order

        lazy_sampler.set_curriculum(epoch)
        for indices, want in zip(lazy_sampler, eager_order, strict=True):
            assert indices == want
            _assert_batches_equal(
                _prepared(lazy.train_dataset, indices),
                _prepared(eager_train, indices),
            )
            checked += 1
        summary = pipeline.end_epoch()
        assert summary is not None
        assert summary["blocks"] == -(-len(RUN_IDS) // run_block_size)
    assert checked > 0

    for split in ("val_dataset", "test_dataset"):
        dataset = getattr(lazy, split)
        assert dataset.heavy_state_ready
        _prepared(dataset, list(range(min(4, len(dataset)))))
    pipeline.shutdown()


@pytest.mark.epiforecaster
@pytest.mark.parametrize("prefetch_blocks", [0, 1, 2])
def test_heavy_state_is_resident_only_in_current_and_prefetched_blocks(
    tmp_path: Path, prefetch_blocks: int
) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    curriculum = _curriculum(run_block_size=1, prefetch_blocks=prefetch_blocks)
    result = _build(_config(path), curriculum)
    pipeline = result.block_pipeline
    runs = result.train_dataset.datasets
    sampler = _sampler(result.train_dataset, curriculum, block_pipeline=pipeline)
    plan = sampler.plan_epoch_blocks(0)
    sampler.set_curriculum(0)
    block_of_batch = [index for index, block in enumerate(plan) for _ in block.batches]

    for indices, block_index in zip(sampler, block_of_batch, strict=True):
        _prepared(result.train_dataset, indices)
        allowed = {
            run
            for later in plan[block_index : block_index + prefetch_blocks + 1]
            for run in later.synth_dataset_indices
        }
        ready = {index for index, run in enumerate(runs) if run.heavy_state_ready}
        current = set(plan[block_index].synth_dataset_indices)
        assert current <= ready, "the current block is fully built"
        assert ready <= allowed, "nothing outside the prefetch window is resident"
        for index, run in enumerate(runs):
            if index not in allowed:
                assert not getattr(run, "_gpu_mobility_cache", None)
                assert not getattr(run, "_context_device_cache", None)
    pipeline.end_epoch()
    # A multi-block epoch drops its last block too.
    assert not any(run.heavy_state_ready for run in runs)
    pipeline.shutdown()


@pytest.mark.epiforecaster
def test_single_block_epochs_keep_heavy_state_resident(tmp_path: Path) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    curriculum = _curriculum(run_block_size=8)
    result = _build(_config(path), curriculum)
    pipeline = result.block_pipeline
    sampler = _sampler(result.train_dataset, curriculum, block_pipeline=pipeline)
    for epoch in range(2):
        sampler.set_curriculum(epoch)
        for indices in sampler:
            _prepared(result.train_dataset, indices)
        summary = pipeline.end_epoch()
        assert summary["blocks"] == 1
        assert all(run.heavy_state_ready for run in result.train_dataset.datasets)
        # The second epoch reuses the resident runs: nothing is rebuilt.
        if epoch == 1:
            assert summary["mean_run_build_s"] == 0.0
    pipeline.shutdown()


@pytest.mark.epiforecaster
def test_profiling_records_cover_startup_blocks_and_epochs(tmp_path: Path) -> None:
    path = tmp_path / "runs.zarr"
    _write_runs(path)
    curriculum = _curriculum(run_block_size=2, prefetch_blocks=1)
    result = _build(_config(path), curriculum)
    pipeline = result.block_pipeline
    records: list[dict] = []
    pipeline.set_record_sink(records.append)  # replays the startup record

    assert [r["kind"] for r in records] == ["startup"]
    assert records[0]["n_runs"] == len(RUN_IDS)
    assert records[0]["light_build_workers"] == 3
    assert records[0]["light_build_s"] > 0

    sampler = _sampler(result.train_dataset, curriculum, block_pipeline=pipeline)
    sampler.set_curriculum(0)
    for indices in sampler:
        _prepared(result.train_dataset, indices)
    summary = pipeline.end_epoch()

    blocks = [r for r in records if r["kind"] == "block"]
    assert [r["block_index"] for r in blocks] == [0, 1]
    assert [r["n_runs"] for r in blocks] == [2, 1]
    for record in blocks:
        assert record["n_runs_built"] == record["n_runs"]
        assert record["build_s"] > 0
        assert 0 < record["run_build_s_mean"] <= record["run_build_s_max"]
        assert record["wait_s"] >= 0
        assert record["prefetch_blocks"] == 1
        assert record["block_builders"] == 2
        assert record["builder_threads"] == 1
        assert "k_hop_s" in record["run_phase_s_mean"]
        assert record["epoch"] == 0
    assert records[-1] == summary
    assert summary["blocks"] == 2
    assert summary["total_wait_s"] == pytest.approx(sum(r["wait_s"] for r in blocks))
    assert summary["mean_block_build_s"] == pytest.approx(
        sum(r["build_s"] for r in blocks) / 2
    )
    pipeline.shutdown()


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
    monkeypatch.setattr(training_utils, "_device_cached_datasets", OrderedDict())
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


@pytest.mark.epiforecaster
def test_released_datasets_do_not_count_against_the_device_cache_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(training_utils, "_device_cached_datasets", OrderedDict())
    monkeypatch.setattr(training_utils, "_device_cache_limit", 2)
    first, second, third = _CachedDataset(), _CachedDataset(), _CachedDataset()
    training_utils._retain_device_caches(first)
    training_utils._retain_device_caches(second)
    # ``first``'s block ended: its caches are dropped by the dataset itself.
    del first._gpu_mobility_cache, first._context_device_cache
    training_utils._retain_device_caches(third)
    assert second._gpu_mobility_cache and third._gpu_mobility_cache


@pytest.mark.epiforecaster
def test_spawned_builder_processes_match_the_eager_reference(
    tmp_path: Path, spawn_builders
) -> None:
    """The real path: spawned processes, shared-memory tensors, bitwise equal."""
    import os

    path = tmp_path / "runs.zarr"
    _write_runs(path)
    config = _config(path)
    curriculum = _curriculum(run_block_size=2, prefetch_blocks=1, block_builders=2)
    lazy = _build(config, curriculum)
    pipeline = lazy.block_pipeline
    records: list[dict] = []
    pipeline.set_record_sink(records.append)
    try:
        runs = lazy.train_dataset.datasets
        eager_runs = _eager_runs(config, runs)
        for run in eager_runs:
            run.release_dense_mobility()
        eager_train = ConcatDataset(eager_runs)
        by_id = dict(zip((run.run_id for run in runs), zip(runs, eager_runs)))

        lazy_sampler = _sampler(lazy.train_dataset, curriculum, block_pipeline=pipeline)
        eager_sampler = _sampler(eager_train, curriculum)
        eager_order = [
            batch
            for block in eager_sampler.plan_epoch_blocks(0)
            for batch in block.batches
        ]
        lazy_sampler.set_curriculum(0)
        compared: set[str] = set()
        for indices, want in zip(lazy_sampler, eager_order, strict=True):
            assert indices == want
            batch = _prepared(lazy.train_dataset, indices)
            _assert_batches_equal(batch, _prepared(eager_train, indices))
            run, eager = by_id[batch.mob_batch.run_id]
            if run.run_id in compared:
                continue
            compared.add(run.run_id)
            for name in HEAVY_STATE_ATTRS:
                want_value, got = getattr(eager, name), getattr(run, name)
                if want_value is None:
                    assert got is None, name
                else:
                    _assert_same_value(got, want_value, name)
            # Built in another process and mapped from shared memory.
            assert run.graph_mobility_weights.is_shared()
            assert run.context_neigh_table.is_shared()
        assert compared == set(RUN_IDS)
        pipeline.end_epoch()
    finally:
        pipeline.shutdown()

    startup = records[0]
    assert startup["kind"] == "startup"
    assert startup["builder_pool_startup_s"] > 0
    assert startup["block_builders"] == 2 and startup["builder_threads"] == 1
    assert len(startup["builder_pids"]) == 2
    assert os.getpid() not in startup["builder_pids"]
    blocks = [r for r in records if r["kind"] == "block"]
    assert len(blocks) == 2
    for record in blocks:
        assert record["builder_pids"], "built by builder processes"
        assert set(record["builder_pids"]) <= set(startup["builder_pids"])
        assert os.getpid() not in record["builder_pids"]
        assert record["builder_torch_threads"] == 1
        assert "install_s" in record["run_phase_s_mean"]
