"""Block pipeline scheduling, fork guard and profiling records (stub runs)."""

from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data.block_pipeline import RunBlockPipeline, live_builder_threads
from training.dataloader_factory import ForkGuardedDataLoader
from training.training_dynamics import (
    BLOCK_BUILD_ARTIFACT_NAME,
    append_block_build_record,
)


class _StubRun:
    """Minimal lazy run; ``gate`` holds its build until set."""

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.run_id = name
        self.heavy_state_ready = False
        self.heavy_build_timings = {"phase_s": 0.25}
        self.block_pipeline: RunBlockPipeline | None = None
        self.started = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.fail = fail
        self.builds = 0
        self.releases = 0

    def build_heavy_state(self) -> float:
        self.started.set()
        assert self.gate.wait(10), "test gate never opened"
        if self.fail:
            raise ValueError(f"build of {self.run_id} failed")
        self.heavy_state_ready = True
        self.builds += 1
        return 0.01

    def release_heavy_state(self) -> None:
        self.heavy_state_ready = False
        self.releases += 1


def _pipeline(runs: list[_StubRun], *, prefetch: int, workers: int = 2):
    pipeline = RunBlockPipeline(prefetch_blocks=prefetch, block_build_workers=workers)
    pipeline.register(runs)
    return pipeline


@pytest.fixture
def blocks() -> list[list[_StubRun]]:
    return [
        [_StubRun("a0"), _StubRun("a1")],
        [_StubRun("b0"), _StubRun("b1")],
        [_StubRun("c0")],
    ]


def _all(blocks: list[list[_StubRun]]) -> list[_StubRun]:
    return [run for block in blocks for run in block]


def test_next_block_builds_in_the_background_while_the_current_one_trains(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=1)
    for run in blocks[1]:
        run.gate.clear()  # the prefetched block is still building
    pipeline.begin_epoch(blocks, epoch=0)

    pipeline.activate(blocks[0][0])
    assert all(run.heavy_state_ready for run in blocks[0])
    assert blocks[1][0].started.wait(5), "block 1 prefetch starts with block 0"
    assert not blocks[2][0].started.is_set(), "only one block ahead"
    # Training block 0 proceeds without waiting for block 1.
    pipeline.activate(blocks[0][1])

    for run in blocks[1]:
        run.gate.set()
    pipeline.activate(blocks[1][0])
    assert all(run.heavy_state_ready for run in blocks[1])
    assert all(not run.heavy_state_ready for run in blocks[0]), "previous block dropped"
    assert blocks[2][0].started.wait(5)
    pipeline.shutdown()


def test_zero_prefetch_builds_each_block_on_demand(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=0)
    pipeline.begin_epoch(blocks, epoch=0)
    assert not any(run.started.is_set() for run in _all(blocks))

    pipeline.activate(blocks[0][0])
    assert all(run.heavy_state_ready for run in blocks[0])
    assert not any(run.started.is_set() for run in blocks[1] + blocks[2])

    pipeline.activate(blocks[1][1])
    assert all(run.heavy_state_ready for run in blocks[1])
    assert not any(run.heavy_state_ready for run in blocks[0])
    assert not blocks[2][0].started.is_set()
    pipeline.shutdown()


def test_prefetch_window_never_exceeds_the_configured_blocks(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=2)
    pipeline.begin_epoch(blocks, epoch=0)
    pipeline.activate(blocks[0][0])
    # Wait until every run inside the window has finished building.
    for run in blocks[1] + blocks[2]:
        assert run.started.wait(5)
    pipeline.shutdown()
    assert [run.builds for run in _all(blocks)] == [1, 1, 1, 1, 1]


def test_fork_guard_leaves_no_builder_thread_and_no_heavy_state(blocks):
    runs = _all(blocks)
    pipeline = _pipeline(runs, prefetch=1)
    for run in blocks[1]:
        run.gate.clear()
    pipeline.begin_epoch(blocks, epoch=0)
    pipeline.activate(blocks[0][0])
    assert blocks[1][0].started.wait(5)
    assert live_builder_threads(), "a prefetch build is in flight"

    # The guard waits for the in-flight build, then drops everything.
    opener = threading.Timer(0.2, lambda: [run.gate.set() for run in blocks[1]])
    opener.start()
    with pipeline.quiesced():
        assert live_builder_threads() == []
        assert not any(run.heavy_state_ready for run in runs)
        with pytest.raises(RuntimeError, match="fork guard"):
            pipeline.activate(blocks[0][0])
        with pipeline.quiesced():  # re-entrant
            assert live_builder_threads() == []
    opener.join()

    # After the guard the epoch resumes: the current block is rebuilt on use.
    pipeline.activate(blocks[0][0])
    assert all(run.heavy_state_ready for run in blocks[0])
    pipeline.shutdown()
    assert live_builder_threads() == []


def test_epoch_plan_published_inside_the_guard_starts_building_on_exit(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=1)
    with pipeline.quiesced():
        pipeline.begin_epoch(blocks, epoch=0)
        assert live_builder_threads() == []
        assert not any(run.started.is_set() for run in _all(blocks))
    assert blocks[0][0].started.wait(5)
    assert blocks[0][1].started.wait(5)
    pipeline.shutdown()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)
@pytest.mark.parametrize("persistent", [True, False])
def test_dataloader_forks_only_inside_the_guard(blocks, monkeypatch, persistent):
    runs = _all(blocks)
    pipeline = _pipeline(runs, prefetch=1)
    for run in blocks[1]:
        run.gate.clear()
    pipeline.begin_epoch(blocks, epoch=0)
    pipeline.activate(blocks[0][0])
    assert blocks[1][0].started.wait(5)

    seen: list[tuple[list[str], list[bool]]] = []
    original = DataLoader._get_iterator

    def spy(self):
        # The fork happens in here.
        seen.append((live_builder_threads(), [run.heavy_state_ready for run in runs]))
        return original(self)

    monkeypatch.setattr(DataLoader, "_get_iterator", spy)

    class _Squares(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(index * index)

    loader = ForkGuardedDataLoader(
        _Squares(),
        batch_size=2,
        num_workers=2,
        multiprocessing_context="fork",
        persistent_workers=persistent,
        fork_guard=pipeline.quiesced,
    )
    opener = threading.Timer(0.2, lambda: [run.gate.set() for run in blocks[1]])
    opener.start()
    expected = [[0, 1], [4, 9], [16, 25], [36, 49]]
    for _ in range(2):
        assert [batch.tolist() for batch in loader] == expected
    opener.join()

    forks = 1 if persistent else 2
    assert len(seen) == forks
    for threads, ready in seen:
        assert threads == [], "no builder thread during the fork"
        assert not any(ready), "no heavy state in the forked workers"
    pipeline.shutdown()


def test_loader_without_workers_does_not_touch_the_pipeline(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=1)
    pipeline.begin_epoch(blocks, epoch=0)
    pipeline.activate(blocks[0][0])

    class _Items(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> int:
            return index

    loader = ForkGuardedDataLoader(
        _Items(), batch_size=1, num_workers=0, fork_guard=pipeline.quiesced
    )
    assert [batch.tolist() for batch in loader] == [[0], [1]]
    assert blocks[0][0].heavy_state_ready, "nothing was released"
    pipeline.shutdown()


def test_build_failures_surface_when_the_block_is_reached(blocks):
    blocks[1][1].fail = True
    pipeline = _pipeline(_all(blocks), prefetch=1)
    pipeline.begin_epoch(blocks, epoch=0)
    pipeline.activate(blocks[0][0])
    with pytest.raises(ValueError, match="build of b1 failed"):
        pipeline.activate(blocks[1][0])
    pipeline.shutdown()


def test_runs_outside_the_plan_are_rejected(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=1)
    pipeline.begin_epoch(blocks[:1], epoch=0)
    with pytest.raises(RuntimeError, match="not in the current epoch plan"):
        pipeline.activate(blocks[2][0])
    with pytest.raises(ValueError, match="not registered"):
        pipeline.begin_epoch([[_StubRun("stranger")]], epoch=1)
    pipeline.shutdown()


def test_registering_a_run_twice_is_rejected(blocks):
    runs = _all(blocks)
    pipeline = _pipeline(runs, prefetch=1)
    with pytest.raises(ValueError, match="already belongs"):
        pipeline.register(runs[:1])


def _wait_until_ready(runs: list[_StubRun]) -> None:
    for run in runs:
        assert run.started.wait(5)
    deadline = threading.Event()
    for _ in range(100):
        if all(run.heavy_state_ready for run in runs):
            return
        deadline.wait(0.05)
    raise AssertionError("builds did not finish")


def test_a_new_plan_reuses_resident_runs_and_drops_the_rest(blocks):
    runs = _all(blocks)
    pipeline = _pipeline(runs, prefetch=1)
    # A plan published but never consumed (e.g. the DataLoader prestart).
    pipeline.begin_epoch(blocks, epoch=0)
    _wait_until_ready(blocks[0])

    pipeline.begin_epoch([blocks[0], blocks[2]], epoch=1)
    pipeline.activate(blocks[0][1])
    assert [run.builds for run in blocks[0]] == [1, 1], "resident runs are reused"
    assert blocks[2][0].started.wait(5), "the new plan's next block is prefetched"
    assert all(run.releases == 0 for run in blocks[1])  # never built, never planned

    # A plan without the resident runs drops them.
    pipeline.end_epoch()
    pipeline.begin_epoch([blocks[1]], epoch=2)
    assert all(run.releases >= 1 for run in blocks[0] + blocks[2])
    assert not any(run.heavy_state_ready for run in blocks[0] + blocks[2])
    pipeline.shutdown()


def test_epoch_records_account_for_waits_and_builds(blocks):
    pipeline = _pipeline(_all(blocks), prefetch=1)
    records: list[dict] = []
    pipeline.record_startup(light_build_s=2.5, n_runs=5, workers=4)
    pipeline.set_record_sink(records.append)  # replays the startup record
    pipeline.begin_epoch(blocks, epoch=3)
    for block in blocks:
        pipeline.activate(block[0])
    summary = pipeline.end_epoch()

    assert records[0] == {
        "kind": "startup",
        "light_build_s": 2.5,
        "n_runs": 5,
        "light_build_workers": 4,
    }
    block_records = [r for r in records if r["kind"] == "block"]
    assert [r["block_index"] for r in block_records] == [0, 1, 2]
    assert [r["n_runs"] for r in block_records] == [2, 2, 1]
    assert all(r["epoch"] == 3 and r["n_blocks"] == 3 for r in block_records)
    assert all(r["run_phase_s_mean"] == {"phase_s": 0.25} for r in block_records)
    assert summary["epoch"] == 3 and summary["blocks"] == 3
    assert summary["total_wait_s"] == pytest.approx(
        sum(r["wait_s"] for r in block_records)
    )
    assert records[-1] == summary
    # Every record is JSON-serializable without NaN.
    for record in records:
        json.dumps(record, allow_nan=False)
    # Multi-block epochs drop the last block at the end.
    assert not any(run.heavy_state_ready for run in _all(blocks))
    pipeline.shutdown()


def test_prefetch_and_worker_counts_are_validated():
    with pytest.raises(ValueError, match="prefetch_blocks"):
        RunBlockPipeline(prefetch_blocks=-1, block_build_workers=1)
    with pytest.raises(ValueError, match="block_build_workers"):
        RunBlockPipeline(prefetch_blocks=0, block_build_workers=0)


def test_block_build_records_append_as_jsonl(tmp_path: Path):
    append_block_build_record(tmp_path, {"kind": "block", "block_index": 0})
    append_block_build_record(tmp_path, {"kind": "epoch", "epoch": 0})
    lines = (tmp_path / BLOCK_BUILD_ARTIFACT_NAME).read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"schema_version": 1, "kind": "block", "block_index": 0},
        {"schema_version": 1, "kind": "epoch", "epoch": 0},
    ]
