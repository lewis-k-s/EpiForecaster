"""Lazy per-block construction of heavy run state with background prefetch.

Curriculum training samples a block of ``run_block_size`` synthetic runs at a
time. A run's *light* state (series, masks, index maps; see ``EpiDataset``) is
built for every run at startup; its *heavy* state (dense mobility reduction,
k-hop closure, graph weights, context tables) is needed only by the main
process and is built here, per block, on demand:

- ``RunBlockPipeline.begin_epoch`` receives the epoch's block plan from the
  sampler and starts building the first block(s) in a small thread pool.
- ``activate`` (main process, once per batch) detects the first batch of a new
  block, releases the previous block's heavy and device state, submits the
  builds of the next ``prefetch_blocks`` blocks and waits only for the builds
  of the block it entered that are still unfinished.
- ``quiesced`` is the fork guard: DataLoader workers are forked, and a fork
  while a builder thread runs can leave a lock (allocator, zarr/blosc, GIL
  hand-off state) held forever in the child, and would hand the child a copy of
  heavy tensors that it keeps alive after the parent drops them. Inside the
  guard no builder thread exists and no heavy state is resident; builds
  requested meanwhile start when the guard exits.

Threads, not processes: a run's heavy build is dominated by zarr/blosc
decompression, numpy elementwise passes and torch tensor ops, all of which
release the GIL, and the result is ~0.5 GB of tensors per run that the main
process must read zero-copy. A process pool would pickle that through pipes
and fork (or spawn) alongside the DataLoader workers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

BUILDER_THREAD_PREFIX = "run-block-build"


class LazyRun(Protocol):
    """A dataset whose heavy state the pipeline builds and drops."""

    run_id: str
    heavy_state_ready: bool
    heavy_build_timings: dict[str, float]
    block_pipeline: RunBlockPipeline | None

    def build_heavy_state(self) -> float: ...

    def release_heavy_state(self) -> None: ...


@dataclass(frozen=True)
class _RunTiming:
    build_s: float
    started: float
    finished: float
    phases: dict[str, float]


def _timed_build(run: LazyRun) -> _RunTiming:
    started = time.perf_counter()
    run.build_heavy_state()
    finished = time.perf_counter()
    return _RunTiming(
        build_s=finished - started,
        started=started,
        finished=finished,
        phases=dict(run.heavy_build_timings),
    )


def live_builder_threads() -> list[str]:
    """Names of alive threads started by any ``RunBlockPipeline``."""
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(BUILDER_THREAD_PREFIX)
    ]


class RunBlockPipeline:
    """Builds and drops the heavy state of curriculum runs block by block.

    All methods are called from the main (consumer) thread; only the builds
    themselves run in pool threads. Runs must be registered once.
    """

    def __init__(self, *, prefetch_blocks: int, block_build_workers: int) -> None:
        if prefetch_blocks < 0:
            raise ValueError(f"prefetch_blocks must be >= 0, got {prefetch_blocks}")
        if block_build_workers < 1:
            raise ValueError(
                f"block_build_workers must be >= 1, got {block_build_workers}"
            )
        self.prefetch_blocks = prefetch_blocks
        self.block_build_workers = block_build_workers
        self._executor: ThreadPoolExecutor | None = None
        self._runs: dict[int, LazyRun] = {}
        self._plan: list[list[LazyRun]] = []
        self._block_of: dict[int, int] = {}
        self._current = -1
        self._epoch = -1
        self._pause_depth = 0
        self._builds: dict[int, Future[_RunTiming]] = {}
        self._timings: dict[int, _RunTiming] = {}
        self._wait_s: dict[int, float] = {}
        self._block_records: list[dict[str, Any]] = []
        self._record_sink: Callable[[dict[str, Any]], None] | None = None
        self._pending_records: list[dict[str, Any]] = []

    # -- registration and records ------------------------------------------

    def register(self, runs: Sequence[LazyRun]) -> None:
        """Attach ``runs`` to this pipeline; their heavy state is managed here."""
        for run in runs:
            if run.block_pipeline is not None:
                raise ValueError(f"Run {run.run_id} already belongs to a pipeline")
            run.block_pipeline = self
            self._runs[id(run)] = run

    def set_record_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        """Receive every profiling record (earlier ones are replayed first)."""
        self._record_sink = sink
        for record in self._pending_records:
            sink(record)
        self._pending_records.clear()

    def _emit(self, record: dict[str, Any]) -> None:
        # Fail on non-JSON values here rather than in a sink far from the cause.
        json.dumps(record, allow_nan=False)
        if self._record_sink is None:
            self._pending_records.append(record)
        else:
            self._record_sink(record)

    def record_startup(
        self, *, light_build_s: float, n_runs: int, workers: int
    ) -> None:
        """Record the wall time of the startup light build of all runs."""
        logger.info(
            "Light build of %d synthetic runs took %.1f s (%d threads)",
            n_runs,
            light_build_s,
            workers,
        )
        self._emit(
            {
                "kind": "startup",
                "light_build_s": light_build_s,
                "n_runs": n_runs,
                "light_build_workers": workers,
            }
        )

    # -- epoch lifecycle ----------------------------------------------------

    def begin_epoch(self, blocks: Sequence[Sequence[LazyRun]], epoch: int) -> None:
        """Adopt the epoch's block plan and start building its first block(s).

        Heavy state of runs the plan does not need soon is dropped; runs whose
        state is already resident (single-block epochs, an identical re-plan)
        are reused.
        """
        if self._current >= 0:
            self.end_epoch()
        plan = [list(block) for block in blocks]
        block_of: dict[int, int] = {}
        for index, block in enumerate(plan):
            for run in block:
                if id(run) not in self._runs:
                    raise ValueError(f"Run {run.run_id} is not registered")
                block_of[id(run)] = index
        self._plan = plan
        self._block_of = block_of
        self._current = -1
        self._epoch = epoch
        self._block_records = []
        self._wait_s = {}

        wanted = {
            id(run) for block in plan[: self.prefetch_blocks + 1] for run in block
        }
        for key, run in self._runs.items():
            if key not in wanted:
                self._discard(run)
        self._submit_through(self.prefetch_blocks - 1)

    def activate(self, run: LazyRun) -> None:
        """Ensure ``run``'s heavy state is built; called before each batch."""
        block = self._block_of.get(id(run))
        if block is None:
            raise RuntimeError(
                f"Run {run.run_id} is not in the current epoch plan; its heavy "
                "state is only built for planned blocks."
            )
        if block == self._current and run.heavy_state_ready:
            return
        if self._pause_depth:
            raise RuntimeError(
                "Heavy run state requested inside the DataLoader fork guard"
            )
        self._enter_block(block)

    def end_epoch(self) -> dict[str, Any] | None:
        """Close the epoch: log the last block and return the epoch summary.

        A multi-block epoch drops the last block and any unused prefetch; a
        single-block epoch keeps its runs resident for the next epoch.
        """
        if self._current < 0:
            self._drop_plan()
            return None
        last = self._current
        self._finish_block(last, release=len(self._plan) > 1)
        for index in range(last + 1, len(self._plan)):
            for run in self._plan[index]:
                self._discard(run)
        records = self._block_records
        summary: dict[str, Any] = {
            "kind": "epoch",
            "epoch": self._epoch,
            "blocks": len(records),
            "total_wait_s": sum(r["wait_s"] for r in records),
            "mean_block_build_s": sum(r["build_s"] for r in records) / len(records),
            "mean_run_build_s": _weighted_mean(records, "run_build_s_mean"),
            "max_run_build_s": max(r["run_build_s_max"] for r in records),
        }
        logger.info(
            "Epoch %d block builds: blocks=%d total_wait_s=%.1f "
            "mean_block_build_s=%.1f mean_run_build_s=%.1f",
            summary["epoch"],
            summary["blocks"],
            summary["total_wait_s"],
            summary["mean_block_build_s"],
            summary["mean_run_build_s"],
        )
        self._emit(summary)
        self._drop_plan()
        return summary

    # -- fork guard -----------------------------------------------------------

    @contextmanager
    def quiesced(self) -> Iterator[None]:
        """No builder thread and no heavy state while the body (a fork) runs.

        Re-entrant. Builds requested inside (``begin_epoch``) start on exit.
        """
        self._pause_depth += 1
        try:
            if self._pause_depth == 1:
                self._drain_and_release()
            yield
        finally:
            self._pause_depth -= 1
            if self._pause_depth == 0 and self._plan:
                self._submit_through(self._current + self.prefetch_blocks)

    def shutdown(self) -> None:
        """Stop builder threads and drop every run's heavy state."""
        self._drain_and_release()
        self._drop_plan()

    # -- internals ------------------------------------------------------------

    def _drop_plan(self) -> None:
        self._plan = []
        self._block_of = {}
        self._current = -1

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.block_build_workers,
                thread_name_prefix=BUILDER_THREAD_PREFIX,
            )
        return self._executor

    def _submit_through(self, last_block: int) -> None:
        """Start builds for every unbuilt run of blocks ``current..last_block``."""
        if self._pause_depth or not self._plan:
            return
        last_block = min(last_block, len(self._plan) - 1)
        for index in range(max(self._current, 0), last_block + 1):
            for run in self._plan[index]:
                if run.heavy_state_ready or id(run) in self._builds:
                    continue
                self._builds[id(run)] = self._ensure_executor().submit(
                    _timed_build, run
                )

    def _enter_block(self, block: int) -> None:
        previous = self._current
        if previous >= 0 and previous != block:
            if block < previous:
                raise RuntimeError(
                    f"Batches left block {previous} for earlier block {block}"
                )
            self._finish_block(previous, release=True)
        self._current = block
        wait_started = time.perf_counter()
        self._submit_through(block + self.prefetch_blocks)
        for run in self._plan[block]:
            future = self._builds.pop(id(run), None)
            if future is not None:
                self._timings[id(run)] = future.result()
        self._wait_s[block] = self._wait_s.get(block, 0.0) + (
            time.perf_counter() - wait_started
        )

    def _finish_block(self, block: int, *, release: bool) -> None:
        runs = self._plan[block]
        timings = [
            self._timings.pop(id(run)) for run in runs if id(run) in self._timings
        ]
        build_s = (
            max(t.finished for t in timings) - min(t.started for t in timings)
            if timings
            else 0.0
        )
        run_times = [t.build_s for t in timings]
        phase_names = sorted({name for t in timings for name in t.phases})
        record: dict[str, Any] = {
            "kind": "block",
            "epoch": self._epoch,
            "block_index": block,
            "n_blocks": len(self._plan),
            "n_runs": len(runs),
            "n_runs_built": len(timings),
            "build_s": build_s,
            "run_build_s_mean": sum(run_times) / len(run_times) if run_times else 0.0,
            "run_build_s_max": max(run_times, default=0.0),
            "wait_s": self._wait_s.pop(block, 0.0),
            "prefetch_blocks": self.prefetch_blocks,
            "block_build_workers": self.block_build_workers,
            "run_phase_s_mean": {
                name: sum(t.phases.get(name, 0.0) for t in timings) / len(timings)
                for name in phase_names
            },
        }
        logger.info(
            "Epoch %d block %d/%d: runs=%d built=%d build_s=%.1f "
            "(per run mean %.1f max %.1f) wait_s=%.1f",
            record["epoch"],
            block + 1,
            record["n_blocks"],
            record["n_runs"],
            record["n_runs_built"],
            record["build_s"],
            record["run_build_s_mean"],
            record["run_build_s_max"],
            record["wait_s"],
        )
        self._block_records.append(record)
        self._emit(record)
        if release:
            for run in runs:
                run.release_heavy_state()

    def _discard(self, run: LazyRun) -> None:
        """Cancel or wait out ``run``'s build, then drop its heavy state."""
        future = self._builds.pop(id(run), None)
        if future is not None and not future.cancel():
            future.exception()  # a finished build is dropped; failures resurface on rebuild
        self._timings.pop(id(run), None)
        if run.heavy_state_ready or future is not None:
            run.release_heavy_state()

    def _drain_and_release(self) -> None:
        for future in self._builds.values():
            if not future.cancel():
                future.exception()
        self._builds.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for run in self._runs.values():
            if run.heavy_state_ready:
                run.release_heavy_state()


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    total = sum(r["n_runs_built"] for r in records)
    if total == 0:
        return 0.0
    return sum(r[key] * r["n_runs_built"] for r in records) / total
