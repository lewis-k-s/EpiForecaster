"""Lazy per-block construction of heavy run state in builder processes.

Curriculum training samples a block of ``run_block_size`` synthetic runs at a
time. A run's *light* state (series, masks, index maps; see ``EpiDataset``) is
built for every run at startup; its *heavy* state (dense mobility reduction,
k-hop closure, graph weights, context tables) is needed only by the main
process and is built here, per block, on demand:

- ``RunBlockPipeline.begin_epoch`` receives the epoch's block plan from the
  sampler and starts building the first block(s).
- ``activate`` (main process, once per batch) detects the first batch of a new
  block, releases the previous block's heavy and device state, submits the
  builds of the next ``prefetch_blocks`` blocks and waits only for the builds
  of the block it entered that are still unfinished, installing each finished
  run on its dataset.
- ``quiesced`` is the DataLoader fork guard (see its docstring).

Builds run in a pool of ``block_builders`` **spawned** processes, each limited
to ``builder_threads`` native threads. A thread pool inside the trainer
measured 3-10x slower per run than the old serial build (MN5 job 48425142) and
stalled training: the builds' Python loops (diagnostics, k-hop) contend with the
training loop for the GIL, and torch/dask/blosc fan out inside the trainer
process. In a builder process neither applies. A builder receives only a small
``HeavyBuildRequest`` (config, run id, node sets, fitted mobility preprocessor),
re-reads the run's mobility from the zarr store, and returns the compact
tensors (~0.46 GB/run) through shared memory, not pickle bytes. The main process
installs them and derives the series-side context tables, which need the light
series it already holds.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from data.epi_dataset import builder_warmup, init_builder_process, run_heavy_build

logger = logging.getLogger(__name__)


class LazyRun(Protocol):
    """A dataset whose heavy state the pipeline builds and drops."""

    run_id: str
    heavy_state_ready: bool
    heavy_build_timings: dict[str, float]
    block_pipeline: RunBlockPipeline | None

    def heavy_build_request(self) -> Any: ...

    def install_heavy_state(self, result: Any) -> None: ...

    def release_heavy_state(self) -> None: ...


@dataclass(frozen=True)
class _RunTiming:
    build_s: float
    started: float
    finished: float
    pid: int
    threads: int
    phases: dict[str, float]


def spawn_builder_pool(block_builders: int, builder_threads: int) -> Executor:
    """The production pool: ``block_builders`` spawned processes.

    ``spawn`` (not fork): a builder starts from a clean interpreter and shares
    nothing with the trainer's threads, locks, CUDA context or DataLoader state.
    """
    return ProcessPoolExecutor(
        max_workers=block_builders,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_builder_process,
        initargs=(builder_threads,),
    )


class RunBlockPipeline:
    """Builds and drops the heavy state of curriculum runs block by block.

    All methods are called from the main (consumer) thread. Runs must be
    registered once. ``executor_factory`` and ``worker_fn`` default to the
    spawned builder pool and ``run_heavy_build``; tests substitute in-process
    equivalents.
    """

    def __init__(
        self,
        *,
        prefetch_blocks: int,
        block_builders: int,
        builder_threads: int,
        executor_factory: Callable[[int, int], Executor] | None = None,
        worker_fn: Callable[[Any], Any] = run_heavy_build,
        warmup_fn: Callable[[], Any] = builder_warmup,
    ) -> None:
        if prefetch_blocks < 0:
            raise ValueError(f"prefetch_blocks must be >= 0, got {prefetch_blocks}")
        if block_builders < 1:
            raise ValueError(f"block_builders must be >= 1, got {block_builders}")
        if builder_threads < 1:
            raise ValueError(f"builder_threads must be >= 1, got {builder_threads}")
        self.prefetch_blocks = prefetch_blocks
        self.block_builders = block_builders
        self.builder_threads = builder_threads
        self._executor_factory = executor_factory or spawn_builder_pool
        self._worker_fn = worker_fn
        self._warmup_fn = warmup_fn
        self._executor: Executor | None = None
        self._pool_started_at = 0.0
        self._warmups: list[Future[Any]] = []
        self._warmup_done_at: list[float] = []
        self._runs: dict[int, LazyRun] = {}
        self._plan: list[list[LazyRun]] = []
        self._block_of: dict[int, int] = {}
        self._current = -1
        self._epoch = -1
        self._pause_depth = 0
        self._builds: dict[int, Future[Any]] = {}
        self._timings: dict[int, _RunTiming] = {}
        self._wait_s: dict[int, float] = {}
        self._block_records: list[dict[str, Any]] = []
        self._record_sink: Callable[[dict[str, Any]], None] | None = None
        self._pending_records: list[dict[str, Any]] = []

    # -- builder pool ---------------------------------------------------------

    @property
    def in_flight_builds(self) -> int:
        """Builds submitted and not yet consumed or discarded."""
        return len(self._builds)

    def start_builders(self) -> None:
        """Start the builder processes and a warm-up task on each (idempotent).

        Returns immediately; the processes import their modules while the
        caller keeps working. ``record_startup`` reports when all are ready.
        """
        if self._executor is not None:
            return
        self._pool_started_at = time.perf_counter()
        self._executor = self._executor_factory(
            self.block_builders, self.builder_threads
        )
        # Dynamic spawning starts one process per submit while none is idle;
        # a fresh process cannot be idle before it has imported its modules.
        for _ in range(self.block_builders):
            future = self._executor.submit(self._warmup_fn)
            future.add_done_callback(
                lambda _f: self._warmup_done_at.append(time.perf_counter())
            )
            self._warmups.append(future)

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
        """Record the light-build wall time and the builder pool's startup time."""
        self.start_builders()
        pids = sorted({future.result() for future in self._warmups})
        pool_startup_s = max(self._warmup_done_at) - self._pool_started_at
        logger.info(
            "Light build of %d synthetic runs took %.1f s (%d threads); "
            "%d builder processes (%d threads each) ready after %.1f s",
            n_runs,
            light_build_s,
            workers,
            len(pids),
            self.builder_threads,
            pool_startup_s,
        )
        self._emit(
            {
                "kind": "startup",
                "light_build_s": light_build_s,
                "n_runs": n_runs,
                "light_build_workers": workers,
                "builder_pool_startup_s": pool_startup_s,
                "block_builders": self.block_builders,
                "builder_threads": self.builder_threads,
                "builder_pids": pids,
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
        """No build in flight and no heavy state resident while the body runs.

        Wrapped around every DataLoader worker fork. Builds no longer run in
        trainer threads, so the original hazard (a fork while a builder thread
        holds an allocator, OMP, zarr or blosc lock) is gone: the builders are
        separate spawned processes. What a fork still must not see is
        (1) heavy tensors: a forked worker inherits the parent's mappings
        (installed state and finished-but-uninstalled builder results) and keeps
        them alive after the parent drops them, which for per-epoch (non
        persistent) workers pins a block's memory for the whole epoch; and
        (2) a build that completes into a half-forked state. The guard therefore
        waits out in-flight builds, discards their results and releases all
        heavy state. The builder pool stays up. Re-entrant. Builds requested
        inside (``begin_epoch``) start on exit. With the default persistent
        workers the only guarded fork is the trainer-init prestart, when nothing
        is built yet, so it costs nothing.
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
        """Stop the builder processes and drop every run's heavy state."""
        self._drain_and_release()
        self._drop_plan()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._warmups.clear()

    # -- internals ------------------------------------------------------------

    def _drop_plan(self) -> None:
        self._plan = []
        self._block_of = {}
        self._current = -1

    def _submit_through(self, last_block: int) -> None:
        """Start builds for every unbuilt run of blocks ``current..last_block``."""
        if self._pause_depth or not self._plan:
            return
        last_block = min(last_block, len(self._plan) - 1)
        self.start_builders()
        assert self._executor is not None
        for index in range(max(self._current, 0), last_block + 1):
            for run in self._plan[index]:
                if run.heavy_state_ready or id(run) in self._builds:
                    continue
                self._builds[id(run)] = self._executor.submit(
                    self._worker_fn, run.heavy_build_request()
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
            if future is None:
                continue
            result = future.result()
            run.install_heavy_state(result)
            self._timings[id(run)] = _RunTiming(
                build_s=result.finished - result.started,
                started=result.started,
                finished=result.finished,
                pid=result.pid,
                threads=result.torch_threads,
                phases=dict(run.heavy_build_timings),
            )
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
            "block_builders": self.block_builders,
            "builder_threads": self.builder_threads,
            "builder_pids": sorted({t.pid for t in timings}),
            "builder_torch_threads": max((t.threads for t in timings), default=0),
            "run_phase_s_mean": {
                name: sum(t.phases.get(name, 0.0) for t in timings) / len(timings)
                for name in phase_names
            },
        }
        logger.info(
            "Epoch %d block %d/%d: runs=%d built=%d build_s=%.1f "
            "(per run mean %.1f max %.1f) wait_s=%.1f builders=%d",
            record["epoch"],
            block + 1,
            record["n_blocks"],
            record["n_runs"],
            record["n_runs_built"],
            record["build_s"],
            record["run_build_s_mean"],
            record["run_build_s_max"],
            record["wait_s"],
            len(record["builder_pids"]),
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
        for run in self._runs.values():
            if run.heavy_state_ready:
                run.release_heavy_state()


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    total = sum(r["n_runs_built"] for r in records)
    if total == 0:
        return 0.0
    return sum(r[key] * r["n_runs_built"] for r in records) / total
