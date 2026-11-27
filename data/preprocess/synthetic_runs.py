"""Build a synthetic corpus with one process per run.

Every step of the synthetic preprocessing works on each run separately, so a
run can be built from the raw corpus on its own. The parent process opens the
corpus, runs the corpus-level mobility QA, builds the first run to define the
output (variables, dtypes and attributes) and creates the store. Worker
processes each build one run, write its shards, read them back and return the
run-level values and the clinical positive-preservation counts. The parent then
writes the run-level arrays and the corpus attributes and packs the store.

Workers are spawned, not forked: Zarr keeps an event-loop thread, and a
spawned worker starts without the parent's open stores or task graphs. The
CPUs of the allocation are shared out between the workers: each worker's Zarr
codec pool gets ``cpus // write_workers`` threads, which compress the inner
chunks of a run in parallel. ``write_workers`` therefore sets the peak memory
(about one run of mobility per worker) and the threads fill the CPUs.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass, replace
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
import warnings

import dask
import dask.array as da
import numpy as np
import xarray as xr

from .zarr_layout import (
    RUN_DIM,
    RunWriteResult,
    collect_run_level,
    create_store,
    finish_store,
    prepare_for_store,
    write_run,
)

if TYPE_CHECKING:
    from .config import PreprocessingConfig
    from .pipeline import OfflinePreprocessingPipeline

POSITIVE_PRESERVATION_ATTR = "positive_clinical_preservation_json"
# Set in the environment of spawned workers, before NumPy loads its BLAS.
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class SyntheticRunResult:
    write: RunWriteResult
    evidence: dict[str, dict[str, int]]


@dataclass(frozen=True)
class _WorkerSetup:
    config: PreprocessingConfig
    store: Path
    run_ids: tuple[str, ...]
    static: dict[str, np.ndarray]
    codec_threads: int


_WORKER: dict[str, Any] = {}


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    # The parent prints the same steps and warnings once, for the first run.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        yield


@contextlib.contextmanager
def _single_thread_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in SINGLE_THREAD_ENVIRONMENT}
    os.environ.update(SINGLE_THREAD_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _init_worker(setup: _WorkerSetup) -> None:
    import zarr

    from .pipeline import OfflinePreprocessingPipeline

    # Zarr encodes and decodes chunks on this thread pool.
    zarr.config.set({"threading.max_workers": setup.codec_threads})
    with _quiet():
        pipeline = OfflinePreprocessingPipeline(setup.config)
        corpus = pipeline.synthetic_processor.open_corpus(
            str(setup.config.synthetic_path)
        )
    _WORKER.update(setup=setup, pipeline=pipeline, corpus=corpus)


def _available_cpus() -> int:
    # The Slurm CPU set of the step, not every CPU of the node.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _static_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _build_run(index: int) -> SyntheticRunResult:
    setup: _WorkerSetup = _WORKER["setup"]
    pipeline: OfflinePreprocessingPipeline = _WORKER["pipeline"]
    run_id = setup.run_ids[index]
    with _quiet():
        dataset = pipeline.build_synthetic_runs(_WORKER["corpus"], [run_id])
        if [str(value) for value in dataset[RUN_DIM].values] != [run_id]:
            raise ValueError(f"Run {run_id!r} built a dataset for other runs")
        evidence = json.loads(dataset.attrs.get(POSITIVE_PRESERVATION_ATTR, "{}"))
        prepared = prepare_for_store(dataset, setup.config.bitround_keepbits)
        result = write_run(setup.store, index, prepared)
    for name, expected in setup.static.items():
        if not _static_equal(expected, result.static[name]):
            raise ValueError(
                f"{name!r} differs between run {setup.run_ids[0]!r} and run "
                f"{run_id!r}; variables without run_id must not depend on the run"
            )
    return SyntheticRunResult(write=replace(result, static={}), evidence=evidence)


def _template(schema: xr.Dataset, run_ids: np.ndarray) -> xr.Dataset:
    """Expand a one-run dataset to every run; run data stays undefined."""
    for name, coord in schema.coords.items():
        if RUN_DIM in coord.dims and name != RUN_DIM:
            raise ValueError(f"Coordinate {name!r} along run_id is not supported")
    static_names = [
        name for name, var in schema.data_vars.items() if RUN_DIM not in var.dims
    ]
    with dask.config.set(scheduler="synchronous"):
        (static,) = dask.compute({name: schema[name].data for name in static_names})
    data_vars = {}
    for name, var in schema.data_vars.items():
        if RUN_DIM in var.dims:
            if var.dims[0] != RUN_DIM:
                raise ValueError(f"{name!r} must have run_id as its first dimension")
            data = da.empty((len(run_ids), *var.shape[1:]), dtype=var.dtype, chunks=-1)
        else:
            data = np.asarray(static[name])
        data_vars[name] = xr.Variable(var.dims, data, attrs=var.attrs)
    coords = {name: coord for name, coord in schema.coords.items() if name != RUN_DIM}
    coords[RUN_DIM] = xr.Variable(RUN_DIM, run_ids)
    return xr.Dataset(data_vars, coords=coords, attrs=schema.attrs)


def _sum_evidence(
    results: list[SyntheticRunResult],
) -> dict[str, dict[str, int]]:
    total: dict[str, dict[str, int]] = {}
    for result in results:
        for variable, counts in result.evidence.items():
            target = total.setdefault(variable, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                target[key] += int(value)
    return dict(sorted(total.items()))


def _run_workers(setup: _WorkerSetup, workers: int) -> list[SyntheticRunResult]:
    count = len(setup.run_ids)
    start = time.monotonic()
    results: list[SyntheticRunResult] = []

    def record(result: SyntheticRunResult) -> None:
        results.append(result)
        print(
            f"    wrote run {len(results)}/{count} "
            f"({result.write.run_id}, {time.monotonic() - start:.0f} s)"
        )

    if workers == 1:
        _init_worker(setup)
        try:
            for index in range(count):
                record(_build_run(index))
        finally:
            _WORKER.clear()
        return results

    with _single_thread_environment():
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(setup,),
        )
    try:
        futures = [pool.submit(_build_run, index) for index in range(count)]
        for future in as_completed(futures):
            record(future.result())
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return sorted(results, key=lambda result: result.write.run_index)


def write_synthetic_dataset(
    pipeline: OfflinePreprocessingPipeline, destination: Path
) -> list[SyntheticRunResult]:
    """Build every corpus run in ``config.write_workers`` processes and pack it."""
    config = pipeline.config
    processor = pipeline.synthetic_processor
    corpus = processor.open_corpus(str(config.synthetic_path))
    provenance = processor.corpus_provenance(corpus)
    run_ids = tuple(str(value) for value in corpus[RUN_DIM].values)
    if not run_ids:
        raise ValueError("The synthetic corpus has no runs to preprocess")

    from .pipeline import SYNTHETIC_CORPUS_PROVENANCE_ATTR

    print(f"Defining the output from run {run_ids[0]}...")
    schema = pipeline.build_synthetic_runs(corpus, list(run_ids[:1]))

    template = _template(schema, corpus[RUN_DIM].values).assign_attrs(
        {SYNTHETIC_CORPUS_PROVENANCE_ATTR: json.dumps(provenance, sort_keys=True)}
    )
    prepared = prepare_for_store(template, config.bitround_keepbits)
    store = create_store(prepared, destination)
    static = {
        name: np.asarray(var.data)
        for name, var in prepared.data_vars.items()
        if RUN_DIM not in var.dims
    }

    workers = min(config.write_workers, len(run_ids))
    codec_threads = max(1, _available_cpus() // workers)
    per_run = [name for name, var in prepared.data_vars.items() if RUN_DIM in var.dims]
    print(
        f"  Building and writing {len(per_run)} run variables for {len(run_ids)} "
        f"runs with {workers} worker(s), {codec_threads} codec thread(s) each"
    )
    setup = _WorkerSetup(
        config=config,
        store=store,
        run_ids=run_ids,
        static=static,
        codec_threads=codec_threads,
    )
    results = _run_workers(setup, workers)

    evidence = _sum_evidence(results)
    for variable, counts in evidence.items():
        print(
            f"  ✓ Positive preservation: {variable} "
            f"({counts['raw_positive_count']} positive observations)"
        )
    writes = [result.write for result in results]
    finish_store(
        store,
        destination,
        run_level=collect_run_level(prepared, writes),
        attrs={POSITIVE_PRESERVATION_ATTR: json.dumps(evidence, sort_keys=True)},
    )
    pipeline._log_postwrite_summary(writes)
    return results
