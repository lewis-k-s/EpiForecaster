# Testing Plan

## Context

- Baseline (2026-02-09): `48%` line coverage across `models/`, `graph/`, `training/`, `data/`, `utils/`.
- Highest priority from current risk: model components and training behavior.
- Known issue to stabilize first: failing test around imported-risk/history dims in `data/epi_dataset.py`.
- Note: parts of graph processing are likely dead code and should not drive short-term test investment.

## Priority Order

1. Model components (`models/`)
2. Training orchestration (`training/`)
3. Preprocessing correctness for active pipelines (interpolation, smoothing, masks, age channels)
4. Utilities directly used in train/eval path
5. Graph processing only for currently active paths; dead code gets minimal smoke coverage or explicit deprecation follow-up

## Checklist

### Phase 0: Stabilize Baseline

- [ ] Reproduce and fix failing `tests/test_epi_dataset_getitem.py::test_imported_risk_gating`.
- [ ] Add regression test(s) locking expected tensor dimensions for imported-risk lags.
- [ ] Ensure full suite passes before coverage-target work.

### Phase 1: Model Component Unit Tests (Top Priority)

- [x] `models/epiforecaster.py`: forward pass shape contracts for minimal configs (with/without mobility and biomarker inputs).
- [x] `models/epiforecaster.py`: masking behavior tests (missingness masks and channel preservation).
- [x] `models/epiforecaster.py`: gradient smoke test (single backward pass on tiny batch).
- [x] `models/transformer_backbone.py`: causal/history window behavior and output dimensionality tests.
- [x] `models/transformer_backbone.py`: edge-case tests (short history, horizon 1, batch size 1).
- [x] `models/aggregators.py`: deterministic tests for each aggregator mode (shape + numeric sanity).
- [x] `models/mobility_gnn.py`: neighborhood aggregation shape/invariance tests on tiny fixed graphs.
- [x] `models/region_losses.py`: unit tests for each loss term, masking rules, and non-finite handling.
- [x] Add parametric tests to reduce duplication across model variants.

### Phase 2: Training Logic Unit Tests

- [x] `training/epiforecaster_trainer.py`: config parsing/validation for critical flags and incompatible settings.
- [x] `training/epiforecaster_trainer.py`: checkpoint discovery/resume branch tests.
- [x] `training/epiforecaster_trainer.py`: scheduler stepping and early-stop decision tests via mocks.
- [x] `training/epiforecaster_trainer.py`: loss composition wiring test (joint inference toggles).
- [x] `training/region2vec_trainer.py`: minimal fit-loop smoke test with tiny synthetic tensors.

### Phase 3: Preprocessing Regression Tests (Interpolation/Smoothing)

- [ ] `data/preprocess/pipeline.py`: unit test source-selection branches (`synthetic`, `cases`, `catalonia_cases`) with expected processor dispatch.
- [ ] `data/preprocess/pipeline.py`: verify optional source failure handling (`hospitalizations`/`deaths`) degrades to `None` and continues.
- [ ] `data/preprocess/pipeline.py`: test `valid_targets` density threshold behavior at exact boundary (`== threshold`) and NaN-only regions.
- [ ] `data/preprocess/pipeline.py`: test `edar_has_source` mask alignment/reindex with partial region overlap.
- [ ] `data/preprocess/pipeline.py`: test saved zarr chunk schema for `run_id`, `date`, and spatial dims.
- [ ] `data/preprocess/processors/hospitalizations_processor.py`: weekly-to-daily interpolation preserves weekly totals per municipality.
- [ ] `data/preprocess/processors/hospitalizations_processor.py`: age channel resets on observation days and caps at 14 during stale periods.
- [ ] `data/preprocess/processors/hospitalizations_processor.py`: Kalman fallback path when parameter fitting fails (use configured fallback variances).
- [ ] `data/preprocess/processors/catalonia_cases_processor.py`: smoothing does not emit inf/-inf and preserves monotonic non-negativity constraints.
- [ ] `data/preprocess/processors/catalonia_cases_processor.py`: mask/age semantics for missing days and leading missing windows.
- [ ] `data/preprocess/processors/edar_processor.py`: Tobit-Kalman censor flag semantics (`0=uncensored`, `1=censored`, `2=missing`) across mixed inputs.
- [ ] `data/preprocess/processors/edar_processor.py`: daily resampling inserts date gaps without fabricating positive measurements.
- [ ] `data/preprocess/processors/edar_processor.py`: age channel normalization and cap for long gaps before/after first observation.
- [ ] `data/preprocess/processors/alignment_processor.py`: EDAR expansion preserves NaN gaps and fills only mask/censor/age defaults.
- [ ] Add parametrized fixtures for sparse, dense, and gap-heavy synthetic time series to reuse across processors.

### Phase 4: Data Path Tests Used by Model

- [x] `data/epi_dataset.py`: `__getitem__` contracts for all enabled feature combinations.
- [x] `data/clinical_series_preprocessor.py` + `data/cases_preprocessor.py`: mask/value/age channel invariants.
- [x] `data/samplers.py`: curriculum and filtering behavior around edge bounds and empty selections.
- [x] Add negative tests for malformed/missing variables in active training datasets.

### Phase 5: Utilities in Critical Path

- [x] `utils/remote_runner.py`: script generation and argument escaping tests (no shell execution).
- [x] `utils/platform.py`: environment/SLURM detection branches.
- [x] `utils/tensor_core.py` and `utils/temporal.py`: pure-function edge-case tests.

### Phase 6: Graph Code (De-prioritized / Dead-Code Aware)

- [ ] Identify active vs dead graph modules (`graph/edge_processor.py`, `graph/node_encoder.py`, `data/preprocess/region_graph_preprocessor.py`).
- [ ] For active graph paths: add focused unit tests for critical transforms only.
- [ ] For dead graph paths: either
  - [ ] add lightweight smoke tests + `TODO(deprecate/remove)`, or
  - [ ] deprecate/remove and stop tracking as coverage target.

## Coverage Targets (After Baseline Is Green)

- [ ] `models/` to `>= 70%`
- [ ] `training/` to `>= 50%`
- [ ] `data/` remains `>= 65%` while increasing coverage in active data path
- [ ] `overall` to `>= 55%` without counting dead code as a hard blocker

## Execution Notes

- Use `uv run pytest` for all test runs.
- Prefer small deterministic fixtures and CPU-only tensor sizes.
- Avoid IO/network dependence in unit tests.
- Where dead code is confirmed, document it and exclude from near-term targets rather than adding high-maintenance tests.
