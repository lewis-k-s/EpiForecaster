# Latent Rescaling and Gradient Recovery Plan

Status: implementation approved, local development in progress
Date: 2026-08-05
Branch: `codex/synthetic-pretraining-routing`

## Objective

Recover useful SIRHD latent training signal during synthetic pretraining without
changing the corrected observation corpus, mechanistic trajectories, model
architecture, observation losses, or residual-head capacity.

The first study isolates latent target semantics and loss scaling. It must not
combine scaling with adaptive weighting, a robust penalty, a rollout curriculum,
or a residual-path change.

## Evidence and working hypothesis

The current loss computes masked fraction-space MSE for S/I/H/R/D, averages the
available component losses, and applies `w_sird_supervision = 0.01`. Earlier
checkpoint-pinned diagnostics found that the aggregate latent gradient on the
selected SIR parameters was many orders of magnitude below the observation
gradient. I/H variation also collapsed.

Different numerical scales are a supported cause of this starvation, but they
are not the only possible cause. Masks, rollout Jacobians, gradient cancellation,
initial-state error, and free-rollout instability can also change each task's
influence. The study therefore measures gradient vectors and latent behavior; it
does not use loss magnitude alone as the success criterion.

## Scientific contracts

### Immutable source data

Use the corrected raw corpus only:

`data/files/raw_synthetic_observations_sirhd_reporting_v1_cb6b4e5.zarr`

The existing raw and processed DVC objects remain immutable. Adding a flow target
requires a new processed dataset identity and DVC artifact. Do not overwrite:

`data/processed/synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped.zarr`

The new processed dataset path is:

`data/processed/synthetic_sirhd_reporting_v1_cb6b4e5_holt_damped_latent_flow_v2.zarr`

The new preprocessing run must record the Git commit, source-bundle hash,
configuration hash, container identity, seeds, raw DVC identity, output path, and
output hash.

### Continuous precision

- Clinical observations remain continuous expected reported counts.
- Mechanistic states remain continuous float32 population states.
- The preprocessor converts latent states and flows to float32 population
  fractions. It does not round or integerize them.
- Wastewater remains raw concentration before its single downstream `log1p`.
- Missing observations use masks or NaNs. Positive raw observations must remain
  positive after preprocessing.
- Latent loss transforms and reductions run in FP32.

### Death-flow target

The model's cumulative `D_trajectory` has shape `[B, H+1]` and starts at
boundary-relative `D0 = 0`. The model's `death_flow` has shape `[B, H]` and each
value belongs to the destination date for one forecast transition.

Replace cumulative-D loss supervision with a separate destination-dated
`death_flow_target` task:

- Raw source of truth: `deaths_true(run_id, region_id, date)` from the corrected
  reporting-v1 corpus. It is the generator's continuous mechanistic daily-death
  flow, not the sparse reported `deaths` observation.
- Canonical processed name: `latent_death_flow_true`.
- Canonical processed dimensions: `(run_id, date, region_id)`.
- Canonical value: continuous float32 population fraction per model step.
- Batch shape: `[B, H]`.
- For forecast date `t+h`, supervise the generator flow over `(t+h-1, t+h]`.
- Rebase each retained cumulative `D_target` window to its boundary value for
  state diagnostics, but do not use it as the fifth loss task.

The immutable raw corpus was inspected with `xarray.open_zarr()`. For run 0,
destination-date alignment between `deaths_true` and
`latent_D_true.diff(date)` had mean absolute error `3.20e-7` raw persons and
maximum absolute error `4.61e-4`; origin-date alignment had mean absolute error
`2.65e-3`. The preprocessor must validate this contract across all runs within a
declared tolerance for float32 cumulative differencing. A full 20-run xarray
check in normalized fraction space found per-run maximum absolute errors between
`4.75e-10` and `8.46e-10`; preprocessing uses `atol=1e-8` and `rtol=1e-5`.

If the raw corpus does not contain `deaths_true`, stop and report a data-contract
failure. Do not silently infer an equivalent target. A derived `D[t+1] - D[t]`
target is valid only under a separately reviewed `dt` and projection contract.

All matched scaling arms must use this corrected death-flow target. The historic
unre-based cumulative-D loss can remain descriptive baseline evidence, but it is
not a causal comparator for the scaling experiment.

### Fixed training-split scales

Fit scales after the train/validation/test split is fixed. Do not fit them in the
preprocessor or from validation/test nodes.

For each latent task `c`, use:

\[
s_c = \max\left(
\operatorname{std}_{\text{train}}(x_c),
s_{c,\min}
\right).
\]

The scale population is the set of unique finite target values from selected
synthetic training runs, training nodes, and supervised destination dates. Include
valid structural zeros and exclude NaNs. Do not weight a date more often because
overlapping forecast windows contain it. Use state fractions for S/I/H/R and the
flow fraction for death flow.

Fit the statistic with float64 accumulators, then store float32 scale constants.
Write an immutable `latent_scale_receipt.json` beside the training artifacts. The
receipt must contain:

- schema version and fitting algorithm;
- task names, standard deviations, floors, final scales, counts, and units;
- dataset and DVC identity;
- run IDs, train-node identity, supervised-date identity, and split seed;
- configuration identity and receipt hash.

### Loss tasks and weights

Use these five loss tasks:

`S`, `I`, `H`, `R`, and `death_flow`.

For the first study, use squared error in every arm. Do not add Huber or
pseudo-Huber yet. Use the same hard masks and reductions in all arms.

Keep the total fixed latent coefficient at `0.01`. With five active complete
synthetic tasks, use fixed task coefficients of `0.002` each. Fail closed if a
configured synthetic-pretraining arm is missing one of the five targets.

The three arms are:

1. `unscaled_mse`
   - raw fraction-space MSE for all five corrected tasks;
2. `std_mse`
   - `T_c(x) = x / s_c` for all five tasks;
3. `asinh_mse`
   - `T_c(x) = x / s_c` for S and R;
   - `T_c(x) = asinh(x / s_c)` for I, H, and death flow.

Each component loss is:

\[
L_c = \operatorname{mean}\left[
\left(T_c(\hat{x}_c)-T_c(x_c)\right)^2
\right].
\]

The combined loss is:

\[
L_{\mathrm{latent}} =
w_S L_S + w_I L_I + w_H L_H + w_R L_R + w_F L_F.
\]

Persist both raw fraction-space MSE and the optimized transformed loss for every
task.

## Gradient telemetry contract

Run latent gradient telemetry as a periodic eager sidecar. Do not put repeated
`autograd.grad` calls inside the compiled training step.

Use one frozen parameter subset that contains the shared representation and the
X0/rate-producing parameters. Record its exact parameter names and hash in the run
artifacts.

For every sampled step and every latent task, persist:

- raw and transformed loss;
- fixed task coefficient;
- coefficient-weighted gradient-vector norm before clipping;
- cosine with the combined observation-loss gradient;
- pairwise latent-task cosines;
- target and prediction standard deviation and their ratio;
- the combined gradient norm before clipping;
- the actual global clipping factor;
- the combined gradient norm after clipping.

Global clipping acts on the already-summed gradient. Do not label a per-task value
as a measured post-clip contribution. If a counterfactual per-task post-clip norm
is shown, label it as the pre-clip vector multiplied by the observed common clip
factor.

## Evaluation contract

Evaluation computation stays in the tested evaluation pipeline. Plotting only
loads canonical outputs.

Report for I, H, and death flow, and retain S/R guardrails:

- MAE and RMSE in canonical fraction space;
- prediction and target standard deviation;
- prediction-to-target standard-deviation ratio;
- Pearson correlation and mean per-window Spearman correlation across the
  forecast horizon, with an eligible-series count;
- metrics by forecast horizon;
- boundary-state, one-step transition, and free-rollout results.

Implement and freeze two diagnostic protocols before using them for arm
selection:

1. Oracle-X0 free rollout: replace predicted boundary S/I/H/R with target boundary
   states and keep the model's predicted future rates.
2. Teacher-forced one-step transitions: reset the state to target S/I/H/R at each
   origin date, apply one predicted transition, and compare the next state and
   destination-dated flows.

Store these as canonical evaluation artifacts with checkpoint, dataset, split,
window, and diagnostic-protocol identities.

## Matched study contract

Each arm must use:

- the same immutable source bundle and container;
- the same new processed dataset and split plan;
- the same initialization checkpoint or identical seeded initialization;
- the same ordered training windows and batches;
- the same optimizer and scheduler state;
- the same observation objectives and fixed observation weights;
- the same gradient clipping and precision policy;
- at least one complete synthetic epoch;
- the same checkpoint-selection and evaluation rules.

The first comparison changes only the latent transform mode. Adaptive weighting,
residual changes, curriculum changes, and robust penalties remain disabled.

## Success criteria

The rescaling study succeeds only if it:

- increases coefficient-weighted latent gradient participation on the frozen
  mechanistic parameter subset;
- recovers I/H/death-flow variation and ranking;
- improves oracle-X0 and teacher-forced transition metrics;
- improves or stabilizes the 28-day free rollout;
- keeps S/R calibration within a tolerance declared before the MN5 jobs start;
- does not materially degrade observation forecasting metrics.

Before job submission, define numerical tolerances and the arm-selection rule in
the versioned study configuration. Do not select an arm from loss magnitude alone.

## Implementation sequence

### Phase 1: corrected flow data contract

1. Validate the raw flow variable name, dimensions, dates, dtype, non-negativity,
   and relation to the generator handoff.
2. Extend synthetic preprocessing to preserve the destination-dated flow as
   `latent_death_flow_true` in float32 fraction space.
3. Add preprocessing QA for finite/nonnegative flow, exact date alignment, and
   positive-value preservation.
4. Add `death_flow_target` to `EpiDatasetItem`, `EpiBatch`, collation, and
   `forward_batch` target dictionaries.
5. Slice it as `[range_end:range_end+H]`; keep state targets as
   `[range_end-1:range_end+H]`.
6. Replace the cumulative-D loss task with the destination-dated flow task.
7. Add focused preprocessing, dataset, collation, model-target, and loss tests.

### Phase 2: scale fitting and loss modes

1. Add canonical configuration fields and validation for the three latent modes,
   task weights, scale floor, and fitted-scale requirements.
2. Add a training-split scale fitter with unique-date accounting and float64
   accumulation.
3. Persist and reload the scale receipt for checkpoint resume and evaluation.
4. Implement FP32 raw, standard-deviation, and `asinh` component losses.
5. Persist per-component raw, transformed, and weighted values.
6. Add unit tests for transforms, masks, floors, weights, leakage prevention,
   receipt determinism, checkpoint resume, and missing-target failure.

### Phase 3: gradient telemetry

1. Define and test the frozen mechanistic parameter subset.
2. Extend the periodic sidecar to calculate per-loss gradients and cosines.
3. Persist pre-clip vectors' norms, the combined norm, clip factor, and actual
   combined post-clip norm.
4. Verify that telemetry leaves live optimizer gradients and model state unchanged.

### Phase 4: canonical diagnostics

1. Add prediction/target standard deviations and their ratio to latent evaluation.
2. Add destination-dated death-flow accuracy and rank metrics.
3. Add horizon-stratified canonical output.
4. Implement oracle-X0 and teacher-forced one-step protocols.
5. Add focused deterministic evaluation tests.

### Phase 5: immutable study preparation

1. Create a new processed dataset path and DVC pointer. Do not replace the existing
   reporting-v1 artifact.
2. Create three versioned MN5 study configurations from the production synthetic
   pretraining template.
3. Set each arm to one complete synthetic epoch for the first matched study.
4. Validate the immutable bundle, configuration hashes, dataset materialization,
   scale receipts, seeds, and expected outputs.
5. Submit batch jobs only after local acceptance and cluster preflight pass.
6. Analyze only terminal runs with complete manifests and canonical artifacts.

Adaptive GradNorm or ReLoBRaLo is a later phase. Add it only if the fixed-scale
study shows that I/H/death-flow gradients remain starved. Use the five nonnegative
latent losses and the same mechanistic parameter subset. Use PCGrad only if the
measured gradient cosines show directional conflict.

## Local validation gates

Run focused tests after each phase. Before immutable MN5 preparation, run:

```bash
uv run ruff check .
uv run pytest tests/
uv run pyright .
```

Pyright is best effort, but all new core training and evaluation paths must have
deterministic focused tests. Do not run production configurations locally.

## Current implementation boundary

Begin with Phase 1. Do not launch training during the data-contract and local-code
implementation stages. Preserve all existing cluster runs, bundles, processed
datasets, DVC objects, and unrelated dirty worktree changes.
