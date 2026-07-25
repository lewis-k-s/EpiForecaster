---
name: training-dynamics-analyze
description: Review EpiForecaster training behavior and author an evidence-backed HTML report from canonical training_dynamics artifacts, DVC and immutable-run provenance, tabular metrics, loss and gradient plots, gradient snapshots, adaptive GradNorm telemetry, timing, and PyTorch profiler traces. Use for completed, failed, pruned, non-finite-halted, local, MN5, cross-validation, or HPO training entities when the user wants convergence, stability, bottleneck, profiling, GradNorm, or artifact-lifecycle interpretation rather than raw statistics.
---

# Training Dynamics Analysis

Produce `report.html`. Do not stop after generating the evidence bundle.

## Resolve the exact training entity

Never auto-select a latest experiment. Resolve the exact immutable run and model,
fold, or trial directory requested by the user. For MN5 discovery or artifact
transfer, load `mn5-dispatch`.

Prefer a model/trial directory or `training_dynamics_manifest.json`; these retain
status, cadence, feature flags, row count, and artifact SHA-256. A direct JSONL
path is acceptable for crash recovery but is unmanifested evidence.

Prefer `training_provenance.json` emitted beside future Snakemake training
artifacts. If the entity remains inside its immutable run, the analyzer resolves
`manifest.yaml` automatically. If artifacts were copied without the sidecar,
provide the exact run manifest with `--run-manifest`.

Do not write analysis into an immutable run. Create a new, explicit report
directory under a local analysis/output root.

## Generate machine evidence

Run:

```bash
uv run --no-sync training-dynamics-analyze <entity> \
  --output-dir <new-empty-report-dir> \
  [--run-manifest <immutable-run>/manifest.yaml]
```

The analyzer must generate only:

- `analysis.json`;
- `provenance.json`;
- `bundle_manifest.json`;
- CSV files under `tables/`;
- plots under `figures/`.

It must not generate qualitative prose or a report. Treat coded signals as
review prompts, not conclusions.

Inventory the evidence coverage before drawing conclusions:

- sampled training loss and learning rate;
- epoch and validation history;
- total and component gradient norms;
- clipping threshold and clip factor;
- adaptive GradNorm controller telemetry;
- gradient-debug snapshots;
- batch/data/model timing;
- PyTorch profiler traces;
- immutable run, DVC input, dependency-contract, and evidence-retention
  provenance.

Say explicitly when evidence is missing. Do not reconstruct missing canonical
metrics from W&B, TensorBoard, console logs, or checkpoints.

## Review and interpret

Read the references required by the available evidence:

- loss and convergence: [references/loss-interpretation.md](references/loss-interpretation.md)
- gradients and GradNorm: [references/gradient-interpretation.md](references/gradient-interpretation.md)
- profiler and timing: [references/profiler-interpretation.md](references/profiler-interpretation.md)
- provenance lifecycle: [references/provenance-lifecycle.md](references/provenance-lifecycle.md)
- HTML composition: [references/html-report.md](references/html-report.md)

Inspect every generated figure. Do not infer the visual shape of a curve from
only its aggregate statistics.

Inspect the relevant CSV tables rather than copying aggregates from console
output. Use `bundle_manifest.json` as the inventory and source-path contract.
Build the provenance section from `provenance.json` and
`tables/provenance.csv`; do not infer DVC identity from a payload path or a Git
SHA.

Align signals by exact persisted step before relating them. Highlight meaningful
coincidences such as:

- a loss spike with clipping;
- a loss regime change with learning-rate or epoch transition;
- component-gradient collapse while total norm remains healthy;
- data-loading delay with lower observed throughput;
- compilation or synchronization inside a profiled step.

Treat correlations and coincidences as associations, not causes.

## Author the HTML report

Use [assets/report-template.html](assets/report-template.html) as the structural
and visual starting point. Write the completed file as `report.html` beside
`bundle_manifest.json`. Keep image paths relative to the report.

Fill all substantive prose during this review. Do not copy placeholder text,
embed generic canned conclusions, or delegate qualitative writing back to the
analyzer.

Lead with a short assessment: improving, plateaued, unstable, worsening, failed,
or insufficient evidence. Then support it with exact statistics and steps.

Separate:

1. observed evidence;
2. interpretation;
3. limitations;
4. testable follow-up checks.

Distinguish component gradient norms from the adaptive GradNorm task-balancing
controller. Squared component-norm shares describe gradient allocation, not
feature importance. Disabled components are not applicable; unsampled components
are missing evidence, not dead.

Do not diagnose overfitting from a training-only loss curve. Do not generalize a
short profiler window to whole-run throughput. Do not prescribe architectural or
optimization changes unless the persisted configuration and observations support
a testable hypothesis.

Verify before returning:

- all template tokens are gone;
- all local image links resolve;
- key claims match `analysis.json` or a named CSV table;
- provenance identities match `provenance.json`, including the explicit
  retention class of the training evidence;
- missing evidence is explicit;
- the HTML opens as a readable standalone report.

Return the HTML report path and summarize its most decision-relevant findings.
