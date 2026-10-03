# Single-run reports

The automatic report contains figures, tables, short labels, and source links.
It contains no interpretation or recommendations. An agent can read the report
and edit a separate copy to add a qualitative review.

## Workflow stage

Set `workflow.target: train_evaluate_report` before preparing an immutable run.
The main training and training smoke templates use this target by default.
Other templates must select it explicitly.

```text
train_model -> export_training_provenance -> evaluate_model -> build_run_report
```

Evaluation uses the selected checkpoint on validation, then test. It always
writes granular outputs for this target. Report generation uses one CPU, without
GPU injection, model loading, inference, or data staging. The output is
`<run-root>/outputs/reports/single_run/report.html`, with a `report_manifest.json`
that records the output hashes. Reports are derived artifacts. Scientific inputs
and the immutable run are not changed.

If an upstream rule fails, no report is produced. There is no error hook. At the
start of the report stage, one input check verifies completion, required files,
artifact hashes, and the training/evaluation checkpoint identity. Missing files
or a recorded incomplete training state produce only status and error-code
tables. These include completed and expected epoch counts. Early stopping can
produce a complete report with fewer epochs than the configured budget.
Malformed inputs and programming errors fail the stage.

## Content and split contracts

The native report includes:

- Saved `train_objective`, `val_wis`, selection scores and their definitions,
  and head scores. `test_wis` is a single selected-checkpoint point. Test scores
  never select the checkpoint. Optional `training_objective_diagnostic` stays
  separate from normal evaluation. Historical diagnostic test traces are read
  only when present in old records.
- Checkpoint head scores, horizon WIS, interval coverage, and example forecasts
  in raw units and target score space. Death examples include signed excess
  deaths against the saved baseline.
- Sampled loss and gradient telemetry, batch-size changes recorded in epoch
  rows, source/input identities, and evidence coverage counts.
- GPU statistics and figures per collection and device. Missing counters stay
  missing. Activity percentages use valid samples. The scope is the full worker
  invocation; no training/evaluation phase is inferred.

Evaluation saves the split contract, window lengths, checkpoint epoch/step,
example history, window support, selected rows, and availability counts. This
code is part of the tested evaluation core. Plotting only loads these exports.

For region splits, examples are selected within each split and head. Sources
use low, middle, and high mean saved WIS among supported candidates. Windows
use the most future observations, then the most uncensored history observations,
then the middle origin. Source IDs break score ties.

For temporal splits, validation defines the source cohort and rank labels.
Test uses those source IDs and selects supported test windows with the same
window rule. Test scores do not select new sources. If a source has no supported
test window, the availability table records the missing cohort member. A
standalone temporal test evaluation without validation selections exports no
example cohort.

The renderer also accepts legacy loss histories and saved head tables. Native
example and score exports that were not recorded are listed in the coverage
table. Old runs are not assumed to have the new export contract.

## Local build and render

To rebuild from a local materialization of a completed run:

```bash
uv run --no-sync python -m scripts.analysis.build_run_report \
  --run-root outputs/runs/<run> \
  --output-dir outputs/reports/<run>
```

Use `--model-root` and `--evaluation-dir` if these canonical artifacts were
materialized at different local paths. This command is a report builder, not
an upstream failure handler.

To render an existing evidence bundle without rebuilding its figures:

```bash
uv run --no-sync python -m scripts.analysis.render_run_report \
  --bundle outputs/reports/<run>/bundle_manifest.json
```

After `uv sync`, the entry points are `run-report-build` and `run-report-render`.
The renderer records input, renderer, and HTML hashes in `report.render.json`.
With fixed inputs and the same environment, repeated builds produce the same
report assets. No current date or generated review text is inserted.

## Evidence layout

`bundle_manifest.json` maps table and figure keys to local files. Its
`report_spec.json` defines presentation order and short labels. All assets and
local links must stay within the bundle directory. Copied metadata must match
its saved hash. Other source artifacts are recorded by path, size, and SHA-256
in `report_inputs.json`.

```json
{
  "schema_version": 2,
  "report_spec": "report_spec.json",
  "report_inputs": "report_inputs.json",
  "tables": {"checkpoint_heads": "tables/checkpoint_heads.csv"},
  "figures": {"loss_epochs": "figures/loss_epochs.png"}
}
```

The spec uses table and figure items. Tables can set `columns`, `labels`,
`decimals`, and `collapsed`. CSV source IDs keep leading zeros. Decimal settings
change display only. Figure items can set `caption` and `collapsed`. The spec
contains no review fields.
