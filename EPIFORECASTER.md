# Native heterogeneous EpiForecaster

This worktree supports the source-native heterogeneous model only. The municipal
model and its training interface belong to the separate legacy worktree.

## Sources and queries

| Head | Source | Target support |
| --- | --- | --- |
| Cases | Municipality | Daily raw counts |
| Admissions | ABS | Complete Monday-Sunday weekly raw counts |
| Deaths | ABS | Daily all-cause counts |
| Wastewater | EDAR | Raw N1, N2, and IP4 assay concentrations |

A query contains a source ID, a forecast origin, and a routed head. ABS
admissions and deaths use the same source fold. Masks exclude missing targets.
Audited graph populations define clinical rates. Evaluation also reports signed
excess deaths against the 2017-19 calendar baseline.

## Model

`NativeSupportDataset` supplies source histories, masks, reporting ages,
calendar covariates, and optional vaccination features. The heterogeneous
encoder uses typed overlap and catchment relations across municipality, ABS,
comarca, MITMA, and EDAR nodes. Comarca is a context support, not a death target.
EDAR point locations and population catchments have separate meanings.

The shared transformer estimates SIRHD initial states, time-varying rates,
and observation context. The differentiable SIR roll-forward produces latent
trajectories. Observation heads apply the reporting kernels and source-specific
operators. Admissions use complete weekly intervals. Wastewater remains an
assay-level output. Future flow values are supervision metadata, not inputs.

Only `native.spatial_encoder=hetero` is supported. With mobility disabled,
OD messages are absent and no OD store is read. With vaccination disabled,
the three vaccination channels are absent from local and relation histories.
With wastewater disabled, no EDAR observations are loaded or queried.

Weekly admissions enter history only when the full Monday-Sunday interval is
complete at the forecast origin. A partial current week cannot reveal its
future total to a daily death query or a relation context.

## Splits and input identity

`data.build_native_source_sets` derives assignments from the same eligible
query inventory that the trainer uses. Its output is a DVC asset directory:
`assignments.parquet` contains typed source IDs and fold indices;
`manifest.json` contains small version-4 metadata. Generated membership files
are not Git source files.

Run `r` uses fold `r` for test, fold `(r+1) mod N` for validation, and all other
folds for training. Chronological origin intervals apply in addition to the
source roles. Earlier origins are removed unless their complete target interval
ends before the next split begins. Every active head must remain in every split.
Visible pre-origin context is allowed across source roles.

The manifest binds the processed input bytes, eligible query inventory,
window length, horizon, stride, mask thresholds, confirmation guard, source
availability, split seed, and fold policy. Training and evaluation reject stale
assets. The Snakemake DAG builds and publishes a run-owned split asset after
input staging and before training. Checkpoints retain the exact split metadata
and assignments for independent evaluation. Changed preprocessing requires a
new asset and a new immutable run.

## Training and evaluation

Local acceptance is bounded:

```bash
uv run python -m data.build_native_source_sets --config configs/train_epifor_smoke.yaml
uv run train epiforecaster --config configs/train_epifor_smoke.yaml
```

Substantive runs use `scripts/workflow/mn5_workflow.sh`; see
[MN5 workflow](docs/MN5_SNAKEMAKE.md). Do not run production configs locally.
The DAG owns final evaluation. `training.run_final_evaluation=false` keeps
per-epoch validation for checkpoint selection but prevents final validation
exports and test evaluation inside training. Explicit diagnostic test epochs
are separate and must be declared in advance.

Checkpoint selection uses mean-target validation WIS. Joint objective loss is
recorded separately. Normal and fixed five-degree Student-t forecasts support
CRPS. Clinical scores use `log1p(per-100k)`; wastewater uses
`log1p(concentration)`. A left-censored wastewater observation uses the configured
forecast distribution's probability below its detection limit. Training and
evaluation call the same censor score. WIS uses uncensored observations and is
reported with point, dispersion, underprediction, and overprediction components.

Standalone evaluation constructs data, model, and loss state without an
optimizer, scheduler, training-dynamics writer, or resume operation. It checks
the saved input and model identities. Canonical granular files are Parquet,
regardless of the training export format. Evaluation also writes aggregate
metrics, WIS summaries, excess deaths, and example-support artifacts. Reporting
loads these canonical outputs.

## Automated coverage

The bounded end-to-end tests derive splits, train through the CLI, load the
selected checkpoint, evaluate validation and test through the CLI, and read
all machine-readable outputs. Strict and non-strict runs cover feature toggles.
Focused unit tests cover weekly rounding, masks and population transforms,
distribution scores and gradients, duplicate counting, configuration parsing,
checkpoint selection, and failure contracts. Plot presentation is outside the
core coverage target.
