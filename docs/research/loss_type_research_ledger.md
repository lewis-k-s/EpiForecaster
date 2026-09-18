# Loss-type research ledger

## 2026-09-18: goal activated

The user authorized the plan and execution with rolling `acc_interactive`
allocations, an `acc_debug` fallback, and local copies of notable results.
The goal is active. The fixed research settings are in
`configs/research/loss_protocol_v1.yaml`.

The live MN5 check found no pending or running jobs for `bsc008913`. Account
`bsc08` permits `acc_interactive` with a two-hour, 40-CPU limit and `acc_debug`
with a two-hour limit. A one-GPU acceptance run needs 20 CPUs. No allocation
has been requested yet; numerical and integration checks must pass first.

The local dataset metadata show 435 dates from 2020-03-01 to 2021-05-09.
The period 2021-02-01 through 2021-05-09 is reserved from this loop's adaptive
decisions. It has been used in past research. Confirmation on it can support
only a retrospective claim. A validated fit/selection/calibration role manifest
and source-overlap audit remain gates for scientific runs.

Work in progress: common WIS selection and exports, Student-t CRPS, numerical
validity checks, and bounded MN5 acceptance. The first 23 numerical reference
tests passed. No forecast improvement has been measured or claimed.

Every notable result will be copied to
`outputs/loss_research/loss-type-research-v1/` with run, checkpoint, config,
source-bundle, environment, input, protocol, and artifact identities. Research
decisions will be appended here. Generated artifacts remain outside Git.

## 2026-09-18: scoring acceptance prepared

Implemented Student-t CRPS, fixed-grid WIS, explicit required-head checkpoint
selection, canonical probabilistic aggregate columns, and granular schema 4
with distribution parameters and observation support. These changes are in the
working tree; the immutable source snapshot records their exact bytes.

Validation: 152 focused tests passed; full suite 1,242 passed and two skipped;
repository Ruff passed. Best-effort Pyright reported five existing config
mapping-type issues, with no errors in the new distribution-score module.
The effective acceptance config and its temporal boundaries were checked.

Prepared run: `loss-score-acceptance-20260918-r1`. The immutable workflow dry-run
passed. Its container SHA-256 is
`b9789057bc686414649c5224bc908ada1556fba17486c8055a76b49cc61c48ba`.
The requested interactive allocation is `46079355`, with one GPU, 20 CPUs,
and one hour. It is pending as of the initial scheduler response. The workflow
will attach only after the allocation is running and its resources are checked.

## 2026-09-18: temporal-scaler defect found before execution

The data-role audit found that `EpiDataset` fits the biomarker scaler on all
dates even when `time_range` limits training windows. This violates temporal
separation. Prepared acceptance run R1 is superseded without execution. The
fix will limit scaler fitting to the training time range and will be tested
with extreme future observations. A new source snapshot is required.

The same audit confirmed a units caveat: clinical targets use log1p(per-100k),
but wastewater uses log1p(concentration). Existing wastewater metric names
contain the legacy `log1p_per_100k` suffix. The protocol now records the actual
units per head; no cross-head raw-unit equivalence is claimed.

The event and calibration module passed 14 focused tests. It has not yet been
connected to the role-manifest and campaign pipeline. It is not sufficient to
open the scientific execution gate.

The corrected source passed 1,257 tests (two skipped), all Ruff checks, and the
immutable workflow dry-run. Prepared replacement:
`loss-score-acceptance-20260918-r2`. Its SIF is unchanged; its source snapshot
contains the temporal-scaler fix. The interactive request remains pending with
no reported start estimate after approximately ten minutes. Use the authorized
debug fallback if the allocation is still pending at dispatch.

## 2026-09-18: acceptance coordinator repair and source audit

Cancelled unused pending interactive request `46079355`. Debug job `46079956`
for R2 failed before training after seven seconds on `as01r4b10`. The dispatch
record correctly names the SIF's Snakemake path, but the coordinator checked
that path on the host. Its empty stdout/stderr and terminal receipt are retained.
The fix restores SIF execution with the documented host Slurm bindings and
preserves batch/interactive completion receipts. All 81 workflow tests passed.

Replacement `loss-score-acceptance-20260918-r3` passed its immutable dry-run
and was submitted as debug job `46080277`. It is pending at this entry.
It contains the scoring and coordinator fixes, not the later preprocessing
changes. It remains a bounded acceptance run, not a candidate comparison.

The source audit found full-series preprocessing fits, future-observation
initialization, and weekly hospitalization totals stamped at week start.
See `loss_data_audit_20260918.md`. New parameter-fit bounds and causal stepping
passed focused perturbation tests. Prepared a separate raw input identity and
new preprocessing configuration. No raw input was altered and no new dataset
has been generated or remotely published yet. The scientific gate is closed.

Dated roles are declared in `configs/research/loss_roles_v1.json`. The research
loader validates their gaps and preprocessing identity. Its default test loader
is calibration; confirmation requires an explicit frozen-finalist config.
The original spatial matrix remains a maximum screen, subject to all-head
support and the short-period uncertainty limit. Unsupported evidence cannot
pass promotion merely because its point estimate improves.

## 2026-09-18: dataset rebuild authorized and inputs published

The user explicitly approved the new research dataset and directed the agent to
assume training authorization within this goal. This covers the planned MN5
training and evaluation loop and the approved new preprocessing identity.

All ten raw-input DVC objects were copied to the remote content-addressed store
without replacing existing objects. Remote MD5 values match the local objects.
Verification receipt: `outputs/loss_research/loss-type-research-v1/raw_remote_md5.txt`.

R3 job `46080277` failed at container startup after seven seconds on
`as01r1b29`: the optional `/opt/.snoopy/lib` mount source does not exist on that
node. The next source version mounts site plugin trees only when present.
It also removes an ineffective HOME environment override. No training ran.
The replacement interactive request is `46080792` (one GPU, 20 CPUs, two hours).

Validation after data roles and checkpoint retention: 1,285 tests passed, two
skipped. The site-mount change then passed all 81 workflow tests. Ruff and
`git diff --check` passed. Best-CRPS checkpoint retention is optional and does
not control WIS early stopping; explicit-score resume rejects changed score,
required-head, objective, family, or degrees-of-freedom settings.

## 2026-09-18: approved preprocessing queued; calibration contracts tested

Prepared and dry-ran `loss-preprocess-20260918-r1` from commit `17b24ec`.
The interactive request `46080792` remained pending with no start estimate;
it was cancelled without execution. Submitted preprocessing on `acc_debug` as
job `46081066`, with one GPU, 20 CPUs, and a 90-minute cap. Attach by SSH jump
only after the scheduler identifies an allocated node owned by this job.

R2 and R3 acceptance failure logs, configs, and dispatch/execution receipts
have been synced to the local acceptance evidence directory. These are failure
records, not forecast results. No notable improvement has been observed.

Added role-checked C0/C1/C2 calibration artifact fitting and scoring. Artifacts
bind the checkpoint SHA-256, role-manifest content hash, forecast identities,
per-head scale fits, and counts. Evaluation rejects fit-role reuse, altered
artifacts, mismatched checkpoints, role-crossing forecast windows, duplicate
identities, invalid observed predictions, and mixed distribution identities.
The canonical scored rows include WIS, CRPS, NLL, PIT, coverage, and width.
These APIs are tested but still need the campaign artifact integration.

Validation: 1,296 tests passed, two skipped; Ruff passed. Pyright passed for
research roles, distribution scores, event/calibration helpers, and calibration
artifacts. Acceptance configuration now pins two epochs and ten batches per
epoch. No scientific comparison is allowed before replacement data QA and
observed support checks pass.

## 2026-09-18: research outputs integrated; preprocessing mount repaired

Preprocessing R1 job `46081066` failed after 14 seconds on `as01r3b26`.
Snakemake started correctly, but its coordinator namespace lacked the SIF
checksum sidecar. Data staging and preprocessing did not run. Logs are saved
under `outputs/loss_research/loss-type-research-v1/preprocessing/r1/logs`.
The next source version mounts the immutable container directory read-only.
The failed run is preserved.

The `evaluation.research_outputs` workflow option now evaluates selection and
calibration from the same checkpoint. It publishes calibrated proper scores,
case event probabilities, fixed and fitted alert scores, source support counts,
and hash-bound manifests from one complete recoverable attempt. A failed
attempt does not publish a completion marker. Hospital event scores remain
unsupported pending a valid source aggregation contract. Tests cover role and
region exclusion, canonical artifact generation, failed publication, and
recovery without repeated evaluation.

Full suite: 1,304 passed, two skipped. A further artifact generation test passed
with all 13 calibration/event tests. Ruff and whitespace checks passed.
The protocol now names the replacement dataset and the March confirmation
boundary from the dated role manifest. Dataset QA remains a required gate;
no scientific loss comparison or improvement claim has been made.

## 2026-09-18: replacement run queued and fixed screen renderer tested

Prepared and dry-ran `loss-preprocess-20260918-r2` from source commit `69f7f88`.
Interactive request `46081822` remained pending with no start estimate and no
steps; it was cancelled unused. The authorized `acc_debug` fallback is job
`46081958` (one GPU, 20 CPUs, 90 minutes). Latest scheduler state: pending for
priority, with no node assigned. Continue this job; do not create a duplicate
because it is queued. When it starts, attach through `mn5` to the observed node.

`scripts/research/loss_screen.py` renders the 48 declared science identities.
It parses every override through the canonical model config and checks that
all non-treatment settings match. It refuses an unpinned replacement dataset.
It prepares templates only; allocation, training, and evaluation remain with
the immutable MN5 launcher. No screen has been submitted.

`evaluation/research_comparison.py` checks exact forecast support and observed
values before producing equal-head paired WIS and per-head diagnostics. It
resamples disjoint target-date blocks that include all regions and heads.
Before any new forecast scores have been obtained, pin a conservative minimum
of four 28-day blocks for a screening uncertainty interval (2,000 draws, seed
852). Shorter evidence is point evidence only and cannot pass promotion. The
short selection period, and possibly confirmation, can therefore support a
promising candidate but not a confirmed superiority claim. This helper does
not replace final event or noninferiority gates.

Validation: 1,315 tests passed, two skipped; Ruff passed. Pyright passed for all
new research modules. After a pandas typing adjustment, all eight comparison
tests passed again. No training has completed and no notable improvement has
been obtained. Keep the goal active through preprocessing QA, acceptance, and
paired training/evaluation within the declared budget.

## 2026-09-18: data-stage mount failure and complete research export fields

R2 job `46081958` ran on `as02r5b31` and failed after 27 seconds. Coordinator
preparation and data dependency hashing passed. Its first scientific job step
failed before DVC staging: the individual checksum-file bind had no destination
parent inside the contained runtime. The scientific wrapper now binds the
immutable container directory read-only, as the coordinator already does.
It also removes the ineffective HOME environment override. R2 logs are synced
under `outputs/loss_research/loss-type-research-v1/preprocessing/r2/logs`.
No preprocessing or model training completed in R2. The next run must be R3.
No live interactive or scheduled allocation was available after this failure.

JSON role and fold manifests now participate in generic scientific dependency
fingerprints. Research schema 1 now exports all 23 WIS quantiles, explicit
log-space target units, last-history-day forecast origins, fixed-rule event
decisions, and nullable fitted-rule decisions. Unsupported decisions remain
missing. These fields were added before any research forecasts were produced.
The immutable-workflow guide documents their meanings and artifact contract.

Validation: 1,317 tests passed, two skipped; Ruff, shell syntax, whitespace, and
research module type checks passed. Host Slurm client dependencies were also
checked inside the pinned SIF on the login node; this is a client namespace
check, not evidence of a successful compute-node science step.

## 2026-09-18: pre-submission publication audit

R3 was prepared from `83c6848` but was not submitted. A source audit found two
publication paths without host Slurm steps: dataset publication and optional
training-evidence retention. With the containerized coordinator, both require
an explicit `srun` step to invoke the host container wrapper. Those paths now
use the same host-step contract as other publication rules. The generated DAG
commands pass focused tests. All direct scientific-container rule calls were
checked for this boundary. Preserve R3 as an unexecuted superseded snapshot;
prepare R4 from this correction. The full suite is being rerun before dispatch.

R4 passed immutable preparation and remote dry-run from `e6d9b36`. Full validation
passed: 1,318 tests, two skipped. It is submitted on `acc_debug` as job
`46082766`, with one GPU, 20 CPUs, and a 90-minute cap. Submission and dry-run
receipts are saved under the local preprocessing `r4` evidence directory.
Continue this exact job and inspect its first science step before any further
retry. The comparison gate remains closed until preprocessing publication,
dataset support QA, and bounded model acceptance pass.

## 2026-09-18: bounded acceptance matrix ready

The renderer now supports `--stage acceptance`. After the replacement dataset
passes QA and its DVC identity is pinned, it creates eight acceptance templates:
Normal/Student-t by NLL/CRPS by mobility on/off, using fold 1 and seed 52.
Each run has two epochs, ten batches per epoch, early-stop patience one, and a
30-minute allocation cap. They use eager execution and the same protected data
roles and research outputs. Their scores are excluded from scientific selection.
The main screen retains the production compilation setting. Enforce the existing
four-GPU-hour cumulative acceptance cap when dispatching; the matrix is not an
authorization to exceed it.

Validation: 1,319 tests passed, two skipped; Ruff, Pyright for the renderer, and
whitespace checks passed. Preprocessing job `46082766` remains live and pending
for priority with no start estimate. Do not resubmit it while it is pending.
No new dataset, forecast score, or notable improvement has been obtained yet.

## 2026-09-18: raw inputs fetched; source mount parent corrected

R4 job `46082766` ran on `as01r4b30` for 45 seconds. Its data step fetched both
DVC targets (7,682 files) and DVC reported them up to date. It then rejected a
missing source mount parent: `outputs/loss_research/loss-type-research-v1`.
The container/runtime fixes passed this stage. Preprocessing did not begin.
An SSH jump was attempted while the job was live; first-host-key setup and the
short failure window prevented a completed attachment. Do not attach to the
ended allocation. Logs are synced in the local preprocessing `r4` directory.

Move the raw pointer from the repository root to
`outputs/loss_research/loss-type-research-v1/raw_inputs.dvc`, beside its payload.
The DVC hash remains `c65715e80dcc5947eb665c7f037ed9ed.dir`; no raw data changed.
The Git ignore exception covers only the pointer; payloads and logs remain
ignored. Local DVC status passes. Preparation now rejects a DVC mount parent
that is absent from the source inventory, even if an ignored local directory
happens to exist. This turns the observed cluster failure into a local check.

Full validation: 1,320 tests passed, two skipped; Ruff and whitespace checks
passed. Prepare the corrected input layout as R5, preserving all prior runs.
The goal remains active and no forecast comparison has been made.

R5 was prepared and dry-ran from `02947ae`, then submitted as `acc_debug` job
`46083788` with one GPU, 20 CPUs, and a 90-minute cap. Local receipts are in
`outputs/loss_research/loss-type-research-v1/preprocessing/r5`. Continue this
job; do not resubmit while it is pending or running. Use a fresh scheduler
check before attaching to its assigned node. The task-specific SSH host-key
file is `/private/tmp/epiforecaster-loss-known-hosts`; accept only new keys and
keep existing-key verification enabled. No science comparison has started.

## 2026-09-18: verified artifact reader and calibrated interval consistency

The research artifact reader now requires a checkpoint SHA-256 from a separate
training receipt or direct checkpoint hash. It checks completion, the protected
role contract, required inventory entries, file hashes, calibration identity,
and the selection-only C0/C1/C2 tables before analysis. It does not refit
calibration or read confirmation labels. Tests reject incomplete, changed,
misidentified, duplicate, and unsafe artifact paths.

An export audit found that calibrated rows retained raw interval bounds while
coverage, widths, and quantiles used the calibrated scale. The export now sets
interval and point-error columns from the same control. A regression test
checks bounds against quantiles and coverage against those bounds.

Validation: 1,329 tests passed, two skipped; Ruff, research module type checks,
and whitespace checks passed. R5 job `46083788` is still pending for priority.
No new dataset or forecast improvement is available. Continue the same job.

## 2026-09-18: approved dataset published and locally verified

R5 job `46083788` completed successfully on `as01r1b15` in 4 minutes 53 seconds.
The SSH jump through `mn5` succeeded while its allocation was active. DVC
staging, preprocessing, publication, and completion steps all exited zero.
The full logs, NetCDF dataset, effective config, QA, and publication receipt are
synced under `outputs/loss_research/loss-type-research-v1/preprocessing/r5`.

Published dataset `real_loss_fit20200930_v1.nc` has DVC MD5
`b5b684d5b9f31369895589c413f39291` and SHA-256
`b9e2b1424abb7f2b3294ab19939c8be7519f8643991530a9ffde52e28c9c1421`.
The local dataset, QA, and effective-config hashes match the receipt. The local
DVC pointer sits beside its payload in the research output root, and the
protocol now pins its MD5. The previous dataset is unchanged.

The dataset has 435 daily dates and 768 regions. The local role check verified
the complete daily grid, 60-day history, 28-day horizon/gaps, and fitted
preprocessing cutoff. Observed cells in fit/selection/calibration are:
cases 21,402/9,686/11,760; hospitalizations 12,775/3,780/3,022;
deaths 31,537/17,144/17,871; N1 wastewater 2,170/823/590.
No observed cell in these checks is non-finite. Confirmation labels were not
inspected. `local_support_qa.json` records the checks. Some municipality Kalman
fits used configured fallback variances; preserve the warnings in the log.
Full model sample eligibility remains an acceptance gate.

Requested one one-GPU, 20-CPU, one-hour `acc_interactive` allocation as job
`46084703` for the bounded acceptance matrix. No other user allocation was
active at the request. Keep its local PTY alive. Do not submit a duplicate
interactive request. There is still no trained comparison or measured forecast
improvement.

After pinning the published input, all 1,329 tests passed again (two skipped).
Ruff, DVC status for the new pointer, and whitespace checks passed. The eight
bounded acceptance templates were rendered under `acceptance/templates-v1`.

The persisted 10-fold manifest exactly matches all 161 valid target regions.
Each planned fold (1, 4, 5) has observed values for all four heads in both its
selection and calibration nodes. Counts are in `local_support_qa.json`.
This is a support check; bounded training must still check sample eligibility.

Prepared and remote-dry-ran from `c446239`:
- `loss-acceptance-normal-nll-m1-f1-s52-r4`;
- `loss-acceptance-student-t-crps-m1-f1-s52-r1`.

Their receipts are saved in the matching local `acceptance` directories. The
renderer outputs are ignored research artifacts, so preparation uses identical
copies in `/private/tmp` through the launcher's external-template interface.
The first attempt to prepare from an ignored in-repository template stopped
locally before transfer or run reservation. No source change was needed.
Neither acceptance run is submitted yet. Interactive job `46084703` remains
pending without a start estimate; its PTY session is `59193`. Recheck it before
any dispatch or fallback. Attach the Normal NLL run first, then Student-t CRPS
if the first run passes and allocation time permits. Use the same immutable
run IDs; do not prepare duplicate runs on a wait timeout.

The scheduler accounting snapshot covers all allocated jobs in this campaign:
393 GPU-seconds (0.1092 GPU-hours), including preprocessing and failed startup.
No pending allocation time is counted. Evidence is in `compute_budget.json`
and `scheduler_accounting_20260918.txt`; refresh after terminal runs.

## 2026-09-18: first acceptance run submitted on debug fallback

Interactive request `46084703` was still pending with no node, no active step,
and no start estimate after about nine minutes. It was cancelled while still
pending; its owning PTY exited. Submitted the existing immutable Normal NLL
run `loss-acceptance-normal-nll-m1-f1-s52-r4` on `acc_debug` as job `46085156`
(one GPU, 20 CPUs requested, 30-minute cap). The submission receipt is saved
with its preparation and dry-run receipts. Continue this exact job. Attach
through SSH only after a fresh scheduler check identifies its live node.
The prepared Student-t CRPS run remains unsubmitted until this pipeline check
passes. No forecast improvement has yet been measured.

## 2026-09-18: native failure during first training initialization

Job `46085156` ran on `as02r5b31`; SSH jump attachment succeeded while active.
It failed after 132 seconds. DVC staging passed. The training subprocess
resolved the 128/17/16 train/selection/calibration node split, then died with
`SIGSEGV` before an optimization step. This is a native-process failure, not
an observed non-finite loss or an empty-fold rejection. The original log has
no native traceback, so the cause is not yet established. Logs are synced in
`acceptance/normal-nll-m1-f1-s52-r4/logs`.

Enable Python fault handling and unbuffered output in the scientific container
wrapper before a new immutable diagnostic acceptance attempt. Preserve R4.
The next attempt will retain the same data, model, loss, seed, and limits;
only diagnostics change. Do not launch the prepared Student-t CRPS R1 while
the common initialization path remains unresolved. No forecast improvement
has been measured.

Diagnostic-wrapper validation passed: 1,329 tests, two skipped; Ruff and shell
syntax checks passed. This change improves failure evidence; it does not claim
to fix the native crash.

R5 `loss-acceptance-normal-nll-m1-f1-s52-r5` was prepared and remote-dry-ran
from `6b30ac1`, then submitted as `acc_debug` job `46086058` with the same
one-GPU/30-minute bound. The receipts are saved under the matching local
acceptance directory. This is the first diagnostic retry of the native failure.
Continue job `46086058`; a wait timeout is not evidence to restart it. Inspect
its fault trace if initialization fails. The existing Student-t CRPS R1 is
still unsubmitted and lacks the new diagnostics; preserve that prepared run.
Refresh the compute budget after terminal state. Allocated use through the
failed R4 run is 525 GPU-seconds (0.1458 GPU-hours).

## 2026-09-18: native traceback identifies the NetCDF4 reopen path

R5 diagnostic job `46086058` ran on `as01r2b22`; SSH jump attachment succeeded.
It failed after 135 seconds. The traceback places the fault in the native
`netCDF4` file open called by xarray's file manager, from
`EpiDataset.load_canonical_dataset` during construction of the training split.
Fold metadata reads had completed; the Dask worker threads were idle in the
captured trace. This narrows the failure to input reopening, before losses,
model forward calls, or optimizer steps. It does not establish the underlying
native-library defect. The trace is saved in the local R5 logs.

Switch the processed NetCDF4 read path to the already declared `h5netcdf`
backend. Preserve xarray's Zarr path and the published dataset. Two real-file
tests check decoded values, attributes, coordinates, masks, run selection,
and repeated lazy opens after a sibling reader closes. Local parity on the
published data passes for all non-mobility fields from fit through calibration
and complete mobility matrices on 2020-05-01, 2020-09-30, and 2021-01-31.
No confirmation labels were read. Evidence is in `netcdf_backend_parity.json`.
The pinned SIF's lock SHA-256 equals the local lock and includes h5netcdf/h5py.
A small earlier local netCDF4 read/close probe also passed, so local success is
not proof that the cluster crash is fixed. Require the next cluster acceptance
run to pass. This will be the second retry of the observed native failure.

Reader validation: 1,331 tests passed, two skipped; Ruff and whitespace checks
passed. Best-effort Pyright reports ten errors in unchanged tensor/topology
sections of `data/epi_dataset.py`, with none at the modified reader. Save the
new reader as a separate source identity before preparing R6.

R6 `loss-acceptance-normal-nll-m1-f1-s52-r6` was prepared and remote-dry-ran
from `a91476f`, then submitted as `acc_debug` job `46087093` with the same
one-GPU/30-minute bound. The receipts are saved under its local acceptance
directory. Continue this exact job and inspect its first training output.
Only claim the reader fix works after successful cluster initialization.
Cumulative allocated use through R5 is 660 GPU-seconds (0.1833 GPU-hours).
No model has completed training and no loss comparison is available yet.

## 2026-09-18: first complete training and research-evaluation acceptance

R6 job `46087093` completed on `as02r1b07` in 367 seconds, with all steps
exiting zero. SSH jump attachment succeeded. The h5netcdf reader passed the
former crash point, and the trainer built 1,942 fit samples, 17 selection
samples, and 16 calibration samples. Training saved best-WIS, best-CRPS, and
final checkpoints. The finite validation WIS was 0.440687; this is acceptance
evidence only and is excluded from scientific selection.

The complete outputs and logs are synced under
`acceptance/normal-nll-m1-f1-s52-r6`. Direct local hashing of `best_model.pt`
matches training receipt SHA-256
`bbfca498698c55f3a36bb73e150ae80be1b37f7eeb48ff2a34104646c7d84c21`.
The research artifact reader verifies all required files, completion markers,
checkpoint/calibration/role identities, and file hashes. It reads 2,358 scored
rows across all four heads and C0/C1/C2, plus 2,562 case-event rows. Evidence is
in `local_artifact_verification.json`. No confirmation evaluation was run.
Optional per-head plotting still assumes Zarr and reported a warning; the
canonical metric and event artifacts were generated and verified.

The production base's separate `compile_backward=true` remained active even
though acceptance set `compile=false`. The acceptance renderer now sets both
flags from the acceptance compilation choice; the full screen keeps production
settings. Preserve R6 as a successful bounded compiled acceptance. Also note
that the existing `max_batches` CLI path limits execution to one epoch, despite
the template's two-epoch ceiling; epoch 0 in its summary is the epoch index.
Neither observation changes the scientific screen's 50-epoch limit.

The renderer correction passes all 1,331 tests (two skipped), Ruff, and
whitespace checks. Next render a new acceptance template set and use a new
Student-t CRPS run with the h5netcdf reader and both compiler paths disabled.
The old prepared Student-t CRPS R1 remains unexecuted and superseded. No
scientific baseline comparison or notable forecast improvement is available.

## 2026-09-18: remaining acceptance matrix prepared; Student-t CRPS submitted

All seven remaining fixed acceptance cases were prepared and remote-dry-ran
from `aad037f`, with the h5netcdf reader and both compilation flags disabled.
The exact identities, template hashes, and states are recorded in
`acceptance/run_inventory_v2.json`. Per-run preparation/dry-run receipts are
saved locally. Normal NLL mobility-on is complete as R6; Student-t CRPS
mobility-on is R2; the other six cases are R1. Do not recreate these runs.

Requested rolling one-GPU/20-CPU/one-hour interactive job `46088104`. It had no
node or active step after about eight minutes and was cancelled while pending;
its PTY exited. Live policy checks show `acc_debug` has MaxJobsPU=1 and
MaxSubmitPU=1. Submit fallback runs one at a time. The interactive partition
is up and contains the single shared GPU login node `alogin3`; no slot was
assigned to this request. Node inventory access via `sinfo` is not permitted.
Do not infer zero capacity from that denied query or bypass its restriction.

Submitted `loss-acceptance-student-t-crps-m1-f1-s52-r2` on `acc_debug` as job
`46088534` (one GPU, 30-minute cap). Continue this exact job. After it passes
artifact verification, use the already prepared Normal CRPS mobility-off R1
case, then the remaining fixed acceptance cases. Retain the four-GPU-hour
acceptance limit and the full research budget. Cumulative allocated use through
Normal NLL R6 is 1,027 GPU-seconds (0.2853 GPU-hours), including preprocessing.

The Normal NLL R6 root config, source/input manifest, data manifest, and
training completion receipt are now also synced under `run_receipts`.
The recorded source bundle SHA-256 is
`2bceb40e6970b5babc21c795a3c86036c18dc4c9b18fe69fff9af31f25867a41`.
Its exported C0 WIS agrees with the trainer's score. This acceptance record
remains excluded from scientific selection; no notable forecast improvement
has been established.
