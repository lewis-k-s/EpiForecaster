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

## 2026-09-18: Student-t CRPS acceptance verified; mobility-off case submitted

Student-t CRPS mobility-on R2 job `46088534` completed on `as03r4b26` in
332 seconds. SSH jump attachment succeeded. Both compilation paths were
disabled, training and evaluation completed, and checkpoints, all evaluation
outputs, logs, root configuration, source/input manifest, and training receipts
are synced locally. Direct checkpoint hashing matches both training receipts:
`4b26aad24e59343ddb2f0ea478415f1c416253d318567fc65037191c950821da`.

The canonical artifact reader verifies all required hashes and protected roles.
There are 2,358 scored rows across the four heads and three controls, plus
2,562 case-event rows. Distribution identity is Student-t with df 5, and the
training objective is CRPS. Acceptance WIS is C0 0.423859, C1 0.381948,
C2 0.382481. These figures are pipeline checks, not scientific ranking or
promotion evidence. No confirmation evaluation was run. The optional plotting
warning is unchanged; canonical metrics and events are complete.

Two of eight acceptance cases are now complete and locally verified.
Submitted the existing `loss-acceptance-normal-crps-m0-f1-s52-r1` run as
`acc_debug` job `46089017` (one GPU, 30-minute cap). Continue this job; the
single-job queue was empty before submission. The other five unsubmitted
cases remain prepared. Use `acceptance/run_inventory_v2.json` for their exact
identities. No duplicate preparation is needed.

Updated scheduler accounting: 1,359 GPU-seconds (0.3775 GPU-hours), including
preprocessing and all failed startup attempts. Keep the goal active through the
remaining acceptance cases, paired scientific screen, and bounded confirmation.
No notable scientific improvement has yet been established.

## 2026-09-18: goal paused and continuation handover prepared

The user requested a goal pause and a handover for a smaller model. The goal
tool confirmed `paused`; the local scheduler monitor was stopped. No further
jobs were submitted. At 14:20:07 UTC, already submitted job `46089017` was
running on `as03r3b03`, with two seconds elapsed and its existing 30-minute
limit. It was not cancelled. Recheck its terminal state before continuation.

The fixed scientific screen has 48 rendered templates under
`outputs/loss_research/loss-type-research-v1/screen/templates-v1`; identity and
template-hash checks passed. No scientific screen run has been prepared or
submitted. Its dispatch gate still requires all eight acceptance cases.

Resume instructions, exact run identities, verification commands, data hashes,
budget gates, and incomplete confirmation work are in
`docs/research/loss_type_autoresearch_handover_20260918.md`. This pause supersedes
the earlier instruction in this ledger to keep the goal active. Do not resume
dispatch until the user resumes the goal.

## 2026-09-18: goal resumed; Normal CRPS mobility-off acceptance verified

The user resumed the goal. Live `sacct` on `mn5` shows job `46089017`
(`loss-acceptance-normal-crps-m0-f1-s52-r1`) COMPLETED with exit 0 after
5 minutes 4 seconds; it ran all four workflow steps clean. Artifacts, logs,
root config, manifest, data manifest, and completion receipt were synced to
`acceptance/normal-crps-m0-f1-s52-r1/`.

The canonical artifact reader verifies the checkpoint SHA-256
`e615499a7833bf1530c4e2338f63eb483de87c4179e5e3563c6a2bf67f9effb2` against
both the training receipt and the completion receipt. It reads 2,358 scored
rows across all four heads and C0/C1/C2, plus 2,562 case-event rows. The
pinned config confirms Normal CRPS, mobility off, both compilation flags
disabled, calibration evaluation role, and `mean_target_wis` selection.
Acceptance WIS C0/C1/C2 is 0.575002/0.469336/0.472429 — a pipeline check
only, excluded from selection. No confirmation evaluation was run.

Inventory state for the case is now `complete_verified`. Scheduler accounting
is refreshed: 1,663 GPU-seconds (0.4619 GPU-hours) through job 46089017.
Live limits on resume: `acc_debug` MaxJobsPU=1, MaxSubmitPU=1; queue empty.
Three of eight acceptance cases are complete and verified.

## 2026-09-18: Normal NLL mobility-off acceptance verified; case five submitted

Job `46090227` (`loss-acceptance-normal-nll-m0-f1-s52-r1`) COMPLETED with exit
0 after 4 minutes 53 seconds. Artifacts, logs, root config, manifest, data
manifest, and completion receipt were synced to
`acceptance/normal-nll-m0-f1-s52-r1/`.

The canonical artifact reader verifies checkpoint SHA-256
`f11739757c2b4967d69aaf84960e75e2924eb436742289585574e9c989286545` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Normal NLL, mobility off,
eager execution, calibration role. Acceptance WIS C0/C1/C2 is
0.573641/0.468483/0.471534 — excluded from selection. Inventory state is
`complete_verified`. Accounting is refreshed to 1,956 GPU-seconds
(0.5433 GPU-hours). Four of eight acceptance cases are complete and verified.

Submitted `loss-acceptance-normal-crps-m1-f1-s52-r1` as job `46092720` on
`acc_debug`. Continue this exact job; do not submit duplicates. Remaining
prepared cases after it: student-t NLL m0, student-t NLL m1, student-t CRPS m0.

## 2026-09-18: Normal CRPS mobility-on acceptance verified; case six submitted

Job `46092720` (`loss-acceptance-normal-crps-m1-f1-s52-r1`) COMPLETED with
exit 0 after 3 minutes 56 seconds. All outputs, logs, root config, manifest,
data manifest, and completion receipt are synced to
`acceptance/normal-crps-m1-f1-s52-r1/`.

The canonical artifact reader verifies checkpoint SHA-256
`ef17e9c5c28f0b6590800a1491e40d0c51a9a3dc682ddb16dd23dfba58030182` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Normal CRPS, mobility on,
eager execution, calibration role. Acceptance WIS C0/C1/C2 is
0.575522/0.469692/0.472778 — excluded from selection. Inventory state is
`complete_verified`. Accounting is refreshed to 2,192 GPU-seconds
(0.6089 GPU-hours). Five of eight acceptance cases are complete and verified.

Submitted `loss-acceptance-student-t-nll-m0-f1-s52-r1` as job `46094660` on
`acc_debug` after confirming the queue was empty. Continue this exact job.
Remaining prepared cases after it: student-t NLL m1, student-t CRPS m0.
All 48 screen templates are staged in `/private/tmp/loss-screen-templates-v1`
with hashes matching `screen.json`, ready for gate-open dispatch.

## 2026-09-18: Student-t NLL mobility-off acceptance verified; case seven submitted

Job `46094660` (`loss-acceptance-student-t-nll-m0-f1-s52-r1`) COMPLETED with
exit 0 after 4 minutes 44 seconds. All outputs, logs, root config, manifest,
data manifest, and completion receipt are synced to
`acceptance/student-t-nll-m0-f1-s52-r1/`.

The canonical artifact reader verifies checkpoint SHA-256
`71da928644dedae169cad1884c73d38f8d7b4a5d22d23b2eac66480a649c3a09` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Student-t NLL with df 5,
mobility off, eager execution, calibration role. Acceptance WIS C0/C1/C2 is
0.551053/0.470208/0.473628 — excluded from selection. Inventory state is
`complete_verified`. Accounting is refreshed to 2,476 GPU-seconds
(0.6878 GPU-hours). Six of eight acceptance cases are complete and verified.

Submitted `loss-acceptance-student-t-nll-m1-f1-s52-r1` as job `46096631` on
`acc_debug` after confirming the queue was empty. Continue this exact job.
One prepared case remains after it: student-t CRPS m0.

## 2026-09-18: Student-t NLL mobility-on acceptance verified; final case submitted

Job `46096631` (`loss-acceptance-student-t-nll-m1-f1-s52-r1`) COMPLETED with
exit 0 after 4 minutes 58 seconds. All outputs, logs, root config, manifest,
data manifest, and completion receipt are synced to
`acceptance/student-t-nll-m1-f1-s52-r1/`.

The canonical artifact reader verifies checkpoint SHA-256
`85b6bbca1500324e0ed5ddf40d4ea123c2016aa26998929733bd17ca5995c0e2` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Student-t NLL with df 5,
mobility on, eager execution, calibration role. Acceptance WIS C0/C1/C2 is
0.551598/0.470571/0.473763 — excluded from selection. Inventory state is
`complete_verified`. Accounting is refreshed to 2,774 GPU-seconds
(0.7706 GPU-hours). Seven of eight acceptance cases are complete and verified.

Submitted the final prepared case `loss-acceptance-student-t-crps-m0-f1-s52-r1`
as job `46098119` on `acc_debug`. Continue this exact job. On its verification
the eight-case acceptance gate opens the 48-identity screen.

## 2026-09-18: acceptance gate passed; screen dispatch open

Job `46098119` (`loss-acceptance-student-t-crps-m0-f1-s52-r1`) COMPLETED with
exit 0 after 5 minutes 12 seconds. All outputs, logs, root config, manifest,
data manifest, and completion receipt are synced to
`acceptance/student-t-crps-m0-f1-s52-r1/`.

The canonical artifact reader verifies checkpoint SHA-256
`429df07c13234af2f79c77a7f049cc4b5e48ebc8db3538889bbd5e0f9b38874e` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Student-t CRPS with df 5,
mobility off, eager execution, calibration role. Acceptance WIS C0/C1/C2 is
0.551055/0.470182/0.473636 — excluded from selection.

All eight acceptance identities in `run_inventory_v2.json` are now
`complete_verified`, each with a `local_artifact_verification.json`,
consistent checkpoint hashes across independent receipts, and full
config-flag checks. The acceptance gate is open: `dispatch_gate.json` now
records `acceptance_complete_dispatch_open`. Accounting is refreshed to
3,086 GPU-seconds (0.8572 GPU-hours), inside the four-GPU-hour acceptance cap.

The screen templates (48 identities, acc_debug one-GPU 120-minute runtime)
are staged in `/private/tmp/loss-screen-templates-v1` with verified hashes.
Screen dispatch has begun: `loss-b-normal-nll-m1-f1-s52-r1` was prepared and
remote-dry-ran, then submitted as `acc_debug` job `46099033`. This first run
measures the full-run cost before committing the remaining budget. No
scientific comparison or improvement claim exists yet; acceptance scores
remain excluded from selection.

## 2026-09-18: screen identity 1 verified; identity 2 submitted

Screen run `loss-b-normal-nll-m1-f1-s52-r1` (job `46099033`) COMPLETED with
exit 0 after 6 minutes 17 seconds. The 50-epoch ceiling was bounded by early
stopping: training dynamics show the run stopped after epoch 6 with patience
5. This measures the practical full-run cost at roughly 0.105 GPU-hours per
identity including staging and evaluation; the 120-minute cap is not binding
at this cost. All 48 identities project to about 5 GPU-hours, well inside
the 128 GPU-hour budget and the confirmation reserve.

The canonical artifact reader verifies checkpoint SHA-256
`bb69d75b0a3ac3831c24ebadef81e816aa28e21640d5e48872db891714375b80` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Pinned config confirms Normal NLL, mobility on,
fold 1, seed 52, full production compilation (both compiler flags active),
calibration evaluation role, and `mean_target_wis` selection. This is the
first scientific-selection-eligible run; WIS C0/C1/C2 is
0.472714/0.448533/0.449230. The screen source snapshot is
`719e577fa90b3d078161444f73e9da324489bd685935889fa7a8733ce66f08da`
(git `b5fdff09`); the source must stay frozen for all 48 identities.

Artifacts and receipts are synced under
`screen/normal-nll-m1-f1-s52-r1/`. Accounting is refreshed to 3,463
GPU-seconds (0.9619 GPU-hours).

The paired identity `loss-b-normal-crps-m1-f1-s52-r1` was prepared,
remote-dry-ran, and submitted as `acc_debug` job `46101000`. Dispatch now
follows `screen.json` order with event-driven `mn5_watch` monitoring; no
fixed-timer wakes remain. No cross-loss comparison is made before all
paired members and their verification complete.

## 2026-09-18: screen identity 2 verified; provenance-noise check resolved

Screen run `loss-b-normal-crps-m1-f1-s52-r1` (job `46101000`) COMPLETED with
exit 0 after 5 minutes 30 seconds, early-stopped. The canonical artifact
reader verifies checkpoint SHA-256
`ce300c78e9f00ea83a39f85a3e87b2ee4725ef560e1bea98f316d0444dc471d4` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal CRPS, mobility on, fold 1, seed 52, full
production compilation. WIS C0/C1/C2 is 0.457270/0.435058/0.436079.
Accounting is refreshed to 3,793 GPU-seconds (1.0536 GPU-hours).

The run's bundle `source_snapshot_sha256` differs from identity 1
(`056ac678…` vs `719e577f…`) at the same git SHA. The difference is dirty
worktree bytes only: the appended research ledger and the user's unrelated
`tex` modification. Provenance-matching was therefore verified at the
scientific-contract level: the `data` and `evaluation` dependency contracts
are hash-identical across the two runs, and the `training` contracts contain
byte-identical `sources` inventory with config differences restricted to
`model_id` and the treatment key `observation_loss`. All screen identities
are provenance-matched at the scientific level despite snapshot drift.
Future identity checks use contract `sources` equality plus treatment-key
config diff, not bundle snapshot equality.

`loss-b-student_t-nll-m1-f1-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46102182`. Per-identity WIS values remain
un-compared until formal paired analysis; no improvement claim is made.

## 2026-09-18: screen identity 3 verified; identity 4 submitted

Screen run `loss-b-student_t-nll-m1-f1-s52-r1` (job `46102182`) COMPLETED
with exit 0 after 5 minutes 19 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`22129ca1f5b54d8f1203993cd7db7ceeaa08f428618ce4173856afcd06f3175e` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t NLL df 5, mobility on, fold 1, seed 52,
full production compilation. WIS C0/C1/C2 is 0.451355/0.430929/0.434834.
Accounting is refreshed to 4,112 GPU-seconds (1.1421 GPU-hours).

`loss-b-student_t-crps-m1-f1-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46103376`. Three of 48 screen identities are
verified; per-identity WIS remains formally un-compared pending the complete
paired matrix and `research_comparison.py` analysis.

## 2026-09-18: screen identity 4 verified; identity 5 submitted

Screen run `loss-b-student_t-crps-m1-f1-s52-r1` (job `46103376`) COMPLETED
with exit 0 after 6 minutes 2 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`776699824f575c5aca4dbede871ba71d2f268f672f8472a4784a84697b89f3d4` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t CRPS df 5, mobility on, fold 1, seed
52, full production compilation. WIS C0/C1/C2 is
0.412586/0.407100/0.409024. Accounting is refreshed to 4,474 GPU-seconds
(1.2427 GPU-hours).

All four mobility-on fold-1 seed-52 identities are now verified. The first
mobility-off pair member `loss-b-normal-nll-m0-f1-s52-r1` was prepared,
remote-dry-ran, and submitted as `acc_debug` job `46104229`. Four of 48
screen identities are verified; formal paired comparison waits for the
complete matrix.

## 2026-09-18: screen identity 5 verified; identity 6 submitted

Screen run `loss-b-normal-nll-m0-f1-s52-r1` (job `46104229`) COMPLETED with
exit 0 after 6 minutes 19 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`26474e26fe3b1b52534858c1483e5c373beef721c70a69bb558d33f10a590736` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal NLL, mobility off, fold 1, seed 52, full
production compilation. WIS C0/C1/C2 is 0.459091/0.440995/0.446337.
Accounting is refreshed to 4,853 GPU-seconds (1.3389 GPU-hours).

`loss-b-normal-crps-m0-f1-s52-r1` was prepared, remote-dry-ran, and submitted
as `acc_debug` job `46105020`. Five of 48 screen identities are verified.

## 2026-09-18: screen identity 6 verified; identity 7 submitted

Screen run `loss-b-normal-crps-m0-f1-s52-r1` (job `46105020`) COMPLETED with
exit 0 after 4 minutes 56 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`a38990a461393956c0ec0dd04bbcba0ee2be09527cfe825546bcbc7d9275ab52` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal CRPS, mobility off, fold 1, seed 52, full
production compilation. WIS C0/C1/C2 is 0.322098/0.315100/0.314453.
Accounting is refreshed to 5,149 GPU-seconds (1.4030 GPU-hours).

`loss-b-student_t-nll-m0-f1-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46105810`. Six of 48 screen identities are
verified; the fold-1 seed-52 quadruple is half complete.

## 2026-09-18: screen identity 7 verified; identity 8 submitted

Screen run `loss-b-student_t-nll-m0-f1-s52-r1` (job `46105810`) COMPLETED
with exit 0 after 6 minutes 4 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`8e99783cc98131b7b6815d350b1ad568e78e31870a3befee30f2f7fd164b7c55` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t NLL df 5, mobility off, fold 1, seed
52, full production compilation. WIS C0/C1/C2 is
0.472855/0.454824/0.457396. Accounting is refreshed to 5,513 GPU-seconds
(1.4708 GPU-hours).

`loss-b-student_t-crps-m0-f1-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46106518`. Seven of 48 screen identities are
verified.

## 2026-09-18: screen identity 8 verified; fold-1 seed-52 octet complete

Screen run `loss-b-student_t-crps-m0-f1-s52-r1` (job `46106518`) COMPLETED
with exit 0 after 6 minutes 5 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`3e084967891f2596869f02e5faf4bb378c08f722fb4d13b0e4e2eb2269aeec0f` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t CRPS df 5, mobility off, fold 1, seed
52, full production compilation. WIS C0/C1/C2 is
0.361844/0.349997/0.350655. Accounting is refreshed to 5,878 GPU-seconds
(1.5376 GPU-hours).

The complete fold-1 seed-52 octet (four loss objectives × mobility on/off)
is now verified. Dispatch proceeds to seed 53: `loss-b-normal-nll-m1-f1-s53-r1`
was prepared, remote-dry-ran, and submitted as `acc_debug` job `46107358`.
Eight of 48 screen identities are verified.

## 2026-09-18: screen identity 9 verified; identity 10 submitted

Screen run `loss-b-normal-nll-m1-f1-s53-r1` (job `46107358`) COMPLETED with
exit 0 after 6 minutes 4 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`32704a89614b30a5403a26c179764fa311502a6641d766068637e1ac1b5e97a1` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal NLL, mobility on, fold 1, seed 53, full
production compilation. WIS C0/C1/C2 is 0.388005/0.373448/0.372570.
Accounting is refreshed to 6,242 GPU-seconds (1.6375 GPU-hours).

`loss-b-normal-crps-m1-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46108553`. Nine of 48 screen identities are
verified.

## 2026-09-18: screen identity 10 verified; identity 11 submitted

Screen run `loss-b-normal-crps-m1-f1-s53-r1` (job `46108553`) COMPLETED with
exit 0 after 6 minutes 59 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`699c05805324ae62df4480ef6e404055518fdd32f08114146e5e3d01046b44bd` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal CRPS, mobility on, fold 1, seed 53, full
production compilation. WIS C0/C1/C2 is 0.298722/0.289189/0.289971.
Accounting is refreshed to 6,649 GPU-seconds (1.7352 GPU-hours).

`loss-b-student_t-nll-m1-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46109466`. Ten of 48 screen identities are
verified.

## 2026-09-18: screen identity 11 verified; identity 12 submitted

Screen run `loss-b-student_t-nll-m1-f1-s53-r1` (job `46109466`) COMPLETED
with exit 0 after 6 minutes 17 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`e42c91ecdc7d7560d0313e8956b385a90dee40a191f219684e8d0a3ac2af60a6` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t NLL df 5, mobility on, fold 1, seed 53,
full production compilation. WIS C0/C1/C2 is 0.482141/0.454598/0.457647.
Accounting is refreshed to 7,026 GPU-seconds (1.8363 GPU-hours).

`loss-b-student_t-crps-m1-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46110659`. Eleven of 48 screen identities are
verified.

## 2026-09-18: screen identity 12 verified; identity 13 submitted

Screen run `loss-b-student_t-crps-m1-f1-s53-r1` (job `46110659`) COMPLETED
with exit 0 after 5 minutes 17 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`e2df9e86ec0bcd099eaec4db7e7dc417912567f158ecc267a8b36a19159ece60` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t CRPS df 5, mobility on, fold 1, seed
53, full production compilation. WIS C0/C1/C2 is
0.418966/0.411857/0.412342. Accounting is refreshed to 7,343 GPU-seconds
(1.9169 GPU-hours).

All four mobility-on fold-1 seed-53 identities are verified. The first
mobility-off seed-53 identity `loss-b-normal-nll-m0-f1-s53-r1` was prepared,
remote-dry-ran, and submitted as `acc_debug` job `46111519`. Twelve of 48
screen identities are verified.

## 2026-09-18: screen identity 13 verified; identity 14 submitted

Screen run `loss-b-normal-nll-m0-f1-s53-r1` (job `46111519`) COMPLETED with
exit 0 after 4 minutes 31 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`5dcad8d7234411b6016550594d3e9554d3059b3961da216a6d76352484bd4bc6` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal NLL, mobility off, fold 1, seed 53, full
production compilation. WIS C0/C1/C2 is 0.462252/0.443568/0.443030.
Accounting is refreshed to 7,614 GPU-seconds (1.9919 GPU-hours).

`loss-b-normal-crps-m0-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46112551`. Thirteen of 48 screen identities
are verified.

## 2026-09-18: screen identity 14 verified; identity 15 submitted

Screen run `loss-b-normal-crps-m0-f1-s53-r1` (job `46112551`) COMPLETED with
exit 0 after 6 minutes 9 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`99301e96c86fb8cce5c15ab5147229d4b78df37c1d4d8b6ebe1c2a90d2fe0bad` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Normal CRPS, mobility off, fold 1, seed 53, full
production compilation. WIS C0/C1/C2 is 0.304507/0.302362/0.302314.
Accounting is refreshed to 7,968 GPU-seconds (2.0625 GPU-hours).

`loss-b-student_t-nll-m0-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46113415`. Fourteen of 48 screen identities
are verified.

## 2026-09-18: screen identity 15 verified; identity 16 submitted

Screen run `loss-b-student_t-nll-m0-f1-s53-r1` (job `46113415`) COMPLETED
with exit 0 after 6 minutes 45 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`95488665a2265465c81c30f99208bf0fabb3b9a1dfa55f035c66e58bc21cb23a` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t NLL df 5, mobility off, fold 1, seed
53, full production compilation. WIS C0/C1/C2 is
0.368633/0.364927/0.366416. Accounting is refreshed to 8,373 GPU-seconds
(2.1750 GPU-hours).

`loss-b-student_t-crps-m0-f1-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46114386`. Fifteen of 48 screen identities
are verified; its verification completes the entire fold-1 matrix.

## 2026-09-19: fold-1 matrix complete; fold-4 dispatch begins

Screen run `loss-b-student_t-crps-m0-f1-s53-r1` (job `46114386`) COMPLETED
with exit 0 after 5 minutes 4 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`ca87ab931cb9981045210c4f12bce245ebf59082ce52622af0de21af1e37344f` against
both receipts and reads 2,358 scored rows across all four heads and C0/C1/C2,
plus 2,562 case-event rows. Student-t CRPS df 5, mobility off, fold 1, seed
53, full production compilation. WIS C0/C1/C2 is
0.303221/0.299424/0.299434. Accounting is refreshed to 8,601 GPU-seconds
(2.2383 GPU-hours).

The complete fold-1 matrix (16 identities: 4 objectives × 2 mobility × 2
seeds) is verified. Fold 4 begins: `loss-b-normal-nll-m1-f4-s52-r1` was
prepared, remote-dry-ran, and submitted as `acc_debug` job `46115553`.
Sixteen of 48 screen identities are verified.

## 2026-09-19: screen identity 17 verified; identity 18 submitted

Screen run `loss-b-normal-nll-m1-f4-s52-r1` (job `46115553`) COMPLETED with
exit 0 after 6 minutes 2 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`d99897e5b189474db86558cc798dce6fb84569a6c20f52fc156a917852312e47` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal NLL, mobility on, fold 4, seed 52, full
production compilation. WIS C0/C1/C2 is 0.485858/0.483626/0.484759. The
row counts differ from fold 1 because fold 4's selection and calibration
nodes have different sizes; the reader's role-contract and support checks
passed. Accounting is refreshed to 8,973 GPU-seconds (2.3417 GPU-hours).

`loss-b-normal-crps-m1-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46116617`. Seventeen of 48 screen identities
are verified.

## 2026-09-19: screen identity 18 verified; identity 19 submitted

Screen run `loss-b-normal-crps-m1-f4-s52-r1` (job `46116617`) COMPLETED with
exit 0 after 4 minutes 34 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`2fc7a980ae52b28a9254934e74736f068c8a9fa0f284c00965beee284b9aef12` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal CRPS, mobility on, fold 4, seed 52, full
production compilation. WIS C0/C1/C2 is 0.490652/0.483027/0.482787.
Accounting is refreshed to 9,129 GPU-seconds (2.3850 GPU-hours).

`loss-b-student_t-nll-m1-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46117470`. Eighteen of 48 screen identities
are verified.

## 2026-09-19: screen identity 19 verified; identity 20 submitted

Screen run `loss-b-student_t-nll-m1-f4-s52-r1` (job `46117470`) COMPLETED
with exit 0 after 5 minutes 19 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`a19125034dec9736990374793be436c2dc3295c3df6b75b9bee96322c5efca46` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t NLL df 5, mobility on, fold 4, seed 52,
full production compilation. WIS C0/C1/C2 is 0.468590/0.469259/0.469091.
Accounting is refreshed to 9,448 GPU-seconds (2.4553 GPU-hours).

`loss-b-student_t-crps-m1-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46118418`. Nineteen of 48 screen identities
are verified.

## 2026-09-19: screen identity 20 verified; identity 21 submitted

Screen run `loss-b-student_t-crps-m1-f4-s52-r1` (job `46118418`) COMPLETED
with exit 0 after 6 minutes 15 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`c959e4a7f7d32c627f1bf4166dfbfce6580b0d844a0423881ff9f1d016e9dee3` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t CRPS df 5, mobility on, fold 4, seed
52, full production compilation. WIS C0/C1/C2 is
0.421463/0.424379/0.423654. Accounting is refreshed to 9,828 GPU-seconds
(2.5594 GPU-hours).

The fold-4 seed-52 mobility-on quartet is complete. The first mobility-off
identity `loss-b-normal-nll-m0-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46119221`. Twenty of 48 screen identities are
verified.

## 2026-09-19: screen identity 21 verified; identity 22 submitted

Screen run `loss-b-normal-nll-m0-f4-s52-r1` (job `46119221`) COMPLETED with
exit 0 after 4 minutes 46 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`71c38e462b3748abd3764601fc1dc1540e4317549cfed4290493ca4ee9c74ce5` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal NLL, mobility off, fold 4, seed 52, full
production compilation. WIS C0/C1/C2 is 0.483810/0.483970/0.483767.
Accounting is refreshed to 10,100 GPU-seconds (2.6347 GPU-hours).

`loss-b-normal-crps-m0-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46120020`. Twenty-one of 48 screen identities
are verified.

## 2026-09-19: screen identity 22 verified; identity 23 submitted

Screen run `loss-b-normal-crps-m0-f4-s52-r1` (job `46120020`) COMPLETED with
exit 0 after 4 minutes 49 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`5e08eecdf0bf5fe20b7af867305768a2b716c6ec997d77df9be17dc902d490c2` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal CRPS, mobility off, fold 4, seed 52, full
production compilation. WIS C0/C1/C2 is 0.322450/0.320779/0.319146.
Accounting is refreshed to 10,360 GPU-seconds (2.7067 GPU-hours).

`loss-b-student_t-nll-m0-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46120744`. Twenty-two of 48 screen identities
are verified.

## 2026-09-19: screen identity 23 verified; identity 24 submitted

Screen run `loss-b-student_t-nll-m0-f4-s52-r1` (job `46120744`) COMPLETED
with exit 0 after 6 minutes 1 second, early-stopped. The canonical reader
verifies checkpoint SHA-256
`40af59a30e3c2ced393c1098f6ba7d2bf8529404da227bf25dad62b2aa984bf3` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t NLL df 5, mobility off, fold 4, seed
52, full production compilation. WIS C0/C1/C2 is
0.486556/0.476978/0.478986. Accounting is refreshed to 10,709 GPU-seconds
(2.8036 GPU-hours).

`loss-b-student_t-crps-m0-f4-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46121689`. Twenty-three of 48 screen
identities are verified.

## 2026-09-19: screen identity 24 verified; halfway point reached

Screen run `loss-b-student_t-crps-m0-f4-s52-r1` (job `46121689`) COMPLETED
with exit 0 after 6 minutes 7 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`65b51b62c33b74a9a2dd0e586e93d67757e5cf890e85621f4cc23f97e5a4ac4c` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t CRPS df 5, mobility off, fold 4, seed
52, full production compilation. WIS C0/C1/C2 is
0.320716/0.320784/0.320318. Accounting is refreshed to 11,083 GPU-seconds
(2.9075 GPU-hours).

The fold-4 seed-52 octet is complete: 24 of 48 screen identities are
verified, exactly halfway. Dispatch continues with seed 53: 
`loss-b-normal-nll-m1-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46122346`.

## 2026-09-19: screen identity 25 verified; identity 26 submitted

Screen run `loss-b-normal-nll-m1-f4-s53-r1` (job `46122346`) COMPLETED with
exit 0 after 6 minutes 14 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`0c224e08a0782c20ebbe1d957e46d27979a38d9aad2e1bc0982caa9603f8c39a` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal NLL, mobility on, fold 4, seed 53, full
production compilation. WIS C0/C1/C2 is 0.430570/0.441803/0.438129.
Accounting is refreshed to 11,457 GPU-seconds (3.0114 GPU-hours).

`loss-b-normal-crps-m1-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46123266`. Twenty-five of 48 screen
identities are verified.

## 2026-09-19: screen identity 26 verified; identity 27 submitted

Screen run `loss-b-normal-crps-m1-f4-s53-r1` (job `46123266`) COMPLETED with
exit 0 after 6 minutes 20 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`20905608ffb3c0380373695e211a8e44e3e3ba0028ba7c7dd14bc943fcab972a` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal CRPS, mobility on, fold 4, seed 53, full
production compilation. WIS C0/C1/C2 is 0.323092/0.327231/0.325066.
Accounting is refreshed to 11,880 GPU-seconds (3.1211 GPU-hours).

`loss-b-student_t-nll-m1-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46124023`. Twenty-six of 48 screen identities
are verified.

## 2026-09-19: screen identity 27 verified; identity 28 submitted

Screen run `loss-b-student_t-nll-m1-f4-s53-r1` (job `46124023`) COMPLETED
with exit 0 after 6 minutes 17 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`6147cc1ff0ee761a03533fce22f3e3ad506118c8b37b0fbbcda412681c8e5f3d` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t NLL df 5, mobility on, fold 4, seed 53,
full production compilation. WIS C0/C1/C2 is 0.401331/0.404201/0.405600.
Accounting is refreshed to 12,309 GPU-seconds (3.2353 GPU-hours).

`loss-b-student_t-crps-m1-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46124776`. Twenty-seven of 48 screen
identities are verified.

## 2026-09-19: screen identity 28 verified; identity 29 submitted

Screen run `loss-b-student_t-crps-m1-f4-s53-r1` (job `46124776`) COMPLETED
with exit 0 after 5 minutes 24 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`ed6d03b17f9e600c351be724fbb13046665a19b7d3207573c4be0c2e2730eb68` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t CRPS df 5, mobility on, fold 4, seed
53, full production compilation. WIS C0/C1/C2 is
0.441331/0.439681/0.436827. Accounting is refreshed to 12,879 GPU-seconds
(3.3556 GPU-hours).

The fold-4 seed-53 mobility-on quartet is complete. The first mobility-off
identity `loss-b-normal-nll-m0-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46125456`. Twenty-eight of 48 screen
identities are verified.

## 2026-09-19: screen identity 29 verified; identity 30 submitted

Screen run `loss-b-normal-nll-m0-f4-s53-r1` (job `46125456`) COMPLETED with
exit 0 after 5 minutes 52 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`aec50ed45bf91f53cb3aeff5d088cdb9472cb6bc079f469796db9f35ead088b2` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal NLL, mobility off, fold 4, seed 53, full
production compilation. WIS C0/C1/C2 is 0.484618/0.482089/0.481359.
Accounting is refreshed to 13,315 GPU-seconds (3.5292 GPU-hours).

`loss-b-normal-crps-m0-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46126221`. Twenty-nine of 48 screen
identities are verified.

## 2026-09-19: screen identity 30 verified; identity 31 submitted

Screen run `loss-b-normal-crps-m0-f4-s53-r1` (job `46126221`) COMPLETED with
exit 0 after 6 minutes 0 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`85848d70fae4d907641fd46dedcc33ec87a0b4d564e75c3da2179b204b8aa64d` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Normal CRPS, mobility off, fold 4, seed 53, full
production compilation. WIS C0/C1/C2 is 0.329919/0.330552/0.330578.
Accounting is refreshed to 13,750 GPU-seconds (3.6528 GPU-hours).

`loss-b-student_t-nll-m0-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46126904`. Thirty of 48 screen identities are
verified.

## 2026-09-19: screen identity 31 verified; identity 32 submitted

Screen run `loss-b-student_t-nll-m0-f4-s53-r1` (job `46126904`) COMPLETED
with exit 0 after 5 minutes 52 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`1d31ef59f51691f4107d55abc513d5719cd8599eb46480b3b7eced9d802d1d3c` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t NLL df 5, mobility off, fold 4, seed
53, full production compilation. WIS C0/C1/C2 is
0.486982/0.474338/0.476344. Accounting is refreshed to 14,190 GPU-seconds
(3.7750 GPU-hours).

`loss-b-student_t-crps-m0-f4-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46127482`. Thirty-one of 48 screen identities
are verified; its verification completes the entire fold-4 matrix.

## 2026-09-19: fold-4 matrix complete; fold-5 dispatch begins

Screen run `loss-b-student_t-crps-m0-f4-s53-r1` (job `46127482`) COMPLETED
with exit 0 after 4 minutes 44 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`dcda3ee105a1d520a7230776059a98b0f9a68d0bc0299fbf1ef63b767d82eff8` against
both receipts and reads 2,430 scored rows across all four heads and C0/C1/C2,
plus 2,274 case-event rows. Student-t CRPS df 5, mobility off, fold 4, seed
53, full production compilation. WIS C0/C1/C2 is
0.349389/0.343507/0.345255. Accounting is refreshed to 14,459 GPU-seconds
(3.8497 GPU-hours).

The complete fold-4 matrix (16 identities) is verified. Fold 5 begins:
`loss-b-normal-nll-m1-f5-s52-r1` was prepared, remote-dry-ran, and submitted
as `acc_debug` job `46128055`. Thirty-two of 48 screen identities are
verified.

## 2026-09-19: screen identity 33 verified; identity 34 submitted

Screen run `loss-b-normal-nll-m1-f5-s52-r1` (job `46128055`) COMPLETED with
exit 0 after 6 minutes 37 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`7a6b5a33c99345b550e1af98f1b3dfda26b83e15caa450985861dd4516473e78` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal NLL, mobility on, fold 5, seed 52, full
production compilation. WIS C0/C1/C2 is 0.439342/0.441775/0.438360. Fold 5's
selection and calibration node sizes differ from folds 1 and 4; the reader's
role-contract and support checks passed. Accounting is refreshed to 14,484
GPU-seconds (4.0233 GPU-hours).

`loss-b-normal-crps-m1-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46128690`. Thirty-three of 48 screen
identities are verified.

## 2026-09-19: screen identity 34 verified; identity 35 submitted

Screen run `loss-b-normal-crps-m1-f5-s52-r1` (job `46128690`) COMPLETED with
exit 0 after 5 minutes 17 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`92e1515b018f307ce78cd82e75aa35ed0b8505e48a754f2d2bb85ce6bec8a7f5` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal CRPS, mobility on, fold 5, seed 52, full
production compilation. WIS C0/C1/C2 is 0.464151/0.442958/0.441712.
Accounting is refreshed to 14,785 GPU-seconds (4.1069 GPU-hours).

`loss-b-student_t-nll-m1-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46129276`. Thirty-four of 48 screen
identities are verified.

## 2026-09-19: screen identity 35 verified; identity 36 submitted

Screen run `loss-b-student_t-nll-m1-f5-s52-r1` (job `46129276`) COMPLETED
with exit 0 after 4 minutes 34 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`b86a4df5b4589f5a3b15f62813c1907ea00bf680e206f9a1fd2d90022ae51378` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t NLL df 5, mobility on, fold 5, seed 52,
full production compilation. WIS C0/C1/C2 is 0.458456/0.434561/0.435171.
Accounting is refreshed to 15,001 GPU-seconds (4.1669 GPU-hours).

`loss-b-student_t-crps-m1-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46129987`. Thirty-five of 48 screen
identities are verified.

## 2026-09-19: screen identity 36 verified; identity 37 submitted

Screen run `loss-b-student_t-crps-m1-f5-s52-r1` (job `46129987`) COMPLETED
with exit 0 after 4 minutes 34 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`147bf8a5575a994fe51bc7bc4ddaaad5c349e0c13dc6794acec99e5d1ddf8a92` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t CRPS df 5, mobility on, fold 5, seed
52, full production compilation. WIS C0/C1/C2 is
0.463918/0.441496/0.440777. Accounting is refreshed to 15,173 GPU-seconds
(4.2147 GPU-hours).

The fold-5 seed-52 mobility-on quartet is complete. The first mobility-off
identity `loss-b-normal-nll-m0-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46130669`. Thirty-six of 48 screen identities
are verified.

## 2026-09-19: screen identity 37 verified; identity 38 submitted

Screen run `loss-b-normal-nll-m0-f5-s52-r1` (job `46130669`) COMPLETED with
exit 0 after 4 minutes 18 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`0666a162a27d663326669dc9a7957ce3a0ca1f1d01af681bbf391874b6f1294d` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal NLL, mobility off, fold 5, seed 52, full
production compilation. WIS C0/C1/C2 is 0.459083/0.432967/0.435042.
Accounting is refreshed to 15,216 GPU-seconds (4.2267 GPU-hours).

`loss-b-normal-crps-m0-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46131338`. Thirty-seven of 48 screen
identities are verified.

## 2026-09-19: screen identity 38 verified; identity 39 submitted

Screen run `loss-b-normal-crps-m0-f5-s52-r1` (job `46131338`) COMPLETED with
exit 0 after 4 minutes 48 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`434a82c83337c305f18af4aa81321e43086dd7704d2627df73bf7334232e7dbc` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal CRPS, mobility off, fold 5, seed 52, full
production compilation. WIS C0/C1/C2 is 0.336354/0.338058/0.338583.
Accounting is refreshed to 15,497 GPU-seconds (4.3047 GPU-hours).

`loss-b-student_t-nll-m0-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46131908`. Thirty-eight of 48 screen
identities are verified.

## 2026-09-19: screen identity 39 verified; identity 40 submitted

Screen run `loss-b-student_t-nll-m0-f5-s52-r1` (job `46131908`) COMPLETED
with exit 0 after 6 minutes 3 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`426c3e60ebd3a40ea6266bc2ebd1fac9ac6bd37efaf6cfcb880c4d2e53738000` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t NLL df 5, mobility off, fold 5, seed
52, full production compilation. WIS C0/C1/C2 is
0.453771/0.427273/0.428725. Accounting is refreshed to 15,806 GPU-seconds
(4.3906 GPU-hours).

`loss-b-student_t-crps-m0-f5-s52-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46132563`. Thirty-nine of 48 screen
identities are verified.

## 2026-09-19: screen identity 40 verified; identity 41 submitted

Screen run `loss-b-student_t-crps-m0-f5-s52-r1` (job `46132563`) COMPLETED
with exit 0 after 6 minutes 9 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`cd864bc164c54235cde1d4a6f408695d0db2cf21ca1064ed17f1961d3f967f63` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t CRPS df 5, mobility off, fold 5, seed
52, full production compilation. WIS C0/C1/C2 is
0.332487/0.342138/0.344488. Accounting is refreshed to 16,415 GPU-seconds
(4.5600 GPU-hours).

The fold-5 seed-52 octet is complete. Dispatch continues with seed 53:
`loss-b-normal-nll-m1-f5-s53-r1` was prepared, remote-dry-ran, and submitted
as `acc_debug` job `46133151`. Forty of 48 screen identities are verified;
eight remain.

## 2026-09-19: screen identity 41 verified; identity 42 submitted

Screen run `loss-b-normal-nll-m1-f5-s53-r1` (job `46133151`) COMPLETED with
exit 0 after 4 minutes 37 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`5049d55d84c6d5744a8ffc1975e7cf922461029eac22e7981b4f27b1509177fa` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal NLL, mobility on, fold 5, seed 53, full
production compilation. WIS C0/C1/C2 is 0.460051/0.440099/0.439779.
Accounting is refreshed to 16,607 GPU-seconds (4.6131 GPU-hours).

`loss-b-normal-crps-m1-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46133598`. Forty-one of 48 screen identities
are verified.

## 2026-09-19: screen identity 42 verified; identity 43 submitted

Screen run `loss-b-normal-crps-m1-f5-s53-r1` (job `46133598`) COMPLETED with
exit 0 after 5 minutes 21 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`7dbef5ad31c23db62b1f1e589c496f35678e1fd3b2a507a01e1cf48758b2f6b9` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal CRPS, mobility on, fold 5, seed 53, full
production compilation. WIS C0/C1/C2 is 0.464954/0.441113/0.440940.
Accounting is refreshed to 16,815 GPU-seconds (4.6708 GPU-hours).

`loss-b-student_t-nll-m1-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46134341`. Forty-two of 48 screen identities
are verified.

## 2026-09-19: screen identity 43 verified; identity 44 submitted

Screen run `loss-b-student_t-nll-m1-f5-s53-r1` (job `46134341`) COMPLETED
with exit 0 after 4 minutes 36 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`d90dec52e329ab640b35cba04b8eea4871f42cbf74d330b5fda098028828a7a5` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t NLL df 5, mobility on, fold 5, seed 53,
full production compilation. WIS C0/C1/C2 is 0.457971/0.436825/0.437414.
Accounting is refreshed to 17,019 GPU-seconds (4.7275 GPU-hours).

`loss-b-student_t-crps-m1-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46134851`. Forty-three of 48 screen
identities are verified.

## 2026-09-19: screen identity 44 verified; identity 45 submitted

Screen run `loss-b-student_t-crps-m1-f5-s53-r1` (job `46134851`) COMPLETED
with exit 0 after 4 minutes 40 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`5b9ad9ca72b584ee75d8c4f9e9e07d6e4b6d16bb651977ad768f290456f10cbb` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t CRPS df 5, mobility on, fold 5, seed
53, full production compilation. WIS C0/C1/C2 is
0.463258/0.439790/0.439795. Accounting is refreshed to 17,227 GPU-seconds
(4.7853 GPU-hours).

The fold-5 seed-53 mobility-on quartet is complete. The first mobility-off
identity `loss-b-normal-nll-m0-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46135377`. Forty-four of 48 screen identities
are verified; three remain.

## 2026-09-19: screen identity 45 verified; identity 46 submitted

Screen run `loss-b-normal-nll-m0-f5-s53-r1` (job `46135377`) COMPLETED with
exit 0 after 4 minutes 33 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`13a721c88d63ed02d3ec6011f4140bef32e74a292692247c956f7229b5811490` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal NLL, mobility off, fold 5, seed 53, full
production compilation. WIS C0/C1/C2 is 0.327086/0.329906/0.329260.
Accounting is refreshed to 17,475 GPU-seconds (4.8542 GPU-hours).

`loss-b-normal-crps-m0-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46135842`. Forty-five of 48 screen identities
are verified.

## 2026-09-19: screen identity 46 verified; identity 47 submitted

Screen run `loss-b-normal-crps-m0-f5-s53-r1` (job `46135842`) COMPLETED with
exit 0 after 4 minutes 17 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`e7617038bf538bb99c3cbdffdc1fac16bad6074392fad30dac215e4b544c9565` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Normal CRPS, mobility off, fold 5, seed 53, full
production compilation. WIS C0/C1/C2 is 0.461645/0.442367/0.441097.
Accounting is refreshed to 17,587 GPU-seconds (4.8853 GPU-hours).

`loss-b-student_t-nll-m0-f5-s53-r1` was prepared, remote-dry-ran, and
submitted as `acc_debug` job `46136383`. Forty-six of 48 screen identities
are verified; two remain.

## 2026-09-19: screen identity 47 verified; final identity submitted

Screen run `loss-b-student_t-nll-m0-f5-s53-r1` (job `46136383`) COMPLETED
with exit 0 after 4 minutes 28 seconds, early-stopped. The canonical reader
verifies checkpoint SHA-256
`65bc71a9e49e99098fdde6c5bd159132fb44b32dc086a9fc97dda171b2359973` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. Student-t NLL df 5, mobility off, fold 5, seed
53, full production compilation. WIS C0/C1/C2 is
0.358408/0.358166/0.357839. Accounting is refreshed to 17,794 GPU-seconds
(4.9428 GPU-hours).

The final screen identity `loss-b-student_t-crps-m0-f5-s53-r1` was prepared,
remote-dry-ran, and submitted as `acc_debug` job `46137006`. Its verification
completes the 48-identity screen and opens the paired comparison stage.

## 2026-09-19: 48-identity screen complete; paired analysis recorded

Final screen run `loss-b-student_t-crps-m0-f5-s53-r1` (job `46137006`)
COMPLETED with exit 0 after 4 minutes 22 seconds, early-stopped. The
canonical reader verifies checkpoint SHA-256
`6019748087f142bc1ab66ae157ef7fdb0031a57b1777fd5c98d2e68227efe8b3` against
both receipts and reads 2,109 scored rows across all four heads and C0/C1/C2,
plus 2,322 case-event rows. All 48 screen identities are now complete,
synced, and locally verified with checkpoint-consistent receipts. Total
allocation cost is 18,056 GPU-seconds (5.0156 GPU-hours), far inside the
128 GPU-hour budget; no run approached the 120-minute cap.

Paired analysis ran through `evaluation.research_comparison.
compare_scored_forecasts` on every declared contrast within matched
fold/seed/control (180 comparisons: 10 contrasts x 3 folds x 2 seeds x
3 controls). Artifacts are recorded under `analysis/`
(`paired_comparisons.json`, `summary_by_contrast.csv`, `ranking.csv`).
All comparisons report `insufficient_temporal_blocks`: the calibration role
spans only 34 days, fewer than the four required 28-day blocks. Per the
frozen protocol this is point evidence only; no promotion or superiority
claim is available from this screen.

Point-evidence findings (C0 identity calibration; C1/C2 agree in direction):

1. Objective: CRPS training improves over NLL training for the Normal
   family. Mobility-on is consistent: positive relative WIS improvement in
   all 6 fold/seed units, mean +6.1% (range +0.05% to +13.1%). Mobility-off
   averages +11.9% but with high variance (-29.1% to +23.3%).
2. Family: Student-t (df 5) does not improve over Normal under CRPS training
   (about -3% both mobility settings) and is noise-level under NLL training.
3. Mobility: mobility-on does not improve calibration-role WIS for any loss
   combination; mobility-off is equal or better in all 6 units for both CRPS
   combinations (mean -11.2% Normal CRPS, -10.8% Student-t CRPS) and mostly
   equal for NLL.
4. Ranking: the lowest mean target WIS is Normal CRPS mobility-off (0.3416),
   followed by Student-t CRPS mobility-off (0.3523). Mobility-on variants
   rank below their mobility-off pairs for every objective.
5. Promotion gate: no contrast passes the full promotion-style gate
   (at least 2% relative WIS improvement with no per-head WIS or MAE
   regression) in more than 1 of 6 paired units, and the required temporal
   uncertainty is unsupported. Per protocol, unsupported gates are
   inconclusive, not passes: the screen establishes no promotable
   difference, and the point evidence favours Normal CRPS with mobility off.

The confirmation data boundary was preserved throughout: no run evaluated
the confirmation role, and no confirmation labels were read. The frozen
final-confirmation workflow (frozen calibrators and event rules consumed
without refitting) remains unimplemented and is a prerequisite for any
future confirmation-stage claim. This screen's result is a valid outcome:
the evidence gates do not support a promotable loss-type change, and the
point ranking indicates Normal CRPS with mobility off as the reference for
any future bounded confirmation design.
