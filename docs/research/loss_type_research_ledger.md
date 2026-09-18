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
