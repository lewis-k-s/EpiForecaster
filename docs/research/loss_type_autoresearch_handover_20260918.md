# Loss-type autoresearch handover

## Status and first actions

**The goal is paused at the user's request. Do not submit more work until the
user resumes it.** This document is a restart guide for a smaller model.
The goal tool confirmed `paused` on 2026-09-18. The local monitor was stopped.
No heartbeat automation is installed.

At **2026-09-18 14:20:07 UTC (16:20:07 Europe/Madrid)**, the already submitted
job **46089017** was `RUNNING` on `as03r3b03`, with two seconds elapsed. It was
not cancelled. Its existing one-GPU, 30-minute limit still applies. This is a
timestamped observation, not a current status guarantee.

After the user resumes:

1. Read the project `AGENTS.md`, model-training skill, MN5 skill and its reference,
   and `docs/MN5_SNAKEMAKE.md`. Reuse the existing goal; do not create a duplicate.
2. Check job 46089017 and all current jobs for `bsc008913` from `mn5`.
3. If terminal, sync its artifacts and verify them as shown below. Do not resubmit
   because a local observer timed out. Record a failure before deciding a retry.
4. Finish the five already prepared acceptance cases, one fallback job at a time.
   Do not prepare them again. Update the inventory and budget after each case.
5. Only after all eight acceptance cases pass, start the fixed 48-identity
   scientific screen. Its templates are already rendered, but no scientific
   screen run has been prepared or submitted.
6. Compare paired, verified evidence. Preserve the confirmation data boundary.
   Sync and record notable results. Continue to the protocol's conclusion or
   budget limit; acceptance alone does not complete the goal.

No scientific improvement has been established. Acceptance scores are excluded
from ranking and promotion. Do not repeat preprocessing or the resolved native
reader investigation.

## Authority, scope, and files

The authoritative checkout is `/Volumes/HUBSSD/code/meprecisa/EpiForecaster` on
the Mac. Branch: `codex/loss-type-research`. Source HEAD before this documentation
change: `48d507b`. The unrelated ` ? tex` change must remain untouched.
Remote source bundles and run roots are immutable, not Git checkouts.

The user authorized training under this goal, the new research preprocessing
run and DVC publication, and local sync of notable results. Those approvals
remain valid after resume. Do not request them again. The pause takes priority
over further dispatch. Do not use subagents unless the user authorizes them.

Read these repository files instead of reconstructing the full conversation:

| File | Purpose |
| --- | --- |
| `configs/research/loss_protocol_v1.yaml` | Frozen settings, gates, and budget; takes precedence over proposed settings in the original plan |
| `configs/research/loss_roles_v1.json` | Exact data roles and source constraints |
| `docs/research/loss_type_autoresearch_plan_20260918.md` | Scientific rationale and stages |
| `docs/research/loss_data_audit_20260918.md` | Data audit and limitations |
| `docs/research/loss_type_research_ledger.md` | Durable evidence and decisions |
| `.agents/skills/model-training/SKILL.md` | Project training rules |
| `/Users/lewis/.codex/skills/mn5-cluster-usage/SKILL.md` | Endpoint and allocation rules; also read `references/mn5-docs.md` beside it |
| `docs/MN5_SNAKEMAKE.md` | Public immutable workflow contract |

Below, `OUT` means the local path
`outputs/loss_research/loss-type-research-v1` relative to the checkout.
`REMOTE` means
`/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/EpiForecaster-DVC`.

Local generated state is ignored by Git but is present on this Mac:

- `OUT/acceptance/run_inventory_v2.json`: exact eight identities, hashes, states,
  job IDs, and verification paths. Reconcile with live Slurm state on resume.
- `OUT/acceptance/templates-v2/`: current eager acceptance templates.
- `OUT/compute_budget.json` and `OUT/scheduler_accounting_20260918.txt`.
- `OUT/screen/templates-v1/screen.json` and 48 YAML files: rendered screen.
- `OUT/screen/dispatch_gate.json`: all identities and template hashes checked;
  dispatch remains gated on eight successful acceptance cases.

If generated templates are unavailable, `scripts/research/loss_screen.py`
can render a new directory with `--dataset-pointer`, `--output`, and
`--stage acceptance` or `--stage screen`. Never overwrite an existing run.

## Acceptance inventory

All cases use fold 1, seed 52, ten batches, and a 30-minute allocation cap.
`m1` enables mobility; `m0` disables it. Student-t uses df 5.
The CLI's `max_batches` path limits training to one epoch despite the template's
two-epoch ceiling. This is known and does not affect the full screen.

| Run ID | Job | State at handover |
| --- | --- | --- |
| `loss-acceptance-normal-nll-m1-f1-s52-r6` | 46087093 | Complete and locally verified |
| `loss-acceptance-student-t-crps-m1-f1-s52-r2` | 46088534 | Complete and locally verified |
| `loss-acceptance-normal-crps-m0-f1-s52-r1` | 46089017 | Running at timestamp above |
| `loss-acceptance-normal-nll-m0-f1-s52-r1` | None | Prepared; remote dry-run passed |
| `loss-acceptance-normal-crps-m1-f1-s52-r1` | None | Prepared; remote dry-run passed |
| `loss-acceptance-student-t-nll-m0-f1-s52-r1` | None | Prepared; remote dry-run passed |
| `loss-acceptance-student-t-nll-m1-f1-s52-r1` | None | Prepared; remote dry-run passed |
| `loss-acceptance-student-t-crps-m0-f1-s52-r1` | None | Prepared; remote dry-run passed |

Normal NLL R6 used source `a91476f`, with backward compilation still enabled.
It is a valid bounded compiled acceptance. The other seven cases were prepared
from `aad037f`, with both `compile=false` and `compile_backward=false`.
The old prepared Student-t CRPS mobility-on R1 was never executed and is
superseded by R2. Do not submit it.

Verified local case directories are `OUT/acceptance/normal-nll-m1-f1-s52-r6/`
and `OUT/acceptance/student-t-crps-m1-f1-s52-r2/`. Each has artifacts, logs,
run receipts, and `local_artifact_verification.json`. Each reader check found
2,358 score rows across all four heads and C0/C1/C2, plus 2,562 event rows.

| Verified identity | Checkpoint SHA-256 |
| --- | --- |
| Normal NLL R6 | `bbfca498698c55f3a36bb73e150ae80be1b37f7eeb48ff2a34104646c7d84c21` |
| Student-t CRPS R2 | `4b26aad24e59343ddb2f0ea478415f1c416253d318567fc65037191c950821da` |

For orientation only, C0/C1/C2 mean WIS was 0.440687/0.382304/0.382858 for
Normal NLL and 0.423859/0.381948/0.382481 for Student-t CRPS. These are pipeline
checks, not evidence that one loss is better.

## MN5 commands and limits

Use `mn5` for scheduler truth, submission, allocations, and `srun`. Use `dt`
for transfer and shared-file inspection. Use `wifi` or `wifi-acc` only for
network dependency preparation. Account: `bsc08`; user: `bsc008913`.

```bash
ssh mn5 'squeue -u bsc008913 -o "%.18i %.16q %.12T %.20N %.12M %R"'
ssh mn5 'sacct -j 46089017 --format=JobID,State,ExitCode,Elapsed -n -P'
ssh mn5 'bash -lc bsc_queues'
ssh mn5 'sacctmgr -nP show qos acc_debug format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxTRESPJ'
```

Live checks on September 18 found `acc_debug` **MaxJobsPU=1, MaxSubmitPU=1**.
Only one submitted fallback job is allowed, including pending work. Refresh
limits on resume. Previous debug jobs often started at 20-minute intervals;
this is an observation, not a schedule guarantee.

Prefer a compatible existing `acc_interactive` allocation. Query its current
steps and remaining time before use. At most one interactive allocation per
user; at most two GPUs/40 CPUs and two hours. Request only resources ready for
use. Previous one-GPU requests stayed pending and were cancelled while unused.
There is no interactive allocation owned by this loop at pause. Last request
46088104 was cancelled and its PTY exited. The interactive partition was UP
on shared node `alogin3`. A denied `sinfo` inventory query is not zero capacity;
do not bypass it. Interactive nodes need not match the `as...` compute pattern.

After resume, if no suitable allocation exists and live policy permits:

```bash
ssh -tt mn5 'salloc -A bsc08 -q acc_interactive -t 01:00:00 -N 1 -n 1 -c 20 --gres=gpu:1 --job-name=loss-acceptance-loop'
```

Keep the owning PTY alive. Dispatch science only through the public launcher:

```bash
# Existing prepared case: do not repeat prepare.
bash scripts/workflow/mn5_workflow.sh run loss-acceptance-normal-nll-m0-f1-s52-r1
# Or attach the prepared run to a compatible active allocation:
bash scripts/workflow/mn5_workflow.sh run-interactive RUN_ID ALLOCATION_ID
# New scientific identity, after the acceptance gate:
bash scripts/workflow/mn5_workflow.sh prepare NEW_RUN_ID --config /private/tmp/TEMPLATE.yaml
bash scripts/workflow/mn5_workflow.sh dry-run NEW_RUN_ID
```

Commands with uppercase placeholders require substitution. Ignored templates
inside `outputs/` are absent from the source snapshot. Copy the exact template
to `/private/tmp` for the supported external-template interface and check its
hash. Existing prepared cases need no template copy or preparation.

For fallback SSH attachment, first obtain the currently assigned node. Never
attach to a stale node after a job ends. Example, only while the job is running:

```bash
ssh -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/private/tmp/epiforecaster-loss-known-hosts \
  -J mn5 bsc008913@CURRENT_NODE 'hostname'
```

Use tool network escalation for SSH/rsync when required. Sandbox DNS failures
do not establish a cluster outage. No automatic approval rejection has occurred.
Do not disable SSH host-key checking.

## Sync and verify each completed case

The live run root is `REMOTE/runs/loss-acceptance-normal-crps-m0-f1-s52-r1`.
On completion, use the following from the checkout. The first three variables
are set for that case; change them together for later cases.

```bash
loss_run=loss-acceptance-normal-crps-m0-f1-s52-r1
loss_remote=/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/EpiForecaster-DVC/runs/$loss_run
loss_local=outputs/loss_research/loss-type-research-v1/acceptance/normal-crps-m0-f1-s52-r1
mkdir -p "$loss_local/artifacts" "$loss_local/logs" "$loss_local/run_receipts"
rsync -a "dt:$loss_remote/outputs/" "$loss_local/artifacts/"
rsync -a "dt:$loss_remote/logs/" "$loss_local/logs/"
rsync -a --include='/config.yaml' --include='/manifest.yaml' \
  --include='/.state/' --include='/.state/data_manifest.json' \
  --include='/.state/training.done' --exclude='*' \
  "dt:$loss_remote/" "$loss_local/run_receipts/"
```

Run this verification with the project Python environment. Do not trust an
evaluation's own checkpoint hash as its independent identity anchor.

```python
from pathlib import Path
import hashlib
import json
from evaluation.research_artifacts import read_research_evidence

root = Path("outputs/loss_research/loss-type-research-v1/acceptance/normal-crps-m0-f1-s52-r1")
model_id = "loss-acceptance-normal-crps-m0-f1-s52"
model = root / "artifacts/training/loss-type-research-v1-acceptance" / model_id
receipt = json.loads((model / "training_manifest.json").read_text())
with (model / "checkpoints/best_model.pt").open("rb") as handle:
    checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
assert checkpoint_hash == receipt["checkpoint_sha256"]
done = json.loads((root / "run_receipts/.state/training.done").read_text())
assert done["checkpoint_sha256"] == checkpoint_hash
evidence = read_research_evidence(
    root / "artifacts/evaluation",
    expected_checkpoint_sha256=checkpoint_hash,
    role_manifest_path=Path("configs/research/loss_roles_v1.json"),
)
print(len(evidence.scores), len(evidence.events))
```

Student-t model IDs and template names use `student_t`; remote run IDs use
`student-t`. The canonical reader checks completion, file inventories and
hashes, checkpoint and role identities, and calibration records. Also inspect
the pinned run config for intended family, df, loss, mobility, and compile flags.
Check all four heads and three controls. Save `local_artifact_verification.json`
using a completed case as the format example, with
`scientific_selection_eligible: false` for acceptance. Only then set inventory
state to `complete_verified`. Record source/config/input identities, not just
the current Git HEAD.

Refresh `sacct -X` accounting after terminal state. Count allocation elapsed
seconds times allocated GPUs, including failures and preprocessing. Pending
time uses no allocated GPU time. At pause the recorded total is **1,359
GPU-seconds = 0.3775 GPU-hours**, through job 46088534. Job 46089017 is not yet
included. Never sum both an allocation and all its steps as separate GPU use.

## Data and environment already verified

The approved replacement dataset is complete, published, materialized locally,
and pinned in the protocol. Do not preprocess again.

| Item | Identity |
| --- | --- |
| Dataset pointer | `OUT/real_loss_fit20200930_v1.nc.dvc` |
| DVC MD5 | `b5b684d5b9f31369895589c413f39291` |
| Payload SHA-256 | `b9e2b1424abb7f2b3294ab19939c8be7519f8643991530a9ffde52e28c9c1421` |
| Payload bytes | 297,505,997 |
| Successful preprocessing | `loss-preprocess-20260918-r5`, job 46083788 |
| QA and publication evidence | `OUT/preprocessing/r5/artifacts/`, including `local_support_qa.json` |
| Raw copied inputs | `OUT/raw_inputs.dvc`, MD5 `c65715e80dcc5947eb665c7f037ed9ed.dir` |
| OD-free Region2Vec | `outputs/region_embeddings_no_mobility/region_embeddings.pt.dvc`, MD5 `e04ff23fe17dbbb245315798fd7cd887` |
| Mobility | `data/files/mobility.zarr.dvc`, MD5 `f3d97d9e1d9128436f8bf498220818eb.dir` |
| Runtime SIF | `containers/epiforecaster-runtime-e2fe44e8b1c4a59ad417.sif` |
| SIF SHA-256 | `b9789057bc686414649c5224bc908ada1556fba17486c8055a76b49cc61c48ba` |

The SIF lock matches `uv.lock` and contains h5netcdf 1.7.2/h5py 3.15.1.
The dataset has 435 dates and 768 regions. Fitted preprocessing stops at
2020-09-30; weekly hospital totals are delayed seven days until the full week
has elapsed. Causal leading missing values remain missing. Source release
vintages are unavailable: revised retrospective data remain a limitation.
QA found zero observed nonfinite values in fit, selection, and calibration.
All four heads have support for folds 1, 4, and 5. The first actual acceptance
split built 1,942 fit, 17 selection, and 16 calibration samples.

The old dataset MD5 `1e695877f8565c40dcec4c214b0813b8.dir` is not valid for this
research protocol. Preserve it; do not use it as the new experiment baseline.

## Scientific contract and remaining work

| Label role | Inclusive dates |
| --- | --- |
| Fit | 2020-05-01 to 2020-09-30 |
| Selection | 2020-10-29 to 2020-11-30 |
| Calibration | 2020-12-29 to 2021-01-31 |
| Confirmation | 2021-03-01 to 2021-05-09 |

Keep 28-day gaps, 60-day history, and full 28-day target windows contained in
their declared role. History may precede the label role. Development uses
`research_evaluation_role=calibration`, `research_confirmation_enabled=false`.
No research confirmation evaluation has been run. The confirmation period was
studied previously, so any final claim is retrospective, not prospective.

The screen is Normal/Student-t(df 5) × NLL/CRPS × mobility on/off × folds
1/4/5 × seeds 52/53 = 48. Preserve paired ordering from `screen.json`.
Use the anchor's four MLP delay kernels and OD-free embeddings, 50 epochs,
early stopping patience 5, and common `mean_target_wis` checkpoint selection.
Retain the best-CRPS diagnostic checkpoint. Full runs retain production
compilation; do not carry the eager acceptance override into them by accident.

C0 is identity calibration, C1 fits a per-head global scale for 80% coverage,
and C2 fits scale for WIS. WIS averages the four required heads equally and
uses 23 quantiles/11 intervals. Student-t scores are in declared log units;
exponentiating Student-t gives an infinite mean and original-scale CRPS.
Use finite original-scale quantiles, not a claimed finite mean or CRPS.

Cases events are a high-incidence research proxy, not clinical outbreak truth.
Fit q90 on unique fit municipality/date observations; q95 is a sensitivity
check. Report fixed probability threshold 0.5 and the calibration-fitted
operating rule with precision floor 0.5 and at least 20 unique alerts/support.
Report precision, recall, Brier score, and undefined metrics with reasons.
Hospital event scoring remains unsupported until source aggregation is defined.

Promotion requires at least 2% lower mean WIS, no more than 2% per-head WIS or
median-MAE regression, 80% coverage error at most 3 percentage points, 95%
error at most 2 points, and supported horizon error at most 5 points. Cases
Brier may regress by at most 2%, recall by at most 2 percentage points.
Require complete paired evidence. Unsupported gates are inconclusive, not passes.

Uncertainty uses disjoint target-date blocks of at least 28 days, grouping all
regions and heads. Require at least four blocks; 2,000 draws, seed 852. Current
short roles may support only point evidence. Do not weaken this rule to claim
superiority. Promising uncertain candidates can enter bounded confirmation,
but cannot be promoted on unsupported evidence.

Budget: 128 GPU-hours or seven elapsed days; at most 204 scientific identities;
four GPU-hours for acceptance; reserve half for confirmation. Up to four new
adaptive settings (24 identities), expanded baseline/finalist checks up to
120 identities, final outer refits up to 12. Reuse only provenance-matched runs.
Measure full-run cost before committing the remaining budget. Stop after three
non-improving complete adaptive comparisons, at the budget limit, or after
frozen confirmation. At most two retries for the same cause. Do not silently
reset elapsed time because the goal was paused.

The final frozen confirmation workflow is **not complete**. Before any final
labels are read, implement and test a separate path that consumes frozen
calibrators and event rules without refitting them. The scalar calibrator has
confirmation-role support, but the event path currently refits from fit and
calibration inputs. The development workflow correctly rejects confirmation.
`research_comparison.py` handles paired screening scores, not the complete
event/promotion/final-confirmation decision. Do not treat it as that final gate.

## Code map, validation, and resolved faults

| Module | Implemented responsibility |
| --- | --- |
| `evaluation/distribution_scores.py` | Normal and Student-t CRPS, WIS, distribution calculations |
| `evaluation/research_calibration.py` | C0/C1/C2 artifacts, protected roles, scores and calibrated intervals |
| `evaluation/research_events.py` | Cases event rules, support, precision/recall and probability metrics |
| `evaluation/research_pipeline.py` | Fit-only thresholds and canonical research outputs |
| `evaluation/research_artifacts.py` | Verified artifact reader; use before comparisons |
| `evaluation/research_comparison.py` | Matched paired score comparisons and block uncertainty |
| `data/research_roles.py` | Role manifest and source/temporal boundary checks |
| `scripts/research/loss_screen.py` | Eight acceptance and 48 screen template renderer |
| `data/epi_dataset.py` | Processed NetCDF via h5netcdf; Zarr via xarray unchanged |

Last full validation: **1,331 tests passed, two skipped; Ruff passed**.
Evidence: `/private/tmp/epiforecaster-loss-pytest-r19.log`. No source change
followed that validation; later changes were records and generated templates.
Best-effort Pyright reported ten errors in unchanged tensor/topology code,
none at the reader change. For future code changes run required tests and lint:

```bash
UV_CACHE_DIR=/private/tmp/epiforecaster-uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=/private/tmp/epiforecaster-uv-cache uv run --no-sync pytest tests/
git diff --check
```

Resolved faults and pitfalls:

- Several early jobs failed on workflow container/bind/source-input setup.
  Those fixes are committed and preprocessing R5 passed. Do not redo setup.
- Normal NLL R4/R5 crashed inside native netCDF4 reopen before training.
  R5 captured the traceback. Commit `a91476f` switched to the declared
  h5netcdf backend, with real-file tests and dataset parity checks. Two complete
  cluster cases now pass this path. Do not retry the old deployed bundles.
- Commit `aad037f` disabled both compiler paths for eager acceptance. A single
  `compile=false` was insufficient because backward compilation is separate.
- Optional per-head plotting still assumes Zarr and warns on the NetCDF file.
  Canonical metrics and event artifacts complete successfully. Do not confuse
  that warning with a failed evidence check or spend queue waits on unrelated
  plotting repairs.
- Existing ablation grouping does not support research outputs; crossval runs
  all declared folds. Use the fixed single-run templates instead of adding a
  second launcher or silently changing the design.
- Use `training-dynamics-analyze` before interpreting convergence/gradients.
  A full dynamics report has not yet been produced.
- Never patch a deployed bundle, run root, or artifact in place. Changed code
  requires a new source snapshot and run ID. Preserve failed runs and receipts.

Keep each iteration small: inspect one terminal run, verify its evidence,
record the decision, then dispatch the next authorized case. Do not turn queue
waits into broad refactors. A negative or inconclusive result is a valid final
outcome if the frozen evidence gates do not pass.
