# MN5 workflow

This document defines the project-specific immutable-run contract. For MN5
endpoint selection, dependency preparation, allocation sizing, interactive use,
and scheduler observation, load and follow the `mn5-cluster-usage` skill.

Snakemake is the production entry point for MN5 preprocessing, training,
evaluation, cross-validation, HPO, ablation, and artifact promotion. Each
workflow uses one ACC-node allocation. Snakemake runs as an
allocation-internal lightweight coordinator and launches scientific work as
one or more `srun` job steps.

The same immutable run supports two execution adapters: `run` requests a batch
allocation, while `run-interactive` attaches to a compatible active
`acc_interactive` allocation through `srun --external-launcher`. Both adapters
use the same snapshot-owned coordinator, DAG, containers, data contracts, logs,
manifests, and scientific artifact rules.

The workflow does not use nested `sbatch` submissions, worker arrays, or a
separate GPP coordinator allocation.

## Allocation contract

Every immutable workflow config requests:

```text
partition=acc
nodes=1
cpus_per_task=20
tasks=N
gres=gpu:N
```

Templates use `acc_bscls`; `acc_debug` is also accepted when its wall-time
limits fit the run. No `--mem` or `--mem-per-cpu` directive is permitted
because MN5 derives host memory from the CPU allocation.

Single training and evaluation use `N=1`. HPO and ablation use their configured
worker count. Cross-validation uses up to `N` concurrent folds, with ten folds
and four GPUs in the production template. `N` must be between one and four.
The coordinator sets the Snakemake CPU, job, and GPU limits from the allocation.

The batch shell owns the allocation and runs the lightweight Snakemake manager.
A parallel worker rule launches:

```text
srun --nodes=1 --ntasks=N --ntasks-per-node=N \
     --cpus-per-task=20 --gpus-per-task=1 --exact \
     --kill-on-bad-exit=1
```

Each rank receives one GPU and 20 CPUs. `SLURM_PROCID` becomes an explicit
logical worker ID. Optuna workers coordinate through their GPFS journal.
Study workers do not
communicate with Snakemake while running. If one rank fails, Slurm terminates
the worker step and the batch job fails.

Cross-validation uses separate training and evaluation rules for each fold.
Each rule requests one GPU and 20 CPUs in one exact `srun` step. There are no
fixed fold waves or campaign-wide training barrier: evaluation depends only on
its own fold's checkpoint and manifests. The finalizer waits for every fold's
declared evaluation files.

After every HPO worker marker exists, separate DAG stages resolve
`study.best_trial`, validate and copy its exact checkpoint, evaluate it, promote
the declared artifacts, and finally delete losing checkpoint directories.
Ablation workers compile a checked-in declarative campaign into a deterministic
`condition × fold × paired-model-seed` queue.  Optuna's JournalStorage is
strictly durable coordination: it performs no TPE or design sampling. Analysis
runs only after the exact declared identity matrix is terminal and provenance
matched; the campaign endpoint and the fold/seed-paired deltas come from run-level
aggregate metrics and never pool granular windows. Granular rows feed only
descriptive region, horizon, and target breakdowns, and any invalid prediction
(non-finite location or scale, or scale of zero or less) stops the analysis.

## Storage and identity

Durable state lives under project GPFS:

```text
/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/
  EpiForecaster-DVC/
    cache/                         # shared, locked DVC cache
    storage/                       # content-addressed DVC remote
    containers/                    # immutable SIFs
    promoted/<run-id>/             # selected artifacts
    runs/
      .reservations/<run-id>/       # permanent no-reuse receipt
      .staging-<run-id>.<token>/    # unpublished transfer location
      by-slurm/<allocation-id>/<run-id>/<attempt-id>
                                      # one-to-many allocation lookup
      <run-id>/
        config.yaml                # immutable rendered config
        .state/dispatch.tsv        # immutable submission bindings
        .state/published.done      # verified publication receipt
        manifest.yaml
        .state/dependencies/       # full-source, stage-labeled contracts
        source/                    # immutable worktree snapshot
        optuna/
        logs/
        outputs/
```

Remote EpiForecaster state belongs under this project root, not under `~/` on
`dt` or MN5. Production workflows use only the `EpiForecaster-DVC/` subtree.

Source is transferred into a unique hidden staging leaf. The runner contained
in that exact source snapshot verifies the config, source inventory, SIF, and
containerized Snakemake version before atomically reserving the run ID and
publishing the final run.
Permanent reservation receipts prevent deleted run IDs from being reused.
The allocation identity is `<job-id>_0`. Container executions add Slurm step
and process rank, yielding `<job-id>_0_<step-id>_<rank>` and preventing scratch
collisions between concurrent ranks.

The manifest records the workflow kind and public target, Git SHA, exact source
snapshot SHA-256, DVC hashes, SIF SHA-256, workflow tooling key, Snakemake
version and location, rendered config SHA-256, and immutable dispatch-request
SHA-256. Study workflows also record their study identity.

Dirty worktrees are supported. Preparation copies tracked and non-ignored
untracked files to an isolated snapshot, hashes and transfers those bytes via
`dt`, verifies them remotely, and makes the source and config read-only. DVC
pointers travel with source; DVC payloads are rejected from the source bundle.

Preparation also checks the SIF against the source. The SIF metadata sidecar
must record the SHA-256 of the snapshot's `uv.lock`; otherwise prepare stops
and names the image to use. `.cluster/mn5.conf` defaults to the digest-named
image, not to the `epiforecaster-runtime.sif` alias. A template can pin
`container.expected_sha256`; a different SIF digest then stops preparation.

## Environments and data

The immutable SIF contains scientific dependencies, DVC, and Snakemake from
the locked `workflow` dependency group. It excludes EpiForecaster source.
Ordinary Python or Snakefile changes therefore create a new run snapshot
without rebuilding the SIF. Packaging the coordinator dependencies in the SIF
provides a sealed coordinator runtime, but using it as the batch control plane
still requires explicit MN5 host-Slurm integration.

The dispatcher, immutable remote runner, and batch coordinator execute their
Python, DVC, and Snakemake commands from the same verified SIF. The coordinator
binds only the MN5 host interfaces needed to launch job steps: the `srun`
client, Slurm libraries and configuration, user/group identity, Munge, and the
site Snoopy and EAR plugin trees. Scientific rules continue to enter the SIF
through `mn5_container_exec.sh`; the coordinator itself does not request
NVIDIA device injection.

The top-level Snakefile includes domain rules from `workflow/rules/`. Every
scientific stage consumes a content-addressed dependency contract and the DVC
data manifest. The public targets are:

```text
train:           prepare_run -> stage_data -> train_model
evaluate:        prepare_run -> stage_data -> evaluate_model
train_evaluate:  train_model -> evaluate_model
hpo:             initialize_study -> launch_optuna_workers -> select_best_trial
                 -> evaluate_model -> promote_best_model
                 -> cleanup_losing_checkpoints
ablation:        initialize_study -> launch_ablation_workers -> finalize_ablation
crossval:        train_crossval_fold[fold] -> evaluate_crossval_fold[fold]
                 -> finalize_crossval
preprocess:      stage_data -> preprocess_dataset -> publish_preprocessed
```

Synthetic pretraining uses the same `train` entity through
`workflow/config/mn5_synth_pretrain.yaml`. Its processed Zarr is materialized
for training, while the raw synthetic Zarr is a provenance-only DVC target.

The corrected 20-run corpus uses
`workflow/config/mn5_preprocess_sirhd_reporting_v1_20run.yaml`. The immutable
run owns the processed NetCDF, QA record, generated DVC pointer, and publication
receipt. Promote the pointer to the source repository only after these records
pass review.

`mn5_train.yaml` pins `workflow.target: train`, so its default terminal artifact
is the training manifest and checkpoint. `train_evaluate` is the explicit
composition when both stages are wanted. Training and HPO trials force
`training.run_final_evaluation=false`; evaluation is owned by the downstream
rule. Ablation trials retain final test metrics because paired-seed aggregation
consumes those per-trial artifacts.

### Declarative ablation campaigns

`configs/campaigns/ablation_default_v2.yaml` is the versioned scientific
authority for the production ablation matrix. It names the baseline, hypotheses,
fixed structured overrides, the allowed hydrated-config difference paths for every
condition, paired model seeds, the persisted representative 10-fold split manifest,
metrics, budget policy, and completeness and provenance requirements.
`workflow/config/mn5_ablation.yaml` supplies only execution resources, study name,
and that campaign path. Every scientific loss, objective, mobility, signal, or
architecture ablation must use this declarative campaign interface; ad hoc
single-run matrices are acceptance or historical-reproduction interfaces only.

Before work is queued, every condition is hydrated through
`EpiForecasterConfig`, including model-parameter promotion and explicit defaults.
Conditions with the same hydrated configuration are rejected, as are differences
outside their declared paths. `equal_budget_policy.paths` is a non-empty list of
resolved configuration paths (currently `training.epochs` and
`training.max_batches`) whose values must be identical for every condition.

Production campaigns evaluate every fold in the persisted manifest. The canonical
10-fold manifest places all 161 valid target regions in the held-out test role
exactly once per condition and paired model seed. `ablation_default_v2.yaml`
therefore declares folds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` and two paired seeds
`[42, 43]`, giving `28 conditions × 10 folds × 2 seeds = 560` identities. Each
identity evaluates its complete eligible held-out support: all observed targets,
forecast horizons, and target dates admitted by the campaign split and masks.
Production ranking cannot use a selected regional or temporal subset.

The same model seed is used for every condition within each fold, so paired
contrasts control initialization, DataLoader order, and other stochastic-training
variation. `training.split_seed` does not change persisted fold membership. Analysis
must report the aggregate endpoint together with matched deltas by target, region,
horizon, fold, and seed; the distribution and fraction of regions improved or
worsened; and predeclared mobility-context strata for mobility contrasts. These
regional effects are descriptive unless source sharing and temporal dependence are
modeled explicitly.

`smoke_fold_indices` (`[1, 4]`) remain a strict subset of the v2 production folds.
The checked-in `workflow/config/mn5_ablation_coverage_smoke.yaml` runs every
default condition and paired model seed over that bounded subset with small batches;
it is an ACC integration acceptance check, not a production estimate. The smaller
`ablation_smoke_v1.yaml` remains the eight-identity infrastructure smoke. Both
smoke campaigns require their exact declared identity matrix, terminal trials, and
provenance, but they cannot support model-selection claims. Campaigns are expanded
in memory; the workflow never generates per-run YAML files. A campaign with fold
indices must declare an existing persisted split manifest, and undeclared
resolved-config drift fails before a worker starts training.

An `ablation_suite` config groups distinct campaigns in one immutable run and
one four-GPU Slurm allocation. The suite runs each campaign in listed order,
with four workers per campaign. Each study keeps a separate journal, worker
receipts, and analysis directory. The suite manifest records both campaign
manifests and their hashes. A later batch submission of the same run resumes
the first incomplete study. The campaigns retain their own split and scoring
contracts; a temporal campaign and a spatial campaign do not share trials.

### Ablation allocation closeout

An ablation config declares `closeout_signal_seconds`. The batch launcher
requests `SIGUSR1` that many seconds before its wall time ends. On this signal,
a worker stops claiming new identities, completes its active trial when
possible, and records a durable `partial` receipt for remaining work. The
parallel launcher then exits with a controlled incomplete status. Snakemake
does not write `workers_complete.done` or run campaign analysis for that
allocation. Submit the same immutable run again to continue its exact identity
matrix; existing terminal trials are retained.

The wrapper scripts ignore `SIGUSR1`, so only the Python worker handles it.
The launcher removes `.state/continuation.required` and old rank receipts when
it starts. It writes the marker again, with the current attempt ID, only after
a close-out. The coordinator accepts a marker of its own attempt only. It then
exits 0 and records `exit_code 75` in `completed.tsv`, and `status` shows
`coordinator=closeout`. Code 75 means that the queue still holds work.

Optuna values always minimize the validation-selection coordination value and
record its actual metric name and value on the trial. They are not the campaign
primary endpoint: the analyzer alone applies the campaign-declared held-out
primary metric and direction to aggregate artifacts. Every ablation config
must set `recover_stale_running: true`. Recovery marks stale `RUNNING`
attempts terminal before enqueuing a replacement. Up to three such recoveries
per identity do not count toward the failed-attempt budget.

A `reuse_receipt` is accepted record by record. A record is reused only when its
dataset and region-embedding DVC hashes, SIF digest and hydrated training config
hash equal those of the executing run, and its source bundle hash equals the
run's or is listed in the campaign field `reuse.allow_code_identities`. The
worker writes the decision to `outputs/training/reuse_acceptance.json`;
rejected identities are trained again. Failed or pruned
historical attempts are retained, but completion requires no nonterminal attempt,
exactly one `COMPLETE` trial, and exactly one canonical output per scientific
identity.

Each output sidecar must agree with its persisted config: `fold` is `null`
unless cross-validation is enabled. Every canonical output has a non-empty
aggregate metric table with `target` and the campaign primary metric
(`mean_target_wis` is a column of `test_main_model_aggregate_metrics.csv`, the
equal-head mean of per-head WIS from the evaluation pipeline). Those run-level
aggregate rows, rather than granular windows, are the sole inputs to campaign
endpoint aggregation and to the paired fold/seed deltas. Reused runs must supply
this aggregate table in the evidence bundle, and the analyzer checks their
recorded identity against the bundled `config.yaml`.

Each data, training, evaluation, cross-validation, HPO, ablation, selection,
and promotion contract fingerprints the same complete scientific source
inventory together with its relevant config fragment and SIF digest. This
deliberately favors conservative invalidation and auditable provenance over
inferring transitive Python dependencies: any scientific source change
invalidates every stage contract, while documentation and tests remain
excluded. Remote run snapshots remain immutable, so changed source is prepared
as a new run rather than mutating an existing one.

Training declares its checkpoint and `training_manifest.json`. Evaluation
declares its summary JSON, node and per-head metric tables, optional granular
Parquet, and a hash-bearing `evaluation_manifest.json`. Ablation declares its
coverage, run-context, aggregate table, and artifact manifest. Completion
markers remain receipts rather than the sole scientific output contract.

`stage_data` materializes the pinned DVC payloads once in the run workspace.
The first scientific step copies them to an allocation-local cache on the node.
The key includes the run config and data manifest. A lock protects construction;
only a complete copy is published. All ranks and later stages mount this copy
read-only. Each process keeps a private temporary directory. Shared leases keep
the cache alive during execution, and the coordinator removes it on exit when
no step still uses it. An abrupt allocation loss can leave node-local files;
a later allocation uses a different cache path. GPFS remains authoritative for
the journal, logs, checkpoints, outputs, and promotion records.

Study journals and worker receipts are durable state. Worker rules own only a
completion output, so Snakemake failure cleanup and reruns cannot delete study
progress. The initialization rule creates the journal before workers start.

Preprocessing and cross-validation keep each attempt under `.state/attempts/`.
Partial attempts remain available for inspection. A completed attempt has a
receipt with the config identity and file hashes. On retry, the runtime checks
those hashes and restores missing public outputs without repeating the work.
Publication uses atomic hard links within the run filesystem, so large payloads
do not require a second copy. Keep attempt directories for the lifetime of the
run: they contain the training logs, dynamics, and recovery evidence. Incomplete
training attempts restart from the configured initial state; completed folds
are reused. Preprocessing declares `<dataset_name>.nc` as a DAG output, and
publication depends on that file as well as its QA evidence.

## Prepare and submit

Single training is the default:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <run-id>
bash scripts/workflow/mn5_workflow.sh dry-run <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>
bash scripts/workflow/mn5_workflow.sh status <run-id>
```

Use `run-interactive` only after the cluster skill has selected a compatible
active allocation:

```bash
bash scripts/workflow/mn5_workflow.sh run-interactive \
  <run-id> <allocation-job-id>
```

Omit the allocation ID only when exactly one compatible `acc_interactive`
allocation is active. The adapter attaches the immutable workflow to that
allocation and never calls `sbatch`.

The coordinator step runs detached (`setsid nohup`, no stdin), so an SSH or VPN
drop does not stop it. `launcher.pid` holds the PID of this detached step. The
allocation must have at least the requested runtime left, or the close-out
lead time plus 60 seconds for runs with close-out.

The selected public target is immutable after preparation. Choose `train` or
`train_evaluate` in the template's `workflow.target` before `prepare`; `run`
does not accept targets, overrides, or arbitrary Snakemake arguments.

Select study templates explicitly:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <hpo-run-id> \
  --config workflow/config/mn5.yaml
bash scripts/workflow/mn5_workflow.sh prepare <ablation-run-id> \
  --config workflow/config/mn5_ablation.yaml
bash scripts/workflow/mn5_workflow.sh prepare <crossval-run-id> \
  --config workflow/config/mn5_crossval.yaml
bash scripts/workflow/mn5_workflow.sh prepare <pretrain-run-id> \
  --config workflow/config/mn5_synth_pretrain.yaml
bash scripts/workflow/mn5_workflow.sh prepare <preprocess-run-id> \
  --config workflow/config/mn5_preprocess_sirhd_reporting_v1_20run.yaml
```

For evaluation without training, copy `workflow/config/mn5_evaluate.yaml`, set
`evaluation.checkpoint` to an absolute GPFS checkpoint path, and pass that
complete config with `--config`. Preparation fails before transfer if the field
is unset or relative.

The snapshot-owned `remote_run.sh` reads resource fields from the immutable
dispatch record. The batch adapter passes them to `sbatch`; the interactive
adapter validates them against the existing allocation. A launch lock and
coordinator lifetime lock prevent duplicate execution.

Every attempt records its mode, allocation, observed resources, coordinator
step, timestamps, logs, and exit status under
`.state/executions/<attempt-id>/`. Status and unlock use the coordinator step
or completion receipt for an interactive attempt; the parent allocation
remaining `RUNNING` does not make a completed workflow appear active.
MN5 `sacct` may display the external-launcher step with the allocation's full
TRES and elapsed lifetime. Use its sibling scientific steps and the run-local
execution receipts for workload accounting and completion.

If the allocation terminates before completion, inspect `status`, unlock stale
Snakemake state, and submit the same immutable run again:

```bash
bash scripts/workflow/mn5_workflow.sh unlock <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>
```

Running the target pinned in the immutable config through the default `all`
rule finishes with `.state/workflow.done`. Direct public targets expose their
declared artifacts without rewriting that default receipt. Existing run
directories and promotion destinations are never overwritten. Do not run
`dvc gc` while any run or promoted artifact may reference the shared cache.

## Research-role evaluation

For a protected research training config, set `evaluation.research_outputs: true`,
`evaluation.split: val`, and `evaluation.granular: true`. The workflow evaluates
selection (`val`) and calibration (`test`) from one checkpoint. This development
path rejects confirmation-enabled configs and changes to the trained observation
objective. It publishes a complete attempt only after both evaluations and the
research artifacts pass validation. Recovery restores the complete attempt.

The evaluation directory includes `research_fit_cases.parquet`,
`research_calibration.json`, `research_scored.parquet`, `research_summary.json`,
`research_events.parquet`, and `research_events.json`. Raw calibration outputs
are under `calibration/`. The top-level evaluation manifest hashes all outputs
and records the checkpoint and dated-role manifest identities. JSON role and
fold manifests are part of scientific stage dependency fingerprints.

Research score schema 1 includes the original distribution parameters, calibrated
scale, observed target, target units, last-history-day forecast origin, data role,
control ID, WIS, CRPS, NLL, median absolute error, PIT, coverage and interval width.
Its 23 `quantile_NNN` columns give probabilities `NNN / 1000` on the declared
WIS grid (for example, `quantile_025` and `quantile_500`). Values use log target
units. Wastewater uses log1p concentration; the clinical heads use log1p per
100,000. Do not interpret exponentiated Student-t values as finite means.

Event schema 1 contains high-incidence case labels and probabilities, fitted
value and alert thresholds, `event_decision_fixed_0_5`, and nullable
`event_decision_operating`. A missing operating decision denotes unsupported
calibration evidence. It is not a negative alert. Summary JSON contains metric
counts, support reasons, and horizon-band results. Hospitalization event scores
remain unsupported until source-level aggregation is defined.
