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

Single training and evaluation use `N=1`. Cross-validation, HPO, and ablation
use their configured fold or worker count, currently `N=4` in production
templates. `N` must be between one and the four GPUs on the ACC node.

The batch shell owns the allocation and runs the lightweight Snakemake manager.
A parallel worker rule launches:

```text
srun --nodes=1 --ntasks=N --ntasks-per-node=N \
     --cpus-per-task=20 --gpus-per-task=1 --exact \
     --kill-on-bad-exit=1
```

Each rank receives one GPU and 20 CPUs. `SLURM_PROCID` becomes an explicit
logical worker ID. Optuna workers coordinate through their GPFS journal;
cross-validation ranks own deterministic fold directories. Workers do not
communicate with Snakemake while running. If one rank fails, Slurm terminates
the worker step and the batch job fails.

After every HPO worker marker exists, separate DAG stages resolve
`study.best_trial`, validate and copy its exact checkpoint, evaluate it, promote
the declared artifacts, and finally delete losing checkpoint directories.
Ablation workers compile a checked-in declarative campaign into a deterministic
`condition × fold × paired-model-seed` queue.  Optuna's JournalStorage is
strictly durable coordination: it performs no TPE or design sampling. Analysis
runs only after the exact declared identity matrix is terminal and provenance
matched; it uses run-level aggregate metrics and never pools granular windows.

## Storage and identity

Durable state lives under project GPFS:

```text
/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/
  EpiForecaster-DVC/
    cache/                         # shared, locked DVC cache
    storage/                       # content-addressed DVC remote
    containers/                    # immutable SIFs
    tooling/
      cache/                       # shared uv cache
      envs/<tooling-key>/          # immutable Snakemake environment
      requests/<tooling-key>/      # lock inputs and ready marker
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
tooling before atomically reserving the run ID and publishing the final run.
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

## Environments and data

The immutable SIF contains scientific dependencies and DVC from the `runtime`
dependency group. It excludes EpiForecaster source and Snakemake. Ordinary
Python or Snakefile changes therefore create a new run snapshot without
rebuilding the SIF.

The host-side coordinator environment contains only the locked `workflow`
group (`snakemake` and `dvc`). Its key is derived from `uv.lock`, Python 3.11,
the dependency-group name, and the transferred tooling-bootstrap digest.
Prepare it idempotently on `wifi-acc` with:

```bash
bash scripts/workflow/mn5_workflow.sh tooling
```

The top-level Snakefile includes domain rules from `workflow/rules/`. Every
scientific stage consumes a content-addressed dependency contract and the DVC
data manifest. The public targets are:

```text
train:           prepare_run -> stage_data -> train_model
evaluate:        prepare_run -> stage_data -> evaluate_model
train_evaluate:  train_model -> evaluate_model
hpo:             launch_optuna_workers -> select_best_trial
                 -> evaluate_model -> promote_best_model
                 -> cleanup_losing_checkpoints
ablation:        launch_ablation_workers -> finalize_ablation
crossval:        train_crossval_folds -> evaluate_crossval_folds
                 -> finalize_crossval
preprocess:      stage_data -> preprocess_dataset -> publish_preprocessed
```

Synthetic pretraining uses the same `train` entity through
`workflow/config/mn5_synth_pretrain.yaml`. Its processed Zarr is materialized
for training, while the raw synthetic Zarr is a provenance-only DVC target.

The corrected 20-run corpus uses
`workflow/config/mn5_preprocess_sirhd_reporting_v1_20run.yaml`. The immutable
run owns the processed Zarr, QA record, generated DVC pointer, and publication
receipt. Promote the pointer to the source repository only after these records
pass review.

`mn5_train.yaml` pins `workflow.target: train`, so its default terminal artifact
is the training manifest and checkpoint. `train_evaluate` is the explicit
composition when both stages are wanted. Training and HPO trials force
`training.run_final_evaluation=false`; evaluation is owned by the downstream
rule. Ablation trials retain final test metrics because paired-seed aggregation
consumes those per-trial artifacts.

### Declarative ablation campaigns

`configs/campaigns/ablation_default_v1.yaml` is the versioned scientific
authority for the production ablation matrix. It names the baseline, hypotheses,
fixed structured overrides, the allowed hydrated-config difference paths for every
condition, paired model seeds, the persisted representative 10-fold split manifest,
metrics, budget policy, and completeness and provenance requirements.
`workflow/config/mn5_ablation.yaml` only supplies execution resources, study name,
and that campaign path.

Before work is queued, every condition is hydrated through
`EpiForecasterConfig`, including model-parameter promotion and explicit defaults.
Conditions with the same hydrated configuration are rejected, as are differences
outside their declared paths. `equal_budget_policy.paths` is a non-empty list of
resolved configuration paths (currently `training.epochs` and
`training.max_batches`) whose values must be identical for every condition.

`ablation_default_v1.yaml` uses the persisted ten-fold manifest but evaluates three
coverage-selected folds (`[1, 4, 5]`) and two paired model seeds (`[42, 43]`) by
default. The selected folds hold out 48 regions: 33 Barcelona, 7 Girona, 4 Lleida,
and 4 Tarragona. This keeps geographic coverage while reducing the production matrix
to `28 conditions × 3 folds × 2 seeds = 168` identities. The same model seed is used
for every condition within each fold, so paired contrasts control initialization,
DataLoader-order, and other stochastic-training variation. `training.split_seed` does
not change these persisted fold memberships.

The default matrix is a screening study. Its results can justify a follow-up that
expands the declared fold or seed replicates. `smoke_fold_indices` (`[1, 4]`) are a
strict subset of the production folds. The checked-in
`workflow/config/mn5_ablation_coverage_smoke.yaml` runs every default condition and
paired model seed over that bounded subset with small batches; it is an ACC integration
acceptance check, not a production estimate. Bounded smoke validation still requires
the exact identity matrix, terminal trials, and provenance, but permits an undefined
primary row when a target has zero observed opportunities. The smaller
`ablation_smoke_v1.yaml` remains the eight-identity infrastructure smoke. Both
campaigns are expanded in memory; the workflow never generates per-run YAML files. A
campaign with fold indices must declare an existing persisted split manifest, and
undeclared resolved-config drift fails before a worker starts training.

### Ablation allocation closeout

An ablation config declares `closeout_signal_seconds`. The batch launcher
requests `SIGUSR1` that many seconds before its wall time ends. On this signal,
a worker stops claiming new identities, completes its active trial when
possible, and records a durable `partial` receipt for remaining work. The
parallel launcher then exits with a controlled incomplete status. Snakemake
does not write `workers_complete.done` or run campaign analysis for that
allocation. Submit the same immutable run again to continue its exact identity
matrix; existing terminal trials are retained.

Optuna values always minimize the validation-selection coordination value and
record its actual metric name and value on the trial. They are not the campaign
primary endpoint: the analyzer alone applies the campaign-declared held-out
primary metric and direction to aggregate artifacts. Recovery marks stale
`RUNNING` attempts terminal before enqueuing a replacement. Failed or pruned
historical attempts are retained, but completion requires no nonterminal attempt,
exactly one `COMPLETE` trial, and exactly one canonical output per scientific
identity.

Each output sidecar must agree with its persisted config: `fold` is `null`
unless cross-validation is enabled. Every canonical output has a non-empty
aggregate metric table with `target` and the campaign primary metric; those
run-level aggregate rows, rather than granular windows, are the sole inputs to
campaign endpoint aggregation.

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
Each worker then copies those inputs into rank-specific node-local scratch and
mounts them read-only over the immutable source tree. GPFS remains authoritative
for the Optuna journal, logs, checkpoints, outputs, and promotion records.

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
