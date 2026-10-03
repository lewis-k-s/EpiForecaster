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
allocation, while `run-interactive` attaches to a compatible held
`acc_interactive` or `acc_debug` allocation through `srun --external-launcher`.
Both adapters use the same snapshot-owned coordinator, DAG, containers, data
contracts, logs, manifests, and scientific artifact rules.

The workflow does not use nested `sbatch` submissions, worker arrays, or a
separate GPP coordinator allocation.

## Native model and derived source sets

This worktree executes only NativeTrainer with the heterogeneous graph encoder.
The municipal model belongs to the separate legacy worktree. All active native
workflow templates stage ABS excess deaths and the version-4 native manifest.

Single training has the dependency chain `stage_data -> native_source_sets ->
train_model -> evaluate_model`. The split step uses the effective training
config and processed inputs. It writes `inputs/native_source_sets/assignments.parquet`
and a small `manifest.json`, then publishes a run-owned DVC pointer under
`outputs/split_publication/`. The receipt records input identities and hashes.
Generated membership is excluded from the source bundle.

Source folds are deterministic for a dataset and eligibility policy. The asset
records processed content hashes, the full eligible query hash, split seed,
window length, horizon, stride, mask thresholds, and calendar boundaries.
Target intervals are purged across time boundaries. Training rejects any
changed input or policy. It records the exact split asset in the checkpoint.
Independent evaluation checks the checkpoint identities and has no trainer
resume lifecycle. Final evaluation honors `training.run_final_evaluation`;
the DAG sets it to false and owns the canonical evaluation exports.

The `mn5_native_wastewater_history_repair.yaml` preprocessing template rebuilds
only `history_log1p` from the pinned parent Zarr's raw concentration and detection
limits. It uses the parent's fixed filter parameters. QA checks all other
variables, coordinates, and attributes for equality, then checks truncated and
future-poisoned histories at calendar and initialization boundaries. The run
writes a new Zarr under `outputs/preprocessing/` and publishes its DVC pointer
and receipt. The parent asset stays unchanged. The QA record binds the parent
DVC identity, source hashes, parameters, and output tree hash.

Select a published version with `native.wastewater_filename` under `native.root`
and stage its DVC pointer. Split generation, training, evaluation, and example
exports use that same filename. The filename defaults to `obs_wastewater.zarr`.
The old native manifest records the parent input bundle; the repair receipt
records the new asset's lineage. It does not imply that the other assets were
rebuilt.


## GPU telemetry

Every command launched with `mn5_container_exec.sh --nv` has a device sampler
in the same worker step. This includes single training, evaluation, HPO,
cross-validation, and ablation workers. CPU commands do not start a sampler.
No extra Slurm allocation or concurrent Snakemake rule is needed.

The sampler queries `nvidia-smi` about every ten seconds. It records GPU and
memory-controller utilisation, used and total device memory in MiB, power in
watts, and temperature in degrees Celsius. Memory-controller utilisation is
activity, not the fraction of device memory occupied. Device memory includes
more than PyTorch tensor allocations. Missing counter values remain `N/A`.

CUDA driver enumeration resolves visible device ordinals to UUIDs without
creating a CUDA context. Queries select those UUIDs explicitly. Missing GPU
visibility, unavailable driver tools, or query failures produce telemetry
warnings; they do not change the scientific command's exit status. The sampler
never falls back to querying every GPU on a shared node. This contract supports
full GPUs; MIG visibility is not supported.

Each worker invocation writes a unique directory under
`outputs/gpu_telemetry/`. It contains `gpu_samples.csv` and `manifest.json`.
The manifest records the execution attempt, Slurm job and step, rank, logical
worker ID, GPU UUIDs, sampling interval, UTC start/end times, sample/error
counts, scientific exit code, and the completed CSV hash. Retries and worker
waves have separate directories. An HPO worker's trace can span several trials;
use the worker ID and UTC timestamps to join it to trial records.

Sampling stops when the command exits. SIGINT and SIGTERM are forwarded to
the scientific process group. A hard scheduler kill can leave a partial CSV
and a running manifest; telemetry does not prove scientific completion.
These are optional diagnostic artifacts, not required completion outputs.
The sampler uses no PyTorch profiler and adds no training-loop synchronisation.
It applies to newly prepared source bundles; existing immutable runs retain
their deployed source.

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
train_evaluate_report: train_model -> evaluate_model -> build_run_report
hpo:             launch_optuna_workers -> select_best_trial
                 -> evaluate_model -> promote_best_model
                 -> cleanup_losing_checkpoints
ablation:        launch_ablation_workers -> finalize_ablation
crossval:        train_crossval_folds -> evaluate_crossval_folds
                 -> finalize_crossval
preprocess:      stage_data -> preprocess_dataset -> publish_preprocessed
```

Synthetic model pretraining is maintained in the municipal worktree. This
worktree retains synthetic preprocessing as a separate data preparation task.

The corrected 20-run corpus uses
`workflow/config/mn5_preprocess_sirhd_reporting_v1_20run.yaml`. The immutable
run owns the processed Zarr, QA record, generated DVC pointer, and publication
receipt. Promote the pointer to the source repository only after these records
pass review.

`mn5_train.yaml` and `mn5_train_smoke.yaml` pin
`workflow.target: train_evaluate_report`. Evaluation writes validation and test
outputs from the selected checkpoint. The final CPU rule produces tables and
figures at `outputs/reports/single_run/report.html`. It does not add a qualitative
review. If an upstream rule fails, no report is produced. See
[single-run reports](SINGLE_RUN_REPORTS.md) for the input contract.
`train` ends at training provenance; `train_evaluate` ends at evaluation.
Training and HPO trials force
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
allocation is active. To attach to an `acc_debug` allocation, start it with
`salloc` and pass its job ID explicitly. The adapter rejects `sbatch` jobs,
attaches the immutable workflow to the held allocation, and never calls
`sbatch` itself.

The selected public target is immutable after preparation. Choose `train`,
`train_evaluate`, or `train_evaluate_report` in the template's `workflow.target`
before `prepare`; `run`
does not accept targets, overrides, or arbitrary Snakemake arguments.
Preparation accepts workflow overrides before the run is frozen; see
`workflow/config/README.md`.

Select study templates explicitly:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <hpo-run-id> \
  --config workflow/config/mn5.yaml
bash scripts/workflow/mn5_workflow.sh prepare <ablation-run-id> \
  --config workflow/config/mn5_ablation.yaml
bash scripts/workflow/mn5_workflow.sh prepare <crossval-run-id> \
  --config workflow/config/mn5_crossval.yaml
bash scripts/workflow/mn5_workflow.sh prepare <preprocess-run-id> \
  --config workflow/config/mn5_preprocess_sirhd_reporting_v1_20run.yaml
```

For evaluation without training, copy `workflow/config/mn5_evaluate.yaml`, set
`evaluation.checkpoint` to an absolute GPFS checkpoint path, and pass that
complete config with `--config`. Preparation fails before transfer if the field
is unset or relative.

For downstream recovery after completed training, an evaluation-only run can set
`evaluation.splits: [val, test]`. To retain the original training records, set
`evaluation.training_model_root` to the absolute producer model directory and
`provenance.training_evidence_retention: dvc`. Preparation verifies the selected
checkpoint against the producer manifest, checks the canonical record hashes,
and pins read-only copies under `inputs/training_evidence`. The new manifest and
retention receipt record the original producer identity separately from the
evaluation run. This does not repeat training or modify the producer run.
The coordinator mounts external checkpoint directories read-only. If the
producer model directory contains the checkpoint, one parent directory mount
covers both inputs. This avoids overlapping mounts and missing file mount
points in the contained filesystem. Scientific stages use the pinned copies.

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
