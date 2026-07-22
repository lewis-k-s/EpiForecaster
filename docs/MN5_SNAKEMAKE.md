# MN5 workflow

Snakemake is the production entry point for MN5 training, evaluation, HPO,
ablation, and artifact promotion. Each workflow uses one preallocated ACC node.
Snakemake runs as the batch process inside that allocation and launches the
scientific work as one or more `srun` job steps.

There are no nested `sbatch` submissions, no worker array, and no separate GPP
coordinator allocation.

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

Single training and evaluation use `N=1`. HPO and ablation use the configured
worker count, currently `N=4` in the production templates. `N` must be between
one and the four GPUs available on the single ACC node.

The batch shell owns the allocation and runs the lightweight Snakemake manager.
A study worker rule launches:

```text
srun --nodes=1 --ntasks=N --ntasks-per-node=N \
     --cpus-per-task=20 --gpus-per-task=1 --exact \
     --kill-on-bad-exit=1
```

Each rank receives one GPU and 20 CPUs. `SLURM_PROCID` selects a stable worker
ID; Optuna workers coordinate only through their journal on GPFS. They do not
communicate with Snakemake while running. If one rank fails, Slurm terminates
the worker step and the batch job fails.

After every HPO worker marker exists, separate DAG stages resolve
`study.best_trial`, validate and copy its exact checkpoint, evaluate it, promote
the declared artifacts, and finally delete losing checkpoint directories.
Ablation analysis likewise runs once after the complete seed/ablation matrix is
terminal; workers never race to launch analysis.

## Storage and identity

Durable state lives under project GPFS:

```text
/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster-DVC/
  cache/                         # shared, locked DVC cache
  storage/                       # content-addressed DVC remote
  containers/                    # immutable SIFs
  tooling/
    cache/                       # shared uv cache
    envs/<tooling-key>/          # immutable Snakemake environment
    requests/<tooling-key>/      # lock inputs and ready marker
  promoted/<run-id>/             # selected artifacts
  runs/
    by-slurm/<job-id>_0          # allocation lookup
    <run-id>/
      config.yaml                # immutable rendered config
      manifest.yaml
      .state/dependencies/       # full-source, stage-labeled contracts
      source/                    # immutable worktree snapshot
      optuna/
      logs/
      outputs/
```

Run IDs are reserved atomically before submission and are never overwritten.
The allocation identity is `<job-id>_0`. Container executions add Slurm step
and process rank, yielding `<job-id>_0_<step-id>_<rank>` and preventing scratch
collisions between concurrent ranks.

The manifest records the workflow kind and public target, Git SHA, exact source
snapshot SHA-256, DVC hashes, SIF SHA-256, workflow tooling key, Snakemake
version and location, and rendered config SHA-256. Study workflows also record
their study identity.

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
and the dependency-group name. Prepare it idempotently on `wifi-acc` with:

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
```

`mn5_train.yaml` pins `workflow.target: train`, so its default terminal artifact
is the training manifest and checkpoint. `train_evaluate` is the explicit
composition when both stages are wanted. Training and HPO trials force
`training.run_final_evaluation=false`; evaluation is owned by the downstream
rule. Ablation trials retain final test metrics because paired-seed aggregation
consumes those per-trial artifacts.

Each data, training, evaluation, HPO, ablation, selection, and promotion
contract fingerprints the same complete scientific source inventory together
with its relevant config fragment and SIF digest. This deliberately favors
conservative invalidation and auditable provenance over inferring transitive
Python dependencies: any scientific source change invalidates every stage
contract, while documentation and tests remain excluded. Remote run snapshots
remain immutable, so changed source is normally prepared as a new run rather
than mutating an existing one.

Training declares its checkpoint and `training_manifest.json`. Evaluation
declares its summary JSON, node and per-head metric tables, optional granular
Parquet, and a hash-bearing `evaluation_manifest.json`. Ablation declares its
coverage, run-context, aggregate table, and artifact manifest. Completion
markers remain receipts rather than the sole scientific output contract.

`stage_data` materializes the pinned DVC payloads once in the run workspace.
Each worker then copies those inputs into rank-specific node-local scratch and
mounts them read-only over the immutable source tree. GPFS remains authoritative
for the Optuna journal, logs, checkpoints, outputs, and promotion records.

## Remote endpoints and queue visibility

The transfer and compute aliases do not expose the same Slurm controller:

- `dt` reaches the `bsc_transfer` controller. Use it for source/data transfer
  and for inspecting files already present on project GPFS.
- `mn5`, `wifi`, and `wifi-acc` reach the `marenostrum5` compute scheduler.
  Use these endpoints for `sbatch`, `squeue`, `sacct`, `scontrol show job`, and
  `scancel`; `wifi-acc` is the VPN-only ACC endpoint.

Consequently, an empty `squeue` or `Invalid job id` on `dt` says nothing about
an MN5 compute job. Shared visibility of GPFS logs does not imply shared Slurm
job visibility. Verify the endpoint when diagnosing this ambiguity:

```bash
ssh dt "scontrol show config | grep -E '^ClusterName|^SlurmctldHost'"
ssh wifi-acc "scontrol show config | grep -E '^ClusterName|^SlurmctldHost'"
```

Monitor ACC scheduler state on `wifi-acc` (or another available MN5 compute
login) and use `dt` only for filesystem evidence such as logs and markers.

## Prepare and submit

Single training is the default:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <run-id>
bash scripts/workflow/mn5_workflow.sh dry-run <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>
bash scripts/workflow/mn5_workflow.sh status <run-id>
```

Use the public target layer rather than internal rule names. On a prepared
training run, `train_evaluate` composes the two stages and `evaluate` pulls in
training when its canonical checkpoint is absent:

```bash
bash scripts/workflow/mn5_workflow.sh run <run-id> train
bash scripts/workflow/mn5_workflow.sh run <run-id> evaluate
bash scripts/workflow/mn5_workflow.sh run <run-id> train_evaluate
```

Select study templates explicitly:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <hpo-run-id> \
  --config workflow/config/mn5.yaml
bash scripts/workflow/mn5_workflow.sh prepare <ablation-run-id> \
  --config workflow/config/mn5_ablation.yaml
```

For evaluation without training, copy `workflow/config/mn5_evaluate.yaml`, set
`evaluation.checkpoint` to an absolute GPFS checkpoint path, and pass that
complete config with `--config`. Preparation fails before transfer if the field
is unset or relative.

`run` reads the allocation fields from the immutable remote config and passes
them to `sbatch`, overriding the script defaults. A launch lock, recorded job
ID, and coordinator lifetime lock prevent duplicate submissions.

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
