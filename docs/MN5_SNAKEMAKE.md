# MN5 workflow

Snakemake is the production entry point for MN5 training, evaluation,
cross-validation, HPO, ablation, and artifact promotion. Each workflow uses one
preallocated ACC node.
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
Ablation analysis likewise runs once after the complete seed/ablation matrix is
terminal; workers never race to launch analysis.

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
      by-slurm/<job-id>_0          # allocation lookup
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
hpo:             launch_optuna_workers -> select_best_trial
                 -> evaluate_model -> promote_best_model
                 -> cleanup_losing_checkpoints
ablation:        launch_ablation_workers -> finalize_ablation
crossval:        train_crossval_folds -> evaluate_crossval_folds
                 -> finalize_crossval
```

Synthetic pretraining uses the same `train` entity through
`workflow/config/mn5_synth_pretrain.yaml`. Its processed Zarr is materialized
for training, while the raw synthetic Zarr is a provenance-only DVC target.

`mn5_train.yaml` pins `workflow.target: train`, so its default terminal artifact
is the training manifest and checkpoint. `train_evaluate` is the explicit
composition when both stages are wanted. Training and HPO trials force
`training.run_final_evaluation=false`; evaluation is owned by the downstream
rule. Ablation trials retain final test metrics because paired-seed aggregation
consumes those per-trial artifacts.

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
```

For evaluation without training, copy `workflow/config/mn5_evaluate.yaml`, set
`evaluation.checkpoint` to an absolute GPFS checkpoint path, and pass that
complete config with `--config`. Preparation fails before transfer if the field
is unset or relative.

The snapshot-owned `remote_run.sh` reads submission fields from the immutable
dispatch record and passes them to `sbatch`, overriding the script defaults. A
launch lock, recorded job ID, and coordinator lifetime lock prevent duplicate
submissions. Dry-run, status, and unlock likewise use the runner contained in
the prepared source rather than code streamed from a caller's current checkout.
Runs without the current dispatch record and transferred runner are rejected.

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
