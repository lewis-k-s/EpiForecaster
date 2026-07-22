---
name: mn5-dispatch
description: Prepare, submit, monitor, and inspect immutable EpiForecaster Snakemake workflows on MareNostrum 5. Use for training, evaluation, cross-validation, HPO, and ablation work on MN5.
allowed-tools: Bash, Read
---

# MN5 Snakemake dispatch

Use `scripts/workflow/mn5_workflow.sh` as the production interface for model
training, evaluation, cross-validation, HPO, and ablations. It snapshots the
current worktree, pins DVC and container identities, transfers through `dt`, and
submits one mode-sized ACC allocation. Do not sync into or execute a mutable
remote checkout for these workflows.

Read `.cluster/mn5.conf`, `docs/MN5_SNAKEMAKE.md`, and the selected workflow
config before dispatch. Never run a config under `configs/production_only/`
locally.

## Endpoint boundary

| Alias | Purpose |
| --- | --- |
| `mn5` | Normal MN5 login; compute submission and scheduler monitoring |
| `wifi` | VPN-only GPP login fallback |
| `wifi-acc` | VPN-only ACC login; dependency fetching, submission, monitoring |
| `dt` | Project-GPFS transfer and log/artifact inspection only |

`dt` uses the `bsc_transfer` Slurm controller. MN5 compute jobs use the
`marenostrum5` controller. An empty queue or unknown job on `dt` says nothing
about an MN5 job. Use `mn5`, `wifi`, or `wifi-acc` for `sbatch`, `squeue`,
`sacct`, `scontrol`, and `scancel`; use `dt` for GPFS files.

When a VPN-only alias is needed, use the `workvpn` skill first. `mn5` and `dt`
normally work without VPN.

## Workflow shapes

All model workflows run Snakemake inside one ACC allocation. MN5 derives host
memory from CPU count, so no path may add `--mem` or `--mem-per-cpu`.

| Workflow | Template | Tasks | GPUs | CPUs/task |
| --- | --- | ---: | ---: | ---: |
| Single training | `workflow/config/mn5_train.yaml` | 1 | 1 | 20 |
| Synthetic pretraining | `workflow/config/mn5_synth_pretrain.yaml` | 1 | 1 | 20 |
| Evaluation only | `workflow/config/mn5_evaluate.yaml` | 1 | 1 | 20 |
| Cross-validation | `workflow/config/mn5_crossval.yaml` | configured folds | same | 20 |
| HPO | `workflow/config/mn5.yaml` | configured workers | same | 20 |
| Ablation | `workflow/config/mn5_ablation.yaml` | configured workers | same | 20 |

Study workers share an Optuna journal on GPFS and run as ranks of fail-fast
`srun` steps. HPO selection, evaluation, promotion, and losing-checkpoint cleanup
are separate dependent rules inside the same allocation. Cross-validation uses
separate multi-rank training and evaluation rules before one finalizer. Ablation
analysis runs once after every expected seed/ablation trial is terminal.

## Commands

The default prepare template is single training:

```bash
bash scripts/workflow/mn5_workflow.sh tooling
bash scripts/workflow/mn5_workflow.sh prepare <run-id>
bash scripts/workflow/mn5_workflow.sh dry-run <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>
bash scripts/workflow/mn5_workflow.sh status <run-id>
```

Select cross-validation, HPO, or ablation explicitly:

```bash
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config workflow/config/mn5_crossval.yaml
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config workflow/config/mn5.yaml
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config workflow/config/mn5_ablation.yaml
```

Pin the public target in `workflow.target` before preparation. The immutable
run command accepts no target, override, or additional Snakemake arguments.

For evaluation-only work, copy `workflow/config/mn5_evaluate.yaml`, set
`evaluation.checkpoint` to an absolute GPFS checkpoint path, then provide the
completed config through `--config`. Preparation rejects an unset or relative
checkpoint.

If a previous coordinator is terminal but left a Snakemake lock:

```bash
bash scripts/workflow/mn5_workflow.sh unlock <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>
```

Source is transferred to a hidden staging leaf, verified by the runner included
in that exact snapshot, and then published under an atomically reserved run ID.
Run IDs are never overwritten or reused. The target pinned in the run config
writes `.state/workflow.done`; re-running `run` while a coordinator is active
is idempotent.

## Monitoring

Query scheduler state on the matching MN5 controller:

```bash
ssh wifi-acc 'squeue -j <job-id> -o "%.18i %.9P %.20j %.2t %.10M %R"'
ssh wifi-acc 'sacct -j <job-id> --format=JobIDRaw,State,ExitCode,Elapsed -X'
```

Inspect durable evidence separately through `dt` under:

```text
/gpfs/projects/bsc08/shared_projects/MePreCiSa/EpiForecaster/EpiForecaster-DVC/runs/<run-id>/
```

Check the coordinator stdout/stderr, rule logs, worker-rank logs, manifest,
completion markers, journal, checkpoint hashes, evaluation outputs, and any
promotion record. Scheduler state and filesystem evidence are complementary;
neither substitutes for the other.

## Failure handling

Diagnose the first causal error, not the final missing-output message. Common
checks are:

- allocation config: one ACC node, 20 CPUs/task, one GPU/task, no memory flag;
- source inventory and snapshot hash verification;
- immutable SIF sidecar verification;
- DVC pointer hashes, cache permissions, and staged bind roots;
- all study worker markers before finalization;
- exact checkpoint path and SHA-256 before evaluation/promotion;
- scratch cleanup warnings versus the scientific process exit status.

Do not delete a run, journal, checkpoint, promotion, or shared DVC cache object
without explicit user authorization. Do not run `dvc gc` while a run or
promotion may still reference the cache.

Preprocessing is the upstream DVC-producing boundary. Model execution is
supported only through the immutable workflow interface, with no direct model
SBATCH, array, or standalone study-worker path.
