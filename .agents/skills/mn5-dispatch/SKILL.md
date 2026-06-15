---
name: mn5-dispatch
description: Submit and manage training, preprocessing, and HPO jobs on the MareNostrum 5 (MN5) cluster. Handles code sync, job submission via cluster_dispatch.sh, monitoring, and result retrieval. Use when user wants to run experiments remotely on MN5.
allowed-tools: Bash, Read
---

# MN5 Remote Dispatch

End-to-end workflow for submitting and managing jobs on the MareNostrum 5 supercomputer via SSH.

## Configuration

Cluster identity, mode registry, and resource specs live in `.cluster/mn5.conf`. The dispatch CLI reads this config via `CLUSTER_CONF` (defaults to `.cluster/dispatch.conf`).

The full mode table (script paths, resource specs) is in `.cluster/mn5.conf` — consult it rather than hardcoding values.

## SSH Hosts

| Alias | Hostname | Purpose |
|-------|----------|---------|
| `mn5` | `glogin1.bsc.es` | Job submission, monitoring, dispatch script |
| `dt` | `transfer1.bsc.es` | Data transfer only (`rsync` of code and data files) |

**Never submit jobs via `dt`.** Transfer nodes have no HPC modules and only the `normal` QoS. Compute QoS names are only available from login nodes.

## Dispatch CLI

The dispatch script is `scripts/cluster/cluster_dispatch.sh`. `mn5_dispatch.sh` remains as a backward-compatible wrapper.

### Commands

```
cluster_dispatch.sh submit <mode> [sbatch-args...]   # Submit via sbatch (or invoke shell modes)
cluster_dispatch.sh alloc  <mode> [salloc-args...]   # Interactive salloc session with mode resources
cluster_dispatch.sh interactive [salloc-args...]      # One-hour MN5 ACC interactive GPU shell
cluster_dispatch.sh run    <mode> [args...]           # Execute .sbatch/.sh directly (no scheduler)
cluster_dispatch.sh status <job-id>                   # Show squeue + sacct
cluster_dispatch.sh logs   <job-id> [task-id]         # List log file paths
cluster_dispatch.sh tail   <job-id> [task-id] [lines] # Tail -f log files (default: 120 lines)
```

### Mode selection

Modes resolve via `MODE_<name>` variables in `.cluster/mn5.conf`. Use the exact mode name (e.g., `single`, `hpsearch`, `pretrain-finetune`). Shell-dispatch modes (`crossval`, `synth-pretrain`, `pretrain-finetune`) invoke their scripts directly instead of wrapping with `sbatch`.

### Remote path

The dispatch script runs on the compute node at `$REMOTE_BASE` (set in config; currently `/home/bsc/bsc008913/EpiForecaster`). All SSH commands use:

```bash
ssh mn5 '$REMOTE_BASE/scripts/cluster/cluster_dispatch.sh <command> ...'
```

---

## 1. Sync Operations

All sync scripts run **locally** and use `dt` (transfer node) for `rsync`.

### Sync code to remote

```bash
bash syncto_mn5.sh
```

### Sync raw data files to remote

```bash
bash syncto_mn5_raw.sh
```

### Sync training results back

```bash
bash syncback_from_mn5.sh [EXPERIMENT_NAME]
```

### Sync HPO study results back

```bash
bash syncback_hpsearch_from_mn5.sh [STUDY_NAME]
```

Always sync code before submitting unless the user explicitly says to skip it.

---

## 2. Submit Operations

All submit commands run **on the compute login node** via `ssh mn5`.

### Command Pattern

```bash
ssh mn5 'CONFIG=<config> /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit <mode> [sbatch-args...]'
```

Environment variables are passed inline:

```bash
ssh mn5 'CONFIG=configs/my_config.yaml OVERRIDES="training.epochs=5" /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=08:00:00'
```

### Dry Run (Smoke Test)

Always do a dry run first for new configurations:

```bash
ssh mn5 'DRY_RUN=1 /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=00:10:00'
```

### Interactive GPU Allocation

For fast iteration/profiling, use the purpose-built MN5 ACC interactive command.
It waits for Slurm once, requests `acc_interactive`, one GPU, 20 CPUs, one task,
and a one-hour walltime, then starts a compute-node shell with
`scripts/cluster/mn5_module_setup.sh` already sourced.

```bash
ssh -tt mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh interactive'
```

Inside that shell, run small batches directly without another Slurm queue wait:

```bash
$UV_RUN train epiforecaster --config configs/production_only/train_epifor_mn5_full.yaml --max-batches 5
```

Pass extra `salloc` arguments after `interactive` when needed:

```bash
ssh -tt mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh interactive --time=00:30:00'
```

The older generic allocation command is still available when you want a mode's
configured resources instead of the ACC interactive queue:

```bash
ssh -tt mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh alloc single'
```

This runs `salloc` with the resource spec from `RESOURCES_single` in config and drops into a shell.

### Direct Execution (no scheduler)

To run a job script directly (useful for local testing or debugging):

```bash
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh run single'
```

Sources `MODULE_SETUP` if configured, then `exec bash` the resolved script.

### Common Environment Variables for Training

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG` | Path to training YAML config | `configs/production_only/train_epifor_mn5_full.yaml` |
| `OVERRIDES` | Space-separated config overrides | empty |
| `MAX_EPOCHS` | Override max epochs | empty |
| `DRY_RUN` | Smoke-test mode (1 epoch, short run) | `0` |
| `EPIFORECASTER_MODEL_ID` | Custom run ID for the experiment | empty |

### Common Environment Variables for Preprocessing

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG` | Path to preprocessing YAML config | `configs/preprocess_full.yaml` |
| `PREPROCESS_TASK` | Preprocess subcommand (`epiforecaster` or `regions`) | `epiforecaster` |

### Common Environment Variables for HPO

| Variable | Description | Default |
|----------|-------------|---------|
| `STUDY_NAME` | HPO study name | `epiforecaster_hpo_v1` |
| `MAX_EPOCHS` | Max epochs per trial | `20` |
| `N_TRIALS` | Number of trials (empty = unlimited until timeout) | empty |
| `SAMPLER` | Optuna sampler: `tpe`, `cmaes`, `random` | `tpe` |
| `PRUNING_START_EPOCH` | Start pruning after N epochs | `10` |
| `SEED` | Reproducibility seed | `42` |

### Common Environment Variables for Crossval

| Variable | Description | Default |
|----------|-------------|---------|
| `CV_SEEDS` | Space-separated seed list | `42 43 44 45 46` |
| `CAMPAIGN_ID` | Campaign identifier | `crossval_<timestamp>` |
| `CROSSVAL_ENABLED` | Enable fold-based splitting | `0` |
| `CROSSVAL_NUM_FOLDS` | Number of folds | `5` |

### Common Environment Variables for Pretrain-Finetune

| Variable | Description | Default |
|----------|-------------|---------|
| `PRETRAIN_CONFIG` | Pretrain config path | `configs/production_only/train_epifor_mn5_synth_pretrain.yaml` |
| `FINETUNE_CONFIG` | Finetune config path | `configs/production_only/train_epifor_mn5_full.yaml` |
| `CHAIN_ID` | Chain identifier | `pretrain_finetune_<timestamp>` |
| `SUBMIT_REAL_EVAL` | Also submit eval on real data (`1`/`0`) | `1` |

### Submit Examples

**Single training run:**
```bash
bash syncto_mn5.sh
ssh mn5 'CONFIG=configs/production_only/train_epifor_mn5_full.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=08:00:00'
```

**Training with overrides:**
```bash
ssh mn5 'CONFIG=configs/train.yaml OVERRIDES="training.epochs=5 training.batch_size=48" /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=04:00:00'
```

**Preprocessing:**
```bash
ssh mn5 'CONFIG=configs/preprocess_full.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit preprocess'
```

**HPO with 8 workers:**
```bash
ssh mn5 'STUDY_NAME=hpo_v3 CONFIG=configs/train.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit hpsearch --array=0-7 --time=08:00:00'
```

**Dry run:**
```bash
ssh mn5 'DRY_RUN=1 /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=00:10:00'
```

**Cross-validation:**
```bash
ssh mn5 'CAMPAIGN_ID=cv_run1 CV_SEEDS="42 43 44 45 46" /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit crossval'
```

**Pretrain then finetune pipeline:**
```bash
ssh mn5 'CHAIN_ID=pf_$(date +%s) /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit pretrain-finetune --time=08:00:00'
```

---

## 3. Monitoring Operations

All monitoring commands run **on the compute login node** via `ssh mn5`.

### Check job status

```bash
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh status <JOB_ID>'
```

### List log files

```bash
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh logs <JOB_ID> [TASK_ID]'
```

### Tail logs (live)

```bash
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh tail <JOB_ID> [TASK_ID] [LINES]'
```

Default: 120 lines. For array jobs, specify `TASK_ID` to filter to a specific task.

**Examples:**
```bash
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh tail 39875209'
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh tail 39875209 0 200'
```

### Slurm Status Codes

| Code | Meaning |
|------|---------|
| `R` | RUNNING |
| `PD` | PENDING (waiting for resources) |
| `CG` | COMPLETING |
| `CD` | COMPLETED |
| `F` | FAILED |
| `PR` | PREEMPTED |

### Common Pending Reasons

| Reason | Meaning |
|--------|---------|
| `Priority` | Higher priority jobs in queue |
| `Resources` | Waiting for resources |
| `QOSGrpNodeLimit` | All nodes for QoS in use |
| `QOSGrpCpuLimit` | All CPUs for QoS in use |

---

## 4. MN5 Hardware Reference

### ACC Partition (GPU training)

| Resource | Per Node |
|----------|----------|
| GPUs | 4x NVIDIA H100 64GB HBM2 |
| CPUs | 2x Intel Xeon Platinum 8460Y+ (80 cores total) |
| Memory | 512 GB DDR5 |
| Local Storage | 480 GB NVMe |
| Network | 800 Gb/s InfiniBand |

### GPP Partition (CPU-only preprocessing, etc.)

| Resource | Per Node |
|----------|----------|
| CPUs | 2x Intel Xeon Platinum 8480+ (112 cores total) |
| Memory | 256 GB DDR5 (1 TB on highmem nodes) |
| Local Storage | 960 GB NVMe |
| Network | 100 Gb/s InfiniBand |

### GPU Allocation Rule (ACC)

**Request 20 CPUs per GPU.** Each ACC node has 80 CPUs and 4 GPUs.

| GPUs | `--gres=gpu:` | `--cpus-per-task` | `--ntasks` |
|------|---------------|-------------------|------------|
| 1 | `gpu:1` | 20 | 1 |
| 2 | `gpu:2` | 20 | 2 |
| 3 | `gpu:3` | 10 | 6 |
| 4 | `gpu:4` | 20 | 4 |

### Available QoS (account bsc08, user bsc008913)

**ACC (GPU):** `acc_bscls`, `acc_debug`, `acc_interactive`
**GPP (CPU):** `gp_bscls`, `gp_debug`, `gp_hbm`, `gp_interactive`

All jobs require `--account=bsc08`.

### Useful Cluster Commands (run via `ssh mn5`)

```bash
ssh mn5 'squeue -u $USER --format="%.10i %.9P %.30j %.2t %.10M %.10l %R"'
ssh mn5 'sacctmgr show assoc where user=$USER format=account,qos -P'
ssh mn5 'sshare -la'
```

---

## 5. End-to-End Examples

### Full training workflow

```bash
# 1. Sync code (via dt)
bash syncto_mn5.sh

# 2. Dry run to verify (via mn5)
ssh mn5 'DRY_RUN=1 CONFIG=configs/production_only/train_epifor_mn5_full.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=00:10:00'

# 3. Submit real job
ssh mn5 'CONFIG=configs/production_only/train_epifor_mn5_full.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit single --time=08:00:00'

# 4. Monitor
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh status <JOB_ID>'
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh tail <JOB_ID>'

# 5. When done, sync results back (via dt)
bash syncback_from_mn5.sh mn5_epiforecaster_full
```

### Preprocessing workflow

```bash
# 1. Sync code + raw data
bash syncto_mn5.sh
bash syncto_mn5_raw.sh

# 2. Submit preprocess job
ssh mn5 'CONFIG=configs/preprocess_full.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit preprocess'

# 3. Monitor
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh status <JOB_ID>'
```

### HPO workflow

```bash
# 1. Sync code
bash syncto_mn5.sh

# 2. Submit HPO
ssh mn5 'STUDY_NAME=my_hpo_study CONFIG=configs/train.yaml /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit hpsearch --array=0-3 --time=08:00:00'

# 3. Monitor specific worker
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh tail <JOB_ID> 0 200'

# 4. Sync HPO results (via dt)
bash syncback_hpsearch_from_mn5.sh my_hpo_study
```

### Pretrain-finetune pipeline

```bash
# 1. Sync code
bash syncto_mn5.sh

# 2. Submit chained pipeline
ssh mn5 'CHAIN_ID=pf_experiment /home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh submit pretrain-finetune --time=08:00:00'

# 3. Sync all results (via dt)
bash syncback_from_mn5.sh mn5_epiforecaster_synth_pretrain
```

### Interactive debugging/profiling on compute node

```bash
# Get a one-hour ACC interactive shell with 1 GPU + 20 CPUs
ssh -tt mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/cluster_dispatch.sh interactive'

# Then run short training/profiling loops inside the allocated compute shell
$UV_RUN train epiforecaster --config configs/production_only/train_epifor_mn5_full.yaml --max-batches 5
```
