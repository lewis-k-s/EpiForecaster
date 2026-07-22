#!/bin/bash
#SBATCH --job-name=epi-snakemake
#SBATCH --account=bsc08
#SBATCH --partition=acc
#SBATCH --qos=acc_bscls
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=07:00:00

set -euo pipefail

RUN_ROOT="${1:?run root is required}"
SNAKEMAKE_BIN="${2:?Snakemake executable is required}"
EXPECTED_TASKS="${3:?expected task count is required}"
EXPECTED_CPUS_PER_TASK="${4:?expected CPUs per task is required}"
shift 4

test -d "$RUN_ROOT/source"
test -r "$RUN_ROOT/config.yaml"
test -x "$SNAKEMAKE_BIN"

if [ "${SLURM_JOB_NUM_NODES:-}" != "1" ] || \
   [ "${SLURM_NTASKS:-}" != "$EXPECTED_TASKS" ] || \
   [ "${SLURM_CPUS_PER_TASK:-}" != "$EXPECTED_CPUS_PER_TASK" ]; then
  echo "Expected one MN5 node with $EXPECTED_TASKS tasks and $EXPECTED_CPUS_PER_TASK CPUs per task" >&2
  exit 2
fi

exec 9>"$RUN_ROOT/.state/coordinator.run.lock"
if ! flock -n 9; then
  echo "A Snakemake coordinator already owns this run" >&2
  exit 1
fi

execution_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}_${SLURM_ARRAY_TASK_ID:-0}"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN_ROOT/.state/coordinator.job_id"
printf '%s\n' "$execution_id" > "$RUN_ROOT/.state/coordinator.execution_id"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUN_ROOT/source"
export XDG_CACHE_HOME="$RUN_ROOT/.cache"

exec "$SNAKEMAKE_BIN" \
  --snakefile "$RUN_ROOT/source/workflow/Snakefile" \
  --configfile "$RUN_ROOT/config.yaml" \
  --workflow-profile "$RUN_ROOT/source/workflow/profiles/mn5" \
  --directory "$RUN_ROOT" \
  --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
  "$@"
