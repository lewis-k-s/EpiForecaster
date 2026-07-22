#!/bin/bash

# Launch all journal-backed study workers as one fail-fast Slurm job step in
# the already allocated ACC node.

set -euo pipefail

RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
WORKERS=""
CPUS_PER_TASK=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \\
  --workers 4 --cpus-per-task 20 -- COMMAND [ARG...]
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --sif)
      SIF="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$RUN_ROOT" ] || [ -z "$SOURCE_ROOT" ] || [ -z "$SIF" ] || [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-4]$ ]] || [ "$CPUS_PER_TASK" != "20" ]; then
  echo "MN5 studies require 1-4 workers with 20 CPUs each" >&2
  exit 2
fi
if [ "${SLURM_JOB_NUM_NODES:-}" != "1" ]; then
  echo "Expected one allocated ACC node, got ${SLURM_JOB_NUM_NODES:-unset}" >&2
  exit 1
fi
if [ "${SLURM_NTASKS:-}" != "$WORKERS" ]; then
  echo "Expected SLURM_NTASKS=$WORKERS, got ${SLURM_NTASKS:-unset}" >&2
  exit 1
fi
if [ "${SLURM_CPUS_PER_TASK:-}" != "$CPUS_PER_TASK" ]; then
  echo "Expected SLURM_CPUS_PER_TASK=$CPUS_PER_TASK, got ${SLURM_CPUS_PER_TASK:-unset}" >&2
  exit 1
fi

worker_script="$SOURCE_ROOT/scripts/workflow/run_optuna_worker.sh"
test -x "$worker_script"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/optuna/workers"

srun \
  --nodes=1 \
  --ntasks="$WORKERS" \
  --ntasks-per-node="$WORKERS" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gpus-per-task=1 \
  --exact \
  --kill-on-bad-exit=1 \
  --open-mode=append \
  --output="$RUN_ROOT/logs/optuna-worker-%t.log" \
  --error="$RUN_ROOT/logs/optuna-worker-%t.log" \
  "$worker_script" \
  --run-root "$RUN_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --sif "$SIF" \
  --cpus "$CPUS_PER_TASK" \
  -- "$@"

for worker_index in $(seq 0 $((WORKERS - 1))); do
  printf -v worker_id '%03d' "$worker_index"
  marker="$RUN_ROOT/optuna/workers/$worker_id.done"
  if [ ! -s "$marker" ]; then
    echo "Worker $worker_id exited without a completion marker: $marker" >&2
    exit 1
  fi
done
