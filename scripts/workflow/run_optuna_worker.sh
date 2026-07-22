#!/bin/bash

# Execute one Optuna worker rank inside the scientific container. This script is
# launched once per srun task and writes its marker only after a clean exit.

set -euo pipefail

RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
CPUS=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \\
  --cpus N -- COMMAND [ARG...]
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
    --cpus)
      CPUS="$2"
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
worker_index="${SLURM_PROCID:?SLURM_PROCID is required for an Optuna worker}"
if [[ ! "$worker_index" =~ ^[0-9]+$ ]]; then
  echo "Worker rank must be a non-negative integer: $worker_index" >&2
  exit 1
fi
printf -v worker_id '%03d' "$worker_index"
marker="$RUN_ROOT/optuna/workers/$worker_id.done"
if [ -s "$marker" ]; then
  echo "Worker $worker_id is already complete; skipping"
  exit 0
fi

"$SOURCE_ROOT/scripts/workflow/mn5_container_exec.sh" \
  --run-root "$RUN_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --sif "$SIF" \
  --stage-data \
  --nv \
  --cpus "$CPUS" \
  -- env "EPIFORECASTER_WORKER_ID=$worker_index" "$@"

temporary="$marker.${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-step}.$worker_index"
printf '%s\n' \
  "worker_id=$worker_index" \
  "job_id=${SLURM_JOB_ID:-}" \
  "step_id=${SLURM_STEP_ID:-}" \
  "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
mv "$temporary" "$marker"
