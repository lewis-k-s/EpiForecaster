#!/bin/bash

set -euo pipefail

RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
CPUS=""
WORKER_GROUP=""
SCIENTIFIC_COMMAND=()

# Print the per-rank worker interface.
usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \
  --cpus N --worker-group NAME -- COMMAND [ARG...]
EOF
}

# Parse rank metadata and retain the scientific command separately.
parse_arguments() {
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
      --worker-group)
        WORKER_GROUP="$2"
        shift 2
        ;;
      --)
        shift
        SCIENTIFIC_COMMAND=("$@")
        return 0
        ;;
      *)
        usage >&2
        return 2
        ;;
    esac
  done
}

# Validate the configured worker command before reading scheduler state.
validate_arguments() {
  if [ -z "$RUN_ROOT" ] \
    || [ -z "$SOURCE_ROOT" ] \
    || [ -z "$SIF" ] \
    || [ -z "$WORKER_GROUP" ] \
    || [ "${#SCIENTIFIC_COMMAND[@]}" -eq 0 ]; then
    usage >&2
    return 2
  fi

  if [[ ! "$WORKER_GROUP" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "Invalid worker group: $WORKER_GROUP" >&2
    return 2
  fi
}

# Validate the scheduler-provided logical worker rank.
validate_worker_index() {
  if [[ ! "$WORKER_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Worker rank must be a non-negative integer: $WORKER_INDEX" >&2
    return 1
  fi
}

# Substitute the deterministic logical rank placeholder in command arguments.
render_worker_arguments() {
  local argument

  WORKER_ARGUMENTS=()
  for argument in "${SCIENTIFIC_COMMAND[@]}"; do
    WORKER_ARGUMENTS+=("${argument//\{worker_id\}/$WORKER_INDEX}")
  done
}

# Publish one completion marker atomically after successful container execution.
publish_completion_marker() {
  local marker="$1"
  local temporary

  temporary="$marker.${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-step}.$WORKER_INDEX"
  printf '%s\n' \
    "worker_group=$WORKER_GROUP" \
    "worker_id=$WORKER_INDEX" \
    "job_id=${SLURM_JOB_ID:-}" \
    "step_id=${SLURM_STEP_ID:-}" \
    "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$temporary"
  mv "$temporary" "$marker"
}

# Execute one rank and publish its durable success receipt.
main() {
  local worker_id marker

  parse_arguments "$@"
  validate_arguments

  WORKER_INDEX="${SLURM_PROCID:?SLURM_PROCID is required for a parallel worker}"
  validate_worker_index
  printf -v worker_id '%03d' "$WORKER_INDEX"
  marker="$RUN_ROOT/.state/workers/$WORKER_GROUP/$worker_id.done"

  if [ -s "$marker" ]; then
    echo "Worker $WORKER_GROUP/$worker_id is already complete; skipping"
    return 0
  fi

  render_worker_arguments

  "$SOURCE_ROOT/scripts/workflow/mn5_container_exec.sh" \
    --run-root "$RUN_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --sif "$SIF" \
    --stage-data \
    --nv \
    --cpus "$CPUS" \
    -- env "EPIFORECASTER_WORKER_ID=$WORKER_INDEX" "${WORKER_ARGUMENTS[@]}"

  publish_completion_marker "$marker"
}

WORKER_INDEX=""
WORKER_ARGUMENTS=()
main "$@"
