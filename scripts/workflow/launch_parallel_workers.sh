#!/bin/bash

set -euo pipefail

# Independent ranks run in deterministic waves over the allocated GPUs.
RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
WORKERS=""
CPUS_PER_TASK=""
WORKER_GROUP=""

# Print the worker-step interface.
usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \
  --workers N --cpus-per-task 20 --worker-group NAME -- COMMAND [ARG...]
EOF
}

# Parse launcher options and leave the scientific command in "$@".
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
      --workers)
        WORKERS="$2"
        shift 2
        ;;
      --cpus-per-task)
        CPUS_PER_TASK="$2"
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

  SCIENTIFIC_COMMAND=()
}

# Reject malformed worker requests before invoking srun.
validate_arguments() {
  if [ -z "$RUN_ROOT" ] \
    || [ -z "$SOURCE_ROOT" ] \
    || [ -z "$SIF" ] \
    || [ -z "$WORKER_GROUP" ] \
    || [ "${#SCIENTIFIC_COMMAND[@]}" -eq 0 ]; then
    usage >&2
    return 2
  fi

  if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || [ "$CPUS_PER_TASK" != "20" ]; then
    echo "MN5 parallel workflows require positive workers with 20 CPUs each" >&2
    return 2
  fi

  if [[ ! "$WORKER_GROUP" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "Invalid worker group: $WORKER_GROUP" >&2
    return 2
  fi
}

# Confirm that the containing allocation can host one 1-4 GPU worker wave.
validate_allocation() {
  if [ "${SLURM_JOB_NUM_NODES:-}" != "1" ]; then
    echo "Expected one allocated ACC node, got ${SLURM_JOB_NUM_NODES:-unset}" >&2
    return 1
  fi

  if ! [[ "${SLURM_NTASKS:-}" =~ ^[1-4]$ ]]; then
    echo "Expected SLURM_NTASKS in [1,4], got ${SLURM_NTASKS:-unset}" >&2
    return 1
  fi

  if [ "${SLURM_CPUS_PER_TASK:-}" != "$CPUS_PER_TASK" ]; then
    echo "Expected SLURM_CPUS_PER_TASK=$CPUS_PER_TASK, got ${SLURM_CPUS_PER_TASK:-unset}" >&2
    return 1
  fi
}

# Require every rank to publish a durable completion or continuation receipt.
verify_worker_receipts() {
  local marker_root="$1"
  local worker_index worker_id complete_marker continuation_marker

  CONTINUATION_REQUIRED=0
  for ((worker_index = 0; worker_index < WORKERS; worker_index++)); do
    printf -v worker_id '%03d' "$worker_index"
    complete_marker="$marker_root/$worker_id.done"
    continuation_marker="$marker_root/$worker_id.partial"

    if [ -s "$complete_marker" ]; then
      continue
    fi
    if [ -s "$continuation_marker" ]; then
      CONTINUATION_REQUIRED=1
      continue
    fi
    echo "Worker $WORKER_GROUP/$worker_id exited without a receipt" >&2
    return 1
  done
}

# Worker receipts describe rank exit, while the journal describes campaign work.
check_ablation_completion() {
  local status

  if [ "${SCIENTIFIC_COMMAND[0]}" != "python" ] \
    || [ "${SCIENTIFIC_COMMAND[1]:-}" != "-m" ] \
    || [ "${SCIENTIFIC_COMMAND[2]:-}" != "scripts.hpo.ablation_worker" ]; then
    return 0
  fi

  set +e
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --gres=none \
    --exact \
    --kill-on-bad-exit=1 \
    "$SOURCE_ROOT/scripts/workflow/mn5_container_exec.sh" \
    --run-root "$RUN_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --sif "$SIF" \
    --cpus 1 \
    -- "${SCIENTIFIC_COMMAND[@]}" --check-complete
  status=$?
  set -e

  case "$status" in
    0) CONTINUATION_REQUIRED=0 ;;
    75) CONTINUATION_REQUIRED=1 ;;
    *) return "$status" ;;
  esac
}

# Receipts describe one allocation. The ablation queue is the source of truth
# for remaining work, so stale rank receipts must not skip a rank.
reset_worker_receipts() {
  local marker_root="$1"

  rm -f "$marker_root"/*.partial
  if [ "${SCIENTIFIC_COMMAND[0]}" = "python" ] \
    && [ "${SCIENTIFIC_COMMAND[1]:-}" = "-m" ] \
    && [ "${SCIENTIFIC_COMMAND[2]:-}" = "scripts.hpo.ablation_worker" ]; then
    rm -f "$marker_root"/*.done
  fi
}

# Launch one gang-scheduled worker step and validate all rank receipts.
main() {
  local worker_script marker_root worker_offset remaining wave_workers

  parse_arguments "$@"
  validate_arguments
  validate_allocation

  worker_script="$SOURCE_ROOT/scripts/workflow/run_parallel_worker.sh"
  marker_root="$RUN_ROOT/.state/workers/$WORKER_GROUP"
  test -x "$worker_script"
  mkdir -p "$RUN_ROOT/logs" "$marker_root"
  # A step failure must not leave the marker of an earlier allocation behind.
  rm -f "$RUN_ROOT/.state/continuation.required"
  reset_worker_receipts "$marker_root"

  worker_offset=0
  while [ "$worker_offset" -lt "$WORKERS" ]; do
    remaining=$((WORKERS - worker_offset))
    wave_workers="${SLURM_NTASKS}"
    if [ "$remaining" -lt "$wave_workers" ]; then
      wave_workers="$remaining"
    fi

    srun \
      --nodes=1 \
      --ntasks="$wave_workers" \
      --ntasks-per-node="$wave_workers" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --gpus-per-task=1 \
      --exact \
      --kill-on-bad-exit=1 \
      --open-mode=append \
      --output="$RUN_ROOT/logs/$WORKER_GROUP-wave-$worker_offset-worker-%t.log" \
      --error="$RUN_ROOT/logs/$WORKER_GROUP-wave-$worker_offset-worker-%t.log" \
      "$worker_script" \
      --run-root "$RUN_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --sif "$SIF" \
      --cpus "$CPUS_PER_TASK" \
      --worker-group "$WORKER_GROUP" \
      --worker-offset "$worker_offset" \
      -- "${SCIENTIFIC_COMMAND[@]}"

    worker_offset=$((worker_offset + wave_workers))
  done

  verify_worker_receipts "$marker_root"
  check_ablation_completion
  if [ "$CONTINUATION_REQUIRED" -eq 1 ]; then
    printf '%s\n' \
      $'schema_version\t1' \
      "attempt_id"$'\t'"${EPIFORECASTER_EXECUTION_ATTEMPT_ID:-}" \
      "continued_at"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >"$RUN_ROOT/.state/continuation.required.$$"
    mv "$RUN_ROOT/.state/continuation.required.$$" \
      "$RUN_ROOT/.state/continuation.required"
    echo "Ablation worker close-out completed; continuation is required" >&2
    return 75
  fi
}

CONTINUATION_REQUIRED=0
main "$@"
