#!/bin/bash

set -euo pipefail

RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
CPUS=""
WORKER_GROUP=""
WORKER_OFFSET="0"
MPS_CLIENTS_PER_GPU="1"
MPS_ROOT=""
MPS_PIPE=""
MPS_GPU_INDEX=""
MPS_LEADER=0
SCIENTIFIC_COMMAND=()
CONTINUATION_EXIT_CODE=75

# Print the per-rank worker interface.
usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \
  --cpus N --worker-group NAME --worker-offset N \
  [--mps-clients-per-gpu K] -- COMMAND [ARG...]
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
      --worker-offset)
        WORKER_OFFSET="$2"
        shift 2
        ;;
      --mps-clients-per-gpu)
        MPS_CLIENTS_PER_GPU="$2"
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
  if [[ ! "$WORKER_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "Worker offset must be a non-negative integer: $WORKER_OFFSET" >&2
    return 2
  fi
  if [[ ! "$MPS_CLIENTS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "MPS clients per GPU must be a positive integer: $MPS_CLIENTS_PER_GPU" >&2
    return 2
  fi
  if [ "$MPS_CLIENTS_PER_GPU" -gt 1 ] && [ "$WORKER_OFFSET" != "0" ]; then
    echo "MPS packing requires every worker rank in one wave" >&2
    return 2
  fi
}

# Start (rank-group leader) or join the CUDA MPS daemon of this rank's GPU.
# Rank r uses GPU r / K, selected by UUID. The daemon directories live under the
# job-local TMPDIR, never under /tmp. Each GPU has its own daemon, started by
# the first rank on that GPU and stopped after all K ranks on it have exited.
mps_setup() {
  local uuid listed waited=0

  MPS_GPU_INDEX=$((WORKER_INDEX / MPS_CLIENTS_PER_GPU))
  if [ -z "${TMPDIR:-}" ] || [ -z "${SLURM_STEP_ID:-}" ]; then
    echo "MPS packing requires the job TMPDIR and a Slurm step" >&2
    return 1
  fi
  listed="$(nvidia-smi --query-gpu=uuid --format=csv,noheader </dev/null)"
  uuid="$(printf '%s\n' "$listed" | sed -n "$((MPS_GPU_INDEX + 1))p" | tr -d ' ')"
  if [[ ! "$uuid" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
    echo "No GPU UUID for MPS GPU index $MPS_GPU_INDEX in: $listed" >&2
    return 1
  fi
  MPS_ROOT="$TMPDIR/mps/step${SLURM_STEP_ID}-gpu${MPS_GPU_INDEX}"
  MPS_PIPE="$MPS_ROOT/pipe"
  if [ $((WORKER_INDEX % MPS_CLIENTS_PER_GPU)) -eq 0 ]; then
    MPS_LEADER=1
    mkdir -p "$MPS_ROOT/log" "$MPS_ROOT/done"
    mkdir -p "$MPS_PIPE"
    env "CUDA_VISIBLE_DEVICES=$uuid" \
      "CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE" \
      "CUDA_MPS_LOG_DIRECTORY=$MPS_ROOT/log" \
      nvidia-cuda-mps-control -d
  fi
  while [ ! -e "$MPS_PIPE/control" ]; do
    if [ "$waited" -ge 120 ]; then
      echo "MPS daemon for GPU $MPS_GPU_INDEX did not start: $MPS_PIPE" >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  export "CUDA_VISIBLE_DEVICES=$uuid"
  echo "MPS: rank $WORKER_INDEX uses GPU $MPS_GPU_INDEX ($uuid) via $MPS_PIPE"
}

# Record this rank as finished; the leader waits for its peers, then stops the
# daemon and keeps the daemon logs next to the other run logs.
mps_finish() {
  local rank_status="$1"
  local done_dir="$MPS_ROOT/done" log_target

  mkdir -p "$done_dir" 2>/dev/null || true
  : >"$done_dir/$WORKER_INDEX"
  if [ "$MPS_LEADER" -ne 1 ]; then
    return 0
  fi
  # A failed or signalled rank means srun is already stopping its peers.
  while [ "$rank_status" -eq 0 ] \
    && [ "$(find "$done_dir" -type f | wc -l)" -lt "$MPS_CLIENTS_PER_GPU" ]; do
    sleep 1
  done
  echo quit | env "CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE" nvidia-cuda-mps-control \
    >/dev/null 2>&1 || true
  log_target="$RUN_ROOT/logs/mps/$WORKER_GROUP-step${SLURM_STEP_ID}-gpu${MPS_GPU_INDEX}"
  mkdir -p "$log_target" 2>/dev/null || true
  cp -r "$MPS_ROOT/log/." "$log_target/" 2>/dev/null || true
  MPS_LEADER=0
}

# Preserve the rank status while MPS cleanup runs on every exit path.
cleanup() {
  local status=$?

  trap - EXIT
  if [ -n "$MPS_ROOT" ]; then
    mps_finish "$status" || true
  fi
  exit "$status"
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

# Publish one atomic receipt after container execution reaches a terminal worker state.
publish_worker_receipt() {
  local marker="$1"
  local outcome="$2"
  local temporary

  temporary="$marker.${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-step}.$WORKER_INDEX"
  printf '%s\n' \
    "worker_group=$WORKER_GROUP" \
    "worker_id=$WORKER_INDEX" \
    "job_id=${SLURM_JOB_ID:-}" \
    "step_id=${SLURM_STEP_ID:-}" \
    "outcome=$outcome" \
    "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$temporary"
  mv "$temporary" "$marker"
}

# Publish one completion marker atomically after successful container execution.
publish_completion_marker() {
  local marker="$1"

  publish_worker_receipt "$marker" complete
}

# Publish one continuation receipt without claiming that the worker completed.
publish_continuation_marker() {
  local marker="$1"

  publish_worker_receipt "$marker" continuation
}

# Execute one rank and publish its durable terminal receipt.
main() {
  local worker_id marker continuation_marker status

  parse_arguments "$@"
  validate_arguments

  WORKER_INDEX="${SLURM_PROCID:?SLURM_PROCID is required for a parallel worker}"
  WORKER_INDEX=$((WORKER_INDEX + WORKER_OFFSET))
  validate_worker_index
  if [ "$MPS_CLIENTS_PER_GPU" -gt 1 ]; then
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    mps_setup
  fi
  printf -v worker_id '%03d' "$WORKER_INDEX"
  marker="$RUN_ROOT/.state/workers/$WORKER_GROUP/$worker_id.done"
  continuation_marker="$RUN_ROOT/.state/workers/$WORKER_GROUP/$worker_id.partial"

  if [ -s "$marker" ]; then
    echo "Worker $WORKER_GROUP/$worker_id is already complete; skipping"
    return 0
  fi

  render_worker_arguments
  MPS_ARGUMENTS=()
  if [ "$MPS_CLIENTS_PER_GPU" -gt 1 ]; then
    MPS_ARGUMENTS=(--mps-pipe-dir "$MPS_PIPE")
  fi

  set +e
  "$SOURCE_ROOT/scripts/workflow/mn5_container_exec.sh" \
    --run-root "$RUN_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --sif "$SIF" \
    --stage-data \
    --nv \
    --cpus "$CPUS" \
    ${MPS_ARGUMENTS[@]+"${MPS_ARGUMENTS[@]}"} \
    -- env "EPIFORECASTER_WORKER_ID=$WORKER_INDEX" "${WORKER_ARGUMENTS[@]}"
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    publish_completion_marker "$marker"
    return 0
  fi
  if [ "$status" -eq "$CONTINUATION_EXIT_CODE" ]; then
    publish_continuation_marker "$continuation_marker"
    return 0
  fi
  return "$status"
}

WORKER_INDEX=""
WORKER_ARGUMENTS=()
MPS_ARGUMENTS=()
# The close-out signal is for the scientific process only; see
# mn5_container_exec.sh.
trap '' USR1
main "$@"
