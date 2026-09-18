#!/bin/bash

# Execute one workflow command in the shared immutable SIF. When requested,
# materialize the run's DVC inputs on this compute node's local /scratch and
# bind those staged directories over the corresponding read-only source paths.

set -euo pipefail

RUN_ROOT=""
SOURCE_ROOT=""
SIF=""
DVC_MODE="none"
DVC_REMOTE=""
DVC_CACHE=""
PROMOTION_ROOT=""
STAGE_DATA=0
USE_NV=0
REQUESTED_CPUS=""
INPUT_PATHS=()
SCIENTIFIC_COMMAND=()
BINDS=()
SINGULARITY_ARGS=()
SCRATCH_ROOT=""

# Print the container execution interface.
usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \\
  [--dvc-mode pull|push --dvc-remote PATH --dvc-cache PATH] \\
  [--promotion-root PATH] [--input-path PATH]... \
  [--stage-data] [--nv] [--cpus N] -- COMMAND [ARG...]
EOF
}

# Parse wrapper options and retain the scientific command as an array.
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
      --dvc-mode)
        DVC_MODE="$2"
        shift 2
        ;;
      --dvc-remote)
        DVC_REMOTE="$2"
        shift 2
        ;;
      --dvc-cache)
        DVC_CACHE="$2"
        shift 2
        ;;
      --promotion-root)
        PROMOTION_ROOT="$2"
        shift 2
        ;;
      --input-path)
        INPUT_PATHS+=("$2")
        shift 2
        ;;
      --stage-data)
        STAGE_DATA=1
        shift
        ;;
      --nv)
        USE_NV=1
        shift
        ;;
      --cpus)
        REQUESTED_CPUS="$2"
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

# Validate the immutable input paths and mode-specific DVC requirements.
validate_arguments() {
  if [ -z "$RUN_ROOT" ] \
    || [ -z "$SOURCE_ROOT" ] \
    || [ -z "$SIF" ] \
    || [ "${#SCIENTIFIC_COMMAND[@]}" -eq 0 ]; then
    usage >&2
    return 2
  fi

  case "$DVC_MODE" in
    none)
      ;;
    pull)
      if [ -z "$DVC_REMOTE" ] || [ -z "$DVC_CACHE" ]; then
        echo "DVC pull mode requires --dvc-remote and --dvc-cache" >&2
        return 2
      fi
      ;;
    push)
      if [ -z "$DVC_REMOTE" ] || [ -z "$DVC_CACHE" ]; then
        echo "DVC push mode requires remote and cache paths" >&2
        return 2
      fi
      ;;
    *)
      echo "Invalid DVC mode: $DVC_MODE" >&2
      return 2
      ;;
  esac

  if [ -n "$REQUESTED_CPUS" ] \
    && [[ ! "$REQUESTED_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--cpus must be a positive integer: $REQUESTED_CPUS" >&2
    return 2
  fi

  test -f "$SIF"
  test -d "$SOURCE_ROOT"
  test -d "$RUN_ROOT"
}

# Load the site module only when Singularity is absent from the current PATH.
load_singularity() {
  local profile_script candidate loaded=0

  if ! command -v module >/dev/null 2>&1; then
    set +u
    for profile_script in \
      /etc/profile.d/00-bsc.sh \
      /etc/profile.d/01-module.sh \
      /apps/modules/LMOD/8.7/lmod/8.7/init/profile; do
      if [ -r "$profile_script" ]; then
        # shellcheck disable=SC1090
        source "$profile_script"
      fi
      if command -v module >/dev/null 2>&1; then
        break
      fi
    done
    set -u
  fi

  if command -v singularity >/dev/null 2>&1; then
    return 0
  fi

  module load EB/apps >/dev/null 2>&1 || true
  for candidate in \
    "${SINGULARITY_MODULE:-singularity/4.1.5}" \
    "Singularity/4.1.5" \
    "SingularityCE/4.1.5"; do
    if module load "$candidate" >/dev/null 2>&1; then
      loaded=1
      break
    fi
  done

  if [ "$loaded" -ne 1 ]; then
    echo "Unable to load Singularity 4.1.5 module" >&2
    return 1
  fi
}

# Reserve a rank-specific scratch directory that cannot collide on retries.
create_scratch() {
  local run_id slurm_job_id slurm_step_id slurm_proc_id execution_id scratch_base

  run_id="$(basename "$RUN_ROOT")"
  slurm_job_id="${SLURM_JOB_ID:-local}"
  slurm_step_id="${SLURM_STEP_ID:-batch}"
  slurm_proc_id="${SLURM_PROCID:-0}"
  execution_id="${slurm_job_id}_${slurm_step_id}_${slurm_proc_id}"
  scratch_base="${TMPDIR:-/scratch/${USER:-unknown}}"
  SCRATCH_ROOT="$scratch_base/epiforecaster/$run_id/$execution_id"

  mkdir -p "$(dirname "$SCRATCH_ROOT")"
  if ! mkdir "$SCRATCH_ROOT"; then
    echo "Scratch execution directory already exists: $SCRATCH_ROOT" >&2
    return 1
  fi
  mkdir "$SCRATCH_ROOT/tmp"
  echo "Workflow execution identity: $execution_id"
}

# Preserve the scientific status while making cleanup failures diagnostic-only.
cleanup() {
  local status=$?

  trap - EXIT
  if ! chmod -R u+w "$SCRATCH_ROOT" 2>/dev/null; then
    echo "Warning: could not make scratch contents writable: $SCRATCH_ROOT" >&2
  fi
  if ! rm -rf "$SCRATCH_ROOT"; then
    echo "Warning: could not remove scratch execution directory: $SCRATCH_ROOT" >&2
  fi
  exit "$status"
}

# Add explicit inputs and DVC storage with the minimum required permissions.
configure_shared_binds() {
  local input_path

  BINDS=(
    "$SOURCE_ROOT:/workspace/source:ro"
    "$RUN_ROOT:/workspace/run:rw"
    "$SCRATCH_ROOT/tmp:/tmp:rw"
  )

  # Publication receipts read the immutable SIF digest from inside the
  # contained runtime. The SIF itself is an execution input, not a bind, so
  # expose its immutable directory. File binds cannot create missing parent
  # directories on all MN5 container mounts.
  if [ -f "${SIF}.sha256" ]; then
    BINDS+=("$(dirname "$SIF"):$(dirname "$SIF"):ro")
  fi

  if [ "${#INPUT_PATHS[@]}" -gt 0 ]; then
    for input_path in "${INPUT_PATHS[@]}"; do
      if [ ! -e "$input_path" ] || [ -L "$input_path" ]; then
        echo "Workflow input must exist and must not be a symlink: $input_path" >&2
        return 1
      fi
      BINDS+=("$input_path:$input_path:ro")
    done
  fi

  case "$DVC_MODE" in
    pull)
      test -d "$DVC_REMOTE"
      test -d "$DVC_CACHE"
      BINDS+=("$DVC_REMOTE:$DVC_REMOTE:ro" "$DVC_CACHE:$DVC_CACHE:rw")
      ;;
    push)
      test -d "$DVC_REMOTE"
      test -d "$DVC_CACHE"
      BINDS+=("$DVC_REMOTE:$DVC_REMOTE:rw" "$DVC_CACHE:$DVC_CACHE:rw")
      if [ -n "$PROMOTION_ROOT" ]; then
        mkdir -p "$PROMOTION_ROOT"
        BINDS+=("$PROMOTION_ROOT:$PROMOTION_ROOT:rw")
      fi
      ;;
  esac
}

# Use one read-only copy per allocation, shared across ranks and stages.
stage_dvc_inputs() {
  if [ "$STAGE_DATA" -ne 1 ]; then
    return 0
  fi
  test -f "$RUN_ROOT/.state/dvc_bind_roots.txt"
  test -d "$RUN_ROOT/.state/dvc-workspace"
  # shellcheck source=scripts/workflow/node_input_cache.sh
  source "$(dirname "${BASH_SOURCE[0]}")/node_input_cache.sh"
  stage_shared_input_cache
}

# Construct the container invocation as an argument array without shell eval.
configure_singularity_arguments() {
  local bind
  local container_path allocated_cpus

  SINGULARITY_ARGS=(exec --cleanenv --containall)
  if [ "$USE_NV" -eq 1 ]; then
    SINGULARITY_ARGS+=(--nv)
  fi
  for bind in "${BINDS[@]}"; do
    SINGULARITY_ARGS+=(--bind "$bind")
  done

  container_path="/opt/epiforecaster/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  allocated_cpus="${REQUESTED_CPUS:-${SLURM_CPUS_PER_TASK:-1}}"
  SINGULARITY_ARGS+=(
    --env "PATH=$container_path"
    --env "PYTHONPATH=/workspace/source"
    --env "PYTHONFAULTHANDLER=1"
    --env "PYTHONUNBUFFERED=1"
    --env "XDG_CACHE_HOME=/tmp/.cache"
    --env "MPLCONFIGDIR=/tmp/matplotlib"
    --env "TMPDIR=/tmp"
    --env "EPFORECASTER_STAGE_TO_NVME=0"
    --env "OMP_NUM_THREADS=$allocated_cpus"
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    --env "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
    --env "SLURM_PROCID=${SLURM_PROCID:-}"
  )
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    SINGULARITY_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi
  # Execute the verified immutable SIF directly from shared GPFS. Node-local
  # image staging is deferred until its end-to-end cost is benchmarked.
  SINGULARITY_ARGS+=(--pwd /workspace/source "$SIF")
}

# Execute the requested scientific command and let the EXIT trap preserve status.
main() {
  parse_arguments "$@"
  validate_arguments
  if [ "$STAGE_DATA" -eq 1 ] && [ "${EPIFORECASTER_INPUT_CACHE_LEASED:-}" != "$RUN_ROOT" ]; then
    # Hold the lease in a parent process for the whole scientific step.
    # shellcheck source=scripts/workflow/node_input_cache.sh
    source "$(dirname "${BASH_SOURCE[0]}")/node_input_cache.sh"
    local cache_root
    cache_root="$(node_cache_run_root "$RUN_ROOT")"
    mkdir -p "$cache_root"
    flock -s "$cache_root/.usage.lock" \
      env "EPIFORECASTER_INPUT_CACHE_LEASED=$RUN_ROOT" \
      bash "${BASH_SOURCE[0]}" "$@"
    return $?
  fi
  load_singularity
  create_scratch
  trap cleanup EXIT

  configure_shared_binds
  stage_dvc_inputs
  configure_singularity_arguments
  if singularity "${SINGULARITY_ARGS[@]}" "${SCIENTIFIC_COMMAND[@]}"; then
    return 0
  else
    local scientific_status=$?
    return "$scientific_status"
  fi
}

main "$@"
