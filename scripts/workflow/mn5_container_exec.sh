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

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-root PATH --source-root PATH --sif PATH \\
  [--dvc-mode pull|push --dvc-remote PATH --dvc-cache PATH] \\
  [--promotion-root PATH] [--input-path PATH]... \
  [--stage-data] [--nv] [--cpus N] -- COMMAND [ARG...]
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
case "$DVC_MODE" in
  none)
    ;;
  pull)
    if [ -z "$DVC_REMOTE" ] || [ -z "$DVC_CACHE" ]; then
      echo "DVC pull mode requires --dvc-remote and --dvc-cache" >&2
      exit 2
    fi
    ;;
  push)
    if [ -z "$DVC_REMOTE" ] || [ -z "$DVC_CACHE" ] || [ -z "$PROMOTION_ROOT" ]; then
      echo "DVC push mode requires remote, cache, and promotion paths" >&2
      exit 2
    fi
    ;;
  *)
    echo "Invalid DVC mode: $DVC_MODE" >&2
    exit 2
    ;;
esac
if [ -n "$REQUESTED_CPUS" ] && [[ ! "$REQUESTED_CPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--cpus must be a positive integer: $REQUESTED_CPUS" >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1; then
  set +u
  for profile_script in /etc/profile.d/00-bsc.sh /etc/profile.d/01-module.sh /apps/modules/LMOD/8.7/lmod/8.7/init/profile; do
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

if ! command -v singularity >/dev/null 2>&1; then
  module load EB/apps >/dev/null 2>&1 || true
  loaded=0
  for candidate in "${SINGULARITY_MODULE:-singularity/4.1.5}" "Singularity/4.1.5" "SingularityCE/4.1.5"; do
    if module load "$candidate" >/dev/null 2>&1; then
      loaded=1
      break
    fi
  done
  if [ "$loaded" -ne 1 ]; then
    echo "Unable to load Singularity 4.1.5 module" >&2
    exit 1
  fi
fi

test -f "$SIF"
test -d "$SOURCE_ROOT"
test -d "$RUN_ROOT"

run_id="$(basename "$RUN_ROOT")"
slurm_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
slurm_task_id="${SLURM_ARRAY_TASK_ID:-0}"
slurm_step_id="${SLURM_STEP_ID:-batch}"
slurm_proc_id="${SLURM_PROCID:-0}"
execution_id="${slurm_job_id}_${slurm_task_id}_${slurm_step_id}_${slurm_proc_id}"
scratch_base="${TMPDIR:-/scratch/${USER:-unknown}}"
scratch_root="$scratch_base/epiforecaster/$run_id/$execution_id"
mkdir -p "$(dirname "$scratch_root")"
if ! mkdir "$scratch_root"; then
  echo "Scratch execution directory already exists: $scratch_root" >&2
  exit 1
fi
mkdir "$scratch_root/tmp"
echo "Workflow execution identity: $execution_id"

cleanup() {
  local status=$?
  trap - EXIT
  if ! chmod -R u+w "$scratch_root" 2>/dev/null; then
    echo "Warning: could not make scratch contents writable: $scratch_root" >&2
  fi
  if ! rm -rf "$scratch_root"; then
    echo "Warning: could not remove scratch execution directory: $scratch_root" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

binds=(
  "$SOURCE_ROOT:/workspace/source:ro"
  "$RUN_ROOT:/workspace/run:rw"
  "$scratch_root/tmp:/tmp:rw"
)
for input_path in "${INPUT_PATHS[@]}"; do
  if [ ! -e "$input_path" ] || [ -L "$input_path" ]; then
    echo "Workflow input must exist and must not be a symlink: $input_path" >&2
    exit 1
  fi
  binds+=("$input_path:$input_path:ro")
done

case "$DVC_MODE" in
  pull)
    test -d "$DVC_REMOTE"
    test -d "$DVC_CACHE"
    binds+=("$DVC_REMOTE:$DVC_REMOTE:ro" "$DVC_CACHE:$DVC_CACHE:rw")
    ;;
  push)
    test -d "$DVC_REMOTE"
    test -d "$DVC_CACHE"
    mkdir -p "$PROMOTION_ROOT"
    binds+=(
      "$DVC_REMOTE:$DVC_REMOTE:rw"
      "$DVC_CACHE:$DVC_CACHE:rw"
      "$PROMOTION_ROOT:$PROMOTION_ROOT:rw"
    )
    ;;
esac

if [ "$STAGE_DATA" -eq 1 ]; then
  bind_roots_file="$RUN_ROOT/.state/dvc_bind_roots.txt"
  dvc_workspace="$RUN_ROOT/.state/dvc-workspace"
  test -f "$bind_roots_file"
  test -d "$dvc_workspace"
  while IFS= read -r bind_root; do
    [ -n "$bind_root" ] || continue
    if [ ! -d "$SOURCE_ROOT/$bind_root" ]; then
      echo "DVC bind root is absent from source snapshot: $bind_root" >&2
      exit 1
    fi
    staged="$scratch_root/staged/$bind_root"
    mkdir -p "$staged"
    rsync -a "$SOURCE_ROOT/$bind_root/" "$staged/"
    rsync -aL "$dvc_workspace/$bind_root/" "$staged/"
    binds+=("$staged:/workspace/source/$bind_root:ro")
  done < "$bind_roots_file"
  echo "Staged DVC inputs at $scratch_root/staged"
fi

singularity_args=(exec --cleanenv --containall)
if [ "$USE_NV" -eq 1 ]; then
  singularity_args+=(--nv)
fi
for bind in "${binds[@]}"; do
  singularity_args+=(--bind "$bind")
done

container_path="/opt/epiforecaster/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
allocated_cpus="${REQUESTED_CPUS:-${SLURM_CPUS_PER_TASK:-1}}"
singularity_args+=(
  --env "PATH=$container_path"
  --env "PYTHONPATH=/workspace/source"
  --env "HOME=/tmp"
  --env "XDG_CACHE_HOME=/tmp/.cache"
  --env "MPLCONFIGDIR=/tmp/matplotlib"
  --env "TMPDIR=/tmp"
  --env "EPFORECASTER_STAGE_TO_NVME=0"
  --env "OMP_NUM_THREADS=$allocated_cpus"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  --env "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
  --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
  --env "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
  --env "SLURM_PROCID=${SLURM_PROCID:-}"
)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  singularity_args+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
singularity_args+=(--pwd /workspace/source "$SIF")

singularity "${singularity_args[@]}" "$@"
