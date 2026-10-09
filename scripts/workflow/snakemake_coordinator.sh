#!/bin/bash

# Run Snakemake as the lightweight control plane inside an existing MN5
# allocation. Scientific rules launch their own exact srun steps.

set -euo pipefail

RUN_ROOT="${1:?run root is required}"
SNAKEMAKE_BIN="${2:?Snakemake executable is required}"
CONTAINER_SIF="${3:?container SIF is required}"
EXPECTED_TASKS="${4:?expected task count is required}"
EXPECTED_CPUS_PER_TASK="${5:?expected CPUs per task is required}"
EXECUTION_MODE="${6:?execution mode is required}"
ATTEMPT_ROOT="${7:?execution attempt root is required}"
shift 7

write_completion() {
  local status=$?
  local receipt="$ATTEMPT_ROOT/completed.tsv"
  local temporary="$receipt.$$"

  trap - EXIT
  local cache_script="$RUN_ROOT/source/scripts/workflow/node_input_cache.sh"
  if [ -f "$cache_script" ] && command -v srun >/dev/null 2>&1; then
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exact \
      bash "$cache_script" clean "$RUN_ROOT" \
      || echo "Warning: allocation input cache cleanup failed" >&2
  fi
  if [ ! -f "$receipt" ]; then
    printf '%s\n' \
      $'schema_version\t1' \
      "exit_code"$'\t'"$status" \
      "completed_at"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >"$temporary"
    mv "$temporary" "$receipt"
  fi
  exit "$status"
}

validate_allocation() {
  test -d "$RUN_ROOT/source"
  test -r "$RUN_ROOT/config.yaml"
  test -f "$CONTAINER_SIF"
  test -f "${CONTAINER_SIF}.sha256"
  test -d "$ATTEMPT_ROOT"

  if [ "$EXECUTION_MODE" != "batch" ] \
    && [ "$EXECUTION_MODE" != "existing-allocation" ]; then
    echo "Unknown execution mode: $EXECUTION_MODE" >&2
    return 2
  fi

  if [ "$EXECUTION_MODE" = "batch" ]; then
    if [ "${SLURM_JOB_NUM_NODES:-}" != "1" ] \
      || [ "${SLURM_NTASKS:-}" != "$EXPECTED_TASKS" ] \
      || [ "${SLURM_CPUS_PER_TASK:-}" != "$EXPECTED_CPUS_PER_TASK" ]; then
      echo "Expected one MN5 node with $EXPECTED_TASKS tasks and $EXPECTED_CPUS_PER_TASK CPUs per task" >&2
      return 2
    fi
  else
    if [ "${SLURM_NNODES:-}" != "1" ]; then
      echo "External launcher does not expose the validated one-node allocation" >&2
      return 2
    fi
    # The special external-launcher step is one lightweight coordinator task;
    # it does not expose the parent allocation's scientific task count or CPU
    # shape. The snapshot-owned adapter already validated and sealed that
    # capacity through scontrol, so publish it to child workflow rules.
    export SLURM_JOB_NUM_NODES=1
    export SLURM_NTASKS="$EXPECTED_TASKS"
    export SLURM_CPUS_PER_TASK="$EXPECTED_CPUS_PER_TASK"
  fi
}

load_singularity() {
  local profile_script candidate

  if command -v singularity >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v module >/dev/null 2>&1; then
    set +u
    for profile_script in \
      /etc/profile.d/00-bsc.sh \
      /etc/profile.d/01-module.sh \
      /apps/modules/LMOD/8.7/lmod/8.7/init/profile; do
      if [ -r "$profile_script" ]; then
        source "$profile_script"
      fi
      if command -v module >/dev/null 2>&1; then
        break
      fi
    done
    set -u
  fi

  module load EB/apps >/dev/null 2>&1 || true
  for candidate in \
    "${SINGULARITY_MODULE:-singularity/4.1.5}" \
    "Singularity/4.1.5" \
    "SingularityCE/4.1.5"; do
    if module load "$candidate" >/dev/null 2>&1; then
      return 0
    fi
  done
  echo "Unable to load Singularity 4.1.5 module" >&2
  return 1
}

trap write_completion EXIT
validate_allocation

# Only one coordinator may mutate a run's Snakemake metadata at a time.
exec 9>"$RUN_ROOT/.state/coordinator.run.lock"
if ! flock -n 9; then
  echo "A Snakemake coordinator already owns this run" >&2
  exit 1
fi

allocation_job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
coordinator_step_id="${SLURM_STEP_ID:-batch}"
execution_id="${allocation_job_id}_${coordinator_step_id}"
printf '%s\n' "$allocation_job_id" >"$RUN_ROOT/.state/coordinator.job_id"
printf '%s\n' "$execution_id" >"$RUN_ROOT/.state/coordinator.execution_id"
printf '%s\n' "$coordinator_step_id" >"$ATTEMPT_ROOT/coordinator.step_id"
printf '%s\n' \
  $'schema_version\t1' \
  "allocation_job_id"$'\t'"$allocation_job_id" \
  "coordinator_step_id"$'\t'"$coordinator_step_id" \
  "execution_mode"$'\t'"$EXECUTION_MODE" \
  "started_at"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$ATTEMPT_ROOT/started.tsv"

export EPIFORECASTER_EXECUTION_MODE="$EXECUTION_MODE"
export EPIFORECASTER_EXECUTION_ATTEMPT_ID
EPIFORECASTER_EXECUTION_ATTEMPT_ID="$(basename "$ATTEMPT_ROOT")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUN_ROOT/source"
export XDG_CACHE_HOME="$RUN_ROOT/.cache"

load_singularity

# The coordinator sees only its own run root. Evaluation and resumed training
# read the declared source checkpoint once to pin it into that run. Bind its
# directory read-only so Singularity can create a directory mount point in the
# contained filesystem. Scientific steps later use the pinned in-run copy.
checkpoint_bind="$(
  singularity exec \
    --cleanenv \
    --containall \
    --bind "$RUN_ROOT:$RUN_ROOT:ro" \
    --pwd "$RUN_ROOT" \
    "$CONTAINER_SIF" \
    /usr/local/bin/python -c \
      'import sys, yaml; config = yaml.safe_load(open(sys.argv[1])); print(config.get("evaluation", {}).get("checkpoint", "") if config.get("workflow", {}).get("kind") == "evaluate" else (config.get("training", {}).get("resume_checkpoint") or config.get("training", {}).get("init_checkpoint") or "") if config.get("workflow", {}).get("kind") == "train" else "")' \
      "$RUN_ROOT/config.yaml"
)"
checkpoint_bind_options=()
if [ -n "$checkpoint_bind" ] && [[ "$checkpoint_bind" != "$RUN_ROOT/"* ]]; then
  if [[ "$checkpoint_bind" != /gpfs/* ]] || [ ! -f "$checkpoint_bind" ]; then
    echo "Source checkpoint is not a readable GPFS file: $checkpoint_bind" >&2
    exit 1
  fi
  checkpoint_directory="$(dirname "$checkpoint_bind")"
  checkpoint_bind_options=(--bind "$checkpoint_directory:$checkpoint_directory:ro")
fi

# A checkpoint evaluation can also retain the completed producer's canonical
# records. Read them once during preparation, then use only the pinned copies.
training_evidence_bind="$(
  singularity exec --cleanenv --containall \
    --bind "$RUN_ROOT:$RUN_ROOT:ro" --pwd "$RUN_ROOT" \
    "$CONTAINER_SIF" /usr/local/bin/python -c \
    'import sys, yaml; config = yaml.safe_load(open(sys.argv[1])); print(config.get("evaluation", {}).get("training_model_root", "") if config.get("workflow", {}).get("kind") == "evaluate" else "")' \
    "$RUN_ROOT/config.yaml"
)"
if [ -n "$training_evidence_bind" ]; then
  if [[ "$training_evidence_bind" != /gpfs/* ]] || [ ! -d "$training_evidence_bind" ]; then
    echo "Producer evidence is not a readable GPFS directory: $training_evidence_bind" >&2
    exit 1
  fi
  # A producer directory can contain the checkpoint. Do not bind a nested file
  # or directory before this parent mount exists in the contained filesystem.
  if [[ "$checkpoint_bind" == "$training_evidence_bind/"* ]]; then
    checkpoint_bind_options=()
  fi
  checkpoint_bind_options+=(--bind "$training_evidence_bind:$training_evidence_bind:ro")
fi

# Snakemake is pinned inside the SIF. Its shell rules still launch host Slurm
# steps, so expose only the site client, configuration, and auth socket.
singularity exec \
  --cleanenv \
  --containall \
  --bind "$RUN_ROOT:$RUN_ROOT:rw" \
  ${checkpoint_bind_options[@]+"${checkpoint_bind_options[@]}"} \
  --bind "${CONTAINER_SIF}.sha256:${CONTAINER_SIF}.sha256:ro" \
  --bind /usr/bin/srun:/usr/bin/srun:ro \
  --bind /usr/lib64/slurm:/usr/lib64/slurm:ro \
  --bind /lib64/libmunge.so.2:/lib64/libmunge.so.2:ro \
  --bind /etc/slurm:/etc/slurm:ro \
  --bind /etc/passwd:/etc/passwd:ro \
  --bind /etc/group:/etc/group:ro \
  --bind /opt/EAR:/opt/EAR:ro \
  --bind /run/munge:/run/munge:rw \
  --env "PATH=/opt/epiforecaster/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "LD_LIBRARY_PATH=/usr/lib64/slurm:/lib64" \
  --env "PYTHONPATH=$RUN_ROOT/source" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "XDG_CACHE_HOME=$RUN_ROOT/.cache" \
  --env "EPIFORECASTER_EXECUTION_MODE=$EXECUTION_MODE" \
  --env "EPIFORECASTER_EXECUTION_ATTEMPT_ID=$EPIFORECASTER_EXECUTION_ATTEMPT_ID" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env "SLURM_JOB_NUM_NODES=1" \
  --env "SLURM_NTASKS=$EXPECTED_TASKS" \
  --env "SLURM_CPUS_PER_TASK=$EXPECTED_CPUS_PER_TASK" \
  --pwd "$RUN_ROOT" \
  "$CONTAINER_SIF" \
  "$SNAKEMAKE_BIN" \
  --snakefile "$RUN_ROOT/source/workflow/Snakefile" \
  --configfile "$RUN_ROOT/config.yaml" \
  --workflow-profile "$RUN_ROOT/source/workflow/profiles/mn5" \
  --directory "$RUN_ROOT" \
  --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
  "$@"
