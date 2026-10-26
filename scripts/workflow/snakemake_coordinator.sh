#!/bin/bash

# Run Snakemake as the lightweight control plane inside an existing MN5
# allocation. Scientific rules launch their own exact srun steps.

set -euo pipefail

RUN_ROOT="${1:?run root is required}"
CONTAINER_SIF="${2:?container SIF is required}"
EXPECTED_TASKS="${3:?expected task count is required}"
EXPECTED_CPUS_PER_TASK="${4:?expected CPUs per task is required}"
EXECUTION_MODE="${5:?execution mode is required}"
ATTEMPT_ROOT="${6:?execution attempt root is required}"
shift 6

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
  if [ ! -f "$CONTAINER_SIF" ]; then
    echo "Missing coordinator SIF: $CONTAINER_SIF" >&2
    return 2
  fi
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

trap write_completion EXIT
validate_allocation

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
    "SingularityCE/4.1.5" \
    "singularity/4.1.0"; do
    if module load "$candidate" >/dev/null 2>&1; then
      loaded=1
      break
    fi
  done

  if [ "$loaded" -ne 1 ]; then
    echo "Unable to load a Singularity 4.1 module (4.1.5 on MN5, 4.1.0 on Nord4)" >&2
    return 1
  fi
}

load_singularity

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

container_args=(
  exec
  --cleanenv
  --containall
  --bind "$RUN_ROOT:$RUN_ROOT:rw"
  --bind "$(dirname "$CONTAINER_SIF"):$(dirname "$CONTAINER_SIF"):ro"
  --bind /usr/bin/srun:/usr/bin/srun:ro
  --bind /usr/lib64/slurm:/usr/lib64/slurm:ro
  --bind /usr/lib64/libmunge.so.2.0.1:/usr/lib64/libmunge.so.2.0.1:ro
  --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2:ro
  --bind /etc/slurm:/etc/slurm:ro
  --bind /etc/passwd:/etc/passwd:ro
  --bind /etc/group:/etc/group:ro
  --bind /run/munge:/run/munge:ro
  --env "PATH=/opt/epiforecaster/.venv/bin:/usr/local/bin:/usr/bin:/bin"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "PYTHONPATH=$RUN_ROOT/source"
  --env "XDG_CACHE_HOME=$RUN_ROOT/.cache"
  --env "TMPDIR=/tmp"
  --pwd "$RUN_ROOT"
)

# Site plugins differ between MN5 nodes. Mount only installed plugin trees.
for optional_site_path in /opt/.snoopy/lib /opt/EAR/current; do
  if [ -d "$optional_site_path" ]; then
    container_args+=(--bind "$optional_site_path:$optional_site_path:ro")
  fi
done

# Clean environments do not automatically preserve the allocation identity
# needed by the host srun client exposed above.
while IFS= read -r variable_name; do
  container_args+=(--env "$variable_name=${!variable_name}")
done < <(compgen -A variable SLURM_)
for variable_name in EPIFORECASTER_EXECUTION_MODE EPIFORECASTER_EXECUTION_ATTEMPT_ID; do
  container_args+=(--env "$variable_name=${!variable_name}")
done


# Keep the host shell alive so completion and cache cleanup always run.
singularity "${container_args[@]}" "$CONTAINER_SIF" /usr/local/bin/snakemake \
  --snakefile "$RUN_ROOT/source/workflow/Snakefile" \
  --configfile "$RUN_ROOT/config.yaml" \
  --workflow-profile "$RUN_ROOT/source/workflow/profiles/mn5" \
  --directory "$RUN_ROOT" \
  --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
  --cores "$((EXPECTED_TASKS * EXPECTED_CPUS_PER_TASK))" \
  --jobs "$EXPECTED_TASKS" \
  --resources "gpu=$EXPECTED_TASKS" \
  "$@"
