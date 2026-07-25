#!/bin/bash

# Run Snakemake as the lightweight control plane inside an existing MN5
# allocation. Scientific rules launch their own exact srun steps.

set -euo pipefail

RUN_ROOT="${1:?run root is required}"
SNAKEMAKE_BIN="${2:?Snakemake executable is required}"
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
  test -x "$SNAKEMAKE_BIN"
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
    if [ "${SLURM_NNODES:-}" != "1" ] \
      || [ "${SLURM_NTASKS:-}" != "$EXPECTED_TASKS" ]; then
      echo "External launcher does not expose the validated one-node, $EXPECTED_TASKS-task allocation" >&2
      return 2
    fi
    # The special external-launcher step omits these batch-shell variables.
    # The snapshot-owned adapter has already validated the containing
    # allocation through scontrol, so publish its sealed capacity to rules.
    export SLURM_JOB_NUM_NODES=1
    export SLURM_CPUS_PER_TASK="$EXPECTED_CPUS_PER_TASK"
  fi
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

"$SNAKEMAKE_BIN" \
  --snakefile "$RUN_ROOT/source/workflow/Snakefile" \
  --configfile "$RUN_ROOT/config.yaml" \
  --workflow-profile "$RUN_ROOT/source/workflow/profiles/mn5" \
  --directory "$RUN_ROOT" \
  --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
  "$@"
