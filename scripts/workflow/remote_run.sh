#!/bin/bash

# Publish and operate one immutable EpiForecaster run from the runner bundled
# with that run's exact source snapshot.  No implementation is streamed from a
# caller's current checkout, and no config-derived values arrive through SSH.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ACTIVE_SOURCE_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
BUNDLE_ROOT="$(dirname "$ACTIVE_SOURCE_ROOT")"
DISPATCH_FILE="$BUNDLE_ROOT/.state/dispatch.tsv"
DISPATCH_DIGEST_FILE="$DISPATCH_FILE.sha256"
CONFIG_PATH="$BUNDLE_ROOT/config.yaml"
SOURCE_FILE_LIST="$BUNDLE_ROOT/.state/source-files.bin"

SCHEMA_VERSION=""
RUN_ID=""
RUN_ROOT=""
RUN_PARENT=""
SOURCE_ROOT=""
SOURCE_SNAPSHOT_SHA256=""
CONFIG_SHA256=""
CONTAINER_SIF=""
CONTAINER_DIGEST_FILE=""
TOOLING_KEY=""
TOOLING_PYTHON=""
SNAKEMAKE_BIN=""
SNAKEMAKE_VERSION=""
WORKFLOW_KIND=""
WORKFLOW_TARGET=""
ACCOUNT=""
PARTITION=""
QOS=""
NODES=""
TASKS=""
CPUS_PER_TASK=""
GRES=""
RUNTIME=""
CLOSEOUT_SIGNAL_SECONDS=""

usage() {
  cat <<EOF
Usage: $(basename "$0") publish|dry-run|submit|attach [allocation-job-id]|status|unlock

The runner discovers config.yaml and .state/dispatch.tsv relative to its own
immutable source snapshot. It accepts no config-derived command-line values.
EOF
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Load the generated TSV without eval or shell sourcing. Values may contain
# spaces, but tabs and newlines are rejected when the request is rendered.
load_dispatch_request() {
  local field value extra

  test -f "$DISPATCH_FILE"
  test -f "$DISPATCH_DIGEST_FILE"
  if [ "$(sha256_file "$DISPATCH_FILE")" != "$(awk '{print $1}' "$DISPATCH_DIGEST_FILE")" ]; then
    echo "Dispatch request failed SHA-256 verification: $DISPATCH_FILE" >&2
    return 1
  fi

  while IFS=$'\t' read -r field value extra; do
    if [ -z "$field" ] || [ -z "$value" ] || [ -n "$extra" ]; then
      echo "Malformed dispatch request line in $DISPATCH_FILE" >&2
      return 1
    fi
    case "$field" in
      schema_version) SCHEMA_VERSION="$value" ;;
      run_id) RUN_ID="$value" ;;
      run_root) RUN_ROOT="$value" ;;
      run_parent) RUN_PARENT="$value" ;;
      source_root) SOURCE_ROOT="$value" ;;
      source_snapshot_sha256) SOURCE_SNAPSHOT_SHA256="$value" ;;
      config_sha256) CONFIG_SHA256="$value" ;;
      container_sif) CONTAINER_SIF="$value" ;;
      container_digest_file) CONTAINER_DIGEST_FILE="$value" ;;
      tooling_key) TOOLING_KEY="$value" ;;
      tooling_python) TOOLING_PYTHON="$value" ;;
      snakemake_bin) SNAKEMAKE_BIN="$value" ;;
      snakemake_version) SNAKEMAKE_VERSION="$value" ;;
      workflow_kind) WORKFLOW_KIND="$value" ;;
      workflow_target) WORKFLOW_TARGET="$value" ;;
      account) ACCOUNT="$value" ;;
      partition) PARTITION="$value" ;;
      qos) QOS="$value" ;;
      nodes) NODES="$value" ;;
      tasks) TASKS="$value" ;;
      cpus_per_task) CPUS_PER_TASK="$value" ;;
      gres) GRES="$value" ;;
      runtime) RUNTIME="$value" ;;
      closeout_signal_seconds) CLOSEOUT_SIGNAL_SECONDS="$value" ;;
      *)
        echo "Unknown dispatch request field: $field" >&2
        return 1
        ;;
    esac
  done <"$DISPATCH_FILE"
}

# Enforce the path and site contracts before any publication or submission.
validate_dispatch_request() {
  local field

  for field in \
    SCHEMA_VERSION RUN_ID RUN_ROOT RUN_PARENT SOURCE_ROOT \
    SOURCE_SNAPSHOT_SHA256 CONFIG_SHA256 CONTAINER_SIF \
    CONTAINER_DIGEST_FILE TOOLING_KEY TOOLING_PYTHON SNAKEMAKE_BIN \
    SNAKEMAKE_VERSION WORKFLOW_KIND WORKFLOW_TARGET ACCOUNT PARTITION QOS \
    NODES TASKS CPUS_PER_TASK GRES RUNTIME CLOSEOUT_SIGNAL_SECONDS; do
    if [ -z "${!field}" ]; then
      echo "Dispatch request is missing $field" >&2
      return 1
    fi
  done

  if [ "$SCHEMA_VERSION" != "1" ]; then
    echo "Unsupported dispatch request schema: $SCHEMA_VERSION" >&2
    return 1
  fi
  if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run ID in dispatch request: $RUN_ID" >&2
    return 1
  fi
  if [ "$RUN_ROOT" != "$RUN_PARENT/$RUN_ID" ] \
    || [ "$SOURCE_ROOT" != "$RUN_ROOT/source" ]; then
    echo "Dispatch request contains inconsistent run paths" >&2
    return 1
  fi
  if [[ ! "$SOURCE_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Dispatch request contains an invalid SHA-256 identity" >&2
    return 1
  fi
  if [ "$PARTITION" = "gpp" ]; then
    # CPU-only preprocessing: one task, memory scales with the CPU count.
    if [ "$WORKFLOW_KIND" != "preprocess" ] \
      || [ "$NODES" != "1" ] \
      || [ "$TASKS" != "1" ] \
      || [[ ! "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] \
      || [ "$CPUS_PER_TASK" -gt 112 ] \
      || [ "$GRES" != "none" ] \
      || [[ "$QOS" != gp_bscls && "$QOS" != gp_debug ]]; then
      echo "Dispatch request violates the MN5 single-node GPP allocation contract" >&2
      return 1
    fi
  elif [ "$PARTITION" = "main" ]; then
    # CPU-only preprocessing on one Nord4 node (48 cores, shared GPFS).
    if [ "$WORKFLOW_KIND" != "preprocess" ] \
      || [ "$NODES" != "1" ] \
      || [ "$TASKS" != "1" ] \
      || [[ ! "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] \
      || [ "$CPUS_PER_TASK" -gt 48 ] \
      || [ "$GRES" != "none" ] \
      || [[ "$QOS" != bsc_ls && "$QOS" != debug ]]; then
      echo "Dispatch request violates the Nord4 single-node allocation contract" >&2
      return 1
    fi
  elif [ "$NODES" != "1" ] \
    || [[ ! "$TASKS" =~ ^[1-4]$ ]] \
    || [ "$CPUS_PER_TASK" != "20" ] \
    || [ "$GRES" != "gpu:$TASKS" ]; then
    echo "Dispatch request violates the MN5 single-node allocation contract" >&2
    return 1
  fi
  if ! [[ "$CLOSEOUT_SIGNAL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Dispatch request contains an invalid close-out signal duration" >&2
    return 1
  fi
}

verify_config() {
  if [ "$(sha256_file "$CONFIG_PATH")" != "$CONFIG_SHA256" ]; then
    echo "Effective workflow config failed SHA-256 verification: $CONFIG_PATH" >&2
    return 1
  fi
}

load_singularity() {
  local status=0

  # Site profile scripts and the Nord4 module function read unset variables
  # and return non-zero, so the module section runs without errexit/nounset.
  set +eu
  load_singularity_module
  status=$?
  set -eu
  return "$status"
}

load_singularity_module() {
  local profile_script candidate loaded=0

  if ! command -v module >/dev/null 2>&1; then
    for profile_script in \
      /etc/profile.d/00-bsc.sh \
      /etc/profile.d/01-module.sh \
      /etc/profile.d/01-modules.sh \
      /apps/modules/LMOD/8.7/lmod/8.7/init/profile; do
      if [ -r "$profile_script" ]; then
        # shellcheck disable=SC1090
        source "$profile_script"
      fi
      if command -v module >/dev/null 2>&1; then
        break
      fi
    done
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

container_exec() {
  local working_directory="$1"
  shift

  load_singularity
  singularity exec \
    --cleanenv \
    --containall \
    --bind "$BUNDLE_ROOT:$BUNDLE_ROOT:rw" \
    --env "PATH=/opt/epiforecaster/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONDONTWRITEBYTECODE=1" \
    --env "PYTHONPATH=$ACTIVE_SOURCE_ROOT" \
    --env "XDG_CACHE_HOME=$BUNDLE_ROOT/.cache" \
    --env "TMPDIR=/tmp" \
    --pwd "$working_directory" \
    "$CONTAINER_SIF" \
    "$@"
}

verify_source() {
  local observed

  observed="$(
    container_exec \
      "$ACTIVE_SOURCE_ROOT" \
      /usr/local/bin/python \
      -m scripts.workflow.run_config verify \
        --config "$CONFIG_PATH" \
        --source-root "$ACTIVE_SOURCE_ROOT" \
        --file-list "$SOURCE_FILE_LIST"
  )"
  if [ "$observed" != "$SOURCE_SNAPSHOT_SHA256" ]; then
    echo "Source snapshot identity does not match dispatch request" >&2
    return 1
  fi
}

verify_container() {
  local recorded_name recorded_sha

  test -f "$CONTAINER_SIF"
  test -f "$CONTAINER_DIGEST_FILE"
  read -r recorded_sha recorded_name <"$CONTAINER_DIGEST_FILE"
  recorded_name="${recorded_name#\*}"
  if [ "$recorded_sha" != "$(sha256_file "$CONTAINER_SIF")" ] \
    || [ "$recorded_name" != "$(basename "$CONTAINER_SIF")" ]; then
    echo "Container failed immutable checksum verification: $CONTAINER_SIF" >&2
    return 1
  fi
}

verify_tooling() {
  local installed_version

  installed_version="$(
    container_exec "$ACTIVE_SOURCE_ROOT" "$SNAKEMAKE_BIN" --version
  )"
  if [ "$installed_version" != "$SNAKEMAKE_VERSION" ]; then
    echo "Snakemake version mismatch: $installed_version" >&2
    return 1
  fi
}

# Verify all immutable inputs represented by the dispatch request.
verify_bundle() {
  verify_config
  verify_source
  verify_container
  verify_tooling
}

PUBLISH_CREATED_ROOT=0
PUBLISH_RESERVATION=""
PUBLISH_TOKEN=""

# Remove only token-owned partial publication state while preserving the first
# causal failure. Successfully published runs and permanent reservations stay.
cleanup_failed_publish() {
  local status=$?
  local owner=""

  trap - EXIT
  if [ "$status" -ne 0 ] && [ -n "$PUBLISH_RESERVATION" ]; then
    if [ -f "$PUBLISH_RESERVATION/owner" ]; then
      owner="$(cat "$PUBLISH_RESERVATION/owner")"
    fi
    if [ "$owner" = "$PUBLISH_TOKEN" ]; then
      if [ ! -f "$RUN_ROOT/.state/published.done" ]; then
        if [ "$PUBLISH_CREATED_ROOT" -eq 1 ]; then
          chmod -R u+w "$RUN_ROOT" 2>/dev/null || true
          rm -rf -- "$RUN_ROOT"
        fi
        chmod -R u+w "$PUBLISH_RESERVATION" 2>/dev/null || true
        rm -rf -- "$PUBLISH_RESERVATION"
      fi
    fi
  fi
  exit "$status"
}

# Publish a verified staging bundle under its immutable scientific run ID.
publish_bundle() {
  local reservation_root receipt temporary_receipt

  if [ "$(dirname "$BUNDLE_ROOT")" != "$RUN_PARENT" ] \
    || [[ "$(basename "$BUNDLE_ROOT")" != ".staging-$RUN_ID."* ]]; then
    echo "Publish must execute from a run-local staging directory" >&2
    return 1
  fi

  verify_bundle

  reservation_root="$RUN_PARENT/.reservations"
  PUBLISH_RESERVATION="$reservation_root/$RUN_ID"
  PUBLISH_TOKEN="$(basename "$BUNDLE_ROOT")"
  mkdir -p "$reservation_root"
  if ! mkdir "$PUBLISH_RESERVATION"; then
    echo "Run ID is already reserved and will not be reused: $RUN_ID" >&2
    return 73
  fi
  printf '%s\n' "$PUBLISH_TOKEN" >"$PUBLISH_RESERVATION/owner"
  trap cleanup_failed_publish EXIT

  if ! mkdir "$RUN_ROOT"; then
    echo "Run already exists and will not be overwritten: $RUN_ROOT" >&2
    return 73
  fi
  PUBLISH_CREATED_ROOT=1

  mv \
    "$ACTIVE_SOURCE_ROOT" \
    "$CONFIG_PATH" \
    "$BUNDLE_ROOT/.state" \
    "$RUN_ROOT/"

  chmod -R a-w "$RUN_ROOT/source"
  chmod 0444 \
    "$RUN_ROOT/config.yaml" \
    "$RUN_ROOT/.state/source-files.bin" \
    "$RUN_ROOT/.state/dispatch.tsv" \
    "$RUN_ROOT/.state/dispatch.tsv.sha256"

  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/.state/workers"
  rmdir "$BUNDLE_ROOT"
  chmod 0444 "$PUBLISH_RESERVATION/owner"
  chmod 0555 "$PUBLISH_RESERVATION"

  # The receipt is the final atomic publication step. Once visible, cleanup
  # preserves both the run and its permanent no-reuse reservation.
  receipt="$RUN_ROOT/.state/published.done"
  temporary_receipt="$receipt.$$"
  printf '%s\n' \
    "schema_version=1" \
    "run_id=$RUN_ID" \
    "source_snapshot_sha256=$SOURCE_SNAPSHOT_SHA256" \
    "config_sha256=$CONFIG_SHA256" \
    "tooling_key=$TOOLING_KEY" >"$temporary_receipt"
  chmod 0444 "$temporary_receipt"
  mv "$temporary_receipt" "$receipt"

  trap - EXIT
  printf 'published:%s' "$RUN_ROOT"
}

require_published_location() {
  if [ "$BUNDLE_ROOT" != "$RUN_ROOT" ] \
    || [ "$ACTIVE_SOURCE_ROOT" != "$SOURCE_ROOT" ] \
    || [ ! -f "$RUN_ROOT/.state/published.done" ]; then
    echo "Run is not a published immutable bundle: $RUN_ROOT" >&2
    return 1
  fi
}

require_published_run() {
  require_published_location
  verify_bundle
}

run_snakemake() {
  container_exec "$RUN_ROOT" "$SNAKEMAKE_BIN" \
    --snakefile "$SOURCE_ROOT/workflow/Snakefile" \
    --configfile "$RUN_ROOT/config.yaml" \
    --workflow-profile "$SOURCE_ROOT/workflow/profiles/mn5" \
    --directory "$RUN_ROOT" \
    --runtime-source-cache-path "$RUN_ROOT/.snakemake/source-cache" \
    --cores "$((TASKS * CPUS_PER_TASK))" \
    --jobs "$TASKS" \
    --resources "gpu=$TASKS" \
    "$@"
}

dry_run() {
  require_published_run
  run_snakemake --dry-run
}

binding_value() {
  local binding="$1"
  local key="$2"

  awk -F '\t' -v key="$key" '$1 == key { print $2; exit }' "$binding"
}

job_field() {
  local record="$1"
  local key="$2"
  local token

  for token in $record; do
    case "$token" in
      "$key="*)
        printf '%s' "${token#*=}"
        return 0
        ;;
    esac
  done
  return 1
}

new_attempt() {
  local mode="$1"
  local attempt_id attempt_root

  attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$mode-$$-$RANDOM"
  attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  mkdir -p "$attempt_root"
  printf '%s' "$attempt_id"
}

activate_attempt() {
  local attempt_id="$1"
  local temporary="$RUN_ROOT/.state/active_execution.$$"

  printf '%s\n' "$attempt_id" >"$temporary"
  mv "$temporary" "$RUN_ROOT/.state/active_execution"
}

write_attempt_binding() {
  local attempt_id="$1"
  local mode="$2"
  local allocation_job_id="$3"
  shift 3
  local attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  local temporary="$attempt_root/binding.tsv.$$"

  printf '%s\n' \
    $'schema_version\t1' \
    "attempt_id"$'\t'"$attempt_id" \
    "mode"$'\t'"$mode" \
    "allocation_job_id"$'\t'"$allocation_job_id" \
    "requested_nodes"$'\t'"$NODES" \
    "requested_tasks"$'\t'"$TASKS" \
    "requested_cpus_per_task"$'\t'"$CPUS_PER_TASK" \
    "requested_gres"$'\t'"$GRES" \
    "requested_runtime_minutes"$'\t'"$RUNTIME" \
    "created_at"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$@" >"$temporary"
  mv "$temporary" "$attempt_root/binding.tsv"
}

record_launcher_completion() {
  local attempt_root="$1"
  local status="$2"
  local receipt="$attempt_root/completed.tsv"
  local temporary="$receipt.$$"

  if [ -f "$receipt" ]; then
    return 0
  fi
  printf '%s\n' \
    $'schema_version\t1' \
    "exit_code"$'\t'"$status" \
    "completed_at"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$temporary"
  mv "$temporary" "$receipt"
}

# Print one compact active response, or nothing when the latest attempt ended.
active_attempt_status() {
  local attempt_id attempt_root binding mode job_id step_id state launcher_pid

  if [ ! -s "$RUN_ROOT/.state/active_execution" ]; then
    return 0
  fi
  attempt_id="$(cat "$RUN_ROOT/.state/active_execution")"
  if [[ ! "$attempt_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid active execution attempt ID: $attempt_id" >&2
    return 1
  fi
  attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  binding="$attempt_root/binding.tsv"
  if [ ! -f "$binding" ] || [ -f "$attempt_root/completed.tsv" ]; then
    return 0
  fi

  mode="$(binding_value "$binding" mode)"
  job_id="$(binding_value "$binding" allocation_job_id)"
  if [ "$mode" = "batch" ]; then
    state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "$state" ]; then
      printf 'active:%s:%s' "$job_id" "$state"
    fi
    return 0
  fi

  if [ "$mode" != "existing-allocation" ]; then
    echo "Unknown execution attempt mode: $mode" >&2
    return 1
  fi
  if [ -s "$attempt_root/coordinator.step_id" ]; then
    step_id="$(cat "$attempt_root/coordinator.step_id")"
    state="$(
      squeue -h --steps -j "$job_id" -o '%i' 2>/dev/null \
        | awk -v id="$job_id.$step_id" '$1 == id { print "RUNNING"; exit }'
    )"
    if [ -n "$state" ]; then
      printf 'active:%s:%s' "$job_id" "$state"
      return 0
    fi
  fi
  if [ -s "$attempt_root/launcher.pid" ]; then
    launcher_pid="$(cat "$attempt_root/launcher.pid")"
    if [[ "$launcher_pid" =~ ^[1-9][0-9]*$ ]] \
      && kill -0 "$launcher_pid" 2>/dev/null; then
      printf 'active:%s:LAUNCHING' "$job_id"
    fi
  fi
}

link_attempt_to_allocation() {
  local job_id="$1"
  local attempt_id="$2"
  local lookup="$RUN_PARENT/by-slurm/$job_id/$RUN_ID"

  mkdir -p "$lookup"
  ln -s "$RUN_ROOT" "$lookup/$attempt_id"
}

# Submit at most one active coordinator using only the sealed dispatch request.
submit_run() {
  local coordinator active job_id result attempt_id attempt_root

  require_published_run
  coordinator="$SOURCE_ROOT/scripts/workflow/snakemake_coordinator.sbatch"
  test -x "$coordinator"

  exec 9>"$RUN_ROOT/.state/coordinator.launch.lock"
  flock 9

  if [ -f "$RUN_ROOT/.state/workflow.done" ]; then
    printf complete
    return 0
  fi

  active="$(active_attempt_status)"
  if [ -n "$active" ]; then
    printf '%s' "$active"
    return 0
  fi

  attempt_id="$(new_attempt batch)"
  attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  local -a submit_arguments=(
    "--chdir=$RUN_ROOT"
    "--output=$RUN_ROOT/logs/snakemake-%j.out"
    "--error=$RUN_ROOT/logs/snakemake-%j.err"
    "--open-mode=append"
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--qos=$QOS"
    "--nodes=$NODES"
    "--ntasks=$TASKS"
    "--ntasks-per-node=$TASKS"
    "--cpus-per-task=$CPUS_PER_TASK"
    "--gres=$GRES"
    "--time=$RUNTIME"
  )
  if [ "$CLOSEOUT_SIGNAL_SECONDS" -gt 0 ]; then
    submit_arguments+=("--signal=USR1@$CLOSEOUT_SIGNAL_SECONDS")
  fi
  submit_arguments+=(
    "$coordinator"
    "$SOURCE_ROOT/scripts/workflow/snakemake_coordinator.sh"
    "$RUN_ROOT"
    "$CONTAINER_SIF"
    "$TASKS"
    "$CPUS_PER_TASK"
    batch
    "$attempt_root"
  )
  job_id="$(sbatch --parsable "${submit_arguments[@]}")"
  job_id="${job_id%%;*}"

  write_attempt_binding "$attempt_id" batch "$job_id"
  activate_attempt "$attempt_id"
  printf '%s\n' "$job_id" >"$RUN_ROOT/.state/coordinator.job_id"
  link_attempt_to_allocation "$job_id" "$attempt_id"
  result="submitted:$job_id"
  printf '%s' "$result"
}

discover_interactive_allocation() {
  local -a candidates

  mapfile -t candidates < <(
    squeue -h -u "$USER" -t RUNNING -o '%i|%q' \
      | awk -F '|' '$2 == "acc_interactive" { print $1 }'
  )
  if [ "${#candidates[@]}" -ne 1 ]; then
    echo "Expected exactly one running acc_interactive allocation, found ${#candidates[@]}" >&2
    return 1
  fi
  printf '%s' "${candidates[0]}"
}

validate_interactive_allocation() {
  local job_id="$1"
  local record state user_id account qos partition nodes tasks cpus alloc_tres
  local gpu_count required_cpus end_time end_epoch now_epoch remaining_seconds

  record="$(scontrol show job -o "$job_id")"
  state="$(job_field "$record" JobState)"
  user_id="$(job_field "$record" UserId)"
  account="$(job_field "$record" Account)"
  qos="$(job_field "$record" QOS)"
  partition="$(job_field "$record" Partition)"
  nodes="$(job_field "$record" NumNodes)"
  tasks="$(job_field "$record" NumTasks)"
  cpus="$(job_field "$record" NumCPUs)"
  alloc_tres="$(job_field "$record" AllocTRES)"
  end_time="$(job_field "$record" EndTime)"

  user_id="${user_id%%(*}"
  if [ "$state" != "RUNNING" ] \
    || [ "$user_id" != "$USER" ] \
    || [ "$account" != "$ACCOUNT" ] \
    || [ "$qos" != "acc_interactive" ] \
    || [[ "$partition" != acc* ]]; then
    echo "Allocation $job_id is not a compatible running ACC interactive allocation" >&2
    return 1
  fi
  if ! [[ "$nodes" =~ ^[1-9][0-9]*$ && "$tasks" =~ ^[1-9][0-9]*$ && "$cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "Allocation $job_id has malformed topology" >&2
    return 1
  fi
  required_cpus=$((TASKS * CPUS_PER_TASK))
  if [ "$nodes" -lt "$NODES" ] || [ "$tasks" -lt "$TASKS" ] || [ "$cpus" -lt "$required_cpus" ]; then
    echo "Allocation $job_id cannot satisfy $NODES node, $TASKS task, $required_cpus CPU request" >&2
    return 1
  fi
  gpu_count=0
  if [[ "$alloc_tres" =~ gres/gpu=([0-9]+) ]]; then
    gpu_count="${BASH_REMATCH[1]}"
  fi
  if [ "$gpu_count" -lt "$TASKS" ]; then
    echo "Allocation $job_id has $gpu_count GPUs, expected at least $TASKS" >&2
    return 1
  fi

  end_epoch="$(date -d "$end_time" +%s)"
  now_epoch="$(date -u +%s)"
  remaining_seconds=$((end_epoch - now_epoch))
  if [ "$remaining_seconds" -lt $((RUNTIME * 60)) ]; then
    echo "Allocation $job_id has insufficient remaining wall time for $RUNTIME minutes" >&2
    return 1
  fi

  printf '%s\n' \
    "observed_partition"$'\t'"$partition" \
    "observed_qos"$'\t'"$qos" \
    "observed_nodes"$'\t'"$nodes" \
    "observed_tasks"$'\t'"$tasks" \
    "observed_cpus"$'\t'"$cpus" \
    "observed_gpus"$'\t'"$gpu_count" \
    "observed_end_time"$'\t'"$end_time"
}

attach_worker() {
  local job_id="$1"
  local attempt_id="$2"
  local attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  local coordinator="$SOURCE_ROOT/scripts/workflow/snakemake_coordinator.sh"
  local status

  require_published_run
  test -x "$coordinator"
  test -f "$attempt_root/binding.tsv"

  set +e
  srun \
    --jobid="$job_id" \
    --external-launcher \
    --open-mode=append \
    --output="$attempt_root/coordinator.out" \
    --error="$attempt_root/coordinator.err" \
    "$coordinator" \
    "$RUN_ROOT" \
    "$CONTAINER_SIF" \
    "$TASKS" \
    "$CPUS_PER_TASK" \
    existing-allocation \
    "$attempt_root"
  status=$?
  set -e
  record_launcher_completion "$attempt_root" "$status"
  return "$status"
}

attach_run() {
  local job_id="${1:-}"
  local active compatibility attempt_id attempt_root launcher_pid

  require_published_run
  exec 9>"$RUN_ROOT/.state/coordinator.launch.lock"
  flock 9

  if [ -f "$RUN_ROOT/.state/workflow.done" ]; then
    printf complete
    return 0
  fi
  active="$(active_attempt_status)"
  if [ -n "$active" ]; then
    printf '%s' "$active"
    return 0
  fi
  if [ -z "$job_id" ]; then
    job_id="$(discover_interactive_allocation)"
  fi
  if ! [[ "$job_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid interactive allocation job ID: $job_id" >&2
    return 2
  fi
  compatibility="$(validate_interactive_allocation "$job_id")"

  attempt_id="$(new_attempt existing-allocation)"
  attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
  write_attempt_binding \
    "$attempt_id" \
    existing-allocation \
    "$job_id" \
    "$compatibility"
  activate_attempt "$attempt_id"
  link_attempt_to_allocation "$job_id" "$attempt_id"

  nohup "$SCRIPT_PATH" attach-worker "$job_id" "$attempt_id" \
    >"$attempt_root/launcher.out" \
    2>"$attempt_root/launcher.err" \
    </dev/null 9>&- &
  launcher_pid=$!
  printf '%s\n' "$launcher_pid" >"$attempt_root/launcher.pid"
  printf 'attached:%s:%s' "$job_id" "$attempt_id"
}

show_status() {
  local attempt_id="" attempt_root="" binding="" mode="" job_id="" active=""
  local exit_code="" step_id=""

  require_published_location
  if [ -s "$RUN_ROOT/.state/active_execution" ]; then
    attempt_id="$(cat "$RUN_ROOT/.state/active_execution")"
    attempt_root="$RUN_ROOT/.state/executions/$attempt_id"
    binding="$attempt_root/binding.tsv"
  fi

  if [ -f "$binding" ]; then
    mode="$(binding_value "$binding" mode)"
    job_id="$(binding_value "$binding" allocation_job_id)"
    active="$(active_attempt_status)"
    if [ -n "$active" ]; then
      echo "coordinator=${active##*:} allocation_job_id=$job_id mode=$mode attempt_id=$attempt_id"
    elif [ -f "$attempt_root/completed.tsv" ]; then
      exit_code="$(binding_value "$attempt_root/completed.tsv" exit_code)"
      step_id="$(cat "$attempt_root/coordinator.step_id" 2>/dev/null || true)"
      echo "coordinator=terminal exit_code=$exit_code allocation_job_id=$job_id coordinator_step_id=${step_id:-unknown} mode=$mode attempt_id=$attempt_id"
    else
      echo "coordinator=not-active allocation_job_id=$job_id mode=$mode attempt_id=$attempt_id"
    fi
  else
    echo coordinator=not-started
  fi

  if [ -f "$RUN_ROOT/.state/workflow.done" ]; then
    echo workflow=complete
  else
    echo workflow=incomplete
  fi

  if [ -n "$attempt_root" ]; then
    tail -n 40 "$attempt_root/coordinator.out" 2>/dev/null || true
    tail -n 40 "$attempt_root/coordinator.err" 2>/dev/null || true
    tail -n 40 "$attempt_root/launcher.err" 2>/dev/null || true
  fi
  if [ "$mode" = "batch" ] && [ -n "$job_id" ]; then
    tail -n 40 "$RUN_ROOT/logs/snakemake-$job_id.out" 2>/dev/null || true
    tail -n 40 "$RUN_ROOT/logs/snakemake-$job_id.err" 2>/dev/null || true
  fi
}

unlock_run() {
  local active

  require_published_run
  active="$(active_attempt_status)"
  if [ -n "$active" ]; then
    echo "Coordinator is still active: $active" >&2
    return 1
  fi
  run_snakemake --unlock
}

main() {
  local command="${1:-}"

  if [ "$#" -gt 0 ]; then
    shift
  fi
  if [ -z "$command" ]; then
    usage >&2
    return 2
  fi

  load_dispatch_request
  validate_dispatch_request
  case "$command" in
    publish)
      [ "$#" -eq 0 ] || return 2
      publish_bundle
      ;;
    dry-run)
      [ "$#" -eq 0 ] || return 2
      dry_run
      ;;
    submit)
      [ "$#" -eq 0 ] || return 2
      submit_run
      ;;
    attach)
      [ "$#" -le 1 ] || return 2
      attach_run "${1:-}"
      ;;
    attach-worker)
      [ "$#" -eq 2 ] || return 2
      attach_worker "$1" "$2"
      ;;
    status)
      [ "$#" -eq 0 ] || return 2
      show_status
      ;;
    unlock)
      [ "$#" -eq 0 ] || return 2
      unlock_run
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
