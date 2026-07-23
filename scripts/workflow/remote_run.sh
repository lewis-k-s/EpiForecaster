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

usage() {
  cat <<EOF
Usage: $(basename "$0") publish|dry-run|submit|status|unlock

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
    NODES TASKS CPUS_PER_TASK GRES RUNTIME; do
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
  if [ "$NODES" != "1" ] \
    || [[ ! "$TASKS" =~ ^[1-4]$ ]] \
    || [ "$CPUS_PER_TASK" != "20" ] \
    || [ "$GRES" != "gpu:$TASKS" ]; then
    echo "Dispatch request violates the MN5 single-node allocation contract" >&2
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
    --env "HOME=/tmp" \
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
    "$@"
}

dry_run() {
  require_published_run
  run_snakemake --dry-run
}

# Submit at most one active coordinator using only the sealed dispatch request.
submit_run() {
  local coordinator previous state job_id result

  require_published_run
  coordinator="$SOURCE_ROOT/scripts/workflow/snakemake_coordinator.sbatch"
  test -x "$coordinator"

  exec 9>"$RUN_ROOT/.state/coordinator.launch.lock"
  flock 9

  if [ -f "$RUN_ROOT/.state/workflow.done" ]; then
    printf complete
    return 0
  fi

  if [ -s "$RUN_ROOT/.state/coordinator.job_id" ]; then
    previous="$(cat "$RUN_ROOT/.state/coordinator.job_id")"
    state="$(squeue -h -j "$previous" -o '%T' 2>/dev/null | head -n 1 || true)"
    case "$state" in
      PENDING | RUNNING | CONFIGURING | COMPLETING)
        printf 'active:%s:%s' "$previous" "$state"
        return 0
        ;;
    esac
  fi

  job_id="$(
    sbatch --parsable \
      --chdir="$RUN_ROOT" \
      --output="$RUN_ROOT/logs/snakemake-%j.out" \
      --error="$RUN_ROOT/logs/snakemake-%j.err" \
      --open-mode=append \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --qos="$QOS" \
      --nodes="$NODES" \
      --ntasks="$TASKS" \
      --ntasks-per-node="$TASKS" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --gres="$GRES" \
      --time="$RUNTIME" \
      "$coordinator" \
      "$RUN_ROOT" \
      "$CONTAINER_SIF" \
      "$TASKS" \
      "$CPUS_PER_TASK"
  )"
  job_id="${job_id%%;*}"

  printf '%s\n' "$job_id" >"$RUN_ROOT/.state/coordinator.job_id"
  mkdir -p "$RUN_PARENT/by-slurm"
  ln -s "$RUN_ROOT" "$RUN_PARENT/by-slurm/${job_id}_0"
  result="submitted:$job_id"
  printf '%s' "$result"
}

show_status() {
  local job_id="" state

  require_published_location
  if [ -f "$RUN_ROOT/.state/coordinator.job_id" ]; then
    job_id="$(cat "$RUN_ROOT/.state/coordinator.job_id")"
  fi

  if [ -n "$job_id" ]; then
    state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "$state" ]; then
      echo "coordinator=$state job_id=$job_id execution_id=${job_id}_0"
    else
      sacct -n -X -j "$job_id" \
        --format=JobIDRaw,State,ExitCode 2>/dev/null | head -n 1 || true
    fi
  else
    echo coordinator=not-submitted
  fi

  if [ -f "$RUN_ROOT/.state/workflow.done" ]; then
    echo workflow=complete
  else
    echo workflow=incomplete
  fi

  if [ -n "$job_id" ]; then
    tail -n 40 "$RUN_ROOT/logs/snakemake-$job_id.out" 2>/dev/null || true
    tail -n 40 "$RUN_ROOT/logs/snakemake-$job_id.err" 2>/dev/null || true
  fi
}

unlock_run() {
  local job_id state

  require_published_run
  if [ -s "$RUN_ROOT/.state/coordinator.job_id" ]; then
    job_id="$(cat "$RUN_ROOT/.state/coordinator.job_id")"
    state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "$state" ]; then
      echo "Coordinator job $job_id is still $state" >&2
      return 1
    fi
  fi
  run_snakemake --unlock
}

main() {
  local command="${1:-}"

  if [ "$#" -ne 1 ]; then
    usage >&2
    return 2
  fi

  load_dispatch_request
  validate_dispatch_request
  case "$command" in
    publish) publish_bundle ;;
    dry-run) dry_run ;;
    submit) submit_run ;;
    status) show_status ;;
    unlock) unlock_run ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
