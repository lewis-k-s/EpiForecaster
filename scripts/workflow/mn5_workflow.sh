#!/bin/bash

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONF="${CLUSTER_CONF:-$REPO_ROOT/.cluster/mn5.conf}"

# shellcheck disable=SC1090
source "$CONF"

COMPUTE_HOST="${COMPUTE_HOST:?COMPUTE_HOST must be configured}"
TRANSFER_HOST="${TRANSFER_HOST:?TRANSFER_HOST must be configured}"
CONTAINER_NETWORK_HOST="${CONTAINER_NETWORK_HOST:-wifi-acc}"
CONTAINER_NETWORK_HOSTNAME="${CONTAINER_NETWORK_HOSTNAME:-}"
REMOTE_WORKFLOW_RUN_ROOT="${REMOTE_WORKFLOW_RUN_ROOT:?REMOTE_WORKFLOW_RUN_ROOT must be configured}"
REMOTE_WORKFLOW_TOOLING_ROOT="${REMOTE_WORKFLOW_TOOLING_ROOT:?REMOTE_WORKFLOW_TOOLING_ROOT must be configured}"
REMOTE_CONTAINER_SIF="${REMOTE_CONTAINER_SIF:?REMOTE_CONTAINER_SIF must be configured}"
DVC_REMOTE_PATH="${DVC_REMOTE_PATH:?DVC_REMOTE_PATH must be configured}"
DVC_CACHE_PATH="${DVC_CACHE_PATH:?DVC_CACHE_PATH must be configured}"
TEMPLATE="${WORKFLOW_CONFIG_TEMPLATE:-$REPO_ROOT/workflow/config/mn5_train.yaml}"
VPN_HELPER="${WORKVPN_HELPER:-/Users/lewis/.codex/skills/workvpn/vpn}"
SNAKEMAKE_EXPECTED_VERSION="9.23.1"
TOOLING_KEY=""
REMOTE_TOOLING_REQUEST=""
REMOTE_TOOLING_ENV=""
REMOTE_SNAKEMAKE_BIN=""
REMOTE_TOOLING_RUNNER=""
TOOLING_BOOTSTRAP="$REPO_ROOT/scripts/workflow/remote_tooling.sh"

# -----------------------------------------------------------------------------
# Command-line help and validation
# -----------------------------------------------------------------------------

# Print the public dispatcher interface. Commands intentionally separate run
# preparation from dry-run, submission, observation, and recovery operations.
usage() {
  cat <<EOF
Usage:
  $(basename "$0") tooling
  $(basename "$0") prepare [run-id] [--config TEMPLATE]
  $(basename "$0") dry-run <run-id>
  $(basename "$0") run <run-id>
  $(basename "$0") status <run-id>
  $(basename "$0") unlock <run-id>
  $(basename "$0") path <run-id>

prepare snapshots this worktree into an atomically reserved project-GPFS run.
The default template is workflow/config/mn5_train.yaml; use --config to select
evaluation, cross-validation, HPO, or ablation.
run submits the one-node ACC allocation declared by that immutable config.
EOF
}

# Reject identifiers that could escape the configured immutable run root.
validate_run_id() {
  local run_id="$1"

  if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run id '$run_id'" >&2
    return 2
  fi
}

# Generate a readable, collision-resistant run ID for ad hoc preparations.
generate_run_id() {
  local branch short_sha timestamp

  branch="$(git branch --show-current)"
  branch="${branch:-detached}"
  branch="$(printf '%s' "$branch" | tr '/ :' '---' | tr -cd 'A-Za-z0-9._-')"
  short_sha="$(git rev-parse --short=10 HEAD)"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s-%s-%s-%s\n' "$branch" "$short_sha" "$timestamp" "$$-$RANDOM"
}

# Resolve a validated run ID below the project-level remote run directory.
run_root() {
  printf '%s/%s\n' "$REMOTE_WORKFLOW_RUN_ROOT" "$1"
}

# Quote one value for interpolation into a remote Bash command string.
shell_quote() {
  printf '%q' "$1"
}

# -----------------------------------------------------------------------------
# Local and remote command helpers
# -----------------------------------------------------------------------------

# Execute one already-installed remote program with safely quoted arguments.
# Implementation code is never generated, serialized, or streamed over SSH.
remote_exec_on() {
  local host="$1"
  shift
  local remote_command=""
  local argument
  local -a ssh_options=()

  if [ "$#" -eq 0 ]; then
    echo "remote_exec_on requires a remote executable" >&2
    return 2
  fi

  if [ "$host" = "$CONTAINER_NETWORK_HOST" ] \
    && [ -n "$CONTAINER_NETWORK_HOSTNAME" ]; then
    ssh_options+=(-o "HostName=$CONTAINER_NETWORK_HOSTNAME")
  fi

  for argument in "$@"; do
    if [ -n "$remote_command" ]; then
      remote_command+=" "
    fi
    remote_command+="$(shell_quote "$argument")"
  done

  ssh "${ssh_options[@]}" "$host" "$remote_command"
}

# Execute a transferred program on the authoritative compute endpoint.
remote_exec() {
  remote_exec_on "$COMPUTE_HOST" "$@"
}

# Compute a SHA-256 digest on Linux or macOS without changing caller behavior.
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# -----------------------------------------------------------------------------
# Locked Snakemake tooling
# -----------------------------------------------------------------------------

# Derive the content-addressed tooling paths once per dispatcher invocation.
set_tooling_identity() {
  if [ -n "$TOOLING_KEY" ]; then
    return 0
  fi

  TOOLING_KEY="$(
    PYTHONDONTWRITEBYTECODE=1 python3 \
      -m scripts.workflow.tooling_identity \
      --lock-file "$REPO_ROOT/uv.lock" \
      --bootstrap "$TOOLING_BOOTSTRAP"
  )"
  REMOTE_TOOLING_REQUEST="$REMOTE_WORKFLOW_TOOLING_ROOT/requests/$TOOLING_KEY"
  REMOTE_TOOLING_ENV="$REMOTE_WORKFLOW_TOOLING_ROOT/envs/$TOOLING_KEY"
  REMOTE_SNAKEMAKE_BIN="$REMOTE_TOOLING_ENV/bin/snakemake"
  REMOTE_TOOLING_RUNNER="$REMOTE_TOOLING_REQUEST/remote_tooling.sh"
}

# Report whether the transferred tooling request and installed environment agree.
remote_tooling_state() {
  remote_exec "$REMOTE_TOOLING_RUNNER" probe
}

# Copy the locked dependency context and its ordinary remote bootstrap script.
# Scientific source remains separate in each immutable run bundle.
sync_tooling_context() {
  local tmp_base tmp_dir request_file tab

  tab=$'\t'
  remote_exec_on \
    "$TRANSFER_HOST" \
    mkdir -p \
    "$REMOTE_TOOLING_REQUEST" \
    "$REMOTE_WORKFLOW_TOOLING_ROOT/envs" \
    "$REMOTE_WORKFLOW_TOOLING_ROOT/cache" \
    "$REMOTE_WORKFLOW_TOOLING_ROOT/python"

  tmp_base="${HUBDIR:-/Volumes/HUBSSD}/tmp/epiforecaster-tooling"
  mkdir -p "$tmp_base"
  tmp_dir="$(mktemp -d "$tmp_base/request.XXXXXX")"
  request_file="$tmp_dir/request.tsv"
  mkdir -p "$tmp_dir/context"
  cp "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock" "$tmp_dir/context/"
  cp "$TOOLING_BOOTSTRAP" "$tmp_dir/remote_tooling.sh"

  if [ -f "$REPO_ROOT/README.md" ]; then
    cp "$REPO_ROOT/README.md" "$tmp_dir/context/"
  fi

  printf '%s\n' \
    "tooling_key${tab}$TOOLING_KEY" \
    "environment${tab}$REMOTE_TOOLING_ENV" \
    "cache_root${tab}$REMOTE_WORKFLOW_TOOLING_ROOT/cache" \
    "python_root${tab}$REMOTE_WORKFLOW_TOOLING_ROOT/python" \
    "expected_version${tab}$SNAKEMAKE_EXPECTED_VERSION" \
    "pyproject_sha256${tab}$(sha256_file "$REPO_ROOT/pyproject.toml")" \
    "uv_lock_sha256${tab}$(sha256_file "$REPO_ROOT/uv.lock")" \
    "bootstrap_sha256${tab}$(sha256_file "$TOOLING_BOOTSTRAP")" \
    >"$request_file"

  rsync -a "$tmp_dir/" "$TRANSFER_HOST:$REMOTE_TOOLING_REQUEST/"
  rm -rf -- "$tmp_dir"
}

# Install and seal one content-addressed coordinator environment by executing
# the transferred bootstrap on the network-enabled host.
install_remote_tooling() {
  remote_exec_on \
    "$CONTAINER_NETWORK_HOST" \
    "$REMOTE_TOOLING_RUNNER" \
    install
}

# Ensure the exact locked coordinator environment exists before preparing or
# operating a run. Ordinary scientific source changes do not rebuild it.
ensure_tooling() {
  local remote_state

  set_tooling_identity
  uv lock --check --project "$REPO_ROOT" >/dev/null
  sync_tooling_context

  remote_state="$(remote_tooling_state)"

  if [ "$remote_state" = "ready" ]; then
    echo "Locked Snakemake tooling is ready: $REMOTE_TOOLING_ENV"
    return 0
  fi

  if [ -x "$VPN_HELPER" ]; then
    "$VPN_HELPER" wait 60
  fi

  install_remote_tooling

  remote_state="$(remote_tooling_state)"
  if [ "$remote_state" != "ready" ]; then
    echo "Remote tooling failed provenance verification: $REMOTE_TOOLING_ENV" >&2
    return 1
  fi

  echo "Locked Snakemake tooling prepared: $REMOTE_TOOLING_ENV"
}

# -----------------------------------------------------------------------------
# Immutable run preparation
# -----------------------------------------------------------------------------

# Resolve the configured discovery link to its immutable SIF artifact.
resolve_remote_container() {
  remote_exec "$REMOTE_TOOLING_RUNNER" resolve-container "$REMOTE_CONTAINER_SIF"
}

# Independently verify an immutable scientific runtime image and its checksum.
verify_remote_container() {
  local resolved_sif="$1"

  remote_exec "$REMOTE_TOOLING_RUNNER" verify-container "$resolved_sif"
}

# Render the effective config, source snapshot, and byte inventory directly
# into the directory layout that will be copied to the remote run root.
render_run_bundle() {
  local run_id="$1"
  local root="$2"
  local resolved_sif="$3"
  local bundle_root="$4"
  local snapshot rendered dispatch_file file_list

  snapshot="$bundle_root/source"
  rendered="$bundle_root/config.yaml"
  dispatch_file="$bundle_root/.state/dispatch.tsv"
  file_list="$bundle_root/.state/source-files.bin"
  mkdir -p "$bundle_root/.state"

  uv run --no-sync python -m scripts.workflow.run_config render \
    --template "$TEMPLATE" \
    --output "$rendered" \
    --dispatch-file "$dispatch_file" \
    --file-list "$file_list" \
    --snapshot-root "$snapshot" \
    --repo-root "$REPO_ROOT" \
    --run-id "$run_id" \
    --run-root "$root" \
    --container-sif "$resolved_sif" \
    --dvc-remote "$DVC_REMOTE_PATH" \
    --dvc-cache "$DVC_CACHE_PATH" \
    --workflow-tooling-key "$TOOLING_KEY" \
    --snakemake-version "$SNAKEMAKE_EXPECTED_VERSION" \
    --snakemake-bin "$REMOTE_SNAKEMAKE_BIN"
}

# Copy the complete directory-shaped bundle into a unique hidden staging leaf.
# No final run path is visible until the transferred runner verifies the bytes.
sync_run_bundle() {
  local bundle_root="$1"
  local staging_root="$2"

  remote_exec_on "$TRANSFER_HOST" test -d "$REMOTE_WORKFLOW_RUN_ROOT"
  rsync -az "$bundle_root/" "$TRANSFER_HOST:$staging_root/"
}

# Execute the ordinary runner carried by the transferred source snapshot.
invoke_run_runner() {
  local bundle_root="$1"
  local command="$2"
  local runner="$bundle_root/source/scripts/workflow/remote_run.sh"

  remote_exec "$runner" "$command"
}

# Assemble and publish one complete immutable run. Any failure after reservation
# removes only the token-owned partial run and leaves existing runs untouched.
prepare_run() {
  local run_id="${1:-}"
  local root resolved_sif tmp_base tmp_dir bundle_root staging_root token result

  if [ -z "$run_id" ]; then
    run_id="$(generate_run_id)"
  fi

  validate_run_id "$run_id"
  root="$(run_root "$run_id")"

  ensure_tooling
  resolved_sif="$(resolve_remote_container)"
  verify_remote_container "$resolved_sif"

  # Build under HUBDIR so large source snapshots never consume the home drive.
  tmp_base="${HUBDIR:-/Volumes/HUBSSD}/tmp/epiforecaster-workflow"
  mkdir -p "$tmp_base"

  tmp_dir="$(mktemp -d "$tmp_base/run.XXXXXX")"
  bundle_root="$tmp_dir/bundle"

  if ! render_run_bundle "$run_id" "$root" "$resolved_sif" "$bundle_root"; then
    rm -rf -- "$tmp_dir"
    return 1
  fi

  # Transfer first, then let the snapshot-owned runner verify and reserve the
  # immutable run ID. The hidden staging leaf is unique to this preparation.
  token="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
  staging_root="$REMOTE_WORKFLOW_RUN_ROOT/.staging-$run_id.$token"

  if ! sync_run_bundle "$bundle_root" "$staging_root"; then
    echo "Partial staging data may require removal: $staging_root" >&2
    rm -rf -- "$tmp_dir"
    return 1
  fi

  if ! result="$(invoke_run_runner "$staging_root" publish)"; then
    rm -rf -- "$tmp_dir"
    return 1
  fi
  if [ "$result" != "published:$root" ]; then
    echo "Unexpected run publication response: $result" >&2
    rm -rf -- "$tmp_dir"
    return 1
  fi

  rm -rf -- "$tmp_dir"

  echo "Prepared immutable MN5 workflow run"
  echo "  run_id: $run_id"
  echo "  run_root: $root"
  echo "  tooling_key: $TOOLING_KEY"
  echo "  container: $resolved_sif"
}

# -----------------------------------------------------------------------------
# Workflow execution and coordinator lifecycle
# -----------------------------------------------------------------------------

# Ask the immutable runner bundled with a run to perform one control operation.
operate_run() {
  local run_id="$1"
  local command="$2"
  local root

  validate_run_id "$run_id"
  root="$(run_root "$run_id")"
  invoke_run_runner "$root" "$command"
}

# Ask Snakemake to construct the sealed run DAG without executing it.
dry_run() {
  operate_run "$1" dry-run
}

# Present the compact machine response from the immutable remote runner.
report_coordinator_launch() {
  local result="$1"
  local run_id="$2"

  case "$result" in
    submitted:*)
      echo "Submitted Snakemake coordinator job ${result#submitted:}"
      ;;
    active:*)
      result="${result#active:}"
      echo "Snakemake coordinator is already active: ${result%%:*} (${result#*:})"
      ;;
    complete)
      echo "Workflow is already complete: $run_id"
      ;;
    *)
      echo "Unexpected coordinator submission response: $result" >&2
      return 1
      ;;
  esac
}

# Submit the allocation declared by a prepared run's immutable configuration.
start_run() {
  local run_id="$1"
  local root result

  validate_run_id "$run_id"
  root="$(run_root "$run_id")"
  result="$(operate_run "$run_id" submit)"

  report_coordinator_launch "$result" "$run_id"
  echo "Run root: $root"
}

# Show authoritative scheduler state, durable completion state, and recent logs.
show_status() {
  operate_run "$1" status
}

# Unlock Snakemake metadata only when no coordinator is active for the run.
unlock_run() {
  operate_run "$1" unlock
}

# -----------------------------------------------------------------------------
# CLI routing
# -----------------------------------------------------------------------------

# Print usage and terminate with the conventional command-line error status.
usage_error() {
  usage >&2
  exit 2
}

# Parse prepare's optional run ID and template without allowing duplicates.
prepare_command() {
  local run_id=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --config)
        if [ "$#" -lt 2 ]; then
          usage_error
        fi

        TEMPLATE="$2"
        shift 2
        ;;
      *)
        if [ -n "$run_id" ]; then
          usage_error
        fi

        run_id="$1"
        shift
        ;;
    esac
  done

  if [ ! -f "$TEMPLATE" ]; then
    echo "Workflow template not found: $TEMPLATE" >&2
    exit 2
  fi

  prepare_run "$run_id"
}

# Dispatch one public subcommand after validating its positional arguments.
main() {
  local command="${1:-}"
  local run_id

  if [ "$#" -gt 0 ]; then
    shift
  fi

  case "$command" in
    tooling)
      if [ "$#" -ne 0 ]; then
        usage_error
      fi
      ensure_tooling
      ;;
    prepare)
      prepare_command "$@"
      ;;
    dry-run)
      if [ "$#" -ne 1 ]; then
        usage_error
      fi
      dry_run "$1"
      ;;
    run)
      if [ "$#" -ne 1 ]; then
        usage_error
      fi
      start_run "$1"
      ;;
    status)
      if [ "$#" -ne 1 ]; then
        usage_error
      fi
      show_status "$1"
      ;;
    unlock)
      if [ "$#" -ne 1 ]; then
        usage_error
      fi
      unlock_run "$1"
      ;;
    path)
      if [ "$#" -ne 1 ]; then
        usage_error
      fi
      validate_run_id "$1"
      run_root "$1"
      ;;
    *)
      usage_error
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
