#!/bin/bash

# Materialize locked dependencies on the network-enabled MN5 login, then
# submit an offline native-SIF build and GPU validation through Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONF="${CLUSTER_CONF:-$REPO_ROOT/.cluster/mn5.conf}"
# shellcheck disable=SC1090
source "$CONF"

COMPUTE_HOST="${COMPUTE_HOST:?COMPUTE_HOST must be configured}"
TRANSFER_HOST="${TRANSFER_HOST:?TRANSFER_HOST must be configured}"
CONTAINER_NETWORK_HOST="${CONTAINER_NETWORK_HOST:-wifi-acc}"
REMOTE_CONTAINER_ROOT="${REMOTE_CONTAINER_ROOT:?REMOTE_CONTAINER_ROOT must be configured}"
REMOTE_CONTAINER_BUILD_ROOT="${REMOTE_CONTAINER_BUILD_ROOT:-$REMOTE_WORKFLOW_PROJECT_ROOT/container-builds}"
REMOTE_CONTAINER_DEP_CACHE="${REMOTE_CONTAINER_DEP_CACHE:-$REMOTE_CONTAINER_BUILD_ROOT/cache}"
REMOTE_CONTAINER_BASE_IMAGE_REF="${REMOTE_CONTAINER_BASE_IMAGE_REF:?REMOTE_CONTAINER_BASE_IMAGE_REF must be configured}"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_runtime_deps_on_wifi.sh"
SBATCH_SCRIPT="$SCRIPT_DIR/build_sif.sbatch"
VPN_HELPER="${WORKVPN_HELPER:-/Users/lewis/.codex/skills/workvpn/vpn}"

# Reservation state is global because the EXIT trap runs after command failure.
RESERVATION_ACTIVE=0
RESERVATION_REQUEST_ROOT=""
RESERVATION_TOKEN=""

# -----------------------------------------------------------------------------
# CLI and local helpers
# -----------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

The dependency identity is derived from the pinned base image, uv.lock, and
the fetch/build scripts. Dependencies are synchronized with
--frozen on $CONTAINER_NETWORK_HOST before an offline MN5 Slurm build starts.
EOF
}

shell_quote() {
  printf '%q' "$1"
}

remote_function_on() {
  local host="$1"
  local function_name="$2"
  shift 2
  local remote_command="bash -s --"
  local argument

  if ! declare -f "$function_name" >/dev/null; then
    echo "Unknown remote function: $function_name" >&2
    return 2
  fi

  for argument in "$@"; do
    remote_command+=" $(shell_quote "$argument")"
  done

  {
    declare -f "$function_name"
    printf '\n%s "$@"\n' "$function_name"
  } | ssh "$host" "$remote_command"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    shasum -a 256 | awk '{print $1}'
  fi
}

# -----------------------------------------------------------------------------
# Remote programs
# -----------------------------------------------------------------------------

# Classify the current immutable artifact/request state without mutating a
# partial artifact set. A verified artifact refreshes the discovery symlinks.
remote_container_state() {
  local final_sif="$1"
  local request_root="$2"
  local container_root="$3"
  local dependency_key="$4"
  local job_id state

  if [ -f "$final_sif" ] \
    && [ -f "$final_sif.sha256" ] \
    && [ -f "$final_sif.build.txt" ] \
    && [ -f "$final_sif.inspect.json" ] \
    && [ -f "$request_root/build.done" ]; then
    if cd "$container_root" \
      && sha256sum -c "$(basename "$final_sif.sha256")" >/dev/null \
      && grep -Fxq "dependency_key=$dependency_key" "$final_sif.build.txt"; then
      ln -sfn "$(basename "$final_sif")" \
        "$container_root/epiforecaster-runtime.sif"
      ln -sfn "$(basename "$final_sif.sha256")" \
        "$container_root/epiforecaster-runtime.sif.sha256"
      printf ready
      return 0
    fi

    printf invalid
    return 0
  fi

  if [ -e "$final_sif" ] \
    || [ -e "$final_sif.sha256" ] \
    || [ -e "$final_sif.build.txt" ] \
    || [ -e "$final_sif.inspect.json" ] \
    || [ -e "$request_root/build.done" ]; then
    printf invalid
    return 0
  fi

  if [ -d "$request_root/dispatching" ]; then
    printf preparing
    return 0
  fi

  if [ -s "$request_root/job.id" ]; then
    job_id="$(cat "$request_root/job.id")"
    state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "$state" ]; then
      printf 'job:%s:%s' "$job_id" "$state"
      return 0
    fi
  fi

  printf missing
}

# Reserve dependency preparation atomically and record its owner token.
remote_reserve_dispatch() {
  local request_root="$1"
  local token="$2"

  mkdir -p "$request_root"
  if mkdir "$request_root/dispatching" 2>/dev/null; then
    printf '%s\n' "$token" >"$request_root/dispatching/owner"
    printf reserved
  else
    printf preparing
  fi
}

# Remove a preparation reservation only when this dispatcher still owns it.
remote_cleanup_dispatch() {
  local request_root="$1"
  local token="$2"
  local owner="$request_root/dispatching/owner"

  if [ -f "$owner" ] && [ "$(cat "$owner")" = "$token" ]; then
    rm -rf -- "$request_root/dispatching"
  fi
}

# Create one or more shared directories on the compute-visible filesystem.
remote_create_directories() {
  mkdir -p "$@"
}

# Run dependency materialization on the network-enabled login node.
remote_fetch_dependencies() {
  set -euo pipefail

  local fetch_script="$1"
  export BASE_IMAGE_REF="$2"
  export DEPENDENCY_KEY="$3"
  export UV_LOCK_SHA256="$4"
  export BUILD_ROOT="$5"
  export CONTAINER_ROOT="$6"
  export UV_CACHE_ROOT="$7"

  bash "$fetch_script"
}

# Print one remote file whose contents form part of the submission contract.
remote_read_file() {
  cat "$1"
}

# Submit at most one active offline build for this dependency identity.
remote_submit_build() {
  set -euo pipefail

  local request_root="$1"
  local base_sif="$2"
  local base_image_ref="$3"
  local prefix_root="$4"
  local dependency_key="$5"
  local lock_sha256="$6"
  local container_root="$7"
  local sbatch_script="$8"
  local existing state job_id

  [ -f "$request_root/deps.ready" ]
  exec 9>"$request_root/submit.lock"
  flock 9

  if [ -s "$request_root/job.id" ]; then
    existing="$(cat "$request_root/job.id")"
    state="$(squeue -h -j "$existing" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [ -n "$state" ]; then
      printf 'existing:%s:%s' "$existing" "$state"
      return 0
    fi
  fi

  job_id="$(
    sbatch --parsable \
      --chdir="$request_root" \
      --output="$request_root/build-%j.out" \
      --error="$request_root/build-%j.err" \
      --export="ALL,BASE_SIF=$base_sif,BASE_IMAGE_REF=$base_image_ref,PREFIX_ROOT=$prefix_root,DEPENDENCY_KEY=$dependency_key,UV_LOCK_SHA256=$lock_sha256,CONTAINER_ROOT=$container_root,BUILD_ROOT=$request_root" \
      "$sbatch_script"
  )"
  job_id="${job_id%%;*}"
  printf '%s\n' "$job_id" >"$request_root/job.id"
  printf 'submitted:%s' "$job_id"
}

# Best-effort EXIT cleanup for a token-owned preparation reservation.
cleanup_reservation() {
  if [ "$RESERVATION_ACTIVE" -eq 1 ]; then
    remote_function_on \
      "$COMPUTE_HOST" \
      remote_cleanup_dispatch \
      "$RESERVATION_REQUEST_ROOT" \
      "$RESERVATION_TOKEN" \
      >/dev/null 2>&1 || true
  fi
}

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=1
  shift
fi
if [ "$#" -ne 0 ]; then
  usage >&2
  exit 2
fi

for required in \
  "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/uv.lock" \
  "$FETCH_SCRIPT" \
  "$SBATCH_SCRIPT"; do
  test -f "$required"
done
uv lock --check --project "$REPO_ROOT" >/dev/null

dependency_key="$({
  printf 'base_image=%s\n' "$REMOTE_CONTAINER_BASE_IMAGE_REF"
  for input in \
    "$REPO_ROOT/uv.lock" \
    "$FETCH_SCRIPT" \
    "$SBATCH_SCRIPT"; do
    printf '%s=%s\n' "${input#"$REPO_ROOT/"}" "$(sha256_file "$input")"
  done
} | sha256_stream)"
build_key="${dependency_key:0:20}"
lock_sha256="$(sha256_file "$REPO_ROOT/uv.lock")"
request_root="$REMOTE_CONTAINER_BUILD_ROOT/$dependency_key"
final_sif="$REMOTE_CONTAINER_ROOT/epiforecaster-runtime-$build_key.sif"

if [ "$dry_run" -eq 1 ]; then
  echo "Dry run; no transfer, VPN, SSH, dependency sync, or Slurm command ran."
  echo "  dependency key: $dependency_key"
  echo "  base image: $REMOTE_CONTAINER_BASE_IMAGE_REF"
  echo "  network fetch host: $CONTAINER_NETWORK_HOST"
  echo "  request: $request_root"
  echo "  output: $final_sif"
  exit 0
fi

remote_state="$(
  remote_function_on \
    "$COMPUTE_HOST" \
    remote_container_state \
    "$final_sif" \
    "$request_root" \
    "$REMOTE_CONTAINER_ROOT" \
    "$dependency_key"
)"

case "$remote_state" in
  ready)
    echo "Reusable SIF already exists: $final_sif"
    exit 0
    ;;
  invalid)
    echo "Existing SIF artifact set is incomplete: $final_sif" >&2
    exit 1
    ;;
  job:*)
    job_id="${remote_state#job:}"
    echo "Container build is already active: ${job_id%%:*} (${job_id#*:})"
    echo "  request: $request_root"
    exit 0
    ;;
  preparing)
    echo "Container dependencies are already being prepared: $request_root"
    exit 0
    ;;
esac

dispatch_token="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
reservation_state="$(
  remote_function_on \
    "$COMPUTE_HOST" \
    remote_reserve_dispatch \
    "$request_root" \
    "$dispatch_token"
)"
if [ "$reservation_state" != "reserved" ]; then
  echo "Container dependencies are already being prepared: $request_root"
  exit 0
fi

RESERVATION_ACTIVE=1
RESERVATION_REQUEST_ROOT="$request_root"
RESERVATION_TOKEN="$dispatch_token"
trap cleanup_reservation EXIT

remote_function_on \
  "$COMPUTE_HOST" \
  remote_create_directories \
  "$request_root/context" \
  "$REMOTE_CONTAINER_ROOT"
rsync -a "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock" \
  "$TRANSFER_HOST:$request_root/context/"
if [ -f "$REPO_ROOT/README.md" ]; then
  rsync -a "$REPO_ROOT/README.md" "$TRANSFER_HOST:$request_root/context/"
fi
rsync -a "$FETCH_SCRIPT" "$TRANSFER_HOST:$request_root/fetch_runtime_deps_on_wifi.sh"
rsync -a "$SBATCH_SCRIPT" "$TRANSFER_HOST:$request_root/build_sif.sbatch"

if [ -x "$VPN_HELPER" ]; then
  "$VPN_HELPER" wait 60
fi

remote_function_on \
  "$CONTAINER_NETWORK_HOST" \
  remote_fetch_dependencies \
  "$request_root/fetch_runtime_deps_on_wifi.sh" \
  "$REMOTE_CONTAINER_BASE_IMAGE_REF" \
  "$dependency_key" \
  "$lock_sha256" \
  "$request_root" \
  "$REMOTE_CONTAINER_ROOT" \
  "$REMOTE_CONTAINER_DEP_CACHE"

base_sif="$(
  remote_function_on \
    "$COMPUTE_HOST" \
    remote_read_file \
    "$request_root/base-sif.path"
)"
prefix_root="$request_root/prefix"
submission="$(
  remote_function_on \
    "$COMPUTE_HOST" \
    remote_submit_build \
    "$request_root" \
    "$base_sif" \
    "$REMOTE_CONTAINER_BASE_IMAGE_REF" \
    "$prefix_root" \
    "$dependency_key" \
    "$lock_sha256" \
    "$REMOTE_CONTAINER_ROOT" \
    "$request_root/build_sif.sbatch"
)"
remote_function_on \
  "$COMPUTE_HOST" \
  remote_cleanup_dispatch \
  "$request_root" \
  "$dispatch_token"
RESERVATION_ACTIVE=0
trap - EXIT

case "$submission" in
  submitted:*)
    job_id="${submission#submitted:}"
    echo "Submitted MN5 container build job $job_id"
    ;;
  existing:*)
    job_id="${submission#existing:}"
    echo "Container build is already active: ${job_id%%:*} (${job_id#*:})"
    job_id="${job_id%%:*}"
    ;;
  *)
    echo "Unexpected container submission response: $submission" >&2
    exit 1
    ;;
esac
echo "  dependency key: $dependency_key"
echo "  locked dependencies: $request_root/prefix"
echo "  expected SIF: $final_sif"
echo "  status: ssh $COMPUTE_HOST 'squeue -j $job_id'"
