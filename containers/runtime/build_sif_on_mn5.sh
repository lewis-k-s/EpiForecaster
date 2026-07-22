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

remote_bash_on() {
  local host="$1"
  local command="$2"
  ssh "$host" "bash -c $(shell_quote "$command")"
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

remote_state="$(remote_bash_on "$COMPUTE_HOST" "
if test -f $(shell_quote "$final_sif") && \
   test -f $(shell_quote "$final_sif.sha256") && \
   test -f $(shell_quote "$final_sif.build.txt") && \
   test -f $(shell_quote "$final_sif.inspect.json") && \
   test -f $(shell_quote "$request_root/build.done"); then
  if cd $(shell_quote "$REMOTE_CONTAINER_ROOT") && \
     sha256sum -c $(shell_quote "$(basename "$final_sif.sha256")") >/dev/null && \
     grep -Fxq $(shell_quote "dependency_key=$dependency_key") $(shell_quote "$final_sif.build.txt"); then
    ln -sfn $(shell_quote "$(basename "$final_sif")") $(shell_quote "$REMOTE_CONTAINER_ROOT/epiforecaster-runtime.sif")
    ln -sfn $(shell_quote "$(basename "$final_sif.sha256")") $(shell_quote "$REMOTE_CONTAINER_ROOT/epiforecaster-runtime.sif.sha256")
    printf 'ready'
    exit 0
  fi
  printf 'invalid'; exit 0
fi
if test -e $(shell_quote "$final_sif") || \
   test -e $(shell_quote "$final_sif.sha256") || \
   test -e $(shell_quote "$final_sif.build.txt") || \
   test -e $(shell_quote "$final_sif.inspect.json") || \
   test -e $(shell_quote "$request_root/build.done"); then
  printf 'invalid'; exit 0
fi
if test -d $(shell_quote "$request_root/dispatching"); then
  printf 'preparing'; exit 0
fi
if test -s $(shell_quote "$request_root/job.id"); then
  job_id=\$(cat $(shell_quote "$request_root/job.id"))
  state=\$(squeue -h -j \"\$job_id\" -o '%T' 2>/dev/null | head -n 1)
  if test -n \"\$state\"; then printf 'job:%s:%s' \"\$job_id\" \"\$state\"; exit 0; fi
fi
printf 'missing'")"

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
reservation_state="$(remote_bash_on "$COMPUTE_HOST" "
mkdir -p $(shell_quote "$request_root")
if mkdir $(shell_quote "$request_root/dispatching") 2>/dev/null; then
  printf '%s\n' $(shell_quote "$dispatch_token") > $(shell_quote "$request_root/dispatching/owner")
  printf reserved
else
  printf preparing
fi")"
if [ "$reservation_state" != "reserved" ]; then
  echo "Container dependencies are already being prepared: $request_root"
  exit 0
fi

reservation_active=1
cleanup_reservation() {
  if [ "$reservation_active" -eq 1 ]; then
    remote_bash_on "$COMPUTE_HOST" "
if test -f $(shell_quote "$request_root/dispatching/owner") && \
   test \"\$(cat $(shell_quote "$request_root/dispatching/owner"))\" = $(shell_quote "$dispatch_token"); then
  rm -rf -- $(shell_quote "$request_root/dispatching")
fi" >/dev/null 2>&1 || true
  fi
}
trap cleanup_reservation EXIT

remote_bash_on "$COMPUTE_HOST" \
  "mkdir -p $(shell_quote "$request_root/context") $(shell_quote "$REMOTE_CONTAINER_ROOT")"
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

fetch_command="set -euo pipefail
export BASE_IMAGE_REF=$(shell_quote "$REMOTE_CONTAINER_BASE_IMAGE_REF")
export DEPENDENCY_KEY=$(shell_quote "$dependency_key")
export UV_LOCK_SHA256=$(shell_quote "$lock_sha256")
export BUILD_ROOT=$(shell_quote "$request_root")
export CONTAINER_ROOT=$(shell_quote "$REMOTE_CONTAINER_ROOT")
export UV_CACHE_ROOT=$(shell_quote "$REMOTE_CONTAINER_DEP_CACHE")
bash $(shell_quote "$request_root/fetch_runtime_deps_on_wifi.sh")"
remote_bash_on "$CONTAINER_NETWORK_HOST" "$fetch_command"

base_sif="$(remote_bash_on "$COMPUTE_HOST" "cat $(shell_quote "$request_root/base-sif.path")")"
prefix_root="$request_root/prefix"
submit_command="set -euo pipefail
test -f $(shell_quote "$request_root/deps.ready")
exec 9>$(shell_quote "$request_root/submit.lock")
flock 9
if test -s $(shell_quote "$request_root/job.id"); then
  existing=\$(cat $(shell_quote "$request_root/job.id"))
  state=\$(squeue -h -j \"\$existing\" -o '%T' 2>/dev/null | head -n 1)
  if test -n \"\$state\"; then printf 'existing:%s:%s' \"\$existing\" \"\$state\"; exit 0; fi
fi
job_id=\$(sbatch --parsable \\
  --chdir=$(shell_quote "$request_root") \\
  --output=$(shell_quote "$request_root/build-%j.out") \\
  --error=$(shell_quote "$request_root/build-%j.err") \\
  --export=ALL,BASE_SIF=$(shell_quote "$base_sif"),BASE_IMAGE_REF=$(shell_quote "$REMOTE_CONTAINER_BASE_IMAGE_REF"),PREFIX_ROOT=$(shell_quote "$prefix_root"),DEPENDENCY_KEY=$(shell_quote "$dependency_key"),UV_LOCK_SHA256=$(shell_quote "$lock_sha256"),CONTAINER_ROOT=$(shell_quote "$REMOTE_CONTAINER_ROOT"),BUILD_ROOT=$(shell_quote "$request_root") \\
  $(shell_quote "$request_root/build_sif.sbatch"))
job_id=\${job_id%%;*}
printf '%s\n' \"\$job_id\" > $(shell_quote "$request_root/job.id")
printf 'submitted:%s' \"\$job_id\""
submission="$(remote_bash_on "$COMPUTE_HOST" "$submit_command")"
remote_bash_on "$COMPUTE_HOST" "rm -rf -- $(shell_quote "$request_root/dispatching")"
reservation_active=0
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
