#!/bin/bash

# Prepare and dispatch isolated MN5 worktree snapshots backed by the shared
# DVC cache and NFS remote. This script runs locally; rsync uses the transfer
# node and all DVC/Slurm operations use the login node.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CONF="${CLUSTER_CONF:-$REPO_ROOT/.cluster/mn5.conf}"

if [ ! -f "$CONF" ]; then
  echo "Cluster config not found: $CONF" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONF"

REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:?REMOTE_RUN_ROOT must be configured}"
DVC_REMOTE_PATH="${DVC_REMOTE_PATH:?DVC_REMOTE_PATH must be configured}"
DVC_CACHE_PATH="${DVC_CACHE_PATH:?DVC_CACHE_PATH must be configured}"
REMOTE_DVC_BIN="${REMOTE_DVC_BIN:?REMOTE_DVC_BIN must be configured}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:?REMOTE_PYTHON_BIN must be configured}"
COMPUTE_HOST="${COMPUTE_HOST:?COMPUTE_HOST must be configured}"
TRANSFER_HOST="${TRANSFER_HOST:?TRANSFER_HOST must be configured}"

DVC_TARGETS=(
  data/files/mobility.zarr.dvc
  data/processed/real_with_id.zarr.dvc
)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") prepare [run-id]
  $(basename "$0") submit <run-id> <mode> [sbatch-args...]
  $(basename "$0") status <run-id> <job-id>
  $(basename "$0") verify <left-run-id> <right-run-id>
  $(basename "$0") path <run-id>

prepare creates a unique remote project, syncs the current worktree, configures
the shared DVC cache, and pulls the pilot raw and processed Zarr datasets.
EOF
}

validate_run_id() {
  local run_id="$1"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run id '$run_id'; use letters, digits, dot, underscore, or dash." >&2
    return 2
  fi
}

generate_run_id() {
  local branch short_sha timestamp
  branch="$(git branch --show-current)"
  branch="${branch:-detached}"
  branch="$(printf '%s' "$branch" | tr '/ :' '---' | tr -cd 'A-Za-z0-9._-')"
  short_sha="$(git rev-parse --short=10 HEAD)"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s-%s-%s-%s\n' "$branch" "$short_sha" "$timestamp" "$$-$RANDOM"
}

remote_project() {
  local run_id="$1"
  printf '%s/%s/project\n' "$REMOTE_RUN_ROOT" "$run_id"
}

shell_quote() {
  printf '%q' "$1"
}

remote_bash() {
  local script="$1"
  ssh "$COMPUTE_HOST" "bash -lc $(shell_quote "$script")"
}

dvc_md5() {
  local dvc_file="$1"
  awk '
    $1 == "-" && $2 == "md5:" { print $3; exit }
    $1 == "md5:" { print $2; exit }
  ' "$dvc_file"
}

prepare_run() {
  local run_id="${1:-}"
  local project git_sha git_branch dirty_state remote_script

  if [ -z "$run_id" ]; then
    run_id="$(generate_run_id)"
  fi
  validate_run_id "$run_id"
  project="$(remote_project "$run_id")"

  git_sha="$(git rev-parse HEAD)"
  git_branch="$(git branch --show-current)"
  git_branch="${git_branch:-detached}"
  if [ -z "$(git status --porcelain --untracked-files=normal)" ]; then
    dirty_state="clean"
  else
    dirty_state="dirty"
  fi

  remote_bash "mkdir -p $(shell_quote "$project")"

  rsync -az --delete \
    --exclude-from="$REPO_ROOT/.gitignore" \
    --exclude=.git \
    --exclude=.venv/ \
    --exclude=.cache/ \
    --exclude=.dvc/cache/ \
    --exclude=.dvc/tmp/ \
    --exclude=outputs/ \
    --exclude=logs/ \
    "$REPO_ROOT/" "$TRANSFER_HOST:$project/"

  remote_script="set -euo pipefail
test -x $(shell_quote "$REMOTE_DVC_BIN")
mkdir -p $(shell_quote "$DVC_REMOTE_PATH") $(shell_quote "$DVC_CACHE_PATH") $(shell_quote "$project/logs") $(shell_quote "$project/outputs")
cd $(shell_quote "$project")
git init -q
$(shell_quote "$REMOTE_DVC_BIN") config --local cache.dir $(shell_quote "$DVC_CACHE_PATH")
$(shell_quote "$REMOTE_DVC_BIN") config --local cache.shared group
$(shell_quote "$REMOTE_DVC_BIN") config --local cache.type symlink
$(shell_quote "$REMOTE_DVC_BIN") remote modify --local mn5-nfs url $(shell_quote "$DVC_REMOTE_PATH")
touch $(shell_quote "$DVC_CACHE_PATH/.dvc-pull.lock")
chmod 0664 $(shell_quote "$DVC_CACHE_PATH/.dvc-pull.lock")
flock -x $(shell_quote "$DVC_CACHE_PATH/.dvc-pull.lock") $(shell_quote "$REMOTE_DVC_BIN") pull ${DVC_TARGETS[*]}
$(shell_quote "$REMOTE_DVC_BIN") status ${DVC_TARGETS[*]}
printf '%s\n' \
  $(shell_quote "run_id=$run_id") \
  $(shell_quote "project_root=$project") \
  $(shell_quote "git_sha=$git_sha") \
  $(shell_quote "git_branch=$git_branch") \
  $(shell_quote "worktree_state=$dirty_state") \
  $(shell_quote "raw_dvc_md5=$(dvc_md5 "$REPO_ROOT/data/files/mobility.zarr.dvc")") \
  $(shell_quote "processed_dvc_md5=$(dvc_md5 "$REPO_ROOT/data/processed/real_with_id.zarr.dvc")") \
  > .mn5-run.env"
  remote_bash "$remote_script"

  echo "Prepared MN5 run"
  echo "  run_id: $run_id"
  echo "  project: $project"
  echo "  worktree: $dirty_state at $git_sha"
}

submit_run() {
  local run_id="$1"
  local mode="$2"
  shift 2
  local project command arg

  validate_run_id "$run_id"
  project="$(remote_project "$run_id")"
  command="set -euo pipefail
test -f $(shell_quote "$project/.mn5-run.env")
cd $(shell_quote "$project")
RUN_ID=$(shell_quote "$run_id") \\
REMOTE_BASE=$(shell_quote "$project") \\
PROJECT_ROOT=$(shell_quote "$project") \\
PYTHON_BIN=$(shell_quote "$REMOTE_PYTHON_BIN") \\
DVC_BIN=$(shell_quote "$REMOTE_DVC_BIN") \\
DVC_REMOTE_PATH=$(shell_quote "$DVC_REMOTE_PATH") \\
DVC_CACHE_PATH=$(shell_quote "$DVC_CACHE_PATH") \\
$(shell_quote "$project/scripts/cluster/cluster_dispatch.sh") submit $(shell_quote "$mode")"
  for arg in "$@"; do
    command+=" $(shell_quote "$arg")"
  done
  remote_bash "$command"
}

status_run() {
  local run_id="$1"
  local job_id="$2"
  local project
  validate_run_id "$run_id"
  project="$(remote_project "$run_id")"
  remote_bash "REMOTE_BASE=$(shell_quote "$project") $(shell_quote "$project/scripts/cluster/cluster_dispatch.sh") status $(shell_quote "$job_id")"
}

verify_runs() {
  local left_id="$1"
  local right_id="$2"
  local left_project right_project raw_md5 processed_md5
  validate_run_id "$left_id"
  validate_run_id "$right_id"
  if [ "$left_id" = "$right_id" ]; then
    echo "Isolation verification requires two different run ids." >&2
    return 2
  fi

  left_project="$(remote_project "$left_id")"
  right_project="$(remote_project "$right_id")"
  raw_md5="$(dvc_md5 "$REPO_ROOT/data/files/mobility.zarr.dvc")"
  processed_md5="$(dvc_md5 "$REPO_ROOT/data/processed/real_with_id.zarr.dvc")"

  remote_bash "$(shell_quote "$REMOTE_PYTHON_BIN") \
    $(shell_quote "$left_project/scripts/cluster/verify_dvc_isolation.py") \
    --left-root $(shell_quote "$left_project") \
    --right-root $(shell_quote "$right_project") \
    --expected-raw-md5 $(shell_quote "$raw_md5") \
    --expected-processed-md5 $(shell_quote "$processed_md5")"
}

main() {
  local command="${1:-}"
  case "$command" in
    prepare)
      shift
      prepare_run "${1:-}"
      ;;
    submit)
      if [ "$#" -lt 3 ]; then
        usage >&2
        return 2
      fi
      shift
      submit_run "$@"
      ;;
    status)
      if [ "$#" -ne 3 ]; then
        usage >&2
        return 2
      fi
      shift
      status_run "$@"
      ;;
    verify)
      if [ "$#" -ne 3 ]; then
        usage >&2
        return 2
      fi
      shift
      verify_runs "$@"
      ;;
    path)
      if [ "$#" -ne 2 ]; then
        usage >&2
        return 2
      fi
      validate_run_id "$2"
      remote_project "$2"
      ;;
    -h | --help | help)
      usage
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
