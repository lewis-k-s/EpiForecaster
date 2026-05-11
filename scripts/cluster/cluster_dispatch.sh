#!/bin/bash

# Generic Slurm cluster dispatch CLI.
# Config-driven: source a .cluster/<name>.conf to set cluster identity,
# mode registry, resource specs, and sync commands.

set -euo pipefail

# --- Repo root (for resolving relative config paths when CWD != repo) ---
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# --- Config loading ---
CONF="${CLUSTER_CONF:-.cluster/dispatch.conf}"

# If CONF is relative and not found from CWD, resolve against repo root
if [ ! -f "$CONF" ] && [[ "$CONF" != /* ]]; then
  CONF="$REPO_ROOT/$CONF"
fi

if [ ! -f "$CONF" ]; then
  echo "Config not found: $CONF" >&2
  echo "Hint: run 'bash syncto_mn5.sh' to sync the cluster config, or set CLUSTER_CONF to an absolute path." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONF"

# Defaults for optional config values
REMOTE_BASE="${REMOTE_BASE:-$PWD}"
LOG_DIR="${LOG_DIR:-logs}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
MODULE_SETUP="${MODULE_SETUP:-}"
SHELL_MODES="${SHELL_MODES:-}"

# --- Helpers ---

usage() {
  cat <<EOF
Usage:
  $(basename "$0") submit <mode> [sbatch-args...]
  $(basename "$0") alloc  <mode> [salloc-args...]
  $(basename "$0") run    <mode> [args...]
  $(basename "$0") status <job-id>
  $(basename "$0") logs   <job-id> [task-id]
  $(basename "$0") tail   <job-id> [task-id] [lines]

Config: $CONF
Cluster: ${CLUSTER_NAME:-<unset>}

Available modes:
EOF
  # List all MODE_* entries from config
  local varname
  for varname in $(compgen -A variable | grep '^MODE_'); do
    local mode="${varname#MODE_}"
    local val="${!varname}"
    printf "  %-20s %s\n" "$mode" "$val"
  done
}

# Resolve a mode name to its script path via config MODE_<mode> variable.
resolve_mode() {
  local mode="$1"
  local ref="MODE_$mode"
  local script="${!ref:-}"

  if [ -z "$script" ]; then
    echo "Unknown mode: $mode" >&2
    echo "Available: $(compgen -A variable | grep '^MODE_' | sed 's/^MODE_//' | tr '\n' ' ')" >&2
    return 1
  fi

  printf '%s\n' "$script"
}

# Resolve per-mode resource spec for alloc.
resolve_resources() {
  local mode="$1"
  local ref="RESOURCES_$mode"
  printf '%s\n' "${!ref:-}"
}

# Check if a mode is a shell-dispatch mode (invoked directly, not via sbatch).
is_shell_mode() {
  local mode="$1"
  local m
  for m in $SHELL_MODES; do
    if [ "$m" = "$mode" ]; then
      return 0
    fi
  done
  return 1
}

repo_cd() {
  cd "$REMOTE_BASE"
  export PROJECT_ROOT="${PROJECT_ROOT:-$REMOTE_BASE}"
}

# --- Commands ---

submit_job() {
  local mode="${1:-}"
  local script_path

  if [ -z "$mode" ]; then
    usage >&2
    return 2
  fi
  shift

  repo_cd
  script_path="$(resolve_mode "$mode")"

  if is_shell_mode "$mode"; then
    "$script_path" "$@"
  else
    sbatch --parsable "$@" "$script_path"
  fi
}

alloc_session() {
  local mode="${1:-}"
  local resources

  if [ -z "$mode" ]; then
    usage >&2
    return 2
  fi
  shift

  repo_cd
  resources="$(resolve_resources "$mode")"

  if [ -z "$resources" ]; then
    echo "No resource spec configured for mode: $mode (set RESOURCES_$mode in config)" >&2
    return 1
  fi

  local account_arg=""
  if [ -n "$SLURM_ACCOUNT" ]; then
    account_arg="--account=$SLURM_ACCOUNT"
  fi

  echo "Allocating session: salloc $resources $account_arg $*"
  salloc $resources $account_arg "$@" bash
}

run_job() {
  local mode="${1:-}"
  local script_path

  if [ -z "$mode" ]; then
    usage >&2
    return 2
  fi
  shift

  repo_cd
  script_path="$(resolve_mode "$mode")"

  if [ -n "$MODULE_SETUP" ] && [ -f "$MODULE_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$MODULE_SETUP"
  fi

  exec bash "$script_path" "$@"
}

show_status() {
  local job_id="${1:-}"

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  repo_cd
  squeue --job "$job_id" --format="%.18i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %R" || true
  sacct --jobs "$job_id" --format=JobID,JobName%32,State,ExitCode,Elapsed,Timelimit,NodeList --parsable2 || true
}

resolve_log_files() {
  local job_id="$1"
  local task_id="${2:-}"
  local -a patterns=()

  if [ -n "$task_id" ]; then
    patterns+=("$LOG_DIR/slurm-*-${job_id}_${task_id}.out")
    patterns+=("$LOG_DIR/slurm-*-${job_id}_${task_id}.err")
  else
    patterns+=("$LOG_DIR/slurm-*-${job_id}.out")
    patterns+=("$LOG_DIR/slurm-*-${job_id}.err")
    patterns+=("$LOG_DIR/slurm-*-${job_id}_*.out")
    patterns+=("$LOG_DIR/slurm-*-${job_id}_*.err")
  fi

  for pattern in "${patterns[@]}"; do
    compgen -G "$pattern" || true
  done | sort -u
}

show_logs() {
  local job_id="${1:-}"
  local task_id="${2:-}"
  local -a files=()
  local file

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  repo_cd
  while IFS= read -r file; do
    files+=("$file")
  done < <(resolve_log_files "$job_id" "$task_id")

  if [ "${#files[@]}" -eq 0 ]; then
    echo "No Slurm logs found for job ${job_id}${task_id:+ task ${task_id}} under $REMOTE_BASE/$LOG_DIR" >&2
    return 1
  fi

  printf '%s\n' "${files[@]}"
}

tail_logs() {
  local job_id="${1:-}"
  local maybe_task="${2:-}"
  local maybe_lines="${3:-}"
  local task_id=""
  local lines="120"
  local file
  local -a files=()

  if [ -z "$job_id" ]; then
    usage >&2
    return 2
  fi

  if [ -n "$maybe_task" ]; then
    task_id="$maybe_task"
  fi
  if [ -n "$maybe_lines" ]; then
    lines="$maybe_lines"
  elif [ -n "$maybe_task" ] && [[ "$maybe_task" =~ ^[0-9]+$ ]]; then
    if ! compgen -G "$REMOTE_BASE/$LOG_DIR/slurm-*-${job_id}_${maybe_task}.out" >/dev/null \
      && ! compgen -G "$REMOTE_BASE/$LOG_DIR/slurm-*-${job_id}_${maybe_task}.err" >/dev/null; then
      task_id=""
      lines="$maybe_task"
    fi
  fi

  repo_cd
  while IFS= read -r file; do
    files+=("$file")
  done < <(resolve_log_files "$job_id" "$task_id")

  if [ "${#files[@]}" -eq 0 ]; then
    echo "No Slurm logs found for job ${job_id}${task_id:+ task ${task_id}} under $REMOTE_BASE/$LOG_DIR" >&2
    return 1
  fi

  tail -n "$lines" -f "${files[@]}"
}

# --- Main dispatch ---

main() {
  local command="${1:-}"

  case "$command" in
    submit)
      shift
      submit_job "$@"
      ;;
    alloc)
      shift
      alloc_session "$@"
      ;;
    run)
      shift
      run_job "$@"
      ;;
    status)
      shift
      show_status "$@"
      ;;
    logs)
      shift
      show_logs "$@"
      ;;
    tail)
      shift
      tail_logs "$@"
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
