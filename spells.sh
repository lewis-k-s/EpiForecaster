#!/usr/bin/env bash

# Source this file to get project helpers in your shell.
# Example: source spells.sh

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SPELLS_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SPELLS_SOURCE="${(%):-%x}"
else
  SPELLS_SOURCE="$0"
fi

SPELLS_DIR="$(cd -- "$(dirname -- "${SPELLS_SOURCE}")" && pwd)"
PROJECT_HOME="${SPELLS_DIR}"

# --- general project utils ---

function h() {
  cd "${PROJECT_HOME}" || return 1
}

function cleanup_logs() {
  local logs_dir="${PROJECT_HOME}/logs"
  local days="${1:-14}"

  if [[ ! -d "${logs_dir}" ]]; then
    echo "logs directory not found: ${logs_dir}" >&2
    return 1
  fi

  local count
  count=$(find "${logs_dir}" -type f -mtime +"${days}" | wc -l | tr -d ' ')

  if [[ "${count}" -eq 0 ]]; then
    echo "no files older than ${days} days in ${logs_dir}"
    return 0
  fi

  echo "removing ${count} files older than ${days} days from ${logs_dir}"
  find "${logs_dir}" -type f -mtime +"${days}" -delete
}

# --- paper helpers (delegated) ---

source "${SPELLS_DIR}/tex/spells.sh"
