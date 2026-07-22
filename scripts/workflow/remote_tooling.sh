#!/bin/bash

# Prepare and inspect the content-addressed host-side Snakemake environment.
# This script is copied into each tooling request and executed from there, so
# installation behavior is visible, versioned, and never streamed with SSH.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REQUEST_ROOT="$(dirname "$SCRIPT_PATH")"
REQUEST_FILE="$REQUEST_ROOT/request.tsv"
CONTEXT="$REQUEST_ROOT/context"
READY_MARKER="$REQUEST_ROOT/ready"

TOOLING_KEY=""
ENVIRONMENT=""
CACHE_ROOT=""
PYTHON_ROOT=""
EXPECTED_VERSION=""
PYPROJECT_SHA256=""
UV_LOCK_SHA256=""
BOOTSTRAP_SHA256=""

usage() {
  cat <<EOF
Usage: $(basename "$0") probe|install|resolve-container PATH|verify-container PATH
EOF
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

load_request() {
  local field value extra

  test -f "$REQUEST_FILE"
  while IFS=$'\t' read -r field value extra; do
    if [ -z "$field" ] || [ -z "$value" ] || [ -n "$extra" ]; then
      echo "Malformed tooling request line in $REQUEST_FILE" >&2
      return 1
    fi
    case "$field" in
      tooling_key) TOOLING_KEY="$value" ;;
      environment) ENVIRONMENT="$value" ;;
      cache_root) CACHE_ROOT="$value" ;;
      python_root) PYTHON_ROOT="$value" ;;
      expected_version) EXPECTED_VERSION="$value" ;;
      pyproject_sha256) PYPROJECT_SHA256="$value" ;;
      uv_lock_sha256) UV_LOCK_SHA256="$value" ;;
      bootstrap_sha256) BOOTSTRAP_SHA256="$value" ;;
      *)
        echo "Unknown tooling request field: $field" >&2
        return 1
        ;;
    esac
  done <"$REQUEST_FILE"

  test -n "$TOOLING_KEY"
  test -n "$ENVIRONMENT"
  test -n "$CACHE_ROOT"
  test -n "$PYTHON_ROOT"
  test -n "$EXPECTED_VERSION"
  test -n "$PYPROJECT_SHA256"
  test -n "$UV_LOCK_SHA256"
  test -n "$BOOTSTRAP_SHA256"
  test "$(sha256_file "$CONTEXT/pyproject.toml")" = "$PYPROJECT_SHA256"
  test "$(sha256_file "$CONTEXT/uv.lock")" = "$UV_LOCK_SHA256"
  test "$(sha256_file "$SCRIPT_PATH")" = "$BOOTSTRAP_SHA256"
}

snakemake_probe() {
  "$ENVIRONMENT/bin/python" -c \
    "import importlib.metadata as m; d=m.distribution('snakemake'); print(d.version); print(d.locate_file(''))"
}

tooling_is_ready() {
  local installed installed_version installed_location

  if [ ! -x "$ENVIRONMENT/bin/snakemake" ] || [ ! -f "$READY_MARKER" ]; then
    return 1
  fi
  if ! installed="$(snakemake_probe 2>/dev/null)"; then
    return 1
  fi
  installed_version="$(printf '%s\n' "$installed" | sed -n '1p')"
  installed_location="$(printf '%s\n' "$installed" | sed -n '2p')"

  [ "$installed_version" = "$EXPECTED_VERSION" ] \
    && grep -Fxq "tooling_key=$TOOLING_KEY" "$READY_MARKER" \
    && grep -Fxq "snakemake_version=$EXPECTED_VERSION" "$READY_MARKER" \
    && grep -Fxq "snakemake_location=$installed_location" "$READY_MARKER" \
    && grep -Fxq "bootstrap_sha256=$BOOTSTRAP_SHA256" "$READY_MARKER"
}

probe_tooling() {
  if tooling_is_ready; then
    printf ready
  else
    printf missing
  fi
}

install_tooling() {
  local installed installed_version installed_location temporary_marker

  export PATH="$HOME/.local/bin:$PATH"
  mkdir -p "$ENVIRONMENT" "$CACHE_ROOT" "$PYTHON_ROOT"
  exec 9>"$REQUEST_ROOT/sync.lock"
  flock 9

  if tooling_is_ready; then
    return 0
  fi

  chmod -R u+w "$ENVIRONMENT"
  command -v uv >/dev/null
  UV_PROJECT_ENVIRONMENT="$ENVIRONMENT" \
    UV_CACHE_DIR="$CACHE_ROOT" \
    UV_PYTHON_INSTALL_DIR="$PYTHON_ROOT" \
    UV_LINK_MODE=copy \
    uv sync \
    --project "$CONTEXT" \
    --frozen \
    --only-group workflow \
    --no-install-project \
    --python 3.11

  PYTHONDONTWRITEBYTECODE=1 "$ENVIRONMENT/bin/snakemake" --version
  installed="$(snakemake_probe)"
  installed_version="$(printf '%s\n' "$installed" | sed -n '1p')"
  installed_location="$(printf '%s\n' "$installed" | sed -n '2p')"
  [ "$installed_version" = "$EXPECTED_VERSION" ]
  [ -n "$installed_location" ]

  chmod -R a-w "$ENVIRONMENT"
  temporary_marker="$READY_MARKER.$$"
  printf '%s\n' \
    "tooling_key=$TOOLING_KEY" \
    "snakemake_version=$installed_version" \
    "snakemake_location=$installed_location" \
    "pyproject_sha256=$PYPROJECT_SHA256" \
    "uv_lock_sha256=$UV_LOCK_SHA256" \
    "bootstrap_sha256=$BOOTSTRAP_SHA256" \
    "uv_version=$(uv --version)" >"$temporary_marker"
  chmod 0444 "$temporary_marker"
  mv "$temporary_marker" "$READY_MARKER"
}

resolve_container() {
  local discovery_path="${1:?container discovery path is required}"

  readlink -f "$discovery_path"
}

verify_container() {
  local resolved_sif="${1:?resolved SIF path is required}"

  test -f "$resolved_sif"
  test -f "$resolved_sif.sha256"
  cd "$(dirname "$resolved_sif")"
  sha256sum -c "$(basename "$resolved_sif.sha256")"
}

main() {
  local command="${1:-}"

  case "$command" in
    probe)
      [ "$#" -eq 1 ] || return 2
      load_request
      probe_tooling
      ;;
    install)
      [ "$#" -eq 1 ] || return 2
      load_request
      install_tooling
      ;;
    resolve-container)
      [ "$#" -eq 2 ] || return 2
      resolve_container "$2"
      ;;
    verify-container)
      [ "$#" -eq 2 ] || return 2
      verify_container "$2"
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
