#!/bin/bash

# Allocation-local immutable inputs. Source this file from the container wrapper.
node_cache_run_root() {
  local base
  # Under the job TMPDIR (/scratch/tmp/<jobid> on MN5), so Slurm removes the
  # copy when the allocation ends even if the step is cancelled.
  base="${EPIFORECASTER_NODE_CACHE_ROOT:-${TMPDIR:-/scratch/tmp}/epiforecaster/inputs}"
  printf '%s/%s/%s\n' "$base" "${SLURM_JOB_ID:?An allocation is required for shared inputs}" "$(basename "$1")"
}

stage_shared_input_cache() {
  local cache_root identity cache_path bind_root digest_command
  cache_root="$(node_cache_run_root "$RUN_ROOT")"
  mkdir -p "$cache_root"
  digest_command=(shasum -a 256)
  if command -v sha256sum >/dev/null 2>&1; then
    digest_command=(sha256sum)
  fi
  identity="$(cat "$RUN_ROOT/config.yaml" "$RUN_ROOT/.state/data_manifest.json" | "${digest_command[@]}" | awk '{print $1}')"
  cache_path="$cache_root/$identity"
  # Keep the locking process alive for the complete copy operation. This
  # form has the same lifetime on the Mac test host and the MN5 host.
  flock -x "$cache_root/$identity.lock" \
    bash "${BASH_SOURCE[0]}" build "$RUN_ROOT" "$SOURCE_ROOT" "$cache_path"
  while IFS= read -r bind_root; do
    [ -n "$bind_root" ] || continue
    BINDS+=("$cache_path/$bind_root:/workspace/source/$bind_root:ro")
  done <"$RUN_ROOT/.state/dvc_bind_roots.txt"
  echo "Using allocation-local DVC inputs at $cache_path"
}

build_shared_input_cache() {
  local run_root="$1" source_root="$2" cache_path="$3"
  local temporary bind_root
  [ ! -f "$cache_path/.ready" ] || return 0
  if [ -e "$cache_path" ]; then
    echo "Unsealed input cache exists: $cache_path" >&2
    return 1
  fi
  temporary="$(mktemp -d "${cache_path%/*}/.staging.XXXXXX")"
  # Failed copies stay separate. Only a complete copy is published.
  while IFS= read -r bind_root; do
    [ -n "$bind_root" ] || continue
    test -d "$source_root/$bind_root"
    mkdir -p "$temporary/$bind_root"
    rsync -a "$source_root/$bind_root/" "$temporary/$bind_root/"
    rsync -aL "$run_root/.state/dvc-workspace/$bind_root/" "$temporary/$bind_root/"
  done <"$run_root/.state/dvc_bind_roots.txt"
  touch "$temporary/.ready"
  chmod -R a-w "$temporary"
  mv "$temporary" "$cache_path"
}

cleanup_shared_input_cache() {
  local cache_root
  cache_root="$(node_cache_run_root "$1")"
  [ -d "$cache_root" ] || return 0
  if ! flock -xn "$cache_root/.usage.lock" \
    bash "${BASH_SOURCE[0]}" remove "$1"; then
    echo "Input cache is still in use or cleanup failed: $cache_root" >&2
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  case "${1:-}" in
    build) build_shared_input_cache "$2" "$3" "$4" ;;
    clean) cleanup_shared_input_cache "${2:?run root is required}" ;;
    remove)
      cache_root="$(node_cache_run_root "${2:?run root is required}")"
      chmod -R u+w "$cache_root"
      rm -rf "$cache_root"
      ;;
    *) exit 2 ;;
  esac
fi
