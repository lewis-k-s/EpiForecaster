#!/bin/bash

# Run only on wifi / wifi-acc. Pull the pinned base and materialize the locked
# Linux environment before any compute-node job is submitted.

set -euo pipefail
umask 0027

BASE_IMAGE_REF="${BASE_IMAGE_REF:?BASE_IMAGE_REF must be set}"
DEPENDENCY_KEY="${DEPENDENCY_KEY:?DEPENDENCY_KEY must be set}"
UV_LOCK_SHA256="${UV_LOCK_SHA256:?UV_LOCK_SHA256 must be set}"
BUILD_ROOT="${BUILD_ROOT:?BUILD_ROOT must be set}"
CONTAINER_ROOT="${CONTAINER_ROOT:?CONTAINER_ROOT must be set}"
UV_CACHE_ROOT="${UV_CACHE_ROOT:-$BUILD_ROOT/uv-cache}"

if [[ ! "$BASE_IMAGE_REF" =~ @sha256:([0-9a-f]{64})$ ]]; then
  echo "BASE_IMAGE_REF must be pinned to an amd64 manifest digest" >&2
  exit 2
fi
base_digest="${BASH_REMATCH[1]}"
if [[ ! "$DEPENDENCY_KEY" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid dependency key: $DEPENDENCY_KEY" >&2
  exit 2
fi

base_key="${base_digest:0:20}"
base_root="$CONTAINER_ROOT/bases"
base_sif="$base_root/uv-python311-bookworm-$base_key.sif"
context_root="$BUILD_ROOT/context"
prefix_root="$BUILD_ROOT/prefix"
cache_root="$UV_CACHE_ROOT"
temporary_root="$BUILD_ROOT/.singularity-tmp"
ready_marker="$BUILD_ROOT/deps.ready"

# Load the network-login Singularity module in non-interactive shells.
load_singularity() {
  if ! command -v module >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source /etc/profile.d/00-bsc.sh 2>/dev/null || true
    # shellcheck disable=SC1091
    source /etc/profile.d/01-module.sh 2>/dev/null || true
    set -u
  fi
  module load EB/apps >/dev/null 2>&1 || true
  module load "${SINGULARITY_FETCH_MODULE:-singularity/3.11.5}"
}

# Pull and atomically publish the pinned base image once per manifest digest.
materialize_base_sif() {
  local temporary

  if [ -f "$base_sif" ]; then
    return 0
  fi

  temporary="$base_root/.uv-python311-bookworm-$base_key-$$.sif"
  trap 'rm -f "$temporary"' EXIT
  singularity pull "$temporary" "docker://$BASE_IMAGE_REF"
  singularity inspect "$temporary" >/dev/null
  chmod 0444 "$temporary"
  mv "$temporary" "$base_sif"
  trap - EXIT
}

# Accept an existing dependency bundle only when its marker and Python agree.
reuse_ready_bundle() {
  local recorded_key

  [ -f "$ready_marker" ] || return 1
  recorded_key="$(awk -F= '$1 == "dependency_key" {print $2}' "$ready_marker")"
  if [ "$recorded_key" = "$DEPENDENCY_KEY" ] \
    && { [ -L "$prefix_root/.venv/bin/python" ] \
      || [ -f "$prefix_root/.venv/bin/python" ]; }; then
    echo "Locked dependency bundle already exists: $prefix_root"
    return 0
  fi

  echo "Dependency marker or prefetched environment is invalid: $ready_marker" >&2
  return 2
}

load_singularity

mkdir -p "$base_root" "$cache_root" "$temporary_root"
exec 9>"$base_root/.uv-python311-bookworm-$base_key.lock"
flock 9
exec 8>"$cache_root/.uv-sync.lock"
export SINGULARITY_CACHEDIR="$cache_root/singularity"
export SINGULARITY_LOCALCACHEDIR="/scratch/tmp/$USER/epiforecaster-$DEPENDENCY_KEY"
export SINGULARITY_TMPDIR="$temporary_root"
export TMPDIR="$temporary_root"
mkdir -p \
  "$SINGULARITY_CACHEDIR" \
  "$SINGULARITY_LOCALCACHEDIR" \
  "$SINGULARITY_TMPDIR"

materialize_base_sif
printf '%s\n' "$base_sif" >"$BUILD_ROOT/base-sif.path"

if [ -f "$ready_marker" ]; then
  if reuse_ready_bundle; then
    exit 0
  fi
  exit 1
fi

test -f "$context_root/pyproject.toml"
test -f "$context_root/uv.lock"
mkdir -p "$prefix_root/.venv" "$cache_root/uv"
flock 8

singularity exec --cleanenv \
  --bind "$context_root:/workspace/source:ro" \
  --bind "$prefix_root:/opt/epiforecaster" \
  --bind "$cache_root/uv:/var/cache/uv" \
  --pwd /workspace/source \
  "$base_sif" \
  env \
  UV_PROJECT_ENVIRONMENT=/opt/epiforecaster/.venv \
  UV_CACHE_DIR=/var/cache/uv \
  UV_CONCURRENT_INSTALLS=1 \
  UV_LINK_MODE=copy \
  UV_PYTHON_DOWNLOADS=never \
  uv sync \
  --frozen \
  --python /usr/local/bin/python3.11 \
  --no-install-project \
  --no-default-groups \
  --group workflow

singularity exec --cleanenv \
  --bind "$prefix_root:/opt/epiforecaster:ro" \
  "$base_sif" \
  /opt/epiforecaster/.venv/bin/python -c \
  'import sys; assert sys.version_info[:2] == (3, 11); import dvc, geopandas, netCDF4, optuna, snakemake, torch, torch_geometric, xarray, zarr; print("locked dependency imports: ok", sys.version, snakemake.__version__, torch.__version__)'

chmod -R a-w "$prefix_root"
printf '%s\n' \
  "dependency_key=$DEPENDENCY_KEY" \
  "base_image=$BASE_IMAGE_REF" \
  "uv_lock_sha256=$UV_LOCK_SHA256" \
  "uv_version=$(singularity exec --cleanenv "$base_sif" uv --version)" >"$ready_marker"
chmod 0444 "$ready_marker" "$BUILD_ROOT/base-sif.path"

echo "Locked dependencies ready: $prefix_root"
echo "Base SIF: $base_sif"
