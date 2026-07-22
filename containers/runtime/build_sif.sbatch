#!/bin/bash
#SBATCH --job-name=epi-sif-build
#SBATCH --account=bsc08
#SBATCH --partition=acc
#SBATCH --qos=acc_bscls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00

# Assemble a native SIF from the network-prefetched base and locked Python
# environment. This job performs no network access.

set -euo pipefail
umask 0027

BASE_SIF="${BASE_SIF:?BASE_SIF must be exported by the dispatcher}"
BASE_IMAGE_REF="${BASE_IMAGE_REF:?BASE_IMAGE_REF must be exported by the dispatcher}"
PREFIX_ROOT="${PREFIX_ROOT:?PREFIX_ROOT must be exported by the dispatcher}"
DEPENDENCY_KEY="${DEPENDENCY_KEY:?DEPENDENCY_KEY must be exported by the dispatcher}"
UV_LOCK_SHA256="${UV_LOCK_SHA256:?UV_LOCK_SHA256 must be exported by the dispatcher}"
CONTAINER_ROOT="${CONTAINER_ROOT:?CONTAINER_ROOT must be exported by the dispatcher}"
BUILD_ROOT="${BUILD_ROOT:?BUILD_ROOT must be exported by the dispatcher}"

if [[ ! "$DEPENDENCY_KEY" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid dependency key: $DEPENDENCY_KEY" >&2
  exit 2
fi
build_key="${DEPENDENCY_KEY:0:20}"
final_sif="$CONTAINER_ROOT/epiforecaster-runtime-$build_key.sif"
digest_sidecar="$final_sif.sha256"
metadata_file="$final_sif.build.txt"
inspect_file="$final_sif.inspect.json"
success_marker="$BUILD_ROOT/build.done"

if ! command -v module >/dev/null 2>&1; then
  set +u
  # shellcheck disable=SC1091
  source /etc/profile.d/00-bsc.sh 2>/dev/null || true
  # shellcheck disable=SC1091
  source /etc/profile.d/01-module.sh 2>/dev/null || true
  set -u
fi
module load EB/apps >/dev/null 2>&1 || true
module load "${SINGULARITY_MODULE:-singularity/4.1.5}"

test -f "$BASE_SIF"
if [ ! -L "$PREFIX_ROOT/.venv/bin/python" ] && \
  [ ! -f "$PREFIX_ROOT/.venv/bin/python" ]; then
  echo "Prefetched environment has no Python launcher: $PREFIX_ROOT/.venv/bin/python" >&2
  exit 1
fi
test -f "$BUILD_ROOT/deps.ready"
recorded_key="$(awk -F= '$1 == "dependency_key" {print $2}' "$BUILD_ROOT/deps.ready")"
if [ "$recorded_key" != "$DEPENDENCY_KEY" ]; then
  echo "Prefetched dependency marker does not match this build" >&2
  exit 1
fi

mkdir -p "$CONTAINER_ROOT" "$BUILD_ROOT"
exec 9>"$CONTAINER_ROOT/.epiforecaster-runtime-$build_key.lock"
flock 9

verify_existing() {
  local expected actual
  test -f "$final_sif" && \
    test -f "$digest_sidecar" && \
    test -f "$metadata_file" && \
    test -f "$inspect_file" && \
    test -f "$success_marker" || return 1
  expected="$(awk 'NR == 1 {print $1}' "$digest_sidecar")"
  actual="$(sha256sum "$final_sif" | awk '{print $1}')"
  test "$actual" = "$expected"
}

if [ -e "$final_sif" ] || \
  [ -e "$digest_sidecar" ] || \
  [ -e "$metadata_file" ] || \
  [ -e "$inspect_file" ] || \
  [ -e "$success_marker" ]; then
  if ! verify_existing; then
    echo "Existing output is incomplete or failed verification: $final_sif" >&2
    exit 1
  fi
  ln -sfn "$(basename "$final_sif")" "$CONTAINER_ROOT/epiforecaster-runtime.sif"
  ln -sfn "$(basename "$digest_sidecar")" \
    "$CONTAINER_ROOT/epiforecaster-runtime.sif.sha256"
  touch "$success_marker"
  echo "Reusable SIF already built: $final_sif"
  exit 0
fi

scratch_leaf="epiforecaster-sif-${SLURM_JOB_ID:-$$}"
scratch_root=""
for scratch_parent in \
  "${SLURM_TMPDIR:-}" \
  "/scratch/tmp/${USER:-unknown}"; do
  [ -n "$scratch_parent" ] || continue
  scratch_candidate="${scratch_parent%/}/$scratch_leaf"
  if mkdir -p "$scratch_candidate" 2>/dev/null; then
    scratch_root="$scratch_candidate"
    break
  fi
done
if [ -z "$scratch_root" ]; then
  scratch_root="$BUILD_ROOT/.scratch-${SLURM_JOB_ID:-$$}"
  mkdir -p "$scratch_root"
fi
mkdir -p "$scratch_root/cache" "$scratch_root/tmp"

sandbox="$scratch_root/rootfs"
scratch_sif="$scratch_root/epiforecaster-runtime.sif"
publish_tmp="$CONTAINER_ROOT/.epiforecaster-runtime-$build_key-${SLURM_JOB_ID:-$$}.sif"
sidecar_tmp="$digest_sidecar.${SLURM_JOB_ID:-$$}.tmp"
metadata_tmp="$metadata_file.${SLURM_JOB_ID:-$$}.tmp"
inspect_tmp="$inspect_file.${SLURM_JOB_ID:-$$}.tmp"
cleanup() {
  set +e
  if [ -d "$scratch_root" ]; then
    chmod -R u+w "$scratch_root" 2>/dev/null
    rm -rf -- "$scratch_root"
  fi
  rm -f "$publish_tmp" "$sidecar_tmp" "$metadata_tmp" "$inspect_tmp"
}
trap cleanup EXIT

export SINGULARITY_CACHEDIR="$scratch_root/cache"
export SINGULARITY_TMPDIR="$scratch_root/tmp"
export TMPDIR="$scratch_root/tmp"

echo "Assembling offline runtime on $(hostname)"
echo "Host architecture: $(uname -m)"
echo "Scratch root: $scratch_root"
singularity build --sandbox "$sandbox" "$BASE_SIF"
mkdir -p "$sandbox/opt/epiforecaster" "$sandbox/usr/local/bin"
cp -a "$PREFIX_ROOT/." "$sandbox/opt/epiforecaster/"
chmod -R u+w "$sandbox/opt/epiforecaster"
chmod -R a+rX "$sandbox/opt/epiforecaster"

# Singularity regenerates its metadata while packing a sandbox, and MN5 does
# not provide the subuid mapping needed for a definition build with fakeroot.
# Use ordinary filesystem launchers instead. The prefetched venv was created at
# its final /opt path, so its console-script shebangs remain valid in the SIF.
for python_command in python python3; do
  rm -f "$sandbox/usr/local/bin/$python_command"
  printf '%s\n' \
    '#!/bin/sh' \
    'exec /opt/epiforecaster/.venv/bin/python "$@"' > \
    "$sandbox/usr/local/bin/$python_command"
  chmod 0755 "$sandbox/usr/local/bin/$python_command"
done
for console_script in "$PREFIX_ROOT/.venv/bin/"*; do
  script_name="${console_script##*/}"
  case "$script_name" in
    activate* | deactivate.bat | python | python3 | python3.11)
      continue
      ;;
  esac
  [ -x "$console_script" ] || continue
  rm -f "$sandbox/usr/local/bin/$script_name"
  printf '%s\n' \
    '#!/bin/sh' \
    "exec /opt/epiforecaster/.venv/bin/$script_name \"\$@\"" > \
    "$sandbox/usr/local/bin/$script_name"
  chmod 0755 "$sandbox/usr/local/bin/$script_name"
done
singularity build --force "$scratch_sif" "$sandbox"

container_arch="$(singularity exec --cleanenv "$scratch_sif" uname -m)"
if [ "$container_arch" != "$(uname -m)" ]; then
  echo "Container architecture $container_arch does not match host $(uname -m)" >&2
  exit 1
fi

singularity exec --cleanenv "$scratch_sif" sh -c '
  printf "runtime python=%s\nruntime dvc=%s\n" \
    "$(command -v python)" "$(command -v dvc)"
  test "$(command -v python)" = /usr/local/bin/python
  test "$(command -v dvc)" = /usr/local/bin/dvc
'
singularity exec --cleanenv "$scratch_sif" python -c \
  'import sys; assert sys.version_info[:2] == (3, 11); assert sys.prefix == "/opt/epiforecaster/.venv"; import dvc, geopandas, netCDF4, optuna, torch, torch_geometric, xarray, zarr; print("dependency imports: ok", sys.version, torch.__version__)'
singularity exec --cleanenv "$scratch_sif" git --version
singularity exec --cleanenv "$scratch_sif" dvc --version
singularity exec --cleanenv --nv "$scratch_sif" nvidia-smi -L
singularity exec --cleanenv --nv "$scratch_sif" python -c \
  'import json, torch; assert torch.cuda.is_available(); print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}))'
singularity inspect --json "$scratch_sif" > "$inspect_tmp"

sif_sha256="$(sha256sum "$scratch_sif" | awk '{print $1}')"
cp "$scratch_sif" "$publish_tmp"
if [ "$(sha256sum "$publish_tmp" | awk '{print $1}')" != "$sif_sha256" ]; then
  echo "Published SIF copy failed SHA-256 verification" >&2
  exit 1
fi
chmod 0444 "$publish_tmp"
printf '%s  %s\n' "$sif_sha256" "$(basename "$final_sif")" > "$sidecar_tmp"
printf '%s\n' \
  "dependency_key=$DEPENDENCY_KEY" \
  "base_image=$BASE_IMAGE_REF" \
  "uv_lock_sha256=$UV_LOCK_SHA256" \
  "sif_sha256=$sif_sha256" \
  "singularity_version=$(singularity version)" \
  "host=$(hostname)" \
  "host_arch=$(uname -m)" \
  "slurm_job_id=${SLURM_JOB_ID:-unknown}" > "$metadata_tmp"

mv "$publish_tmp" "$final_sif"
mv "$sidecar_tmp" "$digest_sidecar"
mv "$metadata_tmp" "$metadata_file"
mv "$inspect_tmp" "$inspect_file"
chmod 0444 "$digest_sidecar" "$metadata_file" "$inspect_file"
ln -sfn "$(basename "$final_sif")" "$CONTAINER_ROOT/epiforecaster-runtime.sif"
ln -sfn "$(basename "$digest_sidecar")" \
  "$CONTAINER_ROOT/epiforecaster-runtime.sif.sha256"
touch "$success_marker"

echo "Shared SIF: $final_sif"
echo "SIF SHA-256: $sif_sha256"
