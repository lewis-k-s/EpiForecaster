#!/bin/bash
#SBATCH --job-name=epi-forecaster
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1                # one CUDA-capable GPU
#SBATCH --cpus-per-task=20          # must be full node
#SBATCH --time=02:00:00             # wall-clock limit
#SBATCH --output=logs/slurm-%x-%j.out    # stdout/stderr log per job
#SBATCH --error=logs/slurm-%x-%j.err    # stdout/stderr log per job

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Path to the training config file (override per run).
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
# Optional fixed max-epochs override (leave empty to use config value).
MAX_EPOCHS="${MAX_EPOCHS:-}"
# Space-separated config overrides (e.g., "training.batch_size=24 training.epochs=5").
OVERRIDES="${OVERRIDES:-}"
# Smoke-test mode:
#   DRY_RUN=1 sbatch --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/train_single_gpu.sbatch
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_MAX_EPOCHS="${DRY_RUN_MAX_EPOCHS:-1}"
# Path to an existing virtual environment (override or comment out if using modules).
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

module load EB/apps EB/install CUDA/12.1.1 || (echo "CUDA module not found" && exit 1)

cd "$PROJECT_ROOT"

# Activate virtualenv if present.
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

# Keep thread counts aligned with allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Build CLI args.
ARGS=(
  train
  epiforecaster
  --config "$CONFIG"
)

if [ "$DRY_RUN" = "1" ]; then
  MAX_EPOCHS="${MAX_EPOCHS:-$DRY_RUN_MAX_EPOCHS}"
  echo "DRY_RUN enabled: max_epochs=${MAX_EPOCHS}"
  echo "Tip: submit with short walltime, e.g. sbatch --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/train_single_gpu.sbatch"
fi

if [ -n "$MAX_EPOCHS" ]; then
  OVERRIDES="${OVERRIDES:+$OVERRIDES }training.epochs=${MAX_EPOCHS}"
fi

if [ -n "$OVERRIDES" ]; then
  for override in $OVERRIDES; do
    ARGS+=(--override "$override")
  done
fi

# Prefer uv runner if available; fall back to plain python.
if command -v uv >/dev/null 2>&1; then
  uv run main "${ARGS[@]}"
else
  python -m cli "${ARGS[@]}"
fi
