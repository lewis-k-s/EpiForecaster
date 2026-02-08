#!/bin/bash
#SBATCH --job-name=epi-hpo
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00 # 5 hours for 50-epoch trials with pruning
#SBATCH --array=0-3 # saturate the node with parallel tasks on each GPU
#SBATCH --output=slurm-%x-%A_%a.out # stdout/stderr log per array task

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base training config (override per run).
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
# Optuna study identity.
STUDY_NAME="${STUDY_NAME:-epiforecaster_hpo_v1}"
# Shared journal file for coordination (must be on shared filesystem).
JOURNAL_FILE="${JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${STUDY_NAME}.journal}"
# Trial outputs root (log_dir override).
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/optuna}"
# Optional speed knobs (set to empty to disable).
EPOCHS="${EPOCHS:-50}"  # Default 50 epochs for meaningful HPO rankings
MAX_BATCHES="${MAX_BATCHES:-}"
SEED="${SEED:-42}"  # Reproducibility seed
SAMPLER="${SAMPLER:-tpe}"  # tpe (default), cmaes, or random
PRUNING_START_EPOCH="${PRUNING_START_EPOCH:-10}"  # Start pruning after N epochs
LOSS_WEIGHT_MODE="${LOSS_WEIGHT_MODE:-fixed}"  # fixed (recommended), bounded
LOSS_WEIGHT_MAX_RATIO="${LOSS_WEIGHT_MAX_RATIO:-2.0}"  # for bounded mode
# Space-separated config overrides applied to base config before HPO (e.g., "data.log_scale=false").
OVERRIDES="${OVERRIDES:-}"
# Worker timeout in seconds (auto-calculated from SLURM time limit minus 15min buffer).
# Falls back to 5 hours - 15 minutes if SLURM_TIME_LIMIT is not set.
if [ -n "${SLURM_TIME_LIMIT:-}" ]; then
    # Convert SLURM_TIME_LIMIT (in minutes) to seconds and subtract 15min buffer
    TIMEOUT_SECS="${TIMEOUT_SECS:-$((SLURM_TIME_LIMIT * 60 - 900))}"
else
    TIMEOUT_SECS="${TIMEOUT_SECS:-17100}"  # 5 hours - 15 minutes (in seconds)
fi
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

ARGS=(
  --config "$CONFIG"
  --study-name "$STUDY_NAME"
  --journal-file "$JOURNAL_FILE"
  --run-root "$RUN_ROOT"
)

# Optional dotted-key overrides are supported by repeating:
#   --override key=value
# In this sbatch script, OVERRIDES is currently not expanded into repeated args.
# Use a custom base config for bulk overrides.

if [ -n "$EPOCHS" ]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [ -n "$MAX_BATCHES" ]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [ -n "$SEED" ]; then
  ARGS+=(--seed "$SEED")
fi
if [ -n "$SAMPLER" ]; then
  ARGS+=(--sampler "$SAMPLER")
fi
if [ -n "$PRUNING_START_EPOCH" ]; then
  ARGS+=(--pruning-start-epoch "$PRUNING_START_EPOCH")
fi
if [ -n "$LOSS_WEIGHT_MODE" ]; then
  ARGS+=(--loss-weight-mode "$LOSS_WEIGHT_MODE")
fi
if [ -n "$LOSS_WEIGHT_MAX_RATIO" ]; then
  ARGS+=(--loss-weight-max-ratio "$LOSS_WEIGHT_MAX_RATIO")
fi
if [ -n "$TIMEOUT_SECS" ]; then
  ARGS+=(--timeout-s "$TIMEOUT_SECS")
fi

# Prefer uv runner if available; fall back to plain python.
if command -v uv >/dev/null 2>&1; then
  uv run python scripts/optuna_epiforecaster_worker.py "${ARGS[@]}"
else
  python scripts/optuna_epiforecaster_worker.py "${ARGS[@]}"
fi
