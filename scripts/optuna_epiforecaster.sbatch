#!/bin/bash
#SBATCH --job-name=epi-hpo
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00 # remember to update the timeout_secs below if you change this
#SBATCH --array=0-3 # saturate the node with parallel tasks on each GPU
#SBATCH --output=slurm-%x-%A_%a.out # stdout/stderr log per array task

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base training config (override per run).
CONFIG="${CONFIG:-configs/train_epifor_mn5_full.yaml}"
# Optuna study identity.
STUDY_NAME="${STUDY_NAME:-epiforecaster_hpo_v1}"
# Shared journal file for coordination (must be on shared filesystem).
JOURNAL_FILE="${JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${STUDY_NAME}.journal}"
# Trial outputs root (log_dir override).
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/optuna}"
# Optional speed knobs (set to empty to disable).
EPOCHS="${EPOCHS:-2}"
MAX_BATCHES="${MAX_BATCHES:-}"
SEED="${SEED:-}"
# Worker timeout in seconds (defaults to SLURM time limit minus buffer).
# Format HH:MM:SS is converted to seconds.
TIMEOUT_SECS="${TIMEOUT_SECS:-14200}" # 4 hours - buffer
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

if [ -n "$EPOCHS" ]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [ -n "$MAX_BATCHES" ]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [ -n "$SEED" ]; then
  ARGS+=(--seed "$SEED")
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
