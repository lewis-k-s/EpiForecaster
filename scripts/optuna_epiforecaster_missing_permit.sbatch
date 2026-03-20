#!/bin/bash
#SBATCH --job-name=epi-missing-permit-hpo
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base training config (override per run).
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"

# New study identity for missing-permit Pareto sweep.
STUDY_NAME="${STUDY_NAME:-epiforecaster_missing_permit_pareto_v1}"
# Shared journal for this new sweep.
JOURNAL_FILE="${JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${STUDY_NAME}.journal}"
# Trial outputs root (log_dir override).
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/optuna}"

# Freeze source (full-HPO study).
FREEZE_STUDY_NAME="${FREEZE_STUDY_NAME:-epiforecaster_hpo_v1}"
FREEZE_JOURNAL_FILE="${FREEZE_JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${FREEZE_STUDY_NAME}.journal}"
# Optional explicit trial number from freeze study (default: best trial in study).
FREEZE_TRIAL_NUMBER="${FREEZE_TRIAL_NUMBER:-}"

# Optional speed knobs (set to empty to disable).
EPOCHS="${EPOCHS:-20}"
MAX_BATCHES="${MAX_BATCHES:-}"
SEED="${SEED:-42}"
SAMPLER="${SAMPLER:-tpe}"
PRUNING_START_EPOCH="${PRUNING_START_EPOCH:-10}"
# Space-separated config overrides applied to base config before HPO.
OVERRIDES="${OVERRIDES:-}"

# Worker timeout in seconds (auto-calculated from SLURM time limit minus 15min buffer).
if [ -n "${SLURM_TIME_LIMIT:-}" ]; then
  TIMEOUT_SECS="${TIMEOUT_SECS:-$((SLURM_TIME_LIMIT * 60 - 900))}"
else
  TIMEOUT_SECS="${TIMEOUT_SECS:-17100}" # 5 hours - 15 minutes
fi

# Path to an existing virtual environment.
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

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
  --freeze-journal-file "$FREEZE_JOURNAL_FILE"
  --freeze-study-name "$FREEZE_STUDY_NAME"
)

# Optional dotted-key overrides are supported by repeating:
#   --override key=value
# In this sbatch script, OVERRIDES is currently not expanded into repeated args.
# Use a custom base config for bulk overrides.

if [ -n "$FREEZE_TRIAL_NUMBER" ]; then
  ARGS+=(--freeze-trial-number "$FREEZE_TRIAL_NUMBER")
fi
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
if [ -n "$TIMEOUT_SECS" ]; then
  ARGS+=(--timeout-s "$TIMEOUT_SECS")
fi

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/optuna_epiforecaster_missing_permit_worker.py "${ARGS[@]}"
else
  python scripts/optuna_epiforecaster_missing_permit_worker.py "${ARGS[@]}"
fi
