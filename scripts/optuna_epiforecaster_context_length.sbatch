#!/bin/bash
#SBATCH --job-name=epi-context-study
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-context_length_${SLURM_ARRAY_JOB_ID:-local}}"
STUDY_NAME="${STUDY_NAME:-epiforecaster_context_length_${CAMPAIGN_ID}}"
JOURNAL_FILE="${JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${STUDY_NAME}.journal}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/optuna}"

# Keep these as an explicit study grid, not a free HPO search space.
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-14 28 42 60 84 112}"
STUDY_SEEDS="${STUDY_SEEDS:-42 43 44}"

EPOCHS="${EPOCHS:-20}"
MAX_BATCHES="${MAX_BATCHES:-}"
N_TRIALS="${N_TRIALS:-}"
SEED="${SEED:-42}"
OVERRIDES="${OVERRIDES:-}"

DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_TIMEOUT_SECS="${DRY_RUN_TIMEOUT_SECS:-600}"
DRY_RUN_N_TRIALS="${DRY_RUN_N_TRIALS:-1}"
SMOKETEST="${SMOKETEST:-0}"

if [ -n "${SLURM_TIME_LIMIT:-}" ]; then
  TIMEOUT_SECS="${TIMEOUT_SECS:-$((SLURM_TIME_LIMIT * 60 - 900))}"
else
  TIMEOUT_SECS="${TIMEOUT_SECS:-17100}" # 5 hours - 15 minutes
fi

VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [ "$DRY_RUN" = "1" ]; then
  N_TRIALS="${N_TRIALS:-$DRY_RUN_N_TRIALS}"
  TIMEOUT_SECS="$DRY_RUN_TIMEOUT_SECS"
  SMOKETEST=1
  echo "DRY_RUN enabled: n_trials=${N_TRIALS}, timeout_s=${TIMEOUT_SECS}"
fi

echo "Starting EpiForecaster context-length study"
echo "  Campaign: $CAMPAIGN_ID"
echo "  Study: $STUDY_NAME"
echo "  Journal: $JOURNAL_FILE"
echo "  Context lengths: $CONTEXT_LENGTHS"
echo "  Seeds: $STUDY_SEEDS"
echo "  N_TRIALS: ${N_TRIALS:-<until-timeout>}"
echo "  TIMEOUT_SECS: $TIMEOUT_SECS"

ARGS=(
  --config "$CONFIG"
  --study-name "$STUDY_NAME"
  --journal-file "$JOURNAL_FILE"
  --campaign-id "$CAMPAIGN_ID"
  --run-root "$RUN_ROOT"
  --context-lengths "$CONTEXT_LENGTHS"
  --seeds "$STUDY_SEEDS"
  --timeout-s "$TIMEOUT_SECS"
  --seed "$SEED"
)

if [ -n "$EPOCHS" ]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [ -n "$MAX_BATCHES" ]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [ -n "$N_TRIALS" ]; then
  ARGS+=(--n-trials "$N_TRIALS")
fi
if [ "$SMOKETEST" = "1" ]; then
  ARGS+=(--smoketest)
  echo "SMOKETEST enabled: max_batches=2 and epochs=1"
fi
if [ -n "$OVERRIDES" ]; then
  for override in $OVERRIDES; do
    ARGS+=(--override "$override")
  done
fi

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/optuna_epiforecaster_context_length_worker.py "${ARGS[@]}"
else
  python scripts/optuna_epiforecaster_context_length_worker.py "${ARGS[@]}"
fi
