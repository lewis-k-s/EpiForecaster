#!/bin/bash
#SBATCH --job-name=epi-ablation-optuna
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-ablation_optuna_${SLURM_ARRAY_JOB_ID:-local}}"
STUDY_NAME="ablation_study_${CAMPAIGN_ID}"
JOURNAL_FILE="${JOURNAL_FILE:-$PROJECT_ROOT/outputs/optuna/${STUDY_NAME}.journal}"
CV_SEEDS="${CV_SEEDS:-42 43 44 45 46}"
N_TRIALS="${N_TRIALS:-}"
# Worker timeout in seconds (auto-calculated from SLURM time limit minus 15min buffer).
if [ -n "${SLURM_TIME_LIMIT:-}" ]; then
  TIMEOUT_SECS="${TIMEOUT_SECS:-$((SLURM_TIME_LIMIT * 60 - 900))}"
else
  TIMEOUT_SECS="${TIMEOUT_SECS:-13500}" # 4 hours - 15 minutes
fi
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_TIMEOUT_SECS="${DRY_RUN_TIMEOUT_SECS:-600}"
DRY_RUN_N_TRIALS="${DRY_RUN_N_TRIALS:-1}"
SMOKETEST="${SMOKETEST:-0}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Starting Optuna Ablation Worker"
echo "  Campaign: $CAMPAIGN_ID"
echo "  Study: $STUDY_NAME"
echo "  Journal: $JOURNAL_FILE"
echo "  Seeds: $CV_SEEDS"
echo "  N_TRIALS: ${N_TRIALS:-<until-timeout>}"
echo "  TIMEOUT_SECS: $TIMEOUT_SECS"

if [ "$DRY_RUN" = "1" ]; then
  N_TRIALS="${N_TRIALS:-$DRY_RUN_N_TRIALS}"
  TIMEOUT_SECS="$DRY_RUN_TIMEOUT_SECS"
  echo "DRY_RUN enabled: n_trials=${N_TRIALS}, timeout_s=${TIMEOUT_SECS}"
fi

ARGS=(
  --config "$CONFIG"
  --study-name "$STUDY_NAME"
  --journal-file "$JOURNAL_FILE"
  --campaign-id "$CAMPAIGN_ID"
  --seeds "$CV_SEEDS"
  --timeout-s "$TIMEOUT_SECS"
)

if [ -n "$N_TRIALS" ]; then
  ARGS+=(--n-trials "$N_TRIALS")
fi

if [ "$SMOKETEST" = "1" ]; then
  ARGS+=(--smoketest)
  echo "SMOKETEST enabled: max_batches=2, epochs=1"
fi

uv run python scripts/optuna_ablation_worker.py "${ARGS[@]}"

# Optional: You can increase --n-trials if workers finish quickly and you want to reuse them.
