#!/bin/bash
#SBATCH --job-name=epi-foldval
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#
#
# Usage example with external sbatch arguments:
#   sbatch --job-name=epi-foldval --account=bsc08 --qos=acc_bscls \
#          --gres=gpu:1 --cpus-per-task=20 --time=04:00:00 --array=0-3 \
#          --output=logs/slurm-%x-%A_%a.out --error=logs/slurm-%x-%A_%a.err \
#          scripts/run_fold_validation.sbatch
#
# All SBATCH directives can be overridden at submission time.

set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base training config (override per run).
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
# Path to an existing virtual environment (override or comment out if using modules).
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

# Campaign and run controls.
CAMPAIGN_ID="${CAMPAIGN_ID:-foldval_${SLURM_ARRAY_JOB_ID:-local}}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
SEED="${SEED:-42}"
# Smoke-test mode:
#   DRY_RUN=1 sbatch --array=0-0 --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/run_fold_validation.sbatch
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_MAX_EPOCHS="${DRY_RUN_MAX_EPOCHS:-1}"

module load EB/apps EB/install CUDA/12.1.1 || (echo "CUDA module not found" && exit 1)

cd "$PROJECT_ROOT"

if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Array task -> split configuration.
# IMPORTANT: index 0 is the baseline split for pairwise seed matching.
declare -a SPLIT_NAMES=(
  "baseline_split" # 0: baseline (pairwise reference)
  "node_v15_t15"   # 1
  "node_v10_t20"   # 2
  "node_v25_t05"   # 3
)

declare -a SPLIT_OVERRIDES=(
  "training.split_strategy=node training.val_split=0.2 training.test_split=0.1"
  "training.split_strategy=node training.val_split=0.15 training.test_split=0.15"
  "training.split_strategy=node training.val_split=0.1 training.test_split=0.2"
  "training.split_strategy=node training.val_split=0.25 training.test_split=0.05"
)

if [ "${#SPLIT_NAMES[@]}" -ne "${#SPLIT_OVERRIDES[@]}" ]; then
  echo "Error: SPLIT_NAMES and SPLIT_OVERRIDES must have the same length."
  exit 1
fi

# Default to task 0 when running locally (outside sbatch array).
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "$TASK_ID" -ge "${#SPLIT_NAMES[@]}" ]; then
  echo "Error: task index $TASK_ID is out of bounds."
  echo "Update --array=0-$((${#SPLIT_NAMES[@]} - 1)) to match configured splits."
  exit 1
fi

SPLIT_NAME="${SPLIT_NAMES[$TASK_ID]}"
SPLIT_OVERRIDE_STRING="${SPLIT_OVERRIDES[$TASK_ID]}"

# Build static override args for this split.
OVERRIDE_ARGS=()
for override in $SPLIT_OVERRIDE_STRING; do
  OVERRIDE_ARGS+=(--override "$override")
done

if [ "$DRY_RUN" = "1" ]; then
  MAX_EPOCHS="${MAX_EPOCHS:-$DRY_RUN_MAX_EPOCHS}"
  echo "DRY_RUN enabled: max_epochs=${MAX_EPOCHS}"
  echo "Tip: submit with single array task and short walltime, e.g. sbatch --array=0-0 --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/run_fold_validation.sbatch"
fi

if [ -n "$MAX_EPOCHS" ]; then
  OVERRIDE_ARGS+=(--override "training.epochs=${MAX_EPOCHS}")
fi

echo "Running split configuration: $SPLIT_NAME (Task ID: $TASK_ID)"
echo "Base config: $CONFIG"
echo "Split overrides: $SPLIT_OVERRIDE_STRING"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Max epochs override: ${MAX_EPOCHS:-<config default>}"
echo "Seed: $SEED"

# Keep runs grouped and uniquely named by split.
EXPERIMENT_NAME_OVERRIDE="output.experiment_name=mn5_foldval__${CAMPAIGN_ID}__${SPLIT_NAME}"
GROUP_OVERRIDE="output.wandb_group=foldval_campaign_${CAMPAIGN_ID}"
OVERRIDE_ARGS+=(--override "$EXPERIMENT_NAME_OVERRIDE" --override "$GROUP_OVERRIDE")

model_id_override="${SLURM_JOB_ID:-local}_f${TASK_ID}_s${SEED}"
tags_override="output.wandb_tags=[mn5,foldval,${SPLIT_NAME},campaign_${CAMPAIGN_ID},seed_${SEED}]"

RUN_ARGS=(
  "${OVERRIDE_ARGS[@]}"
  --override "training.seed=${SEED}"
  --override "$tags_override"
)

echo "Model ID override: $model_id_override"

if command -v uv >/dev/null 2>&1; then
  EPIFORECASTER_MODEL_ID="$model_id_override" \
    uv run main train epiforecaster --config "$CONFIG" "${RUN_ARGS[@]}"
else
  EPIFORECASTER_MODEL_ID="$model_id_override" \
    python -m cli train epiforecaster --config "$CONFIG" "${RUN_ARGS[@]}"
fi

echo "Completed 1 run for split '${SPLIT_NAME}'."
