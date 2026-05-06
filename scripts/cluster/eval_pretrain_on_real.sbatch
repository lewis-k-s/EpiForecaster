#!/bin/bash
#SBATCH --job-name=epi-pretrain-real-eval
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PRETRAIN_EXPERIMENT_NAME="${PRETRAIN_EXPERIMENT_NAME:-mn5_epiforecaster_synth_pretrain}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:?PRETRAIN_RUN_ID must be set}"
PRETRAIN_CHECKPOINT_NAME="${PRETRAIN_CHECKPOINT_NAME:-best_model.pt}"
PRETRAIN_CHECKPOINT_PATH="${PRETRAIN_CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/training/$PRETRAIN_EXPERIMENT_NAME/$PRETRAIN_RUN_ID/checkpoints/$PRETRAIN_CHECKPOINT_NAME}"
REAL_DATASET_PATH="${REAL_DATASET_PATH:-data/processed/real_with_id.zarr}"
REAL_EVAL_EXPERIMENT_NAME="${REAL_EVAL_EXPERIMENT_NAME:-${PRETRAIN_EXPERIMENT_NAME}_real_eval}"
REAL_EVAL_RUN_ID="${REAL_EVAL_RUN_ID:-${PRETRAIN_RUN_ID}}"
REAL_EVAL_OUTPUT_DIR="${REAL_EVAL_OUTPUT_DIR:-$PROJECT_ROOT/outputs/training/$REAL_EVAL_EXPERIMENT_NAME/$REAL_EVAL_RUN_ID}"
REAL_EVAL_SPLITS="${REAL_EVAL_SPLITS:-val test}"
REAL_EVAL_DEVICE="${REAL_EVAL_DEVICE:-auto}"
REAL_EVAL_BATCH_SIZE="${REAL_EVAL_BATCH_SIZE:-}"
REAL_EVAL_NUM_WORKERS="${REAL_EVAL_NUM_WORKERS:-0}"
REAL_EVAL_LOG_EVERY="${REAL_EVAL_LOG_EVERY:-10}"
REAL_EVAL_GRANULAR="${REAL_EVAL_GRANULAR:-0}"
REAL_EVAL_OVERRIDES="${REAL_EVAL_OVERRIDES:-}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/cluster/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ARGS=(
  --checkpoint "$PRETRAIN_CHECKPOINT_PATH"
  --output-dir "$REAL_EVAL_OUTPUT_DIR"
  --real-dataset-path "$REAL_DATASET_PATH"
  --splits $REAL_EVAL_SPLITS
  --device "$REAL_EVAL_DEVICE"
  --num-workers "$REAL_EVAL_NUM_WORKERS"
  --prefetch-factor 0
  --log-every "$REAL_EVAL_LOG_EVERY"
)

if [ -n "$REAL_EVAL_BATCH_SIZE" ]; then
  ARGS+=(--batch-size "$REAL_EVAL_BATCH_SIZE")
fi

if [ "$REAL_EVAL_GRANULAR" = "1" ]; then
  ARGS+=(--granular)
fi

if [ -n "$REAL_EVAL_OVERRIDES" ]; then
  for override in $REAL_EVAL_OVERRIDES; do
    ARGS+=(--override "$override")
  done
fi

echo "Starting real-data eval for synthetic pretrain checkpoint"
echo "  checkpoint: $PRETRAIN_CHECKPOINT_PATH"
echo "  output_dir: $REAL_EVAL_OUTPUT_DIR"
echo "  real_dataset_path: $REAL_DATASET_PATH"
echo "  splits: $REAL_EVAL_SPLITS"

uv run python -m scripts.analysis.eval_checkpoint_on_real "${ARGS[@]}"
