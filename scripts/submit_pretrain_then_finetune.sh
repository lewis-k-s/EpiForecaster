#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/production_only/train_epifor_mn5_synth_pretrain.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"

CHAIN_ID="${CHAIN_ID:-pretrain_finetune_$(date +%Y%m%d_%H%M%S)}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-${CHAIN_ID}_pretrain}"
FINETUNE_RUN_ID="${FINETUNE_RUN_ID:-${CHAIN_ID}_finetune}"

PRETRAIN_EXPERIMENT_NAME="${PRETRAIN_EXPERIMENT_NAME:-mn5_epiforecaster_synth_pretrain}"
PRETRAIN_CHECKPOINT_NAME="${PRETRAIN_CHECKPOINT_NAME:-best_model.pt}"

PRETRAIN_OVERRIDES="${PRETRAIN_OVERRIDES:-}"
FINETUNE_OVERRIDES="${FINETUNE_OVERRIDES:-}"
PRETRAIN_MAX_EPOCHS="${PRETRAIN_MAX_EPOCHS:-}"
FINETUNE_MAX_EPOCHS="${FINETUNE_MAX_EPOCHS:-}"

PRETRAIN_CHECKPOINT_PATH="${PRETRAIN_CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/training/$PRETRAIN_EXPERIMENT_NAME/$PRETRAIN_RUN_ID/checkpoints/$PRETRAIN_CHECKPOINT_NAME}"

combined_finetune_overrides="training.init_checkpoint_path=${PRETRAIN_CHECKPOINT_PATH}"
if [ -n "$FINETUNE_OVERRIDES" ]; then
  combined_finetune_overrides="${combined_finetune_overrides} ${FINETUNE_OVERRIDES}"
fi

PRETRAIN_JOB_ID=$(
  PROJECT_ROOT="$PROJECT_ROOT" \
  CONFIG="$PRETRAIN_CONFIG" \
  OVERRIDES="$PRETRAIN_OVERRIDES" \
  MAX_EPOCHS="$PRETRAIN_MAX_EPOCHS" \
  EPIFORECASTER_MODEL_ID="$PRETRAIN_RUN_ID" \
  sbatch --parsable "$@" scripts/train_single_gpu.sbatch
)

FINETUNE_JOB_ID=$(
  PROJECT_ROOT="$PROJECT_ROOT" \
  CONFIG="$FINETUNE_CONFIG" \
  OVERRIDES="$combined_finetune_overrides" \
  MAX_EPOCHS="$FINETUNE_MAX_EPOCHS" \
  EPIFORECASTER_MODEL_ID="$FINETUNE_RUN_ID" \
  sbatch --parsable --dependency="afterok:${PRETRAIN_JOB_ID}" "$@" scripts/train_single_gpu.sbatch
)

echo "Submitted synth pretrain job: ${PRETRAIN_JOB_ID}"
echo "Submitted real-data fine-tune job: ${FINETUNE_JOB_ID}"
echo "Chain ID: ${CHAIN_ID}"
echo "Pretrain run ID: ${PRETRAIN_RUN_ID}"
echo "Fine-tune run ID: ${FINETUNE_RUN_ID}"
echo "Pretrain checkpoint path: ${PRETRAIN_CHECKPOINT_PATH}"
