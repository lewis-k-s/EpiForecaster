#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_synth_pretrain.yaml}"
OVERRIDES="${OVERRIDES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
EPIFORECASTER_MODEL_ID="${EPIFORECASTER_MODEL_ID:-}"

export PROJECT_ROOT CONFIG OVERRIDES MAX_EPOCHS
if [ -n "$EPIFORECASTER_MODEL_ID" ]; then
  export EPIFORECASTER_MODEL_ID
fi

exec sbatch "$@" scripts/train_single_gpu.sbatch
