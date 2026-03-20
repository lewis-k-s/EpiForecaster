#!/bin/bash
#SBATCH --job-name=epi-crossval
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

resolve_output_log_root() {
  local project_root="$1"
  local output_log_dir="$2"

  if [ -z "$output_log_dir" ]; then
    printf '%s\n' "${project_root%/}"
    return 0
  fi

  case "$output_log_dir" in
    /*) printf '%s\n' "$output_log_dir" ;;
    *) printf '%s/%s\n' "${project_root%/}" "${output_log_dir#./}" ;;
  esac
}

main() {
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
CAMPAIGN_ID="${CAMPAIGN_ID:-crossval_${SLURM_ARRAY_JOB_ID:-local}}"
CV_SEEDS="${CV_SEEDS:-42 43 44 45 46}"
OVERRIDES="${OVERRIDES:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_MAX_EPOCHS="${DRY_RUN_MAX_EPOCHS:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR:-outputs/training}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

read -r -a SEED_LIST <<<"$CV_SEEDS"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAST_TASK_INDEX=$((${#SEED_LIST[@]} - 1))

if [ "$TASK_ID" -gt "$LAST_TASK_INDEX" ]; then
  echo "Error: task index $TASK_ID is out of bounds for CV_SEEDS='$CV_SEEDS'"
  echo "Update --array=0-$LAST_TASK_INDEX to match the configured seed list."
  exit 1
fi

SEED="${SEED_LIST[$TASK_ID]}"
MODEL_ID="${SLURM_ARRAY_JOB_ID:-local}_cv${TASK_ID}_s${SEED}"
OUTPUT_LOG_ROOT="$(resolve_output_log_root "$PROJECT_ROOT" "$OUTPUT_LOG_DIR")"
RUN_DIR="$OUTPUT_LOG_ROOT/crossval__${CAMPAIGN_ID}/${MODEL_ID}"

OVERRIDE_ARGS=()
if [ -n "$OVERRIDES" ]; then
  for override in $OVERRIDES; do
    OVERRIDE_ARGS+=(--override "$override")
  done
fi

if [ "$DRY_RUN" = "1" ]; then
  MAX_EPOCHS="${MAX_EPOCHS:-$DRY_RUN_MAX_EPOCHS}"
  echo "DRY_RUN enabled: max_epochs=${MAX_EPOCHS}"
fi

if [ -n "$MAX_EPOCHS" ]; then
  OVERRIDE_ARGS+=(--override "training.epochs=${MAX_EPOCHS}")
fi

EXPERIMENT_NAME_OVERRIDE="output.experiment_name=crossval__${CAMPAIGN_ID}"
GROUP_OVERRIDE="output.wandb_group=crossval_campaign_${CAMPAIGN_ID}"
TAGS_OVERRIDE="output.wandb_tags=[crossval,campaign_${CAMPAIGN_ID},seed_${SEED}]"

OVERRIDE_ARGS+=(
  --override "output.log_dir=${OUTPUT_LOG_DIR}"
  --override "$EXPERIMENT_NAME_OVERRIDE"
  --override "$GROUP_OVERRIDE"
  --override "output.write_granular_eval=true"
  --override "training.seed=${SEED}"
  --override "$TAGS_OVERRIDE"
)

echo "Running cross-val seed=${SEED} task_id=${TASK_ID}"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Config: $CONFIG"
echo "Model ID: $MODEL_ID"
echo "Run dir: $RUN_DIR"

EPIFORECASTER_MODEL_ID="$MODEL_ID" \
  uv run main train epiforecaster --config "$CONFIG" "${OVERRIDE_ARGS[@]}"

uv run python scripts/eval_crossval_granular.py \
  --run-dir "$RUN_DIR" \
  --splits val test \
  --device "$EVAL_DEVICE" \
  --force

echo "Completed cross-val run for seed=${SEED}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
