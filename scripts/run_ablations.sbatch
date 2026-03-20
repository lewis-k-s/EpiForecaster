#!/bin/bash
#SBATCH --job-name=epi-ablations
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1                # one CUDA-capable GPU
#SBATCH --cpus-per-task=20          # must be full node
#SBATCH --time=04:00:00             # wall-clock limit
#SBATCH --array=0-19                # One task per ablation
#SBATCH --output=logs/slurm-%x-%A_%a.out    # stdout/stderr log per array task
#SBATCH --error=logs/slurm-%x-%A_%a.err    # stdout/stderr log per array task
#
#
# Usage example with external sbatch arguments:
#   sbatch --job-name=epi-ablations --account=bsc08 --qos=acc_bscls \
#          --gres=gpu:1 --cpus-per-task=20 --time=04:00:00 --array=0-19 \
#          --output=logs/slurm-%x-%A_%a.out --error=logs/slurm-%x-%A_%a.err \
#          scripts/run_ablations.sbatch
#
# All SBATCH directives can be overridden at submission time.

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base baseline config that should have the *best hyperparameters* from a previous HPO sweep.
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
# Path to an existing virtual environment (override or comment out if using modules).
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
# Campaign and run controls.
CAMPAIGN_ID="${CAMPAIGN_ID:-ablation_${SLURM_ARRAY_JOB_ID:-local}}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
SEED="${SEED:-42}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
# Smoke-test mode:
#   DRY_RUN=1 sbatch --array=0-0 --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/run_ablations.sbatch
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_MAX_EPOCHS="${DRY_RUN_MAX_EPOCHS:-1}"
DRY_RUN_RUN_ANALYSIS="${DRY_RUN_RUN_ANALYSIS:-0}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

# Keep thread counts aligned with allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Define the ablations. These correspond to Study 4, 5, and 6 in todo/STUDIES.md
# #SBATCH --array=0-19                # One task per ablation
# #SBATCH --output=logs/slurm-%x-%A_%a.out    # stdout/stderr log per array task
# #SBATCH --error=logs/slurm-%x-%A_%a.err    # stdout/stderr log per array task

# ...

declare -a ABLATION_NAMES=(
  "baseline"                # 0: Full model (baseline)
  "no_sir_loss"             # 1: Study 4: -L_sir
  "no_ww_loss"              # 2: Loss OFF, Input ON
  "mask_ww_input"           # 3: Loss ON, Input OFF
  "no_ww"                   # 4: Loss OFF, Input OFF
  "no_cases_loss"           # 5: Loss OFF, Input ON
  "mask_cases_input"        # 6: Loss ON, Input OFF
  "no_cases"                # 7: Loss OFF, Input OFF
  "no_hosp_loss"            # 8: Loss OFF, Input ON
  "mask_hosp_input"         # 9: Loss ON, Input OFF
  "no_hosp"                 # 10: Loss OFF, Input OFF
  "no_deaths_loss"          # 11: Loss OFF, Input ON
  "mask_deaths_input"       # 12: Loss ON, Input OFF
  "no_deaths"               # 13: Loss OFF, Input OFF
  "no_residual_context"     # 14: Study 5: w/o residual context from encoder (assuming residual_scale=0.0 means off)
  "no_mobility"             # 15: Study 6: Remove mobility edges/features
  "no_regions"              # 16: Study 6: Remove static regional embeddings
  "no_mobility_and_regions" # 17: Study 6: Remove both (local-only)
  "mlp_clinical_heads"      # 18: Study 5: delay-kernel vs direct MLP mapping (assuming learnable_kernel=false defaults to direct mapping)
  "fixed_obs_weights"       # 19: Disable adaptive GradNorm weighting
)

declare -a ABLATION_OVERRIDES=(
  ""                                                                                                                                                               # 0: baseline
  "training.loss.joint.w_sir=0.0"                                                                                                                                  # 1: no_sir_loss
  "training.loss.joint.disable_ww=true"                                                                                                                            # 2: no_ww_loss
  "training.loss.joint.mask_input_ww=true"                                                                                                                         # 3: mask_ww_input
  "training.loss.joint.disable_ww=true training.loss.joint.mask_input_ww=true"                                                                                     # 4: no_ww
  "training.loss.joint.disable_cases=true"                                                                                                                         # 5: no_cases_loss
  "training.loss.joint.mask_input_cases=true"                                                                                                                      # 6: mask_cases_input
  "training.loss.joint.disable_cases=true training.loss.joint.mask_input_cases=true"                                                                               # 7: no_cases
  "training.loss.joint.disable_hosp=true"                                                                                                                          # 8: no_hosp_loss
  "training.loss.joint.mask_input_hosp=true"                                                                                                                       # 9: mask_hosp_input
  "training.loss.joint.disable_hosp=true training.loss.joint.mask_input_hosp=true"                                                                                 # 10: no_hosp
  "training.loss.joint.disable_deaths=true"                                                                                                                        # 11: no_deaths_loss
  "training.loss.joint.mask_input_deaths=true"                                                                                                                     # 12: mask_deaths_input
  "training.loss.joint.disable_deaths=true training.loss.joint.mask_input_deaths=true"                                                                             # 13: no_deaths
  "model.observation_heads.residual_scale=0.0"                                                                                                                     # 14: no_residual_context
  "model.type.mobility=false"                                                                                                                                      # 15: no_mobility
  "model.type.regions=false"                                                                                                                                       # 16: no_regions
  "model.type.mobility=false model.type.regions=false"                                                                                                             # 17: no_mobility_and_regions
  "model.observation_heads.learnable_kernel_hosp=false model.observation_heads.learnable_kernel_cases=false model.observation_heads.learnable_kernel_deaths=false" # 18: mlp_clinical_heads
  "training.loss.joint.adaptive_scheme=none"                                                                                                                       # 19: fixed_obs_weights
)

# Default to task 0 when running locally (outside sbatch array).
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Check if task index is beyond our defined ablations.
if [ "$TASK_ID" -ge "${#ABLATION_NAMES[@]}" ]; then
  echo "Error: task index $TASK_ID is out of bounds."
  exit 1
fi

ABLATION_NAME="${ABLATION_NAMES[$TASK_ID]}"
OVERRIDES="${ABLATION_OVERRIDES[$TASK_ID]}"
LAST_TASK_INDEX=$((${#ABLATION_NAMES[@]} - 1))

# Build override arguments.
OVERRIDE_ARGS=()
if [ -n "$OVERRIDES" ]; then
  for override in $OVERRIDES; do
    OVERRIDE_ARGS+=(--override "$override")
  done
fi

if [ "$DRY_RUN" = "1" ]; then
  MAX_EPOCHS="${MAX_EPOCHS:-$DRY_RUN_MAX_EPOCHS}"
  RUN_ANALYSIS="${DRY_RUN_RUN_ANALYSIS}"
  echo "DRY_RUN enabled: max_epochs=${MAX_EPOCHS}, run_analysis=${RUN_ANALYSIS}"
  echo "Tip: submit with single array task and short walltime, e.g. sbatch --array=0-0 --time=00:10:00 --export=ALL,DRY_RUN=1 scripts/run_ablations.sbatch"
fi

if [ -n "$MAX_EPOCHS" ]; then
  OVERRIDE_ARGS+=(--override "training.epochs=${MAX_EPOCHS}")
fi

echo "Running ablation: $ABLATION_NAME (Task ID: $TASK_ID)"
echo "Base config: $CONFIG"
echo "Overrides: $OVERRIDES"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Max epochs override: ${MAX_EPOCHS:-<config default>}"
echo "Seed: $SEED"
echo "Run analysis: $RUN_ANALYSIS"

# Override experiment name so logs don't overwrite
# e.g., mn5_ablation__ablation_12345__no_sir_loss
EXPERIMENT_NAME_OVERRIDE="output.experiment_name=mn5_ablation__${CAMPAIGN_ID}__${ABLATION_NAME}"

# Set W&B group and tags for easy comparison across ablations
GROUP_OVERRIDE="output.wandb_group=ablation_campaign_${CAMPAIGN_ID}"

OVERRIDE_ARGS+=(
  --override "$EXPERIMENT_NAME_OVERRIDE"
  --override "$GROUP_OVERRIDE"
  --override "output.write_granular_eval=true"
)

model_id_override="${SLURM_JOB_ID:-local}_a${TASK_ID}_s${SEED}"
tags_override="output.wandb_tags=[mn5,ablation,${ABLATION_NAME},campaign_${CAMPAIGN_ID},seed_${SEED}]"
RUN_ARGS=(
  "${OVERRIDE_ARGS[@]}"
  --override "training.seed=${SEED}"
  --override "$tags_override"
)

echo "Model ID override: $model_id_override"

EPIFORECASTER_MODEL_ID="$model_id_override" \
  uv run main train epiforecaster --config "$CONFIG" "${RUN_ARGS[@]}"

echo "Completed 1 run for ablation '${ABLATION_NAME}'."

# Optional post-processing on the last array task.
if [ "$RUN_ANALYSIS" = "1" ] && [ "$TASK_ID" -eq "$LAST_TASK_INDEX" ]; then

  echo ""
  echo "========================================"
  echo "Running ablation analysis on last task..."
  echo "========================================"

  ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-outputs/reports/ablation_analysis}"

  uv run python scripts/analyze_ablations.py \
    --training-dir outputs/training \
    --output-dir "$ANALYSIS_OUTPUT_DIR" \
    --campaign-id "$CAMPAIGN_ID" \
    --baseline baseline

  uv run python -m dataviz.ablation_plots \
    "$ANALYSIS_OUTPUT_DIR/ablation_metrics_aggregated.csv" \
    --deltas-csv "$ANALYSIS_OUTPUT_DIR/ablation_metrics_deltas.csv" \
    --cross-head-mean-csv "$ANALYSIS_OUTPUT_DIR/cross_head/cross_head_mean_matrix.csv" \
    --cross-head-std-csv "$ANALYSIS_OUTPUT_DIR/cross_head/cross_head_std_matrix.csv" \
    --output-dir "$ANALYSIS_OUTPUT_DIR" \
    --baseline baseline

  echo ""
  echo "========================================"
  echo "Ablation analysis complete!"
  echo "Results saved to: $ANALYSIS_OUTPUT_DIR"
  echo "========================================"
fi
