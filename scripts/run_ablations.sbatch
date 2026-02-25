#!/bin/bash
#SBATCH --job-name=epi-ablations
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1                # one CUDA-capable GPU
#SBATCH --cpus-per-task=20          # must be full node
#SBATCH --time=04:00:00             # wall-clock limit
#SBATCH --array=0-10                # One task per ablation
#SBATCH --output=logs/slurm-%x-%A_%a.out    # stdout/stderr log per array task
#SBATCH --error=logs/slurm-%x-%A_%a.err    # stdout/stderr log per array task

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Base baseline config that should have the *best hyperparameters* from a previous HPO sweep.
CONFIG="${CONFIG:-configs/production_only/train_epifor_mn5_full.yaml}"
# Path to an existing virtual environment (override or comment out if using modules).
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
# Campaign and repeat controls.
CAMPAIGN_ID="${CAMPAIGN_ID:-ablation_${SLURM_ARRAY_JOB_ID:-local}}"
ABLATION_REPEATS="${ABLATION_REPEATS:-3}"
SEED_BASE="${SEED_BASE:-42}"
TIME_BUFFER_SECONDS="${TIME_BUFFER_SECONDS:-900}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

module load EB/apps EB/install CUDA/12.1.1 || (echo "CUDA module not found" && exit 1)

cd "$PROJECT_ROOT"

# Activate virtualenv if present.
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

# Keep thread counts aligned with allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

time_to_seconds() {
  local t="$1"
  local days=0
  local hours=0
  local mins=0
  local secs=0

  if [ -z "$t" ]; then
    echo "-1"
    return
  fi

  if [ "$t" = "UNLIMITED" ]; then
    echo "2147483647"
    return
  fi

  if [[ "$t" =~ ^([0-9]+)-([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    mins="${BASH_REMATCH[3]}"
    secs="${BASH_REMATCH[4]}"
    echo $((days * 86400 + hours * 3600 + mins * 60 + secs))
    return
  fi

  if [[ "$t" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    hours="${BASH_REMATCH[1]}"
    mins="${BASH_REMATCH[2]}"
    secs="${BASH_REMATCH[3]}"
    echo $((hours * 3600 + mins * 60 + secs))
    return
  fi

  if [[ "$t" =~ ^([0-9]+):([0-9]{2})$ ]]; then
    mins="${BASH_REMATCH[1]}"
    secs="${BASH_REMATCH[2]}"
    echo $((mins * 60 + secs))
    return
  fi

  echo "-1"
}

get_remaining_seconds() {
  if ! command -v squeue >/dev/null 2>&1; then
    echo "-1"
    return
  fi

  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "-1"
    return
  fi

  local remaining_raw
  remaining_raw="$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  time_to_seconds "$remaining_raw"
}

should_stop_for_time() {
  local remaining
  remaining="$(get_remaining_seconds)"

  if [ "$remaining" -lt 0 ]; then
    return 1
  fi

  if [ "$remaining" -le "$TIME_BUFFER_SECONDS" ]; then
    echo "Remaining walltime (${remaining}s) <= TIME_BUFFER_SECONDS (${TIME_BUFFER_SECONDS}s). Stopping repeats."
    return 0
  fi

  return 1
}

wait_for_array_completion() {
  if ! command -v squeue >/dev/null 2>&1; then
    echo "squeue unavailable; skipping array wait gate."
    return
  fi

  if [ -z "${SLURM_ARRAY_JOB_ID:-}" ] || [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    return
  fi

  while true; do
    local pending_count
    pending_count="$(
      squeue -h -j "$SLURM_ARRAY_JOB_ID" -o "%a %T" 2>/dev/null \
        | awk -v self="$SLURM_ARRAY_TASK_ID" '$1 != self {count++} END {print count + 0}'
    )"

    if [ "$pending_count" -eq 0 ]; then
      echo "All sibling array tasks have finished."
      break
    fi

    echo "Waiting for ${pending_count} sibling array tasks before running analysis..."
    sleep 30
  done
}

# Define the ablations. These correspond to Study 4, 5, and 6 in todo/STUDIES.md
declare -a ABLATION_NAMES=(
  "baseline"                             # 0: Full model (baseline)
  "no_sir_loss"                          # 1: Study 4: -L_sir
  "no_ww_loss"                           # 2: Study 4: -L_ww
  "no_cases_loss"                        # 3: Study 4: -L_cases
  "no_hosp_loss"                         # 4: Study 4: -L_hosp
  "no_deaths_loss"                       # 5: Study 4: -L_deaths
  "no_residual_context"                  # 6: Study 5: w/o residual context from encoder (assuming residual_scale=0.0 means off)
  "no_mobility"                          # 7: Study 6: Remove mobility edges/features
  "no_regions"                           # 8: Study 6: Remove static regional embeddings
  "no_mobility_and_regions"              # 9: Study 6: Remove both (local-only)
  "mlp_clinical_heads"                   # 10: Study 5: delay-kernel vs direct MLP mapping (assuming learnable_kernel=false defaults to direct mapping)
)

declare -a ABLATION_OVERRIDES=(
  ""                                                                                # 0: baseline
  "training.loss.joint.w_sir=0.0"                                                   # 1: no_sir_loss
  "training.loss.joint.w_ww=0.0 training.loss.joint.ww_imputed_weight=0.0"          # 2: no_ww_loss
  "training.loss.joint.w_cases=0.0 training.loss.joint.cases_imputed_weight=0.0"    # 3: no_cases_loss
  "training.loss.joint.w_hosp=0.0 training.loss.joint.hosp_imputed_weight=0.0"      # 4: no_hosp_loss
  "training.loss.joint.w_deaths=0.0 training.loss.joint.deaths_imputed_weight=0.0"  # 5: no_deaths_loss
  "model.observation_heads.residual_scale=0.0"                                      # 6: no_residual_context
  "model.type.mobility=false"                                                       # 7: no_mobility
  "model.type.regions=false"                                                        # 8: no_regions
  "model.type.mobility=false model.type.regions=false"                              # 9: no_mobility_and_regions
  "model.observation_heads.learnable_kernel_hosp=false model.observation_heads.learnable_kernel_cases=false model.observation_heads.learnable_kernel_deaths=false" # 10: mlp_clinical_heads
)

# Check if SLURM_ARRAY_TASK_ID is beyond our defined ablations
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ABLATION_NAMES[@]}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of bounds."
  exit 1
fi

ABLATION_NAME="${ABLATION_NAMES[$SLURM_ARRAY_TASK_ID]}"
OVERRIDES="${ABLATION_OVERRIDES[$SLURM_ARRAY_TASK_ID]}"
LAST_TASK_INDEX=$(( ${#ABLATION_NAMES[@]} - 1 ))

echo "Running ablation: $ABLATION_NAME (Task ID: $SLURM_ARRAY_TASK_ID)"
echo "Base config: $CONFIG"
echo "Overrides: $OVERRIDES"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Ablation repeats: $ABLATION_REPEATS"
echo "Seed base: $SEED_BASE"
echo "Time buffer (s): $TIME_BUFFER_SECONDS"
echo "Run analysis: $RUN_ANALYSIS"

# Build override arguments.
OVERRIDE_ARGS=()
if [ -n "$OVERRIDES" ]; then
  for override in $OVERRIDES; do
    OVERRIDE_ARGS+=(--override "$override")
  done
fi

# Override experiment name so logs don't overwrite
# e.g., mn5_ablation__ablation_12345__no_sir_loss
EXPERIMENT_NAME_OVERRIDE="output.experiment_name=mn5_ablation__${CAMPAIGN_ID}__${ABLATION_NAME}"

# Set W&B group and tags for easy comparison across ablations
GROUP_OVERRIDE="output.wandb_group=ablation_campaign_${CAMPAIGN_ID}"

OVERRIDE_ARGS+=(--override "$EXPERIMENT_NAME_OVERRIDE" --override "$GROUP_OVERRIDE")

completed_runs=0

for ((i = 0; i < ABLATION_REPEATS; i++)); do
  if should_stop_for_time; then
    break
  fi

  seed=$((SEED_BASE + i))
  model_id_override="${SLURM_JOB_ID:-local}_a${SLURM_ARRAY_TASK_ID}_s${seed}"
  tags_override="output.wandb_tags=[mn5,ablation,${ABLATION_NAME},campaign_${CAMPAIGN_ID},seed_${seed}]"

  RUN_ARGS=(
    "${OVERRIDE_ARGS[@]}"
    --override "training.seed=${seed}"
    --override "$tags_override"
  )

  echo "----------------------------------------"
  echo "Repeat $((i + 1))/${ABLATION_REPEATS} for $ABLATION_NAME"
  echo "Seed: $seed"
  echo "Model ID override: $model_id_override"
  echo "----------------------------------------"

  # Prefer uv runner if available; fall back to plain python.
  if command -v uv >/dev/null 2>&1; then
    EPIFORECASTER_MODEL_ID="$model_id_override" \
      uv run main train epiforecaster --config "$CONFIG" "${RUN_ARGS[@]}"
  else
    EPIFORECASTER_MODEL_ID="$model_id_override" \
      python -m cli train epiforecaster --config "$CONFIG" "${RUN_ARGS[@]}"
  fi

  completed_runs=$((completed_runs + 1))
done

echo "Completed ${completed_runs} run(s) for ablation '${ABLATION_NAME}'."

# Post-processing: Run analysis on the last array task after siblings finish.
if [ "$RUN_ANALYSIS" = "1" ] && [ "$SLURM_ARRAY_TASK_ID" -eq "$LAST_TASK_INDEX" ]; then
  wait_for_array_completion

  echo ""
  echo "========================================"
  echo "Running ablation analysis on last task..."
  echo "========================================"
  
  ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-outputs/reports/ablation_analysis}"
  
  if command -v uv >/dev/null 2>&1; then
    uv run python scripts/analyze_ablations.py \
      --training-dir outputs/training \
      --output-dir "$ANALYSIS_OUTPUT_DIR" \
      --campaign-id "$CAMPAIGN_ID" \
      --baseline baseline
    
    uv run python -m dataviz.ablation_plots \
      "$ANALYSIS_OUTPUT_DIR/ablation_metrics_aggregated.csv" \
      --deltas-csv "$ANALYSIS_OUTPUT_DIR/ablation_metrics_deltas.csv" \
      --output-dir "$ANALYSIS_OUTPUT_DIR" \
      --baseline baseline
  else
    python scripts/analyze_ablations.py \
      --training-dir outputs/training \
      --output-dir "$ANALYSIS_OUTPUT_DIR" \
      --campaign-id "$CAMPAIGN_ID" \
      --baseline baseline
    
    python -m dataviz.ablation_plots \
      "$ANALYSIS_OUTPUT_DIR/ablation_metrics_aggregated.csv" \
      --deltas-csv "$ANALYSIS_OUTPUT_DIR/ablation_metrics_deltas.csv" \
      --output-dir "$ANALYSIS_OUTPUT_DIR" \
      --baseline baseline
  fi
  
  echo ""
  echo "========================================"
  echo "Ablation analysis complete!"
  echo "Results saved to: $ANALYSIS_OUTPUT_DIR"
  echo "========================================"
fi
