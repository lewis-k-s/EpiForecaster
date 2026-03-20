#!/bin/bash
#SBATCH --job-name=epi-crossval-agg
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
CAMPAIGN_ID="${CAMPAIGN_ID:?CAMPAIGN_ID must be set}"
TRAINING_DIR="${TRAINING_DIR:-outputs/training}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reports/crossval/${CAMPAIGN_ID}/${SPLIT}}"
ANALYZE_VERBOSE="${ANALYZE_VERBOSE:-0}"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/mn5_module_setup.sh"

cd "$PROJECT_ROOT"

echo "Starting crossval aggregate job"
echo "  project_root: $PROJECT_ROOT"
echo "  campaign_id: $CAMPAIGN_ID"
echo "  training_dir: $TRAINING_DIR"
echo "  split: $SPLIT"
echo "  output_dir: $OUTPUT_DIR"
echo "  verbose: $ANALYZE_VERBOSE"

ANALYZE_ARGS=()
if [ "$ANALYZE_VERBOSE" = "1" ]; then
  ANALYZE_ARGS+=(-v)
fi

uv run python scripts/analyze_crossval.py \
  --training-dir "$TRAINING_DIR" \
  --campaign-id "$CAMPAIGN_ID" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  "${ANALYZE_ARGS[@]}"
