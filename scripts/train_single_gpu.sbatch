#!/bin/bash
#SBATCH --job-name=epi-forecaster
#SBATCH --account=bsc08
#SBATCH --qos=acc_bscls
#SBATCH --gres=gpu:1                # one CUDA-capable GPU
#SBATCH --cpus-per-task=20          # must be full node
#SBATCH --time=04:00:00             # wall-clock limit
#SBATCH --output=slurm-%x-%j.out    # stdout/stderr log per job

# Fail fast on errors; print commands for easier debugging.
set -euo pipefail

# Root of the repository (defaults to submission directory).
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
# Path to the training config file (override per run).
CONFIG="${CONFIG:-configs/train_epifor_mn5_full.yaml}"
# Path to an existing virtual environment (override or comment out if using modules).
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

module load EB/apps EB/install CUDA/12.1.1 || (echo "CUDA module not found" && exit 1)

cd "$PROJECT_ROOT"

# Activate virtualenv if present.
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

# Keep thread counts aligned with allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Prefer uv runner if available; fall back to plain python.
if command -v uv >/dev/null 2>&1; then
  uv run main train epiforecaster --config "$CONFIG"
else
  python -m cli train epiforecaster --config "$CONFIG"
fi
