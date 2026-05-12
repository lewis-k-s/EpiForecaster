#!/bin/bash
#SBATCH --job-name=epi-preprocess
#SBATCH --account=bsc08
#SBATCH --qos=gp_bscls
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/preprocess_full.yaml}"
PREPROCESS_TASK="${PREPROCESS_TASK:-epiforecaster}"

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

$UV_RUN preprocess "$PREPROCESS_TASK" --config "$CONFIG"
