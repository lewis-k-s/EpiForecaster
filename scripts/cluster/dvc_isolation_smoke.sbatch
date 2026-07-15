#!/bin/bash
#SBATCH --job-name=epi-dvc-smoke
#SBATCH --account=bsc08
#SBATCH --partition=gpp
#SBATCH --qos=gp_debug
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RUN_ID="${RUN_ID:?RUN_ID must identify the isolated remote workspace}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must point to the shared MN5 environment}"
DVC_BIN="${DVC_BIN:?DVC_BIN must point to the shared MN5 environment}"

cd "$PROJECT_ROOT"

"$DVC_BIN" status \
  data/files/mobility.zarr.dvc \
  data/processed/real_with_id.zarr.dvc

"$PYTHON_BIN" scripts/cluster/dvc_isolation_probe.py \
  --run-id "$RUN_ID" \
  --project-root "$PROJECT_ROOT"
