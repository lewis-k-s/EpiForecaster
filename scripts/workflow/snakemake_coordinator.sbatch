#!/bin/bash
#SBATCH --job-name=epi-snakemake
#SBATCH --account=bsc08
#SBATCH --partition=acc
#SBATCH --qos=acc_bscls
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=07:00:00

# Thin batch adapter for the shared allocation-internal coordinator.
#
# Slurm copies this file into its node-local spool before execution, so
# BASH_SOURCE no longer has the source snapshot as its parent. The submitter
# therefore passes the immutable shared coordinator path explicitly.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Missing shared coordinator path" >&2
  exit 2
fi

COORDINATOR="$1"
shift
if [[ "$COORDINATOR" != /* ]] || [ ! -x "$COORDINATOR" ]; then
  echo "Shared coordinator must be an executable absolute path: $COORDINATOR" >&2
  exit 2
fi

exec "$COORDINATOR" "$@"
