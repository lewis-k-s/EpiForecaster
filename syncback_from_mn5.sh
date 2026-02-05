DEFAULT_EXPERIMENT_NAME="mn5_epiforecaster_full"
EXPERIMENT_NAME="${1:-$DEFAULT_EXPERIMENT_NAME}"

REMOTE_ROOT="dt:/home/bsc/bsc008913/EpiForecaster"

# Sync training runs (includes per-run wandb offline logs under each run dir).
rsync -avz --progress \
    "${REMOTE_ROOT}/outputs/training/${EXPERIMENT_NAME}" \
    ./outputs/training/

# Sync region training runs (if any) for the same experiment name.
rsync -avz --progress \
    "${REMOTE_ROOT}/outputs/region_training/${EXPERIMENT_NAME}" \
    ./outputs/region_training/ || true

# Sync repo-level wandb offline logs (if created by ad-hoc runs).
rsync -avz --progress \
    "${REMOTE_ROOT}/wandb" \
    ./wandb/ || true
