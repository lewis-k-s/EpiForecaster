DEFAULT_EXPERIMENT_NAME="mn5_epiforecaster_full"
EXPERIMENT_NAME="${1:-$DEFAULT_EXPERIMENT_NAME}"

REMOTE_ROOT="dt:/home/bsc/bsc008913/EpiForecaster"
SYNC_WANDB="${SYNC_WANDB:-1}"

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

if [ "$SYNC_WANDB" = "1" ]; then
  if command -v wandb >/dev/null 2>&1; then
    echo "Syncing W&B offline runs..."
    find outputs/training outputs/region_training -type d -name 'offline-run-*' -prune \
      -exec wandb sync {} +
    find wandb -type d -name 'offline-run-*' -prune -exec wandb sync {} + 2>/dev/null || true
  else
    echo "wandb CLI not found. Skipping upload. Set SYNC_WANDB=0 to silence."
  fi
fi
