# Production-Only Configurations

**WARNING: These configurations are designed for MN5-scale computing environments.**

## Critical Warnings
- **DO NOT RUN LOCALLY**: Requires 100GB+ memory and GPU cluster resources
- **OOM Risk**: Running on local machines will cause out-of-memory crashes

## Configuration Files

### `train_epifor_mn5_full.yaml`
- Production training on real data
- Resources: CUDA GPU, 20 CPUs, 64 batch size
- Memory: ~80GB RAM recommended

### `train_epifor_mn5_synth.yaml`
- Production training on synthetic data
- Resources: CUDA GPU, reduced batch (8), 4 workers
- Memory: ~40GB RAM minimum

### `preprocess_mn5_synth.yaml`
- Preprocess synthetic data at MN5 scale
- Memory limit: 100GB configured

## Intended Usage

These configs should only be run on the MN5 cluster via SLURM scripts:

```bash
# For training
sbatch scripts/train_single_gpu.sbatch

# For hyperparameter optimization
sbatch scripts/optuna_epiforecaster.sbatch
```

## Development Alternatives

For local development, use the standard configs instead:

- `configs/train_epifor_full.yaml` - Full model on real data (local-scale)
- `configs/train_epifor_temporal.yaml` - Temporal model (local-scale)
- `configs/train_epifor_synth_local.yaml` - Synthetic data training (local-scale)
