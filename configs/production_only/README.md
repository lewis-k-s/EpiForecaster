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

### `train_epifor_mn5_synth_pretrain.yaml`
- Production synthetic pretraining on processed synthetic data
- Uses the same MN5 training surface as the real-data template
- Enables direct latent `S/I/R/D` supervision for stage-1 pretraining

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
sbatch scripts/cluster/train_single_gpu.sbatch

# For synthetic pretraining with the regular single-GPU sbatch wrapper
CONFIG=configs/production_only/train_epifor_mn5_synth_pretrain.yaml \
sbatch scripts/cluster/train_single_gpu.sbatch

# Convenience wrapper for the same synth-pretrain submission
scripts/cluster/submit_synth_pretrain.sh

# Submit synth pretraining, then fine-tune on real data after the pretrain job succeeds
scripts/cluster/submit_pretrain_then_finetune.sh

# For hyperparameter optimization
sbatch scripts/cluster/optuna_epiforecaster.sbatch
```

The chained submission helper uses a fixed `EPIFORECASTER_MODEL_ID` for the
pretrain run so the fine-tune job can point `training.init_checkpoint_path` at
`.../checkpoints/best_model.pt` deterministically.

The MN5 batch scripts share a common module bootstrap in `scripts/cluster/mn5_module_setup.sh`.
That setup loads the GCC, CUDA, and CMake modules needed for both runtime jobs and
optional ONNX simplification tooling installation.

## Development Alternatives

For local development, use the standard configs instead:

- `configs/train_epifor_full.yaml` - Full model on real data (local-scale)
- `configs/train_epifor_temporal.yaml` - Temporal model (local-scale)
- `configs/train_epifor_synth_local.yaml` - Synthetic data training (local-scale)
