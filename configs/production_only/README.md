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

These configs should only be run through an immutable MN5 Snakemake workflow:

```bash
# Single training (default workflow target)
bash scripts/workflow/mn5_workflow.sh prepare <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>

# Compose training and evaluation in the same immutable run
bash scripts/workflow/mn5_workflow.sh run <run-id> train_evaluate

# HPO
bash scripts/workflow/mn5_workflow.sh prepare <hpo-run-id> \
  --config workflow/config/mn5.yaml

# Ablation study
bash scripts/workflow/mn5_workflow.sh prepare <ablation-run-id> \
  --config workflow/config/mn5_ablation.yaml
```

The default target is single training. Use the public `train_evaluate` or
`evaluate` target when evaluation is required. Use
`--config workflow/config/mn5.yaml` for HPO or
`--config workflow/config/mn5_ablation.yaml` for ablations. Direct production
`train` and `eval` commands are implementation details of these Snakemake rules,
not separate submission interfaces.

To use another production training config, copy
`workflow/config/mn5_train.yaml` and change `training.base_config`; the same
source-aware training and evaluation rules and deterministic artifact contracts
apply.

## Development Alternatives

For local development, use the standard configs instead:

- `configs/train_epifor_real_local.yaml` - Real-data training (local-scale)
- `configs/train_epifor_synth_local.yaml` - Synthetic data training (local-scale)
