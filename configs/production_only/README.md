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
- Invoked only through `workflow/config/mn5_synth_pretrain.yaml`
- Enables direct latent `S/I/R/D` supervision for stage-1 pretraining

### `preprocess_mn5_synth.yaml`
- Canonical MN5 synthetic preprocessing base
- Reads the DVC-tracked raw synthetic Zarr on project GPFS

### `preprocess_mn5_synth_holt_damped.yaml`
- Produces the canonical processed pretraining dataset
- Inherits the MN5 synthetic preprocessing base

## Intended Usage

These configs should only be run through an immutable MN5 Snakemake workflow:

```bash
# Single training (default workflow target)
bash scripts/workflow/mn5_workflow.sh prepare <run-id>
bash scripts/workflow/mn5_workflow.sh run <run-id>

# Compose training and evaluation in the same immutable run by setting
# workflow.target: train_evaluate in a complete workflow config before prepare.
bash scripts/workflow/mn5_workflow.sh prepare <run-id> \
  --config <train-evaluate-workflow-config.yaml>
bash scripts/workflow/mn5_workflow.sh run <run-id>

# HPO
bash scripts/workflow/mn5_workflow.sh prepare <hpo-run-id> \
  --config workflow/config/mn5.yaml

# Ablation study
bash scripts/workflow/mn5_workflow.sh prepare <ablation-run-id> \
  --config workflow/config/mn5_ablation.yaml

# Synthetic pretraining (after the processed DVC pointer is published)
bash scripts/workflow/mn5_workflow.sh prepare <pretrain-run-id> \
  --config workflow/config/mn5_synth_pretrain.yaml
```

The default target is single training. Pin `train_evaluate` or `evaluate` in a
complete workflow config before preparation when evaluation is required. Use
`--config workflow/config/mn5.yaml` for HPO or
`--config workflow/config/mn5_ablation.yaml` for ablations. Direct production
`train` and `eval` commands are implementation details of these Snakemake rules,
not separate submission interfaces.

The checked-in production model configs are deliberately limited to the
canonical real-data model and canonical synthetic pretraining model. Historical
experiment variants belong in immutable run manifests and overrides rather
than additional evergreen YAML files.

## Local Acceptance

Substantive EpiForecaster work uses the MN5 Snakemake interface. The sole local
training config is a bounded parser/runtime acceptance check:

- `configs/train_epifor_smoke.yaml` - 10 training batches with one worker per loader
