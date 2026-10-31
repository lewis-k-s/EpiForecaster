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

### `train_epifor_mn5_synth_pretrain_reactive_n100.yaml`
- Production synthetic pretraining on the processed a97c812 reactive corpus
- Invoked only through `workflow/config/mn5_synth_pretrain_reactive_n100.yaml`
- Enables direct latent `S/I/H/R/death_flow` supervision for stage-1 pretraining

### `preprocess_mn5_synth.yaml`
- Canonical MN5 synthetic preprocessing base
- Reads the DVC-tracked raw synthetic Zarr on project GPFS

### `preprocess_mn5_synth_sirhd_v1_a97c812_reactive_n100.yaml`
- Produces the processed a97c812 reactive corpus (the only synthetic corpus)
- Keeps clinical observations as continuous expected reported counts and stores
  float32 processed supervision so small positive clinical expectations and
  latent fractions cannot underflow back to zero
- Represents missingness only through masks and NaNs and stores wastewater as
  raw concentration for exactly one downstream `log1p`

## Intended Usage

Training and evaluation configs run through the immutable MN5 Snakemake
workflow. Preprocessing is the upstream DVC-producing boundary: run it from an
immutable source bundle in a versioned MN5 run root, validate the generated
Zarr, and publish a new DVC pointer without modifying the source bundle or an
existing artifact.

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
  --config workflow/config/mn5_synth_pretrain_reactive_n100.yaml
```

The pretraining study uses fail-closed templates for initialized
fine-tuning. Copy
`workflow/config/mn5_synth_pretrain_finetune_smoke.template.yaml` for the
bounded real-data smoke, or
`workflow/config/mn5_synth_pretrain_finetune.template.yaml` for the substantive
treatment. Set `training.init_checkpoint` to the absolute GPFS checkpoint from
the accepted pretraining run before immutable preparation.
The workflow validator rejects either template while that field is null, so it
cannot silently execute as a second from-scratch control.

Use `workflow/config/mn5_synth_pretrain_control_smoke.yaml` as the exactly
matched from-scratch smoke.

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
