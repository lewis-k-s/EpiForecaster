---
name: model-training
description: Train EpiForecaster and Region2Vec models locally or remotely on MN5. Covers config-driven YAML workflow, CLI commands, override syntax, and the dataset pipeline. Use when the user wants to run, configure, debug, or iterate on model training.
allowed-tools: Bash, Read
---

# Model Training

End-to-end reference for training EpiForecaster and Region2Vec models — locally or remotely on MN5.

## 1. Quick Start

```bash
# Region2Vec embedding pretraining
uv run train regions --config configs/train_regions.yaml

# EpiForecaster training (synthetic, local)
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml

# Smoke test (truncate after N batches)
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml --max-batches 5
```

## 2. CLI Reference

All commands are invoked via `uv run`. Run `--help` on any subcommand for the full current option reference.

### Command Tree

```
uv run main --help                          # top-level group
uv run train regions --help                 # region embedder
uv run train epiforecaster --help           # epi forecaster
uv run preprocess regions --help            # region graph builder
uv run preprocess epiforecaster --help      # data preprocessing pipeline
```

### Key Flags

| Subcommand | Flag | Purpose |
|------------|------|---------|
| `train epiforecaster` | `--config <yaml>` | Training config (required) |
| | `--resume <checkpoint>` | Resume from checkpoint |
| | `--max-batches <N>` | Truncate run after N batches (smoke test) |
| | `--override key=value` | Override config values (repeatable, dot-path syntax) |
| `train regions` | `--config <yaml>` | Region config (required) |
| | `--epochs <N>` | Override epoch count |
| | `--device <str>` | Override compute device |
| | `--output-dir <path>` | Override output directory |
| | `--dry-run` | Load config and init trainer without running |
| | `--no-cluster` | Skip post-training agglomerative clustering |
| `preprocess epiforecaster` | `--config <yaml>` | Preprocessing config (required) |
| `preprocess regions` | CLI flags only (no YAML) | Builds region graph from GeoJSON + population + mobility |

### Override Syntax

Use `--override` with dot-path keys to override any config value from the CLI:

```bash
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml \
  --override training.epochs=5 \
  --override training.learning_rate=0.001 \
  --override training.device=cpu \
  --override model.type.mobility=false
```

Multiple `--override` flags are allowed. Values are parsed as YAML (so `true`/`false` are booleans, numbers are numeric, quoted strings are strings).

## 3. Dataset Workflow

The canonical pipeline: **raw data → preprocessing → Zarr datasets → training**

### Preprocessing

```bash
# Build region graph (adjacency, flows, features)
uv run preprocess regions \
  --geojson data/files/geo/fl_municipios_catalonia.geojson \
  --population data/files/fl_population_por_municipis.csv \
  --output outputs/region_graph/region_graph.zarr

# Full epi forecaster preprocessing pipeline
uv run preprocess epiforecaster --config configs/preprocess_full.yaml
```

### Training

```bash
# 1. Train region embeddings first (needed by epi forecaster if model.type.regions=true)
uv run train regions --config configs/train_regions.yaml

# 2. Train the epi forecaster
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml
```

## 4. Config-Driven Development

All training runs are driven by YAML configuration files for reproducibility.

### Workflow

1. Copy a template from `configs/` and modify for your experiment
2. Pass the config via `--config <path>` to the train subcommand
3. Override specific values at the CLI level with `--override key=value` rather than editing the YAML
4. Config parsing goes through Python dataclasses — keep the YAML schema and dataclass fields in sync; add/adjust docstrings when new config keys are introduced

### Config Schema Source

| Model | Config Dataclass | Source File |
|-------|-----------------|-------------|
| EpiForecaster | `EpiForecasterConfig`, `ModelConfig`, `DataConfig`, `TrainingParams` | `models/configs.py` |
| Region2Vec | `RegionTrainerConfig`, `EncoderConfig`, `SamplingConfig`, `LossConfig` | `training/region2vec_trainer.py` |

When you need to know what fields are available or what the defaults are, read the source dataclass — it is the authoritative reference.

### Nested Config Sections (EpiForecaster YAML)

```yaml
model:        # ModelConfig — architecture, variant flags, SIR physics
  type:       # ModelVariant — cases, regions, biomarkers, mobility booleans
  sir_physics:  # SIRPhysicsConfig
data:         # DataConfig — dataset paths, missing permits, windowing
  missing_permit:  # MissingPermitConfig
training:     # TrainingParams — optimizer, scheduler, loss, curriculum, device
  curriculum:    # CurriculumConfig
  loss:          # LossConfig → JointLossConfig
  profiler:      # ProfilerConfig
output:       # OutputConfig — logging, checkpoints, wandb
```

## 5. Config Templates

### Local Development (`configs/`)

| Config | Purpose |
|--------|---------|
| `train_epifor_synth_local.yaml` | EpiForecaster on synthetic data (lightweight) |
| `train_epifor_real_local.yaml` | EpiForecaster on real data (local) |
| `train_epifor_curriculum.yaml` | EpiForecaster with curriculum schedule |
| `train_regions.yaml` | Region2Vec embedding pretraining |
| `preprocess_full.yaml` | Full preprocessing pipeline |
| `preprocess_mn5_synth.yaml` | Synthetic preprocessing config |
| `preprocess_real_holt.yaml` | Real data preprocessing (Holt-smoothed) |

### Production (`configs/production_only/`)

> **WARNING**: Do not run `configs/production_only/` configs locally. They require 100GB+ memory and GPU cluster resources and will OOM on dev machines. See `configs/production_only/README.md`.

## 6. Local Training

### Region2Vec

```bash
# Standard training
uv run train regions --config configs/train_regions.yaml

# Quick dry run to verify config loads
uv run train regions --config configs/train_regions.yaml --dry-run

# Override epochs and device
uv run train regions --config configs/train_regions.yaml --epochs 50 --device cpu
```

Outputs: region embeddings (`outputs/region_embeddings/region_embeddings.pt`), training metrics JSON, and optional cluster labels.

### EpiForecaster

```bash
# Full training run
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml

# Quick smoke test
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml --max-batches 3

# Resume from checkpoint
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml --resume outputs/training/<run>/checkpoints/best_model.pt

# Override training params
uv run train epiforecaster --config configs/train_epifor_synth_local.yaml \
  --override training.epochs=2 \
  --override training.batch_size=4 \
  --override training.device=cpu
```

### Device Selection

The `training.device` config key (or `--override training.device=...`) controls compute device:
- `"auto"` — auto-detect (CUDA > MPS > CPU)
- `"cpu"` — force CPU (safe for smoke tests and macOS)
- `"cuda"` / `"mps"` — explicit GPU backend

### MN5 Guard

Configs with `env: "mn5"` will refuse to run locally. The CLI prints an error directing you to use the dispatch script. For local runs, either use a local config or override: `--override env=null`.

## 7. Remote Training (MN5)

For full remote dispatch details, load the **`mn5-dispatch`** skill. The typical loop:

```bash
# 1. Sync code to remote (via dt transfer node)
bash syncto_mn5.sh

# 2. Submit training job (via mn5 login node)
ssh mn5 'CONFIG=configs/production_only/train_epifor_mn5_full.yaml \
  /home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh submit single --time=08:00:00'

# 3. Monitor
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh status <JOB_ID>'
ssh mn5 '/home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh tail <JOB_ID>'

# 4. Sync results back
bash syncback_from_mn5.sh <experiment_name>
```

Overrides work the same way via the `OVERRIDES` env var:
```bash
ssh mn5 'CONFIG=configs/train.yaml OVERRIDES="training.epochs=5 training.batch_size=48" \
  /home/bsc/bsc008913/EpiForecaster/scripts/cluster/mn5_dispatch.sh submit single --time=04:00:00'
```

## 8. Troubleshooting

| Problem | Check |
|---------|-------|
| Config OOM on local | Use a local config, reduce `batch_size`, disable mixed precision, or use `--max-batches` to truncate |
| `env: "mn5"` guard blocks local run | Override with `--override env=null` or use a local config |
| Region embeddings not found | Run `uv run train regions` first; verify `data.region2vec_path` in config |
| Dataset path errors | Check `data.dataset_path` points to a valid Zarr; use `xarray.open_zarr()` to inspect |
| Loss spikes / instability | Load the `training-loss-analysis` skill for TensorBoard diagnostics |
| Slow CPU training | Reduce `batch_size`, disable `enable_mixed_precision`, set `num_workers=0` |
