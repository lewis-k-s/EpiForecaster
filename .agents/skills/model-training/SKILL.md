---
name: model-training
description: Configure EpiForecaster and Region2Vec training, including the MN5 Snakemake production interface and bounded local acceptance checks.
allowed-tools: Bash, Read
---

# Model Training

## Execution policy

Substantive EpiForecaster training, evaluation, HPO, and ablations run through
the immutable MN5 Snakemake workflow. Load the `mn5-dispatch` skill before any
remote preparation, transfer, submission, monitoring, or result inspection.
Do not invoke production configs directly or add alternate local production
interfaces.

Region2Vec remains available locally:

```bash
uv run train regions --config configs/train_regions.yaml
```

The only local EpiForecaster config is a bounded acceptance check:

```bash
uv run train epiforecaster --config configs/train_epifor_smoke.yaml
```

It is fixed to one epoch, at most ten training batches, batch size two, and one
worker for each train, validation, and test loader. It is not evidence that a
production model can run locally.

## Configuration

EpiForecaster YAML is defined by `EpiForecasterConfig` and its nested dataclasses
in `models/configs.py`. Prefer checked-in templates plus repeatable dot-path
overrides:

```bash
uv run train epiforecaster --config configs/train_epifor_smoke.yaml \
  --override model.type.mobility=false
```

Production workflow templates live under `workflow/config/` and point to model
configs under `configs/production_only/`. The public workflow entities are
single training, evaluation, composed train-evaluate, cross-validation, HPO,
ablation, and synthetic pretraining. Synthetic pretraining uses
`workflow/config/mn5_synth_pretrain.yaml`; its raw synthetic pointer contributes
provenance without being staged into every training job.
Training and evaluation remain separate DAG stages with hash-bearing manifests.

## Data pipeline

The canonical data flow is raw inputs, preprocessing, DVC-tracked Zarr outputs,
then workflow staging. A dataset is not a production workflow input until its
DVC pointer is checked in. Inspect Zarr only with `xarray.open_zarr()`.

```bash
uv run preprocess epiforecaster --config configs/preprocess_full.yaml
uv run preprocess regions \
  --geojson data/files/geo/fl_municipios_catalonia.geojson \
  --population data/files/fl_population_por_municipis.csv \
  --output outputs/region_graph/region_graph.zarr
```

Do not run `configs/production_only/` locally. For MN5 commands and the exact
prepare/run/status/log workflow, defer to the `mn5-dispatch` skill rather than
duplicating dispatch instructions here.
