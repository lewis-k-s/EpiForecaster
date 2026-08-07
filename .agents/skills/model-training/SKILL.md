---
name: model-training
description: Configure EpiForecaster and Region2Vec training, validate bounded local acceptance runs, and route substantive training, evaluation, cross-validation, HPO, ablation, and synthetic pretraining through the immutable MN5 Snakemake interface.
---

# Model Training

## Execution boundary

Substantive EpiForecaster work uses `scripts/workflow/mn5_workflow.sh`. Load
`mn5-cluster-usage` before remote preparation, transfer, submission, monitoring,
or artifact inspection. It defines the live MN5 endpoint and allocation policy;
this skill and `docs/MN5_SNAKEMAKE.md` define the immutable workflow interface.

Canonical/general-purpose templates:

| Template | Kind | Target | Purpose |
| --- | --- | --- | --- |
| `mn5_train.yaml` | `train` | `train` | single training |
| `mn5_train_smoke.yaml` | `train` | `train_evaluate` | bounded train/evaluate acceptance |
| `mn5_evaluate.yaml` | `evaluate` | `evaluate` | pinned-checkpoint evaluation |
| `mn5_crossval.yaml` | `crossval` | `crossval` | persisted-fold cross-validation |
| `mn5.yaml` | `hpo` | `hpo` | HPO |
| `mn5_smoke.yaml` | `hpo` | `hpo` | bounded HPO acceptance |
| `mn5_ablation.yaml` | `ablation` | `ablation` | ablation study |
| `mn5_synth_pretrain.yaml` | `train` | `train` | synthetic pretraining |

Targets are pinned before immutable preparation. Do not run configs under
`configs/production_only/` locally.
Treat any specialized or one-off workflow template outside this matrix as an
explicit reproduction/profiling interface and inspect it before use.

## Local acceptance

The local EpiForecaster command is acceptance-only:

```bash
uv run --no-sync train epiforecaster \
  --config configs/train_epifor_smoke.yaml
```

`--max-batches` may tighten its bound but must not expand it into a production
interface.

Region2Vec has a separate local interface. Inspect it before substantive work:

```bash
uv run --no-sync train regions \
  --config configs/train_regions.yaml \
  --dry-run
```

Full Region2Vec training is potentially expensive and requires explicit user
authorization.

## Data and provenance

Production preparation requires every materialized input's checked-in DVC
pointer and cache object. Distinguish:

- materialized epidemiological and mobility Zarr inputs;
- materialized Region2Vec weights;
- provenance-only raw synthetic inputs;
- generated outputs that are not production-ready until their DVC pointer is
  published.

Synthetic pretraining is unavailable while
`data/processed/synthetic_full_holt_damped.zarr.dvc` is absent. Report that gate;
do not create or publish the pointer implicitly.

Preprocessing is a potentially expensive mutation. In particular,
`configs/preprocess_full.yaml` overwrites its configured canonical Zarr. Inspect
worktree/DVC state and obtain explicit authorization before running it. Inspect
Zarr only with `xarray.open_zarr()`.

Every EpiForecaster training run, HPO trial, and cross-validation fold writes
canonical `training_dynamics.jsonl` plus its manifest. Use
`training-dynamics-analyze` for dynamics evidence and `hpsearch-analyze` for
Optuna study evidence.
