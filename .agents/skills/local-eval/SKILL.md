---
name: local-eval
description: Evaluate MN5-trained EpiForecaster models locally with automatic dataset path overrides. Use when user wants to run eval on a checkpoint that was trained on the remote cluster (GPFS/MN5).
allowed-tools: Bash, Read
---

# Local Model Evaluation

Evaluates EpiForecaster checkpoints trained on MN5/GPFS locally, automatically applying dataset path and worker overrides for local execution.

## Quick Start

```
local-eval <checkpoint_path>
local-eval <checkpoint_path> --split val
local-eval <checkpoint_path> --device cpu --output eval/test.png
```

## Purpose

Models trained on MN5 have dataset paths like `/scratch/tmp/...` that don't exist locally. This skill wraps the CLI eval entrypoint with the necessary overrides to run evaluation against the local canonical dataset.

## Default Overrides

| Override | Value | Reason |
|----------|-------|--------|
| `data.dataset_path` | `data/processed/real_with_id.zarr` | Local canonical dataset |
| `training.val_workers` | `0` | macOS multiprocessing compatibility |
| `training.test_workers` | `6` | Parallel test dataloading |
| `model.include_day_of_week` | `true` | Match MN5 training temporal covariates |
| `model.include_holidays` | `true` | Match MN5 training temporal covariates |

Device defaults to `cpu` for memory safety on macOS. Use `--override training.device=auto` for GPU if available.

## Command Construction

The skill translates user input to:

```bash
.venv/bin/python -m cli eval epiforecaster \
  --checkpoint <checkpoint_path> \
  --split <val|test> \
  --override data.dataset_path=data/processed/real_with_id.zarr \
  --override training.val_workers=0 \
  --override training.test_workers=6 \
  --override training.device=cpu \
  --override model.include_day_of_week=true \
  --override model.include_holidays=true \
  [... user overrides ...]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--split` | Which split to evaluate | `test` |
| `--device` | Device override (cpu, cuda, auto) | `cpu` |
| `--output` | Output plot path | auto-resolved |
| `--output-csv` | Node metrics CSV path | auto-resolved |
| `--override` | Additional config overrides | - |
| `--eval-batch-size` | Override batch size | from config |

## Examples

### Basic evaluation (test split)
```bash
local-eval outputs/training/mn5_fresh/36532668/checkpoints/best_model.pt
```

### Validation split
```bash
local-eval outputs/training/mn5_fresh/36532668/checkpoints/best_model.pt --split val
```

### Use GPU (auto-detect)
```bash
local-eval checkpoints/best_model.pt --override training.device=auto
```

### Custom output location
```bash
local-eval checkpoints/best_model.pt --output reports/eval.png --output-csv reports/metrics.csv
```

### Smaller batch size for memory constraints
```bash
local-eval checkpoints/best_model.pt --eval-batch-size 8
```

## Output

The evaluation produces:
1. **Console output**: Evaluation metrics (loss, MAE, RMSE, sMAPE, R²)
2. **Node metrics CSV**: Per-node MAE and sample counts
3. **Forecast plots**: Quartile-based visualization of predictions

## Typical Workflow

1. User identifies a trained checkpoint from MN5 (e.g., `outputs/training/mn5_fresh/36532668/checkpoints/best_model.pt`)
2. Agent runs `local-eval` with appropriate options
3. Agent reviews evaluation metrics and generated plots
4. Agent can iterate with different splits or overrides as needed

## Notes

- The local dataset must exist at `data/processed/real_with_id.zarr`
- Node splits are recomputed using the same seed (42) as training
- Region embeddings are loaded from `outputs/region_embeddings/region_embeddings.pt` if configured
- `val_workers=0` avoids multiprocessing issues with zarr on some systems
- `test_workers=6` enables parallel loading during evaluation
- CPU evaluation is slow (~10+ min for full test set); consider `--eval-batch-size` to reduce memory if using GPU
- The `include_day_of_week` and `include_holidays` overrides are needed because MN5 training data includes 3 temporal covariates
