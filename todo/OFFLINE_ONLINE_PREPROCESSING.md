# Offline/Online Preprocessing Separation

## Background

Synthetic runs share a common mobility base. Mobility variation is limited unless
we sample across many synthetic runs. Today, `EpiDataset` performs global
preprocessing (cases scaling, rolling stats, mobility scaling, biomarker scaling)
per run at load time, which forces us to load full series into memory. That makes
random run sampling expensive and limits batch size.

If we move global series transforms offline, the training-time dataset can sample
different runs without loading each run’s full series, while keeping only
window-level transforms online. This unlocks higher synthetic variation at lower
memory cost.

## Proposal

### 1) Rename “processors” to “loaders”

The current offline `data/preprocess/processors/` modules primarily:
- read raw data formats,
- align them to the canonical schema,
- and assemble the aligned dataset.

Rename this layer to clarify that it is ingestion and alignment, not feature
engineering.

Suggested path:
- `data/preprocess/processors/` -> `data/preprocess/loaders/`

Keep module names largely unchanged (e.g., `cases_processor.py` -> `cases_loader.py`)
so call sites remain obvious.

### 2) Define offline feature engineering stage

Move global series transforms from online dataset classes into offline
preprocessing, producing derived arrays stored in the aligned Zarr.

Targets to move offline (non-exhaustive):
- `data/cases_preprocessor.py` -> compute `cases_processed`, `cases_roll_mean`,
  `cases_roll_std` offline.
- `data/mobility_preprocessor.py` -> store scaled/log mobility, optional clip
  bounds, and any global scaling artifacts.
- `data/biomarker_preprocessor.py` -> store scaled biomarker features where
  possible (per-variant means/stds, masks, age channels).

Online step keeps only window-level transforms:
- normalization anchored to a window (e.g., `make_normalized_window()`)
- subsetting windows/targets
- lightweight masking and concatenation

### 3) Exception: EDAR stays offline

`data/preprocess/processors/edar_processor.py` performs global imputation and
scaling (Tobit-Kalman). That belongs in offline preprocessing and should remain
there as a “feature engineering” exception inside the loader stage.

## Concrete artifacts to add to aligned Zarr

Cases:
- `cases_processed` (T, N, 3)
- `cases_roll_mean` (T, N, 1)
- `cases_roll_std` (T, N, 1)

Mobility:
- `mobility_scaled` (T, N, N) or scaled base + kappa applied (if stored)
- optional: `mobility_mask` if precomputable and stable

Biomarkers:
- `edar_biomarker_*_processed` (aligned and scaled as needed)
- retain raw fields if needed for debugging

## Config implications

- Commit to a fixed `history_length` for synthetic preprocessing.
- Remove `history_length` from hyperparameter search space for synthetic runs.
- Add a preprocessing config flag for `offline_feature_engineering: true` and
  a version tag for the schema to guard compatibility.

## Training-time behavior

- `EpiDataset` checks for precomputed arrays first.
- If present, skip global preprocessing and load the tensors directly.
- If missing, fallback to current online preprocessing (backward compatible).

## Why this helps

- Random run sampling is cheaper (no per-run full-series preprocessing).
- Memory usage drops (training only touches windows and precomputed arrays).
- Scaling is consistent between offline and training, improving reproducibility.

## Migration plan

1) Add loader rename (processors -> loaders) and update imports.
2) Add offline case feature generation and write fields to aligned Zarr.
3) Update `EpiDataset` to use offline fields when present.
4) Extend synthetic preprocessing config to include fixed `history_length`.
5) Update documentation and remove `history_length` from synthetic HP search.

## Open questions

- Should mobility scaling be stored as a full tensor or as base + kappa terms?
- Should we persist scaler metadata (mean/std) for auditability?
- What is the preferred versioning mechanism for aligned dataset schemas?
