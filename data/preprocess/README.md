# Preprocessing Pipeline Guide

This document summarizes how `data/preprocess/pipeline.py` converts raw epidemiological sources into the canonical Zarr dataset that every downstream experiment consumes. The pipeline no longer emits variant-specific layouts—every run writes the same tensor structure so model variants (base, mobility-aware, region embeddings, etc.) can decide how to interpret the shared features.

## Inputs and Canonical Variables

| Variable | Shape | Notes |
| --- | --- | --- |
| `population` | `(region,)` (persisted as `(region, 1)` in Zarr) | Static per administrative region. Sourced from the case metadata. Required for per-100k normalization in preprocessing. |
| `cases` | `(time, region)` | Daily (or configured cadence) case counts. Required input that defines the master time/region grid. |
| `edar_biomarker_<variant>` | `(time, region)` | Per-variant wastewater signal (e.g., N1, N2, IP4) aggregated from EDAR sites to target regions using the contribution matrix. |
| `mobility` | `(time, region, region)` | Origin–destination matrix per timestamp. Optional but expected for the standard build; placeholder tensors are created if mobility is missing so the output schema stays stable. |

Because `cases` and EDAR biomarkers resolve to `(time, region)` series, the combined feature width for temporal signals depends on the number of biomarker variants.

## Processing Stages

1. **Raw loaders (`CasesProcessor`, `EDARProcessor`, `MobilityProcessor`)** – Each processor reads its source (CSV, NetCDF, etc.), performs source-specific cleaning, and emits tensors plus metadata (date ranges, region lists).
2. **Alignment (`AlignmentProcessor`)** – Harmonizes temporal indices to the cases timeline and uses `region_metadata` to translate biomarker/mobility regions into the case region ID space. Coverage stats and any dropped rows are tracked in the stage report.
3. **Static transforms** – All continuous series are log1p-transformed before dtype conversion to prevent float16 overflow:
   - **Clinical series** (`cases`, `hospitalizations`, `deaths`): log1p(per-100k) using population
   - **Mobility**: log1p only
   - **Biomarker values**: log1p only
4. **Validation and storage** – Consistency checks guarantee every batch has identical node/edge counts and no NaNs before persisting via `DatasetStorage.save_dataset()`.
5. **Reporting** – A JSON report (same directory as the Zarr store) captures configuration, alignment outcomes, statistics, and warnings for reproducibility.

## Causal Smoothing Modes

The preprocessing pipeline supports configurable **offline but causal** smoothing.
Each timestep is updated using only current/past observations (no look-ahead over
future dates).

### Clinical series (`cases`, `hospitalizations`, `deaths`)

| Method | Config key | Behavior |
| --- | --- | --- |
| Kalman v2 | `smoothing.clinical_method: "kalman_v2"` | Random-walk Kalman filter with configurable missing policy (`predict` default, optional bounded `momentum`), innovation clipping, and re-entry gain capping to reduce hard jump artifacts after sparse gaps. |
| Damped Holt | `smoothing.clinical_method: "holt_damped"` | Causal level+trend smoother (log space) with damped trend carry through missing intervals to avoid plateaus and spikes. |

### Wastewater series (EDAR variants)

| Method | Config key | Behavior |
| --- | --- | --- |
| Tobit Kalman v2 | `smoothing.wastewater_method: "tobit_kalman_v2"` | Causal Tobit-Kalman filtering for censored observations (LOD-aware), with the same missing-policy and robust re-entry controls as Kalman v2. |

## Wastewater (EDAR) Processing Methodology

The wastewater processor (`EDARProcessor`) implements a sophisticated pipeline to handle the unique challenges of environmental surveillance data, particularly sparsity and limits of detection (LOD).

### 1. Data Cleaning & Flow Calculation
- **Variant Selection:** For each site, the two most prevalent variants (e.g., N1, N2, IP4) are selected based on data availability.
- **Flow Normalization:** Viral loads (CG/L) can be converted to total flow (CG/day) using daily flow rates ($m^3/day$), or kept as concentrations by setting `wastewater_flow_mode: "concentration"`.
- **Temporal Aggregation:** Multiple measurements per day are averaged.

### 2. Imputation and Smoothing (Tobit Kalman Filter)
To address missing data and censored observations (values below the limit of detection), we implement a **Tobit Kalman Filter**:

- **State Space Model:** We model the log-transformed viral flow as a local level process (random walk) observed with noise.
- **Censored Data (Values $\le$ LOD):** Instead of treating these as zeros or missing, we use a Tobit update step. The filter updates the state estimate using the probability mass of the predicted distribution that falls below the LOD. This statistically imputes the most likely value consistent with the "non-detect" observation.
- **Missing Data:** Days with no samples are handled causally according to `smoothing.missing_policy`. Default `predict` uses one-step prediction only; optional `momentum` enables bounded decayed extrapolation (with configurable max steps). Censor flag remains 2 to indicate imputed values.
- **Parameter Estimation:** Process and measurement variances are fitted per-site using Maximum Likelihood Estimation (MLE) on the available non-censored data.

### 3. Spatial Aggregation
Processed site-level data is mapped to administrative regions using a contribution matrix (defined in `region_metadata`). This transforms the sparse site-level signals into a continuous, region-aligned tensor.

### 4. Quality Control & Diagnostics
We provide specialized tools to analyze data quality and the efficacy of the processing:

- **Sparsity Analysis:** `dataviz/analyze_wastewater_sparsity.py`
  - Visualizes temporal availability and below-detection rates.
  - Compares LOCF (Last Observation Carried Forward) vs. true sparsity.
  - Generates "traffic light" heatmaps of data coverage.
  
- **Canonical Output Check:** `dataviz/canonical_biomarker_series.py`
  - Plots the final processed time series per region.
  - Generates heatmaps of the canonical biomarker tensor.
  - Useful for verifying that the imputation produced realistic trends.

## Output Dataset Contract

Running the pipeline always yields a Zarr dataset whose arrays follow this layout:

| Array | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `cases` | `(run_id, time, region, 1)` | float16 | Canonical case counts (log1p per-100k transformed). |
| `cases_mask` | `(run_id, time, region)` | bool | True if observed, False if missing/interpolated. |
| `cases_age` | `(run_id, time, region)` | uint8 | Days since last observation (0-14). 14 = no prior observation. |
| `hospitalizations` | `(run_id, time, region, 1)` | float16 | Hospitalization counts (log1p per-100k, Kalman-smoothed). |
| `hospitalizations_mask` | `(run_id, time, region)` | bool | True if observed, False if missing/interpolated. |
| `hospitalizations_age` | `(run_id, time, region)` | uint8 | Days since last observation (0-14). |
| `deaths` | `(time, region)` | float16 | Death counts (log1p per-100k, Kalman-smoothed). |
| `deaths_mask` | `(time, region)` | bool | True if observed, False if missing/interpolated. |
| `deaths_age` | `(time, region)` | uint8 | Days since last observation (0-14). |
| `edar_biomarker_<variant>` | `(run_id, time, region)` | float16 | Per-variant biomarker intensity (log1p, Kalman-filtered). |
| `edar_biomarker_<variant>_mask` | `(run_id, time, region)` | bool | True if measured (finite and positive), False otherwise. |
| `edar_biomarker_<variant>_censor` | `(run_id, time, region)` | uint8 | 0=uncensored, 1=censored at LOD, 2=missing/imputed. |
| `edar_biomarker_<variant>_age` | `(run_id, time, region)` | uint8 | Days since last measurement (0-14). 14 = no EDAR coverage. |
| `biomarker_data_start` | `(run_id, region)` | int16 | First time index with biomarker data. -1 = no coverage. |
| `mobility` | `(run_id, time, origin, destination)` | float16 | Origin–destination flow tensor (log1p-transformed). |
| `population` | `(region,)` | int32 | Static population per region. |
| `edar_has_source` | `(region,)` | bool | True if region has EDAR site contributions. |
| `valid_targets` | `(run_id, region)` | bool | True if (run, region) meets minimum data density. |
| `temporal_covariates` | `(time, 3)` | float16 | `[dow_sin, dow_cos, is_holiday]` features. |

### Dataset Attributes

The Zarr dataset includes the following attributes to indicate the transforms applied:

| Attribute | Value | Description |
| --- | --- | --- |
| `log_transformed` | `true` | All continuous series (clinical, mobility, biomarkers) are log1p-transformed. |
| `population_norm` | `true` | Clinical series (cases, hospitalizations, deaths) are per-100k normalized. |

### Dtype Rationale

| Category | Dtype | Reason |
| --- | --- | --- |
| Continuous values | float16 | 2 bytes, sufficient precision for log1p-transformed inputs; matches model bf16 compute. Overflow prevented by log1p transform. |
| Binary masks | bool | 1 byte, clear semantics (observed vs missing). |
| Age channels | uint8 | 1 byte, range 0-14 days. |
| Censor flags | uint8 | 1 byte, values 0/1/2. |
| Data start indices | int16 | 2 bytes, needs -1 sentinel, max ~1000 days fits easily. |
| Population | int32 | 4 bytes, populations < 2.1B. |
| Binary flags | bool | 1 byte, clear semantics. |

Any additional metadata (alignment summaries, preprocessing config, pipeline timing) is bundled in the accompanying JSON report and within each `EpiBatch.metadata` field.

## Running the Pipeline

```bash
uv run python -m data.preprocess.pipeline --config configs/preprocess/<your_config>.yaml
```

* Use a single configuration file to describe the desired dataset. Variant-specific switches should live in the training/model configs, not here.
* The pipeline infers the output path via `PreprocessingConfig.get_output_dataset_path()` and overwrites previous artifacts if they share the same dataset name—copy/rename if you need historical snapshots.
* To inspect the result, point `DatasetStorage.validate_dataset(<zarr_path>)` or load via the CLI commands defined in `cli.py`.

## Practical Notes

- Keep raw inputs in `data/raw/` and let the pipeline write to `data/processed/` or `outputs/` via your config. Large artifacts should stay out of version control.
- If mobility data is missing or fails validation, the pipeline logs a warning and writes zero-filled tensors so the output contract stays intact.
- When updating biomarker definitions or region mappings, ensure the `region_metadata` you supply still produces the `(time, region)` series; otherwise the alignment step will raise before writing artifacts.

This guide should be kept in sync with `data/preprocess/pipeline.py` whenever the preprocessing stages or output schema changes.
