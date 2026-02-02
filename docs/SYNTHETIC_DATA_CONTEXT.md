# Synthetic Data Context for Downstream Forecasters

This document explains the synthetic data structure for downstream forecasters who will load, understand, and use the synthetic zarr datasets.

## Overview

The synthetic data generation pipeline creates **counterfactual "twin" scenarios** to benchmark mobility intervention strategies. For each epidemiological profile (fixed R0, incubation period, infectious period), we run multiple scenarios that differ *only* in their intervention strategy (Baseline vs. Global_Timed vs. Local_Static) at varying strengths.

### Key Features

- **Realistic Noise**: Mimics real-world data sources with ascertainment bias, reporting delays, missing data, and wastewater censoring
- **Ground Truth Availability**: True infection/hospitalization/death counts for model evaluation
- **Curriculum Learning Support**: Metadata variables describing noise levels for clean → noisy curriculum progression
- **Factorized Mobility**: Efficient storage avoiding OOM errors on large datasets

## Data Format & Loading

The data is stored in **Zarr format**, a cloud-optimized chunked array storage format compatible with xarray.

### Loading with xarray

```python
import xarray as xr

# Load the zarr dataset
ds = xr.open_zarr("path/to/raw_synthetic_observations.zarr")

# Access variables
cases = ds["cases"]  # Raw case observations with NaN for missing
infections_true = ds["infections_true"]  # Ground truth for evaluation

# List all variables
print(list(ds.data_vars.keys()))
```

### Memory-Mapped Access

For large datasets, use chunked access without loading into memory:

```python
# Access specific runs without loading all data
subset = ds.isel(run_id=slice(0, 10))  # First 10 runs
```

## Variable Reference

### Raw Observations (for Model Input/Preprocessing)

These variables represent the "raw" data that would be available in a real-world scenario. They contain realistic noise, missing data, and censoring patterns.

| Variable | Shape | Description |
|----------|-------|-------------|
| `cases` | `(run_id, date, region_id)` | Reported cases with ascertainment bias (logistic ramp) and missing data (NaN for missing) |
| `hospitalizations` | `(run_id, date, region_id)` | Reported hospitalizations with underreporting (rate ~0.85) and reporting delay (mean ~3 days, std ~1) |
| `deaths` | `(run_id, date, region_id)` | Reported deaths with underreporting (rate ~0.90) and reporting delay (mean ~7 days, std ~2) |
| `edar_biomarker_N1` | `(run_id, date, edar_id)` | Raw N1 gene target concentration with log-normal noise (sigma ~0.5) and censoring (LoD = 375) |
| `edar_biomarker_N2` | `(run_id, date, edar_id)` | Raw N2 gene target concentration with log-normal noise (sigma ~0.8) and censoring (LoD = 500) |
| `edar_biomarker_IP4` | `(run_id, date, edar_id)` | Raw IP4 gene target concentration with log-normal noise (sigma ~0.6) and censoring (LoD = 800) |
| `edar_biomarker_*_censor_hints` | `(run_id, date, edar_id)` | Censoring flags: 0=observed, 1=censored, 2=missing (optional reference) |
| `edar_biomarker_*_LoD` | `(run_id, edar_id)` | Limit of Detection values per EDAR (constant across runs) |
| `mobility_base` | `(origin, target)` | Base mobility matrix (shared across all runs) - factorized format |
| `mobility_kappa0` | `(run_id, date)` | Mobility reduction factor per run and date (κ₀) - factorized format |
| `mobility_time_varying` | `(run_id, origin, target, date)` | Full time-varying mobility matrix per run (optional, large format) |
| `population` | `(run_id, region_id)` | Static population per region |

#### Mobility Storage Formats

Mobility data is available in two formats:

**Factorized Format** (memory-efficient, default):
```python
# Reconstruct full mobility matrix for a specific run and date
mobility_reconstructed = ds["mobility_base"].values * (1 - ds["mobility_kappa0"][run_id, date].values)
```
- **Memory usage**: ~500MB for 100 runs × 100 days × 2850 regions (99.8% reduction)
- **Recommended for**: Most use cases, large datasets

**Time-Varying Format** (optional, direct access):
```python
# Access full mobility matrix directly (no reconstruction needed)
mobility_full = ds["mobility_time_varying"]  # Shape: (run_id, origin, target, date)
```
- **Memory usage**: ~19GB for 23 runs × 114 days × 945 regions
- **Use case**: When direct access is needed without reconstruction overhead

**Check format per run:**
```python
# The synthetic_mobility_type variable indicates which format is used
mobility_type = ds["synthetic_mobility_type"].values
# Values: "factorized" or "time_varying"
```

EpiForecaster's `mobility_processor.py` automatically detects and handles both formats.

### Ground Truth (for Evaluation Only)

These variables contain the *true* values from the simulation. Use them **only** for model evaluation - never for training.

| Variable | Shape | Description |
|----------|-------|-------------|
| `infections_true` | `(run_id, region_id, date)` | Daily infections (sum over ages) |
| `hospitalizations_true` | `(run_id, region_id, date)` | Daily hospitalizations |
| `deaths_true` | `(run_id, region_id, date)` | Daily deaths |

**⚠️ Critical**: Ground truth variables have different dimension ordering than raw observations. See "Coordinate System Note" below.

### Noise Metadata (for Curriculum Ordering)

These metadata variables describe the noise level applied to each run. Use them to build curriculum learning pipelines (clean → noisy progression).

#### Cases Reporting Noise

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_cases_report_rate_min` | `(run_id,)` | Minimum cases ascertainment rate (default: 0.05) |
| `synthetic_cases_report_rate_max` | `(run_id,)` | Maximum cases ascertainment rate (default: 0.60) |
| `synthetic_cases_report_delay_mean` | `(run_id,)` | Cases reporting delay in days (0 = not modeled) |

Cases use a **logistic ramp** for time-varying ascertainment:
```
report_rate(t) = min_rate + (max_rate - min_rate) / (1 + exp(-slope * (t - inflection_day)))
```

#### Hospitalizations Reporting Noise

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_hosp_report_rate` | `(run_id,)` | Hospitalization reporting rate (default: 0.85) |
| `synthetic_hosp_report_delay_mean` | `(run_id,)` | Mean hospitalization reporting delay in days (default: 3) |
| `synthetic_hosp_report_delay_std` | `(run_id,)` | Std dev of hospitalization reporting delay (default: 1) |

#### Deaths Reporting Noise

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_deaths_report_rate` | `(run_id,)` | Deaths reporting rate (default: 0.90) |
| `synthetic_deaths_report_delay_mean` | `(run_id,)` | Mean deaths reporting delay in days (default: 7) |
| `synthetic_deaths_report_delay_std` | `(run_id,)` | Std dev of deaths reporting delay (default: 2) |

#### Wastewater Noise

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_ww_noise_sigma_N1` | `(run_id,)` | Log-normal noise sigma for N1 gene target (default: 0.5) |
| `synthetic_ww_noise_sigma_N2` | `(run_id,)` | Log-normal noise sigma for N2 gene target (default: 0.8) |
| `synthetic_ww_noise_sigma_IP4` | `(run_id,)` | Log-normal noise sigma for IP4 gene target (default: 0.6) |
| `synthetic_ww_transport_loss` | `(run_id,)` | Signal decay in sewer system (default: 50.0 for N1) |

Wastewater uses **age-stratified shedding kernels**:
- **Children (<18)**: Long-tail kernel (α=1.5, β=10.0, mean 15d) - prolonged fecal shedding
- **Adults**: Acute-phase kernel (α=2.5, β=4.0, mean 10d) - correlated with symptoms

#### Sparsity Metadata

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_sparsity_level` | `(run_id,)` | Missing data rate used (default: 0.05) |

### Scenario Metadata (for Analysis)

| Variable | Shape | Description |
|----------|-------|-------------|
| `synthetic_scenario_type` | `(run_id,)` | Scenario type: "Baseline", "Global_Timed", or "Local_Static" |
| `synthetic_strength` | `(run_id,)` | Intervention strength (0.0 to 1.0) |
| `synthetic_mobility_type` | `(run_id,)` | Mobility storage format: "factorized" or "time_varying" |

**Scenario Types**:
- **Baseline**: No intervention (strength = 0.0)
- **Global_Timed**: Reduction of κ₀ only during event window
- **Local_Static**: Structural reduction of mobility matrix (reduces weights of edges connected to X% of nodes)

## Dimension Reference

| Dimension | Description |
|-----------|-------------|
| `run_id` | Simulation run identifier (string, sanitized from directory name) |
| `date` | Daily timestamps (pandas datetime64) |
| `region_id` | Geographic region identifier (municipality ID as string) |
| `edar_id` | Wastewater treatment plant catchment area identifier |
| `origin` | Mobility matrix origin region (same as region_id) |
| `target` | Mobility matrix target region (same as region_id) |

## Key Design Notes

### Factorized Mobility

Mobility is stored in two possible formats:

**Factorized Format** (memory-efficient):
```python
# Base OD matrix (shared across all runs)
mobility_base = ds["mobility_base"]  # (origin, target)

# Time-varying reduction factors
mobility_kappa0 = ds["mobility_kappa0"]  # (run_id, date)

# Reconstruct for specific run and date
kappa0_value = ds["mobility_kappa0"].isel(run_id=0, date=10).values
mobility_at_date = ds["mobility_base"].values * (1 - kappa0_value)
```

**Time-Varying Format** (direct access):
```python
# Full mobility matrix (no reconstruction needed)
mobility_full = ds["mobility_time_varying"]  # (run_id, origin, target, date)
mobility_at_date = mobility_full.isel(run_id=0, date=10).values
```

Check which format a run uses:
```python
mobility_type = ds["synthetic_mobility_type"].isel(run_id=0).values
# Returns: "factorized" or "time_varying"
```

### NaN Conventions

- **Missing data**: Represented as `NaN` (matches real-world format)
- **Censored wastewater**: Values at LoD are set to LoD value (not NaN)
- **Censoring hints**: Use `*_censor_hints` variables to identify censored vs. missing

```python
# Separate missing from censored
is_missing = np.isnan(ds["edar_biomarker_N1"].isel(run_id=0))
is_censored = ds["edar_biomarker_N1_censor_hints"].isel(run_id=0) == 1
```

### Wastewater Censoring

Censoring uses a **probabilistic LoD** curve:

```python
# Detection probability follows sigmoid curve
detection_prob = sigmoid(k * (concentration - LoD))
# where k is the lod_slope (default: 1.5)
```

- **Values below LoD**: Set to LoD (not zero) with censor_hints = 1
- **Missing measurements**: Set to NaN with censor_hints = 2
- **Above LoD**: Observed value with censor_hints = 0

### Age Stratification

Wastewater generation uses **age-stratified shedding kernels**:
- Children have prolonged fecal shedding (long-tail kernel)
- Adults have shedding correlated with symptoms (acute-phase kernel)

This creates realistic "school signatures" (residential areas) vs. "workforce signatures" (business districts).

### Memory Efficiency

The dataset uses:
- **Chunking**: Variables are chunked for efficient partial access (default: 256)
- **Compression**: Zstd compression (level 3) by default
- **Factorized mobility**: Reduces memory by 99.8% (~325GB → ~500MB)
- **Time-varying mobility**: Full format available when direct access is needed (~19GB for 23 runs)

## Usage Examples

### Loading the Dataset

```python
import xarray as xr

# Load from zarr
ds = xr.open_zarr("path/to/raw_synthetic_observations.zarr")

# Inspect structure
print(ds)
print("\nDimensions:", dict(ds.dims))
print("\nCoordinates:", list(ds.coords.keys()))
print("\nData variables:", list(ds.data_vars.keys()))
```

### Accessing Specific Variables

```python
# Raw cases (for preprocessing)
cases = ds["cases"]  # Shape: (run_id, date, region_id)

# Ground truth infections (for evaluation)
infections_true = ds["infections_true"]  # Shape: (run_id, region_id, date)

# Mobility - factorized format (memory-efficient)
base_mobility = ds["mobility_base"].values  # (origin, target)
kappa0 = ds["mobility_kappa0"]  # (run_id, date)
kappa0_t0 = kappa0.isel(run_id=0, date=0).values
mobility_t0 = base_mobility * (1 - kappa0_t0)

# Mobility - time-varying format (direct access)
if "mobility_time_varying" in ds:
    mobility_full = ds["mobility_time_varying"]  # (run_id, origin, target, date)
    mobility_t0 = mobility_full.isel(run_id=0, date=0).values

# Check which mobility format is used
mobility_type = ds["synthetic_mobility_type"].isel(run_id=0).values
print(f"Mobility format: {mobility_type}")  # "factorized" or "time_varying"
```

### Separating Training from Evaluation Data

```python
# Training data: Raw observations (goes into preprocessing pipeline)
training_vars = ["cases", "hospitalizations", "deaths",
                 "edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4",
                 "mobility_base", "mobility_kappa0", "mobility_time_varying",
                 "population"]

# Note: mobility_time_varying is optional; check if present before using
if "mobility_time_varying" not in ds:
    training_vars.remove("mobility_time_varying")

# Evaluation data: Ground truth (for metrics only)
eval_vars = ["infections_true", "hospitalizations_true", "deaths_true"]

# Noise metadata: For curriculum learning
noise_metadata = ["synthetic_cases_report_rate_min", "synthetic_cases_report_rate_max",
                  "synthetic_hosp_report_rate", "synthetic_hosp_report_delay_mean",
                  "synthetic_deaths_report_rate", "synthetic_deaths_report_delay_mean",
                  "synthetic_ww_noise_sigma_N1", "synthetic_ww_noise_sigma_N2",
                  "synthetic_ww_transport_loss", "synthetic_sparsity_level"]
```

### Filtering by Noise Level for Curriculum

```python
# Sort runs by sparsity (clean → noisy)
sparsity = ds["synthetic_sparsity_level"].values
sorted_run_indices = np.argsort(sparsity)
clean_runs = sorted_run_indices[:10]  # 10 cleanest runs
noisy_runs = sorted_run_indices[-10:]  # 10 noisiest runs

# Filter dataset
ds_clean = ds.isel(run_id=clean_runs)
ds_noisy = ds.isel(run_id=noisy_runs)
```

### Filtering by Scenario Type

```python
# Get baseline runs only
baseline_mask = ds["synthetic_scenario_type"].values == "Baseline"
ds_baseline = ds.isel(run_id=np.where(baseline_mask)[0])

# Compare scenarios
for scenario in ["Baseline", "Global_Timed", "Local_Static"]:
    mask = ds["synthetic_scenario_type"].values == scenario
    print(f"{scenario}: {np.sum(mask)} runs")
```

### Accessing Wastewater with Censoring

```python
# Load N1 biomarker with censoring info
n1_conc = ds["edar_biomarker_N1"]  # (run_id, date, edar_id)
n1_censor = ds["edar_biomarker_N1_censor_hints"]  # (run_id, date, edar_id)
n1_lod = ds["edar_biomarker_N1_LoD"]  # (run_id, edar_id)

# For a specific run
run_idx = 0
n1_run = n1_conc.isel(run_id=run_idx)
censor_run = n1_censor.isel(run_id=run_idx)

# Separate observed, censored, and missing
observed_mask = censor_run == 0
censored_mask = censor_run == 1
missing_mask = censor_run == 2

print(f"Observed: {observed_mask.sum().values}")
print(f"Censored: {censored_mask.sum().values}")
print(f"Missing: {missing_mask.sum().values}")
```

## Coordinate System Note

**Important**: Raw observations and ground truth have **different dimension orderings**:

- **Raw observations**: `(run_id, date, region_id)` or `(run_id, date, edar_id)`
- **Ground truth**: `(run_id, region_id, date)`

This design choice optimizes access patterns:
- Raw obs are typically accessed as time series (date dimension first)
- Ground truth is typically accessed for spatial analysis (region dimension first)

```python
# Raw obs: time-series access is fast
cases_region_0 = ds["cases"][:, :, 0]  # All runs, all dates, region 0

# Ground truth: spatial access is fast
infections_date_0 = ds["infections_true"][:, :, 0]  # All runs, all regions, date 0
```

## Wastewater Physics Model

The wastewater generation uses a high-fidelity physical model:

### Signal Generation

1. **Age-stratified shedding**: Different shedding kernels per age group
2. **Aggregation via EMAP**: Municipalities → EDAR catchment areas
3. **Dilution**: Signal normalized by EDAR population (models wastewater flow)
4. **Log-normal noise**: Multiplicative noise on concentration
5. **Censoring**: Probabilistic detection near LoD

### EDAR-Level Aggregation

For EDAR-based wastewater:

```python
# Infections aggregated from regions to EDARs
infections_edar[EDAR] = sum(EMAP[EDAR, region] * infections[region])

# Signal divided by EDAR population (models dilution)
ww_signal = convolve(infections_edar, shedding_kernel) / population_edar
```

The `contribution_ratio` in EMAP represents **wastewater flow fractions**, not population distribution. This correctly models physical dilution in the sewer network.

## Gene Target Properties

| Target | Sensitivity | Noise Sigma | LoD | Transport Loss |
|--------|-------------|-------------|-----|----------------|
| N1 | High (500k) | 0.5 | 375 | 50.0 |
| N2 | Moderate (400k) | 0.8 | 500 | 100.0 |
| IP4 | Low (250k) | 0.6 | 800 | 200.0 |

**Sensitivity Scale**: Relative scaling factor for concentration
**LoD**: Limit of Detection (concentration below this is censored)
**Transport Loss**: Signal decay in sewer system

## Reference Implementation

For the source of truth on output structure, see:
- `python/process_synthetic_outputs.py` - Main processing script

For detailed pipeline documentation:
- `CONTEXT_SYNTHETIC_GEN.md` - Architecture and usage guide

## Troubleshooting

### Common Issues

**Issue**: Shape mismatch when combining raw obs and ground truth
```
ValueError: shapes (100, 435, 2850) and (100, 2850, 435) not aligned
```
**Solution**: Transpose ground truth to match raw obs ordering:
```python
infections_true_T = ds["infections_true"].transpose("run_id", "date", "region_id")
```

**Issue**: Out of memory when loading full dataset
**Solution**: Use chunked access:
```python
# Load only first 10 runs
subset = ds.isel(run_id=slice(0, 10))
```

**Issue**: Cannot interpret edar_id values
**Solution**: EDAR IDs are categorical strings from the input mapping file:
```python
print(ds["edar_id"].values[:10])  # First 10 EDAR IDs
```
