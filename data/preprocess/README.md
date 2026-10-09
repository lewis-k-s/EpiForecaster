# Preprocessing Pipeline Guide

This worktree uses source-native clinical and EDAR inputs with the heterogeneous
model. Municipality preprocessing supplies cases, vaccination, and calendar
features. The municipal training model belongs to the legacy worktree.

## Clinical dataset version 3

Build the opt-in clinical dataset with:

```bash
uv run preprocess epiforecaster --config configs/preprocess_clinical_v3.yaml
```

The output is `data/processed/real_with_id_v3.zarr`. It uses float32 storage,
records `dataset_version: 3`, and refuses to replace an existing dataset.
Existing training configs continue to select `real_with_id.zarr`. Select V3
with a `data.dataset_path` override and generate new native source sets before
using it in an immutable training run. Earlier split manifests bind the V1
input bytes and cannot be reused for V3.

V3 removes municipal wastewater values, masks, censor codes, ages, and
coverage metadata. The clinical pipeline no longer accepts wastewater or
EDAR-to-municipality mapping inputs. Native plant observations and catchment
relations remain in the separate native stores. V1 and V2 assets remain
unchanged; their projected wastewater fields are historical outputs.

Clinical filters operate in `log1p` space with fixed configured
`smoothing.process_var` and `smoothing.measurement_var`. They initialize from
the current observation and expose zero values with mask 0 before the first
observation. Explicit zero counts retain mask 1. Absent municipality-day rows
remain unknown; the case source has no reporting-availability flag that would
justify treating every absent row as an observed zero.

Vaccination coverage and age are computed from all prior dose records before
the output is cropped. Alignment preserves that prior history in the age
channel.

## Source-native contract

Native preprocessing writes versioned artifacts under
`data/processed/native_support/`. These artifacts preserve observation support
instead of projecting every target to municipalities:

| Artifact | Contract |
| --- | --- |
| `native_graph.zarr` | Typed municipality, ABS, comarca, MITMA, and EDAR nodes with audited populations and directed overlap, catchment, and point-location relations |
| `obs_admissions.zarr` | Raw weekly ABS admission counts, masks, ages, and QC-only reported-stratum population sums |
| `obs_wastewater.zarr` | Raw EDAR assay concentrations, detection limits, censor codes, causal histories, flow, and rainfall |
| `od_mitma.zarr` | Primary raw native MITMA OD context |
| `od_municipal.zarr` | Stage-4 projected-context comparison artifact only |
| `edar_catchment_edges.parquet` | Contribution-weighted EDAR relations to municipalities, ABS, and comarques |
| `edar_point_locations.parquet` | Independent point-in-polygon location relations for EDAR plants |
| `obs_deaths_abs_excess.zarr` | Daily ABS observed all-cause deaths, 2017–19 expected deaths, and signed excess; 376 ABS codes match the native graph |
| `native_manifest_v4.json` | Source and artifact identities for the ABS excess-deaths cutover |

The training DAG derives `native_source_sets/assignments.parquet` and a small
version-4 `manifest.json` from the processed inputs. It publishes these through
DVC. Do not commit generated membership JSONs. Changes to preprocessing,
eligibility, or the split policy require a new split asset. Earlier target
windows must end before the next split's origins begin.

Build the ABS mortality store from the downloaded Catalonia death CSV with
`uv run python -m data.preprocess.processors.abs_excess_deaths_processor
--source-csv /path/to/defuncions_a_catalunya_3cre-fa8v_20260928.csv`.
The builder refuses to replace an existing store. It records the source file
name and SHA-256 in the store; the v4 manifest repeats them. The CSV is an
external source input and is not contained in the DVC output pointer.

Municipality cases still come from `data/processed/real_with_id.zarr`. Native
training does not load `hospitalizations_municipality.csv` or
`deaths_municipality.csv`. Wastewater stays at EDAR support: each query owns
an assay target block. The contribution matrix defines the catchment and its
population; the GeoJSON point defines plant location only. These two relation
families are not interchangeable. Projected files remain only for explicit
legacy or comparison work. Missing `(date, source)` rows have mask 0 and are
never converted to observed zeros.

Run native builders through their config-driven preprocessing entry points and
validate the manifest and QC report before training. Do not modify generated
stores in place. Create a new versioned artifact set when inputs or operators
change.

## Municipality input preprocessing

`data/preprocess/pipeline.py` supplies municipality cases, vaccination, and
calendar features in `real_with_id.zarr`. The native forecaster reads these
fields from that store. Native ABS and EDAR observations stay in their own
stores. Projected hospital and death series are not model inputs in this worktree.

## Inputs and Canonical Variables

| Variable | Shape | Notes |
| --- | --- | --- |
| `population` | `(region,)` (persisted as `(region, 1)` in Zarr) | Static per administrative region. Sourced from the case metadata. Required for per-100k normalization in preprocessing. |
| `cases` | `(time, region)` | Daily (or configured cadence) case counts. Required input that defines the master time/region grid. |
| `vaccination_rate` | `(time, region)` | Cumulative first-dose COVID-19 vaccination coverage per municipality, computed from official Generalitat administered-dose records and population. |
| `mobility` | `(time, region, region)` | Origin–destination matrix per timestamp; required for the real clinical build. |

## Processing Stages

1. **Raw loaders** — Read cases, mobility, population, and configured clinical
   sources. Synthetic preprocessing extracts these fields from the raw bundle;
   plant observations in that bundle are not projected to municipalities.
2. **Alignment** — Align cases and mobility by municipality ID. Strict temporal
   alignment requires the configured daily range; outer alignment preserves
   unavailable dates in the observation masks.
3. **Static transforms** — Use `log1p(per-100k)` for clinical counts, `log1p` for
   mobility, and rate space for vaccination coverage.
4. **Storage** — Write Zarr or NetCDF according to `output_format`. The writer
   rejects municipal wastewater fields. V3 uses float32 for continuous values.
5. **Provenance** — Record the resolved config and source identities in attributes.

## Causal Smoothing Modes

The preprocessing pipeline supports configurable **offline but causal** smoothing.
Each timestep is updated using only current/past observations (no look-ahead over
future dates).

### Clinical series (`cases`, `hospitalizations`, `deaths`)

| Method | Config key | Behavior |
| --- | --- | --- |
| Kalman v2 | `smoothing.clinical_method: "kalman_v2"` | Random-walk Kalman filter with configurable missing policy (`predict` default, optional bounded `momentum`), innovation clipping, and re-entry gain capping to reduce hard jump artifacts after sparse gaps. |
| Damped Holt | `smoothing.clinical_method: "holt_damped"` | Causal level+trend smoother in log1p space with damped trend carry through missing intervals. |

### Native wastewater

`native_wastewater_processor.py` preserves each plant and assay. It stores raw
concentration, detection limit, censor code, observation mask, age, flow, and
fixed-parameter causal Tobit-Kalman history. These fields are not merged into
municipality features. `NativeSupportDataset` reads this separate store.

Use `native.wastewater_filename` to select a pinned native wastewater version.
The catchment plot accepts `--wastewater-path` and reads this store directly.
Source sample dates do not establish lab result availability; an as-of study
must account for reporting delay separately.

## Output Dataset Contract

V3 writes a Zarr dataset with the following arrays (NetCDF is also supported):

| Array | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `cases` | `(run_id, time, region)` | float32 | Canonical case counts (log1p per-100k transformed). |
| `cases_mask` | `(run_id, time, region)` | bool | True if observed, False if missing/interpolated. |
| `cases_age` | `(run_id, time, region)` | uint8 | Days since last observation (0-14). 14 = no prior observation. |
| `hospitalizations` | `(run_id, time, region)` | float32 | Hospitalization counts (log1p per-100k, Kalman-smoothed). |
| `hospitalizations_mask` | `(run_id, time, region)` | bool | True if observed, False if missing/interpolated. |
| `hospitalizations_age` | `(run_id, time, region)` | uint8 | Days since last observation (0-14). |
| `deaths` | `(run_id, time, region)` | float32 | Death counts (log1p per-100k, Kalman-smoothed). |
| `deaths_mask` | `(run_id, time, region)` | bool | True if observed, False if missing/interpolated. |
| `deaths_age` | `(run_id, time, region)` | uint8 | Days since last observation (0-14). |
| `vaccination_rate` | `(run_id, time, region)` | float32 | Cumulative first-dose vaccination coverage, clipped to `[0, 1]`. |
| `vaccination_rate_mask` | `(run_id, time, region)` | bool | True if the source had administered first-dose rows for the municipality-date. |
| `vaccination_rate_age` | `(run_id, time, region)` | uint8 | Days since last source first-dose observation (0-14). |
| `mobility` | `(run_id, time, origin, destination)` | float32 | Origin–destination flow tensor (log1p-transformed). |
| `mobility_time_mask` | `(run_id, time)` | bool | True when the mobility tensor has source data for the timestep. False for outer-aligned dates beyond the OD data range. |
| `population` | `(region,)` | int32 | Static population per region. |
| `valid_targets` | `(run_id, region)` | bool | True if (run, region) meets minimum data density. |
| `temporal_covariates` | `(time, covariate)` | float32 | Calendar features such as `[dow_sin, dow_cos, is_holiday, lockdown_severity]` when enabled. |

### Dataset Attributes

The Zarr dataset includes the following attributes to indicate the transforms applied:

| Attribute | Value | Description |
| --- | --- | --- |
| `log_transformed` | `true` | Clinical counts and mobility are log1p-transformed; vaccination remains a rate. |
| `population_norm` | `1` | Clinical series (cases, hospitalizations, deaths) are per-100k normalized. |

### Dtype Rationale

| Category | Dtype | Reason |
| --- | --- | --- |
| Continuous values | float32 | 4 bytes; preserves small positive observations and transformed values. |
| Binary masks | bool | 1 byte, clear semantics (observed vs missing). |
| Age channels | uint8 | 1 byte, range 0-14 days. |
| Population | int32 | 4 bytes, populations < 2.1B. |
| Binary flags | bool | 1 byte, clear semantics. |

The resolved preprocessing config and source provenance are stored in dataset attributes.

## Running the Pipeline

```bash
uv run preprocess epiforecaster --config configs/preprocess_clinical_v3.yaml
```

* Use a single configuration file to describe the desired dataset. Variant-specific switches should live in the training/model configs, not here.
* The output name follows `dataset_name` and `output_format`. The V3 config sets `overwrite_output: false`; use a new version for each rebuild.
* To inspect the result, open the Zarr with `xarray.open_zarr(<zarr_path>)` and validate expected dimensions, masks, and attributes.

## Practical Notes

- Keep raw inputs in `data/raw/` and let the pipeline write to `data/processed/` or `outputs/` via your config. Large artifacts should stay out of version control.
- Real preprocessing requires cases, mobility, and population. Missing required inputs cause an error.
- Changes to native wastewater or catchment relations use separate versioned native artifacts.

This guide should be kept in sync with `data/preprocess/pipeline.py` whenever the preprocessing stages or output schema changes.
