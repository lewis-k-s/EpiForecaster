# Code Housecleaning: Elegance, Reduction, Simplicity & Beauty

*Generated: 2026-02-04*

## Executive Summary

This codebase is well-engineered for scientific computing but suffers from **significant duplication** and **accumulated complexity**. Below are prioritized opportunities for code reduction and elegance improvements.

**Potential Impact:** ~1050 lines of code eliminated, significantly improved maintainability.

---

## 🔴 High-Priority (Do These First)

### 1. Consolidate Three Separate Plotting Systems

**Problem:** Duplicated plotting logic across three locations:
- `utils/plotting.py` (270 lines) - Shared utilities
- `plotting/forecast_plots.py` (318 lines) - Forecast plotting
- `plotting/input_plots.py` - Input data visualization
- `dataviz/` (13 files, ~3000+ lines) - Exploratory analysis

**Duplication Examples:**
- `dataviz/canonical_biomarker_series.py` (718 lines)
- `dataviz/canonical_deaths.py` (similar structure)
- `dataviz/canonical_hospitalizations.py` (similar structure)

All have duplicated: `_load_raw_data`, `plot_region_series`, `plot_heatmap`, `plot_distribution`, etc.

**Solution:**
```bash
# Consolidate into existing dataviz/ package
mv utils/plotting.py dataviz/base.py
mv plotting/forecast_plots.py dataviz/forecast_plots.py
mv plotting/input_plots.py dataviz/input_plots.py
# Extract common patterns from dataviz/canonical_*.py
# into dataviz/canonical_plots.py with template-based rendering
```

**Follow-ups (required to avoid breakage):**
- Update imports to new `dataviz.*` paths in any callers.
- Add or verify `dataviz/__init__.py` if module imports are expected.
- Update any CLI or script entry points that reference `plotting/`.

**Lines Saved:** 400-600
**Files Affected:** 15+

---

### 2. Extract Base Processor Class for Municipality Data

**Problem:** `DeathsProcessor` and `HospitalizationsProcessor` have identical patterns:

```python
# deaths_processor.py:48-62
df = pd.read_csv(deaths_file, dtype={"municipality_code": str})
df = df.rename(columns=self.COLUMN_MAPPING)
df = df[df["municipality_code"].notna() & (df["municipality_code"] != "")]
df["municipality_code"] = df["municipality_code"].astype(str)
# ... date parsing, validation, etc.

# hospitalizations_processor.py:67-79
# IDENTICAL CODE
```

**Solution:** Create `data/preprocess/processors/base.py`:

```python
class BaseMunicipalityProcessor:
    COLUMN_MAPPING: dict[str, str]
    FILE_NAME: str

    def _load_municipality_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, dtype={"municipality_code": str})
        df = df.rename(columns=self.COLUMN_MAPPING)
        df = df[df["municipality_code"].notna() & (df["municipality_code"] != "")]
        df["municipality_code"] = df["municipality_code"].astype(str)
        return df

    def _parse_date_column(self, df: pd.DataFrame, col: str, fmt: str) -> pd.DataFrame:
        # Common date parsing
        ...
```

Then simplify processors to:
```python
class DeathsProcessor(BaseMunicipalityProcessor):
    FILE_NAME = "deaths_municipality.csv"
    COLUMN_MAPPING = {...}

    def process(self, data_dir):
        df = self._load_municipality_csv(data_dir / self.FILE_NAME)
        # deaths-specific logic only (~50 lines instead of 170)
```

**Lines Saved:** 200-300
**Files Affected:** 4 (deaths, hospitalizations, possibly cases, catalonia_cases)

---

### 3. Split the Monolithic Trainer Class

**Problem:** `EpiForecasterTrainer` is 1900+ lines handling too many responsibilities:

| Responsibility | Approx Lines |
|----------------|--------------|
| Data loading/splitting | 200+ |
| Curriculum orchestration | 200+ |
| NVMe staging | 50+ |
| CUDA/fork management | 100+ |
| Profiling | 50+ |
| Checkpointing | 50+ |
| Training loop | 300+ |
| Evaluation | 100+ |
| Cleanup | 100+ |

**Solution:** Split into focused modules:

```
training/
├── data_loading.py      # SplitStrategy, DataLoaderFactory
├── curriculum.py         # CurriculumManager
├── hardware.py           # DeviceManager, NVMeStager
├── checkpointing.py      # CheckpointManager
├── training_loop.py      # TrainingLoop (pure loop logic)
└── trainer.py            # EpiForecasterTrainer (orchestration only, ~200 lines)
```

**Lines Saved:** 0 (reorganization only)
**Complexity Reduced:** Very High
**Maintainability:** Very High

---

## 🟡 Medium-Priority

### 4. Simplify CLI with Decorator Composition

**Problem:** `cli.py` (884 lines) has repeated parameter patterns:

```python
@click.option("--experiment")
@click.option("--run")
@click.option("--checkpoint")
# Repeated across eval_epiforecaster, plot_forecasts, etc.
```

**Solution:** Create `cli_common.py`:

```python
def experiment_options(fn):
    fn = click.option("--experiment")(fn)
    fn = click.option("--run")(fn)
    fn = click.option("--checkpoint")(fn)
    return fn

@eval_group.command("epiforecaster")
@experiment_options
@click.option("--split", ...)
def eval_epiforecaster(experiment, run, checkpoint, split): ...
```

**Lines Saved:** 100-150
**Files Affected:** 1 (cli.py)

---

### 5. Extract Zarr Compatibility Helper

**Problem:** `data/preprocess/pipeline.py:398-429` has 40+ lines of manual encoding cleanup.

**Solution:** Create `utils/zarr_compat.py`:

```python
def normalize_zarr_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Normalize encodings for Zarr v2 output."""
    v3_keys = {"chunks", "compressors", "filters", "serializer", ...}
    for var in chain(ds.data_vars, ds.coords):
        for key in v3_keys:
            var.encoding.pop(key, None)
    return ds
```

**Lines Saved:** 30
**Files Affected:** 2 (pipeline.py + new file)

---

## 🟢 Low-Hanging Fruit

### 6. Extract Constants

**Magic Numbers Found:**
```python
# training/epiforecaster_trainer.py:549
target_sparsities = [0.05, 0.20, 0.40, 0.60, 0.80]

# data/preprocess/pipeline.py:607
max_runs = 5

# Repeated in multiple files:
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
```

**Solution:** Reuse and extend existing `constants.py`:
```python
CURRICULUM_SPARSITY_BUCKETS = [0.05, 0.20, 0.40, 0.60, 0.80]
DEFAULT_MAX_SYNTHETIC_RUNS = 5
```

**Lines Saved:** 50
**Files Affected:** 20+

---

### 7. Dead Code Removal

**Commented Code Found:**
- `training/epiforecaster_trainer.py:1109-1152` (43 lines of commented graph logging)

**Action Items:**
- [ ] Remove commented code blocks

**Lines Saved:** 50-100

---

### 8. Datetime Parsing Helper

**Duplication:**
```python
# Found in multiple processors:
pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
pd.to_datetime(df["week_start"], format="%d/%m/%Y", errors="coerce")
```

**Solution:** Create `utils/dates.py` with common parsing functions.

---

## Summary Table

| Priority | Category | Files Affected | Lines Saved | Complexity Reduced |
|----------|----------|----------------|-------------|-------------------|
| 🔴 High | Plotting consolidation | 15+ | ~500 | High |
| 🔴 High | Processor base class | 4 | ~250 | Medium |
| 🔴 High | Trainer split | 1 | 0 (split) | Very High |
| 🟡 Medium | CLI decorators | 1 | ~120 | Medium |
| 🟡 Medium | Zarr compatibility | 2 | ~30 | Low |
| 🟢 Low | Constants extraction | 20+ | ~50 | Low |
| 🟢 Low | Dead code removal | Various | ~100 | Low |
| **TOTAL** | | **40+** | **~1050** | **Significant** |

---

## Recommended Implementation Steps (Highest Risk Last)

1. **Step 1:** Extract `BaseMunicipalityProcessor` - Quick win, clear scope, immediate value.
2. **Step 2:** Extract Zarr compatibility helper - Small, low-risk change.
3. **Step 3:** Reuse and extend `constants.py` - Low risk, broad cleanup.
4. **Step 4:** CLI decorator composition - Moderate scope, contained to `cli.py`.
5. **Step 5:** Consolidate plotting into existing `dataviz/` (move `input_plots.py`, `forecast_plots.py`, and shared utilities; update imports/entry points).
6. **Step 6 (Highest Risk, Last):** Split `EpiForecasterTrainer` - Largest surface area and highest integration risk.

---

## Design Principles for Future Development

1. **Prefer composition over inheritance** - Use protocol classes for pluggable components
2. **DRY at the function level** - If you copy-paste, extract immediately
3. **Single Responsibility Principle** - One class, one reason to change
4. **Configuration as code** - Use `dataclass` with defaults over nested dicts
5. **Visualization as a library** - Build reusable plotting primitives, not one-off scripts

---

## Checklist

- [ ] Extract `BaseMunicipalityProcessor` class
- [ ] Consolidate plotting into `dataviz/` package
- [ ] Split `EpiForecasterTrainer` into focused modules
- [ ] Create CLI decorator composition helpers
- [ ] Extract Zarr compatibility functions
- [ ] Create `constants.py` for magic numbers
- [ ] Remove dead/commented code
- [ ] Clean up untracked dataviz files
- [ ] Create datetime parsing utilities
