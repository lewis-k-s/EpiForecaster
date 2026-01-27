# Wastewater Variant Detection Data

**Source Repository**: `/Volumes/HUBSSD/code/sars-aigues/`

**Data Source**: ICRA (Catalan Institute for Water Research) - https://sarsaigua.icra.cat/

**Data Owners**: Agència de Salut Pública de Catalunya (ASPCAT) and Agència Catalana de l'Aigua (ACA)

**Zenodo DOI**: [10.5281/zenodo.4147073](https://doi.org/10.5281/zenodo.4147073)

---

## Overview

The sars-aigues repository contains **SARS-CoV-2 variant prevalence data** from wastewater surveillance across 59 wastewater treatment plants (EDAR) in Catalonia. This is the official public export from the ICRA surveillance platform and complements the viral load (CG/L) measurements currently used in the EpiForecaster pipeline.

Unlike the main CSV which tracks overall viral concentration, the variant data tracks **which variants are circulating** and their relative abundances over time. This enables early detection of emerging variants before they appear in clinical surveillance.

---

## Files

| File | Size | Description |
|------|------|-------------|
| `release_with_detection_limits.csv` | 417 KB | Main wastewater data (viral load + LD thresholds) |
| `variants_raw.json` | 1.6 MB | Raw variant proportions per sample (1,331 entries) |
| `variants_grouped.json` | 887 KB | Aggregated variant runs by time window (26 runs) |

---

## Data Structure

### Main CSV (`release_with_detection_limits.csv`)

The same structure as `data/files/wastewater_biomarkers_icra.csv`:

- **id mostra**: Sample identifier (EDAR_CODE-YYYY-MM-DD)
- **depuradora**: EDAR site name
- **LD(CG/L)**: Limit of detection threshold
- **N1(CG/L), N2(CG/L), IP4(CG/L), E(CG/L)**: Viral load concentrations
- **Cabal últimes 24h(m3)**: Flow rate
- **Pluja(mm)**: Rainfall

**Key Difference**: Extends data from **2023-12-27 to 2025-11-10** (+252 samples across 24 EDAR sites)

### Variant JSON (`variants_raw.json`)

```json
{
  "codi": "DABR-2022-06-27",      // Links to wastewater sample ID
  "variants": {
    "BA.5": "0.04141733",          // Variant: relative abundance
    "BA.4": "0.04141733",
    "BF.2": "0.04141733",
    "B.1.127": "0.00177305",
    ...
  },
  "formulari_complet": 1,
  "mostra_invalidada": null,
  "run_id": 53
}
```

**Key fields**:
- `codi`: Links to main CSV sample ID
- `variants`: Map of variant name → proportion (relative abundance)
- `run_id`: Sequencing batch identifier

### Grouped Variant JSON (`variants_grouped.json`)

Aggregated by time window (e.g., weekly runs):

```json
{
  "id": 30,
  "numero_run": 44,
  "data_inici": "2025-11-10",
  "data_final": "2025-11-17",
  "edars": [
    {
      "codi_edar": "DAMP",
      "info_edar": {
        "nom": "AMPOSTA",
        "província": "Tarragona",
        "coordenades": [[0.610031733231528, 40.70406345790177]]
      },
      "variants_of_concern": [
        {
          "variant": "JN.1",
          "detected": true,
          "proportion": "0.234"
        },
        ...
      ]
    }
  ]
}
```

---

## Data Coverage

| Metric | Value |
|--------|-------|
| **Date range** | 2022 to 2025 |
| **EDAR sites with variant data** | 58 of 59 |
| **Total variant entries** | 1,331 |
| **Unique variant lineages** | 2,611 |

### Top Variants (by sample count)

| Variant | Samples | Period |
|---------|---------|--------|
| XBB.1.5 | 364 | 2022-2023 |
| XBB.1.5.21 | 303 | 2023 |
| XBB.1.11.1 | 273 | 2023 |
| XBB.1.5.8 | 273 | 2023 |
| BA.5 | ~200 | 2022 |
| BA.4 | ~150 | 2022 |
| JN.1 | TBD | 2024-2025 |

---

## N1 Dropout Analysis

### Background

The **Omicron N1 dropout** is a known phenomenon in wastewater surveillance where certain Omicron variants (particularly BA.1) have mutations in the nucleocapsid gene that cause the N1 assay to fail or under-detect, while N2 remains functional. This can lead to misleading viral load estimates if only N1 is used.

### Findings from sars-aigues Data

**Does NOT contain explicit N1 dropout information**:
- Variant JSON only includes variant names and proportions
- No gene target failure metadata (e.g., SGTF - S-gene target failure)
- No assay performance notes linked to specific variants

**Empirical analysis of N1/N2 patterns**:

| Period | N1 Dropout Rate* | Notes |
|--------|------------------|-------|
| Pre-Omicron (Delta) | 5.5% | Baseline |
| Omicron BA.1/BA.2 | 2.2% | **Lower** than pre-Omicron |
| Omicron BA.4/BA.5 | 0.5% | Minimal dropout |
| Omicron XBB | 0.2% | Minimal dropout |
| Post-2024 | 3.3% | Similar to baseline |

*N1 dropout = N2 > LD but N1 ≤ LD

**Key observations**:
1. **N1=LD reporting is NOT specific to Omicron** - occurs throughout dataset
2. **100% of "dropout" cases have N1 exactly equal to LD** - suggests reporting convention, not assay failure
3. **N1=LD events correlate with lower viral loads** - N2 median: 3,193 CG/L (dropout) vs 27,157 CG/L (both detectable)
4. **High N1=LD periods** (2020-2021) align with early pandemic when overall viral loads were lower

### Lab Metadata (Observacions Field)

The CSV contains 195 samples with lab notes in the `Observacions` field:

| Observation | Count | Interpretation |
|-------------|-------|----------------|
| "Canvi metodològic" | 48 | Methodological change |
| "Molta materia suspensió" / "Molt material en suspensió" | ~60 | High suspended solids (causes inhibition) |
| "Molta inhibició particularment N1" | 11 | **N1 inhibition due to sample quality** |
| "Diana IP4 detecció <LQ" | 4 | IP4 below quantification limit |
| "Repetida dues vegades" / "Repetició" | ~10 | Sample was re-analyzed |

### Conclusions

**The classic Omicron N1 dropout phenomenon is NOT present in this dataset**. Instead:

1. **N1=LD appears to be a reporting convention** for "below detection limit"
2. **The lab may have updated primers/probes** for Omicron variants, avoiding the N1 dropout issue
3. **Sample quality issues** (high suspended solids, inhibition) affect N1 more than N2
4. **True N1 dropout would show N2 at normal levels with N1 completely absent** - this pattern is not observed

### Recommendations for EpiForecaster

1. **Treat N1=LD values as below-detection** (already done via LD thresholding in current preprocessing)
2. **Prefer N2 as the primary viral load indicator** (more consistently detectable, less affected by inhibition)
3. **The variant data cannot directly correct for N1 dropout** (no gene target metadata)
4. **Optional**: Create an `n1_dropout` feature based on `N1=LD & N2>LD` pattern if needed for modeling

```python
# In edar_processor.py, prioritize N2 for viral load estimates
PRIMARY_VARIANT = "N2(CG/L)"  # More robust than N1

# Optional: N1 dropout indicator
df["n1_dropout"] = (df["N1(CG/L)"] <= df["LD(CG/L)"]) & (df["N2(CG/L)"] > df["LD(CG/L)"])
```

---

## Integration Opportunities

### 1. Update Main Wastewater Source

The `release_with_detection_limits.csv` extends the current wastewater data by **nearly 2 years** (2025-02-24 to 2025-11-10).

**Action**: Update the local copy:
```bash
cp /Volumes/HUBSSD/code/sars-aigues/release_with_detection_limits.csv \
   data/files/wastewater_biomarkers_icra.csv
```

### 2. Variant Proportions as Covariates

Add variant proportions as additional channels to the forecaster:

```python
# In edar_processor.py or new variant_processor.py
VARIANT_COLUMNS = [
    "N1(CG/L)", "N2(CG/L)", "IP4(CG/L)", "E(CG/L)",
    # Add variant proportions:
    "XBB.1.5", "XBB.1.5.21", "BA.5", "JN.1", ...
]
```

**Benefits**:
- Different variants may have different transmission dynamics
- Variant proportions can serve as leading indicators for changes in viral load
- Enables multi-variant forecasting models

### 3. Multi-Variant Forecasting

Model each major variant separately:

```python
# Separate forecasts for major variants
variants_to_model = ["XBB.1.5", "JN.1", "KP.3", "LP.8", ...]

for variant in variants_to_model:
    # Extract time series for this variant across EDARs
    variant_ts = extract_variant_timeseries(variant)

    # Train separate forecaster or multi-task model
    model = train_forecaster(variant_ts)
```

**Benefits**:
- Capture variant-specific transmission parameters
- Early warning for emerging variants
- Understand competitive dynamics between variants

### 4. Variant-Onset Detection

Use variant emergence as a feature for predicting viral load changes:

```python
# Detect new variant emergence
def detect_variant_emergence(variant_ts, threshold=0.05):
    """Return dates when variant proportion exceeds threshold"""
    return variant_ts[variant_ts > threshold].index

# Use as binary feature
X["xbb_emergence"] = detect_variant_emergence(xbb_ts)
```

### 5. Spatiotemporal Variant Spread

Use graph structure to model variant spread between EDARs:

```python
# Transfer entropy between EDARs for specific variants
te_xbb = compute_transfer_entropy(xbb_ts, graph_edges)
te_jn1 = compute_transfer_entropy(jn1_ts, graph_edges)

# Compare leading indicators across variants
```

---

## Potential Research Questions

1. **Variant-specific viral kinetics**: Do different variants show different lag times between wastewater detection and clinical cases?

2. **Competitive displacement**: Can we predict when a new variant will outcompete existing variants based on wastewater dynamics?

3. **Spatial spread patterns**: Do variants spread through the EDAR network in predictable ways? Can we use this for early warning?

4. **Multi-variant modeling**: Does modeling variants separately improve overall forecasting accuracy compared to aggregate viral load?

5. **Variant-specific thresholds**: Do detection limits (LD) vary by variant? Should we adjust preprocessing accordingly?

---

## Preprocessing Considerations

### Data Quality Issues

1. **Sparse sampling**: Variant data is much sparser than viral load (1,331 entries vs 6,718 samples)
2. **Sequencing batch effects**: Different `run_id` values may have different detection thresholds
3. **Variant naming inconsistencies**: Same lineage may be reported as "BA.5" vs "B.1.1.529.5" (use grouping)

### Recommended Preprocessing Steps

```python
# 1. Join variant data to main wastewater samples
df = pd.merge(wastewater_df, variant_df,
              left_on="id mostra", right_on="codi", how="left")

# 2. Filter to high-confidence variants
min_proportion = 0.01  # 1% threshold
min_samples = 10       # Must appear in at least 10 samples

variant_counts = (df["variants"]
    .explode()
    .value_counts())
valid_variants = variant_counts[
    (variant_counts >= min_samples)
].index.tolist()

# 3. Pivot to wide format for time series modeling
variant_ts = df.pivot_table(
    index="date",
    columns="edar_id",
    values=[f"variant_{v}" for v in valid_variants]
)
```

---

## Next Steps

1. **Update main wastewater CSV** from sars-aigues
2. **Run sparsity analysis** on variant data (similar to `analyze_wastewater_sparsity.py`)
3. **Create variant processor** to join variant proportions to main wastewater dataset
4. **Exploratory analysis**: Plot variant dynamics over time, by EDAR
5. **Baseline comparison**: Train forecaster with vs without variant covariates

---

## References

- Zenodo: https://doi.org/10.5281/zenodo.4147073
- ICRA SARS AIGUA platform: https://sarsaigua.icra.cat/
- Current wastewater processor: `data/preprocess/processors/edar_processor.py`
- Sparsity analysis script: `dataviz/analyze_wastewater_sparsity.py`
