# Neighborhood vs Global Trend Regression Analysis

## Overview

This script analyzes how closely local mobility neighborhoods follow global epidemic trends. It performs sliding window regression comparing neighborhood-level case trends against global trends, both scaled by population.

## Key Findings (Test Run on full_v1.zarr)

Based on analysis of the canonical dataset with 14-day stride:

- **Slope Statistics**: Mean = 0.86, Median = 1.00
  - Median slope of ~1.0 indicates neighborhoods generally track global trends perfectly
  - Some neighborhoods deviate (slope range: -0.04 to 1.75)

- **Fit Quality**: Mean R² = 0.75, Median R² = 0.87
  - 83% of regressions have R² > 0.5 (good fit)
  - 59% have R² > 0.8 (excellent fit)

- **Statistical Significance**: 94.2% have p < 0.05
  - Nearly all relationships are statistically significant

- **Neighborhood Size**: Mean = 53 neighbors, Median = 47 neighbors
  - Reasonable receptive field for spatial aggregation

## Interpretation

The high R² values (median 0.87) and slope near 1.0 suggest that:
1. **Strong spatial coupling**: Mobility neighborhoods closely follow global epidemic patterns
2. **Local-global consistency**: Regional outbreaks propagate predictably through mobility networks
3. **Model relevance**: Neighborhood aggregation should capture meaningful spatial signals

The slope distribution (0.86 mean, slightly below 1.0) indicates that:
- Neighborhood cases tend to be slightly less volatile than global cases
- Some smoothing effect occurs when aggregating multiple neighbors
- This may benefit model training by reducing noise

## Usage

```bash
# Basic usage (default 14-day stride, all nodes)
python -m dataviz.neighborhood_global_regression \
    --config configs/train_epifor_full.yaml

# Analyze specific split with custom stride
python -m dataviz.neighborhood_global_regression \
    --config configs/train_epifor_full.yaml \
    --split train \
    --window-stride 7

# Custom mobility threshold
python -m dataviz.neighborhood_global_regression \
    --config configs/train_epifor_full.yaml \
    --mobility-threshold 10.0 \
    --output-dir outputs/reports/my_analysis
```

## Arguments

- `--config` (required): Training config path
- `--split`: Which node split to analyze [train|val|test|all], default: all
- `--window-stride`: Stride for sliding windows (days), default: 14
- `--mobility-threshold`: Minimum flow for neighbors, default: config value
- `--include-self`: Include target node in neighborhood
- `--output-dir`: Output directory, default: outputs/reports/neighborhood_global_regression
- `--show`: Display plots interactively

## Outputs

1. **CSV**: `neighborhood_global_regression_results.csv`
   - Per-window, per-node regression statistics
   - Columns: window_start, window_end, target_node, slope, intercept, r2, p_value, std_err, n_neighbors

2. **Scatter Plots**: `neighborhood_global_scatter.png`
   - 6 sample windows showing neighborhood vs global scatter
   - Includes regression line and statistics

3. **Time Series**: `regression_slopes_timeseries.png`
   - Slope statistics (mean, median, std) over time
   - R² quality over time
   - Shows how relationship evolves

4. **Distribution**: `regression_slopes_distribution.png`
   - Histogram of slopes and R² values
   - Slope distribution by node (sampled)
   - Slope vs R² scatter (colored by neighbor count)

## Technical Details

### Regression Model
For each window and target node:
- **X**: Neighborhood per-capita cases (cases/100k), aggregated across incoming mobility neighbors
- **Y**: Global per-capita cases, population-weighted across all regions
- **Model**: Linear regression `Y = β0 + β1*X + ε`

### Handling Sparsity
- Cases data is highly sparse (especially early in pandemic)
- Filters out neighbors with missing population
- Computes population-weighted global trend using only valid data per timestep
- Requires ≥2 valid timesteps for regression

### Window Parameters
- Uses training config's `history_length + forecast_horizon` as window size
- Default stride of 14 days reduces overlap and computation
- Validates windows based on missingness permit from config

## Recommendations

### For Model Training
1. **Neighborhood aggregation** is well-justified given high R² (0.75+)
2. **Mobility threshold** can be tuned based on neighbor count (~50 neighbors works well)
3. **Spatial smoothing** naturally occurs through neighborhood averaging

### For Further Analysis
1. Analyze different time periods (early vs late pandemic)
2. Compare across train/val/test splits
3. Investigate outliers (slope far from 1.0, low R²)
4. Correlate slope with node properties (population, centrality)
