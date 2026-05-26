---
name: loss-curve-critic
description: Analyze training loss curves from TensorBoard logs to detect spikes, instability, convergence issues, and training anomalies. Use when investigating training dynamics, loss spikes, or convergence problems.
allowed-tools: Bash, Read
---

# Loss Curve Critic

Analyzes TensorBoard event files to extract training loss metrics and diagnose training issues including spikes, instability, convergence problems, and curriculum-related anomalies.

## Quick Start

```
loss-curve-critic <path_to_events_dir>
loss-curve-critic mn5_epiforecaster_full --text
loss-curve-critic outputs/training/sparsity_curriculum/35672072/ --diagnose
```

## Purpose

Use this skill when you want to understand:
- **Loss spikes**: Sudden increases in training loss and their causes
- **Training stability**: Is the loss stable or highly volatile?
- **Convergence health**: Is the model converging properly or stalling?
- **Curriculum effects**: How do curriculum transitions affect loss?
- **Gradient-loss correlation**: Are loss spikes related to gradient spikes?
- **Data loading issues**: Are long data loads correlated with loss spikes?

## Input Formats

### Direct path (simplest)
```
loss-curve-critic <path_to_events_dir>
```
Example: `loss-curve-critic outputs/training/mn5_epiforecaster_full/34198361/`

### Experiment name (finds most recent run)
```
loss-curve-critic <experiment_name>
```
Example: `loss-curve-critic sparsity_curriculum`

### With diagnostic mode
```
loss-curve-critic <path> --diagnose
```
Shows additional diagnostic information including spike timing analysis and correlations.

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "loss-curve-critic",
  "version": "1.0",
  "data": {
    "statistics": { /* loss stats */ },
    "spikes": { /* spike detection */ },
    "stability": { /* volatility metrics */ },
    "convergence": { /* convergence analysis */ },
    "correlations": { /* with other metrics */ }
  },
  "warnings": ["SPIKES: 35 spikes detected (3.7% of steps)"],
  "meta": { "latency_ms": 120, "timestamp": "..." }
}
```

### Console output (--text)
```
================================================================================
LOSS CURVE ANALYSIS
================================================================================

LOSS STATISTICS
--------------------------------------------------------------------------------
Steps:              936
Min:                0.0865
Max:                11.6005
Mean:               0.9164
Median:             0.7259
Std Dev:            1.0367
Coefficient of Variation: 1.131

SPIKE ANALYSIS
--------------------------------------------------------------------------------
High spikes (>3x median):     35 (3.7% of steps)
Severe spikes (>10x median):   5 (0.5% of steps)

First 10 spikes:
  Step 50:   3.13 (4.3x median)
  Step 60:   2.39 (3.3x median)
  Step 80:   2.41 (3.3x median)
  Step 1050: 10.66 (14.7x median) ← SEVERE
  Step 1060:  7.18 (9.9x median)
  ...

STABILITY METRICS
--------------------------------------------------------------------------------
Rolling volatility (window=50): 0.453
Max upward jump:        11.39 (step 1089 → 1090)
Max downward drop:      0.64 (within normal range)
Recovery rate:          72.7% of spikes recover within 10 steps

CONVERGENCE ANALYSIS
--------------------------------------------------------------------------------
Trend slope (log loss):  -0.012 (decreasing → converging)
Plateaus detected:       2 (epochs 3-5, 12-15)
Convergence rate:        0.856 loss units per 1k steps

CURRICULUM ANALYSIS
--------------------------------------------------------------------------------
Epoch boundary spikes:   4 at transitions (epochs 3, 7, 12)
Last epoch loss:         0.523 (32% reduction from start)

CORRELATIONS
--------------------------------------------------------------------------------
Loss-GradNorm correlation:    0.234 (weak positive)
Loss-DataLoad correlation:    0.512 (moderate positive)

FLAGS
--------------------------------------------------------------------------------
  ⚠ SPIKES: 35 high spikes detected (3.7% of steps)
  ⚠ SEVERE: 5 severe spikes detected (>10x median)
  ⚠ VOLATILITY: High loss volatility (CV=1.13)
  ⚠ CURRICULUM: Spike activity at epoch transitions

================================================================================
```

## Metrics Explained

| Metric | Description | Good | Issue |
|--------|-------------|------|-------|
| **High spikes** | Loss values >3x median | <1% | >5% indicates instability |
| **Severe spikes** | Loss values >10x median | 0 | Any indicates serious issue |
| **CV** | Coefficient of variation (std/mean) | <0.5 | >1.0 = high volatility |
| **Rolling volatility** | Std of rolling window | <0.3 | >0.5 = unstable |
| **Recovery rate** | % of spikes that recover | >80% | <50% = training divergence |
| **Trend slope** | Log-loss change per step | Negative | Positive = diverging |
| **Plateaus** | Extended flat loss periods | <2 | Many = stalling |
| **Correlation** | With gradnorm/dataload | - | High = specific cause |

## Interpretation Guide

### High Spike Frequency
- **Flag**: `SPIKES: N high spikes detected (X% of steps)`
- **Meaning**: Frequent loss excursions indicate training instability
- **Causes**:
  - Learning rate too high
  - Batch anomalies (corrupted data, outliers)
  - Numerical instability in loss computation
  - Curriculum transitions (sudden data distribution shift)
- **Actions**:
  - Lower learning rate or add warmup
  - Check data quality and preprocessing
  - Add gradient clipping (check if grad spikes correlate)
  - Smooth curriculum transitions

### Severe Spikes
- **Flag**: `SEVERE: N severe spikes detected (>10x median)`
- **Meaning**: Extreme loss values that may indicate NaN gradients or corrupted batches
- **Causes**:
  - NaN/Inf in gradients or model outputs
  - Division by zero in loss computation
  - Empty batches or graph components
  - Numerical overflow in custom loss components
- **Actions**:
  - Check for NaN/Inf in gradients
  - Add epsilon to division operations
  - Validate batch contents
  - Check loss component implementations

### High Volatility
- **Flag**: `VOLATILITY: High loss volatility (CV=X.XX)`
- **Meaning**: Loss fluctuates wildly step-to-step
- **Causes**:
  - Small batch size
  - High-variance data (especially with synthetic data sparsity)
  - Inconsistent data quality
  - Noisy labels
- **Actions**:
  - Increase batch size
  - Add gradient accumulation
  - Smooth or filter input data
  - Use loss smoothing (exponential moving average)

### Poor Recovery Rate
- **Flag**: `RECOVERY: Only X% of spikes recover (threshold: 50%)`
- **Meaning**: After a spike, loss doesn't return to baseline
- **Causes**:
  - Model state corruption (e.g., dead neurons)
  - Optimization divergence
  - Catastrophic forgetting (curriculum shifts)
- **Actions**:
  - Reduce learning rate
  - Add learning rate decay on spike detection
  - Check for gradient clipping effectiveness
  - Consider checkpoint-based rollback

### Curriculum Transition Spikes
- **Flag**: `CURRICULUM: Spike activity at epoch transitions`
- **Meaning**: Loss increases when curriculum parameters change
- **Causes**:
  - Abrupt data distribution shift (e.g., sparsity jump)
  - Model not adapted to new difficulty level
  - Learning rate too high for new regime
- **Actions**:
  - Smooth curriculum transitions (interpolate parameters)
  - Reduce learning rate at transitions
  - Add warmup period after each transition
  - Validate data pipeline handles parameter changes

### Loss-GradNorm Correlation
- **High positive (>0.5)**: Loss spikes driven by gradient spikes
  - Check gradient clipping threshold
  - Verify loss computation stability
- **Moderate positive (0.2-0.5)**: Some coupling, expected
- **Low/negative**: Loss spikes from other sources
  - Check data loading anomalies
  - Verify batch composition

### Loss-DataLoad Correlation
- **High positive**: Long data loads precede loss spikes
  - Check for data pipeline bottlenecks
  - Verify NVMe staging is working
  - May indicate prefetch issues causing stale data

## Typical Workflow

1. User provides event directory path or experiment name
2. Use `Bash` tool to run: `python scripts/analysis/analyze_loss_curve.py <path>`
3. Interpret the flags section for actionable issues
4. Use `--diagnose` for deeper analysis of spike timing and causes
5. Provide targeted recommendations based on flagged issues

## Commands

```bash
# Analyze specific run
python scripts/analysis/analyze_loss_curve.py outputs/training/sparsity_curriculum/35672072/

# Human-readable output
python scripts/analysis/analyze_loss_curve.py outputs/training/sparsity_curriculum/35672072/ --text

# With detailed diagnostics
python scripts/analysis/analyze_loss_curve.py outputs/training/sparsity_curriculum/35672072/ --diagnose

# By experiment name (finds most recent run)
python scripts/analysis/analyze_loss_curve.py sparsity_curriculum
```

## Implementation Notes

The skill invokes `scripts/analysis/analyze_loss_curve.py` which:
- Uses `tensorboard.backend.event_processing.event_accumulator` to read event files
- Detects spikes using robust thresholds (3x median for high, 10x for severe)
- Computes volatility using rolling standard deviation (window=50 steps)
- Analyzes convergence using log-loss trend slope
- Detects plateaus using extended low-variance windows
- Computes correlations with gradnorm and data load time
- Flags issues based on thresholds defined in the script

Key thresholds (configurable in script):
- High spike: 3x median loss
- Severe spike: 10x median loss
- Volatility warning: CV > 1.0 or rolling std > 0.5
- Recovery threshold: 50% of spikes must recover within 10 steps
- Plateau window: 20 consecutive steps with variance < 1% of mean
- Correlation thresholds: >0.5 = strong, 0.2-0.5 = moderate
