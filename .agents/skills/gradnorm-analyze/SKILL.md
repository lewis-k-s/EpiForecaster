---
name: gradnorm-analyze
description: Analyze TensorBoard gradnorm logs to assess component learning health, training stability, and gradient balance. Use when investigating training dynamics, dead components, or gradient instability.
allowed-tools: Bash, Read
---

# Gradnorm Analysis

Analyzes TensorBoard event files to extract gradient norm metrics and provides insights into model training dynamics at the component level.

## Quick Start

```
gradnorm-analyze <path_to_events_dir>
gradnorm-analyze mn5_epiforecaster_full --text
gradnorm-analyze --hpsearch-trial <trial_path>
```

## Purpose

Use this skill when you want to understand:
- **Component learning health**: Which model components are learning vs. dead?
- **Training stability**: Are there gradient spikes, vanishing gradients, or volatility?
- **Component balance**: Is each module (GNN, ForecasterHead) contributing appropriately?
- **Loss-gradnorm correlation**: How do gradient patterns relate to loss changes?

## Input Formats

### Direct path (simplest)
```
gradnorm-analyze <path_to_events_dir>
```
Example: `gradnorm-analyze outputs/training/mn5_epiforecaster_full/34198361/`

### Optuna trial
```
gradnorm-analyze --hpsearch-trial <trial_path>
```
Example: `gradnorm-analyze --hpsearch-trial outputs/hpsearch/hpsearch-opt/34798953--0_trial28_1767984350052165109/`

### Experiment name (finds most recent run)
```
gradnorm-analyze <experiment_name>
```
Example: `gradnorm-analyze mn5_epiforecaster_full`

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "gradnorm-analyze",
  "version": "1.0",
  "data": { /* component metrics */ },
  "warnings": ["DEAD [Other]: 100.0% steps below floor"],
  "meta": { "latency_ms": 245, "timestamp": "..." }
}
```

Use `--text` flag for human-readable output.

### Console output (--text)
```
================================================================================
GRADNORM ANALYSIS
================================================================================

PER-COMPONENT STATISTICS
--------------------------------------------------------------------------------
Component            Median  95th %ile   Share %   Dead %   Spikes      Vol
--------------------------------------------------------------------------------
MobilityGNN        0.004196   0.010256      7.3%     0.0%        0   1.116
ForecasterHead     0.065995   0.153319     99.7%     0.0%        3   0.514
Other              0.000000   0.000000      0.0%   100.0%        0   0.000
--------------------------------------------------------------------------------
Total_PreClip      0.066448   0.153369     100.0%        -        3   0.513

CLIP RATIO (Clipped_Total / Total_PreClip)
----------------------------------------
Median: 2.5034
95th %ile: 7.2403

LOSS-GRADNORM CORRELATION
----------------------------------------
Correlation: 0.1304
Best lag (4 steps): 0.2475

FLAGS
--------------------------------------------------------------------------------
  ⚠ DEAD [Other]: 100.0% steps below floor (threshold: 5.0%)
  ⚠ SPIKES: 3 spikes detected (1.0 per 1k steps)
  ⚠ Dominance: ForecasterHead at 99.7% (may indicate component imbalance)
```

### JSON output (programmatic)
```
gradnorm-analyze <path> --json
```
Returns structured JSON with all metrics for programmatic analysis.

## Metrics Explained

| Metric | Description | Good | Issue |
|--------|-------------|------|-------|
| **Share %** | Component's contribution to total gradnorm | Balanced per module | One module >> 90% suggests imbalance |
| **Dead %** | % of steps with gradnorm < 1e-6 | <5% | >5% indicates dead/lazy component |
| **Spikes** | Count of gradient spikes (robust z-score >5) | <1 per 100 steps | Frequent spikes = instability |
| **Vol** | Volatility (rolling std of log gradnorm) | <0.5 | >1.0 = high variance |
| **Clip Ratio** | Clipped_Total / Total_PreClip | ~1.0 | <0.9 = frequent clipping |
| **Correlation** | Loss vs log(gradnorm) correlation | Positive | Negative = inverse relationship |

## Interpretation Guide

### Dead Components
- **Flag**: `DEAD [Component]: X% steps below floor`
- **Meaning**: Component gradients are near-zero most of the time
- **Causes**: Over-regularization, dead ReLU, poor initialization, disconnected graph
- **Action**: Check learning rates, regularization, or module connectivity

### Gradient Spikes
- **Flag**: `SPIKES: N spikes detected (X per 1k steps)`
- **Meaning**: Sudden large gradient excursions
- **Causes**: Batch anomalies, numerical instability, learning rate too high
- **Action**: Lower learning rate, add gradient clipping, check data quality

### Component Imbalance
- **Flag**: `IMBALANCE: MobilityGNN under-contributing at X%`
- **Flag**: `Dominance: ForecasterHead at X%`
- **Meaning**: One module dominates gradient signal
- **Causes**: Architecture mismatch, one module much larger than others
- **Action**: Adjust layer sizes, learning rates per module, or loss weighting

### High Volatility
- **Flag**: `VOLATILITY: High gradnorm volatility (X)`
- **Meaning**: Gradients vary wildly step-to-step
- **Causes**: Noisy data, small batch sizes, inconsistent inputs
- **Action**: Increase batch size, smooth inputs, add gradient smoothing

### Frequent Clipping
- **Flag**: `CLIPPING: Frequent gradient clipping detected`
- **Meaning**: Gradients exceed clip threshold often
- **Causes**: Learning rate too high, aggressive targets
- **Action**: Lower learning rate, increase clip threshold

## Typical Workflow

1. User provides event directory path or experiment name
2. Use `Bash` tool to run: `python scripts/analysis/analyze_gradnorm.py <path>`
3. Interpret the flags section for actionable issues
4. For deeper investigation, use `--json` and analyze specific metrics
5. Provide targeted recommendations based on flagged issues

## Commands

```bash
# Analyze specific run
python scripts/analysis/analyze_gradnorm.py outputs/training/mn5_epiforecaster_full/34198361/

# JSON output for programmatic use
python scripts/analysis/analyze_gradnorm.py outputs/training/mn5_epiforecaster_full/34198361/ --json

# Analyze Optuna trial
python scripts/analysis/analyze_gradnorm.py --hpsearch-trial outputs/hpsearch/hpsearch-opt/34798953--0_trial28_1767984350052165109/

# By experiment name (finds most recent run)
python scripts/analysis/analyze_gradnorm.py mn5_epiforecaster_full
```

## Implementation Notes

The skill invokes `scripts/analysis/analyze_gradnorm.py` which:
- Uses `tensorboard.backend.event_processing.event_accumulator` to read event files
- Computes per-component statistics (median, p95, deadness, spikes, volatility)
- Detects spikes using robust z-score (median/MAD) with threshold of 5 MAD
- Calculates component shares relative to Total_PreClip
- Computes loss-gradnorm correlation with lagged analysis
- Flags issues based on thresholds defined in the script

Key thresholds (configurable in script):
- Deadness floor: 1e-6 (flag if >5% of steps below)
- Spike threshold: 5 MAD from median
- MobilityGNN expected share: 5-20%
