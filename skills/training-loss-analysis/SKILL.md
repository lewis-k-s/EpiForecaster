---
name: training-loss-analysis
description: Analyze EpiForecaster training dynamics from TensorBoard logs by combining loss-curve and gradnorm diagnostics. Use when debugging unstable training, loss spikes, poor convergence, dead components, clipping behavior, or curriculum transition regressions.
---

# Training Loss Analysis

Run a two-pass diagnosis:
1. Analyze loss behavior with `loss-curve-critic`.
2. Analyze gradient/component health with `gradnorm-analyze`.
3. Correlate findings and propose targeted training fixes.

## Required Inputs

- TensorBoard event directory path, or
- Experiment name under `outputs/training/<experiment>/` (latest run auto-resolved), or
- Optuna trial path for gradnorm (`--optuna`).

## Commands

```bash
# 1) Loss analysis (JSON)
uv run loss-curve-critic <path-or-experiment>

# 1b) Loss analysis (human-readable)
uv run loss-curve-critic <path-or-experiment> --text --diagnose

# 2) Gradnorm analysis (JSON)
uv run gradnorm-analyze <path-or-experiment>

# 2b) Gradnorm analysis for Optuna trial
uv run gradnorm-analyze <trial-path> --optuna --text
```

## Triage Workflow

1. Run both analyzers on the same run.
2. Prioritize hard failures first:
   - Severe loss spikes or divergence
   - Dead components (`DEAD [...]`)
   - Extreme imbalance (`ForecasterHead` dominance or very low `MobilityGNN` share)
   - Heavy clipping
3. Cross-check correlations:
   - Loss spikes + grad spikes => optimizer/learning-rate instability
   - Loss spikes + low/flat gradnorm => data/curriculum/label or loss-function issue
4. Recommend minimal, testable changes:
   - LR/clip adjustments
   - Curriculum smoothing at transitions
   - Component-specific LR or loss weighting
   - Data quality checks for outlier batches

## Output Contract

Return:
1. Top 3 issues with evidence (metric + value + threshold/expectation).
2. Most likely root cause for each issue.
3. Ordered remediation plan with smallest safe change first.
4. Validation plan (which metrics/flags should improve on next run).

## Implementation Notes

- Loss script: `scripts/analyze_loss_curve.py`
- Gradnorm script: `scripts/analyze_gradnorm.py`
- Prefer `--text` for human debugging and JSON output for structured follow-up.
- If either command fails on path resolution, retry with the explicit event directory.
