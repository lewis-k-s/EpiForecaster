---
name: hpsearch-analyze
description: Analyze HPO journal hyperparameter search results to extract summary statistics, best trials, parameter importance, and search space recommendations.
allowed-tools: Bash, Read
---

# HPO Analysis

Analyze HPO journal hyperparameter search results.

This skill parses Optuna journal files and produces comprehensive analysis of hyperparameter optimization results, including summary statistics, best trials, parameter importance, and recommendations for search space refinement.

## Quick Start

```
/hpsearch-analyze outputs/hpsearch/epiforecaster_hpo_v1.journal
/hpsearch-analyze outputs/hpsearch/epiforecaster_hpo_v1.journal --top 10 --importance --text
```

## Usage

```bash
/hpsearch-analyze outputs/hpsearch/epiforecaster_hpo_v1.journal
```

## Arguments

- `journal_path`: Path to the journal file (e.g., `outputs/hpsearch/epiforecaster_hpo_v1.journal`)

## Optional Arguments

- `--top N`: Show top N trials (default: 5)
- `--importance`: Compute and show parameter importance analysis

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "hpsearch-analyze",
  "version": "1.0",
  "data": {
    "summary": { /* study statistics */ },
    "best_trials": [ /* top trials */ ],
    "param_distributions": { /* param stats */ },
    "recommendations": [ /* refinement suggestions */ ]
  },
  "warnings": [],
  "meta": { "latency_ms": 842, "timestamp": "..." }
}
```

Use `--text` flag for human-readable output.

## What It Does

1. **Parses the journal** - Reads the line-delimited JSON journal file and extracts all trial information including parameters, objective values, and trial states

2. **Prints summary statistics** - Shows total trials, completed/failed/running counts, and objective value statistics (best, median, mean, std, worst)

3. **Shows best trials** - Displays the top N trials with their complete hyperparameter configurations

4. **Analyzes parameter distributions** - For each hyperparameter, shows:
   - Numeric params: min, max, mean, median
   - Categorical params: frequency counts and percentages

5. **Computes parameter importance** - Uses Spearman correlation to rank parameters by their impact on the objective

6. **Provides recommendations** - Suggests narrowed search ranges based on the 80th percentile interval of completed trials

## Example Output

```
============================================================
Study: epiforecaster_hpo_v1
============================================================
Total trials:     53
Completed:        37
Failed:           4
Running:          12

Objective (best_val_loss):
  Best:            0.200761
  Median:          0.237912
  Mean:            0.234699
  Std:             0.016659

...

Parameter Importance (abs(Spearman correlation))
============================================================
  training.learning_rate: 0.788
  model.gnn_depth: 0.407
  training.batch_size: 0.226
  ...
```

## Implementation

The skill runs the `scripts/analysis/analyze_hpsearch_journal.py` script, which can also be invoked directly:

```bash
uv run python scripts/analysis/analyze_hpsearch_journal.py outputs/hpsearch/epiforecaster_hpo_v1.journal --top 10 --importance
```
