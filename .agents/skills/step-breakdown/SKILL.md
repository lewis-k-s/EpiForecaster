---
name: step-breakdown
description: Break down training step timing into phases (data load, forward, backward, optimizer) to identify where time is spent during training iterations. Use when analyzing training step performance or optimizing iteration speed.
allowed-tools: Bash, Read
---

# Training Step Breakdown

Decomposes a single training iteration into timing phases to identify where time is spent.

## Quick Start

```
step-breakdown <experiment_name> <run_id>
step-breakdown mn5_epiforecaster_full 34198361 --text
step-breakdown <path_to_trace.json>
```

## Input Formats

### Run-based analysis (recommended)
```
step-breakdown <experiment_name> <run_id>
```
Example: `step-breakdown mn5_epiforecaster_full 34198361`

### Direct file path
```
step-breakdown <path_to_trace.json>
```
Example: `step-breakdown outputs/training/mn5_epiforecaster_full/34198361/trace.json`

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "step-breakdown",
  "version": "1.0",
  "data": {
    "step_duration_ms": 5432.10,
    "phases": { /* phase timings */ }
  },
  "warnings": [],
  "meta": { "latency_ms": 12, "timestamp": "..." }
}
```

Use `--text` flag for human-readable output.

### Text output (--text) - Single trace
```
======================================================================
TRAINING STEP BREAKDOWN
======================================================================
Step duration:     5432.10 ms

PHASE BREAKDOWN
----------------------------------------------------------------------
Phase            | Time (ms)  | Percentage
----------------------------------------------------------------------
Data Load        |      12.34 |         2.3%
Forward          |    2345.68 |        43.2%
Backward         |    2876.54 |        52.9%
Optimizer        |      54.32 |         1.0%
Overhead         |       6.79 |         0.1%
```

### Multiple traces (aggregated)
Shows mean ± std across all epoch traces:
- Aggregate phase statistics
- Per-trace breakdown
- Identifies timing patterns across training

## Phase Interpretation

| Phase | Expected % | High % Indicates |
|-------|------------|------------------|
| Data Load | 2-5% | Slow workers, I/O bottleneck |
| Forward | 40-45% | Large model, complex ops |
| Backward | 45-55% | Normal for gradient computation |
| Optimizer | 1-2% | Complex optimizer, many params |
| Overhead | <1% | Excessive logging, sync issues |

## Typical Workflow

1. User provides experiment name and run ID
2. Use `Bash` tool to run: `step-breakdown <experiment> <run_id>`
3. Analyze phase percentages
4. Identify phases that deviate from expected ranges
5. Provide targeted optimization recommendations

## Commands

```bash
# Analyze all traces from a run
step-breakdown mn5_epiforecaster_full 34198361

# JSON output for programmatic use
step-breakdown mn5_epiforecaster_full 34198361 --json

# Analyze specific trace file
step-breakdown outputs/training/exp/run/trace.json
```
