---
name: perf-analyze
description: Analyze PyTorch profiler traces to identify performance bottlenecks in training runs. Use when user asks about training performance, GPU utilization, or bottleneck analysis.
allowed-tools: Bash, Read
---

# Performance Trace Analysis

Analyzes PyTorch profiler Chrome trace JSON files to extract performance metrics and identify bottlenecks.

## Quick Start

```
perf-analyze <path_to_trace.json>
perf-analyze <path_to_trace.json> --text
```

## Input Formats

### Direct file path
```
perf-analyze <path_to_trace.json>
```
Example: `perf-analyze outputs/training/mn5_epiforecaster_full/34198361/trace.json`

## Output Format

### JSON (default)
```json
{
  "ok": true,
  "name": "perf-analyze",
  "version": "1.0",
  "data": {
    "trace_duration_ms": 12345.67,
    "gpu_kernels": [ /* kernel timings */ ],
    "cpu_ops": [ /* operation timings */ ]
  },
  "warnings": [],
  "meta": { "latency_ms": 45, "timestamp": "..." }
}
```

Use `--text` flag for human-readable output.

### Text output (--text) - Single trace
```
======================================================================
TRACE METRICS EXTRACTED
======================================================================
Trace duration:    12345678 us
GPU utilization:   80.5 %
CPU/GPU ratio:     0.24
Data/Compute:      0.01
```

### Multiple traces (aggregated)
When analyzing a run with multiple epoch traces:
- Shows aggregate statistics (mean ± std)
- Lists per-trace breakdown
- Identifies patterns across epochs

## Key Metrics

| Metric | Good | Issue |
|--------|------|-------|
| GPU utilization | >70% | <50% = underutilized |
| CPU/GPU ratio | <0.5 | >1 = CPU-bound |
| Data/Compute | <0.2 | >0.5 = data bottleneck |

## Typical Workflow

1. User provides experiment name and run ID
2. Use `Bash` tool to run: `perf-analyze <experiment> <run_id>`
3. Interpret the metrics using the interpretation guide
4. Provide actionable recommendations based on bottlenecks found

## Commands

```bash
# Analyze all traces from a run
perf-analyze mn5_epiforecaster_full 34198361

# JSON output for programmatic use
perf-analyze mn5_epiforecaster_full 34198361 --json

# Analyze specific trace file
perf-analyze outputs/training/exp/run/trace.json
```
