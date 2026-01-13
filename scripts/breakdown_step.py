"""Extract single training step breakdown from PyTorch profiler trace.

This script analyzes Chrome trace format JSON files to identify and categorize
operations within a single training iteration, providing timing breakdown by phase.

Usage:
    # Analyze all traces from a run
    python scripts/breakdown_step.py <experiment_name> <run_id> [--json]

    # Analyze a single trace file (legacy)
    python scripts/breakdown_step.py <path_to_trace_json> [--json]

Phases detected:
    - data_load: Time spent loading/preparing batch
    - forward: Model forward pass
    - backward: Gradient computation (backward pass)
    - optimizer: Optimizer.step() and related operations
    - overhead: Logging, checkpointing, synchronization
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from utils.run_discovery import resolve_trace_paths


@dataclass
class StepEvent:
    """A single event within a training step."""

    name: str
    category: str
    duration_us: float
    start_us: float
    end_us: float
    phase: str  # 'data_load', 'forward', 'backward', 'optimizer', 'overhead'


@dataclass
class StepBreakdown:
    """Breakdown of a single training step."""

    step_duration_us: float
    step_count: int

    # Phase timings (microseconds)
    data_preprocessing_time_us: float  # Actual CPU work (getitem, collate)
    data_wait_time_us: float  # Time blocked on dataloader (enumerate events)
    data_load_time_us: float  # Total (for backward compatibility)
    forward_time_us: float
    backward_time_us: float
    optimizer_time_us: float
    overhead_time_us: float

    # Phase percentages
    data_load_pct: float  # Total (for backward compatibility)
    data_preprocessing_pct: float  # Breakdown percentage
    data_wait_pct: float  # Breakdown percentage
    forward_pct: float
    backward_pct: float
    optimizer_pct: float
    overhead_pct: float

    # Detailed events per phase
    forward_ops: list[tuple[str, float, int]] = field(default_factory=list)
    backward_ops: list[tuple[str, float, int]] = field(default_factory=list)
    optimizer_ops: list[tuple[str, float, int]] = field(default_factory=list)
    data_ops: list[tuple[str, float, int]] = field(default_factory=list)

    # Per-step timing for each detected step
    per_step_times_us: list[float] = field(default_factory=list)


def classify_operation(name: str, category: str) -> str:
    """Classify an operation into a training phase based on name patterns."""
    name_lower = name.lower()

    # Forward pass indicators
    forward_patterns = {
        "forward",
        "model",
        "embedding",
        "gnn",
        "attention",
        "transformer",
        "linear",
        "conv",
        "relu",
        "dropout",
        "layernorm",
        "batchnorm",
        "softmax",
        "cat",
        "stack",
        "addmm",
        "matmul",
        "bmm",
    }

    # Backward pass indicators
    backward_patterns = {
        "backward",
        "grad",
        "autograd",
        "torch::autograd::",
        "accumgrad",
    }

    # Optimizer step indicators
    optimizer_patterns = {
        "optimizer",
        "adam",
        "sgd",
        "step",
        "lr_scheduler",
    }

    # Data loading indicators - split into preprocessing vs wait
    data_wait_patterns = {
        "enumerate",  # Specifically catches DataLoader.__next__ waits
    }

    data_preprocess_patterns = {
        "collate",
        "getitem",
        "worker",
        "mkl",
        "numpy",
        "index",
        "slice",
        "batch",
    }

    # Check patterns - order matters! backward takes precedence
    if any(p in name_lower for p in backward_patterns):
        return "backward"
    if any(p in name_lower for p in optimizer_patterns):
        return "optimizer"
    if any(p in name_lower for p in data_wait_patterns):
        return "data_wait"
    if any(p in name_lower for p in data_preprocess_patterns):
        return "data_preprocessing"
    if any(p in name_lower for p in forward_patterns):
        return "forward"

    # Default: if it's a kernel, likely compute (forward/backward)
    if category == "kernel":
        return "forward"  # Most kernels are forward

    return "overhead"


def extract_step_breakdown(trace_path: str) -> StepBreakdown:
    """Extract step breakdown from Chrome trace JSON."""
    with open(trace_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}", file=sys.stderr)
            sys.exit(1)

    events = data.get("traceEvents", [])
    if not events:
        print("No traceEvents found.", file=sys.stderr)
        sys.exit(1)

    # Filter complete events with duration
    complete_events = [
        e
        for e in events
        if "dur" in e and "ts" in e and e["dur"] > 0
    ]

    if not complete_events:
        print("No complete events found in trace.", file=sys.stderr)
        sys.exit(1)

    # Classify events by phase and aggregate
    phase_times: defaultdict[str, float] = defaultdict(float)
    phase_ops: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for event in complete_events:
        name = event.get("name", "unknown")
        cat = event.get("cat", "unknown")
        duration_us = event["dur"]

        phase = classify_operation(name, cat)
        phase_times[phase] += duration_us
        phase_ops[phase][name].append(duration_us)

    # Build top ops per phase (name, total_duration, count)
    def build_top_ops(phase_ops_dict: defaultdict[str, list[float]]) -> list[tuple[str, float, int]]:
        return [
            (name, sum(durs), len(durs))
            for name, durs in phase_ops_dict.items()
        ]

    forward_ops = sorted(
        build_top_ops(phase_ops["forward"]), key=lambda x: x[1], reverse=True
    )
    backward_ops = sorted(
        build_top_ops(phase_ops["backward"]), key=lambda x: x[1], reverse=True
    )
    optimizer_ops = sorted(
        build_top_ops(phase_ops["optimizer"]), key=lambda x: x[1], reverse=True
    )
    # Combine data_wait and data_preprocessing for data_ops (for backward compatibility)
    data_ops = sorted(
        build_top_ops(phase_ops["data_wait"] | phase_ops["data_preprocessing"]), key=lambda x: x[1], reverse=True
    )

    # Calculate total step duration
    step_start = min(e["ts"] for e in complete_events)
    step_end = max(e["ts"] + e["dur"] for e in complete_events)
    step_duration_us = step_end - step_start

    # Phase times
    data_preprocessing_time_us = phase_times["data_preprocessing"]
    data_wait_time_us = phase_times["data_wait"]
    data_load_time_us = data_preprocessing_time_us + data_wait_time_us  # For compatibility
    forward_time_us = phase_times["forward"]
    backward_time_us = phase_times["backward"]
    optimizer_time_us = phase_times["optimizer"]
    overhead_time_us = phase_times["overhead"]

    # Calculate percentages
    total = step_duration_us or 1  # Avoid div by zero
    data_load_pct = (data_load_time_us / total) * 100
    data_preprocessing_pct = (data_preprocessing_time_us / total) * 100
    data_wait_pct = (data_wait_time_us / total) * 100
    forward_pct = (forward_time_us / total) * 100
    backward_pct = (backward_time_us / total) * 100
    optimizer_pct = (optimizer_time_us / total) * 100
    overhead_pct = (overhead_time_us / total) * 100

    # Estimate step count by looking for iteration markers
    # This is heuristic - we look for patterns that suggest multiple steps
    step_count = 1  # Default to single step

    return StepBreakdown(
        step_duration_us=step_duration_us,
        step_count=step_count,
        data_preprocessing_time_us=data_preprocessing_time_us,
        data_wait_time_us=data_wait_time_us,
        data_load_time_us=data_load_time_us,
        forward_time_us=forward_time_us,
        backward_time_us=backward_time_us,
        optimizer_time_us=optimizer_time_us,
        overhead_time_us=overhead_time_us,
        data_load_pct=data_load_pct,
        data_preprocessing_pct=data_preprocessing_pct,
        data_wait_pct=data_wait_pct,
        forward_pct=forward_pct,
        backward_pct=backward_pct,
        optimizer_pct=optimizer_pct,
        overhead_pct=overhead_pct,
        forward_ops=forward_ops[:15],
        backward_ops=backward_ops[:15],
        optimizer_ops=optimizer_ops[:15],
        data_ops=data_ops[:15],
    )


def format_text(breakdown: StepBreakdown) -> str:
    """Format step breakdown as readable text."""
    lines = [
        "=" * 70,
        "TRAINING STEP BREAKDOWN",
        "=" * 70,
        "",
        f"Step duration:     {breakdown.step_duration_us:>12.0f} us  ({breakdown.step_duration_us / 1000:>8.2f} ms)",
        f"Steps analyzed:    {breakdown.step_count}",
        "",
        "PHASE BREAKDOWN",
        "-" * 70,
        f"{'Phase':<15} | {'Time (us)':>12} | {'Time (ms)':>10} | {'Percentage':>12}",
        "-" * 70,
        f"{'Data Preprocess':<15} | {breakdown.data_preprocessing_time_us:>12.0f} | {breakdown.data_preprocessing_time_us / 1000:>10.2f} | {breakdown.data_preprocessing_pct:>11.1f}%",
        f"{'Data Wait':<15} | {breakdown.data_wait_time_us:>12.0f} | {breakdown.data_wait_time_us / 1000:>10.2f} | {breakdown.data_wait_pct:>11.1f}%",
        f"{'Data Load (total)':<15} | {breakdown.data_load_time_us:>12.0f} | {breakdown.data_load_time_us / 1000:>10.2f} | {breakdown.data_load_pct:>11.1f}%",
        f"{'Forward':<15} | {breakdown.forward_time_us:>12.0f} | {breakdown.forward_time_us / 1000:>10.2f} | {breakdown.forward_pct:>11.1f}%",
        f"{'Backward':<15} | {breakdown.backward_time_us:>12.0f} | {breakdown.backward_time_us / 1000:>10.2f} | {breakdown.backward_pct:>11.1f}%",
        f"{'Optimizer':<15} | {breakdown.optimizer_time_us:>12.0f} | {breakdown.optimizer_time_us / 1000:>10.2f} | {breakdown.optimizer_pct:>11.1f}%",
        f"{'Overhead':<15} | {breakdown.overhead_time_us:>12.0f} | {breakdown.overhead_time_us / 1000:>10.2f} | {breakdown.overhead_pct:>11.1f}%",
        "-" * 70,
        f"{'TOTAL':<15} | {breakdown.step_duration_us:>12.0f} | {breakdown.step_duration_us / 1000:>10.2f} | {100.0:>11.1f}%",
        "",
        "TOP FORWARD OPERATIONS (name | total_us | count | avg_us)",
        "-" * 70,
    ]

    if breakdown.forward_ops:
        for name, dur_us, count in breakdown.forward_ops[:10]:
            avg_us = dur_us / count if count > 0 else 0
            lines.append(f"{name[:45]:<45} | {dur_us:>10.0f} | {count:>4} | {avg_us:>8.1f}")
    else:
        lines.append("(No forward operations detected)")

    lines.extend([
        "",
        "TOP BACKWARD OPERATIONS (name | total_us | count | avg_us)",
        "-" * 70,
    ])

    if breakdown.backward_ops:
        for name, dur_us, count in breakdown.backward_ops[:10]:
            avg_us = dur_us / count if count > 0 else 0
            lines.append(f"{name[:45]:<45} | {dur_us:>10.0f} | {count:>4} | {avg_us:>8.1f}")
    else:
        lines.append("(No backward operations detected)")

    lines.extend([
        "",
        "TOP OPTIMIZER OPERATIONS (name | total_us | count | avg_us)",
        "-" * 70,
    ])

    if breakdown.optimizer_ops:
        for name, dur_us, count in breakdown.optimizer_ops[:10]:
            avg_us = dur_us / count if count > 0 else 0
            lines.append(f"{name[:45]:<45} | {dur_us:>10.0f} | {count:>4} | {avg_us:>8.1f}")
    else:
        lines.append("(No optimizer operations detected)")

    if breakdown.data_ops:
        lines.extend([
            "",
            "TOP DATA LOADING OPERATIONS (name | total_us | count | avg_us)",
            "-" * 70,
        ])
        for name, dur_us, count in breakdown.data_ops[:10]:
            avg_us = dur_us / count if count > 0 else 0
            lines.append(f"{name[:45]:<45} | {dur_us:>10.0f} | {count:>4} | {avg_us:>8.1f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_json(breakdown: StepBreakdown) -> str:
    """Format step breakdown as JSON."""
    data = {
        "step_duration_us": breakdown.step_duration_us,
        "step_duration_ms": breakdown.step_duration_us / 1000,
        "step_count": breakdown.step_count,
        "phases": {
            "data_preprocessing": {
                "time_us": breakdown.data_preprocessing_time_us,
                "time_ms": breakdown.data_preprocessing_time_us / 1000,
                "percentage": breakdown.data_preprocessing_pct,
            },
            "data_wait": {
                "time_us": breakdown.data_wait_time_us,
                "time_ms": breakdown.data_wait_time_us / 1000,
                "percentage": breakdown.data_wait_pct,
            },
            "data_load": {  # Keep for backward compatibility
                "time_us": breakdown.data_load_time_us,
                "time_ms": breakdown.data_load_time_us / 1000,
                "percentage": breakdown.data_load_pct,
            },
            "forward": {
                "time_us": breakdown.forward_time_us,
                "time_ms": breakdown.forward_time_us / 1000,
                "percentage": breakdown.forward_pct,
            },
            "backward": {
                "time_us": breakdown.backward_time_us,
                "time_ms": breakdown.backward_time_us / 1000,
                "percentage": breakdown.backward_pct,
            },
            "optimizer": {
                "time_us": breakdown.optimizer_time_us,
                "time_ms": breakdown.optimizer_time_us / 1000,
                "percentage": breakdown.optimizer_pct,
            },
            "overhead": {
                "time_us": breakdown.overhead_time_us,
                "time_ms": breakdown.overhead_time_us / 1000,
                "percentage": breakdown.overhead_pct,
            },
        },
        "top_forward_ops": [
            {"name": n, "duration_us": d, "count": c} for n, d, c in breakdown.forward_ops
        ],
        "top_backward_ops": [
            {"name": n, "duration_us": d, "count": c}
            for n, d, c in breakdown.backward_ops
        ],
        "top_optimizer_ops": [
            {"name": n, "duration_us": d, "count": c}
            for n, d, c in breakdown.optimizer_ops
        ],
        "top_data_ops": [
            {"name": n, "duration_us": d, "count": c} for n, d, c in breakdown.data_ops
        ],
    }
    return json.dumps(data, indent=2)


def format_aggregated_text(
    all_breakdowns: list[tuple[Path, StepBreakdown]], experiment_name: str, run_id: str
) -> str:
    """Format aggregated step breakdowns from multiple traces as readable text."""
    lines = [
        "=" * 70,
        f"TRAINING STEP BREAKDOWN ({len(all_breakdowns)} traces)",
        "=" * 70,
        "",
        f"Experiment: {experiment_name}",
        f"Run: {run_id}",
        f"Traces found: {len(all_breakdowns)}",
        "",
    ]

    # Calculate aggregate statistics
    durations = [b.step_duration_us for _, b in all_breakdowns]
    forward_pcts = [b.forward_pct for _, b in all_breakdowns]
    backward_pcts = [b.backward_pct for _, b in all_breakdowns]
    data_load_pcts = [b.data_load_pct for _, b in all_breakdowns]
    data_wait_pcts = [b.data_wait_pct for _, b in all_breakdowns]
    data_preprocessing_pcts = [b.data_preprocessing_pct for _, b in all_breakdowns]

    lines.extend([
        "AGGREGATE PHASE STATISTICS (mean ± std)",
        "-" * 70,
        f"Step duration:     {np.mean(durations) / 1000:>8.2f} ± {np.std(durations) / 1000:>5.2f} ms",
        f"Forward:           {np.mean(forward_pcts):>5.1f} ± {np.std(forward_pcts):>4.1f} %",
        f"Backward:          {np.mean(backward_pcts):>5.1f} ± {np.std(backward_pcts):>4.1f} %",
        f"Data Preprocess:   {np.mean(data_preprocessing_pcts):>5.1f} ± {np.std(data_preprocessing_pcts):>4.1f} %",
        f"Data Wait:         {np.mean(data_wait_pcts):>5.1f} ± {np.std(data_wait_pcts):>4.1f} %",
        f"Data Load (total): {np.mean(data_load_pcts):>5.1f} ± {np.std(data_load_pcts):>4.1f} %",
        "",
        "PER-TRACE BREAKDOWN",
        "-" * 70,
    ])

    for i, (trace_path, breakdown) in enumerate(all_breakdowns, 1):
        trace_name = trace_path.stem[:35]
        lines.extend([
            f"Trace {i} ({trace_name}):",
            f"  Step: {breakdown.step_duration_us / 1000:.2f} ms | "
            f"F: {breakdown.forward_pct:.1f}% | "
            f"B: {breakdown.backward_pct:.1f}% | "
            f"DP: {breakdown.data_preprocessing_pct:.1f}% | "
            f"DW: {breakdown.data_wait_pct:.1f}%",
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


def format_aggregated_json(
    all_breakdowns: list[tuple[Path, StepBreakdown]], experiment_name: str, run_id: str
) -> str:
    """Format aggregated step breakdowns as JSON."""
    aggregated = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "trace_count": len(all_breakdowns),
        "traces": [],
    }

    # Aggregate statistics
    durations = [b.step_duration_us for _, b in all_breakdowns]
    forward_pcts = [b.forward_pct for _, b in all_breakdowns]
    backward_pcts = [b.backward_pct for _, b in all_breakdowns]
    data_load_pcts = [b.data_load_pct for _, b in all_breakdowns]
    data_wait_pcts = [b.data_wait_pct for _, b in all_breakdowns]
    data_preprocessing_pcts = [b.data_preprocessing_pct for _, b in all_breakdowns]

    aggregated["aggregate_stats"] = {
        "step_duration_ms": {"mean": float(np.mean(durations) / 1000), "std": float(np.std(durations) / 1000)},
        "forward_pct": {"mean": float(np.mean(forward_pcts)), "std": float(np.std(forward_pcts))},
        "backward_pct": {"mean": float(np.mean(backward_pcts)), "std": float(np.std(backward_pcts))},
        "data_preprocessing_pct": {"mean": float(np.mean(data_preprocessing_pcts)), "std": float(np.std(data_preprocessing_pcts))},
        "data_wait_pct": {"mean": float(np.mean(data_wait_pcts)), "std": float(np.std(data_wait_pcts))},
        "data_load_pct": {"mean": float(np.mean(data_load_pcts)), "std": float(np.std(data_load_pcts))},
    }

    for trace_path, breakdown in all_breakdowns:
        trace_data = json.loads(format_json(breakdown))
        trace_data["trace_name"] = trace_path.stem
        aggregated["traces"].append(trace_data)

    return json.dumps(aggregated, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training step breakdown from PyTorch profiler trace",
        epilog="""
Examples:
  # Analyze all traces from a run
  step-breakdown mn5_epiforecaster_full 34198361

  # Analyze a single trace file
  step-breakdown outputs/training/mn5_epiforecaster_full/34198361/trace.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Either: experiment_name run_id OR path to single trace JSON file",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )

    args = parser.parse_args()

    # Determine input mode
    if len(args.inputs) == 2:
        # Experiment + run ID mode
        experiment_name, run_id = args.inputs
        try:
            trace_paths = resolve_trace_paths(experiment_name=experiment_name, run_id=run_id)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif len(args.inputs) == 1:
        # Single file mode (backward compatible)
        trace_path = Path(args.inputs[0])
        if not trace_path.exists():
            print(f"Error: Trace file not found: {trace_path}", file=sys.stderr)
            sys.exit(1)
        trace_paths = [trace_path]
        experiment_name = run_id = None
    else:
        parser.print_help()
        sys.exit(1)

    # Extract breakdown from all traces
    all_breakdowns: list[tuple[Path, StepBreakdown]] = []
    for trace_path in trace_paths:
        try:
            breakdown = extract_step_breakdown(str(trace_path))
            all_breakdowns.append((trace_path, breakdown))
        except Exception as e:
            print(f"Warning: Failed to analyze {trace_path}: {e}", file=sys.stderr)

    if not all_breakdowns:
        print("Error: No valid traces found", file=sys.stderr)
        sys.exit(1)

    # Output
    if len(all_breakdowns) == 1 and experiment_name is None:
        # Single trace, legacy format
        _, breakdown = all_breakdowns[0]
        if args.json:
            print(format_json(breakdown))
        else:
            print(format_text(breakdown))
    else:
        # Multiple traces or run mode, aggregated format
        if args.json:
            print(format_aggregated_json(all_breakdowns, experiment_name or "unknown", run_id or "unknown"))
        else:
            print(format_aggregated_text(all_breakdowns, experiment_name or "unknown", run_id or "unknown"))


if __name__ == "__main__":
    main()
