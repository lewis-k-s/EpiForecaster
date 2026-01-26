import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils.skill_output import SkillOutputBuilder, print_output


def analyze_trace(trace_path: str) -> dict[str, Any]:
    """Analyze PyTorch profiler trace and return structured data."""
    with open(trace_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}") from e

    events = data.get("traceEvents", [])
    if not events:
        raise ValueError("No traceEvents found in file")

    # Aggregators
    op_duration: defaultdict[str, float] = defaultdict(float)
    op_count: defaultdict[str, int] = defaultdict(int)

    categories = set()

    for event in events:
        if "dur" not in event:
            continue

        name = event.get("name", "unknown")
        cat = event.get("cat", "unknown")
        categories.add(cat)

        duration = event["dur"]
        key = f"[{cat}] {name}"

        op_duration[key] += duration
        op_count[key] += 1

    # Sort by total duration descending
    sorted_ops = sorted(op_duration.items(), key=lambda x: x[1], reverse=True)

    # Separate into categories
    cpu_ops = []
    gpu_kernels = []
    other_ops = []

    for key, duration in sorted_ops:
        if key.startswith("[cpu_op]"):
            cpu_ops.append((key, duration))
        elif key.startswith("[kernel]"):
            gpu_kernels.append((key, duration))
        else:
            other_ops.append((key, duration))

    # Calculate global timeline
    start_times = [e["ts"] for e in events if "ts" in e]
    end_times = [e["ts"] + e["dur"] for e in events if "ts" in e and "dur" in e]

    result: dict[str, Any] = {
        "categories": list(categories),
        "total_events": len(events),
    }

    if start_times and end_times:
        global_start = min(start_times)
        global_end = max(end_times)
        total_trace_duration_us = global_end - global_start

        total_cpu_time = sum(dur for _, dur in cpu_ops)
        total_gpu_time = sum(dur for _, dur in gpu_kernels)

        result["trace_duration_us"] = total_trace_duration_us
        result["trace_duration_ms"] = total_trace_duration_us / 1000.0
        result["total_cpu_time_us"] = total_cpu_time
        result["total_cpu_time_ms"] = total_cpu_time / 1000.0
        result["total_gpu_time_us"] = total_gpu_time
        result["total_gpu_time_ms"] = total_gpu_time / 1000.0

    # Format operations
    def format_ops(items):
        formatted = []
        for key, total_dur_us in items:
            count = op_count[key]
            avg_dur_us = total_dur_us / count
            total_dur_ms = total_dur_us / 1000.0
            clean_name = key.split("] ", 1)[1] if "] " in key else key

            formatted.append(
                {
                    "name": clean_name,
                    "original_key": key,
                    "count": count,
                    "avg_duration_us": avg_dur_us,
                    "total_duration_us": total_dur_us,
                    "total_duration_ms": total_dur_ms,
                }
            )
        return formatted

    result["gpu_kernels"] = format_ops(gpu_kernels)
    result["cpu_ops"] = format_ops(cpu_ops)
    result["other_ops"] = format_ops(other_ops)

    return result


def main():
    """CLI entry point for perf-analyze."""
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch profiler Chrome trace JSON files"
    )
    parser.add_argument("trace_path", help="Path to trace JSON file")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (no indentation)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top operations to show in text mode",
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="perf-analyze",
        input_path=args.trace_path,
    )

    try:
        data = analyze_trace(args.trace_path)
        output = builder.success(data)

        if args.text:
            # Print human-readable output
            print(f"Loading trace: {args.trace_path}")
            print(f"Total events: {data['total_events']}")
            print(f"Event categories found: {data['categories']}")
            print("-" * 60)

            if "trace_duration_ms" in data:
                print(f"\nTotal Trace Duration: {data['trace_duration_ms']:.2f} ms")
                print(f"Total CPU Op Time:    {data['total_cpu_time_ms']:.2f} ms")
                print(f"Total GPU Kernel Time:{data['total_gpu_time_ms']:.2f} ms")

            for title, ops in [
                ("Top GPU Kernels", data["gpu_kernels"]),
                ("Top CPU Operations", data["cpu_ops"]),
                ("Other (Runtime/Annotations)", data["other_ops"]),
            ]:
                print(f"\n--- {title} ---")
                print(
                    f"{'Operation':<60} | {'Count':<8} | {'Total (ms)':<10} | {'Avg (us)':<10}"
                )
                print("-" * 95)
                for op in ops[: args.top]:
                    print(
                        f"{op['name'][:60]:<60} | {op['count']:<8} | "
                        f"{op['total_duration_ms']:<10.2f} | {op['avg_duration_us']:<10.2f}"
                    )
        else:
            indent = 0 if args.compact else 2
            print_output(output, indent=indent)

    except FileNotFoundError:
        print_output(
            builder.error(
                "FileNotFoundError", f"Trace file not found: {args.trace_path}"
            )
        )
    except ValueError as e:
        print_output(builder.error("ValueError", str(e)))
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    main()
