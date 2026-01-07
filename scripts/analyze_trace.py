import json
import sys
from collections import defaultdict


def analyze_trace(trace_path):
    print(f"Loading trace: {trace_path}")
    with open(trace_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    events = data.get("traceEvents", [])
    if not events:
        print("No traceEvents found.")
        return

    print(f"Total events: {len(events)}")

    # Aggregators
    op_duration: defaultdict[str, float] = defaultdict(float)
    op_count: defaultdict[str, int] = defaultdict(int)

    # Categories to potentially filter or split by
    # PyTorch traces often have 'cat' field: 'cpu_op', 'cuda_kernel', etc.

    categories = set()

    for event in events:
        # We only care about complete events (X) or duration events usually,
        # but 'dur' is present in 'X' (Complete) type events.
        # Some traces use B (Begin) and E (End). PyTorch typically uses X.

        if "dur" not in event:
            continue

        name = event.get("name", "unknown")
        cat = event.get("cat", "unknown")
        categories.add(cat)

        # Duration is usually in microseconds
        duration = event["dur"]

        # Key can be name + category to distinguish cpu vs gpu
        key = f"[{cat}] {name}"

        op_duration[key] += duration
        op_count[key] += 1

    print(f"Event categories found: {categories}")
    print("-" * 60)
    print(
        f"{'Operation':<50} | {'Count':<8} | {'Total Time (ms)':<15} | {'Avg Time (us)':<15}"
    )
    print("-" * 60)

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

    def print_table(title, items, limit=20):
        print(f"\n--- {title} ---")
        print(
            f"{'Operation':<60} | {'Count':<8} | {'Total (ms)':<10} | {'Avg (us)':<10}"
        )
        print("-" * 95)
        for _i, (key, total_dur_us) in enumerate(items[:limit]):
            count = op_count[key]
            avg_dur_us = total_dur_us / count
            total_dur_ms = total_dur_us / 1000.0
            # Remove the category prefix for cleaner printing if desired, or keep it
            clean_name = key.split("] ", 1)[1] if "] " in key else key
            print(
                f"{clean_name[:60]:<60} | {count:<8} | {total_dur_ms:<10.2f} | {avg_dur_us:<10.2f}"
            )

    # Calculate global timeline
    start_times = [e["ts"] for e in events if "ts" in e]
    end_times = [e["ts"] + e["dur"] for e in events if "ts" in e and "dur" in e]

    if start_times and end_times:
        global_start = min(start_times)
        global_end = max(end_times)
        total_trace_duration_ms = (global_end - global_start) / 1000.0
        print(f"\nTotal Trace Duration: {total_trace_duration_ms:.2f} ms")

        total_cpu_time = sum(dur for _, dur in cpu_ops) / 1000.0
        total_gpu_time = sum(dur for _, dur in gpu_kernels) / 1000.0

        print(f"Total CPU Op Time:    {total_cpu_time:.2f} ms")
        print(f"Total GPU Kernel Time:{total_gpu_time:.2f} ms")

    print_table("Top GPU Kernels", gpu_kernels)
    print_table("Top CPU Operations", cpu_ops)
    print_table("Other (Runtime/Annotations)", other_ops)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <path_to_trace_json>")
        sys.exit(1)

    analyze_trace(sys.argv[1])
