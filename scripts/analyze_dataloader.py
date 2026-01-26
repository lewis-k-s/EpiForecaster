"""Benchmark dataloader performance using training config YAML.

This script actively benchmarks the dataloader by loading the training config,
creating datasets and dataloaders as specified, and measuring throughput/latency
metrics under various conditions.

Usage:
    # Benchmark using a training config file
    dataloader-analyze configs/train_epifor_full.yaml

    # Quick test (fewer batches)
    dataloader-analyze configs/train_epifor_full.yaml --num-batches 20

    # JSON output
    dataloader-analyze configs/train_epifor_full.yaml --json > benchmark.json

    # Override device
    dataloader-analyze configs/train_epifor_full.yaml --device cpu
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from models.configs import EpiForecasterConfig
from training.epiforecaster_trainer import EpiForecasterTrainer


@dataclass
class DataloaderBenchmarkResult:
    """Benchmark results for a dataloader configuration."""

    # Configuration from training config
    config_path: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    device: str
    multiprocessing_context: str | None

    # Dataset info
    dataset_path: str
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int

    # Benchmark settings
    num_batches_measured: int
    warmup_batches: int

    # Timing metrics
    total_duration_sec: float

    # Throughput (primary metrics)
    batches_per_sec: float
    samples_per_sec: float

    # Latency percentiles
    avg_batch_latency_ms: float
    p50_batch_latency_ms: float
    p95_batch_latency_ms: float
    p99_batch_latency_ms: float

    # Estimated epoch time
    estimated_epoch_time_sec: float


def load_config_from_yaml(config_path: str) -> EpiForecasterConfig:
    """Load training config from YAML file."""
    return EpiForecasterConfig.load(config_path)


def split_dataset_nodes(
    config: EpiForecasterConfig,
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset into train, val, and test node sets.

    Replicates the logic from EpiForecasterTrainer._split_dataset().
    """
    from data.preprocess.config import REGION_COORD

    train_split = 1 - config.training.val_split - config.training.test_split

    aligned_dataset = EpiDataset.load_canonical_dataset(Path(config.data.dataset_path))
    n = aligned_dataset[REGION_COORD].size
    all_nodes = np.arange(n)

    # Check for valid_targets filter
    valid_mask = None
    if config.data.use_valid_targets and "valid_targets" in aligned_dataset:
        valid_mask = aligned_dataset.valid_targets.values.astype(bool)

        if valid_mask is not None:
            all_nodes = all_nodes[valid_mask]
            n = len(all_nodes)

    rng = np.random.default_rng(42)
    rng.shuffle(all_nodes)
    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * config.training.val_split)
    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]

    return list(train_nodes), list(val_nodes), list(test_nodes)


def create_train_dataloader(
    config: EpiForecasterConfig,
    train_nodes: list[int],
    device: torch.device,
) -> tuple[DataLoader, EpiDataset]:
    """Create training dataloader matching trainer logic.

    Replicates the dataset and dataloader creation from EpiForecasterTrainer.
    """
    # Build train dataset with None so it fits scaler internally
    train_dataset = EpiDataset(
        config=config,
        target_nodes=train_nodes,
        context_nodes=train_nodes,
        biomarker_preprocessor=None,
        mobility_preprocessor=None,
    )

    # Dataloader configuration matching _create_data_loaders()
    all_num_workers_zero = config.training.num_workers == 0

    mp_context = "spawn" if device.type == "cuda" and not all_num_workers_zero else None

    pin_memory = config.training.pin_memory and device.type == "cuda"

    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = config.training.num_workers
    if cfg_workers == -1:
        num_workers = avail_cores
    else:
        num_workers = min(avail_cores, cfg_workers)

    persistent_workers = num_workers > 0
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": config.training.batch_size,
        "shuffle": False,  # No shuffling for temporal data
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": EpiForecasterTrainer._collate_fn,
        "multiprocessing_context": mp_context,
    }
    if persistent_workers:
        train_loader_kwargs["persistent_workers"] = True

    if config.training.prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)

    return train_loader, train_dataset


def run_benchmark(
    config: EpiForecasterConfig,
    num_batches: int = 100,
    warmup_batches: int = 5,
    device_override: str | None = None,
) -> DataloaderBenchmarkResult:
    """Run dataloader benchmark using the training config.

    Args:
        config: EpiForecasterConfig loaded from YAML
        num_batches: Number of batches to measure for timing
        warmup_batches: Number of warmup batches (not timed)
        device_override: Override device from config

    Returns:
        DataloaderBenchmarkResult with metrics
    """
    # Setup device
    if device_override:
        device = torch.device(device_override)
    elif config.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.training.device)

    # Split dataset and create dataloader
    print("[1/4] Splitting dataset...", file=sys.stderr, flush=True)
    train_nodes, _val_nodes, _test_nodes = split_dataset_nodes(config)

    print("[2/4] Creating dataloader...", file=sys.stderr, flush=True)
    train_loader, train_dataset = create_train_dataloader(config, train_nodes, device)

    # Get configuration values for result
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = config.training.num_workers
    if cfg_workers == -1:
        num_workers = avail_cores
    else:
        num_workers = min(avail_cores, cfg_workers)

    all_num_workers_zero = num_workers == 0
    mp_context = "spawn" if device.type == "cuda" and not all_num_workers_zero else None
    pin_memory = config.training.pin_memory and device.type == "cuda"
    persistent_workers = num_workers > 0

    dataset_path = str(Path(config.data.dataset_path).resolve())

    # Warmup phase
    print(
        f"[3/4] Warming up ({warmup_batches} batches)...", file=sys.stderr, flush=True
    )
    train_iterator = iter(train_loader)
    for _ in range(warmup_batches):
        try:
            batch = next(train_iterator)
        except StopIteration:
            break

    # Timed phase
    print(f"[4/4] Benchmarking ({num_batches} batches)...", file=sys.stderr, flush=True)
    latencies_ms = []
    start_time = time.perf_counter()
    progress_interval = max(1, num_batches // 10)  # Report 10 times total

    for i in range(num_batches):
        iter_start = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            # Dataset exhausted, create new iterator
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # Match trainer's _forward_batch: transfer ALL tensors
        # Iterate all values and transfer tensors to device
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                v.to(device, non_blocking=True)
            elif hasattr(v, "to"):
                v.to(device)
            # Handle list of PyG Data objects (mobility graphs)
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to"):
                for graph in v:
                    graph.to(device)

        # Synchronize to ensure async transfers complete
        if device.type == "cuda":
            torch.cuda.synchronize()

        iter_end = time.perf_counter()
        latencies_ms.append((iter_end - iter_start) * 1000)

        if (i + 1) % progress_interval == 0:
            print(
                f"  Progress: {i + 1}/{num_batches} batches",
                file=sys.stderr,
                flush=True,
            )

    end_time = time.perf_counter()
    print(
        f"  Complete: {num_batches}/{num_batches} batches", file=sys.stderr, flush=True
    )

    # Cleanup
    del train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Compute metrics
    total_duration_sec = end_time - start_time
    batches_per_sec = num_batches / total_duration_sec
    samples_per_sec = batches_per_sec * config.training.batch_size

    latencies_array = np.array(latencies_ms)
    avg_batch_latency_ms = float(np.mean(latencies_array))
    p50_batch_latency_ms = float(np.percentile(latencies_array, 50))
    p95_batch_latency_ms = float(np.percentile(latencies_array, 95))
    p99_batch_latency_ms = float(np.percentile(latencies_array, 99))

    # Estimated epoch time
    estimated_epoch_time_sec = len(train_dataset) / samples_per_sec

    return DataloaderBenchmarkResult(
        config_path=str(config.training.device),
        batch_size=config.training.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        device=device.type,
        multiprocessing_context=mp_context,
        dataset_path=dataset_path,
        num_train_samples=len(train_dataset),
        num_val_samples=0,  # Not creating val dataset for benchmark
        num_test_samples=0,  # Not creating test dataset for benchmark
        num_batches_measured=num_batches,
        warmup_batches=warmup_batches,
        total_duration_sec=total_duration_sec,
        batches_per_sec=batches_per_sec,
        samples_per_sec=samples_per_sec,
        avg_batch_latency_ms=avg_batch_latency_ms,
        p50_batch_latency_ms=p50_batch_latency_ms,
        p95_batch_latency_ms=p95_batch_latency_ms,
        p99_batch_latency_ms=p99_batch_latency_ms,
        estimated_epoch_time_sec=estimated_epoch_time_sec,
    )


def generate_recommendations(result: DataloaderBenchmarkResult) -> list[str]:
    """Generate actionable recommendations based on benchmark results."""
    recommendations: list[str] = []

    # Throughput assessment
    if result.samples_per_sec > 2000:
        recommendations.append(
            f"✓ Excellent throughput: {result.samples_per_sec:.1f} samples/sec"
        )
    elif result.samples_per_sec > 500:
        recommendations.append(
            f"✓ Good throughput: {result.samples_per_sec:.1f} samples/sec"
        )
    else:
        recommendations.append(
            f"⚠ Low throughput: {result.samples_per_sec:.1f} samples/sec - consider optimization"
        )

    # Worker count assessment
    if result.num_workers == 0:
        recommendations.append(
            "• num_workers=0: Consider increasing for better throughput (try 2-4)"
        )
    elif result.num_workers <= 2:
        recommendations.append(
            f"• num_workers={result.num_workers}: Adequate for small datasets"
        )
    else:
        recommendations.append(
            f"• num_workers={result.num_workers}: Good utilization for this dataset"
        )

    # Pin memory
    if result.device == "cuda":
        if result.pin_memory:
            recommendations.append(
                "• Pin memory is enabled for CUDA - optimal for GPU training"
            )
        else:
            recommendations.append(
                "• Pin memory is disabled - consider enabling for GPU training"
            )
    else:
        if result.pin_memory:
            recommendations.append(
                "• Pin memory is enabled but has no effect on CPU - can disable"
            )

    # Persistent workers
    if result.persistent_workers:
        recommendations.append(
            "• Persistent workers enabled - reduces first-batch latency in multi-epoch training"
        )

    # Multiprocessing context
    if result.multiprocessing_context == "spawn":
        recommendations.append(
            "• Using 'spawn' multiprocessing context (required for CUDA)"
        )

    return recommendations


def format_text(result: DataloaderBenchmarkResult, config_path: str) -> str:
    """Format benchmark results as human-readable text."""
    lines = [
        "=" * 70,
        "DATALOADER BENCHMARK RESULTS",
        "=" * 70,
        "",
        f"Config: {config_path}",
        f"Device: {result.device}",
        f"Dataset: {result.dataset_path}",
        "",
        "CONFIGURATION",
        "-" * 70,
        f"batch_size:              {result.batch_size}",
        f"num_workers:             {result.num_workers}",
        f"pin_memory:              {result.pin_memory}",
        f"persistent_workers:      {result.persistent_workers}",
        f"multiprocessing_context: {result.multiprocessing_context or 'None'}",
        "",
        "DATASET INFO",
        "-" * 70,
        f"Train samples:           {result.num_train_samples}",
        "",
        "BENCHMARK RESULTS",
        "-" * 70,
        f"Measured batches:        {result.num_batches_measured}",
        f"Warmup batches:          {result.warmup_batches}",
        f"Total duration:          {result.total_duration_sec:.2f} sec",
        "",
        "THROUGHPUT",
        "-" * 70,
        f"Samples/sec:             {result.samples_per_sec:.1f}",
        f"Batches/sec:             {result.batches_per_sec:.1f}",
        f"Estimated epoch time:    {result.estimated_epoch_time_sec:.1f} sec",
        "",
        "LATENCY (ms)",
        "-" * 70,
        f"Average:                 {result.avg_batch_latency_ms:.1f}",
        f"P50 (median):            {result.p50_batch_latency_ms:.1f}",
        f"P95:                     {result.p95_batch_latency_ms:.1f}",
        f"P99:                     {result.p99_batch_latency_ms:.1f}",
        "",
        "RECOMMENDATIONS",
        "-" * 70,
    ]

    for rec in generate_recommendations(result):
        lines.append(rec)

    lines.extend(
        [
            "",
            "CONSIDERATIONS",
            "-" * 70,
            "• If GPU utilization is low during training, dataloader may not be bottleneck",
            "• Run perf-analyze on training traces to verify:",
            "  → python -m cli train ... --profile-steps 10",
            "  → perf-analyze <experiment> <run_id>",
            "",
            "• To test different configurations, create new config files and re-run:",
            "  → dataloader-analyze configs/train_epifor_small.yaml",
            "  → dataloader-analyze configs/train_epifor_large.yaml",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def format_json(result: DataloaderBenchmarkResult, config_path: str) -> str:
    """Format benchmark results as JSON."""
    data = {
        "config_path": config_path,
        "device": result.device,
        "dataset_path": result.dataset_path,
        "configuration": {
            "batch_size": result.batch_size,
            "num_workers": result.num_workers,
            "pin_memory": result.pin_memory,
            "persistent_workers": result.persistent_workers,
            "multiprocessing_context": result.multiprocessing_context,
        },
        "dataset_info": {
            "num_train_samples": result.num_train_samples,
            "num_val_samples": result.num_val_samples,
            "num_test_samples": result.num_test_samples,
        },
        "benchmark_config": {
            "num_batches": result.num_batches_measured,
            "warmup_batches": result.warmup_batches,
        },
        "results": {
            "total_duration_sec": result.total_duration_sec,
            "throughput": {
                "samples_per_sec": result.samples_per_sec,
                "batches_per_sec": result.batches_per_sec,
                "estimated_epoch_time_sec": result.estimated_epoch_time_sec,
            },
            "latency_ms": {
                "avg": result.avg_batch_latency_ms,
                "p50": result.p50_batch_latency_ms,
                "p95": result.p95_batch_latency_ms,
                "p99": result.p99_batch_latency_ms,
            },
        },
        "recommendations": generate_recommendations(result),
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dataloader performance using training config",
        epilog="""
Examples:
  # Benchmark using a training config file
  dataloader-analyze configs/train_epifor_full.yaml

  # Quick test (fewer batches)
  dataloader-analyze configs/train_epifor_full.yaml --num-batches 20

  # JSON output
  dataloader-analyze configs/train_epifor_full.yaml --json > benchmark.json

  # Override device
  dataloader-analyze configs/train_epifor_full.yaml --device cpu
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to measure (default: 100)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches (default: 5)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Override device from config",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )

    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    try:
        config = load_config_from_yaml(str(config_path))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    try:
        result = run_benchmark(
            config,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
            device_override=args.device,
        )
        # Fix config_path in result to be the actual path
        result.config_path = str(config_path)
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Output
    if args.json:
        print(format_json(result, str(config_path)))
    else:
        print(format_text(result, str(config_path)))


if __name__ == "__main__":
    main()
