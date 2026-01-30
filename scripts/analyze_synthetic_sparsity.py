#!/usr/bin/env python3
"""
Analyze sparsity in synthetic data vs real data.

This script examines the synthetic observations dataset to understand:
1. Sparsity level metadata across runs
2. Actual cases sparsity in the data
3. Comparison with real data statistics
4. Gap analysis for curriculum learning

Usage:
    python scripts/analyze_synthetic_sparsity.py [--path PATH]

Args:
    path: Path to raw_synthetic_observations.zarr (default: data/files/raw_synthetic_observations.zarr)
"""

import argparse

import numpy as np
import xarray as xr


def analyze_synthetic_sparsity(
    zarr_path: str = "data/files/raw_synthetic_observations.zarr",
) -> dict:
    """Analyze sparsity in synthetic data.

    Args:
        zarr_path: Path to raw synthetic observations zarr

    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("SYNTHETIC DATA SPARSITY ANALYSIS")
    print("=" * 70)
    print()

    # Load with zarr_format=2 to avoid v2/v3 conflict
    ds = xr.open_zarr(zarr_path, zarr_format=2)

    results = {
        "path": zarr_path,
        "dimensions": dict(ds.dims),
        "num_runs": ds.dims["run_id"],
        "num_dates": ds.dims["date"],
        "num_regions": ds.dims["region_id"],
    }

    print(f"Dataset: {zarr_path}")
    print(f"Dimensions: {results['dimensions']}")
    print()

    # Metadata sparsity levels
    sparsity_meta = ds["synthetic_sparsity_level"].values
    results["sparsity_meta"] = {
        "min": float(sparsity_meta.min()),
        "max": float(sparsity_meta.max()),
        "mean": float(sparsity_meta.mean()),
        "unique": sorted(set(sparsity_meta.tolist())),
    }

    print("METADATA: synthetic_sparsity_level")
    print(f"  Min: {sparsity_meta.min():.3f}")
    print(f"  Max: {sparsity_meta.max():.3f}")
    print(f"  Mean: {sparsity_meta.mean():.3f}")
    print(f"  Unique values: {sorted(set(sparsity_meta.tolist()))}")
    print()

    # Actual cases sparsity
    cases = ds["cases"].values
    total_values = cases.size
    nan_values = np.count_nonzero(np.isnan(cases))
    actual_sparsity = 100 * nan_values / total_values

    results["cases_actual"] = {
        "total_values": int(total_values),
        "nan_values": int(nan_values),
        "sparsity_pct": float(actual_sparsity),
    }

    print("ACTUAL CASES DATA")
    print(f"  Total values: {total_values:,}")
    print(f"  NaN values: {nan_values:,}")
    print(f"  Overall sparsity: {actual_sparsity:.2f}%")
    print()

    # Per-run sparsity
    run_sparsity = []
    for i in range(len(sparsity_meta)):
        run_cases = cases[i]
        sparsity_pct = 100 * np.count_nonzero(np.isnan(run_cases)) / run_cases.size
        run_sparsity.append(sparsity_pct)

    run_sparsity = np.array(run_sparsity)
    results["per_run_sparsity"] = {
        "mean": float(run_sparsity.mean()),
        "std": float(run_sparsity.std()),
        "min": float(run_sparsity.min()),
        "max": float(run_sparsity.max()),
    }

    print("PER-RUN SPARSITY DISTRIBUTION")
    print(f"  Mean: {run_sparsity.mean():.2f}%")
    print(f"  Std dev: {run_sparsity.std():.2f}%")
    print(f"  Range: [{run_sparsity.min():.2f}%, {run_sparsity.max():.2f}%]")
    print()

    # Sparsity distribution bins
    bins = [0, 5, 10, 20, 30, 50, 100]
    print("SPARSITY DISTRIBUTION ACROSS RUNS:")
    for i in range(len(bins) - 1):
        count = np.sum((run_sparsity >= bins[i]) & (run_sparsity < bins[i + 1]))
        pct = 100 * count / len(run_sparsity)
        print(f"  [{bins[i]:3d}%, {bins[i+1]:3d}%): {count:3d} runs ({pct:5.1f}%)")
    print()

    # Per-region sparsity
    region_sparsity = []
    for r in range(cases.shape[2]):
        region_data = cases[:, :, r]
        sparsity_pct = 100 * np.count_nonzero(np.isnan(region_data)) / region_data.size
        region_sparsity.append(sparsity_pct)

    region_sparsity = np.array(region_sparsity)
    results["per_region_sparsity"] = {
        "mean": float(region_sparsity.mean()),
        "median": float(np.median(region_sparsity)),
        "std": float(region_sparsity.std()),
        "min": float(region_sparsity.min()),
        "max": float(region_sparsity.max()),
    }

    print("PER-REGION SPARSITY STATISTICS")
    print(f"  Mean: {region_sparsity.mean():.2f}%")
    print(f"  Median: {np.median(region_sparsity):.2f}%")
    print(f"  Std: {region_sparsity.std():.2f}%")
    print(f"  Range: [{region_sparsity.min():.2f}%, {region_sparsity.max():.2f}%]")
    print()

    # High sparsity regions
    high_sparsity_threshold = 10.0
    high_sparsity_count = np.sum(region_sparsity >= high_sparsity_threshold)
    high_sparsity_pct = 100 * high_sparsity_count / len(region_sparsity)

    results["high_sparsity_regions"] = {
        "threshold_pct": high_sparsity_threshold,
        "count": int(high_sparsity_count),
        "pct_of_regions": float(high_sparsity_pct),
    }

    print(f"Regions with >{high_sparsity_threshold}% sparsity:")
    print(f"  {high_sparsity_count} / {len(region_sparsity)} ({high_sparsity_pct:.1f}%)")
    print()

    # Scenario type breakdown
    scenarios = ds["synthetic_scenario_type"].values
    strengths = ds["synthetic_strength"].values

    print("SCENARIO TYPE BREAKDOWN")
    for scenario in np.unique(scenarios):
        mask = scenarios == scenario
        scenario_sparsity = sparsity_meta[mask]
        scenario_strengths = strengths[mask]
        print(f"  {scenario}: {np.sum(mask)} runs")
        print(f"    Sparsity range: [{scenario_sparsity.min():.3f}, {scenario_sparsity.max():.3f}]")
        print(f"    Strength range: [{scenario_strengths.min():.2f}, {scenario_strengths.max():.2f}]")
    print()

    ds.close()

    # Real data comparison (from earlier analysis)
    print("=" * 70)
    print("COMPARISON WITH REAL DATA")
    print("=" * 70)
    print()

    real_sparsity_mean = 72.0
    real_high_sparsity_pct = 96.4
    real_fresh_locf_pct = 37.6

    results["real_data_baseline"] = {
        "sparsity_mean_pct": real_sparsity_mean,
        "high_sparsity_regions_pct": real_high_sparsity_pct,
        "fresh_locf_pct": real_fresh_locf_pct,
    }

    sparsity_gap = real_sparsity_mean - actual_sparsity
    sparsity_ratio = real_sparsity_mean / actual_sparsity if actual_sparsity > 0 else float("inf")

    print(f"{'Metric':<35} {'Synthetic':>15} {'Real':>15} {'Gap':>15}")
    print("-" * 80)
    print(
        f"{'Mean sparsity %':<35} {actual_sparsity:>14.2f}% {real_sparsity_mean:>14.2f}% {sparsity_gap:>+14.2f}%"
    )
    print(
        f"{'Sparsity ratio (real/synth)':<35} {sparsity_ratio:>15.1f}x"
    )
    print(
        f"{'Regions with >10% sparsity':<35} {high_sparsity_pct:>14.1f}% {real_high_sparsity_pct:>14.1f}% N/A"
    )
    print(
        f"{'Fresh data (0-1 days LOCF)':<35} {'~99.9%':>15} {real_fresh_locf_pct:>14.1f}% N/A"
    )
    print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS FOR SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print()

    if len(results["sparsity_meta"]["unique"]) <= 1:
        print("❌ ISSUE: All runs have identical sparsity_level")
        print(f"   Current: {results['sparsity_meta']['unique']}")
        print()
        print("✓ RECOMMENDED: Generate runs with varying sparsity levels")
        print("  Example curriculum: [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]")
        print()
        print("  This enables:")
        print("  1. Curriculum learning (start clean → increase noise)")
        print("  2. Matching real data distribution (~70% sparsity)")
        print("  3. Robustness to missing data")
        print()

    if actual_sparsity < 10.0:
        print("❌ ISSUE: Synthetic data is too clean compared to real")
        print(f"   Current: {actual_sparsity:.1f}% vs Real: {real_sparsity_mean:.1f}%")
        print()
        print("✓ RECOMMENDED: Add runs with higher sparsity targets")
        print("  Target sparsity levels to match real data:")
        print("  - 5-10%: Current (nearly complete)")
        print("  - 20-30%: Moderate missingness")
        print("  - 50-60%: High missingness")
        print("  - 70-80%: Real-world levels")
        print()

    if high_sparsity_pct < 1.0:
        print("❌ ISSUE: Almost no regions have high sparsity")
        print(f"   Current: {high_sparsity_pct:.1f}% regions >10% missing")
        print(f"   Real data: {real_high_sparsity_pct:.1f}% regions >10% missing")
        print()
        print("✓ RECOMMENDED: Ensure high-sparsity runs create")
        print("  realistic sparsity patterns across regions")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total runs analyzed: {results['num_runs']}")
    print(f"Sparsity range in metadata: [{sparsity_meta.min():.3f}, {sparsity_meta.max():.3f}]")
    print(f"Actual data sparsity: {actual_sparsity:.2f}%")
    print(f"Gap to real data: {sparsity_gap:.1f}% (need {sparsity_ratio:.1f}x increase)")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze sparsity in synthetic data vs real data"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/files/raw_synthetic_observations.zarr",
        help="Path to raw_synthetic_observations.zarr",
    )
    args = parser.parse_args()

    analyze_synthetic_sparsity(args.path)
