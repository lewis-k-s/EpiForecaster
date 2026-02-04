#!/usr/bin/env python3
"""
Master runner for all data visualization scripts.

Usage:
    uv run dataviz [TRAIN_CONFIG] [PREPROCESS_CONFIG] [OUTPUT_DIR]

Examples:
    uv run dataviz
    uv run dataviz configs/train_epifor_full.yaml configs/preprocess_full.yaml outputs/reports
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def get_dataset_path(preprocess_config: Path, train_config: Path) -> str | None:
    """Extract dataset path from config files."""
    try:
        if preprocess_config.exists():
            with open(preprocess_config) as f:
                config = yaml.safe_load(f)
                if "output_path" in config:
                    return config["output_path"]
    except Exception:
        pass

    try:
        if train_config.exists():
            with open(train_config) as f:
                config = yaml.safe_load(f)
                dataset_path = config.get("dataset", {}).get("path")
                if dataset_path:
                    return dataset_path
                # Also check data.dataset_path
                dataset_path = config.get("data", {}).get("dataset_path")
                if dataset_path:
                    return dataset_path
    except Exception:
        pass

    return None


def check_dataset_variables(dataset_path: str | None) -> dict[str, bool]:
    """Check which variables are available in the dataset."""
    available = {
        "hospitalizations": False,
        "deaths": False,
        "biomarkers": False,
    }

    if not dataset_path:
        return available

    try:
        import xarray as xr

        ds = xr.open_zarr(dataset_path)
        data_vars = list(ds.data_vars)

        available["hospitalizations"] = "hospitalizations" in data_vars
        available["deaths"] = "deaths" in data_vars
        available["biomarkers"] = any(v.startswith("edar_biomarker") for v in data_vars)

        ds.close()
    except Exception:
        pass

    return available


def run_script(script_path: Path, args: list[str]) -> bool:
    """Run a dataviz script with given arguments."""
    # Convert script path to module name for use with -m flag
    # e.g., "dataviz/raw_hospitalizations.py" -> "dataviz.raw_hospitalizations"
    module_name = str(script_path.with_suffix("")).replace("/", ".").replace("\\", ".")
    cmd = [sys.executable, "-m", module_name] + args
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Failed: {script_path} (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False


def main() -> int:
    """Main entry point for dataviz runner."""
    parser = argparse.ArgumentParser(
        description="Run all data visualization scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run dataviz                                    # Use defaults
  uv run dataviz configs/train_epifor_full.yaml     # Custom train config
  uv run dataviz configs/train.yaml configs/prep.yaml outputs/viz  # Full custom
""",
    )
    parser.add_argument(
        "train_config",
        type=Path,
        nargs="?",
        default=Path("configs/train_epifor_full.yaml"),
        help="Training config YAML (default: configs/train_epifor_full.yaml)",
    )
    parser.add_argument(
        "preprocess_config",
        type=Path,
        nargs="?",
        default=Path("configs/preprocess_full.yaml"),
        help="Preprocessing config YAML (default: configs/preprocess_full.yaml)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/reports"),
        help="Output directory (default: outputs/reports)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only specific visualization type (raw_hospitalizations, raw_deaths, canonical_hospitalizations, canonical_deaths, canonical_biomarkers, or analysis)",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw data visualizations",
    )
    parser.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Skip canonical dataset visualizations",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis visualizations",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EpiForecaster Data Visualization")
    print("=" * 80)
    print(f"Train config: {args.train_config}")
    print(f"Preprocess config: {args.preprocess_config}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Get dataset path for biomarkers
    dataset_path = get_dataset_path(args.preprocess_config, args.train_config)

    # Check which variables are available in the dataset
    available_vars = check_dataset_variables(dataset_path)
    if dataset_path:
        print(f"Dataset: {dataset_path}")
        print(f"Available variables: {available_vars}")
        print("-" * 80)

    # Define all analyses
    analyses: list[tuple[str, Path, list[str]]] = []

    # Raw data visualizations
    if not args.skip_raw and args.only in (None, "raw_hospitalizations"):
        analyses.append(
            (
                "raw_hospitalizations",
                Path("dataviz/raw_hospitalizations.py"),
                [
                    "--hospitalization-csv",
                    "data/files/COVID-19__Persones_hospitalitzades.csv",
                    "--output-dir",
                    str(args.output_dir / "raw_hospitalizations"),
                ],
            )
        )

    if not args.skip_raw and args.only in (None, "raw_deaths"):
        analyses.append(
            (
                "raw_deaths",
                Path("dataviz/raw_deaths.py"),
                [
                    "--deaths-csv",
                    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv",
                    "--output-dir",
                    str(args.output_dir / "raw_deaths"),
                ],
            )
        )

    # Canonical dataset visualizations (only if data is available)
    if (
        not args.skip_canonical
        and args.only in (None, "canonical_hospitalizations")
        and available_vars["hospitalizations"]
    ):
        analyses.append(
            (
                "canonical_hospitalizations",
                Path("dataviz/canonical_hospitalizations.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "canonical_hospitalizations"),
                ],
            )
        )

    if (
        not args.skip_canonical
        and args.only in (None, "canonical_deaths")
        and available_vars["deaths"]
    ):
        analyses.append(
            (
                "canonical_deaths",
                Path("dataviz/canonical_deaths.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "canonical_deaths"),
                ],
            )
        )

    if (
        not args.skip_canonical
        and args.only in (None, "canonical_biomarkers")
        and dataset_path
    ):
        analyses.append(
            (
                "canonical_biomarkers",
                Path("dataviz/canonical_biomarker_series.py"),
                [
                    "--dataset",
                    dataset_path,
                    "--output-dir",
                    str(args.output_dir / "canonical_biomarkers"),
                ],
            )
        )

    # Analysis visualizations (only if no specific 'only' filter or if 'analysis' is requested)
    if not args.skip_analysis and args.only in (None, "analysis"):
        analysis_scripts = [
            (
                "sparsity",
                Path("dataviz/sparsity_analysis.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--geo-path",
                    "data/files/geo/fl_municipios_catalonia.geojson",
                    "--output-dir",
                    str(args.output_dir / "sparsity"),
                    "--window-size",
                    "7",
                ],
            ),
            (
                "interp_norm",
                Path("dataviz/interp_norm_raw_analysis.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "interp_norm_raw_analysis"),
                ],
            ),
            (
                "edge_weight",
                Path("dataviz/edge_weight_analysis.py"),
                [
                    "--output-dir",
                    str(args.output_dir / "edge_weight"),
                ],
            ),
            (
                "khop",
                Path("dataviz/khop_neighbors.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "khop_neighbors"),
                    "--k",
                    "3",
                ],
            ),
            (
                "mobility_lockdown",
                Path("dataviz/mobility_lockdown_analysis.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "mobility_lockdown"),
                ],
            ),
            (
                "mobility_regime",
                Path("dataviz/mobility_regime_analysis.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "mobility_regime"),
                ],
            ),
            (
                "neighborhood_regression",
                Path("dataviz/neighborhood_global_regression.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "neighborhood_regression"),
                    "--target",
                    "cases",
                ],
            ),
            (
                "neighborhood_density",
                Path("dataviz/neighborhood_trace_density.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "neighborhood_density"),
                    "--split",
                    "all",
                ],
            ),
        ]
        analyses.extend(analysis_scripts)

    # Run all analyses
    failed = []
    succeeded = []

    for name, script_path, script_args in analyses:
        if not script_path.exists():
            print(f"\n⚠️  Skipping {name}: script not found at {script_path}")
            continue

        print(f"\n{'─' * 80}")
        print(f"▶️  Running: {name}")
        print(f"{'─' * 80}")

        if run_script(script_path, script_args):
            succeeded.append(name)
            print(f"✅ Completed: {name}")
        else:
            failed.append(name)
            print(f"❌ Failed: {name}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total scripts: {len(succeeded) + len(failed)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed scripts: {', '.join(failed)}")

    print(f"\nOutput directory: {args.output_dir}")
    print("=" * 80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
