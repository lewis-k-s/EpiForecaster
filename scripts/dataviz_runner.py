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
        if train_config.exists():
            with open(train_config) as f:
                config = yaml.safe_load(f) or {}
                dataset_path = config.get("dataset", {}).get("path")
                if dataset_path:
                    return str(dataset_path)
                dataset_path = config.get("data", {}).get("dataset_path")
                if dataset_path:
                    return str(dataset_path)
    except Exception:
        pass

    try:
        if preprocess_config.exists():
            with open(preprocess_config) as f:
                config = yaml.safe_load(f) or {}
                if "output_path" in config:
                    output_path = Path(str(config["output_path"]))
                    dataset_name = config.get("dataset_name")
                    if dataset_name:
                        candidate = output_path / f"{dataset_name}.zarr"
                        if candidate.exists():
                            return str(candidate)
                        return str(candidate)
                    return str(output_path)
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


def get_preprocess_paths(preprocess_config: Path) -> tuple[str, str, str]:
    """Extract raw-data input paths from preprocess config with sensible defaults."""
    defaults = (
        "data/files/hospitalizations_municipality.csv",
        "data/files/deaths_municipality.csv",
        "data/files/mobility.zarr",
    )
    if not preprocess_config.exists():
        return defaults

    try:
        with open(preprocess_config) as f:
            config = yaml.safe_load(f) or {}
        return (
            str(config.get("hospitalizations_file", defaults[0])),
            str(config.get("deaths_file", defaults[1])),
            str(config.get("mobility_path", defaults[2])),
        )
    except Exception:
        return defaults


def has_edge_weight_source_data(
    data_path: Path = Path("data/files/mobility.zarr"),
) -> bool:
    """Return True if mobility.zarr exists for edge-weight analysis."""
    return data_path.exists()


def dataset_has_run_id_dimension(dataset_path: str | None) -> bool:
    """Check whether dataset contains a run_id dimension."""
    if not dataset_path:
        return False
    try:
        import xarray as xr

        ds = xr.open_zarr(dataset_path)
        has_run_id = "run_id" in ds.dims or "run_id" in ds.coords
        ds.close()
        return has_run_id
    except Exception:
        return False


def get_train_run_id(train_config: Path) -> str:
    """Get run_id from train config, defaulting to 'real'."""
    if not train_config.exists():
        return "real"
    try:
        with open(train_config) as f:
            config = yaml.safe_load(f) or {}
        return str(config.get("data", {}).get("run_id", "real"))
    except Exception:
        return "real"


def supports_model_missing_permit(train_config: Path) -> bool:
    """Return True if train config defines missing_permit (data or model section)."""
    if not train_config.exists():
        return False
    try:
        with open(train_config) as f:
            config = yaml.safe_load(f) or {}
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        return "missing_permit" in model_cfg or "missing_permit" in data_cfg
    except Exception:
        return False


def run_script(script_path: Path, args: list[str]) -> bool:
    """Run a dataviz script with given arguments."""
    if str(script_path).startswith("scripts/"):
        cmd = [sys.executable, str(script_path)] + args
    else:
        module_name = (
            str(script_path.with_suffix("")).replace("/", ".").replace("\\", ".")
        )
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

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EpiForecaster Data Visualization")
    print("=" * 80)
    print(f"Train config: {args.train_config}")
    print(f"Preprocess config: {args.preprocess_config}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    dataset_path = get_dataset_path(args.preprocess_config, args.train_config)
    hosp_csv, deaths_csv, mobility_path = get_preprocess_paths(args.preprocess_config)
    run_id = get_train_run_id(args.train_config)
    biomarker_compatible = not dataset_has_run_id_dimension(dataset_path)
    regression_compatible = supports_model_missing_permit(args.train_config)

    available_vars = check_dataset_variables(dataset_path)
    if dataset_path:
        print(f"Dataset: {dataset_path}")
        print(f"Available variables: {available_vars}")
        print("-" * 80)

    analyses: list[tuple[str, Path, list[str]]] = [
        (
            "raw_hospitalizations",
            Path("dataviz/raw_hospitalizations.py"),
            [
                "--hospitalization-csv",
                hosp_csv,
                "--output-dir",
                str(args.output_dir / "raw_hospitalizations"),
            ],
        ),
        (
            "raw_deaths",
            Path("dataviz/raw_deaths.py"),
            [
                "--deaths-csv",
                deaths_csv,
                "--output-dir",
                str(args.output_dir / "raw_deaths"),
            ],
        ),
        (
            "input_series_plots",
            Path("dataviz/input_series_plots.py"),
            [
                "--config",
                str(args.train_config),
                "--output-dir",
                str(args.output_dir / "input_series"),
                "--num-samples",
                "5",
            ],
        ),
        (
            "age_mask_plots",
            Path("dataviz/age_mask_plots.py"),
            [
                "--config",
                str(args.train_config),
                "--output-dir",
                str(args.output_dir / "age_mask"),
            ],
        ),
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
            "mobility_regime",
            Path("dataviz/mobility_regime_analysis.py"),
            [
                "--mobility-path",
                mobility_path,
                "--output-dir",
                str(args.output_dir / "mobility_regime"),
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

    if available_vars["hospitalizations"]:
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

    if available_vars["deaths"]:
        analyses.append(
            (
                "canonical_deaths",
                Path("dataviz/canonical_deaths.py"),
                [
                    "--dataset",
                    str(dataset_path),
                    "--run-id",
                    run_id,
                    "--output-dir",
                    str(args.output_dir / "canonical_deaths"),
                ],
            )
        )

    if available_vars["biomarkers"] and dataset_path and biomarker_compatible:
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

    if regression_compatible:
        regression_results_path = (
            args.output_dir
            / "neighborhood_regression"
            / "neighborhood_global_regression_results.csv"
        )
        analyses.append(
            (
                "neighborhood_regression",
                Path("dataviz/neighborhood_global_regression.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--output-dir",
                    str(args.output_dir / "neighborhood_regression"),
                ],
            )
        )
        analyses.append(
            (
                "mobility_lockdown",
                Path("dataviz/mobility_lockdown_analysis.py"),
                [
                    "--config",
                    str(args.train_config),
                    "--regression-results",
                    str(regression_results_path),
                    "--output-dir",
                    str(args.output_dir / "mobility_lockdown"),
                ],
            )
        )

    if has_edge_weight_source_data():
        analyses.append(("edge_weight", Path("dataviz/edge_weight_analysis.py"), []))

    failed = []
    succeeded = []

    for name, script_path, script_args in analyses:
        print(f"\n{'─' * 80}")
        print(f"▶️  Running: {name}")
        print(f"{'─' * 80}")
        try:
            if run_script(script_path, script_args):
                succeeded.append(name)
                print(f"✅ Completed: {name}")
            else:
                failed.append(name)
                print(f"❌ Failed: {name}")
        except Exception as e:
            failed.append(name)
            print(f"❌ Exception in {name}: {e}")

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
