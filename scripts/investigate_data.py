#!/usr/bin/env python3
"""
Data Investigation Script for COVID-19 Forecasting.

This script runs the complete preprocessing pipeline and analyzes the data
for zero-window prevalence, alignment quality, and distribution characteristics
before model training begins.
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Simple argument parsing for data investigation
def parse_arguments():
    """Parse command line arguments for data investigation."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Investigation for COVID-19 Forecasting")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing data files")
    parser.add_argument("--mobility", type=str, default="files/daily_dynpop_mitma/", help="Path to mobility data")
    parser.add_argument("--auxiliary_data_dir", type=str, default="files/", help="Directory containing auxiliary data")
    parser.add_argument("--cases_file", type=str, default="files/flowmaps_cat_municipio_cases.csv", help="Path to cases file")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD)")

    # Processing arguments
    parser.add_argument("--cases_normalization", type=str, default="log1p", choices=["log1p", "standard", "none"])
    parser.add_argument("--min_cases_threshold", type=int, default=0)
    parser.add_argument("--cases_fill_missing", type=str, default="forward_fill", choices=["forward_fill", "interpolate", "zero"])

    # Windowing and preprocessing
    parser.add_argument("--windowing_stride", type=int, default=1)
    parser.add_argument("--min_flow_threshold", type=int, default=10)
    parser.add_argument("--enable_preprocessing_hooks", action="store_true")
    parser.add_argument("--use_edar_data", action="store_true")
    parser.add_argument("--edar_hidden_dim", type=int, default=64)
    parser.add_argument("--edar_biomarker_features", nargs="+", default=None)

    # Output arguments
    parser.add_argument("--investigation_output", type=str, default="outputs/data_investigation")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--no_plots", action="store_true")

    # Dataset alignment
    parser.add_argument("--target_dataset", type=str, default="cases")
    parser.add_argument("--padding_strategy", type=str, default="interpolate", choices=["zero", "interpolate", "forward_fill"])
    parser.add_argument("--crop_datasets", action="store_true")
    parser.add_argument("--alignment_buffer_days", type=int, default=0)
    parser.add_argument("--interpolation_method", type=str, default="linear", choices=["linear", "cubic", "spline", "smart"])
    parser.add_argument("--validate_alignment", action="store_true")

    return parser.parse_args()
from data.cases_loader import create_cases_loader
from data.dataset_alignment import create_alignment_manager
from data.mobility_loader import MobilityDataLoader
from utils.data_analytics import DataAnalytics

logger = logging.getLogger(__name__)


class DataInvestigator:
    """
    Comprehensive data investigation pipeline that runs preprocessing
    and analyzes data quality before model training.
    """

    def __init__(self, args, output_dir: Optional[str] = None):
        """
        Initialize data investigator.

        Args:
            args: Parsed command line arguments
            output_dir: Optional output directory override
        """
        self.args = args
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(args.output_dir) / "data_investigation"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analytics module
        self.analytics = DataAnalytics(str(self.output_dir))

        # Data containers
        self.cases_loader = None
        self.mobility_loader = None
        self.alignment_manager = None
        self.datasets = {}
        self.dataset_dates = {}
        self.dataset_entities = {}
        self.max_mobility_timepoints = getattr(args, "mobility_timepoints", 90)

    def run_investigation(self) -> Dict[str, Any]:
        """
        Run complete data investigation pipeline.

        Returns:
            Dictionary containing all investigation results
        """
        logger.info("🔍 Starting data investigation pipeline")
        logger.info(f"Output directory: {self.output_dir}")

        results = {
            "investigation_timestamp": datetime.now().isoformat(),
            "args": vars(self.args),
            "data_loading": {},
            "alignment": {},
            "zero_window_analysis": {},
            "distribution_analysis": {},
            "visualizations": {},
            "recommendations": [],
        }

        try:
            # Step 1: Load all data
            results["data_loading"] = self.load_all_data()

            # Step 2: Analyze zero-window prevalence
            if "cases" in self.datasets:
                logger.info("📊 Analyzing zero-window prevalence...")
                zero_window_results = self.analytics.analyze_zero_windows(
                    self.datasets["cases"],
                    thresholds=[0.0, 0.01, 0.1, 0.5, 1.0],
                    normalization=self.args.cases_normalization,
                )
                results["zero_window_analysis"] = zero_window_results

            # Step 3: Validate dataset alignment
            if len(self.datasets) > 1:
                logger.info("🔗 Validating dataset alignment...")
                alignment_results = self.analytics.validate_alignment(
                    self.datasets,
                    self.dataset_dates,
                    self.dataset_entities,
                    target_dataset=self.args.target_dataset,
                )
                results["alignment"] = alignment_results

            # Step 4: Analyze data distributions
            logger.info("📈 Analyzing data distributions...")
            distribution_results = self.analytics.analyze_data_distributions(
                self.datasets, dataset_names=list(self.datasets.keys())
            )
            results["distribution_analysis"] = distribution_results

            # Step 5: Create visualizations
            logger.info("📊 Creating data visualizations...")
            visualizations = self.create_visualizations()
            results["visualizations"] = visualizations

            # Step 6: Generate comprehensive report
            logger.info("📝 Generating comprehensive report...")
            report_path = self.analytics.generate_comprehensive_report(
                zero_window_results=results.get("zero_window_analysis", {}),
                alignment_results=results.get("alignment", {}),
                distribution_results=results.get("distribution_analysis", {}),
            )
            results["report_path"] = report_path

            # Step 7: Generate executive summary
            logger.info("📋 Generating executive summary...")
            summary = self.generate_executive_summary(results)
            results["executive_summary"] = summary

            # Collect all recommendations
            all_recommendations = []
            all_recommendations.extend(
                results.get("zero_window_analysis", {}).get("recommendations", [])
            )
            all_recommendations.extend(
                results.get("alignment", {}).get("recommendations", [])
            )
            results["recommendations"] = all_recommendations

            logger.info("✅ Data investigation completed successfully!")
            return results

        except Exception as e:
            logger.error(f"❌ Data investigation failed: {e}")
            raise

    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all datasets using the same pipeline as training.

        Returns:
            Dictionary with data loading results
        """
        logger.info("📂 Loading all datasets...")

        loading_results = {
            "success": True,
            "datasets_loaded": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Load COVID cases data
            logger.info("Loading COVID cases data...")
            cases_file_path = (
                str(Path(self.args.data_dir) / self.args.cases_file)
                if not Path(self.args.cases_file).is_absolute()
                else self.args.cases_file
            )

            fill_missing = getattr(
                self.args,
                "cases_fill_missing",
                getattr(self.args, "fill_missing", "forward_fill"),
            )

            logger.info(
                "Using %s strategy for missing case data", fill_missing
            )

            self.cases_loader = create_cases_loader(
                cases_file=cases_file_path,
                normalization=self.args.cases_normalization,
                min_cases_threshold=self.args.min_cases_threshold,
                fill_missing=fill_missing,
            )

            if self.cases_loader.cases_tensor is None:
                raise RuntimeError("Cases tensor not materialized after loading")

            # Derive temporal coverage
            start, end = (self.cases_loader.date_range or (None, None))
            if start is None or pd.isna(start):
                start = (
                    self.cases_loader.cases_df["evstart"].min()
                    if self.cases_loader.cases_df is not None
                    else None
                )
            if end is None or pd.isna(end):
                if self.cases_loader.cases_df is not None:
                    end_candidates = self.cases_loader.cases_df["evend"].dropna()
                    if end_candidates.empty:
                        end = self.cases_loader.cases_df["evstart"].max()
                    else:
                        end = end_candidates.max()

            if start is None or end is None or pd.isna(start) or pd.isna(end):
                raise ValueError("Unable to determine temporal coverage for cases data")

            # Persist the cleaned date range on the loader for downstream consumers
            self.cases_loader.date_range = (start, end)

            case_dates = pd.date_range(start=start, end=end, freq="D")
            case_dates_list = list(case_dates.to_pydatetime())

            cases_tensor = self.cases_loader.cases_tensor
            self.datasets["cases"] = cases_tensor
            self.dataset_dates["cases"] = case_dates_list
            self.dataset_entities["cases"] = list(
                self.cases_loader.municipality_mapping.keys()
            )

            stats = self.cases_loader.get_statistics()
            loading_results["datasets_loaded"].append("cases")
            loading_results["statistics"]["cases"] = {
                "shape": list(cases_tensor.shape),
                "total_cases": stats.get("total_cases"),
                "municipalities": stats.get("active_municipalities"),
                "timepoints": stats.get("timepoints"),
                "date_range": f"{case_dates_list[0].date()} to {case_dates_list[-1].date()}",
                "normalization": self.args.cases_normalization,
            }

            logger.info(f"✅ COVID cases loaded: {cases_tensor.shape}")

            # Load mobility data
            logger.info("Loading mobility data...")
            mobility_files = self._find_mobility_files()
            if not mobility_files:
                logger.warning("No mobility files found; skipping mobility analysis")
            else:
                self.mobility_loader = MobilityDataLoader(
                    min_flow_threshold=self.args.min_flow_threshold,
                    normalize_flows=True,
                    undirected=False,
                    allow_self_loops=False,
                    edge_selector="nonzero",
                    node_stats=("sum", "mean", "count_nonzero"),
                    engine="h5netcdf",
                    chunks={"time": 1},
                )

                mobility_features: list[torch.Tensor] = []
                mobility_dates: list[Any] = []
                sampled_files: list[str] = []

                for file_path in mobility_files:
                    logger.info(f"Streaming mobility data from: {file_path}")
                    sampled_files.append(str(file_path))
                    try:
                        stream = self.mobility_loader.stream_dataset(
                            netcdf_filepath=str(file_path),
                            population_filepath=None,
                            edge_vars=["person_hours"],
                            time_slice=None,
                        )

                        for step_idx, graph in enumerate(stream):
                            mobility_features.append(graph.x)
                            timestamp = self._extract_timestamp(
                                graph, len(mobility_dates)
                            )
                            mobility_dates.append(timestamp)
                            if len(mobility_features) >= self.max_mobility_timepoints:
                                break

                        if len(mobility_features) >= self.max_mobility_timepoints:
                            break

                    except Exception as stream_error:
                        logger.error(
                            f"Failed to stream mobility data from {file_path}: {stream_error}"
                        )
                        continue

                if mobility_features:
                    mobility_tensor = torch.stack(
                        mobility_features, dim=1
                    )  # [nodes, timepoints, features]

                    self.datasets["mobility"] = mobility_tensor
                    self.dataset_dates["mobility"] = mobility_dates
                    zone_ids = (
                        list(self.mobility_loader.zone_ids)
                        if self.mobility_loader and self.mobility_loader.zone_ids
                        else list(range(mobility_tensor.shape[0]))
                    )
                    if len(zone_ids) != mobility_tensor.shape[0]:
                        logger.warning(
                            "Zone registry size mismatch; using positional indices"
                        )
                        zone_ids = list(range(mobility_tensor.shape[0]))
                    self.dataset_entities["mobility"] = zone_ids

                    loading_results["datasets_loaded"].append("mobility")
                    loading_results["statistics"]["mobility"] = {
                        "shape": list(mobility_tensor.shape),
                        "nodes": mobility_tensor.shape[0],
                        "timepoints": mobility_tensor.shape[1],
                        "features": mobility_tensor.shape[2],
                        "sampled_files": sampled_files,
                    }

                    logger.info(f"✅ Mobility data loaded: {mobility_tensor.shape}")

            # Run dataset alignment if enabled
            if self.args.crop_datasets and len(self.datasets) > 1:
                logger.info("Running dataset alignment...")
                self.alignment_manager = create_alignment_manager(
                    target_dataset_name=self.args.target_dataset,
                    padding_strategy=self.args.padding_strategy,
                    crop_datasets=self.args.crop_datasets,
                    interpolation_method=self.args.interpolation_method,
                )

                aligned_datasets = self.alignment_manager.align_datasets(
                    datasets=self.datasets,
                    dataset_dates=self.dataset_dates,
                    dataset_entities=self.dataset_entities,
                )

                if aligned_datasets:
                    # Update datasets with aligned versions
                    for name, aligned_data in aligned_datasets.items():
                        if name in self.datasets:
                            original_shape = self.datasets[name].shape
                            self.datasets[name] = aligned_data
                            loading_results["statistics"][name]["aligned_shape"] = list(
                                aligned_data.shape
                            )
                            loading_results["statistics"][name]["alignment_change"] = (
                                f"{original_shape} -> {aligned_data.shape}"
                            )

                    logger.info("✅ Dataset alignment completed")

        except Exception as e:
            loading_results["success"] = False
            loading_results["errors"].append(str(e))
            logger.error(f"❌ Error loading data: {e}")

        return loading_results

    def _find_mobility_files(self) -> list[Path]:
        """Locate mobility NetCDF files honoring optional date filters."""
        data_dir = Path(self.args.data_dir)
        mobility_path = Path(self.args.mobility)
        if not mobility_path.is_absolute():
            mobility_path = data_dir / mobility_path

        if not mobility_path.exists():
            logger.warning(f"Mobility path not found: {mobility_path}")
            return []

        if mobility_path.is_file():
            if mobility_path.suffix != ".nc":
                logger.warning(
                    f"Mobility file must be NetCDF (.nc), got: {mobility_path}"
                )
                return []
            return [mobility_path]

        if mobility_path.is_dir():
            netcdf_files = sorted(mobility_path.glob("*.nc"))
            if not netcdf_files:
                logger.warning(
                    f"No NetCDF files found in mobility directory: {mobility_path}"
                )
                return []

            start_date = self._parse_date(getattr(self.args, "start_date", None))
            end_date = self._parse_date(getattr(self.args, "end_date", None))

            if not start_date and not end_date:
                return netcdf_files

            filtered_files = []
            pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")

            for file_path in netcdf_files:
                match = pattern.search(file_path.name)
                if not match:
                    filtered_files.append(file_path)
                    continue

                file_start = datetime.strptime(match.group(1), "%Y-%m-%d")
                file_end = datetime.strptime(match.group(2), "%Y-%m-%d")

                include = True
                if start_date and file_end < start_date:
                    include = False
                if end_date and file_start > end_date:
                    include = False

                if include:
                    filtered_files.append(file_path)

            if not filtered_files:
                logger.warning(
                    "No mobility files matched the provided date range; using all files"
                )
                return netcdf_files

            return filtered_files

        logger.warning(f"Unrecognized mobility path: {mobility_path}")
        return []

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format (expected YYYY-MM-DD): {value}")
            return None

    @staticmethod
    def _extract_timestamp(graph: Any, default_index: int) -> Any:
        """Convert graph timestamp tensor into a pandas Timestamp when possible."""
        try:
            if hasattr(graph, "t") and graph.t is not None and graph.t.numel() > 0:
                timestamp_value = int(graph.t.view(-1)[0].item())
                return pd.to_datetime(timestamp_value, unit="ns", utc=True)
        except Exception:
            pass
        return default_index

    def create_visualizations(self) -> Dict[str, str]:
        """
        Create all data visualization plots.

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        try:
            # Zero-window analysis visualization
            if "cases" in self.datasets:
                zero_window_viz = self.analytics.visualize_zero_windows(
                    self.datasets["cases"], thresholds=[0.0, 0.01, 0.1, 0.5, 1.0]
                )
                visualizations["zero_window_analysis"] = zero_window_viz

            # Data distribution visualization
            if len(self.datasets) > 0:
                distribution_viz = self.analytics.visualize_data_distributions(
                    self.datasets, dataset_names=list(self.datasets.keys())
                )
                visualizations["data_distributions"] = distribution_viz

            # Alignment visualization (if multiple datasets)
            if len(self.datasets) > 1:
                alignment_viz = self.create_alignment_visualization()
                if alignment_viz:
                    visualizations["alignment_analysis"] = alignment_viz

        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            visualizations["error"] = str(e)

        return visualizations

    def create_alignment_visualization(self) -> Optional[str]:
        """
        Create alignment quality visualization.

        Returns:
            Path to saved visualization or None if error
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle("Dataset Alignment Analysis", fontsize=16, fontweight="bold")

            # Temporal coverage comparison
            datasets = list(self.datasets.keys())
            coverage_data = []

            for name in datasets:
                dates = self.dataset_dates.get(name, [])
                if dates:
                    # Convert to datetime if needed
                    if isinstance(dates[0], str):
                        dates = [pd.to_datetime(d) for d in dates]
                    elif not isinstance(dates[0], datetime):
                        dates = [datetime.fromisoformat(str(d)) for d in dates]

                    coverage_data.append(
                        {
                            "dataset": name,
                            "start_date": min(dates),
                            "end_date": max(dates),
                            "coverage": len(dates),
                        }
                    )

            if coverage_data:
                coverage_df = pd.DataFrame(coverage_data)

                # Temporal coverage bar plot
                axes[0].bar(
                    coverage_df["dataset"],
                    coverage_df["coverage"],
                    color="skyblue",
                    alpha=0.7,
                )
                axes[0].set_title("Temporal Coverage by Dataset")
                axes[0].set_ylabel("Number of Timepoints")
                axes[0].tick_params(axis="x", rotation=45)

                # Entity overlap heatmap
                if len(datasets) >= 2:
                    overlap_matrix = np.zeros((len(datasets), len(datasets)))
                    for i, dataset1 in enumerate(datasets):
                        entities1 = set(self.dataset_entities.get(dataset1, []))
                        for j, dataset2 in enumerate(datasets):
                            entities2 = set(self.dataset_entities.get(dataset2, []))
                            if entities1 and entities2:
                                overlap = len(entities1.intersection(entities2))
                                overlap_matrix[i, j] = (
                                    overlap / max(len(entities1), len(entities2)) * 100
                                )

                    im = axes[1].imshow(overlap_matrix, cmap="Blues", aspect="auto")
                    axes[1].set_xticks(range(len(datasets)))
                    axes[1].set_yticks(range(len(datasets)))
                    axes[1].set_xticklabels(datasets, rotation=45)
                    axes[1].set_yticklabels(datasets)
                    axes[1].set_title("Entity Overlap (%)")
                    plt.colorbar(im, ax=axes[1])

            plt.tight_layout()
            viz_path = self.output_dir / "alignment_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Alignment visualization saved to: {viz_path}")
            return str(viz_path)

        except Exception as e:
            logger.error(f"❌ Error creating alignment visualization: {e}")
            return None

    def generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary of investigation findings.

        Args:
            results: Complete investigation results

        Returns:
            Dictionary with executive summary
        """
        summary = {
            "key_findings": [],
            "critical_issues": [],
            "recommendations": [],
            "next_steps": [],
        }

        # Analyze zero-window prevalence
        zero_window_analysis = results.get("zero_window_analysis", {})
        threshold_analysis = zero_window_analysis.get("threshold_analysis", {})

        if threshold_analysis:
            # Check for critical zero-window issues
            for threshold, stats in threshold_analysis.items():
                threshold_val = float(threshold.split("_")[1])
                if threshold_val <= 0.1 and stats["zero_percentage"] > 30:
                    summary["critical_issues"].append(
                        f"High zero prevalence ({stats['zero_percentage']:.1f}%) at threshold {threshold_val}"
                    )

        # Analyze alignment quality
        alignment_analysis = results.get("alignment", {})
        alignment_quality = alignment_analysis.get("alignment_quality", {})

        for dataset, quality in alignment_quality.items():
            if quality["temporal_coverage"] < 50:
                summary["critical_issues"].append(
                    f"Poor temporal coverage for {dataset} ({quality['temporal_coverage']:.1f}%)"
                )
            if quality["entity_coverage"] < 50:
                summary["critical_issues"].append(
                    f"Poor entity coverage for {dataset} ({quality['entity_coverage']:.1f}%)"
                )

        # Generate key findings
        if "cases" in results.get("data_loading", {}).get("datasets_loaded", []):
            cases_stats = results["data_loading"]["statistics"].get("cases", {})
            summary["key_findings"].append(
                f"COVID data: {cases_stats.get('municipalities', 0)} municipalities, "
                f"{cases_stats.get('timepoints', 0)} timepoints, "
                f"{cases_stats.get('normalization', 'unknown')} normalization"
            )

        if "mobility" in results.get("data_loading", {}).get("datasets_loaded", []):
            mobility_stats = results["data_loading"]["statistics"].get("mobility", {})
            summary["key_findings"].append(
                f"Mobility data: {mobility_stats.get('nodes', 0)} nodes, "
                f"{mobility_stats.get('timepoints', 0)} timepoints, "
                f"{mobility_stats.get('features', 0)} features"
            )

        # Add recommendations
        summary["recommendations"] = results.get("recommendations", [])

        # Suggest next steps
        if summary["critical_issues"]:
            summary["next_steps"].append(
                "Address critical data quality issues before training"
            )
        else:
            summary["next_steps"].append(
                "Proceed with model training with minor data quality concerns"
            )

        summary["next_steps"].append("Consider implementing case threshold filtering")
        summary["next_steps"].append("Evaluate different normalization strategies")

        return summary

    def save_results(self, results: Dict[str, Any]) -> str:
        """
        Save investigation results to JSON file.

        Args:
            results: Investigation results dictionary

        Returns:
            Path to saved results file
        """
        import json

        results_path = self.output_dir / "investigation_results.json"

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        # Convert all numpy/torch objects
        results_json = json.dumps(results, default=convert_numpy, indent=2)

        with open(results_path, "w", encoding="utf-8") as f:
            f.write(results_json)

        logger.info(f"Investigation results saved to: {results_path}")
        return str(results_path)


def main():
    """Main function for data investigation."""
    # Parse arguments
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("🚀 Starting COVID-19 Data Investigation")

    try:
        # Create investigator
        investigator = DataInvestigator(args)

        # Run investigation
        results = investigator.run_investigation()

        # Save results
        results_path = investigator.save_results(results)

        # Print executive summary
        print("\n" + "=" * 60)
        print("📋 EXECUTIVE SUMMARY")
        print("=" * 60)

        executive_summary = results.get("executive_summary", {})
        key_findings = executive_summary.get("key_findings", [])
        critical_issues = executive_summary.get("critical_issues", [])
        recommendations = executive_summary.get("recommendations", [])

        if key_findings:
            print("\n🔍 Key Findings:")
            for i, finding in enumerate(key_findings, 1):
                print(f"  {i}. {finding}")

        if critical_issues:
            print("\n⚠️  Critical Issues:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")

        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n📊 Full investigation results saved to: {results_path}")
        print(f"📈 Visualizations saved to: {investigator.output_dir}")
        print(f"📝 Comprehensive report: {results.get('report_path', 'N/A')}")

        return 0

    except Exception as e:
        logger.error(f"❌ Data investigation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
