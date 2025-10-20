"""
Data Analytics Module for Epidemiological Forecasting.

This module provides comprehensive data analysis functions for investigating
data quality, zero-window prevalence, and alignment validation in COVID-19
forecasting datasets.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class DataAnalytics:
    """
    Comprehensive data analytics for epidemiological forecasting datasets.

    Provides statistical analysis, zero-window detection, alignment validation,
    and visualization capabilities for COVID-19 forecasting data.
    """

    def __init__(self, output_dir: str = "outputs/data_investigation"):
        """
        Initialize data analytics.

        Args:
            output_dir: Directory for saving analysis results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def analyze_zero_windows(
        self,
        target_data: Tensor,
        thresholds: List[float] = [0.0, 0.01, 0.1, 0.5, 1.0],
        normalization: str = "log1p",
    ) -> Dict[str, Any]:
        """
        Analyze prevalence of zero-windows in target data.

        Args:
            target_data: Target tensor [num_entities, num_timepoints]
            thresholds: List of thresholds to consider as "zero"
            normalization: Type of normalization applied ('log1p', 'standard', 'none')

        Returns:
            Dictionary with zero-window analysis results
        """
        logger.info(f"Analyzing zero-windows with {normalization} normalization")

        results = {
            "normalization": normalization,
            "data_shape": list(target_data.shape),
            "total_values": target_data.numel(),
            "threshold_analysis": {},
            "temporal_analysis": {},
            "entity_analysis": {},
            "recommendations": [],
        }

        # Analyze different thresholds
        for threshold in thresholds:
            zero_mask = target_data <= threshold
            zero_count = zero_mask.sum().item()
            zero_percentage = (zero_count / target_data.numel()) * 100

            results["threshold_analysis"][f"threshold_{threshold}"] = {
                "zero_count": int(zero_count),
                "zero_percentage": zero_percentage,
                "mean_non_zero": target_data[~zero_mask].mean().item()
                if (~zero_mask).any()
                else 0,
                "std_non_zero": target_data[~zero_mask].std().item()
                if (~zero_mask).any()
                else 0,
            }

        # Temporal analysis (by timepoint)
        temporal_zeros = (target_data <= 0.1).sum(
            dim=0
        )  # Using 0.1 as default threshold
        results["temporal_analysis"] = {
            "mean_zeros_per_timepoint": temporal_zeros.float().mean().item(),
            "max_zeros_per_timepoint": temporal_zeros.max().item(),
            "min_zeros_per_timepoint": temporal_zeros.min().item(),
            "high_zero_periods": (temporal_zeros > (target_data.shape[0] * 0.5))
            .sum()
            .item(),
        }

        # Entity analysis (by municipality/zone)
        entity_zeros = (target_data <= 0.1).sum(dim=1)
        results["entity_analysis"] = {
            "mean_zeros_per_entity": entity_zeros.float().mean().item(),
            "max_zeros_per_entity": entity_zeros.max().item(),
            "min_zeros_per_entity": entity_zeros.min().item(),
            "high_zero_entities": (entity_zeros > (target_data.shape[1] * 0.5))
            .sum()
            .item(),
        }

        # Generate recommendations
        self._generate_zero_window_recommendations(results)

        return results

    def validate_alignment(
        self,
        datasets: Dict[str, Tensor],
        dataset_dates: Dict[str, List[datetime]],
        dataset_entities: Dict[str, List[Union[int, str]]],
        target_dataset: str = "cases",
    ) -> Dict[str, Any]:
        """
        Validate dataset alignment quality and coverage.

        Args:
            datasets: Dictionary of dataset tensors
            dataset_dates: Dictionary of date lists for each dataset
            dataset_entities: Dictionary of entity lists for each dataset
            target_dataset: Name of target dataset

        Returns:
            Dictionary with alignment validation results
        """
        logger.info(f"Validating dataset alignment with target: {target_dataset}")

        results = {
            "target_dataset": target_dataset,
            "datasets_validated": list(datasets.keys()),
            "alignment_quality": {},
            "coverage_analysis": {},
            "temporal_overlap": {},
            "entity_overlap": {},
            "recommendations": [],
        }

        if target_dataset not in datasets:
            results["error"] = f"Target dataset {target_dataset} not found in datasets"
            return results

        target_dates = set(dataset_dates[target_dataset])
        target_entities = set(dataset_entities[target_dataset])

        # Analyze each dataset against target
        for dataset_name, dataset_tensor in datasets.items():
            if dataset_name == target_dataset:
                continue

            dates = set(dataset_dates[dataset_name])
            entities = set(dataset_entities[dataset_name])

            # Temporal overlap
            temporal_overlap = len(target_dates.intersection(dates))
            temporal_coverage = temporal_overlap / len(target_dates) * 100

            # Entity overlap
            entity_overlap = len(target_entities.intersection(entities))
            entity_coverage = entity_overlap / len(target_entities) * 100

            # Dataset shape compatibility
            shape_compatible = (
                dataset_tensor.shape[0] == datasets[target_dataset].shape[0]
                and dataset_tensor.shape[1] == datasets[target_dataset].shape[1]
            )

            results["alignment_quality"][dataset_name] = {
                "temporal_overlap": temporal_overlap,
                "temporal_coverage": temporal_coverage,
                "entity_overlap": entity_overlap,
                "entity_coverage": entity_coverage,
                "shape_compatible": shape_compatible,
            }

            # Coverage analysis
            results["coverage_analysis"][dataset_name] = {
                "target_timepoints": len(target_dates),
                "dataset_timepoints": len(dates),
                "missing_timepoints": len(target_dates - dates),
                "target_entities": len(target_entities),
                "dataset_entities": len(entities),
                "missing_entities": len(target_entities - entities),
            }

        # Overall temporal overlap analysis
        all_dates = set()
        for dates in dataset_dates.values():
            all_dates.update(dates)

        results["temporal_overlap"] = {
            "total_unique_dates": len(all_dates),
            "target_date_range": f"{min(target_dates)} to {max(target_dates)}",
            "dataset_coverage": {
                name: len(set(dates)) / len(all_dates) * 100
                for name, dates in dataset_dates.items()
            },
        }

        # Overall entity overlap analysis
        all_entities = set()
        for entities in dataset_entities.values():
            all_entities.update(entities)

        results["entity_overlap"] = {
            "total_unique_entities": len(all_entities),
            "target_entities": len(target_entities),
            "dataset_coverage": {
                name: len(set(entities)) / len(all_entities) * 100
                for name, entities in dataset_entities.items()
            },
        }

        # Generate recommendations
        self._generate_alignment_recommendations(results)

        return results

    def analyze_data_distributions(
        self, datasets: Dict[str, Tensor], dataset_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze data distributions across all datasets.

        Args:
            datasets: Dictionary of dataset tensors
            dataset_names: Optional list of dataset names to analyze

        Returns:
            Dictionary with distribution analysis results
        """
        if dataset_names is None:
            dataset_names = list(datasets.keys())

        logger.info(f"Analyzing data distributions for: {dataset_names}")

        results = {
            "datasets_analyzed": dataset_names,
            "basic_stats": {},
            "distribution_analysis": {},
            "correlation_analysis": {},
            "quality_metrics": {},
        }

        # Basic statistics for each dataset
        for name in dataset_names:
            if name not in datasets:
                continue

            data = datasets[name]
            flat_data = data.flatten()

            results["basic_stats"][name] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "min": data.min().item(),
                "max": data.max().item(),
                "mean": data.mean().item(),
                "median": data.median().item(),
                "std": data.std().item(),
                "zeros": (data == 0).sum().item(),
                "missing": torch.isnan(data).sum().item(),
            }

            # Distribution analysis
            if data.numel() > 0:
                results["distribution_analysis"][name] = {
                    "percentiles": {
                        "5th": torch.quantile(
                            flat_data[~torch.isnan(flat_data)], 0.05
                        ).item(),
                        "25th": torch.quantile(
                            flat_data[~torch.isnan(flat_data)], 0.25
                        ).item(),
                        "75th": torch.quantile(
                            flat_data[~torch.isnan(flat_data)], 0.75
                        ).item(),
                        "95th": torch.quantile(
                            flat_data[~torch.isnan(flat_data)], 0.95
                        ).item(),
                    },
                    "skewness": self._calculate_skewness(flat_data),
                    "kurtosis": self._calculate_kurtosis(flat_data),
                }

        # Correlation analysis (for datasets with same shape)
        compatible_datasets = [
            name
            for name in dataset_names
            if name in datasets
            and datasets[name].shape == datasets[dataset_names[0]].shape
        ]

        if len(compatible_datasets) > 1:
            correlation_matrix = torch.zeros(
                (len(compatible_datasets), len(compatible_datasets))
            )
            for i, name1 in enumerate(compatible_datasets):
                for j, name2 in enumerate(compatible_datasets):
                    if i <= j:
                        data1 = datasets[name1].flatten()
                        data2 = datasets[name2].flatten()
                        # Remove NaN values for correlation
                        mask = ~torch.isnan(data1) & ~torch.isnan(data2)
                        if mask.sum() > 1:
                            corr = torch.corrcoef(
                                torch.stack([data1[mask], data2[mask]])
                            )[0, 1]
                            correlation_matrix[i, j] = (
                                corr if not torch.isnan(corr) else 0
                            )
                        else:
                            correlation_matrix[i, j] = 0
                    else:
                        correlation_matrix[i, j] = correlation_matrix[j, i]

            results["correlation_analysis"] = {
                "datasets": compatible_datasets,
                "correlation_matrix": correlation_matrix.tolist(),
                "high_correlations": [],
            }

            # Find high correlations (> 0.7 or < -0.7)
            for i in range(len(compatible_datasets)):
                for j in range(i + 1, len(compatible_datasets)):
                    corr = correlation_matrix[i, j].item()
                    if abs(corr) > 0.7:
                        results["correlation_analysis"]["high_correlations"].append(
                            {
                                "dataset1": compatible_datasets[i],
                                "dataset2": compatible_datasets[j],
                                "correlation": corr,
                            }
                        )

        # Quality metrics
        for name in dataset_names:
            if name not in datasets:
                continue

            data = datasets[name]
            results["quality_metrics"][name] = {
                "completeness": (1 - torch.isnan(data).sum() / data.numel()).item(),
                "variance": data.var().item(),
                "range": (data.max() - data.min()).item(),
                "coefficient_of_variation": (data.std() / data.mean()).item()
                if data.mean() != 0
                else float("inf"),
            }

        return results

    def visualize_zero_windows(
        self,
        target_data: Tensor,
        thresholds: List[float] = [0.0, 0.01, 0.1, 0.5, 1.0],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create visualizations for zero-window analysis.

        Args:
            target_data: Target tensor [num_entities, num_timepoints]
            thresholds: List of thresholds to visualize
            save_path: Optional path to save the plot

        Returns:
            Path to saved visualization
        """
        if save_path is None:
            save_path = self.output_dir / "zero_window_analysis.png"

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Zero-Window Analysis", fontsize=16, fontweight="bold")

        # Threshold analysis
        zero_percentages = []
        for threshold in thresholds:
            zero_mask = target_data <= threshold
            zero_percentage = (zero_mask.sum() / target_data.numel()) * 100
            zero_percentages.append(zero_percentage)

        axes[0, 0].bar(
            [str(t) for t in thresholds], zero_percentages, color="skyblue", alpha=0.7
        )
        axes[0, 0].set_title("Zero-Window Percentage by Threshold")
        axes[0, 0].set_xlabel("Threshold")
        axes[0, 0].set_ylabel("Percentage of Values (%)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Temporal distribution of zeros
        temporal_zeros = (target_data <= 0.1).sum(dim=0).float()
        axes[0, 1].plot(temporal_zeros.numpy(), color="red", linewidth=2)
        axes[0, 1].axhline(
            y=temporal_zeros.mean().item(),
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Mean",
        )
        axes[0, 1].set_title("Temporal Distribution of Zeros (threshold=0.1)")
        axes[0, 1].set_xlabel("Time Index")
        axes[0, 1].set_ylabel("Number of Zero Entities")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Entity distribution of zeros
        entity_zeros = (target_data <= 0.1).sum(dim=1).float()
        axes[1, 0].hist(
            entity_zeros.numpy(), bins=30, color="green", alpha=0.7, edgecolor="black"
        )
        axes[1, 0].axvline(
            x=entity_zeros.mean().item(),
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Mean",
        )
        axes[1, 0].set_title("Entity Distribution of Zeros (threshold=0.1)")
        axes[1, 0].set_xlabel("Number of Zero Timepoints")
        axes[1, 0].set_ylabel("Number of Entities")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Heatmap of zero patterns (first 50 entities x first 50 timepoints)
        sample_size = min(50, target_data.shape[0], target_data.shape[1])
        zero_heatmap = (target_data[:sample_size, :sample_size] <= 0.1).float()
        im = axes[1, 1].imshow(zero_heatmap.numpy(), cmap="RdYlBu_r", aspect="auto")
        axes[1, 1].set_title(
            f"Zero Pattern Heatmap (first {sample_size}x{sample_size})"
        )
        axes[1, 1].set_xlabel("Time Index")
        axes[1, 1].set_ylabel("Entity Index")
        plt.colorbar(im, ax=axes[1, 1], label="Zero (1) vs Non-Zero (0)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Zero-window visualization saved to: {save_path}")
        return str(save_path)

    def visualize_data_distributions(
        self,
        datasets: Dict[str, Tensor],
        dataset_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create visualizations for data distributions.

        Args:
            datasets: Dictionary of dataset tensors
            dataset_names: Optional list of dataset names to visualize
            save_path: Optional path to save the plot

        Returns:
            Path to saved visualization
        """
        if dataset_names is None:
            dataset_names = list(datasets.keys())

        if save_path is None:
            save_path = self.output_dir / "data_distributions.png"

        n_datasets = len(dataset_names)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Data Distribution Analysis", fontsize=16, fontweight="bold")

        # Flatten datasets for distribution analysis
        flat_data = {}
        for name in dataset_names:
            if name in datasets:
                data = datasets[name].flatten()
                # Remove NaN and infinite values
                data = data[torch.isfinite(data)]
                flat_data[name] = data.numpy()

        # Distribution plots
        colors = plt.cm.Set3(np.linspace(0, 1, len(flat_data)))

        # Histogram
        axes[0, 0].set_title("Value Distributions")
        for i, (name, data) in enumerate(flat_data.items()):
            if len(data) > 0:
                axes[0, 0].hist(
                    data, bins=50, alpha=0.6, label=name, color=colors[i], density=True
                )
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Box plots
        axes[0, 1].set_title("Statistical Summary")
        box_data = []
        box_labels = []
        for name, data in flat_data.items():
            if len(data) > 0:
                box_data.append(data)
                box_labels.append(name)

        if box_data:
            axes[0, 1].boxplot(box_data, labels=box_labels)
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].grid(True, alpha=0.3)

        # Basic statistics comparison
        stats_comparison = pd.DataFrame(
            {
                name: {
                    "Mean": np.mean(data) if len(data) > 0 else 0,
                    "Std": np.std(data) if len(data) > 0 else 0,
                    "Min": np.min(data) if len(data) > 0 else 0,
                    "Max": np.max(data) if len(data) > 0 else 0,
                }
                for name, data in flat_data.items()
            }
        )

        # Statistics heatmap
        if len(stats_comparison.columns) > 0:
            im = axes[1, 0].imshow(
                stats_comparison.values, cmap="viridis", aspect="auto"
            )
            axes[1, 0].set_xticks(range(len(stats_comparison.columns)))
            axes[1, 0].set_xticklabels(stats_comparison.columns, rotation=45)
            axes[1, 0].set_yticks(range(len(stats_comparison.index)))
            axes[1, 0].set_yticklabels(stats_comparison.index)
            axes[1, 0].set_title("Statistics Comparison")
            plt.colorbar(im, ax=axes[1, 0])

        # Data quality metrics
        quality_metrics = pd.DataFrame(
            {
                name: {
                    "Completeness": (1 - np.isnan(data).sum() / len(data))
                    if len(data) > 0
                    else 0,
                    "Variance": np.var(data) if len(data) > 0 else 0,
                    "Range": (np.max(data) - np.min(data)) if len(data) > 0 else 0,
                }
                for name, data in flat_data.items()
            }
        )

        if len(quality_metrics.columns) > 0:
            quality_metrics.plot(kind="bar", ax=axes[1, 1], rot=45)
            axes[1, 1].set_title("Data Quality Metrics")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].legend(title="Metrics")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Data distribution visualization saved to: {save_path}")
        return str(save_path)

    def generate_comprehensive_report(
        self,
        zero_window_results: Dict[str, Any],
        alignment_results: Dict[str, Any],
        distribution_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            zero_window_results: Results from zero-window analysis
            alignment_results: Results from alignment validation
            distribution_results: Results from distribution analysis
            save_path: Optional path to save the report

        Returns:
            Path to saved report
        """
        if save_path is None:
            save_path = self.output_dir / "data_investigation_report.md"

        report = f"""# Data Investigation Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report provides a comprehensive analysis of the COVID-19 forecasting dataset,
focusing on zero-window prevalence, data alignment quality, and distribution characteristics.

## 1. Zero-Window Analysis

### Key Findings:
- **Normalization Method**: {zero_window_results.get("normalization", "Unknown")}
- **Data Shape**: {zero_window_results.get("data_shape", "Unknown")}
- **Total Values**: {zero_window_results.get("total_values", "Unknown")}

### Threshold Analysis:
"""

        # Add threshold analysis
        threshold_analysis = zero_window_results.get("threshold_analysis", {})
        for threshold, stats in threshold_analysis.items():
            report += f"""
- **{threshold}**: {stats["zero_percentage"]:.2f}% zeros ({stats["zero_count"]} values)
  - Mean non-zero: {stats["mean_non_zero"]:.4f}
  - Std non-zero: {stats["std_non_zero"]:.4f}
"""

        # Add temporal analysis
        temporal_analysis = zero_window_results.get("temporal_analysis", {})
        if temporal_analysis:
            report += f"""
### Temporal Patterns:
- Average zeros per timepoint: {temporal_analysis.get("mean_zeros_per_timepoint", 0):.1f}
- Maximum zeros in a timepoint: {temporal_analysis.get("max_zeros_per_timepoint", 0):.0f}
- High-zero periods (>50% entities): {temporal_analysis.get("high_zero_periods", 0)}
"""

        # Add entity analysis
        entity_analysis = zero_window_results.get("entity_analysis", {})
        if entity_analysis:
            report += f"""
### Entity Patterns:
- Average zeros per entity: {entity_analysis.get("mean_zeros_per_entity", 0):.1f}
- Maximum zeros for an entity: {entity_analysis.get("max_zeros_per_entity", 0):.0f}
- High-zero entities (>50% timepoints): {entity_analysis.get("high_zero_entities", 0)}
"""

        # Add alignment results
        report += f"""
## 2. Data Alignment Analysis

### Target Dataset: {alignment_results.get("target_dataset", "Unknown")}

### Alignment Quality:
"""

        alignment_quality = alignment_results.get("alignment_quality", {})
        for dataset, quality in alignment_quality.items():
            report += f"""
#### {dataset} vs Target:
- **Temporal Coverage**: {quality["temporal_coverage"]:.1f}% ({quality["temporal_overlap"]}/{quality.get("temporal_overlap", 0) + quality.get("missing_timepoints", 0)} timepoints)
- **Entity Coverage**: {quality["entity_coverage"]:.1f}% ({quality["entity_overlap"]}/{quality.get("entity_overlap", 0) + quality.get("missing_entities", 0)} entities)
- **Shape Compatible**: {"✓" if quality["shape_compatible"] else "✗"}
"""

        # Add distribution results
        report += f"""
## 3. Data Distribution Analysis

### Datasets Analyzed: {", ".join(distribution_results.get("datasets_analyzed", []))}

### Key Statistics:
"""

        basic_stats = distribution_results.get("basic_stats", {})
        for dataset, stats in basic_stats.items():
            report += f"""
#### {dataset}:
- **Shape**: {stats["shape"]}
- **Range**: [{stats["min"]:.4f}, {stats["max"]:.4f}]
- **Mean**: {stats["mean"]:.4f}
- **Std**: {stats["std"]:.4f}
- **Zeros**: {stats["zeros"]} ({stats["zeros"] / stats.get("shape", [1, 1])[0] / stats.get("shape", [1, 1])[1] * 100:.2f}%)
- **Missing**: {stats["missing"]}
"""

        # Add recommendations
        all_recommendations = []
        all_recommendations.extend(zero_window_results.get("recommendations", []))
        all_recommendations.extend(alignment_results.get("recommendations", []))

        if all_recommendations:
            report += """
## 4. Recommendations

"""
            for i, rec in enumerate(all_recommendations, 1):
                report += f"{i}. {rec}\n"

        # Save report
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Comprehensive report saved to: {save_path}")
        return str(save_path)

    def _calculate_skewness(self, data: Tensor) -> float:
        """Calculate skewness of data."""
        if data.numel() < 3:
            return 0.0

        data = data[torch.isfinite(data)]
        if data.numel() < 3:
            return 0.0

        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0

        skewness = ((data - mean) ** 3).mean() / (std**3)
        return skewness.item()

    def _calculate_kurtosis(self, data: Tensor) -> float:
        """Calculate kurtosis of data."""
        if data.numel() < 4:
            return 0.0

        data = data[torch.isfinite(data)]
        if data.numel() < 4:
            return 0.0

        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0

        kurtosis = ((data - mean) ** 4).mean() / (std**4) - 3
        return kurtosis.item()

    def _generate_zero_window_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate recommendations based on zero-window analysis."""
        recommendations = []

        threshold_analysis = results.get("threshold_analysis", {})
        temporal_analysis = results.get("temporal_analysis", {})
        entity_analysis = results.get("entity_analysis", {})

        # Check for excessive zeros
        if threshold_analysis:
            high_zero_threshold = None
            for threshold in [0.1, 0.5, 1.0]:
                key = f"threshold_{threshold}"
                if (
                    key in threshold_analysis
                    and threshold_analysis[key]["zero_percentage"] > 50
                ):
                    high_zero_threshold = threshold
                    break

            if high_zero_threshold:
                recommendations.append(
                    f"High prevalence of zero values (>50%) at threshold {high_zero_threshold}. "
                    f"Consider implementing case threshold filtering or weighted loss functions."
                )

        # Check temporal patterns
        if temporal_analysis.get("high_zero_periods", 0) > 0:
            recommendations.append(
                f"Found {temporal_analysis['high_zero_periods']} high-zero periods. "
                "Consider temporal stratification or focusing on high-incidence periods."
            )

        # Check entity patterns
        if entity_analysis.get("high_zero_entities", 0) > 0:
            recommendations.append(
                f"Found {entity_analysis['high_zero_entities']} high-zero entities. "
                "Consider entity filtering or minimum case requirements per municipality."
            )

        # Normalization recommendations
        normalization = results.get("normalization", "log1p")
        if normalization == "log1p":
            recommendations.append(
                "Current log1p normalization may be creating near-zero values. "
                "Consider testing standard normalization or adjusting the loss function."
            )

        results["recommendations"] = recommendations

    def _generate_alignment_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate recommendations based on alignment analysis."""
        recommendations = []

        alignment_quality = results.get("alignment_quality", {})

        for dataset, quality in alignment_quality.items():
            if quality["temporal_coverage"] < 80:
                recommendations.append(
                    f"Low temporal coverage for {dataset} ({quality['temporal_coverage']:.1f}%). "
                    "Consider extending data collection or adjusting alignment parameters."
                )

            if quality["entity_coverage"] < 80:
                recommendations.append(
                    f"Low entity coverage for {dataset} ({quality['entity_coverage']:.1f}%). "
                    "Consider improving entity matching or using spatial interpolation."
                )

            if not quality["shape_compatible"]:
                recommendations.append(
                    f"Shape incompatibility for {dataset}. "
                    "Check alignment and preprocessing steps."
                )

        results["recommendations"] = recommendations
