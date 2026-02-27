"""Tests for cross-head impact analysis.

Tests cover:
1. Matrix computation logic (pairwise deltas, aggregation)
2. Data loading from joint loss CSV files
3. Seed extraction and matching
4. Edge cases (missing data, NaN handling, baseline=0)
5. Visualization functions (heatmap generation)

This is a panel of tests corresponding to the cross-head impact analysis
module (scripts/analyze_cross_head_impact.py).
"""

import re
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# Import the module under test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze_cross_head_impact import (
    ABLATION_TO_HEAD,
    HEADS,
    HEAD_TO_COLUMN,
    aggregate_cross_head_matrix,
    compute_pairwise_deltas,
    extract_seed_from_config,
    format_impact_matrix,
    get_run_seed,
    load_joint_loss_metrics,
)


class TestHeadConstants:
    """Tests for head-related constants."""

    def test_heads_list(self):
        """Test that HEADS contains expected observation heads."""
        assert "ww" in HEADS
        assert "cases" in HEADS
        assert "hosp" in HEADS
        assert "deaths" in HEADS
        assert "sir" not in HEADS  # SIR should be excluded
        assert len(HEADS) == 4

    def test_head_to_column_mapping(self):
        """Test that all heads have corresponding CSV columns."""
        for head in HEADS:
            assert head in HEAD_TO_COLUMN
            col = HEAD_TO_COLUMN[head]
            assert col.startswith("joint_loss_")
            assert col.endswith("_median")

    def test_ablation_to_head_mapping(self):
        """Test ablation names map to correct heads."""
        assert ABLATION_TO_HEAD["no_ww_loss"] == "ww"
        assert ABLATION_TO_HEAD["no_cases_loss"] == "cases"
        assert ABLATION_TO_HEAD["no_hosp_loss"] == "hosp"
        assert ABLATION_TO_HEAD["no_deaths_loss"] == "deaths"


class TestSeedExtraction:
    """Tests for seed extraction from config and run directories."""

    def test_extract_seed_from_config_success(self, tmp_path):
        """Test successful seed extraction from config.yaml."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        config = {"training": {"seed": 42}, "model": {"type": "epiforecaster"}}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.dump(config))

        seed = extract_seed_from_config(run_dir)
        assert seed == 42

    def test_extract_seed_from_config_missing(self, tmp_path):
        """Test handling of missing config file."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        seed = extract_seed_from_config(run_dir)
        assert seed is None

    def test_extract_seed_from_config_no_training_section(self, tmp_path):
        """Test handling of config without training section."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        config = {"model": {"type": "epiforecaster"}}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.dump(config))

        seed = extract_seed_from_config(run_dir)
        assert seed is None

    def test_get_run_seed_prefers_config(self, tmp_path):
        """Test that config seed is preferred over directory parsing."""
        run_dir = tmp_path / "run_seed99"
        run_dir.mkdir()

        config = {"training": {"seed": 42}}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.dump(config))

        seed = get_run_seed(run_dir)
        assert seed == 42  # Config takes precedence


class TestLoadJointLossMetrics:
    """Tests for loading per-head losses from joint loss CSV."""

    def test_load_joint_loss_metrics_success(self, tmp_path):
        """Test successful loading of joint loss metrics."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        # Create test CSV
        csv_content = """model,folds,joint_loss_ww_median,joint_loss_hosp_median,joint_loss_cases_median,joint_loss_deaths_median
epiforecaster,1,0.5,0.3,0.4,0.2
"""
        metrics_file = run_dir / "test_main_model_joint_loss_aggregate.csv"
        metrics_file.write_text(csv_content)

        metrics = load_joint_loss_metrics(run_dir, split="test")

        assert metrics is not None
        assert metrics["ww"] == 0.5
        assert metrics["hosp"] == 0.3
        assert metrics["cases"] == 0.4
        assert metrics["deaths"] == 0.2

    def test_load_joint_loss_metrics_missing_file(self, tmp_path):
        """Test handling of missing metrics file."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        metrics = load_joint_loss_metrics(run_dir, split="test")
        assert metrics is None

    def test_load_joint_loss_metrics_with_nan(self, tmp_path):
        """Test handling of NaN values in metrics."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        csv_content = """model,folds,joint_loss_ww_median,joint_loss_hosp_median,joint_loss_cases_median,joint_loss_deaths_median
epiforecaster,1,0.5,,0.4,0.2
"""
        metrics_file = run_dir / "test_main_model_joint_loss_aggregate.csv"
        metrics_file.write_text(csv_content)

        metrics = load_joint_loss_metrics(run_dir, split="test")

        assert metrics is not None
        assert metrics["ww"] == 0.5
        assert metrics["hosp"] is None  # NaN becomes None
        assert metrics["cases"] == 0.4
        assert metrics["deaths"] == 0.2

    def test_load_joint_loss_metrics_with_inf(self, tmp_path):
        """Test handling of infinite values in metrics."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        csv_content = """model,folds,joint_loss_ww_median,joint_loss_hosp_median,joint_loss_cases_median,joint_loss_deaths_median
epiforecaster,1,0.5,inf,0.4,0.2
"""
        metrics_file = run_dir / "test_main_model_joint_loss_aggregate.csv"
        metrics_file.write_text(csv_content)

        metrics = load_joint_loss_metrics(run_dir, split="test")

        assert metrics is not None
        assert metrics["ww"] == 0.5
        assert metrics["hosp"] is None  # Inf becomes None
        assert metrics["cases"] == 0.4
        assert metrics["deaths"] == 0.2


class TestComputePairwiseDeltas:
    """Tests for pairwise delta computation."""

    def test_compute_pairwise_deltas_basic(self, tmp_path):
        """Test basic pairwise delta computation."""
        # Create baseline runs
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()

        for seed in [42, 43]:
            run_dir = baseline_dir / f"run_{seed}"
            run_dir.mkdir()
            config = {"training": {"seed": seed}}
            (run_dir / "config.yaml").write_text(yaml.dump(config))

            # Create metrics file
            csv_content = f"""model,folds,joint_loss_ww_median,joint_loss_hosp_median
epiforecaster,1,{1.0 + seed * 0.01},{0.5 + seed * 0.01}
"""
            (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(
                csv_content
            )

        # Create ablation runs
        ablation_dir = tmp_path / "no_ww"
        ablation_dir.mkdir()

        for seed in [42, 43]:
            run_dir = ablation_dir / f"run_{seed}"
            run_dir.mkdir()
            config = {"training": {"seed": seed}}
            (run_dir / "config.yaml").write_text(yaml.dump(config))

            # Create metrics file (higher loss when ww is ablated)
            csv_content = f"""model,folds,joint_loss_ww_median,joint_loss_hosp_median
epiforecaster,1,{1.5 + seed * 0.01},{0.6 + seed * 0.01}
"""
            (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(
                csv_content
            )

        # Create CrossHeadRun objects
        from analyze_cross_head_impact import CrossHeadRun

        baseline_runs = [
            CrossHeadRun("baseline", "test", baseline_dir, baseline_dir / "run_42", 42),
            CrossHeadRun("baseline", "test", baseline_dir, baseline_dir / "run_43", 43),
        ]

        ablation_runs = [
            CrossHeadRun(
                "no_ww_loss", "test", ablation_dir, ablation_dir / "run_42", 42
            ),
            CrossHeadRun(
                "no_ww_loss", "test", ablation_dir, ablation_dir / "run_43", 43
            ),
        ]

        result = compute_pairwise_deltas(baseline_runs, ablation_runs, split="test")

        assert result is not None
        assert len(result) == 4  # 2 seeds × 2 measured heads

        # Check structure
        assert "ablated_head" in result.columns
        assert "measured_head" in result.columns
        assert "baseline_loss" in result.columns
        assert "ablation_loss" in result.columns
        assert "delta_abs" in result.columns
        assert "delta_pct" in result.columns
        assert "seed" in result.columns

    def test_compute_pairwise_deltas_no_matching_seeds(self, tmp_path):
        """Test when no seeds match between baseline and ablation."""
        from analyze_cross_head_impact import CrossHeadRun

        baseline_runs = [
            CrossHeadRun("baseline", "test", tmp_path, tmp_path / "run_1", 42),
        ]

        ablation_runs = [
            CrossHeadRun(
                "no_ww_loss", "test", tmp_path, tmp_path / "run_1", 99
            ),  # Different seed
        ]

        result = compute_pairwise_deltas(baseline_runs, ablation_runs, split="test")
        assert result is None

    def test_compute_pairwise_deltas_missing_metrics(self, tmp_path):
        """Test handling when metrics files are missing."""
        from analyze_cross_head_impact import CrossHeadRun

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        baseline_runs = [
            CrossHeadRun("baseline", "test", tmp_path, run_dir, 42),
        ]

        ablation_runs = [
            CrossHeadRun("no_ww_loss", "test", tmp_path, run_dir, 42),
        ]

        result = compute_pairwise_deltas(baseline_runs, ablation_runs, split="test")
        assert result is None


class TestAggregateCrossHeadMatrix:
    """Tests for matrix aggregation."""

    def test_aggregate_cross_head_matrix_basic(self):
        """Test basic matrix aggregation."""
        # Create sample pairwise data
        data = {
            "ablated_head": ["ww", "ww", "cases", "cases"],
            "measured_head": ["hosp", "deaths", "hosp", "deaths"],
            "delta_pct": [10.0, 5.0, 3.0, 2.0],
            "seed": [42, 42, 43, 43],
            "baseline_loss": [1.0, 1.0, 1.0, 1.0],
            "ablation_loss": [1.1, 1.05, 1.03, 1.02],
            "delta_abs": [0.1, 0.05, 0.03, 0.02],
        }
        pairwise_df = pd.DataFrame(data)

        mean_matrix, std_matrix, count_matrix = aggregate_cross_head_matrix(pairwise_df)

        # Check structure
        assert "ww" in mean_matrix.index
        assert "cases" in mean_matrix.index
        assert "hosp" in mean_matrix.columns
        assert "deaths" in mean_matrix.columns

        # Check values
        assert mean_matrix.loc["ww", "hosp"] == 10.0
        assert mean_matrix.loc["ww", "deaths"] == 5.0
        assert count_matrix.loc["ww", "hosp"] == 1

    def test_aggregate_cross_head_matrix_multiple_seeds(self):
        """Test aggregation with multiple seeds per ablation."""
        data = {
            "ablated_head": ["ww", "ww", "ww", "ww"],
            "measured_head": ["hosp", "hosp", "hosp", "hosp"],
            "delta_pct": [10.0, 12.0, 8.0, 14.0],
            "seed": [42, 43, 44, 45],
            "baseline_loss": [1.0, 1.0, 1.0, 1.0],
            "ablation_loss": [1.1, 1.12, 1.08, 1.14],
            "delta_abs": [0.1, 0.12, 0.08, 0.14],
        }
        pairwise_df = pd.DataFrame(data)

        mean_matrix, std_matrix, count_matrix = aggregate_cross_head_matrix(pairwise_df)

        # Mean should be average of [10, 12, 8, 14]
        expected_mean = 11.0
        assert mean_matrix.loc["ww", "hosp"] == pytest.approx(expected_mean, abs=1e-6)

        # Count should be 4
        assert count_matrix.loc["ww", "hosp"] == 4

        # Std should be computed
        assert std_matrix.loc["ww", "hosp"] > 0


class TestFormatImpactMatrix:
    """Tests for matrix formatting."""

    def test_format_impact_matrix_basic(self):
        """Test basic formatting with mean ± std."""
        mean_data = {"hosp": [10.0, 5.0], "deaths": [3.0, 2.0]}
        mean_matrix = pd.DataFrame(mean_data, index=["ww", "cases"])

        std_data = {"hosp": [1.0, 0.5], "deaths": [0.3, 0.2]}
        std_matrix = pd.DataFrame(std_data, index=["ww", "cases"])

        count_data = {"hosp": [3, 3], "deaths": [3, 3]}
        count_matrix = pd.DataFrame(count_data, index=["ww", "cases"])

        formatted = format_impact_matrix(mean_matrix, std_matrix, count_matrix)

        # Check formatting
        assert "+10.0±1.0 (n=3)" in formatted.loc["ww", "hosp"]
        assert "+5.0±0.5 (n=3)" in formatted.loc["cases", "hosp"]

    def test_format_impact_matrix_with_nan(self):
        """Test formatting with NaN values."""
        mean_data = {"hosp": [10.0, np.nan]}
        mean_matrix = pd.DataFrame(mean_data, index=["ww", "cases"])

        std_data = {"hosp": [1.0, np.nan]}
        std_matrix = pd.DataFrame(std_data, index=["ww", "cases"])

        count_data = {"hosp": [3, 0]}
        count_matrix = pd.DataFrame(count_data, index=["ww", "cases"])

        formatted = format_impact_matrix(mean_matrix, std_matrix, count_matrix)

        assert "+10.0±1.0 (n=3)" in formatted.loc["ww", "hosp"]
        assert formatted.loc["cases", "hosp"] == "N/A"


class TestEdgeCases:
    """Edge case tests."""

    def test_baseline_zero_handling(self, tmp_path):
        """Test handling when baseline loss is zero."""
        from analyze_cross_head_impact import CrossHeadRun

        # Create baseline with zero loss
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        run_dir = baseline_dir / "run_42"
        run_dir.mkdir()
        config = {"training": {"seed": 42}}
        (run_dir / "config.yaml").write_text(yaml.dump(config))

        csv_content = """model,folds,joint_loss_ww_median,joint_loss_hosp_median
epiforecaster,1,0.0,0.5
"""
        (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(csv_content)

        # Create ablation with non-zero loss
        ablation_dir = tmp_path / "no_ww"
        ablation_dir.mkdir()
        run_dir = ablation_dir / "run_42"
        run_dir.mkdir()
        config = {"training": {"seed": 42}}
        (run_dir / "config.yaml").write_text(yaml.dump(config))

        csv_content = """model,folds,joint_loss_ww_median,joint_loss_hosp_median
epiforecaster,1,0.5,0.6
"""
        (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(csv_content)

        baseline_runs = [
            CrossHeadRun("baseline", "test", baseline_dir, baseline_dir / "run_42", 42),
        ]

        ablation_runs = [
            CrossHeadRun(
                "no_ww_loss", "test", ablation_dir, ablation_dir / "run_42", 42
            ),
        ]

        result = compute_pairwise_deltas(baseline_runs, ablation_runs, split="test")

        # When baseline is 0, delta_pct should be NaN
        ww_row = result[result["measured_head"] == "ww"].iloc[0]
        assert np.isnan(ww_row["delta_pct"])

        # But delta_abs should still be computed
        assert ww_row["delta_abs"] == 0.5

    def test_negative_delta(self, tmp_path):
        """Test handling when ablation improves loss (negative delta)."""
        from analyze_cross_head_impact import CrossHeadRun

        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        run_dir = baseline_dir / "run_42"
        run_dir.mkdir()
        config = {"training": {"seed": 42}}
        (run_dir / "config.yaml").write_text(yaml.dump(config))

        csv_content = """model,folds,joint_loss_ww_median
epiforecaster,1,1.0
"""
        (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(csv_content)

        ablation_dir = tmp_path / "no_ww"
        ablation_dir.mkdir()
        run_dir = ablation_dir / "run_42"
        run_dir.mkdir()
        config = {"training": {"seed": 42}}
        (run_dir / "config.yaml").write_text(yaml.dump(config))

        # Lower loss (improvement)
        csv_content = """model,folds,joint_loss_ww_median
epiforecaster,1,0.8
"""
        (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(csv_content)

        baseline_runs = [
            CrossHeadRun("baseline", "test", baseline_dir, baseline_dir / "run_42", 42),
        ]

        ablation_runs = [
            CrossHeadRun(
                "no_ww_loss", "test", ablation_dir, ablation_dir / "run_42", 42
            ),
        ]

        result = compute_pairwise_deltas(baseline_runs, ablation_runs, split="test")

        # Should be negative (improvement)
        assert result.iloc[0]["delta_pct"] == pytest.approx(
            -20.0, abs=1e-10
        )  # (0.8 - 1.0) / 1.0 * 100


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
        reason="matplotlib not installed",
    )
    def test_plot_cross_head_impact_heatmap(self, tmp_path):
        """Test heatmap generation doesn't crash."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        from dataviz.ablation_plots import plot_cross_head_impact_heatmap

        # Create test matrix
        mean_data = {"hosp": [10.0, 5.0], "deaths": [3.0, 2.0]}
        mean_df = pd.DataFrame(mean_data, index=["ww", "cases"])

        mean_csv = tmp_path / "cross_head_mean_matrix.csv"
        mean_df.to_csv(mean_csv)

        # Should not raise
        output_path = plot_cross_head_impact_heatmap(mean_csv, output_dir=tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".png"

    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
        reason="matplotlib not installed",
    )
    def test_plot_cross_head_impact_with_std(self, tmp_path):
        """Test heatmap with std annotations."""
        import matplotlib

        matplotlib.use("Agg")

        from dataviz.ablation_plots import plot_cross_head_impact_heatmap

        mean_data = {"hosp": [10.0], "deaths": [3.0]}
        mean_df = pd.DataFrame(mean_data, index=["ww"])
        mean_csv = tmp_path / "mean.csv"
        mean_df.to_csv(mean_csv)

        std_data = {"hosp": [1.0], "deaths": [0.3]}
        std_df = pd.DataFrame(std_data, index=["ww"])
        std_csv = tmp_path / "std.csv"
        std_df.to_csv(std_csv)

        output_path = plot_cross_head_impact_heatmap(
            mean_csv, std_csv, output_dir=tmp_path
        )

        assert output_path.exists()


class TestIntegration:
    """Integration tests with realistic data patterns."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from runs to matrices."""
        from analyze_cross_head_impact import (
            CrossHeadRun,
            compute_pairwise_deltas,
            aggregate_cross_head_matrix,
        )

        # Create baseline runs (3 seeds)
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()

        baseline_runs = []
        for seed in [42, 43, 44]:
            run_dir = baseline_dir / f"run_{seed}"
            run_dir.mkdir()
            config = {"training": {"seed": seed}}
            (run_dir / "config.yaml").write_text(yaml.dump(config))

            # Baseline losses vary slightly by seed
            ww_loss = 1.0 + seed * 0.001
            hosp_loss = 0.5 + seed * 0.001
            cases_loss = 0.8 + seed * 0.001
            deaths_loss = 0.3 + seed * 0.001

            csv_content = f"""model,folds,joint_loss_ww_median,joint_loss_hosp_median,joint_loss_cases_median,joint_loss_deaths_median
epiforecaster,1,{ww_loss},{hosp_loss},{cases_loss},{deaths_loss}
"""
            (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(
                csv_content
            )

            baseline_runs.append(
                CrossHeadRun("baseline", "test", baseline_dir, run_dir, seed)
            )

        # Create ablation runs (no_ww, 3 seeds matching baseline)
        ablation_dir = tmp_path / "no_ww"
        ablation_dir.mkdir()

        ablation_runs = []
        for seed in [42, 43, 44]:
            run_dir = ablation_dir / f"run_{seed}"
            run_dir.mkdir()
            config = {"training": {"seed": seed}}
            (run_dir / "config.yaml").write_text(yaml.dump(config))

            # When ww is ablated, ww loss increases (weight=0)
            # Other heads may also be affected
            ww_loss = 2.0 + seed * 0.001  # Higher without ww head
            hosp_loss = 0.6 + seed * 0.001  # Slightly higher
            cases_loss = 0.9 + seed * 0.001
            deaths_loss = 0.35 + seed * 0.001

            csv_content = f"""model,folds,joint_loss_ww_median,joint_loss_hosp_median,joint_loss_cases_median,joint_loss_deaths_median
epiforecaster,1,{ww_loss},{hosp_loss},{cases_loss},{deaths_loss}
"""
            (run_dir / "test_main_model_joint_loss_aggregate.csv").write_text(
                csv_content
            )

            ablation_runs.append(
                CrossHeadRun("no_ww_loss", "test", ablation_dir, run_dir, seed)
            )

        # Compute pairwise deltas
        pairwise_df = compute_pairwise_deltas(
            baseline_runs, ablation_runs, split="test"
        )

        assert pairwise_df is not None
        assert len(pairwise_df) == 12  # 3 seeds × 4 heads

        # Aggregate
        mean_matrix, std_matrix, count_matrix = aggregate_cross_head_matrix(pairwise_df)

        assert mean_matrix.loc["ww", "ww"] > 50  # Large increase when ablated
        assert mean_matrix.loc["ww", "hosp"] > 10  # Moderate cross-effect
        assert count_matrix.loc["ww", "ww"] == 3  # 3 seeds

        # All std should be small (similar seeds)
        assert std_matrix.loc["ww", "ww"] < 1.0
