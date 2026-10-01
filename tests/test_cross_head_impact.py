import yaml


class TestIntegration:
    """Integration tests with realistic data patterns."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from runs to matrices."""
        from scripts.analysis.analyze_cross_head_impact import (
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

        # Create ablation runs (sig:ww:aux, 3 seeds matching baseline)
        ablation_dir = tmp_path / "sig_ww_aux"
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
                CrossHeadRun("sig:ww:aux", "test", ablation_dir, run_dir, seed)
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
