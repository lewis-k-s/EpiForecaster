from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from evaluation.selection import WindowSelectionSpec

import cli as cli_module


def test_eval_cli_uses_output_directory_for_all_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_extract_run_from_checkpoint_path(_checkpoint: Path):
        return None

    def _fake_eval_checkpoint(**kwargs):
        captured["eval_checkpoint_kwargs"] = kwargs
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=2)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1, 1: 0.2}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    def _fake_select_nodes_by_loss(**kwargs):
        captured["select_nodes_kwargs"] = kwargs
        return {"Q1 (Best MAE)": [0], "Q4 (Worst MAE)": [1]}

    def _fake_generate_forecast_plots(**kwargs):
        captured["generate_forecast_plots_kwargs"] = kwargs
        return {
            "selected_nodes": [0, 1],
            "node_groups": {"Q1 (Best MAE)": [0], "Q4 (Worst MAE)": [1]},
        }

    def _fake_render_eval_per_head_plots(**kwargs):
        captured["render_eval_per_head_plots_kwargs"] = kwargs
        return {
            "perf_vs_population_cases": kwargs["output_dir"]
            / "perf_vs_population_cases.png"
        }

    def _fake_render_baseline_delta_plots(**kwargs):
        captured["render_baseline_delta_plots_kwargs"] = kwargs
        output_path = kwargs["output_dir"] / "test_baseline_comparison_sarima.png"
        output_path.write_text("plot", encoding="utf-8")
        return {"baseline_comparison_sarima": output_path}

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        _fake_extract_run_from_checkpoint_path,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        _fake_select_nodes_by_loss,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        _fake_generate_forecast_plots,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        _fake_render_eval_per_head_plots,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_baseline_delta_plots",
        _fake_render_baseline_delta_plots,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_csv_path"] is None
    assert eval_kwargs["node_metrics_target"] == "hospitalizations"
    assert "output.write_granular_eval=false" in eval_kwargs["overrides"]

    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["output_path"] == output_dir / "test_forecasts.png"

    per_head_kwargs = captured["render_eval_per_head_plots_kwargs"]
    assert per_head_kwargs["per_head_node_metrics_csv"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert per_head_kwargs["output_dir"] == output_dir


def test_eval_cli_creates_missing_output_directory_for_all_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "fresh_eval_outputs"

    captured: dict[str, object] = {}

    def _fake_extract_run_from_checkpoint_path(_checkpoint: Path):
        return None

    def _fake_eval_checkpoint(**kwargs):
        captured["eval_checkpoint_kwargs"] = kwargs
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=1)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        _fake_extract_run_from_checkpoint_path,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        lambda **kwargs: {"Q1 (Best MAE)": [0]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: {"selected_nodes": [0], "node_groups": {"Q1 (Best MAE)": [0]}},
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_dir.is_dir()
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_csv_path"] is None


def test_eval_cli_granular_flag_enables_auto_granular_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        lambda _checkpoint: None,
    )

    def _fake_eval_checkpoint(**kwargs):
        captured["eval_checkpoint_kwargs"] = kwargs
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=1)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        lambda **kwargs: {"Q1 (Best MAE)": [0]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: {"selected_nodes": [0], "node_groups": {"Q1 (Best MAE)": [0]}},
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_dir),
            "--granular",
        ],
    )

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_csv_path"] == output_dir / "test_granular.csv"
    assert "output.write_granular_eval=true" in eval_kwargs["overrides"]


def test_eval_cli_full_split_is_accepted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        lambda _checkpoint: None,
    )

    def _fake_eval_checkpoint(**kwargs):
        captured["eval_checkpoint_kwargs"] = kwargs
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=1)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        lambda **kwargs: {"Q1 (Best MAE)": [0]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: {"selected_nodes": [0], "node_groups": {"Q1 (Best MAE)": [0]}},
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "full",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["split"] == "full"
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "full_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "full_node_metrics_per_head.csv"
    )


def test_eval_cli_compare_evals_runs_fresh_baselines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_extract_run_from_checkpoint_path(_checkpoint: Path):
        return None

    def _fake_eval_checkpoint(**kwargs):
        captured["eval_checkpoint_kwargs"] = kwargs
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=2)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1, 1: 0.2}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    def _fake_select_nodes_by_loss(**kwargs):
        return {"Q1 (Best MAE)": [0]}

    def _fake_generate_forecast_plots(**kwargs):
        return {"selected_nodes": [0], "node_groups": {"Q1 (Best MAE)": [0]}}

    def _fake_render_eval_per_head_plots(**kwargs):
        return {}

    def _fake_render_baseline_delta_plots(**kwargs):
        captured["render_baseline_delta_plots_kwargs"] = kwargs
        output_path = kwargs["output_dir"] / "test_baseline_comparison_sarima.png"
        output_path.write_text("plot", encoding="utf-8")
        return {"baseline_comparison_sarima": output_path}

    def _fake_run_same_slice_baseline_evaluation(**kwargs):
        captured["run_same_slice_baseline_evaluation_kwargs"] = kwargs
        baseline_dir = kwargs["output_dir"]
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_csv = baseline_dir / "baseline_aggregate_metrics.csv"
        baseline_csv.write_text(
            "model,target,mae_mean,rmse_mean,r2_mean\n", encoding="utf-8"
        )
        (baseline_dir / "baseline_metadata.json").write_text(
            '{"comparison_scope":"same_eval_slice"}',
            encoding="utf-8",
        )
        return {
            "baseline_aggregate_metrics": baseline_csv,
            "baseline_metadata": baseline_dir / "baseline_metadata.json",
        }

    def _fake_compare_model_metrics_against_baselines(**kwargs):
        captured["compare_kwargs"] = kwargs
        kwargs["output_csv"].write_text(
            (
                "target,baseline_model,metric,model_value,baseline_value\n"
                "cases,sarima,mae,0.3,0.4\n"
                "cases,sarima,r2,0.2,0.1\n"
            ),
            encoding="utf-8",
        )
        return kwargs["output_csv"]

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        _fake_extract_run_from_checkpoint_path,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        _fake_select_nodes_by_loss,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        _fake_generate_forecast_plots,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        _fake_render_eval_per_head_plots,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_baseline_delta_plots",
        _fake_render_baseline_delta_plots,
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.run_same_slice_baseline_evaluation",
        _fake_run_same_slice_baseline_evaluation,
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.compare_model_metrics_against_baselines",
        _fake_compare_model_metrics_against_baselines,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_dir),
            "--granular",
            "--compare-evals",
            "--compare-baselines",
            str(tmp_path / "ignored.csv"),
        ],
    )

    assert result.exit_code == 0, result.output
    baseline_kwargs = captured["run_same_slice_baseline_evaluation_kwargs"]
    assert baseline_kwargs["models"] == [
        "exp_smoothing",
        "last_observed",
        "sarima",
        "var",
    ]
    assert baseline_kwargs["split"] == "test"
    assert (
        baseline_kwargs["output_dir"] == output_dir / "test_baseline_eval_same_window"
    )

    compare_kwargs = captured["compare_kwargs"]
    assert compare_kwargs["baseline_results_csv"] == (
        output_dir / "test_baseline_eval_same_window" / "baseline_aggregate_metrics.csv"
    )
    baseline_plot_kwargs = captured["render_baseline_delta_plots_kwargs"]
    assert (
        baseline_plot_kwargs["baseline_deltas_csv"]
        == output_dir / "test_baseline_deltas.csv"
    )
    assert baseline_plot_kwargs["output_dir"] == output_dir
    assert (
        "--compare-evals supplied; ignoring explicit --compare-baselines path"
        in result.output
    )


def test_eval_cli_compare_evals_rejects_full_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        lambda _checkpoint: None,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        lambda **kwargs: {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=1)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        },
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        lambda **kwargs: {"Q1 (Best MAE)": [0]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: {"selected_nodes": [0], "node_groups": {"Q1 (Best MAE)": [0]}},
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "full",
            "--output",
            str(output_dir),
            "--compare-evals",
        ],
    )

    assert result.exit_code != 0
    assert "--compare-evals is not supported with --split full" in result.output


def test_eval_cli_granular_window_selection_uses_generated_granular_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}

    def _fake_extract_run_from_checkpoint_path(_checkpoint: Path):
        return None

    def _fake_eval_checkpoint(**kwargs):
        granular_csv = kwargs["granular_csv_path"]
        granular_csv.write_text(
            "split,target,node_id,window_start,abs_error\n", encoding="utf-8"
        )
        return {
            "config": SimpleNamespace(training=SimpleNamespace(num_forecast_samples=2)),
            "model": object(),
            "loader": object(),
            "node_mae": {"hospitalizations": {0: 0.1}},
            "eval_loss": 0.5,
            "eval_metrics": {
                "mae": 0.1,
                "mae_hosp_log1p_per_100k": 0.2,
                "mae_ww_log1p_per_100k": 0.3,
                "mae_cases_log1p_per_100k": 0.4,
                "mae_deaths_log1p_per_100k": 0.5,
            },
        }

    def _fake_load_window_selection_specs_from_granular(**kwargs):
        captured["load_window_specs_kwargs"] = kwargs
        return [
            WindowSelectionSpec(
                node_id=7,
                window_start=11,
                score=0.25,
                observed_targets=("cases", "hospitalizations"),
                observed_points=6,
            )
        ]

    def _fake_select_windows_by_loss(**kwargs):
        captured["select_windows_kwargs"] = kwargs
        return {"Q1 (Best MAE)": kwargs["window_specs"]}

    def _fake_generate_forecast_plots(**kwargs):
        captured["generate_forecast_plots_kwargs"] = kwargs
        return {
            "selected_nodes": [7],
            "node_groups": {},
            "window_groups": kwargs["window_groups"],
        }

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        _fake_extract_run_from_checkpoint_path,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint",
        _fake_eval_checkpoint,
    )
    monkeypatch.setattr(
        "evaluation.selection.load_window_selection_specs_from_granular",
        _fake_load_window_selection_specs_from_granular,
    )
    monkeypatch.setattr(
        "evaluation.selection.select_windows_by_loss",
        _fake_select_windows_by_loss,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        _fake_generate_forecast_plots,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss",
        lambda **kwargs: {"unused": []},
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.compare_model_metrics_against_baselines",
        lambda **kwargs: kwargs["output_csv"],
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        lambda **kwargs: {},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--output",
            str(output_dir),
            "--granular",
            "--selection-mode",
            "granular_window_quartile",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        captured["load_window_specs_kwargs"]["granular_csv"]
        == output_dir / "test_granular.csv"
    )
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["node_groups"] is None
    assert plot_kwargs["window_groups"]["Q1 (Best MAE)"][0].window_start == 11


def test_plot_forecasts_window_quartile_uses_granular_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"dummy")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path",
        lambda _checkpoint: None,
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.load_model_from_checkpoint",
        lambda checkpoint, device="auto", overrides=None: (
            __import__("torch").nn.Linear(1, 1),
            SimpleNamespace(
                output=SimpleNamespace(
                    resolve_granular_eval_filename=lambda split: f"{split}_granular.csv"
                ),
                training=SimpleNamespace(),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.build_loader_from_config",
        lambda config, split, device="auto": (
            SimpleNamespace(
                dataset=type(
                    "_Dataset",
                    (),
                    {
                        "_valid_window_starts_by_node": {},
                        "__len__": lambda self: 0,
                    },
                )()
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        "evaluation.selection.load_window_selection_specs_from_granular",
        lambda **kwargs: [
            WindowSelectionSpec(3, 9, 0.2, ("cases",), 2),
        ],
    )
    monkeypatch.setattr(
        "evaluation.selection.select_windows_by_loss",
        lambda **kwargs: {"Q1 (Best MAE)": kwargs["window_specs"]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: captured.setdefault("generate_forecast_plots_kwargs", kwargs),
    )

    granular_csv = checkpoint_path.parent.parent / "test_granular.csv"
    granular_csv.write_text(
        "split,target,node_id,window_start,abs_error\n", encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "plot",
            "forecasts",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--nodes",
            "window_quartile:1",
        ],
    )

    assert result.exit_code == 0, result.output
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["node_groups"] is None
    assert plot_kwargs["window_groups"]["Q1 (Best MAE)"][0].node_id == 3


def test_plot_forecasts_window_worst_uses_granular_tail_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"dummy")
    granular_csv = checkpoint_path.parent.parent / "test_granular.csv"
    granular_csv.write_text(
        "split,target,node_id,window_start,abs_error\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.load_model_from_checkpoint",
        lambda checkpoint, device="auto", overrides=None: (
            __import__("torch").nn.Linear(1, 1),
            SimpleNamespace(
                output=SimpleNamespace(
                    resolve_granular_eval_filename=lambda split: f"{split}_granular.csv"
                ),
                training=SimpleNamespace(),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.build_loader_from_config",
        lambda config, split, device="auto": (
            SimpleNamespace(
                dataset=type(
                    "_Dataset",
                    (),
                    {
                        "_valid_window_starts_by_node": {},
                        "__len__": lambda self: 0,
                    },
                )()
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        "evaluation.selection.load_window_selection_specs_from_granular",
        lambda **kwargs: [
            WindowSelectionSpec(3, 9, 0.2, ("cases",), 2),
        ],
    )
    monkeypatch.setattr(
        "evaluation.selection.select_worst_windows_by_loss",
        lambda **kwargs: {"Worst MAE": kwargs["window_specs"]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: captured.setdefault("generate_forecast_plots_kwargs", kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "plot",
            "forecasts",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--nodes",
            "window_worst:1",
            "--include-sird-latents",
        ],
    )

    assert result.exit_code == 0, result.output
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["node_groups"] is None
    assert plot_kwargs["window_groups"]["Worst MAE"][0].node_id == 3
    assert plot_kwargs["include_sird_latents"] is True
