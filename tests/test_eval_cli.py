from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import cli as cli_module
from evaluation.selection import WindowSelectionSpec
from tests.eval_test_helpers import (
    install_eval_cli_harness,
    install_plot_forecast_harness,
    make_checkpoint,
    make_eval_result,
)


def _invoke_eval(
    checkpoint_path: Path,
    output_dir: Path,
    *extra_args: str,
    split: str = "test",
):
    return CliRunner().invoke(
        cli_module.cli,
        [
            "eval",
            "epiforecaster",
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            split,
            "--output",
            str(output_dir),
            *extra_args,
        ],
    )


def test_eval_cli_uses_output_directory_for_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}
    groups = {"Q1 (Best MAE)": [0], "Q4 (Worst MAE)": [1]}
    eval_result = make_eval_result(
        num_forecast_samples=2,
        node_mae={0: 0.1, 1: 0.2},
    )
    install_eval_cli_harness(
        monkeypatch,
        captured,
        eval_result=eval_result,
        node_groups=groups,
    )

    def fake_render_baseline_delta_plots(**kwargs):
        captured["render_baseline_delta_plots_kwargs"] = kwargs
        output_path = kwargs["output_dir"] / "test_baseline_comparison_sarima.png"
        output_path.write_text("plot", encoding="utf-8")
        return {"baseline_comparison_sarima": output_path}

    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_baseline_delta_plots",
        fake_render_baseline_delta_plots,
    )

    result = _invoke_eval(checkpoint_path, output_dir)

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_output_path"] is None
    assert eval_kwargs["node_metrics_target"] == "hospitalizations"
    assert "output.write_granular_eval=false" in eval_kwargs["overrides"]

    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["output_path"] == output_dir / "test_forecasts.png"
    per_head_kwargs = captured["render_eval_per_head_plots_kwargs"]
    assert per_head_kwargs["per_head_node_metrics_csv"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert per_head_kwargs["output_dir"] == output_dir
    assert per_head_kwargs["model"] is eval_result["model"]
    assert per_head_kwargs["config"] is eval_result["config"]
    assert per_head_kwargs["loader"] is eval_result["loader"]


def test_eval_cli_creates_missing_output_directory_for_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "fresh_eval_outputs"
    captured: dict[str, object] = {}
    install_eval_cli_harness(monkeypatch, captured)

    result = _invoke_eval(checkpoint_path, output_dir)

    assert result.exit_code == 0, result.output
    assert output_dir.is_dir()
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_output_path"] is None


def test_eval_cli_granular_flag_enables_auto_granular_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}
    install_eval_cli_harness(monkeypatch, captured)

    result = _invoke_eval(checkpoint_path, output_dir, "--granular")

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "test_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "test_node_metrics_per_head.csv"
    )
    assert eval_kwargs["granular_output_path"] == output_dir / "test_granular.parquet"
    assert eval_kwargs["granular_eval_format"] is None
    assert "output.write_granular_eval=true" in eval_kwargs["overrides"]


def test_eval_cli_relevant_latent_toggle_enables_granular_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}
    install_eval_cli_harness(monkeypatch, captured)

    result = _invoke_eval(
        checkpoint_path,
        output_dir,
        "--include-relevant-latent",
    )

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["granular_output_path"] == (
        output_dir / "test_granular.parquet"
    )
    assert "output.write_granular_eval=true" in eval_kwargs["overrides"]
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["include_relevant_latent"] is True
    assert plot_kwargs["granular_eval_path"] == (
        output_dir / "test_granular.parquet"
    )


def test_eval_cli_writes_machine_readable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    summary_path = output_dir / "test_summary.json"
    captured: dict[str, object] = {}
    eval_result = make_eval_result()
    eval_result["eval_loss"] = float("nan")
    eval_result["eval_metrics"]["mae_by_horizon"] = [0.2, float("nan")]
    eval_result["eval_metrics"]["nested"] = {"unsupported": float("inf")}
    install_eval_cli_harness(monkeypatch, captured, eval_result=eval_result)

    result = _invoke_eval(
        checkpoint_path,
        output_dir,
        "--summary-json",
        str(summary_path),
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(summary_path.read_text())
    assert summary == {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "split": "test",
        "eval_loss": None,
        "metrics": {
            "mae": 0.1,
            "mae_cases_log1p_per_100k": 0.4,
            "mae_deaths_log1p_per_100k": 0.5,
            "mae_hosp_log1p_per_100k": 0.2,
            "mae_by_horizon": [0.2, None],
            "mae_ww_log1p_per_100k": 0.3,
            "nested": {"unsupported": None},
        },
    }


def test_eval_cli_full_split_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}
    install_eval_cli_harness(monkeypatch, captured)

    result = _invoke_eval(checkpoint_path, output_dir, split="full")

    assert result.exit_code == 0, result.output
    eval_kwargs = captured["eval_checkpoint_kwargs"]
    assert eval_kwargs["split"] == "full"
    assert eval_kwargs["node_metrics_csv_path"] == output_dir / "full_node_metrics.csv"
    assert eval_kwargs["per_head_node_metrics_csv_path"] == (
        output_dir / "full_node_metrics_per_head.csv"
    )


def test_eval_cli_compare_evals_runs_fresh_baselines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path, nested=True)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}
    install_eval_cli_harness(
        monkeypatch,
        captured,
        eval_result=make_eval_result(num_forecast_samples=2, node_mae={0: 0.1, 1: 0.2}),
    )

    def fake_render_baseline_delta_plots(**kwargs):
        captured["render_baseline_delta_plots_kwargs"] = kwargs
        output_path = kwargs["output_dir"] / "test_baseline_comparison_sarima.png"
        output_path.write_text("plot", encoding="utf-8")
        return {"baseline_comparison_sarima": output_path}

    def fake_run_same_slice_baseline_evaluation(**kwargs):
        captured["run_same_slice_baseline_evaluation_kwargs"] = kwargs
        baseline_dir = kwargs["output_dir"]
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_csv = baseline_dir / "baseline_aggregate_metrics.csv"
        baseline_csv.write_text(
            "model,target,mae_mean,rmse_mean,r2_mean\n", encoding="utf-8"
        )
        metadata = baseline_dir / "baseline_metadata.json"
        metadata.write_text('{"comparison_scope":"same_eval_slice"}', encoding="utf-8")
        return {
            "baseline_aggregate_metrics": baseline_csv,
            "baseline_metadata": metadata,
        }

    def fake_compare_model_metrics_against_baselines(**kwargs):
        captured["compare_kwargs"] = kwargs
        kwargs["output_csv"].write_text(
            "target,baseline_model,metric,model_value,baseline_value\n"
            "cases,sarima,mae,0.3,0.4\n"
            "cases,sarima,r2,0.2,0.1\n",
            encoding="utf-8",
        )
        return kwargs["output_csv"]

    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_baseline_delta_plots",
        fake_render_baseline_delta_plots,
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.run_same_slice_baseline_evaluation",
        fake_run_same_slice_baseline_evaluation,
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.compare_model_metrics_against_baselines",
        fake_compare_model_metrics_against_baselines,
    )

    result = _invoke_eval(
        checkpoint_path,
        output_dir,
        "--granular",
        "--compare-evals",
        "--compare-baselines",
        str(tmp_path / "ignored.csv"),
    )

    assert result.exit_code == 0, result.output
    baseline_kwargs = captured["run_same_slice_baseline_evaluation_kwargs"]
    assert baseline_kwargs["models"] == [
        "exp_smoothing",
        "last_observed",
        "sarima",
        "sarimax",
        "varmax",
    ]
    assert baseline_kwargs["split"] == "test"
    assert baseline_kwargs["output_dir"] == (
        output_dir / "test_baseline_eval_same_window"
    )
    compare_kwargs = captured["compare_kwargs"]
    assert compare_kwargs["baseline_results_csv"] == (
        output_dir / "test_baseline_eval_same_window" / "baseline_aggregate_metrics.csv"
    )
    baseline_plot_kwargs = captured["render_baseline_delta_plots_kwargs"]
    assert baseline_plot_kwargs["baseline_deltas_csv"] == (
        output_dir / "test_baseline_deltas.csv"
    )
    assert baseline_plot_kwargs["output_dir"] == output_dir
    assert (
        "--compare-evals supplied; ignoring explicit --compare-baselines path"
        in result.output
    )


def test_eval_cli_compare_evals_rejects_full_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    install_eval_cli_harness(monkeypatch, {})

    result = _invoke_eval(checkpoint_path, output_dir, "--compare-evals", split="full")

    assert result.exit_code != 0
    assert "--compare-evals is not supported with --split full" in result.output


def test_eval_cli_granular_window_selection_uses_generated_granular_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "eval_outputs"
    output_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_eval_checkpoint(**kwargs):
        kwargs["granular_output_path"].write_text(
            "split,target,node_id,window_start,abs_error\n", encoding="utf-8"
        )
        return make_eval_result(num_forecast_samples=2)

    install_eval_cli_harness(monkeypatch, captured, eval_callback=fake_eval_checkpoint)

    def fake_load_window_selection_specs_from_granular(**kwargs):
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

    def fake_select_windows_by_loss(**kwargs):
        captured["select_windows_kwargs"] = kwargs
        return {"Q1 (Best MAE)": kwargs["window_specs"]}

    def fake_generate_forecast_plots(**kwargs):
        captured["generate_forecast_plots_kwargs"] = kwargs
        return {
            "selected_nodes": [7],
            "node_groups": {},
            "window_groups": kwargs["window_groups"],
        }

    monkeypatch.setattr(
        "evaluation.selection.load_window_selection_specs_from_granular",
        fake_load_window_selection_specs_from_granular,
    )
    monkeypatch.setattr(
        "evaluation.selection.select_windows_by_loss", fake_select_windows_by_loss
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        fake_generate_forecast_plots,
    )
    monkeypatch.setattr(
        "evaluation.baseline_eval.compare_model_metrics_against_baselines",
        lambda **kwargs: kwargs["output_csv"],
    )

    result = _invoke_eval(
        checkpoint_path,
        output_dir,
        "--granular",
        "--selection-mode",
        "granular_window_quartile",
    )

    assert result.exit_code == 0, result.output
    assert captured["load_window_specs_kwargs"]["granular_csv"] == (
        output_dir / "test_granular.parquet"
    )
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["node_groups"] is None
    assert plot_kwargs["window_groups"]["Q1 (Best MAE)"][0].window_start == 11


@pytest.mark.parametrize(
    ("node_selector", "selection_function", "group_name", "include_latents"),
    [
        (
            "window_quartile:1",
            "select_windows_by_loss",
            "Q1 (Best MAE)",
            False,
        ),
        ("window_worst:1", "select_worst_windows_by_loss", "Worst MAE", True),
    ],
)
def test_plot_forecasts_uses_granular_window_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_selector: str,
    selection_function: str,
    group_name: str,
    include_latents: bool,
) -> None:
    checkpoint_path = make_checkpoint(tmp_path, nested=True)
    granular_output = checkpoint_path.parent.parent / "test_granular.parquet"
    granular_output.write_text(
        "split,target,node_id,window_start,abs_error\n", encoding="utf-8"
    )
    captured: dict[str, object] = {}
    install_plot_forecast_harness(
        monkeypatch,
        captured,
        selection_function=selection_function,
        group_name=group_name,
    )
    args = [
        "plot",
        "forecasts",
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "test",
        "--nodes",
        node_selector,
    ]
    if include_latents:
        args.append("--include-sird-latents")

    result = CliRunner().invoke(cli_module.cli, args)

    assert result.exit_code == 0, result.output
    plot_kwargs = captured["generate_forecast_plots_kwargs"]
    assert plot_kwargs["node_groups"] is None
    assert plot_kwargs["window_groups"][group_name][0].node_id == 3
    if include_latents:
        assert plot_kwargs["include_sird_latents"] is True
