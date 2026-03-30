from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from dataviz.eval_head_plots import (
    _compute_canonical_sparsity,
    _select_representative_window_specs_for_target,
    render_baseline_delta_plots,
    render_eval_per_head_plots,
)
from dataviz.granular_comparison import _format_target_label
from evaluation.selection import WindowSelectionSpec
from evaluation.selection import select_nodes_by_loss


def _write_test_dataset(path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    regions = np.array(["08001", "08002"])
    run_ids = np.array([0, 1])

    ds = xr.Dataset(
        data_vars={
            "population": ((REGION_COORD,), np.array([1000.0, 10000.0])),
            "cases_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.array(
                    [
                        [[1, 1], [1, 0], [0, 1], [0, 1]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    dtype=np.float32,
                ),
            ),
            "hosp_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.array(
                    [
                        [[1, 1], [1, 1], [0, 0], [0, 0]],
                        [[1, 1], [1, 1], [1, 1], [1, 1]],
                    ],
                    dtype=np.float32,
                ),
            ),
            "deaths_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.array(
                    [
                        [[1, 0], [1, 0], [1, 1], [1, 1]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    dtype=np.float32,
                ),
            ),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.array(
                    [
                        [[1, 0], [1, 0], [1, 0], [1, 1]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    dtype=np.float32,
                ),
            ),
            "edar_biomarker_N2_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.array(
                    [
                        [[0, 0], [0, 1], [0, 0], [0, 1]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "run_id": run_ids,
            TEMPORAL_COORD: dates,
            REGION_COORD: regions,
        },
    )
    ds.to_zarr(path)


def test_compute_canonical_sparsity_uses_run_filter_and_combined_ww_rule(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    sparsity_df = _compute_canonical_sparsity(dataset_path=dataset_path, run_id=0)

    cases_region_a = sparsity_df[
        (sparsity_df["target"] == "cases") & (sparsity_df["region_id"] == "08001")
    ].iloc[0]
    ww_region_a = sparsity_df[
        (sparsity_df["target"] == "wastewater") & (sparsity_df["region_id"] == "08001")
    ].iloc[0]
    ww_region_b = sparsity_df[
        (sparsity_df["target"] == "wastewater") & (sparsity_df["region_id"] == "08002")
    ].iloc[0]

    assert cases_region_a["sparsity_pct"] == pytest.approx(50.0)
    assert ww_region_a["sparsity_pct"] == pytest.approx(0.0)
    assert ww_region_b["sparsity_pct"] == pytest.approx(50.0)


def test_render_eval_per_head_plots_writes_expected_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(
        [
            {
                "target": target,
                "node_id": node_id,
                "region_id": region_id,
                "region_label": label,
                "population": population,
                "observed_count": observed_count,
                "mae": mae,
                "rmse": rmse,
            }
            for target, node_id, region_id, label, population, observed_count, mae, rmse in [
                ("hospitalizations", 0, "08001", "Region A", 1000.0, 4, 0.3, 0.4),
                ("hospitalizations", 1, "08002", "Region B", 10000.0, 4, 0.6, 0.7),
                ("wastewater", 0, "08001", "Region A", 1000.0, 4, 0.2, 0.3),
                ("wastewater", 1, "08002", "Region B", 10000.0, 2, 0.5, 0.6),
                ("cases", 0, "08001", "Region A", 1000.0, 2, 0.7, 0.8),
                ("cases", 1, "08002", "Region B", 10000.0, 3, 0.4, 0.5),
                ("deaths", 0, "08001", "Region A", 1000.0, 3, 0.1, 0.2),
                ("deaths", 1, "08002", "Region B", 10000.0, 2, 0.9, 1.0),
            ]
        ]
    ).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )

    artifacts = render_eval_per_head_plots(per_head_node_metrics_csv=csv_path)

    expected_names = {
        "perf_vs_population_hospitalizations",
        "perf_vs_population_wastewater",
        "perf_vs_population_cases",
        "perf_vs_population_deaths",
        "perf_vs_sparsity_hospitalizations",
        "perf_vs_sparsity_wastewater",
        "perf_vs_sparsity_cases",
        "perf_vs_sparsity_deaths",
    }
    assert set(artifacts) == expected_names
    for path in artifacts.values():
        assert path.exists()


def test_render_baseline_delta_plots_writes_expected_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_baseline_deltas.csv"
    pd.DataFrame(
        [
            {
                "target": "cases",
                "baseline_model": "sarima",
                "metric": "mae",
                "model_value": 0.30,
                "baseline_value": 0.40,
            },
            {
                "target": "cases",
                "baseline_model": "sarima",
                "metric": "r2",
                "model_value": 0.20,
                "baseline_value": 0.10,
            },
            {
                "target": "hospitalizations",
                "baseline_model": "sarima",
                "metric": "mae",
                "model_value": 0.25,
                "baseline_value": 0.35,
            },
            {
                "target": "hospitalizations",
                "baseline_model": "sarima",
                "metric": "r2",
                "model_value": 0.40,
                "baseline_value": 0.22,
            },
            {
                "target": "joint_observation",
                "baseline_model": "sarima",
                "metric": "mae",
                "model_value": 1.0,
                "baseline_value": 2.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    artifacts = render_baseline_delta_plots(baseline_deltas_csv=csv_path)

    assert set(artifacts) == {"baseline_comparison_mae", "baseline_comparison_r2"}
    assert artifacts["baseline_comparison_mae"].exists()
    assert artifacts["baseline_comparison_r2"].exists()


def test_render_baseline_delta_plots_groups_all_baselines_by_metric(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "test_baseline_deltas.csv"
    pd.DataFrame(
        [
            {
                "target": "cases",
                "baseline_model": "sarima",
                "metric": "mae",
                "model_value": 0.30,
                "baseline_value": 0.40,
            },
            {
                "target": "cases",
                "baseline_model": "sarima",
                "metric": "r2",
                "model_value": 0.20,
                "baseline_value": 0.10,
            },
            {
                "target": "cases",
                "baseline_model": "exp_smoothing",
                "metric": "mae",
                "model_value": 0.30,
                "baseline_value": 0.45,
            },
            {
                "target": "cases",
                "baseline_model": "exp_smoothing",
                "metric": "r2",
                "model_value": 0.20,
                "baseline_value": 0.05,
            },
            {
                "target": "hospitalizations",
                "baseline_model": "var",
                "metric": "mae",
                "model_value": 0.25,
                "baseline_value": 0.32,
            },
            {
                "target": "hospitalizations",
                "baseline_model": "var",
                "metric": "r2",
                "model_value": 0.30,
                "baseline_value": 0.18,
            },
        ]
    ).to_csv(csv_path, index=False)

    artifacts = render_baseline_delta_plots(baseline_deltas_csv=csv_path)

    assert set(artifacts) == {
        "baseline_comparison_mae",
        "baseline_comparison_r2",
    }
    for path in artifacts.values():
        assert path.exists()


def test_select_nodes_by_loss_quartiles_rank_low_mae_as_best() -> None:
    quartiles = select_nodes_by_loss(
        node_mae={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        target_name="hospitalizations",
        strategy="quartile",
        samples_per_group=1,
        rng=np.random.default_rng(0),
    )

    assert list(quartiles) == [
        "Q1 (Best MAE)",
        "Q2 (Good MAE)",
        "Q3 (Poor MAE)",
        "Q4 (Worst MAE)",
    ]
    assert quartiles["Q1 (Best MAE)"] == [0]
    assert quartiles["Q4 (Worst MAE)"] == [3]


def test_select_nodes_by_loss_supports_non_hospitalization_targets() -> None:
    best = select_nodes_by_loss(
        node_mae={7: 0.4, 8: 0.1, 9: 0.2},
        target_name="cases",
        strategy="best",
        k=2,
    )

    assert best == {"Best": [8, 9]}


def test_render_eval_per_head_plots_adds_forecast_example_artifacts_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(
        [
            {
                "target": "hospitalizations",
                "node_id": 0,
                "region_id": "08001",
                "region_label": "Region A",
                "population": 1000.0,
                "observed_count": 4,
                "mae": 0.3,
                "rmse": 0.4,
            },
            {
                "target": "hospitalizations",
                "node_id": 1,
                "region_id": "08002",
                "region_label": "Region B",
                "population": 10000.0,
                "observed_count": 4,
                "mae": 0.6,
                "rmse": 0.7,
            },
        ]
    ).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "target": "hospitalizations",
                "node_id": 0,
                "window_start": 5,
                "abs_error": 0.1,
            },
            {
                "split": "test",
                "target": "hospitalizations",
                "node_id": 0,
                "window_start": 5,
                "abs_error": 0.3,
            },
            {
                "split": "test",
                "target": "hospitalizations",
                "node_id": 1,
                "window_start": 6,
                "abs_error": 0.4,
            },
            {
                "split": "test",
                "target": "hospitalizations",
                "node_id": 1,
                "window_start": 6,
                "abs_error": 0.8,
            },
        ]
    ).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        pass

    class _DummyLoader:
        pass

    def _fake_load_model_from_checkpoint(*args, **kwargs):
        return _DummyModel(), _DummyConfig(), {}

    def _fake_build_loader_from_config(*args, **kwargs):
        return _DummyLoader(), None

    def _fake_collect_forecast_samples_for_window_specs(**kwargs):
        return [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "hosp": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                    }
                },
            }
            for spec in kwargs["window_specs"]
        ]

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        _fake_load_model_from_checkpoint,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        _fake_build_loader_from_config,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        _fake_collect_forecast_samples_for_window_specs,
    )

    artifacts = render_eval_per_head_plots(
        per_head_node_metrics_csv=csv_path,
        samples_per_quartile=1,
    )

    forecast_key = "forecast_examples_quartiles_hospitalizations"
    assert forecast_key in artifacts
    assert artifacts[forecast_key].exists()


def test_render_eval_per_head_plots_uses_four_examples_per_quartile_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    rows: list[dict[str, object]] = []
    for node_id in range(16):
        rows.append(
            {
                "target": "hospitalizations",
                "node_id": node_id,
                "region_id": f"{8000 + node_id:05d}",
                "region_label": f"Region {node_id}",
                "population": 1000.0 + node_id,
                "observed_count": 4,
                "mae": 0.1 + (node_id * 0.1),
                "rmse": 0.2 + (node_id * 0.1),
            }
        )

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    granular_rows: list[dict[str, object]] = []
    for node_id in range(16):
        granular_rows.extend(
            [
                {
                    "split": "test",
                    "target": "hospitalizations",
                    "node_id": node_id,
                    "window_start": 100 + node_id,
                    "abs_error": 0.1 + (node_id * 0.1),
                },
                {
                    "split": "test",
                    "target": "hospitalizations",
                    "node_id": node_id,
                    "window_start": 100 + node_id,
                    "abs_error": 0.1 + (node_id * 0.1),
                },
            ]
        )
    pd.DataFrame(granular_rows).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        pass

    class _DummyLoader:
        pass

    captured_window_specs: list[WindowSelectionSpec] = []
    captured_figures: list[dict[str, object]] = []

    def _fake_load_model_from_checkpoint(*args, **kwargs):
        return _DummyModel(), _DummyConfig(), {}

    def _fake_build_loader_from_config(*args, **kwargs):
        return _DummyLoader(), None

    def _fake_collect_forecast_samples_for_window_specs(**kwargs):
        captured_window_specs.extend(kwargs["window_specs"])
        return [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "hosp": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                    }
                },
            }
            for spec in kwargs["window_specs"]
        ]

    def _fake_make_forecast_figure(**kwargs):
        captured_figures.append(kwargs)
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        return fig

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        _fake_load_model_from_checkpoint,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        _fake_build_loader_from_config,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        _fake_collect_forecast_samples_for_window_specs,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.make_forecast_figure",
        _fake_make_forecast_figure,
    )

    artifacts = render_eval_per_head_plots(per_head_node_metrics_csv=csv_path)

    forecast_key = "forecast_examples_quartiles_hospitalizations"
    assert forecast_key in artifacts
    assert artifacts[forecast_key].exists()
    assert len(captured_window_specs) == 16
    assert len({spec.node_id for spec in captured_window_specs}) == 16
    assert len(captured_figures) == 1
    grouped_samples = captured_figures[0]["samples"]
    assert list(grouped_samples) == [
        "Q1 (Best MAE)",
        "Q2 (Good MAE)",
        "Q3 (Poor MAE)",
        "Q4 (Worst MAE)",
    ]
    assert all(len(samples) == 4 for samples in grouped_samples.values())
    assert captured_figures[0]["figure_title"] == f"{_format_target_label('hospitalizations')} (MAE)"
    assert (
        captured_figures[0]["shared_xlabel"]
        == "Time (days relative to forecast start)"
    )


def test_render_eval_per_head_plots_emits_latent_forecast_artifacts_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(
        [
            {
                "target": "hospitalizations",
                "node_id": node_id,
                "region_id": f"{8000 + node_id:05d}",
                "region_label": f"Region {node_id}",
                "population": 1000.0 + node_id,
                "observed_count": 4,
                "mae": 0.1 + (node_id * 0.1),
                "rmse": 0.2 + (node_id * 0.1),
            }
            for node_id in range(4)
        ]
    ).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "target": "hospitalizations",
                "node_id": node_id,
                "window_start": 100 + node_id,
                "abs_error": 0.1 + (node_id * 0.1),
            }
            for node_id in range(4)
        ]
    ).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        pass

    class _DummyLoader:
        pass

    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        lambda *args, **kwargs: (_DummyModel(), _DummyConfig(), {}),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        lambda *args, **kwargs: (_DummyLoader(), None),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        lambda **kwargs: [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "hosp": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                    }
                },
                "latents": {
                    "latent_s": {
                        "actual_context": np.array(
                            [np.nan, np.nan, 0.7, 0.6, np.nan], dtype=np.float32
                        ),
                        "prediction": np.array([0.7, 0.6], dtype=np.float32),
                        "target": np.array([np.nan, np.nan], dtype=np.float32),
                        "history": np.array([], dtype=np.float32),
                    }
                },
            }
            for spec in kwargs["window_specs"]
        ],
    )

    def _fake_make_forecast_figure(**kwargs):
        captured_calls.append(kwargs)
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        return fig

    monkeypatch.setattr(
        "dataviz.eval_head_plots.make_forecast_figure",
        _fake_make_forecast_figure,
    )

    artifacts = render_eval_per_head_plots(
        per_head_node_metrics_csv=csv_path,
        samples_per_quartile=1,
    )

    assert "forecast_examples_quartiles_hospitalizations" in artifacts
    assert len(captured_calls) == 1
    assert captured_calls[0]["overlay_target"] == "latent_i"
    assert captured_calls[0]["overlay_label"] == "Latent I"


def test_render_eval_per_head_plots_emits_joint_latent_quartile_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(
        [
            {
                "target": "hospitalizations",
                "node_id": node_id,
                "region_id": f"{8000 + node_id:05d}",
                "region_label": f"Region {node_id}",
                "population": 1000.0 + node_id,
                "observed_count": 4,
                "mae": 0.1 + (node_id * 0.1),
                "rmse": 0.2 + (node_id * 0.1),
            }
            for node_id in range(4)
        ]
    ).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "target": target,
                "node_id": node_id,
                "window_start": 100 + node_id,
                "abs_error": 0.1 + (node_id * 0.1),
            }
            for node_id in range(4)
            for target in ["cases", "hospitalizations", "deaths", "wastewater"]
        ]
    ).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        pass

    class _DummyLoader:
        pass

    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        lambda *args, **kwargs: (_DummyModel(), _DummyConfig(), {}),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        lambda *args, **kwargs: (_DummyLoader(), None),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        lambda **kwargs: [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "cases": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                    }
                },
                "latents": {
                    "latent_s": {
                        "actual_context": np.array(
                            [np.nan, np.nan, 0.7, 0.6, np.nan], dtype=np.float32
                        ),
                        "prediction": np.array([0.7, 0.6], dtype=np.float32),
                        "target": np.array([np.nan, np.nan], dtype=np.float32),
                        "history": np.array([], dtype=np.float32),
                    },
                    "latent_i": {
                        "actual_context": np.array(
                            [np.nan, np.nan, 0.2, 0.1, np.nan], dtype=np.float32
                        ),
                        "prediction": np.array([0.2, 0.1], dtype=np.float32),
                        "target": np.array([np.nan, np.nan], dtype=np.float32),
                        "history": np.array([], dtype=np.float32),
                    },
                    "latent_r": {
                        "actual_context": np.array(
                            [np.nan, np.nan, 0.05, 0.08, np.nan], dtype=np.float32
                        ),
                        "prediction": np.array([0.05, 0.08], dtype=np.float32),
                        "target": np.array([np.nan, np.nan], dtype=np.float32),
                        "history": np.array([], dtype=np.float32),
                    },
                    "latent_d": {
                        "actual_context": np.array(
                            [np.nan, np.nan, 0.01, 0.02, np.nan], dtype=np.float32
                        ),
                        "prediction": np.array([0.01, 0.02], dtype=np.float32),
                        "target": np.array([np.nan, np.nan], dtype=np.float32),
                        "history": np.array([], dtype=np.float32),
                    },
                },
            }
            for spec in kwargs["window_specs"]
        ],
    )

    def _fake_make_forecast_figure(**kwargs):
        captured_calls.append(kwargs)
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        return fig

    monkeypatch.setattr(
        "dataviz.eval_head_plots.make_forecast_figure",
        _fake_make_forecast_figure,
    )

    artifacts = render_eval_per_head_plots(
        per_head_node_metrics_csv=csv_path,
        samples_per_quartile=1,
    )

    assert "forecast_examples_quartiles_joint_latent_s" in artifacts
    assert "forecast_examples_quartiles_joint_latent_i" in artifacts
    assert "forecast_examples_quartiles_joint_latent_r" in artifacts
    assert "forecast_examples_quartiles_joint_latent_d" in artifacts
    joint_latent_calls = [
        call
        for call in captured_calls
        if call.get("payload_collection") == "latents"
    ]
    assert {call["target"] for call in joint_latent_calls} == {
        "latent_s",
        "latent_i",
        "latent_r",
        "latent_d",
    }


def test_select_representative_window_specs_for_target_uses_median_and_tiebreaks() -> None:
    target_df = pd.DataFrame(
        [
            {"target": "cases", "node_id": 7, "mae": 0.5},
        ]
    )
    granular_df = pd.DataFrame(
        [
            {"target": "cases", "node_id": 7, "window_start": 12, "abs_error": 0.2},
            {"target": "cases", "node_id": 7, "window_start": 12, "abs_error": 0.4},
            {"target": "cases", "node_id": 7, "window_start": 15, "abs_error": 0.2},
            {"target": "cases", "node_id": 7, "window_start": 15, "abs_error": 0.6},
            {"target": "cases", "node_id": 7, "window_start": 15, "abs_error": 0.4},
            {"target": "cases", "node_id": 7, "window_start": 19, "abs_error": 0.5},
            {"target": "cases", "node_id": 7, "window_start": 19, "abs_error": 0.5},
        ]
    )

    grouped = _select_representative_window_specs_for_target(
        target_df=target_df,
        target_name="cases",
        granular_df=granular_df,
        samples_per_quartile=1,
    )

    assert list(grouped) == ["Q1 (Best MAE)"]
    spec = grouped["Q1 (Best MAE)"][0]
    assert spec.node_id == 7
    assert spec.window_start == 15
    assert spec.score == pytest.approx(0.4)
    assert spec.observed_points == 3


def test_render_eval_per_head_plots_backfills_same_quartile_with_eligible_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    rows: list[dict[str, object]] = []
    for node_id, mae in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], start=0):
        rows.append(
            {
                "target": "cases",
                "node_id": node_id,
                "region_id": f"{8000 + node_id:05d}",
                "region_label": f"Region {node_id}",
                "population": 1000.0 + node_id,
                "observed_count": 4,
                "mae": mae,
                "rmse": mae + 0.1,
            }
        )

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 20, "abs_error": 0.2},
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 20, "abs_error": 0.2},
            {"split": "test", "target": "cases", "node_id": 3, "window_start": 21, "abs_error": 0.3},
            {"split": "test", "target": "cases", "node_id": 3, "window_start": 21, "abs_error": 0.3},
        ]
    ).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        def __init__(self) -> None:
            self.training = False

        def parameters(self):
            import torch

            yield torch.nn.Parameter(torch.tensor(0.0))

    class _DummyLoader:
        pass

    captured_figures: list[dict[str, list[dict[str, object]]]] = []

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        lambda *args, **kwargs: (_DummyModel(), _DummyConfig(), {}),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        lambda *args, **kwargs: (_DummyLoader(), None),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        lambda **kwargs: [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "cases": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                        "window_mae": float(spec.score),
                    }
                },
                "window_mae": float(spec.score),
            }
            for spec in kwargs["window_specs"]
        ],
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.make_forecast_figure",
        lambda **kwargs: captured_figures.append(kwargs["samples"]) or __import__("matplotlib.pyplot").pyplot.figure(),
    )

    artifacts = render_eval_per_head_plots(
        per_head_node_metrics_csv=csv_path,
        samples_per_quartile=1,
    )

    assert "forecast_examples_quartiles_cases" in artifacts
    grouped_samples = captured_figures[0]
    assert grouped_samples["Q1 (Best MAE)"][0]["node_id"] == 1
    assert grouped_samples["Q2 (Good MAE)"][0]["node_id"] == 3


def test_render_eval_per_head_plots_regression_uses_non_last_representative_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    _write_test_dataset(dataset_path)

    csv_path = tmp_path / "test_node_metrics_per_head.csv"
    pd.DataFrame(
        [
            {
                "target": "cases",
                "node_id": node_id,
                "region_id": f"{8000 + node_id:05d}",
                "region_label": f"Region {node_id}",
                "population": 1000.0,
                "observed_count": 4,
                "mae": mae,
                "rmse": mae + 0.1,
            }
            for node_id, mae in [
                (0, 0.1),
                (1, 0.2),
                (2, 0.3),
                (3, 0.4),
                (4, 0.8),
                (5, 0.9),
                (6, 1.0),
                (7, 1.1),
            ]
        ]
    ).to_csv(csv_path, index=False)
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(tmp_path / "checkpoint.ckpt"),
                "dataset_path": str(dataset_path),
                "run_id": 0,
                "schema_version": "1",
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"split": "test", "target": "cases", "node_id": 0, "window_start": 10, "abs_error": 0.1},
            {"split": "test", "target": "cases", "node_id": 0, "window_start": 10, "abs_error": 0.3},
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 11, "abs_error": 0.2},
            {"split": "test", "target": "cases", "node_id": 1, "window_start": 11, "abs_error": 0.2},
            {"split": "test", "target": "cases", "node_id": 2, "window_start": 12, "abs_error": 0.3},
            {"split": "test", "target": "cases", "node_id": 2, "window_start": 12, "abs_error": 0.3},
        ]
    ).to_csv(tmp_path / "test_granular.csv", index=False)

    class _DummyModelConfig:
        input_window_length = 3
        forecast_horizon = 2

    class _DummyConfig:
        model = _DummyModelConfig()

    class _DummyModel:
        def __init__(self) -> None:
            self.training = False

        def parameters(self):
            import torch

            yield torch.nn.Parameter(torch.tensor(0.0))

    class _DummyLoader:
        pass

    captured_figures: list[dict[str, list[dict[str, object]]]] = []

    monkeypatch.setattr(
        "dataviz.eval_head_plots.load_model_from_checkpoint",
        lambda *args, **kwargs: (_DummyModel(), _DummyConfig(), {}),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.build_loader_from_config",
        lambda *args, **kwargs: (_DummyLoader(), None),
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.collect_forecast_samples_for_window_specs",
        lambda **kwargs: [
            {
                "node_id": spec.node_id,
                "window_start": spec.window_start,
                "node_label": f"Region {spec.node_id}",
                "t_rel": np.arange(-2, 3),
                "H": 2,
                "targets": {
                    "cases": {
                        "actual_context": np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        "prediction": np.array([4, 5], dtype=np.float32),
                        "target": np.array([4, 5], dtype=np.float32),
                        "history": np.array([1, 2, 3], dtype=np.float32),
                        "window_mae": float(spec.score),
                    }
                },
                "window_mae": float(spec.score),
            }
            for spec in kwargs["window_specs"]
        ],
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.make_forecast_figure",
        lambda **kwargs: captured_figures.append(kwargs["samples"]) or __import__("matplotlib.pyplot").pyplot.figure(),
    )

    render_eval_per_head_plots(
        per_head_node_metrics_csv=csv_path,
        samples_per_quartile=2,
    )

    grouped_samples = captured_figures[0]
    assert [sample["node_id"] for sample in grouped_samples["Q1 (Best MAE)"]] == [0, 1]
    assert [sample["window_start"] for sample in grouped_samples["Q1 (Best MAE)"]] == [10, 11]
