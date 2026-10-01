from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from evaluation.selection import WindowSelectionSpec


EVAL_METRICS = {
    "mae": 0.1,
    "mae_hosp_log1p_per_100k": 0.2,
    "mae_ww_log1p_per_100k": 0.3,
    "mae_cases_log1p_per_100k": 0.4,
    "mae_deaths_log1p_per_100k": 0.5,
}


def make_eval_result(
    *,
    num_forecast_samples: int = 1,
    node_mae: dict[int, float] | None = None,
) -> dict[str, object]:
    return {
        "config": SimpleNamespace(
            training=SimpleNamespace(num_forecast_samples=num_forecast_samples)
        ),
        "model": object(),
        "loader": object(),
        "node_mae": {"hospitalizations": node_mae or {0: 0.1}},
        "eval_loss": 0.5,
        "eval_metrics": EVAL_METRICS.copy(),
    }


def make_checkpoint(tmp_path: Path, *, nested: bool = False) -> Path:
    parent = tmp_path / "run" / "checkpoints" if nested else tmp_path
    parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = parent / "best_model.pt"
    checkpoint_path.write_bytes(b"dummy")
    return checkpoint_path


def install_eval_cli_harness(
    monkeypatch: Any,
    captured: dict[str, object],
    *,
    eval_callback: Any | None = None,
    eval_result: dict[str, object] | None = None,
    node_groups: dict[str, list[int]] | None = None,
) -> None:
    groups = node_groups or {"Q1 (Best MAE)": [0]}

    def fake_eval_checkpoint(**kwargs: object) -> dict[str, object]:
        captured["eval_checkpoint_kwargs"] = kwargs
        if eval_callback is not None:
            return eval_callback(**kwargs)
        return eval_result or make_eval_result()

    def fake_select_nodes_by_loss(**kwargs: object) -> dict[str, list[int]]:
        captured["select_nodes_kwargs"] = kwargs
        return groups

    def fake_generate_forecast_plots(**kwargs: object) -> dict[str, object]:
        captured["generate_forecast_plots_kwargs"] = kwargs
        return {
            "selected_nodes": [node for nodes in groups.values() for node in nodes],
            "node_groups": groups,
        }

    def fake_render_eval_per_head_plots(**kwargs: object) -> dict[str, Path]:
        captured["render_eval_per_head_plots_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path", lambda _path: None
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.eval_checkpoint", fake_eval_checkpoint
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.select_nodes_by_loss", fake_select_nodes_by_loss
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        fake_generate_forecast_plots,
    )
    monkeypatch.setattr(
        "dataviz.eval_head_plots.render_eval_per_head_plots",
        fake_render_eval_per_head_plots,
    )


def install_plot_forecast_harness(
    monkeypatch: Any,
    captured: dict[str, object],
    *,
    selection_function: str,
    group_name: str,
) -> None:
    import torch

    monkeypatch.setattr(
        "utils.run_discovery.extract_run_from_checkpoint_path", lambda _path: None
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.load_model_from_checkpoint",
        lambda checkpoint, device="auto", overrides=None: (
            torch.nn.Linear(1, 1),
            SimpleNamespace(
                output=SimpleNamespace(
                    resolve_granular_eval_filename=lambda split: (
                        f"{split}_granular.parquet"
                    )
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
                    {"_valid_window_starts_by_node": {}, "__len__": lambda self: 0},
                )()
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        "evaluation.selection.load_window_selection_specs_from_granular",
        lambda **kwargs: [WindowSelectionSpec(3, 9, 0.2, ("cases",), 2)],
    )
    monkeypatch.setattr(
        f"evaluation.selection.{selection_function}",
        lambda **kwargs: {group_name: kwargs["window_specs"]},
    )
    monkeypatch.setattr(
        "evaluation.epiforecaster_eval.generate_forecast_plots",
        lambda **kwargs: captured.setdefault("generate_forecast_plots_kwargs", kwargs),
    )


def make_granular_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "test",
        "target": "cases",
        "node_id": 1,
        "region_id": "08001",
        "region_label": "region-a",
        "window_start": 0,
        "window_start_date": "2024-01-01",
        "horizon": 1,
        "target_index": 3,
        "target_date": "2024-01-04",
        "observed": 10.0,
        "abs_error": 2.0,
        "sq_error": 4.0,
    }
    row.update(overrides)
    return row


def write_granular_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
