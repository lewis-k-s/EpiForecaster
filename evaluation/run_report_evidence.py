"""Canonical report context and example exports, without plotting imports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.native_example_support import (
    eligible_windows,
    load_example_history,
    window_support,
)
from evaluation.native_forecast_examples import select_native_forecast_examples
from models.configs import EpiForecasterConfig


def report_context(config: EpiForecasterConfig) -> dict:
    contract = (
        json.loads(Path(config.native.fold_manifest).read_text())
        if config.native.fold_index is not None
        else {"kind": "temporal_tail"}
    )
    strategy = (
        "temporal"
        if contract.get("kind") in {"rolling_time", "temporal_tail"}
        else "region"
    )
    return {
        "schema_version": 1,
        "native_support": True,
        "split_strategy": strategy,
        "fold_index": config.native.fold_index,
        "split_contract": contract,
        "input_window_length": config.model.input_window_length,
        "forecast_horizon": config.model.forecast_horizon,
        "expected_epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "config": config.to_dict(),
    }


def export_native_report_examples(
    config: EpiForecasterConfig, output_dir: Path, split: str
) -> list[Path]:
    """Persist history, support, selections and availability for report consumers."""
    context = report_context(config)
    rows = pd.read_parquet(output_dir / f"{split}_granular.parquet")
    real_path = config.data.real_dataset_path or config.data.dataset_path
    history = load_example_history(
        Path(real_path).parent,
        rows,
        native_root=Path(config.native.root),
        cases_path=Path(real_path),
        wastewater_filename=config.native.wastewater_filename,
    )
    support = eligible_windows(
        window_support(rows, history, config.model.input_window_length)
    )
    cohort = None
    if context["split_strategy"] == "temporal" and split == "test":
        # A standalone test evaluation does not define a validation cohort.
        path = output_dir / "val_example_selection.csv"
        cohort = (
            pd.read_csv(path, dtype={"source_id": str})
            if path.is_file()
            else pd.DataFrame(
                columns=[
                    "target",
                    "source_type",
                    "source_id",
                    "example_rank",
                    "source_mean_wis",
                ]
            )
        )
    if support.eligible.any():
        selected, selection = select_native_forecast_examples(
            rows,
            input_window_length=config.model.input_window_length,
            support=support,
            source_cohort=cohort,
        )
    else:
        selected = rows.iloc[:0]
        selection = pd.DataFrame(
            columns=[
                "target",
                "source_type",
                "source_id",
                "example_rank",
                "source_mean_wis",
                "window_start",
                "origin_index",
                "observed_points",
            ]
        )
    selection["selection_split"] = (
        "val" if context["split_strategy"] == "temporal" else split
    )
    outputs = []
    for suffix, frame, fmt in (
        ("example_history", history, "parquet"),
        ("example_rows", selected, "parquet"),
        ("window_support", support, "csv"),
        ("example_selection", selection, "csv"),
    ):
        path = output_dir / f"{split}_{suffix}.{fmt}"
        if fmt == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        outputs.append(path)
    counts = []
    for target, group in support.groupby("target", sort=True):
        chosen = selection[selection.target == target]
        reference_count = (
            len(cohort[cohort.target == target]) if cohort is not None else len(chosen)
        )
        counts.append(
            {
                "split": split,
                "target": target,
                "candidate_windows": len(group),
                "eligible_windows": int(group.eligible.sum()),
                "eligible_sources": group[group.eligible].source_id.nunique(),
                "selected_sources": len(chosen),
                "missing_cohort_sources": reference_count - len(chosen),
                "cohort_available": cohort is None or not cohort.empty,
            }
        )
    path = output_dir / f"{split}_example_availability.csv"
    pd.DataFrame(counts).to_csv(path, index=False)
    outputs.append(path)
    return outputs
