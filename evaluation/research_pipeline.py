"""Canonical research artifact generation from observed granular forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset_factory import split_nodes_by_ratio
from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from data.research_roles import validate_preprocessing_role
from evaluation.research_calibration import (
    fit_research_calibrators,
    score_research_calibrators,
)
from evaluation.research_events import score_research_events
from models.configs import EpiForecasterConfig


RESEARCH_ARTIFACTS = (
    "research_fit_cases.parquet",
    "research_calibration.json",
    "research_scored.parquet",
    "research_summary.json",
    "research_events.parquet",
    "research_events.json",
)


def load_fit_cases(config: EpiForecasterConfig, manifest: dict) -> pd.DataFrame:
    """Read unique fit-role case labels from the canonical data, without a model."""
    dataset = EpiDataset.load_canonical_dataset(
        Path(config.data.dataset_path),
        run_id=config.data.run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    try:
        validate_preprocessing_role(dataset.attrs, manifest)
        if not dataset.attrs.get("log_transformed") or not dataset.attrs.get(
            "population_norm"
        ):
            raise ValueError(
                "Case event fitting requires canonical log1p(per-100k) targets"
            )
        if config.training.crossval_enabled:
            nodes, _, _ = split_nodes_by_ratio(config=config)
        else:
            nodes = list(range(dataset.sizes[REGION_COORD]))
        bounds = manifest["roles"]["fit"]
        subset = (
            dataset[["cases", "cases_mask"]]
            .isel({REGION_COORD: nodes})
            .sel({TEMPORAL_COORD: slice(bounds["label_start"], bounds["label_end"])})
        )
        if "run_id" in subset.dims:
            subset = subset.squeeze("run_id", drop=True)
        values = subset.cases.transpose(TEMPORAL_COORD, REGION_COORD).values.astype(
            np.float32
        )
        mask = subset.cases_mask.transpose(TEMPORAL_COORD, REGION_COORD).values > 0
        # Same eligible observed targets as EpiDataset._load_target_values_and_mask.
        mask &= np.isfinite(values) & (values >= 0)
        time_index, node_index = np.nonzero(mask)
        return pd.DataFrame(
            {
                "role": "fit",
                "target": "cases",
                "source_id": subset[REGION_COORD].values[node_index].astype(str),
                "target_date": pd.DatetimeIndex(
                    subset[TEMPORAL_COORD].values[time_index]
                ).strftime("%Y-%m-%d"),
                "observed": values[time_index, node_index],
                "observed_mask": True,
            }
        )
    finally:
        dataset.close()


def write_research_outputs(
    *,
    config: EpiForecasterConfig,
    manifest: dict,
    checkpoint_sha256: str,
    selection_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Write metric artifacts only after the complete role contract passes."""
    if (
        config.training.split_strategy != "research"
        or config.model.forecast_horizon != 28
    ):
        raise ValueError("Research outputs require the declared 28-day research split")
    if config.training.research_evaluation_role != "calibration":
        raise ValueError("Development research cannot read confirmation labels")
    selection = pd.read_parquet(selection_path).assign(role="selection")
    calibration = pd.read_parquet(calibration_path).assign(role="calibration")
    fit_cases = load_fit_cases(config, manifest)
    artifact = fit_research_calibrators(
        calibration, manifest, checkpoint_sha256=checkpoint_sha256
    )
    scored, summary = score_research_calibrators(
        selection,
        artifact,
        manifest,
        role="selection",
        checkpoint_sha256=checkpoint_sha256,
    )
    events, event_summary = score_research_events(
        fit_cases, calibration, scored, artifact, manifest
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("research_fit_cases.parquet", fit_cases),
        ("research_scored.parquet", scored),
        ("research_events.parquet", events),
    ]:
        frame.to_parquet(output_dir / name, index=False)
    for name, payload in [
        ("research_calibration.json", artifact),
        ("research_summary.json", summary),
        ("research_events.json", event_summary),
    ]:
        (output_dir / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return [output_dir / name for name in RESEARCH_ARTIFACTS]
