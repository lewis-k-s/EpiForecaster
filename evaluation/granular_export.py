from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SMAPE_EPSILON = 1e-6
GRANULAR_SCHEMA_VERSION = "1"
GRANULAR_KEY_COLUMNS = ["split", "target", "node_id", "window_start", "horizon"]
GRANULAR_FIELDNAMES = [
    "split",
    "target",
    "node_id",
    "region_id",
    "region_label",
    "window_start",
    "window_start_date",
    "horizon",
    "target_index",
    "target_date",
    "observed",
    "abs_error",
    "sq_error",
    "smape_num",
    "smape_den",
]

_TARGET_NAMES = {
    "ww": "wastewater",
    "hosp": "hospitalizations",
    "cases": "cases",
    "deaths": "deaths",
}


def _normalize_split_name(split_name: str) -> str:
    return split_name.strip().lower()


def _format_metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.datetime64,)):
        return np.datetime_as_string(value, unit="D")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _format_temporal_coord(
    temporal_coords: Sequence[Any] | None,
    index: int,
) -> str | None:
    if temporal_coords is None or index < 0 or index >= len(temporal_coords):
        return None
    value = temporal_coords[index]
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_granular_metadata_sidecar(
    granular_csv_path: Path,
    metadata: dict[str, Any],
) -> Path:
    sidecar_path = granular_csv_path.with_suffix(
        f"{granular_csv_path.suffix}.meta.json"
    )
    existing: dict[str, Any] = {}
    if sidecar_path.exists():
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
    merged = {**existing, **metadata}
    merged["schema_version"] = GRANULAR_SCHEMA_VERSION
    normalized = {key: _format_metadata_value(value) for key, value in merged.items()}
    sidecar_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


class GranularEvalWriter:
    """Stream granular eval rows directly to CSV during metric accumulation."""

    def __init__(
        self,
        *,
        path: Path,
        split_name: str,
        observed_only: bool = True,
    ) -> None:
        self.path = path
        self.split_name = _normalize_split_name(split_name)
        self.observed_only = observed_only
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=GRANULAR_FIELDNAMES)
        self._writer.writeheader()

    def close(self) -> None:
        self._file.close()

    def write_rows(
        self,
        *,
        batch_data: Any,
        target_name: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        temporal_coords: Sequence[Any] | None,
        region_ids: Sequence[str] | None,
        region_labels: Sequence[str] | None,
    ) -> None:
        canonical_target = _TARGET_NAMES[target_name]
        pred = torch.nan_to_num(
            predictions.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).cpu()
        target = torch.nan_to_num(
            targets.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).cpu()
        observed_weights = torch.nan_to_num(weights.detach().float(), nan=0.0).cpu()
        abs_error = (pred - target).abs()
        sq_error = (pred - target) ** 2
        smape_num = 2.0 * abs_error
        smape_den = pred.abs() + target.abs() + _SMAPE_EPSILON

        batch_size, horizon_size = target.shape
        target_nodes = batch_data.target_node.detach().cpu().tolist()
        window_starts = batch_data.window_start.detach().cpu().tolist()
        input_window_length = 0
        if hasattr(batch_data, "hosp_hist") and batch_data.hosp_hist is not None:
            input_window_length = int(batch_data.hosp_hist.shape[1])

        for batch_index in range(batch_size):
            node_id = int(target_nodes[batch_index])
            window_start = int(window_starts[batch_index])
            window_start_date = _format_temporal_coord(temporal_coords, window_start)
            region_id = ""
            region_label = ""
            if region_ids is not None and 0 <= node_id < len(region_ids):
                region_id = str(region_ids[node_id])
            if region_labels is not None and 0 <= node_id < len(region_labels):
                region_label = str(region_labels[node_id])
            if not region_label:
                region_label = region_id

            for horizon_index in range(horizon_size):
                is_observed = bool(
                    observed_weights[batch_index, horizon_index].item() > 0
                )
                if self.observed_only and not is_observed:
                    continue

                target_index = window_start + input_window_length + horizon_index
                row = {
                    "split": self.split_name,
                    "target": canonical_target,
                    "node_id": node_id,
                    "region_id": region_id,
                    "region_label": region_label,
                    "window_start": window_start,
                    "window_start_date": window_start_date,
                    "horizon": horizon_index + 1,
                    "target_index": target_index,
                    "target_date": _format_temporal_coord(
                        temporal_coords, target_index
                    ),
                    "observed": float(target[batch_index, horizon_index].item()),
                    "abs_error": float(abs_error[batch_index, horizon_index].item()),
                    "sq_error": float(sq_error[batch_index, horizon_index].item()),
                    "smape_num": float(smape_num[batch_index, horizon_index].item()),
                    "smape_den": float(smape_den[batch_index, horizon_index].item()),
                }
                self._writer.writerow(row)
