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


def write_metadata_sidecar(
    output_path: Path,
    metadata: dict[str, Any],
    *,
    schema_version: str,
) -> Path:
    sidecar_path = output_path.with_suffix(f"{output_path.suffix}.meta.json")
    existing: dict[str, Any] = {}
    if sidecar_path.exists():
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
    merged = {**existing, **metadata}
    merged["schema_version"] = schema_version
    normalized = {key: _format_metadata_value(value) for key, value in merged.items()}
    sidecar_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def write_granular_metadata_sidecar(
    granular_output_path: Path,
    metadata: dict[str, Any],
) -> Path:
    return write_metadata_sidecar(
        granular_output_path,
        metadata,
        schema_version=GRANULAR_SCHEMA_VERSION,
    )


def infer_granular_eval_format(path: Path, configured: str | None = None) -> str:
    if configured:
        normalized = configured.lower()
    else:
        normalized = "csv" if path.suffix.lower() == ".csv" else "parquet"
    if normalized not in {"csv", "parquet"}:
        raise ValueError(
            "granular eval format must be one of ['csv', 'parquet'], "
            f"got {normalized!r}"
        )
    return normalized


def _row_dicts_from_batch(
    *,
    split_name: str,
    observed_only: bool,
    batch_data: Any,
    target_name: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    temporal_coords: Sequence[Any] | None,
    region_ids: Sequence[str] | None,
    region_labels: Sequence[str] | None,
) -> list[dict[str, Any]]:
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

    rows: list[dict[str, Any]] = []
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
            is_observed = bool(observed_weights[batch_index, horizon_index].item() > 0)
            if observed_only and not is_observed:
                continue

            target_index = window_start + input_window_length + horizon_index
            rows.append(
                {
                    "split": split_name,
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
            )
    return rows


class _CsvGranularEvalWriter:
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
        for row in _row_dicts_from_batch(
            split_name=self.split_name,
            observed_only=self.observed_only,
            batch_data=batch_data,
            target_name=target_name,
            predictions=predictions,
            targets=targets,
            weights=weights,
            temporal_coords=temporal_coords,
            region_ids=region_ids,
            region_labels=region_labels,
        ):
            self._writer.writerow(row)


class _ParquetGranularEvalWriter:
    """Buffered Parquet writer for large granular eval exports."""

    def __init__(
        self,
        *,
        path: Path,
        split_name: str,
        observed_only: bool = True,
        row_group_size: int = 50_000,
    ) -> None:
        self.path = path
        self.split_name = _normalize_split_name(split_name)
        self.observed_only = observed_only
        self.row_group_size = row_group_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._writer: Any | None = None
        self._schema = None

    @staticmethod
    def _build_schema() -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                ("split", pa.string()),
                ("target", pa.string()),
                ("node_id", pa.int64()),
                ("region_id", pa.string()),
                ("region_label", pa.string()),
                ("window_start", pa.int64()),
                ("window_start_date", pa.string()),
                ("horizon", pa.int64()),
                ("target_index", pa.int64()),
                ("target_date", pa.string()),
                ("observed", pa.float64()),
                ("abs_error", pa.float64()),
                ("sq_error", pa.float64()),
                ("smape_num", pa.float64()),
                ("smape_den", pa.float64()),
            ]
        )

    def _flush(self) -> None:
        if self._schema is None:
            self._schema = self._build_schema()
        if not self._rows:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                self._schema,
                compression="zstd",
            )
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        if self._schema is None:
            self._schema = self._build_schema()
        if self._rows:
            self._flush()
        if self._writer is None:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist([], schema=self._schema), self.path)
        else:
            self._writer.close()

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
        self._rows.extend(
            _row_dicts_from_batch(
                split_name=self.split_name,
                observed_only=self.observed_only,
                batch_data=batch_data,
                target_name=target_name,
                predictions=predictions,
                targets=targets,
                weights=weights,
                temporal_coords=temporal_coords,
                region_ids=region_ids,
                region_labels=region_labels,
            )
        )
        if len(self._rows) >= self.row_group_size:
            self._flush()


class GranularEvalWriter:
    """Format-aware granular eval writer preserving the existing call surface."""

    def __init__(
        self,
        *,
        path: Path,
        split_name: str,
        observed_only: bool = True,
        output_format: str | None = None,
    ) -> None:
        resolved_format = infer_granular_eval_format(path, output_format)
        writer_cls = (
            _CsvGranularEvalWriter
            if resolved_format == "csv"
            else _ParquetGranularEvalWriter
        )
        self.path = path
        self.output_format = resolved_format
        self._writer = writer_cls(
            path=path,
            split_name=split_name,
            observed_only=observed_only,
        )

    def close(self) -> None:
        self._writer.close()

    def write_rows(self, **kwargs: Any) -> None:
        self._writer.write_rows(**kwargs)
