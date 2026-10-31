from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

GRANULAR_SCHEMA_VERSION = "4"
GRANULAR_KEY_COLUMNS = ["split", "target", "node_id", "window_start", "horizon"]
GRANULAR_INTERVAL_LEVEL_NAMES = ("50", "80", "95")
GRANULAR_REQUIRED_FIELDNAMES = [
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
]
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
    "prediction",
    "mechanistic_prediction",
    "residual_contribution",
    "relevant_latent_name",
    "relevant_latent_value",
    "abs_error",
    "sq_error",
    "nll",
    "crps",
    "wis",
    "distribution_loc",
    "distribution_scale",
    "distribution_family",
    "distribution_df",
    "observed_mask",
    "observation_weight",
    "interval_low_50",
    "interval_high_50",
    "covered_50",
    "interval_low_80",
    "interval_high_80",
    "covered_80",
    "interval_low_95",
    "interval_high_95",
    "covered_95",
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
        suffix = path.suffix.lower()
        if suffix == ".csv":
            normalized = "csv"
        elif suffix in {"", ".parquet"}:
            normalized = "parquet"
        else:
            raise ValueError(
                "granular eval output path must end in .csv or .parquet when "
                f"format is not configured, got {path}"
            )
    if normalized not in {"csv", "parquet"}:
        raise ValueError(
            "granular eval format must be one of ['csv', 'parquet'], "
            f"got {normalized!r}"
        )
    return normalized


def _flat_cpu(tensor: torch.Tensor, keep: np.ndarray) -> np.ndarray:
    """Return a [B, H] tensor as float64 values of the kept row-major cells."""
    return tensor.detach().float().cpu().numpy().reshape(-1)[keep].astype(np.float64)


def _granular_columns_from_batch(
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
    nll: torch.Tensor | None = None,
    crps: torch.Tensor | None = None,
    wis: torch.Tensor | None = None,
    distribution_loc: torch.Tensor | None = None,
    distribution_scale: torch.Tensor | None = None,
    distribution_family: str | None = None,
    distribution_df: float | None = None,
    intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    mechanistic_predictions: torch.Tensor | None = None,
    residual_contributions: torch.Tensor | None = None,
    relevant_latent_name: str | None = None,
    relevant_latent_values: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build one batch of granular rows as columns in ``GRANULAR_FIELDNAMES`` order.

    Rows are sample-major then horizon, as in the per-row export. Numeric
    columns are float64/int64/bool arrays; text columns are Python lists.
    """
    canonical_target = _TARGET_NAMES[target_name]
    pred = torch.nan_to_num(
        predictions.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).cpu()
    target = torch.nan_to_num(
        targets.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).cpu()
    observed_weights = torch.nan_to_num(weights.detach().float(), nan=0.0).cpu()
    abs_error = (pred - target).abs()
    sq_error = (pred - target) ** 2

    batch_size, horizon_size = target.shape
    observed_cells = (observed_weights > 0).numpy().reshape(-1)
    keep = observed_cells if observed_only else np.ones_like(observed_cells)
    row_count = int(keep.sum())
    nan_column = np.full(row_count, np.nan)

    def optional(value: torch.Tensor | None) -> np.ndarray:
        return nan_column.copy() if value is None else _flat_cpu(value, keep)

    target_nodes = batch_data.target_node.detach().cpu().numpy().astype(np.int64)
    window_starts = batch_data.window_start.detach().cpu().numpy().astype(np.int64)
    input_window_length = 0
    if hasattr(batch_data, "hosp_hist") and batch_data.hosp_hist is not None:
        input_window_length = int(batch_data.hosp_hist.shape[1])

    sample_index = np.repeat(np.arange(batch_size), horizon_size)[keep]
    horizon_index = np.tile(np.arange(horizon_size), batch_size)[keep]
    row_window_starts = window_starts[sample_index]
    target_indices = row_window_starts + input_window_length + horizon_index

    sample_region_ids: list[str] = []
    sample_region_labels: list[str] = []
    for node_id in target_nodes.tolist():
        region_id = ""
        region_label = ""
        if region_ids is not None and 0 <= node_id < len(region_ids):
            region_id = str(region_ids[node_id])
        if region_labels is not None and 0 <= node_id < len(region_labels):
            region_label = str(region_labels[node_id])
        sample_region_ids.append(region_id)
        sample_region_labels.append(region_label or region_id)

    date_cache: dict[int, str | None] = {}

    def date_at(index: int) -> str | None:
        if index not in date_cache:
            date_cache[index] = _format_temporal_coord(temporal_coords, index)
        return date_cache[index]

    rows = sample_index.tolist()
    observed_column = _flat_cpu(target, keep)
    observed_mask = observed_cells[keep]
    columns: dict[str, Any] = {
        "split": [split_name] * row_count,
        "target": [canonical_target] * row_count,
        "node_id": target_nodes[sample_index],
        "region_id": [sample_region_ids[row] for row in rows],
        "region_label": [sample_region_labels[row] for row in rows],
        "window_start": row_window_starts,
        "window_start_date": [date_at(index) for index in row_window_starts.tolist()],
        "horizon": horizon_index + 1,
        "target_index": target_indices,
        "target_date": [date_at(index) for index in target_indices.tolist()],
        "observed": observed_column,
        "prediction": _flat_cpu(pred, keep),
        "mechanistic_prediction": optional(mechanistic_predictions),
        "residual_contribution": optional(residual_contributions),
        "relevant_latent_name": [relevant_latent_name or ""] * row_count,
        "relevant_latent_value": optional(relevant_latent_values),
        "abs_error": _flat_cpu(abs_error, keep),
        "sq_error": _flat_cpu(sq_error, keep),
        "nll": optional(nll),
        "crps": optional(crps),
        "wis": optional(wis),
        "distribution_loc": optional(distribution_loc),
        "distribution_scale": optional(distribution_scale),
        "distribution_family": [distribution_family or ""] * row_count,
        "distribution_df": [distribution_df] * row_count,
        "observed_mask": observed_mask,
        "observation_weight": _flat_cpu(observed_weights, keep),
    }
    interval_bounds = {str(name): bounds for name, bounds in (intervals or {}).items()}
    for level_name in GRANULAR_INTERVAL_LEVEL_NAMES:
        if level_name in interval_bounds:
            low_tensor, high_tensor = interval_bounds[level_name]
            low = _flat_cpu(low_tensor, keep)
            high = _flat_cpu(high_tensor, keep)
            inside = (low <= observed_column) & (observed_column <= high)
            covered = np.where(observed_mask, inside.astype(np.float64), np.nan)
        else:
            low, high, covered = (nan_column.copy() for _ in range(3))
        columns[f"interval_low_{level_name}"] = low
        columns[f"interval_high_{level_name}"] = high
        columns[f"covered_{level_name}"] = covered
    return {name: columns[name] for name in GRANULAR_FIELDNAMES}


def _rows_from_columns(columns: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert granular columns to row dicts with plain Python scalars."""
    values = [
        column.tolist() if isinstance(column, np.ndarray) else column
        for column in columns.values()
    ]
    names = list(columns)
    return [dict(zip(names, row, strict=True)) for row in zip(*values, strict=True)]


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
        nll: torch.Tensor | None = None,
        crps: torch.Tensor | None = None,
        wis: torch.Tensor | None = None,
        distribution_loc: torch.Tensor | None = None,
        distribution_scale: torch.Tensor | None = None,
        distribution_family: str | None = None,
        distribution_df: float | None = None,
        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        mechanistic_predictions: torch.Tensor | None = None,
        residual_contributions: torch.Tensor | None = None,
        relevant_latent_name: str | None = None,
        relevant_latent_values: torch.Tensor | None = None,
    ) -> None:
        columns = _granular_columns_from_batch(
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
            nll=nll,
            crps=crps,
            wis=wis,
            distribution_loc=distribution_loc,
            distribution_scale=distribution_scale,
            distribution_family=distribution_family,
            distribution_df=distribution_df,
            intervals=intervals,
            mechanistic_predictions=mechanistic_predictions,
            residual_contributions=residual_contributions,
            relevant_latent_name=relevant_latent_name,
            relevant_latent_values=relevant_latent_values,
        )
        self._writer.writerows(_rows_from_columns(columns))


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
        self._tables: list[Any] = []
        self._buffered_rows = 0
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
                ("prediction", pa.float64()),
                ("mechanistic_prediction", pa.float64()),
                ("residual_contribution", pa.float64()),
                ("relevant_latent_name", pa.string()),
                ("relevant_latent_value", pa.float64()),
                ("abs_error", pa.float64()),
                ("sq_error", pa.float64()),
                ("nll", pa.float64()),
                ("crps", pa.float64()),
                ("wis", pa.float64()),
                ("distribution_loc", pa.float64()),
                ("distribution_scale", pa.float64()),
                ("distribution_family", pa.string()),
                ("distribution_df", pa.float64()),
                ("observed_mask", pa.bool_()),
                ("observation_weight", pa.float64()),
                ("interval_low_50", pa.float64()),
                ("interval_high_50", pa.float64()),
                ("covered_50", pa.float64()),
                ("interval_low_80", pa.float64()),
                ("interval_high_80", pa.float64()),
                ("covered_80", pa.float64()),
                ("interval_low_95", pa.float64()),
                ("interval_high_95", pa.float64()),
                ("covered_95", pa.float64()),
            ]
        )

    def _flush(self) -> None:
        if self._schema is None:
            self._schema = self._build_schema()
        if not self._tables:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.concat_tables(self._tables).combine_chunks()
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                self._schema,
                compression="zstd",
            )
        self._writer.write_table(table)
        self._tables.clear()
        self._buffered_rows = 0

    def close(self) -> None:
        if self._schema is None:
            self._schema = self._build_schema()
        if self._tables:
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
        nll: torch.Tensor | None = None,
        crps: torch.Tensor | None = None,
        wis: torch.Tensor | None = None,
        distribution_loc: torch.Tensor | None = None,
        distribution_scale: torch.Tensor | None = None,
        distribution_family: str | None = None,
        distribution_df: float | None = None,
        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        mechanistic_predictions: torch.Tensor | None = None,
        residual_contributions: torch.Tensor | None = None,
        relevant_latent_name: str | None = None,
        relevant_latent_values: torch.Tensor | None = None,
    ) -> None:
        import pyarrow as pa

        if self._schema is None:
            self._schema = self._build_schema()
        columns = _granular_columns_from_batch(
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
            nll=nll,
            crps=crps,
            wis=wis,
            distribution_loc=distribution_loc,
            distribution_scale=distribution_scale,
            distribution_family=distribution_family,
            distribution_df=distribution_df,
            intervals=intervals,
            mechanistic_predictions=mechanistic_predictions,
            residual_contributions=residual_contributions,
            relevant_latent_name=relevant_latent_name,
            relevant_latent_values=relevant_latent_values,
        )
        table = pa.Table.from_pydict(columns, schema=self._schema)
        self._tables.append(table)
        self._buffered_rows += table.num_rows
        if self._buffered_rows >= self.row_group_size:
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
