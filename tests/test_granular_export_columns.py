from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from evaluation.granular_export import (
    GRANULAR_FIELDNAMES,
    GranularEvalWriter,
    _granular_columns_from_batch,
    _rows_from_columns,
)

pytestmark = pytest.mark.epiforecaster


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        target_node=torch.tensor([1, 7]),
        window_start=torch.tensor([0, 2]),
        hosp_hist=torch.zeros(2, 3, 3),
    )


def _columns(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "split_name": "val",
        "observed_only": True,
        "batch_data": _batch(),
        "target_name": "hosp",
        "predictions": torch.tensor([[1.0, float("nan")], [float("inf"), 4.0]]),
        "targets": torch.tensor([[2.0, 0.5], [3.0, float("-inf")]]),
        "weights": torch.tensor([[1.0, 0.0], [float("nan"), 2.0]]),
        "temporal_coords": [np.datetime64("2024-01-01") + i for i in range(6)],
        "region_ids": ["A", "B"],
        "region_labels": ["", "Label B"],
        "intervals": {
            "50": (
                torch.tensor([[1.5, 0.0], [0.0, -1.0]]),
                torch.tensor([[2.5, 1.0], [1.0, 1.0]]),
            )
        },
        "distribution_df": 4.0,
    }
    kwargs.update(overrides)
    return _granular_columns_from_batch(**kwargs)  # type: ignore[arg-type]


def test_columns_follow_schema_order_and_keep_observed_cells() -> None:
    columns = _columns()

    assert list(columns) == GRANULAR_FIELDNAMES
    # Cells (0, 0) and (1, 1) are observed; NaN weights count as unobserved.
    assert columns["horizon"].tolist() == [1, 2]
    assert columns["node_id"].tolist() == [1, 7]
    assert columns["window_start_date"] == ["2024-01-01", "2024-01-03"]
    assert columns["target_index"].tolist() == [3, 6]
    assert columns["target_date"] == ["2024-01-04", None]
    assert columns["observation_weight"].tolist() == [1.0, 2.0]
    assert columns["observed_mask"].tolist() == [True, True]


def test_columns_sanitize_non_finite_predictions_and_targets() -> None:
    columns = _columns(observed_only=False)

    assert columns["prediction"].tolist() == [1.0, 0.0, 0.0, 4.0]
    assert columns["observed"].tolist() == [2.0, 0.5, 3.0, 0.0]
    assert columns["abs_error"].tolist() == [1.0, 0.5, 3.0, 4.0]
    assert columns["sq_error"].tolist() == [1.0, 0.25, 9.0, 16.0]


def test_columns_resolve_region_fallbacks() -> None:
    columns = _columns(observed_only=False)

    # Node 1 has an empty label and falls back to its id; node 7 is out of range.
    assert columns["region_id"] == ["B", "B", "", ""]
    assert columns["region_label"] == ["Label B", "Label B", "", ""]


def test_columns_mark_coverage_only_for_observed_cells() -> None:
    columns = _columns(observed_only=False)

    covered = columns["covered_50"].tolist()
    assert covered[0] == 1.0
    assert math.isnan(covered[1]) and math.isnan(covered[2])
    assert covered[3] == 1.0
    # Levels that the batch does not provide stay NaN.
    assert all(math.isnan(value) for value in columns["interval_low_80"].tolist())
    assert all(math.isnan(value) for value in columns["covered_95"].tolist())


def test_columns_fill_missing_optional_outputs_with_nan() -> None:
    columns = _columns()

    for name in (
        "nll",
        "crps",
        "wis",
        "mechanistic_prediction",
        "relevant_latent_value",
    ):
        assert all(math.isnan(value) for value in columns[name].tolist())
    assert columns["relevant_latent_name"] == ["", ""]
    assert columns["distribution_family"] == ["", ""]
    assert columns["distribution_df"] == [4.0, 4.0]


def test_rows_from_columns_use_plain_python_scalars() -> None:
    rows = _rows_from_columns(_columns())

    assert len(rows) == 2
    assert list(rows[0]) == GRANULAR_FIELDNAMES
    assert type(rows[0]["node_id"]) is int
    assert type(rows[0]["prediction"]) is float
    assert type(rows[0]["observed_mask"]) is bool


def test_parquet_writer_flushes_row_groups_and_keeps_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "granular.parquet"
    writer = GranularEvalWriter(path=path, split_name="Val")
    writer._writer.row_group_size = 6
    batch_kwargs = {
        "batch_data": _batch(),
        "target_name": "cases",
        "predictions": torch.ones(2, 2),
        "targets": torch.zeros(2, 2),
        "weights": torch.ones(2, 2),
        "temporal_coords": None,
        "region_ids": None,
        "region_labels": None,
    }
    for _ in range(3):
        writer.write_rows(**batch_kwargs)
    writer.close()

    table = pq.read_table(path)
    assert table.column_names == GRANULAR_FIELDNAMES
    assert table.num_rows == 12
    assert pq.ParquetFile(path).num_row_groups == 2
    assert set(table["target"].to_pylist()) == {"cases"}
