"""Stage-4 matched-comparison analysis tests."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.native_stage4_comparison import (
    CAVEATS,
    HISTORICAL_RELATIVE_PATH,
    compare_native_runs,
    load_native_runs,
    run_comparison,
)

pytestmark = pytest.mark.epiforecaster


def _granular_rows(value: float, *, omit_second: bool = False) -> pd.DataFrame:
    rows = []
    for source_id, day in (
        ("001", "2020-11-01"),
        ("001", "2020-11-02"),
        ("002", "2020-11-02"),
    ):
        if omit_second and source_id == "002":
            continue
        source_value = value if source_id == "001" else 2.0 * value
        rows.append(
            {
                "target": "cases",
                "source_type": "municipality",
                "source_id": source_id,
                "target_date": day,
                "horizon": 1,
                "observed": 1.0,
                "wis": source_value,
                "wis_dispersion": source_value * 0.2,
                "wis_underprediction": source_value * 0.1,
                "wis_overprediction": source_value * 0.1,
                "wis_point": source_value * 0.6,
                "abs_error": source_value + 1.0,
                "abs_error_raw": source_value + 2.0,
                "crps": source_value + 3.0,
                "nll": source_value + 4.0,
                "covered_50": 0.0,
                "covered_80": 1.0,
                "covered_95": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _write_matrix(root: Path) -> list[Path]:
    paths: list[Path] = []
    factors = itertools.product(
        ("pooled", "hetero"), (False, True), ("nll", "crps"), ("mitma", "municipal")
    )
    for encoder, mobility, objective, od_source in factors:
        name = f"{encoder}_{mobility}_{objective}_{od_source}"
        path = root / name / "val_granular.csv"
        path.parent.mkdir(parents=True)
        value = (
            10.0
            + (1.0 if encoder == "hetero" else 0.0)
            + (2.0 if mobility else 0.0)
            + (3.0 if objective == "crps" else 0.0)
            + (4.0 if od_source == "mitma" else 0.0)
        )
        omit_second = (
            encoder == "hetero"
            and mobility
            and objective == "nll"
            and od_source == "mitma"
        )
        _granular_rows(value, omit_second=omit_second).to_csv(path, index=False)
        metadata = {
            "schema_version": "4",
            "spatial_encoder": encoder,
            "mobility_enabled": mobility,
            "objective": objective,
            "od_source": od_source,
            "fold": 0,
            "seed": 52,
            "model_parameters": 100 if encoder == "pooled" else 120,
            "optimizer_steps": 10,
            "wall_time_s_before_final_eval": 1.0,
        }
        path.with_suffix(".csv.meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        paths.append(path)
    return paths


def _write_historical(root: Path) -> None:
    path = root / HISTORICAL_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"family": "normal", "objective": "nll", "target": "cases"},
            {"family": "normal", "objective": "crps", "target": "cases"},
            {"family": "student_t", "objective": "nll", "target": "cases"},
        ]
    ).to_csv(path, index=False)


def test_stage4_pairing_uses_common_source_date_subset(tmp_path: Path) -> None:
    runs = load_native_runs(_write_matrix(tmp_path / "runs"))
    table = compare_native_runs(runs)

    assert len(table) == 20  # 8 encoder + 8 mobility + 4 mobility-on OD pairs
    pair = table[
        (table["comparison"] == "encoder")
        & (table["objective"] == "nll")
        & table["baseline_mobility"]
        & (table["baseline_od_source"] == "mitma")
    ].iloc[0]
    assert pair["baseline_rows"] == 3
    assert pair["candidate_rows"] == 2
    assert pair["common_rows"] == 2
    assert pair["common_sources"] == 1
    assert pair["wis_change"] == pytest.approx(-1.0)
    assert pair["baseline_parameters"] == 100
    assert pair["candidate_parameters"] == 120

    equal_source_pair = table[
        (table["comparison"] == "encoder")
        & (table["objective"] == "nll")
        & ~table["baseline_mobility"]
        & (table["baseline_od_source"] == "mitma")
    ].iloc[0]
    # Source 001 has two dates and gain -1; source 002 has one date and gain
    # -2. Equal-source aggregation is (-1 + -2) / 2, not row mean -4/3.
    assert equal_source_pair["wis_change"] == pytest.approx(-1.5)


def test_stage4_end_to_end_outputs_separate_historical_reference(
    tmp_path: Path,
) -> None:
    paths = _write_matrix(tmp_path / "runs")
    historical_root = tmp_path / "historical"
    output_dir = tmp_path / "reports"
    _write_historical(historical_root)

    summary = run_comparison(
        native_paths=paths,
        historical_root=historical_root,
        output_dir=output_dir,
    )

    native = pd.read_csv(output_dir / "native_stage4_paired_metrics.csv")
    historical = pd.read_csv(output_dir / "historical_municipal_reference.csv")
    persisted = json.loads(
        (output_dir / "native_stage4_summary.json").read_text(encoding="utf-8")
    )
    assert len(native) == 20
    assert set(historical["family"]) == {"normal"}
    assert set(historical["objective"]) == {"nll", "crps"}
    assert persisted == summary
    assert persisted["caveats"] == CAVEATS
    assert "not directly comparable" in " ".join(persisted["caveats"])
    assert persisted["common_subset_identity"] == [
        "target",
        "source_type",
        "source_id",
        "target_date",
        "horizon",
    ]
    assert persisted["mobility_off_neutralized_inputs"] == [
        "od_context_zero_filled",
        "od_graph_messages",
    ]


def test_stage4_requires_complete_fold_seed_matrix(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path / "runs")
    with pytest.raises(ValueError, match="requires 16"):
        load_native_runs(paths[:-1])
