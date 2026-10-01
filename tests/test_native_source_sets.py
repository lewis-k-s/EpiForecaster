"""Tests for eligibility-backed multi-resolution source sets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from data.native_source_sets import (
    SCHEMA,
    SOURCE_TYPES,
    QueryEligibilityRecord,
    SourceBalanceRecord,
    assemble_source_set_manifest,
    assign_balanced_source_sets,
    freeze_calendar_split,
    load_source_set_manifest,
    rotating_fold_roles,
    split_queries_for_run,
    write_source_set_manifest,
)

pytestmark = pytest.mark.epiforecaster


HEAD_BY_TYPE = {
    "municipality": "cases",
    "abs": "hosp+deaths",
    "edar": "ww",
}


def _records(per_type: int = 9) -> list[SourceBalanceRecord]:
    records: list[SourceBalanceRecord] = []
    for type_index, source_type in enumerate(SOURCE_TYPES):
        for index in range(per_type):
            records.append(
                SourceBalanceRecord(
                    source_type=source_type,
                    source_id=f"{source_type}-{index:02d}",
                    head=HEAD_BY_TYPE[source_type],
                    eligible_queries=10 + index,
                    support_population=1000.0 * (index + 1),
                    target_burden=float((index + type_index) % 5),
                    geography=f"g{index % 3}",
                    assay_observations=(20 + index) if source_type == "edar" else 0,
                    censored_fraction=(index / 20) if source_type == "edar" else 0,
                    flow_coverage=(0.5 + index / 20)
                    if source_type == "edar"
                    else 0,
                )
            )
    return records


def _queries(records: list[SourceBalanceRecord]) -> list[QueryEligibilityRecord]:
    return [
        QueryEligibilityRecord(
            source_type=record.source_type,
            source_id=record.source_id,
            head=record.head,
            origin=origin,
        )
        for origin in range(12)
        for record in records
    ]


def test_assigns_every_source_once_and_balances_each_resolution() -> None:
    records = _records()
    assignments = assign_balanced_source_sets(records, num_folds=3, seed=7)

    assert len(assignments) == len(records)
    assert assignments == assign_balanced_source_sets(records, num_folds=3, seed=7)
    for source_type in SOURCE_TYPES:
        counts = Counter(
            fold
            for (record_type, _source_id), fold in assignments.items()
            if record_type == source_type
        )
        assert set(counts) == {0, 1, 2}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_inventory_requires_every_supervised_type_and_enough_sources() -> None:
    records = _records()
    with pytest.raises(ValueError, match="lacks source types"):
        assign_balanced_source_sets(
            [record for record in records if record.source_type != "abs"],
            num_folds=3,
        )
    with pytest.raises(ValueError, match="fewer than 10 folds"):
        assign_balanced_source_sets(records, num_folds=10)


def test_calendar_and_rotating_roles_produce_three_disjoint_splits() -> None:
    records = _records()
    queries = _queries(records)
    assignments = assign_balanced_source_sets(records, num_folds=3, seed=7)
    calendar = freeze_calendar_split(
        queries, validation_fraction=0.25, test_fraction=0.25, forecast_horizon=1
    )
    roles = rotating_fold_roles(0, 3)

    train, validation, test = split_queries_for_run(
        queries, assignments, calendar, roles
    )

    assert roles.test_fold == 0
    assert roles.validation_fold == 1
    assert roles.train_folds == (2,)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert {queries[index].head for index in train} == set(HEAD_BY_TYPE.values())
    assert {queries[index].head for index in validation} == set(
        HEAD_BY_TYPE.values()
    )
    assert {queries[index].head for index in test} == set(HEAD_BY_TYPE.values())
    assert max(queries[index].origin for index in train) < min(
        queries[index].origin for index in validation
    )
    assert max(queries[index].origin for index in validation) < min(
        queries[index].origin for index in test
    )


def test_abs_admissions_and_deaths_keep_one_source_fold() -> None:
    records = _records()
    queries = [
        QueryEligibilityRecord(record.source_type, record.source_id, head, origin)
        for origin in range(12)
        for record in records
        for head in (("hosp", "deaths") if record.source_type == "abs" else (record.head,))
    ]
    assignments = assign_balanced_source_sets(records, num_folds=3, seed=7)
    calendar = freeze_calendar_split(
        queries, validation_fraction=0.25, test_fraction=0.25, forecast_horizon=1
    )
    splits = split_queries_for_run(
        queries, assignments, calendar, rotating_fold_roles(0, 3)
    )
    for positions in splits:
        abs_queries = {
            (queries[index].source_id, queries[index].origin, queries[index].head)
            for index in positions
            if queries[index].source_type == "abs"
        }
        for source_id, origin, head in abs_queries:
            paired = "deaths" if head == "hosp" else "hosp"
            assert (source_id, origin, paired) in abs_queries


def test_manifest_records_four_source_sets_and_all_run_roles() -> None:
    records = _records()
    queries = _queries(records)
    assignments = assign_balanced_source_sets(records, num_folds=3, seed=7)
    calendar = freeze_calendar_split(
        queries, validation_fraction=0.25, test_fraction=0.25, forecast_horizon=1
    )

    payload = assemble_source_set_manifest(
        records, assignments, calendar, num_folds=3, seed=7
    )

    assert payload["schema"] == SCHEMA
    assert len(payload["folds"]) == 3
    assert len(payload["run_roles"]) == 3
    for fold in payload["folds"]:
        assert set(fold["source_sets"]) == set(SOURCE_TYPES)
    assert set(payload["fold_metrics"]) == set(SOURCE_TYPES)


def test_manifest_round_trip_and_duplicate_guard(tmp_path: Path) -> None:
    records = _records()
    queries = _queries(records)
    assignments = assign_balanced_source_sets(records, num_folds=3, seed=7)
    calendar = freeze_calendar_split(
        queries, validation_fraction=0.25, test_fraction=0.25, forecast_horizon=1
    )
    payload = assemble_source_set_manifest(
        records, assignments, calendar, num_folds=3, seed=7
    )
    path = tmp_path / "source_sets.json"
    write_source_set_manifest(payload, path)

    restored = load_source_set_manifest(path)
    assert restored.assignments == assignments
    assert restored.run_roles[0] == rotating_fold_roles(0, 3)

    import pandas as pd
    import hashlib
    table_path = path.parent / "assignments.parquet"
    table = pd.read_parquet(table_path)
    pd.concat([table, table.iloc[:1]]).to_parquet(table_path, index=False)
    tampered = json.loads(path.read_text())
    tampered["assignments_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="multiple folds"):
        load_source_set_manifest(path)
