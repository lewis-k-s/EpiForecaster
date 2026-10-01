"""Eligibility-backed source sets and rotating train/validation/test roles.

Version 4 does not derive every source fold from a reduced municipality set.
Municipalities, ABS, and EDAR plants are assigned directly from
their eligible query inventory. A common fold index coordinates repeated
runs, but each source type has its own balanced source set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.crossval_manifest import hash_region_ids

SCHEMA = "native_source_sets.v4"
DEFAULT_NUM_FOLDS = 10
SOURCE_TYPES = ("municipality", "abs", "edar")


@dataclass(frozen=True)
class SourceBalanceRecord:
    """One supervised source and the quantities balanced across folds."""

    source_type: str
    source_id: str
    head: str
    eligible_queries: int
    support_population: float
    target_burden: float = 0.0
    geography: str = "unknown"
    assay_observations: int = 0
    censored_fraction: float = 0.0
    flow_coverage: float = 0.0

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unsupported source type {self.source_type!r}")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.eligible_queries <= 0:
            raise ValueError("eligible_queries must be positive")
        for name in (
            "support_population",
            "target_burden",
            "censored_fraction",
            "flow_coverage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.censored_fraction <= 1:
            raise ValueError("censored_fraction must be in [0, 1]")
        if not 0 <= self.flow_coverage <= 1:
            raise ValueError("flow_coverage must be in [0, 1]")


@dataclass(frozen=True)
class QueryEligibilityRecord:
    """One eligible source-origin query used to freeze calendar intervals."""

    source_type: str
    source_id: str
    head: str
    origin: int


@dataclass(frozen=True)
class CalendarSplit:
    """Positions of chronological train, validation, and test intervals."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    validation_start_origin: int
    test_start_origin: int
    forecast_horizon: int


@dataclass(frozen=True)
class FoldRoles:
    """Source-fold roles for one repeated run."""

    run_index: int
    train_folds: tuple[int, ...]
    validation_fold: int
    test_fold: int


@dataclass(frozen=True)
class SourceSetManifest:
    """Validated version-4 source-set manifest."""

    path: Path
    num_folds: int
    assignments: dict[tuple[str, str], int]
    validation_start_origin: int
    test_start_origin: int
    forecast_horizon: int
    provenance: dict
    run_roles: tuple[FoldRoles, ...]


def _tie_value(seed: int, source_type: str, source_id: str, fold: int) -> int:
    payload = f"{seed}|{source_type}|{source_id}|{fold}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _numeric_vector(record: SourceBalanceRecord) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(record.eligible_queries),
            float(record.support_population),
            float(record.target_burden),
            float(record.assay_observations),
            float(record.censored_fraction),
            float(record.flow_coverage),
        ],
        dtype=np.float64,
    )


def _objective(
    records: Sequence[SourceBalanceRecord],
    assignments: Mapping[str, int],
    num_folds: int,
) -> float:
    """Normalized fold imbalance for one source type."""
    vectors = np.stack([_numeric_vector(record) for record in records])
    totals = vectors.sum(axis=0)
    active = totals > 0
    target = totals / num_folds
    weights = np.asarray([3.0, 4.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    score = 0.0
    by_fold = np.zeros((num_folds, vectors.shape[1]), dtype=np.float64)
    geography_counts: dict[tuple[int, str], int] = defaultdict(int)
    geography_total: dict[str, int] = defaultdict(int)
    for record, vector in zip(records, vectors, strict=True):
        fold = assignments[record.source_id]
        by_fold[fold] += vector
        geography_counts[(fold, record.geography)] += 1
        geography_total[record.geography] += 1
    normalized = np.zeros_like(by_fold)
    normalized[:, active] = (by_fold[:, active] - target[active]) / totals[active]
    score += float(np.sum(weights[active] * normalized[:, active] ** 2))
    for geography, total in geography_total.items():
        expected = total / num_folds
        if expected == 0:
            continue
        for fold in range(num_folds):
            delta = geography_counts[(fold, geography)] - expected
            score += 0.5 * (delta / total) ** 2
    return score


def _assign_one_type(
    records: Sequence[SourceBalanceRecord], *, num_folds: int, seed: int
) -> dict[str, int]:
    if len(records) < num_folds:
        raise ValueError(
            f"source type {records[0].source_type!r} has {len(records)} sources, "
            f"fewer than {num_folds} folds"
        )
    source_ids = [record.source_id for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(
            f"source type {records[0].source_type!r} contains duplicate source IDs"
        )

    totals = np.stack([_numeric_vector(record) for record in records]).sum(axis=0)
    scale = np.where(totals > 0, totals, 1.0)
    ordered = sorted(
        records,
        key=lambda record: (
            -float(np.sum(_numeric_vector(record) / scale)),
            _tie_value(seed, record.source_type, record.source_id, 0),
        ),
    )
    assignments: dict[str, int] = {}
    fold_vectors = np.zeros((num_folds, len(scale)), dtype=np.float64)
    fold_counts = np.zeros(num_folds, dtype=np.int64)
    max_sources_per_fold = math.ceil(len(records) / num_folds)
    geography_counts: dict[tuple[int, str], int] = defaultdict(int)

    for position, record in enumerate(ordered):
        vector = _numeric_vector(record)
        candidate_folds = (
            [position]
            if position < num_folds
            else [
                fold
                for fold in range(num_folds)
                if fold_counts[fold] < max_sources_per_fold
            ]
        )
        best_fold: int | None = None
        best_score: float | None = None
        for fold in candidate_folds:
            projected = fold_vectors.copy()
            projected[fold] += vector
            numeric_score = float(
                np.sum(((projected - totals / num_folds) / scale) ** 2)
            )
            geography_score = float(geography_counts[(fold, record.geography)])
            score = numeric_score + 0.5 * geography_score
            if best_score is None or score < best_score - 1e-15:
                best_score, best_fold = score, fold
            elif math.isclose(score, best_score, rel_tol=0, abs_tol=1e-15):
                assert best_fold is not None
                if _tie_value(
                    seed, record.source_type, record.source_id, fold
                ) < _tie_value(seed, record.source_type, record.source_id, best_fold):
                    best_fold = fold
        assert best_fold is not None
        assignments[record.source_id] = best_fold
        fold_vectors[best_fold] += vector
        fold_counts[best_fold] += 1
        geography_counts[(best_fold, record.geography)] += 1

    min_sources_per_fold = len(records) // num_folds
    while int(fold_counts.min()) < min_sources_per_fold:
        receiver = int(np.argmin(fold_counts))
        donors = {
            fold
            for fold in range(num_folds)
            if fold_counts[fold] > min_sources_per_fold
        }
        candidates = [
            record for record in records if assignments[record.source_id] in donors
        ]
        best_record = min(
            candidates,
            key=lambda record: (
                _objective(
                    records,
                    {
                        **assignments,
                        record.source_id: receiver,
                    },
                    num_folds,
                ),
                _tie_value(seed, record.source_type, record.source_id, receiver),
            ),
        )
        donor = assignments[best_record.source_id]
        assignments[best_record.source_id] = receiver
        fold_counts[donor] -= 1
        fold_counts[receiver] += 1

    swap_order = sorted(records, key=lambda record: record.source_id)
    for _pass in range(8):
        baseline = _objective(records, assignments, num_folds)
        improved = False
        for left, right in zip(swap_order, swap_order[1:]):
            left_fold = assignments[left.source_id]
            right_fold = assignments[right.source_id]
            if left_fold == right_fold:
                continue
            candidate = dict(assignments)
            candidate[left.source_id] = right_fold
            candidate[right.source_id] = left_fold
            score = _objective(records, candidate, num_folds)
            if score < baseline - 1e-12:
                assignments = candidate
                improved = True
                break
        if not improved:
            break
    return assignments


def assign_balanced_source_sets(
    records: Iterable[SourceBalanceRecord],
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = 42,
) -> dict[tuple[str, str], int]:
    """Assign each eligible native source exactly once to a balanced fold."""
    if num_folds < 3:
        raise ValueError("at least three folds are required for rotating roles")
    collected = list(records)
    if not collected:
        raise ValueError("source-set assignment requires at least one source")
    by_type: dict[str, list[SourceBalanceRecord]] = defaultdict(list)
    for record in collected:
        by_type[record.source_type].append(record)
    expected_types: set[str] = {"municipality", "abs"}
    missing_types = sorted(expected_types - set(by_type))
    if missing_types:
        raise ValueError(f"source-set inventory lacks source types: {missing_types}")

    result: dict[tuple[str, str], int] = {}
    for source_type in SOURCE_TYPES:
        typed = sorted(by_type[source_type], key=lambda record: record.source_id)
        if not typed:
            continue
        assignments = _assign_one_type(typed, num_folds=num_folds, seed=seed)
        result.update(
            ((source_type, source_id), fold) for source_id, fold in assignments.items()
        )
    return result


def freeze_calendar_split(
    queries: Sequence[QueryEligibilityRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    forecast_horizon: int,
    min_queries_per_head: int = 1,
) -> CalendarSplit:
    """Choose chronological intervals and require every head in every split."""
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    if not queries:
        raise ValueError("calendar split requires eligible queries")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    if min_queries_per_head <= 0:
        raise ValueError("min_queries_per_head must be positive")
    origins = sorted({query.origin for query in queries})
    if len(origins) < 3:
        raise ValueError("calendar split requires at least three distinct origins")
    n_test = max(1, math.ceil(test_fraction * len(origins)))
    n_validation = max(1, math.ceil(validation_fraction * len(origins)))
    if n_test + n_validation >= len(origins):
        raise ValueError("calendar fractions leave no training origins")
    validation_start = origins[-(n_test + n_validation)]
    test_start = origins[-n_test]
    train = tuple(
        i
        for i, query in enumerate(queries)
        if query.origin + forecast_horizon < validation_start
    )
    validation = tuple(
        i
        for i, query in enumerate(queries)
        if validation_start <= query.origin
        and query.origin + forecast_horizon < test_start
    )
    test = tuple(i for i, query in enumerate(queries) if query.origin >= test_start)
    heads = sorted({query.head for query in queries})
    for name, positions in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        counts = {head: 0 for head in heads}
        for position in positions:
            counts[queries[position].head] += 1
        deficient = {
            head: count
            for head, count in counts.items()
            if count < min_queries_per_head
        }
        if deficient:
            raise ValueError(
                f"calendar {name} interval lacks minimum head queries: {deficient}"
            )
    return CalendarSplit(
        train=train,
        validation=validation,
        test=test,
        validation_start_origin=validation_start,
        test_start_origin=test_start,
        forecast_horizon=forecast_horizon,
    )


def rotating_fold_roles(run_index: int, num_folds: int) -> FoldRoles:
    """Use fold ``r`` for test, ``r+1`` for validation, and all others to train."""
    if not 0 <= run_index < num_folds:
        raise ValueError(f"run_index must be in 0..{num_folds - 1}")
    test_fold = run_index
    validation_fold = (run_index + 1) % num_folds
    train_folds = tuple(
        fold for fold in range(num_folds) if fold not in {test_fold, validation_fold}
    )
    return FoldRoles(
        run_index=run_index,
        train_folds=train_folds,
        validation_fold=validation_fold,
        test_fold=test_fold,
    )


def split_queries_for_run(
    queries: Sequence[QueryEligibilityRecord],
    assignments: Mapping[tuple[str, str], int],
    calendar: CalendarSplit,
    roles: FoldRoles,
) -> tuple[list[int], list[int], list[int]]:
    """Apply both source-fold roles and frozen chronological intervals."""
    calendar_sets = {
        "train": set(calendar.train),
        "validation": set(calendar.validation),
        "test": set(calendar.test),
    }
    positions = {"train": [], "validation": [], "test": []}
    for index, query in enumerate(queries):
        key = query.source_type, query.source_id
        if key not in assignments:
            raise ValueError(f"Eligible source missing from split asset: {key}")
        fold = assignments[key]
        if index in calendar_sets["train"] and fold in roles.train_folds:
            positions["train"].append(index)
        elif index in calendar_sets["validation"] and fold == roles.validation_fold:
            positions["validation"].append(index)
        elif index in calendar_sets["test"] and fold == roles.test_fold:
            positions["test"].append(index)
    for name, selected in positions.items():
        if not selected:
            raise ValueError(f"run {roles.run_index} has no {name} queries")
    return positions["train"], positions["validation"], positions["test"]


def assemble_source_set_manifest(
    records: Sequence[SourceBalanceRecord],
    assignments: Mapping[tuple[str, str], int],
    calendar: CalendarSplit,
    *,
    num_folds: int,
    seed: int,
) -> dict[str, object]:
    """Create the deterministic version-4 source-set manifest payload."""
    folds = []
    for fold in range(num_folds):
        source_sets = {
            source_type: sorted(
                source_id
                for (record_type, source_id), assigned in assignments.items()
                if record_type == source_type and assigned == fold
            )
            for source_type in SOURCE_TYPES
        }
        folds.append({"fold_index": fold, "source_sets": source_sets})
    source_hashes = {
        source_type: hash_region_ids(
            sorted(
                source_id
                for record_type, source_id in assignments
                if record_type == source_type
            )
        )
        for source_type in SOURCE_TYPES
    }
    metrics: dict[str, list[dict[str, float | int]]] = {}
    for source_type in SOURCE_TYPES:
        typed = [record for record in records if record.source_type == source_type]
        metrics[source_type] = []
        for fold in range(num_folds):
            selected = [
                record
                for record in typed
                if assignments[(record.source_type, record.source_id)] == fold
            ]
            metrics[source_type].append(
                {
                    "sources": len(selected),
                    "eligible_queries": sum(r.eligible_queries for r in selected),
                    "support_population": float(
                        sum(r.support_population for r in selected)
                    ),
                    "target_burden": float(sum(r.target_burden for r in selected)),
                    "assay_observations": sum(r.assay_observations for r in selected),
                }
            )
    return {
        "schema": SCHEMA,
        "num_folds": num_folds,
        "seed": seed,
        "derivation": "direct_eligibility_backed_multi_source_balance",
        "generalization_claim": (
            "held-out native sources with frozen future calendar intervals; "
            "local pre-origin history remains visible"
        ),
        "source_id_hashes": source_hashes,
        "calendar": {
            "validation_start_origin": calendar.validation_start_origin,
            "test_start_origin": calendar.test_start_origin,
            "forecast_horizon": calendar.forecast_horizon,
        },
        "folds": folds,
        "fold_metrics": metrics,
        "run_roles": [
            asdict(rotating_fold_roles(run, num_folds)) for run in range(num_folds)
        ],
    }


def write_source_set_manifest(payload: Mapping[str, object], path: Path) -> None:
    """Write typed assignments as Parquet and a small JSON metadata sidecar."""
    if path.suffix != ".json":
        path = path / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(payload)
    rows = [
        (source_type, source_id, fold["fold_index"])
        for fold in metadata.pop("folds")
        for source_type, ids in fold["source_sets"].items()
        for source_id in ids
    ]
    table = pd.DataFrame(rows, columns=["source_type", "source_id", "fold_index"])
    table = table.astype(
        {"source_type": "string", "source_id": "string", "fold_index": "int32"}
    )
    assignments = path.parent / "assignments.parquet"
    table.sort_values(["source_type", "source_id"]).to_parquet(assignments, index=False)
    metadata["assignments_sha256"] = hashlib.sha256(
        assignments.read_bytes()
    ).hexdigest()
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_source_set_manifest(path: Path) -> SourceSetManifest:
    """Load and validate exact-once source membership and rotating roles."""
    if path.is_dir():
        path = path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"Unsupported native source-set schema in {path}: {payload.get('schema')!r}"
        )
    num_folds = int(payload.get("num_folds", -1))
    if num_folds < 3:
        raise ValueError("native source-set manifest requires at least three folds")
    assignments_path = path.parent / "assignments.parquet"
    if hashlib.sha256(assignments_path.read_bytes()).hexdigest() != payload.get(
        "assignments_sha256"
    ):
        raise ValueError("native source-set assignment hash mismatch")
    schema = pq.read_schema(assignments_path)
    if (
        schema.names != ["source_type", "source_id", "fold_index"]
        or not all(
            pa.types.is_string(schema.field(name).type)
            or pa.types.is_large_string(schema.field(name).type)
            for name in ("source_type", "source_id")
        )
        or schema.field("fold_index").type != pa.int32()
    ):
        raise ValueError("Invalid native source-set Parquet field types")
    table = pd.read_parquet(assignments_path)
    if (
        list(table.columns) != ["source_type", "source_id", "fold_index"]
        or table.isna().any().any()
    ):
        raise ValueError("Invalid native source-set assignment table")
    if table.duplicated(["source_type", "source_id"]).any():
        raise ValueError("source assigned to multiple folds")
    if set(table.source_type) - set(SOURCE_TYPES) or set(table.fold_index) != set(
        range(num_folds)
    ):
        raise ValueError("Invalid native source-set assignment roles")
    fold_entries = [
        {
            "fold_index": fold,
            "source_sets": {
                kind: table.loc[
                    (table.fold_index == fold) & (table.source_type == kind),
                    "source_id",
                ].tolist()
                for kind in SOURCE_TYPES
            },
        }
        for fold in range(num_folds)
    ]
    assignments: dict[tuple[str, str], int] = {}
    by_type: dict[str, list[str]] = {source_type: [] for source_type in SOURCE_TYPES}
    for entry in fold_entries:
        fold = int(entry["fold_index"])
        source_sets = entry.get("source_sets")
        if not isinstance(source_sets, dict) or set(source_sets) != set(SOURCE_TYPES):
            raise ValueError(
                f"fold {fold} source_sets must contain exactly {SOURCE_TYPES}"
            )
        for source_type in SOURCE_TYPES:
            values = [str(value) for value in source_sets[source_type]]
            if len(values) != len(set(values)):
                raise ValueError(
                    f"fold {fold} {source_type} source set contains duplicates"
                )
            for source_id in values:
                key = source_type, source_id
                if key in assignments:
                    raise ValueError(f"source assigned to multiple folds: {key}")
                assignments[key] = fold
            by_type[source_type].extend(values)
    hashes = payload.get("source_id_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("native source-set manifest lacks source_id_hashes")
    for source_type in SOURCE_TYPES:
        expected = hash_region_ids(sorted(by_type[source_type]))
        if hashes.get(source_type) != expected:
            raise ValueError(f"native source-set hash mismatch for {source_type}")

    calendar = payload.get("calendar")
    if not isinstance(calendar, dict):
        raise ValueError("native source-set manifest lacks calendar")
    validation_start = int(calendar["validation_start_origin"])
    test_start = int(calendar["test_start_origin"])
    if validation_start >= test_start:
        raise ValueError("validation_start_origin must precede test_start_origin")

    forecast_horizon = int(calendar["forecast_horizon"])
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    raw_roles = payload.get("run_roles")
    if not isinstance(raw_roles, list) or len(raw_roles) != num_folds:
        raise ValueError("native source-set manifest must define one role per run")
    roles = tuple(
        FoldRoles(
            run_index=int(entry["run_index"]),
            train_folds=tuple(int(value) for value in entry["train_folds"]),
            validation_fold=int(entry["validation_fold"]),
            test_fold=int(entry["test_fold"]),
        )
        for entry in sorted(raw_roles, key=lambda entry: int(entry["run_index"]))
    )
    for run, role in enumerate(roles):
        if role != rotating_fold_roles(run, num_folds):
            raise ValueError(f"native source-set role mismatch for run {run}")
    return SourceSetManifest(
        path=path,
        num_folds=num_folds,
        assignments=assignments,
        validation_start_origin=validation_start,
        test_start_origin=test_start,
        forecast_horizon=forecast_horizon,
        provenance=payload.get("provenance", {}),
        run_roles=roles,
    )
