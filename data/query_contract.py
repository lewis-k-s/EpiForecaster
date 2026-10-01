"""Native-support forecast-query contract (plan sections 11.1-11.2).

Defines the frozen query interface for the heterogeneous-graph refactor: the
source-type to observation-head routing table, per-head reporting cadences,
the ``ForecastQuery`` sample identity, and the validation/merge operations
that guard every downstream native-support consumer (dataset indexing, batch
collation, head routing, loss composition). Pure stdlib — no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

SourceType = Literal["municipality", "abs", "comarca", "mitma", "edar"]

# Real-data query routing (supervised heads only; synthetic may request
# unions). Head names stay the canonical set ``ww | hosp | cases | deaths``
# used by evaluation/losses.py and the export modules. ABS carries weekly
# admissions and daily all-cause deaths; excess is derived at evaluation.
HEADS_FOR_SOURCE: dict[SourceType, frozenset[str]] = {
    "municipality": frozenset({"cases"}),
    "abs": frozenset({"hosp", "deaths"}),
    "comarca": frozenset(),  # geographic context only
    "mitma": frozenset(),  # context-only, never queried
    "edar": frozenset({"ww"}),
}

# Reporting interval per canonical head, in days on the shared daily grid.
REPORTING_CADENCE_DAYS: dict[str, int] = {
    "cases": 1,
    "hosp": 7,
    "deaths": 1,
    "ww": 7,
}

# Only observed sources may be queried; ``unobserved`` and ``unmapped`` are
# explicit registry statuses (plan section 11.3) that must fail validation.
OBSERVED = "observed"


class UnsupportedQueryError(ValueError):
    """A forecast query failed validation or merge; never passed silently."""


class SourceStatusRegistry(Protocol):
    """Minimal registry surface needed to validate forecast queries.

    REG-1's ``SourceRegistry`` satisfies this structurally. ``source_status``
    returns one of ``{"observed", "unobserved", "unmapped"}``, or ``None``
    when the ``(source_type, source_id)`` pair is absent from the registry.
    """

    def source_status(self, source_type: SourceType, source_id: str) -> str | None:
        """Return the source's observation status, or None if unknown."""
        ...


@dataclass(frozen=True)
class ForecastQuery:
    """One source population at one forecast origin, with requested heads.

    Trajectory identity is source-conditioned, not head-conditioned: for a
    fixed source, origin, realization and permitted context, one latent
    rollout serves every requested compatible head (plan section 4).
    """

    source_type: SourceType
    source_id: str  # native registry ID ("08121", "088", "06", ...)
    forecast_origin: int  # index into the shared daily date coordinate
    horizon_days: int  # default config.model.forecast_horizon
    requested_heads: frozenset[str]
    realization_id: str = "real"  # dataset/run identity — synthetic runs never merge


def validate_query(query: ForecastQuery, registry: SourceStatusRegistry) -> None:
    """Validate a query against the routing table and source registry.

    Raises ``UnsupportedQueryError`` on: unknown source type or unknown
    ``(source_type, source_id)`` pair; empty ``requested_heads``; heads not
    in ``HEADS_FOR_SOURCE[source_type]``; and any registry status other than
    ``"observed"`` (unobserved and unmapped sources are never queried).
    """
    if query.source_type not in HEADS_FOR_SOURCE:
        raise UnsupportedQueryError(
            f"Unknown source type {query.source_type!r}; "
            f"expected one of {sorted(HEADS_FOR_SOURCE)}"
        )
    status = registry.source_status(query.source_type, query.source_id)
    if status is None:
        raise UnsupportedQueryError(
            f"Unknown source: ({query.source_type!r}, {query.source_id!r}) "
            "is not in the registry"
        )
    if not query.requested_heads:
        raise UnsupportedQueryError(
            f"Query for {query.source_type} {query.source_id!r} requested no heads"
        )
    allowed = HEADS_FOR_SOURCE[query.source_type]
    unsupported = query.requested_heads - allowed
    if unsupported:
        raise UnsupportedQueryError(
            f"Head(s) {sorted(unsupported)} unsupported for source type "
            f"{query.source_type!r}; allowed: {sorted(allowed)}"
        )
    if status != OBSERVED:
        raise UnsupportedQueryError(
            f"Source ({query.source_type!r}, {query.source_id!r}) has status "
            f"{status!r}; only 'observed' sources may be queried"
        )


def merge_queries(queries: Iterable[ForecastQuery]) -> ForecastQuery:
    """Merge repeated requests into one query with the union of heads.

    Queries sharing the trajectory identity ``(source_type, source_id,
    forecast_origin, realization_id)`` collapse into a single request whose
    ``requested_heads`` is their union. Heterogeneous identities, an empty
    input sequence, and conflicting ``horizon_days`` within one identity are
    refused.
    """
    collected = list(queries)
    if not collected:
        raise UnsupportedQueryError("Cannot merge an empty sequence of queries")
    first = collected[0]
    identity = (
        first.source_type,
        first.source_id,
        first.forecast_origin,
        first.realization_id,
    )
    heads: set[str] = set()
    for query in collected:
        other = (
            query.source_type,
            query.source_id,
            query.forecast_origin,
            query.realization_id,
        )
        if other != identity:
            raise UnsupportedQueryError(
                "Cannot merge queries with heterogeneous identities: "
                f"{identity} vs {other}"
            )
        if query.horizon_days != first.horizon_days:
            raise UnsupportedQueryError(
                f"Cannot merge queries for identity {identity} with "
                f"conflicting horizon_days: {first.horizon_days} vs "
                f"{query.horizon_days}"
            )
        heads.update(query.requested_heads)
    return replace(first, requested_heads=frozenset(heads))
