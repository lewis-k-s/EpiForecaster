"""Unit tests for the native-support forecast-query contract (QC-1).

Covers the frozen routing table of plan section 11.1, the validation
failures of section 11.2 (unknown source, empty heads, head/source
mismatch, non-observed source status), and head-union merging over the
trajectory identity. Pure in-memory fixtures; no I/O.
"""

from __future__ import annotations

import pytest

from data.query_contract import (
    HEADS_FOR_SOURCE,
    REPORTING_CADENCE_DAYS,
    ForecastQuery,
    UnsupportedQueryError,
    merge_queries,
    validate_query,
)


class _FakeRegistry:
    """Structural stand-in for REG-1's SourceRegistry (Protocol)."""

    def __init__(self, statuses: dict[tuple[str, str], str | None]) -> None:
        self._statuses = dict(statuses)

    def source_status(self, source_type: str, source_id: str) -> str | None:
        return self._statuses.get((source_type, source_id))


REGISTRY = _FakeRegistry(
    {
        ("municipality", "08121"): "observed",
        ("abs", "088"): "observed",
        ("comarca", "06"): "observed",
        ("edar", "EDIAB"): "observed",
        ("mitma", "08019_AM"): "observed",
        ("municipality", "08002"): "unobserved",
        ("abs", "089"): "unmapped",
    }
)


def _query(**overrides: object) -> ForecastQuery:
    fields: dict[str, object] = {
        "source_type": "municipality",
        "source_id": "08121",
        "forecast_origin": 100,
        "horizon_days": 14,
        "requested_heads": frozenset({"cases"}),
    }
    fields.update(overrides)
    return ForecastQuery(**fields)  # type: ignore[arg-type]


class TestRoutingTable:
    @pytest.mark.epiforecaster
    def test_heads_for_source_matches_frozen_contract(self) -> None:
        assert HEADS_FOR_SOURCE == {
            "municipality": frozenset({"cases"}),
            "abs": frozenset({"hosp"}),
            "comarca": frozenset({"deaths"}),
            "mitma": frozenset(),
            "edar": frozenset({"ww"}),
        }
        assert all(isinstance(heads, frozenset) for heads in HEADS_FOR_SOURCE.values())

    @pytest.mark.epiforecaster
    def test_reporting_cadence_matches_frozen_contract(self) -> None:
        assert REPORTING_CADENCE_DAYS == {"cases": 1, "hosp": 7, "deaths": 1, "ww": 7}

    @pytest.mark.epiforecaster
    def test_head_names_stay_the_canonical_four(self) -> None:
        """No fifth head name: the weekly ABS variant stays ``hosp``."""
        routed = {head for heads in HEADS_FOR_SOURCE.values() for head in heads}
        assert routed == {"cases", "hosp", "deaths", "ww"}
        assert set(REPORTING_CADENCE_DAYS) == routed

    @pytest.mark.epiforecaster
    def test_default_realization_id_is_real(self) -> None:
        assert _query().realization_id == "real"


class TestValidateQuery:
    @pytest.mark.epiforecaster
    @pytest.mark.parametrize(
        ("source_type", "source_id", "heads"),
        [
            ("municipality", "08121", {"cases"}),
            ("abs", "088", {"hosp"}),
            ("comarca", "06", {"deaths"}),
            ("edar", "EDIAB", {"ww"}),
        ],
    )
    def test_observed_queries_for_every_supervised_source_type_pass(
        self, source_type: str, source_id: str, heads: set[str]
    ) -> None:
        validate_query(
            _query(
                source_type=source_type,
                source_id=source_id,
                requested_heads=frozenset(heads),
            ),
            REGISTRY,
        )

    @pytest.mark.epiforecaster
    def test_unknown_source_type_raises(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="Unknown source type"):
            validate_query(_query(source_type="province"), REGISTRY)

    @pytest.mark.epiforecaster
    def test_unknown_source_id_raises(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="not in the registry"):
            validate_query(_query(source_id="99999"), REGISTRY)

    @pytest.mark.epiforecaster
    def test_empty_requested_heads_raises(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="requested no heads"):
            validate_query(_query(requested_heads=frozenset()), REGISTRY)

    @pytest.mark.epiforecaster
    @pytest.mark.parametrize(
        ("source_type", "source_id", "heads"),
        [
            ("municipality", "08121", {"hosp"}),
            ("abs", "088", {"cases"}),
            ("comarca", "06", {"deaths", "cases"}),
            ("mitma", "08019_AM", {"cases"}),
            ("edar", "EDIAB", {"ww", "deaths"}),
        ],
    )
    def test_head_source_mismatch_raises(
        self, source_type: str, source_id: str, heads: set[str]
    ) -> None:
        with pytest.raises(UnsupportedQueryError, match="unsupported for source type"):
            validate_query(
                _query(
                    source_type=source_type,
                    source_id=source_id,
                    requested_heads=frozenset(heads),
                ),
                REGISTRY,
            )

    @pytest.mark.epiforecaster
    def test_mitma_with_empty_heads_also_raises(self) -> None:
        """MITMA is context-only: any request must fail, here on empty heads."""
        with pytest.raises(UnsupportedQueryError, match="requested no heads"):
            validate_query(
                _query(
                    source_type="mitma",
                    source_id="08019_AM",
                    requested_heads=frozenset(),
                ),
                REGISTRY,
            )

    @pytest.mark.epiforecaster
    @pytest.mark.parametrize(
        ("source_type", "source_id", "heads", "status"),
        [
            ("municipality", "08002", {"cases"}, "unobserved"),
            ("abs", "089", {"hosp"}, "unmapped"),
        ],
    )
    def test_non_observed_source_status_raises(
        self, source_type: str, source_id: str, heads: set[str], status: str
    ) -> None:
        with pytest.raises(UnsupportedQueryError, match=f"has status '{status}'"):
            validate_query(
                _query(
                    source_type=source_type,
                    source_id=source_id,
                    requested_heads=frozenset(heads),
                ),
                REGISTRY,
            )


class TestMergeQueries:
    @pytest.mark.epiforecaster
    def test_merges_union_of_heads_over_identical_identity(self) -> None:
        first = _query(requested_heads=frozenset({"cases"}))
        second = _query(requested_heads=frozenset({"deaths"}))
        merged = merge_queries([first, second])
        assert merged == _query(requested_heads=frozenset({"cases", "deaths"}))

    @pytest.mark.epiforecaster
    def test_duplicate_requests_collapse_without_changing_inputs(self) -> None:
        first = _query()
        second = _query()
        assert merge_queries([first, second]) == first
        assert second.requested_heads == frozenset({"cases"})

    @pytest.mark.epiforecaster
    def test_single_query_round_trips(self) -> None:
        query = _query(realization_id="synthetic_run_3")
        assert merge_queries([query]) == query

    @pytest.mark.epiforecaster
    def test_refuses_mixed_realization_id(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="heterogeneous identities"):
            merge_queries([_query(), _query(realization_id="synthetic_run_3")])

    @pytest.mark.epiforecaster
    def test_refuses_different_forecast_origins(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="heterogeneous identities"):
            merge_queries([_query(forecast_origin=100), _query(forecast_origin=120)])

    @pytest.mark.epiforecaster
    def test_refuses_different_source(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="heterogeneous identities"):
            merge_queries([_query(source_id="08121"), _query(source_id="08002")])
        with pytest.raises(UnsupportedQueryError, match="heterogeneous identities"):
            merge_queries([_query(), _query(source_type="abs", source_id="088")])

    @pytest.mark.epiforecaster
    def test_refuses_conflicting_horizon_within_one_identity(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="conflicting horizon_days"):
            merge_queries([_query(horizon_days=14), _query(horizon_days=28)])

    @pytest.mark.epiforecaster
    def test_refuses_empty_sequence(self) -> None:
        with pytest.raises(UnsupportedQueryError, match="empty"):
            merge_queries([])
