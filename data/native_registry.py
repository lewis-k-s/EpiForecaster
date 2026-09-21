"""Native source registries (plan sections 11.2-11.3, slice REG-1).

Registries of :class:`SourceRecord` per native source type, keyed on string
IDs with leading zeros preserved (``"088"``, ``"00"``, ``"01001"``). Statuses
follow plan section 11.3:

- ``observed``: at least one observation row falls inside the study window
  (2020-03-01..2021-05-09). A reported row that sums to zero still counts:
  presence of a reported observation is the criterion, not its value.
- ``unobserved``: observations exist, but only outside the window.
- ``unmapped``: the code exists on only one side of a reconciliation
  (observations vs geometry/membership). Overrides ``observed``; one-sided
  codes are surfaced as statuses, never as errors.

Orchestrator ruling (2026-09-21): ``poblacio`` in the ABS CSV is a
per-(sex, age)-stratum, case-only population, so the CSV cannot supply an ABS
population denominator. ABS records carry ``population=None``; the audited ABS
denominator is assigned later at ``native_graph`` assembly from overlap-implied
municipal population (plan sections 10.3(4) and 11.3). The sentinel filter is
the structural triple only (``codi_abs=="000"`` / ``codi_regio=="00"`` /
``"No disponible"`` text fields); ``index_socioeconomic==-1`` is carried by
real ABS with genuine admissions and must never be filtered on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

STUDY_WINDOW_START = date(2020, 3, 1)
STUDY_WINDOW_END = date(2021, 5, 9)

Window = tuple[date, date]

SOURCE_MUNICIPALITY = "municipality"
SOURCE_ABS = "abs"
SOURCE_COMARCA = "comarca"

STATUS_OBSERVED = "observed"
STATUS_UNOBSERVED = "unobserved"
STATUS_UNMAPPED = "unmapped"

# Sentinel text fields on the ABS Catalonia-total rows.
SENTINEL_LABEL = "No disponible"

POP_SOURCE_MUNICIPAL_AUDITED = "fl_population_por_municipis"
POP_SOURCE_COMARCA_MEMBER_SUM = "member_municipality_sum"

ABS_POPULATION_NOTE = (
    "poblacio in the ABS CSV is a per-(sex, age)-stratum case-only population; "
    "registry population is None and the audited ABS denominator is assigned "
    "at native_graph assembly (plan sections 10.3(4) and 11.3)"
)


@dataclass(frozen=True)
class SourceRecord:
    """One native source's registry entry (plan sections 11.2-11.3).

    ``population`` is the audited denominator when a sound one exists
    (municipalities: ``fl_population_por_municipis``; comarcas: the sum of
    member municipal populations) and ``None`` where it does not (ABS).
    ``population_source`` notes where a non-None population came from.
    ``first_date``/``last_date`` span all observation dates, not just the
    study window; mapping-only records have no observations and keep them
    ``None``.
    """

    source_type: str
    source_id: str
    status: str
    population: float | None
    first_date: date | None
    last_date: date | None
    population_source: str | None = None


class SourceRegistry:
    """Registry of :class:`SourceRecord` keyed on ``(source_type, source_id)``.

    Satisfies ``data.query_contract.SourceStatusRegistry`` structurally via
    :meth:`source_status`.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], SourceRecord] = {}

    def add(self, record: SourceRecord) -> None:
        """Insert or replace the record for ``(source_type, source_id)``."""
        self._records[(record.source_type, record.source_id)] = record

    def extend(self, records: Iterable[SourceRecord]) -> None:
        """Insert or replace every record in ``records``."""
        for record in records:
            self.add(record)

    def get(self, source_type: str, source_id: str) -> SourceRecord | None:
        """Return the record for ``(source_type, source_id)``, or None."""
        return self._records.get((source_type, source_id))

    def records_for(self, source_type: str) -> list[SourceRecord]:
        """All records of one source type, sorted by ``source_id``."""
        return sorted(
            (r for r in self._records.values() if r.source_type == source_type),
            key=lambda r: r.source_id,
        )

    def source_status(self, source_type: str, source_id: str) -> str | None:
        """Status string, or None when the pair is absent from the registry."""
        record = self.get(source_type, source_id)
        return None if record is None else record.status

    def counts_by_status(self, source_type: str) -> dict[str, int]:
        """Record count per status for one source type (status-sorted)."""
        counts: dict[str, int] = {}
        for record in self.records_for(source_type):
            counts[record.status] = counts.get(record.status, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[SourceRecord]:
        return iter(self._records.values())


@dataclass(frozen=True)
class AbsRegistryReport:
    """Reconciliation numbers for :func:`build_abs_registry`."""

    total_rows: int
    sentinel_rows_dropped: int
    real_codes: int
    observed_codes: int
    unobserved_codes: int
    abs_week_observations_in_window: int
    casos_in_window: int
    geometry_ids_provided: bool
    geometry_only_codes: tuple[str, ...]
    csv_codes_missing_geometry: tuple[str, ...]
    population_note: str


@dataclass(frozen=True)
class ComarcaRegistryReport:
    """Reconciliation numbers for :func:`build_comarca_registry`."""

    deaths_rows: int
    deaths_codes: int
    membership_codes: int
    shared_codes: int
    deaths_only_codes: tuple[str, ...]
    membership_only_codes: tuple[str, ...]
    observed_codes: int
    unobserved_codes: int
    deaths_in_window: int
    total_population: float


@dataclass(frozen=True)
class MunicipalRegistryReport:
    """Reconciliation numbers for :func:`build_municipal_registry`."""

    population_rows: int
    reference_rows: int
    matched_codes: int
    population_only_codes: tuple[str, ...]
    reference_only_codes: tuple[str, ...]
    total_population: float


def _in_window(day: date, window: Window) -> bool:
    start, end = window
    return start <= day <= end


def build_abs_registry(
    csv_path: str | Path,
    *,
    window: Window = (STUDY_WINDOW_START, STUDY_WINDOW_END),
    geometry_ids: Iterable[str | int] | None = None,
) -> tuple[SourceRegistry, AbsRegistryReport]:
    """Build the ABS registry from ``COVID-19__Persones_hospitalitzades.csv``.

    Rows are per (epi-week, ABS, sex, age group). Sentinel Catalonia-total
    rows (structural triple: ``codi_abs=="000"`` / ``codi_regio=="00"`` /
    ``"No disponible"`` text fields) are dropped and counted. The observation
    grain is one (abs, week) row with the sex/age strata ``casos`` summed
    (plan section 12.2); ``observed`` requires at least one such row whose
    ``data_inici`` week start falls inside ``window``. ``poblacio`` is never
    promoted to a registry population (see module docstring); all ABS
    populations are ``None``.

    ``geometry_ids`` (e.g. ``ABS.shp`` ``CODABSa`` codes, normalized to
    3-digit zero-padded strings) reconciles the CSV against the geometry:
    geometry-only codes become ``unmapped`` records, CSV codes without
    geometry are forced ``unmapped`` (dates retained).
    """
    frame = pd.read_csv(csv_path, dtype=str)
    total_rows = len(frame)
    sentinel_mask = (
        frame["codi_abs"].eq("000")
        | frame["codi_regio"].eq("00")
        | frame["nom_regio"].eq(SENTINEL_LABEL)
        | frame["nom_ambit"].eq(SENTINEL_LABEL)
        | frame["nom_abs"].eq(SENTINEL_LABEL)
    )
    sentinel_rows_dropped = int(sentinel_mask.sum())
    frame = frame.loc[~sentinel_mask].copy()
    frame["week_start"] = pd.to_datetime(frame["data_inici"], format="%d/%m/%Y").dt.date
    frame["casos_n"] = pd.to_numeric(frame["casos"], errors="raise").astype(int)

    weekly = frame.groupby(["codi_abs", "week_start"], sort=True)["casos_n"].sum()

    registry = SourceRegistry()
    codes = sorted(weekly.index.get_level_values("codi_abs").unique())
    for code in codes:
        weeks = weekly.loc[code].index
        in_window = [week for week in weeks if _in_window(week, window)]
        status = STATUS_OBSERVED if in_window else STATUS_UNOBSERVED
        registry.add(
            SourceRecord(
                source_type=SOURCE_ABS,
                source_id=code,
                status=status,
                population=None,
                first_date=min(weeks),
                last_date=max(weeks),
            )
        )

    abs_weeks_in_window = sum(
        1 for _code, week in weekly.index if _in_window(week, window)
    )
    casos_in_window = int(
        sum(
            value for (_code, week), value in weekly.items() if _in_window(week, window)
        )
    )

    geometry_ids_provided = geometry_ids is not None
    geometry_only: tuple[str, ...] = ()
    csv_missing: tuple[str, ...] = ()
    if geometry_ids is not None:
        geometry_set = {str(code).strip().zfill(3) for code in geometry_ids}
        csv_set = set(codes)
        for code in sorted(geometry_set - csv_set):
            registry.add(
                SourceRecord(
                    source_type=SOURCE_ABS,
                    source_id=code,
                    status=STATUS_UNMAPPED,
                    population=None,
                    first_date=None,
                    last_date=None,
                )
            )
        geometry_only = tuple(sorted(geometry_set - csv_set))
        csv_missing = tuple(sorted(csv_set - geometry_set))
        for code in csv_missing:
            registry.add(
                replace(registry.get(SOURCE_ABS, code), status=STATUS_UNMAPPED)
            )

    counts = registry.counts_by_status(SOURCE_ABS)
    report = AbsRegistryReport(
        total_rows=total_rows,
        sentinel_rows_dropped=sentinel_rows_dropped,
        real_codes=len(codes),
        observed_codes=counts.get(STATUS_OBSERVED, 0),
        unobserved_codes=counts.get(STATUS_UNOBSERVED, 0),
        abs_week_observations_in_window=abs_weeks_in_window,
        casos_in_window=casos_in_window,
        geometry_ids_provided=geometry_ids_provided,
        geometry_only_codes=geometry_only,
        csv_codes_missing_geometry=csv_missing,
        population_note=ABS_POPULATION_NOTE,
    )
    return registry, report


def _read_mpis_membership(
    mpis_csv_path: str | Path,
) -> dict[str, list[str]]:
    """Comarca -> member 6-digit municipal codes from ``mpiscatalunya.csv``.

    The file has three preamble lines before the header and a trailing comma
    in the header row (one extra empty column); codes are whitespace-stripped.
    """
    mpis = pd.read_csv(mpis_csv_path, skiprows=3, dtype=str, encoding="utf-8-sig")
    membership: dict[str, list[str]] = {}
    for muni, comarca in zip(
        mpis["Codi"].str.strip(), mpis["Codi comarca"].str.strip(), strict=True
    ):
        membership.setdefault(comarca, []).append(muni)
    return membership


def _read_population_bridge(
    population_csv_path: str | Path, reference_csv_path: str | Path
) -> tuple[dict[str, str], dict[str, float]]:
    """Six-to-five digit municipal code map and 5-digit-keyed populations.

    ``municipality_id_reference.csv`` bridges the mpis 6-digit codes to
    5-digit codes (verified transform: ``muni_code_5digit ==
    muni_code_6digit[:5]``); ``fl_population_por_municipis.csv`` keys
    population on the 5-digit ``id``.
    """
    reference = pd.read_csv(reference_csv_path, dtype=str)
    six_to_five = dict(
        zip(
            reference["muni_code_6digit"].str.strip(),
            reference["muni_code_5digit"].str.strip(),
            strict=True,
        )
    )
    population = pd.read_csv(population_csv_path, dtype={"id": str})
    pop_by_id = dict(
        zip(
            population["id"].astype(str).str.strip(),
            pd.to_numeric(population["d.population"], errors="raise"),
            strict=True,
        )
    )
    return six_to_five, pop_by_id


def build_comarca_registry(
    deaths_csv_path: str | Path,
    mpis_csv_path: str | Path,
    population_csv_path: str | Path,
    reference_csv_path: str | Path,
    *,
    window: Window = (STUDY_WINDOW_START, STUDY_WINDOW_END),
) -> tuple[SourceRegistry, ComarcaRegistryReport]:
    """Build the comarca registry from the deaths CSV + mpis membership.

    Deaths are summed per (comarca, date) over all sex rows, including the
    unclassified-sex rows (``Codi Sexe`` is NaN there, so it is never keyed
    on). Membership and the population denominator (sum of member municipal
    populations) come from ``mpiscatalunya.csv`` bridged through
    ``municipality_id_reference.csv`` to ``fl_population_por_municipis.csv``.
    Deaths-only codes (real data: ``"00"`` = "No classificat", not a comarca)
    are ``unmapped`` with their observation dates retained and population
    ``None``; membership-only codes (real data: ``"43"``, zero death rows)
    are ``unmapped`` with no dates but an audited member-sum population.
    """
    deaths = pd.read_csv(deaths_csv_path, dtype=str)
    deaths["date"] = pd.to_datetime(deaths["Data defunció"], format="%d/%m/%Y").dt.date
    deaths["n"] = pd.to_numeric(deaths["Nombre defuncions"], errors="raise").astype(int)
    daily = deaths.groupby(["Codi Comarca", "date"], sort=True)["n"].sum()

    membership = _read_mpis_membership(mpis_csv_path)
    six_to_five, pop_by_id = _read_population_bridge(
        population_csv_path, reference_csv_path
    )

    def comarca_population(comarca: str) -> float:
        total = 0.0
        for member in membership.get(comarca, []):
            five = six_to_five.get(member)
            if five is not None:
                total += pop_by_id.get(five, 0.0)
        return float(total)

    deaths_codes = set(daily.index.get_level_values("Codi Comarca"))
    membership_codes = set(membership)

    registry = SourceRegistry()
    for code in sorted(deaths_codes | membership_codes):
        shared = code in deaths_codes and code in membership_codes
        if shared:
            dates = daily.loc[code].index
            observed = any(_in_window(day, window) for day in dates)
            registry.add(
                SourceRecord(
                    source_type=SOURCE_COMARCA,
                    source_id=code,
                    status=STATUS_OBSERVED if observed else STATUS_UNOBSERVED,
                    population=comarca_population(code),
                    first_date=min(dates),
                    last_date=max(dates),
                    population_source=POP_SOURCE_COMARCA_MEMBER_SUM,
                )
            )
        elif code in deaths_codes:
            dates = daily.loc[code].index
            registry.add(
                SourceRecord(
                    source_type=SOURCE_COMARCA,
                    source_id=code,
                    status=STATUS_UNMAPPED,
                    population=None,
                    first_date=min(dates),
                    last_date=max(dates),
                )
            )
        else:
            registry.add(
                SourceRecord(
                    source_type=SOURCE_COMARCA,
                    source_id=code,
                    status=STATUS_UNMAPPED,
                    population=comarca_population(code),
                    first_date=None,
                    last_date=None,
                    population_source=POP_SOURCE_COMARCA_MEMBER_SUM,
                )
            )

    records = registry.records_for(SOURCE_COMARCA)
    report = ComarcaRegistryReport(
        deaths_rows=len(deaths),
        deaths_codes=len(deaths_codes),
        membership_codes=len(membership_codes),
        shared_codes=len(deaths_codes & membership_codes),
        deaths_only_codes=tuple(sorted(deaths_codes - membership_codes)),
        membership_only_codes=tuple(sorted(membership_codes - deaths_codes)),
        observed_codes=sum(r.status == STATUS_OBSERVED for r in records),
        unobserved_codes=sum(r.status == STATUS_UNOBSERVED for r in records),
        deaths_in_window=int(
            sum(
                value
                for (_code, day), value in daily.items()
                if _in_window(day, window)
            )
        ),
        total_population=float(
            sum(r.population for r in records if r.population is not None)
        ),
    )
    return registry, report


def build_municipal_registry(
    population_csv_path: str | Path,
    reference_csv_path: str | Path,
) -> tuple[SourceRegistry, MunicipalRegistryReport]:
    """Build the municipal registry from population ∩ reference codes.

    REG-1 has no municipal observation input, so matched municipalities are
    ``unobserved`` with no dates; downstream slices refine their statuses from
    the observation stores. Either-side-only codes become ``unmapped``
    records (real files: none — the join is 947/947 exact).
    """
    population = pd.read_csv(population_csv_path, dtype={"id": str})
    ids = population["id"].astype(str).str.strip()
    pop_by_id = dict(
        zip(
            ids,
            pd.to_numeric(population["d.population"], errors="raise"),
            strict=True,
        )
    )
    reference = pd.read_csv(reference_csv_path, dtype=str)
    reference_ids = set(reference["muni_code_5digit"].str.strip())

    registry = SourceRegistry()
    population_set = set(ids)
    for code in sorted(population_set & reference_ids):
        registry.add(
            SourceRecord(
                source_type=SOURCE_MUNICIPALITY,
                source_id=code,
                status=STATUS_UNOBSERVED,
                population=float(pop_by_id[code]),
                first_date=None,
                last_date=None,
                population_source=POP_SOURCE_MUNICIPAL_AUDITED,
            )
        )
    for code in sorted(population_set - reference_ids):
        registry.add(
            SourceRecord(
                source_type=SOURCE_MUNICIPALITY,
                source_id=code,
                status=STATUS_UNMAPPED,
                population=float(pop_by_id[code]),
                first_date=None,
                last_date=None,
                population_source=POP_SOURCE_MUNICIPAL_AUDITED,
            )
        )
    for code in sorted(reference_ids - population_set):
        registry.add(
            SourceRecord(
                source_type=SOURCE_MUNICIPALITY,
                source_id=code,
                status=STATUS_UNMAPPED,
                population=None,
                first_date=None,
                last_date=None,
            )
        )

    records = registry.records_for(SOURCE_MUNICIPALITY)
    report = MunicipalRegistryReport(
        population_rows=len(population),
        reference_rows=len(reference),
        matched_codes=len(population_set & reference_ids),
        population_only_codes=tuple(sorted(population_set - reference_ids)),
        reference_only_codes=tuple(sorted(reference_ids - population_set)),
        total_population=float(
            sum(r.population for r in records if r.population is not None)
        ),
    )
    return registry, report
