"""Native-support query dataset (plan sections 11.2, 11.4, 11.5; slice DS-1).

Indexes per-head ``(source, origin)`` forecast queries over the native
support artifacts and yields one item per query — the ``NativeEpiBatch``
field set of section 11.4 minus collation (BAT-1 owns collation). Reads
only the native root (``native_graph.zarr``, ``obs_admissions.zarr``,
``obs_deaths.zarr``, optional ``od_mitma.zarr``) plus
``real_with_id.zarr``.

Window geometry. The shared daily grid is the ``date`` coordinate of
``real_with_id.zarr`` (435 days, 2020-03-01..2021-05-09). A query with
origin ``o`` covers ``L`` history days ``[o-L+1, o]`` (nowcast day
included, mirroring the municipal path) and ``H`` target days
``(o, o+H]`` — the same open interval the weekly reporting operator
(:mod:`models.reporting_operators`) aggregates.

Eligibility. A ``(source, origin)`` pair is indexed when the source is
registry-``observed`` for its routed head, both windows clear the
minimum-observation counts of ``DataConfig._resolve_min_observed_map``
(reused verbatim) at the head's native cadence: cases/deaths count
observed days (cadence 1); the weekly ``hosp`` head counts complete
reported Mon-Sun weeks (cadence 7) fully inside each window. With
``confirmation_guard`` on (default, ruling R9) an origin is additionally
rejected when its target window ``(origin, origin+H]`` intersects the
reserved ``CONFIRMATION_WINDOW`` (plan section 10.3(10)); an explicit
``confirmation_guard=False`` opts in for calibration-period work.

Histories ``hist [L, F_native]`` — three clinical channels (value, mask,
age) followed by the ``k`` shared temporal covariates. VALUE channels are
type-aware consumption transforms (ruling R10) of raw-at-rest stores:
masks and ages are raw, unreported days keep value 0, and TARGETS STAY
RAW COUNTS (the loss owns target-side transforms):

- municipality: daily cases from ``real_with_id.zarr`` as stored —
  ``log1p(per-100k)`` float16, already transformed by provenance — with
  cases mask and age.
- abs: weekly admissions on the daily grid with interval placement —
  the weekly value and its reporting age sit on the week-start (Monday)
  day only; all other days carry value 0 with mask 0. The value channel
  is ``log1p((weekly_count / pop_graph) · 1e5)`` computed from the raw
  stored count and the ABS graph population.
- comarca: daily deaths from ``obs_deaths.zarr`` (raw counts at rest,
  plan section 11.3); the value channel is
  ``log1p((daily_count / pop_graph) · 1e5)``; mask and age are raw.

Targets are raw counts plus the graph population denominator: municipal
cases are inverted from the stored ``log1p(per-100k)`` values at index
time; deaths and weekly admissions are already raw. The weekly ``hosp``
block additionally carries ``target_stratum_pop_sum`` (per-week QC-only
lower bound, section 10.3(4)) beside ``target_population_graph``.

Context ``ctx_feat [L, D_ctx]``. Fixed pooling over the query source's
municipal overlap neighbors from ``native_graph.zarr`` edges, taken in
the ``(src=municipality, dst=query source)`` direction. Weights are the
RAW destination-population fractions, NEVER renormalized (ruling R6):
they sum to ``ctx_coverage <= 1`` (the population share reachable
through municipal overlaps) and the uncovered deficit acts as an
explicit zero-valued slot — a sole ``0.6``-weight neighbor contributes
``0.6 × feature`` and ``0.4 × zeros`` — so ``ctx_coverage`` keeps
reporting the deficit instead of the pooling faking completeness.
Municipal query sources have no municipal overlap neighbors: their
context is zero with coverage 0 (Stage-2 property; ENC-1 replaces fixed
pooling). With ``include_od_context`` (default off) one extra constant
feature column carries the origin-day incoming trip share into the
source's overlapping MITMA zones: the sum of ``od_mitma`` trips
targeting those zones at the origin day, divided by the total incoming
trips over all zones that day, after the consumption-time OD policy
(plan sections 11.6-11.7): flows below ``od_min_flow_raw`` are dropped
and ``od_self_flows="drop"`` clears the diagonal before the shares are
taken.

Deterministic ordering: items sort by ``(source_type, source_id,
forecast_origin)`` and identical ``(source, origin)`` pairs appear once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np
import torch
import xarray as xr
import zarr
from torch.utils.data import Dataset

from data.native_registry import (
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_MUNICIPALITY,
    STATUS_OBSERVED,
    STATUS_UNMAPPED,
    STATUS_UNOBSERVED,
    SourceRecord,
    SourceRegistry,
)
from data.query_contract import ForecastQuery, validate_query
from models.configs import DataConfig, NATIVE_OD_SELF_FLOW_MODES

#: Stable enum index per source type (BAT-1 collation consumes this).
SOURCE_TYPE_IDS: dict[str, int] = {
    "municipality": 0,
    "abs": 1,
    "comarca": 2,
    "mitma": 3,
    "edar": 4,
}

#: Routed observation head -> the single source type that carries it.
HEAD_SOURCE_TYPE: dict[str, str] = {
    "cases": SOURCE_MUNICIPALITY,
    "hosp": SOURCE_ABS,
    "deaths": SOURCE_COMARCA,
}

#: Native head name -> key in ``TARGET_OBSERVATION_CADENCE_DAYS``.
_LEGACY_HEAD_KEY = {"cases": "cases", "hosp": "hospitalizations", "deaths": "deaths"}

_DAYS_PER_WEEK = 7
_PER_100K = 100_000.0
_POOL_RELATIONS = ("muni_abs", "muni_comarca")
_OD_RELATIONS = ("mitma_abs", "mitma_comarca")

#: Reserved model-confirmation window (plan section 10.3(10), ruling R9).
#: [D1] used 2020-12-31..2021-01-27 for calibration only, so native
#: evaluation must not score targets inside it unless an explicit
#: ``confirmation_guard=False`` override opts in. Inclusive on both ends.
CONFIRMATION_WINDOW: tuple[np.datetime64, np.datetime64] = (
    np.datetime64("2020-12-31", "D"),
    np.datetime64("2021-01-27", "D"),
)


def _day_of_week(date: np.datetime64) -> int:
    """Weekday (0=Monday..6=Sunday) of a ``datetime64[D]`` value."""
    return (int(date.astype("datetime64[D]").astype(np.int64)) + 3) % _DAYS_PER_WEEK


@dataclass
class NativeDatasetConfig:
    """Window and eligibility configuration for :class:`NativeSupportDataset`.

    Fractions mirror ``DataConfig``'s uniform observed-fraction semantics;
    native cadences are applied per head at resolution time.
    """

    input_window_length: int = 60
    forecast_horizon: int = 28
    window_stride: int = 1
    min_history_observed_fraction: float = 0.5
    min_horizon_observed_fraction: float = 0.5
    # OD summary context row (plan section 11.4 D_ctx extension); default off.
    include_od_context: bool = False
    # Consumption-time OD policy applied when ``include_od_context`` is on
    # (plan sections 11.6-11.7, ruling R10).
    od_min_flow_raw: float = 10.0
    od_self_flows: str = "retain"
    # Ruling R9: reject origins whose target window intersects
    # ``CONFIRMATION_WINDOW`` unless explicitly disabled for calibration work.
    confirmation_guard: bool = True

    def __post_init__(self) -> None:
        if self.input_window_length <= 0 or self.forecast_horizon <= 0:
            raise ValueError("window lengths must be positive")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        for name in (
            "min_history_observed_fraction",
            "min_horizon_observed_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.od_self_flows not in NATIVE_OD_SELF_FLOW_MODES:
            raise ValueError(
                f"od_self_flows must be one of {NATIVE_OD_SELF_FLOW_MODES}, "
                f"got {self.od_self_flows!r}"
            )
        if (
            not math.isfinite(float(self.od_min_flow_raw))
            or self.od_min_flow_raw < 0
        ):
            raise ValueError("od_min_flow_raw must be finite and non-negative")

    def _min_observed(self, window_length: int, fraction: float, head: str) -> int:
        """``DataConfig._resolve_min_observed_map`` at the head's cadence."""
        resolved = DataConfig._resolve_min_observed_map(window_length, fraction)
        return resolved[_LEGACY_HEAD_KEY[head]]

    def min_history_observed(self, head: str) -> int:
        """Minimum observed opportunities inside the ``L``-day history."""
        return self._min_observed(
            self.input_window_length, self.min_history_observed_fraction, head
        )

    def min_horizon_observed(self, head: str) -> int:
        """Minimum observed opportunities inside the ``H``-day horizon."""
        return self._min_observed(
            self.forecast_horizon, self.min_horizon_observed_fraction, head
        )


class NativeDatasetItem(TypedDict):
    """One query's item — ``NativeEpiBatch`` fields minus collation."""

    query: ForecastQuery
    head: str
    hist: torch.Tensor  # [L, F_native] float32; channels documented above
    hist_mask: torch.Tensor  # [L] bool
    ctx_feat: torch.Tensor  # [L, D_ctx] float32
    ctx_coverage: float
    temporal_covariates: torch.Tensor  # [L, k] float32
    source_type_id: int
    source_pop: float
    origin_dow: int
    target: torch.Tensor  # [H] daily (cases/deaths) | [W] weekly (hosp)
    target_mask: torch.Tensor  # same shape as target
    target_population_graph: float
    target_stratum_pop_sum: torch.Tensor | None  # [W] float32, hosp only


class _GraphNodes(NamedTuple):
    """Node table of ``native_graph.zarr`` plus the OD-zone reference."""

    source_id: np.ndarray  # [node] str
    node_type: np.ndarray  # [node] str
    status: np.ndarray  # [node] str
    population: np.ndarray  # [node] float32
    od_zone_ref: np.ndarray  # [od_zone] str


class _ObsSeries:
    """Per-source observation series aligned to their own date coordinate."""

    def __init__(
        self,
        dates: np.ndarray,
        source_ids: list[str],
        values: np.ndarray,
        masks: np.ndarray,
        ages: np.ndarray,
    ) -> None:
        self.dates = dates
        self.source_ids = source_ids
        self.values = values
        self.masks = masks
        self.ages = ages
        self.columns = {source_id: i for i, source_id in enumerate(source_ids)}
        self._rows = {date: i for i, date in enumerate(dates)}

    def row_of(self, date: np.datetime64) -> int | None:
        return self._rows.get(date.astype("datetime64[D]"))


class _CasesSeries(_ObsSeries):
    """``real_with_id`` cases block: 3-channel series + shared covariates."""

    def __init__(
        self,
        dates: np.ndarray,
        region_ids: list[str],
        channels: np.ndarray,
        masks: np.ndarray,
        covariates: np.ndarray,
    ) -> None:
        super().__init__(dates, region_ids, channels[:, :, 0], masks, channels[:, :, 2])
        self.channels = channels  # [T, M, 3] float32 value/mask/age
        self.covariates = covariates  # [T, k] float32


class _AdmSeries(_ObsSeries):
    """Weekly admissions with the QC stratum-population column."""

    def __init__(
        self,
        dates: np.ndarray,
        source_ids: list[str],
        values: np.ndarray,
        masks: np.ndarray,
        ages: np.ndarray,
        stratum: np.ndarray,
    ) -> None:
        super().__init__(dates, source_ids, values, masks, ages)
        self.stratum = stratum


class _OdStore(NamedTuple):
    """``od_mitma.zarr``: daily raw OD trips over the MITMA zone set."""

    dates: np.ndarray  # [date] datetime64[D]
    zone_ids: list[str]
    mobility: np.ndarray  # [date, origin, target] float32

    def row_of(self, date: np.datetime64) -> int | None:
        rows = np.flatnonzero(self.dates == date.astype("datetime64[D]"))
        return int(rows[0]) if rows.size else None


def _align_daily(
    store: _ObsSeries, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place a daily store onto the shared grid: value/mask/age ``[T, S]``.

    Store dates absent from the grid contribute nothing; grid days absent
    from the store carry mask 0 (plan section 11.7 absent-row policy).
    """
    rows = {date: i for i, date in enumerate(grid)}
    values = np.zeros((len(grid), len(store.source_ids)), dtype=np.float32)
    masks = np.zeros((len(grid), len(store.source_ids)), dtype=bool)
    ages = np.zeros((len(grid), len(store.source_ids)), dtype=np.float32)
    for row, date in enumerate(store.dates):
        grid_row = rows.get(date)
        if grid_row is not None:
            values[grid_row] = store.values[row]
            masks[grid_row] = store.masks[row]
            ages[grid_row] = store.ages[row]
    return values, masks, ages


class NativeSupportDataset(Dataset):
    """Per-head ``(source, origin)`` forecast-query dataset over native stores."""

    def __init__(
        self,
        native_root: str | Path,
        real_with_id_path: str | Path,
        config: NativeDatasetConfig,
    ) -> None:
        self._config = config
        root = Path(native_root)
        self._graph_nodes, self._node_index = self._load_graph_nodes(root)
        needed_rels = _POOL_RELATIONS + (
            _OD_RELATIONS if config.include_od_context else ()
        )
        self._edges = self._load_graph_edges(root, needed_rels)
        self._od = self._load_od_store(root) if config.include_od_context else None
        self._adm = self._load_admissions(root / "obs_admissions.zarr")
        self._deaths = self._load_store(root / "obs_deaths.zarr", "comarca_id")
        self._cases = self._load_cases(real_with_id_path)
        self._grid = self._cases.dates
        # Deaths aligned onto the shared grid (absent dates -> mask 0).
        self._deaths_values, self._deaths_masks, self._deaths_ages = _align_daily(
            self._deaths, self._grid
        )
        self._L = config.input_window_length
        self._H = config.forecast_horizon
        if self._od is not None:
            self._od_zone_index = {
                zone: i for i, zone in enumerate(self._graph_nodes.od_zone_ref)
            }
        self._registry = self._build_registry()
        self._pools = self._build_pools()
        self._items = self._build_index()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def config(self) -> NativeDatasetConfig:
        return self._config

    @property
    def registry(self) -> SourceRegistry:
        """Registry derived from the stores and graph statuses (per head)."""
        return self._registry

    @property
    def queries(self) -> list[ForecastQuery]:
        """All indexed queries in deterministic ``(type, id, origin)`` order."""
        return [query for query, _ in self._items]

    @property
    def head_index(self) -> dict[str, list[tuple[str, str, int]]]:
        """Per-head ``(source_type, source_id, forecast_origin)`` index."""
        index: dict[str, list[tuple[str, str, int]]] = {}
        for query, head in self._items:
            index.setdefault(head, []).append(
                (query.source_type, query.source_id, query.forecast_origin)
            )
        return index

    @property
    def head_counts(self) -> dict[str, int]:
        """Items per head — the section 7 sampling audit."""
        counts: dict[str, int] = {}
        for _, head in self._items:
            counts[head] = counts.get(head, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> NativeDatasetItem:
        query, head = self._items[idx]
        origin = query.forecast_origin
        hist_slice = slice(origin - self._L + 1, origin + 1)
        target_slice = slice(origin + 1, origin + self._H + 1)
        covariates = self._cases.covariates[hist_slice].astype(np.float32)
        stratum: np.ndarray | None = None

        if query.source_type == SOURCE_MUNICIPALITY:
            hist, hist_mask = self._municipal_history(query.source_id, hist_slice)
            target, target_mask = self._municipal_target(query.source_id, target_slice)
        elif query.source_type == SOURCE_ABS:
            hist, hist_mask = self._abs_history(query.source_id, hist_slice)
            target, target_mask, stratum = self._abs_target(query.source_id, origin)
        else:
            hist, hist_mask = self._comarca_history(query.source_id, hist_slice)
            target, target_mask = self._comarca_target(query.source_id, target_slice)
        hist = np.concatenate([hist, covariates], axis=1)

        pooled, coverage = self._pooled_context(
            query.source_type, query.source_id, hist_slice
        )
        if self._od is not None:
            od_column = np.full((self._L, 1), self._od_share(query), dtype=np.float32)
            pooled = np.concatenate([pooled, od_column], axis=1)

        node = self._node_index[f"{query.source_type}:{query.source_id}"]
        population = float(self._graph_nodes.population[node])
        return {
            "query": query,
            "head": head,
            "hist": torch.from_numpy(hist),
            "hist_mask": torch.from_numpy(hist_mask),
            "ctx_feat": torch.from_numpy(pooled),
            "ctx_coverage": coverage,
            "temporal_covariates": torch.from_numpy(covariates),
            "source_type_id": SOURCE_TYPE_IDS[query.source_type],
            "source_pop": population,
            "origin_dow": _day_of_week(self._grid[origin]),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "target_population_graph": population,
            "target_stratum_pop_sum": (
                None if stratum is None else torch.from_numpy(stratum)
            ),
        }

    # ------------------------------------------------------------------
    # Registry and index construction
    # ------------------------------------------------------------------

    def _load_graph_nodes(self, root: Path) -> tuple[_GraphNodes, dict[str, int]]:
        ds = xr.open_zarr(root / "native_graph.zarr").compute()
        source_id = np.asarray(ds["source_id"].values).astype(str)
        node_type = np.asarray(ds["node_type"].values).astype(str)
        status = np.asarray(ds["status"].values).astype(str)
        population = np.asarray(ds["population"].values, dtype=np.float32)
        od_zone_ref = (
            np.asarray(ds["od_zone_ref"].values).astype(str)
            if "od_zone_ref" in ds
            else np.empty(0, dtype="<U1")
        )
        nodes = _GraphNodes(source_id, node_type, status, population, od_zone_ref)
        index = {f"{t}:{s}": i for i, (t, s) in enumerate(zip(node_type, source_id))}
        return nodes, index

    def _load_graph_edges(
        self, root: Path, relations: tuple[str, ...]
    ) -> dict[str, dict[str, np.ndarray]]:
        # Consolidated metadata predates the edges subgroup (it is appended
        # after the xarray node-table write), so it must be bypassed here.
        group = cast(
            zarr.Group,
            zarr.open_group(
                str(root / "native_graph.zarr"), mode="r", use_consolidated=False
            )["edges"],
        )
        edges: dict[str, dict[str, np.ndarray]] = {}
        for relation in relations:
            sub = cast(zarr.Group, group[relation])  # KeyError -> incomplete graph
            edges[relation] = {
                name: np.asarray(cast(zarr.Array, sub[name])[:])
                for name in ("src_idx", "dst_idx", "weight")
            }
        return edges

    def _load_od_store(self, root: Path) -> _OdStore:
        ds = xr.open_zarr(root / "od_mitma.zarr")
        mobility = np.asarray(ds["mobility"].values, dtype=np.float32)
        dates = np.asarray(ds["date"].values).astype("datetime64[D]")
        zone_ids = [str(z) for z in ds["zone_id"].values]
        return _OdStore(dates, zone_ids, mobility)

    def _load_admissions(self, path: Path) -> _AdmSeries:
        """Load the weekly admissions store; ``stratum_pop_sum`` is required."""
        loaded = self._load_store(path, "abs_id")
        if not isinstance(loaded, _AdmSeries):
            raise ValueError(f"{path} is missing the stratum_pop_sum variable")
        return loaded

    def _load_store(self, path: Path, dim_name: str) -> _ObsSeries | _AdmSeries:
        ds = xr.open_zarr(path).compute()
        dates = np.asarray(ds["date"].values).astype("datetime64[D]")
        source_ids = [str(s) for s in ds[dim_name].values]
        values = np.asarray(ds["value"].values, dtype=np.float32)
        masks = np.asarray(ds["mask"].values, dtype=bool)
        ages = np.asarray(ds["age"].values, dtype=np.float32)
        order = np.argsort(dates, kind="stable")
        dates, values, masks, ages = (
            dates[order],
            values[order],
            masks[order],
            ages[order],
        )
        if "stratum_pop_sum" in ds:
            stratum = np.asarray(ds["stratum_pop_sum"].values, dtype=np.float32)[order]
            return _AdmSeries(dates, source_ids, values, masks, ages, stratum)
        return _ObsSeries(dates, source_ids, values, masks, ages)

    def _load_cases(self, real_with_id_path: str | Path) -> _CasesSeries:
        ds = xr.open_zarr(real_with_id_path)
        needed = ["cases", "cases_mask", "cases_age", "temporal_covariates"]
        ds = ds[needed]
        if "run_id" in ds.dims:
            run_ids = [str(r).strip() for r in ds["run_id"].values]
            ds = ds.isel(run_id=run_ids.index("real") if "real" in run_ids else 0)
        ds = ds.compute()
        dates = np.asarray(ds["date"].values).astype("datetime64[D]")
        region_ids = [str(r) for r in ds["region_id"].values]
        channels = np.stack(
            [
                np.asarray(ds["cases"].values, dtype=np.float32),
                np.asarray(ds["cases_mask"].values, dtype=np.float32),
                np.asarray(ds["cases_age"].values, dtype=np.float32),
            ],
            axis=-1,
        )
        masks = np.asarray(ds["cases_mask"].values, dtype=bool)
        covariates = np.asarray(ds["temporal_covariates"].values, dtype=np.float32)
        return _CasesSeries(dates, region_ids, channels, masks, covariates)

    def _build_registry(self) -> SourceRegistry:
        """Derive per-head source statuses from the stores + graph statuses."""
        registry = SourceRegistry()
        for i, source_id in enumerate(self._cases.source_ids):
            node = self._node_index.get(f"{SOURCE_MUNICIPALITY}:{source_id}")
            if node is None or not math.isfinite(
                float(self._graph_nodes.population[node])
            ):
                continue  # no audited graph denominator -> never queried
            observed = bool(self._cases.masks[:, i].any())
            registry.add(
                SourceRecord(
                    source_type=SOURCE_MUNICIPALITY,
                    source_id=source_id,
                    status=STATUS_OBSERVED if observed else STATUS_UNOBSERVED,
                    population=float(self._graph_nodes.population[node]),
                    first_date=None,
                    last_date=None,
                )
            )
        for store, source_type in (
            (self._adm, SOURCE_ABS),
            (self._deaths, SOURCE_COMARCA),
        ):
            for j, source_id in enumerate(store.source_ids):
                node = self._node_index.get(f"{source_type}:{source_id}")
                if (
                    node is None
                    or str(self._graph_nodes.status[node]) == STATUS_UNMAPPED
                ):
                    status = STATUS_UNMAPPED
                elif not bool(store.masks[:, j].any()):
                    status = STATUS_UNOBSERVED
                else:
                    status = STATUS_OBSERVED
                population = (
                    float(self._graph_nodes.population[node])
                    if node is not None
                    else None
                )
                registry.add(
                    SourceRecord(
                        source_type=source_type,
                        source_id=source_id,
                        status=status,
                        population=(
                            population
                            if population is not None and math.isfinite(population)
                            else None
                        ),
                        first_date=None,
                        last_date=None,
                    )
                )
        return registry

    def _origin_grid(self) -> np.ndarray:
        """Admissible forecast origins on the shared grid for ``L + H`` windows."""
        return np.arange(
            self._L - 1, len(self._grid) - self._H, self._config.window_stride
        )

    def _eligibility_counts(
        self, head: str, source_ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Observed-opportunity counts per (source, origin), ``[S, O]`` each.

        Daily heads (cadence 1) count observed days; the weekly ``hosp``
        head counts complete reported Mon-Sun weeks fully inside each
        window — ``DataConfig._resolve_min_observed_map`` semantics at the
        head's native cadence.
        """
        origins = self._origin_grid()
        if head == "hosp":
            dates = self._adm.dates
            columns = np.array([self._adm.columns[s] for s in source_ids])
            cum = np.vstack(
                [
                    np.zeros((1, self._adm.masks.shape[1])),
                    np.cumsum(self._adm.masks, axis=0),
                ]
            )[:, columns]

            def week_bounds(
                a: np.ndarray, b: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
                """Week-index range of Mon-Sun weeks fully inside ``[a, b]``."""
                lo = np.searchsorted(dates, self._grid[a], side="left")
                hi = np.searchsorted(
                    dates,
                    self._grid[b] - np.timedelta64(_DAYS_PER_WEEK - 1, "D"),
                    side="right",
                )
                return lo, hi

            h_lo, h_hi = week_bounds(origins - self._L + 1, origins)
            t_lo, t_hi = week_bounds(origins + 1, origins + self._H)
            hist = (cum[h_hi] - cum[h_lo]).T
            target = (cum[t_hi] - cum[t_lo]).T
            return hist, target
        masks = self._cases.masks if head == "cases" else self._deaths_masks
        store = self._cases if head == "cases" else self._deaths
        columns = np.array([store.columns[s] for s in source_ids])
        cum = np.vstack([np.zeros((1, masks.shape[1])), np.cumsum(masks, axis=0)])
        cum = cum[:, columns]
        hist = (cum[origins + 1] - cum[origins - self._L + 1]).T
        target = (cum[origins + self._H + 1] - cum[origins + 1]).T
        return hist, target

    def _reserved_origin_mask(self, origins: np.ndarray) -> np.ndarray:
        """Origins whose target window ``(origin, origin+H]`` intersects
        ``CONFIRMATION_WINDOW`` (plan section 10.3(10), ruling R9)."""
        win_start, win_end = CONFIRMATION_WINDOW
        target_start = self._grid[origins + 1]
        target_end = self._grid[origins + self._H]
        return (target_start <= win_end) & (target_end >= win_start)

    def _build_index(self) -> list[tuple[ForecastQuery, str]]:
        """Deterministic per-head ``(source, origin)`` index (sorted, deduped)."""
        origins = self._origin_grid()
        reserved = (
            self._reserved_origin_mask(origins)
            if self._config.confirmation_guard
            else None
        )
        entries: list[tuple[ForecastQuery, str]] = []
        for head in sorted(HEAD_SOURCE_TYPE):
            source_type = HEAD_SOURCE_TYPE[head]
            records = [
                record
                for record in self._registry.records_for(source_type)
                if record.status == STATUS_OBSERVED
            ]
            if not records:
                continue
            hist_counts, target_counts = self._eligibility_counts(
                head, [record.source_id for record in records]
            )
            min_history = self._config.min_history_observed(head)
            min_horizon = self._config.min_horizon_observed(head)
            for row, record in enumerate(records):
                eligible = (hist_counts[row] >= min_history) & (
                    target_counts[row] >= min_horizon
                )
                if reserved is not None:
                    eligible &= ~reserved
                for offset in np.flatnonzero(eligible):
                    query = ForecastQuery(
                        source_type=source_type,  # type: ignore[arg-type]
                        source_id=record.source_id,
                        forecast_origin=int(origins[offset]),
                        horizon_days=self._H,
                        requested_heads=frozenset({head}),
                    )
                    validate_query(query, self._registry)
                    entries.append((query, head))
        entries.sort(
            key=lambda entry: (
                entry[0].source_type,
                entry[0].source_id,
                entry[0].forecast_origin,
            )
        )
        return entries

    # ------------------------------------------------------------------
    # Context pooling
    # ------------------------------------------------------------------

    def _build_pools(self) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        """Per-source (cases columns, raw destination fractions, coverage).

        Neighbors are case-bearing municipalities; the coverage scalar is
        the query source's population share reachable through municipal
        overlaps with case observations. Municipal queries have no
        municipal overlap neighbors: empty pooling, coverage 0.
        """
        node_to_cases = np.full(len(self._graph_nodes.source_id), -1, dtype=np.int64)
        for column, region_id in enumerate(self._cases.source_ids):
            node = self._node_index.get(f"{SOURCE_MUNICIPALITY}:{region_id}")
            if node is not None:
                node_to_cases[node] = column
        pools: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for source_type in (SOURCE_ABS, SOURCE_COMARCA, SOURCE_MUNICIPALITY):
            for record in self._registry.records_for(source_type):
                if f"{source_type}:{record.source_id}" not in self._node_index:
                    continue  # graph-absent sources are never queried
                pools[f"{source_type}:{record.source_id}"] = self._pool_for(
                    source_type, record.source_id, node_to_cases
                )
        return pools

    def _pool_for(
        self, source_type: str, source_id: str, node_to_cases: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        node = self._node_index[f"{source_type}:{source_id}"]
        is_muni = self._graph_nodes.node_type == SOURCE_MUNICIPALITY
        columns: list[int] = []
        weights: list[float] = []
        for relation in _POOL_RELATIONS:
            edges = self._edges[relation]
            for row in np.flatnonzero(edges["dst_idx"] == node):
                src = int(edges["src_idx"][row])
                case_column = int(node_to_cases[src])
                if is_muni[src] and case_column >= 0:
                    columns.append(case_column)
                    weights.append(float(edges["weight"][row]))
        raw = np.asarray(weights, dtype=np.float32)
        coverage = float(raw.sum())
        # Ruling R6: raw destination fractions, never renormalized — the
        # uncovered deficit stays an implicit zero-contribution slot.
        return np.asarray(columns, dtype=np.int64), raw, coverage

    def _pooled_context(
        self, source_type: str, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, float]:
        """Fixed pooling of municipal daily cases channels, ``[L, 3]``."""
        columns, weights, coverage = self._pools[f"{source_type}:{source_id}"]
        pooled = np.zeros((self._L, 3), dtype=np.float32)
        if columns.size:
            block = self._cases.channels[hist_slice][:, columns, :]  # [L, M, 3]
            pooled = (block * weights[None, :, None]).sum(axis=1, dtype=np.float32)
        return pooled.astype(np.float32, copy=False), coverage

    def _od_share(self, query: ForecastQuery) -> float:
        """Origin-day incoming trip share into the source's overlapping zones.

        Consumption-time OD policy (plan sections 11.6-11.7, ruling R10):
        flows below ``od_min_flow_raw`` are dropped and
        ``od_self_flows="drop"`` clears the diagonal (origin==target)
        before the shares are taken; the store stays raw at rest.
        """
        assert self._od is not None
        node = self._node_index[f"{query.source_type}:{query.source_id}"]
        zone_columns: set[int] = set()
        for relation in _OD_RELATIONS:
            edges = self._edges[relation]
            for row in np.flatnonzero(edges["dst_idx"] == node):
                src = int(edges["src_idx"][row])
                if self._graph_nodes.node_type[src] == "mitma":
                    zone_columns.add(
                        self._od_zone_index[str(self._graph_nodes.source_id[src])]
                    )
        if not zone_columns:
            return 0.0
        od_row = self._od.row_of(self._grid[query.forecast_origin])
        if od_row is None:
            return 0.0
        flows = self._od.mobility[od_row].astype(np.float32, copy=True)
        if self._config.od_self_flows == "drop":
            np.fill_diagonal(flows, 0.0)
        flows[flows < self._config.od_min_flow_raw] = 0.0
        columns = sorted(zone_columns)
        incoming = float(flows[:, columns].sum())
        total = float(flows.sum())
        return incoming / total if total > 0.0 else 0.0

    # ------------------------------------------------------------------
    # History and target assembly
    # ------------------------------------------------------------------

    def _municipal_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Municipal cases history, ``[L, 3]``.

        The stored ``log1p(per-100k)`` value channel is consumed as-is
        (ruling R10: already transformed by provenance); targets are
        inverted to raw counts separately.
        """
        column = self._cases.columns[source_id]
        block = self._cases.channels[hist_slice, column, :].astype(np.float32)
        return block, self._cases.masks[hist_slice, column].copy()

    def _municipal_target(
        self, source_id: str, target_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw daily counts inverted from the stored ``log1p(per-100k)``."""
        column = self._cases.columns[source_id]
        values = self._cases.channels[target_slice, column, 0]
        mask = self._cases.masks[target_slice, column]
        node = self._node_index[f"{SOURCE_MUNICIPALITY}:{source_id}"]
        population = float(self._graph_nodes.population[node])
        raw = np.expm1(values) * (population / _PER_100K)
        return np.where(mask, raw, 0.0).astype(np.float32), mask.copy()

    def _abs_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weekly admissions interval-placed on the daily grid, ``[L, 3]``.

        Consumption transform (ruling R10): the raw-at-rest weekly count
        becomes ``log1p((count / pop_graph) · 1e5)`` using the ABS graph
        population. Mask/age channels are raw; targets stay raw.
        """
        column = self._adm.columns[source_id]
        node = self._node_index[f"{SOURCE_ABS}:{source_id}"]
        population = float(self._graph_nodes.population[node])
        hist_start = int(hist_slice.start)
        start = int(
            np.searchsorted(self._adm.dates, self._grid[hist_start], side="left")
        )
        end = int(
            np.searchsorted(
                self._adm.dates, self._grid[int(hist_slice.stop) - 1], side="right"
            )
        )
        block = np.zeros((self._L, 3), dtype=np.float32)
        for row in range(start, end):
            offset = int(
                (self._adm.dates[row] - self._grid[hist_start]).astype(np.int64)
            )
            if offset < 0 or offset >= self._L or not self._adm.masks[row, column]:
                continue
            weekly_count = float(self._adm.values[row, column])
            block[offset] = (
                math.log1p(weekly_count / population * _PER_100K),
                1.0,
                float(self._adm.ages[row, column]),
            )
        return block, block[:, 1] > 0

    def _abs_target(
        self, source_id: str, origin: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weekly hosp block aligned with the ROP-1 week slots, ``[W]``."""
        column = self._adm.columns[source_id]
        num_weeks = -(-self._H // _DAYS_PER_WEEK)
        origin_dow = _day_of_week(self._grid[origin])
        first_offset = (-(origin_dow + 1)) % _DAYS_PER_WEEK
        values = np.zeros(num_weeks, dtype=np.float32)
        mask = np.zeros(num_weeks, dtype=bool)
        stratum = np.zeros(num_weeks, dtype=np.float32)
        for week in range(num_weeks):
            start_day = origin + 1 + first_offset + _DAYS_PER_WEEK * week
            if start_day + _DAYS_PER_WEEK - 1 > origin + self._H:
                continue  # incomplete Mon-Sun interval -> mask 0 (section 11.5)
            row = self._adm.row_of(self._grid[start_day])
            if row is None or not self._adm.masks[row, column]:
                continue
            values[week] = self._adm.values[row, column]
            mask[week] = True
            stratum[week] = self._adm.stratum[row, column]
        return values, mask, stratum

    def _comarca_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Daily deaths history, ``[L, 3]``.

        Consumption transform (ruling R10): the raw-at-rest count becomes
        ``log1p((count / pop_graph) · 1e5)`` using the comarca graph
        population, applied only where the mask is 1 (unreported days stay
        0). Mask/age channels are raw; targets stay raw.
        """
        column = self._deaths.columns[source_id]
        node = self._node_index[f"{SOURCE_COMARCA}:{source_id}"]
        population = float(self._graph_nodes.population[node])
        values = self._deaths_values[hist_slice, column]
        mask = self._deaths_masks[hist_slice, column]
        value = np.log1p(
            np.where(mask, values, 0.0) / population * _PER_100K
        )
        block = np.stack(
            [
                value.astype(np.float32),
                mask.astype(np.float32),
                self._deaths_ages[hist_slice, column],
            ],
            axis=-1,
        )
        return block.astype(np.float32, copy=False), mask.copy()

    def _comarca_target(
        self, source_id: str, target_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        column = self._deaths.columns[source_id]
        values = self._deaths_values[target_slice, column]
        mask = self._deaths_masks[target_slice, column]
        return np.where(mask, values, 0.0).astype(np.float32), mask.copy()
