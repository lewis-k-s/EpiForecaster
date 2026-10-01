"""Native-support query dataset (plan sections 11.2, 11.4, 11.5; slice DS-1).

Indexes per-head ``(source, origin)`` forecast queries over the native
support artifacts and yields one item per query — the ``NativeEpiBatch``
field set of section 11.4 minus collation (BAT-1 owns collation). Reads
only the native root (``native_graph.zarr``, ``obs_admissions.zarr``,
``obs_deaths_abs_excess.zarr``, optional ``obs_wastewater.zarr``, optional
``od_mitma.zarr``) plus ``real_with_id.zarr``. Native wastewater also reads
the EDAR-to-municipality catchment table.

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

Histories ``hist [L, F_native]`` contain three clinical channels (value,
mask, age), or four EDAR channels (assay-history mean, sample presence,
minimum sample age, last causal flow). By default, three municipality-only
vaccination channels (first-dose rate, observation mask, age / 14) follow;
other source types carry zeros. The ``k`` shared temporal covariates come
last. Vaccination and graph context use only days through the forecast
origin. Clinical value channels are
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
- abs: daily all-cause deaths from ``obs_deaths_abs_excess.zarr`` share the
  ABS row with weekly admissions. Both heads use the same six-channel
  history (admissions value/mask/age plus deaths value/mask/age). Death
  values are observed counts transformed to ``log1p(per-100k)``; expected
  pre-pandemic counts are carried beside targets for excess evaluation.
- edar: causal assay history in ``log1p(concentration)`` space. The target is
  the assay-specific raw concentration block with detection-limit and censor
  metadata. Catchment population comes from contribution-weighted
  municipalities; flow at the forecast target is supervision metadata only.

Targets are raw counts plus the graph population denominator: municipal
cases are inverted from the stored ``log1p(per-100k)`` values at index
time; all-cause deaths and weekly admissions are already raw. The weekly ``hosp``
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
from typing import NamedTuple, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from torch.utils.data import Dataset

from data.native_registry import (
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_EDAR,
    SOURCE_MUNICIPALITY,
    STATUS_OBSERVED,
    STATUS_UNMAPPED,
    STATUS_UNOBSERVED,
    SourceRecord,
    SourceRegistry,
)
from data.query_contract import ForecastQuery, validate_query
from models.configs import (
    DataConfig,
    NATIVE_OD_SELF_FLOW_MODES,
    NATIVE_OD_SOURCES,
    NATIVE_SPATIAL_ENCODERS,
)

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
    "deaths": SOURCE_ABS,
    "ww": SOURCE_EDAR,
}

#: Native head name -> key in ``TARGET_OBSERVATION_CADENCE_DAYS``.
_LEGACY_HEAD_KEY = {
    "cases": "cases",
    "hosp": "hospitalizations",
    "deaths": "deaths",
    "ww": "wastewater",
}

_DAYS_PER_WEEK = 7
_PER_100K = 100_000.0
_POOL_RELATIONS = ("muni_abs",)
_OD_RELATIONS = ("mitma_muni", "mitma_abs", "mitma_comarca")
_HETERO_MITMA_RELATIONS = ("mitma_muni", "mitma_abs", "mitma_comarca")
_EDAR_CONTEXT_RELATION = "muni_edar_catchment"

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
    od_graph_cache_size: int = 16
    cache_od_target_weights: bool = False
    od_self_flows: str = "retain"
    # Ruling R9: reject origins whose target window intersects
    # ``CONFIRMATION_WINDOW`` unless explicitly disabled for calibration work.
    confirmation_guard: bool = True
    spatial_encoder: str = "hetero"
    od_source: str = "mitma"
    mobility_enabled: bool = True
    wastewater_enabled: bool = False
    include_vaccination: bool = True

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
        if not math.isfinite(float(self.od_min_flow_raw)) or self.od_min_flow_raw < 0:
            raise ValueError("od_min_flow_raw must be finite and non-negative")
        if type(self.od_graph_cache_size) is not int or self.od_graph_cache_size < 1:
            raise ValueError("od_graph_cache_size must be a positive integer")
        if self.spatial_encoder not in NATIVE_SPATIAL_ENCODERS:
            raise ValueError(
                f"spatial_encoder must be one of {NATIVE_SPATIAL_ENCODERS}, "
                f"got {self.spatial_encoder!r}"
            )
        if self.od_source not in NATIVE_OD_SOURCES:
            raise ValueError(
                f"od_source must be one of {NATIVE_OD_SOURCES}, got {self.od_source!r}"
            )

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
    target_extra: NotRequired[dict[str, torch.Tensor]]
    relation_ctx: NotRequired[dict[str, torch.Tensor]]  # relation -> [L, 3]
    od_node_features: NotRequired[torch.Tensor]  # [N, 3 or 6]
    od_adjacency: NotRequired[torch.Tensor]  # [N, N]
    od_target_weights: NotRequired[torch.Tensor]  # [N]


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
        vaccination: np.ndarray | None,
    ) -> None:
        super().__init__(dates, region_ids, channels[:, :, 0], masks, channels[:, :, 2])
        self.channels = channels  # [T, M, 3] float32 value/mask/age
        self.covariates = covariates  # [T, k] float32
        self.vaccination = vaccination  # [T, M, 3] or None when disabled


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


class _DeathsSeries(_ObsSeries):
    """ABS observed all-cause deaths with a fixed expected-count baseline."""

    def __init__(
        self,
        dates: np.ndarray,
        source_ids: list[str],
        values: np.ndarray,
        masks: np.ndarray,
        ages: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        super().__init__(dates, source_ids, values, masks, ages)
        self.expected = expected


class _WastewaterSeries:
    """EDAR-level assay observations aligned to the shared daily grid."""

    def __init__(self, dataset: xr.Dataset) -> None:
        self.dates = np.asarray(dataset["date"].values).astype("datetime64[D]")
        self.source_ids = [str(value) for value in dataset["edar_id"].values]
        self.assays = [str(value) for value in dataset["assay"].values]
        self.columns = {
            source_id: index for index, source_id in enumerate(self.source_ids)
        }
        self.concentration = np.asarray(
            dataset["concentration_raw"].values, dtype=np.float32
        )
        self.detection_limit = np.asarray(
            dataset["detection_limit"].values, dtype=np.float32
        )
        self.mask = np.asarray(dataset["observed_mask"].values, dtype=bool)
        self.censor_code = np.asarray(dataset["censor_code"].values, dtype=np.uint8)
        self.history = np.asarray(dataset["history_log1p"].values, dtype=np.float32)
        self.history_available = np.asarray(
            dataset["history_available_mask"].values, dtype=bool
        )
        self.age = np.asarray(dataset["history_age_days"].values, dtype=np.float32)
        self.flow = np.asarray(dataset["flow_m3_24h"].values, dtype=np.float32)
        self.flow_mask = np.asarray(dataset["flow_mask"].values, dtype=bool)
        self.flow_history_log1p = np.full_like(self.flow, np.nan, dtype=np.float32)
        for column in range(self.flow.shape[1]):
            last = np.nan
            for row in range(self.flow.shape[0]):
                if self.flow_mask[row, column]:
                    last = np.float32(np.log1p(self.flow[row, column]))
                self.flow_history_log1p[row, column] = last


class _OdStore:
    """Lazy daily OD access with source-specific at-rest decoding."""

    def __init__(self, dataset: xr.Dataset, *, log1p_stored: bool) -> None:
        self._dataset = dataset
        self.dates = np.asarray(dataset["date"].values).astype("datetime64[D]")
        self.zone_ids = [str(zone) for zone in dataset["zone_id"].values]
        self.log1p_stored = log1p_stored
        self._rows = {date: row for row, date in enumerate(self.dates)}

    def row_of(self, date: np.datetime64) -> int | None:
        return self._rows.get(date.astype("datetime64[D]"))

    def flows_at(self, row: int) -> np.ndarray:
        flows = np.asarray(
            self._dataset["mobility"].isel(date=row).values, dtype=np.float32
        )
        if self.log1p_stored:
            flows = np.expm1(flows)
        return flows


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
        hetero_relations = (
            _HETERO_MITMA_RELATIONS
            if config.spatial_encoder == "hetero"
            and config.mobility_enabled
            and config.od_source == "mitma"
            else ()
        )
        needed_rels = tuple(
            dict.fromkeys(
                _POOL_RELATIONS
                + (
                    _OD_RELATIONS
                    if config.include_od_context and config.mobility_enabled
                    else ()
                )
                + hetero_relations
            )
        )
        self._edges = self._load_graph_edges(root, needed_rels)
        needs_od = config.mobility_enabled and (
            config.include_od_context or config.spatial_encoder == "hetero"
        )
        self._od = self._load_od_store(root, config.od_source) if needs_od else None
        self._adm = self._load_admissions(root / "obs_admissions.zarr")
        self._deaths = self._load_abs_deaths(root / "obs_deaths_abs_excess.zarr")
        self._cases = self._load_cases(real_with_id_path)
        self._grid = self._cases.dates
        if not np.array_equal(self._deaths.dates, self._grid):
            raise ValueError("ABS excess deaths date grid differs from cases")
        self._deaths_expected = self._deaths.expected
        self._ww: _WastewaterSeries | None = None
        self._edar_population: dict[str, float] = {}
        self._edar_case_pools: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self._edar_municipality_weights: dict[str, dict[str, float]] = {}
        wastewater_path = root / "obs_wastewater.zarr"
        catchment_path = root / "edar_catchment_edges.parquet"
        if config.wastewater_enabled:
            if not wastewater_path.exists() or not catchment_path.exists():
                raise FileNotFoundError(
                    "native wastewater requires obs_wastewater.zarr and "
                    "edar_catchment_edges.parquet"
                )
            self._ww = self._load_wastewater(wastewater_path)
            self._load_edar_catchment(catchment_path)
        # Deaths aligned onto the shared grid (absent dates -> mask 0).
        self._deaths_values, self._deaths_masks, self._deaths_ages = _align_daily(
            self._deaths, self._grid
        )
        self._L = config.input_window_length
        self._H = config.forecast_horizon
        if self._od is not None:
            self._od_zone_index = {zone: i for i, zone in enumerate(self._od.zone_ids)}
        self._registry = self._build_registry()
        self._relation_pools = self._build_relation_pools()
        self._pools = self._build_pools()
        self._od_feature_weights = self._build_od_feature_weights()
        self._od_graph_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._od_target_weight_cache: dict[tuple[str, str], np.ndarray] = {}
        self._items = self._build_index()
        self._context_source_roles: dict[str, str] | None = None

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

    def set_context_source_roles(self, source_roles: dict[str, str]) -> None:
        """Keep spatial context within each query's frozen source role."""
        self._context_source_roles = dict(source_roles)

    def _context_allowed(self, query_key: str, context_key: str) -> bool:
        roles = self._context_source_roles
        return roles is None or (
            roles.get(query_key) in {"train", "validation", "test"}
            and roles.get(query_key) == roles.get(context_key)
        )

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

    @property
    def date_grid(self) -> np.ndarray:
        """Shared daily coordinate used by query origins and evaluation."""
        return self._grid.copy()

    @property
    def wastewater_assays(self) -> tuple[str, ...]:
        """Stable assay order of the native wastewater target block."""
        return () if self._ww is None else tuple(self._ww.assays)

    def graph_node_index(self, source_type: str, source_id: str) -> int:
        """Stable native-graph node index for schema-v4 granular exports."""
        key = f"{source_type}:{source_id}"
        if key in self._node_index:
            return self._node_index[key]
        if source_type == SOURCE_EDAR and self._ww is not None:
            return len(self._graph_nodes.source_id) + self._ww.columns[source_id]
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._items)

    def head_at(self, index: int) -> str:
        """Routed head for an index without materializing its graph tensors."""
        return self._items[index][1]

    def __getitem__(self, idx: int) -> NativeDatasetItem:
        query, head = self._items[idx]
        origin = query.forecast_origin
        hist_slice = slice(origin - self._L + 1, origin + 1)
        target_slice = slice(origin + 1, origin + self._H + 1)
        covariates = self._cases.covariates[hist_slice].astype(np.float32)
        stratum: np.ndarray | None = None

        target_extra: dict[str, torch.Tensor] | None = None
        if query.source_type == SOURCE_MUNICIPALITY:
            hist, hist_mask = self._municipal_history(query.source_id, hist_slice)
            target, target_mask = self._municipal_target(query.source_id, target_slice)
        elif query.source_type == SOURCE_ABS:
            hist, hist_mask = self._abs_joint_history(query.source_id, hist_slice)
            if head == "deaths":
                target, target_mask = self._abs_deaths_target(
                    query.source_id, target_slice
                )
                target_extra = {
                    "expected_deaths": torch.from_numpy(
                        self._deaths_expected[
                            target_slice, self._deaths.columns[query.source_id]
                        ].copy()
                    )
                }
            else:
                target, target_mask, stratum = self._abs_target(query.source_id, origin)
        else:
            hist, hist_mask = self._edar_history(query.source_id, hist_slice)
            target, target_mask, extra = self._edar_target(
                query.source_id, target_slice
            )
            target_extra = {
                name: torch.from_numpy(value) for name, value in extra.items()
            }
        if hist.shape[1] < 6:
            hist = np.concatenate(
                [hist, np.zeros((self._L, 6 - hist.shape[1]), dtype=np.float32)],
                axis=1,
            )
        if self._cases.vaccination is not None:
            vaccination = np.zeros((self._L, 3), dtype=np.float32)
            if query.source_type == SOURCE_MUNICIPALITY:
                column = self._cases.columns[query.source_id]
                vaccination = self._cases.vaccination[hist_slice, column].copy()
            hist = np.concatenate([hist, vaccination], axis=1)
        hist = np.concatenate([hist, covariates], axis=1)

        relation_ctx, pooled, coverage = self._relation_context(
            query.source_type, query.source_id, hist_slice
        )
        if self._config.include_od_context:
            od_share = self._od_share(query) if self._config.mobility_enabled else 0.0
            od_column = np.full((self._L, 1), od_share, dtype=np.float32)
            pooled = np.concatenate([pooled, od_column], axis=1)

        population = self._source_population(query.source_type, query.source_id)
        item: NativeDatasetItem = {
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
            "relation_ctx": {
                name: torch.from_numpy(value) for name, value in relation_ctx.items()
            },
        }
        if target_extra is not None:
            item["target_extra"] = target_extra
        if self._config.spatial_encoder == "hetero" and self._config.mobility_enabled:
            od_features, od_adjacency = self._od_graph_context(query.forecast_origin)
            item["od_node_features"] = torch.from_numpy(od_features)
            item["od_adjacency"] = torch.from_numpy(od_adjacency)
            item["od_target_weights"] = torch.from_numpy(
                self._od_target_weights(query.source_type, query.source_id)
            )
        return item

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

    def _load_od_store(self, root: Path, od_source: str) -> _OdStore:
        ds = xr.open_zarr(root / f"od_{od_source}.zarr")
        return _OdStore(ds, log1p_stored=od_source == "municipal")

    def _load_admissions(self, path: Path) -> _AdmSeries:
        """Load the weekly admissions store; ``stratum_pop_sum`` is required."""
        loaded = self._load_store(path, "abs_id")
        if not isinstance(loaded, _AdmSeries):
            raise ValueError(f"{path} is missing the stratum_pop_sum variable")
        return loaded

    def _load_wastewater(self, path: Path) -> _WastewaterSeries:
        dataset = xr.open_zarr(path).compute()
        if dataset.attrs.get("schema") != "native_wastewater_observations.v1":
            raise ValueError(f"Unsupported native wastewater schema in {path}")
        if not np.array_equal(
            np.asarray(dataset["date"].values).astype("datetime64[D]"), self._grid
        ):
            dataset = dataset.reindex(date=self._grid).compute()
        return _WastewaterSeries(dataset)

    def _load_edar_catchment(self, path: Path) -> None:
        edges = pd.read_parquet(path)
        municipal = edges[edges["target_type"] == SOURCE_MUNICIPALITY].copy()
        if municipal.empty:
            raise ValueError("EDAR catchment table has no municipality rows")
        self._edar_population = {
            str(edar_id): float(values.iloc[0])
            for edar_id, values in municipal.groupby("edar_id")[
                "catchment_population_est"
            ]
        }
        case_columns = {
            source_id: index for index, source_id in enumerate(self._cases.source_ids)
        }
        for edar_id, rows in municipal.groupby("edar_id"):
            columns: list[int] = []
            weights: list[float] = []
            for row in rows.itertuples(index=False):
                column = case_columns.get(str(row.target_id))
                if column is not None:
                    columns.append(column)
                    weights.append(cast(float, row.frac_of_edar_catchment))
            weight_array = np.asarray(weights, dtype=np.float32)
            self._edar_case_pools[str(edar_id)] = (
                np.asarray(columns, dtype=np.int64),
                weight_array,
                float(weight_array.sum()),
            )
            self._edar_municipality_weights[str(edar_id)] = {
                str(row.target_id): cast(float, row.frac_of_edar_catchment)
                for row in rows.itertuples(index=False)
            }

    def _source_population(self, source_type: str, source_id: str) -> float:
        if source_type == SOURCE_EDAR:
            return self._edar_population[source_id]
        node = self._node_index[f"{source_type}:{source_id}"]
        return float(self._graph_nodes.population[node])

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

    def _load_abs_deaths(self, path: Path) -> _DeathsSeries:
        ds = xr.open_zarr(path).compute()
        if ds.attrs.get("schema") != "obs_deaths_abs_excess":
            raise ValueError("ABS excess deaths store has an unexpected schema")
        dates = np.asarray(ds["date"].values).astype("datetime64[D]")
        source_ids = [str(value) for value in ds["abs_id"].values]
        values = np.asarray(ds["value"].values, dtype=np.float32)
        masks = np.asarray(ds["mask"].values, dtype=bool)
        ages = np.asarray(ds["age"].values, dtype=np.float32)
        expected = np.asarray(ds["expected"].values, dtype=np.float32)
        excess = np.asarray(ds["excess"].values, dtype=np.float32)
        if (
            np.any(values < 0)
            or not np.isfinite(expected).all()
            or np.any(expected < 0)
            or not np.allclose(excess, values - expected, atol=1e-4)
        ):
            raise ValueError("ABS excess deaths store has invalid count or baseline")
        return _DeathsSeries(dates, source_ids, values, masks, ages, expected)

    def _load_cases(self, real_with_id_path: str | Path) -> _CasesSeries:
        ds = xr.open_zarr(real_with_id_path)
        needed = ["cases", "cases_mask", "cases_age", "temporal_covariates"]
        if self._config.include_vaccination:
            vaccination_names = (
                "vaccination_rate",
                "vaccination_rate_mask",
                "vaccination_rate_age",
            )
            missing = [name for name in vaccination_names if name not in ds]
            if missing:
                raise ValueError(f"native vaccination input missing: {missing}")
            needed.extend(vaccination_names)
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
        vaccination = None
        if self._config.include_vaccination:
            vaccination = np.stack(
                [
                    np.asarray(ds["vaccination_rate"].values, dtype=np.float32),
                    np.asarray(ds["vaccination_rate_mask"].values, dtype=np.float32),
                    np.clip(
                        np.asarray(ds["vaccination_rate_age"].values, dtype=np.float32)
                        / 14.0,
                        0.0,
                        1.0,
                    ),
                ],
                axis=-1,
            )
            vaccination[:, :, 0] = np.nan_to_num(
                vaccination[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0
            )
        return _CasesSeries(dates, region_ids, channels, masks, covariates, vaccination)

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
        for source_id in sorted(
            set(self._adm.source_ids) | set(self._deaths.source_ids)
        ):
            source_type = SOURCE_ABS
            observed = any(
                source_id in store.columns
                and bool(store.masks[:, store.columns[source_id]].any())
                for store in (self._adm, self._deaths)
            )
            node = self._node_index.get(f"{source_type}:{source_id}")
            if node is None or str(self._graph_nodes.status[node]) == STATUS_UNMAPPED:
                status = STATUS_UNMAPPED
            elif not observed:
                status = STATUS_UNOBSERVED
            else:
                status = STATUS_OBSERVED
            population = (
                float(self._graph_nodes.population[node]) if node is not None else None
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
        if self._ww is not None:
            for column, source_id in enumerate(self._ww.source_ids):
                population = self._edar_population.get(source_id)
                observed = bool(self._ww.mask[:, column, :].any())
                status = (
                    STATUS_OBSERVED
                    if observed and population is not None and population > 0
                    else STATUS_UNMAPPED
                    if population is None
                    else STATUS_UNOBSERVED
                )
                registry.add(
                    SourceRecord(
                        source_type=SOURCE_EDAR,
                        source_id=source_id,
                        status=status,
                        population=population,
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
        if head == "ww":
            assert self._ww is not None
            masks = self._ww.mask.any(axis=2)
            store = self._ww
        else:
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
                and record.population is not None
                and math.isfinite(record.population)
                and record.population > 0
                and (head != "hosp" or record.source_id in self._adm.columns)
                and (head != "deaths" or record.source_id in self._deaths.columns)
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
                entry[1],
            )
        )
        return entries

    # ------------------------------------------------------------------
    # Context pooling
    # ------------------------------------------------------------------

    def _build_relation_pools(
        self,
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]]:
        """Municipal-neighbor pools kept separate by typed relation."""
        node_to_cases = np.full(len(self._graph_nodes.source_id), -1, dtype=np.int64)
        for column, region_id in enumerate(self._cases.source_ids):
            node = self._node_index.get(f"{SOURCE_MUNICIPALITY}:{region_id}")
            if node is not None:
                node_to_cases[node] = column
        relation_pools: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {
            name: {} for name in _POOL_RELATIONS
        }
        for relation in _POOL_RELATIONS:
            for source_type in (SOURCE_ABS, SOURCE_COMARCA, SOURCE_MUNICIPALITY):
                for record in self._registry.records_for(source_type):
                    key = f"{source_type}:{record.source_id}"
                    if key not in self._node_index:
                        continue
                    relation_pools[relation][key] = self._pool_for_relation(
                        relation, source_type, record.source_id, node_to_cases
                    )
        return relation_pools

    def _pool_for_relation(
        self,
        relation: str,
        source_type: str,
        source_id: str,
        node_to_cases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        node = self._node_index[f"{source_type}:{source_id}"]
        is_muni = self._graph_nodes.node_type == SOURCE_MUNICIPALITY
        columns: list[int] = []
        weights: list[float] = []
        edges = self._edges[relation]
        for row in np.flatnonzero(edges["dst_idx"] == node):
            src = int(edges["src_idx"][row])
            case_column = int(node_to_cases[src])
            if is_muni[src] and case_column >= 0:
                columns.append(case_column)
                weights.append(float(edges["weight"][row]))
        raw = np.asarray(weights, dtype=np.float32)
        return np.asarray(columns, dtype=np.int64), raw, float(raw.sum())

    def _build_pools(self) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        """Per-source (cases columns, raw destination fractions, coverage).

        Neighbors are case-bearing municipalities; the coverage scalar is
        the query source's population share reachable through municipal
        overlaps with case observations. Municipal queries have no
        municipal overlap neighbors: empty pooling, coverage 0.
        """
        pools: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for source_type in (SOURCE_ABS, SOURCE_COMARCA, SOURCE_MUNICIPALITY):
            for record in self._registry.records_for(source_type):
                key = f"{source_type}:{record.source_id}"
                if key not in self._node_index:
                    continue  # graph-absent sources are never queried
                blocks = [self._relation_pools[name][key] for name in _POOL_RELATIONS]
                columns = np.concatenate([block[0] for block in blocks])
                weights = np.concatenate([block[1] for block in blocks])
                pools[key] = (columns, weights, sum(block[2] for block in blocks))
        for source_id, pool in self._edar_case_pools.items():
            pools[f"{SOURCE_EDAR}:{source_id}"] = pool
        return pools

    def _relation_context(
        self, source_type: str, source_id: str, hist_slice: slice
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
        """Bidirectional typed messages plus the unchanged pooled baseline."""
        key = f"{source_type}:{source_id}"
        relation_ctx: dict[str, np.ndarray] = {}
        for relation in _POOL_RELATIONS:
            width = 6 if self._cases.vaccination is not None else 3
            block = np.zeros((self._L, width), dtype=np.float32)
            if source_type != SOURCE_EDAR:
                columns, weights, _ = self._relation_pools[relation][key]
                if columns.size:
                    allowed = np.asarray(
                        [
                            self._context_allowed(
                                key,
                                f"{SOURCE_MUNICIPALITY}:{self._cases.source_ids[column]}",
                            )
                            for column in columns
                        ],
                        dtype=bool,
                    )
                    municipal = self._municipal_features(hist_slice, columns[allowed])
                    block = (municipal * weights[allowed][None, :, None]).sum(
                        axis=1, dtype=np.float32
                    )
                if source_type == SOURCE_MUNICIPALITY:
                    block[:, :3] += self._reverse_relation_context(
                        relation, source_id, hist_slice
                    )
            relation_ctx[relation] = block.astype(np.float32, copy=False)
        pooled, coverage = self._pooled_context(source_type, source_id, hist_slice)
        if self._config.wastewater_enabled:
            width = 6 if self._cases.vaccination is not None else 3
            edar_context = np.zeros((self._L, width), dtype=np.float32)
            if source_type == SOURCE_EDAR:
                edar_context[:, :3] = pooled
                if self._cases.vaccination is not None:
                    edar_context[:, 3:] = self._pooled_vaccination(
                        source_type, source_id, hist_slice
                    )
            relation_ctx[_EDAR_CONTEXT_RELATION] = edar_context
        return relation_ctx, pooled, coverage

    def _reverse_relation_context(
        self, relation: str, municipality_id: str, hist_slice: slice
    ) -> np.ndarray:
        """ABS/comarca histories sent back to a municipal target."""
        target = self._node_index[f"{SOURCE_MUNICIPALITY}:{municipality_id}"]
        expected_type = SOURCE_ABS if relation == "muni_abs" else SOURCE_COMARCA
        block = np.zeros((self._L, 3), dtype=np.float32)
        edges = self._edges[relation]
        for edge in np.flatnonzero(edges["dst_idx"] == target):
            src = int(edges["src_idx"][edge])
            if self._graph_nodes.node_type[src] != expected_type:
                continue
            population = float(self._graph_nodes.population[src])
            if not math.isfinite(population) or population <= 0:
                continue
            source_id = str(self._graph_nodes.source_id[src])
            if not self._context_allowed(
                f"{SOURCE_MUNICIPALITY}:{municipality_id}",
                f"{expected_type}:{source_id}",
            ):
                continue
            if expected_type == SOURCE_ABS:
                if source_id not in self._adm.columns:
                    continue
                history, _ = self._abs_history(source_id, hist_slice)
            else:
                continue  # comarca nodes no longer carry death observations
            block += history * float(edges["weight"][edge])
        return block

    def _pooled_context(
        self, source_type: str, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, float]:
        """Fixed pooling of municipal daily cases channels, ``[L, 3]``."""
        columns, weights, coverage = self._pools[f"{source_type}:{source_id}"]
        pooled = np.zeros((self._L, 3), dtype=np.float32)
        if columns.size:
            allowed = np.asarray(
                [
                    self._context_allowed(
                        f"{source_type}:{source_id}",
                        f"{SOURCE_MUNICIPALITY}:{self._cases.source_ids[column]}",
                    )
                    for column in columns
                ],
                dtype=bool,
            )
            columns, weights = columns[allowed], weights[allowed]
            coverage = float(weights.sum())
            block = self._cases.channels[hist_slice][:, columns, :]  # [L, M, 3]
            pooled = (block * weights[None, :, None]).sum(axis=1, dtype=np.float32)
        return pooled.astype(np.float32, copy=False), coverage

    def _pooled_vaccination(
        self, source_type: str, source_id: str, hist_slice: slice
    ) -> np.ndarray:
        """Pool municipality vaccination with the same catchment weights."""
        assert self._cases.vaccination is not None
        columns, weights, _ = self._pools[f"{source_type}:{source_id}"]
        allowed = np.asarray(
            [
                self._context_allowed(
                    f"{source_type}:{source_id}",
                    f"{SOURCE_MUNICIPALITY}:{self._cases.source_ids[column]}",
                )
                for column in columns
            ],
            dtype=bool,
        )
        vaccination = self._cases.vaccination[hist_slice][:, columns[allowed]]
        return (vaccination * weights[allowed][None, :, None]).sum(
            axis=1, dtype=np.float32
        )

    def _municipal_features(
        self, hist_slice: slice, columns: np.ndarray | None = None
    ) -> np.ndarray:
        """Municipal case and optional vaccination history at the query origin."""
        cases = self._cases.channels[hist_slice]
        if columns is not None:
            cases = cases[:, columns]
        if self._cases.vaccination is None:
            return cases
        vaccination = self._cases.vaccination[hist_slice]
        if columns is not None:
            vaccination = vaccination[:, columns]
        return np.concatenate([cases, vaccination], axis=-1)

    def _build_od_feature_weights(self) -> np.ndarray:
        """Map municipal features into the configured OD node set."""
        if (
            self._config.spatial_encoder != "hetero"
            or not self._config.mobility_enabled
            or self._od is None
        ):
            return np.zeros((0, len(self._cases.source_ids)), dtype=np.float32)
        zone_index = {zone: index for index, zone in enumerate(self._od.zone_ids)}
        case_index = {
            source_id: index for index, source_id in enumerate(self._cases.source_ids)
        }
        weights = np.zeros(
            (len(self._od.zone_ids), len(self._cases.source_ids)), dtype=np.float32
        )
        if self._config.od_source == "municipal":
            for zone, row in zone_index.items():
                column = case_index.get(zone)
                if column is not None:
                    weights[row, column] = 1.0
            return weights

        edges = self._edges["mitma_muni"]
        for edge in range(len(edges["src_idx"])):
            src = int(edges["src_idx"][edge])
            dst = int(edges["dst_idx"][edge])
            if (
                self._graph_nodes.node_type[src] != SOURCE_MUNICIPALITY
                or self._graph_nodes.node_type[dst] != "mitma"
            ):
                continue
            zone = zone_index.get(str(self._graph_nodes.source_id[dst]))
            column = case_index.get(str(self._graph_nodes.source_id[src]))
            if zone is not None and column is not None:
                weights[zone, column] += float(edges["weight"][edge])
        return weights

    def _od_graph_context(self, origin: int) -> tuple[np.ndarray, np.ndarray]:
        """Origin-specific OD node features and incoming-normalized adjacency."""
        cached = self._od_graph_cache.get(origin)
        if cached is not None:
            return cached
        assert self._od is not None
        features = (
            self._od_feature_weights
            @ self._municipal_features(slice(origin, origin + 1))[0]
        )
        od_row = self._od.row_of(self._grid[origin])
        if od_row is None:
            flows = np.zeros(
                (len(self._od.zone_ids), len(self._od.zone_ids)), dtype=np.float32
            )
        else:
            flows = self._od.flows_at(od_row).astype(np.float32, copy=True)
        if self._config.od_self_flows == "drop":
            np.fill_diagonal(flows, 0.0)
        flows[(flows <= 0.0) | (flows < self._config.od_min_flow_raw)] = 0.0
        incoming = flows.sum(axis=0, dtype=np.float64).astype(np.float32)
        adjacency = flows.T
        np.divide(
            adjacency,
            incoming[:, None],
            out=adjacency,
            where=incoming[:, None] > 0,
        )
        result = (
            features.astype(np.float32, copy=False),
            adjacency.astype(np.float32, copy=False),
        )
        if len(self._od_graph_cache) >= self._config.od_graph_cache_size:
            self._od_graph_cache.pop(next(iter(self._od_graph_cache)))
        self._od_graph_cache[origin] = result
        return result

    def _od_target_weights(self, source_type: str, source_id: str) -> np.ndarray:
        """Typed hop from configured OD nodes back to one clinical target."""
        if not self._config.cache_od_target_weights:
            return self._build_od_target_weights(source_type, source_id)
        key = (source_type, source_id)
        cached = self._od_target_weight_cache.get(key)
        if cached is None:
            cached = self._build_od_target_weights(source_type, source_id)
            self._od_target_weight_cache[key] = cached
        return cached

    def _build_od_target_weights(self, source_type: str, source_id: str) -> np.ndarray:
        """Build fixed source weights without an origin-dependent input."""
        assert self._od is not None
        weights = np.zeros(len(self._od.zone_ids), dtype=np.float32)
        zone_index = {zone: index for index, zone in enumerate(self._od.zone_ids)}
        if source_type == SOURCE_EDAR:
            municipality_weights = self._edar_municipality_weights[source_id]
            if self._config.od_source == "municipal":
                for municipality_id, catchment_weight in municipality_weights.items():
                    zone = zone_index.get(municipality_id)
                    if zone is not None:
                        weights[zone] += catchment_weight
                return weights
            edges = self._edges["mitma_muni"]
            for municipality_id, catchment_weight in municipality_weights.items():
                target = self._node_index.get(
                    f"{SOURCE_MUNICIPALITY}:{municipality_id}"
                )
                if target is None:
                    continue
                for edge in np.flatnonzero(edges["dst_idx"] == target):
                    src = int(edges["src_idx"][edge])
                    if self._graph_nodes.node_type[src] != "mitma":
                        continue
                    zone = zone_index.get(str(self._graph_nodes.source_id[src]))
                    if zone is not None:
                        weights[zone] += catchment_weight * float(edges["weight"][edge])
            return weights
        if self._config.od_source == "municipal":
            if source_type == SOURCE_MUNICIPALITY:
                zone = zone_index.get(source_id)
                if zone is not None:
                    weights[zone] = 1.0
                return weights
            relation = "muni_abs" if source_type == SOURCE_ABS else "muni_comarca"
            expected_type = SOURCE_MUNICIPALITY
        else:
            relation = f"mitma_{source_type.replace('municipality', 'muni')}"
            expected_type = "mitma"
        target = self._node_index[f"{source_type}:{source_id}"]
        edges = self._edges[relation]
        for edge in np.flatnonzero(edges["dst_idx"] == target):
            src = int(edges["src_idx"][edge])
            if self._graph_nodes.node_type[src] != expected_type:
                continue
            zone = zone_index.get(str(self._graph_nodes.source_id[src]))
            if zone is not None:
                weights[zone] += float(edges["weight"][edge])
        return weights

    def _od_share(self, query: ForecastQuery) -> float:
        """Origin-day incoming trip share into the source's overlapping zones.

        Consumption-time OD policy (plan sections 11.6-11.7, ruling R10):
        flows below ``od_min_flow_raw`` are dropped and
        ``od_self_flows="drop"`` clears the diagonal (origin==target)
        before the shares are taken; the store stays raw at rest.
        """
        assert self._od is not None
        if query.source_type == SOURCE_EDAR:
            zone_weights = self._od_target_weights(query.source_type, query.source_id)
            zone_columns = set(np.flatnonzero(zone_weights > 0).tolist())
        else:
            node = self._node_index[f"{query.source_type}:{query.source_id}"]
            zone_columns = set()
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
        flows = self._od.flows_at(od_row).astype(np.float32, copy=True)
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
            # A weekly total is available only after its complete interval.
            if (
                self._adm.dates[row] + np.timedelta64(6, "D")
                > self._grid[int(hist_slice.stop) - 1]
            ):
                continue
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

    def _abs_joint_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the same admission and all-cause history for both ABS heads."""
        admissions = deaths = np.zeros((self._L, 3), dtype=np.float32)
        admission_mask = death_mask = np.zeros(self._L, dtype=bool)
        if source_id in self._adm.columns:
            admissions, admission_mask = self._abs_history(source_id, hist_slice)
        if source_id in self._deaths.columns:
            deaths, death_mask = self._abs_deaths_history(source_id, hist_slice)
        return np.concatenate([admissions, deaths], axis=1), admission_mask | death_mask

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

    def _abs_deaths_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Daily ABS all-cause death counts, ``[L, 3]``.

        Consumption transform (ruling R10): the raw-at-rest count becomes
        ``log1p((count / pop_graph) · 1e5)`` using the ABS graph
        population, applied only where the mask is 1 (unreported days stay
        0). Mask/age channels are raw; targets stay raw.
        """
        column = self._deaths.columns[source_id]
        node = self._node_index[f"{SOURCE_ABS}:{source_id}"]
        population = float(self._graph_nodes.population[node])
        values = self._deaths_values[hist_slice, column]
        mask = self._deaths_masks[hist_slice, column]
        value = np.log1p(np.where(mask, values, 0.0) / population * _PER_100K)
        block = np.stack(
            [
                value.astype(np.float32),
                mask.astype(np.float32),
                self._deaths_ages[hist_slice, column],
            ],
            axis=-1,
        )
        return block.astype(np.float32, copy=False), mask.copy()

    def _abs_deaths_target(
        self, source_id: str, target_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        column = self._deaths.columns[source_id]
        values = self._deaths_values[target_slice, column]
        mask = self._deaths_masks[target_slice, column]
        return np.where(mask, values, 0.0).astype(np.float32), mask.copy()

    def _edar_history(
        self, source_id: str, hist_slice: slice
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assay-aware EDAR history compressed to four causal channels.

        Assay targets remain separate. The conditioning value is the masked
        mean of per-assay causal ``log1p`` histories, followed by raw sample
        presence, minimum assay age, and causally carried ``log1p`` 24-hour
        flow. No target-window flow enters model input.
        """
        assert self._ww is not None
        column = self._ww.columns[source_id]
        history = self._ww.history[hist_slice, column, :]
        available = self._ww.history_available[hist_slice, column, :]
        count = available.sum(axis=1)
        value = np.divide(
            np.where(available, history, 0.0).sum(axis=1),
            count,
            out=np.zeros(self._L, dtype=np.float32),
            where=count > 0,
        )
        observed = np.asarray(
            self._ww.mask[hist_slice, column, :].any(axis=1), dtype=bool
        )
        ages = self._ww.age[hist_slice, column, :]
        age = np.full(self._L, 0.0, dtype=np.float32)
        for row in range(self._L):
            finite = ages[row][np.isfinite(ages[row])]
            if finite.size:
                age[row] = float(finite.min())
        flow = np.nan_to_num(
            self._ww.flow_history_log1p[hist_slice, column], nan=0.0
        ).astype(np.float32)
        block = np.stack(
            [value, observed.astype(np.float32), age, flow], axis=-1
        ).astype(np.float32)
        return block, observed.copy()

    def _edar_target(
        self, source_id: str, target_slice: slice
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Raw ``[H, assay]`` concentrations plus censor and flow extras."""
        assert self._ww is not None
        column = self._ww.columns[source_id]
        mask = self._ww.mask[target_slice, column, :]
        concentration = self._ww.concentration[target_slice, column, :]
        target = np.where(mask, concentration, 0.0).astype(np.float32)
        extra = {
            "detection_limit": np.nan_to_num(
                self._ww.detection_limit[target_slice, column, :], nan=0.0
            ).astype(np.float32),
            "censor_code": self._ww.censor_code[target_slice, column, :].astype(
                np.uint8
            ),
            "flow_m3_24h": np.nan_to_num(
                self._ww.flow[target_slice, column], nan=0.0
            ).astype(np.float32),
            "flow_mask": self._ww.flow_mask[target_slice, column].copy(),
        }
        return target, mask.copy(), extra
