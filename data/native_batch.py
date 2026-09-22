"""Native-support batch container and collation (plan sections 11.4, 12.2 BAT-1).

Collates DS-1 ``NativeDatasetItem`` dicts (:mod:`data.native_dataset`) into
the unified :class:`NativeEpiBatch`: stacked ``[B, L, 5 + F_native]``
history — a frozen source-type one-hot block (review R12, one channel per
:data:`SOURCE_TYPE_ORDER` entry, repeated over ``L``) followed by the
per-query native channels — source conditioning, and per-head supervised
blocks routed by each query's ``requested_heads``. Pure tensor work — no
I/O and no dataset access.

Routing is per requested head, so a synthetic multi-head query contributes
its row to every requested head's block; real-data routing (plan section
11.1) requests exactly one head per query, which makes the routed blocks a
disjoint partition of the batch rows.

The frozen section 11.4 fields keep their exact names and shapes. Two
additive fields extend the frozen set and are appended after it:
``ctx_coverage`` (``[B]`` float32 passthrough of the pooled-context
coverage) and ``head_extra`` (per-head QC extras — ``stratum_pop_sum`` for
the weekly ``hosp`` head, plan section 10.3(4)). They are additive by
construction: consumers of the frozen fields can ignore them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from data.native_dataset import SOURCE_TYPE_IDS as _DATASET_SOURCE_TYPE_IDS

if TYPE_CHECKING:
    from data.native_dataset import NativeDatasetItem
    from data.query_contract import ForecastQuery

#: Frozen source-type enum order (plan section 11.1). ``source_type_id``
#: values index into this tuple.
SOURCE_TYPE_ORDER: tuple[str, ...] = (
    "municipality",
    "abs",
    "comarca",
    "mitma",
    "edar",
)

#: Enum index per source type — must stay identical to DS-1's mapping
#: (``data.native_dataset.SOURCE_TYPE_IDS``); the import-time guard below
#: fails fast on drift between the two modules.
SOURCE_TYPE_IDS: dict[str, int] = {
    name: index for index, name in enumerate(SOURCE_TYPE_ORDER)
}

if SOURCE_TYPE_IDS != _DATASET_SOURCE_TYPE_IDS:
    raise ImportError(
        "SOURCE_TYPE_ORDER drifted from data.native_dataset.SOURCE_TYPE_IDS: "
        f"{SOURCE_TYPE_ORDER} vs {sorted(_DATASET_SOURCE_TYPE_IDS.items())}"
    )

#: Supervised heads with routed per-head blocks (plan section 11.4).
ROUTED_HEADS: tuple[str, ...] = ("cases", "hosp", "deaths")

#: Head QC extras carried beside the frozen blocks (additive ``head_extra``).
_HEAD_EXTRA_KEY = "stratum_pop_sum"


def _move_tensors(value: object, device: torch.device, non_blocking: bool) -> object:
    """Recursively move tensors (including inside per-head dicts) to device."""
    if isinstance(value, torch.Tensor):
        if value.device != device:
            return value.to(device=device, non_blocking=non_blocking)
        return value
    if isinstance(value, dict):
        return {
            key: _move_tensors(item, device, non_blocking)
            for key, item in value.items()
        }
    return value


def _pin_tensors(value: object) -> object:
    """Recursively pin CPU tensors (including inside per-head dicts)."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory() if value.device.type == "cpu" else value
    if isinstance(value, dict):
        return {key: _pin_tensors(item) for key, item in value.items()}
    return value


@dataclass
class NativeEpiBatch:
    """Unified native-support batch (plan section 11.4).

    The first twelve fields are the frozen section 11.4 contract — exact
    names and shapes. ``ctx_coverage`` and ``head_extra`` are additive
    extensions documented in the module docstring.
    """

    # -- Frozen plan section 11.4 fields --------------------------------
    queries: list[ForecastQuery]  # length B, order-defining
    hist: torch.Tensor  # (B, L, 5 + F_native): leading source-type one-hot + native channels
    hist_mask: torch.Tensor  # (B, L) observed-validity for the query's own series
    source_type_id: torch.Tensor  # (B,) int enum index
    source_pop: torch.Tensor  # (B,) audited denominator for per-100k transform
    origin_dow: torch.Tensor  # (B,) weekday of forecast_origin (0=Mon)
    ctx_feat: torch.Tensor  # (B, L, D_ctx) fixed-pooling context
    temporal_covariates: torch.Tensor  # (B, L, k), shared with the municipal path
    # Per-head supervised blocks, gathered by routing index (length B_h <= B):
    head_index: dict[str, torch.Tensor | None]  # {"cases","hosp","deaths"} row positions
    head_target: dict[str, torch.Tensor | None]  # cases/deaths: (B_·, H) daily; hosp: (B_h, W) weekly
    head_target_mask: dict[str, torch.Tensor | None]
    head_population: dict[str, torch.Tensor | None]  # per-head denominator
    # -- Additive extensions (documented; not part of the frozen block) --
    ctx_coverage: torch.Tensor  # (B,) float32 pooled-context coverage <= 1
    head_extra: dict[str, dict[str, torch.Tensor | None]]  # per-head QC extras
    # ENC-1 graph context. The pooled path leaves these empty. Graph tensors
    # are grouped by unique forecast origin, so identical-origin queries share
    # one mobility encoding inside a forward pass.
    relation_ctx: dict[str, torch.Tensor] = field(default_factory=dict)
    od_node_features: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, 3)
    )  # (G, N, 3)
    od_adjacency: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, 0)
    )  # (G, N, N)
    od_target_weights: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0)
    )  # (B, N)
    od_group_index: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )  # (B,)

    @property
    def b(self) -> int:
        """Batch size B."""
        return len(self.queries)

    @property
    def t(self) -> int:
        """History window length L."""
        return int(self.hist.shape[1])

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> "NativeEpiBatch":
        """Move batch tensors to the target device without changing dtypes."""
        if isinstance(device, str):
            device = torch.device(device)
        safe_non_blocking = bool(non_blocking) and device.type == "cuda"
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            setattr(
                self, field_name, _move_tensors(value, device, safe_non_blocking)
            )
        return self

    def pin_memory(self) -> "NativeEpiBatch":
        """Pin CPU tensors recursively so CUDA transfers can be non-blocking."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, _pin_tensors(getattr(self, field_name)))
        return self


def _stack_uniform(values: Sequence[torch.Tensor], label: str) -> torch.Tensor:
    """Stack tensors of identical shape; a shape mismatch is a batch error."""
    shape = tuple(values[0].shape)
    for position, value in enumerate(values[1:], start=1):
        if tuple(value.shape) != shape:
            raise ValueError(
                f"Cannot stack {label!r}: item 0 has shape {shape} but item "
                f"{position} has shape {tuple(value.shape)}"
            )
    return torch.stack(list(values), dim=0)


def _collect_head_extra(
    routed: Sequence[NativeDatasetItem], head: str
) -> dict[str, torch.Tensor | None]:
    """Stack row-aligned QC extras for one routed head.

    ``None`` when no routed item provides the extra; partial provision is a
    contract violation because extras are row-aligned with the routed block
    and a missing slot cannot be represented.
    """
    values = [item.get("target_stratum_pop_sum") for item in routed]
    provided = [value for value in values if value is not None]
    if not provided:
        return {_HEAD_EXTRA_KEY: None}
    if len(provided) != len(routed):
        raise ValueError(
            f"Head {head!r}: {len(provided)} of {len(routed)} routed items "
            f"provide {_HEAD_EXTRA_KEY!r}; extras are row-aligned and must "
            "be provided by all or none"
        )
    return {_HEAD_EXTRA_KEY: _stack_uniform(provided, _HEAD_EXTRA_KEY)}


def _source_type_one_hot(
    source_ids: torch.Tensor, length: int, dtype: torch.dtype
) -> torch.Tensor:
    """``[B, L, 5]`` frozen one-hot of each row's source type (review R12).

    One channel per :data:`SOURCE_TYPE_ORDER` entry, identical across all
    ``L`` history rows of an item — including pad rows, whose native
    channels are zero but whose type is still known.
    """
    block = torch.zeros(
        source_ids.shape[0], length, len(SOURCE_TYPE_ORDER), dtype=dtype
    )
    block.scatter_(2, source_ids.view(-1, 1, 1).expand(-1, length, 1), 1.0)
    return block


def collate_native_batch(items: Sequence[NativeDatasetItem]) -> NativeEpiBatch:
    """Collate DS-1 items into a :class:`NativeEpiBatch`.

    Validates source-type one-hot consistency
    (``item["source_type_id"]`` vs the query's enum index in
    :data:`SOURCE_TYPE_ORDER`), asserts uniform per-slot shapes, and routes
    each row into the blocks of every head its query requests. The batch
    ``hist`` gains the frozen leading source-type one-hot block (review
    R12): ``[B, L, 5 + F_native]`` with the item's native channels kept
    verbatim behind the one-hot; ``source_type_id`` stays alongside it.
    """
    if not items:
        raise ValueError("collate_native_batch requires a non-empty item list")
    queries: list[ForecastQuery] = [item["query"] for item in items]

    for position, (query, item) in enumerate(zip(queries, items, strict=True)):
        expected = SOURCE_TYPE_IDS[query.source_type]
        actual = int(item["source_type_id"])
        if actual != expected:
            raise ValueError(
                f"Item {position} source_type_id {actual} does not match its "
                f"query source type {query.source_type!r} (enum index "
                f"{expected} in {SOURCE_TYPE_ORDER})"
            )
        head = item.get("head")
        if head is not None and head not in query.requested_heads:
            raise ValueError(
                f"Item {position} routed head {head!r} is not among its "
                f"query requested_heads {sorted(query.requested_heads)}"
            )
        origin_dow = int(item["origin_dow"])
        if not 0 <= origin_dow <= 6:
            raise ValueError(
                f"Item {position} origin_dow {origin_dow} outside 0..6"
            )

    head_index: dict[str, torch.Tensor | None] = {}
    head_target: dict[str, torch.Tensor | None] = {}
    head_target_mask: dict[str, torch.Tensor | None] = {}
    head_population: dict[str, torch.Tensor | None] = {}
    head_extra: dict[str, dict[str, torch.Tensor | None]] = {}

    for head in ROUTED_HEADS:
        positions = [
            index
            for index, query in enumerate(queries)
            if head in query.requested_heads
        ]
        if not positions:
            head_index[head] = None
            head_target[head] = None
            head_target_mask[head] = None
            head_population[head] = None
            head_extra[head] = {}
            continue
        routed: list[NativeDatasetItem] = [items[index] for index in positions]
        target = _stack_uniform([item["target"] for item in routed], "target")
        target_mask = _stack_uniform(
            [item["target_mask"] for item in routed], "target_mask"
        )
        if tuple(target_mask.shape) != tuple(target.shape):
            raise ValueError(
                f"Head {head!r}: target_mask shape "
                f"{tuple(target_mask.shape)} does not match target shape "
                f"{tuple(target.shape)}"
            )
        head_index[head] = torch.tensor(positions, dtype=torch.long)
        head_target[head] = target
        head_target_mask[head] = target_mask
        head_population[head] = torch.tensor(
            [float(item["target_population_graph"]) for item in routed],
            dtype=torch.float32,
        )
        head_extra[head] = _collect_head_extra(routed, head)

    source_ids = torch.tensor(
        [int(item["source_type_id"]) for item in items], dtype=torch.long
    )
    hist = _stack_uniform([item["hist"] for item in items], "hist")
    hist = torch.cat(
        [_source_type_one_hot(source_ids, hist.shape[1], hist.dtype), hist], dim=2
    )

    relation_names = sorted(
        {name for item in items for name in item.get("relation_ctx", {})}
    )
    relation_ctx = {
        name: _stack_uniform(
            [
                item.get("relation_ctx", {}).get(
                    name, torch.zeros_like(item["ctx_feat"][:, :3])
                )
                for item in items
            ],
            f"relation_ctx.{name}",
        )
        for name in relation_names
    }

    od_presence = [item.get("od_node_features") is not None for item in items]
    if any(od_presence) and not all(od_presence):
        raise ValueError("OD graph context must be provided by all batch items or none")
    has_od = all(od_presence)
    if has_od:
        origin_to_group: dict[int, int] = {}
        grouped_features: list[torch.Tensor] = []
        grouped_adjacency: list[torch.Tensor] = []
        target_weights: list[torch.Tensor] = []
        group_index: list[int] = []
        for item in items:
            node_features = item.get("od_node_features")
            adjacency = item.get("od_adjacency")
            target_weight = item.get("od_target_weights")
            if node_features is None or adjacency is None or target_weight is None:
                raise ValueError("OD graph context is missing a required tensor")
            origin = int(item["query"].forecast_origin)
            group = origin_to_group.get(origin)
            if group is None:
                group = len(grouped_features)
                origin_to_group[origin] = group
                grouped_features.append(node_features)
                grouped_adjacency.append(adjacency)
            elif not torch.equal(
                grouped_features[group], node_features
            ) or not torch.equal(grouped_adjacency[group], adjacency):
                raise ValueError(
                    f"OD graph context differs within forecast origin {origin}"
                )
            target_weights.append(target_weight)
            group_index.append(group)
        od_node_features = _stack_uniform(grouped_features, "od_node_features")
        od_adjacency = _stack_uniform(grouped_adjacency, "od_adjacency")
        od_target_weights = _stack_uniform(
            target_weights, "od_target_weights"
        )
        od_group_index = torch.tensor(group_index, dtype=torch.long)
    else:
        od_node_features = torch.zeros(0, 0, 3)
        od_adjacency = torch.zeros(0, 0, 0)
        od_target_weights = torch.zeros(len(items), 0)
        od_group_index = torch.zeros(len(items), dtype=torch.long)

    return NativeEpiBatch(
        queries=queries,
        hist=hist,
        hist_mask=_stack_uniform([item["hist_mask"] for item in items], "hist_mask"),
        source_type_id=source_ids,
        source_pop=torch.tensor(
            [float(item["source_pop"]) for item in items], dtype=torch.float32
        ),
        origin_dow=torch.tensor(
            [int(item["origin_dow"]) for item in items], dtype=torch.long
        ),
        ctx_feat=_stack_uniform([item["ctx_feat"] for item in items], "ctx_feat"),
        temporal_covariates=_stack_uniform(
            [item["temporal_covariates"] for item in items], "temporal_covariates"
        ),
        head_index=head_index,
        head_target=head_target,
        head_target_mask=head_target_mask,
        head_population=head_population,
        ctx_coverage=torch.tensor(
            [float(item["ctx_coverage"]) for item in items], dtype=torch.float32
        ),
        head_extra=head_extra,
        relation_ctx=relation_ctx,
        od_node_features=od_node_features,
        od_adjacency=od_adjacency,
        od_target_weights=od_target_weights,
        od_group_index=od_group_index,
    )
