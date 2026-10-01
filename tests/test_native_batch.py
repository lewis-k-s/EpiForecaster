"""BAT-1 tests: ``NativeEpiBatch`` collation over DS-1 items (plan section 12.2).

Synthetic items are hand-built to the DS-1 item contract
(``data.native_dataset.NativeDatasetItem``) — no dataset or I/O involved.
The real-artifact integration test is skipif-guarded like
``tests/test_native_dataset.py``. Weekly ``hosp`` width W is origin-dow
dependent (plan section 11.5), so the integration draw keeps one width
class; within-head width uniformity is exactly what collation asserts.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from data.native_batch import (
    SOURCE_TYPE_IDS,
    SOURCE_TYPE_ORDER,
    collate_native_batch,
)
from data.native_dataset import (
    SOURCE_TYPE_IDS as DATASET_SOURCE_TYPE_IDS,
    NativeDatasetConfig,
    NativeDatasetItem,
    NativeSupportDataset,
)
from data.query_contract import ForecastQuery, SourceType

pytestmark = pytest.mark.epiforecaster

L = 8
F_NATIVE = 5  # per-item native channels: 3 clinical + 2 temporal covariates
N_TYPES = len(SOURCE_TYPE_ORDER)  # frozen leading one-hot block width (R12)
D_CTX = 3
H = 4  # daily target width (cases/deaths)
W = 2  # weekly target width (hosp)


def _item(
    source_type: SourceType,
    heads: frozenset[str] | set[str],
    *,
    source_id: str = "1",
    origin: int = 10,
    origin_dow: int = 3,
    pop: float = 1000.0,
    coverage: float = 0.5,
    pad_rows: tuple[int, ...] = (),
    weekly: bool = False,
    stratum: bool = False,
    source_type_id: int | None = None,
    target_len: int | None = None,
) -> NativeDatasetItem:
    """Hand-built DS-1 item with deterministic values derived from source_id."""
    base = float(source_id)
    hist = torch.arange(L * F_NATIVE, dtype=torch.float32).reshape(L, F_NATIVE) + base
    hist_mask = torch.ones(L, dtype=torch.bool)
    for row in pad_rows:
        hist[row] = 0.0
        hist_mask[row] = False
    width = target_len if target_len is not None else (W if weekly else H)
    query = ForecastQuery(
        source_type=source_type,
        source_id=source_id,
        forecast_origin=origin,
        horizon_days=H,
        requested_heads=frozenset(heads),
    )
    item: NativeDatasetItem = {
        "query": query,
        "head": sorted(heads)[0],
        "hist": hist,
        "hist_mask": hist_mask,
        "ctx_feat": torch.full((L, D_CTX), base, dtype=torch.float32),
        "ctx_coverage": coverage,
        "temporal_covariates": torch.full((L, 2), base, dtype=torch.float32),
        "source_type_id": (
            SOURCE_TYPE_IDS[source_type]
            if source_type_id is None
            else source_type_id
        ),
        "source_pop": pop,
        "origin_dow": origin_dow,
        "target": torch.full((width,), base, dtype=torch.float32),
        "target_mask": torch.ones(width, dtype=torch.bool),
        "target_population_graph": pop,
        "target_stratum_pop_sum": (
            torch.arange(width, dtype=torch.float32) if stratum else None
        ),
    }
    return item


@pytest.fixture()
def mixed_items() -> list[NativeDatasetItem]:
    """One real-data-style item per head: municipality, abs, comarca."""
    return [
        _item(
            "municipality",
            {"cases"},
            source_id="08121",
            origin=11,
            origin_dow=2,
            pop=250.0,
        ),
        _item(
            "abs",
            {"hosp"},
            source_id="088",
            origin=12,
            origin_dow=3,
            pop=70_000.0,
            weekly=True,
            stratum=True,
            coverage=0.25,
        ),
        _item(
            "comarca",
            {"deaths"},
            source_id="06",
            origin=13,
            origin_dow=4,
            pop=180_000.0,
        ),
    ]


# ----------------------------------------------------------------------
# Shapes, enum mapping, passthrough
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("source_type", "expected_id"), [("municipality", 0), ("abs", 1), ("comarca", 2), ("edar", 4)])
def test_source_type_id_matches_enum(
    source_type: SourceType, expected_id: int
) -> None:
    assert SOURCE_TYPE_IDS == DATASET_SOURCE_TYPE_IDS
    item = _item(source_type, {"cases"}, source_id="01001")
    batch = collate_native_batch([item])
    assert batch.source_type_id.tolist() == [expected_id]


def test_source_type_id_mismatch_raises() -> None:
    item = _item("municipality", {"cases"}, source_type_id=3)
    with pytest.raises(ValueError, match="source_type_id"):
        collate_native_batch([item])


def test_hist_leading_block_is_exactly_source_type_one_hot(
    mixed_items: list[NativeDatasetItem],
) -> None:
    batch = collate_native_batch(mixed_items)
    expected = torch.zeros(3, L, N_TYPES)
    for row, item in enumerate(mixed_items):
        expected[row, :, SOURCE_TYPE_IDS[item["query"].source_type]] = 1.0
    block = batch.hist[:, :, :N_TYPES]
    # Each row's 5-block matches its query type; every other slot is zero.
    assert torch.equal(block, expected)
    # EXACTLY one-hot: every history row sums to 1 with 0/1 entries only.
    assert torch.equal(block.sum(dim=-1), torch.ones(3, L))
    assert bool(((block == 0.0) | (block == 1.0)).all())
    # The per-query native channels pass through verbatim behind the block.
    for row, item in enumerate(mixed_items):
        assert torch.equal(batch.hist[row, :, N_TYPES:], item["hist"])


def test_item_head_outside_requested_heads_raises() -> None:
    item = _item("municipality", {"cases"}, source_id="1")
    item["head"] = "deaths"
    with pytest.raises(ValueError, match="requested_heads"):
        collate_native_batch([item])


def test_origin_dow_out_of_range_raises() -> None:
    item = _item("municipality", {"cases"}, source_id="1", origin_dow=7)
    with pytest.raises(ValueError, match="origin_dow"):
        collate_native_batch([item])


# ----------------------------------------------------------------------
# Routing: coverage, disjointness, multi-head synthetic rows
# ----------------------------------------------------------------------


def test_routing_covers_each_row_exactly_once(
    mixed_items: list[NativeDatasetItem],
) -> None:
    batch = collate_native_batch(mixed_items)
    seen: list[int] = []
    for index in batch.head_index.values():
        if index is not None:
            seen.extend(index.tolist())
    assert sorted(seen) == list(range(len(mixed_items)))


def test_routing_matches_requested_heads(
    mixed_items: list[NativeDatasetItem],
) -> None:
    batch = collate_native_batch(mixed_items)
    for head, index in batch.head_index.items():
        for position in [] if index is None else index.tolist():
            assert head in mixed_items[position]["query"].requested_heads
    for position, item in enumerate(mixed_items):
        for head in item["query"].requested_heads:
            assert position in batch.head_index[head].tolist()


def test_multi_head_synthetic_item_lands_in_each_requested_head() -> None:
    items = [
        _item("municipality", {"cases"}, source_id="01001"),
        _item("comarca", {"deaths"}, source_id="02"),
        _item("municipality", {"cases", "deaths"}, source_id="01003"),
        _item("abs", {"hosp"}, source_id="088", weekly=True, stratum=True),
    ]
    batch = collate_native_batch(items)
    assert batch.head_index["cases"].tolist() == [0, 2]
    assert batch.head_index["deaths"].tolist() == [1, 2]
    assert batch.head_index["hosp"].tolist() == [3]
    assert batch.head_target["cases"][:, 0].tolist() == pytest.approx(
        [1001.0, 1003.0]
    )
    assert batch.head_target["deaths"][:, 0].tolist() == pytest.approx([2.0, 1003.0])
    assert batch.head_population["deaths"].tolist() == pytest.approx(
        [1000.0, 1000.0]
    )
    extra = batch.head_extra["hosp"]
    assert extra is not None and extra["stratum_pop_sum"] is not None
    assert extra["stratum_pop_sum"].shape == (1, W)


# ----------------------------------------------------------------------
# Padding, absent heads, extras
# ----------------------------------------------------------------------


def test_padding_zeros_carry_mask_zero() -> None:
    item = _item("municipality", {"cases"}, source_id="01001", pad_rows=(1, 4))
    batch = collate_native_batch([item])
    assert batch.hist_mask[0, [1, 4]].tolist() == [False, False]
    assert int(batch.hist_mask[0].sum()) == L - 2
    # Native channels stay zero on pad rows; the one-hot type block is
    # repeated over every row, pads included (type is still known there).
    assert float(batch.hist[0, 1, N_TYPES:].abs().sum()) == 0.0
    assert float(batch.hist[0, 4, N_TYPES:].abs().sum()) == 0.0
    assert float(batch.hist[0, 0, N_TYPES:].abs().sum()) > 0.0
    assert batch.hist[0, :, :N_TYPES].sum(dim=-1).tolist() == [1.0] * L
    assert batch.hist[0, 1, SOURCE_TYPE_IDS["municipality"]].item() == 1.0


def test_absent_head_blocks_are_none() -> None:
    items = [
        _item("municipality", {"cases"}, source_id="01001"),
        _item("comarca", {"deaths"}, source_id="02"),
    ]
    batch = collate_native_batch(items)
    assert batch.head_index["hosp"] is None
    assert batch.head_target["hosp"] is None
    assert batch.head_target_mask["hosp"] is None
    assert batch.head_population["hosp"] is None
    assert batch.head_extra["hosp"] == {}
    assert batch.head_extra["cases"] == {"stratum_pop_sum": None}


def test_stratum_pop_sum_gathered_for_hosp(
    mixed_items: list[NativeDatasetItem],
) -> None:
    batch = collate_native_batch(mixed_items)
    extra = batch.head_extra["hosp"]
    assert extra is not None and extra["stratum_pop_sum"] is not None
    stratum = cast("torch.Tensor", extra["stratum_pop_sum"])
    assert stratum.shape == (1, W)
    expected = mixed_items[1]["target_stratum_pop_sum"]
    assert expected is not None
    assert torch.equal(stratum[0], expected)


def test_partial_stratum_extra_raises() -> None:
    items = [
        _item("abs", {"hosp"}, source_id="1", weekly=True, stratum=True),
        _item("abs", {"hosp"}, source_id="2", weekly=True, stratum=False),
    ]
    with pytest.raises(ValueError, match="row-aligned"):
        collate_native_batch(items)


# ----------------------------------------------------------------------
# Uniformity assertions and guards
# ----------------------------------------------------------------------


def test_hist_shape_mismatch_raises() -> None:
    item = _item("municipality", {"cases"}, source_id="1")
    bad = _item("abs", {"hosp"}, source_id="2", weekly=True)
    bad["hist"] = torch.zeros(L, F_NATIVE + 1)
    with pytest.raises(ValueError, match="hist"):
        collate_native_batch([item, bad])


def test_target_width_mismatch_within_head_raises() -> None:
    items = [
        _item("abs", {"hosp"}, source_id="1", weekly=True, target_len=3),
        _item("abs", {"hosp"}, source_id="2", weekly=True, target_len=2),
    ]
    with pytest.raises(ValueError, match="target"):
        collate_native_batch(items)


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        collate_native_batch([])


# ----------------------------------------------------------------------
# to / pin_memory
# ----------------------------------------------------------------------


def test_to_cpu_round_trip(mixed_items: list[NativeDatasetItem]) -> None:
    batch = collate_native_batch(mixed_items)
    hist_before = batch.hist.clone()
    moved = batch.to("cpu")
    assert moved is batch
    assert moved.hist.device.type == "cpu"
    assert torch.equal(moved.hist, hist_before)
    assert moved.hist.dtype == torch.float32
    assert moved.source_type_id.dtype == torch.long
    assert moved.queries[0] is mixed_items[0]["query"]


def test_to_recurses_into_head_dicts(mixed_items: list[NativeDatasetItem]) -> None:
    batch = collate_native_batch(mixed_items).to(torch.device("meta"))
    assert batch.hist.is_meta
    assert batch.hist_mask.is_meta
    for blocks in (
        batch.head_index,
        batch.head_target,
        batch.head_target_mask,
        batch.head_population,
    ):
        for value in blocks.values():
            assert value is None or value.is_meta
    for extras in batch.head_extra.values():
        for value in extras.values():
            assert value is None or value.is_meta
    assert batch.queries == [item["query"] for item in mixed_items]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_round_trip(mixed_items: list[NativeDatasetItem]) -> None:
    batch = collate_native_batch(mixed_items)
    pinned = batch.pin_memory()
    assert pinned is batch
    assert batch.hist.is_pinned()
    assert batch.head_target["cases"] is not None
    assert cast("torch.Tensor", batch.head_target["cases"]).is_pinned()


# ----------------------------------------------------------------------
# Real-artifact integration (skipped when artifacts are absent)
# ----------------------------------------------------------------------

REAL_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "native_support"
)
REAL_CASES = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "real_with_id.zarr"
)

requires_real_artifacts = pytest.mark.skipif(
    not (REAL_ROOT / "native_graph.zarr").exists() or not REAL_CASES.exists(),
    reason="real native-support artifacts not present",
)


@requires_real_artifacts
def test_real_artifacts_collate_matches_head_counts() -> None:
    ds = NativeSupportDataset(REAL_ROOT, REAL_CASES, NativeDatasetConfig())
    # Weekly hosp width W is origin-dow dependent (plan section 11.5); keep
    # one width class so the batch satisfies the uniformity contract.
    items: list[NativeDatasetItem] = []
    hosp_width: int | None = None
    for position in range(len(ds)):
        if len(items) == 8:
            break
        item = ds[position]
        if item["head"] == "hosp":
            width = int(item["target"].shape[0])
            if hosp_width is None:
                hosp_width = width
            elif width != hosp_width:
                continue
        items.append(item)
    assert len(items) == 8

    batch = collate_native_batch(items)
    assert batch.b == 8
    assert batch.t == 60  # NativeDatasetConfig default input window
    assert batch.hist.shape == (8, 60, N_TYPES + int(items[0]["hist"].shape[1]))
    for row, item in enumerate(items):
        one_hot = batch.hist[row, :, :N_TYPES]
        assert one_hot.sum(dim=-1).tolist() == [1.0] * 60
        assert one_hot[:, int(item["source_type_id"])].eq(1.0).all()
        assert torch.equal(batch.hist[row, :, N_TYPES:], item["hist"])

    # Real-data routing is single-head (plan section 11.1): routed blocks
    # partition the batch rows and never exceed the dataset's head_counts.
    routed_total = sum(
        int(index.numel())
        for index in batch.head_index.values()
        if index is not None
    )
    assert routed_total == 8
    weekly_width = hosp_width if hosp_width is not None else H
    for head, index in batch.head_index.items():
        if index is None:
            continue
        assert head in ds.head_counts
        assert index.tolist() == sorted(index.tolist())
        assert int(index.numel()) <= ds.head_counts[head]
        targets = batch.head_target[head]
        masks = batch.head_target_mask[head]
        populations = batch.head_population[head]
        assert targets is not None and masks is not None and populations is not None
        expected_width = weekly_width if head == "hosp" else 28
        assert targets.shape == (int(index.numel()), expected_width)
        assert masks.shape == (int(index.numel()), expected_width)
        for row, position in enumerate(index.tolist()):
            assert populations[row].item() == pytest.approx(
                float(items[position]["target_population_graph"])
            )

    assert batch.origin_dow.tolist() == [item["origin_dow"] for item in items]
    assert batch.source_type_id.tolist() == [
        int(item["source_type_id"]) for item in items
    ]
    if batch.head_index["hosp"] is not None:
        extra = batch.head_extra["hosp"]
        assert extra is not None
        stratum = extra["stratum_pop_sum"]
        assert stratum is not None
        assert stratum.shape == (
            sum(
                1
                for item in items
                if item["head"] == "hosp"
            ),
            hosp_width,
        )
