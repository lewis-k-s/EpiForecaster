"""MOD-1 tests: native forecaster adapter (plan section 12.2).

CPU-only and seeded. Tiny ``NativeEpiForecaster`` instances prove the plan
section 9 "queries and batching are correct" criterion: single-query vs
mixed-batch equivalence, routing-invariant rollouts, weekly ``W_max``
normalization aligned with the ROP-1 reporting operators, gradient reach
through adapter/backbone/heads, and the section 11.4 output shape contract.
"""

from __future__ import annotations

import math

import pytest
import torch

from data.native_batch import (
    SOURCE_TYPE_IDS,
    NativeEpiBatch,
    collate_native_batch,
)
from data.native_dataset import NativeDatasetItem
from data.query_contract import ForecastQuery, SourceType
from models.configs import ObservationHeadConfig
from models.native_forecaster import (
    NativeEpiForecaster,
    NativeForecasterConfig,
    _weekly_hosp_mechanics,
)

pytestmark = pytest.mark.epiforecaster

L = 12
F_NATIVE = 5  # 3 clinical channels + 2 shared covariate columns
D_CTX = 4
D_IN = 24
OBS_CONTEXT_DIM = 16
ORIGIN = 100
POP_MUNI = 19_000.0
POP_ABS = 50_000.0
POP_COMARCA = 60_000.0


def _model(horizon: int) -> NativeEpiForecaster:
    torch.manual_seed(1234)
    config = NativeForecasterConfig(
        f_native=F_NATIVE,
        d_ctx=D_CTX,
        forecast_horizon=horizon,
        sequence_length=L,
        d_in=D_IN,
        adapter_hidden_dim=32,
        type_embedding_dim=8,
        head_d_model=32,
        head_n_heads=2,
        head_num_layers=2,
        observation_heads=ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM),
    )
    model = NativeEpiForecaster(config)
    model.eval()
    return model


def _item(
    source_type: SourceType,
    heads: frozenset[str],
    *,
    source_id: str,
    origin_dow: int,
    horizon: int,
    pop: float,
) -> NativeDatasetItem:
    """Hand-built DS-1 item to the collation contract (no dataset, no I/O)."""
    weekly = "hosp" in heads
    target_width = -(-horizon // 7) if weekly else horizon
    value = float(int(source_id)) % 7.0
    return {
        "query": ForecastQuery(
            source_type=source_type,
            source_id=source_id,
            forecast_origin=ORIGIN,
            horizon_days=horizon,
            requested_heads=heads,
        ),
        "head": sorted(heads)[0],
        "hist": torch.arange(L * F_NATIVE, dtype=torch.float32).reshape(
            L, F_NATIVE
        ) + value,
        "hist_mask": torch.ones(L, dtype=torch.bool),
        "ctx_feat": torch.full((L, D_CTX), value),
        "ctx_coverage": 0.5,
        "temporal_covariates": torch.full((L, 2), value),
        "source_type_id": SOURCE_TYPE_IDS[source_type],
        "source_pop": pop,
        "origin_dow": origin_dow,
        "target": torch.arange(target_width, dtype=torch.float32) + 3.0,
        "target_mask": torch.ones(target_width, dtype=torch.bool),
        "target_population_graph": pop,
        "target_stratum_pop_sum": None,
    }


def test_single_query_matches_mixed_batch() -> None:
    """The same query produces identical outputs alone vs inside a batch."""
    model = _model(horizon=14)
    abs_item = _item(
        "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
        horizon=14, pop=POP_ABS,
    )
    single = model(collate_native_batch([abs_item]))
    mixed_batch = collate_native_batch(
        [
            _item(
                "municipality", frozenset({"cases"}), source_id="08019",
                origin_dow=0, horizon=14, pop=POP_MUNI,
            ),
            abs_item,
            _item(
                "comarca", frozenset({"deaths"}), source_id="06",
                origin_dow=3, horizon=14, pop=POP_COMARCA,
            ),
        ]
    )
    mixed = model(mixed_batch)
    row = 1  # abs query position inside the mixed batch
    for key in (
        "S_trajectory",
        "I_trajectory",
        "H_trajectory",
        "R_trajectory",
        "D_trajectory",
        "hospitalization_flow",
        "death_flow",
        "dist_cases_loc",
        "dist_cases_scale",
        "dist_deaths_loc",
        "dist_deaths_scale",
    ):
        assert torch.allclose(single[key], mixed[key][row], atol=1e-6), key
    # Weekly blocks are gathered over hosp-routed rows: both batches have
    # exactly one, so its block index is 0 in both.
    for key in (
        "dist_hosp_weekly_loc",
        "dist_hosp_weekly_scale",
        "mechanistic_hosp_weekly",
    ):
        assert torch.allclose(single[key], mixed[key][0], atol=1e-6), key
    assert torch.equal(single["hosp_weekly_mask"][0], mixed["hosp_weekly_mask"][0])
    # Eval-mode rerun is bit-identical (seeded CPU determinism).
    assert torch.equal(mixed["I_trajectory"], model(mixed_batch)["I_trajectory"])


def test_requested_heads_do_not_change_trajectories() -> None:
    """Adding a head to a query's routing leaves its rollout untouched."""
    model = _model(horizon=14)

    def manual_batch(muni_heads: frozenset[str]) -> NativeEpiBatch:
        has_hosp = "hosp" in muni_heads
        w_max = 2
        queries = [
            ForecastQuery(
                source_type="municipality",
                source_id="08019",
                forecast_origin=ORIGIN,
                horizon_days=14,
                requested_heads=muni_heads,
            ),
            ForecastQuery(
                source_type="comarca",
                source_id="06",
                forecast_origin=ORIGIN,
                horizon_days=14,
                requested_heads=frozenset({"deaths"}),
            ),
        ]
        hist = torch.stack(
            [
                torch.arange(L * F_NATIVE, dtype=torch.float32).reshape(
                    L, F_NATIVE
                ),
                torch.arange(L * F_NATIVE, dtype=torch.float32).reshape(
                    L, F_NATIVE
                ) + 7.0,
            ]
        )
        return NativeEpiBatch(
            queries=queries,
            hist=hist,
            hist_mask=torch.ones(2, L, dtype=torch.bool),
            source_type_id=torch.tensor([0, 2]),
            source_pop=torch.tensor([POP_MUNI, POP_COMARCA]),
            origin_dow=torch.tensor([0, 3]),
            ctx_feat=torch.arange(2 * L * D_CTX, dtype=torch.float32).reshape(
                2, L, D_CTX
            ),
            temporal_covariates=torch.arange(2 * L * 2, dtype=torch.float32)
            .reshape(2, L, 2),
            head_index={
                "cases": torch.tensor([0]),
                "hosp": torch.tensor([0]) if has_hosp else None,
                "deaths": torch.tensor([1]),
            },
            head_target={
                "cases": torch.arange(14, dtype=torch.float32).unsqueeze(0),
                "hosp": (
                    torch.arange(2, dtype=torch.float32).unsqueeze(0)
                    if has_hosp
                    else None
                ),
                "deaths": torch.arange(14, dtype=torch.float32).unsqueeze(0),
            },
            head_target_mask={
                "cases": torch.ones(1, 14, dtype=torch.bool),
                "hosp": (
                    torch.ones(1, w_max, dtype=torch.bool) if has_hosp else None
                ),
                "deaths": torch.ones(1, 14, dtype=torch.bool),
            },
            head_population={
                "cases": torch.tensor([POP_MUNI]),
                "hosp": torch.tensor([POP_MUNI]) if has_hosp else None,
                "deaths": torch.tensor([POP_COMARCA]),
            },
            ctx_coverage=torch.tensor([0.75, 1.0]),
            head_extra={"cases": {}, "hosp": {}, "deaths": {}},
        )

    cases_only = model(manual_batch(frozenset({"cases"})))
    cases_plus_hosp = model(manual_batch(frozenset({"cases", "hosp"})))
    for key in (
        "S_trajectory",
        "I_trajectory",
        "H_trajectory",
        "R_trajectory",
        "D_trajectory",
        "hospitalization_flow",
        "death_flow",
    ):
        assert torch.equal(cases_only[key], cases_plus_hosp[key]), key
    # The hosp routing block itself appears only in the union batch.
    assert cases_only["head_index"]["hosp"] is None
    assert cases_plus_hosp["head_index"]["hosp"].tolist() == [0]
    assert cases_only["dist_hosp_weekly_loc"].shape == (0, 2)
    assert cases_plus_hosp["dist_hosp_weekly_loc"].shape == (1, 2)


def _naive_weekly_sums(daily: torch.Tensor, origin_dow: int) -> list[float]:
    """Independent Python reference for complete Mon-Sun week sums.

    ``daily`` holds the fraction-space flow for days origin+1..origin+H
    (no nowcast). Week ``w`` starts at the ``w``-th Monday inside the
    interval and is complete iff its Sunday is inside too.
    """
    values = daily.detach().double().tolist()
    first_monday = (-(origin_dow + 1)) % 7
    sums = []
    start = first_monday
    while start + 7 <= len(values):
        sums.append(sum(values[start : start + 7]))
        start += 7
    return sums


def test_weekly_outputs_match_reporting_operator() -> None:
    """Weekly keys align with hand-computed ROP-1 semantics (H=14)."""
    model = _model(horizon=14)
    batch = collate_native_batch(
        [
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=0,
                horizon=14, pop=POP_ABS,
            ),
            _item(
                "abs", frozenset({"hosp"}), source_id="089", origin_dow=6,
                horizon=14, pop=POP_ABS,
            ),
        ]
    )
    out = model(batch)
    loc = out["mechanistic_hosp_weekly"]
    mask = out["hosp_weekly_mask"]
    scale = out["dist_hosp_weekly_scale"]
    source = out["observation_source_hosp_weekly"]
    assert loc.shape == (2, 2)
    assert mask.shape == (2, 2) and scale.shape == (2, 2)
    assert source.shape == (2, 15)
    # Per-dow variation per ROP-1: Monday origin (dow=0) has one complete
    # week; Sunday origin (dow=6) has both weeks complete.
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [True, True]
    assert out["pred_hosp_weekly"][0, 1].item() == 0.0

    population = batch.head_population["hosp"]
    for row, dow in ((0, 0), (1, 6)):
        sums = _naive_weekly_sums(source[row, 1:], dow)
        for week, week_sum in enumerate(sums):
            expected = math.log1p(week_sum * population[row].item() * 1e5)
            assert loc[row, week].item() == pytest.approx(expected, rel=1e-5)
            # Inverting the single transform recovers the raw weekly sum:
            # sum-before-transform, not a sum of transformed days.
            recovered = math.expm1(loc[row, week].item()) / (
                population[row].item() * 1e5
            )
            assert recovered == pytest.approx(week_sum, rel=1e-4)
    # Masked trailing slot is an exact zero sum -> log1p(0) location.
    assert loc[0, 1].item() == 0.0
    assert bool(torch.isfinite(scale).all())


def test_weekly_w_max_normalization_h21() -> None:
    """H=21 always emits W_max=3 slots; trailing slots zero + mask 0."""
    model = _model(horizon=21)
    assert model.weekly_slots == 3
    batch = collate_native_batch(
        [
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=0,
                horizon=21, pop=POP_ABS,
            ),
            _item(
                "abs", frozenset({"hosp"}), source_id="089", origin_dow=6,
                horizon=21, pop=POP_ABS,
            ),
        ]
    )
    out = model(batch)
    loc = out["mechanistic_hosp_weekly"]
    mask = out["hosp_weekly_mask"]
    assert loc.shape == (2, 3)
    assert mask[0].tolist() == [True, True, False]
    assert mask[1].tolist() == [True, True, True]
    # The masked Monday-origin slot carries a zero sum and zero location.
    assert loc[0, 2].item() == 0.0
    sums, week_mask = _weekly_hosp_mechanics(
        out["hospitalization_flow"],
        out["origin_dow"],
        batch.head_population["hosp"],
        21,
    )[:2]
    assert week_mask[0].tolist() == [True, True, False]
    assert sums[0, 2].item() == 0.0


def test_gradients_reach_adapter_backbone_and_heads() -> None:
    """loss = mean(cases loc + deaths loc + weekly hosp loc) reaches params."""
    model = _model(horizon=14)
    out = model(
        collate_native_batch(
            [
                _item(
                    "municipality", frozenset({"cases"}), source_id="08019",
                    origin_dow=0, horizon=14, pop=POP_MUNI,
                ),
                _item(
                    "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
                    horizon=14, pop=POP_ABS,
                ),
                _item(
                    "comarca", frozenset({"deaths"}), source_id="06",
                    origin_dow=3, horizon=14, pop=POP_COMARCA,
                ),
            ]
        )
    )
    loss = (
        out["dist_cases_loc"].mean()
        + out["dist_deaths_loc"].mean()
        + out["dist_hosp_weekly_loc"].mean()
        # Scale terms: the predictive scales are distribution parameters
        # outside the locations, so they are added to exercise every head's
        # scale-projection route (the hosp head's only live parameter path).
        + out["dist_cases_scale"].mean()
        + out["dist_deaths_scale"].mean()
        + out["dist_hosp_weekly_scale"].mean()
    )
    loss.backward()
    assert torch.isfinite(loss)

    for label, module in (
        ("adapter", model.input_adapter),
        ("backbone", model.base.backbone),
    ):
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{label}.{name}"
            assert torch.isfinite(param.grad).all(), f"{label}.{name}"

    head_paths = {
        model.base.cases_head: (
            "delay_kernel",
            "scale",
            "residual_proj",
            "alpha",
            "distribution_scale_proj",
        ),
        model.base.deaths_head: (
            "delay_kernel",
            "scale",
            "residual_proj",
            "alpha",
            "distribution_scale_proj",
        ),
        # The weekly path deliberately bypasses the hosp kernel; its scale
        # projection is the live parameter route (module docstring).
        model.base.hosp_head: ("distribution_scale_proj",),
    }
    for head, prefixes in head_paths.items():
        for name, param in head.named_parameters():
            if name.startswith(prefixes):
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name


def test_output_shape_contract() -> None:
    """All outputs follow the section 11.4 conventions [B, .] / [B_h, W_max]."""
    model = _model(horizon=14)
    batch = collate_native_batch(
        [
            _item(
                "municipality", frozenset({"cases"}), source_id="08019",
                origin_dow=0, horizon=14, pop=POP_MUNI,
            ),
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
                horizon=14, pop=POP_ABS,
            ),
            _item(
                "comarca", frozenset({"deaths"}), source_id="06", origin_dow=3,
                horizon=14, pop=POP_COMARCA,
            ),
        ]
    )
    out = model(batch)
    assert model.input_adapter(batch).shape == (3, L, D_IN)
    for key in (
        "S_trajectory",
        "I_trajectory",
        "H_trajectory",
        "R_trajectory",
        "D_trajectory",
    ):
        assert out[key].shape == (3, 15), key
    for key in ("hospitalization_flow", "death_flow", "physics_residual", "beta_t"):
        assert out[key].shape == (3, 14), key
    assert out["obs_context"].shape == (3, 14, OBS_CONTEXT_DIM)
    assert out["initial_states"].shape == (3, 4)
    for key in ("dist_cases_loc", "dist_cases_scale"):
        assert out[key].shape == (3, 15), key
    for key in ("dist_deaths_loc", "dist_deaths_scale"):
        assert out[key].shape == (3, 14), key
    for key in (
        "pred_hosp_weekly",
        "dist_hosp_weekly_loc",
        "dist_hosp_weekly_scale",
        "mechanistic_hosp_weekly",
        "residual_hosp_weekly",
    ):
        assert out[key].shape == (1, 2), key
    assert out["hosp_weekly_mask"].shape == (1, 2)
    assert out["hosp_weekly_mask"].dtype == torch.bool
    # Frozen v1 weekly residual: loc == mechanistic exactly.
    assert torch.equal(out["pred_hosp_weekly"], out["mechanistic_hosp_weekly"])
    assert torch.equal(out["pred_cases"], out["dist_cases_loc"])
    assert torch.equal(out["pred_deaths"], out["dist_deaths_loc"])
    # Query metadata echo and routing.
    assert out["source_type_id"].tolist() == [0, 1, 2]
    assert out["origin_dow"].tolist() == [0, 6, 3]
    assert out["head_index"]["cases"].tolist() == [0]
    assert out["head_index"]["hosp"].tolist() == [1]
    assert out["head_index"]["deaths"].tolist() == [2]


def test_per_source_embedding_requires_source_index() -> None:
    """The optional per-source embedding raises without an index tensor."""
    torch.manual_seed(1234)
    config = NativeForecasterConfig(
        f_native=F_NATIVE,
        d_ctx=D_CTX,
        forecast_horizon=14,
        sequence_length=L,
        d_in=D_IN,
        use_per_source_embedding=True,
        num_sources=4,
        head_d_model=32,
        head_n_heads=2,
        head_num_layers=2,
        observation_heads=ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM),
    )
    model = NativeEpiForecaster(config)
    model.eval()
    batch = collate_native_batch(
        [
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
                horizon=14, pop=POP_ABS,
            )
        ]
    )
    with pytest.raises(ValueError, match="source_index is required"):
        model.input_adapter(batch)
    indexed = model.input_adapter(batch, source_index=torch.tensor([2]))
    assert indexed.shape == (1, L, D_IN)
    with pytest.raises(ValueError, match="num_sources"):
        NativeForecasterConfig(
            f_native=F_NATIVE,
            d_ctx=D_CTX,
            use_per_source_embedding=True,
            num_sources=0,
        )
