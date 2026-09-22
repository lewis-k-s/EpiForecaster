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
from models.reporting_operators import (
    weekly_interval_sum,
    weekly_log1p_per100k_location,
)

pytestmark = pytest.mark.epiforecaster

L = 12
F_NATIVE = 5  # 3 clinical channels + 2 shared covariate columns (item-level)
D_CTX = 4
D_IN = 24
OBS_CONTEXT_DIM = 16
ORIGIN = 100
POP_MUNI = 19_000.0
POP_ABS = 50_000.0
POP_COMARCA = 60_000.0


def _model(
    horizon: int,
    f_native: int,
    *,
    observation_heads: ObservationHeadConfig | None = None,
) -> NativeEpiForecaster:
    """Tiny model whose history width matches the batch under test.

    ``f_native`` is derived from the actual batch ``hist`` width (the
    collated batch prepends the frozen 5-channel source-type one-hot ahead
    of the item's native channels), never assumed from a constant.
    """
    torch.manual_seed(1234)
    config = NativeForecasterConfig(
        f_native=f_native,
        d_ctx=D_CTX,
        forecast_horizon=horizon,
        sequence_length=L,
        d_in=D_IN,
        adapter_hidden_dim=32,
        type_embedding_dim=8,
        head_d_model=32,
        head_n_heads=2,
        head_num_layers=2,
        observation_heads=(
            observation_heads
            if observation_heads is not None
            else ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM)
        ),
    )
    model = NativeEpiForecaster(config)
    model.eval()
    return model


def _identity_hosp_heads() -> ObservationHeadConfig:
    """Hosp head config whose delay kernel is the exact identity.

    A length-1 kernel support softmax-normalizes to the single weight 1.0,
    so the delay kernel is ``k(x) = x`` regardless of learnability — the
    regression anchor under which the R5 weekly composition reduces to the
    raw ROP-1 formula on the admission flow.
    """
    return ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM, kernel_length_hosp=1)


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
    abs_item = _item(
        "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
        horizon=14, pop=POP_ABS,
    )
    single_batch = collate_native_batch([abs_item])
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
    model = _model(horizon=14, f_native=mixed_batch.hist.shape[2])
    single = model(single_batch)
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

    cases_only_batch = manual_batch(frozenset({"cases"}))
    cases_plus_hosp_batch = manual_batch(frozenset({"cases", "hosp"}))
    model = _model(horizon=14, f_native=cases_only_batch.hist.shape[2])
    cases_only = model(cases_only_batch)
    cases_plus_hosp = model(cases_plus_hosp_batch)
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
    """Weekly keys align with hand-computed ROP-1 semantics (H=14).

    Under the identity hosp kernel the R5 composition (delay kernel, then
    interval sum, then ``log1p(sums · clinical_scale · 1e5)``)
    reduces to the previous raw-flow formula with the head's clinical
    scale (init 0.05, the ~5% hospitalization rate) folded in.
    """
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
    model = _model(
        horizon=14,
        f_native=batch.hist.shape[2],
        observation_heads=_identity_hosp_heads(),
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

    clinical_scale = model.base.hosp_head.get_scale().item()
    for row, dow in ((0, 0), (1, 6)):
        sums = _naive_weekly_sums(source[row, 1:], dow)
        for week, week_sum in enumerate(sums):
            expected = math.log1p(week_sum * clinical_scale * 1e5)
            assert loc[row, week].item() == pytest.approx(expected, rel=1e-5)
            # Inverting the single transform recovers the raw weekly sum:
            # sum-before-transform, not a sum of transformed days.
            recovered = math.expm1(loc[row, week].item()) / (
                clinical_scale * 1e5
            )
            assert recovered == pytest.approx(week_sum, rel=1e-4)
    # Regression anchor: with the identity kernel the composition equals
    # the raw-flow reporting operator with the clinical scale folded in
    # (R5 reduces to the pre-R5 formula up to the learned scale).
    reference_sums, _ = weekly_interval_sum(source, batch.origin_dow, horizon=14)
    reference = weekly_log1p_per100k_location(reference_sums * clinical_scale)
    assert torch.allclose(loc, reference, atol=1e-6)
    # Masked trailing slot is an exact zero sum -> log1p(0) location.
    assert loc[0, 1].item() == 0.0
    assert bool(torch.isfinite(scale).all())


def test_weekly_w_max_normalization_h21() -> None:
    """H=21 always emits W_max=3 slots; trailing slots zero + mask 0."""
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
    model = _model(
        horizon=21,
        f_native=batch.hist.shape[2],
        observation_heads=_identity_hosp_heads(),
    )
    assert model.weekly_slots == 3
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
        21,
        model.base.hosp_head.delay_kernel,
        model.base.hosp_head.get_scale(),
    )[:2]
    assert week_mask[0].tolist() == [True, True, False]
    assert sums[0, 2].item() == 0.0


def test_gradients_reach_adapter_backbone_and_heads() -> None:
    """loss = mean(cases loc + deaths loc + weekly hosp loc) reaches params."""
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
                "comarca", frozenset({"deaths"}), source_id="06",
                origin_dow=3, horizon=14, pop=POP_COMARCA,
            ),
        ]
    )
    model = _model(horizon=14, f_native=batch.hist.shape[2])
    out = model(batch)
    loss = (
        out["dist_cases_loc"].mean()
        + out["dist_deaths_loc"].mean()
        + out["dist_hosp_weekly_loc"].mean()
        # Scale terms: the predictive scales are distribution parameters
        # outside the locations, so they are added to exercise every head's
        # scale-projection route.
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
        # The weekly path routes through the hosp kernel and clinical scale
        # (R5); under the default config the kernel is frozen (buffers only),
        # so its live parameter routes are the scale and the scale projection.
        model.base.hosp_head: ("delay_kernel", "scale", "distribution_scale_proj"),
    }
    for head, prefixes in head_paths.items():
        for name, param in head.named_parameters():
            if name.startswith(prefixes):
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name


def test_output_shape_contract() -> None:
    """All outputs follow the section 11.4 conventions [B, .] / [B_h, W_max]."""
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
    model = _model(horizon=14, f_native=batch.hist.shape[2])
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


def test_per_source_embedding_raises_not_implemented() -> None:
    """R13: the per-source static embedding is deferred beyond v1."""
    config = NativeForecasterConfig(
        f_native=F_NATIVE,
        d_ctx=D_CTX,
        forecast_horizon=14,
        sequence_length=L,
        d_in=D_IN,
        use_per_source_embedding=True,
        head_d_model=32,
        head_n_heads=2,
        head_num_layers=2,
        observation_heads=ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM),
    )
    with pytest.raises(NotImplementedError, match="deferred beyond v1"):
        NativeEpiForecaster(config)


def test_weekly_hosp_gradients_reach_kernel_and_scale() -> None:
    """R5: a weekly-location loss reaches the hosp delay kernel and scale.

    The default config freezes the hosp kernel (buffers hold no
    parameters), so connectivity is proven with a learnable-kernel config.
    """
    heads = ObservationHeadConfig(
        obs_context_dim=OBS_CONTEXT_DIM,
        learnable_kernel_hosp=True,
        learnable_scale_hosp=True,
    )
    batch = collate_native_batch(
        [
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
                horizon=14, pop=POP_ABS,
            ),
            _item(
                "abs", frozenset({"hosp"}), source_id="089", origin_dow=3,
                horizon=14, pop=POP_ABS,
            ),
        ]
    )
    model = _model(
        horizon=14, f_native=batch.hist.shape[2], observation_heads=heads
    )
    assert list(model.base.hosp_head.delay_kernel.named_parameters())
    out = model(batch)
    out["dist_hosp_weekly_loc"].sum().backward()
    for name, param in model.base.hosp_head.delay_kernel.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
    scale = model.base.hosp_head.scale
    assert scale.grad is not None
    assert torch.isfinite(scale.grad).all()
    assert scale.grad.abs().item() > 0


def test_cases_anchoring_wiring() -> None:
    """R15: delta forecasting anchors only cases-routed rows from history.

    Hand example: the anchored location continues the last valid cases
    history value (masked decoys are ignored); rows without a valid anchor
    and rows not routed to cases stay untouched, as do the rollout, the
    deaths head, the weekly hosp block, and the cases scale path.
    """
    off_heads = ObservationHeadConfig(obs_context_dim=OBS_CONTEXT_DIM)
    on_heads = ObservationHeadConfig(
        obs_context_dim=OBS_CONTEXT_DIM, delta_forecasting=True
    )
    observed = _item(
        "municipality", frozenset({"cases"}), source_id="08019",
        origin_dow=0, horizon=14, pop=POP_MUNI,
    )
    hist_observed = torch.zeros(L, F_NATIVE)
    hist_observed[:, 0] = torch.arange(1, L + 1) * 0.25
    hist_observed[:10, 1] = 1.0  # last valid observation: index 9 -> 2.5
    hist_observed[10:, 0] = 99.0  # masked decoys must not become the anchor
    observed["hist"] = hist_observed
    blind = _item(
        "municipality", frozenset({"cases"}), source_id="08020",
        origin_dow=0, horizon=14, pop=POP_MUNI,
    )
    hist_blind = torch.zeros(L, F_NATIVE)
    hist_blind[:, 0] = 5.0  # all-zero mask: no valid anchor
    blind["hist"] = hist_blind
    batch = collate_native_batch(
        [
            observed,
            blind,
            _item(
                "abs", frozenset({"hosp"}), source_id="088", origin_dow=6,
                horizon=14, pop=POP_ABS,
            ),
        ]
    )
    off = _model(horizon=14, f_native=batch.hist.shape[2], observation_heads=off_heads)
    on = _model(horizon=14, f_native=batch.hist.shape[2], observation_heads=on_heads)
    assert torch.equal(off.base.cases_head.scale, on.base.cases_head.scale)
    out_off, out_on = off(batch), on(batch)

    # Anchoring never touches the rollout, the deaths head, the weekly hosp
    # block, or the obs-context-driven cases scale.
    for key in (
        "I_trajectory",
        "hospitalization_flow",
        "dist_deaths_loc",
        "dist_deaths_scale",
        "dist_cases_scale",
        "dist_hosp_weekly_loc",
        "dist_hosp_weekly_scale",
        "hosp_weekly_mask",
    ):
        assert torch.equal(out_off[key], out_on[key]), key

    raw = out_off["dist_cases_loc"]
    # Row 0: anchored on the last valid history value (2.5), decoys ignored.
    expected_row = torch.clamp(2.5 + raw[0] - raw[0, :1], min=0.0)
    assert torch.allclose(out_on["dist_cases_loc"][0], expected_row, atol=1e-6)
    assert not torch.allclose(out_on["dist_cases_loc"][0], raw[0])
    # Row 1 has no valid observation; row 2 is not routed to cases.
    assert torch.equal(out_on["dist_cases_loc"][1], raw[1])
    assert torch.equal(out_on["dist_cases_loc"][2], raw[2])
