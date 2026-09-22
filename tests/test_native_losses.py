"""LOS-1 tests: ``NativeJointInferenceLoss`` routed loss (plan section 12.2).

Synthetic, CPU, seeded: the loss is driven by hand-built model-output dicts
and hand-built :class:`~data.native_batch.NativeEpiBatch` head blocks — no
MOD-1 forward and no I/O. Expected scores use the closed-form normal
NLL/CRPS formulas computed independently here, never the implementation's
own helpers.
"""

from __future__ import annotations

import math

import pytest
import torch

from data.native_batch import SOURCE_TYPE_IDS, NativeEpiBatch
from data.query_contract import ForecastQuery, SourceType
from evaluation.losses import (
    NativeJointInferenceLoss,
    _log1p_per100k_targets,
    audit_head_counts,
    get_native_loss_from_config,
)
from models.configs import (
    JointLossConfig,
    LatentSupervisionConfig,
    LossConfig,
    ObservationHeadConfig,
)

pytestmark = pytest.mark.epiforecaster

H = 14  # daily horizon -> W_max = ceil(14 / 7) = 2
W_MAX = 2
L = 4
F_NATIVE = 3
D_CTX = 2

_POP: dict[str, float] = {
    "municipality": 250.0,
    "abs": 70_000.0,
    "comarca": 180_000.0,
    "mitma": 1.0,
    "edar": 1.0,
}


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Determinism (no randomness is consumed, but seed anyway)."""
    torch.manual_seed(1234)


# ----------------------------------------------------------------------
# Closed-form scoring references
# ----------------------------------------------------------------------


def _normal_nll(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    z = (target - loc) / scale
    return 0.5 * z**2 + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)


def _normal_crps(
    loc: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    z = (target - loc) / scale
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


# ----------------------------------------------------------------------
# Synthetic batch / model-output builders
# ----------------------------------------------------------------------


def _query(
    source_type: SourceType, source_id: str, origin: int, heads: set[str]
) -> ForecastQuery:
    return ForecastQuery(
        source_type=source_type,
        source_id=source_id,
        forecast_origin=origin,
        horizon_days=H,
        requested_heads=frozenset(heads),
    )


def _make_batch(
    *,
    queries: list[ForecastQuery],
    cases_rows: tuple[int, ...] = (),
    hosp_rows: tuple[int, ...] = (),
    deaths_rows: tuple[int, ...] = (),
    daily_width: int = H,
    hosp_width: int = W_MAX,
    cases_mask: torch.Tensor | None = None,
    deaths_mask: torch.Tensor | None = None,
    hosp_mask: torch.Tensor | None = None,
    raw_counts: torch.Tensor | None = None,
    hosp_pop: torch.Tensor | None = None,
    origin_dow: tuple[int, ...] | None = None,
) -> NativeEpiBatch:
    """Hand-built batch: only head blocks carry meaningful values; the shared
    history/context fields are deterministic filler the loss never reads."""
    b = len(queries)
    head_index: dict[str, torch.Tensor | None] = {
        "cases": None,
        "hosp": None,
        "deaths": None,
    }
    head_target = dict(head_index)
    head_target_mask = dict(head_index)
    head_population = dict(head_index)

    if cases_rows:
        head_index["cases"] = torch.tensor(cases_rows, dtype=torch.long)
        head_target["cases"] = (
            0.1 * torch.arange(1, daily_width + 1, dtype=torch.float32)
        ).repeat(len(cases_rows), 1)
        head_target_mask["cases"] = (
            cases_mask.clone()
            if cases_mask is not None
            else torch.ones(len(cases_rows), daily_width, dtype=torch.bool)
        )
        head_population["cases"] = torch.full((len(cases_rows),), 250.0)
    if deaths_rows:
        head_index["deaths"] = torch.tensor(deaths_rows, dtype=torch.long)
        head_target["deaths"] = (
            0.2 * torch.arange(1, daily_width + 1, dtype=torch.float32)
        ).repeat(len(deaths_rows), 1)
        head_target_mask["deaths"] = (
            deaths_mask.clone()
            if deaths_mask is not None
            else torch.ones(len(deaths_rows), daily_width, dtype=torch.bool)
        )
        head_population["deaths"] = torch.full((len(deaths_rows),), 180_000.0)
    if hosp_rows:
        head_index["hosp"] = torch.tensor(hosp_rows, dtype=torch.long)
        head_target["hosp"] = (
            raw_counts.clone()
            if raw_counts is not None
            else (
                3.0 * torch.arange(1, hosp_width + 1, dtype=torch.float32)
            ).repeat(len(hosp_rows), 1)
        )
        head_target_mask["hosp"] = (
            hosp_mask.clone()
            if hosp_mask is not None
            else torch.ones(len(hosp_rows), hosp_width, dtype=torch.bool)
        )
        head_population["hosp"] = (
            hosp_pop.clone()
            if hosp_pop is not None
            else torch.full((len(hosp_rows),), 70_000.0)
        )

    return NativeEpiBatch(
        queries=queries,
        hist=torch.zeros(b, L, F_NATIVE),
        hist_mask=torch.ones(b, L, dtype=torch.bool),
        source_type_id=torch.tensor(
            [SOURCE_TYPE_IDS[str(q.source_type)] for q in queries], dtype=torch.long
        ),
        source_pop=torch.tensor([_POP[str(q.source_type)] for q in queries]),
        origin_dow=(
            torch.tensor(origin_dow, dtype=torch.long)
            if origin_dow is not None
            else torch.zeros(b, dtype=torch.long)
        ),
        ctx_feat=torch.zeros(b, L, D_CTX),
        temporal_covariates=torch.zeros(b, L, 2),
        head_index=head_index,
        head_target=head_target,
        head_target_mask=head_target_mask,
        head_population=head_population,
        ctx_coverage=torch.ones(b),
        head_extra={"cases": {}, "hosp": {}, "deaths": {}},
    )


def _default_three_head_batch() -> NativeEpiBatch:
    """Real-data-style routing: municipality/cases, abs/hosp, comarca/deaths.

    The hosp row uses origin dow 3 (Thursday), whose second Mon-Sun week is
    partial for H=14 -> weekly completeness mask ``[True, False]``.
    """
    return _make_batch(
        queries=[
            _query("municipality", "08121", 10, {"cases"}),
            _query("abs", "088", 11, {"hosp"}),
            _query("comarca", "06", 12, {"deaths"}),
        ],
        cases_rows=(0,),
        hosp_rows=(1,),
        deaths_rows=(2,),
        origin_dow=(0, 3, 4),
    )


def _daily_outputs(b: int, width: int = H) -> dict[str, torch.Tensor]:
    loc = 1.0 + 0.25 * torch.arange(b * width, dtype=torch.float32).reshape(b, width)
    scale = 0.5 + 0.05 * torch.arange(b * width, dtype=torch.float32).reshape(b, width)
    return {
        "pred_cases": loc,
        "dist_cases_loc": loc,
        "dist_cases_scale": scale,
        "pred_deaths": 1.5 * loc,
        "dist_deaths_loc": 1.5 * loc,
        "dist_deaths_scale": 2.0 * scale,
    }


def _model_shaped_daily_outputs(b: int) -> dict[str, torch.Tensor]:
    """Real-model daily widths: cases ``[B, H + 1]`` with nowcast, deaths ``[B, H]``."""
    cases = _daily_outputs(b, width=H + 1)
    deaths = _daily_outputs(b, width=H)
    return {
        "pred_cases": cases["pred_cases"],
        "dist_cases_loc": cases["dist_cases_loc"],
        "dist_cases_scale": cases["dist_cases_scale"],
        "pred_deaths": deaths["pred_deaths"],
        "dist_deaths_loc": deaths["dist_deaths_loc"],
        "dist_deaths_scale": deaths["dist_deaths_scale"],
    }


def _weekly_outputs(
    rows: int,
    weekly_mask: torch.Tensor,
    slots: int = W_MAX,
    loc_offset: float = 20.0,
) -> dict[str, torch.Tensor]:
    loc = loc_offset + 0.5 * torch.arange(
        rows * slots, dtype=torch.float32
    ).reshape(rows, slots)
    scale = 1.0 + 0.25 * torch.arange(
        rows * slots, dtype=torch.float32
    ).reshape(rows, slots)
    return {
        "pred_hosp_weekly": loc,
        "dist_hosp_weekly_loc": loc,
        "dist_hosp_weekly_scale": scale,
        "hosp_weekly_mask": weekly_mask,
    }


# ----------------------------------------------------------------------
# 1. Per-head masked NLL/CRPS vs hand-computed formulas
# ----------------------------------------------------------------------


@pytest.mark.parametrize("observation_loss", ["nll", "crps"])
def test_per_head_losses_match_manual_formulas(observation_loss: str) -> None:
    """2-row 3-step example per head, one masked point each (5 active slots).

    R2: batch targets are RAW daily counts; the loss scores them against
    ``log1p(count / pop * 1e5)`` (cases pop 250, deaths pop 180000), so the
    closed-form references consume the hand-transformed targets.
    """
    queries = [
        _query("municipality", "08121", 10, {"cases", "deaths"}),
        _query("municipality", "08122", 11, {"cases", "deaths"}),
    ]
    batch = _make_batch(
        queries=queries,
        cases_rows=(0, 1),
        deaths_rows=(0, 1),
        daily_width=3,
    )
    loc_c = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 2.0]])
    scale_c = torch.tensor([[0.5, 1.0, 1.5], [0.75, 0.5, 1.25]])
    raw_c = torch.tensor([[1.5, 0.0, 2.5], [0.0, 1.0, 3.0]])
    mask_c = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    loc_d = torch.tensor([[0.5, 1.5, 2.5], [1.0, 0.0, 1.5]])
    scale_d = torch.tensor([[0.8, 1.2, 0.9], [1.1, 0.7, 1.3]])
    raw_d = torch.tensor([[1.0, 2.0, 1.5], [0.5, 0.5, 2.0]])
    mask_d = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    batch.head_target["cases"] = raw_c.clone()
    batch.head_target["deaths"] = raw_d.clone()
    batch.head_target_mask["cases"] = mask_c.bool()
    batch.head_target_mask["deaths"] = mask_d.bool()

    loss = NativeJointInferenceLoss(horizon_days=3, observation_loss=observation_loss)
    point = _normal_nll if observation_loss == "nll" else _normal_crps
    target_c = torch.log1p(raw_c / 250.0 * 1e5)
    target_d = torch.log1p(raw_d / 180_000.0 * 1e5)
    expected_cases = point(loc_c, scale_c, target_c)[mask_c > 0.5].mean()
    expected_deaths = point(loc_d, scale_d, target_d)[mask_d > 0.5].mean()

    components = loss(
        {
            "pred_cases": loc_c,
            "dist_cases_loc": loc_c,
            "dist_cases_scale": scale_c,
            "pred_deaths": loc_d,
            "dist_deaths_loc": loc_d,
            "dist_deaths_scale": scale_d,
        },
        batch,
    )
    torch.testing.assert_close(components["cases"], expected_cases)
    torch.testing.assert_close(components["deaths"], expected_deaths)
    assert components["cases_n_eff"].item() == 5.0
    assert components["deaths_n_eff"].item() == 5.0


# ----------------------------------------------------------------------
# 1b. R1: internal nowcast alignment of daily-head model tensors
# ----------------------------------------------------------------------


def test_cases_nowcast_column_dropped_internally() -> None:
    """R1: model-shaped [B, H+1] cases tensors score like manually dropped t0."""
    batch = _default_three_head_batch()
    outputs = {
        **_model_shaped_daily_outputs(3),
        **_weekly_outputs(1, torch.tensor([[True, False]])),
    }
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    components = loss(outputs, batch)

    dropped = {
        **outputs,
        "pred_cases": outputs["pred_cases"][:, 1:],
        "dist_cases_loc": outputs["dist_cases_loc"][:, 1:],
        "dist_cases_scale": outputs["dist_cases_scale"][:, 1:],
    }
    expected = loss(dropped, batch)
    torch.testing.assert_close(components["cases"], expected["cases"])
    torch.testing.assert_close(components["deaths"], expected["deaths"])

    # Hand-computed cases reference: drop t0, transform the raw targets
    # with the routed population block (250.0), score against the wide loc.
    raw = batch.head_target["cases"]
    case_rows = batch.head_index["cases"]
    expected_cases = _normal_nll(
        outputs["dist_cases_loc"].index_select(0, case_rows)[:, 1:],
        outputs["dist_cases_scale"].index_select(0, case_rows)[:, 1:],
        torch.log1p(raw / 250.0 * 1e5),
    ).mean()
    torch.testing.assert_close(components["cases"], expected_cases)
    assert components["cases_n_eff"].item() == float(H)
    assert components["deaths_n_eff"].item() == float(H)


def test_misaligned_daily_widths_raise_clear_error() -> None:
    """R1: non-conforming daily widths fail loudly instead of being truncated."""
    batch = _default_three_head_batch()
    outputs = {
        **_model_shaped_daily_outputs(3),
        **_weekly_outputs(1, torch.tensor([[True, False]])),
    }
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")

    # Deaths has no nowcast (municipal convention): [B, H+1] cannot align.
    deaths_wide = dict(outputs)
    deaths_wide["dist_deaths_loc"] = torch.cat(
        [outputs["dist_deaths_loc"], outputs["dist_deaths_loc"][:, -1:]], dim=1
    )
    with pytest.raises(ValueError, match="deaths"):
        loss(deaths_wide, batch)

    # Cases width H+2 is neither the target width nor the nowcast width.
    cases_wide = dict(outputs)
    cases_wide["dist_cases_loc"] = torch.cat(
        [outputs["dist_cases_loc"], outputs["dist_cases_loc"][:, -1:]], dim=1
    )
    with pytest.raises(ValueError, match="cases"):
        loss(cases_wide, batch)


# ----------------------------------------------------------------------
# 2. Weekly target transform + mask composition
# ----------------------------------------------------------------------


def test_weekly_target_transform_and_mask_composition() -> None:
    """log1p(counts / P * 1e5) targets; weekly completeness excludes slots."""
    batch = _make_batch(
        queries=[
            _query("abs", "088", 20, {"hosp"}),  # dow 6: two complete weeks
            _query("abs", "089", 21, {"hosp"}),  # dow 3: second week partial
        ],
        hosp_rows=(0, 1),
        raw_counts=torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        hosp_pop=torch.tensor([70_000.0, 50_000.0]),
        origin_dow=(6, 3),
    )
    weekly_mask = torch.tensor([[True, True], [True, False]])
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    components = loss(_weekly_outputs(2, weekly_mask), batch)

    # Worked number: row 0 slot 0 -> log1p(3 / 70000 * 1e5).
    active_target = torch.tensor(
        [
            math.log1p(3.0 / 70_000.0 * 1e5),
            math.log1p(4.0 / 70_000.0 * 1e5),
            math.log1p(5.0 / 50_000.0 * 1e5),
        ]
    )
    active_loc = torch.tensor([20.0, 20.5, 21.0])
    active_scale = torch.tensor([1.0, 1.25, 1.5])
    expected = _normal_nll(active_loc, active_scale, active_target).mean()
    torch.testing.assert_close(components["hosp"], expected)
    assert components["hosp_n_eff"].item() == 3.0

    # A target-mask-0 slot is weight-0: junk in its raw count cannot leak in.
    masked = _make_batch(
        queries=[
            _query("abs", "088", 20, {"hosp"}),
            _query("abs", "089", 21, {"hosp"}),
        ],
        hosp_rows=(0, 1),
        raw_counts=torch.tensor([[3.0, 999.0], [5.0, 6.0]]),
        hosp_pop=torch.tensor([70_000.0, 50_000.0]),
        hosp_mask=torch.tensor([[True, False], [True, True]]),
        origin_dow=(6, 3),
    )
    clean = _make_batch(
        queries=[
            _query("abs", "088", 20, {"hosp"}),
            _query("abs", "089", 21, {"hosp"}),
        ],
        hosp_rows=(0, 1),
        raw_counts=torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        hosp_pop=torch.tensor([70_000.0, 50_000.0]),
        hosp_mask=torch.tensor([[True, False], [True, True]]),
        origin_dow=(6, 3),
    )
    comp_masked = loss(_weekly_outputs(2, weekly_mask), masked)
    comp_clean = loss(_weekly_outputs(2, weekly_mask), clean)
    torch.testing.assert_close(comp_masked["hosp"], comp_clean["hosp"])
    # Effective weights = [[T,F],[T,T]] & [[T,T],[T,F]] -> 2 active slots.
    assert comp_masked["hosp_n_eff"].item() == 2.0
    assert comp_clean["hosp_n_eff"].item() == 2.0


def test_weekly_narrow_target_width_pads_trailing_slots() -> None:
    """Per-dow width W=1 < W_max=2: padded trailing slots get weight 0."""
    batch = _make_batch(
        queries=[
            _query("abs", "088", 20, {"hosp"}),
            _query("abs", "089", 21, {"hosp"}),
        ],
        hosp_rows=(0, 1),
        hosp_width=1,
        raw_counts=torch.tensor([[3.0], [5.0]]),
        hosp_pop=torch.tensor([70_000.0, 50_000.0]),
        origin_dow=(6, 3),
    )
    # Weekly completeness says both slots exist; the emitted target class
    # carries only slot 0, so the padded slot 1 must stay weight-0 even with
    # garbage in the model's loc there.
    weekly_mask = torch.tensor([[True, True], [True, True]])
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    components = loss(_weekly_outputs(2, weekly_mask), batch)

    active_target = torch.tensor(
        [math.log1p(3.0 / 70_000.0 * 1e5), math.log1p(5.0 / 50_000.0 * 1e5)]
    )
    active_loc = torch.tensor([20.0, 21.0])
    active_scale = torch.tensor([1.0, 1.5])
    expected = _normal_nll(active_loc, active_scale, active_target).mean()
    torch.testing.assert_close(components["hosp"], expected)
    assert components["hosp_n_eff"].item() == 2.0


# ----------------------------------------------------------------------
# 2b. R2: daily raw-count targets transformed at consumption
# ----------------------------------------------------------------------


def test_daily_targets_transform_log1p_per100k_at_consumption() -> None:
    """R2/R17: raw counts -> log1p(count / pop * 1e5) after masking.

    Hand-computed for pop 250: raw 2.0 -> log1p(800), raw 4.0 ->
    log1p(1600). The masked slot's junk raw value (NaN here, a negative in
    the invariance check) cannot leak in: masked slots are set to target 0.
    """
    pop = torch.tensor([250.0])
    weights = torch.tensor([[1.0, 0.0, 1.0]])
    target = _log1p_per100k_targets(
        torch.tensor([[2.0, float("nan"), 4.0]]), pop, weights
    )
    expected = torch.tensor(
        [[math.log1p(2.0 / 250.0 * 1e5), 0.0, math.log1p(4.0 / 250.0 * 1e5)]]
    )
    torch.testing.assert_close(target, expected)

    junk = _log1p_per100k_targets(torch.tensor([[2.0, -3.0, 4.0]]), pop, weights)
    torch.testing.assert_close(junk, expected)


def test_daily_target_transform_is_equal_rate_invariant() -> None:
    """R17: equal count/population rates have equal transformed targets."""
    transformed = _log1p_per100k_targets(
        torch.tensor([[10.0], [100.0]]),
        torch.tensor([1_000.0, 10_000.0]),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(transformed[0], transformed[1])
    torch.testing.assert_close(transformed, torch.full((2, 1), math.log1p(1_000.0)))


@pytest.mark.parametrize("population", [0.0, -1.0, float("inf"), float("nan")])
def test_daily_target_transform_rejects_invalid_population(population: float) -> None:
    with pytest.raises(ValueError, match="population"):
        _log1p_per100k_targets(
            torch.ones(1, 1), torch.tensor([population]), torch.ones(1, 1)
        )


def test_daily_raw_count_targets_score_in_log1p_per100k_space() -> None:
    """R2 end-to-end: raw-count cases score against the transformed targets,
    and a junk raw value in a masked slot leaves the loss unchanged."""
    queries = [_query("municipality", "08121", 10, {"cases"})]
    batch = _make_batch(queries=queries, cases_rows=(0,), daily_width=3)
    batch.head_target["cases"] = torch.tensor([[2.0, float("nan"), 4.0]])
    batch.head_target_mask["cases"] = torch.tensor([[True, False, True]])
    loc = torch.tensor([[0.5, 1.0, 2.0]])
    scale = torch.tensor([[1.0, 1.0, 1.0]])
    loss = NativeJointInferenceLoss(horizon_days=3, observation_loss="nll")
    components = loss({"dist_cases_loc": loc, "dist_cases_scale": scale}, batch)

    expected = _normal_nll(
        loc,
        scale,
        torch.tensor(
            [[math.log1p(2.0 / 250.0 * 1e5), 0.0, math.log1p(4.0 / 250.0 * 1e5)]]
        ),
    )[torch.tensor([[True, False, True]])].mean()
    torch.testing.assert_close(components["cases"], expected)
    assert components["cases_n_eff"].item() == 2.0

    # Invariance: cleaning the junk masked slot to 0.0 changes nothing.
    batch.head_target["cases"] = torch.tensor([[2.0, 0.0, 4.0]])
    clean = loss({"dist_cases_loc": loc, "dist_cases_scale": scale}, batch)
    torch.testing.assert_close(clean["cases"], components["cases"])


def test_daily_head_population_falls_back_to_source_pop() -> None:
    """R2: without a head_population block, pop is gathered from source_pop."""
    queries = [
        _query("municipality", "08121", 10, {"cases"}),
        _query("municipality", "08122", 11, {"cases"}),
    ]
    batch = _make_batch(queries=queries, cases_rows=(0, 1), daily_width=2)
    batch.head_population["cases"] = None
    batch.source_pop = torch.tensor([111.0, 222.0])
    batch.head_target["cases"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch.head_target_mask["cases"] = torch.ones(2, 2, dtype=torch.bool)
    loc = torch.tensor([[0.5, 1.0], [2.0, 3.0]])
    scale = torch.ones(2, 2)
    loss = NativeJointInferenceLoss(horizon_days=2, observation_loss="nll")
    components = loss({"dist_cases_loc": loc, "dist_cases_scale": scale}, batch)

    expected_target = torch.tensor(
        [
            [math.log1p(1.0 / 111.0 * 1e5), math.log1p(2.0 / 111.0 * 1e5)],
            [math.log1p(3.0 / 222.0 * 1e5), math.log1p(4.0 / 222.0 * 1e5)],
        ]
    )
    expected = _normal_nll(loc, scale, expected_target).mean()
    torch.testing.assert_close(components["cases"], expected)
    assert components["cases_n_eff"].item() == 4.0


# ----------------------------------------------------------------------
# 3. Inactive head, zero-weighting, weight renormalization
# ----------------------------------------------------------------------


def test_inactive_head_zero_weighted_and_weights_renormalize() -> None:
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    batch = _default_three_head_batch()
    outputs = {
        **_daily_outputs(3),
        **_weekly_outputs(1, torch.tensor([[True, False]])),
    }
    components = loss(outputs, batch)
    assert components["obs_active_mask"].tolist() == [False, True, True, True]
    torch.testing.assert_close(
        components["obs_weights"],
        torch.tensor([0.0, 0.95 / 3.0, 0.95 / 3.0, 0.95 / 3.0]),
    )
    for key in (
        "ww",
        "hosp",
        "cases",
        "deaths",
        "sird_supervision",
        "sird_supervision_raw",
        "ww_weighted",
        "hosp_weighted",
        "cases_weighted",
        "deaths_weighted",
        "sird_supervision_weighted",
        "ww_n_eff",
        "hosp_n_eff",
        "cases_n_eff",
        "deaths_n_eff",
        "obs_weights",
        "obs_active_mask",
        "obs_total",
        "total",
    ):
        assert key in components
    torch.testing.assert_close(
        components["total"], components["obs_total"]
    )

    # Empty hosp routing: component anchored to zero and 0-weighted; the
    # equal split renormalizes over the two remaining active heads.
    batch.head_index["hosp"] = None
    batch.head_target["hosp"] = None
    batch.head_target_mask["hosp"] = None
    batch.head_population["hosp"] = None
    components = loss(_daily_outputs(3), batch)
    assert components["obs_active_mask"].tolist() == [False, False, True, True]
    torch.testing.assert_close(
        components["obs_weights"], torch.tensor([0.0, 0.0, 0.475, 0.475])
    )
    assert components["hosp"].item() == 0.0
    assert components["hosp_weighted"].item() == 0.0
    assert components["hosp_n_eff"].item() == 0.0
    assert components["ww"].item() == 0.0
    assert components["ww_weighted"].item() == 0.0
    torch.testing.assert_close(
        components["total"], 0.475 * (components["cases"] + components["deaths"])
    )


def test_disable_flag_deactivates_routed_head() -> None:
    loss = NativeJointInferenceLoss(
        horizon_days=H, observation_loss="nll", disable_hosp=True
    )
    batch = _default_three_head_batch()
    outputs = {
        **_daily_outputs(3),
        **_weekly_outputs(1, torch.tensor([[True, False]])),
    }
    components = loss(outputs, batch)
    assert components["obs_active_mask"].tolist() == [False, False, True, True]
    assert components["hosp"].item() == 0.0


# ----------------------------------------------------------------------
# 4. Count-once audit
# ----------------------------------------------------------------------


def test_audit_head_counts_flags_duplicate_routing() -> None:
    batch = _default_three_head_batch()
    audit = audit_head_counts(batch)
    assert audit.duplicates == 0
    assert audit.supervised_points == {"cases": H, "hosp": W_MAX, "deaths": H, "ww": 0}
    assert audit.routed_rows == {"cases": 1, "hosp": 1, "deaths": 1, "ww": 0}
    assert audit.duplicate_row_positions == {"cases": (), "hosp": (), "deaths": (), "ww": ()}
    assert audit.as_log_dict()["duplicates"] == 0

    # A duplicated query collated twice routes row 0 into the cases block
    # twice; the audit flags the identity double-count (the loss dedupes it
    # before scoring — see the R7 enforcement test below).
    duplicated = _default_three_head_batch()
    duplicated.head_index["cases"] = torch.tensor([0, 0], dtype=torch.long)
    duplicated.head_target["cases"] = torch.stack(
        [duplicated.head_target["cases"][0]] * 2
    )
    duplicated.head_target_mask["cases"] = torch.ones(2, H, dtype=torch.bool)
    duplicated.head_population["cases"] = torch.tensor([250.0, 250.0])
    audit_dup = audit_head_counts(duplicated)
    assert audit_dup.duplicates == 1
    assert audit_dup.duplicate_row_positions["cases"] == (0,)
    assert audit_dup.supervised_points["cases"] == 2 * H
    assert audit_dup.as_log_dict()["duplicates"] == 1


def test_duplicate_routing_does_not_change_loss_or_counts() -> None:
    """R7: a duplicated routed query scores identically to the deduplicated
    batch — same total, per-head losses, n_eff, and weights — while the raw
    batch is still flagged by the audit."""
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    weekly = _weekly_outputs(1, torch.tensor([[True, False]]))
    clean_batch = _default_three_head_batch()
    clean = loss({**_daily_outputs(3), **weekly}, clean_batch)

    duplicated_batch = _default_three_head_batch()
    duplicated_batch.head_index["cases"] = torch.tensor([0, 0], dtype=torch.long)
    duplicated_batch.head_target["cases"] = torch.stack(
        [clean_batch.head_target["cases"][0]] * 2
    )
    duplicated_batch.head_target_mask["cases"] = torch.ones(2, H, dtype=torch.bool)
    duplicated_batch.head_population["cases"] = torch.full((2,), 250.0)
    duplicated_batch.head_index["hosp"] = torch.tensor([1, 1], dtype=torch.long)
    duplicated_batch.head_target["hosp"] = clean_batch.head_target["hosp"].repeat(
        2, 1
    )
    duplicated_batch.head_target_mask["hosp"] = torch.ones(2, W_MAX, dtype=torch.bool)
    duplicated_batch.head_population["hosp"] = torch.full((2,), 70_000.0)
    duplicated_weekly = {
        key: value.repeat(2, 1) if value.ndim == 2 else value
        for key, value in weekly.items()
    }
    duplicated = loss(
        {**_daily_outputs(3), **duplicated_weekly}, duplicated_batch
    )

    audit = audit_head_counts(duplicated_batch)
    assert audit.duplicates == 2  # malformed batch still flagged
    assert audit.duplicate_row_positions["cases"] == (0,)
    assert audit.duplicate_row_positions["hosp"] == (1,)

    torch.testing.assert_close(duplicated["total"], clean["total"])
    for head in ("cases", "hosp", "deaths"):
        torch.testing.assert_close(duplicated[head], clean[head])
        assert duplicated[f"{head}_n_eff"].item() == clean[f"{head}_n_eff"].item()
    torch.testing.assert_close(duplicated["obs_weights"], clean["obs_weights"])
    assert duplicated["cases_n_eff"].item() == float(H)
    assert duplicated["hosp_n_eff"].item() == 1.0
    assert duplicated["deaths_n_eff"].item() == float(H)


def test_audit_head_counts_narrow_weekly_block() -> None:
    batch = _make_batch(
        queries=[_query("abs", "088", 20, {"hosp"})],
        hosp_rows=(0,),
        hosp_width=1,
        origin_dow=(3,),
    )
    audit = audit_head_counts(batch)
    assert audit.supervised_points["hosp"] == 1  # block mask sum, not W_max
    assert audit.duplicates == 0


# ----------------------------------------------------------------------
# 5. Gradient flow
# ----------------------------------------------------------------------


def test_gradients_flow_to_each_active_head() -> None:
    batch = _default_three_head_batch()
    daily = _daily_outputs(3)
    cases_loc = daily["dist_cases_loc"].clone().requires_grad_(True)
    cases_scale = daily["dist_cases_scale"].clone().requires_grad_(True)
    deaths_loc = daily["dist_deaths_loc"].clone().requires_grad_(True)
    deaths_scale = daily["dist_deaths_scale"].clone().requires_grad_(True)
    weekly = _weekly_outputs(1, torch.tensor([[True, False]]))
    hosp_loc = weekly["dist_hosp_weekly_loc"].clone().requires_grad_(True)
    hosp_scale = weekly["dist_hosp_weekly_scale"].clone().requires_grad_(True)

    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    components = loss(
        {
            "pred_cases": cases_loc,
            "dist_cases_loc": cases_loc,
            "dist_cases_scale": cases_scale,
            "pred_deaths": deaths_loc,
            "dist_deaths_loc": deaths_loc,
            "dist_deaths_scale": deaths_scale,
            "pred_hosp_weekly": hosp_loc,
            "dist_hosp_weekly_loc": hosp_loc,
            "dist_hosp_weekly_scale": hosp_scale,
            "hosp_weekly_mask": weekly["hosp_weekly_mask"],
        },
        batch,
    )
    components["total"].backward()
    for name, tensor in (
        ("cases_loc", cases_loc),
        ("cases_scale", cases_scale),
        ("deaths_loc", deaths_loc),
        ("deaths_scale", deaths_scale),
        ("hosp_loc", hosp_loc),
        ("hosp_scale", hosp_scale),
    ):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.norm() > 0, name


# ----------------------------------------------------------------------
# 6. Latent supervision rejection + factory plumbing
# ----------------------------------------------------------------------


@pytest.mark.parametrize("latent_total_weight", [0.1, 1.0])
def test_latent_supervision_rejected_in_native_mode(
    latent_total_weight: float,
) -> None:
    with pytest.raises(ValueError, match="real-data-only"):
        NativeJointInferenceLoss(
            horizon_days=H, latent_total_weight=latent_total_weight
        )


def test_invalid_horizon_days_rejected() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        NativeJointInferenceLoss(horizon_days=0)


def test_factory_builds_from_defaults() -> None:
    loss = get_native_loss_from_config(
        None, observation_heads_config=ObservationHeadConfig(), horizon_days=H
    )
    assert loss.weekly_slots == W_MAX
    assert loss.observation_loss == "nll"
    assert loss.obs_weight_sum == JointLossConfig().gradnorm_obs_weight_sum
    components = loss(
        {
            **_daily_outputs(3),
            **_weekly_outputs(1, torch.tensor([[True, False]])),
        },
        _default_three_head_batch(),
    )
    assert torch.isfinite(components["total"])


def test_factory_rejects_latent_config() -> None:
    config = LossConfig(
        joint=JointLossConfig(latent=LatentSupervisionConfig(total_weight=0.3))
    )
    with pytest.raises(ValueError, match="real-data-only"):
        get_native_loss_from_config(
            config, observation_heads_config=ObservationHeadConfig(), horizon_days=H
        )
