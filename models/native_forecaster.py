"""Native-support forecaster adapter (plan sections 12.2 MOD-1, 11.4-11.5).

Composes the existing :class:`~models.epiforecaster.EpiForecaster`: the base
contributes its temporal backbone, SIRHD roll-forward and observation heads
unchanged, constructed with a minimal ``ModelVariant`` (all municipal input
features disabled) so its own width arithmetic is bypassed. The base's
``forward`` is never called; its head-composition pattern is mirrored here so
the native path stays free of municipal four-channel semantics.

W_max normalization (orchestrator ruling resolving BAT-1's width finding):
weekly hospitalization outputs are always computed at the fixed width
``W_max = ceil(horizon / 7)`` slots for every row, regardless of the origin
weekday. :func:`~models.reporting_operators.weekly_interval_sum` zeroes the
sums of absent/partial slots and ``hosp_weekly_mask`` marks completeness, so
a Monday-origin H=21 row emits three slots with the third masked off while a
Sunday-origin row uses all three. The dataset may emit per-dow widths
``W <= W_max`` in ``head_target``; consumers gather against this module's
``W_max`` outputs, never the reverse.

Weekly scale and residual (frozen v1 choices, plan section 11.5):

- The weekly mechanistic location is
  :func:`~models.reporting_operators.weekly_log1p_per100k_location` applied
  to the summed fraction-space admission flow with the head's audited graph
  population (sum before transform). The hosp head's delay kernel is NOT
  routed: its ~21-day reporting kernel would smear across weekly slots.
- The predictive scale reuses the hosp head's obs-context-driven scale
  projection (:meth:`ClinicalObservationHead.get_distribution_scale`) on a
  week-aligned context: the mean of the t0-augmented ``obs_context`` over
  each week's days. Complete weeks average 7 days, partial weeks their
  available days; fully absent weeks (mask 0) receive a zero context and are
  loss-masked downstream.
- The weekly residual is frozen to zero in v1 and weekly anchoring /
  delta-forecasting is disabled: the head's residual projection is not
  directly callable without the kernel path, because the residual
  contribution is defined by the head's composition internals (alpha
  scaling, anchor composition, residual log-location budget); reusing the
  bare projection would duplicate private semantics.

The adapter consumes ``hist``, ``ctx_feat``, ``source_type_id`` and
``source_pop``. The standalone ``temporal_covariates`` batch field is
deliberately not consumed a second time: DS-1 already embeds the shared
covariates in ``hist``'s trailing columns. ``ctx_coverage`` and ``head_extra``
belong to ENC-1/LOS-1, not the model forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from einops import repeat
import torch.nn as nn
import torch.nn.functional as F

from data.native_batch import SOURCE_TYPE_ORDER, NativeEpiBatch
from models.configs import (
    InitWeightsConfig,
    ModelVariant,
    ObservationHeadConfig,
    SIRPhysicsConfig,
)
from models.epiforecaster import EpiForecaster
from models.reporting_operators import (
    weekly_interval_sum,
    weekly_log1p_per100k_location,
)
from utils.precision_policy import MODEL_PARAM_DTYPE

logger = logging.getLogger(__name__)

__all__ = [
    "NativeEpiForecaster",
    "NativeForecasterConfig",
    "NativeInputAdapter",
]


@dataclass(frozen=True)
class NativeForecasterConfig:
    """Widths and base-model wiring for :class:`NativeEpiForecaster`.

    ``f_native`` is the unified history width from the batch (the query's
    three clinical channels plus the ``k`` shared temporal covariates),
    ``d_ctx`` the fixed-pooling context width. ``d_in`` is the backbone input
    width the adapter emits; it is the single source of truth for the
    composed base's input arithmetic.
    """

    f_native: int
    d_ctx: int
    forecast_horizon: int = 14
    sequence_length: int = 14
    d_in: int = 64
    adapter_hidden_dim: int = 64
    type_embedding_dim: int = 8
    use_per_source_embedding: bool = False
    num_sources: int = 0
    head_d_model: int = 96
    head_n_heads: int = 2
    head_num_layers: int = 2
    head_dropout: float = 0.1
    sir_physics: SIRPhysicsConfig = field(default_factory=SIRPhysicsConfig)
    observation_heads: ObservationHeadConfig = field(default_factory=ObservationHeadConfig)
    init_weights: InitWeightsConfig = field(default_factory=InitWeightsConfig)
    strict: bool = True

    def __post_init__(self) -> None:
        for name in ("f_native", "forecast_horizon", "sequence_length", "d_in",
                     "adapter_hidden_dim", "type_embedding_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_ctx < 0:
            raise ValueError(f"d_ctx must be non-negative, got {self.d_ctx}")
        if self.use_per_source_embedding and self.num_sources <= 0:
            raise ValueError(
                "use_per_source_embedding requires num_sources > 0, "
                f"got {self.num_sources}"
            )


class NativeInputAdapter(nn.Module):
    """Project unified native histories into the backbone input width.

    Concatenated along features: the unified history block ``hist
    [B, L, F_native]``, the fixed-pooling context ``ctx_feat [B, L, D_ctx]``,
    the source-type embedding (``num_embeddings=5`` per
    ``SOURCE_TYPE_ORDER``) expanded over ``L``, and a ``log1p(population)``
    scalar channel expanded over ``L``. A small MLP (Linear → SiLU → Linear)
    maps the concat to ``d_in``.

    Initialization is standard small Xavier with zero biases — deliberately
    NOT zero-init so the adapter contributes signal from the first step; the
    backbone's ReZero gates already damp its early influence.

    The optional per-source static embedding (plan section 11.7) is OFF by
    default; when enabled, :meth:`forward` requires a ``source_index [B]``
    tensor of registry positions.
    """

    def __init__(self, config: NativeForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.type_embedding = nn.Embedding(
            num_embeddings=len(SOURCE_TYPE_ORDER),
            embedding_dim=config.type_embedding_dim,
        )
        input_dim = config.f_native + config.d_ctx + config.type_embedding_dim + 1
        if config.use_per_source_embedding:
            self.per_source_embedding: nn.Embedding | None = nn.Embedding(
                num_embeddings=config.num_sources,
                embedding_dim=config.type_embedding_dim,
            )
            input_dim += config.type_embedding_dim
        else:
            self.per_source_embedding = None
        self.input_proj = nn.Linear(input_dim, config.adapter_hidden_dim)
        self.output_proj = nn.Linear(config.adapter_hidden_dim, config.d_in)
        nn.init.xavier_normal_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.to(MODEL_PARAM_DTYPE)

    def forward(
        self,
        batch: NativeEpiBatch,
        source_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble per-row native features; returns ``[B, L, d_in]``.

        Raises:
            ValueError: If the per-source static embedding is enabled but
                ``source_index`` is not provided.
        """
        hist = batch.hist
        seq_len = hist.shape[1]
        features: list[torch.Tensor] = [
            hist,
            batch.ctx_feat,
            repeat(
                self.type_embedding(batch.source_type_id.long()),
                "b e -> b l e",
                l=seq_len,
            ),
            repeat(
                torch.log1p(batch.source_pop.float()).to(hist.dtype),
                "b -> b l 1",
                l=seq_len,
            ),
        ]
        if self.per_source_embedding is not None:
            if source_index is None:
                raise ValueError(
                    "source_index is required when the per-source static "
                    "embedding is enabled"
                )
            features.append(
                repeat(
                    self.per_source_embedding(source_index.long()),
                    "b e -> b l e",
                    l=seq_len,
                )
            )
        hidden = F.silu(self.input_proj(torch.cat(features, dim=-1)))
        return self.output_proj(hidden)


def _build_base_forecaster(config: NativeForecasterConfig) -> EpiForecaster:
    """Minimal base forecaster: backbone + SIRHD + heads, nothing else.

    All variant features are disabled, so the base's input width arithmetic
    reduces to the adapter output width declared via
    ``temporal_covariates_dim=d_in`` (the adapter output enters the backbone
    as a pre-assembled feature block). ``use_population`` stays off because
    population enters through the adapter's ``log1p(population)`` channel.
    """
    return EpiForecaster(
        variant_type=ModelVariant(
            cases=False, biomarkers=False, mobility=False, regions=False
        ),
        sir_physics=config.sir_physics,
        observation_heads=config.observation_heads,
        init_weights=config.init_weights,
        temporal_input_dim=0,
        sequence_length=config.sequence_length,
        forecast_horizon=config.forecast_horizon,
        use_population=False,
        temporal_covariates_dim=config.d_in,
        head_d_model=config.head_d_model,
        head_n_heads=config.head_n_heads,
        head_num_layers=config.head_num_layers,
        head_dropout=config.head_dropout,
        strict=config.strict,
    )


def _weekly_hosp_mechanics(
    hospitalization_flow: torch.Tensor,
    origin_dow: torch.Tensor,
    population: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weekly ABS mechanistic block (plan section 11.5).

    Prepends the zero nowcast column so :func:`weekly_interval_sum` drops it
    (the weekly head observes the admission interval flow, not a t=0
    nowcast), sums complete Mon-Sun weeks, and applies the single weekly
    ``log1p(per-100k)`` transform with the audited graph population.

    Returns:
        ``(mechanistic_loc [B, W_max], week_mask [B, W_max] bool,
        observation_source [B, H + 1])``.
    """
    batch_size = hospitalization_flow.shape[0]
    source = torch.cat(
        [
            torch.zeros(
                batch_size, 1, device=hospitalization_flow.device,
                dtype=hospitalization_flow.dtype,
            ),
            hospitalization_flow,
        ],
        dim=1,
    )
    sums, week_mask = weekly_interval_sum(source, origin_dow, horizon=horizon)
    loc = weekly_log1p_per100k_location(sums, population.to(sums.dtype))
    return loc, week_mask, source


def _week_aligned_obs_context(
    obs_context_with_init: torch.Tensor,
    origin_dow: torch.Tensor,
    horizon: int,
    weekly_slots: int,
) -> torch.Tensor:
    """Mean t0-augmented obs context per epi week, ``[B, H + 1, C] -> [B, W_max, C]``.

    The week partition mirrors :func:`weekly_interval_sum` exactly: week
    ``w`` starts at the ``w``-th Monday inside ``(origin, origin + H]``.
    Complete weeks average their 7 days, partial weeks the days they
    contain, and weeks fully outside the horizon yield a zero context (their
    outputs are loss-masked downstream).
    """
    batch_size, _, context_dim = obs_context_with_init.shape
    device = obs_context_with_init.device
    dtype = obs_context_with_init.dtype
    first_monday = torch.remainder(-origin_dow.long() - 1, 7)
    day_offsets = torch.arange(horizon, device=device)
    rel = day_offsets.unsqueeze(0) - first_monday.unsqueeze(1)
    week_of_day = torch.div(rel, 7, rounding_mode="floor").clamp(
        0, weekly_slots - 1
    )
    in_interval = (rel >= 0).to(dtype)

    ctx_days = obs_context_with_init[:, 1:, :]
    week_idx = week_of_day.unsqueeze(-1).expand(-1, -1, context_dim)
    sums = torch.zeros(
        batch_size, weekly_slots, context_dim, device=device, dtype=dtype
    )
    sums = sums.scatter_add(1, week_idx, ctx_days * in_interval.unsqueeze(-1))
    counts = torch.zeros(batch_size, weekly_slots, device=device, dtype=dtype)
    counts = counts.scatter_add(1, week_of_day, in_interval)
    weekly_mean = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return weekly_mean * (counts > 0).unsqueeze(-1).to(dtype)


class NativeEpiForecaster(nn.Module):
    """Native-support forecaster composing a minimal :class:`EpiForecaster`.

    One shared per-row rollout (adapter -> backbone -> SIRHD) feeds every
    routed head; routing decides which supervised blocks are produced, never
    the trajectory (plan section 4). Cases and deaths distributions are
    produced for all rows; the weekly ABS block is produced for the rows
    routed to ``hosp``. Output keys follow the base's conventions plus the
    weekly keys (``pred_hosp_weekly``, ``dist_hosp_weekly_loc/scale``,
    ``mechanistic_hosp_weekly``, ``residual_hosp_weekly``,
    ``hosp_weekly_mask``, ``observation_source_hosp_weekly``) and the echoed
    query metadata (``source_type_id``, ``origin_dow``, ``head_index``).
    """

    def __init__(self, config: NativeForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.forecast_horizon = config.forecast_horizon
        self.weekly_slots = -(-config.forecast_horizon // 7)
        self.input_adapter = NativeInputAdapter(config)
        self.base = _build_base_forecaster(config)
        logger.info(
            "Initialized NativeEpiForecaster: f_native=%d d_ctx=%d d_in=%d "
            "horizon=%d W_max=%d",
            config.f_native,
            config.d_ctx,
            config.d_in,
            config.forecast_horizon,
            self.weekly_slots,
        )

    def forward(self, batch: NativeEpiBatch) -> dict[str, torch.Tensor | dict[str, torch.Tensor | None]]:
        """Run the native-support forward pass over a collated batch."""
        x_seq = self.input_adapter(batch)
        encoder_outputs = self.base.backbone(x_seq)
        beta_t = encoder_outputs["beta_t"]
        obs_context = encoder_outputs["obs_context"]
        initial_states = F.softmax(encoder_outputs["initial_states_logits"], dim=-1)

        # SIRHD in fraction space, matching the base composition: the state
        # vector sums to 1 at t0, so the effective population is N=1.
        sir_outputs = self.base.sir_rollforward(
            beta_t=beta_t,
            gamma_t=encoder_outputs["gamma_t"],
            mortality_t=encoder_outputs["mortality_t"],
            S0=initial_states[:, 0],
            I0=initial_states[:, 1],
            R0=initial_states[:, 3],
            H0=initial_states[:, 2],
            hospitalization_rate_t=encoder_outputs["hospitalization_rate_t"],
            hospital_recovery_t=encoder_outputs["hospital_recovery_t"],
            hospital_mortality_t=encoder_outputs["hospital_mortality_t"],
            population=torch.ones_like(beta_t[:, 0]),
        )
        i_traj = sir_outputs["I_trajectory"]
        death_flow = sir_outputs["death_flow"]
        hospitalization_flow = sir_outputs["hospitalization_flow"]

        obs_context_with_init = torch.cat(
            [obs_context[:, :1, :], obs_context], dim=1
        )

        # Anchoring disabled for native v1 (plan section 11.5):
        # last_observed stays None, which keeps the anchor composition inert
        # regardless of the shared head anchor_mode configuration.
        dist_cases = self.base.cases_head.forward_distribution(
            I_trajectory=i_traj,
            obs_context=obs_context_with_init,
            last_observed=None,
            last_observed_mask=None,
        )
        dist_deaths = self.base.deaths_head.forward_distribution(
            I_trajectory=death_flow,
            obs_context=obs_context,
            last_observed=None,
        )

        weekly = self._weekly_hosp_outputs(
            batch, hospitalization_flow, obs_context_with_init
        )

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor | None]] = {
            "beta_t": beta_t,
            "gamma_t": encoder_outputs["gamma_t"],
            "mortality_t": encoder_outputs["mortality_t"],
            "hospitalization_rate_t": encoder_outputs["hospitalization_rate_t"],
            "hospital_recovery_t": encoder_outputs["hospital_recovery_t"],
            "hospital_mortality_t": encoder_outputs["hospital_mortality_t"],
            "S_trajectory": sir_outputs["S_trajectory"],
            "I_trajectory": i_traj,
            "H_trajectory": sir_outputs["H_trajectory"],
            "R_trajectory": sir_outputs["R_trajectory"],
            "D_trajectory": sir_outputs["D_trajectory"],
            "hospitalization_flow": hospitalization_flow,
            "death_flow": death_flow,
            "physics_residual": sir_outputs["physics_residual"],
            "obs_context": obs_context,
            "initial_states": initial_states,
            "pred_cases": dist_cases["loc"],
            "dist_cases_loc": dist_cases["loc"],
            "dist_cases_scale": dist_cases["scale"],
            "mechanistic_cases": dist_cases["mechanistic_loc"],
            "residual_cases": dist_cases["residual_loc"],
            "residual_input_cases": dist_cases["residual_input"],
            "observation_source_cases": i_traj,
            "pred_deaths": dist_deaths["loc"],
            "dist_deaths_loc": dist_deaths["loc"],
            "dist_deaths_scale": dist_deaths["scale"],
            "mechanistic_deaths": dist_deaths["mechanistic_loc"],
            "residual_deaths": dist_deaths["residual_loc"],
            "residual_input_deaths": dist_deaths["residual_input"],
            "observation_source_deaths": death_flow,
            **weekly,
            "source_type_id": batch.source_type_id,
            "origin_dow": batch.origin_dow,
            "head_index": dict(batch.head_index),
        }
        return outputs

    def _weekly_hosp_outputs(
        self,
        batch: NativeEpiBatch,
        hospitalization_flow: torch.Tensor,
        obs_context_with_init: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Weekly ABS head block at fixed ``W_max`` width (module docstring)."""
        rows = batch.head_index.get("hosp")
        horizon = self.forecast_horizon
        slots = self.weekly_slots
        device = hospitalization_flow.device
        dtype = hospitalization_flow.dtype
        if rows is None or rows.numel() == 0:
            empty = torch.zeros(0, slots, device=device, dtype=dtype)
            return {
                "pred_hosp_weekly": empty,
                "dist_hosp_weekly_loc": empty,
                "dist_hosp_weekly_scale": empty,
                "mechanistic_hosp_weekly": empty,
                "residual_hosp_weekly": empty,
                "hosp_weekly_mask": torch.zeros(
                    0, slots, device=device, dtype=torch.bool
                ),
                "observation_source_hosp_weekly": torch.zeros(
                    0, horizon + 1, device=device, dtype=dtype
                ),
            }

        rows = rows.long()
        population = batch.head_population.get("hosp")
        if population is None:
            raise ValueError(
                "hosp-routed batch is missing head_population['hosp']"
            )
        origin_dow = batch.origin_dow.index_select(0, rows)
        mechanistic_loc, week_mask, source = _weekly_hosp_mechanics(
            hospitalization_flow.index_select(0, rows),
            origin_dow,
            population,
            horizon,
        )
        week_context = _week_aligned_obs_context(
            obs_context_with_init.index_select(0, rows),
            origin_dow,
            horizon,
            slots,
        )
        # Frozen v1 choices (module docstring): scale from the hosp head's
        # obs-context scale projection on the week-aligned context; residual
        # exactly zero with anchoring disabled.
        scale = self.base.hosp_head.get_distribution_scale(
            obs_context=week_context, like=mechanistic_loc
        )
        residual = torch.zeros_like(mechanistic_loc)
        loc = mechanistic_loc + residual
        return {
            "pred_hosp_weekly": loc,
            "dist_hosp_weekly_loc": loc,
            "dist_hosp_weekly_scale": scale,
            "mechanistic_hosp_weekly": mechanistic_loc,
            "residual_hosp_weekly": residual,
            "hosp_weekly_mask": week_mask,
            "observation_source_hosp_weekly": source,
        }
