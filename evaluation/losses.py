from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from evaluation.student_t_scores import student_t5_cdf, student_t5_crps

from models.configs import (
    DataConfig,
    JointLossConfig,
    LossConfig,
    ObservationHeadConfig,
)
from utils.device import sync_to_device
from utils.training_utils import drop_nowcast

if TYPE_CHECKING:
    from data.epi_batch import EpiBatch
    from data.native_batch import NativeEpiBatch

_LOSS_VALUE_CLAMP = 1.0e6
_RMSE_EPS = 1.0e-8
_LOG_TWO_PI = math.log(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def get_loss_from_config(
    loss_config: LossConfig | None,
    *,
    observation_heads_config: ObservationHeadConfig,
    latent_scales: dict[str, float] | None = None,
    data_config: DataConfig | None = None,
    input_window_length: int | None = None,
    forecast_horizon: int | None = None,
) -> JointInferenceLoss:
    """Build the only supported EpiForecaster loss: JointInferenceLoss.

    Predictive-scale parameterization belongs to the model observation heads, so
    their scale floor is also the canonical floor used for scoring.
    """
    if loss_config is None:
        joint_cfg = JointLossConfig()
    else:
        name_lower = (loss_config.name or "").lower()
        if name_lower != "joint_inference":
            raise ValueError(
                "Only training.loss.name='joint_inference' is supported for "
                f"EpiForecaster, got {loss_config.name!r}."
            )
        joint_cfg = loss_config.joint
    if joint_cfg.adaptive_scheme == "gradnorm" and joint_cfg.observation_loss == "nll":
        raise ValueError(
            "adaptive_scheme='gradnorm' requires a non-negative observation "
            "loss; observation_loss='nll' is signed. Use fixed weighting or "
            "a non-negative objective such as 'crps'."
        )

    zero_min_obs = {
        "cases": 0,
        "hospitalizations": 0,
        "deaths": 0,
        "wastewater": 0,
    }
    min_history_obs = zero_min_obs
    min_horizon_obs = zero_min_obs
    if (
        data_config is not None
        and input_window_length is not None
        and forecast_horizon is not None
    ):
        min_history_obs = data_config.resolve_min_history_observed_map(
            input_window_length=int(input_window_length)
        )
        min_horizon_obs = data_config.resolve_min_horizon_observed_map(
            forecast_horizon=int(forecast_horizon)
        )

    return JointInferenceLoss(
        observation_loss=joint_cfg.observation_loss,
        distribution_family=joint_cfg.distribution_family,
        distribution_df=joint_cfg.distribution_df,
        distribution_scale_floor=observation_heads_config.distribution_scale_floor,
        interval_levels=joint_cfg.interval_levels,
        obs_weight_sum=joint_cfg.gradnorm_obs_weight_sum,
        latent_total_weight=joint_cfg.latent.total_weight,
        latent_scaling=joint_cfg.latent.scaling,
        latent_component_weights=joint_cfg.latent.component_weights,
        latent_scales=latent_scales,
        latent_require_complete_targets=joint_cfg.latent.require_complete_targets,
        disable_ww=joint_cfg.disable_ww,
        disable_hosp=joint_cfg.disable_hosp,
        disable_cases=joint_cfg.disable_cases,
        disable_deaths=joint_cfg.disable_deaths,
        mask_input_ww=joint_cfg.mask_input_ww,
        mask_input_hosp=joint_cfg.mask_input_hosp,
        mask_input_cases=joint_cfg.mask_input_cases,
        mask_input_deaths=joint_cfg.mask_input_deaths,
        ww_min_history_observed=min_history_obs["wastewater"],
        hosp_min_history_observed=min_history_obs["hospitalizations"],
        cases_min_history_observed=min_history_obs["cases"],
        deaths_min_history_observed=min_history_obs["deaths"],
        ww_min_observed=min_horizon_obs["wastewater"],
        hosp_min_observed=min_horizon_obs["hospitalizations"],
        cases_min_observed=min_horizon_obs["cases"],
        deaths_min_observed=min_horizon_obs["deaths"],
    )


class JointInferenceLoss(nn.Module):
    """
    Joint inference loss combining observation and latent SIRHD supervision losses.

    This loss is designed for the joint inference framework where the model outputs
    latent SIR states and observation predictions rather than direct forecasts.
    """

    _OBS_HEADS = ("ww", "hosp", "cases", "deaths")
    _HEAD_TARGET_KEY = {
        "ww": "ww",
        "hosp": "hosp",
        "cases": "cases",
        "deaths": "deaths",
    }
    _HEAD_MASK_KEY = {
        "ww": "ww_mask",
        "hosp": "hosp_mask",
        "cases": "cases_mask",
        "deaths": "deaths_mask",
    }
    _HEAD_DISABLE_ATTR = {
        "ww": "disable_ww",
        "hosp": "disable_hosp",
        "cases": "disable_cases",
        "deaths": "disable_deaths",
    }
    _HEAD_MIN_OBS_ATTR = {
        "ww": "ww_min_observed",
        "hosp": "hosp_min_observed",
        "cases": "cases_min_observed",
        "deaths": "deaths_min_observed",
    }
    _HEAD_MIN_HISTORY_OBS_ATTR = {
        "ww": "ww_min_history_observed",
        "hosp": "hosp_min_history_observed",
        "cases": "cases_min_history_observed",
        "deaths": "deaths_min_history_observed",
    }
    _HEAD_PRED_KEY = {
        "ww": "pred_ww",
        "hosp": "pred_hosp",
        "cases": "pred_cases",
        "deaths": "pred_deaths",
    }
    _HEAD_HAS_NOWCAST = {
        "ww": True,
        "hosp": True,
        "cases": True,
        "deaths": False,
    }
    _LATENT_COMPONENTS = (
        ("latent_s", "s", "S_trajectory", "S_target"),
        ("latent_i", "i", "I_trajectory", "I_target"),
        ("latent_h", "h", "H_trajectory", "H_target"),
        ("latent_r", "r", "R_trajectory", "R_target"),
        (
            "latent_death_flow",
            "death_flow",
            "death_flow",
            "death_flow_target",
        ),
    )
    _LATENT_TASKS = ("s", "i", "h", "r", "death_flow")

    def __init__(
        self,
        observation_loss: str = "mse",
        distribution_family: str = "normal",
        distribution_df: float = 5.0,
        distribution_scale_floor: float = 1.0e-3,
        interval_levels: list[float] | tuple[float, ...] | None = None,
        obs_weight_sum: float = 0.95,
        latent_total_weight: float = 0.0,
        latent_scaling: str = "none",
        latent_component_weights: dict[str, float] | None = None,
        latent_scales: dict[str, float] | None = None,
        latent_require_complete_targets: bool = False,
        sir_residual_clip: float | None = None,
        disable_ww: bool = False,
        disable_hosp: bool = False,
        disable_cases: bool = False,
        disable_deaths: bool = False,
        mask_input_ww: bool = False,
        mask_input_hosp: bool = False,
        mask_input_cases: bool = False,
        mask_input_deaths: bool = False,
        ww_min_history_observed: int = 0,
        hosp_min_history_observed: int = 0,
        cases_min_history_observed: int = 0,
        deaths_min_history_observed: int = 0,
        ww_min_observed: int = 0,
        hosp_min_observed: int = 0,
        cases_min_observed: int = 0,
        deaths_min_observed: int = 0,
    ):
        super().__init__()
        self.observation_loss = observation_loss.lower()
        self.distribution_family = distribution_family.lower()
        self.distribution_df = float(distribution_df)
        self.distribution_scale_floor = float(distribution_scale_floor)
        self.interval_levels = tuple(
            float(v) for v in (interval_levels or (0.5, 0.8, 0.95))
        )
        self.obs_weight_sum = float(obs_weight_sum)
        del sir_residual_clip
        self.latent_total_weight = float(latent_total_weight)
        self.latent_scaling = latent_scaling.lower()
        self.latent_require_complete_targets = bool(latent_require_complete_targets)
        self.latent_component_weights = {
            name: float(value)
            for name, value in (
                latent_component_weights or {name: 0.2 for name in self._LATENT_TASKS}
            ).items()
        }
        self.latent_scales = (
            None
            if latent_scales is None
            else {name: float(value) for name, value in latent_scales.items()}
        )
        self.disable_ww = bool(disable_ww)
        self.disable_hosp = bool(disable_hosp)
        self.disable_cases = bool(disable_cases)
        self.disable_deaths = bool(disable_deaths)
        self.mask_input_ww = bool(mask_input_ww)
        self.mask_input_hosp = bool(mask_input_hosp)
        self.mask_input_cases = bool(mask_input_cases)
        self.mask_input_deaths = bool(mask_input_deaths)
        self.ww_min_history_observed = int(ww_min_history_observed)
        self.hosp_min_history_observed = int(hosp_min_history_observed)
        self.cases_min_history_observed = int(cases_min_history_observed)
        self.deaths_min_history_observed = int(deaths_min_history_observed)
        self.ww_min_observed = int(ww_min_observed)
        self.hosp_min_observed = int(hosp_min_observed)
        self.cases_min_observed = int(cases_min_observed)
        self.deaths_min_observed = int(deaths_min_observed)
        if self.obs_weight_sum <= 0:
            raise ValueError(
                f"obs_weight_sum must be positive, got {self.obs_weight_sum}"
            )
        if self.observation_loss not in {"mse", "rmse", "mae", "nll", "crps"}:
            raise ValueError(
                "observation_loss must be one of "
                "['crps', 'mae', 'mse', 'nll', 'rmse'], "
                f"got {self.observation_loss!r}"
            )
        if self.distribution_family not in {"normal", "student_t"}:
            raise ValueError(
                "distribution_family must be one of ['normal', 'student_t'], "
                f"got {self.distribution_family!r}"
            )
        if (
            self.observation_loss == "crps"
            and self.distribution_family == "student_t"
            and self.distribution_df != 5.0
        ):
            raise ValueError("Student-t CRPS requires distribution_df=5.0")
        if self.distribution_scale_floor <= 0:
            raise ValueError(
                "distribution_scale_floor must be positive, "
                f"got {self.distribution_scale_floor}"
            )
        if self.distribution_family == "student_t" and self.distribution_df <= 2.0:
            raise ValueError(
                f"distribution_df must be > 2 for student_t, got {self.distribution_df}"
            )
        if not self.interval_levels:
            raise ValueError("interval_levels must not be empty")
        for level in self.interval_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(
                    f"interval_levels values must be in (0, 1), got {level}"
                )
        if not math.isfinite(self.latent_total_weight) or self.latent_total_weight < 0:
            raise ValueError(
                "latent_total_weight must be finite and non-negative, "
                f"got {self.latent_total_weight}"
            )
        if self.latent_scaling not in {"none", "std", "asinh"}:
            raise ValueError(
                "latent_scaling must be one of ['asinh', 'none', 'std'], "
                f"got {self.latent_scaling!r}"
            )
        expected_latent_tasks = set(self._LATENT_TASKS)
        if set(self.latent_component_weights) != expected_latent_tasks:
            raise ValueError(
                "latent_component_weights must contain exactly "
                f"{sorted(expected_latent_tasks)}"
            )
        latent_weight_sum = sum(self.latent_component_weights.values())
        if any(
            not math.isfinite(value) or value < 0
            for value in self.latent_component_weights.values()
        ) or not math.isclose(latent_weight_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError(
                "latent_component_weights must be finite, non-negative, and sum to 1.0"
            )
        if self.latent_total_weight > 0 and self.latent_scaling != "none":
            if self.latent_scales is None:
                raise ValueError(
                    f"latent_scales are required for latent_scaling={self.latent_scaling!r}"
                )
            if set(self.latent_scales) != expected_latent_tasks:
                raise ValueError(
                    "latent_scales must contain exactly "
                    f"{sorted(expected_latent_tasks)}"
                )
            if any(
                not math.isfinite(value) or value <= 0
                for value in self.latent_scales.values()
            ):
                raise ValueError("latent_scales values must be finite and positive")

    @staticmethod
    def fixed_obs_weights(
        *,
        active_mask: torch.Tensor,
        obs_weight_sum: float,
    ) -> torch.Tensor:
        active_mask = active_mask.to(dtype=torch.bool)
        active_f32 = active_mask.to(dtype=torch.float32)
        num_active = active_f32.sum()
        safe_num_active = num_active.clamp_min(1.0)
        equal_weight = (
            torch.as_tensor(
                float(obs_weight_sum),
                dtype=torch.float32,
                device=active_mask.device,
            )
            / safe_num_active
        )
        weights = active_f32 * equal_weight
        return torch.where(num_active > 0, weights, torch.zeros_like(weights))

    def compose_total_loss(
        self,
        *,
        components: dict[str, torch.Tensor],
        obs_active_mask: torch.Tensor,
        obs_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compose total loss from raw components with fixed or provided obs weights."""
        obs_losses = torch.stack(
            [
                components["ww"].float(),
                components["hosp"].float(),
                components["cases"].float(),
                components["deaths"].float(),
            ]
        )

        if obs_weights is None:
            obs_weights = self.fixed_obs_weights(
                active_mask=obs_active_mask, obs_weight_sum=self.obs_weight_sum
            ).to(obs_losses.device)
        else:
            obs_weights = obs_weights.float().to(obs_losses.device)

        obs_weighted = obs_weights * obs_losses
        obs_total = obs_weighted.sum()
        sird_supervision_weighted = (
            self.latent_total_weight * components["sird_supervision"]
        )
        total = obs_total + sird_supervision_weighted
        return {
            "obs_weights": obs_weights,
            "obs_total": obs_total,
            "total": total,
            "ww_weighted": obs_weighted[0],
            "hosp_weighted": obs_weighted[1],
            "cases_weighted": obs_weighted[2],
            "deaths_weighted": obs_weighted[3],
            "sird_supervision_weighted": sird_supervision_weighted,
        }

    @staticmethod
    def _build_supervision_weights(
        *,
        target: torch.Tensor,
        observed_mask: torch.Tensor | None,
        history_observed_mask: torch.Tensor | None = None,
        min_history_observed: int = 0,
        min_horizon_observed: int = 0,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Build effective per-point supervision weights for a head using strict hard masking."""
        target, observed_mask = sync_to_device(target, observed_mask, device=device)

        target_f32 = target.float()
        finite_mask = torch.isfinite(target_f32).to(dtype=torch.float32)
        if observed_mask is None:
            observed = finite_mask
        else:
            observed = torch.nan_to_num(
                observed_mask.float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(min=0.0, max=1.0)

        # STRICT HARD MASK: Only supervise where observed_mask > 0.5
        observed_binary = (observed > 0.5).to(dtype=torch.float32) * finite_mask

        if min_horizon_observed > 0:
            observed_counts = observed_binary.sum(dim=1, keepdim=True)
            eligible = (observed_counts >= float(min_horizon_observed)).to(
                observed_binary.dtype
            )
            observed_binary = observed_binary * eligible

        if min_history_observed > 0:
            if history_observed_mask is None:
                raise ValueError(
                    "History observation mask is required when "
                    "min_history_observed is positive"
                )
            history_mask = history_observed_mask.to(device=device).float()
            history_binary = (
                torch.nan_to_num(
                    history_mask,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                > 0.5
            ).to(observed_binary.dtype)
            history_counts = history_binary.sum(dim=1, keepdim=True)
            history_eligible = (history_counts >= float(min_history_observed)).to(
                observed_binary.dtype
            )
            observed_binary = observed_binary * history_eligible

        n_eff = observed_binary.sum()
        return {
            "target": target_f32,
            "weights": observed_binary,  # Weights are strictly the binary mask
            "observed_binary": observed_binary,
            "n_eff": n_eff,
            "active": n_eff > 0,
        }

    @staticmethod
    def _weighted_masked_error_from_weights(
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute weighted masked error from precomputed effective weights."""
        prediction_f32 = prediction.float()
        target_f32 = target.float()
        weights_f32 = weights.float()
        active = weights_f32 > 0

        prediction_clean = torch.nan_to_num(
            prediction_f32,
            nan=0.0,
            posinf=_LOSS_VALUE_CLAMP,
            neginf=-_LOSS_VALUE_CLAMP,
        ).clamp(min=-_LOSS_VALUE_CLAMP, max=_LOSS_VALUE_CLAMP)
        target_clean = torch.nan_to_num(
            target_f32,
            nan=0.0,
            posinf=_LOSS_VALUE_CLAMP,
            neginf=-_LOSS_VALUE_CLAMP,
        ).clamp(min=-_LOSS_VALUE_CLAMP, max=_LOSS_VALUE_CLAMP)
        prediction_safe = torch.where(
            active, prediction_clean, torch.zeros_like(prediction_clean)
        )
        target_safe = torch.where(active, target_clean, torch.zeros_like(target_clean))

        abs_error = (prediction_safe - target_safe).abs()
        if loss_type == "mae":
            point_loss = abs_error
        else:
            point_loss = abs_error**2

        mean_loss = (point_loss * weights_f32).sum() / weights_f32.sum().clamp_min(1e-8)
        if loss_type == "rmse":
            return torch.sqrt(mean_loss + _RMSE_EPS) - torch.sqrt(
                torch.as_tensor(
                    _RMSE_EPS, dtype=mean_loss.dtype, device=mean_loss.device
                )
            )
        return mean_loss

    def _weighted_masked_observation_loss_from_weights(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._weighted_masked_error_from_weights(
            prediction=prediction,
            target=target,
            weights=weights,
            loss_type=self.observation_loss,
        )

    @staticmethod
    def _sanitize_distribution_loc(loc: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            loc.float(),
            nan=0.0,
            posinf=_LOSS_VALUE_CLAMP,
            neginf=-_LOSS_VALUE_CLAMP,
        ).clamp(min=-_LOSS_VALUE_CLAMP, max=_LOSS_VALUE_CLAMP)

    def _sanitize_distribution_scale(self, scale: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            scale.float(),
            nan=self.distribution_scale_floor,
            posinf=_LOSS_VALUE_CLAMP,
            neginf=self.distribution_scale_floor,
        ).clamp(min=self.distribution_scale_floor, max=_LOSS_VALUE_CLAMP)

    @staticmethod
    def _normal_crps(
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        z = (target - loc) / scale
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)
        return crps.clamp_min(0.0)

    def _distribution_point_loss(
        self,
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.observation_loss == "nll":
            if self.distribution_family == "normal":
                z = (target - loc) / scale
                return 0.5 * z**2 + torch.log(scale) + 0.5 * _LOG_TWO_PI

            df = torch.as_tensor(
                self.distribution_df,
                dtype=loc.dtype,
                device=loc.device,
            )
            dist = torch.distributions.StudentT(df, loc=loc, scale=scale)
            return -dist.log_prob(target)

        if self.observation_loss == "crps":
            if self.distribution_family == "student_t":
                return student_t5_crps(loc=loc, scale=scale, target=target)
            return self._normal_crps(loc=loc, scale=scale, target=target)

        raise ValueError(
            "distribution loss requested for unsupported observation_loss "
            f"{self.observation_loss!r}"
        )

    def _weighted_masked_distribution_loss_from_weights(
        self,
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked NLL/CRPS from distribution parameters."""
        loc_clean = self._sanitize_distribution_loc(loc)
        scale_clean = self._sanitize_distribution_scale(scale)
        target_f32 = target.float()
        weights_f32 = weights.float()
        active = weights_f32 > 0
        target_clean = torch.nan_to_num(
            target_f32,
            nan=0.0,
            posinf=_LOSS_VALUE_CLAMP,
            neginf=-_LOSS_VALUE_CLAMP,
        ).clamp(min=-_LOSS_VALUE_CLAMP, max=_LOSS_VALUE_CLAMP)

        point_loss = self._distribution_point_loss(
            loc=loc_clean,
            scale=scale_clean,
            target=target_clean,
        )
        if self.observation_loss == "nll":
            point_loss = torch.nan_to_num(
                point_loss,
                nan=_LOSS_VALUE_CLAMP,
                posinf=_LOSS_VALUE_CLAMP,
                neginf=-_LOSS_VALUE_CLAMP,
            ).clamp(min=-_LOSS_VALUE_CLAMP, max=_LOSS_VALUE_CLAMP)
        else:
            point_loss = torch.nan_to_num(
                point_loss,
                nan=_LOSS_VALUE_CLAMP,
                posinf=_LOSS_VALUE_CLAMP,
                neginf=0.0,
            ).clamp(min=0.0, max=_LOSS_VALUE_CLAMP)
        point_loss = torch.where(active, point_loss, torch.zeros_like(point_loss))
        return (point_loss * weights_f32).sum() / weights_f32.sum().clamp_min(1e-8)

    @classmethod
    def _align_head_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        head_name: str,
        horizon: int,
    ) -> torch.Tensor:
        if cls._HEAD_HAS_NOWCAST[head_name]:
            return drop_nowcast(tensor, horizon)
        return tensor

    def _get_head_distribution_params(
        self,
        model_outputs: dict[str, torch.Tensor],
        *,
        head_name: str,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_key = self._HEAD_PRED_KEY[head_name]
        loc = model_outputs.get(f"dist_{head_name}_loc", model_outputs[pred_key])
        scale = model_outputs.get(f"dist_{head_name}_scale")
        loc = self._align_head_tensor(loc, head_name=head_name, horizon=horizon)
        if scale is None:
            scale = torch.ones_like(loc)
        else:
            scale = self._align_head_tensor(scale, head_name=head_name, horizon=horizon)
        return loc, scale

    def _compute_head_observation_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        *,
        head_name: str,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        horizon = int(target.shape[1])
        loc, scale = self._get_head_distribution_params(
            model_outputs, head_name=head_name, horizon=horizon
        )
        if self.observation_loss in {"nll", "crps"}:
            return self._weighted_masked_distribution_loss_from_weights(
                loc=loc,
                scale=scale,
                target=target,
                weights=weights,
            )

        return self._weighted_masked_observation_loss_from_weights(
            prediction=loc,
            target=target,
            weights=weights,
        )

    @classmethod
    def _weighted_masked_mse_from_weights(
        cls,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted masked MSE from precomputed effective weights."""
        return cls._weighted_masked_error_from_weights(
            prediction=prediction,
            target=target,
            weights=weights,
            loss_type="mse",
        )

    def _transform_latent_pair(
        self,
        *,
        task_name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform one latent prediction-target pair in FP32."""
        prediction_f32 = prediction.float()
        target_f32 = target.float()
        if self.latent_scaling == "none":
            return prediction_f32, target_f32

        assert self.latent_scales is not None
        scale = torch.as_tensor(
            self.latent_scales[task_name],
            dtype=torch.float32,
            device=prediction_f32.device,
        )
        prediction_scaled = prediction_f32 / scale
        target_scaled = target_f32 / scale
        if self.latent_scaling == "std" or task_name in {"s", "r"}:
            return prediction_scaled, target_scaled
        return torch.asinh(prediction_scaled), torch.asinh(target_scaled)

    def compute_observation_supervision(
        self,
        targets: dict[str, torch.Tensor | None],
        *,
        device: torch.device,
        batch_data: EpiBatch | None = None,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """Single source of truth for per-head active status and effective weights."""
        out: dict[str, dict[str, torch.Tensor | None]] = {}
        zero = torch.tensor(0.0, device=device)
        inactive = torch.tensor(False, device=device, dtype=torch.bool)

        for head_name in self._OBS_HEADS:
            target_key = self._HEAD_TARGET_KEY[head_name]
            disable_attr = self._HEAD_DISABLE_ATTR[head_name]
            target = targets.get(target_key)
            if bool(getattr(self, disable_attr)) or target is None:
                out[head_name] = {
                    "target": None,
                    "weights": None,
                    "observed_binary": None,
                    "n_eff": zero,
                    "active": inactive,
                }
                continue

            mask_key = self._HEAD_MASK_KEY[head_name]
            min_observed_attr = self._HEAD_MIN_OBS_ATTR[head_name]
            min_history_observed_attr = self._HEAD_MIN_HISTORY_OBS_ATTR[head_name]
            out[head_name] = self._build_supervision_weights(
                target=target,
                observed_mask=targets.get(mask_key),
                history_observed_mask=self._get_history_observed_mask(
                    batch_data, head_name
                ),
                min_history_observed=int(getattr(self, min_history_observed_attr)),
                min_horizon_observed=int(getattr(self, min_observed_attr)),
                device=device,
            )

        return out

    @staticmethod
    def _get_history_observed_mask(
        batch_data: EpiBatch | None,
        head_name: str,
    ) -> torch.Tensor | None:
        if batch_data is None:
            return None
        if head_name == "ww":
            return getattr(batch_data, "ww_hist_mask", None)

        history = {
            "hosp": getattr(batch_data, "hosp_hist", None),
            "cases": getattr(batch_data, "cases_hist", None),
            "deaths": getattr(batch_data, "deaths_hist", None),
        }[head_name]
        if history is None:
            return None
        return history[..., 1]

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: EpiBatch | None = None,
    ) -> torch.Tensor:
        components = self.compute_components(model_outputs, targets, batch_data)
        return components["total"]

    def compute_components(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: EpiBatch | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute joint inference loss components.

        Args:
            model_outputs: Dict from EpiForecaster.forward() containing:
                - pred_ww: [B, H+1] predicted wastewater (includes t=0 nowcast)
                - pred_hosp: [B, H+1] predicted hospitalizations (includes t=0 nowcast)
                - pred_cases: [B, H+1] predicted reported cases (includes t=0 nowcast)
                - pred_deaths: [B, H] predicted deaths (no nowcast needed)
                - S_trajectory/I_trajectory/H_trajectory/R_trajectory/D_trajectory: [B, H+1]
            targets: Dict containing target tensors:
                - ww: [B, H] wastewater targets (optional)
                - hosp: [B, H] hospitalization targets (optional)
                - cases: [B, H] reported cases targets (optional)
                - deaths: [B, H] deaths targets (optional)
                - S_target/I_target/H_target/R_target: [B, H+1] state targets (optional)
                - death_flow_target: [B, H] destination-dated flow target (optional)
            batch_data: Batch history masks used for per-head window eligibility.

        Returns:
            Dict with unweighted and weighted component losses plus total:
                - ww, hosp, cases, deaths, latent_s, latent_i, latent_h,
                  latent_r, latent_death_flow, sird_supervision
                - ww_weighted, hosp_weighted, cases_weighted, deaths_weighted,
                  sird_supervision_weighted
                - total
        """
        # Keep losses attached to graph while avoiding NaN propagation from non-finite preds.
        zero_anchor = (
            torch.nan_to_num(
                model_outputs["pred_ww"].float(), nan=0.0, posinf=0.0, neginf=0.0
            ).sum()
            * 0.0
        )
        ww_loss = zero_anchor
        hosp_loss = zero_anchor
        cases_loss = zero_anchor
        deaths_loss = zero_anchor
        latent_losses = {
            component_name: zero_anchor
            for component_name, _task_name, _output_key, _target_key in self._LATENT_COMPONENTS
        }
        latent_raw_losses = dict(latent_losses)
        latent_weighted_losses = dict(latent_losses)
        sird_supervision_raw_loss = zero_anchor
        sird_supervision_loss = zero_anchor
        obs_supervision = self.compute_observation_supervision(
            targets,
            device=zero_anchor.device,
            batch_data=batch_data,
        )
        obs_active_mask = torch.stack(
            [
                cast(torch.Tensor, obs_supervision["ww"]["active"]),
                cast(torch.Tensor, obs_supervision["hosp"]["active"]),
                cast(torch.Tensor, obs_supervision["cases"]["active"]),
                cast(torch.Tensor, obs_supervision["deaths"]["active"]),
            ]
        ).to(dtype=torch.bool)

        ww_n_eff = cast(torch.Tensor, obs_supervision["ww"]["n_eff"]).float()
        hosp_n_eff = cast(torch.Tensor, obs_supervision["hosp"]["n_eff"]).float()
        cases_n_eff = cast(torch.Tensor, obs_supervision["cases"]["n_eff"]).float()
        deaths_n_eff = cast(torch.Tensor, obs_supervision["deaths"]["n_eff"]).float()

        ww_target = obs_supervision["ww"]["target"]
        ww_weights = obs_supervision["ww"]["weights"]
        if ww_target is not None and ww_weights is not None:
            ww_base = self._compute_head_observation_loss(
                model_outputs,
                head_name="ww",
                target=ww_target,
                weights=ww_weights,
            )
            ww_loss = ww_base

        hosp_target = obs_supervision["hosp"]["target"]
        hosp_weights = obs_supervision["hosp"]["weights"]
        if hosp_target is not None and hosp_weights is not None:
            hosp_base = self._compute_head_observation_loss(
                model_outputs,
                head_name="hosp",
                target=hosp_target,
                weights=hosp_weights,
            )
            hosp_loss = hosp_base

        cases_target = obs_supervision["cases"]["target"]
        cases_weights = obs_supervision["cases"]["weights"]
        if cases_target is not None and cases_weights is not None:
            cases_base = self._compute_head_observation_loss(
                model_outputs,
                head_name="cases",
                target=cases_target,
                weights=cases_weights,
            )
            cases_loss = cases_base

        deaths_target = obs_supervision["deaths"]["target"]
        deaths_weights = obs_supervision["deaths"]["weights"]
        if deaths_target is not None and deaths_weights is not None:
            deaths_base = self._compute_head_observation_loss(
                model_outputs,
                head_name="deaths",
                target=deaths_target,
                weights=deaths_weights,
            )
            deaths_loss = deaths_base

        if self.latent_total_weight > 0:
            missing_targets = [
                target_key
                for _component_name, _task_name, _output_key, target_key in self._LATENT_COMPONENTS
                if targets.get(target_key) is None
            ]
            if self.latent_require_complete_targets and missing_targets:
                raise ValueError(
                    "Complete latent supervision requires all five targets; "
                    f"missing {missing_targets}"
                )
            if self.latent_require_complete_targets:
                invalid_targets = []
                for (
                    _component_name,
                    _task_name,
                    _output_key,
                    target_key,
                ) in self._LATENT_COMPONENTS:
                    target = targets[target_key]
                    assert isinstance(target, torch.Tensor)
                    target_f32 = target.float()
                    if (
                        not torch.isfinite(target_f32).all()
                        or not (target_f32 >= 0).all()
                    ):
                        invalid_targets.append(target_key)
                if invalid_targets:
                    raise ValueError(
                        "Complete latent supervision targets must be finite and "
                        f"nonnegative; invalid {invalid_targets}"
                    )

            active_components: list[tuple[str, str]] = []
            for (
                component_name,
                task_name,
                output_key,
                target_key,
            ) in self._LATENT_COMPONENTS:
                target = targets.get(target_key)
                if target is None:
                    continue
                supervision = self._build_supervision_weights(
                    target=target,
                    observed_mask=None,
                    min_horizon_observed=0,
                    device=zero_anchor.device,
                )
                latent_target = supervision["target"]
                latent_weights = supervision["weights"]
                if latent_target is None or latent_weights is None:
                    continue
                raw_loss = self._weighted_masked_mse_from_weights(
                    prediction=model_outputs[output_key],
                    target=latent_target,
                    weights=latent_weights,
                )
                transformed_prediction, transformed_target = (
                    self._transform_latent_pair(
                        task_name=task_name,
                        prediction=model_outputs[output_key],
                        target=latent_target,
                    )
                )
                transformed_loss = self._weighted_masked_mse_from_weights(
                    prediction=transformed_prediction,
                    target=transformed_target,
                    weights=latent_weights,
                )
                latent_raw_losses[component_name] = raw_loss
                latent_losses[component_name] = transformed_loss
                active_components.append((component_name, task_name))

            active_weight_sum = sum(
                self.latent_component_weights[task_name]
                for _component_name, task_name in active_components
            )
            if active_weight_sum > 0:
                for component_name, task_name in active_components:
                    normalized_weight = (
                        self.latent_component_weights[task_name] / active_weight_sum
                    )
                    latent_weighted_losses[component_name] = (
                        normalized_weight * latent_losses[component_name]
                    )
                    sird_supervision_raw_loss = (
                        sird_supervision_raw_loss
                        + normalized_weight * latent_raw_losses[component_name]
                    )
                    sird_supervision_loss = (
                        sird_supervision_loss + latent_weighted_losses[component_name]
                    )

        components = {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            **latent_losses,
            **{f"{name}_raw": value for name, value in latent_raw_losses.items()},
            **{
                f"{name}_weighted": value
                for name, value in latent_weighted_losses.items()
            },
            "sird_supervision_raw": sird_supervision_raw_loss,
            "sird_supervision": sird_supervision_loss,
        }
        totals = self.compose_total_loss(
            components=components,
            obs_active_mask=obs_active_mask,
        )

        return {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            **latent_losses,
            **{f"{name}_raw": value for name, value in latent_raw_losses.items()},
            **{
                f"{name}_weighted": value
                for name, value in latent_weighted_losses.items()
            },
            "sird_supervision_raw": sird_supervision_raw_loss,
            "sird_supervision": sird_supervision_loss,
            "ww_weighted": totals["ww_weighted"],
            "hosp_weighted": totals["hosp_weighted"],
            "cases_weighted": totals["cases_weighted"],
            "deaths_weighted": totals["deaths_weighted"],
            "sird_supervision_weighted": totals["sird_supervision_weighted"],
            "ww_n_eff": ww_n_eff,
            "hosp_n_eff": hosp_n_eff,
            "cases_n_eff": cases_n_eff,
            "deaths_n_eff": deaths_n_eff,
            "obs_weights": totals["obs_weights"],
            "obs_active_mask": obs_active_mask,
            "obs_total": totals["obs_total"],
            "total": totals["total"],
        }

    def compute_components_train(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: EpiBatch | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compile-safe train path (kept separate to avoid eval-path regressions)."""
        return self.compute_components(model_outputs, targets, batch_data)


# ---------------------------------------------------------------------------
# Native-support routed loss (plan sections 12.2 LOS-1, 11.1, 11.5)
# ---------------------------------------------------------------------------

#: Per-100k constant of the shared ``log1p(per-100k * interval)`` target
#: space (plan sections 11.4/11.5; mirrors ``models.reporting_operators``).
_PER_100K = 100_000.0

#: Heads with routed per-head blocks in :class:`NativeEpiBatch` (section 11.4).
_NATIVE_ROUTED_HEADS = ("ww", "cases", "hosp", "deaths")


def device_zero(device: torch.device) -> torch.Tensor:
    """Zero scalar tensor on ``device`` (inactive-head anchor)."""
    return torch.tensor(0.0, device=device)


def device_false(device: torch.device) -> torch.Tensor:
    """False bool scalar tensor on ``device`` (inactive-head flag)."""
    return torch.tensor(False, device=device, dtype=torch.bool)


def _binary_target_mask(mask: torch.Tensor) -> torch.Tensor:
    """Strict binary supervision mask (base ``_build_supervision_weights`` semantics)."""
    observed = torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return observed > 0.5


@dataclass(frozen=True)
class NativeHeadCountAudit:
    """Per-head section 7 supervision counts and count-once audit (LOS-1).

    ``supervised_points`` counts weight-eligible target slots per head block
    (``head_target_mask > 0.5``; the loss additionally intersects this with
    the weekly completeness mask for ``hosp``). ``duplicate_row_positions``
    lists, per head, the batch-row positions whose routing identity
    ``(source_type, source_id, forecast_origin, horizon_days, realization_id)``
    occurs more than once in that head's ``head_index``; ``duplicates`` is the
    total number of beyond-once occurrences across heads.
    """

    supervised_points: dict[str, int]
    routed_rows: dict[str, int]
    duplicate_row_positions: dict[str, tuple[int, ...]]
    duplicates: int

    def as_log_dict(self) -> dict[str, object]:
        """Plain-dict view for section 7 structured logging."""
        return {
            "supervised_points": dict(self.supervised_points),
            "routed_rows": dict(self.routed_rows),
            "duplicate_row_positions": {
                head: list(positions)
                for head, positions in self.duplicate_row_positions.items()
            },
            "duplicates": self.duplicates,
        }


def _routing_identity(
    native_batch: NativeEpiBatch, position: int
) -> tuple[object, str, int, int, str]:
    """Count-once identity of one batch row: ``(source, origin, interval, run)``."""
    query = native_batch.queries[position]
    return (
        query.source_type,
        query.source_id,
        query.forecast_origin,
        query.horizon_days,
        query.realization_id,
    )


def _count_once_slots(native_batch: NativeEpiBatch, rows: torch.Tensor) -> torch.Tensor:
    """Routed-block slots kept under the count-once rule (review ruling R7).

    A query routed twice into one head must not change any component loss,
    ``n_eff``, or supervised count: the first occurrence in routed-block
    order wins and later duplicates are dropped before any gather.
    ``audit_head_counts`` still flags the malformed batch for logging.
    """
    cached = native_batch.cached_count_once_slots(rows)
    if cached is not None:
        return cached
    seen: set[tuple[object, str, int, int, str]] = set()
    keep: list[int] = []
    for slot, position in enumerate(rows.tolist()):
        identity = _routing_identity(native_batch, int(position))
        if identity not in seen:
            seen.add(identity)
            keep.append(slot)
    return torch.tensor(keep, dtype=torch.long, device=rows.device)


def _log1p_per100k_targets(
    raw_counts: torch.Tensor,
    population: torch.Tensor,
    weights: torch.Tensor,
    *,
    validate_population: bool = True,
) -> torch.Tensor:
    """Daily-head target transform at consumption (review ruling R2).

    ``cases``/``deaths`` targets are raw daily counts in the batch (DS-1
    emits raw); the loss transforms them once, per routed row, in the shared
    ``log1p(per-100k)`` target space — mirroring the weekly hosp transform in
    ``_weekly_hosp_loss`` (whose own semantics are unchanged):

        ``target = log1p((raw_count / pop_audited) * 1e5)``

    The transform runs AFTER masking: masked slots (weight 0) are set to
    target 0 so junk raw values there (negative or NaN) cannot yield
    NaN/inf targets.
    """
    pop = population.to(dtype=raw_counts.dtype).unsqueeze(1)
    if validate_population and (
        not bool(torch.isfinite(pop).all()) or not bool((pop > 0).all())
    ):
        raise ValueError("population entries must be positive and finite")
    safe_counts = torch.where(weights > 0, raw_counts, torch.zeros_like(raw_counts))
    return torch.log1p(safe_counts / pop * _PER_100K)


def audit_head_counts(native_batch: NativeEpiBatch) -> NativeHeadCountAudit:
    """Count per-head supervised points and flag duplicate routing identities.

    Each routed row should appear at most once per head: DS-1/BAT-1 routing
    keys a block row by the query's trajectory identity, so a duplicated query
    (same ``(source, origin, interval)`` routed twice into one head) would be
    counted twice by the loss. The loss enforces count-once itself
    (``_count_once_slots``, review ruling R7); this audit stays for section 7
    logging and still flags malformed batches.
    """
    supervised_points: dict[str, int] = {}
    routed_rows: dict[str, int] = {}
    duplicate_row_positions: dict[str, tuple[int, ...]] = {}
    duplicates = 0

    for head in _NATIVE_ROUTED_HEADS:
        rows = native_batch.head_index.get(head)
        mask = native_batch.head_target_mask.get(head)
        if rows is None or mask is None or int(rows.numel()) == 0:
            supervised_points[head] = 0
            routed_rows[head] = 0
            duplicate_row_positions[head] = ()
            continue

        supervised_points[head] = int(_binary_target_mask(mask).sum().item())
        routed_rows[head] = int(rows.numel())
        identities = [
            _routing_identity(native_batch, position) for position in rows.tolist()
        ]
        counts = Counter(identities)
        duplicate_row_positions[head] = tuple(
            position
            for position in sorted(set(rows.tolist()))
            if counts[_routing_identity(native_batch, position)] > 1
        )
        duplicates += sum(count - 1 for count in counts.values() if count > 1)

    return NativeHeadCountAudit(
        supervised_points=supervised_points,
        routed_rows=routed_rows,
        duplicate_row_positions=duplicate_row_positions,
        duplicates=duplicates,
    )


def get_native_loss_from_config(
    loss_config: LossConfig | None,
    *,
    observation_heads_config: ObservationHeadConfig,
    horizon_days: int,
) -> NativeJointInferenceLoss:
    """Build :class:`NativeJointInferenceLoss` from the training loss config.

    Mirrors the :func:`get_loss_from_config` kwargs plumbing. Municipal
    window-eligibility gates (``min_history_observed``/``min_observed``) do
    not apply in native mode: eligibility is the routing block plus the
    weekly completeness mask (plan section 11.5).
    """
    if loss_config is None:
        joint_cfg = JointLossConfig()
    else:
        name_lower = (loss_config.name or "").lower()
        if name_lower != "joint_inference":
            raise ValueError(
                "Only training.loss.name='joint_inference' is supported for "
                f"EpiForecaster, got {loss_config.name!r}."
            )
        joint_cfg = loss_config.joint
    if joint_cfg.adaptive_scheme == "gradnorm" and joint_cfg.observation_loss == "nll":
        raise ValueError(
            "adaptive_scheme='gradnorm' requires a non-negative observation "
            "loss; observation_loss='nll' is signed. Use fixed weighting or "
            "a non-negative objective such as 'crps'."
        )

    return NativeJointInferenceLoss(
        horizon_days=horizon_days,
        observation_loss=joint_cfg.observation_loss,
        distribution_family=joint_cfg.distribution_family,
        distribution_df=joint_cfg.distribution_df,
        distribution_scale_floor=observation_heads_config.distribution_scale_floor,
        interval_levels=joint_cfg.interval_levels,
        obs_weight_sum=joint_cfg.gradnorm_obs_weight_sum,
        latent_total_weight=joint_cfg.latent.total_weight,
        latent_scaling=joint_cfg.latent.scaling,
        latent_component_weights=joint_cfg.latent.component_weights,
        disable_ww=joint_cfg.disable_ww,
        disable_hosp=joint_cfg.disable_hosp,
        disable_cases=joint_cfg.disable_cases,
        disable_deaths=joint_cfg.disable_deaths,
    )


class NativeJointInferenceLoss(JointInferenceLoss):
    """Routed loss for native support (plan section 12.2 LOS-1).

    Additive subclass of :class:`JointInferenceLoss`: the municipal path is
    untouched, and scoring reuses the base class's weighted-masked
    distribution helpers (no reimplemented NLL/CRPS math).

    Head slots keep the base task order ``("ww", "hosp", "cases", "deaths")``
    (section 11.1; :attr:`~training.gradnorm.GradNormController.task_names`
    is unchanged, so adaptive weights for native mode are a TRN-1
    integration concern — this loss only exposes components compatible with
    the existing controller):

    - ``cases``/``deaths``: masked distribution loss (NLL or CRPS per
      ``observation_loss``, normal family) on the routed rows ``[B_., H]``
      in the existing daily ``log1p(per-100k)`` space. Targets arrive as RAW
      daily counts (review ruling R2: DS-1 emits raw); the loss transforms
      at consumption, per routed row, ``log1p((count / pop_audited) * 1e5)``
      after masking (masked slots get target 0). ``pop_audited`` is
      ``head_population[head]`` when present, else ``batch.source_pop``
      gathered at the routed rows. Gathered model tensors are aligned
      internally (review ruling R1, municipal ``_HEAD_HAS_NOWCAST``
      convention): ``cases`` may carry the model's t=0 nowcast column
      ``[B_., H + 1]`` and loses it; ``deaths`` emits ``[B_., H]``.
    - ``hosp``: weekly masked distribution loss on ``[B_h, W_max]`` in weekly
      ``log1p(per-100k * week)`` space. The weekly target is transformed here
      from the batch's interval-summed raw weekly counts and the audited
      graph population: ``log1p((counts / P) * 1e5)`` (section 11.5 — the model
      side sums fractions before the transform; the target side is
      interval-summed raw counts transformed once). Effective weights are
      ``head_target_mask & hosp_weekly_mask``: partial/absent weeks, including
      ``W_max`` trailing slots beyond a narrower per-dow target width, are
      weight-0. The weekly component fills the daily-``hosp`` 4-vector slot.
    - ``ww``: assay-level EDAR loss in ``log1p(concentration)`` space.
      Uncensored observations use the configured distribution score.
      Left-censored observations use the configured distribution probability below the
      recorded detection limit.

    ``compose_total_loss`` is inherited unchanged: equal active split or
    GradNorm weights against ``obs_weight_sum`` over the 4-vector
    ``obs_active_mask``.

    Count-once is enforced before scoring (review ruling R7): routed rows
    are deduplicated by routing identity
    ``(source_type, source_id, forecast_origin, horizon_days,
    realization_id)`` per head, first occurrence winning, so a duplicated
    query changes no component loss, ``n_eff``, or count.
    ``audit_head_counts`` still audits the raw batch.

    Latent SIRHD supervision is rejected (frozen v1 choice, section 11.7:
    native-mode training is real-data-only; synthetic stays on the municipal
    path).
    """

    def __init__(
        self,
        *,
        horizon_days: int,
        observation_loss: str = "mse",
        distribution_family: str = "normal",
        distribution_df: float = 5.0,
        distribution_scale_floor: float = 1.0e-3,
        interval_levels: list[float] | tuple[float, ...] | None = None,
        obs_weight_sum: float = 0.95,
        latent_total_weight: float = 0.0,
        latent_scaling: str = "none",
        latent_component_weights: dict[str, float] | None = None,
        disable_ww: bool = False,
        disable_hosp: bool = False,
        disable_cases: bool = False,
        disable_deaths: bool = False,
    ) -> None:
        if not isinstance(horizon_days, int) or isinstance(horizon_days, bool):
            raise ValueError(
                f"horizon_days must be an int, got {type(horizon_days).__name__}"
            )
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}")
        if float(latent_total_weight) > 0:
            raise ValueError(
                "Native mode trains real-data-only in v1 (plan section 11.7): "
                "latent SIRHD supervision (latent_total_weight > 0) is not "
                "supported by NativeJointInferenceLoss; synthetic latent "
                "supervision stays on the municipal JointInferenceLoss path."
            )
        super().__init__(
            observation_loss=observation_loss,
            distribution_family=distribution_family,
            distribution_df=distribution_df,
            distribution_scale_floor=distribution_scale_floor,
            interval_levels=interval_levels,
            obs_weight_sum=obs_weight_sum,
            latent_total_weight=latent_total_weight,
            latent_scaling=latent_scaling,
            latent_component_weights=latent_component_weights,
            disable_ww=disable_ww,
            disable_hosp=disable_hosp,
            disable_cases=disable_cases,
            disable_deaths=disable_deaths,
        )
        self.horizon_days = int(horizon_days)
        #: Fixed weekly output width (MOD-1 W_max normalization ruling):
        #: ``W_max = ceil(H / 7)`` slots for every row regardless of origin dow.
        self.weekly_slots = -(-self.horizon_days // 7)

    @staticmethod
    def _zero_anchor(model_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Graph-connected zero scalar, mirroring the base anchor in ``compute_components``."""
        for key in (
            "pred_cases",
            "dist_cases_loc",
            "pred_deaths",
            "dist_deaths_loc",
            "pred_hosp_weekly",
            "dist_hosp_weekly_loc",
        ):
            tensor = model_outputs.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                return (
                    torch.nan_to_num(
                        tensor.float(), nan=0.0, posinf=0.0, neginf=0.0
                    ).sum()
                    * 0.0
                )
        return torch.zeros(())

    def _align_daily_head_tensor(
        self,
        tensor: torch.Tensor,
        *,
        head_name: str,
        target_width: int,
    ) -> torch.Tensor:
        """Drop a gathered daily-head tensor's t=0 nowcast column (R1).

        Mirrors the municipal ``_HEAD_HAS_NOWCAST`` convention: gathered
        ``cases`` tensors may arrive ``[B_h, H + 1]`` and lose their t=0
        column; ``deaths`` emits ``[B_h, H]`` and passes through untouched.
        Any other width is left alone and fails the subsequent shape check
        with a clear error (``drop_nowcast`` would silently truncate).
        """
        if (
            self._HEAD_HAS_NOWCAST[head_name]
            and tensor.ndim >= 2
            and int(tensor.shape[1]) == target_width + 1
        ):
            return drop_nowcast(tensor, target_width)
        return tensor

    def _daily_head_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        native_batch: NativeEpiBatch,
        *,
        head_name: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routed daily ``cases``/``deaths`` loss; returns (loss, n_eff, active)."""
        rows = native_batch.head_index.get(head_name)
        target = native_batch.head_target.get(head_name)
        mask = native_batch.head_target_mask.get(head_name)
        if rows is None or int(rows.numel()) == 0:
            return device_zero(device), device_zero(device), device_false(device)
        if target is None or mask is None:
            raise ValueError(
                f"Head {head_name!r} has routed rows but is missing its "
                "target or target_mask block"
            )
        rows = rows.to(device=device, dtype=torch.long)
        target_width = int(target.shape[1])
        population = native_batch.head_population.get(head_name)
        # R7 count-once: drop later duplicates of an already-scored routing
        # identity before any gather, so a repeated query changes no
        # component loss, n_eff, or count (audit_head_counts still flags it).
        slots = _count_once_slots(native_batch, rows)
        if int(slots.numel()) != int(rows.numel()):
            rows = rows.index_select(0, slots)
            target = target.index_select(0, slots)
            mask = mask.index_select(0, slots)
            if population is not None:
                population = population.index_select(0, slots)
        # R1: gathered cases tensors may carry the model's t=0 nowcast
        # column ([B_h, H + 1]) and lose it; deaths passes through. The loc
        # resolution mirrors the municipal pred-key fallback convention.
        loc_tensor = model_outputs.get(f"dist_{head_name}_loc")
        if loc_tensor is None:
            loc_tensor = model_outputs[self._HEAD_PRED_KEY[head_name]]
        loc = self._align_daily_head_tensor(
            loc_tensor.to(device).index_select(0, rows),
            head_name=head_name,
            target_width=target_width,
        )
        scale = model_outputs.get(f"dist_{head_name}_scale")
        scale = (
            torch.ones_like(loc)
            if scale is None
            else self._align_daily_head_tensor(
                scale.to(device).index_select(0, rows),
                head_name=head_name,
                target_width=target_width,
            )
        )
        if tuple(loc.shape) != tuple(target.shape):
            raise ValueError(
                f"Head {head_name!r}: model distribution shape "
                f"{tuple(loc.shape)} does not match routed target shape "
                f"{tuple(target.shape)}"
            )
        target = target.to(device=device, dtype=torch.float32)
        weights = _binary_target_mask(mask.to(device)).float()
        if population is None:
            # R2 denominator fallback: the batch-level audited population
            # gathered at the (deduplicated) routed rows.
            population = native_batch.source_pop.index_select(0, rows)
        target = _log1p_per100k_targets(
            target,
            population.to(device=device, dtype=torch.float32),
            weights,
            validate_population=not native_batch.loss_metadata_valid(head_name),
        )
        loss = self._weighted_masked_distribution_loss_from_weights(
            loc=loc, scale=scale, target=target, weights=weights
        )
        return loss, weights.sum(), weights.sum() > 0

    def left_censored_nll(
        self, loc: torch.Tensor, scale: torch.Tensor, detection_limit: torch.Tensor
    ) -> torch.Tensor:
        """Score a raw detection limit with the configured forecast family."""
        safe_scale = torch.nan_to_num(
            scale.float(), nan=self.distribution_scale_floor
        ).clamp_min(self.distribution_scale_floor)
        z = (
            torch.log1p(detection_limit.float().clamp_min(0.0)) - loc.float()
        ) / safe_scale
        if self.distribution_family == "student_t":
            if self.distribution_df != 5.0:
                raise ValueError(
                    "Left-censored Student-t wastewater requires distribution_df=5.0"
                )
            score = -student_t5_cdf(z).clamp_min(1.0e-12).log()
        else:
            score = -torch.special.log_ndtr(z)
        return torch.nan_to_num(score, nan=0.0, posinf=_LOSS_VALUE_CLAMP)

    def _wastewater_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        native_batch: NativeEpiBatch,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assay-level EDAR loss with left-censoring at the reported LoD."""
        rows = native_batch.head_index.get("ww")
        raw_target = native_batch.head_target.get("ww")
        target_mask = native_batch.head_target_mask.get("ww")
        extras = native_batch.head_extra.get("ww", {})
        if rows is None or int(rows.numel()) == 0:
            return (
                device_zero(device),
                device_zero(device),
                device_false(device),
                device_zero(device),
                device_zero(device),
            )
        detection_limit = extras.get("detection_limit")
        censor_code = extras.get("censor_code")
        if (
            raw_target is None
            or target_mask is None
            or detection_limit is None
            or censor_code is None
        ):
            raise ValueError(
                "Head 'ww' requires target, target_mask, detection_limit, "
                "and censor_code blocks"
            )

        rows = rows.to(device=device, dtype=torch.long)
        slots = _count_once_slots(native_batch, rows)
        deduped = int(slots.numel()) != int(rows.numel())
        if deduped:
            raw_target = raw_target.index_select(0, slots)
            target_mask = target_mask.index_select(0, slots)
            detection_limit = detection_limit.index_select(0, slots)
            censor_code = censor_code.index_select(0, slots)

        loc = model_outputs["dist_ww_loc"].to(device=device, dtype=torch.float32)
        scale = model_outputs["dist_ww_scale"].to(device=device, dtype=torch.float32)
        if deduped:
            loc = loc.index_select(0, slots)
            scale = scale.index_select(0, slots)
        target = raw_target.to(device=device, dtype=torch.float32)
        mask = _binary_target_mask(target_mask.to(device))
        code = censor_code.to(device=device, dtype=torch.uint8)
        lod = detection_limit.to(device=device, dtype=torch.float32)
        if loc.shape != target.shape or scale.shape != target.shape:
            raise ValueError(
                "Head 'ww' model and target shapes disagree: "
                f"{tuple(loc.shape)}, {tuple(scale.shape)}, {tuple(target.shape)}"
            )
        if code.shape != target.shape or lod.shape != target.shape:
            raise ValueError("Head 'ww' censor metadata must match target shape")

        uncensored = mask & (code == 0)
        censored = mask & (code == 1)
        invalid = mask & (code > 1)
        metadata_valid = native_batch.loss_metadata_valid("ww")
        if not metadata_valid and bool(invalid.any()):
            raise ValueError("Observed wastewater slots must use censor_code 0 or 1")
        if (
            (native_batch._ww_has_censored if metadata_valid else bool(censored.any()))
            and self.distribution_family == "student_t"
            and self.distribution_df != 5.0
        ):
            raise ValueError(
                "Left-censored Student-t wastewater requires distribution_df=5.0"
            )

        safe_target = torch.log1p(
            torch.where(uncensored, target, torch.zeros_like(target)).clamp_min(0.0)
        )
        uncensored_weights = uncensored.float()
        uncensored_loss = self._weighted_masked_distribution_loss_from_weights(
            loc=loc,
            scale=scale,
            target=safe_target,
            weights=uncensored_weights,
        )
        uncensored_count = uncensored_weights.sum()

        censored_nll = self.left_censored_nll(
            loc, scale, torch.where(censored, lod, torch.zeros_like(lod))
        )
        censored_sum = torch.where(
            censored, censored_nll, torch.zeros_like(censored_nll)
        ).sum()
        censored_count = censored.float().sum()
        n_eff = uncensored_count + censored_count
        total = uncensored_loss * uncensored_count + censored_sum
        loss = torch.where(n_eff > 0, total / n_eff.clamp_min(1.0), total * 0.0)
        return loss, n_eff, n_eff > 0, uncensored_loss, uncensored_count

    def _weekly_hosp_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        native_batch: NativeEpiBatch,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Weekly ``hosp`` loss in weekly ``log1p(per-100k * week)`` space."""
        rows = native_batch.head_index.get("hosp")
        raw_counts = native_batch.head_target.get("hosp")
        target_mask = native_batch.head_target_mask.get("hosp")
        population = native_batch.head_population.get("hosp")
        if rows is None or int(rows.numel()) == 0:
            return device_zero(device), device_zero(device), device_false(device)
        if raw_counts is None or target_mask is None or population is None:
            raise ValueError(
                "Head 'hosp' has routed rows but is missing its target, "
                "target_mask, or population block"
            )
        # R7 count-once: same deduplication as the daily heads — the weekly
        # model block is row-aligned with the routed block, so all seven
        # tensors are sliced together by kept slot.
        n_routed = int(rows.numel())
        slots = _count_once_slots(native_batch, rows)
        deduped = int(slots.numel()) != n_routed
        if deduped:
            rows = rows.index_select(0, slots)
            raw_counts = raw_counts.index_select(0, slots)
            target_mask = target_mask.index_select(0, slots)
            population = population.index_select(0, slots)
        loc = model_outputs["dist_hosp_weekly_loc"].to(device)
        scale = model_outputs["dist_hosp_weekly_scale"].to(device)
        weekly_mask = model_outputs["hosp_weekly_mask"].to(device)
        if deduped:
            loc = loc.index_select(0, slots)
            scale = scale.index_select(0, slots)
            weekly_mask = weekly_mask.index_select(0, slots)
        if loc.shape[1] != self.weekly_slots:
            raise ValueError(
                f"Weekly hosp distribution width {loc.shape[1]} does not "
                f"match W_max={self.weekly_slots} "
                f"(ceil({self.horizon_days}/7))"
            )
        if loc.shape != scale.shape or tuple(weekly_mask.shape) != tuple(loc.shape):
            raise ValueError(
                f"Weekly hosp loc/scale/mask shapes disagree: "
                f"{tuple(loc.shape)} vs {tuple(scale.shape)} vs "
                f"{tuple(weekly_mask.shape)}"
            )

        raw = raw_counts.to(device=device, dtype=torch.float32)
        if raw.shape[1] > self.weekly_slots:
            raise ValueError(
                f"Weekly hosp target width {raw.shape[1]} exceeds "
                f"W_max={self.weekly_slots}; per-dow target widths are "
                "bounded by the MOD-1 W_max normalization ruling"
            )
        # Target side of section 11.5: interval-summed raw weekly counts,
        # transformed once — log1p((counts / P) * 1e5).
        pop = population.to(device=device, dtype=torch.float32).unsqueeze(1)
        if not native_batch.loss_metadata_valid("hosp") and (
            not bool(torch.isfinite(pop).all()) or not bool((pop > 0).all())
        ):
            raise ValueError("population entries must be positive and finite")
        safe_raw = torch.where(
            _binary_target_mask(target_mask.to(device)), raw, torch.zeros_like(raw)
        )
        target = torch.log1p(safe_raw / pop * _PER_100K)
        target_mask_binary = _binary_target_mask(target_mask.to(device))
        if raw.shape[1] < self.weekly_slots:
            # Per-dow narrower target class (MOD-1 W_max ruling): pad on the
            # right with weight-0 slots — those slots are incomplete weeks.
            pad = self.weekly_slots - raw.shape[1]
            target = torch.cat([target, target.new_zeros(target.shape[0], pad)], dim=1)
            target_mask_binary = torch.cat(
                [
                    target_mask_binary,
                    torch.zeros(
                        target_mask_binary.shape[0],
                        pad,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
        if tuple(target.shape) != tuple(loc.shape):
            raise ValueError(
                f"Weekly hosp target shape {tuple(target.shape)} does not "
                f"match model distribution shape {tuple(loc.shape)}"
            )
        weights = (target_mask_binary & weekly_mask.to(dtype=torch.bool)).float()
        loss = self._weighted_masked_distribution_loss_from_weights(
            loc=loc, scale=scale, target=target, weights=weights
        )
        return loss, weights.sum(), weights.sum() > 0

    def compute_components(
        self,
        model_outputs: dict[str, torch.Tensor],
        native_batch: NativeEpiBatch,
    ) -> dict[str, torch.Tensor]:
        """Routed components mirroring the base ``compute_components`` keys."""
        zero_anchor = self._zero_anchor(model_outputs)
        device = zero_anchor.device

        if self.disable_ww:
            ww_loss = zero_anchor
            ww_n_eff = device_zero(device)
            ww_active = device_false(device)
            ww_uncensored_loss = zero_anchor
            ww_uncensored_count = device_zero(device)
        else:
            ww_loss, ww_n_eff, ww_active, ww_uncensored_loss, ww_uncensored_count = (
                self._wastewater_loss(model_outputs, native_batch, device=device)
            )

        losses: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for head in ("cases", "hosp", "deaths"):
            disable_attr = self._HEAD_DISABLE_ATTR[head]
            if bool(getattr(self, disable_attr)):
                losses[head] = (
                    zero_anchor,
                    device_zero(device),
                    device_false(device),
                )
            elif head == "hosp":
                losses[head] = self._weekly_hosp_loss(
                    model_outputs, native_batch, device=device
                )
            else:
                losses[head] = self._daily_head_loss(
                    model_outputs, native_batch, head_name=head, device=device
                )

        components: dict[str, torch.Tensor] = {
            "ww": ww_loss,
            "hosp": losses["hosp"][0],
            "cases": losses["cases"][0],
            "deaths": losses["deaths"][0],
            "sird_supervision": zero_anchor,
            "sird_supervision_raw": zero_anchor,
        }
        obs_active_mask = torch.stack(
            [
                ww_active,
                losses["hosp"][2],
                losses["cases"][2],
                losses["deaths"][2],
            ]
        ).to(dtype=torch.bool)
        totals = self.compose_total_loss(
            components=components,
            obs_active_mask=obs_active_mask,
        )
        return {
            "ww": ww_loss,
            "hosp": losses["hosp"][0],
            "cases": losses["cases"][0],
            "deaths": losses["deaths"][0],
            "sird_supervision": zero_anchor,
            "sird_supervision_raw": zero_anchor,
            "ww_weighted": totals["ww_weighted"],
            "hosp_weighted": totals["hosp_weighted"],
            "cases_weighted": totals["cases_weighted"],
            "deaths_weighted": totals["deaths_weighted"],
            "sird_supervision_weighted": totals["sird_supervision_weighted"],
            "ww_n_eff": ww_n_eff,
            "ww_uncensored_objective": ww_uncensored_loss,
            "ww_uncensored_count": ww_uncensored_count,
            "hosp_n_eff": losses["hosp"][1],
            "cases_n_eff": losses["cases"][1],
            "deaths_n_eff": losses["deaths"][1],
            "obs_weights": totals["obs_weights"],
            "obs_active_mask": obs_active_mask,
            "obs_total": totals["obs_total"],
            "total": totals["total"],
        }

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        native_batch: NativeEpiBatch,
    ) -> dict[str, torch.Tensor]:
        """Routed loss components over a native batch (native trainer path)."""
        return self.compute_components(model_outputs, native_batch)
