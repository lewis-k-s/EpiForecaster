from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from evaluation.distribution_scores import (
    distribution_wis,
    standard_wis_half_widths,
    student_t_crps,
)

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
        if self.observation_loss not in {"mse", "rmse", "mae", "nll", "crps", "wis"}:
            raise ValueError(
                "observation_loss must be one of "
                "['crps', 'mae', 'mse', 'nll', 'rmse', 'wis'], "
                f"got {self.observation_loss!r}"
            )
        if self.distribution_family not in {"normal", "student_t"}:
            raise ValueError(
                "distribution_family must be one of ['normal', 'student_t'], "
                f"got {self.distribution_family!r}"
            )
        if self.distribution_scale_floor <= 0:
            raise ValueError(
                "distribution_scale_floor must be positive, "
                f"got {self.distribution_scale_floor}"
            )
        if self.distribution_family == "student_t" and self.distribution_df <= 2.0:
            raise ValueError(
                f"distribution_df must be > 2 for student_t, got {self.distribution_df}"
            )
        if not math.isfinite(self.distribution_df):
            raise ValueError("distribution_df must be finite")
        self.register_buffer(
            "wis_half_widths",
            torch.tensor(
                standard_wis_half_widths(self.distribution_family, self.distribution_df)
            ),
            persistent=False,
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
                return student_t_crps(loc, scale, target, self.distribution_df)
            return self._normal_crps(loc=loc, scale=scale, target=target)

        if self.observation_loss == "wis":
            return distribution_wis(loc, scale, target, self.wis_half_widths.to(loc))

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
        if self.observation_loss in {"nll", "crps", "wis"}:
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
            # Target values are validated once per split by
            # data.latent_supervision.validate_complete_latent_targets; a
            # value-dependent branch here would break the compiled graph.
            if self.latent_require_complete_targets and missing_targets:
                raise ValueError(
                    "Complete latent supervision requires all five targets; "
                    f"missing {missing_targets}"
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
