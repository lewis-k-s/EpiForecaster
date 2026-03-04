from __future__ import annotations

import logging
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import zarr.errors

from data.epi_dataset import EpiDataset, collate_epiforecaster_batch
from data.preprocess.config import REGION_COORD
from evaluation.metrics import TorchMaskedMetricAccumulator
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.normalization import unscale_forecasts
from utils.dtypes import sync_to_device
from utils.training_utils import drop_nowcast
from models.configs import DataConfig, EpiForecasterConfig, LossConfig
from models.epiforecaster import EpiForecaster
from plotting.forecast_plots import (
    DEFAULT_PLOT_TARGETS,
    collect_forecast_samples_for_target_nodes,
    make_forecast_figure,
    make_joint_forecast_figure,
)

logger = logging.getLogger(__name__)

# Global seeded RNG for reproducibility across evaluation/plotting
_GLOBAL_RNG = np.random.default_rng(42)
_LOSS_VALUE_CLAMP = 1.0e6


def _ensure_wandb_run(
    *,
    config: EpiForecasterConfig | None,
    log_dir: Path | None,
    name: str,
    job_type: str,
) -> wandb.sdk.wandb_run.Run | None:
    if wandb.run is not None:
        return wandb.run
    if log_dir is None:
        return None
    project = config.output.wandb_project if config is not None else "epiforecaster"
    entity = config.output.wandb_entity if config is not None else None
    group = None
    mode = "online"
    if config is not None:
        group = config.output.wandb_group or config.output.experiment_name
        mode = config.output.wandb_mode
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        dir=str(log_dir),
        config=config.to_dict() if config is not None else None,
        job_type=job_type,
        mode=mode,
    )


class ForecastLoss(nn.Module):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class WrappedTorchLoss(ForecastLoss):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        _ = (target_mean, target_scale)
        return self.loss_fn(predictions, targets)


class SMAPELoss(ForecastLoss):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        pred_unscaled, targets_unscaled = unscale_forecasts(
            predictions, targets, target_mean, target_scale
        )
        numerator = 2 * (pred_unscaled - targets_unscaled).abs()
        denominator = pred_unscaled.abs() + targets_unscaled.abs() + self.epsilon
        return (numerator / denominator).mean()


class UnscaledMSELoss(ForecastLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        pred_unscaled, targets_unscaled = unscale_forecasts(
            predictions, targets, target_mean, target_scale
        )
        diff = pred_unscaled - targets_unscaled
        return (diff**2).mean()


class CompositeLoss(ForecastLoss):
    def __init__(self, components: list[tuple[ForecastLoss, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _weight in components])
        self.loss_fns: list[ForecastLoss] = [loss for loss, _weight in components]
        self.weights = [float(weight) for _loss, weight in components]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        total = predictions.new_zeros(())
        for loss_fn, weight in zip(self.loss_fns, self.weights, strict=False):
            if weight == 0:
                continue
            total = total + weight * loss_fn.forward(
                predictions, targets, target_mean, target_scale
            )
        return total


class JointInferenceLoss(nn.Module):
    """
    Joint inference loss combining wastewater, hospitalization, and SIR physics losses.

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
    _HEAD_N_EFF_REF_ATTR = {
        "ww": "ww_n_eff_reference",
        "hosp": "hosp_n_eff_reference",
        "cases": "cases_n_eff_reference",
        "deaths": "deaths_n_eff_reference",
    }
    _HEAD_INDEX = {head_name: idx for idx, head_name in enumerate(_OBS_HEADS)}

    def __init__(
        self,
        obs_weight_sum: float = 0.95,
        w_sir: float = 0.1,
        w_continuity: float = 0.0,
        sir_residual_clip: float = 1.0e3,
        disable_ww: bool = False,
        disable_hosp: bool = False,
        disable_cases: bool = False,
        disable_deaths: bool = False,
        mask_input_ww: bool = False,
        mask_input_hosp: bool = False,
        mask_input_cases: bool = False,
        mask_input_deaths: bool = False,
        ww_min_observed: int = 0,
        hosp_min_observed: int = 0,
        cases_min_observed: int = 0,
        deaths_min_observed: int = 0,
        obs_n_eff_power: float = 0.0,
        obs_n_eff_reference: float = 0.0,
        ww_n_eff_reference: float = 0.0,
        hosp_n_eff_reference: float = 0.0,
        cases_n_eff_reference: float = 0.0,
        deaths_n_eff_reference: float = 0.0,
        forecast_horizon: int | None = None,
        horizon_norm_enabled: bool = True,
        horizon_norm_ema_decay: float = 0.9,
        horizon_norm_eps: float = 1.0e-6,
        horizon_norm_scale_floor: float = 1.0e-3,
        horizon_weight_mode: str = "exp_decay",
        horizon_weight_gamma: float = 0.85,
        horizon_weight_power: float = 1.0,
    ):
        super().__init__()
        self.obs_weight_sum = float(obs_weight_sum)
        self.w_sir = w_sir
        self.w_continuity = w_continuity
        self.sir_residual_clip = float(sir_residual_clip)
        self.disable_ww = bool(disable_ww)
        self.disable_hosp = bool(disable_hosp)
        self.disable_cases = bool(disable_cases)
        self.disable_deaths = bool(disable_deaths)
        self.mask_input_ww = bool(mask_input_ww)
        self.mask_input_hosp = bool(mask_input_hosp)
        self.mask_input_cases = bool(mask_input_cases)
        self.mask_input_deaths = bool(mask_input_deaths)
        self.ww_min_observed = int(ww_min_observed)
        self.hosp_min_observed = int(hosp_min_observed)
        self.cases_min_observed = int(cases_min_observed)
        self.deaths_min_observed = int(deaths_min_observed)
        self.obs_n_eff_power = float(obs_n_eff_power)
        self.obs_n_eff_reference = float(obs_n_eff_reference)
        self.ww_n_eff_reference = float(ww_n_eff_reference)
        self.hosp_n_eff_reference = float(hosp_n_eff_reference)
        self.cases_n_eff_reference = float(cases_n_eff_reference)
        self.deaths_n_eff_reference = float(deaths_n_eff_reference)
        self.horizon_norm_enabled = bool(horizon_norm_enabled)
        self.horizon_norm_ema_decay = float(horizon_norm_ema_decay)
        self.horizon_norm_eps = float(horizon_norm_eps)
        self.horizon_norm_scale_floor = float(horizon_norm_scale_floor)
        self.horizon_weight_mode = str(horizon_weight_mode).lower()
        self.horizon_weight_gamma = float(horizon_weight_gamma)
        self.horizon_weight_power = float(horizon_weight_power)
        if self.obs_weight_sum <= 0:
            raise ValueError(
                f"obs_weight_sum must be positive, got {self.obs_weight_sum}"
            )
        if self.obs_n_eff_power < 0:
            raise ValueError(
                f"obs_n_eff_power must be non-negative, got {self.obs_n_eff_power}"
            )
        for name, value in [
            ("obs_n_eff_reference", self.obs_n_eff_reference),
            ("ww_n_eff_reference", self.ww_n_eff_reference),
            ("hosp_n_eff_reference", self.hosp_n_eff_reference),
            ("cases_n_eff_reference", self.cases_n_eff_reference),
            ("deaths_n_eff_reference", self.deaths_n_eff_reference),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not (0 <= self.horizon_norm_ema_decay < 1):
            raise ValueError(
                "horizon_norm_ema_decay must be in [0, 1), "
                f"got {self.horizon_norm_ema_decay}"
            )
        if self.horizon_norm_eps <= 0:
            raise ValueError(
                f"horizon_norm_eps must be positive, got {self.horizon_norm_eps}"
            )
        if self.horizon_norm_scale_floor <= 0:
            raise ValueError(
                "horizon_norm_scale_floor must be positive, "
                f"got {self.horizon_norm_scale_floor}"
            )
        if self.horizon_weight_mode not in {
            "uniform",
            "exp_decay",
            "exp_growth",
            "linear_decay",
        }:
            raise ValueError(
                "horizon_weight_mode must be one of "
                "['uniform', 'exp_decay', 'exp_growth', 'linear_decay'], "
                f"got {self.horizon_weight_mode!r}"
            )
        if not (0 < self.horizon_weight_gamma <= 1):
            raise ValueError(
                "horizon_weight_gamma must be in (0, 1], "
                f"got {self.horizon_weight_gamma}"
            )
        if self.horizon_weight_power <= 0:
            raise ValueError(
                f"horizon_weight_power must be positive, got {self.horizon_weight_power}"
            )
        if forecast_horizon is not None and int(forecast_horizon) < 1:
            raise ValueError(
                f"forecast_horizon must be >= 1 when provided, got {forecast_horizon}"
            )

        initial_horizon = int(forecast_horizon) if forecast_horizon is not None else 0
        self.register_buffer(
            "horizon_rms_scales",
            torch.ones((len(self._OBS_HEADS), initial_horizon), dtype=torch.float32),
        )
        self.register_buffer(
            "horizon_scale_initialized",
            torch.zeros((len(self._OBS_HEADS), initial_horizon), dtype=torch.bool),
        )
        self.register_buffer(
            "horizon_weights",
            self._build_horizon_weights(initial_horizon),
        )

    def _build_horizon_weights(self, horizon: int) -> torch.Tensor:
        if horizon <= 0:
            return torch.zeros(0, dtype=torch.float32)
        if self.horizon_weight_mode == "uniform":
            raw = torch.ones(horizon, dtype=torch.float32)
        elif self.horizon_weight_mode == "exp_decay":
            raw = torch.pow(
                torch.as_tensor(self.horizon_weight_gamma, dtype=torch.float32),
                torch.arange(horizon, dtype=torch.float32),
            )
        elif self.horizon_weight_mode == "exp_growth":
            raw = torch.pow(
                torch.as_tensor(self.horizon_weight_gamma, dtype=torch.float32),
                torch.arange(horizon - 1, -1, -1, dtype=torch.float32),
            )
        else:
            raw = torch.pow(
                torch.arange(horizon, 0, -1, dtype=torch.float32),
                self.horizon_weight_power,
            )
        return raw / raw.sum().clamp_min(1e-8)

    def _ensure_horizon_state(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        current_horizon = int(self.horizon_weights.numel())
        if current_horizon == horizon:
            return

        target_device = self.horizon_rms_scales.device
        new_scales = torch.ones(
            (len(self._OBS_HEADS), horizon),
            dtype=torch.float32,
            device=target_device,
        )
        new_initialized = torch.zeros(
            (len(self._OBS_HEADS), horizon),
            dtype=torch.bool,
            device=target_device,
        )
        overlap = min(current_horizon, horizon)
        if overlap > 0:
            new_scales[:, :overlap] = self.horizon_rms_scales[:, :overlap]
            new_initialized[:, :overlap] = self.horizon_scale_initialized[:, :overlap]
        self.horizon_rms_scales = new_scales
        self.horizon_scale_initialized = new_initialized
        self.horizon_weights = self._build_horizon_weights(horizon).to(target_device)

    def _ensure_horizon_state_device(self, device: torch.device) -> None:
        if self.horizon_rms_scales.device == device:
            return
        self.horizon_rms_scales = self.horizon_rms_scales.to(device)
        self.horizon_scale_initialized = self.horizon_scale_initialized.to(device)
        self.horizon_weights = self.horizon_weights.to(device)

    @staticmethod
    def _align_prediction_to_target_horizon(
        prediction: torch.Tensor,
        target_horizon: int,
    ) -> torch.Tensor:
        if prediction.ndim < 2:
            return prediction
        if target_horizon < 1:
            return prediction
        current_horizon = int(prediction.shape[1])
        if current_horizon <= target_horizon:
            return prediction
        return prediction[:, :target_horizon]

    def _resolve_batch_horizon(
        self,
        *,
        obs_supervision: dict[str, dict[str, torch.Tensor | None]],
        model_outputs: dict[str, torch.Tensor],
    ) -> int | None:
        for head_name in self._OBS_HEADS:
            target = obs_supervision[head_name]["target"]
            if target is not None:
                return int(target.shape[1])
        for key in ("pred_deaths", "physics_residual"):
            value = model_outputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return int(value.shape[1])
        return None

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
        sir_weighted = self.w_sir * components["sir"]
        continuity_weighted = self.w_continuity * components["continuity"]
        total = obs_total + sir_weighted + continuity_weighted
        return {
            "obs_weights": obs_weights,
            "obs_total": obs_total,
            "total": total,
            "ww_weighted": obs_weighted[0],
            "hosp_weighted": obs_weighted[1],
            "cases_weighted": obs_weighted[2],
            "deaths_weighted": obs_weighted[3],
            "sir_weighted": sir_weighted,
            "continuity_weighted": continuity_weighted,
        }

    @staticmethod
    def _build_supervision_weights(
        *,
        target: torch.Tensor,
        observed_mask: torch.Tensor | None,
        min_observed: int,
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

        if min_observed > 0:
            observed_counts = observed_binary.sum(dim=1, keepdim=True)
            eligible = (observed_counts >= float(min_observed)).to(
                observed_binary.dtype
            )
            observed_binary = observed_binary * eligible

        n_eff = observed_binary.sum()
        return {
            "target": target_f32,
            "weights": observed_binary,  # Weights are strictly the binary mask
            "observed_binary": observed_binary,
            "n_eff": n_eff,
            "active": n_eff > 0,
        }

    @staticmethod
    def _weighted_masked_mse_from_weights(
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted masked MSE from precomputed effective weights."""
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

        sq = (prediction_safe - target_safe) ** 2

        # PER-SERIES NORMALIZATION
        # 1. Sum squared errors per series
        series_sq_sum = (sq * weights_f32).sum(dim=1)

        # 2. Divide by number of active points in that series
        series_weight_sum = weights_f32.sum(dim=1)
        series_mse = series_sq_sum / series_weight_sum.clamp_min(1.0)

        # 3. The final loss is the average across all active series in the batch
        batch_active_series = (series_weight_sum > 0).to(weights_f32.dtype)
        total_active_series = batch_active_series.sum()

        return (series_mse * batch_active_series).sum() / total_active_series.clamp_min(
            1e-8
        )

    @staticmethod
    def _compute_per_h_mse(
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        sq = (prediction_safe - target_safe) ** 2
        sq_sum_h = (sq * weights_f32).sum(dim=0)
        count_h = weights_f32.sum(dim=0)
        mse_h = sq_sum_h / count_h.clamp_min(1.0)
        return mse_h, count_h

    def _maybe_update_horizon_scales(
        self,
        *,
        head_idx: int,
        mse_h: torch.Tensor,
        active_h: torch.Tensor,
    ) -> None:
        if not self.horizon_norm_enabled:
            return
        with torch.no_grad():
            rms_h = torch.sqrt(mse_h.detach().clamp_min(0.0))
            active_h_bool = active_h
            scale_row = self.horizon_rms_scales[head_idx]
            initialized_row = self.horizon_scale_initialized[head_idx]

            newly_active = active_h_bool & (~initialized_row)
            scale_row_next = torch.where(newly_active, rms_h, scale_row)
            initialized_next = initialized_row | newly_active

            steady = active_h_bool & initialized_next
            d = self.horizon_norm_ema_decay
            ema_value = d * scale_row_next + (1.0 - d) * rms_h
            scale_row_next = torch.where(steady, ema_value, scale_row_next)

            clamped = scale_row_next.clamp_min(self.horizon_norm_scale_floor)
            scale_row_next = torch.where(
                active_h_bool,
                clamped,
                scale_row_next,
            )
            scale_row.copy_(scale_row_next)
            initialized_row.copy_(initialized_next)

    def _use_legacy_observation_objective(self) -> bool:
        """Preserve historical loss behavior when horizon weighting is effectively off."""
        return (not self.horizon_norm_enabled) and self.horizon_weight_mode == "uniform"

    def _head_horizon_normalized_mse_loss(
        self,
        *,
        head_name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        head_idx = self._HEAD_INDEX[head_name]
        mse_h, count_h = self._compute_per_h_mse(
            prediction=prediction,
            target=target,
            weights=weights,
        )
        active_h = count_h > 0
        scale_h = self.horizon_rms_scales[head_idx]
        if self._use_legacy_observation_objective():
            legacy_loss = self._weighted_masked_mse_from_weights(
                prediction=prediction,
                target=target,
                weights=weights,
            )
            horizon_weights = self.horizon_weights
            weighted_active = horizon_weights * active_h.to(dtype=mse_h.dtype)
            denom = weighted_active.sum()
            raw_loss = torch.where(
                denom > 0,
                (weighted_active * mse_h).sum() / denom.clamp_min(self.horizon_norm_eps),
                mse_h.new_zeros(()),
            )
            contrib = torch.where(
                denom > 0,
                (weighted_active * mse_h) / denom.clamp_min(self.horizon_norm_eps),
                torch.zeros_like(mse_h),
            )
            return legacy_loss, {
                "raw": mse_h.detach(),
                "norm": mse_h.detach(),
                "contrib": contrib.detach(),
                "scale": scale_h.detach(),
                "raw_loss": raw_loss.detach(),
            }

        if self.horizon_norm_enabled:
            mse_norm_h = mse_h / (scale_h.detach().pow(2) + self.horizon_norm_eps)
        else:
            mse_norm_h = mse_h

        horizon_weights = self.horizon_weights
        weighted_active = horizon_weights * active_h.to(dtype=mse_h.dtype)
        denom = weighted_active.sum()
        raw_loss = torch.where(
            denom > 0,
            (weighted_active * mse_h).sum() / denom.clamp_min(self.horizon_norm_eps),
            mse_h.new_zeros(()),
        )
        numerator = (weighted_active * mse_norm_h).sum()
        loss = torch.where(
            denom > 0,
            numerator / denom.clamp_min(self.horizon_norm_eps),
            mse_h.new_zeros(()),
        )
        contrib = torch.where(
            denom > 0,
            (weighted_active * mse_norm_h) / denom.clamp_min(self.horizon_norm_eps),
            torch.zeros_like(mse_h),
        )

        if self.training and self.horizon_norm_enabled:
            self._maybe_update_horizon_scales(
                head_idx=head_idx,
                mse_h=mse_h,
                active_h=active_h,
            )

        return loss, {
            "raw": mse_h.detach(),
            "norm": mse_norm_h.detach(),
            "contrib": contrib.detach(),
            "scale": scale_h.detach(),
            "raw_loss": raw_loss.detach(),
        }

    def _head_n_eff_scale(self, *, head_name: str, n_eff: torch.Tensor) -> torch.Tensor:
        if self.obs_n_eff_power <= 0:
            return torch.as_tensor(1.0, device=n_eff.device, dtype=torch.float32)
        head_ref_attr = self._HEAD_N_EFF_REF_ATTR[head_name]
        reference_value = float(getattr(self, head_ref_attr))
        if reference_value <= 0:
            reference_value = float(self.obs_n_eff_reference)
        if reference_value <= 0:
            reference_value = 1.0
        ref = torch.as_tensor(reference_value, device=n_eff.device, dtype=torch.float32)
        ratio = (n_eff.float() / ref).clamp(min=0.0, max=1.0)
        return torch.pow(ratio, self.obs_n_eff_power)

    def compute_observation_supervision(
        self,
        targets: dict[str, torch.Tensor | None],
        *,
        device: torch.device,
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
            out[head_name] = self._build_supervision_weights(
                target=target,
                observed_mask=targets.get(mask_key),
                min_observed=int(getattr(self, min_observed_attr)),
                device=device,
            )

        return out

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        components = self.compute_components(model_outputs, targets, batch_data)
        return components["total"]

    def compute_components(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: dict[str, torch.Tensor] | None = None,
        *,
        emit_horizon_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute joint inference loss components.

        Args:
            model_outputs: Dict from EpiForecaster.forward() containing:
                - pred_ww: [B, H+1] predicted wastewater (includes t=0 nowcast)
                - pred_hosp: [B, H+1] predicted hospitalizations (includes t=0 nowcast)
                - pred_cases: [B, H+1] predicted reported cases (includes t=0 nowcast)
                - pred_deaths: [B, H] predicted deaths (no nowcast needed)
                - physics_residual: [B, H] SIR dynamics residual
            targets: Dict containing target tensors:
                - ww: [B, H] wastewater targets (optional)
                - hosp: [B, H] hospitalization targets (optional)
                - cases: [B, H] reported cases targets (optional)
                - deaths: [B, H] deaths targets (optional)
            batch_data: Optional dict containing historical data for continuity penalty:
                - HospHist: [B, T, 3] hospitalization history
                - CasesHist: [B, T, 3] cases history
                - DeathsHist: [B, T, 3] deaths history

        Returns:
            Dict with unweighted and weighted component losses plus total:
                - ww, hosp, cases, deaths, sir, continuity
                - ww_weighted, hosp_weighted, cases_weighted, deaths_weighted, sir_weighted, continuity_weighted
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
        ww_raw_loss = zero_anchor
        hosp_loss = zero_anchor
        hosp_raw_loss = zero_anchor
        cases_loss = zero_anchor
        cases_raw_loss = zero_anchor
        deaths_loss = zero_anchor
        deaths_raw_loss = zero_anchor
        sir_loss = zero_anchor
        continuity_loss = zero_anchor
        obs_supervision = self.compute_observation_supervision(
            targets,
            device=zero_anchor.device,
        )
        batch_horizon = self._resolve_batch_horizon(
            obs_supervision=obs_supervision,
            model_outputs=model_outputs,
        )
        if batch_horizon is not None:
            self._ensure_horizon_state(batch_horizon)
        self._ensure_horizon_state_device(zero_anchor.device)
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
        horizon_diagnostics: dict[str, dict[str, torch.Tensor]] = {}

        ww_target = obs_supervision["ww"]["target"]
        ww_weights = obs_supervision["ww"]["weights"]
        if ww_target is not None and ww_weights is not None:
            ww_base, ww_diag = self._head_horizon_normalized_mse_loss(
                head_name="ww",
                prediction=drop_nowcast(model_outputs["pred_ww"], ww_target.shape[1]),
                target=ww_target,
                weights=ww_weights,
            )
            horizon_diagnostics["ww"] = ww_diag
            ww_scale = self._head_n_eff_scale(head_name="ww", n_eff=ww_n_eff)
            ww_loss = ww_base * ww_scale
            ww_raw_loss = cast(torch.Tensor, ww_diag["raw_loss"]) * ww_scale

        hosp_target = obs_supervision["hosp"]["target"]
        hosp_weights = obs_supervision["hosp"]["weights"]
        if hosp_target is not None and hosp_weights is not None:
            hosp_base, hosp_diag = self._head_horizon_normalized_mse_loss(
                head_name="hosp",
                prediction=drop_nowcast(
                    model_outputs["pred_hosp"], hosp_target.shape[1]
                ),
                target=hosp_target,
                weights=hosp_weights,
            )
            horizon_diagnostics["hosp"] = hosp_diag
            hosp_scale = self._head_n_eff_scale(
                head_name="hosp",
                n_eff=hosp_n_eff,
            )
            hosp_loss = hosp_base * hosp_scale
            hosp_raw_loss = cast(torch.Tensor, hosp_diag["raw_loss"]) * hosp_scale

        cases_target = obs_supervision["cases"]["target"]
        cases_weights = obs_supervision["cases"]["weights"]
        if cases_target is not None and cases_weights is not None:
            cases_base, cases_diag = self._head_horizon_normalized_mse_loss(
                head_name="cases",
                prediction=drop_nowcast(
                    model_outputs["pred_cases"], cases_target.shape[1]
                ),
                target=cases_target,
                weights=cases_weights,
            )
            horizon_diagnostics["cases"] = cases_diag
            cases_scale = self._head_n_eff_scale(
                head_name="cases",
                n_eff=cases_n_eff,
            )
            cases_loss = cases_base * cases_scale
            cases_raw_loss = cast(torch.Tensor, cases_diag["raw_loss"]) * cases_scale

        deaths_target = obs_supervision["deaths"]["target"]
        deaths_weights = obs_supervision["deaths"]["weights"]
        if deaths_target is not None and deaths_weights is not None:
            deaths_prediction = self._align_prediction_to_target_horizon(
                model_outputs["pred_deaths"],
                deaths_target.shape[1],
            )
            deaths_base, deaths_diag = self._head_horizon_normalized_mse_loss(
                head_name="deaths",
                prediction=deaths_prediction,
                target=deaths_target,
                weights=deaths_weights,
            )
            horizon_diagnostics["deaths"] = deaths_diag
            deaths_scale = self._head_n_eff_scale(
                head_name="deaths",
                n_eff=deaths_n_eff,
            )
            deaths_loss = deaths_base * deaths_scale
            deaths_raw_loss = cast(torch.Tensor, deaths_diag["raw_loss"]) * deaths_scale

        # SIR physics loss (always computed from residual)
        if self.w_sir > 0:
            physics_residual = model_outputs["physics_residual"]
            if self.sir_residual_clip > 0:
                physics_residual = torch.clamp(
                    physics_residual,
                    min=-self.sir_residual_clip,
                    max=self.sir_residual_clip,
                )
            ww_mask = targets.get("ww_mask")
            hosp_mask = targets.get("hosp_mask")
            cases_mask = targets.get("cases_mask")
            deaths_mask = targets.get("deaths_mask")

            combined_mask: torch.Tensor | None = None
            masks = [
                m
                for m in [ww_mask, hosp_mask, cases_mask, deaths_mask]
                if m is not None
            ]
            if masks:
                combined_mask = torch.nan_to_num(
                    masks[0].float().to(physics_residual.device),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=1.0)
                for mask in masks[1:]:
                    cleaned = torch.nan_to_num(
                        mask.float().to(physics_residual.device),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(min=0.0, max=1.0)
                    combined_mask = torch.maximum(combined_mask, cleaned)

            if combined_mask is None:
                sir_loss = physics_residual.mean()
            else:
                sir_loss = self._weighted_masked_mse_from_weights(
                    prediction=physics_residual,
                    target=torch.zeros_like(physics_residual),
                    weights=combined_mask.float(),
                )

        # Nowcast continuity penalty
        if self.w_continuity > 0 and batch_data is not None:
            continuity_loss = self._compute_continuity_loss(model_outputs, batch_data)

        components = {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            "sir": sir_loss,
            "continuity": continuity_loss,
        }
        totals = self.compose_total_loss(
            components=components,
            obs_active_mask=obs_active_mask,
        )

        horizon_terms: dict[str, torch.Tensor] = {}
        if emit_horizon_diagnostics:
            horizon_count = int(self.horizon_weights.numel())
            for head_name in self._OBS_HEADS:
                head_diag = horizon_diagnostics.get(head_name)
                if head_diag is None:
                    zeros = zero_anchor.new_zeros((horizon_count,))
                    scale_h = self.horizon_rms_scales[self._HEAD_INDEX[head_name]]
                    head_diag = {
                        "raw": zeros,
                        "norm": zeros,
                        "contrib": zeros,
                        "scale": scale_h.detach(),
                    }
                for h_idx in range(horizon_count):
                    horizon_terms[f"loss_{head_name}_h{h_idx + 1}_raw"] = head_diag[
                        "raw"
                    ][h_idx]
                    horizon_terms[f"loss_{head_name}_h{h_idx + 1}_norm"] = head_diag[
                        "norm"
                    ][h_idx]
                    horizon_terms[f"loss_{head_name}_h{h_idx + 1}_contrib"] = head_diag[
                        "contrib"
                    ][h_idx]
                    horizon_terms[f"horizon_scale_{head_name}_h{h_idx + 1}"] = head_diag[
                        "scale"
                    ][h_idx]

        return {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            "sir": sir_loss,
            "continuity": continuity_loss,
            "ww_weighted": totals["ww_weighted"],
            "hosp_weighted": totals["hosp_weighted"],
            "cases_weighted": totals["cases_weighted"],
            "deaths_weighted": totals["deaths_weighted"],
            "ww_raw": ww_raw_loss,
            "hosp_raw": hosp_raw_loss,
            "cases_raw": cases_raw_loss,
            "deaths_raw": deaths_raw_loss,
            "sir_weighted": totals["sir_weighted"],
            "continuity_weighted": totals["continuity_weighted"],
            "ww_n_eff": ww_n_eff,
            "hosp_n_eff": hosp_n_eff,
            "cases_n_eff": cases_n_eff,
            "deaths_n_eff": deaths_n_eff,
            "obs_weights": totals["obs_weights"],
            "obs_active_mask": obs_active_mask,
            "obs_total": totals["obs_total"],
            "total": totals["total"],
            **horizon_terms,
        }

    def compute_components_train(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        batch_data: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compile-safe train path (kept separate to avoid eval-path regressions)."""
        return self.compute_components(
            model_outputs,
            targets,
            batch_data,
            emit_horizon_diagnostics=False,
        )

    def _compute_continuity_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute nowcast continuity penalty.

        Penalizes the discontinuity between the last observed value and the
        model's first forecast prediction (t=0, the nowcast).

        Args:
            model_outputs: Dict containing predictions with t=0 (nowcast)
            batch_data: Dict containing historical observations

        Returns:
            Scalar continuity loss
        """
        zero_anchor = (
            torch.nan_to_num(
                model_outputs["pred_hosp"].float(), nan=0.0, posinf=0.0, neginf=0.0
            ).sum()
            * 0.0
        )
        component_losses: list[torch.Tensor] = []

        def _masked_mse(
            nowcast_pred: torch.Tensor, last_observed: torch.Tensor
        ) -> torch.Tensor:
            valid_mask = torch.isfinite(last_observed)
            valid_f = valid_mask.to(
                device=nowcast_pred.device, dtype=nowcast_pred.dtype
            )
            last_observed_safe = torch.nan_to_num(
                last_observed.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).to(device=nowcast_pred.device, dtype=nowcast_pred.dtype)
            sq = (nowcast_pred - last_observed_safe) ** 2
            numerator = (sq * valid_f).sum()
            denominator = valid_f.sum().clamp_min(1.0)
            return numerator / denominator

        if "HospHist" in batch_data and model_outputs["pred_hosp"].shape[1] > 0:
            component_losses.append(
                _masked_mse(
                    model_outputs["pred_hosp"][:, 0], batch_data["HospHist"][:, -1, 0]
                )
            )

        if "CasesHist" in batch_data and model_outputs["pred_cases"].shape[1] > 0:
            component_losses.append(
                _masked_mse(
                    model_outputs["pred_cases"][:, 0], batch_data["CasesHist"][:, -1, 0]
                )
            )

        if "DeathsHist" in batch_data and model_outputs["pred_deaths"].shape[1] > 0:
            component_losses.append(
                _masked_mse(
                    model_outputs["pred_deaths"][:, 0],
                    batch_data["DeathsHist"][:, -1, 0],
                )
            )

        if component_losses:
            return torch.stack(component_losses).mean()
        return zero_anchor


def get_loss_function(name: str) -> ForecastLoss:
    name_lower = name.lower()
    if name_lower == "mse":
        return WrappedTorchLoss(nn.MSELoss())
    elif name_lower == "mse_unscaled":
        return UnscaledMSELoss()
    elif name_lower in ("mae", "l1"):
        return WrappedTorchLoss(nn.L1Loss())
    elif name_lower == "smape":
        return SMAPELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_loss_from_config(
    loss_config: LossConfig | None,
    *,
    data_config: DataConfig | None = None,
    forecast_horizon: int | None = None,
) -> ForecastLoss | JointInferenceLoss:
    if loss_config is None:
        return get_loss_function("smape")
    name_lower = loss_config.name.lower()
    if name_lower == "joint_inference":
        # Joint inference loss for SIR + observation heads
        joint_cfg = loss_config.joint
        min_obs = {"cases": 0, "hospitalizations": 0, "deaths": 0, "wastewater": 0}
        if data_config is not None and forecast_horizon is not None:
            min_obs = data_config.resolve_min_observed_map(
                forecast_horizon=int(forecast_horizon)
            )
        return JointInferenceLoss(
            obs_weight_sum=joint_cfg.gradnorm_obs_weight_sum,
            w_sir=joint_cfg.w_sir,
            w_continuity=joint_cfg.w_continuity,
            sir_residual_clip=joint_cfg.sir_residual_clip,
            disable_ww=joint_cfg.disable_ww,
            disable_hosp=joint_cfg.disable_hosp,
            disable_cases=joint_cfg.disable_cases,
            disable_deaths=joint_cfg.disable_deaths,
            mask_input_ww=joint_cfg.mask_input_ww,
            mask_input_hosp=joint_cfg.mask_input_hosp,
            mask_input_cases=joint_cfg.mask_input_cases,
            mask_input_deaths=joint_cfg.mask_input_deaths,
            ww_min_observed=min_obs["wastewater"],
            hosp_min_observed=min_obs["hospitalizations"],
            cases_min_observed=min_obs["cases"],
            deaths_min_observed=min_obs["deaths"],
            obs_n_eff_power=joint_cfg.obs_n_eff_power,
            obs_n_eff_reference=joint_cfg.obs_n_eff_reference,
            ww_n_eff_reference=joint_cfg.ww_n_eff_reference,
            hosp_n_eff_reference=joint_cfg.hosp_n_eff_reference,
            cases_n_eff_reference=joint_cfg.cases_n_eff_reference,
            deaths_n_eff_reference=joint_cfg.deaths_n_eff_reference,
            forecast_horizon=int(forecast_horizon) if forecast_horizon is not None else None,
            horizon_norm_enabled=joint_cfg.horizon_norm_enabled,
            horizon_norm_ema_decay=joint_cfg.horizon_norm_ema_decay,
            horizon_norm_eps=joint_cfg.horizon_norm_eps,
            horizon_norm_scale_floor=joint_cfg.horizon_norm_scale_floor,
            horizon_weight_mode=joint_cfg.horizon_weight_mode,
            horizon_weight_gamma=joint_cfg.horizon_weight_gamma,
            horizon_weight_power=joint_cfg.horizon_weight_power,
        )
    if name_lower == "composite":
        if not loss_config.components:
            raise ValueError("Composite loss requires components")
        components: list[tuple[ForecastLoss, float]] = []
        for component in loss_config.components:
            components.append((get_loss_function(component.name), component.weight))
        return CompositeLoss(components)
    return get_loss_function(loss_config.name)


def resolve_device(device: str) -> torch.device:
    """Resolve the torch device string using the same priority as training."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> tuple[EpiForecaster, EpiForecasterConfig, dict[str, Any]]:
    """Load an EpiForecaster model + config from a saved trainer checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on
        overrides: Optional list of dotted-key config overrides applied before model creation

    Returns:
        Tuple of (model, config, checkpoint_dict)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is missing required keys or has invalid config
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate required keys
    required_keys = ["model_state_dict", "config"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            f"This checkpoint may be from an incompatible version or corrupted."
        )

    config_raw = checkpoint["config"]

    # Handle backwards compatibility: old checkpoints have pickled EpiForecasterConfig,
    # new checkpoints have plain dicts (robust to config class changes)
    if isinstance(config_raw, dict):
        # New format: plain dict (YAML-compatible)
        config = EpiForecasterConfig.from_dict(config_raw)
    elif isinstance(config_raw, EpiForecasterConfig):
        # Old format: pickled EpiForecasterConfig (for backwards compatibility)
        config = config_raw
    else:
        raise ValueError(
            f"Checkpoint config has invalid type: {type(config_raw).__name__}. "
            f"Expected EpiForecasterConfig or dict. "
            f"Please check that the checkpoint was created with a compatible version."
        )

    # Apply overrides BEFORE model creation (for architecture-affecting params)
    if overrides:
        config = EpiForecasterConfig.apply_overrides(config, overrides)
        logger.info(f"Applied {len(overrides)} config overrides before model creation")

    model = EpiForecaster(
        variant_type=config.model.type,
        temporal_input_dim=config.model.cases_dim,
        biomarkers_dim=config.model.biomarkers_dim,
        region_embedding_dim=config.model.region_embedding_dim,
        mobility_embedding_dim=config.model.mobility_embedding_dim,
        gnn_depth=config.model.gnn_depth,
        sequence_length=config.model.input_window_length,
        forecast_horizon=config.model.forecast_horizon,
        use_population=config.model.use_population,
        population_dim=config.model.population_dim,
        device=resolve_device(device),
        gnn_module=config.model.gnn_module,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        head_d_model=config.model.head_d_model,
        head_n_heads=config.model.head_n_heads,
        head_num_layers=config.model.head_num_layers,
        head_dropout=config.model.head_dropout,
        sir_physics=config.model.sir_physics,
        observation_heads=config.model.observation_heads,
        temporal_covariates_dim=config.model.temporal_covariates_dim,
    )
    # Strip _orig_mod. prefix from compiled model checkpoints
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(resolve_device(device))
    return model, config, checkpoint


def _maybe_load_criterion_state_from_checkpoint(
    *,
    criterion: JointInferenceLoss,
    checkpoint: dict[str, Any],
) -> None:
    state_dict = checkpoint.get("criterion_state_dict")
    if state_dict is None:
        logger.info(
            "[eval] Checkpoint missing criterion_state_dict; "
            "using freshly initialized loss state."
        )
        return
    try:
        missing, unexpected = criterion.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "[eval] Loaded criterion_state_dict with mismatch "
                f"(missing={missing}, unexpected={unexpected})."
            )
    except RuntimeError as exc:
        logger.warning(
            "[eval] Failed to load criterion_state_dict (%s). "
            "Using freshly initialized loss state.",
            exc,
        )


def split_nodes(config: EpiForecasterConfig) -> tuple[list[int], list[int], list[int]]:
    """Match the node holdout split logic used during training."""
    train_split = 1 - config.training.val_split - config.training.test_split
    if not config.data.run_id:
        raise ValueError("run_id must be specified in config")
    aligned_dataset = EpiDataset.load_canonical_dataset(
        Path(config.data.dataset_path),
        run_id=config.data.run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    N = aligned_dataset[REGION_COORD].size
    all_nodes = np.arange(N)
    if config.data.use_valid_targets:
        valid_mask = EpiDataset.get_valid_nodes(
            dataset_path=Path(config.data.dataset_path),
            run_id=config.data.run_id,
        )
        all_nodes = all_nodes[valid_mask]
    rng = np.random.default_rng(42)
    rng.shuffle(all_nodes)
    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * config.training.val_split)
    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]
    return list(train_nodes), list(val_nodes), list(test_nodes)


def _suppress_zarr_warnings(worker_id: int) -> None:
    """Suppress zarr/numcodecs warnings in DataLoader worker processes."""
    import warnings

    warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


def build_loader_from_config(
    config: EpiForecasterConfig,
    *,
    split: str,
    batch_size: int | None = None,
    device: str = "auto",
) -> tuple[DataLoader[EpiDataset], torch.Tensor | None]:
    """Build a DataLoader for the given split from the checkpoint config.

    Returns:
        Tuple of (DataLoader, region_embeddings). Region embeddings are pre-loaded
        to the target device to avoid repeated transfers during evaluation.
    """
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    if config.training.split_strategy == "time":
        train_end: str = config.training.train_end_date or ""
        val_end: str = config.training.val_end_date or ""
        test_end: str | None = config.training.test_end_date
        _train_dataset, val_dataset, test_dataset = EpiDataset.create_temporal_splits(
            config=config,
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end,
        )
        dataset = val_dataset if split_key == "val" else test_dataset
    else:
        train_nodes, val_nodes, test_nodes = split_nodes(config)
        if split_key == "val":
            dataset = EpiDataset(
                config=config,
                target_nodes=val_nodes,
                context_nodes=train_nodes + val_nodes,
            )
        else:
            dataset = EpiDataset(
                config=config,
                target_nodes=test_nodes,
                context_nodes=train_nodes + val_nodes,
            )

    # Worker configuration mirrors training defaults:
    # - validation loader uses val_workers
    # - test loader uses test_workers
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = (
        config.training.val_workers
        if split_key == "val"
        else getattr(config.training, "test_workers", 0)
    )
    if cfg_workers == -1:
        num_workers = max(0, avail_cores)
    else:
        num_workers = min(max(0, avail_cores), cfg_workers)

    resolved_batch = batch_size or config.training.batch_size
    resolved_device = resolve_device(device)
    pin_memory = bool(config.training.pin_memory) and resolved_device.type == "cuda"

    # Pre-load region embeddings to device to avoid repeated transfers
    region_embeddings = getattr(dataset, "region_embeddings", None)
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(resolved_device)

    loader = DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(
            collate_epiforecaster_batch,
            require_region_index=bool(config.model.type.regions),
        ),
        worker_init_fn=_suppress_zarr_warnings if num_workers > 0 else None,
    )
    return loader, region_embeddings


def select_nodes_by_loss(
    *,
    node_mae: dict[int, float],
    strategy: str = "quartile",
    k: int = 5,
    samples_per_group: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """
    Select nodes by different loss-based strategies using in-memory node_mae.

    Args:
        node_mae: Dict mapping node_id → average MAE
        strategy: "topk", "quartile", "worst", "best", "random"
        k: Number of nodes for topk/worst/best strategies
        samples_per_group: Number of nodes per group for quartile strategy (default 4)
        rng: Random generator for deterministic sampling (default: global seeded RNG)

    Returns:
        Dict mapping group name → list of node_ids
        Examples:
            strategy="topk": {"Top-k": [1, 2, 3, 4, 5]}
            strategy="quartile": {"Q1 (Worst)": [...], "Q2 (Poor)": [...], ...}
            strategy="worst": {"Worst": [1, 2, 3, 4, 5]}
    """
    if rng is None:
        rng = _GLOBAL_RNG

    if not node_mae:
        logger.warning("[eval] No node MAE values available for selection")
        return {
            "Q1 (Worst)": [],
            "Q2 (Poor)": [],
            "Q3 (Average)": [],
            "Q4 (Best)": [],
        }

    if strategy == "random":
        all_nodes = list(node_mae.keys())
        k = min(k, len(all_nodes))
        selected = rng.choice(all_nodes, size=k, replace=False).tolist()
        return {"Random": selected}

    # Sort by MAE for other strategies
    sorted_nodes = sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))

    if strategy == "topk":
        top_k = [node_id for node_id, _mae in sorted_nodes[:k]]
        return {"Top-k": top_k}

    elif strategy == "best":
        top_k = [node_id for node_id, _mae in sorted_nodes[:k]]
        return {"Best": top_k}

    elif strategy == "worst":
        bottom_k = [node_id for node_id, _mae in sorted_nodes[-k:]]
        return {"Worst": bottom_k}

    elif strategy == "quartile":
        maes = [mae for _node_id, mae in sorted_nodes]
        q1_cutoff = np.percentile(maes, 25)
        q2_cutoff = np.percentile(maes, 50)
        q3_cutoff = np.percentile(maes, 75)

        quartile_groups: dict[str, list[int]] = {
            "Q1 (Worst)": [],
            "Q2 (Poor)": [],
            "Q3 (Average)": [],
            "Q4 (Best)": [],
        }

        for node_id, mae in sorted_nodes:
            if mae <= q1_cutoff:
                quartile_groups["Q1 (Worst)"].append(node_id)
            elif mae <= q2_cutoff:
                quartile_groups["Q2 (Poor)"].append(node_id)
            elif mae <= q3_cutoff:
                quartile_groups["Q3 (Average)"].append(node_id)
            else:
                quartile_groups["Q4 (Best)"].append(node_id)

        # Sample from each quartile
        for quartile_name, nodes in quartile_groups.items():
            k = min(samples_per_group, len(nodes))
            quartile_groups[quartile_name] = (
                rng.choice(nodes, k, replace=False).tolist() if nodes else []
            )

        return quartile_groups

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def topk_target_nodes_by_mae(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    region_embeddings: torch.Tensor | None = None,
    k: int = 5,
) -> list[int]:
    """Compute top-k target node ids by average per-window MAE over the loader."""
    device = next(model.parameters()).device
    forward_model = cast(EpiForecaster, model)

    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            eval_iter = loader
            for batch in eval_iter:
                from utils.training_utils import inject_gpu_mobility

                inject_gpu_mobility(batch, eval_iter.dataset, device)

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch,
                    region_embeddings=region_embeddings,
                )
                predictions = model_outputs.get("pred_hosp")
                targets = targets_dict.get("hosp")
                mask = targets_dict.get("hosp_mask")
                if predictions is None or targets is None:
                    raise ValueError(
                        "topk_target_nodes_by_mae requires hospitalization targets "
                        "('HospTarget') to be present in the batch."
                    )
                if mask is None:
                    mask = torch.ones_like(targets)
                abs_diff = (predictions - targets).abs()
                valid_per_sample = mask.sum(dim=1) > 0
                per_sample_mae = (abs_diff * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1.0)
                target_nodes = batch["TargetNode"]
                for sample_mae, target_node, is_valid in zip(
                    per_sample_mae, target_nodes, valid_per_sample, strict=False
                ):
                    if not bool(is_valid):
                        continue
                    node_id = int(target_node)
                    if node_id not in node_mae_sum:
                        node_mae_sum[node_id] = torch.tensor(0.0, device=device)
                    node_mae_sum[node_id] += sample_mae.detach()
                    node_mae_count[node_id] = node_mae_count.get(node_id, 0) + 1
    finally:
        if model_was_training:
            model.train()

    if not node_mae_sum:
        return []

    node_mae = {
        node_id: (node_mae_sum[node_id] / max(1, node_mae_count[node_id])).item()
        for node_id in node_mae_sum
    }
    return [
        node_id
        for node_id, _mae in sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))[:k]
    ]


def evaluate_checkpoint_topk_forecasts(
    *,
    checkpoint_path: Path,
    split: str = "val",
    k: int = 5,
    device: str = "auto",
    window: str = "last",
    output_path: Path | None = None,
    log_dir: Path | None = None,
    eval_csv_path: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    End-to-end: load checkpoint, compute top-k nodes, collect series, and (optionally) save figure.

    Returns a dict containing: model, config, loader, topk_nodes, samples, figure.
    """

    start_time = time.time()
    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device=device
    )
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        f"[eval] Building {split} loader from dataset: {config.data.dataset_path}"
    )
    loader, region_embeddings = build_loader_from_config(
        config, split=split, device=device, batch_size=batch_size
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(f"[eval] {split} samples: {len(dataset)}")
    logger.info(f"[eval] Scanning for top-k nodes by MAE (k={k})...")

    topk_nodes = topk_target_nodes_by_mae(
        model=model, loader=loader, region_embeddings=region_embeddings, k=k
    )
    logger.debug(f"[eval] Top-k scan done in {time.time() - start_time:.2f}s")
    logger.info("[eval] Collecting forecast samples for top-k nodes...")
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=topk_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=30,
        context_post=30,
    )

    fig = make_forecast_figure(
        samples=samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=30,
        context_post=30,
    )
    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    node_mae_dict: dict[int, float] = {}
    try:
        criterion = get_loss_from_config(
            config.training.loss,
            data_config=config.data,
            forecast_horizon=config.model.forecast_horizon,
        )
        if not isinstance(criterion, JointInferenceLoss):
            raise ValueError(
                "Evaluation now requires JointInferenceLoss. "
                "Set training.loss.name=joint_inference in the config."
            )
        _maybe_load_criterion_state_from_checkpoint(
            criterion=criterion,
            checkpoint=checkpoint,
        )
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            output_csv_path=eval_csv_path,
        )
    except Exception as exc:  # pragma: no cover - evaluation best-effort
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    if log_dir is not None or wandb.run is not None:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"eval_{split}_{checkpoint_path.parent.parent.name}"
        _ensure_wandb_run(
            config=config, log_dir=log_dir, name=run_name, job_type="eval"
        )
        if wandb.run is not None:
            log_data: dict[str, Any] = {}
            if math.isfinite(eval_loss):
                log_data[f"loss_{split}"] = eval_loss
            for key in ("mae", "rmse", "smape", "r2"):
                if key in eval_metrics:
                    log_data[f"{key}_{split}"] = eval_metrics[key]
            if log_data:
                wandb.log(log_data, step=0)

    return {
        "checkpoint": checkpoint,
        "config": config,
        "model": model,
        "loader": loader,
        "topk_nodes": topk_nodes,
        "samples": samples,
        "figure": fig,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
        "node_mae": node_mae_dict,
        "log_dir": log_dir,
    }


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    def _fmt(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"

    rows = [
        ("Loss", _fmt(loss)),
        ("MAE", _fmt(metrics.get("mae"))),
        ("RMSE", _fmt(metrics.get("rmse"))),
        ("sMAPE", _fmt(metrics.get("smape"))),
        ("R2", _fmt(metrics.get("r2"))),
    ]
    table = ["| Metric | Value |", "|---|---|"]
    for name, value in rows:
        table.append(f"| {name} | {value} |")
    return "\n".join(table)


def evaluate_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: JointInferenceLoss,
    horizon: int,
    device: torch.device,
    region_embeddings: torch.Tensor | None = None,
    split_name: str = "Eval",
    max_batches: int | None = None,
    output_csv_path: Path | None = None,
) -> tuple[float, dict[str, Any], dict[int, float]]:
    """Evaluate a loader and compute loss/metrics matching trainer behavior.

    Uses device-local metric accumulation to minimize CPU-GPU synchronization.
    """
    logger.info(f"{split_name} evaluation started...")
    criterion = criterion.to(device)
    # Device-local accumulators - avoid sync until end
    total_loss = torch.tensor(0.0, device=device)
    hosp_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    ww_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    cases_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    deaths_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    loss_ww_sum = torch.tensor(0.0, device=device)
    loss_hosp_sum = torch.tensor(0.0, device=device)
    loss_cases_sum = torch.tensor(0.0, device=device)
    loss_deaths_sum = torch.tensor(0.0, device=device)
    loss_sir_sum = torch.tensor(0.0, device=device)
    loss_ww_raw_sum = torch.tensor(0.0, device=device)
    loss_hosp_raw_sum = torch.tensor(0.0, device=device)
    loss_cases_raw_sum = torch.tensor(0.0, device=device)
    loss_deaths_raw_sum = torch.tensor(0.0, device=device)
    loss_ww_weighted_sum = torch.tensor(0.0, device=device)
    loss_hosp_weighted_sum = torch.tensor(0.0, device=device)
    loss_cases_weighted_sum = torch.tensor(0.0, device=device)
    loss_deaths_weighted_sum = torch.tensor(0.0, device=device)
    loss_sir_weighted_sum = torch.tensor(0.0, device=device)
    horizon_metric_sums: dict[str, torch.Tensor] = {}
    horizon_metric_prefixes = (
        "loss_ww_h",
        "loss_hosp_h",
        "loss_cases_h",
        "loss_deaths_h",
        "horizon_scale_",
    )

    # For node-level MAE, accumulate in dict but defer item() calls
    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    num_batches = len(loader)
    eval_iter = loader
    log_every = 10

    model_was_training = model.training
    criterion_was_training = criterion.training
    model.eval()
    criterion.eval()
    forward_model = cast(EpiForecaster, model)
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            for batch_idx, batch_data in enumerate(eval_iter):
                if max_batches and batch_idx >= max_batches:
                    break
                if batch_idx % log_every == 0:
                    logger.info(f"{split_name} evaluation: {batch_idx}/{num_batches}")

                from utils.training_utils import inject_gpu_mobility

                inject_gpu_mobility(batch_data, eval_iter.dataset, device)

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                    mask_cases=criterion.mask_input_cases,
                    mask_ww=criterion.mask_input_ww,
                    mask_hosp=criterion.mask_input_hosp,
                    mask_deaths=criterion.mask_input_deaths,
                )

                # Slice predictions to match target horizon (remove t=0 nowcast)
                from utils.training_utils import drop_nowcast

                # Create sliced model outputs for metric computation
                sliced_model_outputs = {
                    k: drop_nowcast(v, horizon)
                    if k.startswith("pred_") and isinstance(v, torch.Tensor)
                    else v
                    for k, v in model_outputs.items()
                }

                # Compute loss with batch_data for continuity penalty
                components = criterion.compute_components(
                    model_outputs,
                    targets_dict,
                    batch_data,
                    emit_horizon_diagnostics=True,
                )
                metric_supervision = criterion.compute_observation_supervision(
                    targets_dict,
                    device=device,
                )
                loss = components["total"]
                total_loss += loss.detach()
                loss_ww_sum += components["ww"].detach()
                loss_hosp_sum += components["hosp"].detach()
                loss_cases_sum += components["cases"].detach()
                loss_deaths_sum += components["deaths"].detach()
                loss_sir_sum += components["sir"].detach()
                loss_ww_raw_sum += components["ww_raw"].detach()
                loss_hosp_raw_sum += components["hosp_raw"].detach()
                loss_cases_raw_sum += components["cases_raw"].detach()
                loss_deaths_raw_sum += components["deaths_raw"].detach()
                if "continuity" in components:
                    pass  # Don't accumulate continuity loss in metrics
                loss_ww_weighted_sum += components["ww_weighted"].detach()
                loss_hosp_weighted_sum += components["hosp_weighted"].detach()
                loss_cases_weighted_sum += components["cases_weighted"].detach()
                loss_deaths_weighted_sum += components["deaths_weighted"].detach()
                loss_sir_weighted_sum += components["sir_weighted"].detach()
                for key, value in components.items():
                    if not isinstance(value, torch.Tensor) or value.ndim != 0:
                        continue
                    if not key.startswith(horizon_metric_prefixes):
                        continue
                    if key not in horizon_metric_sums:
                        horizon_metric_sums[key] = torch.tensor(0.0, device=device)
                    horizon_metric_sums[key] += value.detach()

                # Log sparsity-loss correlation during evaluation (moved from training)
                if batch_idx % 10 == 0:
                    log_sparsity_loss_correlation(
                        batch=batch_data,
                        model_outputs=model_outputs,
                        targets=targets_dict,
                        wandb_run=None,
                        step=batch_idx,
                        epoch=0,
                    )

                pred_hosp = sliced_model_outputs.get("pred_hosp")
                hosp_targets = metric_supervision["hosp"]["target"]
                hosp_mask = targets_dict.get("hosp_mask")
                hosp_weights = metric_supervision["hosp"]["weights"]
                if (
                    pred_hosp is not None
                    and hosp_targets is not None
                    and hosp_weights is not None
                ):
                    _diff, abs_diff, weights = hosp_metrics.update(
                        predictions=pred_hosp,
                        targets=hosp_targets,
                        observed_mask=hosp_mask,
                        sample_weights=hosp_weights,
                    )
                    # Per-node MAE - keep tensors on device until end
                    valid_per_sample = weights.sum(dim=1) > 0
                    per_sample_mae = (abs_diff * weights).sum(dim=1) / weights.sum(
                        dim=1
                    ).clamp_min(1e-8)
                    target_nodes = batch_data["TargetNode"]
                    for sample_mae, target_node, is_valid in zip(
                        per_sample_mae, target_nodes, valid_per_sample, strict=False
                    ):
                        if not bool(is_valid):
                            continue
                        node_id = int(target_node)
                        if node_id not in node_mae_sum:
                            node_mae_sum[node_id] = torch.tensor(0.0, device=device)
                        node_mae_sum[node_id] += sample_mae.detach()
                        node_mae_count[node_id] = node_mae_count.get(node_id, 0) + 1

                pred_ww = sliced_model_outputs.get("pred_ww")
                ww_targets = metric_supervision["ww"]["target"]
                ww_mask = targets_dict.get("ww_mask")
                ww_weights = metric_supervision["ww"]["weights"]
                if (
                    pred_ww is not None
                    and ww_targets is not None
                    and ww_weights is not None
                ):
                    ww_metrics.update(
                        predictions=pred_ww,
                        targets=ww_targets,
                        observed_mask=ww_mask,
                        sample_weights=ww_weights,
                    )

                pred_cases = sliced_model_outputs.get("pred_cases")
                cases_targets = metric_supervision["cases"]["target"]
                cases_mask = targets_dict.get("cases_mask")
                cases_weights = metric_supervision["cases"]["weights"]
                if (
                    pred_cases is not None
                    and cases_targets is not None
                    and cases_weights is not None
                ):
                    cases_metrics.update(
                        predictions=pred_cases,
                        targets=cases_targets,
                        observed_mask=cases_mask,
                        sample_weights=cases_weights,
                    )

                pred_deaths = sliced_model_outputs.get("pred_deaths")
                deaths_targets = metric_supervision["deaths"]["target"]
                deaths_mask = targets_dict.get("deaths_mask")
                deaths_weights = metric_supervision["deaths"]["weights"]
                if (
                    pred_deaths is not None
                    and deaths_targets is not None
                    and deaths_weights is not None
                ):
                    deaths_metrics.update(
                        predictions=pred_deaths,
                        targets=deaths_targets,
                        observed_mask=deaths_mask,
                        sample_weights=deaths_weights,
                    )

    finally:
        if model_was_training:
            model.train()
        if criterion_was_training:
            criterion.train()

    # Final sync - transfer metrics to CPU once
    mean_loss = (total_loss / max(1, num_batches)).item()
    hosp_summary = hosp_metrics.finalize()
    ww_summary = ww_metrics.finalize()
    cases_summary = cases_metrics.finalize()
    deaths_summary = deaths_metrics.finalize()

    # Convert node MAE tensors to scalars
    node_mae = {
        node_id: (node_mae_sum[node_id] / max(1, node_mae_count[node_id])).item()
        for node_id in node_mae_sum
    }

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as csv_lib

        with open(output_csv_path, "w", newline="") as f:
            writer = csv_lib.writer(f)
            writer.writerow(["node_id", "mae", "num_samples"])
            for node_id in sorted(node_mae.keys()):
                writer.writerow([node_id, node_mae[node_id], node_mae_count[node_id]])

    metrics = {
        # Legacy primary metrics (hospitalizations)
        "mae": hosp_summary.mae,
        "rmse": hosp_summary.rmse,
        "smape": hosp_summary.smape,
        "r2": hosp_summary.r2,
        "mae_per_h": hosp_summary.mae_per_h,
        "rmse_per_h": hosp_summary.rmse_per_h,
        # Hospitalization metrics in log1p(per-100k) space
        "mae_hosp_log1p_per_100k": hosp_summary.mae,
        "rmse_hosp_log1p_per_100k": hosp_summary.rmse,
        "smape_hosp_log1p_per_100k": hosp_summary.smape,
        "r2_hosp_log1p_per_100k": hosp_summary.r2,
        "observed_count_hosp": hosp_summary.observed_count,
        "effective_count_hosp": hosp_summary.effective_count,
        # Wastewater metrics in log1p(per-100k) space
        "mae_ww_log1p_per_100k": ww_summary.mae,
        "rmse_ww_log1p_per_100k": ww_summary.rmse,
        "smape_ww_log1p_per_100k": ww_summary.smape,
        "r2_ww_log1p_per_100k": ww_summary.r2,
        "observed_count_ww": ww_summary.observed_count,
        "effective_count_ww": ww_summary.effective_count,
        # Cases metrics in log1p(per-100k) space
        "mae_cases_log1p_per_100k": cases_summary.mae,
        "rmse_cases_log1p_per_100k": cases_summary.rmse,
        "smape_cases_log1p_per_100k": cases_summary.smape,
        "r2_cases_log1p_per_100k": cases_summary.r2,
        "observed_count_cases": cases_summary.observed_count,
        "effective_count_cases": cases_summary.effective_count,
        # Deaths metrics in log1p(per-100k) space
        "mae_deaths_log1p_per_100k": deaths_summary.mae,
        "rmse_deaths_log1p_per_100k": deaths_summary.rmse,
        "smape_deaths_log1p_per_100k": deaths_summary.smape,
        "r2_deaths_log1p_per_100k": deaths_summary.r2,
        "observed_count_deaths": deaths_summary.observed_count,
        "effective_count_deaths": deaths_summary.effective_count,
        # Joint loss components (averaged per batch, same reduction as mean_loss)
        "loss_ww": (loss_ww_sum / max(1, num_batches)).item(),
        "loss_hosp": (loss_hosp_sum / max(1, num_batches)).item(),
        "loss_cases": (loss_cases_sum / max(1, num_batches)).item(),
        "loss_deaths": (loss_deaths_sum / max(1, num_batches)).item(),
        "loss_ww_raw": (loss_ww_raw_sum / max(1, num_batches)).item(),
        "loss_hosp_raw": (loss_hosp_raw_sum / max(1, num_batches)).item(),
        "loss_cases_raw": (loss_cases_raw_sum / max(1, num_batches)).item(),
        "loss_deaths_raw": (loss_deaths_raw_sum / max(1, num_batches)).item(),
        "loss_sir": (loss_sir_sum / max(1, num_batches)).item(),
        "loss_ww_weighted": (loss_ww_weighted_sum / max(1, num_batches)).item(),
        "loss_hosp_weighted": (loss_hosp_weighted_sum / max(1, num_batches)).item(),
        "loss_cases_weighted": (loss_cases_weighted_sum / max(1, num_batches)).item(),
        "loss_deaths_weighted": (loss_deaths_weighted_sum / max(1, num_batches)).item(),
        "loss_sir_weighted": (loss_sir_weighted_sum / max(1, num_batches)).item(),
    }
    for key, value in horizon_metric_sums.items():
        metrics[key] = (value / max(1, num_batches)).item()

    logger.info("EVAL COMPLETE")
    return mean_loss, metrics, node_mae


def generate_forecast_plots(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    node_groups: dict[str, list[int]],
    window: str = "last",
    context_pre: int = 30,
    context_post: int = 30,
    output_path: Path | None = None,
    log_dir: Path | None = None,
    target_names: list[str] | None = None,
    wandb_prefix: str = "forecasts",
) -> dict[str, Any]:
    """
    Generate forecast plots for given node groups (generic).

    Args:
        model: The trained model
        loader: Original DataLoader for data access
        node_groups: Dict mapping group name → list of node IDs
                     (could be quartiles, topk, worst, random, anything!)
        window: Which time window to plot ("last" or "random")
        context_pre: Days before forecast start
        context_post: Days after forecast end
        output_path: Optional path to save figure
    log_dir: Optional W&B run directory for eval metrics

    Returns:
        Dict with figure, all_samples, selected_nodes, node_groups
    """
    # Flatten all nodes to collect samples once
    all_selected_nodes: list[int] = []
    for group_nodes in node_groups.values():
        all_selected_nodes.extend(group_nodes)

    if not all_selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "all_samples": [],
            "selected_nodes": [],
            "node_groups": {},
        }

    logger.info(
        f"[plot] Collecting forecast samples for {len(all_selected_nodes)} nodes..."
    )

    # Use existing function - it handles subset creation internally
    resolved_targets = target_names or list(DEFAULT_PLOT_TARGETS)

    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=all_selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
    )

    # Group samples by original group names
    node_to_group: dict[int, str] = {}
    for group_name, nodes in node_groups.items():
        for node_id in nodes:
            node_to_group[node_id] = group_name

    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        node_id = sample["node_id"]
        if node_id in node_to_group:
            group_name = node_to_group[node_id]
            if group_name not in grouped_samples:
                grouped_samples[group_name] = []
            grouped_samples[group_name].append(sample)

    # Generate figure using existing generic function
    dataset = cast(EpiDataset, loader.dataset)
    config = dataset.config
    fig = make_joint_forecast_figure(
        samples=grouped_samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    separate_figures: dict[str, Any] = {}
    for target_name in resolved_targets:
        target_fig = make_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=context_pre,
            context_post=context_post,
            target=target_name,
        )
        if target_fig is None:
            continue
        separate_figures[target_name] = target_fig
        if output_path is not None:
            target_output_path = output_path.with_name(
                f"{output_path.stem}_{target_name}{output_path.suffix}"
            )
            target_fig.savefig(target_output_path, dpi=200, bbox_inches="tight")
            logger.info(f"[plot] Saved figure to: {target_output_path}")

    if fig is not None and (log_dir is not None or wandb.run is not None):
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        _ensure_wandb_run(
            config=dataset.config,
            log_dir=log_dir,
            name="forecast_plots",
            job_type="eval",
        )
        if wandb.run is not None:
            log_payload: dict[str, Any] = {}
            if fig is not None:
                log_payload[f"{wandb_prefix}/joint"] = wandb.Image(fig)
            for target_name, target_fig in separate_figures.items():
                log_payload[f"{wandb_prefix}/{target_name}"] = wandb.Image(target_fig)
            if log_payload:
                wandb.log(log_payload, step=0)

    return {
        "figure": fig,
        "joint_figure": fig,
        "separate_figures": separate_figures,
        "all_samples": samples,
        "selected_nodes": all_selected_nodes,
        "node_groups": node_groups,
    }


def eval_checkpoint(
    *,
    checkpoint_path: Path,
    split: str = "val",
    device: str = "auto",
    log_dir: Path | None = None,
    overrides: list[str] | None = None,
    output_csv_path: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate checkpoint - pure evaluation, no selection or plotting.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Which split to evaluate ("val" or "test")
        device: Device to use for evaluation (overridden by training.device in overrides)
    log_dir: Optional W&B run directory for forecast plots
        overrides: Optional list of dotted-key config overrides (e.g., ["training.val_workers=4"])
        output_csv_path: Optional path to save node-level metrics CSV

    Returns:
        Dict with: checkpoint, config, model, loader, node_mae_dict,
                   eval_loss, eval_metrics
    """
    # Extract training.device from overrides if present
    resolved_device = device
    if overrides:
        for ov in overrides:
            if ov.startswith("training.device="):
                resolved_device = ov.split("=", 1)[1]
                break

    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        overrides=list(overrides) if overrides else None,
    )
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        f"[eval] Building {split} loader from dataset: {config.data.dataset_path}"
    )
    loader, region_embeddings = build_loader_from_config(
        config, split=split, device=resolved_device, batch_size=batch_size
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(f"[eval] {split} samples: {len(dataset)}")

    # Run evaluation - returns node_mae_dict as third value
    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    node_mae_dict: dict[int, float] = {}
    try:
        criterion = get_loss_from_config(
            config.training.loss,
            data_config=config.data,
            forecast_horizon=config.model.forecast_horizon,
        )
        if not isinstance(criterion, JointInferenceLoss):
            raise ValueError(
                "Evaluation now requires JointInferenceLoss. "
                "Set training.loss.name=joint_inference in the config."
            )
        _maybe_load_criterion_state_from_checkpoint(
            criterion=criterion,
            checkpoint=checkpoint,
        )
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            output_csv_path=output_csv_path,
        )
    except Exception as exc:
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    forecast_plot_result: dict[str, Any] | None = None
    if split.lower() == "test" and node_mae_dict:
        k = max(1, int(config.training.num_forecast_samples))
        worst_nodes = select_nodes_by_loss(
            node_mae=node_mae_dict, strategy="worst", k=k
        ).get("Worst", [])
        best_nodes = select_nodes_by_loss(
            node_mae=node_mae_dict, strategy="best", k=k
        ).get("Best", [])
        node_groups = {"Poorly-performing": worst_nodes, "Well-performing": best_nodes}

        if any(node_groups.values()):
            output_path = None
            if log_dir is not None:
                output_path = log_dir / f"{split}_forecasts_joint.png"
            forecast_plot_result = generate_forecast_plots(
                model=model,
                loader=loader,
                node_groups=node_groups,
                window="last",
                context_pre=30,
                context_post=30,
                output_path=output_path,
                log_dir=log_dir,
                target_names=list(DEFAULT_PLOT_TARGETS),
                wandb_prefix=f"forecasts_{split}",
            )
        else:
            logger.warning("[plot] Could not select test nodes for forecast plots")

    if log_dir is not None or wandb.run is not None:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"eval_{split}_{checkpoint_path.parent.parent.name}"
        _ensure_wandb_run(
            config=config, log_dir=log_dir, name=run_name, job_type="eval"
        )
        if wandb.run is not None:
            log_data: dict[str, Any] = {}
            if math.isfinite(eval_loss):
                log_data[f"loss_{split}"] = eval_loss
            for key in ("mae", "rmse", "smape", "r2"):
                if key in eval_metrics:
                    log_data[f"{key}_{split}"] = eval_metrics[key]
            if log_data:
                wandb.log(log_data, step=0)

    return {
        "checkpoint": checkpoint,
        "config": config,
        "model": model,
        "loader": loader,
        "node_mae": node_mae_dict,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
        "forecast_plots": forecast_plot_result,
    }


def plot_forecasts_from_csv(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    samples_per_quartile: int = 2,
    window: str = "last",
    device: str = "auto",
    output_path: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Load evaluation CSV, sample nodes from quartiles, and generate forecast plots.

    Args:
        csv_path: Path to CSV with columns node_id, mae, num_samples
        checkpoint_path: Path to model checkpoint
        samples_per_quartile: Number of nodes to sample from each quartile (default 2)
        window: Which window to plot ('last' or 'random')
        device: Device to use for inference
        output_path: Optional path to save the figure

    Returns:
        Dict containing: figure, selected_nodes, quartile_groups, config
    """
    import csv as csv_lib

    logger.info(f"[plot] Loading evaluation CSV: {csv_path}")
    node_mae_list: list[tuple[int, float, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            node_id = int(row["node_id"])
            mae = float(row["mae"])
            num_samples = int(row["num_samples"])
            node_mae_list.append((node_id, mae, num_samples))

    if not node_mae_list:
        logger.warning("[plot] No valid nodes found in CSV")
        return {
            "figure": None,
            "selected_nodes": [],
            "quartile_groups": {},
            "config": None,
        }

    node_mae_list.sort(key=lambda x: x[1])

    maes = [mae for _, mae, _ in node_mae_list]
    q1_cutoff = np.percentile(maes, 25)
    q2_cutoff = np.percentile(maes, 50)
    q3_cutoff = np.percentile(maes, 75)

    quartile_groups: dict[str, list[int]] = {
        "Q1 (Worst)": [],
        "Q2 (Poor)": [],
        "Q3 (Average)": [],
        "Q4 (Best)": [],
    }

    for node_id, mae, num_samples in node_mae_list:
        if mae <= q1_cutoff:
            quartile_groups["Q1 (Worst)"].append(node_id)
        elif mae <= q2_cutoff:
            quartile_groups["Q2 (Poor)"].append(node_id)
        elif mae <= q3_cutoff:
            quartile_groups["Q3 (Average)"].append(node_id)
        else:
            quartile_groups["Q4 (Best)"].append(node_id)

    selected_nodes: list[int] = []
    import random

    for quartile_name, nodes in quartile_groups.items():
        available = len(nodes)
        k = min(samples_per_quartile, available)
        sampled = random.sample(nodes, k)
        quartile_groups[quartile_name] = sampled
        selected_nodes.extend(sampled)
        logger.info(
            f"[plot] {quartile_name}: sampled {k} nodes (available: {available})"
        )

    if not selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "selected_nodes": [],
            "quartile_groups": {},
            "config": None,
        }

    logger.info(f"[plot] Loading checkpoint: {checkpoint_path}")
    model, config, _checkpoint = load_model_from_checkpoint(
        checkpoint_path, device=device
    )

    loader, _region_embeddings = build_loader_from_config(
        config, split="val", device=device, batch_size=batch_size
    )
    logger.info(
        f"[plot] Collecting forecast samples for {len(selected_nodes)} nodes..."
    )
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=30,
        context_post=30,
    )

    quartile_samples: dict[str, list[dict[str, Any]]] = {
        name: [] for name in quartile_groups.keys()
    }
    node_to_quartile: dict[int, str] = {}
    for quartile_name, nodes in quartile_groups.items():
        for node_id in nodes:
            node_to_quartile[node_id] = quartile_name

    for sample in samples:
        node_id = sample["node_id"]
        if node_id in node_to_quartile:
            quartile_samples[node_to_quartile[node_id]].append(sample)

    fig = make_forecast_figure(
        samples=quartile_samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=30,
        context_post=30,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    return {
        "figure": fig,
        "selected_nodes": selected_nodes,
        "quartile_groups": quartile_groups,
        "samples": samples,
        "config": config,
    }
