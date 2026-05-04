from __future__ import annotations

from typing import TYPE_CHECKING, cast
import torch
import torch.nn as nn

from models.configs import DataConfig, JointLossConfig, LossConfig
from utils.device import sync_to_device
from utils.training_utils import drop_nowcast

if TYPE_CHECKING:
    from data.epi_batch import EpiBatch

_LOSS_VALUE_CLAMP = 1.0e6


def get_loss_from_config(
    loss_config: LossConfig | None,
    *,
    data_config: DataConfig | None = None,
    forecast_horizon: int | None = None,
) -> JointInferenceLoss:
    """Build the only supported EpiForecaster loss: JointInferenceLoss."""
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

    min_obs = {"cases": 0, "hospitalizations": 0, "deaths": 0, "wastewater": 0}
    if (
        data_config is not None
        and forecast_horizon is not None
        and hasattr(data_config, "resolve_min_observed_map")
    ):
        min_obs = data_config.resolve_min_observed_map(
            forecast_horizon=int(forecast_horizon)
        )

    return JointInferenceLoss(
        obs_weight_sum=joint_cfg.gradnorm_obs_weight_sum,
        w_sird_supervision=joint_cfg.w_sird_supervision,
        w_continuity=joint_cfg.w_continuity,
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
    )


class JointInferenceLoss(nn.Module):
    """
    Joint inference loss combining observation and latent SIRD supervision losses.

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
    _CONTINUITY_HEADS = (
        ("hosp", "hosp_hist", "pred_hosp"),
        ("cases", "cases_hist", "pred_cases"),
        ("deaths", "deaths_hist", "pred_deaths"),
    )
    _LATENT_COMPONENTS = (
        ("latent_s", "S_trajectory", "S_target", "S_target_mask"),
        ("latent_i", "I_trajectory", "I_target", "I_target_mask"),
        ("latent_r", "R_trajectory", "R_target", "R_target_mask"),
        ("latent_d", "D_trajectory", "D_target", "D_target_mask"),
    )

    def __init__(
        self,
        obs_weight_sum: float = 0.95,
        w_sird_supervision: float = 0.0,
        w_sir: float | None = None,
        w_continuity: float = 0.0,
        sir_residual_clip: float | None = None,
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
    ):
        super().__init__()
        self.obs_weight_sum = float(obs_weight_sum)
        del sir_residual_clip
        if w_sir is not None and w_sird_supervision == 0.0:
            w_sird_supervision = float(w_sir)
        self.w_sird_supervision = float(w_sird_supervision)
        self.w_continuity = w_continuity
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
        if self.obs_weight_sum <= 0:
            raise ValueError(
                f"obs_weight_sum must be positive, got {self.obs_weight_sum}"
            )
        if self.w_sird_supervision < 0:
            raise ValueError(
                "w_sird_supervision must be non-negative, "
                f"got {self.w_sird_supervision}"
            )

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
            self.w_sird_supervision * components["sird_supervision"]
        )
        continuity_weighted = self.w_continuity * components["continuity"]
        total = obs_total + sird_supervision_weighted + continuity_weighted
        return {
            "obs_weights": obs_weights,
            "obs_total": obs_total,
            "total": total,
            "ww_weighted": obs_weighted[0],
            "hosp_weighted": obs_weighted[1],
            "cases_weighted": obs_weighted[2],
            "deaths_weighted": obs_weighted[3],
            "sird_supervision_weighted": sird_supervision_weighted,
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
        return (sq * weights_f32).sum() / weights_f32.sum().clamp_min(1e-8)

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
                - S_trajectory/I_trajectory/R_trajectory/D_trajectory: [B, H+1]
            targets: Dict containing target tensors:
                - ww: [B, H] wastewater targets (optional)
                - hosp: [B, H] hospitalization targets (optional)
                - cases: [B, H] reported cases targets (optional)
                - deaths: [B, H] deaths targets (optional)
                - S_target/I_target/R_target/D_target: [B, H+1] latent targets (optional)
            batch_data: Optional EpiBatch containing historical data for continuity:
                - hosp_hist: [B, L, 3] hospitalization history
                - cases_hist: [B, L, 3] cases history
                - deaths_hist: [B, L, 3] deaths history

        Returns:
            Dict with unweighted and weighted component losses plus total:
                - ww, hosp, cases, deaths, latent_s, latent_i, latent_r, latent_d, sird_supervision, continuity
                - ww_weighted, hosp_weighted, cases_weighted, deaths_weighted,
                  sird_supervision_weighted, continuity_weighted
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
        latent_s_loss = zero_anchor
        latent_i_loss = zero_anchor
        latent_r_loss = zero_anchor
        latent_d_loss = zero_anchor
        sird_supervision_loss = zero_anchor
        continuity_loss = zero_anchor
        obs_supervision = self.compute_observation_supervision(
            targets,
            device=zero_anchor.device,
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
            ww_base = self._weighted_masked_mse_from_weights(
                prediction=drop_nowcast(model_outputs["pred_ww"], ww_target.shape[1]),
                target=ww_target,
                weights=ww_weights,
            )
            ww_loss = ww_base

        hosp_target = obs_supervision["hosp"]["target"]
        hosp_weights = obs_supervision["hosp"]["weights"]
        if hosp_target is not None and hosp_weights is not None:
            hosp_base = self._weighted_masked_mse_from_weights(
                prediction=drop_nowcast(
                    model_outputs["pred_hosp"], hosp_target.shape[1]
                ),
                target=hosp_target,
                weights=hosp_weights,
            )
            hosp_loss = hosp_base

        cases_target = obs_supervision["cases"]["target"]
        cases_weights = obs_supervision["cases"]["weights"]
        if cases_target is not None and cases_weights is not None:
            cases_base = self._weighted_masked_mse_from_weights(
                prediction=drop_nowcast(
                    model_outputs["pred_cases"], cases_target.shape[1]
                ),
                target=cases_target,
                weights=cases_weights,
            )
            cases_loss = cases_base

        deaths_target = obs_supervision["deaths"]["target"]
        deaths_weights = obs_supervision["deaths"]["weights"]
        if deaths_target is not None and deaths_weights is not None:
            deaths_base = self._weighted_masked_mse_from_weights(
                prediction=model_outputs["pred_deaths"],
                target=deaths_target,
                weights=deaths_weights,
            )
            deaths_loss = deaths_base

        if self.w_sird_supervision > 0:
            latent_component_losses: list[torch.Tensor] = []
            latent_component_map: dict[str, torch.Tensor] = {}
            for component_name, output_key, target_key, mask_key in self._LATENT_COMPONENTS:
                target = targets.get(target_key)
                if target is None:
                    continue
                supervision = self._build_supervision_weights(
                    target=target,
                    observed_mask=targets.get(mask_key),
                    min_observed=0,
                    device=zero_anchor.device,
                )
                latent_target = supervision["target"]
                latent_weights = supervision["weights"]
                if latent_target is None or latent_weights is None:
                    continue
                latent_loss = self._weighted_masked_mse_from_weights(
                    prediction=model_outputs[output_key],
                    target=latent_target,
                    weights=latent_weights,
                )
                latent_component_losses.append(latent_loss)
                latent_component_map[component_name] = latent_loss

            latent_s_loss = latent_component_map.get("latent_s", zero_anchor)
            latent_i_loss = latent_component_map.get("latent_i", zero_anchor)
            latent_r_loss = latent_component_map.get("latent_r", zero_anchor)
            latent_d_loss = latent_component_map.get("latent_d", zero_anchor)
            if latent_component_losses:
                sird_supervision_loss = torch.stack(latent_component_losses).mean()

        # Nowcast continuity penalty
        if self.w_continuity > 0 and batch_data is not None:
            continuity_loss = self._compute_continuity_loss(
                model_outputs=model_outputs,
                batch_data=batch_data,
                obs_supervision=obs_supervision,
            )

        components = {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            "latent_s": latent_s_loss,
            "latent_i": latent_i_loss,
            "latent_r": latent_r_loss,
            "latent_d": latent_d_loss,
            "sird_supervision": sird_supervision_loss,
            "continuity": continuity_loss,
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
            "latent_s": latent_s_loss,
            "latent_i": latent_i_loss,
            "latent_r": latent_r_loss,
            "latent_d": latent_d_loss,
            "sird_supervision": sird_supervision_loss,
            "continuity": continuity_loss,
            "ww_weighted": totals["ww_weighted"],
            "hosp_weighted": totals["hosp_weighted"],
            "cases_weighted": totals["cases_weighted"],
            "deaths_weighted": totals["deaths_weighted"],
            "sird_supervision_weighted": totals["sird_supervision_weighted"],
            "continuity_weighted": totals["continuity_weighted"],
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

    def _compute_continuity_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch_data: EpiBatch,
        obs_supervision: dict[str, dict[str, torch.Tensor | None]],
    ) -> torch.Tensor:
        """
        Compute nowcast continuity penalty for active observation heads only.

        Penalizes the discontinuity between the last observed value and the
        model's first forecast prediction (t=0, the nowcast).

        Args:
            model_outputs: Dict containing predictions with t=0 (nowcast)
            batch_data: EpiBatch dataclass containing historical observations
            obs_supervision: Per-head supervision info including active flags

        Returns:
            Scalar continuity loss averaged over active continuity heads
        """
        zero_anchor = (
            torch.nan_to_num(
                model_outputs["pred_hosp"].float(), nan=0.0, posinf=0.0, neginf=0.0
            ).sum()
            * 0.0
        )
        component_losses: list[torch.Tensor] = []
        active_flags: list[torch.Tensor] = []

        def _masked_mse(
            nowcast_pred: torch.Tensor, last_observed: torch.Tensor
        ) -> torch.Tensor:
            valid_mask = torch.isfinite(last_observed)
            valid_f = valid_mask.to(device=nowcast_pred.device, dtype=nowcast_pred.dtype)
            last_observed_safe = torch.nan_to_num(
                last_observed.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).to(device=nowcast_pred.device, dtype=nowcast_pred.dtype)
            sq = (nowcast_pred - last_observed_safe) ** 2
            numerator = (sq * valid_f).sum()
            denominator = valid_f.sum().clamp_min(1.0)
            return numerator / denominator

        for head_name, hist_field, pred_key in self._CONTINUITY_HEADS:
            active = cast(torch.Tensor, obs_supervision[head_name]["active"])

            prediction = model_outputs.get(pred_key)
            if prediction is None or prediction.shape[1] == 0:
                continue
            hist_tensor = getattr(batch_data, hist_field, None)
            if hist_tensor is None:
                continue

            head_loss = _masked_mse(prediction[:, 0], hist_tensor[:, -1, 0])
            active_f = active.to(dtype=head_loss.dtype)
            component_losses.append(head_loss * active_f)
            active_flags.append(active_f)

        if component_losses:
            stacked_losses = torch.stack(component_losses)
            stacked_flags = torch.stack(active_flags)
            return stacked_losses.sum() / stacked_flags.sum().clamp_min(1.0)
        return zero_anchor
