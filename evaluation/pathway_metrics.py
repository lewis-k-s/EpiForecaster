from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EPSILON = 1.0e-12
_VALUE_CLAMP = 1.0e6


def _clean(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=_VALUE_CLAMP,
        neginf=-_VALUE_CLAMP,
    ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) <= _EPSILON:
        return float("nan")
    return numerator / denominator


@dataclass
class PathwayMetricResult:
    observed_count: int
    effective_count: float
    mean_abs_mechanistic: float
    mean_abs_residual: float
    residual_abs_fraction: float
    residual_to_mechanistic_abs_ratio: float
    mechanistic_variance: float
    residual_variance: float
    full_variance: float
    mechanistic_variance_share: float
    residual_variance_share: float
    mechanistic_only_mae: float
    mechanistic_only_rmse: float
    reconstruction_max_abs_error: float


class TorchPathwayMetricAccumulator:
    """Accumulate observation-head pathway diagnostics on the evaluation device."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.weight_sum = torch.tensor(0.0, device=device)
        self.observed_count = 0
        self.abs_mechanistic_sum = torch.tensor(0.0, device=device)
        self.abs_residual_sum = torch.tensor(0.0, device=device)
        self.mechanistic_error_abs_sum = torch.tensor(0.0, device=device)
        self.mechanistic_error_sq_sum = torch.tensor(0.0, device=device)
        self.reconstruction_max_abs_error = torch.tensor(0.0, device=device)
        self.weighted_sums = {
            name: torch.tensor(0.0, device=device)
            for name in ("full", "mechanistic", "residual")
        }
        self.weighted_sq_sums = {
            name: torch.tensor(0.0, device=device)
            for name in ("full", "mechanistic", "residual")
        }

    def update(
        self,
        *,
        full: torch.Tensor,
        mechanistic: torch.Tensor,
        residual: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        finite = (
            torch.isfinite(full.float())
            & torch.isfinite(mechanistic.float())
            & torch.isfinite(residual.float())
            & torch.isfinite(targets.float())
        )
        effective_weights = (
            torch.nan_to_num(
                weights.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            * finite.to(dtype=torch.float32)
        )
        full_clean = _clean(full)
        mechanistic_clean = _clean(mechanistic)
        residual_clean = _clean(residual)
        targets_clean = _clean(targets)

        self.weight_sum += effective_weights.sum()
        self.observed_count += int((effective_weights > 0).sum().item())
        self.abs_mechanistic_sum += (
            mechanistic_clean.abs() * effective_weights
        ).sum()
        self.abs_residual_sum += (residual_clean.abs() * effective_weights).sum()
        mechanistic_error = mechanistic_clean - targets_clean
        self.mechanistic_error_abs_sum += (
            mechanistic_error.abs() * effective_weights
        ).sum()
        self.mechanistic_error_sq_sum += (
            mechanistic_error.square() * effective_weights
        ).sum()

        active = effective_weights > 0
        if active.any():
            reconstruction_error = (
                full_clean - mechanistic_clean - residual_clean
            ).abs()
            self.reconstruction_max_abs_error = torch.maximum(
                self.reconstruction_max_abs_error,
                reconstruction_error[active].max(),
            )

        for name, value in (
            ("full", full_clean),
            ("mechanistic", mechanistic_clean),
            ("residual", residual_clean),
        ):
            self.weighted_sums[name] += (value * effective_weights).sum()
            self.weighted_sq_sums[name] += (
                value.square() * effective_weights
            ).sum()

    def finalize(self) -> PathwayMetricResult:
        effective_count = float(self.weight_sum.item())
        if effective_count <= 0:
            nan = float("nan")
            return PathwayMetricResult(
                observed_count=0,
                effective_count=0.0,
                mean_abs_mechanistic=nan,
                mean_abs_residual=nan,
                residual_abs_fraction=nan,
                residual_to_mechanistic_abs_ratio=nan,
                mechanistic_variance=nan,
                residual_variance=nan,
                full_variance=nan,
                mechanistic_variance_share=nan,
                residual_variance_share=nan,
                mechanistic_only_mae=nan,
                mechanistic_only_rmse=nan,
                reconstruction_max_abs_error=nan,
            )

        mean_abs_mechanistic = float(
            (self.abs_mechanistic_sum / self.weight_sum).item()
        )
        mean_abs_residual = float((self.abs_residual_sum / self.weight_sum).item())
        variances: dict[str, float] = {}
        for name in self.weighted_sums:
            mean = self.weighted_sums[name] / self.weight_sum
            variance = self.weighted_sq_sums[name] / self.weight_sum - mean.square()
            variances[name] = float(variance.clamp_min(0.0).item())

        return PathwayMetricResult(
            observed_count=self.observed_count,
            effective_count=effective_count,
            mean_abs_mechanistic=mean_abs_mechanistic,
            mean_abs_residual=mean_abs_residual,
            residual_abs_fraction=_safe_ratio(
                mean_abs_residual,
                mean_abs_mechanistic + mean_abs_residual,
            ),
            residual_to_mechanistic_abs_ratio=_safe_ratio(
                mean_abs_residual,
                mean_abs_mechanistic,
            ),
            mechanistic_variance=variances["mechanistic"],
            residual_variance=variances["residual"],
            full_variance=variances["full"],
            mechanistic_variance_share=_safe_ratio(
                variances["mechanistic"],
                variances["full"],
            ),
            residual_variance_share=_safe_ratio(
                variances["residual"],
                variances["full"],
            ),
            mechanistic_only_mae=float(
                (self.mechanistic_error_abs_sum / self.weight_sum).item()
            ),
            mechanistic_only_rmse=float(
                torch.sqrt(self.mechanistic_error_sq_sum / self.weight_sum).item()
            ),
            reconstruction_max_abs_error=float(
                self.reconstruction_max_abs_error.item()
            ),
        )


@dataclass
class LatentMetricResult:
    count: int
    mae: float
    rmse: float
    pearson: float
    mae_over_target_std: float
    nonnegative_violation_fraction: float


class _TorchLatentPairAccumulator:
    def __init__(self, *, device: torch.device) -> None:
        self.count = torch.tensor(0.0, device=device)
        self.abs_error_sum = torch.tensor(0.0, device=device)
        self.sq_error_sum = torch.tensor(0.0, device=device)
        self.pred_sum = torch.tensor(0.0, device=device)
        self.target_sum = torch.tensor(0.0, device=device)
        self.pred_sq_sum = torch.tensor(0.0, device=device)
        self.target_sq_sum = torch.tensor(0.0, device=device)
        self.cross_sum = torch.tensor(0.0, device=device)
        self.pred_finite_count = torch.tensor(0.0, device=device)
        self.pred_negative_count = torch.tensor(0.0, device=device)

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor | None,
    ) -> None:
        pred_f32 = prediction.float()
        pred_finite = torch.isfinite(pred_f32)
        self.pred_finite_count += pred_finite.sum()
        self.pred_negative_count += (pred_finite & (pred_f32 < 0)).sum()
        if target is None:
            return

        target_f32 = target.float()
        finite = pred_finite & torch.isfinite(target_f32)
        count = finite.sum()
        if not count.any():
            return
        pred = _clean(pred_f32)[finite]
        truth = _clean(target_f32)[finite]
        error = pred - truth
        self.count += count
        self.abs_error_sum += error.abs().sum()
        self.sq_error_sum += error.square().sum()
        self.pred_sum += pred.sum()
        self.target_sum += truth.sum()
        self.pred_sq_sum += pred.square().sum()
        self.target_sq_sum += truth.square().sum()
        self.cross_sum += (pred * truth).sum()

    def finalize(self) -> LatentMetricResult:
        pred_finite_count = float(self.pred_finite_count.item())
        negative_fraction = (
            float((self.pred_negative_count / self.pred_finite_count).item())
            if pred_finite_count > 0
            else float("nan")
        )
        count = int(self.count.item())
        if count <= 0:
            return LatentMetricResult(
                count=0,
                mae=float("nan"),
                rmse=float("nan"),
                pearson=float("nan"),
                mae_over_target_std=float("nan"),
                nonnegative_violation_fraction=negative_fraction,
            )

        pred_mean = self.pred_sum / self.count
        target_mean = self.target_sum / self.count
        pred_variance = (self.pred_sq_sum / self.count - pred_mean.square()).clamp_min(
            0.0
        )
        target_variance = (
            self.target_sq_sum / self.count - target_mean.square()
        ).clamp_min(0.0)
        covariance = self.cross_sum / self.count - pred_mean * target_mean
        denominator = torch.sqrt(pred_variance * target_variance)
        pearson = (
            float((covariance / denominator).item())
            if float(denominator.item()) > _EPSILON
            else float("nan")
        )
        mae = float((self.abs_error_sum / self.count).item())
        target_std = float(torch.sqrt(target_variance).item())
        return LatentMetricResult(
            count=count,
            mae=mae,
            rmse=float(torch.sqrt(self.sq_error_sum / self.count).item()),
            pearson=pearson,
            mae_over_target_std=_safe_ratio(mae, target_std),
            nonnegative_violation_fraction=negative_fraction,
        )


class TorchLatentCoherenceAccumulator:
    """Accumulate latent truth agreement and structural SIRHD diagnostics."""

    _LATENT_KEYS = {
        "s": ("S_trajectory", "S_target"),
        "i": ("I_trajectory", "I_target"),
        "h": ("H_trajectory", "H_target"),
        "r": ("R_trajectory", "R_target"),
        "d": ("D_trajectory", "D_target"),
    }

    def __init__(self, *, device: torch.device) -> None:
        self.latents = {
            name: _TorchLatentPairAccumulator(device=device)
            for name in self._LATENT_KEYS
        }
        self.mass_error_sum = torch.tensor(0.0, device=device)
        self.mass_error_max = torch.tensor(0.0, device=device)
        self.mass_count = torch.tensor(0.0, device=device)
        self.death_transition_error_sum = torch.tensor(0.0, device=device)
        self.death_transition_count = torch.tensor(0.0, device=device)
        self.death_monotonic_violation_count = torch.tensor(0.0, device=device)
        self.flow_finite_counts = {
            name: torch.tensor(0.0, device=device)
            for name in ("hospitalization", "death")
        }
        self.flow_negative_counts = {
            name: torch.tensor(0.0, device=device)
            for name in ("hospitalization", "death")
        }
        self.physics_residual_abs_sum = torch.tensor(0.0, device=device)
        self.physics_residual_count = torch.tensor(0.0, device=device)

    def update(
        self,
        *,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
    ) -> None:
        for name, (prediction_key, target_key) in self._LATENT_KEYS.items():
            prediction = model_outputs.get(prediction_key)
            if prediction is None:
                continue
            self.latents[name].update(
                prediction=prediction,
                target=targets.get(target_key),
            )

        if not all(
            prediction_key in model_outputs
            for prediction_key, _target_key in self._LATENT_KEYS.values()
        ):
            return
        trajectories = [
            model_outputs[prediction_key].float()
            for prediction_key, _target_key in self._LATENT_KEYS.values()
        ]
        mass_finite = torch.stack(
            [torch.isfinite(trajectory) for trajectory in trajectories]
        ).all(dim=0)
        mass = torch.stack([_clean(trajectory) for trajectory in trajectories]).sum(
            dim=0
        )
        if mass_finite.any():
            mass_error = (mass[mass_finite] - 1.0).abs()
            self.mass_error_sum += mass_error.sum()
            self.mass_error_max = torch.maximum(
                self.mass_error_max,
                mass_error.max(),
            )
            self.mass_count += mass_finite.sum()

        death_trajectory = model_outputs["D_trajectory"].float()
        death_flow_tensor = model_outputs.get("death_flow")
        if death_flow_tensor is None:
            return
        death_flow = death_flow_tensor.float()
        death_delta = death_trajectory[:, 1:] - death_trajectory[:, :-1]
        transition_finite = torch.isfinite(death_delta) & torch.isfinite(death_flow)
        if transition_finite.any():
            transition_error = (
                _clean(death_delta) - _clean(death_flow)
            )[transition_finite].abs()
            self.death_transition_error_sum += transition_error.sum()
            self.death_transition_count += transition_finite.sum()
            self.death_monotonic_violation_count += (
                death_delta[transition_finite] < 0
            ).sum()

        for name, key in (
            ("hospitalization", "hospitalization_flow"),
            ("death", "death_flow"),
        ):
            flow_tensor = model_outputs.get(key)
            if flow_tensor is None:
                continue
            flow = flow_tensor.float()
            finite = torch.isfinite(flow)
            self.flow_finite_counts[name] += finite.sum()
            self.flow_negative_counts[name] += (finite & (flow < 0)).sum()

        physics_residual_tensor = model_outputs.get("physics_residual")
        if physics_residual_tensor is None:
            return
        physics_residual = physics_residual_tensor.float()
        physics_finite = torch.isfinite(physics_residual)
        if physics_finite.any():
            self.physics_residual_abs_sum += physics_residual[
                physics_finite
            ].abs().sum()
            self.physics_residual_count += physics_finite.sum()

    @staticmethod
    def _fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        if float(denominator.item()) <= 0:
            return float("nan")
        return float((numerator / denominator).item())

    def finalize(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for name, accumulator in self.latents.items():
            result = accumulator.finalize()
            metrics.update(
                {
                    f"latent_count_{name}": result.count,
                    f"latent_mae_{name}_fraction": result.mae,
                    f"latent_rmse_{name}_fraction": result.rmse,
                    f"latent_pearson_{name}": result.pearson,
                    f"latent_mae_over_target_std_{name}": result.mae_over_target_std,
                    f"latent_nonnegative_violation_fraction_{name}": (
                        result.nonnegative_violation_fraction
                    ),
                }
            )

        mass_count = float(self.mass_count.item())
        metrics["latent_mass_conservation_mae"] = (
            float((self.mass_error_sum / self.mass_count).item())
            if mass_count > 0
            else float("nan")
        )
        metrics["latent_mass_conservation_max_abs_error"] = (
            float(self.mass_error_max.item()) if mass_count > 0 else float("nan")
        )
        metrics["latent_death_transition_mae"] = (
            float(
                (
                    self.death_transition_error_sum / self.death_transition_count
                ).item()
            )
            if float(self.death_transition_count.item()) > 0
            else float("nan")
        )
        metrics["latent_death_monotonic_violation_fraction"] = self._fraction(
            self.death_monotonic_violation_count,
            self.death_transition_count,
        )
        for name in ("hospitalization", "death"):
            metrics[f"latent_{name}_flow_negative_fraction"] = self._fraction(
                self.flow_negative_counts[name],
                self.flow_finite_counts[name],
            )
        metrics["latent_physics_residual_mae"] = (
            float(
                (
                    self.physics_residual_abs_sum / self.physics_residual_count
                ).item()
            )
            if float(self.physics_residual_count.item()) > 0
            else float("nan")
        )
        return metrics
