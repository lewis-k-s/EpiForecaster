from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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


def _masked_average_tie_ranks(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return masked ``(..., H)`` average ranks via an O(H²) pair grid."""
    target_valid = valid.unsqueeze(-1)
    comparator_valid = valid.unsqueeze(-2)
    active_pairs = target_valid & comparator_valid
    target = values.unsqueeze(-1)
    comparator = values.unsqueeze(-2)
    less_count = (active_pairs & (comparator < target)).sum(dim=-1)
    tie_count = (active_pairs & (comparator == target)).sum(dim=-1)
    return less_count.float() + (tie_count.float() - 1.0) / 2.0


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
        self.observed_count = torch.tensor(0, device=device, dtype=torch.long)
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
        effective_weights = torch.nan_to_num(
            weights.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0) * finite.to(dtype=torch.float32)
        full_clean = _clean(full)
        mechanistic_clean = _clean(mechanistic)
        residual_clean = _clean(residual)
        targets_clean = _clean(targets)

        self.weight_sum += effective_weights.sum()
        self.observed_count += (effective_weights > 0).sum()
        self.abs_mechanistic_sum += (mechanistic_clean.abs() * effective_weights).sum()
        self.abs_residual_sum += (residual_clean.abs() * effective_weights).sum()
        mechanistic_error = mechanistic_clean - targets_clean
        self.mechanistic_error_abs_sum += (
            mechanistic_error.abs() * effective_weights
        ).sum()
        self.mechanistic_error_sq_sum += (
            mechanistic_error.square() * effective_weights
        ).sum()

        active = effective_weights > 0
        reconstruction_error = (full_clean - mechanistic_clean - residual_clean).abs()
        batch_reconstruction_max = torch.where(
            active,
            reconstruction_error,
            self.reconstruction_max_abs_error,
        ).max()
        self.reconstruction_max_abs_error = torch.maximum(
            self.reconstruction_max_abs_error,
            batch_reconstruction_max,
        )

        for name, value in (
            ("full", full_clean),
            ("mechanistic", mechanistic_clean),
            ("residual", residual_clean),
        ):
            self.weighted_sums[name] += (value * effective_weights).sum()
            self.weighted_sq_sums[name] += (value.square() * effective_weights).sum()

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
            observed_count=int(self.observed_count.item()),
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
    prediction_std: float
    target_std: float
    prediction_target_std_ratio: float
    mean_window_spearman: float
    window_spearman_eligible_count: int
    horizon_mae: tuple[float, ...]
    horizon_rmse: tuple[float, ...]
    nonnegative_violation_fraction: float


@dataclass(frozen=True)
class WindowSpearmanMetricResult:
    """Coverage and mean rank agreement for a family of forecast windows."""

    mean: float
    total_count: int
    candidate_count: int
    eligible_count: int
    candidate_fraction: float
    eligible_fraction: float
    eligible_given_candidate_fraction: float


class _TorchWindowSpearmanAccumulator:
    """Accumulate masked, tie-aware Spearman values."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        # Per-batch (correlations, eligible) rows, summed once in finalize.
        self.correlation_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.count = torch.tensor(0.0, device=device)
        self.total_window_count = torch.tensor(0, device=device, dtype=torch.long)
        self.candidate_window_count = torch.tensor(0, device=device, dtype=torch.long)

    def update(
        self,
        *,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if prediction.shape != reference.shape or prediction.shape != valid.shape:
            raise ValueError(
                "Prediction, reference, and validity mask must have identical shapes."
            )
        # Flatten all leading dimensions; rank each horizon row independently.
        prediction_rows = prediction.float().reshape(-1, prediction.shape[-1])
        reference_rows = reference.float().reshape(-1, reference.shape[-1])
        valid_rows = valid.reshape(-1, valid.shape[-1]).bool()
        valid_count = valid_rows.sum(dim=-1)
        candidate = valid_count >= 2

        row_count, horizon = prediction_rows.shape
        # Sort invalid sentinel indices last, then gather compact valid rows.
        column_index = torch.arange(
            horizon,
            device=prediction_rows.device,
        ).expand(row_count, horizon)
        compacted_index = (
            torch.where(valid_rows, column_index, horizon)
            .sort(dim=-1)
            .values.clamp_max(horizon - 1)
        )
        compacted_valid = column_index < valid_count.unsqueeze(-1)
        prediction_compact = prediction_rows.gather(dim=-1, index=compacted_index)
        reference_compact = reference_rows.gather(dim=-1, index=compacted_index)
        prediction_ranks = _masked_average_tie_ranks(
            prediction_compact,
            compacted_valid,
        )
        reference_ranks = _masked_average_tie_ranks(
            reference_compact,
            compacted_valid,
        )

        correlations = torch.zeros(
            row_count,
            device=prediction_rows.device,
            dtype=prediction_rows.dtype,
        )
        eligible = torch.zeros(
            row_count,
            device=prediction_rows.device,
            dtype=torch.bool,
        )
        window_sizes = [
            int(window_size)
            for window_size in valid_count.unique().detach().cpu().tolist()
            if int(window_size) >= 2
        ]
        for window_size in window_sizes:
            prediction_window = prediction_ranks[:, :window_size]
            reference_window = reference_ranks[:, :window_size]
            prediction_centered = prediction_window - prediction_window.mean(
                dim=-1, keepdim=True
            )
            reference_centered = reference_window - reference_window.mean(
                dim=-1, keepdim=True
            )
            denominator = torch.sqrt(
                prediction_centered.square().sum(dim=-1)
                * reference_centered.square().sum(dim=-1)
            )
            window_eligible = (valid_count == window_size) & (denominator > _EPSILON)
            window_correlation = (prediction_centered * reference_centered).sum(
                dim=-1
            ) / denominator.clamp_min(_EPSILON)
            correlations = torch.where(
                window_eligible,
                window_correlation,
                correlations,
            )
            eligible |= window_eligible

        self.total_window_count += prediction_rows.shape[0]
        self.candidate_window_count += candidate.sum()
        self.count += eligible.sum()
        # Keep rows on the device; selecting them here would sync every batch.
        self.correlation_batches.append((correlations, eligible))

    def correlation_sum(self) -> torch.Tensor:
        """Return the float32 sum of all eligible correlations in update order."""
        if not self.correlation_batches:
            return torch.tensor(0.0, device=self.device)
        correlations = torch.cat([batch for batch, _ in self.correlation_batches])
        eligible = torch.cat([rows for _, rows in self.correlation_batches])
        # Preserve the legacy row-wise float32 addition order. Ineligible rows
        # added zero, so only eligible rows change the sum. NumPy accumulate adds
        # strictly in sequence, unlike a device reduction or a CPU cumsum.
        running = np.concatenate(
            [np.zeros(1, dtype=np.float32), correlations[eligible].cpu().numpy()]
        )
        total = np.add.accumulate(running, dtype=np.float32)[-1]
        return torch.tensor(float(total), device=self.device)

    def finalize(self) -> WindowSpearmanMetricResult:
        total_count = int(self.total_window_count.item())
        candidate_count = int(self.candidate_window_count.item())
        eligible_count = int(self.count.item())
        return WindowSpearmanMetricResult(
            mean=(
                float((self.correlation_sum() / self.count).item())
                if eligible_count > 0
                else float("nan")
            ),
            total_count=total_count,
            candidate_count=candidate_count,
            eligible_count=eligible_count,
            candidate_fraction=(
                candidate_count / total_count if total_count > 0 else 0.0
            ),
            eligible_fraction=eligible_count / total_count if total_count > 0 else 0.0,
            eligible_given_candidate_fraction=(
                eligible_count / candidate_count if candidate_count > 0 else 0.0
            ),
        )


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
        self.horizon_abs_error_sum: torch.Tensor | None = None
        self.horizon_sq_error_sum: torch.Tensor | None = None
        self.horizon_count: torch.Tensor | None = None
        self.window_spearman = _TorchWindowSpearmanAccumulator(device=device)

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
        error_full = _clean(pred_f32) - _clean(target_f32)
        error = error_full[finite]
        self.count += count
        self.abs_error_sum += error.abs().sum()
        self.sq_error_sum += error.square().sum()
        self.pred_sum += pred.sum()
        self.target_sum += truth.sum()
        self.pred_sq_sum += pred.square().sum()
        self.target_sq_sum += truth.square().sum()
        self.cross_sum += (pred * truth).sum()
        reduce_dims = tuple(range(finite.ndim - 1))
        horizon_count = finite.float().sum(dim=reduce_dims)
        horizon_abs_error = torch.where(finite, error_full.abs(), 0.0).sum(
            dim=reduce_dims
        )
        horizon_sq_error = torch.where(finite, error_full.square(), 0.0).sum(
            dim=reduce_dims
        )
        if self.horizon_count is None:
            self.horizon_count = torch.zeros_like(horizon_count)
            self.horizon_abs_error_sum = torch.zeros_like(horizon_abs_error)
            self.horizon_sq_error_sum = torch.zeros_like(horizon_sq_error)
        if self.horizon_count.shape != horizon_count.shape:
            raise ValueError(
                "Latent metric horizon changed between evaluation batches."
            )
        self.horizon_count += horizon_count
        assert self.horizon_abs_error_sum is not None
        assert self.horizon_sq_error_sum is not None
        self.horizon_abs_error_sum += horizon_abs_error
        self.horizon_sq_error_sum += horizon_sq_error

        self.window_spearman.update(
            prediction=pred_f32,
            reference=target_f32,
            valid=finite,
        )

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
                prediction_std=float("nan"),
                target_std=float("nan"),
                prediction_target_std_ratio=float("nan"),
                mean_window_spearman=float("nan"),
                window_spearman_eligible_count=0,
                horizon_mae=(),
                horizon_rmse=(),
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
        prediction_std = float(torch.sqrt(pred_variance).item())
        target_std = float(torch.sqrt(target_variance).item())
        assert self.horizon_count is not None
        assert self.horizon_abs_error_sum is not None
        assert self.horizon_sq_error_sum is not None
        valid_horizon = self.horizon_count > 0
        horizon_mae_tensor = torch.where(
            valid_horizon,
            self.horizon_abs_error_sum / self.horizon_count.clamp_min(1),
            torch.full_like(self.horizon_count, float("nan")),
        )
        horizon_rmse_tensor = torch.where(
            valid_horizon,
            torch.sqrt(self.horizon_sq_error_sum / self.horizon_count.clamp_min(1)),
            torch.full_like(self.horizon_count, float("nan")),
        )
        window_spearman = self.window_spearman.finalize()
        return LatentMetricResult(
            count=count,
            mae=mae,
            rmse=float(torch.sqrt(self.sq_error_sum / self.count).item()),
            pearson=pearson,
            mae_over_target_std=_safe_ratio(mae, target_std),
            prediction_std=prediction_std,
            target_std=target_std,
            prediction_target_std_ratio=_safe_ratio(prediction_std, target_std),
            mean_window_spearman=window_spearman.mean,
            window_spearman_eligible_count=window_spearman.eligible_count,
            horizon_mae=tuple(float(value) for value in horizon_mae_tensor.tolist()),
            horizon_rmse=tuple(float(value) for value in horizon_rmse_tensor.tolist()),
            nonnegative_violation_fraction=negative_fraction,
        )


@dataclass(frozen=True)
class ObservationLatentMetricResult:
    observed_count: int
    full_window_spearman: WindowSpearmanMetricResult
    mechanistic_only_window_spearman: WindowSpearmanMetricResult


class TorchObservationLatentCoherenceAccumulator:
    """Measure full and mechanistic-only forecast movement against latent sources."""

    def __init__(self, *, device: torch.device) -> None:
        self.observed_count = torch.tensor(0, device=device, dtype=torch.long)
        self.full_window_spearman = _TorchWindowSpearmanAccumulator(device=device)
        self.mechanistic_only_window_spearman = _TorchWindowSpearmanAccumulator(
            device=device
        )

    def update(
        self,
        *,
        full_prediction: torch.Tensor,
        relevant_latent: torch.Tensor | None,
        weights: torch.Tensor,
        mechanistic_projection: torch.Tensor | None = None,
    ) -> None:
        if relevant_latent is None:
            return
        if (
            full_prediction.shape != relevant_latent.shape
            or full_prediction.shape != weights.shape
        ):
            raise ValueError(
                "Full prediction, relevant latent, and weights must have identical "
                "shapes."
            )
        full_prediction_f32 = full_prediction.float()
        latent_f32 = relevant_latent.float()
        weights_f32 = weights.float()
        observed = (
            torch.isfinite(latent_f32) & torch.isfinite(weights_f32) & (weights_f32 > 0)
        )
        full_valid = observed & torch.isfinite(full_prediction_f32)
        self.observed_count += observed.sum()
        self.full_window_spearman.update(
            prediction=full_prediction_f32,
            reference=latent_f32,
            valid=full_valid,
        )
        if mechanistic_projection is not None:
            if mechanistic_projection.shape != full_prediction.shape:
                raise ValueError(
                    "Mechanistic projection and full prediction must have identical "
                    "shapes."
                )
            mechanistic_f32 = mechanistic_projection.float()
            mechanistic_valid = observed & torch.isfinite(mechanistic_f32)
            self.mechanistic_only_window_spearman.update(
                prediction=mechanistic_f32,
                reference=latent_f32,
                valid=mechanistic_valid,
            )

    def finalize(self) -> ObservationLatentMetricResult:
        return ObservationLatentMetricResult(
            observed_count=int(self.observed_count.item()),
            full_window_spearman=self.full_window_spearman.finalize(),
            mechanistic_only_window_spearman=(
                self.mechanistic_only_window_spearman.finalize()
            ),
        )


class TorchLatentCoherenceAccumulator:
    """Accumulate latent truth agreement and structural SIRHD diagnostics.

    The model's five predicted S/I/H/R/D states form its internal population
    partition. Exported supervision may be only a subset of a richer upstream
    compartment model, so model mass and exported-target mass are reported as
    distinct quantities.
    """

    _STATE_KEYS = {
        "s": ("S_trajectory", "S_target"),
        "i": ("I_trajectory", "I_target"),
        "h": ("H_trajectory", "H_target"),
        "r": ("R_trajectory", "R_target"),
        "d": ("D_trajectory", "D_target"),
    }
    _LATENT_KEYS = {
        **_STATE_KEYS,
        "death_flow": ("death_flow", "death_flow_target"),
    }

    def __init__(self, *, device: torch.device) -> None:
        self.latents = {
            name: _TorchLatentPairAccumulator(device=device)
            for name in self._LATENT_KEYS
        }
        self.model_mass_error_sum = torch.tensor(0.0, device=device)
        self.model_mass_error_max = torch.tensor(0.0, device=device)
        self.model_mass_count = torch.tensor(0.0, device=device)
        self.target_exported_mass_sum = torch.tensor(0.0, device=device)
        self.target_exported_mass_min = torch.tensor(float("inf"), device=device)
        self.target_exported_mass_max = torch.tensor(float("-inf"), device=device)
        self.target_exported_mass_gap_sum = torch.tensor(0.0, device=device)
        self.target_exported_mass_count = torch.tensor(0.0, device=device)
        self.model_target_exported_mass_error_sum = torch.tensor(0.0, device=device)
        self.model_target_exported_mass_count = torch.tensor(0.0, device=device)
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
            for prediction_key, _target_key in self._STATE_KEYS.values()
        ):
            return
        trajectories = [
            model_outputs[prediction_key].float()
            for prediction_key, _target_key in self._STATE_KEYS.values()
        ]
        model_mass_finite = torch.stack(
            [torch.isfinite(trajectory) for trajectory in trajectories]
        ).all(dim=0)
        model_mass = torch.stack(
            [_clean(trajectory) for trajectory in trajectories]
        ).sum(dim=0)
        if model_mass_finite.any():
            model_mass_error = (model_mass[model_mass_finite] - 1.0).abs()
            self.model_mass_error_sum += model_mass_error.sum()
            self.model_mass_error_max = torch.maximum(
                self.model_mass_error_max,
                model_mass_error.max(),
            )
            self.model_mass_count += model_mass_finite.sum()

        target_trajectories = [
            targets.get(target_key)
            for _prediction_key, target_key in self._STATE_KEYS.values()
        ]
        if all(isinstance(target, torch.Tensor) for target in target_trajectories):
            target_tensors = [
                target.float()
                for target in target_trajectories
                if isinstance(target, torch.Tensor)
            ]
            target_mass_finite = torch.stack(
                [torch.isfinite(target) for target in target_tensors]
            ).all(dim=0)
            target_mass = torch.stack(
                [_clean(target) for target in target_tensors]
            ).sum(dim=0)
            if target_mass_finite.any():
                finite_target_mass = target_mass[target_mass_finite]
                self.target_exported_mass_sum += finite_target_mass.sum()
                self.target_exported_mass_min = torch.minimum(
                    self.target_exported_mass_min,
                    finite_target_mass.min(),
                )
                self.target_exported_mass_max = torch.maximum(
                    self.target_exported_mass_max,
                    finite_target_mass.max(),
                )
                self.target_exported_mass_gap_sum += (
                    (finite_target_mass - 1.0).abs().sum()
                )
                self.target_exported_mass_count += target_mass_finite.sum()

            model_target_finite = model_mass_finite & target_mass_finite
            if model_target_finite.any():
                model_target_error = (
                    model_mass[model_target_finite] - target_mass[model_target_finite]
                ).abs()
                self.model_target_exported_mass_error_sum += model_target_error.sum()
                self.model_target_exported_mass_count += model_target_finite.sum()

        death_trajectory = model_outputs["D_trajectory"].float()
        death_flow_tensor = model_outputs.get("death_flow")
        if death_flow_tensor is None:
            return
        death_flow = death_flow_tensor.float()
        death_delta = death_trajectory[:, 1:] - death_trajectory[:, :-1]
        transition_finite = torch.isfinite(death_delta) & torch.isfinite(death_flow)
        if transition_finite.any():
            transition_error = (_clean(death_delta) - _clean(death_flow))[
                transition_finite
            ].abs()
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
            self.physics_residual_abs_sum += (
                physics_residual[physics_finite].abs().sum()
            )
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
                    f"latent_prediction_std_{name}": result.prediction_std,
                    f"latent_target_std_{name}": result.target_std,
                    f"latent_prediction_target_std_ratio_{name}": (
                        result.prediction_target_std_ratio
                    ),
                    f"latent_mean_window_spearman_{name}": (
                        result.mean_window_spearman
                    ),
                    f"latent_window_spearman_eligible_count_{name}": (
                        result.window_spearman_eligible_count
                    ),
                    f"latent_nonnegative_violation_fraction_{name}": (
                        result.nonnegative_violation_fraction
                    ),
                }
            )
            for horizon_index, (mae, rmse) in enumerate(
                zip(result.horizon_mae, result.horizon_rmse, strict=True),
                start=1 if name == "death_flow" else 0,
            ):
                metrics[f"latent_mae_{name}_horizon_{horizon_index:02d}_fraction"] = mae
                metrics[f"latent_rmse_{name}_horizon_{horizon_index:02d}_fraction"] = (
                    rmse
                )

        model_mass_count = float(self.model_mass_count.item())
        metrics["latent_model_sirhd_mass_error_from_one_mae"] = (
            float((self.model_mass_error_sum / self.model_mass_count).item())
            if model_mass_count > 0
            else float("nan")
        )
        metrics["latent_model_sirhd_mass_error_from_one_max"] = (
            float(self.model_mass_error_max.item())
            if model_mass_count > 0
            else float("nan")
        )
        target_mass_count = float(self.target_exported_mass_count.item())
        metrics["latent_target_exported_sirhd_mass_mean"] = (
            float(
                (self.target_exported_mass_sum / self.target_exported_mass_count).item()
            )
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_min"] = (
            float(self.target_exported_mass_min.item())
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_max"] = (
            float(self.target_exported_mass_max.item())
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_gap_from_one_mae"] = (
            float(
                (
                    self.target_exported_mass_gap_sum / self.target_exported_mass_count
                ).item()
            )
            if target_mass_count > 0
            else float("nan")
        )
        model_target_mass_count = float(self.model_target_exported_mass_count.item())
        metrics["latent_model_vs_target_exported_sirhd_mass_mae"] = (
            float(
                (
                    self.model_target_exported_mass_error_sum
                    / self.model_target_exported_mass_count
                ).item()
            )
            if model_target_mass_count > 0
            else float("nan")
        )
        metrics["latent_death_transition_mae"] = (
            float(
                (self.death_transition_error_sum / self.death_transition_count).item()
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
            float((self.physics_residual_abs_sum / self.physics_residual_count).item())
            if float(self.physics_residual_count.item()) > 0
            else float("nan")
        )
        return metrics


class TorchLatentProtocolAccumulator:
    """Accumulate canonical metrics for one latent diagnostic protocol."""

    _KEYS = {
        "s": ("S_trajectory", "S_target"),
        "i": ("I_trajectory", "I_target"),
        "h": ("H_trajectory", "H_target"),
        "r": ("R_trajectory", "R_target"),
        "death_flow": ("death_flow", "death_flow_target"),
    }

    def __init__(self, *, protocol: str, device: torch.device) -> None:
        self.protocol = protocol
        self.latents = {
            task: _TorchLatentPairAccumulator(device=device) for task in self._KEYS
        }

    def update(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
    ) -> None:
        for task, (prediction_key, target_key) in self._KEYS.items():
            prediction = predictions.get(prediction_key)
            if prediction is None:
                continue
            self.latents[task].update(
                prediction=prediction,
                target=targets.get(target_key),
            )

    def finalize(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        prefix = f"latent_diag_{self.protocol}"
        for task, accumulator in self.latents.items():
            result = accumulator.finalize()
            metrics.update(
                {
                    f"{prefix}_count_{task}": result.count,
                    f"{prefix}_mae_{task}_fraction": result.mae,
                    f"{prefix}_rmse_{task}_fraction": result.rmse,
                    f"{prefix}_pearson_{task}": result.pearson,
                    f"{prefix}_prediction_std_{task}": result.prediction_std,
                    f"{prefix}_target_std_{task}": result.target_std,
                    f"{prefix}_prediction_target_std_ratio_{task}": (
                        result.prediction_target_std_ratio
                    ),
                    f"{prefix}_mean_window_spearman_{task}": (
                        result.mean_window_spearman
                    ),
                    f"{prefix}_window_spearman_eligible_count_{task}": (
                        result.window_spearman_eligible_count
                    ),
                }
            )
            for horizon_index, (mae, rmse) in enumerate(
                zip(result.horizon_mae, result.horizon_rmse, strict=True),
                start=1,
            ):
                metrics[f"{prefix}_mae_{task}_horizon_{horizon_index:02d}"] = mae
                metrics[f"{prefix}_rmse_{task}_horizon_{horizon_index:02d}"] = rmse
        return metrics
