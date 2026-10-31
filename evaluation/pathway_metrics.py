from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from evaluation.metrics import (
    MetricSums,
    compiled_metric_step,
    distinct_metric_sums,
)

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


_PATHWAY_PARTS = ("full", "mechanistic", "residual")


def _pathway_metric_step(
    sums: MetricSums,
    full: torch.Tensor,
    mechanistic: torch.Tensor,
    residual: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> MetricSums:
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
    mechanistic_error = mechanistic_clean - targets_clean
    active = effective_weights > 0
    reconstruction_error = (full_clean - mechanistic_clean - residual_clean).abs()
    batch_reconstruction_max = torch.where(
        active,
        reconstruction_error,
        sums["reconstruction_max"],
    ).max()

    new_sums = {
        "weight": sums["weight"] + effective_weights.sum(),
        "observed": sums["observed"] + active.sum(),
        "abs_mechanistic": sums["abs_mechanistic"]
        + (mechanistic_clean.abs() * effective_weights).sum(),
        "abs_residual": sums["abs_residual"]
        + (residual_clean.abs() * effective_weights).sum(),
        "mechanistic_error_abs": sums["mechanistic_error_abs"]
        + (mechanistic_error.abs() * effective_weights).sum(),
        "mechanistic_error_sq": sums["mechanistic_error_sq"]
        + (mechanistic_error.square() * effective_weights).sum(),
        "reconstruction_max": torch.maximum(
            sums["reconstruction_max"], batch_reconstruction_max
        ),
    }
    for name, value in zip(
        _PATHWAY_PARTS, (full_clean, mechanistic_clean, residual_clean), strict=True
    ):
        new_sums[name] = sums[name] + (value * effective_weights).sum()
        new_sums[f"{name}_sq"] = (
            sums[f"{name}_sq"] + (value.square() * effective_weights).sum()
        )
    return new_sums


class TorchPathwayMetricAccumulator:
    """Accumulate observation-head pathway diagnostics on the evaluation device."""

    def __init__(self, *, device: torch.device, compile_step: bool = False) -> None:
        self.device = device
        zero = torch.tensor(0.0, device=device)
        self.sums: MetricSums = {
            "weight": zero,
            "observed": torch.tensor(0, device=device, dtype=torch.long),
            "abs_mechanistic": zero,
            "abs_residual": zero,
            "mechanistic_error_abs": zero,
            "mechanistic_error_sq": zero,
            "reconstruction_max": zero,
        }
        for name in _PATHWAY_PARTS:
            self.sums[name] = zero
            self.sums[f"{name}_sq"] = zero
        self.sums = distinct_metric_sums(self.sums)
        self._step = (
            compiled_metric_step(_pathway_metric_step)
            if compile_step
            else _pathway_metric_step
        )

    def update(
        self,
        *,
        full: torch.Tensor,
        mechanistic: torch.Tensor,
        residual: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self.sums = self._step(self.sums, full, mechanistic, residual, targets, weights)

    def finalize(self) -> PathwayMetricResult:
        effective_count = float(self.sums["weight"].item())
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
            (self.sums["abs_mechanistic"] / self.sums["weight"]).item()
        )
        mean_abs_residual = float(
            (self.sums["abs_residual"] / self.sums["weight"]).item()
        )
        variances: dict[str, float] = {}
        for name in _PATHWAY_PARTS:
            mean = self.sums[name] / self.sums["weight"]
            variance = self.sums[f"{name}_sq"] / self.sums["weight"] - mean.square()
            variances[name] = float(variance.clamp_min(0.0).item())

        return PathwayMetricResult(
            observed_count=int(self.sums["observed"].item()),
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
                (self.sums["mechanistic_error_abs"] / self.sums["weight"]).item()
            ),
            mechanistic_only_rmse=float(
                torch.sqrt(
                    self.sums["mechanistic_error_sq"] / self.sums["weight"]
                ).item()
            ),
            reconstruction_max_abs_error=float(self.sums["reconstruction_max"].item()),
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


def _window_spearman_step(
    counts: MetricSums,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[MetricSums, torch.Tensor, torch.Tensor]:
    """Score each window row; return (counts, correlations, eligible)."""
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

    # Center over each row's own valid prefix with masked sums, so rows of
    # every window size are scored at once without a host sync.
    valid_weight = compacted_valid.to(prediction_ranks.dtype)
    window_size = valid_count.clamp_min(1).unsqueeze(-1).to(valid_weight.dtype)
    prediction_centered = (
        prediction_ranks
        - (prediction_ranks * valid_weight).sum(dim=-1, keepdim=True) / window_size
    ) * valid_weight
    reference_centered = (
        reference_ranks
        - (reference_ranks * valid_weight).sum(dim=-1, keepdim=True) / window_size
    ) * valid_weight
    denominator = torch.sqrt(
        prediction_centered.square().sum(dim=-1)
        * reference_centered.square().sum(dim=-1)
    )
    eligible = candidate & (denominator > _EPSILON)
    correlations = torch.where(
        eligible,
        (prediction_centered * reference_centered).sum(dim=-1)
        / denominator.clamp_min(_EPSILON),
        0.0,
    ).to(prediction_rows.dtype)
    new_counts = {
        "total": counts["total"] + row_count,
        "candidate": counts["candidate"] + candidate.sum(),
        "eligible": counts["eligible"] + eligible.sum(),
    }
    return new_counts, correlations, eligible


class _TorchWindowSpearmanAccumulator:
    """Accumulate masked, tie-aware Spearman values."""

    def __init__(self, *, device: torch.device, compile_step: bool = False) -> None:
        self.device = device
        # Per-batch (correlations, eligible) rows, summed once in finalize.
        self.correlation_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.counts: MetricSums = {
            "total": torch.tensor(0, device=device, dtype=torch.long),
            "candidate": torch.tensor(0, device=device, dtype=torch.long),
            "eligible": torch.tensor(0.0, device=device),
        }
        self._step = (
            compiled_metric_step(_window_spearman_step)
            if compile_step
            else _window_spearman_step
        )

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
        self.record(*self._step(self.counts, prediction, reference, valid))

    def record(
        self,
        counts: MetricSums,
        correlations: torch.Tensor,
        eligible: torch.Tensor,
    ) -> None:
        """Store the result of a ``_window_spearman_step`` run by a caller."""
        self.counts = counts
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
        total_count = int(self.counts["total"].item())
        candidate_count = int(self.counts["candidate"].item())
        eligible_count = int(self.counts["eligible"].item())
        return WindowSpearmanMetricResult(
            mean=(
                float((self.correlation_sum() / self.counts["eligible"]).item())
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


def _latent_pair_step(
    sums: MetricSums,
    spearman_counts: MetricSums,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[MetricSums, MetricSums, torch.Tensor, torch.Tensor]:
    """Add one batch of finite prediction-target pairs to ``sums``."""
    pred_f32 = prediction.float()
    target_f32 = target.float()
    pred_finite = torch.isfinite(pred_f32)
    finite = pred_finite & torch.isfinite(target_f32)
    pred = _clean(pred_f32)
    truth = _clean(target_f32)
    error = pred - truth
    reduce_dims = tuple(range(finite.ndim - 1))
    new_sums = {
        "pred_finite": sums["pred_finite"] + pred_finite.sum(),
        "pred_negative": sums["pred_negative"] + (pred_finite & (pred_f32 < 0)).sum(),
        "count": sums["count"] + finite.sum(),
        "abs_error": sums["abs_error"] + torch.where(finite, error.abs(), 0.0).sum(),
        "sq_error": sums["sq_error"] + torch.where(finite, error.square(), 0.0).sum(),
        "pred": sums["pred"] + torch.where(finite, pred, 0.0).sum(),
        "target": sums["target"] + torch.where(finite, truth, 0.0).sum(),
        "pred_sq": sums["pred_sq"] + torch.where(finite, pred.square(), 0.0).sum(),
        "target_sq": sums["target_sq"] + torch.where(finite, truth.square(), 0.0).sum(),
        "cross": sums["cross"] + torch.where(finite, pred * truth, 0.0).sum(),
        "horizon_count": sums["horizon_count"] + finite.float().sum(dim=reduce_dims),
        "horizon_abs_error": sums["horizon_abs_error"]
        + torch.where(finite, error.abs(), 0.0).sum(dim=reduce_dims),
        "horizon_sq_error": sums["horizon_sq_error"]
        + torch.where(finite, error.square(), 0.0).sum(dim=reduce_dims),
    }
    new_counts, correlations, eligible = _window_spearman_step(
        spearman_counts, pred_f32, target_f32, finite
    )
    # A batch without finite pairs does not count toward the window total.
    new_counts["total"] = torch.where(
        finite.any(), new_counts["total"], spearman_counts["total"]
    )
    return new_sums, new_counts, correlations, eligible


def _prediction_sign_step(sums: MetricSums, prediction: torch.Tensor) -> MetricSums:
    pred_f32 = prediction.float()
    pred_finite = torch.isfinite(pred_f32)
    return {
        **sums,
        "pred_finite": sums["pred_finite"] + pred_finite.sum(),
        "pred_negative": sums["pred_negative"] + (pred_finite & (pred_f32 < 0)).sum(),
    }


class _TorchLatentPairAccumulator:
    def __init__(self, *, device: torch.device, compile_step: bool = False) -> None:
        self.device = device
        zero = torch.tensor(0.0, device=device)
        self.sums: MetricSums = {
            name: zero
            for name in (
                "pred_finite",
                "pred_negative",
                "count",
                "abs_error",
                "sq_error",
                "pred",
                "target",
                "pred_sq",
                "target_sq",
                "cross",
            )
        }
        self.sums = distinct_metric_sums(self.sums)
        self.horizon: int | None = None
        self.window_spearman = _TorchWindowSpearmanAccumulator(device=device)
        self._pair_step = (
            compiled_metric_step(_latent_pair_step)
            if compile_step
            else _latent_pair_step
        )
        self._sign_step = (
            compiled_metric_step(_prediction_sign_step)
            if compile_step
            else _prediction_sign_step
        )

    def update(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor | None,
    ) -> None:
        if target is None:
            self.sums = self._sign_step(self.sums, prediction)
            return

        horizon = prediction.shape[-1]
        if self.horizon is None:
            self.horizon = horizon
            zeros = torch.zeros(horizon, device=self.device)
            for name in ("horizon_count", "horizon_abs_error", "horizon_sq_error"):
                self.sums[name] = zeros
            self.sums = distinct_metric_sums(self.sums)
        if self.horizon != horizon:
            raise ValueError(
                "Latent metric horizon changed between evaluation batches."
            )
        self.sums, counts, correlations, eligible = self._pair_step(
            self.sums, self.window_spearman.counts, prediction, target
        )
        self.window_spearman.record(counts, correlations, eligible)

    def finalize(self) -> LatentMetricResult:
        pred_finite_count = float(self.sums["pred_finite"].item())
        negative_fraction = (
            float((self.sums["pred_negative"] / self.sums["pred_finite"]).item())
            if pred_finite_count > 0
            else float("nan")
        )
        count = int(self.sums["count"].item())
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

        sums = self.sums
        pred_mean = sums["pred"] / sums["count"]
        target_mean = sums["target"] / sums["count"]
        pred_variance = (
            sums["pred_sq"] / sums["count"] - pred_mean.square()
        ).clamp_min(0.0)
        target_variance = (
            sums["target_sq"] / sums["count"] - target_mean.square()
        ).clamp_min(0.0)
        covariance = sums["cross"] / sums["count"] - pred_mean * target_mean
        denominator = torch.sqrt(pred_variance * target_variance)
        pearson = (
            float((covariance / denominator).item())
            if float(denominator.item()) > _EPSILON
            else float("nan")
        )
        mae = float((sums["abs_error"] / sums["count"]).item())
        prediction_std = float(torch.sqrt(pred_variance).item())
        target_std = float(torch.sqrt(target_variance).item())
        horizon_count = sums["horizon_count"]
        valid_horizon = horizon_count > 0
        horizon_mae_tensor = torch.where(
            valid_horizon,
            sums["horizon_abs_error"] / horizon_count.clamp_min(1),
            torch.full_like(horizon_count, float("nan")),
        )
        horizon_rmse_tensor = torch.where(
            valid_horizon,
            torch.sqrt(sums["horizon_sq_error"] / horizon_count.clamp_min(1)),
            torch.full_like(horizon_count, float("nan")),
        )
        window_spearman = self.window_spearman.finalize()
        return LatentMetricResult(
            count=count,
            mae=mae,
            rmse=float(torch.sqrt(sums["sq_error"] / sums["count"]).item()),
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


def _observation_latent_step(
    observed_count: torch.Tensor,
    full_counts: MetricSums,
    mechanistic_counts: MetricSums,
    full_prediction: torch.Tensor,
    relevant_latent: torch.Tensor,
    weights: torch.Tensor,
    mechanistic_projection: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    tuple[MetricSums, torch.Tensor, torch.Tensor],
    tuple[MetricSums, torch.Tensor, torch.Tensor] | None,
]:
    full_prediction_f32 = full_prediction.float()
    latent_f32 = relevant_latent.float()
    weights_f32 = weights.float()
    observed = (
        torch.isfinite(latent_f32) & torch.isfinite(weights_f32) & (weights_f32 > 0)
    )
    full_valid = observed & torch.isfinite(full_prediction_f32)
    full = _window_spearman_step(
        full_counts, full_prediction_f32, latent_f32, full_valid
    )
    mechanistic = None
    if mechanistic_projection is not None:
        mechanistic_f32 = mechanistic_projection.float()
        mechanistic_valid = observed & torch.isfinite(mechanistic_f32)
        mechanistic = _window_spearman_step(
            mechanistic_counts, mechanistic_f32, latent_f32, mechanistic_valid
        )
    return observed_count + observed.sum(), full, mechanistic


class TorchObservationLatentCoherenceAccumulator:
    """Measure full and mechanistic-only forecast movement against latent sources."""

    def __init__(self, *, device: torch.device, compile_step: bool = False) -> None:
        self.observed_count = torch.tensor(0, device=device, dtype=torch.long)
        self.full_window_spearman = _TorchWindowSpearmanAccumulator(device=device)
        self.mechanistic_only_window_spearman = _TorchWindowSpearmanAccumulator(
            device=device
        )
        self._step = (
            compiled_metric_step(_observation_latent_step)
            if compile_step
            else _observation_latent_step
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
        if (
            mechanistic_projection is not None
            and mechanistic_projection.shape != full_prediction.shape
        ):
            raise ValueError(
                "Mechanistic projection and full prediction must have identical shapes."
            )
        self.observed_count, full, mechanistic = self._step(
            self.observed_count,
            self.full_window_spearman.counts,
            self.mechanistic_only_window_spearman.counts,
            full_prediction,
            relevant_latent,
            weights,
            mechanistic_projection,
        )
        self.full_window_spearman.record(*full)
        if mechanistic is not None:
            self.mechanistic_only_window_spearman.record(*mechanistic)

    def finalize(self) -> ObservationLatentMetricResult:
        return ObservationLatentMetricResult(
            observed_count=int(self.observed_count.item()),
            full_window_spearman=self.full_window_spearman.finalize(),
            mechanistic_only_window_spearman=(
                self.mechanistic_only_window_spearman.finalize()
            ),
        )


def _latent_coherence_step(
    sums: MetricSums,
    trajectories: tuple[torch.Tensor, ...],
    target_trajectories: tuple[torch.Tensor, ...] | None,
    death_flow: torch.Tensor | None,
    hospitalization_flow: torch.Tensor | None,
    physics_residual: torch.Tensor | None,
) -> MetricSums:
    """Add one batch of structural SIRHD diagnostics to ``sums``.

    Masked sums replace boolean selection, so the step has no host sync.
    """
    sums = dict(sums)
    states = [trajectory.float() for trajectory in trajectories]
    model_mass_finite = torch.stack([torch.isfinite(state) for state in states]).all(
        dim=0
    )
    model_mass = torch.stack([_clean(state) for state in states]).sum(dim=0)
    # Errors are nonnegative, so a zero fill keeps the running max unchanged.
    model_mass_error = torch.where(model_mass_finite, (model_mass - 1.0).abs(), 0.0)
    sums["model_mass_error"] = sums["model_mass_error"] + model_mass_error.sum()
    sums["model_mass_error_max"] = torch.maximum(
        sums["model_mass_error_max"], model_mass_error.max()
    )
    sums["model_mass_count"] = sums["model_mass_count"] + model_mass_finite.sum()

    if target_trajectories is not None:
        target_states = [target.float() for target in target_trajectories]
        target_mass_finite = torch.stack(
            [torch.isfinite(target) for target in target_states]
        ).all(dim=0)
        target_mass = torch.stack([_clean(target) for target in target_states]).sum(
            dim=0
        )
        sums["target_mass"] = (
            sums["target_mass"]
            + torch.where(target_mass_finite, target_mass, 0.0).sum()
        )
        sums["target_mass_min"] = torch.minimum(
            sums["target_mass_min"],
            torch.where(target_mass_finite, target_mass, float("inf")).min(),
        )
        sums["target_mass_max"] = torch.maximum(
            sums["target_mass_max"],
            torch.where(target_mass_finite, target_mass, float("-inf")).max(),
        )
        sums["target_mass_gap"] = (
            sums["target_mass_gap"]
            + torch.where(target_mass_finite, (target_mass - 1.0).abs(), 0.0).sum()
        )
        sums["target_mass_count"] = sums["target_mass_count"] + target_mass_finite.sum()
        model_target_finite = model_mass_finite & target_mass_finite
        sums["model_target_mass_error"] = (
            sums["model_target_mass_error"]
            + torch.where(
                model_target_finite, (model_mass - target_mass).abs(), 0.0
            ).sum()
        )
        sums["model_target_mass_count"] = (
            sums["model_target_mass_count"] + model_target_finite.sum()
        )

    if death_flow is None:
        return sums
    death_trajectory = states[-1]
    death_flow_f32 = death_flow.float()
    death_delta = death_trajectory[:, 1:] - death_trajectory[:, :-1]
    transition_finite = torch.isfinite(death_delta) & torch.isfinite(death_flow_f32)
    sums["death_transition_error"] = (
        sums["death_transition_error"]
        + torch.where(
            transition_finite, (_clean(death_delta) - _clean(death_flow_f32)).abs(), 0.0
        ).sum()
    )
    sums["death_transition_count"] = (
        sums["death_transition_count"] + transition_finite.sum()
    )
    sums["death_monotonic_violation"] = (
        sums["death_monotonic_violation"]
        + (transition_finite & (death_delta < 0)).sum()
    )

    for name, flow_tensor in (
        ("hospitalization", hospitalization_flow),
        ("death", death_flow),
    ):
        if flow_tensor is None:
            continue
        flow = flow_tensor.float()
        finite = torch.isfinite(flow)
        sums[f"{name}_flow_finite"] = sums[f"{name}_flow_finite"] + finite.sum()
        sums[f"{name}_flow_negative"] = (
            sums[f"{name}_flow_negative"] + (finite & (flow < 0)).sum()
        )

    if physics_residual is None:
        return sums
    residual = physics_residual.float()
    residual_finite = torch.isfinite(residual)
    sums["physics_residual_abs"] = (
        sums["physics_residual_abs"]
        + torch.where(residual_finite, residual.abs(), 0.0).sum()
    )
    sums["physics_residual_count"] = (
        sums["physics_residual_count"] + residual_finite.sum()
    )
    return sums


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

    def __init__(self, *, device: torch.device, compile_step: bool = False) -> None:
        self.latents = {
            name: _TorchLatentPairAccumulator(device=device, compile_step=compile_step)
            for name in self._LATENT_KEYS
        }
        zero = torch.tensor(0.0, device=device)
        self.sums: MetricSums = {
            name: zero
            for name in (
                "model_mass_error",
                "model_mass_error_max",
                "model_mass_count",
                "target_mass",
                "target_mass_gap",
                "target_mass_count",
                "model_target_mass_error",
                "model_target_mass_count",
                "death_transition_error",
                "death_transition_count",
                "death_monotonic_violation",
                "hospitalization_flow_finite",
                "hospitalization_flow_negative",
                "death_flow_finite",
                "death_flow_negative",
                "physics_residual_abs",
                "physics_residual_count",
            )
        }
        self.sums["target_mass_min"] = torch.tensor(float("inf"), device=device)
        self.sums["target_mass_max"] = torch.tensor(float("-inf"), device=device)
        self.sums = distinct_metric_sums(self.sums)
        self._step = (
            compiled_metric_step(_latent_coherence_step)
            if compile_step
            else _latent_coherence_step
        )

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
        target_trajectories = [
            targets.get(target_key)
            for _prediction_key, target_key in self._STATE_KEYS.values()
        ]
        self.sums = self._step(
            self.sums,
            tuple(
                model_outputs[prediction_key]
                for prediction_key, _target_key in self._STATE_KEYS.values()
            ),
            tuple(target_trajectories)
            if all(isinstance(target, torch.Tensor) for target in target_trajectories)
            else None,
            model_outputs.get("death_flow"),
            model_outputs.get("hospitalization_flow"),
            model_outputs.get("physics_residual"),
        )

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

        model_mass_count = float(self.sums["model_mass_count"].item())
        metrics["latent_model_sirhd_mass_error_from_one_mae"] = (
            float(
                (self.sums["model_mass_error"] / self.sums["model_mass_count"]).item()
            )
            if model_mass_count > 0
            else float("nan")
        )
        metrics["latent_model_sirhd_mass_error_from_one_max"] = (
            float(self.sums["model_mass_error_max"].item())
            if model_mass_count > 0
            else float("nan")
        )
        target_mass_count = float(self.sums["target_mass_count"].item())
        metrics["latent_target_exported_sirhd_mass_mean"] = (
            float((self.sums["target_mass"] / self.sums["target_mass_count"]).item())
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_min"] = (
            float(self.sums["target_mass_min"].item())
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_max"] = (
            float(self.sums["target_mass_max"].item())
            if target_mass_count > 0
            else float("nan")
        )
        metrics["latent_target_exported_sirhd_mass_gap_from_one_mae"] = (
            float(
                (self.sums["target_mass_gap"] / self.sums["target_mass_count"]).item()
            )
            if target_mass_count > 0
            else float("nan")
        )
        model_target_mass_count = float(self.sums["model_target_mass_count"].item())
        metrics["latent_model_vs_target_exported_sirhd_mass_mae"] = (
            float(
                (
                    self.sums["model_target_mass_error"]
                    / self.sums["model_target_mass_count"]
                ).item()
            )
            if model_target_mass_count > 0
            else float("nan")
        )
        metrics["latent_death_transition_mae"] = (
            float(
                (
                    self.sums["death_transition_error"]
                    / self.sums["death_transition_count"]
                ).item()
            )
            if float(self.sums["death_transition_count"].item()) > 0
            else float("nan")
        )
        metrics["latent_death_monotonic_violation_fraction"] = self._fraction(
            self.sums["death_monotonic_violation"],
            self.sums["death_transition_count"],
        )
        for name in ("hospitalization", "death"):
            metrics[f"latent_{name}_flow_negative_fraction"] = self._fraction(
                self.sums[f"{name}_flow_negative"],
                self.sums[f"{name}_flow_finite"],
            )
        metrics["latent_physics_residual_mae"] = (
            float(
                (
                    self.sums["physics_residual_abs"]
                    / self.sums["physics_residual_count"]
                ).item()
            )
            if float(self.sums["physics_residual_count"].item()) > 0
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

    def __init__(
        self, *, protocol: str, device: torch.device, compile_step: bool = False
    ) -> None:
        self.protocol = protocol
        self.latents = {
            task: _TorchLatentPairAccumulator(device=device, compile_step=compile_step)
            for task in self._KEYS
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
