from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_EPSILON = 1e-6
_VALUE_CLAMP = 1.0e6


def _compute_per_horizon_metrics(
    abs_sum: torch.Tensor,
    mse_sum: torch.Tensor,
    count_sum: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Return per-horizon metrics, preserving unsupported horizons as NaN."""
    mae = torch.full_like(abs_sum, float("nan"))
    rmse = torch.full_like(mse_sum, float("nan"))
    valid = count_sum > 0
    if torch.any(valid):
        mae[valid] = abs_sum[valid] / count_sum[valid]
        rmse[valid] = torch.sqrt(mse_sum[valid] / count_sum[valid])
    return mae.tolist(), rmse.tolist()


def _compute_per_horizon_metrics_numpy(
    abs_sum: np.ndarray,
    mse_sum: np.ndarray,
    count_sum: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return per-horizon metrics, preserving unsupported horizons as NaN."""
    mae = np.full_like(abs_sum, np.nan, dtype=np.float64)
    rmse = np.full_like(mse_sum, np.nan, dtype=np.float64)
    valid = count_sum > 0
    if np.any(valid):
        mae[valid] = abs_sum[valid] / count_sum[valid]
        rmse[valid] = np.sqrt(mse_sum[valid] / count_sum[valid])
    return mae.tolist(), rmse.tolist()


@dataclass
class MaskedMetricResult:
    mae: float
    rmse: float
    r2: float
    observed_count: int
    effective_count: float
    mae_per_h: list[float]
    rmse_per_h: list[float]


@dataclass
class ProbabilisticBatchMetrics:
    nll: torch.Tensor
    crps: torch.Tensor
    intervals: dict[str, tuple[torch.Tensor, torch.Tensor]]
    weights: torch.Tensor


@dataclass
class ProbabilisticMetricResult:
    nll: float
    crps: float
    observed_count: int
    effective_count: float
    nll_per_h: list[float]
    crps_per_h: list[float]
    coverage_by_level: dict[str, float]
    interval_width_by_level: dict[str, float]
    coverage_per_h_by_level: dict[str, list[float]]
    interval_width_per_h_by_level: dict[str, list[float]]


@dataclass
class WeightedIntervalScoreBatch:
    """Per-point WIS values and their additive decomposition."""

    wis: torch.Tensor
    dispersion: torch.Tensor
    underprediction: torch.Tensor
    overprediction: torch.Tensor
    point: torch.Tensor
    weights: torch.Tensor


@dataclass
class WeightedIntervalScoreResult:
    """Masked mean WIS, components, counts, and per-horizon means."""

    wis: float
    dispersion: float
    underprediction: float
    overprediction: float
    point: float
    observed_count: int
    effective_count: float
    wis_per_h: list[float]
    dispersion_per_h: list[float]
    underprediction_per_h: list[float]
    overprediction_per_h: list[float]
    point_per_h: list[float]


class TorchWeightedIntervalScoreAccumulator:
    """Accumulate the standard weighted interval score on-device.

    ``interval_levels`` are central coverage levels, so interval ``k`` has
    ``alpha_k = 1 - level_k``. With ``K`` intervals, the score is

    ``(0.5 * |y-m| + sum(alpha_k / 2 * IS_alpha_k)) / (K + 0.5)``.

    The returned components are additive: ``wis == point + dispersion +
    underprediction + overprediction`` up to floating-point tolerance.
    """

    _COMPONENTS = (
        "wis",
        "dispersion",
        "underprediction",
        "overprediction",
        "point",
    )

    def __init__(
        self,
        *,
        device: torch.device,
        interval_levels: list[float] | tuple[float, ...],
        horizon: int | None = None,
    ) -> None:
        self.device = device
        self.horizon = horizon
        self.interval_levels = tuple(float(level) for level in interval_levels)
        if not self.interval_levels:
            raise ValueError("WIS requires at least one central interval level")
        if any(not 0.0 < level < 1.0 for level in self.interval_levels):
            raise ValueError("WIS interval levels must be strictly between 0 and 1")
        self.level_names = tuple(
            interval_level_name(level) for level in self.interval_levels
        )
        self._sums = {
            name: torch.tensor(0.0, device=device) for name in self._COMPONENTS
        }
        self._per_h_sums = {
            name: torch.zeros(horizon, device=device)
            for name in self._COMPONENTS
            if horizon is not None
        }
        self._weight_sum = torch.tensor(0.0, device=device)
        self._per_h_weight_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self._observed_count = torch.tensor(0, device=device, dtype=torch.long)

    def update(
        self,
        *,
        median: torch.Tensor,
        targets: torch.Tensor,
        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]],
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
    ) -> WeightedIntervalScoreBatch:
        missing = sorted(set(self.level_names) - set(intervals))
        if missing:
            raise ValueError(f"WIS intervals missing levels: {missing}")
        median_clean = torch.nan_to_num(
            median.float(), nan=0.0, posinf=_VALUE_CLAMP, neginf=-_VALUE_CLAMP
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
        target_clean = torch.nan_to_num(
            targets.float(), nan=0.0, posinf=_VALUE_CLAMP, neginf=-_VALUE_CLAMP
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP).to(median_clean.device)
        finite_target = torch.isfinite(targets.float()).to(median_clean.device).float()
        if sample_weights is not None:
            weights = torch.nan_to_num(
                sample_weights.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
        elif observed_mask is None:
            weights = finite_target
        else:
            weights = torch.nan_to_num(
                observed_mask.float(), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
        weights = weights.to(median_clean.device) * finite_target

        normalizer = len(self.interval_levels) + 0.5
        point = 0.5 * (median_clean - target_clean).abs() / normalizer
        dispersion = torch.zeros_like(point)
        underprediction = torch.zeros_like(point)
        overprediction = torch.zeros_like(point)
        for level, level_name in zip(
            self.interval_levels, self.level_names, strict=True
        ):
            alpha = 1.0 - level
            low, high = intervals[level_name]
            low = torch.nan_to_num(low.float(), nan=0.0).to(median_clean.device)
            high = torch.nan_to_num(high.float(), nan=0.0).to(median_clean.device)
            width = (high - low).clamp_min(0.0)
            dispersion += (alpha / 2.0) * width / normalizer
            underprediction += (
                (target_clean - high).clamp_min(0.0) / normalizer
            )
            overprediction += (
                (low - target_clean).clamp_min(0.0) / normalizer
            )
        wis = point + dispersion + underprediction + overprediction
        batch = WeightedIntervalScoreBatch(
            wis=wis,
            dispersion=dispersion,
            underprediction=underprediction,
            overprediction=overprediction,
            point=point,
            weights=weights,
        )
        weight_sum = weights.sum()
        self._weight_sum += weight_sum
        self._observed_count += (weights > 0).sum()
        if self.horizon is not None:
            if wis.ndim != 2 or int(wis.shape[1]) != self.horizon:
                raise ValueError(
                    f"WIS horizon expected [B, {self.horizon}], got {tuple(wis.shape)}"
                )
            assert self._per_h_weight_sum is not None
            self._per_h_weight_sum += weights.sum(dim=0)
        for name in self._COMPONENTS:
            values = getattr(batch, name)
            self._sums[name] += (values * weights).sum()
            if self.horizon is not None:
                self._per_h_sums[name] += (values * weights).sum(dim=0)
        return batch

    def _mean(self, name: str) -> float:
        if float(self._weight_sum.item()) <= 0:
            return float("nan")
        return float((self._sums[name] / self._weight_sum).item())

    def _per_h(self, name: str) -> list[float]:
        if self.horizon is None or self._per_h_weight_sum is None:
            return []
        out = torch.full_like(self._per_h_sums[name], float("nan"))
        valid = self._per_h_weight_sum > 0
        out[valid] = self._per_h_sums[name][valid] / self._per_h_weight_sum[valid]
        return out.tolist()

    def finalize(self) -> WeightedIntervalScoreResult:
        return WeightedIntervalScoreResult(
            wis=self._mean("wis"),
            dispersion=self._mean("dispersion"),
            underprediction=self._mean("underprediction"),
            overprediction=self._mean("overprediction"),
            point=self._mean("point"),
            observed_count=int(self._observed_count.item()),
            effective_count=float(self._weight_sum.item()),
            wis_per_h=self._per_h("wis"),
            dispersion_per_h=self._per_h("dispersion"),
            underprediction_per_h=self._per_h("underprediction"),
            overprediction_per_h=self._per_h("overprediction"),
            point_per_h=self._per_h("point"),
        )


def interval_level_name(level: float) -> str:
    pct = float(level) * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return str(int(round(pct)))
    return f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _normal_crps(
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    z = (target - loc) / scale
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return (
        scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    ).clamp_min(0.0)


def standard_interval_half_width(
    *,
    level: float,
    distribution_family: str,
    distribution_df: float,
) -> float:
    quantile = (1.0 + float(level)) / 2.0
    if distribution_family == "normal":
        normal = torch.distributions.Normal(0.0, 1.0)
        return float(normal.icdf(torch.tensor(quantile, dtype=torch.float64)).item())

    from scipy import stats

    return float(stats.t.ppf(quantile, distribution_df))


class TorchMaskedMetricAccumulator:
    """Accumulate masked metrics; per-horizon mode requires [B, H]."""

    def __init__(self, *, device: torch.device, horizon: int | None = None):
        self.device = device
        self.horizon = horizon
        self.mae_sum = torch.tensor(0.0, device=device)
        self.mse_sum = torch.tensor(0.0, device=device)
        self.weight_sum = torch.tensor(0.0, device=device)
        self.target_weighted_sum = torch.tensor(0.0, device=device)
        self.target_weighted_sq_sum = torch.tensor(0.0, device=device)
        self.observed_count = torch.tensor(0, device=device, dtype=torch.long)
        self.per_h_mae_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.per_h_mse_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.per_h_count_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update accumulators and return cleaned (diff, abs_diff, effective_mask)."""
        pred = torch.nan_to_num(
            predictions.float(),
            nan=0.0,
            posinf=_VALUE_CLAMP,
            neginf=-_VALUE_CLAMP,
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
        target = torch.nan_to_num(
            targets.float(),
            nan=0.0,
            posinf=_VALUE_CLAMP,
            neginf=-_VALUE_CLAMP,
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)

        finite_target = torch.isfinite(targets.float()).to(pred.dtype)
        if sample_weights is not None:
            mask = (
                torch.nan_to_num(
                    sample_weights.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0)
                * finite_target
            )
        elif observed_mask is None:
            mask = finite_target
        else:
            mask = (
                torch.nan_to_num(
                    observed_mask.float(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=1.0)
                * finite_target
            ).to(pred.dtype)

        diff = pred - target
        abs_diff = diff.abs()
        weighted_abs_error = abs_diff * mask
        weighted_sq_error = (diff**2) * mask
        mask_sum = mask.sum()
        self.mae_sum += weighted_abs_error.sum()
        self.mse_sum += weighted_sq_error.sum()
        self.weight_sum += mask_sum
        self.target_weighted_sum += (target * mask).sum()
        self.target_weighted_sq_sum += ((target**2) * mask).sum()
        self.observed_count += (mask > 0).sum()

        if self.horizon is not None:
            assert self.per_h_mae_sum is not None
            assert self.per_h_mse_sum is not None
            assert self.per_h_count_sum is not None
            self.per_h_mae_sum += weighted_abs_error.sum(dim=0)
            self.per_h_mse_sum += weighted_sq_error.sum(dim=0)
            self.per_h_count_sum += mask.sum(dim=0)

        return diff, abs_diff, mask

    def finalize(self) -> MaskedMetricResult:
        if float(self.weight_sum.item()) <= 0:
            return MaskedMetricResult(
                mae=float("nan"),
                rmse=float("nan"),
                r2=float("nan"),
                observed_count=0,
                effective_count=0.0,
                mae_per_h=[],
                rmse_per_h=[],
            )

        weight_sum = self.weight_sum.clamp_min(1e-8)
        mae = (self.mae_sum / weight_sum).item()
        rmse = math.sqrt((self.mse_sum / weight_sum).item())
        ss_res = self.mse_sum.item()
        ss_tot = (
            self.target_weighted_sq_sum - (self.target_weighted_sum**2) / weight_sum
        ).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mae_per_h: list[float] = []
        rmse_per_h: list[float] = []
        if self.horizon is not None:
            assert self.per_h_mae_sum is not None
            assert self.per_h_mse_sum is not None
            assert self.per_h_count_sum is not None
            mae_per_h, rmse_per_h = _compute_per_horizon_metrics(
                abs_sum=self.per_h_mae_sum,
                mse_sum=self.per_h_mse_sum,
                count_sum=self.per_h_count_sum,
            )

        return MaskedMetricResult(
            mae=mae,
            rmse=rmse,
            r2=r2,
            observed_count=int(self.observed_count.item()),
            effective_count=float(self.weight_sum.item()),
            mae_per_h=mae_per_h,
            rmse_per_h=rmse_per_h,
        )


class TorchDistributionMetricAccumulator:
    """Accumulate masked probabilistic forecast scores on-device."""

    def __init__(
        self,
        *,
        device: torch.device,
        horizon: int | None = None,
        distribution_family: str = "normal",
        distribution_df: float = 5.0,
        scale_floor: float = 1.0e-3,
        interval_levels: list[float] | tuple[float, ...] = (0.5, 0.8, 0.95),
    ) -> None:
        self.device = device
        self.horizon = horizon
        self.distribution_family = distribution_family.lower()
        self.distribution_df = float(distribution_df)
        self.scale_floor = float(scale_floor)
        self.interval_levels = tuple(float(level) for level in interval_levels)
        if self.distribution_family not in {"normal", "student_t"}:
            raise ValueError(
                "distribution_family must be one of ['normal', 'student_t'], "
                f"got {self.distribution_family!r}"
            )
        if self.distribution_family == "student_t" and self.distribution_df <= 2.0:
            raise ValueError(
                f"distribution_df must be > 2 for student_t, got {self.distribution_df}"
            )
        if self.scale_floor <= 0:
            raise ValueError(f"scale_floor must be positive, got {self.scale_floor}")

        self.level_names = tuple(
            interval_level_name(level) for level in self.interval_levels
        )
        self.interval_half_widths = {
            level_name: torch.tensor(
                standard_interval_half_width(
                    level=level,
                    distribution_family=self.distribution_family,
                    distribution_df=self.distribution_df,
                ),
                device=device,
                dtype=torch.float32,
            )
            for level_name, level in zip(
                self.level_names, self.interval_levels, strict=True
            )
        }

        self.nll_sum = torch.tensor(0.0, device=device)
        self.nll_weight_sum = torch.tensor(0.0, device=device)
        self.crps_sum = torch.tensor(0.0, device=device)
        self.crps_weight_sum = torch.tensor(0.0, device=device)
        self.observed_count = torch.tensor(0, device=device, dtype=torch.long)
        self.per_h_nll_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.per_h_nll_count = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.per_h_crps_sum = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.per_h_crps_count = (
            torch.zeros(horizon, device=device) if horizon is not None else None
        )
        self.coverage_sum = {
            level_name: torch.tensor(0.0, device=device)
            for level_name in self.level_names
        }
        self.interval_width_sum = {
            level_name: torch.tensor(0.0, device=device)
            for level_name in self.level_names
        }
        self.interval_weight_sum = {
            level_name: torch.tensor(0.0, device=device)
            for level_name in self.level_names
        }
        self.per_h_coverage_sum = {
            level_name: torch.zeros(horizon, device=device)
            for level_name in self.level_names
            if horizon is not None
        }
        self.per_h_interval_width_sum = {
            level_name: torch.zeros(horizon, device=device)
            for level_name in self.level_names
            if horizon is not None
        }
        self.per_h_interval_count = {
            level_name: torch.zeros(horizon, device=device)
            for level_name in self.level_names
            if horizon is not None
        }

    def _effective_mask(
        self,
        *,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        finite_target = torch.isfinite(targets.float()).to(dtype=torch.float32)
        if sample_weights is not None:
            mask = torch.nan_to_num(
                sample_weights.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=0.0)
        elif observed_mask is None:
            mask = finite_target
        else:
            mask = torch.nan_to_num(
                observed_mask.float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(min=0.0, max=1.0)
        return mask * finite_target

    def _nll(
        self,
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.distribution_family == "normal":
            z = (target - loc) / scale
            return 0.5 * z**2 + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)

        df = torch.as_tensor(
            self.distribution_df,
            dtype=loc.dtype,
            device=loc.device,
        )
        dist = torch.distributions.StudentT(df, loc=loc, scale=scale)
        return -dist.log_prob(target)

    def update(
        self,
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
    ) -> ProbabilisticBatchMetrics:
        loc_clean = torch.nan_to_num(
            loc.float(),
            nan=0.0,
            posinf=_VALUE_CLAMP,
            neginf=-_VALUE_CLAMP,
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
        scale_clean = torch.nan_to_num(
            scale.float(),
            nan=self.scale_floor,
            posinf=_VALUE_CLAMP,
            neginf=self.scale_floor,
        ).clamp(min=self.scale_floor, max=_VALUE_CLAMP)
        target_clean = torch.nan_to_num(
            targets.float(),
            nan=0.0,
            posinf=_VALUE_CLAMP,
            neginf=-_VALUE_CLAMP,
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
        target_clean = target_clean.to(loc_clean.device)
        mask = self._effective_mask(
            targets=targets,
            observed_mask=observed_mask,
            sample_weights=sample_weights,
        ).to(loc_clean.device)

        nll = torch.nan_to_num(
            self._nll(loc=loc_clean, scale=scale_clean, target=target_clean),
            nan=_VALUE_CLAMP,
            posinf=_VALUE_CLAMP,
            neginf=-_VALUE_CLAMP,
        ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
        weighted_nll = nll * mask
        mask_sum = mask.sum()
        mask_per_h = mask.sum(dim=0) if self.horizon is not None else None
        self.nll_sum += weighted_nll.sum()
        self.nll_weight_sum += mask_sum
        self.observed_count += (mask > 0).sum()

        if self.distribution_family == "normal":
            crps = torch.nan_to_num(
                _normal_crps(loc=loc_clean, scale=scale_clean, target=target_clean),
                nan=0.0,
                posinf=_VALUE_CLAMP,
                neginf=0.0,
            ).clamp(min=0.0, max=_VALUE_CLAMP)
            weighted_crps = crps * mask
            self.crps_sum += weighted_crps.sum()
            self.crps_weight_sum += mask_sum
        else:
            crps = torch.full_like(nll, float("nan"))

        if self.horizon is not None:
            assert self.per_h_nll_sum is not None
            assert self.per_h_nll_count is not None
            assert mask_per_h is not None
            self.per_h_nll_sum += weighted_nll.sum(dim=0)
            self.per_h_nll_count += mask_per_h
            if self.distribution_family == "normal":
                assert self.per_h_crps_sum is not None
                assert self.per_h_crps_count is not None
                self.per_h_crps_sum += weighted_crps.sum(dim=0)
                self.per_h_crps_count += mask_per_h

        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for level_name in self.level_names:
            half_width = scale_clean * self.interval_half_widths[level_name].to(
                scale_clean.dtype
            )
            low = loc_clean - half_width
            high = loc_clean + half_width
            intervals[level_name] = (low, high)
            covered = ((target_clean >= low) & (target_clean <= high)).to(
                dtype=mask.dtype
            )
            width = (high - low).clamp_min(0.0)
            weighted_covered = covered * mask
            weighted_width = width * mask
            self.coverage_sum[level_name] += weighted_covered.sum()
            self.interval_width_sum[level_name] += weighted_width.sum()
            self.interval_weight_sum[level_name] += mask_sum
            if self.horizon is not None:
                assert mask_per_h is not None
                self.per_h_coverage_sum[level_name] += weighted_covered.sum(dim=0)
                self.per_h_interval_width_sum[level_name] += weighted_width.sum(dim=0)
                self.per_h_interval_count[level_name] += mask_per_h

        return ProbabilisticBatchMetrics(
            nll=nll,
            crps=crps,
            intervals=intervals,
            weights=mask,
        )

    @staticmethod
    def _mean_from_sums(
        *,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> float:
        if float(denominator.item()) <= 0:
            return float("nan")
        return float((numerator / denominator.clamp_min(1e-8)).item())

    def _per_h_from_sums(
        self,
        *,
        numerator: torch.Tensor | None,
        denominator: torch.Tensor | None,
    ) -> list[float]:
        if self.horizon is None or numerator is None or denominator is None:
            return []
        out = torch.full_like(numerator, float("nan"))
        valid = denominator > 0
        if torch.any(valid):
            out[valid] = numerator[valid] / denominator[valid]
        return out.tolist()

    def finalize(self) -> ProbabilisticMetricResult:
        coverage_by_level: dict[str, float] = {}
        interval_width_by_level: dict[str, float] = {}
        coverage_per_h_by_level: dict[str, list[float]] = {}
        interval_width_per_h_by_level: dict[str, list[float]] = {}
        for level_name in self.level_names:
            weight_sum = self.interval_weight_sum[level_name]
            coverage_by_level[level_name] = self._mean_from_sums(
                numerator=self.coverage_sum[level_name],
                denominator=weight_sum,
            )
            interval_width_by_level[level_name] = self._mean_from_sums(
                numerator=self.interval_width_sum[level_name],
                denominator=weight_sum,
            )
            coverage_per_h_by_level[level_name] = self._per_h_from_sums(
                numerator=self.per_h_coverage_sum.get(level_name),
                denominator=self.per_h_interval_count.get(level_name),
            )
            interval_width_per_h_by_level[level_name] = self._per_h_from_sums(
                numerator=self.per_h_interval_width_sum.get(level_name),
                denominator=self.per_h_interval_count.get(level_name),
            )

        return ProbabilisticMetricResult(
            nll=self._mean_from_sums(
                numerator=self.nll_sum,
                denominator=self.nll_weight_sum,
            ),
            crps=self._mean_from_sums(
                numerator=self.crps_sum,
                denominator=self.crps_weight_sum,
            ),
            observed_count=int(self.observed_count.item()),
            effective_count=float(self.nll_weight_sum.item()),
            nll_per_h=self._per_h_from_sums(
                numerator=self.per_h_nll_sum,
                denominator=self.per_h_nll_count,
            ),
            crps_per_h=self._per_h_from_sums(
                numerator=self.per_h_crps_sum,
                denominator=self.per_h_crps_count,
            ),
            coverage_by_level=coverage_by_level,
            interval_width_by_level=interval_width_by_level,
            coverage_per_h_by_level=coverage_per_h_by_level,
            interval_width_per_h_by_level=interval_width_per_h_by_level,
        )


def compute_masked_metrics_numpy(
    predictions: np.ndarray,
    targets: np.ndarray,
    observed_mask: np.ndarray | None,
    sample_weights: np.ndarray | None = None,
    horizon: int | None = None,
) -> MaskedMetricResult:
    # Detect overflow issues early
    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            f"Non-finite values detected in predictions (inf={np.isinf(predictions).sum()}, "
            f"nan={np.isnan(predictions).sum()}). This may indicate float16 overflow. "
            f"Ensure data is upcast to float32/float64 before metric computation."
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError(
            f"Non-finite values detected in targets (inf={np.isinf(targets).sum()}, "
            f"nan={np.isnan(targets).sum()}). This may indicate float16 overflow. "
            f"Ensure data is upcast to float32/float64 before metric computation."
        )

    pred = np.nan_to_num(predictions.astype(np.float64), nan=0.0)
    target = np.nan_to_num(targets.astype(np.float64), nan=0.0)
    finite_target = np.isfinite(targets).astype(np.float64)
    if sample_weights is not None:
        mask = np.clip(
            np.nan_to_num(sample_weights.astype(np.float64), nan=0.0),
            0.0,
            np.inf,
        )
        mask = mask * finite_target
    elif observed_mask is None:
        mask = finite_target
    else:
        mask = np.clip(
            np.nan_to_num(observed_mask.astype(np.float64), nan=0.0), 0.0, 1.0
        )
        mask = mask * finite_target

    observed_count = int((mask > 0).sum())
    effective_count = float(mask.sum())
    if effective_count <= 0:
        return MaskedMetricResult(
            mae=float("nan"),
            rmse=float("nan"),
            r2=float("nan"),
            observed_count=0,
            effective_count=0.0,
            mae_per_h=[],
            rmse_per_h=[],
        )

    diff = pred - target
    abs_diff = np.abs(diff)
    mae = float((abs_diff * mask).sum() / max(_EPSILON, effective_count))
    rmse = float(np.sqrt(((diff**2) * mask).sum() / max(_EPSILON, effective_count)))
    ss_res = float(((diff**2) * mask).sum())
    weighted_target_sum = float((target * mask).sum())
    weighted_target_sq_sum = float(((target**2) * mask).sum())
    ss_tot = weighted_target_sq_sum - (weighted_target_sum**2) / max(
        _EPSILON, effective_count
    )
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mae_per_h: list[float] = []
    rmse_per_h: list[float] = []
    if horizon is not None and pred.ndim == 2 and pred.shape[1] == horizon:
        per_h_abs_sum = (abs_diff * mask).sum(axis=0)
        per_h_sq_sum = ((diff**2) * mask).sum(axis=0)
        per_h_count = mask.sum(axis=0)
        mae_per_h, rmse_per_h = _compute_per_horizon_metrics_numpy(
            abs_sum=per_h_abs_sum,
            mse_sum=per_h_sq_sum,
            count_sum=per_h_count,
        )

    return MaskedMetricResult(
        mae=mae,
        rmse=rmse,
        r2=r2,
        observed_count=observed_count,
        effective_count=effective_count,
        mae_per_h=mae_per_h,
        rmse_per_h=rmse_per_h,
    )
