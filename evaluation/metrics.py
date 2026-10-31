from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from evaluation.distribution_scores import (
    distribution_wis,
    standard_wis_half_widths,
    student_t_crps,
)

_EPSILON = 1e-6
_VALUE_CLAMP = 1.0e6

MetricSums = dict[str, torch.Tensor]


def _batch_input(value: object) -> object:
    if isinstance(value, torch.Tensor):
        value = value.contiguous()
        if value.ndim > 0:
            torch._dynamo.maybe_mark_dynamic(value, 0)
        return value
    if isinstance(value, tuple):
        return tuple(_batch_input(item) for item in value)
    return value


@functools.cache
def compiled_metric_step(step: Callable, **compile_options: object) -> Callable:
    """Return one shared ``torch.compile`` wrapper for a pure metric step.

    A step takes the running sums as a dict and returns new sums, so the
    compiled graph holds no in-place state or Python side effects. The
    wrapper is shared, so all heads and splits reuse the same graphs.

    Heads pass contiguous tensors or views of wider nowcast tensors, and the
    last batch is smaller. Contiguous inputs with a dynamic batch dim let one
    graph serve all of them; static guards gave five graphs per step. Python
    constants (for example the Student-t df) stay specialized.
    """
    compiled = torch.compile(step, **compile_options)

    @functools.wraps(step)
    def run(*args: object, **kwargs: object) -> object:
        return compiled(*(_batch_input(arg) for arg in args), **kwargs)

    return run


def distinct_metric_sums(sums: MetricSums) -> MetricSums:
    """Give every key its own tensor.

    Dynamo guards on tensors shared between keys. The first step returns
    distinct tensors, so shared initial zeros would force one more compile.
    """
    return {name: value.clone() for name, value in sums.items()}


def _clean_values(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=_VALUE_CLAMP,
        neginf=-_VALUE_CLAMP,
    ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)


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
    wis: torch.Tensor
    intervals: dict[str, tuple[torch.Tensor, torch.Tensor]]
    weights: torch.Tensor


@dataclass
class ProbabilisticMetricResult:
    nll: float
    crps: float
    wis: float
    invalid_prediction_count: int
    scale_floor_count: int
    observed_count: int
    effective_count: float
    nll_per_h: list[float]
    crps_per_h: list[float]
    wis_per_h: list[float]
    coverage_by_level: dict[str, float]
    interval_width_by_level: dict[str, float]
    coverage_per_h_by_level: dict[str, list[float]]
    interval_width_per_h_by_level: dict[str, list[float]]


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


def _masked_metric_step(
    sums: MetricSums,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
) -> tuple[MetricSums, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add one batch to ``sums``; return (sums, diff, abs_diff, effective_mask)."""
    pred = _clean_values(predictions)
    target = _clean_values(targets)

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
    new_sums = {
        "mae": sums["mae"] + weighted_abs_error.sum(),
        "mse": sums["mse"] + weighted_sq_error.sum(),
        "weight": sums["weight"] + mask.sum(),
        "target": sums["target"] + (target * mask).sum(),
        "target_sq": sums["target_sq"] + ((target**2) * mask).sum(),
        "observed": sums["observed"] + (mask > 0).sum(),
    }
    if "per_h_mae" in sums:
        new_sums["per_h_mae"] = sums["per_h_mae"] + weighted_abs_error.sum(dim=0)
        new_sums["per_h_mse"] = sums["per_h_mse"] + weighted_sq_error.sum(dim=0)
        new_sums["per_h_count"] = sums["per_h_count"] + mask.sum(dim=0)
    return new_sums, diff, abs_diff, mask


class TorchMaskedMetricAccumulator:
    """Accumulate masked metrics; per-horizon mode requires [B, H]."""

    def __init__(
        self,
        *,
        device: torch.device,
        horizon: int | None = None,
        compile_step: bool = False,
    ):
        self.device = device
        self.horizon = horizon
        zero = torch.tensor(0.0, device=device)
        self.sums: MetricSums = {
            "mae": zero,
            "mse": zero,
            "weight": zero,
            "target": zero,
            "target_sq": zero,
            "observed": torch.tensor(0, device=device, dtype=torch.long),
        }
        if horizon is not None:
            zeros = torch.zeros(horizon, device=device)
            self.sums.update(per_h_mae=zeros, per_h_mse=zeros, per_h_count=zeros)
        self.sums = distinct_metric_sums(self.sums)
        self._step = (
            compiled_metric_step(_masked_metric_step)
            if compile_step
            else _masked_metric_step
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update accumulators and return cleaned (diff, abs_diff, effective_mask)."""
        self.sums, diff, abs_diff, mask = self._step(
            self.sums, predictions, targets, observed_mask, sample_weights
        )
        return diff, abs_diff, mask

    def finalize(self) -> MaskedMetricResult:
        if float(self.sums["weight"].item()) <= 0:
            return MaskedMetricResult(
                mae=float("nan"),
                rmse=float("nan"),
                r2=float("nan"),
                observed_count=0,
                effective_count=0.0,
                mae_per_h=[],
                rmse_per_h=[],
            )

        weight_sum = self.sums["weight"].clamp_min(1e-8)
        mae = (self.sums["mae"] / weight_sum).item()
        rmse = math.sqrt((self.sums["mse"] / weight_sum).item())
        ss_res = self.sums["mse"].item()
        ss_tot = (
            self.sums["target_sq"] - (self.sums["target"] ** 2) / weight_sum
        ).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mae_per_h: list[float] = []
        rmse_per_h: list[float] = []
        if self.horizon is not None:
            mae_per_h, rmse_per_h = _compute_per_horizon_metrics(
                abs_sum=self.sums["per_h_mae"],
                mse_sum=self.sums["per_h_mse"],
                count_sum=self.sums["per_h_count"],
            )

        return MaskedMetricResult(
            mae=mae,
            rmse=rmse,
            r2=r2,
            observed_count=int(self.sums["observed"].item()),
            effective_count=float(self.sums["weight"].item()),
            mae_per_h=mae_per_h,
            rmse_per_h=rmse_per_h,
        )


def _distribution_effective_mask(
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


def _distribution_nll(
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    distribution_family: str,
    distribution_df: float,
) -> torch.Tensor:
    if distribution_family == "normal":
        z = (target - loc) / scale
        return 0.5 * z**2 + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)

    df = torch.as_tensor(distribution_df, dtype=loc.dtype, device=loc.device)
    dist = torch.distributions.StudentT(df, loc=loc, scale=scale)
    return -dist.log_prob(target)


def _distribution_metric_step(
    sums: MetricSums,
    loc: torch.Tensor,
    scale: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    *,
    distribution_family: str,
    distribution_df: float,
    scale_floor: float,
    wis_half_widths: torch.Tensor,
    interval_half_widths: dict[str, torch.Tensor],
) -> tuple[MetricSums, ProbabilisticBatchMetrics]:
    """Add one batch of probabilistic scores to ``sums``."""
    loc_clean = _clean_values(loc)
    scale_clean = torch.nan_to_num(
        scale.float(),
        nan=scale_floor,
        posinf=_VALUE_CLAMP,
        neginf=scale_floor,
    ).clamp(min=scale_floor, max=_VALUE_CLAMP)
    target_clean = _clean_values(targets).to(loc_clean.device)
    mask = _distribution_effective_mask(
        targets=targets,
        observed_mask=observed_mask,
        sample_weights=sample_weights,
    ).to(loc_clean.device)
    active = mask > 0
    invalid = (~torch.isfinite(loc) | ~torch.isfinite(scale) | (scale <= 0)) & active
    per_h = "per_h_nll" in sums

    nll = torch.nan_to_num(
        _distribution_nll(
            loc=loc_clean,
            scale=scale_clean,
            target=target_clean,
            distribution_family=distribution_family,
            distribution_df=distribution_df,
        ),
        nan=_VALUE_CLAMP,
        posinf=_VALUE_CLAMP,
        neginf=-_VALUE_CLAMP,
    ).clamp(min=-_VALUE_CLAMP, max=_VALUE_CLAMP)
    if distribution_family == "normal":
        crps = _normal_crps(loc=loc_clean, scale=scale_clean, target=target_clean)
    else:
        crps = student_t_crps(loc_clean, scale_clean, target_clean, distribution_df)
    wis = distribution_wis(loc_clean, scale_clean, target_clean, wis_half_widths)
    weighted_nll = nll * mask
    weighted_crps = crps * mask
    weighted_wis = wis * mask

    new_sums = {
        "nll": sums["nll"] + weighted_nll.sum(),
        "crps": sums["crps"] + weighted_crps.sum(),
        "wis": sums["wis"] + weighted_wis.sum(),
        "weight": sums["weight"] + mask.sum(),
        "observed": sums["observed"] + active.sum(),
        "invalid": sums["invalid"] + invalid.sum(),
        "scale_floor": sums["scale_floor"] + ((scale <= scale_floor) & active).sum(),
    }
    if per_h:
        new_sums["per_h_nll"] = sums["per_h_nll"] + weighted_nll.sum(dim=0)
        new_sums["per_h_crps"] = sums["per_h_crps"] + weighted_crps.sum(dim=0)
        new_sums["per_h_wis"] = sums["per_h_wis"] + weighted_wis.sum(dim=0)
        new_sums["per_h_weight"] = sums["per_h_weight"] + mask.sum(dim=0)

    intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for level_name, standard_half_width in interval_half_widths.items():
        half_width = scale_clean * standard_half_width.to(scale_clean.dtype)
        low = loc_clean - half_width
        high = loc_clean + half_width
        intervals[level_name] = (low, high)
        covered = ((target_clean >= low) & (target_clean <= high)).to(mask.dtype)
        weighted_covered = covered * mask
        weighted_width = (high - low).clamp_min(0.0) * mask
        new_sums[f"coverage_{level_name}"] = (
            sums[f"coverage_{level_name}"] + weighted_covered.sum()
        )
        new_sums[f"width_{level_name}"] = (
            sums[f"width_{level_name}"] + weighted_width.sum()
        )
        if per_h:
            new_sums[f"per_h_coverage_{level_name}"] = sums[
                f"per_h_coverage_{level_name}"
            ] + weighted_covered.sum(dim=0)
            new_sums[f"per_h_width_{level_name}"] = sums[
                f"per_h_width_{level_name}"
            ] + weighted_width.sum(dim=0)

    return new_sums, ProbabilisticBatchMetrics(
        nll=nll,
        crps=crps,
        wis=wis,
        intervals=intervals,
        weights=mask,
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
        compile_step: bool = False,
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
        self.wis_half_widths = torch.tensor(
            standard_wis_half_widths(self.distribution_family, self.distribution_df),
            device=device,
            dtype=torch.float32,
        )

        zero = torch.tensor(0.0, device=device)
        zero_count = torch.tensor(0, device=device, dtype=torch.long)
        self.sums: MetricSums = {
            "nll": zero,
            "crps": zero,
            "wis": zero,
            "weight": zero,
            "observed": zero_count,
            "invalid": zero_count,
            "scale_floor": zero_count,
        }
        for level_name in self.level_names:
            self.sums[f"coverage_{level_name}"] = zero
            self.sums[f"width_{level_name}"] = zero
        if horizon is not None:
            zeros = torch.zeros(horizon, device=device)
            for name in ("nll", "crps", "wis", "weight"):
                self.sums[f"per_h_{name}"] = zeros
            for level_name in self.level_names:
                self.sums[f"per_h_coverage_{level_name}"] = zeros
                self.sums[f"per_h_width_{level_name}"] = zeros
        self.sums = distinct_metric_sums(self.sums)
        self._step = (
            compiled_metric_step(_distribution_metric_step)
            if compile_step
            else _distribution_metric_step
        )

    def update(
        self,
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        targets: torch.Tensor,
        observed_mask: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
    ) -> ProbabilisticBatchMetrics:
        self.sums, batch = self._step(
            self.sums,
            loc,
            scale,
            targets,
            observed_mask,
            sample_weights,
            distribution_family=self.distribution_family,
            distribution_df=self.distribution_df,
            scale_floor=self.scale_floor,
            wis_half_widths=self.wis_half_widths,
            interval_half_widths=self.interval_half_widths,
        )
        return batch

    @staticmethod
    def _mean_from_sums(
        *,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> float:
        if float(denominator.item()) <= 0:
            return float("nan")
        return float((numerator / denominator.clamp_min(1e-8)).item())

    def _per_h_from_sums(self, name: str) -> list[float]:
        if self.horizon is None:
            return []
        numerator = self.sums[f"per_h_{name}"]
        denominator = self.sums["per_h_weight"]
        out = torch.full_like(numerator, float("nan"))
        valid = denominator > 0
        if torch.any(valid):
            out[valid] = numerator[valid] / denominator[valid]
        return out.tolist()

    def finalize(self) -> ProbabilisticMetricResult:
        weight_sum = self.sums["weight"]
        coverage_by_level: dict[str, float] = {}
        interval_width_by_level: dict[str, float] = {}
        coverage_per_h_by_level: dict[str, list[float]] = {}
        interval_width_per_h_by_level: dict[str, list[float]] = {}
        for level_name in self.level_names:
            coverage_by_level[level_name] = self._mean_from_sums(
                numerator=self.sums[f"coverage_{level_name}"],
                denominator=weight_sum,
            )
            interval_width_by_level[level_name] = self._mean_from_sums(
                numerator=self.sums[f"width_{level_name}"],
                denominator=weight_sum,
            )
            coverage_per_h_by_level[level_name] = self._per_h_from_sums(
                f"coverage_{level_name}"
            )
            interval_width_per_h_by_level[level_name] = self._per_h_from_sums(
                f"width_{level_name}"
            )

        invalid_prediction_count = int(self.sums["invalid"].item())
        return ProbabilisticMetricResult(
            nll=self._mean_from_sums(
                numerator=self.sums["nll"], denominator=weight_sum
            ),
            crps=self._mean_from_sums(
                numerator=self.sums["crps"], denominator=weight_sum
            ),
            wis=self._mean_from_sums(numerator=self.sums["wis"], denominator=weight_sum)
            if invalid_prediction_count == 0
            else float("nan"),
            invalid_prediction_count=invalid_prediction_count,
            scale_floor_count=int(self.sums["scale_floor"].item()),
            observed_count=int(self.sums["observed"].item()),
            effective_count=float(weight_sum.item()),
            nll_per_h=self._per_h_from_sums("nll"),
            crps_per_h=self._per_h_from_sums("crps"),
            wis_per_h=self._per_h_from_sums("wis")
            if invalid_prediction_count == 0
            else [float("nan")] * (self.horizon or 0),
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
