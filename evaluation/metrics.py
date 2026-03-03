from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_EPSILON = 1e-6
_VALUE_CLAMP = 1.0e6


@dataclass
class MaskedMetricResult:
    mae: float
    rmse: float
    smape: float
    r2: float
    observed_count: int
    effective_count: float
    mae_per_h: list[float]
    rmse_per_h: list[float]


class TorchMaskedMetricAccumulator:
    """Accumulate masked regression metrics on-device and finalize once."""

    def __init__(self, *, device: torch.device, horizon: int | None = None):
        self.device = device
        self.horizon = horizon
        self.mae_sum = torch.tensor(0.0, device=device)
        self.mse_sum = torch.tensor(0.0, device=device)
        self.smape_sum = torch.tensor(0.0, device=device)
        self.weight_sum = torch.tensor(0.0, device=device)
        self.target_weighted_sum = torch.tensor(0.0, device=device)
        self.target_weighted_sq_sum = torch.tensor(0.0, device=device)
        self.observed_count = 0
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
        self.mae_sum += (abs_diff * mask).sum()
        self.mse_sum += ((diff**2) * mask).sum()
        self.smape_sum += (
            2 * abs_diff / (pred.abs() + target.abs() + _EPSILON) * mask
        ).sum()
        self.weight_sum += mask.sum()
        self.target_weighted_sum += (target * mask).sum()
        self.target_weighted_sq_sum += ((target**2) * mask).sum()
        self.observed_count += int((mask > 0).sum().item())

        if self.horizon is not None and self.per_h_mae_sum is not None:
            self.per_h_mae_sum += (abs_diff * mask).sum(dim=0)
            self.per_h_mse_sum += ((diff**2) * mask).sum(dim=0)
            self.per_h_count_sum += mask.sum(dim=0)

        return diff, abs_diff, mask

    def finalize(self) -> MaskedMetricResult:
        if float(self.weight_sum.item()) <= 0:
            return MaskedMetricResult(
                mae=float("nan"),
                rmse=float("nan"),
                smape=float("nan"),
                r2=float("nan"),
                observed_count=0,
                effective_count=0.0,
                mae_per_h=[],
                rmse_per_h=[],
            )

        weight_sum = self.weight_sum.clamp_min(1e-8)
        mae = (self.mae_sum / weight_sum).item()
        rmse = math.sqrt((self.mse_sum / weight_sum).item())
        smape = (self.smape_sum / weight_sum).item()
        ss_res = self.mse_sum.item()
        ss_tot = (
            self.target_weighted_sq_sum
            - (self.target_weighted_sum**2) / weight_sum
        ).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mae_per_h: list[float] = []
        rmse_per_h: list[float] = []
        if self.horizon is not None and self.per_h_count_sum is not None:
            denom = self.per_h_count_sum.clamp_min(1.0)
            mae_per_h = (self.per_h_mae_sum / denom).tolist()
            rmse_per_h = (self.per_h_mse_sum / denom).sqrt().tolist()

        return MaskedMetricResult(
            mae=mae,
            rmse=rmse,
            smape=smape,
            r2=r2,
            observed_count=int(self.observed_count),
            effective_count=float(self.weight_sum.item()),
            mae_per_h=mae_per_h,
            rmse_per_h=rmse_per_h,
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
            smape=float("nan"),
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
    smape = float(
        (2 * abs_diff / (np.abs(pred) + np.abs(target) + _EPSILON) * mask).sum()
        / max(_EPSILON, effective_count)
    )
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
        per_h_count = np.maximum(per_h_count, 1.0)
        mae_per_h = (per_h_abs_sum / per_h_count).tolist()
        rmse_per_h = np.sqrt(per_h_sq_sum / per_h_count).tolist()

    return MaskedMetricResult(
        mae=mae,
        rmse=rmse,
        smape=smape,
        r2=r2,
        observed_count=observed_count,
        effective_count=effective_count,
        mae_per_h=mae_per_h,
        rmse_per_h=rmse_per_h,
    )
