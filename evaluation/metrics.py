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
        self.target_mean_acc = torch.tensor(0.0, device=device)
        self.target_m2 = torch.tensor(0.0, device=device)
        self.total_count = 0
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
        if observed_mask is None:
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

        flat_targets = target[mask > 0].detach().float().reshape(-1)
        batch_count = flat_targets.numel()
        if batch_count > 0:
            batch_mean = flat_targets.mean()
            batch_m2 = ((flat_targets - batch_mean) ** 2).sum()

            delta = batch_mean - self.target_mean_acc
            new_count = self.total_count + batch_count
            self.target_mean_acc += delta * batch_count / new_count
            self.target_m2 += (
                batch_m2 + (delta**2) * (self.total_count * batch_count) / new_count
            )
            self.total_count = new_count

        if self.horizon is not None and self.per_h_mae_sum is not None:
            self.per_h_mae_sum += (abs_diff * mask).sum(dim=0)
            self.per_h_mse_sum += ((diff**2) * mask).sum(dim=0)
            self.per_h_count_sum += mask.sum(dim=0)

        return diff, abs_diff, mask

    def finalize(self) -> MaskedMetricResult:
        if self.total_count <= 0:
            return MaskedMetricResult(
                mae=float("nan"),
                rmse=float("nan"),
                smape=float("nan"),
                r2=float("nan"),
                observed_count=0,
                mae_per_h=[],
                rmse_per_h=[],
            )

        mae = (self.mae_sum / max(1, self.total_count)).item()
        rmse = math.sqrt((self.mse_sum / max(1, self.total_count)).item())
        smape = (self.smape_sum / max(1, self.total_count)).item()
        ss_res = self.mse_sum.item()
        ss_tot = self.target_m2.item()
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
            observed_count=int(self.total_count),
            mae_per_h=mae_per_h,
            rmse_per_h=rmse_per_h,
        )


def compute_masked_metrics_numpy(
    predictions: np.ndarray,
    targets: np.ndarray,
    observed_mask: np.ndarray | None,
) -> MaskedMetricResult:
    pred = np.nan_to_num(predictions.astype(np.float64), nan=0.0)
    target = np.nan_to_num(targets.astype(np.float64), nan=0.0)
    finite_target = np.isfinite(targets).astype(np.float64)
    if observed_mask is None:
        mask = finite_target
    else:
        mask = np.clip(
            np.nan_to_num(observed_mask.astype(np.float64), nan=0.0), 0.0, 1.0
        )
        mask = mask * finite_target

    observed_count = int(mask.sum())
    if observed_count <= 0:
        return MaskedMetricResult(
            mae=float("nan"),
            rmse=float("nan"),
            smape=float("nan"),
            r2=float("nan"),
            observed_count=0,
            mae_per_h=[],
            rmse_per_h=[],
        )

    diff = pred - target
    abs_diff = np.abs(diff)
    mae = float((abs_diff * mask).sum() / max(1, observed_count))
    rmse = float(np.sqrt(((diff**2) * mask).sum() / max(1, observed_count)))
    smape = float(
        (2 * abs_diff / (np.abs(pred) + np.abs(target) + _EPSILON) * mask).sum()
        / max(1, observed_count)
    )
    observed_targets = target[mask > 0]
    if observed_targets.size == 0:
        r2 = float("nan")
    else:
        ss_res = float(((diff**2) * mask).sum())
        centered = observed_targets - observed_targets.mean()
        ss_tot = float((centered**2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return MaskedMetricResult(
        mae=mae,
        rmse=rmse,
        smape=smape,
        r2=r2,
        observed_count=observed_count,
        mae_per_h=[],
        rmse_per_h=[],
    )
