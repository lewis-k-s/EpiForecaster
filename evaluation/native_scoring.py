"""Native evaluation settings and selection, without training-loss weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch

from evaluation.student_t_scores import student_t5_cdf
from models.configs import EvaluationConfig

NATIVE_HEADS = ("ww", "cases", "hosp", "deaths")


def native_selection(
    metrics: Mapping[str, object], config: EvaluationConfig
) -> tuple[float, dict[str, Any]]:
    """Apply the declared selection rule to observed heads, without fallback."""
    values = []
    heads = []
    for head in NATIVE_HEADS:
        count = metrics.get(f"observed_count_{head}")
        if count == 0:
            continue
        value = metrics.get(f"{config.selection_metric}_{head}")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            if isinstance(count, int | float) and count > 0:
                raise ValueError(
                    f"Observed head {head!r} has no finite {config.selection_metric.upper()}"
                )
            continue
        values.append(float(value))
        heads.append(head)
    if not values:
        raise ValueError(
            f"Native selection requires at least one finite observed {config.selection_metric.upper()}"
        )
    definition = {
        "metric": config.selection_metric,
        "aggregation": config.selection_aggregation,
        "direction": "minimize",
        "heads": heads,
        "head_weighting": "equal weight over heads with observed exact targets",
        "within_head_aggregation": "mean over exact observed source/horizon query targets",
        "target_space": "clinical log1p(per-100k), weekly admissions; wastewater log1p(concentration)",
        "distribution_family": config.distribution_family,
        "distribution_df": config.distribution_df,
        "interval_levels": list(config.interval_levels),
    }
    return float(np.mean(values)), definition


@dataclass(frozen=True)
class NativeEvaluationScorer:
    """Distribution and alignment settings needed by held-out metric views."""

    config: EvaluationConfig
    horizon_days: int
    distribution_scale_floor: float

    @property
    def weekly_slots(self) -> int:
        return -(-self.horizon_days // 7)

    @property
    def distribution_family(self) -> str:
        return self.config.distribution_family

    @property
    def distribution_df(self) -> float:
        return self.config.distribution_df

    @property
    def interval_levels(self) -> list[float]:
        return self.config.interval_levels

    @staticmethod
    def _align_daily_head_tensor(
        tensor: torch.Tensor, *, head_name: str, target_width: int
    ) -> torch.Tensor:
        if head_name == "cases" and tensor.shape[1] == target_width + 1:
            return tensor[:, 1:]
        return tensor

    def left_censored_nll(
        self, loc: torch.Tensor, scale: torch.Tensor, detection_limit: torch.Tensor
    ) -> torch.Tensor:
        """Companion score for P(concentration <= the recorded detection limit)."""
        safe_scale = torch.nan_to_num(
            scale.float(), nan=self.distribution_scale_floor
        ).clamp_min(self.distribution_scale_floor)
        z = (
            torch.log1p(detection_limit.float().clamp_min(0.0)) - loc.float()
        ) / safe_scale
        if self.distribution_family == "student_t":
            score = -student_t5_cdf(z).clamp_min(1.0e-12).log()
        else:
            score = -torch.special.log_ndtr(z)
        return torch.nan_to_num(score, nan=0.0, posinf=1.0e6)
