from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LossOutlierEvent:
    """Diagnostic emitted when a scalar loss is unusually high."""

    value: float
    mean: float
    variance: float
    stddev: float
    threshold: float
    z_score: float
    count: int


class EmaLossStats:
    """Track an exponential moving mean/variance for scalar training losses."""

    def __init__(self, *, decay: float = 0.99, stddev_threshold: float = 5.0):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if stddev_threshold <= 0.0:
            raise ValueError(
                f"stddev_threshold must be positive, got {stddev_threshold}"
            )
        self.decay = float(decay)
        self.stddev_threshold = float(stddev_threshold)
        self.count = 0
        self.mean = 0.0
        self._second_moment = 0.0

    @property
    def variance(self) -> float:
        return max(0.0, self._second_moment - self.mean * self.mean)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def update(self, value: float) -> LossOutlierEvent | None:
        """Update statistics and return an outlier event if value exceeds history."""
        if not math.isfinite(value):
            return None

        event = None
        if self.count >= 2:
            stddev = self.stddev
            threshold = self.mean + self.stddev_threshold * stddev
            if value > threshold:
                z_score = (value - self.mean) / stddev if stddev > 0.0 else math.inf
                event = LossOutlierEvent(
                    value=value,
                    mean=self.mean,
                    variance=self.variance,
                    stddev=stddev,
                    threshold=threshold,
                    z_score=z_score,
                    count=self.count,
                )
        return self._update_with(value, event=event)

    def _update_with(
        self, value: float, *, event: LossOutlierEvent | None
    ) -> LossOutlierEvent | None:
        if self.count == 0:
            self.mean = value
            self._second_moment = value * value
        else:
            keep = self.decay
            add = 1.0 - self.decay
            self.mean = keep * self.mean + add * value
            self._second_moment = keep * self._second_moment + add * value * value
        self.count += 1
        return event


def _to_list(value: object) -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def stringify_epi_batch_indices(
    batch: Any,
    *,
    max_items: int = 8,
) -> str:
    """Format EpiBatch sample identifiers using human-readable dataset metadata."""
    target_nodes = _to_list(batch.target_node)
    window_starts = _to_list(batch.window_start)
    labels = list(batch.node_labels)
    region_ids = list(getattr(batch, "region_ids", []))
    if not region_ids:
        region_ids = ["unknown"] * len(target_nodes)

    total = min(len(target_nodes), len(window_starts), len(labels), len(region_ids))
    displayed = min(total, max_items)
    entries = []
    for idx in range(displayed):
        entries.append(
            f"{labels[idx]} (region_id={region_ids[idx]}, "
            f"target_node={target_nodes[idx]}, window_start={window_starts[idx]})"
        )
    if total > displayed:
        entries.append(f"... {total - displayed} more")
    return "; ".join(entries)
