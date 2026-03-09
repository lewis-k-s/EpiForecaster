"""Gradient debugging utilities for EpiForecaster.

This module provides toggleable gradient diagnostics to help identify sources of
non-finite gradients during training. When disabled, it has zero overhead.

Usage:
    # In config
    training:
      enable_gradient_debug: true
      gradient_debug_log_dir: "outputs/grad_debug"

    # In trainer
    debugger = GradientDebugger(logger, log_dir)
    if debugger.check_gradients(model):
        stats = debugger.capture_snapshot(model, loss, batch_info)
        debugger.save_report(stats, step)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Statistics for a single parameter or layer's gradients."""

    name: str
    shape: list[int] = field(default_factory=list)
    numel: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    norm: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    finite_ratio: float = 1.0


@dataclass
class GradientSnapshot:
    """Complete snapshot of gradient state at a specific step."""

    step: int
    epoch: int
    batch_idx: int
    loss: float | None = None
    loss_finite: bool = True
    global_grad_norm: float = 0.0
    has_non_finite: bool = False
    layer_stats: list[GradientStats] = field(default_factory=list)
    non_finite_layers: list[str] = field(default_factory=list)
    vanishing_layers: list[str] = field(default_factory=list)
    exploding_layers: list[str] = field(default_factory=list)
    head_supervision: dict[str, dict[str, float | int | bool]] = field(
        default_factory=dict
    )
    head_gradient_health: dict[str, dict[str, float | int | bool]] = field(
        default_factory=dict
    )
    head_coverage: dict[str, dict[str, float | int]] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


class GradientDebugger:
    """Toggleable gradient debugger with layer-level diagnostics.

    When disabled (enabled=False), all methods are no-ops with minimal overhead.
    When enabled, captures detailed statistics about gradient health.

    Attributes:
        enabled: Whether debugging is active
        log_dir: Directory to save diagnostic reports
        logger: Logger instance for output
    """

    _OBS_HEAD_MARKERS = ("ww", "hosp", "cases", "deaths")

    def __init__(
        self,
        enabled: bool = False,
        log_dir: str | Path | None = None,
        vanishing_threshold: float = 1.0e-8,
        exploding_threshold: float = 1.0e2,
        snapshot_top_k: int = 5,
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize gradient debugger.

        Args:
            enabled: Whether to enable gradient debugging
            log_dir: Directory to save diagnostic JSON files
            logger_instance: Logger to use for diagnostic output
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir) if log_dir else None
        self.vanishing_threshold = float(vanishing_threshold)
        self.exploding_threshold = float(exploding_threshold)
        self.snapshot_top_k = int(snapshot_top_k)
        self.logger = logger_instance or logger

        if self.vanishing_threshold < 0:
            raise ValueError(
                "vanishing_threshold must be non-negative, got "
                f"{self.vanishing_threshold}"
            )
        if self.exploding_threshold <= 0:
            raise ValueError(
                "exploding_threshold must be positive, got "
                f"{self.exploding_threshold}"
            )
        if self.snapshot_top_k < 1:
            raise ValueError(
                f"snapshot_top_k must be >= 1, got {self.snapshot_top_k}"
            )

        if self.enabled and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Gradient debugging enabled. Reports will be saved to {self.log_dir}"
            )

    def check_gradients(self, model: nn.Module) -> bool:
        """Check if any model parameters have non-finite gradients.

        Args:
            model: The model to check

        Returns:
            True if non-finite gradients detected, False otherwise
        """
        if not self.enabled:
            return False

        has_non_finite = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_non_finite = True
                    break

        return has_non_finite

    def compute_layer_stats(
        self, name: str, param: nn.Parameter
    ) -> GradientStats | None:
        """Compute gradient statistics for a single parameter.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            GradientStats object or None if parameter has no gradient
        """
        if param.grad is None:
            return None

        grad = param.grad.detach()
        stats = GradientStats(
            name=name,
            shape=list(grad.shape),
            numel=grad.numel(),
        )

        # Count NaN and Inf
        nan_mask = torch.isnan(grad)
        inf_mask = torch.isinf(grad)
        stats.nan_count = int(nan_mask.sum().item())
        stats.inf_count = int(inf_mask.sum().item())
        stats.finite_ratio = float(
            (grad.numel() - stats.nan_count - stats.inf_count) / grad.numel()
        )

        # Compute statistics on finite values only
        finite_grad = grad[torch.isfinite(grad)]
        if finite_grad.numel() > 0:
            stats.mean = float(finite_grad.mean().item())
            stats.std = (
                float(finite_grad.std().item()) if finite_grad.numel() > 1 else 0.0
            )
            stats.min = float(finite_grad.min().item())
            stats.max = float(finite_grad.max().item())

        # Compute norm (will be inf if any non-finite values exist)
        stats.norm = float(grad.norm().item())

        return stats

    def capture_snapshot(
        self,
        model: nn.Module,
        loss: torch.Tensor | None = None,
        step_info: dict | None = None,
        head_supervision: dict[str, dict[str, float | int | bool]] | None = None,
        head_coverage: dict[str, dict[str, float | int]] | None = None,
    ) -> GradientSnapshot:
        """Capture a complete snapshot of gradient state.

        Args:
            model: Model to capture gradients from
            loss: Loss tensor (optional)
            step_info: Dictionary with step/epoch/batch_idx info

        Returns:
            GradientSnapshot with complete diagnostics
        """
        if not self.enabled:
            # Return empty snapshot when disabled
            return GradientSnapshot(step=0, epoch=0, batch_idx=0)

        step_info = step_info or {}
        snapshot = GradientSnapshot(
            step=step_info.get("step", -1),
            epoch=step_info.get("epoch", -1),
            batch_idx=step_info.get("batch_idx", -1),
            loss=float(loss.item()) if loss is not None else None,
            loss_finite=bool(torch.isfinite(loss).item()) if loss is not None else True,
            head_supervision=head_supervision or {},
            head_coverage=head_coverage or {},
        )

        # Compute stats for all parameters with gradients
        layer_stats = []
        non_finite_layers = []
        vanishing_layers = []
        exploding_layers = []
        total_norm_sq = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                stats = self.compute_layer_stats(name, param)
                if stats:
                    layer_stats.append(stats)

                    # Track non-finite layers
                    if stats.nan_count > 0 or stats.inf_count > 0:
                        non_finite_layers.append(name)
                        snapshot.has_non_finite = True
                    elif stats.norm <= self.vanishing_threshold:
                        vanishing_layers.append(name)
                    elif stats.norm >= self.exploding_threshold:
                        exploding_layers.append(name)

                    # Accumulate squared norm for global norm (only finite contributions)
                    if torch.isfinite(param.grad).all():
                        total_norm_sq += stats.norm**2

        snapshot.layer_stats = layer_stats
        snapshot.non_finite_layers = non_finite_layers
        snapshot.vanishing_layers = vanishing_layers
        snapshot.exploding_layers = exploding_layers
        snapshot.head_gradient_health = self._build_head_gradient_health(snapshot)
        snapshot.global_grad_norm = float(total_norm_sq**0.5)

        # Generate summary statistics
        snapshot.summary = self._generate_summary(snapshot)

        return snapshot

    def _generate_summary(self, snapshot: GradientSnapshot) -> dict:
        """Generate high-level summary statistics from snapshot."""
        if not snapshot.layer_stats:
            return {}

        total_params = sum(s.numel for s in snapshot.layer_stats)
        total_nan = sum(s.nan_count for s in snapshot.layer_stats)
        total_inf = sum(s.inf_count for s in snapshot.layer_stats)
        finite_layer_stats = [
            s
            for s in snapshot.layer_stats
            if s.nan_count == 0 and s.inf_count == 0 and s.finite_ratio == 1.0
        ]
        finite_layer_stats.sort(key=lambda s: s.norm)
        bottom_k = [
            {"name": s.name, "norm": s.norm}
            for s in finite_layer_stats[: self.snapshot_top_k]
        ]
        top_k = [
            {"name": s.name, "norm": s.norm}
            for s in finite_layer_stats[-self.snapshot_top_k :][::-1]
        ]
        finite_norms = [s.norm for s in finite_layer_stats]

        # Find layers with highest non-finite ratios
        problematic_layers = [
            {
                "name": s.name,
                "nan_count": s.nan_count,
                "inf_count": s.inf_count,
                "finite_ratio": s.finite_ratio,
            }
            for s in snapshot.layer_stats
            if s.nan_count > 0 or s.inf_count > 0
        ]

        # Sort by severity (lower finite_ratio = more severe)
        problematic_layers.sort(key=lambda x: x["finite_ratio"])

        return {
            "total_parameters": total_params,
            "total_layers_with_grads": len(snapshot.layer_stats),
            "total_nan_values": total_nan,
            "total_inf_values": total_inf,
            "non_finite_layers_count": len(snapshot.non_finite_layers),
            "non_finite_layers": snapshot.non_finite_layers,
            "vanishing_layers_count": len(snapshot.vanishing_layers),
            "exploding_layers_count": len(snapshot.exploding_layers),
            "min_finite_layer_norm": min(finite_norms) if finite_norms else 0.0,
            "median_finite_layer_norm": (
                sorted(finite_norms)[len(finite_norms) // 2] if finite_norms else 0.0
            ),
            "max_finite_layer_norm": max(finite_norms) if finite_norms else 0.0,
            "lowest_norm_layers": bottom_k,
            "highest_norm_layers": top_k,
            "head_supervision": snapshot.head_supervision,
            "head_gradient_health": snapshot.head_gradient_health,
            "head_coverage": snapshot.head_coverage,
            "most_problematic_layers": problematic_layers[:10],  # Top 10
        }

    def refresh_snapshot_summary(self, snapshot: GradientSnapshot) -> GradientSnapshot:
        """Recompute the derived summary after caller-side metadata updates."""
        snapshot.summary = self._generate_summary(snapshot)
        return snapshot

    @classmethod
    def infer_observation_head(cls, layer_name: str) -> str | None:
        """Map parameter names to observation head names when applicable."""
        for head in cls._OBS_HEAD_MARKERS:
            if f".{head}_head." in layer_name or layer_name.startswith(f"{head}_head."):
                return head
        return None

    def _build_head_gradient_health(
        self, snapshot: GradientSnapshot
    ) -> dict[str, dict[str, float | int | bool]]:
        """Summarize expected vs unexpected zero-gradient head behavior."""
        if not snapshot.head_supervision:
            return {}

        vanishing_by_head: dict[str, list[str]] = {head: [] for head in self._OBS_HEAD_MARKERS}
        for layer_name in snapshot.vanishing_layers:
            head = self.infer_observation_head(layer_name)
            if head is not None:
                vanishing_by_head[head].append(layer_name)

        head_health: dict[str, dict[str, float | int | bool]] = {}
        for head, supervision in snapshot.head_supervision.items():
            active = bool(supervision.get("active", False))
            vanishing_layers = vanishing_by_head.get(head, [])
            head_health[head] = {
                "active": active,
                "vanishing_layer_count": len(vanishing_layers),
                "has_vanishing_layers": bool(vanishing_layers),
                "expected_zero": (not active) and bool(vanishing_layers),
                "unexpected_zero": active and bool(vanishing_layers),
                "vanishing_layers": vanishing_layers,
            }

        return head_health

    def build_snapshot_log_data(
        self, snapshot: GradientSnapshot
    ) -> dict[str, float | int]:
        """Build compact numeric metrics suitable for console/W&B logging."""
        summary = snapshot.summary
        log_data: dict[str, float | int] = {
            "grad_snapshot_global_norm": snapshot.global_grad_norm,
            "grad_snapshot_layers_with_grads": int(
                summary.get("total_layers_with_grads", 0)
            ),
            "grad_snapshot_non_finite_layers": int(
                summary.get("non_finite_layers_count", 0)
            ),
            "grad_snapshot_vanishing_layers": int(
                summary.get("vanishing_layers_count", 0)
            ),
            "grad_snapshot_exploding_layers": int(
                summary.get("exploding_layers_count", 0)
            ),
            "grad_snapshot_min_layer_norm": float(
                summary.get("min_finite_layer_norm", 0.0)
            ),
            "grad_snapshot_median_layer_norm": float(
                summary.get("median_finite_layer_norm", 0.0)
            ),
            "grad_snapshot_max_layer_norm": float(
                summary.get("max_finite_layer_norm", 0.0)
            ),
        }
        for head, supervision in snapshot.head_supervision.items():
            log_data[f"grad_snapshot_head_{head}_active"] = int(
                bool(supervision.get("active", False))
            )
            log_data[f"grad_snapshot_head_{head}_n_eff"] = float(
                supervision.get("n_eff", 0.0)
            )
            log_data[f"grad_snapshot_head_{head}_valid_points"] = int(
                supervision.get("valid_points", 0)
            )
            log_data[f"grad_snapshot_head_{head}_valid_series"] = int(
                supervision.get("valid_series", 0)
            )
        for head, health in snapshot.head_gradient_health.items():
            log_data[f"grad_snapshot_head_{head}_expected_zero"] = int(
                bool(health.get("expected_zero", False))
            )
            log_data[f"grad_snapshot_head_{head}_unexpected_zero"] = int(
                bool(health.get("unexpected_zero", False))
            )
            log_data[f"grad_snapshot_head_{head}_vanishing_layers"] = int(
                health.get("vanishing_layer_count", 0)
            )
        for head, coverage in snapshot.head_coverage.items():
            log_data[f"grad_snapshot_head_{head}_pass_rate"] = float(
                coverage.get("pass_rate", 0.0)
            )
            log_data[f"grad_snapshot_head_{head}_zero_when_active_rate"] = float(
                coverage.get("zero_when_active_rate", 0.0)
            )
            log_data[f"grad_snapshot_head_{head}_zero_when_inactive_rate"] = float(
                coverage.get("zero_when_inactive_rate", 0.0)
            )
        return log_data

    def format_snapshot_status(self, snapshot: GradientSnapshot) -> str:
        """Build a concise status line for periodic snapshot captures."""
        summary = snapshot.summary
        highest = summary.get("highest_norm_layers", [])
        lowest = summary.get("lowest_norm_layers", [])
        top_layer = highest[0] if highest else None
        bottom_layer = lowest[0] if lowest else None

        top_desc = (
            f"{top_layer['name']}={top_layer['norm']:.2e}"
            if top_layer is not None
            else "n/a"
        )
        bottom_desc = (
            f"{bottom_layer['name']}={bottom_layer['norm']:.2e}"
            if bottom_layer is not None
            else "n/a"
        )
        status = (
            f"Gradient snapshot @ step {snapshot.step}: "
            f"median={summary.get('median_finite_layer_norm', 0.0):.2e} | "
            f"max={summary.get('max_finite_layer_norm', 0.0):.2e} | "
            f"vanishing={summary.get('vanishing_layers_count', 0)} | "
            f"exploding={summary.get('exploding_layers_count', 0)} | "
            f"low={bottom_desc} | high={top_desc}"
        )
        if snapshot.head_gradient_health:
            head_terms = []
            for head in self._OBS_HEAD_MARKERS:
                if head not in snapshot.head_gradient_health:
                    continue
                health = snapshot.head_gradient_health[head]
                supervision = snapshot.head_supervision.get(head, {})
                active = "on" if bool(health.get("active", False)) else "off"
                zero_tag = ""
                if bool(health.get("unexpected_zero", False)):
                    zero_tag = " unexpected-zero"
                elif bool(health.get("expected_zero", False)):
                    zero_tag = " expected-zero"
                head_terms.append(
                    f"{head}={active}/n={float(supervision.get('n_eff', 0.0)):.0f}{zero_tag}"
                )
            if head_terms:
                status = f"{status} | " + ", ".join(head_terms)
        return status

    def save_report(self, snapshot: GradientSnapshot, step: int | None = None) -> Path:
        """Save snapshot report to JSON file.

        Args:
            snapshot: The snapshot to save
            step: Training step (for filename)

        Returns:
            Path to saved file
        """
        if not self.enabled or not self.log_dir:
            return Path()

        step = step or snapshot.step
        filename = f"grad_debug_step_{step:06d}_epoch_{snapshot.epoch}.json"
        filepath = self.log_dir / filename

        # Convert to dict for JSON serialization
        report_data = asdict(snapshot)

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"Gradient debug report saved to {filepath}")
        return filepath

    def log_summary(
        self, snapshot: GradientSnapshot, level: int = logging.WARNING
    ) -> None:
        """Log a human-readable summary of the snapshot.

        Args:
            snapshot: The snapshot to log
            level: Logging level to use
        """
        if not self.enabled:
            return

        lines = [
            "=" * 60,
            "GRADIENT DEBUG REPORT",
            "=" * 60,
            f"Step: {snapshot.step}, Epoch: {snapshot.epoch}, Batch: {snapshot.batch_idx}",
            f"Loss: {snapshot.loss} (finite: {snapshot.loss_finite})",
            f"Global Grad Norm: {snapshot.global_grad_norm:.4f}",
            f"Has Non-Finite: {snapshot.has_non_finite}",
            "",
            "Summary:",
        ]

        for key, value in snapshot.summary.items():
            if key != "most_problematic_layers":
                lines.append(f"  {key}: {value}")

        if snapshot.non_finite_layers:
            lines.append("")
            lines.append("Non-Finite Layers:")
            for layer in snapshot.non_finite_layers[:20]:  # Limit output
                lines.append(f"  - {layer}")

        lines.append("=" * 60)

        self.logger.log(level, "\n".join(lines))


def create_gradient_debugger(
    config: dict | Mapping | None = None,
    logger_instance: logging.Logger | None = None,
) -> GradientDebugger:
    """Factory function to create GradientDebugger from config.

    Args:
        config: Configuration dictionary with keys:
            - enable_gradient_debug: bool
            - gradient_debug_log_dir: str | None
        logger_instance: Logger to use

    Returns:
        GradientDebugger instance (disabled if config is None)
    """
    if config is None:
        return GradientDebugger(enabled=False, logger_instance=logger_instance)

    enabled = getattr(config, "enable_gradient_debug", False)
    log_dir = getattr(config, "gradient_debug_log_dir", None)
    vanishing_threshold = getattr(config, "gradient_vanishing_threshold", 1.0e-8)
    exploding_threshold = getattr(config, "gradient_exploding_threshold", 1.0e2)
    snapshot_top_k = getattr(config, "gradient_snapshot_top_k", 5)

    return GradientDebugger(
        enabled=enabled,
        log_dir=log_dir,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        snapshot_top_k=snapshot_top_k,
        logger_instance=logger_instance,
    )


# Convenience function for quick checks
def has_non_finite_gradients(model: nn.Module) -> bool:
    """Quick check for non-finite gradients (no overhead from debugger setup).

    Args:
        model: Model to check

    Returns:
        True if any non-finite gradients exist
    """
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return True
    return False
    _OBS_HEAD_MARKERS = ("ww", "hosp", "cases", "deaths")
