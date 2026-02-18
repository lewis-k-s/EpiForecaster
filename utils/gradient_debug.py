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

    def __init__(
        self,
        enabled: bool = False,
        log_dir: str | Path | None = None,
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
        self.logger = logger_instance or logger

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
            stats.std = float(finite_grad.std().item())
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
        )

        # Compute stats for all parameters with gradients
        layer_stats = []
        non_finite_layers = []
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

                    # Accumulate squared norm for global norm (only finite contributions)
                    if torch.isfinite(param.grad).all():
                        total_norm_sq += stats.norm**2

        snapshot.layer_stats = layer_stats
        snapshot.non_finite_layers = non_finite_layers
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
            "most_problematic_layers": problematic_layers[:10],  # Top 10
        }

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

    return GradientDebugger(
        enabled=enabled,
        log_dir=log_dir,
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
