#!/usr/bin/env python3
"""
Analyze TensorBoard gradnorm logs for EpiForecaster model training.

This script extracts gradient norm metrics from TensorBoard event files and
provides insights into:
- Component-level learning health (dead vs healthy gradients)
- Training instability (spikes, vanishing, volatility)
- Component balance (is each module contributing?)
- Loss-gradnorm correlations

Usage:
    python scripts/analyze_gradnorm.py <path_to_events_dir>
    python scripts/analyze_gradnorm.py <experiment_name> <run_id>
    python scripts/analyze_gradnorm.py --optuna <trial_path> --json
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.skill_output import SkillOutputBuilder, print_output


# Try to import tensorboard, provide helpful error if missing
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print(
        "Error: tensorboard package required. Install with: pip install tensorboard",
        file=sys.stderr,
    )
    sys.exit(1)


# Flag thresholds (based on Oracle guidance)
DEADNESS_FLOOR = 1e-6
DEADNESS_THRESHOLD_PCT = 5.0  # Flag if >5% of steps below floor
SPIKE_MAD_THRESHOLD = 5.0  # Flag spikes >5 MAD from median
SPIKE_MAD_SEVERE = 7.0  # Severe spike >7 MAD
MOBILITY_GNN_MIN_SHARE = 5.0  # Minimum expected % share
MOBILITY_GNN_MAX_SHARE = 20.0  # Maximum expected % share
ROLLING_WINDOW = 50  # Steps for rolling metrics
REGIME_WINDOW = 10  # Windows for regime shift detection


def ascii_sparkline(values: list[float], width: int = 20) -> str:
    """Generate ASCII sparkline from values."""
    if not values or len(values) < 2:
        return "▬" * width

    # Normalize to 0-1 range
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return "▬" * width

    # Unicode blocks for sparkline: ▁ ▂ ▃ ▄ ▅ ▆ ▇ █
    blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    # Resample to requested width
    indices = np.linspace(0, len(values) - 1, width, dtype=int)
    resampled = [values[i] for i in indices]

    sparkline = []
    for v in resampled:
        normalized = (v - min_val) / range_val
        block_idx = int(normalized * (len(blocks) - 1))
        sparkline.append(blocks[min(block_idx, len(blocks) - 1)])

    return "".join(sparkline)


@dataclass
class ClipDiagnostics:
    """Detailed clip diagnostics."""

    clip_rate_pct: float  # % steps where clipping occurred
    avg_clip_factor: float  # Average factor when clipped
    max_clip_factor: float  # Maximum clip factor observed
    unclipped_rate_pct: float  # % steps with no clipping


@dataclass
class ComponentMetrics:
    """Metrics for a single model component."""

    name: str
    values: list[float]
    steps: list[int]

    # Computed statistics
    median: float = field(init=False)
    p95: float = field(init=False)
    min_val: float = field(init=False)
    max_val: float = field(init=False)
    range_ratio: float = field(init=False)
    deadness_rate: float = field(init=False)
    slope: float = field(init=False)
    log_slope: float = field(init=False)
    spike_count: int = field(init=False)
    spike_count_per_1k: float = field(init=False)
    volatility: float = field(init=False)

    # Share relative to total (time series)
    share_median: float = field(init=False)
    share_values: list[float] = field(init=False, default_factory=list)
    share_cv: float = field(init=False)  # Coefficient of variation
    share_trend_slope: float = field(init=False)
    regime_shifts: list[tuple[int, str]] = field(init=False, default_factory=list)

    # Effective learning signal (grad * LR)
    effective_signal_median: float = field(init=False)

    def __post_init__(self):
        if not self.values:
            self.median = 0.0
            self.p95 = 0.0
            self.min_val = 0.0
            self.max_val = 0.0
            self.range_ratio = 0.0
            self.deadness_rate = 0.0
            self.slope = 0.0
            self.log_slope = 0.0
            self.spike_count = 0
            self.spike_count_per_1k = 0.0
            self.volatility = 0.0
            self.share_median = 0.0
            self.share_values = []
            self.share_cv = 0.0
            self.share_trend_slope = 0.0
            self.regime_shifts = []
            self.effective_signal_median = 0.0
            return

        arr = np.array(self.values)
        steps_arr = np.array(self.steps)

        self.median = float(np.median(arr))
        self.p95 = float(np.percentile(arr, 95))
        self.min_val = float(np.min(arr))
        self.max_val = float(np.max(arr))
        self.range_ratio = (
            self.max_val / max(self.min_val, DEADNESS_FLOOR)
            if self.min_val > 0
            else float("inf")
        )

        # Deadness: % of steps below floor
        below_floor = np.sum(arr < DEADNESS_FLOOR)
        self.deadness_rate = 100.0 * below_floor / len(arr)

        # Slope: linear fit on log scale
        log_vals = np.log(np.maximum(arr, DEADNESS_FLOOR))
        if len(steps_arr) > 1:
            self.log_slope = float(
                np.polyfit(steps_arr, log_vals, 1)[0]
            )  # First-order coefficient
            self.slope = float(np.polyfit(steps_arr, arr, 1)[0])
        else:
            self.log_slope = 0.0
            self.slope = 0.0

        # Spike detection using robust z-score (median/MAD)
        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val))
        mad = max(mad, 1e-9)  # Avoid division by zero
        z_scores = np.abs(arr - median_val) / mad
        self.spike_count = int(np.sum(z_scores > SPIKE_MAD_THRESHOLD))

        # Spike count per 1k steps
        step_range = max(steps_arr[-1] - steps_arr[0], 1)
        self.spike_count_per_1k = 1000.0 * self.spike_count / step_range

        # Volatility: rolling std of log values
        if len(arr) >= ROLLING_WINDOW:
            log_rolling_std = [
                np.std(log_vals[max(0, i - ROLLING_WINDOW) : i + 1])
                for i in range(len(log_vals))
            ]
            self.volatility = float(np.mean(log_rolling_std))
        else:
            self.volatility = float(np.std(log_vals))

        self.share_median = 0.0  # Will be set by caller
        self.share_values = []
        self.share_cv = 0.0
        self.share_trend_slope = 0.0
        self.regime_shifts = []
        self.effective_signal_median = 0.0


@dataclass
class GradnormAnalysis:
    """Complete gradnorm analysis results."""

    # Component metrics
    total_preclip: ComponentMetrics
    mobility_gnn: ComponentMetrics
    forecaster_head: ComponentMetrics
    other: ComponentMetrics
    clipped_total: ComponentMetrics

    # Learning rate (for effective signal computation)
    lr_median: float = 0.0
    lr_values: list[float] = field(default_factory=list)

    # Clip diagnostics
    clip_diagnostics: ClipDiagnostics | None = None

    # Loss correlation
    loss_gradnorm_corr: float | None = None
    loss_gradnorm_lag_corr: dict[int, float] = field(default_factory=dict)

    # Per-component loss correlation
    component_loss_corr: dict[str, float] = field(default_factory=dict)

    # Flags
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "components": {
                "Total_PreClip": self._component_to_dict(self.total_preclip),
                "MobilityGNN": self._component_to_dict(self.mobility_gnn),
                "ForecasterHead": self._component_to_dict(self.forecaster_head),
                "Other": self._component_to_dict(self.other),
                "Clipped_Total": self._component_to_dict(self.clipped_total),
            },
            "learning_rate": {"median": self.lr_median},
            "clip_diagnostics": (
                {
                    "clip_rate_pct": self.clip_diagnostics.clip_rate_pct,
                    "avg_clip_factor": self.clip_diagnostics.avg_clip_factor,
                    "max_clip_factor": self.clip_diagnostics.max_clip_factor,
                    "unclipped_rate_pct": self.clip_diagnostics.unclipped_rate_pct,
                }
                if self.clip_diagnostics
                else None
            ),
            "loss_correlation": self.loss_gradnorm_corr,
            "lag_correlation": self.loss_gradnorm_lag_corr,
            "component_loss_correlation": self.component_loss_corr,
            "flags": self.flags,
        }

    @staticmethod
    def _component_to_dict(c: ComponentMetrics) -> dict:
        return {
            "median": c.median,
            "p95": c.p95,
            "min": c.min_val,
            "max": c.max_val,
            "range_ratio": c.range_ratio,
            "deadness_rate_pct": c.deadness_rate,
            "slope": c.slope,
            "log_slope": c.log_slope,
            "spike_count": c.spike_count,
            "spike_count_per_1k": c.spike_count_per_1k,
            "volatility": c.volatility,
            "share_median_pct": c.share_median,
            "share_cv": c.share_cv,
            "share_trend_slope": c.share_trend_slope,
            "effective_signal_median": c.effective_signal_median,
            "regime_shifts": [{"step": s, "direction": d} for s, d in c.regime_shifts],
        }


def load_scalars(
    ea: event_accumulator.EventAccumulator, tag: str
) -> tuple[list[int], list[float]]:
    """Load scalar events as (steps, values) tuples."""
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def compute_clip_diagnostics(
    clipped: ComponentMetrics, total: ComponentMetrics, clip_threshold: float = 1.0
) -> ClipDiagnostics:
    """Compute detailed clip diagnostics."""
    if not total.values or not clipped.values:
        return ClipDiagnostics(0.0, 0.0, 0.0, 0.0)

    clipped_steps = 0
    clip_factors = []
    unclipped_steps = 0

    for c_val, t_val in zip(clipped.values, total.values, strict=False):
        if t_val > DEADNESS_FLOOR:
            ratio = c_val / t_val
            if ratio < clip_threshold - 0.01:  # Allow small numerical tolerance
                clipped_steps += 1
                clip_factors.append(clip_threshold / t_val)
            else:
                unclipped_steps += 1

    total_steps = clipped_steps + unclipped_steps
    if total_steps == 0:
        return ClipDiagnostics(0.0, 0.0, 0.0, 0.0)

    clip_rate = 100.0 * clipped_steps / total_steps
    avg_factor = float(np.mean(clip_factors)) if clip_factors else 0.0
    max_factor = float(np.max(clip_factors)) if clip_factors else 0.0
    unclipped_rate = 100.0 * unclipped_steps / total_steps

    return ClipDiagnostics(clip_rate, avg_factor, max_factor, unclipped_rate)


def compute_shares(
    components: dict[str, ComponentMetrics], total: ComponentMetrics
) -> None:
    """Compute component shares relative to total (time series)."""
    for name, comp in components.items():
        if name == "Total_PreClip" or not total.values:
            comp.share_median = 0.0
            comp.share_values = []
            comp.share_cv = 0.0
            comp.share_trend_slope = 0.0
            comp.regime_shifts = []
            continue

        # Align by index
        shares = []
        for c_val, t_val in zip(comp.values, total.values, strict=False):
            if t_val > DEADNESS_FLOOR:
                shares.append(100.0 * c_val / t_val)
            else:
                shares.append(0.0)

        comp.share_values = shares
        comp.share_median = float(np.median(shares)) if shares else 0.0

        # Coefficient of variation of share
        if len(shares) > 1 and np.mean(shares) > 0:
            comp.share_cv = float(np.std(shares) / np.mean(shares))
        else:
            comp.share_cv = 0.0

        # Trend slope of share
        if len(shares) >= REGIME_WINDOW:
            comp.share_trend_slope = float(np.polyfit(range(len(shares)), shares, 1)[0])
        else:
            comp.share_trend_slope = 0.0

        # Regime shift detection
        comp.regime_shifts = detect_regime_shifts(shares, total.steps)


def detect_regime_shifts(
    shares: list[float], steps: list[int], window: int = REGIME_WINDOW
) -> list[tuple[int, str]]:
    """Detect regime shifts in component share over time."""
    if len(shares) < window * 2:
        return []

    shifts = []
    for i in range(window, len(shares) - window):
        before_mean = np.mean(shares[i - window : i])
        after_mean = np.mean(shares[i : i + window])

        # Detect significant shift (>20% relative change)
        if before_mean > 0:
            rel_change = abs(after_mean - before_mean) / before_mean
            if rel_change > 0.2:
                direction = "INCREASING" if after_mean > before_mean else "DECREASING"
                shifts.append((steps[i], direction))

    return shifts


def compute_effective_signal(comp: ComponentMetrics, lr_values: list[float]) -> None:
    """Compute effective learning signal (grad * LR)."""
    if not comp.values or not lr_values:
        comp.effective_signal_median = 0.0
        return

    # Use median LR
    lr_median = np.median(lr_values)

    # Effective signal = grad * LR
    effective = [g * lr_median for g in comp.values]
    comp.effective_signal_median = float(np.median(effective))


def compute_loss_correlation(
    loss_steps: list[int],
    loss_values: list[float],
    gradnorm: ComponentMetrics,
    max_lag: int = 5,
) -> tuple[float | None, dict[int, float]]:
    """Compute loss-gradnorm correlation (rolling and lagged)."""
    if not loss_values or not gradnorm.values:
        return None, {}

    # Simple correlation on aligned data
    min_len = min(len(loss_values), len(gradnorm.values))
    if min_len < 10:
        return None, {}

    loss_arr = np.array(loss_values[:min_len])
    grad_arr = np.array(gradnorm.values[:min_len])

    # Log-scale gradnorm for correlation
    log_grad = np.log(np.maximum(grad_arr, DEADNESS_FLOOR))

    # Pearson correlation
    corr_matrix = np.corrcoef(loss_arr, log_grad)
    corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else None

    # Lagged correlation
    lag_corrs = {}
    for lag in range(1, min(max_lag + 1, min_len // 2)):
        if lag >= min_len:
            break
        lagged_grad = log_grad[:-lag]
        aligned_loss = loss_arr[lag:]
        if len(lagged_grad) > 0:
            lag_corr = np.corrcoef(aligned_loss, lagged_grad)[0, 1]
            if not np.isnan(lag_corr):
                lag_corrs[lag] = float(lag_corr)

    return corr, lag_corrs


def compute_component_loss_correlation(
    loss_steps: list[int],
    loss_values: list[float],
    component: ComponentMetrics,
) -> float | None:
    """Compute correlation between component share and loss."""
    if not loss_values or not component.share_values:
        return None

    min_len = min(len(loss_values), len(component.share_values))
    if min_len < 10:
        return None

    loss_arr = np.array(loss_values[:min_len])
    share_arr = np.array(component.share_values[:min_len])

    corr_matrix = np.corrcoef(loss_arr, share_arr)
    if corr_matrix.shape == (2, 2):
        return float(corr_matrix[0, 1])
    return None


def analyze_events(event_dir: str | Path) -> tuple[GradnormAnalysis, list[str]]:
    """Analyze gradnorm from TensorBoard event directory."""
    event_dir = Path(event_dir)
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory not found: {event_dir}")

    # Load event files
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()

    # Load all gradnorm scalars
    tags = [
        "GradNorm/Total_PreClip",
        "GradNorm/MobilityGNN",
        "GradNorm/ForecasterHead",
        "GradNorm/Other",
        "GradNorm/Clipped_Total",
    ]

    components = {}
    for tag in tags:
        name = tag.replace("GradNorm/", "")
        steps, values = load_scalars(ea, tag)
        components[name] = ComponentMetrics(name=name, values=values, steps=steps)

    # Load learning rate
    lr_steps, lr_values = load_scalars(ea, "Learning_Rate/step")
    if not lr_values:
        lr_steps, lr_values = load_scalars(ea, "Learning_Rate")
    lr_median = float(np.median(lr_values)) if lr_values else 0.0

    # Load loss for correlation
    loss_steps, loss_values = load_scalars(ea, "Loss/Train_step")
    if not loss_values:
        loss_steps, loss_values = load_scalars(ea, "Loss/Train")

    # Compute component shares (with time series)
    total = components["Total_PreClip"]
    compute_shares(components, total)

    # Compute effective signal for each component
    for name, comp in components.items():
        if name in ["MobilityGNN", "ForecasterHead", "Other"]:
            compute_effective_signal(comp, lr_values)

    # Clip diagnostics
    clip_diagnostics = compute_clip_diagnostics(components["Clipped_Total"], total)

    # Loss correlation
    loss_corr, lag_corrs = compute_loss_correlation(loss_steps, loss_values, total)

    # Per-component loss correlation
    component_loss_corr = {}
    for name, comp in [
        ("MobilityGNN", components["MobilityGNN"]),
        ("ForecasterHead", components["ForecasterHead"]),
    ]:
        corr = compute_component_loss_correlation(loss_steps, loss_values, comp)
        if corr is not None:
            component_loss_corr[name] = corr

    analysis = GradnormAnalysis(
        total_preclip=total,
        mobility_gnn=components["MobilityGNN"],
        forecaster_head=components["ForecasterHead"],
        other=components["Other"],
        clipped_total=components["Clipped_Total"],
        lr_median=lr_median,
        lr_values=lr_values,
        clip_diagnostics=clip_diagnostics,
        loss_gradnorm_corr=loss_corr,
        loss_gradnorm_lag_corr=lag_corrs,
        component_loss_corr=component_loss_corr,
    )

    # Generate flags
    flags = generate_flags(analysis)
    analysis.flags = flags
    return analysis, flags


def generate_flags(analysis: GradnormAnalysis) -> list[str]:
    """Generate warning flags based on thresholds."""
    flags = []

    # Check for dead components
    for name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
        ("Other", analysis.other),
    ]:
        if comp.deadness_rate > DEADNESS_THRESHOLD_PCT:
            flags.append(
                f"DEAD [{name}]: {comp.deadness_rate:.1f}% steps below floor "
                f"(threshold: {DEADNESS_THRESHOLD_PCT}%)"
            )

    # Check for spike activity
    if analysis.total_preclip.spike_count > 0:
        flags.append(
            f"SPIKES: {analysis.total_preclip.spike_count} spikes detected "
            f"({analysis.total_preclip.spike_count_per_1k:.1f} per 1k steps)"
        )

    # Check component balance
    gnn_share = analysis.mobility_gnn.share_median
    if gnn_share < MOBILITY_GNN_MIN_SHARE:
        flags.append(
            f"IMBALANCE: MobilityGNN under-contributing at {gnn_share:.1f}% "
            f"(expected: {MOBILITY_GNN_MIN_SHARE}-{MOBILITY_GNN_MAX_SHARE}%)"
        )
    elif gnn_share > MOBILITY_GNN_MAX_SHARE:
        flags.append(
            f"IMBALANCE: MobilityGNN dominating at {gnn_share:.1f}% "
            f"(expected: {MOBILITY_GNN_MIN_SHARE}-{MOBILITY_GNN_MAX_SHARE}%)"
        )

    # Check head dominance
    head_share = analysis.forecaster_head.share_median
    if head_share > 95:
        flags.append(
            f"Dominance: ForecasterHead at {head_share:.1f}% "
            f"(may indicate component imbalance)"
        )

    # Check volatility
    if analysis.total_preclip.volatility > 1.0:
        flags.append(
            f"VOLATILITY: High gradnorm volatility "
            f"({analysis.total_preclip.volatility:.3f})"
        )

    # Check share stability
    for name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
    ]:
        if comp.share_cv > 0.5:
            flags.append(
                f"UNSTABLE [{name}]: Share CV {comp.share_cv:.2f} "
                f"(unstable contribution over time)"
            )

    # Check for regime shifts
    for comp_name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
    ]:
        if comp.regime_shifts:
            shift_str = ", ".join(f"{s} ({d})" for s, d in comp.regime_shifts[:3])
            flags.append(f"REGIME SHIFT [{comp_name}]: {shift_str}")

    # Check clip diagnostics
    if analysis.clip_diagnostics and analysis.clip_diagnostics.clip_rate_pct > 30:
        flags.append(
            f"CLIPPING: {analysis.clip_diagnostics.clip_rate_pct:.1f}% of steps clipped "
            f"(indicates aggressive clipping)"
        )

    return flags


def print_analysis(analysis: GradnormAnalysis) -> None:
    """Print formatted analysis to console."""
    print("=" * 80)
    print("GRADNORM ANALYSIS")
    print("=" * 80)

    # Component summary table
    print("\nPER-COMPONENT STATISTICS")
    print("-" * 80)
    print(
        f"{'Component':<16} {'Median':>10} {'95th':>10} {'Share %':>9} "
        f"{'CV':>6} {'Dead %':>8} {'Signal':>10}"
    )
    print("-" * 80)

    for name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
        ("Other", analysis.other),
    ]:
        sparkline = ascii_sparkline(
            comp.share_values[-20:] if comp.share_values else []
        )
        print(
            f"{name:<16} {comp.median:10.6f} {comp.p95:10.6f} "
            f"{comp.share_median:8.1f}% {comp.share_cv:5.2f} {comp.deadness_rate:7.1f}% "
            f"{comp.effective_signal_median:9.2e}"
        )
        if comp.share_values:
            print(
                f"  Trend: {sparkline:<60} "
                f"({comp.share_values[0]:.1f}% → {comp.share_values[-1]:.1f}%)"
            )

    print("-" * 80)

    # Clip diagnostics
    if analysis.clip_diagnostics:
        print("\nCLIP DIAGNOSTICS")
        print("-" * 40)
        print(
            f"Clip rate:     {analysis.clip_diagnostics.clip_rate_pct:6.2f}% "
            f"of steps clipped"
        )
        print(
            f"Avg factor:    {analysis.clip_diagnostics.avg_clip_factor:6.4f}x "
            f"when clipped"
        )
        print(
            f"Max factor:    {analysis.clip_diagnostics.max_clip_factor:6.4f}x "
            f"max reduction"
        )
        print(
            f"Unclipped:     {analysis.clip_diagnostics.unclipped_rate_pct:6.2f}% "
            f"of steps unclipped"
        )

    # Loss correlation
    if analysis.loss_gradnorm_corr is not None:
        print("\nLOSS-GRADNORM CORRELATION")
        print("-" * 40)
        print(f"Total correlation: {analysis.loss_gradnorm_corr:+.4f}")

        if analysis.component_loss_corr:
            print("Per-component:")
            for name, corr in analysis.component_loss_corr.items():
                print(f"  {name}: {corr:+.4f}")

        if analysis.loss_gradnorm_lag_corr:
            best_lag = max(
                analysis.loss_gradnorm_lag_corr.items(),
                key=lambda x: abs(x[1]),
            )
            print(f"Best lag ({best_lag[0]} steps): {best_lag[1]:+.4f}")

    # Temporal trends
    print("\nTEMPORAL TRENDS")
    print("-" * 40)
    print(f"Learning rate:  {analysis.lr_median:.2e} (median)")
    print(
        f"Total gradnorm: {analysis.total_preclip.min_val:.6f} → "
        f"{analysis.total_preclip.max_val:.6f}"
    )
    print(
        f"  Log slope: {analysis.total_preclip.log_slope:+.6f} "
        f"({'decreasing' if analysis.total_preclip.log_slope < 0 else 'increasing'})"
    )

    for name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
    ]:
        if comp.share_trend_slope != 0:
            direction = "increasing" if comp.share_trend_slope > 0 else "decreasing"
            print(
                f"{name} share: {direction} ({comp.share_trend_slope:+.4f}% per step)"
            )

    # Regime shifts
    for name, comp in [
        ("MobilityGNN", analysis.mobility_gnn),
        ("ForecasterHead", analysis.forecaster_head),
    ]:
        if comp.regime_shifts:
            shifts_str = ", ".join(f"{s} ({d})" for s, d in comp.regime_shifts[:3])
            print(f"{name} regime shifts: {shifts_str}")

    # Flags
    if analysis.flags:
        print("\nFLAGS")
        print("-" * 80)
        for flag in analysis.flags:
            print(f"  ⚠ {flag}")
    else:
        print("\nNo flags - all metrics within normal range.")

    print("\n" + "=" * 80)


def resolve_event_path(
    path: str,
    optuna: bool = False,
    experiment: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Resolve event directory path from various input formats."""
    if optuna:
        # Optuna trial path - use directly
        return Path(path)

    if experiment and run_id:
        # Experiment/run format: outputs/training/<experiment>/<run_id>/
        base = Path("outputs/training") / experiment / run_id
    elif Path(path).exists():
        # Direct path
        return Path(path)
    else:
        # Try as experiment name only (find most recent run)
        base = Path("outputs/training") / path
        if base.exists():
            runs = [d for d in base.iterdir() if d.is_dir()]
            if runs:
                base = sorted(runs, reverse=True)[0]
            else:
                base = base / "unknown"
        else:
            raise FileNotFoundError(f"Cannot resolve path: {path}")

    return base


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard gradnorm logs for EpiForecaster"
    )
    parser.add_argument(
        "path",
        help="Path to event directory, experiment name, or trial path",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Treat path as Optuna trial directory",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment name (if using experiment/run format)",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID (if using experiment/run format)",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output results as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (no indentation)"
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="gradnorm-analyze",
        input_path=args.path,
    )

    try:
        event_path = resolve_event_path(
            args.path,
            optuna=args.optuna,
            experiment=args.experiment,
            run_id=args.run_id,
        )
        analysis, flags = analyze_events(event_path)

        data = analysis.to_dict()
        builder.warnings = flags

        output = builder.success(data)

        if args.text:
            print_analysis(analysis)
        else:
            indent = 0 if args.compact else 2
            print_output(output, indent=indent)

        return 0

    except FileNotFoundError as e:
        print_output(builder.error("FileNotFoundError", str(e)))
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    sys.exit(main())
