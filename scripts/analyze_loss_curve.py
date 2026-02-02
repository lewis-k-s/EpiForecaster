#!/usr/bin/env python3
"""
Analyze TensorBoard training loss curves for EpiForecaster model training.

This script extracts training loss metrics from TensorBoard event files and
provides insights into:
- Loss spikes (frequency, severity, timing)
- Training stability (volatility, variance)
- Convergence health (trends, plateaus)
- Curriculum transition effects
- Correlations with gradnorm, data loading, and sparsity

Usage:
    python scripts/analyze_loss_curve.py <path_to_events_dir>
    python scripts/analyze_loss_curve.py <experiment_name> --text
    python scripts/analyze_loss_curve.py <path> --diagnose
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


# Flag thresholds
SPIKE_HIGH_THRESHOLD = 3.0  # Multiple of median for "high" spike
SPIKE_SEVERE_THRESHOLD = 10.0  # Multiple of median for "severe" spike
SPIKE_PCT_WARNING = 5.0  # Warn if >5% of steps are spikes
CV_WARNING_THRESHOLD = 1.0  # Coefficient of variation warning
ROLLING_VOL_WARNING = 0.5  # Rolling volatility warning
RECOVERY_WARNING_THRESHOLD = 50.0  # Warn if <50% of spikes recover
PLATEAU_WINDOW = 20  # Steps to detect plateau
PLATEAU_VAR_THRESHOLD = 0.01  # Variance threshold for plateau
ROLLING_WINDOW = 50  # Rolling window for volatility
RECOVERY_WINDOW = 10  # Steps to check for spike recovery
CURRICULUM_WINDOW = 20  # Steps before/after epoch boundary to check
SPIKE_GROUP_GAP_THRESHOLD = 50  # Max step gap for consecutive spike grouping


def ascii_sparkline(values: list[float], width: int = 40) -> str:
    """Generate ASCII sparkline from values."""
    if not values or len(values) < 2:
        return "▬" * width

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return "▬" * width

    blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    indices = np.linspace(0, len(values) - 1, width, dtype=int)
    resampled = [values[i] for i in indices]

    sparkline = []
    for v in resampled:
        normalized = (v - min_val) / range_val
        block_idx = int(normalized * (len(blocks) - 1))
        sparkline.append(blocks[min(block_idx, len(blocks) - 1)])

    return "".join(sparkline)


@dataclass
class SpikeAnalysis:
    """Loss spike analysis results."""

    high_spikes: list[tuple[int, float]]  # (step, value) for spikes >3x median
    severe_spikes: list[tuple[int, float]]  # (step, value) for spikes >10x median
    high_spike_pct: float
    severe_spike_pct: float
    consecutive_groups: list[list[tuple[int, float]]]
    recovery_rate: float  # % of spikes that recover
    max_upward_jump: float  # Max increase between consecutive steps
    max_downward_drop: float  # Max decrease between consecutive steps
    max_upward_step: int  # Step where max upward jump occurred


@dataclass
class StabilityMetrics:
    """Training stability metrics."""

    rolling_volatility: float  # Std of rolling window
    coefficient_of_variation: float  # std/mean
    max_drawdown: float  # Max drop from peak
    max_drawup: float  # Max rise from trough


@dataclass
class ConvergenceAnalysis:
    """Convergence analysis results."""

    log_loss_slope: float  # Trend of log(loss)
    convergence_rate: float  # Loss reduction per 1k steps
    plateaus: list[tuple[int, int]]  # (start_step, end_step) of plateaus
    initial_loss: float
    final_loss: float
    reduction_pct: float


@dataclass
class CurriculumTransition:
    """Single curriculum transition analysis."""

    epoch: int
    epoch_step: int  # Step at epoch boundary
    loss_before: float  # Mean loss before transition
    loss_after: float  # Mean loss after transition
    loss_delta: float  # Change in loss
    spike_count_near: int  # Spikes near transition
    sparsity_delta: float | None  # Change in sparsity if available


@dataclass
class CurriculumAnalysis:
    """Curriculum transition analysis."""

    transitions: list[CurriculumTransition]
    total_transitions: int
    problematic_transitions: int  # Transitions with significant loss increase
    avg_loss_delta: float


@dataclass
class Correlations:
    """Correlations with other metrics."""

    loss_gradnorm_corr: float | None
    loss_dataload_corr: float | None
    loss_sparsity_corr: float | None


@dataclass
class LossCurveAnalysis:
    """Complete loss curve analysis results."""

    # Basic statistics
    steps: list[int]
    values: list[float]
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float

    # Analysis components
    spikes: SpikeAnalysis
    stability: StabilityMetrics
    convergence: ConvergenceAnalysis
    curriculum: CurriculumAnalysis | None
    correlations: Correlations

    # Flags
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "statistics": {
                "steps": len(self.steps),
                "min": self.min_val,
                "max": self.max_val,
                "mean": self.mean_val,
                "median": self.median_val,
                "std": self.std_val,
                "cv": self.stability.coefficient_of_variation,
            },
            "spikes": {
                "high_count": len(self.spikes.high_spikes),
                "high_pct": self.spikes.high_spike_pct,
                "severe_count": len(self.spikes.severe_spikes),
                "severe_pct": self.spikes.severe_spike_pct,
                "recovery_rate": self.spikes.recovery_rate,
                "max_upward_jump": self.spikes.max_upward_jump,
                "max_downward_drop": self.spikes.max_downward_drop,
                "first_10_spikes": [
                    {"step": s, "value": v, "x_median": v / self.median_val}
                    for s, v in self.spikes.high_spikes[:10]
                ],
            },
            "stability": {
                "rolling_volatility": self.stability.rolling_volatility,
                "coefficient_of_variation": self.stability.coefficient_of_variation,
                "max_drawdown": self.stability.max_drawdown,
                "max_drawup": self.stability.max_drawup,
            },
            "convergence": {
                "log_loss_slope": self.convergence.log_loss_slope,
                "convergence_rate": self.convergence.convergence_rate,
                "plateaus": [
                    {"start": s, "end": e} for s, e in self.convergence.plateaus
                ],
                "initial_loss": self.convergence.initial_loss,
                "final_loss": self.convergence.final_loss,
                "reduction_pct": self.convergence.reduction_pct,
            },
            "curriculum": (
                {
                    "transitions": [
                        {
                            "epoch": t.epoch,
                            "epoch_step": t.epoch_step,
                            "loss_before": t.loss_before,
                            "loss_after": t.loss_after,
                            "loss_delta": t.loss_delta,
                            "spike_count_near": t.spike_count_near,
                            "sparsity_delta": t.sparsity_delta,
                        }
                        for t in self.curriculum.transitions
                    ],
                    "total_transitions": self.curriculum.total_transitions,
                    "problematic_transitions": self.curriculum.problematic_transitions,
                    "avg_loss_delta": self.curriculum.avg_loss_delta,
                }
                if self.curriculum
                else None
            ),
            "correlations": {
                "loss_gradnorm": self.correlations.loss_gradnorm_corr,
                "loss_dataload": self.correlations.loss_dataload_corr,
                "loss_sparsity": self.correlations.loss_sparsity_corr,
            },
            "flags": self.flags,
        }


def load_scalars(
    ea: event_accumulator.EventAccumulator, tag: str
) -> tuple[list[int], list[float]]:
    """Load scalar events as (steps, values) tuples."""
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def align_by_step(
    steps_a: list[int], values_a: list[int | float],
    steps_b: list[int], values_b: list[int | float]
) -> tuple[np.ndarray, np.ndarray]:
    """Align two time series by step using interpolation.

    Returns aligned values for steps_a (interpolating from b).
    """
    if not steps_a or not steps_b:
        return np.array([]), np.array([])

    steps_a_arr = np.array(steps_a)
    values_a_arr = np.array(values_a)
    steps_b_arr = np.array(steps_b)
    values_b_arr = np.array(values_b)

    # Interpolate values_b onto steps_a
    aligned_b = np.interp(steps_a_arr, steps_b_arr, values_b_arr, left=np.nan, right=np.nan)

    # Filter out NaN values
    valid_mask = ~np.isnan(aligned_b)
    return values_a_arr[valid_mask], aligned_b[valid_mask]


def analyze_spikes(
    steps: list[int], values: list[float], median: float
) -> SpikeAnalysis:
    """Analyze loss spikes."""
    arr = np.array(values)
    high_threshold = median * SPIKE_HIGH_THRESHOLD
    severe_threshold = median * SPIKE_SEVERE_THRESHOLD

    # Find spikes
    high_spikes = [
        (s, v) for s, v in zip(steps, values) if v > high_threshold and v < severe_threshold
    ]
    severe_spikes = [(s, v) for s, v in zip(steps, values) if v >= severe_threshold]

    # Combine for convenience
    all_spikes = sorted(high_spikes + severe_spikes, key=lambda x: x[0])

    # Consecutive spike groups - use actual step spacing
    # Group spikes that are close together (within threshold steps)
    consecutive_groups = []
    if len(all_spikes) > 1:
        current_group = [all_spikes[0]]
        for s, v in all_spikes[1:]:
            # Check if this spike is close to the previous one
            if s - current_group[-1][0] <= SPIKE_GROUP_GAP_THRESHOLD:
                current_group.append((s, v))
            else:
                if current_group:
                    consecutive_groups.append(current_group)
                current_group = [(s, v)]
        if current_group:
            consecutive_groups.append(current_group)

    # Recovery rate: % of spikes that recover (loss returns to <2x median within window)
    # FIXED: Use step-to-index mapping for O(n) lookup instead of O(n²) steps.index()
    step_to_idx = {s: i for i, s in enumerate(steps)}
    recoveries = 0
    for s, _ in all_spikes:
        if s not in step_to_idx:
            continue
        idx = step_to_idx[s]
        # Look ahead in the loss array for recovery
        for i in range(idx + 1, min(idx + RECOVERY_WINDOW, len(values))):
            if values[i] < median * 2:
                recoveries += 1
                break

    recovery_rate = 100.0 * recoveries / len(all_spikes) if all_spikes else 100.0

    # Max jumps between consecutive steps
    diffs = np.diff(arr)
    max_upward_jump = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    max_downward_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    max_upward_step = int(np.argmax(diffs)) if len(diffs) > 0 else 0

    return SpikeAnalysis(
        high_spikes=high_spikes,
        severe_spikes=severe_spikes,
        high_spike_pct=100.0 * len(high_spikes) / len(steps) if steps else 0.0,
        severe_spike_pct=100.0 * len(severe_spikes) / len(steps) if steps else 0.0,
        consecutive_groups=consecutive_groups,
        recovery_rate=recovery_rate,
        max_upward_jump=max_upward_jump,
        max_downward_drop=max_downward_drop,
        max_upward_step=steps[max_upward_step] if max_upward_step < len(steps) else 0,
    )


def analyze_stability(steps: list[int], values: list[float]) -> StabilityMetrics:
    """Analyze training stability."""
    arr = np.array(values)

    # Rolling volatility
    if len(arr) >= ROLLING_WINDOW:
        rolling_std = [
            np.std(arr[max(0, i - ROLLING_WINDOW) : i + 1]) for i in range(len(arr))
        ]
        rolling_volatility = float(np.mean(rolling_std))
    else:
        rolling_volatility = float(np.std(arr))

    # FIXED: Coefficient of variation - use std/mean, not rolling_volatility/mean
    std_val = float(np.std(arr))
    mean_val = float(np.mean(arr))
    cv = std_val / mean_val if mean_val > 0 else float("inf")

    # Max drawdown (peak to trough)
    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = cumulative_max - arr
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Max drawup (trough to peak)
    cumulative_min = np.minimum.accumulate(arr)
    drawups = arr - cumulative_min
    max_drawup = float(np.max(drawups)) if len(drawups) > 0 else 0.0

    return StabilityMetrics(
        rolling_volatility=rolling_volatility,
        coefficient_of_variation=cv,
        max_drawdown=max_drawdown,
        max_drawup=max_drawup,
    )


def analyze_convergence(
    steps: list[int], values: list[float]
) -> ConvergenceAnalysis:
    """Analyze convergence behavior."""
    steps_arr = np.array(steps)
    arr = np.array(values)

    # Log-loss slope
    log_arr = np.log(np.maximum(arr, 1e-9))
    if len(steps_arr) > 1:
        log_slope = float(np.polyfit(steps_arr, log_arr, 1)[0])
    else:
        log_slope = 0.0

    # Convergence rate (loss reduction per 1k steps)
    if len(arr) > 1:
        convergence_rate = 1000.0 * (arr[-1] - arr[0]) / (steps_arr[-1] - steps_arr[0])
    else:
        convergence_rate = 0.0

    # Detect plateaus (extended periods of low variance)
    plateaus = []
    if len(arr) >= PLATEAU_WINDOW:
        i = 0
        while i <= len(arr) - PLATEAU_WINDOW:
            window = arr[i : i + PLATEAU_WINDOW]
            window_var = float(np.var(window))
            window_mean = float(np.mean(window))
            if window_mean > 0 and window_var < PLATEAU_VAR_THRESHOLD * window_mean:
                # Extend plateau
                j = i + PLATEAU_WINDOW
                while j <= len(arr) - PLATEAU_WINDOW:
                    next_window = arr[j : j + PLATEAU_WINDOW]
                    next_var = float(np.var(next_window))
                    if next_var < PLATEAU_VAR_THRESHOLD * window_mean:
                        j += 1
                    else:
                        break
                plateaus.append((steps[i], steps[min(j, len(arr)) - 1]))
                i = j
            else:
                i += 1

    return ConvergenceAnalysis(
        log_loss_slope=log_slope,
        convergence_rate=convergence_rate,
        plateaus=plateaus,
        initial_loss=float(arr[0]) if len(arr) > 0 else 0.0,
        final_loss=float(arr[-1]) if len(arr) > 0 else 0.0,
        reduction_pct=(
            100.0 * (arr[0] - arr[-1]) / arr[0] if len(arr) > 0 and arr[0] > 0 else 0.0
        ),
    )


def analyze_curriculum(
    loss_steps: list[int],
    loss_values: list[float],
    spike_steps: set[int],
    epoch_steps: list[int] | None = None,
    sparsity_steps: list[int] | None = None,
    sparsity_values: list[float] | None = None,
) -> CurriculumAnalysis | None:
    """Analyze curriculum transition effects.

    Args:
        loss_steps: Training loss step indices
        loss_values: Training loss values
        spike_steps: Set of step indices where spikes occurred
        epoch_steps: Step indices at epoch boundaries (if available)
        sparsity_steps: Step indices for sparsity metric (if available)
        sparsity_values: Sparsity values (if available)
    """
    if not epoch_steps:
        return None

    # Build step-to-value mappings for quick lookup
    loss_dict = {s: v for s, v in zip(loss_steps, loss_values)}
    sparsity_dict = {s: v for s, v in zip(sparsity_steps, sparsity_values)} if sparsity_steps else None

    transitions = []
    problematic_count = 0
    loss_deltas = []

    for i, epoch_step in enumerate(epoch_steps):
        # Get window before and after epoch boundary
        before_start = epoch_step - CURRICULUM_WINDOW
        before_end = epoch_step - 1
        after_start = epoch_step + 1
        after_end = epoch_step + CURRICULUM_WINDOW

        # Collect loss values in windows
        before_losses = [loss_dict.get(s) for s in range(before_start, before_end + 1) if s in loss_dict]
        after_losses = [loss_dict.get(s) for s in range(after_start, after_end + 1) if s in loss_dict]

        # Filter None values
        before_losses = [v for v in before_losses if v is not None]
        after_losses = [v for v in after_losses if v is not None]

        if not before_losses or not after_losses:
            continue

        loss_before = float(np.mean(before_losses))
        loss_after = float(np.mean(after_losses))
        loss_delta = loss_after - loss_before
        loss_deltas.append(loss_delta)

        # Count spikes near transition
        spike_count = sum(1 for s in spike_steps if after_start <= s <= after_end)

        # Get sparsity change if available
        sparsity_delta = None
        if sparsity_dict:
            sparsity_before = sparsity_dict.get(before_end)
            sparsity_after = sparsity_dict.get(after_start)
            if sparsity_before is not None and sparsity_after is not None:
                sparsity_delta = sparsity_after - sparsity_before

        # Flag problematic transitions (loss increases significantly)
        is_problematic = loss_delta > 0 and abs(loss_delta) > np.std(loss_values) * 0.5
        if is_problematic:
            problematic_count += 1

        transitions.append(CurriculumTransition(
            epoch=i,
            epoch_step=epoch_step,
            loss_before=loss_before,
            loss_after=loss_after,
            loss_delta=loss_delta,
            spike_count_near=spike_count,
            sparsity_delta=sparsity_delta,
        ))

    avg_loss_delta = float(np.mean(loss_deltas)) if loss_deltas else 0.0

    return CurriculumAnalysis(
        transitions=transitions,
        total_transitions=len(transitions),
        problematic_transitions=problematic_count,
        avg_loss_delta=avg_loss_delta,
    )


def compute_correlations(
    loss_steps: list[int],
    loss_values: list[float],
    gradnorm_steps: list[int],
    gradnorm_values: list[float],
    dataload_steps: list[int],
    dataload_values: list[float],
    sparsity_steps: list[int] | None = None,
    sparsity_values: list[float] | None = None,
) -> Correlations:
    """Compute correlations with other metrics using step-aligned interpolation."""
    loss_gradnorm_corr = None
    loss_dataload_corr = None
    loss_sparsity_corr = None

    # Loss-gradnorm correlation (step-aligned)
    if loss_values and gradnorm_values:
        loss_aligned, grad_aligned = align_by_step(loss_steps, loss_values, gradnorm_steps, gradnorm_values)
        if len(loss_aligned) > 10:
            corr_matrix = np.corrcoef(loss_aligned, grad_aligned)
            if corr_matrix.shape == (2, 2):
                loss_gradnorm_corr = float(corr_matrix[0, 1])

    # Loss-dataload correlation (step-aligned)
    if loss_values and dataload_values:
        loss_aligned, data_aligned = align_by_step(loss_steps, loss_values, dataload_steps, dataload_values)
        if len(loss_aligned) > 10:
            corr_matrix = np.corrcoef(loss_aligned, data_aligned)
            if corr_matrix.shape == (2, 2):
                loss_dataload_corr = float(corr_matrix[0, 1])

    # Loss-sparsity correlation (step-aligned)
    if loss_values and sparsity_values and sparsity_steps:
        loss_aligned, sparsity_aligned = align_by_step(loss_steps, loss_values, sparsity_steps, sparsity_values)
        if len(loss_aligned) > 10:
            corr_matrix = np.corrcoef(loss_aligned, sparsity_aligned)
            if corr_matrix.shape == (2, 2):
                loss_sparsity_corr = float(corr_matrix[0, 1])

    return Correlations(
        loss_gradnorm_corr=loss_gradnorm_corr,
        loss_dataload_corr=loss_dataload_corr,
        loss_sparsity_corr=loss_sparsity_corr,
    )


def generate_flags(analysis: LossCurveAnalysis) -> list[str]:
    """Generate warning flags based on thresholds."""
    flags = []

    # Spike frequency
    total_spikes = len(analysis.spikes.high_spikes) + len(analysis.spikes.severe_spikes)
    if total_spikes > 0:
        flags.append(
            f"SPIKES: {total_spikes} high spikes detected "
            f"({analysis.spikes.high_spike_pct:.1f}% of steps)"
        )

    # Severe spikes
    if len(analysis.spikes.severe_spikes) > 0:
        flags.append(
            f"SEVERE: {len(analysis.spikes.severe_spikes)} severe spikes detected "
            f"(>{SPIKE_SEVERE_THRESHOLD}x median)"
        )

    # Volatility warnings
    if analysis.stability.coefficient_of_variation > CV_WARNING_THRESHOLD:
        flags.append(
            f"VOLATILITY: High loss volatility "
            f"(CV={analysis.stability.coefficient_of_variation:.2f})"
        )

    if analysis.stability.rolling_volatility > ROLLING_VOL_WARNING:
        flags.append(
            f"VOLATILITY: High rolling volatility "
            f"({analysis.stability.rolling_volatility:.3f})"
        )

    # Recovery rate
    if analysis.spikes.recovery_rate < RECOVERY_WARNING_THRESHOLD:
        flags.append(
            f"RECOVERY: Only {analysis.spikes.recovery_rate:.1f}% of spikes recover "
            f"(threshold: {RECOVERY_WARNING_THRESHOLD}%)"
        )

    # Convergence issues
    if analysis.convergence.log_loss_slope > 0:
        flags.append(
            f"DIVERGENCE: Loss is increasing "
            f"(log slope: {analysis.convergence.log_loss_slope:+.4f})"
        )

    if len(analysis.convergence.plateaus) > 3:
        flags.append(
            f"PLATEAUS: {len(analysis.convergence.plateaus)} plateaus detected "
            f"(possible stalling)"
        )

    # Curriculum transition issues
    if analysis.curriculum:
        if analysis.curriculum.problematic_transitions > 0:
            flags.append(
                f"CURRICULUM: {analysis.curriculum.problematic_transitions}/{analysis.curriculum.total_transitions} "
                f"transitions caused loss spikes"
            )

        # Check for sparsity jumps specifically
        large_sparsity_jumps = [
            t for t in analysis.curriculum.transitions
            if t.sparsity_delta and abs(t.sparsity_delta) > 0.3
        ]
        if large_sparsity_jumps:
            jump_info = ", ".join(f"epoch {t.epoch} (Δ={t.sparsity_delta:+.2f})" for t in large_sparsity_jumps[:3])
            flags.append(f"CURRICULUM: Large sparsity jumps at {jump_info}")

    # Correlation flags
    if analysis.correlations.loss_gradnorm_corr and analysis.correlations.loss_gradnorm_corr > 0.5:
        flags.append(
            f"CORRELATION: High loss-gradnorm correlation "
            f"({analysis.correlations.loss_gradnorm_corr:.3f}) - spikes driven by gradients"
        )

    if (
        analysis.correlations.loss_dataload_corr
        and analysis.correlations.loss_dataload_corr > 0.5
    ):
        flags.append(
            f"CORRELATION: High loss-dataload correlation "
            f"({analysis.correlations.loss_dataload_corr:.3f}) - check data pipeline"
        )

    if analysis.correlations.loss_sparsity_corr and abs(analysis.correlations.loss_sparsity_corr) > 0.5:
        direction = "positive" if analysis.correlations.loss_sparsity_corr > 0 else "negative"
        flags.append(
            f"CORRELATION: Strong {direction} loss-sparsity correlation "
            f"({analysis.correlations.loss_sparsity_corr:.3f}) - sparsity affects loss"
        )

    return flags


def analyze_events(
    event_dir: str | Path, diagnose: bool = False
) -> tuple[LossCurveAnalysis, list[str]]:
    """Analyze loss curve from TensorBoard event directory."""
    event_dir = Path(event_dir)
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory not found: {event_dir}")

    # Load event files
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()

    # Load training loss
    loss_steps, loss_values = load_scalars(ea, "Loss/Train_step")
    if not loss_values:
        loss_steps, loss_values = load_scalars(ea, "Loss/Train")

    if not loss_values:
        raise ValueError("No training loss data found in event files")

    # Basic statistics
    arr = np.array(loss_values)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr))

    # Analyze components
    spikes = analyze_spikes(loss_steps, loss_values, median_val)
    stability = analyze_stability(loss_steps, loss_values)
    convergence = analyze_convergence(loss_steps, loss_values)

    # Load additional metrics for correlation
    grad_steps, grad_values = load_scalars(ea, "GradNorm/Total_PreClip")
    data_steps, data_values = load_scalars(ea, "Time/DataLoad_s")

    # Load curriculum metrics if available
    epoch_steps, _ = load_scalars(ea, "epoch")
    sparsity_steps, sparsity_values = load_scalars(ea, "Train/Sparsity")

    # Compute correlations with step-aligned interpolation
    correlations = compute_correlations(
        loss_steps, loss_values,
        grad_steps, grad_values,
        data_steps, data_values,
        sparsity_steps, sparsity_values,
    )

    # Curriculum analysis
    spike_step_set = set(s for s, _ in spikes.high_spikes + spikes.severe_spikes)
    curriculum = analyze_curriculum(
        loss_steps, loss_values, spike_step_set,
        epoch_steps if epoch_steps else None,
        sparsity_steps if sparsity_steps else None,
        sparsity_values if sparsity_values else None,
    )

    analysis = LossCurveAnalysis(
        steps=loss_steps,
        values=loss_values,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        median_val=median_val,
        std_val=std_val,
        spikes=spikes,
        stability=stability,
        convergence=convergence,
        curriculum=curriculum,
        correlations=correlations,
    )

    # Generate flags
    flags = generate_flags(analysis)
    analysis.flags = flags

    return analysis, flags


def print_analysis(analysis: LossCurveAnalysis) -> None:
    """Print formatted analysis to console."""
    print("=" * 80)
    print("LOSS CURVE ANALYSIS")
    print("=" * 80)

    # Basic statistics
    print("\nLOSS STATISTICS")
    print("-" * 40)
    print(f"Steps:              {len(analysis.steps)}")
    print(f"Min:                {analysis.min_val:.4f}")
    print(f"Max:                {analysis.max_val:.4f}")
    print(f"Mean:               {analysis.mean_val:.4f}")
    print(f"Median:             {analysis.median_val:.4f}")
    print(f"Std Dev:            {analysis.std_val:.4f}")
    print(
        f"Coefficient of Variation: {analysis.stability.coefficient_of_variation:.3f}"
    )

    # Sparkline
    sparkline = ascii_sparkline(analysis.values[-100:] if len(analysis.values) > 100 else analysis.values)
    print(f"\nLoss trend (last {min(100, len(analysis.values))} steps):")
    print(f"  {sparkline}")

    # Spike analysis
    print("\nSPIKE ANALYSIS")
    print("-" * 40)
    total_spikes = len(analysis.spikes.high_spikes) + len(analysis.spikes.severe_spikes)
    print(f"High spikes (>{SPIKE_HIGH_THRESHOLD}x median): {len(analysis.spikes.high_spikes)} "
          f"({analysis.spikes.high_spike_pct:.1f}% of steps)")
    print(f"Severe spikes (>{SPIKE_SEVERE_THRESHOLD}x median): {len(analysis.spikes.severe_spikes)} "
          f"({analysis.spikes.severe_spike_pct:.1f}% of steps)")

    if total_spikes > 0:
        all_spikes = sorted(analysis.spikes.high_spikes + analysis.spikes.severe_spikes, key=lambda x: x[0])
        print(f"\nFirst 10 spikes:")
        for s, v in all_spikes[:10]:
            x_median = v / analysis.median_val
            marker = " ← SEVERE" if v >= analysis.median_val * SPIKE_SEVERE_THRESHOLD else ""
            print(f"  Step {s:5d}: {v:6.2f} ({x_median:4.1f}x median){marker}")

    print(f"\nRecovery rate: {analysis.spikes.recovery_rate:.1f}% of spikes recover within {RECOVERY_WINDOW} steps")
    print(f"Max upward jump: {analysis.spikes.max_upward_jump:.2f} (step {analysis.spikes.max_upward_step})")
    print(f"Max downward drop: {analysis.spikes.max_downward_drop:.2f}")

    # Stability metrics
    print("\nSTABILITY METRICS")
    print("-" * 40)
    print(f"Rolling volatility (window={ROLLING_WINDOW}): {analysis.stability.rolling_volatility:.3f}")
    print(f"Max drawdown: {analysis.stability.max_drawdown:.4f}")
    print(f"Max drawup: {analysis.stability.max_drawup:.4f}")

    # Convergence analysis
    print("\nCONVERGENCE ANALYSIS")
    print("-" * 40)
    direction = "decreasing → converging" if analysis.convergence.log_loss_slope < 0 else "increasing → diverging"
    print(f"Trend slope (log loss): {analysis.convergence.log_loss_slope:+.4f} ({direction})")
    print(f"Convergence rate: {analysis.convergence.convergence_rate:.4f} loss units per 1k steps")

    if analysis.convergence.plateaus:
        print(f"Plateaus detected: {len(analysis.convergence.plateaus)}")
        for start, end in analysis.convergence.plateaus[:5]:
            print(f"  Steps {start} → {end}")

    print(f"\nInitial loss: {analysis.convergence.initial_loss:.4f}")
    print(f"Final loss: {analysis.convergence.final_loss:.4f}")
    print(f"Reduction: {analysis.convergence.reduction_pct:.1f}%")

    # Curriculum analysis
    if analysis.curriculum and analysis.curriculum.transitions:
        print("\nCURRICULUM TRANSITION ANALYSIS")
        print("-" * 80)
        print(f"{'Epoch':>6} {'Step':>8} {'Loss Before':>12} {'Loss After':>12} {'Δ Loss':>10} {'Spikes':>8} {'Δ Sparsity':>10}")
        print("-" * 80)
        for t in analysis.curriculum.transitions:
            sparsity_str = f"{t.sparsity_delta:+.2f}" if t.sparsity_delta is not None else "N/A"
            delta_str = f"{t.loss_delta:+.4f}"
            marker = " ⚠" if t.loss_delta > 0.1 else ""
            print(
                f"{t.epoch:>6} {t.epoch_step:>8} {t.loss_before:>12.4f} {t.loss_after:>12.4f} "
                f"{delta_str:>10} {t.spike_count_near:>8} {sparsity_str:>10}{marker}"
            )

        print(f"\nProblematic transitions: {analysis.curriculum.problematic_transitions}/{analysis.curriculum.total_transitions}")
        print(f"Average loss delta: {analysis.curriculum.avg_loss_delta:+.4f}")

    # Correlations
    print("\nCORRELATIONS")
    print("-" * 40)
    if analysis.correlations.loss_gradnorm_corr is not None:
        grad_status = "weak positive" if 0 < analysis.correlations.loss_gradnorm_corr < 0.2 else \
                      "moderate positive" if 0.2 <= analysis.correlations.loss_gradnorm_corr < 0.5 else \
                      "strong positive" if analysis.correlations.loss_gradnorm_corr >= 0.5 else "negative"
        print(f"Loss-GradNorm correlation: {analysis.correlations.loss_gradnorm_corr:+.3f} ({grad_status})")
    else:
        print("Loss-GradNorm correlation: N/A")

    if analysis.correlations.loss_dataload_corr is not None:
        data_status = "weak positive" if 0 < analysis.correlations.loss_dataload_corr < 0.2 else \
                      "moderate positive" if 0.2 <= analysis.correlations.loss_dataload_corr < 0.5 else \
                      "strong positive" if analysis.correlations.loss_dataload_corr >= 0.5 else "negative"
        print(f"Loss-DataLoad correlation: {analysis.correlations.loss_dataload_corr:+.3f} ({data_status})")
    else:
        print("Loss-DataLoad correlation: N/A")

    if analysis.correlations.loss_sparsity_corr is not None:
        sparsity_status = "weak positive" if 0 < analysis.correlations.loss_sparsity_corr < 0.2 else \
                          "moderate positive" if 0.2 <= analysis.correlations.loss_sparsity_corr < 0.5 else \
                          "strong positive" if analysis.correlations.loss_sparsity_corr >= 0.5 else \
                          "strong negative" if analysis.correlations.loss_sparsity_corr <= -0.5 else "negative"
        print(f"Loss-Sparsity correlation: {analysis.correlations.loss_sparsity_corr:+.3f} ({sparsity_status})")
    else:
        print("Loss-Sparsity correlation: N/A")

    # Flags
    if analysis.flags:
        print("\nFLAGS")
        print("-" * 80)
        for flag in analysis.flags:
            print(f"  ⚠ {flag}")
    else:
        print("\nNo flags - all metrics within normal range.")

    print("\n" + "=" * 80)


def resolve_event_path(path: str) -> Path:
    """Resolve event directory path from various input formats."""
    input_path = Path(path)

    if input_path.exists():
        return input_path

    # Try as experiment name
    base = Path("outputs/training") / path
    if base.exists():
        runs = [d for d in base.iterdir() if d.is_dir()]
        if runs:
            return sorted(runs, reverse=True)[0]
        else:
            return base / "unknown"

    raise FileNotFoundError(f"Cannot resolve path: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard training loss curves for EpiForecaster"
    )
    parser.add_argument(
        "path",
        help="Path to event directory or experiment name",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output results as human-readable text (default: JSON)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show additional diagnostic information",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (no indentation)"
    )

    args = parser.parse_args()

    builder = SkillOutputBuilder(
        skill_name="loss-curve-critic",
        input_path=args.path,
    )

    try:
        event_path = resolve_event_path(args.path)
        analysis, flags = analyze_events(event_path, diagnose=args.diagnose)

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
    except ValueError as e:
        print_output(builder.error("ValueError", str(e)))
    except Exception as e:
        print_output(builder.error(type(e).__name__, str(e), {"traceback": str(e)}))


if __name__ == "__main__":
    sys.exit(main())
