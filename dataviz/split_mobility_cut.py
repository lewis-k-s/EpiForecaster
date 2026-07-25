#!/usr/bin/env python
"""Plot persisted coverage and mobility-retention split diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import Palette


STRATEGY_COLORS = {
    "Independent stratified seeds": Palette.SPLIT["train"],
    "Representative 10-fold": Palette.SPLIT["val"],
    "Refined representative 10-fold": Palette.TARGET["wastewater"],
}


def plot_split_mobility_cut(
    *,
    metrics_path: Path,
    summary_path: Path,
    optimization_history_path: Path,
    balance_audit_path: Path,
    output_path: Path,
) -> None:
    """Render persisted split results without reopening source data."""
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    history = pd.read_csv(optimization_history_path)
    balance_audit = pd.read_csv(balance_audit_path)
    strategies = summary["strategy"].tolist()
    colors = [STRATEGY_COLORS[strategy] for strategy in strategies]

    figure = plt.figure(figsize=(12.5, 8.8))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        bottom=0.14,
        top=0.88,
        hspace=0.34,
        wspace=0.30,
    )
    coverage_ax = figure.add_subplot(grid[0, 0])
    retention_ax = figure.add_subplot(grid[0, 1])
    isolation_ax = figure.add_subplot(grid[1, 0])
    optimization_ax = figure.add_subplot(grid[1, 1])

    coverage = 100.0 * summary["test_region_coverage"].to_numpy()
    bars = coverage_ax.bar(strategies, coverage, color=colors, width=0.62)
    coverage_ax.set_ylabel("Valid target regions covered (%)")
    coverage_ax.set_ylim(0, 108)
    coverage_ax.set_title("Exactly-once test coverage")
    coverage_ax.tick_params(axis="x", rotation=14)
    for bar, (_, row) in zip(bars, summary.iterrows(), strict=True):
        coverage_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{int(row['unique_test_regions'])}/{int(row['valid_region_count'])}",
            ha="center",
            va="bottom",
            fontweight="semibold",
        )
        coverage_ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(4.0, bar.get_height() - 9.0),
            f"{int(row['repeated_test_assignments'])} repeated slots",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    retention_columns = [
        ("mean_incoming_flow_retained", "Mean target"),
        ("p10_incoming_flow_retained", "10th percentile"),
        ("minimum_incoming_flow_retained", "Worst target"),
    ]
    positions = np.arange(len(retention_columns))
    offsets = np.linspace(-0.20, 0.20, len(strategies))
    for offset, strategy in zip(offsets, strategies, strict=True):
        strategy_metrics = metrics[metrics["strategy"] == strategy]
        color = STRATEGY_COLORS[strategy]
        for x_position, (column, _label) in zip(
            positions,
            retention_columns,
            strict=True,
        ):
            values = 100.0 * strategy_metrics[column].to_numpy()
            retention_ax.scatter(
                np.full_like(values, x_position + offset),
                values,
                s=32,
                alpha=0.70,
                color=color,
                edgecolor="none",
            )
            retention_ax.plot(
                [x_position + offset - 0.06, x_position + offset + 0.06],
                [np.mean(values), np.mean(values)],
                color=color,
                linewidth=3,
            )
    retention_ax.axhline(
        70.0,
        color=Palette.TARGET["deaths"],
        linestyle="--",
        linewidth=1.2,
        label="Hard floor",
    )
    retention_ax.axhline(
        75.0,
        color=Palette.TARGET["hospitalizations"],
        linestyle=":",
        linewidth=1.2,
        label="Preferred floor",
    )
    retention_ax.set_xticks(
        positions,
        [label for _, label in retention_columns],
    )
    retention_ax.set_ylabel("Incoming mobility flow retained (%)")
    retention_ax.set_ylim(0, 103)
    retention_ax.set_title("Test targets retain cross-boundary context")
    retention_ax.legend(frameon=False, fontsize=8, loc="lower left")

    for strategy in strategies:
        strategy_metrics = metrics[metrics["strategy"] == strategy]
        isolation_ax.scatter(
            100.0 * strategy_metrics["test_internal_flow_fraction"],
            100.0 * strategy_metrics["minimum_incoming_flow_retained"],
            s=52,
            alpha=0.78,
            color=STRATEGY_COLORS[strategy],
            label=strategy,
        )
    isolation_ax.set_xlabel("Flow trapped inside test fold (%)")
    isolation_ax.set_ylabel("Worst target retained flow (%)")
    isolation_ax.set_title("Complete directed-network isolation")
    isolation_ax.legend(frameon=False, fontsize=8, loc="lower left")

    optimization_ax.plot(
        history["iteration"],
        100.0 * history["worst_fold_internal_flow_fraction"],
        color=STRATEGY_COLORS["Refined representative 10-fold"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Worst fold internal flow",
    )
    optimization_ax.plot(
        history["iteration"],
        100.0 * history["minimum_target_retained_flow"],
        color=Palette.TARGET["hospitalizations"],
        linewidth=2.2,
        marker=".",
        label="Worst target retained flow",
    )
    optimization_ax.set_xlabel("Accepted constrained swap")
    optimization_ax.set_ylabel("Flow fraction (%)")
    optimization_ax.set_title("Deterministic local-swap refinement")
    optimization_ax.legend(frameon=False, fontsize=8)

    balance_ok = bool(balance_audit["within_baseline_envelope"].all())
    figure.suptitle(
        "Coverage-safe mobility-network split refinement",
        fontsize=16,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.048,
        "Complete directed mobility network; no top-K filtering. "
        f"All balance features within baseline envelope: {balance_ok}.",
        ha="center",
        va="center",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.020,
        "Cross-boundary flow is available as context; within-test flow is unavailable.",
        ha="center",
        va="center",
        fontsize=9,
    )

    for axis in (
        coverage_ax,
        retention_ax,
        isolation_ax,
        optimization_ax,
    ):
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--optimization-history", type=Path, required=True)
    parser.add_argument("--balance-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_split_mobility_cut(
        metrics_path=args.metrics,
        summary_path=args.summary,
        optimization_history_path=args.optimization_history,
        balance_audit_path=args.balance_audit,
        output_path=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
