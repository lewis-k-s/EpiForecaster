#!/usr/bin/env python
"""Render mobility-processing QC plots from persisted CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import Palette

SCOPE_COLORS = {
    "All regions": Palette.SPLIT["train"],
    "Valid targets": Palette.SPLIT["val"],
    "Training context": Palette.SPLIT["test"],
}


def plot_value_pipeline(input_dir: Path, output_path: Path) -> None:
    """Show threshold sensitivity, retained edges, degree, and clipping."""
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    scopes = pd.read_csv(input_dir / "scope_metrics.csv")
    sensitivity = pd.read_csv(input_dir / "threshold_sensitivity.csv")
    daily = pd.read_csv(input_dir / "daily_training_metrics.csv")

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        "Mobility processing QC: raw-flow selection before weight clipping",
        fontsize=18,
        fontweight="semibold",
    )

    axis = axes[0, 0]
    for scope, rows in sensitivity.groupby("scope", sort=False):
        rows = rows[rows["raw_flow_threshold"] > 0]
        axis.plot(
            rows["raw_flow_threshold"],
            100 * rows["positive_observation_retained_fraction"],
            marker="o",
            linewidth=2,
            label=scope,
            color=SCOPE_COLORS.get(scope),
        )
    threshold = float(summary["mobility_min_flow_raw"])
    axis.axvline(
        threshold, color="#222222", linestyle="--", label=f"selected = {threshold:g}"
    )
    axis.set_xscale("log")
    axis.set_xlabel("Minimum raw flow")
    axis.set_ylabel("Positive edge observations retained (%)")
    axis.set_title("Unit-correct threshold sensitivity")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    colors = [SCOPE_COLORS.get(scope, "#777777") for scope in scopes["scope"]]
    bars = axis.bar(
        scopes["scope"],
        100 * scopes["positive_observation_retained_fraction"],
        color=colors,
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Positive observations retained (%)")
    axis.set_title(f"Effective edge mask at raw flow ≥ {threshold:g}")
    axis.tick_params(axis="x", rotation=12)

    axis = axes[1, 0]
    dates = pd.to_datetime(daily["date"])
    axis.plot(
        dates,
        daily["mean_incoming_degree_unthresholded"],
        color="#9CA3AF",
        linewidth=1.8,
        label="Before: any positive flow",
    )
    axis.plot(
        dates,
        daily["mean_incoming_degree_effective"],
        color=SCOPE_COLORS["Training context"],
        linewidth=2.2,
        label="After: shared edge mask",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Mean incoming neighbors per training region")
    axis.set_title("Topology and clinical context now use the same degree")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.plot(
        dates,
        100 * daily["clipped_edge_fraction"],
        color="#D1495B",
        linewidth=2,
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Selected edges upper-clipped (%)")
    axis.set_title(
        "Clipping is separate from selection "
        f"(log-space max {summary['mobility_weight_clip_max']:g})"
    )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_effective_graph(input_dir: Path, output_path: Path) -> None:
    """Compare old positive-flow density with the unified effective graph."""
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    scopes = pd.read_csv(input_dir / "scope_metrics.csv")

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    figure.suptitle(
        "Effective graph and model-input mask provenance",
        fontsize=18,
        fontweight="semibold",
    )

    axis = axes[0]
    positions = np.arange(len(scopes))
    width = 0.36
    before = axis.bar(
        positions - width / 2,
        scopes["mean_incoming_degree_unthresholded"],
        width,
        color="#CBD5E1",
        label="Before: positive-flow topology",
    )
    after = axis.bar(
        positions + width / 2,
        scopes["mean_incoming_degree_effective"],
        width,
        color=[SCOPE_COLORS.get(scope, "#777777") for scope in scopes["scope"]],
        label="After: raw-flow edge mask",
    )
    axis.bar_label(before, fmt="%.1f", padding=3)
    axis.bar_label(after, fmt="%.1f", padding=3)
    axis.set_xticks(positions, scopes["scope"], rotation=12)
    axis.set_ylabel("Mean incoming neighbors per region-day")
    axis.set_title("One threshold controls the dynamic graph")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    mask_sources = summary["mask_sources"]
    rows = [
        ("GNN weights", mask_sources["gnn_weights"]),
        ("K-hop topology", mask_sources["k_hop_topology"]),
        ("Clinical context", mask_sources["clinical_context"]),
        ("Imported risk", mask_sources["imported_risk"]),
        ("GPU adjacency", mask_sources["gpu_adjacency"]),
        ("Self-loops", mask_sources["self_loops"]),
    ]
    source_names = list(dict.fromkeys(source for _, source in rows))
    matrix = np.zeros((len(rows), len(source_names)), dtype=float)
    for row_index, (_, source) in enumerate(rows):
        matrix[row_index, source_names.index(source)] = 1.0
    axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        np.arange(len(source_names)),
        [source.replace("_", " ") for source in source_names],
        rotation=18,
        ha="right",
    )
    axis.set_yticks(np.arange(len(rows)), [label for label, _ in rows])
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                "●" if matrix[row_index, column_index] else "",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
            )
    axis.set_title("Mask/weight source used by each model input")
    axis.tick_params(length=0)

    figure.text(
        0.5,
        0.02,
        "Static queen/KNN runs substitute their adjacency for GNN, k-hop, and clinical context; "
        "imported risk remains mobility-derived.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.22,
        top=0.84,
        wspace=0.42,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    plot_value_pipeline(
        args.input_dir,
        output_dir / "mobility_value_pipeline.png",
    )
    plot_effective_graph(
        args.input_dir,
        output_dir / "mobility_effective_graph.png",
    )
    print(f"Wrote mobility QC plots to {output_dir}")


if __name__ == "__main__":
    main()
