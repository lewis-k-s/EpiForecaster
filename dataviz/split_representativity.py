#!/usr/bin/env python
"""Plot persisted cross-validation representativity diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


DISTRIBUTION_METRICS = {
    "population": "Population",
    "population_density_per_km2": "Population density",
    "cases_mean_observed_raw": "Cases burden",
    "hospitalizations_mean_observed_raw": "Hospitalizations burden",
    "deaths_mean_observed_raw": "Deaths burden",
    "wastewater_mean_observed_raw": "Wastewater burden",
}
AVAILABILITY_METRICS = {
    "cases_availability_fraction": "Cases",
    "hospitalizations_availability_fraction": "Hospitalizations",
    "deaths_availability_fraction": "Deaths",
    "wastewater_availability_fraction": "Wastewater",
}
SHARE_METRICS = {
    "population": "Population",
    "cases_total_observed_count_equivalent": "Cases load",
    "hospitalizations_total_observed_count_equivalent": "Hospitalizations load",
    "deaths_total_observed_count_equivalent": "Deaths load",
    "wastewater_total_observed_raw": "Wastewater load",
}
BALANCE_CMAP = LinearSegmentedColormap.from_list(
    "balance",
    ["#4C78A8", "#F7F7F7", "#F2A541"],
)


def _metric_matrix(
    balance: pd.DataFrame,
    *,
    metrics: dict[str, str],
    value_column: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    selected = balance[balance["metric"].isin(metrics)].copy()
    split_order = sorted(
        selected["split_id"].unique(),
        key=lambda value: int(value.split()[-1]),
    )
    matrix = (
        selected.pivot(index="metric", columns="split_id", values=value_column)
        .reindex(index=metrics, columns=split_order)
        .to_numpy(dtype=np.float64)
    )
    return matrix, list(metrics.values()), split_order


def _annotate_heatmap(
    axis: plt.Axes,
    display_values: np.ndarray,
    *,
    formatter: str,
) -> None:
    for row in range(display_values.shape[0]):
        for column in range(display_values.shape[1]):
            value = display_values[row, column]
            label = formatter.format(value) if np.isfinite(value) else "—"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#24313D",
            )


def _draw_heatmap(
    *,
    axis: plt.Axes,
    values: np.ndarray,
    display_values: np.ndarray,
    row_labels: list[str],
    split_labels: list[str],
    title: str,
    formatter: str,
    limit: float,
) -> None:
    image = axis.imshow(
        values,
        cmap=BALANCE_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xticks(
        np.arange(len(split_labels)),
        [label.replace("Fold ", "") for label in split_labels],
    )
    axis.set_xlabel("Test fold")
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.tick_params(length=0)
    axis.set_xticks(
        np.arange(-0.5, len(split_labels), 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(row_labels), 1),
        minor=True,
    )
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    _annotate_heatmap(axis, display_values, formatter=formatter)
    plt.colorbar(image, ax=axis, fraction=0.022, pad=0.015)


def plot_split_representativity(
    *,
    balance_path: Path,
    output_path: Path,
) -> None:
    """Render distribution, availability, and aggregate-share diagnostics."""
    balance = pd.read_csv(balance_path)

    distribution, distribution_labels, split_labels = _metric_matrix(
        balance,
        metrics=DISTRIBUTION_METRICS,
        value_column="standardized_mean_difference",
    )
    availability, availability_labels, availability_splits = _metric_matrix(
        balance,
        metrics=AVAILABILITY_METRICS,
        value_column="availability_difference_pp",
    )
    share_ratio, share_labels, share_splits = _metric_matrix(
        balance,
        metrics=SHARE_METRICS,
        value_column="share_ratio_to_region_count",
    )
    if availability_splits != split_labels or share_splits != split_labels:
        raise ValueError("All metric panels must contain the same split IDs")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11.5),
        gridspec_kw={"height_ratios": [1.5, 1.0, 1.25]},
        constrained_layout=True,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.94))
    _draw_heatmap(
        axis=axes[0],
        values=distribution,
        display_values=distribution,
        row_labels=distribution_labels,
        split_labels=split_labels,
        title="Distribution balance (standardized mean difference)",
        formatter="{:+.2f}",
        limit=max(0.6, float(np.nanmax(np.abs(distribution)))),
    )
    _draw_heatmap(
        axis=axes[1],
        values=availability,
        display_values=availability,
        row_labels=availability_labels,
        split_labels=split_labels,
        title="Availability relative to the valid-region universe (percentage points)",
        formatter="{:+.1f}",
        limit=max(6.5, float(np.nanmax(np.abs(availability)))),
    )

    log_share_ratio = np.log2(share_ratio)
    _draw_heatmap(
        axis=axes[2],
        values=log_share_ratio,
        display_values=share_ratio,
        row_labels=share_labels,
        split_labels=split_labels,
        title="Share of additive total relative to the fold's region share",
        formatter="{:.2f}×",
        limit=max(2.1, float(np.nanmax(np.abs(log_share_ratio)))),
    )

    figure.suptitle(
        "Refined 10-fold target-series representativity",
        fontsize=16,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.002,
        "Burden distributions use observed canonical targets on raw scale; "
        "standardized differences are computed after log1p transformation. "
        "A share ratio of 1× means an additive total is proportional to fold size.",
        ha="center",
        va="center",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_split_representativity(
        balance_path=args.balance,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
