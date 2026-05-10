"""Overview diagnostics for raw OD mobility matrices.

This script reads ``data/files/mobility.zarr`` with xarray and produces a compact
set of mobility indicators useful before deeper regime or epidemiological
coupling analyses.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("mobility_overview_analysis")


LOCKDOWN_PERIODS = [
    ("Spain-wide lockdown", "2020-03-15", "2020-06-21"),
    ("Catalunya perimeter+weekend", "2020-10-30", "2020-11-14"),
    ("Catalunya perimeter+municipal", "2021-01-07", "2021-01-18"),
]


@dataclass
class MobilityOverviewConfig:
    """Runtime settings for overview mobility diagnostics."""

    mobility_path: Path
    output_dir: Path
    top_edge_fraction: float = 0.01
    top_regions: int = 25
    log_every: int = 50


def open_mobility_dataset(path: Path) -> xr.Dataset:
    """Open a mobility Zarr dataset and normalize destination naming."""
    ds = xr.open_zarr(path, chunks={"date": 1})
    if "target" in ds.dims:
        ds = ds.rename({"target": "destination"})
    if "mobility" not in ds:
        raise ValueError("Mobility dataset must contain a 'mobility' variable")
    if not {"date", "origin", "destination"}.issubset(ds.dims):
        raise ValueError("Mobility data must have date, origin, and destination dims")
    return ds


def compute_daily_metrics(
    mobility: xr.DataArray,
    top_edge_fraction: float,
    log_every: int,
) -> pd.DataFrame:
    """Compute per-snapshot mobility indicators."""
    mobility = mobility.transpose("date", "origin", "destination")
    dates = pd.DatetimeIndex(mobility["date"].values)
    n_edges = mobility.sizes["origin"] * mobility.sizes["destination"]
    rows: list[dict[str, Any]] = []

    for idx, date in enumerate(dates):
        matrix = np.asarray(mobility.isel(date=idx).values, dtype=float)
        matrix[matrix < 0] = 0.0
        total = float(matrix.sum())
        active_mask = matrix > 0
        active_edges = int(active_mask.sum())
        edge_values = matrix[active_mask]
        diagonal = float(np.trace(matrix))
        off_diagonal = total - diagonal
        row_strength = matrix.sum(axis=1)
        col_strength = matrix.sum(axis=0)

        rows.append(
            {
                "date": date,
                "total_volume": total,
                "log_total_volume": float(np.log1p(total)),
                "active_edges": active_edges,
                "edge_density": active_edges / n_edges,
                "self_flow_fraction": diagonal / total if total > 0 else 0.0,
                "off_diagonal_volume": off_diagonal,
                "mean_active_edge": float(edge_values.mean())
                if edge_values.size
                else 0.0,
                "median_active_edge": float(np.median(edge_values))
                if edge_values.size
                else 0.0,
                "edge_gini": gini_coefficient(edge_values),
                "top_edge_fraction": top_fraction(edge_values, total, top_edge_fraction),
                "reciprocity_ratio": reciprocity_ratio(matrix),
                "origin_strength_gini": gini_coefficient(row_strength),
                "destination_strength_gini": gini_coefficient(col_strength),
                "net_flow_abs_share": float(np.abs(col_strength - row_strength).sum())
                / (2.0 * total)
                if total > 0
                else 0.0,
            }
        )

        if (idx + 1) % log_every == 0 or idx + 1 == len(dates):
            logger.info("Processed %d/%d snapshots", idx + 1, len(dates))

    return pd.DataFrame(rows).set_index("date").sort_index()


def compute_region_strengths(mobility: xr.DataArray, top_regions: int) -> pd.DataFrame:
    """Rank regions by average outgoing, incoming, and net mobility volume."""
    mobility = mobility.transpose("date", "origin", "destination")
    outgoing = mobility.sum(dim=("date", "destination")).compute()
    incoming = mobility.sum(dim=("date", "origin")).compute()

    out_df = outgoing.to_series().rename("outgoing_total").reset_index()
    in_df = incoming.to_series().rename("incoming_total").reset_index()
    out_df = out_df.rename(columns={"origin": "region_id"})
    in_df = in_df.rename(columns={"destination": "region_id"})
    region_df = out_df.merge(in_df, on="region_id", how="outer").fillna(0.0)
    region_df["net_inflow"] = region_df["incoming_total"] - region_df["outgoing_total"]
    region_df["total_touch"] = region_df["incoming_total"] + region_df["outgoing_total"]
    region_df["incoming_rank"] = region_df["incoming_total"].rank(
        ascending=False, method="min"
    )
    region_df["outgoing_rank"] = region_df["outgoing_total"].rank(
        ascending=False, method="min"
    )
    return (
        region_df.sort_values("total_touch", ascending=False)
        .head(top_regions)
        .reset_index(drop=True)
    )


def compute_period_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily metrics inside named public-health periods."""
    rows: list[dict[str, Any]] = []
    for name, start, end in LOCKDOWN_PERIODS:
        period = daily_df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if period.empty:
            continue
        before = daily_df.loc[
            pd.Timestamp(start) - pd.Timedelta(days=14) : pd.Timestamp(start)
            - pd.Timedelta(days=1)
        ]
        row: dict[str, Any] = {
            "period": name,
            "start": start,
            "end": end,
            "n_days": int(len(period)),
        }
        for col in [
            "total_volume",
            "edge_density",
            "self_flow_fraction",
            "reciprocity_ratio",
            "edge_gini",
            "net_flow_abs_share",
        ]:
            period_mean = float(period[col].mean())
            baseline_mean = float(before[col].mean()) if not before.empty else np.nan
            row[f"{col}_mean"] = period_mean
            row[f"{col}_delta_vs_prev14_pct"] = (
                (period_mean - baseline_mean) / baseline_mean * 100
                if np.isfinite(baseline_mean) and baseline_mean != 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_plots(daily_df: pd.DataFrame, output_dir: Path) -> None:
    """Write overview plots to disk."""
    plot_columns = [
        ("total_volume", "Total Mobility Volume"),
        ("edge_density", "Active OD Edge Density"),
        ("self_flow_fraction", "Self-Flow Fraction"),
        ("reciprocity_ratio", "Reciprocity Ratio"),
        ("edge_gini", "Active Edge Gini"),
        ("net_flow_abs_share", "Absolute Net Flow Share"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, (column, title) in zip(axes.ravel(), plot_columns, strict=True):
        ax.plot(daily_df.index, daily_df[column], color="#2f6f73", linewidth=1.5)
        add_period_shading(ax)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "mobility_overview_timeseries.png", dpi=200)
    plt.close(fig)

    weekday = daily_df.assign(weekday=daily_df.index.day_name()).groupby("weekday")
    weekday_df = weekday[
        ["total_volume", "edge_density", "self_flow_fraction", "reciprocity_ratio"]
    ].mean()
    weekday_df = weekday_df.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    weekday_df["total_volume"].plot(kind="bar", ax=ax, color="#8f5f2a")
    ax.set_title("Average Total Mobility by Weekday")
    ax.set_xlabel("")
    ax.set_ylabel("Total volume")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mobility_weekday_volume.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        daily_df["edge_density"],
        daily_df["total_volume"],
        c=daily_df["self_flow_fraction"],
        cmap="viridis",
        s=28,
        alpha=0.8,
    )
    ax.set_xlabel("Active OD edge density")
    ax.set_ylabel("Total mobility volume")
    ax.set_title("Volume vs Network Density")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Self-flow fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "mobility_volume_density_scatter.png", dpi=200)
    plt.close(fig)


def add_period_shading(ax: plt.Axes) -> None:
    """Add public-health restriction period shading to a time axis."""
    for name, start, end in LOCKDOWN_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#c44e52", alpha=0.12)
    ax.legend(["Daily value", "Restriction period"], loc="best", fontsize=8)


def write_report(
    daily_df: pd.DataFrame,
    period_df: pd.DataFrame,
    top_regions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a markdown summary and machine-readable metadata."""
    summary = {
        "date_start": str(daily_df.index.min().date()),
        "date_end": str(daily_df.index.max().date()),
        "n_days": int(len(daily_df)),
        "mean_total_volume": float(daily_df["total_volume"].mean()),
        "min_total_volume_date": str(daily_df["total_volume"].idxmin().date()),
        "max_total_volume_date": str(daily_df["total_volume"].idxmax().date()),
        "mean_edge_density": float(daily_df["edge_density"].mean()),
        "mean_self_flow_fraction": float(daily_df["self_flow_fraction"].mean()),
    }
    with (output_dir / "mobility_overview_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    report = [
        "# Mobility Overview Analysis",
        "",
        f"Dataset covers {summary['n_days']} daily OD snapshots from "
        f"{summary['date_start']} to {summary['date_end']}.",
        "",
        "## Key Files",
        "",
        "- `mobility_daily_indicators.csv`",
        "- `mobility_lockdown_period_summary.csv`",
        "- `mobility_top_regions.csv`",
        "- `mobility_overview_timeseries.png`",
        "- `mobility_weekday_volume.png`",
        "- `mobility_volume_density_scatter.png`",
        "",
        "## Summary",
        "",
        f"- Mean total volume: {summary['mean_total_volume']:.3f}",
        f"- Lowest-volume day: {summary['min_total_volume_date']}",
        f"- Highest-volume day: {summary['max_total_volume_date']}",
        f"- Mean edge density: {summary['mean_edge_density']:.4f}",
        f"- Mean self-flow fraction: {summary['mean_self_flow_fraction']:.4f}",
        "",
        "## Indicator Plan",
        "",
        "- Baseline-normalized volume shocks by weekday and holiday status.",
        "- Origin/destination strength concentration and persistent mobility hubs.",
        "- Directional imbalance via net inflow/outflow and reciprocity.",
        "- Network fragmentation using thresholded connected components.",
        "- Spatial reach using entropy of destination distributions per origin.",
        "- Regime-aligned epidemiological coupling against cases and wastewater.",
    ]
    if not period_df.empty:
        report.extend(["", "## Restriction Period Summary", ""])
        report.append(format_markdown_table(period_df))
    if not top_regions_df.empty:
        report.extend(["", "## Top Regions by Mobility Touch", ""])
        report.append(format_markdown_table(top_regions_df.head(10)))
    (output_dir / "README.md").write_text("\n".join(report) + "\n")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table without optional pandas dependencies."""
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    headers = [str(col) for col in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.astype(str).itertuples(index=False):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def top_fraction(values: np.ndarray, total: float, fraction: float) -> float:
    """Return share of total mobility carried by the top fraction of active edges."""
    if values.size == 0 or total <= 0:
        return 0.0
    top_k = max(int(np.ceil(values.size * fraction)), 1)
    return float(np.sort(values)[-top_k:].sum() / total)


def reciprocity_ratio(matrix: np.ndarray) -> float:
    """Measure how much directed OD volume is reciprocated by reverse edges."""
    total = float(matrix.sum())
    if total <= 0:
        return 0.0
    return float(np.minimum(matrix, matrix.T).sum() / total)


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient for non-negative values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    total = sorted_values.sum()
    if total == 0:
        return 0.0
    n = sorted_values.size
    weights = np.arange(1, n + 1)
    return float((2 * np.sum(weights * sorted_values)) / (n * total) - (n + 1) / n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw mobility overview metrics")
    parser.add_argument(
        "--mobility-path",
        type=Path,
        default=Path("data/files/mobility.zarr"),
        help="Path to the mobility Zarr store",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/mobility_overview"),
        help="Directory for analysis artifacts",
    )
    parser.add_argument("--top-edge-fraction", type=float, default=0.01)
    parser.add_argument("--top-regions", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = MobilityOverviewConfig(
        mobility_path=args.mobility_path,
        output_dir=args.output_dir,
        top_edge_fraction=args.top_edge_fraction,
        top_regions=args.top_regions,
        log_every=args.log_every,
    )
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening mobility dataset from %s", config.mobility_path)
    ds = open_mobility_dataset(config.mobility_path)
    daily_df = compute_daily_metrics(
        ds["mobility"], config.top_edge_fraction, config.log_every
    )
    region_df = compute_region_strengths(ds["mobility"], config.top_regions)
    period_df = compute_period_summary(daily_df)

    daily_df.to_csv(config.output_dir / "mobility_daily_indicators.csv")
    region_df.to_csv(config.output_dir / "mobility_top_regions.csv", index=False)
    period_df.to_csv(config.output_dir / "mobility_lockdown_period_summary.csv", index=False)
    generate_plots(daily_df, config.output_dir)
    write_report(daily_df, period_df, region_df, config.output_dir)
    logger.info("Wrote mobility overview outputs to %s", config.output_dir)


if __name__ == "__main__":
    main()
