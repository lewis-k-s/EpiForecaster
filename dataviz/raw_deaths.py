"""
Visualize raw daily municipality-level deaths data from CSV.

Outputs:
- Daily time series by municipality (line plot)
- Municipality heatmap (municipality × date)
- Distribution by municipality (violin plots)
- Cumulative deaths progression
- Peak days analysis (bar chart)
- Sex breakdown (stacked bars by sex per municipality)
- Summary statistics table (console and CSV)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

logger = logging.getLogger(__name__)

# Column mapping for municipality-level deaths data (from dasymetric_mob)
COLUMN_MAPPING = {
    "Data defunció": "date",
    "Codi Sexe": "sex_code",
    "Sexe": "sex",
    "municipality_code": "municipality_code",
    "municipality_name": "municipality_name",
    "defuncions_muni": "deaths",
}

DTYPES: dict[str, type] = {
    "Data defunció": str,
    "Codi Sexe": str,
    "Sexe": str,
    "municipality_code": str,
    "defuncions_muni": float,
}


def _load_raw_data(deaths_file: Path) -> pd.DataFrame:
    """Load raw deaths data from CSV."""
    if not deaths_file.exists():
        raise FileNotFoundError(f"Deaths file not found: {deaths_file}")

    logger.info("Loading deaths from %s", deaths_file)

    df = pd.read_csv(
        deaths_file,
        dtype=DTYPES,  # type: ignore[arg-type]
    )

    df = df.rename(columns=COLUMN_MAPPING)

    df["date"] = pd.to_datetime(
        df["date"], format="%d/%m/%Y", errors="coerce"
    ).dt.tz_localize(None)

    # Drop rows with missing municipality_code
    df = df[
        df["municipality_code"].notna() & (df["municipality_code"] != "")
    ]

    df = df.dropna(subset=["date", "municipality_code", "deaths"])
    df = df[df["deaths"] >= 0]

    logger.info(
        "Loaded %d records, %d municipalities, dates: %s to %s",
        len(df),
        df["municipality_code"].nunique(),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    return df


def _aggregate_to_comarca_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deaths to comarca-date level (summing across sex)."""
    aggregated = (
        df.groupby(["date", "municipality_code", "municipality_name"], dropna=False)["deaths"]
        .sum()
        .reset_index()
    )
    return aggregated


def plot_comarca_series(
    municipality_daily: pd.DataFrame,
    output_path: Path,
    max_comarcas: int = 20,
    seed: int = 7,
) -> None:
    """Plot daily time series for selected comarcas."""
    unique_municipalities = municipality_daily["municipality_code"].unique()
    n_comarcas = min(max_comarcas, len(unique_municipalities))
    rng = np.random.default_rng(seed)
    selected_municipalities = rng.choice(unique_municipalities, size=n_comarcas, replace=False)

    fig, ax = plt.subplots(figsize=(14, 6))

    for municipality_code in selected_municipalities:
        municipality_data = municipality_daily[municipality_daily["municipality_code"] == municipality_code]
        municipality_data = municipality_data.sort_values("date")
        municipality_name = (
            municipality_data["municipality_name"].iloc[0]
            if len(municipality_data) > 0
            else municipality_code
        )
        ax.plot(
            municipality_data["date"],
            municipality_data["deaths"],
            alpha=0.6,
            linewidth=1.5,
            label=f"{municipality_name}",
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Deaths")
    ax.set_title(f"Daily Deaths by Comarca (showing {n_comarcas} random comarcas)")
    ax.grid(True, alpha=0.3)

    if n_comarcas <= 10:
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved comarca series to %s", output_path)


def plot_comarca_heatmap(
    municipality_daily: pd.DataFrame,
    output_path: Path,
    max_comarcas: int = 50,
    seed: int = 7,
) -> None:
    """Plot heatmap of deaths (comarca × date)."""
    unique_municipalities = municipality_daily["municipality_code"].unique()
    n_comarcas = min(max_comarcas, len(unique_municipalities))
    rng = np.random.default_rng(seed)
    selected_municipalities = rng.choice(unique_municipalities, size=n_comarcas, replace=False)

    subset = municipality_daily[municipality_daily["municipality_code"].isin(selected_municipalities)]
    pivot = subset.pivot_table(
        index="municipality_name",
        columns="date",
        values="deaths",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(14, max(6, n_comarcas * 0.3)))

    vmax = (
        float(np.percentile(pivot.values[pivot.values > 0], 95))
        if (pivot.values > 0).any()
        else None
    )

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Daily Deaths"},
        xticklabels=False,
    )

    step = max(1, len(pivot.columns) // 8)
    ax.set_xticks(np.arange(0, len(pivot.columns), step))
    ax.set_xticklabels(
        [
            pd.Timestamp(d).strftime("%Y-%m")
            if isinstance(d, str)
            else d.strftime("%Y-%m")
            for d in pivot.columns[::step]
        ],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Comarca")
    ax.set_title(f"Deaths Heatmap by Comarca (showing {n_comarcas} random comarcas)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved comarca heatmap to %s", output_path)


def plot_comarca_distribution(
    municipality_daily: pd.DataFrame,
    output_path: Path,
    max_comarcas: int = 15,
) -> None:
    """Plot violin plots of death distribution by comarca."""
    comarca_totals = (
        municipality_daily.groupby("municipality_code")["deaths"]
        .sum()
        .sort_values(ascending=False)
    )
    selected_municipalities = comarca_totals.head(max_comarcas).index.tolist()

    plot_data = municipality_daily[
        municipality_daily["municipality_code"].isin(selected_municipalities)
    ].copy()

    fig, ax = plt.subplots(figsize=(max(12, len(selected_municipalities) * 0.6), 6))

    sns.violinplot(
        data=plot_data,
        x="municipality_name",
        y="deaths",
        ax=ax,
        palette="Set2",
    )

    ax.set_xlabel("Comarca")
    ax.set_ylabel("Daily Deaths")
    ax.set_title(
        f"Distribution of Daily Deaths by Comarca (top {len(selected_municipalities)})"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comarca distribution to %s", output_path)


def plot_cumulative_deaths(
    municipality_daily: pd.DataFrame,
    output_path: Path,
    max_comarcas: int = 20,
    seed: int = 7,
) -> None:
    """Plot cumulative deaths by comarca."""
    unique_municipalities = municipality_daily["municipality_code"].unique()
    n_comarcas = min(max_comarcas, len(unique_municipalities))
    rng = np.random.default_rng(seed)
    selected_municipalities = rng.choice(unique_municipalities, size=n_comarcas, replace=False)

    fig, ax = plt.subplots(figsize=(14, 6))

    for municipality_code in selected_municipalities:
        municipality_data = municipality_daily[municipality_daily["municipality_code"] == municipality_code]
        municipality_data = municipality_data.sort_values("date")
        municipality_data["cumulative"] = municipality_data["deaths"].cumsum()
        municipality_name = (
            municipality_data["municipality_name"].iloc[0]
            if len(municipality_data) > 0
            else municipality_code
        )
        ax.plot(
            municipality_data["date"],
            municipality_data["cumulative"],
            alpha=0.6,
            linewidth=1.5,
            label=f"{municipality_name}",
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Deaths")
    ax.set_title(f"Cumulative Deaths by Comarca (showing {n_comarcas} random comarcas)")
    ax.grid(True, alpha=0.3)

    if n_comarcas <= 10:
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved cumulative deaths to %s", output_path)


def plot_peak_days(
    municipality_daily: pd.DataFrame,
    output_path: Path,
    max_comarcas: int = 20,
) -> None:
    """Plot bar chart of peak death days by comarca."""
    peak_data = []
    for municipality_code in municipality_daily["municipality_code"].unique():
        municipality_data = municipality_daily[municipality_daily["municipality_code"] == municipality_code]
        if len(municipality_data) > 0:
            peak_idx = municipality_data["deaths"].idxmax()
            peak_row = municipality_data.loc[peak_idx]
            peak_data.append(
                {
                    "municipality_code": municipality_code,
                    "municipality_name": peak_row["municipality_name"],
                    "peak_date": peak_row["date"],
                    "peak_deaths": peak_row["deaths"],
                }
            )

    peak_df = pd.DataFrame(peak_data)
    peak_df = peak_df.sort_values("peak_deaths", ascending=False).head(max_comarcas)

    fig, ax = plt.subplots(figsize=(max(12, len(peak_df) * 0.6), 6))

    colors = plt.get_cmap("YlOrRd")(np.linspace(0.3, 0.9, len(peak_df)))
    ax.bar(peak_df["municipality_name"], peak_df["peak_deaths"], color=colors)

    ax.set_xlabel("Comarca")
    ax.set_ylabel("Peak Daily Deaths")
    ax.set_title(f"Peak Death Days by Comarca (top {len(peak_df)})")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved peak days to %s", output_path)


def plot_sex_breakdown(
    raw_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot stacked bar chart of deaths by sex per comarca."""
    # Aggregate by comarca and sex
    sex_data = raw_df.groupby(["municipality_name", "sex"])["deaths"].sum().reset_index()

    pivot = sex_data.pivot(index="municipality_name", columns="sex", values="deaths")
    pivot = pivot.fillna(0)

    # Sort by total deaths
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).head(20)
    pivot = pivot.drop("total", axis=1)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.4), 6))

    bottom = np.zeros(len(pivot))
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for Home, Orange for Dona

    for idx, sex in enumerate(pivot.columns):
        values = np.asarray(pivot[sex].values)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=sex,
            color=colors[idx % len(colors)],
        )
        bottom = bottom + values

    ax.set_xlabel("Comarca")
    ax.set_ylabel("Total Deaths")
    ax.set_title("Deaths by Sex per Comarca (top 20 by total deaths)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Sex", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sex breakdown to %s", output_path)


def compute_summary_statistics(
    municipality_daily: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary statistics for deaths."""
    stats = []

    total_deaths = municipality_daily["deaths"].sum()
    n_days = municipality_daily["date"].nunique()
    n_comarcas = municipality_daily["municipality_code"].nunique()

    daily_totals = municipality_daily.groupby("date")["deaths"].sum()
    peak_day = daily_totals.idxmax()
    peak_value = daily_totals.max()

    comarca_totals = (
        municipality_daily.groupby("municipality_code")["deaths"]
        .sum()
        .sort_values(ascending=False)
    )
    most_affected = comarca_totals.index[0]
    most_affected_name = municipality_daily[municipality_daily["municipality_code"] == most_affected][
        "municipality_name"
    ].iloc[0]
    most_affected_value = comarca_totals.iloc[0]

    # Sex breakdown
    sex_totals = raw_df.groupby("sex")["deaths"].sum()
    male_deaths = sex_totals.get("Home", 0)
    female_deaths = sex_totals.get("Dona", 0)

    stats.append(
        {"metric": "Total Deaths", "value": int(total_deaths), "unit": "cases"}
    )
    stats.append({"metric": "Number of Days", "value": n_days, "unit": "days"})
    stats.append(
        {"metric": "Number of Comarcas", "value": n_comarcas, "unit": "comarcas"}
    )
    # Format peak day
    if isinstance(peak_day, pd.Timestamp):
        peak_day_str = peak_day.strftime("%Y-%m-%d")
    else:
        peak_day_str = str(peak_day)

    stats.append(
        {
            "metric": "Peak Day",
            "value": peak_day_str,
            "unit": "date",
        }
    )
    stats.append(
        {"metric": "Peak Day Deaths", "value": int(peak_value), "unit": "cases"}
    )
    stats.append(
        {
            "metric": "Most Affected Comarca",
            "value": most_affected_name,
            "unit": "comarca",
        }
    )
    stats.append(
        {
            "metric": "Most Affected Deaths",
            "value": int(most_affected_value),
            "unit": "cases",
        }
    )
    stats.append({"metric": "Male Deaths", "value": int(male_deaths), "unit": "cases"})
    stats.append(
        {"metric": "Female Deaths", "value": int(female_deaths), "unit": "cases"}
    )

    stats_df = pd.DataFrame(stats)
    return stats_df


def print_summary_statistics(stats_df: pd.DataFrame) -> None:
    """Print summary statistics table to console."""
    print("\n" + "=" * 80)
    print("RAW DEATHS SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deaths-csv",
        type=Path,
        required=True,
        help="Path to deaths CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/raw_deaths"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--max-comarcas",
        type=int,
        default=20,
        help="Max comarcas to include in line plot",
    )
    parser.add_argument(
        "--max-comarcas-heatmap",
        type=int,
        default=50,
        help="Max comarcas to include in heatmap",
    )
    parser.add_argument(
        "--max-comarcas-distribution",
        type=int,
        default=15,
        help="Max comarcas to show in distribution plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling comarcas",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path to save CSV statistics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw_data(args.deaths_csv)

    if raw_df.empty:
        logger.warning("No death data found")
        return

    municipality_daily = _aggregate_to_comarca_day(raw_df)

    stats_df = compute_summary_statistics(municipality_daily, raw_df)
    print_summary_statistics(stats_df)

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(args.stats_output, index=False)
        logger.info("Saved statistics to %s", args.stats_output)

    plot_comarca_series(
        municipality_daily,
        args.output_dir / "comarca_series.png",
        max_comarcas=args.max_comarcas,
        seed=args.seed,
    )

    plot_comarca_heatmap(
        municipality_daily,
        args.output_dir / "comarca_heatmap.png",
        max_comarcas=args.max_comarcas_heatmap,
        seed=args.seed,
    )

    plot_comarca_distribution(
        municipality_daily,
        args.output_dir / "comarca_distribution.png",
        max_comarcas=args.max_comarcas_distribution,
    )

    plot_cumulative_deaths(
        municipality_daily,
        args.output_dir / "cumulative_deaths.png",
        max_comarcas=args.max_comarcas,
        seed=args.seed,
    )

    plot_peak_days(
        municipality_daily,
        args.output_dir / "peak_days.png",
        max_comarcas=args.max_comarcas,
    )

    plot_sex_breakdown(
        raw_df,
        args.output_dir / "sex_breakdown.png",
    )


if __name__ == "__main__":
    main()
