"""
Visualize raw weekly municipality-level hospitalization data from CSV.

Outputs:
- Weekly time series by municipality (line plot)
- Municipality heatmap (municipality × week)
- Distribution by comarca (violin plots)
- Demographic breakdown (stacked bar charts by sex/age group)
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

# Column mapping from hospitalizations_processor (new municipality-level format)
COLUMN_MAPPING = {
    "setmana_epidemiologica": "epi_week",
    "any": "year",
    "data_inici": "week_start",
    "data_final": "week_end",
    "codi_regio": "municipality_comarca_code",
    "nom_regio": "municipality_comarca_name",
    "codi_ambit": "ambit_code",
    "nom_ambit": "ambit_name",
    "sexe": "sex",
    "grup_edat": "age_group",
    "index_socioeconomic": "ses_index",
    "municipality_code": "municipality_code",
    "municipality_name": "municipality_name",
    "casos_muni": "hospitalizations",
    "poblacio_muni": "population",
}

DTYPES: dict[str, type] = {
    "setmana_epidemiologica": int,
    "any": int,
    "data_inici": str,
    "data_final": str,
    "codi_regio": str,
    "sexe": str,
    "grup_edat": str,
    "municipality_code": str,
    "casos_muni": float,
    "poblacio_muni": float,
}


def _load_raw_data(hosp_file: Path) -> pd.DataFrame:
    """Load raw hospitalization data from CSV."""
    if not hosp_file.exists():
        raise FileNotFoundError(f"Hospitalization file not found: {hosp_file}")

    logger.info("Loading hospitalizations from %s", hosp_file)

    df = pd.read_csv(
        hosp_file,
        dtype=DTYPES,  # type: ignore[arg-type]
        usecols=list(COLUMN_MAPPING.keys()),
    )

    df = df.rename(columns=COLUMN_MAPPING)

    df["week_start"] = pd.to_datetime(
        df["week_start"], format="%d/%m/%Y", errors="coerce"
    ).dt.tz_localize(None)

    # Drop rows with missing municipality_code
    df = df[
        df["municipality_code"].notna() & (df["municipality_code"] != "")
    ]

    df = df.dropna(
        subset=["week_start", "municipality_code", "hospitalizations"]
    )

    df = df[
        (df["sex"] != "No disponible")
    ]

    df = df[df["hospitalizations"] >= 0]

    logger.info(
        "Loaded %d records, %d municipalities, weeks: %s to %s",
        len(df),
        df["municipality_code"].nunique(),
        df["week_start"].min().strftime("%Y-%m-%d"),
        df["week_start"].max().strftime("%Y-%m-%d"),
    )

    return df


def _aggregate_to_abs_week(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hospitalizations to ABS-week level (sum across sex/age)."""
    aggregated = (
        df.groupby(["week_start", "municipality_comarca_code", "municipality_code"], dropna=False)[
            "hospitalizations"
        ]
        .sum()
        .reset_index()
    )
    logger.info("Aggregated to %d ABS-week records", len(aggregated))
    return aggregated


def _load_comarca_mapping(muni_ref_file: Path | None) -> pd.DataFrame | None:
    """Load municipality reference for comarca names."""
    if muni_ref_file is None or not muni_ref_file.exists():
        return None

    df = pd.read_csv(muni_ref_file)
    if "municipality_comarca_code" in df.columns and "municipality_comarca_name" in df.columns:
        return df[["municipality_comarca_code", "municipality_comarca_name"]].drop_duplicates()
    return None


def plot_municipality_weekly_series(
    municipality_weekly: pd.DataFrame,
    output_path: Path,
    max_abs: int = 20,
    seed: int = 7,
) -> None:
    """Plot weekly time series for selected ABS areas."""
    unique_municipalities = municipality_weekly["municipality_code"].unique()
    n_municipalities = min(max_abs, len(unique_municipalities))
    rng = np.random.default_rng(seed)
    selected_municipalities = rng.choice(unique_municipalities, size=n_municipalities, replace=False)

    fig, ax = plt.subplots(figsize=(14, 6))

    for municipality_code in selected_municipalities:
        municipality_data = municipality_weekly[municipality_weekly["municipality_code"] == municipality_code]
        municipality_data = municipality_data.sort_values("week_start")
        ax.plot(
            municipality_data["week_start"],
            municipality_data["hospitalizations"],
            alpha=0.6,
            linewidth=1.5,
            label=f"ABS {municipality_code}",
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_xlabel("Week Start Date")
    ax.set_ylabel("Weekly Hospitalizations")
    ax.set_title(f"Weekly Hospitalizations by ABS Area (showing {n_municipalities} random ABS)")
    ax.grid(True, alpha=0.3)

    if n_municipalities <= 10:
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ABS weekly series to %s", output_path)


def plot_abs_heatmap(
    municipality_weekly: pd.DataFrame,
    output_path: Path,
    max_abs: int = 50,
    seed: int = 7,
) -> None:
    """Plot heatmap of hospitalizations (ABS × week)."""
    unique_municipalities = municipality_weekly["municipality_code"].unique()
    n_municipalities = min(max_abs, len(unique_municipalities))
    rng = np.random.default_rng(seed)
    selected_municipalities = rng.choice(unique_municipalities, size=n_municipalities, replace=False)

    subset = municipality_weekly[municipality_weekly["municipality_code"].isin(selected_municipalities)]
    pivot = subset.pivot_table(
        index="municipality_code",
        columns="week_start",
        values="hospitalizations",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(14, max(6, n_municipalities * 0.3)))

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
        cbar_kws={"label": "Weekly Hospitalizations"},
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

    ax.set_xlabel("Week")
    ax.set_ylabel("ABS Code")
    ax.set_title(f"Hospitalizations Heatmap by ABS (showing {n_municipalities} random ABS)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ABS heatmap to %s", output_path)


def plot_comarca_distribution(
    municipality_weekly: pd.DataFrame,
    municipality_comarca_names: pd.DataFrame | None,
    output_path: Path,
    max_comarques: int = 15,
) -> None:
    """Plot violin plots of hospitalization distribution by comarca."""
    weekly_by_comarca = (
        municipality_weekly.groupby(["week_start", "municipality_comarca_code"])["hospitalizations"]
        .sum()
        .reset_index()
    )

    comarca_totals = (
        weekly_by_comarca.groupby("municipality_comarca_code")["hospitalizations"]
        .sum()
        .sort_values(ascending=False)
    )
    selected_comarcas = comarca_totals.head(max_comarques).index.tolist()

    plot_data = weekly_by_comarca[
        weekly_by_comarca["municipality_comarca_code"].isin(selected_comarcas)
    ].copy()

    if municipality_comarca_names is not None:
        plot_data = plot_data.merge(municipality_comarca_names, on="municipality_comarca_code", how="left")
        plot_data["comarca_label"] = plot_data["municipality_comarca_name"].fillna(
            plot_data["municipality_comarca_code"]
        )
    else:
        plot_data["comarca_label"] = plot_data["municipality_comarca_code"]

    fig, ax = plt.subplots(figsize=(max(12, len(selected_comarcas) * 0.6), 6))

    sns.violinplot(
        data=plot_data,
        x="comarca_label",
        y="hospitalizations",
        ax=ax,
        palette="Set2",
    )

    ax.set_xlabel("Comarca")
    ax.set_ylabel("Weekly Hospitalizations")
    ax.set_title(
        f"Distribution of Weekly Hospitalizations by Comarca (top {len(selected_comarcas)})"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comarca distribution to %s", output_path)


def plot_demographic_breakdown(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot stacked bar charts by sex and age group."""
    demo_data = df.groupby(["sex", "age_group"])["hospitalizations"].sum().reset_index()

    pivot = demo_data.pivot(index="sex", columns="age_group", values="hospitalizations")
    pivot = pivot.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(pivot))
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(pivot.columns)))

    for idx, (age_group, color) in enumerate(zip(pivot.columns, colors)):
        values = np.asarray(pivot[age_group].values)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=age_group,
            color=color,
        )
        bottom = bottom + values

    ax.set_xlabel("Sex")
    ax.set_ylabel("Total Hospitalizations")
    ax.set_title("Hospitalizations by Sex and Age Group")
    ax.legend(title="Age Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved demographic breakdown to %s", output_path)


def compute_summary_statistics(
    municipality_weekly: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary statistics for hospitalizations."""
    stats = []

    total_hosp = municipality_weekly["hospitalizations"].sum()
    n_weeks = municipality_weekly["week_start"].nunique()
    n_municipalities = municipality_weekly["municipality_code"].nunique()

    weekly_totals = municipality_weekly.groupby("week_start")["hospitalizations"].sum()
    peak_week = weekly_totals.idxmax()
    peak_value = weekly_totals.max()

    abs_totals = (
        municipality_weekly.groupby("municipality_code")["hospitalizations"]
        .sum()
        .sort_values(ascending=False)
    )
    most_affected_abs = abs_totals.index[0]
    most_affected_value = abs_totals.iloc[0]

    stats.append(
        {
            "metric": "Total Hospitalizations",
            "value": int(total_hosp),
            "unit": "cases",
        }
    )
    stats.append(
        {
            "metric": "Number of Weeks",
            "value": n_weeks,
            "unit": "weeks",
        }
    )
    stats.append(
        {
            "metric": "Number of ABS Areas",
            "value": n_municipalities,
            "unit": "areas",
        }
    )
    # Ensure peak_week is formatted correctly
    if isinstance(peak_week, pd.Timestamp):
        peak_week_str = peak_week.strftime("%Y-%m-%d")
    else:
        peak_week_str = str(peak_week)

    stats.append(
        {
            "metric": "Peak Week",
            "value": peak_week_str,
            "unit": "date",
        }
    )
    stats.append(
        {
            "metric": "Peak Week Hospitalizations",
            "value": int(peak_value),
            "unit": "cases",
        }
    )
    stats.append(
        {
            "metric": "Most Affected ABS",
            "value": str(most_affected_abs),
            "unit": "ABS code",
        }
    )
    stats.append(
        {
            "metric": "Most Affected ABS Count",
            "value": int(most_affected_value),
            "unit": "cases",
        }
    )

    stats_df = pd.DataFrame(stats)
    return stats_df


def print_summary_statistics(stats_df: pd.DataFrame) -> None:
    """Print summary statistics table to console."""
    print("\n" + "=" * 80)
    print("RAW HOSPITALIZATIONS SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hospitalization-csv",
        type=Path,
        required=True,
        help="Path to hospitalization CSV file",
    )
    parser.add_argument(
        "--municipality-ref",
        type=Path,
        default=None,
        help="Path to municipality reference CSV (for comarca names)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/raw_hospitalizations"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--max-abs",
        type=int,
        default=20,
        help="Max ABS areas to include in line plot",
    )
    parser.add_argument(
        "--max-abs-heatmap",
        type=int,
        default=50,
        help="Max ABS areas to include in heatmap",
    )
    parser.add_argument(
        "--max-comarques",
        type=int,
        default=15,
        help="Max comarques to show in distribution plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling ABS areas",
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

    raw_df = _load_raw_data(args.hospitalization_csv)

    if raw_df.empty:
        logger.warning("No hospitalization data found")
        return

    municipality_weekly = _aggregate_to_abs_week(raw_df)

    municipality_comarca_names = None
    if args.municipality_ref:
        municipality_comarca_names = _load_comarca_mapping(args.municipality_ref)

    stats_df = compute_summary_statistics(municipality_weekly, raw_df)
    print_summary_statistics(stats_df)

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(args.stats_output, index=False)
        logger.info("Saved statistics to %s", args.stats_output)

    plot_municipality_weekly_series(
        municipality_weekly,
        args.output_dir / "municipality_weekly_series.png",
        max_abs=args.max_abs,
        seed=args.seed,
    )

    plot_abs_heatmap(
        municipality_weekly,
        args.output_dir / "abs_heatmap.png",
        max_abs=args.max_abs_heatmap,
        seed=args.seed,
    )

    plot_comarca_distribution(
        municipality_weekly,
        municipality_comarca_names,
        args.output_dir / "comarca_distribution.png",
        max_comarques=args.max_comarques,
    )

    plot_demographic_breakdown(
        raw_df,
        args.output_dir / "demographic_breakdown.png",
    )


if __name__ == "__main__":
    main()
