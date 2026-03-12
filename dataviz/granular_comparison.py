from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
import pandas as pd
import seaborn as sns

from evaluation.granular_export import GRANULAR_FIELDNAMES, GRANULAR_KEY_COLUMNS

logger = logging.getLogger("granular_comparison")

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

_VALUE_COLUMNS = ["observed", "abs_error", "sq_error", "smape_num", "smape_den"]
_METADATA_COLUMNS = [
    "region_id",
    "region_label",
    "window_start_date",
    "target_index",
    "target_date",
]
TARGET_ORDER = ["wastewater", "cases", "hospitalizations", "deaths"]
TARGET_LABELS = {
    "wastewater": "Wastewater",
    "cases": "Cases",
    "hospitalizations": "Hosp.",
    "deaths": "Deaths",
}
DEFAULT_KNOCKOUT_ABLATIONS = ["no_mobility", "no_regions", "no_mobility_and_regions"]
ERROR_DIVERGING_CMAP = "RdYlGn_r"
DEFAULT_REGION_GEOJSON = Path("data/files/geo/fl_municipios_catalonia.geojson")


def _ordered_targets(values: pd.Series) -> list[str]:
    targets = [target for target in TARGET_ORDER if target in set(values.dropna())]
    for target in sorted(values.dropna().unique()):
        if target not in targets:
            targets.append(str(target))
    return targets


def _format_target_label(target: str) -> str:
    return TARGET_LABELS.get(target, str(target).replace("_", " ").title())


def _signed_norm(values: pd.Series | list[float]) -> colors.Normalize:
    series = pd.Series(values, dtype=float)
    max_abs = float(series.abs().max()) if not series.empty else 0.0
    max_abs = max(max_abs, 1e-6)
    return colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _value_colors(values: pd.Series | list[float]) -> list[tuple[float, float, float, float]]:
    cmap = colormaps[ERROR_DIVERGING_CMAP]
    norm = _signed_norm(values)
    return [cmap(norm(float(value))) for value in values]


def _load_granular_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"region_id": str})
    missing = [column for column in GRANULAR_FIELDNAMES if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required granular columns: {missing}")
    duplicate_mask = df.duplicated(GRANULAR_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"{path} contains {duplicate_count} duplicate granular keys; strict pairing requires uniqueness"
        )
    df["region_id"] = df["region_id"].fillna("").astype(str)
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    df["window_start_date"] = pd.to_datetime(df["window_start_date"], errors="coerce")
    return df


def _validate_join_coverage(
    *,
    baseline_rows: int,
    candidate_rows: int,
    matched_rows: int,
    min_join_coverage: float,
) -> None:
    baseline_coverage = matched_rows / max(1, baseline_rows)
    candidate_coverage = matched_rows / max(1, candidate_rows)
    if (
        baseline_coverage + 1e-12 < min_join_coverage
        or candidate_coverage + 1e-12 < min_join_coverage
    ):
        raise ValueError(
            "Granular join coverage below threshold: "
            f"baseline={baseline_coverage:.3f}, candidate={candidate_coverage:.3f}, "
            f"required>={min_join_coverage:.3f}"
        )


def _prepare_joined_frame(
    *,
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    min_join_coverage: float,
) -> pd.DataFrame:
    merged = baseline_df.merge(
        candidate_df,
        on=GRANULAR_KEY_COLUMNS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    _validate_join_coverage(
        baseline_rows=len(baseline_df),
        candidate_rows=len(candidate_df),
        matched_rows=len(merged),
        min_join_coverage=min_join_coverage,
    )

    for column in _METADATA_COLUMNS:
        baseline_col = f"{column}_baseline"
        candidate_col = f"{column}_candidate"
        if baseline_col in merged.columns and candidate_col in merged.columns:
            mismatched = merged[baseline_col].fillna("").astype(str) != merged[
                candidate_col
            ].fillna("").astype(str)
            if mismatched.any():
                raise ValueError(
                    f"Granular join matched rows with conflicting metadata column '{column}'"
                )
            merged[column] = merged[baseline_col]
            merged = merged.drop(columns=[baseline_col, candidate_col])

    for value_column in _VALUE_COLUMNS:
        merged[f"{value_column}_delta"] = (
            merged[f"{value_column}_candidate"] - merged[f"{value_column}_baseline"]
        )
        if value_column != "observed":
            merged[f"{value_column}_uplift"] = (
                merged[f"{value_column}_baseline"] - merged[f"{value_column}_candidate"]
            )

    merged["smape_baseline"] = merged["smape_num_baseline"] / merged[
        "smape_den_baseline"
    ].clip(lower=1e-6)
    merged["smape_candidate"] = merged["smape_num_candidate"] / merged[
        "smape_den_candidate"
    ].clip(lower=1e-6)
    merged["smape_delta"] = merged["smape_candidate"] - merged["smape_baseline"]
    merged["smape_uplift"] = merged["smape_baseline"] - merged["smape_candidate"]
    return merged


def _aggregate_pairs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            count=("abs_error_baseline", "size"),
            observed_mean=("observed_baseline", "mean"),
            baseline_mae=("abs_error_baseline", "mean"),
            candidate_mae=("abs_error_candidate", "mean"),
            baseline_mse=("sq_error_baseline", "mean"),
            candidate_mse=("sq_error_candidate", "mean"),
            baseline_smape=("smape_baseline", "mean"),
            candidate_smape=("smape_candidate", "mean"),
            abs_error_delta_mean=("abs_error_delta", "mean"),
            abs_error_uplift_mean=("abs_error_uplift", "mean"),
            sq_error_delta_mean=("sq_error_delta", "mean"),
            sq_error_uplift_mean=("sq_error_uplift", "mean"),
            smape_delta_mean=("smape_delta", "mean"),
            smape_uplift_mean=("smape_uplift", "mean"),
        )
        .reset_index()
    )
    grouped["baseline_rmse"] = grouped["baseline_mse"].clip(lower=0.0).map(math.sqrt)
    grouped["candidate_rmse"] = grouped["candidate_mse"].clip(lower=0.0).map(math.sqrt)
    grouped["rmse_delta"] = grouped["candidate_rmse"] - grouped["baseline_rmse"]
    grouped["rmse_uplift"] = grouped["baseline_rmse"] - grouped["candidate_rmse"]
    grouped = grouped.drop(columns=["baseline_mse", "candidate_mse"])
    if "target" in grouped.columns:
        grouped["target"] = pd.Categorical(
            grouped["target"],
            categories=_ordered_targets(grouped["target"]),
            ordered=True,
        )
        grouped = grouped.sort_values(
            ["target", *[col for col in group_cols if col != "target"]]
        )
    return grouped


def _save_region_time_heatmap(
    region_time_df: pd.DataFrame,
    output_path: Path,
    *,
    top_regions: int = 20,
    max_date_ticks: int = 4,
) -> Path:
    targets = _ordered_targets(region_time_df["target"])
    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(18, max(5, len(targets) * 5)),
        squeeze=False,
    )
    for axis, target in zip(axes.flatten(), targets, strict=False):
        target_df = region_time_df[region_time_df["target"] == target].copy()
        top_region_labels = (
            target_df.groupby("region_label")["count"].sum().nlargest(top_regions).index
        )
        target_df = target_df[target_df["region_label"].isin(top_region_labels)]
        pivot = target_df.pivot_table(
            index="region_label",
            columns="target_date",
            values="abs_error_uplift_mean",
            aggfunc="mean",
        )
        if pivot.empty:
            axis.set_visible(False)
            continue
        sns.heatmap(
            pivot,
            ax=axis,
            cmap=ERROR_DIVERGING_CMAP,
            center=0.0,
            cbar=True,
            cbar_kws={"label": "MAE uplift"},
        )
        columns = list(pivot.columns)
        if columns:
            tick_count = min(max_date_ticks, len(columns))
            if tick_count == 1:
                tick_positions = [0]
            else:
                tick_positions = [
                    round(i * (len(columns) - 1) / (tick_count - 1))
                    for i in range(tick_count)
                ]
            tick_positions = sorted(set(tick_positions))
            tick_labels = []
            for pos in tick_positions:
                value = pd.Timestamp(columns[pos])
                tick_labels.append(value.strftime("%Y-%m-%d"))
            axis.set_xticks([pos + 0.5 for pos in tick_positions])
            axis.set_xticklabels(tick_labels, rotation=0, ha="center")
        axis.set_title(f"{_format_target_label(target)}: region-time MAE uplift")
        axis.set_xlabel("Target date")
        axis.set_ylabel("Region")
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_rolling_time_plot(
    time_df: pd.DataFrame,
    output_path: Path,
    *,
    rolling_window: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for target in _ordered_targets(time_df["target"]):
        target_df = time_df[time_df["target"] == target]
        ordered = target_df.sort_values("target_date").copy()
        ordered["rolling_abs_error_uplift"] = (
            ordered["abs_error_uplift_mean"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )
        ax.plot(
            ordered["target_date"],
            ordered["rolling_abs_error_uplift"],
            label=_format_target_label(str(target)),
            linewidth=2.0,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Rolling MAE uplift over time")
    ax.set_xlabel("Target date")
    ax.set_ylabel("MAE uplift")
    ax.legend(title="Target")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_horizon_curve(horizon_df: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=horizon_df,
        x="horizon",
        y="abs_error_uplift_mean",
        hue="target",
        marker="o",
        ax=ax,
        hue_order=_ordered_targets(horizon_df["target"]),
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_text(_format_target_label(text.get_text()))
    ax.set_title("MAE uplift by forecast horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE uplift")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_region_bars(
    region_df: pd.DataFrame,
    output_path: Path,
    *,
    n_regions: int = 8,
) -> Path:
    targets = _ordered_targets(region_df["target"])
    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(12, max(4, len(targets) * 4)),
        squeeze=False,
    )
    for axis, target in zip(axes.flatten(), targets, strict=False):
        target_df = region_df[region_df["target"] == target].copy()
        winners = target_df.nlargest(n_regions, "abs_error_uplift_mean")
        losers = target_df.nsmallest(n_regions, "abs_error_uplift_mean")
        plot_df = pd.concat([losers, winners], ignore_index=True).drop_duplicates(
            subset=["region_label"]
        )
        plot_df = plot_df.sort_values("abs_error_uplift_mean")
        if plot_df.empty:
            axis.set_visible(False)
            continue
        sns.barplot(
            data=plot_df,
            x="abs_error_uplift_mean",
            y="region_label",
            hue="region_label",
            orient="h",
            ax=axis,
            palette=_value_colors(plot_df["abs_error_uplift_mean"]),
            legend=False,
        )
        axis.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
        axis.set_title(f"{_format_target_label(target)}: region gain/loss ranking")
        axis.set_xlabel("MAE uplift")
        axis.set_ylabel("Region")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_target_summary(target_df: pd.DataFrame, output_path: Path) -> Path:
    melted = target_df.melt(
        id_vars=["target"],
        value_vars=["abs_error_uplift_mean", "rmse_uplift", "smape_uplift_mean"],
        var_name="metric",
        value_name="uplift",
    )
    metric_labels = {
        "abs_error_uplift_mean": "MAE uplift",
        "rmse_uplift": "RMSE uplift",
        "smape_uplift_mean": "sMAPE uplift",
    }
    melted["metric"] = melted["metric"].map(metric_labels)
    melted["target_label"] = melted["target"].map(_format_target_label)
    target_order = [_format_target_label(target) for target in _ordered_targets(melted["target"])]
    metric_palette = {
        "MAE uplift": "#4C72B0",
        "RMSE uplift": "#DD8452",
        "sMAPE uplift": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="target_label",
        y="uplift",
        order=target_order,
        hue="metric",
        palette=metric_palette,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Target-level uplift summary")
    ax.set_xlabel("Target")
    ax.set_ylabel("Uplift")
    for patch, uplift in zip(ax.patches, melted["uplift"], strict=False):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        va = "bottom" if uplift >= 0 else "top"
        y_text = y + 0.003 if uplift >= 0 else y - 0.003
        ax.text(
            x,
            y_text,
            f"{uplift:.3f}",
            ha="center",
            va=va,
            fontsize=8,
            color=_value_colors([uplift])[0],
        )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_target_horizon_heatmap(horizon_df: pd.DataFrame, output_path: Path) -> Path:
    pivot = (
        horizon_df.assign(target_label=horizon_df["target"].map(_format_target_label))
        .pivot_table(
            index="target_label",
            columns="horizon",
            values="abs_error_uplift_mean",
            aggfunc="mean",
            observed=False,
        )
        .reindex(
            [_format_target_label(target) for target in _ordered_targets(horizon_df["target"])]
        )
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(
        pivot,
        cmap=ERROR_DIVERGING_CMAP,
        center=0.0,
        annot=True,
        fmt=".3f",
        linewidths=0.4,
        cbar_kws={"label": "MAE uplift"},
        ax=ax,
    )
    ax.set_title("MAE uplift by target and horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_uplift_distribution(paired_df: pd.DataFrame, output_path: Path) -> Path:
    plot_df = paired_df[["target", "abs_error_uplift"]].copy()
    plot_df["target_label"] = plot_df["target"].map(_format_target_label)
    ordered_labels = [
        _format_target_label(target) for target in _ordered_targets(plot_df["target"])
    ]
    median_by_target = plot_df.groupby("target_label", sort=False)["abs_error_uplift"].median()
    palette = {
        label: _value_colors([median_by_target.get(label, 0.0)])[0]
        for label in ordered_labels
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxenplot(
        data=plot_df,
        x="target_label",
        y="abs_error_uplift",
        order=ordered_labels,
        hue="target_label",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Per-example MAE uplift distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("MAE uplift")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_region_scatter(region_df: pd.DataFrame, output_path: Path) -> Path:
    plot_df = region_df.copy()
    plot_df["target_label"] = plot_df["target"].map(_format_target_label)
    norm = _signed_norm(plot_df["abs_error_uplift_mean"])
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=plot_df,
        x="baseline_mae",
        y="abs_error_uplift_mean",
        hue="abs_error_uplift_mean",
        palette=ERROR_DIVERGING_CMAP,
        hue_norm=norm,
        style="target_label",
        size="count",
        sizes=(20, 220),
        alpha=0.8,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Regional uplift vs baseline error")
    ax.set_xlabel("Baseline MAE")
    ax.set_ylabel("MAE uplift")

    label_df = pd.concat(
        [
            plot_df.nlargest(6, "abs_error_uplift_mean"),
            plot_df.nsmallest(6, "abs_error_uplift_mean"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["target", "region_label"])
    for row in label_df.itertuples():
        ax.text(
            row.baseline_mae,
            row.abs_error_uplift_mean,
            str(row.region_label),
            fontsize=8,
            alpha=0.85,
        )

    ax.legend(title="Uplift / target / count", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_target_choropleths(
    region_df: pd.DataFrame,
    output_path: Path,
    *,
    region_geojson_path: str | Path = DEFAULT_REGION_GEOJSON,
) -> Path:
    geo_df = gpd.read_file(region_geojson_path)
    geo_df["region_id"] = geo_df["id"].astype(str).str.zfill(5)

    target_order = _ordered_targets(region_df["target"])
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    norm = _signed_norm(region_df["abs_error_uplift_mean"])

    for axis, target in zip(axes, TARGET_ORDER, strict=False):
        if target not in target_order:
            axis.set_axis_off()
            continue

        target_df = region_df[region_df["target"] == target].copy()
        target_df["region_id"] = target_df["region_id"].fillna("").astype(str).str.zfill(5)
        merged = geo_df.merge(
            target_df[["region_id", "abs_error_uplift_mean"]],
            on="region_id",
            how="left",
        )
        merged.plot(
            column="abs_error_uplift_mean",
            cmap=ERROR_DIVERGING_CMAP,
            norm=norm,
            legend=False,
            missing_kwds={"color": "#d9d9d9"},
            edgecolor="#444444",
            linewidth=0.1,
            ax=axis,
        )
        axis.set_title(_format_target_label(target))
        axis.set_axis_off()

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=ERROR_DIVERGING_CMAP)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Mean MAE uplift")
    fig.suptitle("Per-target mean regional MAE uplift", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.92, top=0.90, bottom=0.02, wspace=0.02)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _find_single_granular_csv(
    *,
    training_dir: Path,
    campaign_id: str,
    ablation_name: str,
    split: str,
) -> Path:
    experiment_dir = training_dir / f"mn5_ablation__{campaign_id}__{ablation_name}"
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Ablation experiment not found: {experiment_dir}")

    matches = sorted(experiment_dir.glob(f"*/{split}_granular_metrics.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No {split}_granular_metrics.csv found under {experiment_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one granular CSV for {ablation_name}, found {len(matches)}"
        )
    return matches[0]


def compare_ablation_suite(
    *,
    training_dir: str | Path,
    campaign_id: str,
    baseline_ablation: str,
    candidate_ablations: list[str],
    output_dir: str | Path,
    split: str = "test",
    min_join_coverage: float = 1.0,
    rolling_window: int = 7,
    region_geojson_path: str | Path = DEFAULT_REGION_GEOJSON,
) -> dict[str, dict[str, object]]:
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = _find_single_granular_csv(
        training_dir=training_dir,
        campaign_id=campaign_id,
        ablation_name=baseline_ablation,
        split=split,
    )
    results: dict[str, dict[str, object]] = {}

    for candidate_ablation in candidate_ablations:
        candidate_csv = _find_single_granular_csv(
            training_dir=training_dir,
            campaign_id=campaign_id,
            ablation_name=candidate_ablation,
            split=split,
        )
        comparison_dir = output_dir / f"{baseline_ablation}__vs__{candidate_ablation}"
        logger.info(
            "Comparing %s against %s -> %s",
            candidate_ablation,
            baseline_ablation,
            comparison_dir,
        )
        results[candidate_ablation] = compare_granular_csvs(
            baseline_csv=baseline_csv,
            candidate_csv=candidate_csv,
            output_dir=comparison_dir,
            min_join_coverage=min_join_coverage,
            rolling_window=rolling_window,
            region_geojson_path=region_geojson_path,
        )

    summary_rows = []
    for candidate_ablation, artifacts in results.items():
        target_df = pd.read_csv(artifacts["tables"]["target_aggregates"])
        for row in target_df.itertuples():
            summary_rows.append(
                {
                    "baseline_ablation": baseline_ablation,
                    "candidate_ablation": candidate_ablation,
                    "target": row.target,
                    "count": row.count,
                    "baseline_mae": row.baseline_mae,
                    "candidate_mae": row.candidate_mae,
                    "mae_uplift": row.abs_error_uplift_mean,
                    "baseline_rmse": row.baseline_rmse,
                    "candidate_rmse": row.candidate_rmse,
                    "rmse_uplift": row.rmse_uplift,
                    "baseline_smape": row.baseline_smape,
                    "candidate_smape": row.candidate_smape,
                    "smape_uplift": row.smape_uplift_mean,
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_dir / "suite_target_summary.csv", index=False)
    return results


def compare_granular_csvs(
    *,
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    output_dir: str | Path,
    min_join_coverage: float = 1.0,
    rolling_window: int = 7,
    region_geojson_path: str | Path = DEFAULT_REGION_GEOJSON,
) -> dict[str, object]:
    """Compare two granular eval CSVs, write paired tables, and render uplift plots."""
    baseline_path = Path(baseline_csv)
    candidate_path = Path(candidate_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = _load_granular_csv(baseline_path)
    candidate_df = _load_granular_csv(candidate_path)
    paired_df = _prepare_joined_frame(
        baseline_df=baseline_df,
        candidate_df=candidate_df,
        min_join_coverage=min_join_coverage,
    )

    region_df = _aggregate_pairs(
        paired_df,
        ["target", "node_id", "region_id", "region_label"],
    )
    time_df = _aggregate_pairs(paired_df, ["target", "target_date"])
    region_time_df = _aggregate_pairs(
        paired_df,
        ["target", "node_id", "region_label", "target_date"],
    )
    horizon_df = _aggregate_pairs(paired_df, ["target", "horizon"])
    target_df = _aggregate_pairs(paired_df, ["target"])

    tables = {
        "paired_rows": output_dir / "paired_row_deltas.csv",
        "region_aggregates": output_dir / "region_aggregates.csv",
        "time_aggregates": output_dir / "time_aggregates.csv",
        "region_time_aggregates": output_dir / "region_time_aggregates.csv",
        "horizon_aggregates": output_dir / "horizon_aggregates.csv",
        "target_aggregates": output_dir / "target_aggregates.csv",
    }
    paired_df.to_csv(tables["paired_rows"], index=False)
    region_df.to_csv(tables["region_aggregates"], index=False)
    time_df.to_csv(tables["time_aggregates"], index=False)
    region_time_df.to_csv(tables["region_time_aggregates"], index=False)
    horizon_df.to_csv(tables["horizon_aggregates"], index=False)
    target_df.to_csv(tables["target_aggregates"], index=False)

    plots = {
        "region_time_heatmap": _save_region_time_heatmap(
            region_time_df, output_dir / "region_time_heatmap.png"
        ),
        "rolling_time_uplift": _save_rolling_time_plot(
            time_df,
            output_dir / "rolling_time_uplift.png",
            rolling_window=rolling_window,
        ),
        "horizon_uplift_curve": _save_horizon_curve(
            horizon_df, output_dir / "horizon_uplift_curve.png"
        ),
        "region_gain_loss_bars": _save_region_bars(
            region_df, output_dir / "region_gain_loss_bars.png"
        ),
        "target_summary": _save_target_summary(
            target_df, output_dir / "target_summary.png"
        ),
        "target_horizon_heatmap": _save_target_horizon_heatmap(
            horizon_df, output_dir / "target_horizon_heatmap.png"
        ),
        "uplift_distribution": _save_uplift_distribution(
            paired_df, output_dir / "uplift_distribution.png"
        ),
        "region_scatter": _save_region_scatter(
            region_df, output_dir / "region_scatter.png"
        ),
        "target_choropleths": _save_target_choropleths(
            region_df,
            output_dir / "target_choropleths.png",
            region_geojson_path=region_geojson_path,
        ),
    }

    return {
        "baseline_rows": len(baseline_df),
        "candidate_rows": len(candidate_df),
        "matched_rows": len(paired_df),
        "tables": tables,
        "plots": plots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render granular comparison artifacts for ablation eval CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m dataviz.granular_comparison \
      --baseline-csv outputs/.../baseline/test_granular_metrics.csv \
      --candidate-csv outputs/.../no_mobility/test_granular_metrics.csv \
      --output-dir outputs/reports/granular_compare/no_mobility

  uv run python -m dataviz.granular_comparison \
      --training-dir outputs/training \
      --campaign-id ablation_37693513 \
      --output-dir outputs/reports/granular_compare/ablation_37693513
        """,
    )
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--campaign-id", type=str, default=None)
    parser.add_argument("--baseline-ablation", type=str, default="baseline")
    parser.add_argument(
        "--candidate-ablation",
        action="append",
        dest="candidate_ablations",
        help="Candidate ablation name. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--region-geojson-path",
        type=Path,
        default=DEFAULT_REGION_GEOJSON,
    )
    parser.add_argument("--min-join-coverage", type=float, default=1.0)
    parser.add_argument("--rolling-window", type=int, default=7)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.baseline_csv is not None or args.candidate_csv is not None:
        if args.baseline_csv is None or args.candidate_csv is None:
            raise ValueError("Pairwise mode requires both --baseline-csv and --candidate-csv")
        artifacts = compare_granular_csvs(
            baseline_csv=args.baseline_csv,
            candidate_csv=args.candidate_csv,
            output_dir=args.output_dir,
            min_join_coverage=args.min_join_coverage,
            rolling_window=args.rolling_window,
            region_geojson_path=args.region_geojson_path,
        )
        logger.info(
            "Compared %s vs %s with %d matched rows",
            args.candidate_csv,
            args.baseline_csv,
            artifacts["matched_rows"],
        )
        return 0

    if args.campaign_id is None:
        raise ValueError(
            "Batch mode requires --campaign-id unless --baseline-csv/--candidate-csv are used"
        )

    candidate_ablations = args.candidate_ablations or DEFAULT_KNOCKOUT_ABLATIONS
    results = compare_ablation_suite(
        training_dir=args.training_dir,
        campaign_id=args.campaign_id,
        baseline_ablation=args.baseline_ablation,
        candidate_ablations=candidate_ablations,
        output_dir=args.output_dir,
        split=args.split,
        min_join_coverage=args.min_join_coverage,
        rolling_window=args.rolling_window,
        region_geojson_path=args.region_geojson_path,
    )
    logger.info("Generated %d granular comparison suites", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
