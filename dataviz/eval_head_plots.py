from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from data.preprocess.config import REGION_COORD
from dataviz.granular_comparison import (
    TARGET_ORDER,
    _format_target_label,
    _ordered_targets,
)
from dataviz.sparsity_analysis import get_wastewater_sparsity_mask
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.selection import WindowSelectionSpec, select_windows_by_loss
from plotting.forecast_plots import collect_forecast_samples_for_window_specs, make_forecast_figure

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

PER_HEAD_NODE_METRICS_REQUIRED_COLUMNS = [
    "target",
    "node_id",
    "region_id",
    "region_label",
    "population",
    "observed_count",
    "mae",
    "rmse",
]

BASELINE_DELTA_REQUIRED_COLUMNS = [
    "target",
    "baseline_model",
    "metric",
    "model_value",
    "baseline_value",
]
GRANULAR_METRICS_REQUIRED_COLUMNS = [
    "target",
    "node_id",
    "window_start",
    "abs_error",
]

_TARGET_TO_MASK_VARIABLES = {
    "cases": ["cases_mask"],
    "hospitalizations": ["hosp_mask", "hospitalizations_mask"],
    "deaths": ["deaths_mask"],
}
_TARGET_TO_PLOT_NAME = {
    "hospitalizations": "hosp",
    "wastewater": "ww",
    "cases": "cases",
    "deaths": "deaths",
}
_TARGET_TO_LATENT_OVERLAY = {
    "hospitalizations": ("latent_i", "Latent I"),
    "wastewater": ("latent_i", "Latent I"),
    "cases": ("latent_i", "Latent I"),
    "deaths": ("latent_d", "Latent D"),
}
_BASELINE_COMPARISON_METRICS = {
    "mae": "MAE",
    "r2": "R²",
}
_TARGET_PLOT_BASELINES = {"sarima", "exp_smoothing", "var"}


def _format_baseline_model_label(name: str) -> str:
    return name.replace("_", " ")


def _baseline_model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_per_head_node_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"region_id": str})
    missing = [
        column
        for column in PER_HEAD_NODE_METRICS_REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["region_id"] = df["region_id"].fillna("").astype(str)
    df["target"] = pd.Categorical(
        df["target"],
        categories=[target for target in TARGET_ORDER if target in set(df["target"])],
        ordered=True,
    )
    return df


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(f"{path.suffix}.meta.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing node metrics sidecar: {sidecar_path}")
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _resolve_granular_metrics_csv_path(
    *,
    per_head_node_metrics_csv: Path,
    sidecar: dict[str, Any],
    granular_metrics_csv: str | Path | None,
) -> Path | None:
    if granular_metrics_csv is not None:
        return Path(granular_metrics_csv)

    split = str(sidecar.get("split", "test")).strip() or "test"
    candidates = [
        per_head_node_metrics_csv.parent / f"{split}_granular.csv",
        per_head_node_metrics_csv.parent / f"{split}_granular_metrics.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_granular_metrics(
    *,
    path: Path,
    split: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [
        column
        for column in GRANULAR_METRICS_REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.strip().lower()].copy()

    if df.empty:
        return df

    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
    df["window_start"] = pd.to_numeric(df["window_start"], errors="coerce")
    df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
    df = df.dropna(subset=["target", "node_id", "window_start", "abs_error"]).copy()
    if df.empty:
        return df
    df["node_id"] = df["node_id"].astype(int)
    df["window_start"] = df["window_start"].astype(int)
    return df


def _load_baseline_delta_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [
        column for column in BASELINE_DELTA_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[
        df["metric"].isin(_BASELINE_COMPARISON_METRICS)
        & (df["target"].astype(str) != "joint_observation")
    ].copy()
    if df.empty:
        return df

    df = df[df["baseline_model"].astype(str).isin(_TARGET_PLOT_BASELINES)].copy()
    if df.empty:
        return df

    ordered_targets = [target for target in TARGET_ORDER if target in set(df["target"])]
    df["target"] = pd.Categorical(df["target"], categories=ordered_targets, ordered=True)
    df["metric"] = pd.Categorical(
        df["metric"],
        categories=list(_BASELINE_COMPARISON_METRICS.keys()),
        ordered=True,
    )
    df["model_value"] = pd.to_numeric(df["model_value"], errors="coerce")
    df["baseline_value"] = pd.to_numeric(df["baseline_value"], errors="coerce")
    return df


def _select_run(dataset: xr.Dataset, run_id: Any) -> xr.Dataset:
    if run_id is None or (
        "run_id" not in dataset.dims and "run_id" not in dataset.coords
    ):
        return dataset

    candidates = [run_id]
    if isinstance(run_id, str):
        stripped = run_id.strip()
        candidates.append(stripped)
        try:
            candidates.append(int(stripped))
        except ValueError:
            pass
    else:
        candidates.append(str(run_id))

    for candidate in candidates:
        try:
            return dataset.sel(run_id=candidate)
        except Exception:
            continue
    logger.warning(
        "Could not select run_id=%s from dataset; using full dataset", run_id
    )
    return dataset


def _get_mask_array(dataset: xr.Dataset, target: str) -> np.ndarray:
    if target == "wastewater":
        return get_wastewater_sparsity_mask(dataset)

    for mask_name in _TARGET_TO_MASK_VARIABLES[target]:
        if mask_name in dataset:
            mask = dataset[mask_name]
            if "run_id" in mask.dims:
                mask = mask.isel(run_id=0)
            return (mask.values == 0).astype(bool)

    raise ValueError(f"No canonical mask found for target '{target}'")


def _compute_canonical_sparsity(
    *,
    dataset_path: Path,
    run_id: Any,
) -> pd.DataFrame:
    dataset = xr.open_zarr(dataset_path)
    try:
        dataset = _select_run(dataset, run_id)
        region_ids = dataset[REGION_COORD].values
        rows: list[dict[str, Any]] = []
        for target in ["hospitalizations", "wastewater", "cases", "deaths"]:
            try:
                sparsity_mask = _get_mask_array(dataset, target)
            except ValueError:
                continue

            sparsity_pct = sparsity_mask.mean(axis=0) * 100.0
            for region_id, sparsity_value in zip(
                region_ids, sparsity_pct, strict=False
            ):
                rows.append(
                    {
                        "target": target,
                        "region_id": str(region_id),
                        "sparsity_pct": float(sparsity_value),
                    }
                )
        return pd.DataFrame(rows)
    finally:
        dataset.close()


def _save_scatter_with_trend(
    *,
    plot_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y="mae",
        size="observed_count",
        sizes=(30, 240),
        alpha=0.75,
        color="#4C72B0",
        ax=ax,
        legend=False,
    )
    if len(plot_df) >= 2:
        sns.regplot(
            data=plot_df,
            x=x_col,
            y="mae",
            scatter=False,
            color="#222222",
            line_kws={"linestyle": "--", "linewidth": 1.5},
            ax=ax,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("MAE")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _select_quartile_nodes_in_order(
    *,
    node_mae: dict[int, float],
) -> dict[str, list[int]]:
    quartile_names = [
        "Q1 (Best MAE)",
        "Q2 (Good MAE)",
        "Q3 (Poor MAE)",
        "Q4 (Worst MAE)",
    ]
    if not node_mae:
        return {name: [] for name in quartile_names}

    sorted_nodes = sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))
    maes = [mae for _node_id, mae in sorted_nodes]
    q1_cutoff = np.percentile(maes, 25)
    q2_cutoff = np.percentile(maes, 50)
    q3_cutoff = np.percentile(maes, 75)
    quartiles = {name: [] for name in quartile_names}
    for node_id, mae in sorted_nodes:
        if mae <= q1_cutoff:
            quartiles["Q1 (Best MAE)"].append(node_id)
        elif mae <= q2_cutoff:
            quartiles["Q2 (Good MAE)"].append(node_id)
        elif mae <= q3_cutoff:
            quartiles["Q3 (Poor MAE)"].append(node_id)
        else:
            quartiles["Q4 (Worst MAE)"].append(node_id)
    return quartiles


def _select_representative_window_specs_for_target(
    *,
    target_df: pd.DataFrame,
    target_name: str,
    granular_df: pd.DataFrame,
    samples_per_quartile: int,
) -> dict[str, list[WindowSelectionSpec]]:
    node_mae = {
        int(row.node_id): float(row.mae)
        for row in target_df.itertuples()
        if pd.notna(row.mae)
    }
    node_groups = _select_quartile_nodes_in_order(node_mae=node_mae)
    if granular_df.empty:
        return {}

    target_granular = granular_df[granular_df["target"].astype(str) == target_name].copy()
    if target_granular.empty:
        return {}

    per_window = (
        target_granular.groupby(["node_id", "window_start"], dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            observed_points=("abs_error", "size"),
        )
        .reset_index()
    )
    if per_window.empty:
        return {}

    candidate_specs: dict[int, WindowSelectionSpec] = {}
    for node_id, node_group in per_window.groupby("node_id", dropna=False):
        maes = pd.to_numeric(node_group["mae"], errors="coerce").dropna()
        if maes.empty:
            continue
        median_mae = float(maes.median())
        ranked = node_group.assign(
            median_distance=(pd.to_numeric(node_group["mae"], errors="coerce") - median_mae).abs()
        ).sort_values(
            by=["median_distance", "observed_points", "mae", "window_start"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        best = ranked.iloc[0]
        candidate_specs[int(node_id)] = WindowSelectionSpec(
            node_id=int(node_id),
            window_start=int(best["window_start"]),
            score=float(best["mae"]),
            observed_targets=(target_name,),
            observed_points=int(best["observed_points"]),
        )

    grouped_specs: dict[str, list[WindowSelectionSpec]] = {}
    for group_name, node_ids in node_groups.items():
        specs: list[WindowSelectionSpec] = []
        for node_id in node_ids:
            spec = candidate_specs.get(int(node_id))
            if spec is None:
                continue
            specs.append(spec)
            if len(specs) >= samples_per_quartile:
                break
        if specs:
            grouped_specs[group_name] = specs
    return grouped_specs


def _group_samples_by_region_quartile(
    *,
    target_df: pd.DataFrame,
    target_name: str,
    plot_target: str,
    granular_df: pd.DataFrame,
    model: Any,
    loader: Any,
    samples_per_quartile: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped_specs = _select_representative_window_specs_for_target(
        target_df=target_df,
        target_name=target_name,
        granular_df=granular_df,
        samples_per_quartile=samples_per_quartile,
    )
    if not grouped_specs:
        return {}

    ordered_specs = [spec for specs in grouped_specs.values() for spec in specs]
    samples = collect_forecast_samples_for_window_specs(
        window_specs=ordered_specs,
        model=model,
        loader=loader,
        context_pre=30,
        context_post=30,
        target_names=[plot_target],
    )
    samples_by_key = {
        (int(sample["node_id"]), int(sample["window_start"])): sample for sample in samples
    }
    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for group_name, specs in grouped_specs.items():
        group_samples = [
            samples_by_key[(spec.node_id, spec.window_start)]
            for spec in specs
            if (spec.node_id, spec.window_start) in samples_by_key
        ]
        if group_samples:
            grouped_samples[group_name] = group_samples
    return grouped_samples


def _select_joint_window_specs_by_quartile(
    *,
    granular_df: pd.DataFrame,
    samples_per_quartile: int,
) -> dict[str, list[WindowSelectionSpec]]:
    if granular_df.empty:
        return {}

    per_window = (
        granular_df.groupby(["node_id", "window_start"], dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            observed_points=("abs_error", "size"),
            observed_targets=("target", lambda vals: tuple(sorted(set(vals.astype(str))))),
        )
        .reset_index()
    )
    if per_window.empty:
        return {}

    window_specs = [
        WindowSelectionSpec(
            node_id=int(row.node_id),
            window_start=int(row.window_start),
            score=float(row.mae),
            observed_targets=tuple(row.observed_targets),
            observed_points=int(row.observed_points),
        )
        for row in per_window.itertuples()
        if pd.notna(row.mae)
    ]
    if not window_specs:
        return {}

    grouped_specs = select_windows_by_loss(
        window_specs=window_specs,
        samples_per_group=samples_per_quartile,
    )
    return {group: specs for group, specs in grouped_specs.items() if specs}


def _render_joint_latent_quartile_plots(
    *,
    granular_df: pd.DataFrame,
    sidecar: dict[str, Any],
    output_dir: Path,
    samples_per_quartile: int,
) -> dict[str, Path]:
    checkpoint_path_raw = sidecar.get("checkpoint_path")
    if not checkpoint_path_raw or granular_df.empty:
        return {}
    if granular_df["target"].astype(str).nunique() < 2:
        return {}

    grouped_specs = _select_joint_window_specs_by_quartile(
        granular_df=granular_df,
        samples_per_quartile=samples_per_quartile,
    )
    if not grouped_specs:
        return {}

    checkpoint_path = Path(checkpoint_path_raw)
    model, config, _checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device="auto",
    )
    loader, _region_embeddings = build_loader_from_config(
        config,
        split=str(sidecar.get("split", "test")),
        device="auto",
    )

    ordered_specs = [spec for specs in grouped_specs.values() for spec in specs]
    samples = collect_forecast_samples_for_window_specs(
        window_specs=ordered_specs,
        model=model,
        loader=loader,
        context_pre=30,
        context_post=30,
        target_names=["cases"],
    )
    samples_by_key = {
        (int(sample["node_id"]), int(sample["window_start"])): sample for sample in samples
    }
    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for group_name, specs in grouped_specs.items():
        matched_samples = [
            samples_by_key[(spec.node_id, spec.window_start)]
            for spec in specs
            if (spec.node_id, spec.window_start) in samples_by_key
        ]
        if matched_samples:
            grouped_samples[group_name] = matched_samples
    if not grouped_samples:
        return {}
    if not any(
        isinstance(sample.get("latents"), dict) and sample["latents"]
        for samples in grouped_samples.values()
        for sample in samples
    ):
        return {}

    latent_specs = {
        "latent_s": "Latent S",
        "latent_i": "Latent I",
        "latent_r": "Latent R",
        "latent_d": "Latent D",
    }
    artifacts: dict[str, Path] = {}
    for latent_name, latent_label in latent_specs.items():
        fig = make_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=30,
            context_post=30,
            target=latent_name,
            target_label=latent_label,
            figure_title=f"{latent_label} (joint-window MAE quartiles)",
            shared_xlabel="Time (days relative to forecast start)",
            payload_collection="latents",
            connect_from_history=False,
        )
        if fig is None:
            continue
        output_path = output_dir / f"forecast_examples_quartiles_joint_{latent_name}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        artifacts[f"forecast_examples_quartiles_joint_{latent_name}"] = output_path

    return artifacts


def _render_forecast_quartile_plots(
    *,
    metrics_df: pd.DataFrame,
    granular_df: pd.DataFrame,
    sidecar: dict[str, Any],
    output_dir: Path,
    samples_per_quartile: int,
) -> dict[str, Path]:
    checkpoint_path_raw = sidecar.get("checkpoint_path")
    if not checkpoint_path_raw:
        logger.info(
            "Skipping forecast example plots because checkpoint_path is absent from sidecar"
        )
        return {}

    checkpoint_path = Path(checkpoint_path_raw)
    model, config, _checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device="auto",
    )
    loader, _region_embeddings = build_loader_from_config(
        config,
        split=str(sidecar.get("split", "test")),
        device="auto",
    )

    artifacts: dict[str, Path] = {}
    for target in _ordered_targets(metrics_df["target"].astype(str)):
        plot_target = _TARGET_TO_PLOT_NAME.get(target)
        if plot_target is None:
            continue
        target_df = metrics_df[metrics_df["target"].astype(str) == target].copy()
        grouped_samples = _group_samples_by_region_quartile(
            target_df=target_df,
            target_name=target,
            plot_target=plot_target,
            granular_df=granular_df,
            model=model,
            loader=loader,
            samples_per_quartile=samples_per_quartile,
        )
        if not grouped_samples:
            continue
        fig = make_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=30,
            context_post=30,
            target=plot_target,
            target_label=_format_target_label(target),
            figure_title=f"{_format_target_label(target)} (MAE)",
            shared_xlabel="Time (days relative to forecast start)",
            overlay_target=_TARGET_TO_LATENT_OVERLAY[target][0],
            overlay_label=_TARGET_TO_LATENT_OVERLAY[target][1],
        )
        if fig is None:
            continue
        output_path = output_dir / f"forecast_examples_quartiles_{target}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        artifacts[f"forecast_examples_quartiles_{target}"] = output_path

    return artifacts


def render_baseline_delta_plots(
    *,
    baseline_deltas_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    csv_path = Path(baseline_deltas_csv)
    output_dir = Path(output_dir) if output_dir is not None else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_df = _load_baseline_delta_metrics(csv_path)
    if delta_df.empty:
        logger.info("Skipping baseline delta plots because %s has no MAE/R² rows", csv_path)
        return {}

    split_prefix = csv_path.stem.removesuffix("_baseline_deltas")
    artifacts: dict[str, Path] = {}
    baseline_models = sorted(delta_df["baseline_model"].dropna().astype(str).unique())
    baseline_labels = {
        baseline_model: _format_baseline_model_label(baseline_model)
        for baseline_model in baseline_models
    }
    hue_order = ["Our model", *[baseline_labels[name] for name in baseline_models]]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][: len(hue_order)]

    for metric_key, metric_label in _BASELINE_COMPARISON_METRICS.items():
        metric_df = delta_df[delta_df["metric"] == metric_key].copy()
        metric_df = metric_df.dropna(subset=["model_value", "baseline_value"])
        if metric_df.empty:
            continue

        model_rows = (
            metric_df[["target", "model_value"]]
            .drop_duplicates(subset=["target"])
            .rename(columns={"model_value": "value"})
        )
        model_rows["series"] = "Our model"

        baseline_rows = metric_df[["target", "baseline_model", "baseline_value"]].rename(
            columns={"baseline_value": "value"}
        )
        baseline_rows["series"] = baseline_rows["baseline_model"].astype(str).map(
            baseline_labels
        )
        baseline_rows = baseline_rows.drop(columns=["baseline_model"])

        plot_df = pd.concat([model_rows, baseline_rows], ignore_index=True)
        plot_df["target_label"] = plot_df["target"].astype(str).map(_format_target_label)
        target_order = [
            _format_target_label(target)
            for target in _ordered_targets(metric_df["target"].astype(str))
        ]

        fig, axis = plt.subplots(figsize=(10.5, 5.5))
        sns.barplot(
            data=plot_df,
            x="target_label",
            y="value",
            hue="series",
            order=target_order,
            hue_order=hue_order,
            palette=palette,
            ax=axis,
        )
        title_suffix = "(lower is better)" if metric_key == "mae" else "(higher is better)"
        axis.set_title(f"{metric_label} {title_suffix}")
        axis.set_xlabel("Target")
        axis.set_ylabel(metric_label)
        if metric_key == "r2":
            axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)

        handles, labels = axis.get_legend_handles_labels()
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

        fig.suptitle(f"Model vs baselines by target: {metric_label}")
        if handles and labels:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                frameon=True,
            )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        output_path = output_dir / f"{split_prefix}_baseline_comparison_{metric_key}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        artifacts[f"baseline_comparison_{metric_key}"] = output_path

    return artifacts


def render_eval_per_head_plots(
    *,
    per_head_node_metrics_csv: str | Path,
    output_dir: str | Path | None = None,
    samples_per_quartile: int = 4,
    granular_metrics_csv: str | Path | None = None,
    forecast_window: str = "last",
) -> dict[str, Path]:
    csv_path = Path(per_head_node_metrics_csv)
    output_dir = Path(output_dir) if output_dir is not None else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _load_per_head_node_metrics(csv_path)
    sidecar = _load_sidecar(csv_path)
    granular_csv_path = _resolve_granular_metrics_csv_path(
        per_head_node_metrics_csv=csv_path,
        sidecar=sidecar,
        granular_metrics_csv=granular_metrics_csv,
    )
    dataset_path = Path(sidecar["dataset_path"])
    sparsity_df = _compute_canonical_sparsity(
        dataset_path=dataset_path,
        run_id=sidecar.get("run_id"),
    )
    joined = metrics_df.merge(
        sparsity_df,
        on=["target", "region_id"],
        how="left",
        validate="many_to_one",
    )
    joined["log10_population"] = np.log10(joined["population"].clip(lower=1.0))
    granular_df = pd.DataFrame(columns=GRANULAR_METRICS_REQUIRED_COLUMNS)
    if granular_csv_path is None:
        logger.warning(
            "Skipping representative per-head forecast plots because no granular metrics CSV was found near %s",
            csv_path,
        )
    elif not granular_csv_path.exists():
        logger.warning(
            "Skipping representative per-head forecast plots because granular metrics CSV does not exist: %s",
            granular_csv_path,
        )
    else:
        granular_df = _load_granular_metrics(
            path=granular_csv_path,
            split=str(sidecar.get("split", "test")),
        )

    artifacts: dict[str, Path] = {}
    for target in _ordered_targets(joined["target"].astype(str)):
        target_df = joined[joined["target"].astype(str) == target].copy()
        if target_df.empty:
            continue
        target_label = _format_target_label(target)
        population_path = output_dir / f"perf_vs_population_{target}.png"
        sparsity_path = output_dir / f"perf_vs_sparsity_{target}.png"
        artifacts[f"perf_vs_population_{target}"] = _save_scatter_with_trend(
            plot_df=target_df.dropna(subset=["log10_population", "mae"]),
            x_col="log10_population",
            x_label="log10(Population)",
            title=f"{target_label}: MAE vs Population",
            output_path=population_path,
        )
        artifacts[f"perf_vs_sparsity_{target}"] = _save_scatter_with_trend(
            plot_df=target_df.dropna(subset=["sparsity_pct", "mae"]),
            x_col="sparsity_pct",
            x_label="Input sparsity (%)",
            title=f"{target_label}: MAE vs Input Sparsity",
            output_path=sparsity_path,
        )

    artifacts.update(
        _render_forecast_quartile_plots(
            metrics_df=metrics_df,
            granular_df=granular_df,
            sidecar=sidecar,
            output_dir=output_dir,
            samples_per_quartile=samples_per_quartile,
        )
    )
    artifacts.update(
        _render_joint_latent_quartile_plots(
            granular_df=granular_df,
            sidecar=sidecar,
            output_dir=output_dir,
            samples_per_quartile=samples_per_quartile,
        )
    )

    return artifacts
