#!/usr/bin/env python
"""Quantify municipality-level representation skew and its quality tradeoffs.

Diagnoses:
1. Per-municipality valid window count distribution per split (skew, Gini)
2. Observation quality vs density correlation (temporal coverage, regularity)
3. Effective gradient influence per municipality across an epoch

Usage:
    python scripts/analyze_municipality_representation.py \\
        --config configs/train_epifor_real_local.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats as sp_stats

from data.dataset_factory import split_nodes_by_ratio
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WW_COMPONENTS = ("edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4")
HEAD_NAMES = ("cases", "hospitalizations", "deaths", "wastewater")

SPLIT_COLORS = {
    "train": "#2A6F97",
    "val": "#D9822B",
    "test": "#7A4EA3",
}

HEAD_COLORS = {
    "(any)": "#111111",
    "cases": "#1f77b4",
    "hospitalizations": "#ff7f0e",
    "deaths": "#d62728",
    "wastewater": "#2ca02c",
}

WEEKLY_HEADS = {"hospitalizations", "wastewater"}
HEAD_FREQ = {
    "cases": "daily",
    "hospitalizations": "weekly",
    "deaths": "daily",
    "wastewater": "weekly",
}


@dataclass(frozen=True)
class MunicipalityDiagnostics:
    node_idx: int
    n_windows: dict[str, int]
    obs_density: dict[str, float]
    temporal_coverage: dict[str, int]
    obs_regularity: dict[str, float]
    population: float


def _target_mask(dataset: xr.Dataset, var_name: str) -> np.ndarray | None:
    da = dataset[var_name]
    if "run_id" in da.dims:
        da = da.squeeze("run_id", drop=True)
    da = da.transpose(TEMPORAL_COORD, REGION_COORD)
    values = np.asarray(da.values)

    mask_name = f"{var_name}_mask"
    if mask_name in dataset:
        mask_da = dataset[mask_name]
        if "run_id" in mask_da.dims:
            mask_da = mask_da.squeeze("run_id", drop=True)
        mask_da = mask_da.transpose(TEMPORAL_COORD, REGION_COORD)
        observed = np.asarray(mask_da.values) > 0
    else:
        observed = np.isfinite(values)

    return (observed & np.isfinite(values) & (values >= 0.0)).astype(bool)


def _wastewater_mask(dataset: xr.Dataset) -> np.ndarray | None:
    component_masks = [
        mask
        for var_name in WW_COMPONENTS
        if (mask := _target_mask(dataset, var_name)) is not None
    ]
    if not component_masks:
        return None
    return np.any(np.stack(component_masks, axis=0), axis=0)


def _load_masks_by_head(
    dataset: xr.Dataset,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for head, var_name in {
        "cases": "cases",
        "hospitalizations": "hospitalizations",
        "deaths": "deaths",
    }.items():
        mask = _target_mask(dataset, var_name)
        if mask is not None:
            masks[head] = mask
    ww_mask = _wastewater_mask(dataset)
    if ww_mask is not None:
        masks["wastewater"] = ww_mask
    return masks


def _valid_targets(dataset: xr.Dataset, config: EpiForecasterConfig) -> list[int]:
    num_nodes = int(dataset.sizes[REGION_COORD])
    if not config.data.use_valid_targets or "valid_targets" not in dataset:
        return list(range(num_nodes))
    vt = dataset["valid_targets"]
    if "run_id" in vt.dims:
        vt = vt.squeeze("run_id", drop=True)
    return np.flatnonzero(np.asarray(vt.values).astype(bool)).astype(int).tolist()


def _count_valid_windows_per_node(
    *,
    mask_by_head: dict[str, np.ndarray],
    target_nodes: list[int],
    context_length: int,
    forecast_horizon: int,
    window_stride: int,
    max_lag: int,
    missing_permit_map: dict[str, dict[str, int]],
    total_time_steps: int,
) -> dict[int, int]:
    segment_length = context_length + forecast_horizon
    if total_time_steps < segment_length:
        return {n: 0 for n in target_nodes}

    starts = np.asarray(
        list(range(max_lag, total_time_steps - segment_length + 1, window_stride)),
        dtype=np.int64,
    )
    if starts.size == 0:
        return {n: 0 for n in target_nodes}

    num_nodes = next(iter(mask_by_head.values())).shape[1]
    valid_mask = np.zeros((len(starts), num_nodes), dtype=bool)

    for head_name, head_mask in mask_by_head.items():
        observed = head_mask.astype(np.int32)
        cumsum = np.concatenate(
            [np.zeros((1, num_nodes), dtype=np.int32), np.cumsum(observed, axis=0)],
            axis=0,
        )
        history_counts = cumsum[context_length:] - cumsum[:-context_length]
        target_counts = (
            cumsum[context_length + forecast_horizon :]
            - cumsum[context_length:-forecast_horizon]
        )
        history_counts = history_counts[starts]
        target_counts = target_counts[starts]

        input_permit = int(missing_permit_map["input"].get(head_name, 0))
        horizon_permit = int(missing_permit_map["horizon"].get(head_name, 0))
        history_threshold = max(0, context_length - input_permit)
        target_threshold = max(0, forecast_horizon - horizon_permit)
        valid_mask |= (history_counts >= history_threshold) & (
            target_counts >= target_threshold
        )

    return {n: int(valid_mask[:, n].sum()) for n in target_nodes}


def _count_valid_windows_per_head(
    *,
    mask_by_head: dict[str, np.ndarray],
    target_nodes: list[int],
    context_length: int,
    forecast_horizon: int,
    window_stride: int,
    max_lag: int,
    missing_permit_map: dict[str, dict[str, int]],
    total_time_steps: int,
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for head_name, head_mask in mask_by_head.items():
        result[head_name] = _count_valid_windows_per_node(
            mask_by_head={head_name: head_mask},
            target_nodes=target_nodes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            window_stride=window_stride,
            max_lag=max_lag,
            missing_permit_map=missing_permit_map,
            total_time_steps=total_time_steps,
        )
    return result


def _observation_density(mask: np.ndarray, node_idx: int) -> float:
    col = mask[:, node_idx]
    return float(col.sum()) / len(col) if len(col) > 0 else 0.0


def _temporal_coverage(mask: np.ndarray, node_idx: int) -> int:
    col = mask[:, node_idx]
    dates = np.where(col)[0]
    if len(dates) == 0:
        return 0
    month_indices = dates // 30
    return int(len(np.unique(month_indices)))


def _observation_regularity(mask: np.ndarray, node_idx: int) -> float:
    col = mask[:, node_idx]
    obs_days = np.where(col)[0]
    if len(obs_days) < 2:
        return float("inf")
    gaps = np.diff(obs_days).astype(float)
    mean_gap = gaps.mean()
    if mean_gap == 0:
        return 0.0
    return float(gaps.std() / mean_gap)


def _gini(values: np.ndarray) -> float:
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_v).sum() / (n * sorted_v.sum())) - (n + 1) / n)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "max_min_ratio": 0.0,
            "gini": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "cv": float(values.std() / values.mean()) if values.mean() > 0 else 0.0,
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "max_min_ratio": float(values.max() / values.min()) if values.min() > 0 else float("inf"),
        "gini": _gini(values),
    }


def compute_diagnostics(config: EpiForecasterConfig) -> tuple[
    dict[str, dict[int, int]],
    dict[str, dict[str, dict[int, int]]],
    dict[int, MunicipalityDiagnostics],
    list[int],
    dict[str, list[int]],
]:
    dataset = xr.open_zarr(config.data.dataset_path)
    if "run_id" in dataset.dims:
        dataset = dataset.sel(run_id=config.data.run_id)

    valid_targets = _valid_targets(dataset, config)
    logger.info("Total valid target nodes: %d", len(valid_targets))

    if config.training.split_strategy == "node":
        train_nodes, val_nodes, test_nodes = split_nodes_by_ratio(config)
    else:
        train_nodes = val_nodes = test_nodes = valid_targets

    split_nodes_map = {
        "train": train_nodes,
        "val": val_nodes,
        "test": test_nodes,
    }

    mask_by_head = _load_masks_by_head(dataset)
    logger.info("Loaded masks for heads: %s", list(mask_by_head.keys()))

    total_time_steps = int(dataset.sizes[TEMPORAL_COORD])
    L = int(config.model.input_window_length)
    H = int(config.model.forecast_horizon)
    stride = int(config.data.window_stride)
    max_lag = max(config.data.mobility_lags, default=0)
    missing_permit_map = config.data.resolve_missing_permit_map()

    population = np.asarray(dataset["population"].values).astype(float)

    window_counts_by_split: dict[str, dict[int, int]] = {}
    head_window_counts_by_split: dict[str, dict[str, dict[int, int]]] = {}

    for split_name, nodes in split_nodes_map.items():
        logger.info("Computing window counts for %s (%d nodes)...", split_name, len(nodes))

        window_counts_by_split[split_name] = _count_valid_windows_per_node(
            mask_by_head=mask_by_head,
            target_nodes=nodes,
            context_length=L,
            forecast_horizon=H,
            window_stride=stride,
            max_lag=max_lag,
            missing_permit_map=missing_permit_map,
            total_time_steps=total_time_steps,
        )

        head_window_counts_by_split[split_name] = _count_valid_windows_per_head(
            mask_by_head=mask_by_head,
            target_nodes=nodes,
            context_length=L,
            forecast_horizon=H,
            window_stride=stride,
            max_lag=max_lag,
            missing_permit_map=missing_permit_map,
            total_time_steps=total_time_steps,
        )

    all_diagnostic_nodes = sorted(
        set(train_nodes) | set(val_nodes) | set(test_nodes)
    )
    diagnostics: dict[int, MunicipalityDiagnostics] = {}

    for node_idx in all_diagnostic_nodes:
        n_windows: dict[str, int] = {}
        for split_name, counts in window_counts_by_split.items():
            n_windows[split_name] = counts.get(node_idx, 0)

        obs_density: dict[str, float] = {}
        temp_cov: dict[str, int] = {}
        obs_reg: dict[str, float] = {}

        for head_name, head_mask in mask_by_head.items():
            obs_density[head_name] = _observation_density(head_mask, node_idx)
            temp_cov[head_name] = _temporal_coverage(head_mask, node_idx)
            obs_reg[head_name] = _observation_regularity(head_mask, node_idx)

        diagnostics[node_idx] = MunicipalityDiagnostics(
            node_idx=node_idx,
            n_windows=n_windows,
            obs_density=obs_density,
            temporal_coverage=temp_cov,
            obs_regularity=obs_reg,
            population=population[node_idx] if node_idx < len(population) else 0.0,
        )

    return (
        window_counts_by_split,
        head_window_counts_by_split,
        diagnostics,
        all_diagnostic_nodes,
        split_nodes_map,
    )


def print_summary_tables(
    window_counts_by_split: dict[str, dict[int, int]],
    head_window_counts_by_split: dict[str, dict[str, dict[int, int]]],
    diagnostics: dict[int, MunicipalityDiagnostics],
    split_nodes_map: dict[str, list[int]],
) -> None:
    print("\n" + "=" * 80)
    print("PART 1: Representation Skew — Valid Windows per Municipality")
    print("=" * 80)

    for split_name, nodes in split_nodes_map.items():
        counts = np.array([window_counts_by_split[split_name].get(n, 0) for n in nodes])
        s = _summary_stats(counts)
        total_windows = int(counts.sum())
        n_zero = int((counts == 0).sum())

        print(f"\n--- {split_name.upper()} ({len(nodes)} municipalities) ---")
        print(f"  Total valid (node, window) pairs: {total_windows}")
        print(f"  Municipalities with 0 windows:    {n_zero}")
        print(f"  Mean:  {s['mean']:.1f}   Std: {s['std']:.1f}   CV: {s['cv']:.2f}")
        print(
            f"  Min:   {s['min']:.0f}   P25: {s['p25']:.1f}   "
            f"Median: {s['median']:.1f}   P75: {s['p75']:.1f}   Max: {s['max']:.0f}"
        )
        print(f"  Max/Min ratio: {s['max_min_ratio']:.1f}   Gini: {s['gini']:.3f}")

    print("\n\nPer-head breakdown (TRAIN split):")
    train_nodes = split_nodes_map["train"]
    for head_name in HEAD_NAMES:
        if head_name not in head_window_counts_by_split.get("train", {}):
            continue
        counts = np.array(
            [head_window_counts_by_split["train"][head_name].get(n, 0) for n in train_nodes]
        )
        s = _summary_stats(counts)
        print(
            f"  {head_name:16s}  mean={s['mean']:6.1f}  std={s['std']:6.1f}  "
            f"CV={s['cv']:.2f}  Gini={s['gini']:.3f}  total={int(counts.sum())}"
        )


def print_quality_correlations(
    diagnostics: dict[int, MunicipalityDiagnostics],
) -> None:
    print("\n" + "=" * 80)
    print("PART 2: Observation Quality vs Valid Window Count Correlation")
    print("=" * 80)

    node_indices = sorted(diagnostics.keys())
    train_windows = np.array([diagnostics[n].n_windows.get("train", 0) for n in node_indices])
    populations = np.array([diagnostics[n].population for n in node_indices])

    quality_metrics = ["obs_density", "temporal_coverage", "obs_regularity"]

    print(f"\n{'Head':<16s} {'Quality metric':<22s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman ρ':>11s} {'p-value':>10s}")
    print("-" * 82)

    for head_name in HEAD_NAMES:
        for metric_name in quality_metrics:
            metric_values = []
            for n in node_indices:
                d = diagnostics[n]
                if metric_name == "obs_density":
                    val = d.obs_density.get(head_name, 0.0)
                elif metric_name == "temporal_coverage":
                    val = float(d.temporal_coverage.get(head_name, 0))
                elif metric_name == "obs_regularity":
                    val = d.obs_regularity.get(head_name, float("inf"))
                else:
                    val = 0.0
                metric_values.append(val)

            metric_arr = np.array(metric_values)
            finite_mask = np.isfinite(metric_arr) & np.isfinite(train_windows)
            if finite_mask.sum() < 5:
                continue

            pr, p_p = sp_stats.pearsonr(
                train_windows[finite_mask], metric_arr[finite_mask]
            )
            sr, p_s = sp_stats.spearmanr(
                train_windows[finite_mask], metric_arr[finite_mask]
            )
            print(
                f"{head_name:<16s} {metric_name:<22s} {pr:>10.3f} {p_p:>10.4f} "
                f"{sr:>11.3f} {p_s:>10.4f}"
            )

    if (populations > 0).sum() >= 5:
        valid_pop = populations > 0
        pr, p_p = sp_stats.pearsonr(
            train_windows[valid_pop], np.log10(populations[valid_pop])
        )
        sr, p_s = sp_stats.spearmanr(
            train_windows[valid_pop], np.log10(populations[valid_pop])
        )
        print(
            f"{'(overall)':<16s} {'log10(population)':<22s} {pr:>10.3f} {p_p:>10.4f} "
            f"{sr:>11.3f} {p_s:>10.4f}"
        )


def print_gradient_influence(
    window_counts_by_split: dict[str, dict[int, int]],
    diagnostics: dict[int, MunicipalityDiagnostics],
    split_nodes_map: dict[str, list[int]],
) -> None:
    print("\n" + "=" * 80)
    print("PART 3: Effective Gradient Influence per Municipality (TRAIN)")
    print("=" * 80)

    train_nodes = split_nodes_map["train"]
    counts = np.array([window_counts_by_split["train"].get(n, 0) for n in train_nodes])
    total = counts.sum()

    if total == 0:
        print("  No valid windows in train split.")
        return

    w_eff = counts / total
    n_munis = len(train_nodes)
    w_uniform = 1.0 / n_munis
    over_rep = w_eff / w_uniform

    print(f"\n  N municipalities:  {n_munis}")
    print(f"  Total windows:     {int(total)}")
    print(f"  Uniform weight:    {w_uniform:.5f}")
    print(f"  Effective weights: mean={w_eff.mean():.5f}  std={w_eff.std():.5f}")
    print("  Over-representation ratio (w_eff/w_uniform):")
    print(
        f"    mean={over_rep.mean():.2f}  std={over_rep.std():.2f}  "
        f"min={over_rep.min():.2f}  median={np.median(over_rep):.2f}  max={over_rep.max():.2f}"
    )

    populations = np.array([diagnostics[n].population for n in train_nodes])
    if (populations > 0).sum() >= 5:
        valid = populations > 0
        pr, _ = sp_stats.pearsonr(w_eff[valid], np.log10(populations[valid]))
        sr, _ = sp_stats.spearmanr(w_eff[valid], np.log10(populations[valid]))
        print("\n  Correlation with log10(population):")
        print(f"    Pearson r = {pr:.3f}   Spearman ρ = {sr:.3f}")


def write_csv(
    diagnostics: dict[int, MunicipalityDiagnostics],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    node_indices = sorted(diagnostics.keys())

    fieldnames = [
        "node_idx",
        "population",
        "n_windows_train",
        "n_windows_val",
        "n_windows_test",
    ]
    for head_name in HEAD_NAMES:
        fieldnames.extend([
            f"density_{head_name}",
            f"temporal_coverage_{head_name}",
            f"regularity_{head_name}",
        ])

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for n in node_indices:
            d = diagnostics[n]
            row: dict[str, str | float | int] = {
                "node_idx": d.node_idx,
                "population": d.population,
                "n_windows_train": d.n_windows.get("train", 0),
                "n_windows_val": d.n_windows.get("val", 0),
                "n_windows_test": d.n_windows.get("test", 0),
            }
            for head_name in HEAD_NAMES:
                row[f"density_{head_name}"] = f"{d.obs_density.get(head_name, 0.0):.4f}"
                row[f"temporal_coverage_{head_name}"] = d.temporal_coverage.get(head_name, 0)
                reg = d.obs_regularity.get(head_name, float("inf"))
                row[f"regularity_{head_name}"] = f"{reg:.4f}" if np.isfinite(reg) else "inf"
            writer.writerow(row)

    logger.info("Wrote per-municipality CSV: %s", output_csv)


def plot_window_histogram(
    window_counts_by_split: dict[str, dict[int, int]],
    split_nodes_map: dict[str, list[int]],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, split_name in zip(axes, ("train", "val", "test")):
        nodes = split_nodes_map[split_name]
        counts = np.array([window_counts_by_split[split_name].get(n, 0) for n in nodes])
        ax.hist(counts, bins=40, color=SPLIT_COLORS[split_name], edgecolor="white", alpha=0.85)
        ax.set_title(f"{split_name} ({len(nodes)} munis)")
        ax.set_xlabel("Valid windows")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Count")
    fig.suptitle("Distribution of valid windows per municipality", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote histogram: %s", output_png)


def plot_lorenz_curve(
    window_counts_by_split: dict[str, dict[int, int]],
    split_nodes_map: dict[str, list[int]],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    for split_name in ("train", "val", "test"):
        nodes = split_nodes_map[split_name]
        counts = np.array(
            [window_counts_by_split[split_name].get(n, 0) for n in nodes]
        )
        sorted_counts = np.sort(counts)
        total = sorted_counts.sum()
        if total == 0:
            continue
        cumulative_share = np.cumsum(sorted_counts) / total
        n = len(sorted_counts)
        pop_share = np.arange(1, n + 1) / n
        gini_val = _gini(counts)
        ax.plot(
            pop_share,
            cumulative_share,
            color=SPLIT_COLORS[split_name],
            label=f"{split_name} (G={gini_val:.3f})",
            linewidth=1.5,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Equal")
    ax.set_xlabel("Cumulative share of municipalities")
    ax.set_ylabel("Cumulative share of gradient updates")
    ax.set_title("Lorenz curve: gradient influence inequality")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote Lorenz curve: %s", output_png)


def plot_density_vs_windows(
    diagnostics: dict[int, MunicipalityDiagnostics],
    split_nodes_map: dict[str, list[int]],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    train_nodes = set(split_nodes_map["train"])
    train_diags = [diagnostics[n] for n in sorted(diagnostics.keys()) if n in train_nodes]

    if not train_diags:
        return

    n_heads = len(HEAD_NAMES)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4), squeeze=False)

    for ax, head_name in zip(axes[0], HEAD_NAMES):
        windows = np.array([d.n_windows.get("train", 0) for d in train_diags])
        density = np.array([d.obs_density.get(head_name, 0.0) for d in train_diags])

        mask = np.isfinite(windows) & np.isfinite(density)
        if mask.sum() < 3:
            continue

        ax.scatter(density[mask], windows[mask], alpha=0.4, s=18, color=HEAD_COLORS[head_name])
        pr, _ = sp_stats.pearsonr(density[mask], windows[mask])
        sr, _ = sp_stats.spearmanr(density[mask], windows[mask])
        ax.set_title(f"{head_name}\nr={pr:.2f}, ρ={sr:.2f}", fontsize=10)
        ax.set_xlabel("Observation density")
        ax.set_ylabel("Valid train windows")

    fig.suptitle("Observation density vs valid training windows", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote density scatter: %s", output_png)


def plot_population_vs_weight(
    diagnostics: dict[int, MunicipalityDiagnostics],
    window_counts_by_split: dict[str, dict[int, int]],
    split_nodes_map: dict[str, list[int]],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    train_nodes = split_nodes_map["train"]
    counts = np.array([window_counts_by_split["train"].get(n, 0) for n in train_nodes])
    total = counts.sum()
    if total == 0:
        return

    w_eff = counts / total
    n_munis = len(train_nodes)
    w_uniform = 1.0 / n_munis
    over_rep = w_eff / w_uniform

    populations = np.array([diagnostics[n].population for n in train_nodes])
    valid = populations > 0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if valid.sum() > 0:
        ax.scatter(
            np.log10(populations[valid]),
            over_rep[valid],
            alpha=0.5,
            s=22,
            color="#2A6F97",
        )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Equal rep.")
        ax.set_xlabel("log10(population)")
        ax.set_ylabel("Over-representation ratio (w_eff / w_uniform)")
        ax.set_title("Population vs gradient influence")

        pr, _ = sp_stats.pearsonr(
            np.log10(populations[valid]), over_rep[valid]
        )
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {pr:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.legend(frameon=False)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote population vs weight scatter: %s", output_png)


def _scale_permit_map(
    base_map: dict[str, dict[str, int]],
    scale: float,
    L: int,
    H: int,
) -> dict[str, dict[str, int]]:
    return {
        window_type: {
            head: min(int(permit * scale), window_len)
            for head, permit in permits.items()
        }
        for window_type, permits, window_len in [
            ("input", base_map["input"], L),
            ("horizon", base_map["horizon"], H),
        ]
    }


@dataclass(frozen=True)
class SweepPoint:
    scale: float
    head: str
    freq: str
    total_windows: int
    n_municipalities: int
    n_excluded: int
    gini: float
    mean_windows: float
    std_windows: float
    min_obs_input: int
    min_obs_horizon: int
    coverage_input: float
    coverage_horizon: float


def _perfect_obs(window_len: int, freq: str) -> int:
    if freq == "weekly":
        return int(np.ceil(window_len / 7))
    return window_len


def compute_sweep(
    config: EpiForecasterConfig,
    scales: list[float],
) -> tuple[list[SweepPoint], dict[str, list[int]], dict[str, np.ndarray]]:
    dataset = xr.open_zarr(config.data.dataset_path)
    if "run_id" in dataset.dims:
        dataset = dataset.sel(run_id=config.data.run_id)

    valid_targets = _valid_targets(dataset, config)

    if config.training.split_strategy == "node":
        train_nodes, _, _ = split_nodes_by_ratio(config)
    else:
        train_nodes = valid_targets

    mask_by_head = _load_masks_by_head(dataset)
    total_time_steps = int(dataset.sizes[TEMPORAL_COORD])
    L = int(config.model.input_window_length)
    H = int(config.model.forecast_horizon)
    stride = int(config.data.window_stride)
    max_lag = max(config.data.mobility_lags, default=0)
    base_permit_map = config.data.resolve_missing_permit_map()

    sweep_points: list[SweepPoint] = []

    for scale in scales:
        scaled_map = _scale_permit_map(base_permit_map, scale, L, H)

        per_head = _count_valid_windows_per_head(
            mask_by_head=mask_by_head,
            target_nodes=train_nodes,
            context_length=L,
            forecast_horizon=H,
            window_stride=stride,
            max_lag=max_lag,
            missing_permit_map=scaled_map,
            total_time_steps=total_time_steps,
        )

        combined_counts = _count_valid_windows_per_node(
            mask_by_head=mask_by_head,
            target_nodes=train_nodes,
            context_length=L,
            forecast_horizon=H,
            window_stride=stride,
            max_lag=max_lag,
            missing_permit_map=scaled_map,
            total_time_steps=total_time_steps,
        )
        combined_arr = np.array([combined_counts.get(n, 0) for n in train_nodes])
        sweep_points.append(
            SweepPoint(
                scale=scale,
                head="(any)",
                freq="mixed",
                total_windows=int(combined_arr.sum()),
                n_municipalities=len(train_nodes),
                n_excluded=int((combined_arr == 0).sum()),
                gini=_gini(combined_arr),
                mean_windows=float(combined_arr.mean()),
                std_windows=float(combined_arr.std()),
                min_obs_input=0,
                min_obs_horizon=0,
                coverage_input=0.0,
                coverage_horizon=0.0,
            )
        )

        for head_name in HEAD_NAMES:
            if head_name not in per_head:
                continue
            counts = np.array([per_head[head_name].get(n, 0) for n in train_nodes])
            total_windows = int(counts.sum())
            head_min_input = L - scaled_map["input"].get(head_name, 0)
            head_min_horizon = H - scaled_map["horizon"].get(head_name, 0)
            freq = HEAD_FREQ.get(head_name, "daily")
            perf_in = _perfect_obs(L, freq)
            perf_hz = _perfect_obs(H, freq)
            sweep_points.append(
                SweepPoint(
                    scale=scale,
                    head=head_name,
                    freq=freq,
                    total_windows=total_windows,
                    n_municipalities=len(train_nodes),
                    n_excluded=int((counts == 0).sum()),
                    gini=_gini(counts) if total_windows > 0 else float("nan"),
                    mean_windows=float(counts.mean()),
                    std_windows=float(counts.std()),
                    min_obs_input=head_min_input,
                    min_obs_horizon=head_min_horizon,
                    coverage_input=head_min_input / perf_in if perf_in > 0 else 0.0,
                    coverage_horizon=head_min_horizon / perf_hz if perf_hz > 0 else 0.0,
                )
            )

    return sweep_points, {"train": train_nodes}, mask_by_head


def print_sweep_table(sweep_points: list[SweepPoint]) -> None:
    print("\n" + "=" * 118)
    print("SWEEP: Missingness Permit Scale vs Representation Equity (TRAIN split)")
    print("=" * 118)
    print(
        f"\n{'scale':>6s}  {'head':<16s} {'freq':>7s}  {'Gini':>6s}  {'total':>8s}  "
        f"{'excl':>5s}  {'mean':>7s}  {'cov_in':>7s}  {'cov_hz':>7s}  "
        f"{'min_in':>6s}  {'min_hz':>6s}"
    )
    print("-" * 118)

    for pt in sweep_points:
        if pt.head == "(any)":
            print(
                f"{pt.scale:>6.2f}  {pt.head:<16s} {pt.freq:>7s}  {pt.gini:>6.3f}  "
                f"{pt.total_windows:>8d}  {pt.n_excluded:>5d}  {pt.mean_windows:>7.1f}  "
                f"{'---':>7s}  {'---':>7s}  "
                f"{'---':>6s}  {'---':>6s}"
            )
            continue
        print(
            f"{pt.scale:>6.2f}  {pt.head:<16s} {pt.freq:>7s}  {pt.gini:>6.3f}  "
            f"{pt.total_windows:>8d}  {pt.n_excluded:>5d}  {pt.mean_windows:>7.1f}  "
            f"{pt.coverage_input:>6.0%}  {pt.coverage_horizon:>6.0%}  "
            f"{pt.min_obs_input:>6d}  {pt.min_obs_horizon:>6d}"
        )

    print(
        "\n  cov_in/cov_hz = min required observations / perfect-coverage count "
        "(L/7 for weekly, L for daily)"
    )


def write_sweep_csv(sweep_points: list[SweepPoint], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    any_by_scale = {pt.scale: pt for pt in sweep_points if pt.head == "(any)"}
    fieldnames = [
        "scale",
        "head",
        "freq",
        "total_windows",
        "n_municipalities",
        "n_excluded",
        "gini",
        "mean_windows",
        "std_windows",
        "min_obs_input",
        "min_obs_horizon",
        "coverage_input",
        "coverage_horizon",
        "any_total_windows",
        "any_n_excluded",
        "any_gini",
        "any_mean_windows",
        "any_std_windows",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pt in sweep_points:
            any_pt = any_by_scale.get(pt.scale)
            writer.writerow({
                "scale": pt.scale,
                "head": pt.head,
                "freq": pt.freq,
                "total_windows": pt.total_windows,
                "n_municipalities": pt.n_municipalities,
                "n_excluded": pt.n_excluded,
                "gini": pt.gini,
                "mean_windows": pt.mean_windows,
                "std_windows": pt.std_windows,
                "min_obs_input": pt.min_obs_input,
                "min_obs_horizon": pt.min_obs_horizon,
                "coverage_input": f"{pt.coverage_input:.4f}",
                "coverage_horizon": f"{pt.coverage_horizon:.4f}",
                "any_total_windows": any_pt.total_windows if any_pt else "",
                "any_n_excluded": any_pt.n_excluded if any_pt else "",
                "any_gini": any_pt.gini if any_pt else "",
                "any_mean_windows": any_pt.mean_windows if any_pt else "",
                "any_std_windows": any_pt.std_windows if any_pt else "",
            })
    logger.info("Wrote sweep CSV: %s", output_csv)


def plot_gini_vs_windows(
    sweep_points: list[SweepPoint],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for head_name in ("(any)", *HEAD_NAMES):
        pts = [p for p in sweep_points if p.head == head_name]
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p.scale)
        totals = [
            p.total_windows for p in pts_sorted if p.total_windows > 0 and np.isfinite(p.gini)
        ]
        ginis = [
            p.gini for p in pts_sorted if p.total_windows > 0 and np.isfinite(p.gini)
        ]
        if not totals:
            continue
        color = HEAD_COLORS[head_name]
        linestyle = "--" if head_name == "(any)" else "-"
        linewidth = 2.2 if head_name == "(any)" else 1.5
        marker = "D" if head_name == "(any)" else "o"

        ax.plot(
            totals,
            ginis,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=linewidth,
            linestyle=linestyle,
            label="any head" if head_name == "(any)" else head_name,
        )

        annotated_scales = {0.0, 0.98, 1.0} if head_name == "(any)" else set()
        for pt in pts_sorted:
            if (
                pt.scale in annotated_scales
                and pt.total_windows > 0
                and np.isfinite(pt.gini)
            ):
                scale_label = f"s={pt.scale:.2f}" if pt.scale == 0.98 else f"s={pt.scale:.1f}"
                ax.annotate(
                    scale_label,
                    (pt.total_windows, pt.gini),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                    color=color,
                )

    ax.set_xlabel("Total valid (node, window) pairs")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Representation equity vs data support\n(missingness permit sweep)")
    ax.legend(frameon=False, title="Head")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote Gini tradeoff curve: %s", output_png)


def plot_exclusion_vs_scale(
    sweep_points: list[SweepPoint],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    any_pts = sorted(
        [p for p in sweep_points if p.head == "(any)"],
        key=lambda p: p.scale,
    )
    if any_pts:
        ax.plot(
            [p.scale for p in any_pts],
            [p.n_excluded for p in any_pts],
            color="black",
            marker="D",
            markersize=4,
            linewidth=2.0,
            linestyle="--",
            label="(any head)",
            zorder=5,
        )

    for head_name in HEAD_NAMES:
        pts = [p for p in sweep_points if p.head == head_name]
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p.scale)
        scales = [p.scale for p in pts_sorted]
        excluded = [p.n_excluded for p in pts_sorted]

        ax.plot(
            scales,
            excluded,
            color=HEAD_COLORS[head_name],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=head_name,
        )

    ax.set_xlabel("Missingness permit scale factor")
    ax.set_ylabel("Municipalities with 0 windows (excluded)")
    ax.set_title("Municipality exclusion vs permit strictness")
    ax.legend(frameon=False, title="Head")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote exclusion plot: %s", output_png)


SWEEP_SCALES = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.92, 0.94, 0.96, 0.98, 1.0,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify municipality representation skew and quality tradeoffs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_epifor_real_local.yaml"),
        help="Training config to inspect.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/reports/municipality_representation.csv"),
        help="Per-municipality CSV output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tex/EpiForecaster/plots"),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run missingness permit sweep and produce tradeoff plots.",
    )
    args = parser.parse_args()

    config = EpiForecasterConfig.load(args.config)
    logger.info("Loaded config: %s", args.config)

    if args.sweep:
        sweep_points, _, _ = compute_sweep(config, SWEEP_SCALES)
        print_sweep_table(sweep_points)
        output_dir = args.output_dir
        plot_gini_vs_windows(
            sweep_points,
            output_dir / "missingness_sweep_gini_vs_windows.png",
        )
        plot_exclusion_vs_scale(
            sweep_points,
            output_dir / "missingness_sweep_exclusion.png",
        )
        write_sweep_csv(
            sweep_points,
            Path("outputs/reports/missingness_sweep.csv"),
        )
        logger.info("Sweep done.")
        return

    (
        window_counts_by_split,
        head_window_counts_by_split,
        diagnostics,
        all_nodes,
        split_nodes_map,
    ) = compute_diagnostics(config)

    print_summary_tables(
        window_counts_by_split,
        head_window_counts_by_split,
        diagnostics,
        split_nodes_map,
    )
    print_quality_correlations(diagnostics)
    print_gradient_influence(window_counts_by_split, diagnostics, split_nodes_map)

    write_csv(diagnostics, args.output_csv)

    output_dir = args.output_dir
    plot_window_histogram(
        window_counts_by_split,
        split_nodes_map,
        output_dir / "municipality_window_histogram.png",
    )
    plot_lorenz_curve(
        window_counts_by_split,
        split_nodes_map,
        output_dir / "municipality_lorenz_curve.png",
    )
    plot_density_vs_windows(
        diagnostics,
        split_nodes_map,
        output_dir / "municipality_density_vs_windows.png",
    )
    plot_population_vs_weight(
        diagnostics,
        window_counts_by_split,
        split_nodes_map,
        output_dir / "municipality_population_vs_weight.png",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
