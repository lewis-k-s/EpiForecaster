from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def collect_case_window_samples(
    loader: DataLoader,
    n: int = 5,
    cases_feature_idx: int = 0,
    biomarker_feature_idx: int | None = 0,
    include_biomarkers: bool = True,
    include_mobility: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Collect case windows (history + target horizon) from the DataLoader's dataset.

    We intentionally avoid collation/batching here: for plotting we only need
    single samples, and collation would only stack tensors without changing
    values. Using the dataset ensures we capture exactly what the DataLoader
    would yield (pre-stack) for each sample.
    """
    samples: list[dict[str, Any]] = []

    dataset = loader.dataset
    k = min(int(n), len(dataset))
    if shuffle:
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), k=k)
    else:
        indices = list(range(k))

    for idx in indices:
        item = dataset[idx]
        case_hist = item["case_node"]
        targets = item["target"]

        if not isinstance(case_hist, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            raise TypeError(
                "collect_case_window_samples expects dataset items with torch tensors."
            )

        if case_hist.ndim != 2:
            raise ValueError("Expected `case_node` with shape (L, C).")

        history = case_hist[:, cases_feature_idx]
        if targets.ndim == 1:
            future = targets
        elif targets.ndim == 2:
            future = targets[:, cases_feature_idx]
        else:
            raise ValueError("Expected `target` with shape (H,) or (H, C).")

        window = torch.cat([history, future], dim=0).detach().cpu().numpy()
        sample: dict[str, Any] = {
            "node_id": int(item["target_node"]),
            "node_label": str(item.get("node_label", "")),
            "series": np.asarray(window, dtype=np.float32),
        }

        if include_biomarkers:
            biomarkers = item.get("bio_node")
            if isinstance(biomarkers, torch.Tensor):
                if biomarkers.ndim != 2:
                    raise ValueError("Expected `bio_node` with shape (L, B).")
                if biomarker_feature_idx is None:
                    biomarker_series = biomarkers.mean(dim=-1)
                else:
                    biomarker_series = biomarkers[:, biomarker_feature_idx]
                sample["biomarkers"] = (
                    biomarker_series.detach().cpu().numpy().astype(np.float32)
                )

        if include_mobility:
            mob_graphs = item.get("mob")
            if isinstance(mob_graphs, list):
                incoming = []
                for g in mob_graphs:
                    edge_weight = getattr(g, "edge_weight", None)
                    if edge_weight is None:
                        edge_weight = getattr(g, "edge_attr", None)
                    if edge_weight is None:
                        incoming.append(0.0)
                    else:
                        incoming.append(float(edge_weight.detach().sum().cpu().item()))
                sample["mobility_incoming"] = np.asarray(incoming, dtype=np.float32)

        samples.append(sample)

    return samples


def make_cases_window_figure(samples: list[dict[str, Any]], history_length: int):
    """
    Build a seaborn figure to visualize case windows (history + horizon).

    If samples include biomarker or mobility series, those are plotted alongside
    cases and each series is normalized to the same 0-1 scale for comparability.

    Returns a matplotlib Figure suitable for saving or TensorBoard logging.
    """
    if not samples:
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    n = len(samples)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(12, max(2.5 * n, 3.0)), sharex=True
    )
    if n == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples, strict=False):
        series = np.asarray(sample["series"], dtype=np.float32).reshape(-1)
        total_len = series.shape[0]

        biomarker_series = sample.get("biomarkers")
        mobility_series = sample.get("mobility_incoming")

        plot_series: dict[str, np.ndarray] = {"Cases": series}
        if biomarker_series is not None:
            biomarker_series = np.asarray(biomarker_series, dtype=np.float32).reshape(
                -1
            )
            biomarker_aligned = np.full(total_len, np.nan, dtype=np.float32)
            biomarker_aligned[: min(history_length, biomarker_series.shape[0])] = (
                biomarker_series[:history_length]
            )
            plot_series["Biomarkers"] = biomarker_aligned
        if mobility_series is not None:
            mobility_series = np.asarray(mobility_series, dtype=np.float32).reshape(-1)
            mobility_aligned = np.full(total_len, np.nan, dtype=np.float32)
            mobility_aligned[: min(history_length, mobility_series.shape[0])] = (
                mobility_series[:history_length]
            )
            plot_series["Incoming mobility"] = mobility_aligned

        plot_rows = []
        for label, values_series in plot_series.items():
            series_values = values_series
            plot_rows.append(
                pd.DataFrame(
                    {
                        "t": np.arange(total_len),
                        "value": series_values,
                        "series": label,
                    }
                )
            )

        df = pd.concat(plot_rows, axis=0, ignore_index=True)
        sns.lineplot(data=df, x="t", y="value", hue="series", ax=ax)
        ax.axvline(history_length - 0.5, color="black", linestyle="--", alpha=0.5)

        node_label = sample.get("node_label", "")
        node_id = sample.get("node_id", None)
        title = f"{node_label}".strip()
        if node_id is not None:
            title = f"{title} (id={node_id})" if title else f"id={node_id}"
        ax.set_title(title)
        ax.set_ylabel("Cases")
        legend = ax.get_legend()
        if legend is not None and ax is not axes[0]:
            legend.remove()

    axes[-1].set_xlabel("Time index (history → horizon)")
    fig.tight_layout()
    return fig
