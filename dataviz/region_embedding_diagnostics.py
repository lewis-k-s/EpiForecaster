#!/usr/bin/env python
"""Diagnostics for final Region2Vec embedding artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from libpysal import weights
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)

from utils.plotting import add_grid, save_figure

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["area", "perimeter", "population", "density", "lon", "lat"]
DEFAULT_K_VALUES = (1, 5, 10, 20)


@dataclass(frozen=True)
class DiagnosticsConfig:
    embeddings: Path
    region_graph: Path
    output_dir: Path
    geojson: Path | None = None
    cluster_count: int = 14
    max_hops: int = 5
    flow_threshold: float | None = None
    max_scatter_points: int = 20_000
    seed: int = 42


@dataclass(frozen=True)
class EmbeddingArtifact:
    embeddings: np.ndarray
    region_ids: list[str]
    config: dict[str, Any]


@dataclass(frozen=True)
class RegionGraphData:
    features: np.ndarray
    edge_index: np.ndarray
    region_ids: list[str]
    flows: np.ndarray | None
    metadata: dict[str, Any]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x_valid = x[mask]
    y_valid = y[mask]
    if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return None
    result = spearmanr(x_valid, y_valid)
    return _safe_float(result.statistic)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def load_embedding_artifact(path: str | Path) -> EmbeddingArtifact:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Embedding artifact must be a dict, got {type(artifact)!r}")
    embeddings = artifact.get("embeddings")
    region_ids = artifact.get("region_ids")
    if embeddings is None:
        raise ValueError("Embedding artifact missing 'embeddings'")
    if region_ids is None:
        raise ValueError("Embedding artifact missing 'region_ids'")
    embedding_array = torch.as_tensor(embeddings).detach().cpu().float().numpy()
    return EmbeddingArtifact(
        embeddings=embedding_array,
        region_ids=[str(region_id) for region_id in region_ids],
        config=dict(artifact.get("config") or {}),
    )


def load_region_graph(path: str | Path) -> RegionGraphData:
    ds = xr.open_zarr(path, consolidated=False, mask_and_scale=False)
    required = {"features", "edge_index", "region_ids"}
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise ValueError(f"Region graph missing required variables: {missing}")

    metadata_raw = ds.attrs.get("metadata", {})
    if isinstance(metadata_raw, str):
        metadata = json.loads(metadata_raw)
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        metadata = {}

    flows = np.asarray(ds["flows"].values, dtype=np.float64) if "flows" in ds else None
    raw_edge_index = np.asarray(ds["edge_index"].values)
    finite_edges = np.isfinite(raw_edge_index).all(axis=0)
    if not finite_edges.all():
        logger.warning(
            "Dropping %d edge_index columns with non-finite values",
            int((~finite_edges).sum()),
        )
    edge_index = raw_edge_index[:, finite_edges].astype(np.int64)
    return RegionGraphData(
        features=np.asarray(ds["features"].values, dtype=np.float64),
        edge_index=edge_index,
        region_ids=[str(region_id) for region_id in ds["region_ids"].values.tolist()],
        flows=flows,
        metadata=metadata,
    )


def align_graph_to_artifact(
    artifact: EmbeddingArtifact, graph: RegionGraphData
) -> tuple[EmbeddingArtifact, RegionGraphData, dict[str, Any]]:
    artifact_ids = artifact.region_ids
    graph_ids = graph.region_ids
    artifact_set = set(artifact_ids)
    graph_set = set(graph_ids)
    missing_in_graph = sorted(artifact_set - graph_set)
    missing_in_artifact = sorted(graph_set - artifact_set)
    if missing_in_graph:
        raise ValueError(
            "Embedding artifact contains region IDs not present in graph: "
            f"{missing_in_graph[:5]}"
        )

    graph_index = {region_id: idx for idx, region_id in enumerate(graph_ids)}
    order = np.array(
        [graph_index[region_id] for region_id in artifact_ids], dtype=np.int64
    )
    old_to_new = np.full(len(graph_ids), -1, dtype=np.int64)
    old_to_new[order] = np.arange(len(order), dtype=np.int64)

    edge_index = graph.edge_index
    keep_edges = (old_to_new[edge_index[0]] >= 0) & (old_to_new[edge_index[1]] >= 0)
    remapped_edges = old_to_new[edge_index[:, keep_edges]]
    flows = graph.flows[np.ix_(order, order)] if graph.flows is not None else None
    aligned_graph = RegionGraphData(
        features=graph.features[order],
        edge_index=remapped_edges,
        region_ids=artifact_ids,
        flows=flows,
        metadata=graph.metadata,
    )
    integrity = {
        "artifact_regions": len(artifact_ids),
        "graph_regions": len(graph_ids),
        "aligned_regions": len(order),
        "missing_in_artifact": len(missing_in_artifact),
        "missing_in_graph": len(missing_in_graph),
        "dropped_graph_edges": int((~keep_edges).sum()),
    }
    return artifact, aligned_graph, integrity


def adjacency_matrix(edge_index: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    if edge_index.size == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    data = np.ones(edge_index.shape[1], dtype=np.float64)
    adj = sparse.csr_matrix(
        (data, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )
    return adj.maximum(adj.T)


def compute_hop_distances(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    adj = adjacency_matrix(edge_index, num_nodes)
    return shortest_path(
        csgraph=adj,
        method="FW",
        directed=False,
        unweighted=True,
        return_predecessors=False,
    )


def compute_pca(embeddings: np.ndarray) -> tuple[np.ndarray, PCA]:
    n_components = min(10, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(embeddings)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords, pca


def compute_clusters(
    embeddings: np.ndarray, requested_clusters: int, seed: int
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    n_nodes = embeddings.shape[0]
    n_clusters = max(1, min(requested_clusters, n_nodes))
    if n_clusters == 1:
        labels = np.zeros(n_nodes, dtype=np.int64)
    else:
        labels = KMeans(
            n_clusters=n_clusters, n_init=20, random_state=seed
        ).fit_predict(embeddings)

    metrics: dict[str, float | int | None] = {"n_clusters": int(n_clusters)}
    if 1 < np.unique(labels).size < n_nodes:
        metrics["silhouette"] = _safe_float(silhouette_score(embeddings, labels))
        metrics["davies_bouldin"] = _safe_float(
            davies_bouldin_score(embeddings, labels)
        )
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
    return labels, metrics


def cluster_contiguity(labels: np.ndarray, edge_index: np.ndarray) -> dict[str, Any]:
    graph = nx.Graph()
    graph.add_nodes_from(range(labels.size))
    graph.add_edges_from(
        zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False)
    )
    components_per_cluster: dict[str, int] = {}
    for cluster_id in sorted(np.unique(labels).tolist()):
        nodes = np.flatnonzero(labels == cluster_id).tolist()
        if not nodes:
            components_per_cluster[str(cluster_id)] = 0
            continue
        components_per_cluster[str(cluster_id)] = nx.number_connected_components(
            graph.subgraph(nodes)
        )
    return {
        "components_per_cluster": components_per_cluster,
        "mean_components": _safe_float(np.mean(list(components_per_cluster.values()))),
        "max_components": int(max(components_per_cluster.values(), default=0)),
    }


def flow_concentration(labels: np.ndarray, flows: np.ndarray | None) -> float | None:
    if flows is None:
        return None
    flow = np.asarray(flows, dtype=np.float64).copy()
    np.fill_diagonal(flow, 0.0)
    finite = np.isfinite(flow)
    total = flow[finite].sum()
    if total <= 0:
        return None
    same = labels[:, None] == labels[None, :]
    return float(flow[same & finite].sum() / total)


def compute_spatial_autocorrelation(
    embeddings: np.ndarray, edge_index: np.ndarray
) -> pd.DataFrame:
    num_nodes = embeddings.shape[0]
    neighbors: dict[int, list[int]] = {idx: [] for idx in range(num_nodes)}
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False):
        if src == dst:
            continue
        if dst not in neighbors[src]:
            neighbors[src].append(dst)
        if src not in neighbors[dst]:
            neighbors[dst].append(src)
    w = weights.W(neighbors, silence_warnings=True)
    rows: list[dict[str, Any]] = []
    try:
        import esda
    except ImportError:
        for dim in range(embeddings.shape[1]):
            rows.append(
                {
                    "dimension": dim,
                    "moran_i": np.nan,
                    "p_norm": np.nan,
                    "status": "esda_unavailable",
                }
            )
        return pd.DataFrame(rows)

    for dim in range(embeddings.shape[1]):
        values = embeddings[:, dim]
        if np.nanstd(values) <= 1e-12 or np.unique(values).size < 2:
            rows.append(
                {
                    "dimension": dim,
                    "moran_i": np.nan,
                    "p_norm": np.nan,
                    "status": "constant",
                }
            )
            continue
        try:
            moran = esda.Moran(values, w, permutations=0)
            rows.append(
                {
                    "dimension": dim,
                    "moran_i": float(moran.I),
                    "p_norm": _safe_float(getattr(moran, "p_norm", np.nan)),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dimension": dim,
                    "moran_i": np.nan,
                    "p_norm": np.nan,
                    "status": f"failed:{type(exc).__name__}",
                }
            )
    return pd.DataFrame(rows)


def compute_neighbor_overlaps(
    distance_matrix: np.ndarray,
    hop_distances: np.ndarray,
    flows: np.ndarray | None,
    *,
    flow_threshold: float,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1)
    rows: list[dict[str, float | int | None]] = []
    one_hop = hop_distances == 1
    two_hop = (hop_distances > 0) & (hop_distances <= 2)
    for k in k_values:
        effective_k = min(k, max(1, distance_matrix.shape[0] - 1))
        nearest = order[:, :effective_k]
        one_scores = []
        two_scores = []
        flow_scores = []
        for node_idx in range(distance_matrix.shape[0]):
            neigh = nearest[node_idx]
            one_scores.append(float(one_hop[node_idx, neigh].mean()))
            two_scores.append(float(two_hop[node_idx, neigh].mean()))
            if flows is not None:
                flow_candidates = np.flatnonzero(flows[node_idx] > flow_threshold)
                flow_candidates = flow_candidates[flow_candidates != node_idx]
                denom = min(effective_k, flow_candidates.size)
                if denom > 0:
                    flow_scores.append(
                        len(set(neigh.tolist()) & set(flow_candidates.tolist())) / denom
                    )
        rows.append(
            {
                "k": effective_k,
                "one_hop_overlap": float(np.mean(one_scores)) if one_scores else None,
                "two_hop_overlap": float(np.mean(two_scores)) if two_scores else None,
                "flow_topk_recall": float(np.mean(flow_scores))
                if flow_scores
                else None,
            }
        )
    return pd.DataFrame(rows)


def compute_flow_metrics(
    flows: np.ndarray | None,
    distance_matrix: np.ndarray,
    *,
    flow_threshold: float,
) -> dict[str, Any]:
    if flows is None:
        return {
            "available": False,
            "positive_pairs": 0,
            "spearman_flow_similarity": None,
            "roc_auc_high_flow": None,
            "average_precision_high_flow": None,
        }
    flow = np.asarray(flows, dtype=np.float64).copy()
    np.fill_diagonal(flow, 0.0)
    offdiag = ~np.eye(flow.shape[0], dtype=bool)
    finite = np.isfinite(flow) & np.isfinite(distance_matrix) & offdiag
    labels = (flow[finite] > flow_threshold).astype(np.int64)
    scores = -distance_matrix[finite]
    positive_mask = flow[finite] > 0
    metrics: dict[str, Any] = {
        "available": True,
        "positive_pairs": int(labels.sum()),
        "flow_threshold": float(flow_threshold),
        "spearman_flow_similarity": _safe_spearman(
            flow[finite][positive_mask], scores[positive_mask]
        ),
        "roc_auc_high_flow": None,
        "average_precision_high_flow": None,
    }
    if np.unique(labels).size == 2:
        metrics["roc_auc_high_flow"] = _safe_float(roc_auc_score(labels, scores))
        metrics["average_precision_high_flow"] = _safe_float(
            average_precision_score(labels, scores)
        )
    return metrics


def compute_region_metrics(
    artifact: EmbeddingArtifact,
    graph: RegionGraphData,
    distance_matrix: np.ndarray,
    pca_coords: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    nearest_idx = np.argmin(distances, axis=1)
    degree = np.asarray(
        adjacency_matrix(graph.edge_index, len(artifact.region_ids)).sum(axis=1)
    ).reshape(-1)
    flows = graph.flows
    if flows is None:
        out_flow = np.full(len(artifact.region_ids), np.nan)
        in_flow = np.full(len(artifact.region_ids), np.nan)
    else:
        flow = flows.copy()
        np.fill_diagonal(flow, 0.0)
        out_flow = np.nansum(flow, axis=1)
        in_flow = np.nansum(flow, axis=0)

    data: dict[str, Any] = {
        "region_id": artifact.region_ids,
        "embedding_norm": np.linalg.norm(artifact.embeddings, axis=1),
        "nearest_neighbor_id": [artifact.region_ids[idx] for idx in nearest_idx],
        "nearest_neighbor_distance": distances[
            np.arange(distances.shape[0]), nearest_idx
        ],
        "graph_degree": degree,
        "out_flow": out_flow,
        "in_flow": in_flow,
        "total_flow": out_flow + in_flow,
        "pca1": pca_coords[:, 0],
        "pca2": pca_coords[:, 1],
        "cluster": labels,
    }
    for idx, name in enumerate(FEATURE_COLUMNS[: graph.features.shape[1]]):
        data[name] = graph.features[:, idx]
    return pd.DataFrame(data)


def compute_metrics(
    artifact: EmbeddingArtifact,
    graph: RegionGraphData,
    config: DiagnosticsConfig,
) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    embeddings = np.asarray(artifact.embeddings, dtype=np.float64)
    num_nodes = embeddings.shape[0]
    distance_matrix = squareform(pdist(embeddings, metric="euclidean"))
    condensed = distance_matrix[np.triu_indices(num_nodes, k=1)]
    norms = np.linalg.norm(embeddings, axis=1)
    pca_coords, pca = compute_pca(embeddings)
    labels, cluster_metrics = compute_clusters(
        embeddings, config.cluster_count, config.seed
    )
    hop_distances = compute_hop_distances(graph.edge_index, num_nodes)
    flow_threshold = (
        float(config.flow_threshold)
        if config.flow_threshold is not None
        else float(
            artifact.config.get("sampling", {}).get("min_flow_threshold", 0.0)
            if isinstance(artifact.config.get("sampling"), dict)
            else 0.0
        )
    )

    edge_mask = adjacency_matrix(graph.edge_index, num_nodes).toarray().astype(bool)
    upper = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    edge_distances = distance_matrix[upper & edge_mask]
    nonedge_distances = distance_matrix[upper & ~edge_mask]
    hop_eval_mask = (
        upper & np.isfinite(hop_distances) & (hop_distances <= config.max_hops)
    )
    finite_hops = hop_distances[hop_eval_mask]
    finite_hop_distances = distance_matrix[hop_eval_mask]

    overlap_df = compute_neighbor_overlaps(
        distance_matrix,
        hop_distances,
        graph.flows,
        flow_threshold=flow_threshold,
    )
    region_df = compute_region_metrics(
        artifact, graph, distance_matrix, pca_coords, labels
    )
    moran_df = compute_spatial_autocorrelation(embeddings, graph.edge_index)

    singular_values = np.linalg.svd(
        embeddings - embeddings.mean(axis=0), compute_uv=False
    )
    singular_share = (
        singular_values / singular_values.sum()
        if singular_values.sum() > 0
        else singular_values
    )
    entropy = -np.sum(singular_share * np.log(singular_share + 1e-12))
    effective_rank = float(np.exp(entropy)) if singular_share.size else 0.0

    metrics: dict[str, Any] = {
        "integrity": {
            "num_regions": int(num_nodes),
            "embedding_dim": int(embeddings.shape[1]),
            "finite_fraction": float(np.isfinite(embeddings).mean()),
            "duplicate_pairs_eps_1e-6": int((condensed < 1.0e-6).sum()),
        },
        "geometry": {
            "norms": _summary(norms),
            "pairwise_distance": _summary(condensed),
            "effective_rank": effective_rank,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "spatial_alignment": {
            "mean_edge_distance": _safe_float(np.mean(edge_distances))
            if edge_distances.size
            else None,
            "mean_nonedge_distance": _safe_float(np.mean(nonedge_distances))
            if nonedge_distances.size
            else None,
            "edge_nonedge_distance_ratio": (
                _safe_float(np.mean(edge_distances) / np.mean(nonedge_distances))
                if edge_distances.size
                and nonedge_distances.size
                and np.mean(nonedge_distances) > 0
                else None
            ),
            "spearman_hop_distance": _safe_spearman(finite_hops, finite_hop_distances),
            "knn_overlap": overlap_df.to_dict(orient="records"),
        },
        "flow_alignment": compute_flow_metrics(
            graph.flows, distance_matrix, flow_threshold=flow_threshold
        ),
        "clusters": {
            **cluster_metrics,
            "contiguity": cluster_contiguity(labels, graph.edge_index),
            "within_cluster_flow_share": flow_concentration(labels, graph.flows),
        },
    }
    return metrics, region_df, moran_df, overlap_df, distance_matrix, hop_distances


def plot_embedding_pca_scatter(region_df: pd.DataFrame, output_dir: Path) -> Path:
    color_columns = [
        col
        for col in ["population", "density", "graph_degree", "cluster"]
        if col in region_df
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    for ax, color_col in zip(axes.ravel(), color_columns, strict=False):
        sns.scatterplot(
            data=region_df,
            x="pca1",
            y="pca2",
            hue=color_col,
            palette="viridis" if color_col != "cluster" else "tab20",
            s=22,
            linewidth=0,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"PCA colored by {color_col}")
        add_grid(ax)
    for ax in axes.ravel()[len(color_columns) :]:
        ax.axis("off")
    path = output_dir / "embedding_pca_scatter.png"
    save_figure(fig, path, log_msg="Saved PCA scatter diagnostics")
    return path


def plot_distance_by_hop(
    distance_matrix: np.ndarray, hop_distances: np.ndarray, output_dir: Path
) -> Path:
    upper = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    finite = upper & np.isfinite(hop_distances) & (hop_distances > 0)
    plot_df = pd.DataFrame(
        {
            "hop": hop_distances[finite].astype(int),
            "embedding_distance": distance_matrix[finite],
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No finite hop distances", ha="center", va="center")
        ax.axis("off")
    else:
        sns.boxplot(data=plot_df, x="hop", y="embedding_distance", ax=ax)
        ax.set_title("Embedding Distance by Spatial Hop")
        add_grid(ax, axis="y")
    path = output_dir / "distance_by_hop.png"
    save_figure(fig, path, log_msg="Saved distance by hop plot")
    return path


def plot_flow_vs_embedding_similarity(
    flows: np.ndarray | None,
    distance_matrix: np.ndarray,
    output_dir: Path,
    *,
    max_points: int,
    seed: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    if flows is None:
        ax.text(0.5, 0.5, "No flow matrix available", ha="center", va="center")
        ax.axis("off")
    else:
        flow = flows.copy()
        np.fill_diagonal(flow, 0.0)
        mask = np.isfinite(flow) & (flow > 0)
        idx = np.argwhere(mask)
        if idx.size == 0:
            ax.text(0.5, 0.5, "No positive flows available", ha="center", va="center")
            ax.axis("off")
        else:
            rng = np.random.default_rng(seed)
            if len(idx) > max_points:
                idx = idx[rng.choice(len(idx), size=max_points, replace=False)]
            sampled_flow = flow[idx[:, 0], idx[:, 1]]
            sampled_distance = distance_matrix[idx[:, 0], idx[:, 1]]
            ax.scatter(
                np.log1p(sampled_flow),
                sampled_distance,
                s=8,
                alpha=0.35,
                linewidths=0,
            )
            ax.set_xlabel("log1p(flow)")
            ax.set_ylabel("embedding distance")
            ax.set_title("Flow Strength vs Embedding Distance")
            add_grid(ax)
    path = output_dir / "flow_vs_embedding_similarity.png"
    save_figure(fig, path, log_msg="Saved flow similarity plot")
    return path


def plot_nearest_neighbor_overlap(overlap_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for column, label in [
        ("one_hop_overlap", "1-hop overlap"),
        ("two_hop_overlap", "2-hop overlap"),
        ("flow_topk_recall", "flow top-k recall"),
    ]:
        if column in overlap_df and overlap_df[column].notna().any():
            ax.plot(overlap_df["k"], overlap_df[column], marker="o", label=label)
    ax.set_xlabel("embedding nearest-neighbor k")
    ax.set_ylabel("mean overlap / recall")
    ax.set_ylim(0, 1)
    ax.set_title("Embedding kNN Alignment")
    ax.legend()
    add_grid(ax)
    path = output_dir / "nearest_neighbor_overlap.png"
    save_figure(fig, path, log_msg="Saved nearest-neighbor overlap plot")
    return path


def plot_norms_and_pca_variance(
    region_df: pd.DataFrame, pca_variance: list[float], output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(region_df["embedding_norm"], bins=40, ax=axes[0])
    axes[0].set_title("Embedding Norm Distribution")
    add_grid(axes[0], axis="y")

    cumulative = np.cumsum(np.asarray(pca_variance, dtype=float))
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    axes[1].set_xlabel("PCA components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_ylim(0, min(1.05, max(0.05, float(cumulative[-1]) + 0.05)))
    axes[1].set_title("PCA Explained Variance")
    add_grid(axes[1])
    path = output_dir / "embedding_norms_and_pca_variance.png"
    save_figure(fig, path, log_msg="Saved norm/PCA plot")
    return path


def _load_geojson_for_regions(
    geojson_path: Path,
    region_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> Any:
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    id_field = str(metadata.get("geojson_id_field") or "id")
    if id_field not in gdf.columns:
        candidates = [
            col for col in ("id", "region_id", "codi", "codigo") if col in gdf
        ]
        if not candidates:
            raise ValueError(
                f"GeoJSON id field {id_field!r} missing and no fallback ID column found"
            )
        id_field = candidates[0]
    gdf = gdf.copy()
    gdf["region_id"] = gdf[id_field].astype(str)
    return gdf.merge(region_df, on="region_id", how="inner")


def plot_map_clusters(
    geojson_path: Path | None,
    region_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    if geojson_path is None or not geojson_path.exists():
        logger.warning("Skipping cluster map; GeoJSON path unavailable")
        return None
    try:
        merged = _load_geojson_for_regions(geojson_path, region_df, metadata)
        fig, ax = plt.subplots(figsize=(8, 8))
        merged.plot(
            column="cluster", categorical=True, legend=True, ax=ax, linewidth=0.1
        )
        ax.set_axis_off()
        ax.set_title("Region Embedding Clusters")
        path = output_dir / "embedding_map_clusters.png"
        save_figure(fig, path, log_msg="Saved cluster map")
        return path
    except Exception as exc:
        logger.warning("Skipping cluster map: %s", exc)
        return None


def plot_map_pc1_pc2(
    geojson_path: Path | None,
    region_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    if geojson_path is None or not geojson_path.exists():
        logger.warning("Skipping PCA maps; GeoJSON path unavailable")
        return None
    try:
        merged = _load_geojson_for_regions(geojson_path, region_df, metadata)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, column in zip(axes, ["pca1", "pca2"], strict=True):
            merged.plot(
                column=column, legend=True, ax=ax, cmap="viridis", linewidth=0.1
            )
            ax.set_axis_off()
            ax.set_title(column.upper())
        path = output_dir / "embedding_map_pc1_pc2.png"
        save_figure(fig, path, log_msg="Saved PCA maps")
        return path
    except Exception as exc:
        logger.warning("Skipping PCA maps: %s", exc)
        return None


def write_outputs(
    config: DiagnosticsConfig,
    artifact: EmbeddingArtifact,
    graph: RegionGraphData,
    metrics: dict[str, Any],
    region_df: pd.DataFrame,
    moran_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    hop_distances: np.ndarray,
) -> dict[str, str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = config.output_dir / "metrics.json"
    region_metrics_path = config.output_dir / "region_embedding_metrics.csv"
    moran_path = config.output_dir / "embedding_dim_spatial_autocorr.csv"
    overlap_path = config.output_dir / "nearest_neighbor_overlap.csv"

    metrics_path.write_text(
        json.dumps(_to_jsonable(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    region_df.to_csv(region_metrics_path, index=False)
    moran_df.to_csv(moran_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    plot_paths: dict[str, Path | None] = {
        "embedding_pca_scatter": plot_embedding_pca_scatter(
            region_df, config.output_dir
        ),
        "distance_by_hop": plot_distance_by_hop(
            distance_matrix, hop_distances, config.output_dir
        ),
        "flow_vs_embedding_similarity": plot_flow_vs_embedding_similarity(
            graph.flows,
            distance_matrix,
            config.output_dir,
            max_points=config.max_scatter_points,
            seed=config.seed,
        ),
        "nearest_neighbor_overlap_plot": plot_nearest_neighbor_overlap(
            overlap_df, config.output_dir
        ),
        "embedding_norms_and_pca_variance": plot_norms_and_pca_variance(
            region_df,
            metrics["geometry"]["pca_explained_variance_ratio"],
            config.output_dir,
        ),
        "embedding_map_clusters": plot_map_clusters(
            config.geojson, region_df, graph.metadata, config.output_dir
        ),
        "embedding_map_pc1_pc2": plot_map_pc1_pc2(
            config.geojson, region_df, graph.metadata, config.output_dir
        ),
    }

    artifacts = {
        "metrics": str(metrics_path),
        "region_metrics": str(region_metrics_path),
        "spatial_autocorr": str(moran_path),
        "nearest_neighbor_overlap": str(overlap_path),
    }
    artifacts.update(
        {name: str(path) for name, path in plot_paths.items() if path is not None}
    )
    artifacts_path = config.output_dir / "artifacts.json"
    artifacts["artifacts"] = str(artifacts_path)
    artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


def run_diagnostics(config: DiagnosticsConfig) -> dict[str, str]:
    artifact = load_embedding_artifact(config.embeddings)
    graph = load_region_graph(config.region_graph)
    artifact, graph, integrity = align_graph_to_artifact(artifact, graph)
    metrics, region_df, moran_df, overlap_df, distance_matrix, hop_distances = (
        compute_metrics(artifact, graph, config)
    )
    metrics["integrity"].update(integrity)
    artifacts = write_outputs(
        config,
        artifact,
        graph,
        metrics,
        region_df,
        moran_df,
        overlap_df,
        distance_matrix,
        hop_distances,
    )
    logger.info("Wrote Region2Vec diagnostics to %s", config.output_dir)
    return artifacts


def parse_args() -> DiagnosticsConfig:
    parser = argparse.ArgumentParser(
        description="Compute diagnostics and static plots for Region2Vec embeddings."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--region-graph", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-count", type=int, default=14)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--flow-threshold", type=float, default=None)
    parser.add_argument("--max-scatter-points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return DiagnosticsConfig(
        embeddings=args.embeddings,
        region_graph=args.region_graph,
        geojson=args.geojson,
        output_dir=args.output_dir,
        cluster_count=args.cluster_count,
        max_hops=args.max_hops,
        flow_threshold=args.flow_threshold,
        max_scatter_points=args.max_scatter_points,
        seed=args.seed,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    sns.set_theme(style="whitegrid")
    artifacts = run_diagnostics(parse_args())
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
