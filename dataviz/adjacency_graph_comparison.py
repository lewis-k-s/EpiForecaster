"""Compare dynamic mobility and static spatial-KNN adjacency graphs.

This script visualizes how the graph topology changes when the mobility GNN is
fed a municipality centroid KNN graph instead of OD mobility adjacency. It uses
the same core assumptions as the dataloader-side feature mask:

- adjacency[i, j] means node i can contribute to target/frontier node j;
- receptive fields expand through incoming graph edges;
- the center node is always included in its ego graph;
- a configurable neighbor ceiling can limit newly added ego nodes at each hop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString

sys_path = str(Path(__file__).parent.parent)
if sys_path not in sys.path:
    sys.path.append(sys_path)

from data.epi_dataset import EpiDataset  # noqa: E402
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from models.configs import EpiForecasterConfig  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_REGION_SOURCE = Path("data/files/geo/fl_municipios_catalonia.geojson")
SPATIAL_KNN_CRS = "EPSG:25831"


@dataclass(frozen=True)
class GraphPair:
    region_ids: list[str]
    region_names: list[str]
    centroids: np.ndarray
    mobility_adjacency: np.ndarray
    mobility_weights: np.ndarray
    spatial_adjacency: np.ndarray
    spatial_scores: np.ndarray
    regions_gdf: gpd.GeoDataFrame
    date_label: str


def _resolve_region_source(config: EpiForecasterConfig, fallback: Path) -> Path:
    configured = config.data.regions_data_path.strip()
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = (Path.cwd() / configured_path).resolve()
        if configured_path.exists():
            return configured_path
        logger.warning("Configured regions_data_path does not exist: %s", configured_path)

    fallback_path = fallback if fallback.is_absolute() else (Path.cwd() / fallback)
    if fallback_path.exists():
        return fallback_path.resolve()
    raise FileNotFoundError(
        "No region GeoJSON found. Set data.regions_data_path or pass --regions-geojson."
    )


def _ordered_regions(
    region_source: Path, region_ids: list[str]
) -> tuple[gpd.GeoDataFrame, list[str], np.ndarray]:
    regions = gpd.read_file(region_source)
    if "id" not in regions.columns:
        raise ValueError(f"{region_source} must contain an 'id' property")

    name_by_id = {}
    if "name" in regions.columns:
        name_by_id = dict(zip(regions["id"].astype(str), regions["name"].astype(str)))

    regions = regions.copy()
    regions["id"] = regions["id"].astype(str)
    regions = regions.drop_duplicates(subset="id", keep="first").set_index("id")

    missing = [region_id for region_id in region_ids if region_id not in regions.index]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"{region_source} is missing {len(missing)} dataset region IDs: {preview}"
        )

    ordered = regions.loc[region_ids].copy()
    if ordered.crs is None:
        logger.warning("GeoJSON has no CRS metadata; assuming %s", SPATIAL_KNN_CRS)
        ordered = ordered.set_crs(SPATIAL_KNN_CRS)
    else:
        ordered = ordered.to_crs(SPATIAL_KNN_CRS)

    centroids_geom = ordered.geometry.centroid
    centroids = np.column_stack(
        [centroids_geom.x.to_numpy(), centroids_geom.y.to_numpy()]
    )
    names = [name_by_id.get(region_id, region_id) for region_id in region_ids]
    ordered["region_name"] = names
    return ordered, names, centroids


def _mobility_array(dataset: xr.Dataset) -> xr.DataArray:
    if "mobility" not in dataset:
        raise ValueError("Canonical dataset is missing the 'mobility' variable")

    mobility = dataset["mobility"]
    if "run_id" in mobility.dims:
        mobility = mobility.squeeze("run_id", drop=True)

    dims = set(mobility.dims)
    if {TEMPORAL_COORD, "origin", "destination"}.issubset(dims):
        return mobility.transpose(TEMPORAL_COORD, "origin", "destination")
    if {TEMPORAL_COORD, REGION_COORD, "region_id_to"}.issubset(dims):
        return mobility.transpose(TEMPORAL_COORD, REGION_COORD, "region_id_to")
    if {TEMPORAL_COORD, REGION_COORD, REGION_COORD}.issubset(dims):
        return mobility.transpose(TEMPORAL_COORD, REGION_COORD, REGION_COORD)

    raise ValueError(f"Unsupported mobility dimensions: {mobility.dims}")


def _select_time_index(
    mobility: xr.DataArray, *, date: str | None, time_index: int | None
) -> tuple[int, str]:
    dates = pd.to_datetime(mobility[TEMPORAL_COORD].values)
    if date is not None:
        target = pd.Timestamp(date)
        matches = np.flatnonzero(dates == target)
        if len(matches) == 0:
            raise ValueError(f"Date {date!r} not found in mobility date coordinate")
        idx = int(matches[0])
    elif time_index is not None:
        idx = int(time_index)
    else:
        idx = len(dates) // 2

    if idx < 0 or idx >= len(dates):
        raise ValueError(f"time_index {idx} is outside [0, {len(dates) - 1}]")
    return idx, str(dates[idx].date())


def _build_spatial_knn(
    centroids: np.ndarray, *, k: int
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = centroids.shape[0]
    k = min(int(k), max(0, num_nodes - 1))
    adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
    scores = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if k == 0:
        return adjacency, scores

    deltas = centroids[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    np.fill_diagonal(distances, np.inf)

    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    row_idx = np.repeat(np.arange(num_nodes), k)
    col_idx = nearest.reshape(-1)
    adjacency[row_idx, col_idx] = True
    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, False)

    finite_distances = distances[np.isfinite(distances)]
    max_distance = float(finite_distances.max()) if finite_distances.size else 1.0
    scores[adjacency] = max_distance - distances[adjacency]
    return adjacency, scores


def load_graph_pair(
    config: EpiForecasterConfig,
    *,
    regions_geojson: Path,
    date: str | None,
    time_index: int | None,
    mobility_threshold: float,
    knn_k: int,
) -> GraphPair:
    dataset = EpiDataset.load_canonical_dataset(
        Path(config.data.dataset_path),
        run_id=config.data.run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    try:
        region_ids = [str(region_id) for region_id in dataset[REGION_COORD].values]
        regions_gdf, region_names, centroids = _ordered_regions(regions_geojson, region_ids)
        mobility = _mobility_array(dataset)
        idx, date_label = _select_time_index(
            mobility, date=date, time_index=time_index
        )
        mobility_matrix = np.asarray(mobility.isel({TEMPORAL_COORD: idx}).values)
    finally:
        dataset.close()

    mobility_matrix = np.nan_to_num(mobility_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mobility_adjacency = mobility_matrix > mobility_threshold
    np.fill_diagonal(mobility_adjacency, False)

    spatial_adjacency, spatial_scores = _build_spatial_knn(centroids, k=knn_k)
    return GraphPair(
        region_ids=region_ids,
        region_names=region_names,
        centroids=centroids,
        mobility_adjacency=mobility_adjacency,
        mobility_weights=mobility_matrix.astype(np.float32),
        spatial_adjacency=spatial_adjacency,
        spatial_scores=spatial_scores,
        regions_gdf=regions_gdf,
        date_label=date_label,
    )


def _edge_count(adjacency: np.ndarray, *, directed: bool) -> int:
    count = int(adjacency.sum())
    if directed:
        return count
    return int(np.triu(adjacency, k=1).sum())


def graph_summary(pair: GraphPair) -> pd.DataFrame:
    mobility = pair.mobility_adjacency
    mobility_undirected = mobility | mobility.T
    spatial = pair.spatial_adjacency
    n = len(pair.region_ids)
    directed_denominator = n * max(n - 1, 1)
    undirected_denominator = n * max(n - 1, 1) / 2

    overlap = mobility_undirected & spatial
    union = mobility_undirected | spatial
    return pd.DataFrame(
        [
            {
                "graph": "mobility_directed",
                "edges": _edge_count(mobility, directed=True),
                "density": float(mobility.sum() / directed_denominator),
                "mean_degree": float(mobility.sum(axis=0).mean()),
                "median_degree": float(np.median(mobility.sum(axis=0))),
            },
            {
                "graph": "mobility_undirected",
                "edges": _edge_count(mobility_undirected, directed=False),
                "density": float(np.triu(mobility_undirected, k=1).sum() / undirected_denominator),
                "mean_degree": float(mobility_undirected.sum(axis=0).mean()),
                "median_degree": float(np.median(mobility_undirected.sum(axis=0))),
            },
            {
                "graph": "spatial_knn",
                "edges": _edge_count(spatial, directed=False),
                "density": float(np.triu(spatial, k=1).sum() / undirected_denominator),
                "mean_degree": float(spatial.sum(axis=0).mean()),
                "median_degree": float(np.median(spatial.sum(axis=0))),
            },
            {
                "graph": "overlap_undirected",
                "edges": _edge_count(overlap, directed=False),
                "density": float(np.triu(overlap, k=1).sum() / undirected_denominator),
                "edge_jaccard_vs_union": float(overlap.sum() / union.sum())
                if union.any()
                else 0.0,
            },
        ]
    )


def _ego_receptive_field(
    adjacency: np.ndarray,
    scores: np.ndarray,
    center: int,
    *,
    max_hops: int,
    neighbor_ceiling: int | None,
) -> set[int]:
    seen = {int(center)}
    frontier = {int(center)}

    for _ in range(max_hops):
        candidates: dict[int, float] = {}
        for dst in frontier:
            incoming = np.flatnonzero(adjacency[:, dst])
            for src in incoming:
                src = int(src)
                if src in seen:
                    continue
                candidates[src] = max(candidates.get(src, 0.0), float(scores[src, dst]))

        if neighbor_ceiling is not None and len(candidates) > neighbor_ceiling:
            ranked = sorted(candidates, key=lambda node: (-candidates[node], node))
            candidates = {node: candidates[node] for node in ranked[:neighbor_ceiling]}

        frontier = set(candidates)
        if not frontier:
            break
        seen.update(frontier)

    return seen


def ego_comparison(
    pair: GraphPair,
    *,
    max_hops: int,
    neighbor_ceiling: int | None,
) -> pd.DataFrame:
    rows = []
    for idx, (region_id, region_name) in enumerate(zip(pair.region_ids, pair.region_names)):
        mobility_nodes = _ego_receptive_field(
            pair.mobility_adjacency,
            pair.mobility_weights,
            idx,
            max_hops=max_hops,
            neighbor_ceiling=neighbor_ceiling,
        )
        spatial_nodes = _ego_receptive_field(
            pair.spatial_adjacency,
            pair.spatial_scores,
            idx,
            max_hops=max_hops,
            neighbor_ceiling=neighbor_ceiling,
        )
        overlap = mobility_nodes & spatial_nodes
        union = mobility_nodes | spatial_nodes
        rows.append(
            {
                "node_idx": idx,
                "region_id": region_id,
                "region_name": region_name,
                "mobility_ego_size": len(mobility_nodes),
                "spatial_knn_ego_size": len(spatial_nodes),
                "overlap_size": len(overlap),
                "mobility_only_size": len(mobility_nodes - spatial_nodes),
                "spatial_only_size": len(spatial_nodes - mobility_nodes),
                "ego_jaccard": len(overlap) / len(union) if union else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _sample_edges(
    adjacency: np.ndarray,
    *,
    directed: bool,
    max_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if directed:
        edges = np.column_stack(np.flatnonzero(adjacency).reshape(-1, 1))
        rows, cols = np.where(adjacency)
    else:
        rows, cols = np.where(np.triu(adjacency, k=1))
    edges = np.column_stack([rows, cols])
    if len(edges) > max_edges:
        keep = rng.choice(len(edges), size=max_edges, replace=False)
        edges = edges[keep]
    return edges


def _edge_geometries(pair: GraphPair, edges: np.ndarray) -> gpd.GeoDataFrame:
    if len(edges) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=SPATIAL_KNN_CRS)
    lines = [
        LineString([tuple(pair.centroids[src]), tuple(pair.centroids[dst])])
        for src, dst in edges
    ]
    return gpd.GeoDataFrame(geometry=lines, crs=SPATIAL_KNN_CRS)


def plot_degree_distribution(pair: GraphPair, output_path: Path) -> None:
    mobility_in = pair.mobility_adjacency.sum(axis=0)
    mobility_out = pair.mobility_adjacency.sum(axis=1)
    spatial_degree = pair.spatial_adjacency.sum(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = max(12, int(np.sqrt(len(pair.region_ids))))
    ax.hist(mobility_in, bins=bins, alpha=0.55, label="Mobility in-degree")
    ax.hist(mobility_out, bins=bins, alpha=0.45, label="Mobility out-degree")
    ax.hist(spatial_degree, bins=bins, alpha=0.55, label="Spatial KNN degree")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Municipality count")
    ax.set_title(f"Adjacency Degree Distribution ({pair.date_label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ego_summary(ego_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        ego_df["mobility_ego_size"],
        ego_df["spatial_knn_ego_size"],
        c=ego_df["ego_jaccard"],
        cmap="viridis",
        alpha=0.75,
        s=26,
    )
    max_size = max(
        int(ego_df["mobility_ego_size"].max()),
        int(ego_df["spatial_knn_ego_size"].max()),
    )
    axes[0].plot([0, max_size], [0, max_size], color="0.25", linewidth=1)
    axes[0].set_xlabel("Mobility ego size")
    axes[0].set_ylabel("Spatial KNN ego size")
    axes[0].set_title("Receptive Field Size")

    axes[1].hist(ego_df["ego_jaccard"], bins=25, color="#3b6ea8", alpha=0.85)
    axes[1].set_xlabel("Ego-node Jaccard")
    axes[1].set_ylabel("Municipality count")
    axes[1].set_title("Receptive Field Overlap")
    fig.tight_layout()
    fig.savefig(output_dir / "ego_receptive_field_summary.png", dpi=180)
    plt.close(fig)


def plot_edge_overlay(
    pair: GraphPair,
    output_path: Path,
    *,
    max_edges: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    mobility_edges = _sample_edges(
        pair.mobility_adjacency | pair.mobility_adjacency.T,
        directed=False,
        max_edges=max_edges,
        rng=rng,
    )
    spatial_edges = _sample_edges(
        pair.spatial_adjacency,
        directed=False,
        max_edges=max_edges,
        rng=rng,
    )
    mobility_lines = _edge_geometries(pair, mobility_edges)
    spatial_lines = _edge_geometries(pair, spatial_edges)

    fig, ax = plt.subplots(figsize=(9, 9))
    pair.regions_gdf.boundary.plot(ax=ax, linewidth=0.25, color="0.8")
    if not mobility_lines.empty:
        mobility_lines.plot(ax=ax, color="#d95f02", linewidth=0.35, alpha=0.25)
    if not spatial_lines.empty:
        spatial_lines.plot(ax=ax, color="#1b9e77", linewidth=0.45, alpha=0.35)
    ax.set_title(
        f"Edge Overlay Sample ({pair.date_label})\n"
        "orange: mobility skeleton, green: spatial KNN"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ego_difference_maps(
    pair: GraphPair,
    ego_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_hops: int,
    neighbor_ceiling: int | None,
    top_n: int,
) -> None:
    selected = ego_df.sort_values(
        ["ego_jaccard", "overlap_size"], ascending=[True, True]
    ).head(top_n)

    for _, row in selected.iterrows():
        center = int(row["node_idx"])
        mobility_nodes = _ego_receptive_field(
            pair.mobility_adjacency,
            pair.mobility_weights,
            center,
            max_hops=max_hops,
            neighbor_ceiling=neighbor_ceiling,
        )
        spatial_nodes = _ego_receptive_field(
            pair.spatial_adjacency,
            pair.spatial_scores,
            center,
            max_hops=max_hops,
            neighbor_ceiling=neighbor_ceiling,
        )
        shared = mobility_nodes & spatial_nodes
        mobility_only = mobility_nodes - spatial_nodes
        spatial_only = spatial_nodes - mobility_nodes

        plot_gdf = pair.regions_gdf.copy()
        categories = np.full(len(plot_gdf), "outside", dtype=object)
        categories[list(shared)] = "shared"
        categories[list(mobility_only)] = "mobility_only"
        categories[list(spatial_only)] = "spatial_only"
        categories[center] = "center"
        plot_gdf["category"] = categories

        colors = {
            "outside": "#f0f0f0",
            "shared": "#7570b3",
            "mobility_only": "#d95f02",
            "spatial_only": "#1b9e77",
            "center": "#000000",
        }

        fig, ax = plt.subplots(figsize=(8, 8))
        for category, color in colors.items():
            subset = plot_gdf[plot_gdf["category"] == category]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=color,
                    edgecolor="white",
                    linewidth=0.15,
                    alpha=0.95 if category != "outside" else 0.35,
                    label=category,
                )
        ax.set_title(
            f"Ego Difference: {row['region_name']} ({row['region_id']})\n"
            f"Jaccard={row['ego_jaccard']:.2f}, mobility={row['mobility_ego_size']}, "
            f"spatial={row['spatial_knn_ego_size']}"
        )
        ax.set_axis_off()
        handles = [
            Patch(facecolor=color, edgecolor="white", label=category)
            for category, color in colors.items()
            if category in set(categories)
        ]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
        fig.tight_layout()
        output_name = f"ego_difference_node_{center}_{row['region_id']}.png"
        fig.savefig(output_dir / output_name, dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mobility and spatial-KNN adjacency receptive fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted config override. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/adjacency_graph_comparison"),
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--regions-geojson",
        type=Path,
        default=None,
        help="GeoJSON with municipality polygons. Defaults to config then repo GeoJSON.",
    )
    parser.add_argument("--date", type=str, default=None, help="Exact date to compare")
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Time index to compare when --date is not provided. Defaults to midpoint.",
    )
    parser.add_argument(
        "--mobility-threshold",
        type=float,
        default=0.0,
        help="Mobility edge threshold.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=None,
        help="K for centroid KNN. Defaults to model.max_neighbors.",
    )
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum ego hops")
    parser.add_argument(
        "--neighbor-ceiling",
        type=int,
        default=None,
        help="Maximum newly added ego nodes per hop. Omit for uncapped.",
    )
    parser.add_argument(
        "--top-ego-maps",
        type=int,
        default=6,
        help="Number of most different ego neighborhoods to map.",
    )
    parser.add_argument(
        "--max-overlay-edges",
        type=int,
        default=2500,
        help="Maximum edges per graph in the overlay map.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for edge samples")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    config = EpiForecasterConfig.load(str(args.config), overrides=args.overrides)
    region_source = (
        args.regions_geojson.resolve()
        if args.regions_geojson is not None
        else _resolve_region_source(config, DEFAULT_REGION_SOURCE)
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(args.mobility_threshold)
    knn_k = int(args.knn_k) if args.knn_k is not None else int(config.model.max_neighbors)
    neighbor_ceiling = (
        int(args.neighbor_ceiling) if args.neighbor_ceiling is not None else None
    )

    logger.info("Loading graph pair from %s", config.data.dataset_path)
    pair = load_graph_pair(
        config,
        regions_geojson=region_source,
        date=args.date,
        time_index=args.time_index,
        mobility_threshold=threshold,
        knn_k=knn_k,
    )

    summary_df = graph_summary(pair)
    ego_df = ego_comparison(
        pair,
        max_hops=int(args.max_hops),
        neighbor_ceiling=neighbor_ceiling,
    )

    summary_path = output_dir / "adjacency_graph_summary.csv"
    ego_path = output_dir / "ego_receptive_field_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    ego_df.to_csv(ego_path, index=False)

    plot_degree_distribution(pair, output_dir / "degree_distribution.png")
    plot_ego_summary(ego_df, output_dir)
    plot_edge_overlay(
        pair,
        output_dir / "edge_overlay_sample.png",
        max_edges=int(args.max_overlay_edges),
        seed=int(args.seed),
    )
    if args.top_ego_maps > 0:
        plot_ego_difference_maps(
            pair,
            ego_df,
            output_dir,
            max_hops=int(args.max_hops),
            neighbor_ceiling=neighbor_ceiling,
            top_n=int(args.top_ego_maps),
        )

    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", ego_path)
    logger.info("Wrote plots to %s", output_dir)


if __name__ == "__main__":
    main()
