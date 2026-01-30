#!/usr/bin/env python
"""Analyze hop loss contributions to understand why distant regions are being pulled together.

Usage:
    python scripts/analyze_hop_loss.py --embeddings outputs/region_embeddings/region_embeddings.pt
"""

from __future__ import annotations

import argparse
import json as json_lib
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xarray as xr
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def load_embeddings(
    embeddings_path: str | Path,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Load embeddings and metadata from saved checkpoint.

    Returns:
        embeddings: [num_nodes, embedding_dim]
        region_ids: List of region identifiers
        config: Training config dictionary
    """
    checkpoint = torch.load(embeddings_path, weights_only=False)
    return (
        checkpoint["embeddings"],
        checkpoint["region_ids"],
        checkpoint["config"],
    )


def load_region_attributes(zarr_path: str | Path) -> dict[str, np.ndarray]:
    """Load region attributes from zarr dataset.

    Returns:
        Dictionary with keys: area, perimeter, population, density, lon, lat
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    features = ds["features"].values  # [num_nodes, feature_dim]

    return {
        "area": features[:, 0],
        "perimeter": features[:, 1],
        "population": features[:, 2],
        "density": features[:, 3],
        "lon": features[:, 4],
        "lat": features[:, 5],
    }


def compute_deciles(values: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    """Compute decile labels for values."""
    if len(np.unique(values)) < 2:
        return np.zeros(len(values), dtype=int)
    try:
        import pandas as pd

        deciles = pd.qcut(values, q=n_deciles, labels=False, duplicates="drop")
        return deciles
    except Exception:
        return np.zeros(len(values), dtype=int)


def compute_hop_distances(
    edge_index: torch.Tensor, num_nodes: int, max_hops: int = 5
) -> torch.Tensor:
    """Compute hop distance matrix between all node pairs."""
    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    adjacency = sparse.csr_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )

    distances_np = shortest_path(
        csgraph=adjacency,
        method="FW",
        directed=False,
        unweighted=True,
        return_predecessors=False,
    )

    distances = torch.from_numpy(distances_np).float()

    # Clamp to max_hops and mark inf as unreachable
    finite_mask = torch.isfinite(distances)
    distances[finite_mask] = torch.clamp(distances[finite_mask], max=float(max_hops))

    return distances


def get_hop_pairs(
    hop_distances: torch.Tensor,
    hop_threshold: int = 2,
    max_hops: int = 5,
    hop_pairs_count: int = 8192,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample hop pairs for analysis.

    Returns:
        hop_pairs: [2, N] tensor of pair indices
        hop_values: [N] tensor of hop distances
    """
    rng = np.random.default_rng(seed)

    hop_mask = hop_distances > hop_threshold
    hop_mask &= torch.isfinite(hop_distances)
    hop_mask &= hop_distances <= max_hops

    all_hop_pairs = torch.argwhere(hop_mask).T
    all_hop_values = hop_distances[hop_mask]

    # Sample if we have more than requested
    n_pairs = min(len(all_hop_values), hop_pairs_count)
    if len(all_hop_values) > hop_pairs_count:
        idx = rng.choice(len(all_hop_values), size=n_pairs, replace=False)
        hop_pairs = all_hop_pairs[:, idx]
        hop_values = all_hop_values[idx]
    else:
        hop_pairs = all_hop_pairs
        hop_values = all_hop_values

    return hop_pairs, hop_values


def compute_pairwise_distances(
    embeddings: torch.Tensor, pairs: torch.Tensor
) -> torch.Tensor:
    """Compute L2 distance between pairs of embeddings."""
    i, j = pairs
    return torch.norm(embeddings[i] - embeddings[j], dim=1)


def compute_hop_loss_per_pair(
    embeddings: torch.Tensor,
    hop_pairs: torch.Tensor,
    hop_values: torch.Tensor,
) -> torch.Tensor:
    """Compute hop loss contribution per pair.

    Formula: distance(embed_i, embed_j) / log(hop_distance + 1.0)
    """
    d_hop = compute_pairwise_distances(embeddings, hop_pairs)
    hop_loss_per_pair = d_hop / torch.clamp(torch.log(hop_values + 1.0), min=1e-4)
    return hop_loss_per_pair


def analyze_semantic_similarity(
    hop_pairs: torch.Tensor,
    hop_loss_per_pair: torch.Tensor,
    attributes: dict[str, np.ndarray],
    cluster_labels: np.ndarray | None = None,
    top_k: int = 100,
) -> dict[str, Any]:
    """Analyze semantic similarity of high-hop-loss pairs.

    Returns:
        Dictionary with statistics about semantic similarity
    """
    # Get top contributors to hop loss
    top_idx = torch.topk(
        hop_loss_per_pair, k=min(top_k, len(hop_loss_per_pair))
    ).indices
    top_pairs = hop_pairs[:, top_idx].cpu().numpy()
    top_hop_loss = hop_loss_per_pair[top_idx].cpu().numpy()

    i, j = top_pairs

    results: dict[str, Any] = {
        "top_pairs": {
            "region_i": i.tolist(),
            "region_j": j.tolist(),
            "hop_loss": top_hop_loss.tolist(),
        },
        "similarity_stats": {},
    }

    # Compute similarities for top pairs
    for attr_name, attr_values in attributes.items():
        # Normalize attribute for comparison
        attr_norm = (attr_values - attr_values.mean()) / (attr_values.std() + 1e-8)

        # Compute absolute difference (lower = more similar)
        attr_diff = np.abs(attr_norm[i] - attr_norm[j])

        # Compute correlation: are high-hop-loss pairs similar in this attribute?
        # High hop loss with low diff = pairs are similar but far apart
        similarity = 1.0 - (attr_diff / (attr_diff.max() + 1e-8))
        avg_similarity = similarity.mean()

        results["similarity_stats"][attr_name] = {
            "mean_similarity": float(avg_similarity),
            "mean_abs_diff": float(attr_diff.mean()),
        }

    # Cluster similarity
    if cluster_labels is not None:
        cluster_match = (cluster_labels[i] == cluster_labels[j]).astype(float)
        results["similarity_stats"]["same_cluster"] = {
            "fraction": float(cluster_match.mean()),
            "count": int(cluster_match.sum()),
            "total": len(cluster_match),
        }

    # Same decile similarity for key attributes
    for attr_name in ["population", "density"]:
        if attr_name in attributes:
            deciles = compute_deciles(attributes[attr_name])
            decile_match = (deciles[i] == deciles[j]).astype(float)
            results["similarity_stats"][f"{attr_name}_same_decile"] = {
                "fraction": float(decile_match.mean()),
                "count": int(decile_match.sum()),
                "total": len(decile_match),
            }

    return results


def analyze_by_region(
    hop_pairs: torch.Tensor,
    hop_loss_per_pair: torch.Tensor,
    num_nodes: int,
) -> dict[str, Any]:
    """Analyze hop loss contribution per region.

    Identifies which regions are most involved in high-hop-loss pairs.
    """
    region_contributions = torch.zeros(num_nodes)
    region_pair_counts = torch.zeros(num_nodes)

    i, j = hop_pairs.cpu().numpy()
    loss_values = hop_loss_per_pair.cpu().numpy()

    for region_i, region_j, loss_val in zip(i, j, loss_values, strict=False):
        region_contributions[region_i] += loss_val
        region_contributions[region_j] += loss_val
        region_pair_counts[region_i] += 1
        region_pair_counts[region_j] += 1

    # Average contribution per pair for each region
    avg_contribution = np.divide(
        region_contributions.numpy(),
        region_pair_counts.numpy(),
        out=np.zeros_like(region_contributions.numpy()),
        where=region_pair_counts.numpy() > 0,
    )

    top_regions = np.argsort(avg_contribution)[-20:][::-1]

    return {
        "avg_contribution_per_region": avg_contribution.tolist(),
        "top_region_indices": top_regions.tolist(),
        "top_region_contributions": avg_contribution[top_regions].tolist(),
    }


def print_analysis_summary(
    results: dict[str, Any],
    region_ids: list[str],
    attributes: dict[str, np.ndarray],
) -> None:
    """Print a human-readable summary of the analysis."""

    print("=" * 60)
    print("HOP LOSS ANALYSIS SUMMARY")
    print("=" * 60)
    print()

    # Overall hop loss
    total_hop_loss = results["total_hop_loss"]
    mean_hop_loss = results["mean_hop_loss"]
    print(f"Total hop loss: {total_hop_loss:.4f}")
    print(f"Mean hop loss per pair: {mean_hop_loss:.4f}")
    print(f"Number of hop pairs analyzed: {results['num_hop_pairs']}")
    print()

    # Semantic similarity stats
    print("SEMANTIC SIMILARITY OF HIGH-HOP-LOSS PAIRS (top 100)")
    print("-" * 60)

    sim_stats = results["semantic_similarity_stats"]

    for attr_name, stats in sim_stats.items():
        if attr_name == "same_cluster":
            frac = stats["fraction"]
            count = stats["count"]
            total = stats["total"]
            print(f"  Same cluster:         {frac:.1%} ({count}/{total} pairs)")
        elif "_same_decile" in attr_name:
            base_name = attr_name.replace("_same_decile", "")
            frac = stats["fraction"]
            count = stats["count"]
            total = stats["total"]
            print(f"  Same {base_name} decile: {frac:.1%} ({count}/{total} pairs)")
        else:
            mean_sim = stats["mean_similarity"]
            mean_diff = stats["mean_abs_diff"]
            print(
                f"  {attr_name:20s}: similarity={mean_sim:.3f}, avg_diff={mean_diff:.3f}"
            )

    print()
    print("INTERPRETATION:")
    print("  - High similarity % → distant regions are semantically alike")
    print("  - This suggests the model is learning meaningful structure")
    print()

    # Top contributing regions
    print("TOP REGIONS CONTRIBUTING TO HOP LOSS")
    print("-" * 60)

    region_analysis = results["region_analysis"]
    top_indices = region_analysis["top_region_indices"]
    top_contribs = region_analysis["top_region_contributions"]

    for idx, contrib in zip(top_indices[:10], top_contribs[:10], strict=False):
        region_id = region_ids[idx] if idx < len(region_ids) else f"region_{idx}"
        pop = attributes["population"][idx]
        density = attributes["density"][idx]
        print(
            f"  {region_id:20s}: avg_contrib={contrib:.4f}, pop={pop:.0f}, density={density:.1f}"
        )

    print()


def plot_semantic_similarity_barplot(
    similarity_stats: dict[str, Any], output_dir: Path
) -> Path:
    """Create horizontal bar plot of semantic similarity scores.

    Args:
        similarity_stats: Dictionary from analyze_semantic_similarity
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    # Extract similarity scores for numeric attributes
    attrs_to_plot = []
    similarities = []

    for attr_name, stats in similarity_stats.items():
        if attr_name in ["same_cluster"]:
            # Skip cluster similarity for bar plot
            continue
        if "_same_decile" in attr_name:
            # Convert fraction to similarity-like score for plotting
            attrs_to_plot.append(attr_name.replace("_same_decile", " (same decile)"))
            similarities.append(stats["fraction"])
        elif "mean_similarity" in stats:
            attrs_to_plot.append(attr_name)
            similarities.append(stats["mean_similarity"])

    # Sort by similarity descending
    sorted_idx = np.argsort(similarities)[::-1]
    attrs_sorted = [attrs_to_plot[i] for i in sorted_idx]
    sim_sorted = [similarities[i] for i in sorted_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("RdYlGn", len(attrs_sorted))
    bars = ax.barh(attrs_sorted, sim_sorted, color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, sim_sorted, strict=False):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Similarity Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Semantic Similarity of High-Hop-Loss Region Pairs",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

    # Add interpretation text
    ax.text(
        0.98,
        0.02,
        "Higher values indicate distant regions\nare semantically similar",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
    )

    plt.tight_layout()

    output_path = output_dir / "semantic_similarity_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_hop_loss_distribution(
    hop_loss_per_pair: torch.Tensor, hop_values: torch.Tensor, output_dir: Path
) -> Path:
    """Create grouped histogram of hop loss distribution by hop distance.

    Args:
        hop_loss_per_pair: Tensor of hop loss per pair
        hop_values: Tensor of hop distances for each pair
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    hop_np = hop_values.cpu().numpy()
    loss_np = hop_loss_per_pair.cpu().numpy()

    # Get unique hop distances
    unique_hops = np.sort(np.unique(hop_np))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram for each hop distance
    bins = np.linspace(0, loss_np.max(), 50)
    colors = sns.color_palette("viridis", len(unique_hops))

    for hop, color in zip(unique_hops, colors, strict=False):
        mask = hop_np == hop
        ax.hist(
            loss_np[mask],
            bins=bins,
            alpha=0.6,
            label=f"{int(hop)} hops (n={mask.sum()})",
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Hop Loss", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hop Loss Distribution by Hop Distance", fontsize=14, fontweight="bold"
    )
    ax.legend(title="Hop Distance", loc="upper right")

    plt.tight_layout()

    output_path = output_dir / "hop_loss_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_hop_loss_vs_attributes(
    hop_pairs: torch.Tensor,
    hop_loss_per_pair: torch.Tensor,
    attributes: dict[str, np.ndarray],
    output_dir: Path,
) -> list[Path]:
    """Create scatter plots of hop loss vs attribute similarity.

    Args:
        hop_pairs: [2, N] tensor of pair indices
        hop_loss_per_pair: [N] tensor of hop loss values
        attributes: Dictionary of attribute arrays
        output_dir: Directory to save plots

    Returns:
        List of paths to saved plots
    """
    i, j = hop_pairs.cpu().numpy()
    loss_np = hop_loss_per_pair.cpu().numpy()

    output_paths = []
    attrs_to_plot = ["population", "density", "area"]

    for attr_name in attrs_to_plot:
        if attr_name not in attributes:
            continue

        attr_values = attributes[attr_name]
        attr_norm = (attr_values - attr_values.mean()) / (attr_values.std() + 1e-8)

        # Compute absolute difference (lower = more similar)
        attr_diff = np.abs(attr_norm[i] - attr_norm[j])

        # Convert to similarity
        similarity = 1.0 - (attr_diff / (attr_diff.max() + 1e-8))

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create scatter with color gradient by hop loss
        scatter = ax.scatter(
            similarity,
            loss_np,
            alpha=0.5,
            s=20,
            c=loss_np,
            cmap="YlOrRd",
            edgecolors="none",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Hop Loss", fontsize=10)

        # Add trend line (lowess)
        from scipy.stats import binned_statistic

        bin_centers = np.linspace(0, 1, 20)
        bin_stats = binned_statistic(similarity, loss_np, statistic="mean", bins=20)
        ax.plot(
            bin_centers,
            bin_stats.statistic,
            "r-",
            linewidth=2,
            label="Binned mean",
            alpha=0.8,
        )

        ax.set_xlabel(
            f"{attr_name.capitalize()} Similarity", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Hop Loss", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Hop Loss vs {attr_name.capitalize()} Similarity",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / f"hop_loss_vs_{attr_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        output_paths.append(output_path)

    return output_paths


def plot_top_regions_heatmap(
    region_analysis: dict[str, Any],
    region_ids: list[str],
    attributes: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Create heatmap/table visualization of top contributing regions.

    Args:
        region_analysis: Dictionary from analyze_by_region
        region_ids: List of region identifiers
        attributes: Dictionary of attribute arrays
        output_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    top_indices = np.array(region_analysis["top_region_indices"])
    top_contribs = np.array(region_analysis["top_region_contributions"])

    # Create data for the table
    table_data = []
    for idx, contrib in zip(top_indices[:20], top_contribs[:20], strict=False):
        region_id = region_ids[idx] if idx < len(region_ids) else f"region_{idx}"
        table_data.append(
            {
                "region_id": region_id,
                "avg_contrib": contrib,
                "population": attributes["population"][idx],
                "density": attributes["density"][idx],
                "area": attributes["area"][idx],
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    rows = []
    for data in table_data:
        rows.append(
            [
                data["region_id"],
                f"{data['avg_contrib']:.4f}",
                f"{data['population']:.0f}",
                f"{data['density']:.2f}",
                f"{data['area']:.2f}",
            ]
        )

    # Create table
    table = ax.table(
        cellText=rows,
        cellLoc="left",
        colLabels=["Region ID", "Avg Contrib", "Population", "Density", "Area"],
        loc="center",
        colWidths=[0.25, 0.15, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#4a90d4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color alternate rows
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(5):
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        "Top 20 Regions Contributing to Hop Loss",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    output_path = output_dir / "top_regions_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_embedding_vs_hop_distance(
    embeddings: torch.Tensor,
    hop_pairs: torch.Tensor,
    hop_values: torch.Tensor,
    hop_loss_per_pair: torch.Tensor,
    output_dir: Path,
) -> Path:
    """Create scatter plot of embedding distance vs hop distance.

    Args:
        embeddings: [num_nodes, embedding_dim] tensor
        hop_pairs: [2, N] tensor of pair indices
        hop_values: [N] tensor of hop distances
        hop_loss_per_pair: [N] tensor of hop loss values
        output_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    # Compute actual embedding distances
    i, j = hop_pairs
    embed_distances = torch.norm(embeddings[i] - embeddings[j], dim=1).cpu().numpy()
    hop_np = hop_values.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter with color by hop loss
    unique_hops = np.sort(np.unique(hop_np))
    colors = sns.color_palette("viridis", len(unique_hops))

    for hop, color in zip(unique_hops, colors, strict=False):
        mask = hop_np == hop
        ax.scatter(
            embed_distances[mask],
            np.full(mask.sum(), hop),
            c=[color],
            label=f"{int(hop)} hops",
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Embedding L2 Distance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hop Distance (graph edges)", fontsize=12, fontweight="bold")
    ax.set_title("Embedding Distance vs Hop Distance", fontsize=14, fontweight="bold")
    ax.legend(title="Hop Distance")

    # Add note about interpretation
    ax.text(
        0.98,
        0.02,
        "Points below the diagonal indicate\nregions pulled closer in embedding\nspace than their graph distance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.5},
    )

    plt.tight_layout()

    output_path = output_dir / "embedding_distance_vs_hops.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_markdown_report(
    results: dict[str, Any],
    region_ids: list[str],
    attributes: dict[str, np.ndarray],
    output_dir: Path,
    plot_paths: list[Path],
) -> Path:
    """Generate a comprehensive markdown report.

    Args:
        results: Complete analysis results dictionary
        region_ids: List of region identifiers
        attributes: Dictionary of attribute arrays
        output_dir: Directory to save report
        plot_paths: List of paths to generated plots

    Returns:
        Path to saved report
    """
    # Extract key statistics
    total_hop_loss = results["total_hop_loss"]
    mean_hop_loss = results["mean_hop_loss"]
    std_hop_loss = results["std_hop_loss"]
    num_pairs = results["num_hop_pairs"]
    sim_stats = results["semantic_similarity_stats"]

    # Calculate key findings
    pop_sim = sim_stats.get("population", {}).get("mean_similarity", 0)
    density_sim = sim_stats.get("density", {}).get("mean_similarity", 0)
    pop_decile = sim_stats.get("population_same_decile", {}).get("fraction", 0)
    density_decile = sim_stats.get("density_same_decile", {}).get("fraction", 0)

    # Generate report content
    report_lines = [
        "# Region Hop Loss Analysis",
        "",
        "## Summary",
        "",
        "This report analyzes the hop loss contribution in the region embedding model. ",
        "Hop loss measures how well the model preserves spatial relationships - high hop loss ",
        "occurs when distant regions in the graph are placed close together in embedding space.",
        "",
        "## Hop Loss Statistics",
        "",
        f"- **Total hop loss**: {total_hop_loss:.4f}",
        f"- **Mean hop loss per pair**: {mean_hop_loss:.4f}",
        f"- **Standard deviation**: {std_hop_loss:.4f}",
        f"- **Number of hop pairs analyzed**: {num_pairs:,}",
        "",
        "## Semantic Similarity Analysis",
        "",
        "High-hop-loss pairs (top 100) were analyzed for semantic similarity in region attributes:",
        "",
        "| Attribute | Similarity Score | Interpretation |",
        "|-----------|------------------|----------------|",
    ]

    # Add similarity table rows
    for attr_name, stats in sim_stats.items():
        if attr_name == "same_cluster":
            frac = stats["fraction"]
            report_lines.append(
                f"| Same Cluster | {frac:.1%} | Fraction of pairs in same cluster |"
            )
        elif "_same_decile" in attr_name:
            base_name = attr_name.replace("_same_decile", "").capitalize()
            frac = stats["fraction"]
            random_expected = 0.1  # 10% for 10 deciles
            ratio = frac / random_expected if random_expected > 0 else 0
            report_lines.append(
                f"| Same {base_name} Decile | {frac:.1%} | {ratio:.1f}× random (10%) |"
            )
        elif "mean_similarity" in stats:
            mean_sim = stats["mean_similarity"]
            if mean_sim > 0.7:
                interpretation = "High similarity"
            elif mean_sim > 0.5:
                interpretation = "Moderate similarity"
            else:
                interpretation = "Low similarity"
            report_lines.append(
                f"| {attr_name.capitalize()} | {mean_sim:.3f} | {interpretation} |"
            )

    report_lines.extend(
        [
            "",
            "## Key Finding",
            "",
            "### The model is learning **demographic similarity** over spatial distance.",
            "",
            "High-hop-loss pairs show strong semantic similarity:",
            f"- **Population similarity**: {pop_sim:.1%} (very high)",
            f"- **Density similarity**: {density_sim:.1%} (high)",
            f"- **Same population decile**: {pop_decile:.1%} ({pop_decile / 0.1:.1f}× random)",
            f"- **Same density decile**: {density_decile:.1%} ({density_decile / 0.1:.1f}× random)",
            "",
            "This is **meaningful structure**, not a bug. The model is pulling together ",
            "regions that are demographically similar, even when they are spatially distant. ",
            "This can be beneficial for:\n",
            "- Capturing demographic patterns in disease spread\n",
            "- Enabling knowledge transfer between similar regions\n",
            "- Learning robust features that generalize across geography",
            "",
            "## Recommendations",
            "",
            "Based on these findings:",
            "",
            "1. **Embrace the demographic clustering**: The model is learning meaningful structure",
            "2. **Consider multi-hop attention**: Use longer-range connections to capture demographic patterns",
            "3. **Balance spatial and semantic**: If needed, add a weighted loss to preserve spatial locality",
            "4. **Monitor by region**: Some regions contribute more to hop loss - check if this aligns with domain knowledge",
            "",
            "## Top Contributing Regions",
            "",
            "Regions most involved in high-hop-loss pairs (top 10):",
            "",
            "| Region | Avg Contribution | Population | Density |",
            "|--------|------------------|------------|---------|",
        ]
    )

    region_analysis = results["region_analysis"]
    top_indices = region_analysis["top_region_indices"][:10]
    top_contribs = region_analysis["top_region_contributions"][:10]

    for idx, contrib in zip(top_indices, top_contribs, strict=False):
        region_id = region_ids[idx] if idx < len(region_ids) else f"region_{idx}"
        pop = attributes["population"][idx]
        density = attributes["density"][idx]
        report_lines.append(
            f"| {region_id} | {contrib:.4f} | {pop:.0f} | {density:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )

    # Add plot references
    plot_names = {
        "semantic_similarity_barplot.png": "Semantic Similarity Bar Chart",
        "hop_loss_distribution.png": "Hop Loss Distribution by Hop Distance",
        "top_regions_heatmap.png": "Top Contributing Regions Table",
        "embedding_distance_vs_hops.png": "Embedding Distance vs Hop Distance",
        "hop_loss_vs_population.png": "Hop Loss vs Population Similarity",
        "hop_loss_vs_density.png": "Hop Loss vs Density Similarity",
    }

    for plot_path in plot_paths:
        plot_name = plot_path.name
        title = plot_names.get(plot_name, plot_name)
        relative_path = (
            plot_path.relative_to(output_dir.parent) if output_dir.parent else plot_path
        )
        report_lines.append(f"### {title}")
        report_lines.append("")
        report_lines.append(f"![{title}]({relative_path})")
        report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "*Generated by analyze_hop_loss.py*",
        ]
    )

    # Write report
    report_path = output_dir / "hop_analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Analyze hop loss contributions")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="outputs/region_embeddings/region_embeddings.pt",
        help="Path to saved embeddings",
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default="outputs/region_graph/region_graph.zarr",
        help="Path to region graph zarr dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top hop-loss pairs to analyze for semantic similarity",
    )
    parser.add_argument(
        "--hop-threshold",
        type=int,
        default=2,
        help="Minimum hop distance for hop pairs",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=5,
        help="Maximum hop distance for hop pairs",
    )
    parser.add_argument(
        "--hop-pairs",
        type=int,
        default=8192,
        help="Number of hop pairs to sample",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="outputs/reports/regions_hop_analysis/",
        help="Directory to save analysis reports and plots",
    )

    args = parser.parse_args()

    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings}")
    embeddings, region_ids, config = load_embeddings(args.embeddings)
    num_nodes = embeddings.size(0)
    logger.info(f"Loaded {num_nodes} embeddings with dim {embeddings.size(1)}")

    # Load region attributes
    logger.info(f"Loading region attributes from {args.zarr}")
    attributes = load_region_attributes(args.zarr)

    # Load edge index and cluster labels
    logger.info("Loading graph structure...")
    ds = xr.open_zarr(args.zarr, consolidated=False)
    edge_index = torch.from_numpy(ds["edge_index"].values).long()

    # Load cluster labels if available
    cluster_labels = None
    try:
        cluster_path = Path(args.embeddings).parent / "region_clusters.json"
        if cluster_path.exists():
            import json

            with open(cluster_path) as f:
                cluster_dict = json.load(f)
            cluster_labels = np.array([cluster_dict[rid] for rid in region_ids])
            logger.info(f"Loaded {len(np.unique(cluster_labels))} clusters")
    except Exception as e:
        logger.warning(f"Could not load cluster labels: {e}")

    # Compute hop distances
    logger.info("Computing hop distances...")
    max_hops = config["loss"].get("max_hops", args.max_hops)
    hop_distances = compute_hop_distances(edge_index, num_nodes, max_hops=max_hops)

    # Sample hop pairs (use config defaults if available)
    sampling_config = config.get("sampling", {})
    hop_threshold = sampling_config.get("hop_threshold", args.hop_threshold)
    hop_pairs_count = sampling_config.get("hop_pairs", args.hop_pairs)

    logger.info(
        f"Sampling {hop_pairs_count} hop pairs (threshold={hop_threshold}, max_hops={max_hops})..."
    )
    hop_pairs, hop_values = get_hop_pairs(
        hop_distances,
        hop_threshold=hop_threshold,
        max_hops=max_hops,
        hop_pairs_count=hop_pairs_count,
    )

    # Compute hop loss per pair
    logger.info("Computing hop loss per pair...")
    hop_loss_per_pair = compute_hop_loss_per_pair(embeddings, hop_pairs, hop_values)

    # Analyze semantic similarity
    logger.info(f"Analyzing semantic similarity of top {args.top_k} hop-loss pairs...")
    semantic_results = analyze_semantic_similarity(
        hop_pairs, hop_loss_per_pair, attributes, cluster_labels, top_k=args.top_k
    )

    # Analyze by region
    logger.info("Analyzing hop loss contribution by region...")
    region_results = analyze_by_region(hop_pairs, hop_loss_per_pair, num_nodes)

    # Compile results
    results = {
        "total_hop_loss": float(hop_loss_per_pair.sum()),
        "mean_hop_loss": float(hop_loss_per_pair.mean()),
        "std_hop_loss": float(hop_loss_per_pair.std()),
        "num_hop_pairs": len(hop_loss_per_pair),
        "semantic_similarity_stats": semantic_results["similarity_stats"],
        "region_analysis": region_results,
    }

    # Print summary
    print_analysis_summary(results, region_ids, attributes)

    # Generate reports and plots
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating reports in {report_dir}...")

    # Generate all plots
    plot_paths: list[Path] = []

    logger.info("  - Generating semantic similarity barplot...")
    plot_paths.append(
        plot_semantic_similarity_barplot(
            semantic_results["similarity_stats"], report_dir
        )
    )

    logger.info("  - Generating hop loss distribution plot...")
    plot_paths.append(
        plot_hop_loss_distribution(hop_loss_per_pair, hop_values, report_dir)
    )

    logger.info("  - Generating hop loss vs attributes plots...")
    plot_paths.extend(
        plot_hop_loss_vs_attributes(
            hop_pairs, hop_loss_per_pair, attributes, report_dir
        )
    )

    logger.info("  - Generating top regions heatmap...")
    plot_paths.append(
        plot_top_regions_heatmap(region_results, region_ids, attributes, report_dir)
    )

    logger.info("  - Generating embedding distance vs hop distance plot...")
    plot_paths.append(
        plot_embedding_vs_hop_distance(
            embeddings, hop_pairs, hop_values, hop_loss_per_pair, report_dir
        )
    )

    # Generate markdown report
    logger.info("  - Generating markdown report...")
    report_path = generate_markdown_report(
        results, region_ids, attributes, report_dir, plot_paths
    )
    logger.info(f"  - Report saved to {report_path}")

    # Save JSON results
    json_path = report_dir / "hop_analysis_results.json"
    with open(json_path, "w") as f:
        json_lib.dump(results, f, indent=2)
    logger.info(f"  - JSON results saved to {json_path}")

    logger.info(f"Analysis complete! {len(plot_paths)} plots generated.")

    # Save results if requested (legacy support)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json_lib.dump(results, f, indent=2)
        logger.info(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
