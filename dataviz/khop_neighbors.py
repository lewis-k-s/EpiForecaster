"""K-hop neighbor visualization and analysis for mobility graphs.

This module provides tools to visualize and quantify k-hop neighbors in the mobility
graph data to understand the receptive field of spatial aggregation for minibatch tuning.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch_geometric.utils import k_hop_subgraph

# Optional dependencies with graceful fallbacks
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print(
        "Warning: networkx not available. Graph visualization features will be limited."
    )

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Interactive visualizations will be disabled.")


class KHopNeighborAnalyzer:
    """Analyzes k-hop neighborhoods in mobility graphs."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int | None = None):
        """
        Initialize the analyzer with a graph structure.

        Args:
            edge_index: Edge connectivity in COO format [2, num_edges]
            num_nodes: Total number of nodes in the graph
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes or int(edge_index.max().item()) + 1

        # Build adjacency list representation for faster neighbor queries
        self.adj_list = self._build_adjacency_list()

        # Cache for k-hop neighborhoods
        self.khop_cache = {}

    def _build_adjacency_list(self) -> dict[int, set[int]]:
        """Build adjacency list from edge index."""
        adj_list = defaultdict(set)
        edges = self.edge_index.t().numpy()

        for src, dst in edges:
            adj_list[int(src)].add(int(dst))
            adj_list[int(dst)].add(
                int(src)
            )  # Treat as undirected for neighbor analysis

        return dict(adj_list)

    def get_khop_neighbors(
        self, node_idx: int, k: int, include_self: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get k-hop neighbors of a node using PyTorch Geometric utilities.

        Args:
            node_idx: Index of the center node
            k: Number of hops
            include_self: Whether to include the center node

        Returns:
            subset: Node indices in the k-hop neighborhood
            sub_edge_index: Edge index of the subgraph
        """
        cache_key = (node_idx, k, include_self)
        if cache_key in self.khop_cache:
            return self.khop_cache[cache_key]

        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=int(node_idx),  # Ensure it's a Python int
            num_hops=k,
            edge_index=self.edge_index,
            relabel_nodes=False,
            num_nodes=self.num_nodes,
        )

        if not include_self:
            mask = subset != node_idx
            subset = subset[mask]

        self.khop_cache[cache_key] = (subset, sub_edge_index)
        return subset, sub_edge_index

    def analyze_receptive_field(
        self, max_k: int = 5, sample_nodes: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Analyze receptive field growth with increasing k.

        Args:
            max_k: Maximum number of hops to analyze
            sample_nodes: Specific nodes to sample (if None, samples randomly)

        Returns:
            DataFrame with receptive field statistics
        """
        if sample_nodes is None:
            # Sample 10% of nodes or at least 10 nodes
            n_samples = max(10, self.num_nodes // 10)
            sample_nodes = np.random.choice(
                self.num_nodes, min(n_samples, self.num_nodes), replace=False
            )

        results = []
        for node_idx in sample_nodes:
            for k in range(1, max_k + 1):
                neighbors, _ = self.get_khop_neighbors(node_idx, k, include_self=False)

                results.append(
                    {
                        "node_idx": node_idx,
                        "k": k,
                        "num_neighbors": len(neighbors),
                        "coverage": len(neighbors) / self.num_nodes,
                    }
                )

        return pd.DataFrame(results)

    def compute_neighbor_statistics(self, k: int = 2) -> dict:
        """
        Compute statistics about k-hop neighborhoods across all nodes.

        Args:
            k: Number of hops

        Returns:
            Dictionary with statistics
        """
        neighbor_counts = []
        coverages = []

        for node_idx in range(self.num_nodes):
            neighbors, _ = self.get_khop_neighbors(node_idx, k, include_self=False)
            neighbor_counts.append(len(neighbors))
            coverages.append(len(neighbors) / self.num_nodes)

        return {
            "k": k,
            "mean_neighbors": np.mean(neighbor_counts),
            "std_neighbors": np.std(neighbor_counts),
            "min_neighbors": np.min(neighbor_counts),
            "max_neighbors": np.max(neighbor_counts),
            "median_neighbors": np.median(neighbor_counts),
            "mean_coverage": np.mean(coverages),
            "std_coverage": np.std(coverages),
            "percentiles": {
                "25": np.percentile(neighbor_counts, 25),
                "50": np.percentile(neighbor_counts, 50),
                "75": np.percentile(neighbor_counts, 75),
                "90": np.percentile(neighbor_counts, 90),
                "95": np.percentile(neighbor_counts, 95),
                "99": np.percentile(neighbor_counts, 99),
            },
        }


class KHopVisualizer:
    """Visualize k-hop neighborhoods and receptive fields."""

    def __init__(self, analyzer: KHopNeighborAnalyzer):
        """
        Initialize visualizer with an analyzer.

        Args:
            analyzer: KHopNeighborAnalyzer instance
        """
        self.analyzer = analyzer

    def plot_khop_subgraph(
        self,
        node_idx: int,
        k: int,
        node_labels: dict | None = None,
        edge_weights: torch.Tensor | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Visualize k-hop neighborhood of a specific node.

        Args:
            node_idx: Center node index
            k: Number of hops
            node_labels: Optional node labels dictionary
            edge_weights: Optional edge weights
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for graph visualization. Install with: pip install networkx"
            )
        # Get k-hop subgraph
        subset, sub_edge_index = self.analyzer.get_khop_neighbors(node_idx, k)

        # Create NetworkX graph for visualization
        G = nx.Graph()

        # Add nodes
        for node in subset.numpy():
            label = node_labels.get(node, str(node)) if node_labels else str(node)
            G.add_node(node, label=label)

        # Add edges
        edges = sub_edge_index.t().numpy()
        for src, dst in edges:
            if src in subset and dst in subset:
                weight = 1.0
                if edge_weights is not None:
                    edge_idx = np.where((edges[:, 0] == src) & (edges[:, 1] == dst))[0]
                    if len(edge_idx) > 0:
                        weight = float(edge_weights[edge_idx[0]])
                G.add_edge(src, dst, weight=weight)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Draw graph with spring layout
        pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

        # Color nodes based on distance from center
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == node_idx:
                node_colors.append("red")
                node_sizes.append(500)
            else:
                # Calculate actual hop distance
                try:
                    distance = nx.shortest_path_length(G, node_idx, node)
                    node_colors.append(plt.cm.viridis(distance / k))
                    node_sizes.append(300 - distance * 50)
                except Exception:
                    node_colors.append("gray")
                    node_sizes.append(200)

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, ax=ax1, alpha=0.8
        )

        if edge_weights is not None:
            edge_widths = [G[u][v]["weight"] for u, v in G.edges()]
            edge_widths = np.array(edge_widths)
            edge_widths = 1 + 4 * (edge_widths - edge_widths.min()) / (
                edge_widths.max() - edge_widths.min() + 1e-8
            )
        else:
            edge_widths = 1

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax1)

        # Draw labels
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)

        ax1.set_title(f"{k}-hop Neighborhood of Node {node_idx}")
        ax1.axis("off")

        # Plot degree distribution in the subgraph
        degrees = [G.degree(node) for node in G.nodes()]
        ax2.hist(degrees, bins=20, edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Degree Distribution in {k}-hop Subgraph")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"K-hop Analysis: {len(subset)} nodes, {len(edges)} edges")
        plt.tight_layout()

        return fig

    def plot_receptive_field_growth(
        self, max_k: int = 5, sample_size: int = 50
    ) -> plt.Figure:
        """
        Plot how receptive field grows with k.

        Args:
            max_k: Maximum number of hops
            sample_size: Number of nodes to sample

        Returns:
            Matplotlib figure
        """
        # Sample nodes
        sample_nodes = np.random.choice(
            self.analyzer.num_nodes,
            min(sample_size, self.analyzer.num_nodes),
            replace=False,
        )

        # Analyze receptive field
        df = self.analyzer.analyze_receptive_field(max_k, sample_nodes)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Average number of neighbors vs k
        ax = axes[0, 0]
        stats = df.groupby("k")["num_neighbors"].agg(["mean", "std", "min", "max"])
        ax.plot(stats.index, stats["mean"], marker="o", label="Mean", linewidth=2)
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.3,
            label="±1 std",
        )
        ax.plot(stats.index, stats["min"], ":", label="Min", alpha=0.7)
        ax.plot(stats.index, stats["max"], ":", label="Max", alpha=0.7)
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Number of neighbors")
        ax.set_title("Receptive Field Size Growth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Coverage percentage vs k
        ax = axes[0, 1]
        coverage_stats = df.groupby("k")["coverage"].agg(["mean", "std", "min", "max"])
        ax.plot(
            coverage_stats.index,
            coverage_stats["mean"] * 100,
            marker="o",
            linewidth=2,
            color="green",
        )
        ax.fill_between(
            coverage_stats.index,
            (coverage_stats["mean"] - coverage_stats["std"]) * 100,
            (coverage_stats["mean"] + coverage_stats["std"]) * 100,
            alpha=0.3,
            color="green",
        )
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Graph coverage (%)")
        ax.set_title("Percentage of Graph Covered")
        ax.grid(True, alpha=0.3)

        # Plot 3: Distribution of neighbor counts for each k
        ax = axes[1, 0]
        k_values = df["k"].unique()
        positions = []
        data_to_plot = []
        for k in k_values:
            k_data = df[df["k"] == k]["num_neighbors"].values
            data_to_plot.append(k_data)
            positions.append(k)

        ax.boxplot(data_to_plot, positions=positions, widths=0.6)
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Number of neighbors")
        ax.set_title("Distribution of Receptive Field Sizes")
        ax.grid(True, alpha=0.3)

        # Plot 4: Heatmap of coverage by node and k
        ax = axes[1, 1]
        pivot = df.pivot_table(values="coverage", index="node_idx", columns="k")
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Node index (sampled)")
        ax.set_title("Coverage Heatmap")
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        plt.colorbar(im, ax=ax, label="Coverage")

        plt.suptitle(
            f"Receptive Field Analysis (Graph with {self.analyzer.num_nodes} nodes)"
        )
        plt.tight_layout()

        return fig

    def plot_minibatch_recommendations(
        self, max_k: int = 5, target_coverage: float = 0.8
    ) -> plt.Figure:
        """
        Plot recommendations for minibatch size based on receptive field analysis.

        Args:
            max_k: Maximum number of hops to consider
            target_coverage: Target graph coverage per minibatch

        Returns:
            Matplotlib figure with recommendations
        """
        # Compute statistics for each k
        stats = []
        for k in range(1, max_k + 1):
            stat = self.analyzer.compute_neighbor_statistics(k)
            stats.append(stat)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Recommended batch sizes for different k values
        ax = axes[0, 0]
        k_values = [s["k"] for s in stats]
        mean_coverages = [s["mean_coverage"] for s in stats]
        recommended_batch_sizes = [
            int(target_coverage / cov) if cov > 0 else self.analyzer.num_nodes
            for cov in mean_coverages
        ]

        ax.bar(k_values, recommended_batch_sizes, color="skyblue", edgecolor="navy")
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Recommended batch size")
        ax.set_title(f"Batch Size for {target_coverage * 100:.0f}% Coverage")
        ax.grid(True, alpha=0.3, axis="y")

        # Add text annotations
        for k, size in zip(k_values, recommended_batch_sizes, strict=False):
            ax.text(k, size + 0.5, str(size), ha="center", va="bottom")

        # Plot 2: Memory estimate (simplified)
        ax = axes[0, 1]
        mean_neighbors = [s["mean_neighbors"] for s in stats]
        memory_factor = np.array(mean_neighbors) * np.array(recommended_batch_sizes)

        ax.plot(
            k_values,
            memory_factor / self.analyzer.num_nodes,
            marker="o",
            linewidth=2,
            color="orange",
        )
        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Relative memory usage")
        ax.set_title("Memory Usage Estimate (relative to full graph)")
        ax.axhline(y=1.0, color="red", linestyle="--", label="Full graph", alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Percentile analysis
        ax = axes[1, 0]
        percentiles = ["25", "50", "75", "90", "95"]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(percentiles)))

        for i, p in enumerate(percentiles):
            values = [s["percentiles"][p] for s in stats]
            ax.plot(
                k_values,
                values,
                marker="o",
                label=f"{p}th percentile",
                color=colors[i],
                linewidth=2,
            )

        ax.set_xlabel("Number of hops (k)")
        ax.set_ylabel("Number of neighbors")
        ax.set_title("Percentile Analysis of Receptive Fields")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis("tight")
        ax.axis("off")

        # Create summary table
        table_data = []
        for stat in stats:
            k = stat["k"]
            row = [
                f"{k}",
                f"{stat['mean_neighbors']:.1f}",
                f"{stat['std_neighbors']:.1f}",
                f"{stat['mean_coverage'] * 100:.1f}%",
                f"{int(target_coverage / stat['mean_coverage']) if stat['mean_coverage'] > 0 else 'N/A'}",
            ]
            table_data.append(row)

        columns = [
            "K-hops",
            "Mean\nNeighbors",
            "Std\nNeighbors",
            "Mean\nCoverage",
            f"Batch Size\n({target_coverage * 100:.0f}% coverage)",
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.15, 0.2, 0.2, 0.2, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        plt.suptitle(
            f"Minibatch Tuning Recommendations\n"
            f"Graph: {self.analyzer.num_nodes} nodes, "
            f"{self.analyzer.edge_index.shape[1]} edges"
        )
        plt.tight_layout()

        return fig

    def create_interactive_khop_viz(
        self, node_idx: int, max_k: int = 3, node_labels: dict | None = None
    ) -> go.Figure:
        """
        Create interactive visualization of k-hop neighborhoods using Plotly.

        Args:
            node_idx: Center node index
            max_k: Maximum number of hops to visualize
            node_labels: Optional node labels

        Returns:
            Plotly figure
        """
        if not HAS_PLOTLY:
            raise ImportError(
                "plotly is required for interactive visualizations. Install with: pip install plotly"
            )

        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for graph layout. Install with: pip install networkx"
            )
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=max_k,
            subplot_titles=[f"{k}-hop neighbors" for k in range(1, max_k + 1)],
            specs=[[{"type": "scatter"} for _ in range(max_k)]],
        )

        colors = px.colors.qualitative.Set1

        for k in range(1, max_k + 1):
            # Get k-hop subgraph
            subset, sub_edge_index = self.analyzer.get_khop_neighbors(node_idx, k)

            # Create NetworkX graph for layout
            G = nx.Graph()
            for node in subset.numpy():
                G.add_node(int(node))

            edges = sub_edge_index.t().numpy()
            for src, dst in edges:
                if src in subset and dst in subset:
                    G.add_edge(int(src), int(dst))

            # Calculate layout
            pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

            # Prepare edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line={"width": 0.5, "color": "#888"},
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            )

            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                label = node_labels.get(node, str(node)) if node_labels else str(node)

                if node == node_idx:
                    node_colors.append("red")
                    node_text.append(f"Center: {label}")
                else:
                    try:
                        distance = nx.shortest_path_length(G, node_idx, node)
                        node_colors.append(colors[min(distance, len(colors) - 1)])
                        node_text.append(f"{label}<br>Distance: {distance}")
                    except Exception:
                        node_colors.append("gray")
                        node_text.append(f"{label}<br>Distance: N/A")

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=[str(n) for n in G.nodes()],
                hovertext=node_text,
                marker={
                    "color": node_colors,
                    "size": 10,
                    "line": {"width": 2, "color": "white"},
                },
                textposition="top center",
                textfont={"size": 8},
                showlegend=False,
            )

            # Add traces to subplot
            fig.add_trace(edge_trace, row=1, col=k)
            fig.add_trace(node_trace, row=1, col=k)

            # Update axes
            fig.update_xaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=1, col=k
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=1, col=k
            )

        # Update layout
        fig.update_layout(
            title=f"K-hop Neighborhoods of Node {node_idx}",
            height=500,
            showlegend=False,
            hovermode="closest",
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
        )

        return fig


def load_mobility_graph_from_nc(filepath: str) -> tuple[torch.Tensor, dict]:
    """
    Load mobility graph from NetCDF file.

    Args:
        filepath: Path to NetCDF file

    Returns:
        edge_index: Edge connectivity tensor
        metadata: Dictionary with additional information
    """
    filepath = Path(filepath)

    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    if not filepath.suffix == ".nc":
        raise ValueError(f"Expected NetCDF file (.nc), got: {filepath.suffix}")

    try:
        # Load NetCDF file with error handling
        ds = xr.open_dataset(str(filepath))
        print(f"Successfully opened NetCDF file: {filepath.name}")

        # Validate required variables and dimensions
        if "person_hours" not in ds.data_vars:
            available_vars = list(ds.data_vars.keys())
            raise ValueError(
                f"Expected 'person_hours' variable not found. Available variables: {available_vars}"
            )

        if "time" not in ds.dims:
            raise ValueError("Expected 'time' dimension not found in NetCDF file")

        if "home" not in ds.coords:
            raise ValueError("Expected 'home' coordinate not found in NetCDF file")

        print(f"NetCDF structure: {dict(ds.sizes)}")
        print(f"Available variables: {list(ds.data_vars.keys())}")

        # Get first time slice for structure analysis
        data = ds["person_hours"].isel(time=0).values

        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D data array for person_hours, got {data.ndim}D"
            )

        print(f"Mobility matrix shape: {data.shape}")

        # Create unified node mapping for both home and destination
        # First, get all unique municipality codes
        home_codes = ds.coords["home"].values
        dest_codes = ds.coords["destination"].values

        # Create unified node space
        all_codes = np.union1d(home_codes, dest_codes)
        code_to_node = {code: i for i, code in enumerate(all_codes)}
        node_to_code = dict(enumerate(all_codes))

        # Create mapping from original indices to unified node indices
        home_to_node = np.array([code_to_node[code] for code in home_codes])
        dest_to_node = np.array([code_to_node[code] for code in dest_codes])

        print(
            f"Unified node space: {len(all_codes)} nodes from {len(home_codes)} home + {len(dest_codes)} dest codes"
        )

        # Create edge index from non-zero flows using unified node indices
        edges = []
        edge_weights = []

        for i in range(data.shape[0]):  # home dimension
            for j in range(data.shape[1]):  # destination dimension
                if (
                    not np.isnan(data[i, j]) and data[i, j] > 0
                ):  # Only include valid, positive flows
                    src_node = home_to_node[i]  # Map home index to unified node index
                    dst_node = dest_to_node[j]  # Map dest index to unified node index
                    edges.append([src_node, dst_node])
                    edge_weights.append(data[i, j])

        if len(edges) == 0:
            raise ValueError(
                "No valid edges found in the mobility data (all flows are zero or NaN)"
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        print(f"Created {len(edges)} edges from mobility flows")

        # Create node labels using the unified mapping
        node_labels = {i: str(code) for i, code in enumerate(all_codes)}

        metadata = {
            "num_nodes": len(all_codes),  # Use unified node count
            "num_edges": edge_index.shape[1],
            "edge_weights": edge_weights,
            "node_labels": node_labels,
            "node_to_code": node_to_code,
            "code_to_node": code_to_node,
            "home_codes": home_codes.tolist(),
            "dest_codes": dest_codes.tolist(),
            "time_steps": len(ds.coords["time"]),
            "filename": filepath.name,
            "data_shape": data.shape,
            "time_range": (
                str(ds.coords["time"].values[0]),
                str(ds.coords["time"].values[-1]),
            ),
            "flow_stats": {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "nonzero_count": int(np.sum(data > 0)),
            },
        }

        ds.close()

        return edge_index, metadata

    except Exception as e:
        if "ds" in locals():
            try:
                ds.close()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to load mobility graph from {filepath}: {str(e)}"
        ) from e
