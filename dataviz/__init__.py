"""
Data visualization module for mobility graph analysis.

This module provides tools for visualizing and analyzing k-hop neighborhoods
in mobility graphs to optimize minibatch sizes for GraphSAGE training.

Main components:
- KHopNeighborAnalyzer: Analyze receptive fields and neighborhood growth
- KHopVisualizer: Create visualizations of k-hop subgraphs

Example usage:
    from dataviz.khop_neighbors import KHopNeighborAnalyzer, load_mobility_graph_from_nc

    # Load graph
    edge_index, metadata = load_mobility_graph_from_nc('data/mobility.nc')

    # Analyze k-hop neighborhoods
    analyzer = KHopNeighborAnalyzer(edge_index, metadata['num_nodes'])
"""

from .khop_neighbors import (
    KHopNeighborAnalyzer,
    KHopVisualizer,
    load_mobility_graph_from_nc,
)

__all__ = [
    "KHopNeighborAnalyzer",
    "KHopVisualizer",
    "load_mobility_graph_from_nc",
]
