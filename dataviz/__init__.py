"""Data visualization package."""

from importlib import import_module

__all__ = [
    "KHopNeighborAnalyzer",
    "KHopVisualizer",
    "load_mobility_graph_from_nc",
]


def __getattr__(name: str):
    """Lazy export to avoid importing heavy modules at package import time."""
    if name in __all__:
        module = import_module("dataviz.khop_neighbors")
        return getattr(module, name)
    raise AttributeError(f"module 'dataviz' has no attribute {name!r}")
