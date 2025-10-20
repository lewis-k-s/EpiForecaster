"""
EpiForecaster: Epidemiological forecasting with neural networks and graph learning for mobility data

A PyTorch Geometric implementation for epidemiological forecasting using inductive
graph neural networks on origin-destination mobility data with demographic and case data integration.
"""

__version__ = "0.1.0"
__author__ = "Lewis Knox"
__description__ = "Graph Neural Network for Mobility-Based Epidemiological Forecasting"

# Make main modules easily accessible
from . import data, graph, models

__all__ = ["data", "graph", "models"]
