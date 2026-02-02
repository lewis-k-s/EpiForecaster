"""
EpiForecaster: Epidemiological forecasting with neural networks and graph learning for mobility data

A PyTorch Geometric implementation for epidemiological forecasting using inductive
graph neural networks on origin-destination mobility data with demographic and case data integration.
"""

__version__ = "0.1.0"
__author__ = "Lewis Knox"
__description__ = "Graph Neural Network for Mobility-Based Epidemiological Forecasting"

# Import with absolute imports to avoid issues when run directly
# These will be available when the package is properly installed

try:
    from models import sir_rollforward
    from models.sir_rollforward import SIRRollForward

    __all__ = ["sir_rollforward", "SIRRollForward"]
except ImportError:
    # When run directly without package installation
    __all__ = []
