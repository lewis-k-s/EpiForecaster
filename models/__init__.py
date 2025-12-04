"""
Graph neural network models for epidemiological forecasting.
"""

from .aggregators import (
    AttentionAggregator,
    MaxPoolAggregator,
    MeanAggregator,
    create_aggregator,
)
from .epiforecaster import (
    EpiForecaster,
    create_epiforecaster_variant,
)
from .forecaster_head import (
    ForecasterHead,
    create_forecaster_head,
)

# Note: dual_graph_forecaster, dual_graph_sage, and graphsage_od were deleted
# These imports are removed to fix broken dependencies
from .mobility_gnn import (
    MobilityGNN,
    create_mobility_gnn,
)

__all__ = [
    "MeanAggregator",
    "AttentionAggregator",
    "MaxPoolAggregator",
    "create_aggregator",
    # New three-layer architecture components
    "EpiForecaster",
    "create_epiforecaster_variant",
    "ForecasterHead",
    "create_forecaster_head",
    "MobilityGNN",
    "create_mobility_gnn",
]
