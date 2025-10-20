"""
Graph neural network models for epidemiological forecasting.
"""

from .aggregators import (
    AttentionAggregator,
    MaxPoolAggregator,
    MeanAggregator,
    create_aggregator,
)
from .dual_graph_forecaster import (
    DualGraphForecaster,
    SimpleDualGraphForecaster,
    create_dual_graph_forecaster,
)
from .dual_graph_sage import (
    DualGraphSAGE,
)
from .graphsage_od import (
    GraphSAGE_OD,
    TemporalGraphSAGE_OD,
    create_graphsage_model,
)
from .region_embedding import (
    RegionEmbedder,
    RegionGCN,
    SpatialRegionalizer,
    create_region_embedder,
)
from .region_losses import (
    CommunityOrientedLoss,
    FlowWeightedContrastiveLoss,
    SpatialAutocorrelationLoss,
    SpatialContiguityPrior,
    create_community_loss,
)

__all__ = [
    "GraphSAGE_OD",
    "TemporalGraphSAGE_OD",
    "create_graphsage_model",
    "DualGraphSAGE",
    "MeanAggregator",
    "AttentionAggregator",
    "MaxPoolAggregator",
    "create_aggregator",
    "DualGraphForecaster",
    "SimpleDualGraphForecaster",
    "create_dual_graph_forecaster",
    "RegionEmbedder",
    "RegionGCN",
    "SpatialRegionalizer",
    "create_region_embedder",
    "CommunityOrientedLoss",
    "FlowWeightedContrastiveLoss",
    "SpatialContiguityPrior",
    "SpatialAutocorrelationLoss",
    "create_community_loss",
]
