"""
Graph neural network models for epidemiological forecasting.
"""

from .aggregators import (
    AttentionAggregator,
    MaxPoolAggregator,
    MeanAggregator,
    create_aggregator,
)
from .epiforecaster import EpiForecaster
from .forecaster_head import ForecasterHead
from .mobility_gnn import MobilityGNN
from .observation_heads import (
    ClinicalObservationHead,
    CompositeObservationLoss,
    DelayKernel,
    MAELoss,
    MSELoss,
    SheddingConvolution,
    SMAPELoss,
    UnscaledMSELoss,
    WastewaterObservationHead,
)
from .sir_rollforward import SIRRollForward

__all__ = [
    "MeanAggregator",
    "AttentionAggregator",
    "MaxPoolAggregator",
    "create_aggregator",
    "EpiForecaster",
    "ForecasterHead",
    "MobilityGNN",
    "SIRRollForward",
    "DelayKernel",
    "SheddingConvolution",
    "ClinicalObservationHead",
    "WastewaterObservationHead",
    # Loss classes
    "MSELoss",
    "MAELoss",
    "SMAPELoss",
    "UnscaledMSELoss",
    "CompositeObservationLoss",
]
