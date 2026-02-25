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
from .transformer_backbone import TransformerBackbone
from .mobility_gnn import MobilityDenseEncoder
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
    "TransformerBackbone",
    "MobilityDenseEncoder",
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
