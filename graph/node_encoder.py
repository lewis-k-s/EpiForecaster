"""
Inductive node feature encoding for epidemiological graphs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

logger = logging.getLogger(__name__)


class RegionEncoderArtifact(TypedDict, total=False):
    """Serialized payload emitted by :class:`RegionEmbedderTrainer`.

    The trainer stores both the frozen encoder weights and the already-computed
    embeddings so downstream components can either reuse the tensor directly or
    regenerate embeddings for new graphs.
    """

    embeddings: torch.Tensor
    region_ids: list[str]
    config: dict[str, Any]
    encoder_state_dict: dict[str, torch.Tensor]
    feature_dim: int


class Region2Vec(nn.Module):
    """
    Inductive node encoder that learns to aggregate features from local neighborhoods
    rather than learning fixed embeddings for specific nodes.

    Based on GraphSAGE methodology (Hamilton et al., 2017) adapted for epidemiological
    forecasting with graphs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int | None = None,
        num_layers: int = 2,
        aggregation: str = "mean",
        dropout: float = 0.5,
        activation: str = "relu",
        normalize: bool = True,
        residual: bool = False,
    ):
        """
        Initialize inductive node encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (defaults to hidden_dim)
            num_layers: Number of GraphSAGE layers
            aggregation: Aggregation method ('mean', 'max', 'lstm', 'attention')
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            normalize: Whether to apply L2 normalization
            residual: Whether to use residual connections
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.dropout = dropout
        self.normalize = normalize
        self.residual = residual

        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = self.output_dim if i == num_layers - 1 else hidden_dim

            # Choose convolution type based on aggregation
            if aggregation == "attention":
                conv = GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
            else:
                conv = SAGEConv(in_dim, out_dim, aggr=aggregation, normalize=normalize)

            self.convs.append(conv)

            # Batch normalization (not for last layer)
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        if residual and input_dim != self.output_dim:
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through inductive encoder.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Store original input for residual connection
        residual = x

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch normalization (except last layer)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            # Apply activation (except last layer)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        # Add residual connection
        if self.residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        # Final normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_weights(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> tuple[Region2Vec, RegionEncoderArtifact]:
        """
        Restore a pretrained encoder from a Region2Vec artifact.

        Returns the instantiated encoder (set to eval mode) and the raw artifact
        dictionary so callers can access the stored embeddings/region_ids.
        """

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Region encoder checkpoint not found: {checkpoint_path}"
            )

        artifact = torch.load(checkpoint_path, map_location=map_location or "cpu")
        if not isinstance(artifact, dict):
            raise ValueError(
                "Region encoder checkpoint must contain a dictionary payload. "
                f"Got type={type(artifact)!r}"
            )
        region_artifact: RegionEncoderArtifact = cast(RegionEncoderArtifact, artifact)

        feature_dim = region_artifact.get("feature_dim")
        if feature_dim is None:
            raise ValueError(
                "Region encoder artifact missing 'feature_dim'; unable to rebuild encoder."
            )

        encoder_cfg = (region_artifact.get("config") or {}).get("encoder", {})
        embedding_dim = encoder_cfg.get("embedding_dim")
        if embedding_dim is None:
            embedding_dim = encoder_cfg.get("hidden_dim", 128)

        encoder = cls(
            input_dim=int(feature_dim),
            hidden_dim=int(encoder_cfg.get("hidden_dim", 128)),
            output_dim=int(embedding_dim),
            num_layers=int(encoder_cfg.get("num_layers", 2)),
            aggregation=str(encoder_cfg.get("aggregation", "mean")),
            dropout=float(encoder_cfg.get("dropout", 0.5)),
            activation=str(encoder_cfg.get("activation", "relu")),
            normalize=bool(encoder_cfg.get("normalize", True)),
            residual=bool(encoder_cfg.get("residual", False)),
        )

        state_dict = region_artifact.get("encoder_state_dict")
        if state_dict is None:
            raise ValueError(
                "Region encoder artifact missing 'encoder_state_dict'; cannot load weights."
            )
        encoder.load_state_dict(state_dict)
        encoder.eval()
        return encoder, region_artifact
