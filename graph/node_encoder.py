"""
Inductive node feature encoding for epidemiological graphs.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, SAGEConv

logger = logging.getLogger(__name__)


class InductiveNodeEncoder(nn.Module):
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
        output_dim: Optional[int] = None,
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

    def encode_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode node features using neighbor sampling for large graphs.

        Args:
            x: Node features
            edge_index: Edge connectivity
            batch_size: Batch size for neighbor sampling

        Returns:
            Node embeddings
        """
        if batch_size is None or x.size(0) <= batch_size:
            return self.forward(x, edge_index)

        # Mini-batch processing for large graphs
        embeddings = []
        num_nodes = x.size(0)

        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)

            # Extract subgraph for this batch
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[start_idx:end_idx] = True

            # Get edges within this subgraph
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask]

            # Remap node indices
            node_mapping = torch.cumsum(node_mask, dim=0) - 1
            sub_edge_index = node_mapping[sub_edge_index]

            # Encode subgraph
            sub_x = x[node_mask]
            sub_embeddings = self.forward(sub_x, sub_edge_index)
            embeddings.append(sub_embeddings)

        return torch.cat(embeddings, dim=0)


class TemporalNodeEncoder(nn.Module):
    """
    Temporal node encoder that captures both spatial and temporal patterns.

    Combines spatial graph encoding with temporal sequence modeling for
    time-evolving epidemiological graphs.
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        temporal_hidden_dim: int = 64,
        sequence_length: int = 7,
        temporal_layers: int = 1,
        bidirectional: bool = False,
    ):
        """
        Initialize temporal node encoder.

        Args:
            spatial_encoder: Spatial node encoder (InductiveNodeEncoder)
            temporal_hidden_dim: Hidden dimension for temporal modeling
            sequence_length: Length of temporal sequences
            temporal_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.spatial_encoder = spatial_encoder
        self.temporal_hidden_dim = temporal_hidden_dim
        self.sequence_length = sequence_length

        # Determine spatial output dimension
        if hasattr(spatial_encoder, "output_dim"):
            spatial_dim = spatial_encoder.output_dim
        else:
            spatial_dim = spatial_encoder.hidden_dim

        # Temporal sequence encoder
        self.temporal_lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if temporal_layers > 1 else 0.0,
        )

        # Output dimension
        lstm_output_dim = temporal_hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_dim, spatial_dim)

    def forward(
        self, graph_sequence: list[HeteroData], node_type: str = "region"
    ) -> torch.Tensor:
        """
        Encode temporal sequence of graphs.

        Args:
            graph_sequence: List of HeteroData graphs in temporal order
            node_type: Node type to encode (for heterogeneous graphs)

        Returns:
            Temporal node embeddings
        """
        # Encode each graph spatially
        spatial_embeddings = []

        for graph in graph_sequence:
            spatial_emb = self.spatial_encoder(graph.x, graph.edge_index)

            if spatial_emb is not None:
                spatial_embeddings.append(spatial_emb)

        if len(spatial_embeddings) == 0:
            raise ValueError(f"No embeddings found for node type: {node_type}")

        # Stack spatial embeddings into temporal sequence
        # Shape: [num_nodes, sequence_length, spatial_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)

        # Apply temporal LSTM
        lstm_output, (hidden, cell) = self.temporal_lstm(temporal_sequence)

        # Use last output or mean of sequence
        temporal_embeddings = lstm_output[:, -1, :]  # Last time step

        # Project back to spatial dimension
        output_embeddings = self.output_projection(temporal_embeddings)

        return output_embeddings


def create_node_encoder(
    config: dict[str, Any],
    input_dims: dict[str, int],
) -> nn.Module:
    """
    Factory function to create appropriate node encoder based on configuration.

    Args:
        config: Configuration dictionary
        input_dims: Input dimensions (single int)

    Returns:
        Configured node encoder
    """
    encoder_type = config.get("encoder_type", "inductive")

    if encoder_type == "inductive":
        # Homogeneous inductive encoder
        if isinstance(input_dims, dict):
            input_dim = input_dims.get("region", 64)  # Default for region nodes
        else:
            input_dim = input_dims

        return InductiveNodeEncoder(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 128),
            output_dim=config.get("output_dim"),
            num_layers=config.get("num_layers", 2),
            aggregation=config.get("aggregation", "mean"),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "relu"),
            normalize=config.get("normalize", True),
            residual=config.get("residual", False),
        )

    elif encoder_type == "temporal":
        # Temporal encoder (requires base spatial encoder)
        base_config = config.get("spatial_encoder", {})
        base_encoder = create_node_encoder(base_config, input_dims)

        return TemporalNodeEncoder(
            spatial_encoder=base_encoder,
            temporal_hidden_dim=config.get("temporal_hidden_dim", 64),
            sequence_length=config.get("sequence_length", 7),
            temporal_layers=config.get("temporal_layers", 1),
            bidirectional=config.get("bidirectional", False),
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Example usage and testing

    # Test inductive encoder
    input_dim = 64
    encoder = InductiveNodeEncoder(
        input_dim=input_dim, hidden_dim=128, num_layers=2, aggregation="mean"
    )

    # Create dummy data
    num_nodes = 100
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))

    # Test encoding
    embeddings = encoder(x, edge_index)
    print(f"Inductive encoder output shape: {embeddings.shape}")

    print("Node encoders initialized successfully")
