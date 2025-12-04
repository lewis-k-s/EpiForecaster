"""
Inductive node feature encoding for epidemiological graphs.
"""

import logging
from pathlib import Path
from typing import Any

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

    def encode_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int | None = None,
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
