"""
Mobility-based Graph Neural Network for epidemiological forecasting.

This module implements the per-time-step graph neural network component from the
design document. It processes mobility-weighted case signals independently at
each time step, creating enhanced region representations for the temporal forecaster.

Key Design Principles:
1. Non-recurrent: Each time step is processed independently
2. Mobility-weighted aggregation: Uses incoming flow normalization
3. Configurable inputs: cases ± biomarkers ± region embeddings
4. Efficient batch processing: Handles multiple regions simultaneously

Architecture:
- Input: node_features_t [num_nodes, in_dim], mobility_matrix_t [num_nodes, num_nodes]
- Processing: Mobility-weighted message passing using existing aggregators
- Output: mobility_embeddings_t [num_nodes, out_dim]
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

logger = logging.getLogger(__name__)


class MobilityPyGEncoder(nn.Module):
    """
    Lightweight PyG-based encoder that can switch between GCN and GAT.

    Supports variable depth, residual connections, and normalization.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 2,
        module_type: str = "gcn",
        dropout: float = 0.1,
        heads: int = 1,
    ):
        super().__init__()

        self.module_type = module_type
        self.depth = depth
        self.dropout_val = dropout
        self.activation = nn.ReLU()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        if depth == 1:
            self.layers.append(self._make_layer(in_dim, out_dim, heads, module_type))
        else:
            # Input layer
            self.layers.append(self._make_layer(in_dim, hidden_dim, heads, module_type))
            self.norms.append(nn.LayerNorm(hidden_dim))

            if in_dim != hidden_dim:
                self.skips.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.skips.append(nn.Identity())

            # Hidden layers
            for _ in range(depth - 2):
                self.layers.append(
                    self._make_layer(hidden_dim, hidden_dim, heads, module_type)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.skips.append(nn.Identity())

            # Output layer
            self.layers.append(
                self._make_layer(hidden_dim, out_dim, heads, module_type)
            )

        logger.info(
            f"Initialized MobilityPyGEncoder: {in_dim}->{hidden_dim}->{out_dim}, "
            f"module_type={module_type}, layers={depth}, heads={heads}"
        )
        self._initialize_skip_layers()

    def _make_layer(self, in_c, out_c, heads, module_type):
        if module_type == "gcn":
            return GCNConv(in_c, out_c, add_self_loops=True)
        elif module_type == "gat":
            return GATConv(
                in_c,
                out_c,
                heads=heads,
                concat=False,
                add_self_loops=True,
                dropout=self.dropout_val,
            )
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")

    def _initialize_skip_layers(self) -> None:
        for layer in self.skips:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights aligned to edge_index
        """
        h = x

        for i, layer in enumerate(self.layers):
            h_in = h

            h = layer(h, edge_index, edge_weight)

            if i < len(self.layers) - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout_val, training=self.training)

                if i < len(self.skips):
                    skip = self.skips[i](h_in)
                    h = h + skip

        return h
