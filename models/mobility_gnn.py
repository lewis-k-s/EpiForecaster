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
- Input: node_features_t [num_graphs, num_nodes, in_dim], adj [num_graphs, num_nodes, num_nodes]
- Processing: Mobility-weighted message passing using dense GNN layers
- Output: mobility_embeddings_t [num_graphs, num_nodes, out_dim]
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import DenseGATConv, DenseGCNConv

logger = logging.getLogger(__name__)


class MobilityDenseEncoder(nn.Module):
    """Dense adjacency mobility encoder for small/moderately dense graphs."""

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
            self.layers.append(self._make_layer(in_dim, hidden_dim, heads, module_type))
            self.norms.append(nn.LayerNorm(hidden_dim))

            if in_dim != hidden_dim:
                self.skips.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.skips.append(nn.Identity())

            for _ in range(depth - 2):
                self.layers.append(
                    self._make_layer(hidden_dim, hidden_dim, heads, module_type)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.skips.append(nn.Identity())

            self.layers.append(
                self._make_layer(hidden_dim, out_dim, heads, module_type)
            )

        logger.info(
            "Initialized MobilityDenseEncoder: %d->%d->%d, module_type=%s, layers=%d, heads=%d",
            in_dim,
            hidden_dim,
            out_dim,
            module_type,
            depth,
            heads,
        )
        self._initialize_skip_layers()

    def _make_layer(self, in_c, out_c, heads, module_type):
        if module_type == "gcn":
            return DenseGCNConv(in_c, out_c)
        if module_type == "gat":
            return DenseGATConv(in_c, out_c, heads=heads, concat=False)
        raise ValueError(f"Unsupported module_type: {module_type}")

    def _initialize_skip_layers(self) -> None:
        for layer in self.skips:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_graphs, num_nodes, in_dim]
            adj: Dense adjacency [num_adj, num_nodes, num_nodes], with
                self-loops already present. num_adj is num_graphs, or a divisor
                of it when graphs share adjacency: graph g uses adj[g % num_adj]
                (all samples of a sample-major [B*T] batch share one window).
            target_index: Optional local target node per graph [num_graphs].
                When set, the output is [num_graphs, out_dim] for these nodes.
        """
        num_graphs, num_adj = x.size(0), adj.size(0)
        if num_graphs % num_adj:
            raise ValueError(
                f"adj has {num_adj} matrices; it must divide num_graphs={num_graphs}"
            )
        graph_ids = torch.arange(num_graphs, device=x.device)
        adj_ids = graph_ids % num_adj
        if self.module_type != "gcn" and num_adj != num_graphs:
            adj = adj[adj_ids]
        # All GCN layers use the same normalized adjacency (no added loops, no
        # mask), so normalize it once. DenseGCNConv would make one copy per layer.
        adj_norm = gcn_normalize_adjacency(adj) if self.module_type == "gcn" else None
        last = len(self.layers) - 1
        h = x

        for i, layer in enumerate(self.layers):
            h_in = h
            if adj_norm is None:
                h = layer(h, adj, add_loop=False)
            else:
                h = layer.lin(h)
                if i == last and target_index is not None:
                    # Only the target rows of the last layer are used.
                    rows = adj_norm[adj_ids, target_index].unsqueeze(1)
                    h = torch.matmul(rows, h).squeeze(1)
                else:
                    h = _propagate(adj_norm, h)
                if layer.bias is not None:
                    h = h + layer.bias

            if i < len(self.layers) - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout_val, training=self.training)

            if i < len(self.skips):
                skip = self.skips[i](h_in)
                h = h + skip

        if target_index is not None and h.dim() == 3:
            h = h[graph_ids, target_index]
        return h


def _propagate(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """adj[g % A] @ h[g] for adj [A, N, N] and h [G, N, D]."""
    num_adj = adj.size(0)
    if num_adj == h.size(0):
        return torch.matmul(adj, h)
    # Fold the samples that share each matrix into the feature dim, so the
    # shared adjacency is never expanded to [G, N, N].
    shared = rearrange(h, "(s a) n d -> a n (s d)", a=num_adj)
    out = torch.matmul(adj, shared)
    return rearrange(out, "a n (s d) -> (s a) n d", d=h.size(-1))


def gcn_normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric GCN normalization D^-1/2 A D^-1/2, as in DenseGCNConv."""
    deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
    return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
