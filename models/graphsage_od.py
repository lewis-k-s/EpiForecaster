"""
GraphSAGE implementation for Origin-Destination mobility data.
"""

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Import our custom components
from .aggregators import create_aggregator

logger = logging.getLogger(__name__)


class GraphSAGE_OD(nn.Module):
    """
    GraphSAGE implementation specialized for Origin-Destination mobility data.

    This model implements the inductive learning framework from Hamilton et al. (2017)
    adapted for epidemiological forecasting with mobility flow data. It supports both
    homogeneous and heterogeneous graph structures.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        aggregator_type: str = "mean",
        dropout: float = 0.5,
        normalize: bool = True,
        residual: bool = True,
        edge_dim: Optional[int] = None,
        batch_norm: bool = True,
    ):
        """
        Initialize GraphSAGE for O-D data.

        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GraphSAGE layers
            aggregator_type: Type of aggregator ('mean', 'attention', 'max', 'lstm', 'hybrid')
            dropout: Dropout probability
            normalize: Whether to normalize embeddings
            residual: Whether to use residual connections
            edge_dim: Edge feature dimension (for attention aggregator)
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.normalize = normalize
        self.residual = residual
        self.edge_dim = edge_dim
        self.batch_norm = batch_norm

        # Build SAGE layers
        self.sage_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        logger.info("GraphSAGE_OD layer configuration:")
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            logger.info(
                f"  Layer {i}: {in_dim} -> {out_dim} (aggregator: {aggregator_type})"
            )

            # Create aggregator-specific layer
            if aggregator_type in ["mean", "max"]:
                # Use PyTorch Geometric's SAGEConv for standard aggregators
                layer = SAGEConv(
                    in_dim, out_dim, aggr=aggregator_type, normalize=normalize
                )
            else:
                # Use our custom aggregators
                layer = create_aggregator(
                    aggregator_type=aggregator_type,
                    input_dim=in_dim,
                    output_dim=out_dim,
                    edge_dim=edge_dim,
                    normalize=False,  # We'll handle normalization separately
                )

            self.sage_layers.append(layer)

            # Batch normalization (not for last layer)
            if batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Residual connection projection if needed
        if residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        for layer in self.sage_layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAGE.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Store input for residual connection
        x_input = x

        # Apply SAGE layers
        for i, layer in enumerate(self.sage_layers):
            # Apply layer
            if (
                hasattr(layer, "forward")
                and edge_attr is not None
                and self.aggregator_type != "mean"
            ):
                # Custom aggregators that support edge attributes
                x = layer(x, edge_index, edge_attr)
            else:
                # Standard PyTorch Geometric layers
                x = layer(x, edge_index)

            # Apply batch normalization (except last layer)
            if self.batch_norms is not None and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            # Apply activation and dropout (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        # Add residual connection
        if self.residual:
            if self.residual_proj is not None:
                x_input = self.residual_proj(x_input)
            x = x + x_input

        # Final normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x

    def encode_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode large graphs using subgraph sampling.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge attributes (optional)
            batch_size: Batch size for subgraph processing

        Returns:
            Node embeddings
        """
        if batch_size is None or x.size(0) <= batch_size:
            return self.forward(x, edge_index, edge_attr)

        # Process in mini-batches
        embeddings = []
        num_nodes = x.size(0)

        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)

            # Extract subgraph
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
            node_mask[start_idx:end_idx] = True

            # Get edges within subgraph and k-hop neighbors
            subgraph_data = self._extract_k_hop_subgraph(
                node_mask, edge_index, edge_attr, k=self.num_layers
            )

            # Encode subgraph
            sub_embeddings = self.forward(
                subgraph_data["x"],
                subgraph_data["edge_index"],
                subgraph_data["edge_attr"],
            )

            # Extract embeddings for target nodes
            target_embeddings = sub_embeddings[subgraph_data["target_mask"]]
            embeddings.append(target_embeddings)

        return torch.cat(embeddings, dim=0)

    def _extract_k_hop_subgraph(
        self,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        k: int,
    ) -> dict[str, torch.Tensor]:
        """
        Extract k-hop subgraph around target nodes.

        Args:
            node_mask: Boolean mask for target nodes
            edge_index: Full graph edge index
            edge_attr: Full graph edge attributes
            k: Number of hops

        Returns:
            Dictionary with subgraph data
        """
        device = edge_index.device
        num_nodes = node_mask.size(0)

        # Start with target nodes
        current_nodes = node_mask.clone()

        # Expand k hops
        for _ in range(k):
            # Find neighbors of current nodes
            edge_mask = current_nodes[edge_index[0]] | current_nodes[edge_index[1]]
            neighbor_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

            # Add all nodes connected to current nodes
            connected_edges = edge_index[:, edge_mask]
            neighbor_nodes[connected_edges.flatten()] = True

            # Update current nodes
            current_nodes = current_nodes | neighbor_nodes

        # Extract subgraph
        subgraph_nodes = current_nodes.nonzero().flatten()
        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_mapping[subgraph_nodes] = torch.arange(len(subgraph_nodes), device=device)

        # Extract edges within subgraph
        edge_mask = current_nodes[edge_index[0]] & current_nodes[edge_index[1]]
        sub_edge_index = edge_index[:, edge_mask]
        sub_edge_index = node_mapping[sub_edge_index]

        sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        # Extract node features
        sub_x = torch.zeros(len(subgraph_nodes), node_mask.size(0), device=device)
        # This is a simplified version - in practice, you'd extract actual node features

        # Create target mask for subgraph
        target_mask = node_mask[subgraph_nodes]

        return {
            "x": sub_x,
            "edge_index": sub_edge_index,
            "edge_attr": sub_edge_attr,
            "target_mask": target_mask,
            "node_mapping": node_mapping,
        }


class TemporalGraphSAGE_OD(nn.Module):
    """
    Temporal GraphSAGE that processes sequences of graphs for time-series forecasting.

    Combines spatial GraphSAGE encoding with temporal sequence modeling using LSTM
    to capture both spatial and temporal patterns in epidemiological data.
    """

    def __init__(
        self,
        spatial_encoder: GraphSAGE_OD,
        temporal_hidden_dim: int = 64,
        sequence_length: int = 7,
        temporal_layers: int = 1,
        bidirectional: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize temporal GraphSAGE.

        Args:
            spatial_encoder: Spatial graph encoder
            temporal_hidden_dim: Hidden dimension for temporal modeling
            sequence_length: Length of input sequences
            temporal_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            output_dim: Output dimension (defaults to spatial encoder output)
        """
        super().__init__()

        self.spatial_encoder = spatial_encoder
        self.temporal_hidden_dim = temporal_hidden_dim
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional

        # Determine spatial output dimension
        spatial_dim = spatial_encoder.output_dim
        self.output_dim = output_dim or spatial_dim

        # Temporal sequence encoder
        self.temporal_lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if temporal_layers > 1 else 0.0,
        )

        # Output projection
        lstm_output_dim = temporal_hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_dim, self.output_dim)

    def forward(self, graph_sequence: list[Any]) -> torch.Tensor:
        """
        Process temporal sequence of graphs.

        Args:
            graph_sequence: List of graph data objects in temporal order

        Returns:
            Temporal embeddings for the last time step
        """
        # Encode each graph spatially
        spatial_embeddings = []

        for graph_data in graph_sequence:
            # Homogeneous encoding
            spatial_emb = self.spatial_encoder(
                graph_data.x,
                graph_data.edge_index,
                getattr(graph_data, "edge_attr", None),
            )

            if spatial_emb is not None:
                spatial_embeddings.append(spatial_emb)

        # Stack into temporal sequence [num_nodes, sequence_length, spatial_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)

        # Apply temporal LSTM
        lstm_output, (hidden, cell) = self.temporal_lstm(temporal_sequence)

        # Use last output
        final_output = lstm_output[:, -1, :]

        # Project to output dimension
        output = self.output_projection(final_output)

        return output


def create_graphsage_model(
    config: dict[str, Any],
    input_dims: Union[int, dict[str, int]],
    node_types: Optional[list[str]] = None,
    edge_types: Optional[list[tuple[str, str, str]]] = None,
) -> nn.Module:
    """
    Factory function to create GraphSAGE models based on configuration.

    Args:
        config: Model configuration
        input_dims: Input dimensions (int for homogeneous, dict for heterogeneous)
        node_types: Node types for heterogeneous graphs
        edge_types: Edge types for heterogeneous graphs

    Returns:
        Configured GraphSAGE model
    """
    model_type = config.get("model_type", "homogeneous")

    if model_type == "homogeneous":
        if isinstance(input_dims, dict):
            input_dim = input_dims.get("region", 64)
        else:
            input_dim = input_dims

        return GraphSAGE_OD(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 128),
            output_dim=config.get("output_dim", 64),
            num_layers=config.get("num_layers", 2),
            aggregator_type=config.get("aggregator_type", "mean"),
            dropout=config.get("dropout", 0.5),
            normalize=config.get("normalize", True),
            residual=config.get("residual", True),
            edge_dim=config.get("edge_dim"),
            batch_norm=config.get("batch_norm", True),
        )

    elif model_type == "heterogeneous":
        raise ValueError("Heterogeneous GraphSAGE not implemented")

    elif model_type == "temporal":
        # Create base spatial encoder first
        spatial_config = config.get("spatial_encoder", {})
        spatial_encoder = create_graphsage_model(
            spatial_config, input_dims, node_types, edge_types
        )

        return TemporalGraphSAGE_OD(
            spatial_encoder=spatial_encoder,
            temporal_hidden_dim=config.get("temporal_hidden_dim", 64),
            sequence_length=config.get("sequence_length", 7),
            temporal_layers=config.get("temporal_layers", 1),
            bidirectional=config.get("bidirectional", False),
            output_dim=config.get("output_dim"),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage and testing

    # Test homogeneous GraphSAGE
    input_dim = 64
    model = GraphSAGE_OD(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=32,
        num_layers=2,
        aggregator_type="attention",
    )

    # Create test data
    num_nodes, num_edges = 100, 200
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)

    # Test forward pass
    embeddings = model(x, edge_index, edge_attr)
    print(f"GraphSAGE_OD output shape: {embeddings.shape}")
