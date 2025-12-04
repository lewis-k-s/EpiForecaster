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

logger = logging.getLogger(__name__)


class MobilityGNNLayer(nn.Module):
    """
    Single layer of mobility-weighted graph neural network.

    Implements message passing with mobility flow weights for epidemiological
    case signal aggregation from origin regions to destination regions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        aggregator_type: str = "mean",
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
    ):
        """
        Initialize MobilityGNN layer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            aggregator_type: Type of neighborhood aggregation ('mean', 'attention', 'max')
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        # Create simple aggregator to avoid type inspection issues
        # For now, use a simple linear layer approach
        self.input_transform = nn.Linear(input_dim, output_dim)
        self.output_transform = nn.Linear(output_dim, output_dim)
        self.aggregator_type = aggregator_type

        # Linear transformation for self-features (for residual connection)
        if residual and input_dim != output_dim:
            self.self_transform = nn.Linear(input_dim, output_dim)
        else:
            self.self_transform = None

        # Layer normalization
        if layer_norm:
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        mobility_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of MobilityGNN layer.

        Args:
            node_features: Node features [num_nodes, input_dim]
            mobility_matrix: Mobility flow matrix [num_nodes, num_nodes]
                           mobility_matrix[j, i] = flow from origin j to destination i

        Returns:
            Updated node features [num_nodes, output_dim]
        """
        # Normalize mobility matrix by incoming flows per destination
        # This matches design doc specification: A_t[i,j] = M_t[j,i] / sum_k M_t[k,i]
        normalized_mobility = self._normalize_incoming_flows(mobility_matrix)

        # Create edge list and edge attributes from dense mobility matrix
        edge_index, edge_attr = self._mobility_to_edges(normalized_mobility)

        # Skip aggregation if no edges
        if edge_index.size(1) == 0:
            output = torch.zeros(
                node_features.size(0),
                self.output_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            if self.self_transform is not None:
                output = output + self.self_transform(node_features)
            return output

        # Simple message passing without complex aggregation for now
        if edge_index.size(1) == 0:
            # No edges, just transform self features
            aggregated = self.input_transform(node_features)
        else:
            # Transform node features
            transformed_features = self.input_transform(node_features)

            # Simple mean aggregation (placeholder for more sophisticated aggregation)
            aggregated = torch.zeros_like(transformed_features)

            # For each destination node, aggregate from incoming edges
            num_nodes = node_features.size(0)
            for i in range(num_nodes):
                # Find edges going to node i (edge_index[1, :] == i)
                dest_mask = edge_index[1, :] == i
                if dest_mask.any():
                    # Get source nodes for these edges
                    src_nodes = edge_index[0, dest_mask]
                    # Get edge weights
                    edge_weights = (
                        edge_attr[dest_mask].squeeze(-1)
                        if edge_attr is not None
                        else torch.ones_like(src_nodes, dtype=torch.float)
                    )

                    # Weighted aggregation
                    weighted_sum = torch.sum(
                        transformed_features[src_nodes] * edge_weights.unsqueeze(-1),
                        dim=0,
                    )
                    weight_sum = torch.sum(edge_weights)

                    aggregated[i] = weighted_sum / (weight_sum + 1e-8)
                else:
                    aggregated[i] = transformed_features[i]

            # Apply output transformation
            aggregated = self.output_transform(aggregated)

        # Apply dropout
        aggregated = self.dropout_layer(aggregated)

        # Residual connection
        if self.residual:
            if self.self_transform is not None:
                residual = self.self_transform(node_features)
            else:
                residual = node_features
            output = aggregated + residual
        else:
            output = aggregated

        # Layer normalization
        if self.norm is not None:
            output = self.norm(output)

        return output

    def _normalize_incoming_flows(
        self, mobility_matrix: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Normalize mobility matrix by incoming flows per destination.

        Implements design doc specification:
        A_t[i,j] = M_t[j,i] / sum_k M_t[k,i]

        Args:
            mobility_matrix: Mobility matrix [num_nodes, num_nodes]
            eps: Small constant to avoid division by zero

        Returns:
            Normalized mobility matrix [num_nodes, num_nodes]
        """
        # Sum over origins (dim=0) for each destination
        incoming_sums = torch.sum(
            mobility_matrix, dim=0, keepdim=True
        )  # [1, num_nodes]
        normalized = mobility_matrix / (incoming_sums + eps)
        return normalized

    def _mobility_to_edges(
        self, mobility_matrix: torch.Tensor, threshold: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dense mobility matrix to edge format for PyTorch Geometric.

        Args:
            mobility_matrix: Normalized mobility matrix [num_nodes, num_nodes]
            threshold: Minimum flow weight to create edge

        Returns:
            edge_index: [2, num_edges] tensor of edge connections
            edge_attr: [num_edges, 1] tensor of edge weights
        """
        num_nodes = mobility_matrix.size(0)

        # Find edges with flow above threshold
        edge_mask = mobility_matrix > threshold

        # Get edge indices
        origin_indices, dest_indices = torch.where(edge_mask)

        # Create edge_index in PyG format: [2, num_edges]
        edge_index = torch.stack([origin_indices, dest_indices], dim=0)

        # Edge attributes are the flow weights
        edge_attr = mobility_matrix[edge_mask].unsqueeze(-1)  # [num_edges, 1]

        return edge_index, edge_attr


class MobilityGNN(nn.Module):
    """
    Mobility-based Graph Neural Network for per-time-step processing.

    Implements the non-recurrent GNN component from the design document.
    Processes case and biomarker signals using mobility-weighted aggregation
    independently at each time step.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        aggregator_type: str = "mean",
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize MobilityGNN.

        Args:
            in_dim: Input feature dimension (cases + biomarkers + optional embeddings)
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of GNN layers
            aggregator_type: Type of neighborhood aggregation
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to apply layer normalization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.activation = activation

        # Build layers
        self.layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.layers.append(
            MobilityGNNLayer(
                input_dim=in_dim,
                output_dim=hidden_dim,
                aggregator_type=aggregator_type,
                dropout=dropout,
                residual=residual,
                layer_norm=layer_norm,
            )
        )

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(
                MobilityGNNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    aggregator_type=aggregator_type,
                    dropout=dropout,
                    residual=residual,
                    layer_norm=layer_norm,
                )
            )

        # Last layer: hidden_dim -> out_dim
        if num_layers > 1:
            self.layers.append(
                MobilityGNNLayer(
                    input_dim=hidden_dim,
                    output_dim=out_dim,
                    aggregator_type=aggregator_type,
                    dropout=dropout,
                    residual=residual,
                    layer_norm=layer_norm,
                )
            )

        # Final activation function
        if activation == "relu":
            self.final_activation = F.relu
        elif activation == "gelu":
            self.final_activation = F.gelu
        elif activation == "tanh":
            self.final_activation = torch.tanh
        else:
            self.final_activation = F.relu

        # Log initialization
        logger.info(
            f"Initialized MobilityGNN: {in_dim}->{hidden_dim}->{out_dim}, "
            f"layers={num_layers}, agg={aggregator_type}"
        )

    def forward(
        self,
        node_features_t: torch.Tensor,
        mobility_matrix_t: torch.Tensor,
        region_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of MobilityGNN for a single time step.

        Args:
            node_features_t: Node features at time t [num_nodes, in_dim]
                Should contain cases, biomarkers, and optionally region embeddings
            mobility_matrix_t: Mobility flows at time t [num_nodes, num_nodes]
                mobility_matrix_t[j, i] = flow from origin j to destination i
            region_embeddings: Optional static region embeddings [num_nodes, embed_dim]
                If provided, will be concatenated to node_features_t

        Returns:
            Mobility-enhanced node embeddings [num_nodes, out_dim]
        """
        # Add static region embeddings if provided
        if region_embeddings is not None:
            # Concatenate region embeddings to node features
            node_features_t = torch.cat([node_features_t, region_embeddings], dim=1)

        # Validate input dimensions
        expected_in_dim = self.in_dim
        if region_embeddings is not None:
            expected_in_dim -= region_embeddings.size(1)

        if node_features_t.size(1) != expected_in_dim:
            raise ValueError(
                f"Expected input dim {expected_in_dim}, got {node_features_t.size(1)}. "
                f"If using region_embeddings, they should be concatenated to node_features_t."
            )

        # Apply GNN layers
        h = node_features_t

        for i, layer in enumerate(self.layers):
            h = layer(h, mobility_matrix_t)

            # Apply activation (except for final layer)
            if i < len(self.layers) - 1:
                h = self.final_activation(h)

        # Final activation
        h = self.final_activation(h)

        return h

    def forward_batch(
        self,
        node_features_batch: torch.Tensor,  # [batch_size, num_nodes, in_dim]
        mobility_matrices_batch: torch.Tensor,  # [batch_size, num_nodes, num_nodes]
        region_embeddings: torch.Tensor | None = None,  # [num_nodes, embed_dim]
    ) -> torch.Tensor:
        """
        Forward pass for batch processing of multiple time steps/samples.

        Args:
            node_features_batch: Batch of node features [batch_size, num_nodes, in_dim]
            mobility_matrices_batch: Batch of mobility matrices [batch_size, num_nodes, num_nodes]
            region_embeddings: Optional static region embeddings [num_nodes, embed_dim]

        Returns:
            Batch of mobility-enhanced embeddings [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = node_features_batch.shape

        # Process each batch element (can be optimized with proper batching)
        outputs = []

        for b in range(batch_size):
            output_b = self.forward(
                node_features_batch[b],
                mobility_matrices_batch[b],
                region_embeddings,
            )
            outputs.append(output_b)

        return torch.stack(outputs, dim=0)

    def get_output_dimension(self) -> int:
        """Get the output dimension of the MobilityGNN."""
        return self.out_dim


def create_mobility_gnn(
    in_dim: int,
    hidden_dim: int = 64,
    out_dim: int = 64,
    num_layers: int = 2,
    aggregator_type: str = "mean",
    dropout: float = 0.1,
    **kwargs,
) -> MobilityGNN:
    """
    Factory function to create MobilityGNN with common configurations.

    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output embedding dimension
        num_layers: Number of GNN layers
        aggregator_type: Type of neighborhood aggregation
        dropout: Dropout probability
        **kwargs: Additional arguments for MobilityGNN

    Returns:
        Configured MobilityGNN instance
    """
    return MobilityGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        aggregator_type=aggregator_type,
        dropout=dropout,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)

    # Configuration
    num_nodes = 10
    in_dim = 5  # cases + biomarkers
    hidden_dim = 16
    out_dim = 8
    batch_size = 3

    # Create test data
    node_features = torch.randn(batch_size, num_nodes, in_dim).abs()  # Positive values
    mobility_matrices = torch.abs(torch.randn(batch_size, num_nodes, num_nodes)) * 100

    # Remove self-loops
    for b in range(batch_size):
        mobility_matrices[b].fill_diagonal_(0)

    # Create MobilityGNN
    mobility_gnn = create_mobility_gnn(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        aggregator_type="mean",
    )

    # Test forward pass
    print(f"Input node features shape: {node_features.shape}")
    print(f"Input mobility matrices shape: {mobility_matrices.shape}")

    # Process batch
    output_embeddings = mobility_gnn.forward_batch(node_features, mobility_matrices)

    print(f"Output embeddings shape: {output_embeddings.shape}")
    print(f"Output dimension: {mobility_gnn.get_output_dimension()}")

    # Test with region embeddings
    region_embeddings = torch.randn(num_nodes, 4)
    output_with_regions = mobility_gnn.forward_batch(
        node_features, mobility_matrices, region_embeddings
    )

    print(f"Output with region embeddings shape: {output_with_regions.shape}")

    print("MobilityGNN test completed successfully!")
