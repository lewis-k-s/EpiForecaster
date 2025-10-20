"""
Neighborhood aggregation functions for epidemiological GraphSAGE.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.utils import softmax

logger = logging.getLogger(__name__)


class MeanAggregator(MessagePassing):
    """
    Mean aggregator for GraphSAGE - baseline uniform importance for all neighbors.

    Computes simple average of neighbor features, treating all connections equally.
    Good baseline for mobility data where all flows have similar importance.
    """

    def __init__(self, input_dim: int, output_dim: int, normalize: bool = True):
        """
        Initialize mean aggregator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            normalize: Whether to normalize output
        """
        super().__init__(aggr="mean")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Linear transformation for neighbor features
        self.neighbor_transform = nn.Linear(input_dim, output_dim)

        # Linear transformation for self features
        self.self_transform = nn.Linear(input_dim, output_dim)

        # Final combination layer
        self.combine = nn.Linear(2 * output_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.neighbor_transform.weight)
        nn.init.xavier_uniform_(self.self_transform.weight)
        nn.init.xavier_uniform_(self.combine.weight)

        nn.init.zeros_(self.neighbor_transform.bias)
        nn.init.zeros_(self.self_transform.bias)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of mean aggregator.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Aggregated node features [num_nodes, output_dim]
        """
        # Transform self features
        self_features = self.self_transform(x)

        # Aggregate neighbor features
        neighbor_features = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Combine self and neighbor features
        combined = torch.cat([self_features, neighbor_features], dim=1)
        output = self.combine(combined)

        # Apply activation and normalization
        output = F.relu(output)

        if self.normalize:
            output = F.normalize(output, p=2, dim=1)

        return output

    def message(
        self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create messages from neighbors.

        Args:
            x_j: Neighbor node features [num_edges, input_dim]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Messages [num_edges, output_dim]
        """
        # Transform neighbor features
        messages = self.neighbor_transform(x_j)

        # Edge attributes can be used to weight messages, but for mean aggregator
        # we keep it simple and just return transformed features
        return messages


class AttentionAggregator(MessagePassing):
    """
    Attention-based aggregator that weights neighbors by flow volume and demographic similarity.

    Uses multi-head attention to learn importance weights for different neighbors,
    particularly useful for mobility data where flow volumes vary significantly.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize attention aggregator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            edge_dim: Edge attribute dimension (optional)
            num_heads: Number of attention heads
            dropout: Dropout probability
            normalize: Whether to normalize output
        """
        super().__init__(aggr="add")  # Use 'add' since we'll handle weighting manually

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.normalize = normalize

        # Ensure output_dim is divisible by num_heads
        assert output_dim % num_heads == 0
        self.head_dim = output_dim // num_heads

        # Query, Key, Value transformations
        self.query_transform = nn.Linear(input_dim, output_dim, bias=False)
        self.key_transform = nn.Linear(input_dim, output_dim, bias=False)
        self.value_transform = nn.Linear(input_dim, output_dim, bias=False)

        # Edge attribute integration (if provided)
        if edge_dim is not None:
            self.edge_transform = nn.Linear(edge_dim, output_dim, bias=False)

        # Self feature transformation
        self.self_transform = nn.Linear(input_dim, output_dim)

        # Final combination
        self.combine = nn.Linear(2 * output_dim, output_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Log parameter counts for debugging
        total_params = sum(p.numel() for p in self.parameters())
        logger.debug(f"AttentionAggregator ({input_dim}->{output_dim}) parameters:")
        logger.debug(f"  query_transform: {self.query_transform.weight.numel()}")
        logger.debug(f"  key_transform: {self.key_transform.weight.numel()}")
        logger.debug(f"  value_transform: {self.value_transform.weight.numel()}")
        logger.debug(
            f"  self_transform: {self.self_transform.weight.numel() + self.self_transform.bias.numel()}"
        )
        logger.debug(
            f"  combine: {self.combine.weight.numel() + self.combine.bias.numel()}"
        )
        logger.debug(f"  Total: {total_params}")

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters using scaled initialization."""
        # Use scaled initialization for attention layers
        for module in [self.query_transform, self.key_transform, self.value_transform]:
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))

        if hasattr(self, "edge_transform"):
            nn.init.xavier_uniform_(self.edge_transform.weight)

        nn.init.xavier_uniform_(self.self_transform.weight)
        nn.init.xavier_uniform_(self.combine.weight)

        nn.init.zeros_(self.self_transform.bias)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of attention aggregator.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Aggregated node features [num_nodes, output_dim]
        """
        # Transform self features
        self_features = self.self_transform(x)

        # Aggregate neighbor features with attention
        neighbor_features = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Combine self and neighbor features
        combined = torch.cat([self_features, neighbor_features], dim=1)
        output = self.combine(combined)

        # Apply activation and normalization
        output = F.relu(output)
        output = self.dropout_layer(output)

        if self.normalize:
            output = F.normalize(output, p=2, dim=1)

        return output

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index_i: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create attention-weighted messages.

        Args:
            x_i: Target node features [num_edges, input_dim]
            x_j: Source node features [num_edges, input_dim]
            edge_index_i: Target node indices [num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Attention-weighted messages [num_edges, output_dim]
        """
        # Transform features to query, key, value
        query = self.query_transform(x_i)  # [num_edges, output_dim]
        key = self.key_transform(x_j)  # [num_edges, output_dim]
        value = self.value_transform(x_j)  # [num_edges, output_dim]

        # Reshape for multi-head attention
        query = query.view(
            -1, self.num_heads, self.head_dim
        )  # [num_edges, num_heads, head_dim]
        key = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores
        attention_scores = (query * key).sum(dim=-1) / math.sqrt(
            self.head_dim
        )  # [num_edges, num_heads]

        # Incorporate edge attributes into attention if available
        if edge_attr is not None and hasattr(self, "edge_transform"):
            edge_contribution = self.edge_transform(
                edge_attr
            )  # [num_edges, output_dim]
            edge_contribution = edge_contribution.view(
                -1, self.num_heads, self.head_dim
            )

            # Add edge contribution to attention scores
            edge_scores = (query * edge_contribution).sum(dim=-1) / math.sqrt(
                self.head_dim
            )
            attention_scores = attention_scores + edge_scores

        # Apply softmax attention (per target node)
        attention_weights = softmax(
            attention_scores, edge_index_i, dim=0
        )  # [num_edges, num_heads]

        # Apply attention to values
        attention_weights = attention_weights.unsqueeze(-1)  # [num_edges, num_heads, 1]
        attended_values = attention_weights * value  # [num_edges, num_heads, head_dim]

        # Concatenate heads
        attended_values = attended_values.view(
            -1, self.output_dim
        )  # [num_edges, output_dim]

        return attended_values


class MaxPoolAggregator(MessagePassing):
    """
    Max pooling aggregator that captures peak transmission risk from high-flow connections.

    Takes element-wise maximum of neighbor features, useful for epidemiological modeling
    where we want to capture the strongest transmission pathways.
    """

    def __init__(self, input_dim: int, output_dim: int, normalize: bool = True):
        """
        Initialize max pool aggregator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            normalize: Whether to normalize output
        """
        super().__init__(aggr="max")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Linear transformations
        self.neighbor_transform = nn.Linear(input_dim, output_dim)
        self.self_transform = nn.Linear(input_dim, output_dim)
        self.combine = nn.Linear(2 * output_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.neighbor_transform.weight)
        nn.init.xavier_uniform_(self.self_transform.weight)
        nn.init.xavier_uniform_(self.combine.weight)

        nn.init.zeros_(self.neighbor_transform.bias)
        nn.init.zeros_(self.self_transform.bias)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of max pool aggregator.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Aggregated node features [num_nodes, output_dim]
        """
        # Transform self features
        self_features = self.self_transform(x)

        # Aggregate neighbor features using max pooling
        neighbor_features = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Handle nodes with no neighbors (return zeros for neighbor component)
        if neighbor_features.size(0) != self_features.size(0):
            neighbor_features = torch.zeros_like(self_features)

        # Combine self and neighbor features
        combined = torch.cat([self_features, neighbor_features], dim=1)
        output = self.combine(combined)

        # Apply activation and normalization
        output = F.relu(output)

        if self.normalize:
            output = F.normalize(output, p=2, dim=1)

        return output

    def message(
        self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create messages for max pooling.

        Args:
            x_j: Neighbor node features [num_edges, input_dim]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)

        Returns:
            Messages [num_edges, output_dim]
        """
        messages = self.neighbor_transform(x_j)

        # For epidemiological applications, we can weight by flow volume before max pooling
        if edge_attr is not None and edge_attr.size(1) > 0:
            # Use first edge attribute as flow weight (assuming it's flow volume)
            flow_weights = edge_attr[:, 0].unsqueeze(1)  # [num_edges, 1]
            messages = messages * flow_weights

        return messages


def create_aggregator(
    aggregator_type: str, input_dim: int, output_dim: int, **kwargs
) -> MessagePassing:
    """
    Factory function to create aggregators based on type string.

    Args:
        aggregator_type: Type of aggregator ('mean', 'attention', 'max', 'lstm', 'hybrid')
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        **kwargs: Additional arguments for specific aggregators

    Returns:
        Aggregator instance
    """
    if aggregator_type == "mean":
        return MeanAggregator(input_dim, output_dim, **kwargs)
    elif aggregator_type == "attention":
        return AttentionAggregator(input_dim, output_dim, **kwargs)
    elif aggregator_type == "max":
        return MaxPoolAggregator(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")


if __name__ == "__main__":
    # Example usage and testing
    input_dim, output_dim = 64, 32
    num_nodes, num_edges = 100, 200

    # Create test data
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)  # 4-dimensional edge attributes

    # Test different aggregators
    aggregators = {
        "mean": MeanAggregator(input_dim, output_dim),
        "attention": AttentionAggregator(input_dim, output_dim, edge_dim=4),
        "max": MaxPoolAggregator(input_dim, output_dim),
    }

    for name, aggregator in aggregators.items():
        output = aggregator(x, edge_index, edge_attr)
        print(f"{name} aggregator output shape: {output.shape}")

    print("All aggregators tested successfully")
