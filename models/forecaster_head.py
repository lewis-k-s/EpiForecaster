"""
Transformer-based Forecaster Head for epidemiological time series forecasting.

This module implements the temporal modeling component from the design document.
It uses Transformer architecture with multi-head self-attention to model
temporal dependencies in epidemiological time series for multi-step forecasting.

Key Design Principles:
1. Temporal attention: Multi-head self-attention across time dimension
2. Positional encoding: Sinusoidal or learned encodings for time steps
3. Sequence-to-vector: Use final time step for forecasting
4. Configurable architecture: Variable layers, heads, dimensions

Architecture:
- Input: x_seq [batch_size, seq_len, in_dim] containing sequences of per-time-step features
- Processing: Transformer encoder with positional encoding
- Output: forecasts [batch_size, horizon] for future time steps

The forecaster consumes sequences containing:
- Local epidemic features (cases, biomarkers)
- Mobility-enhanced embeddings from MobilityGNN
- Static region embeddings
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer time series processing.

    Implements sinusoidal positional encoding as described in the original
    Transformer paper, adapted for time series forecasting.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1, device=None
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            device: Target device for tensor placement
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dropout = nn.Dropout(dropout)

        # Create sinusoidal positional encodings on the correct device
        pe = torch.zeros(max_len, d_model, device=self.device)
        position = torch.arange(
            0, max_len, dtype=torch.float, device=self.device
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=self.device).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input sequence.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for more flexible temporal modeling.

    Uses learnable embedding vectors for each time position instead of
    fixed sinusoidal encodings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned positional encoding to input sequence.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        position_ids = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_embeddings = self.pos_embedding(position_ids)
        return self.dropout(x + pos_embeddings)


class ForecasterHead(nn.Module):
    """
    Transformer-based forecaster head for epidemiological time series.

    Implements the temporal modeling component from the design document using
    multi-head self-attention to capture temporal dependencies in sequences
    of per-time-step features.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        horizon: int = 7,
        dropout: float = 0.1,
        activation: str = "gelu",
        positional_encoding: str = "sinusoidal",
        max_seq_len: int = 1000,
        use_layer_norm: bool = True,
        device=None,
    ):
        """
        Initialize ForecasterHead.

        Args:
            in_dim: Input feature dimension (local + mobility + region features)
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            num_layers: Number of Transformer encoder layers
            horizon: Forecasting horizon (number of future time steps)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            positional_encoding: Type of positional encoding ('sinusoidal', 'learned')
            max_seq_len: Maximum sequence length for positional encoding
            use_layer_norm: Whether to use layer normalization in encoder
        """
        super().__init__()

        self.device = device or torch.device("cpu")
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.horizon = horizon
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding = positional_encoding
        self.max_seq_len = max_seq_len

        # Input projection layer
        self.input_projection = nn.Linear(in_dim, d_model)

        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(
                d_model, max_seq_len, dropout, device=self.device
            )
        elif positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding}")

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=use_layer_norm,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection for sequence-to-vector forecasting
        # We use final time step representation for forecasting
        self.output_projection = nn.Linear(d_model, horizon)

        # Optional output normalization
        self.output_norm = nn.LayerNorm(horizon) if use_layer_norm else None

        # Log initialization
        logger.info(
            f"Initialized ForecasterHead: {in_dim}->{d_model}->{horizon}, "
            f"layers={num_layers}, heads={n_heads}"
        )

    def forward(
        self, x_seq: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of ForecasterHead.

        Args:
            x_seq: Input sequence [batch_size, seq_len, in_dim]
                Contains concatenated local features, mobility embeddings, and region embeddings
            mask: Optional attention mask [batch_size, seq_len]
                True values indicate positions that should be masked

        Returns:
            Forecasts [batch_size, horizon] for future time steps
        """
        batch_size, seq_len, _ = x_seq.shape

        # Input projection to model dimension
        x = self.input_projection(x_seq)  # [batch_size, seq_len, d_model]

        # Apply positional encoding
        x = self.pos_encoding(x)  # [batch_size, seq_len, d_model]

        # Handle attention mask (PyTorch uses different mask convention)
        attn_mask = None
        if mask is not None:
            # Convert from boolean mask to additive mask
            # True in input mask means valid positions, False means masked
            attn_mask = ~mask  # Flip mask convention
            attn_mask = attn_mask.unsqueeze(1).expand(
                -1, seq_len, -1
            )  # [batch_size, seq_len, seq_len]

        # Apply Transformer encoder
        # Note: PyTorch's TransformerEncoder expects src_key_padding_mask for padding
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=attn_mask if attn_mask is not None else None
        )

        # Sequence-to-vector: use final time step for forecasting
        # This matches design document specification for sequence-to-vector head
        final_hidden = encoded[:, -1, :]  # [batch_size, d_model]

        # Output projection to forecast horizon
        forecasts = self.output_projection(final_hidden)  # [batch_size, horizon]

        # Optional output normalization
        if self.output_norm is not None:
            forecasts = self.output_norm(forecasts)

        return forecasts

    def forward_with_attention(
        self, x_seq: torch.Tensor, return_attention: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with attention weights for interpretability.

        Args:
            x_seq: Input sequence [batch_size, seq_len, in_dim]
            return_attention: Whether to return attention weights

        Returns:
            forecasts: [batch_size, horizon] forecasts
            attention_weights: Optional attention weights for analysis
        """
        if not return_attention:
            return self.forward(x_seq), None

        # For attention visualization, we'd need to modify the TransformerEncoder
        # For now, just return the forecasts
        forecasts = self.forward(x_seq)
        attention_weights = None  # TODO: Implement attention extraction if needed

        return forecasts, attention_weights

    def get_forecast_dimension(self) -> int:
        """Get the forecasting horizon."""
        return self.horizon

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiHorizonForecaster(nn.Module):
    """
    Extended forecaster that supports multiple horizons with separate heads.

    Useful for scenarios where different forecasting horizons are needed
    (e.g., short-term and long-term predictions).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        horizons: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        positional_encoding: str = "sinusoidal",
        device=None,
    ):
        """
        Initialize multi-horizon forecaster.

        Args:
            in_dim: Input feature dimension
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            num_layers: Number of Transformer encoder layers
            horizons: List of forecasting horizons
            dropout: Dropout probability
            activation: Activation function
            positional_encoding: Type of positional encoding
        """
        super().__init__()

        self.device = device or torch.device("cpu")
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.horizons = horizons if horizons is not None else [1, 7, 14]
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding = positional_encoding

        # Shared components
        self.input_projection = nn.Linear(in_dim, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, dropout=dropout, device=self.device
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Multiple output heads for different horizons
        self.output_heads = nn.ModuleDict()
        for horizon in horizons:
            self.output_heads[str(horizon)] = nn.Linear(d_model, horizon)

    def forward(self, x_seq: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Forward pass for multiple horizons.

        Args:
            x_seq: Input sequence [batch_size, seq_len, in_dim]

        Returns:
            Dictionary mapping horizons to forecasts
        """
        # Shared encoding
        x = self.input_projection(x_seq)
        x = self.pos_encoding(x)
        encoded = self.transformer_encoder(x)

        # Use final time step for all horizons
        final_hidden = encoded[:, -1, :]

        # Generate forecasts for each horizon
        forecasts = {}
        for horizon in self.horizons:
            forecasts[horizon] = self.output_heads[str(horizon)](final_hidden)

        return forecasts


def create_forecaster_head(
    in_dim: int,
    d_model: int = 128,
    n_heads: int = 4,
    num_layers: int = 3,
    horizon: int = 7,
    **kwargs,
) -> ForecasterHead:
    """
    Factory function to create ForecasterHead with common configurations.

    Args:
        in_dim: Input feature dimension
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        num_layers: Number of Transformer encoder layers
        horizon: Forecasting horizon
        **kwargs: Additional arguments for ForecasterHead

    Returns:
        Configured ForecasterHead instance
    """
    return ForecasterHead(
        in_dim=in_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        horizon=horizon,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage and testing
    torch.manual_seed(42)

    # Configuration
    batch_size = 4
    seq_len = 14  # History window length
    in_dim = 20  # Combined features dimension
    horizon = 7  # Forecast 7 days ahead
    d_model = 64
    n_heads = 4
    num_layers = 2

    # Create test data
    x_seq = torch.randn(batch_size, seq_len, in_dim)

    print(f"Input sequence shape: {x_seq.shape}")

    # Test basic forecaster
    forecaster = create_forecaster_head(
        in_dim=in_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        horizon=horizon,
    )

    print(f"Forecaster parameters: {forecaster.count_parameters():,}")

    # Forward pass
    forecasts = forecaster(x_seq)
    print(f"Forecasts shape: {forecasts.shape}")
    print(f"Expected horizon: {horizon}")

    # Test with attention mask (simulate some padding)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -2:] = False  # Mask last 2 positions
    forecasts_masked = forecaster(x_seq, mask)
    print(f"Masked forecasts shape: {forecasts_masked.shape}")

    # Test multi-horizon forecaster
    multi_forecaster = MultiHorizonForecaster(
        in_dim=in_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        horizons=[1, 3, 7, 14],
    )

    multi_forecasts = multi_forecaster(x_seq)
    print("Multi-horizon forecasts:")
    for horizon, forecast in multi_forecasts.items():
        print(f"  Horizon {horizon}: {forecast.shape}")

    print("ForecasterHead test completed successfully!")
