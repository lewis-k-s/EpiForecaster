"""
Dual Graph SAGE Model

Implements separate GraphSAGE encoders for mobility and EDAR graphs,
with attention masking to combine signals appropriately.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .attention_mask import AttentionMaskProcessor, MultiScaleAttentionMask
from .graphsage_od import GraphSAGE_OD

logger = logging.getLogger(__name__)


class DualGraphSAGE(nn.Module):
    """
    Dual Graph SAGE model with separate encoders for mobility and EDAR networks.

    This model maintains complete separation between the two graphs, using
    EDAR-municipality contribution ratios as an attention mask to filter
    relevant wastewater signals for each municipality's forecast.
    """

    def __init__(
        self,
        # Mobility graph parameters
        mobility_input_dim: int,
        mobility_hidden_dim: int = 128,
        mobility_output_dim: int = 64,
        mobility_num_layers: int = 2,
        mobility_aggregator: str = "mean",
        # EDAR graph parameters
        edar_input_dim: int = None,
        edar_hidden_dim: int = 64,
        edar_output_dim: int = 32,
        edar_num_layers: int = 2,
        edar_aggregator: str = "attention",
        # Attention mask parameters
        use_learnable_attention: bool = True,
        attention_temperature: float = 1.0,
        attention_fusion: str = "weighted",
        use_multiscale: bool = False,
        # General parameters
        dropout: float = 0.2,
        normalize_embeddings: bool = True,
        residual: bool = True,
    ):
        """
        Initialize Dual Graph SAGE model.

        Args:
            mobility_input_dim: Input dimension for municipality nodes
            mobility_hidden_dim: Hidden dimension for mobility encoder
            mobility_output_dim: Output dimension for mobility embeddings
            mobility_num_layers: Number of layers in mobility encoder
            mobility_aggregator: Aggregation method for mobility graph

            edar_input_dim: Input dimension for EDAR nodes (defaults to mobility_input_dim)
            edar_hidden_dim: Hidden dimension for EDAR encoder
            edar_output_dim: Output dimension for EDAR embeddings
            edar_num_layers: Number of layers in EDAR encoder
            edar_aggregator: Aggregation method for EDAR graph

            use_learnable_attention: Whether to learn additional attention weights
            attention_temperature: Temperature for attention softmax
            attention_fusion: Method to combine mask and learned attention
            use_multiscale: Whether to use multi-scale attention

            dropout: Dropout probability
            normalize_embeddings: Whether to normalize final embeddings
            residual: Whether to use residual connections
        """
        super().__init__()

        # Set EDAR input dimension
        if edar_input_dim is None:
            edar_input_dim = mobility_input_dim

        self.mobility_output_dim = mobility_output_dim
        self.edar_output_dim = edar_output_dim
        self.normalize_embeddings = normalize_embeddings

        # Mobility graph encoder
        self.mobility_encoder = GraphSAGE_OD(
            input_dim=mobility_input_dim,
            hidden_dim=mobility_hidden_dim,
            output_dim=mobility_output_dim,
            num_layers=mobility_num_layers,
            aggregator_type=mobility_aggregator,
            dropout=dropout,
            normalize=False,  # We'll normalize after combination
            residual=residual,
        )

        # EDAR graph encoder (if EDAR data is available)
        self.edar_encoder = GraphSAGE_OD(
            input_dim=edar_input_dim,
            hidden_dim=edar_hidden_dim,
            output_dim=edar_output_dim,
            num_layers=edar_num_layers,
            aggregator_type=edar_aggregator,
            dropout=dropout,
            normalize=False,
            residual=residual,
        )

        # Attention mask processor
        attention_processor = AttentionMaskProcessor(
            edar_embedding_dim=edar_output_dim,
            municipality_embedding_dim=mobility_output_dim,
            use_learnable_weights=use_learnable_attention,
            temperature=attention_temperature,
            dropout=dropout,
            fusion_method=attention_fusion,
        )

        if use_multiscale:
            self.attention_processor = MultiScaleAttentionMask(
                base_processor=attention_processor, num_scales=3, scale_fusion="concat"
            )
        else:
            self.attention_processor = attention_processor

        # Fusion layer to combine mobility and EDAR signals
        self.fusion_layer = nn.Sequential(
            nn.Linear(mobility_output_dim * 2, mobility_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mobility_output_dim, mobility_output_dim),
        )

        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(mobility_output_dim * 2, mobility_output_dim), nn.Sigmoid()
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(mobility_output_dim)

        logger.info(
            f"Initialized DualGraphSAGE with mobility_dim={mobility_output_dim}, "
            f"edar_dim={edar_output_dim}"
        )

    def forward(
        self,
        mobility_data: Data,
        edar_data: Optional[Data],
        edar_muni_mask: Optional[torch.Tensor],
        return_separate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through dual graph model.

        Args:
            mobility_data: PyG Data object for mobility graph
                - x: Node features [n_municipalities, mobility_input_dim]
                - edge_index: Edge connectivity [2, n_edges]
                - edge_attr: Optional edge features [n_edges, edge_feat_dim]
            edar_data: Optional PyG Data object for EDAR graph
                - x: Node features [n_edars, edar_input_dim]
                - edge_index: Edge connectivity [2, n_edar_edges]
            edar_muni_mask: Optional attention mask [n_municipalities, n_edars]
            return_separate: Whether to return separate embeddings

        Returns:
            Dictionary containing:
                - 'embeddings': Combined municipality embeddings [n_municipalities, mobility_output_dim]
                - 'mobility_embeddings': Mobility graph embeddings [n_municipalities, mobility_output_dim]
                - 'edar_signals': Masked EDAR signals [n_municipalities, edar_output_dim]
                - 'attention_stats': Attention statistics (if EDAR data available)
        """
        # Encode mobility graph
        mobility_embeddings = self.mobility_encoder(
            mobility_data.x,  # [n_municipalities, mobility_input_dim]
            mobility_data.edge_index,  # [2, n_edges]
            getattr(
                mobility_data, "edge_attr", None
            ),  # Optional [n_edges, edge_feat_dim]
        )
        # Output shape: [n_municipalities, mobility_output_dim]
        assert mobility_embeddings.shape == (
            mobility_data.x.shape[0],
            self.mobility_output_dim,
        ), (
            f"Mobility embeddings shape mismatch: expected ({mobility_data.x.shape[0]}, {self.mobility_output_dim}), got {mobility_embeddings.shape}"
        )

        outputs = {"mobility_embeddings": mobility_embeddings}

        # Process EDAR signals if available
        if edar_data is not None and edar_muni_mask is not None:
            # Encode EDAR graph
            edar_embeddings = self.edar_encoder(
                edar_data.x,  # [n_edars, edar_input_dim]
                edar_data.edge_index,  # [2, n_edar_edges]
                getattr(
                    edar_data, "edge_attr", None
                ),  # Optional [n_edar_edges, edge_feat_dim]
            )
            # Output shape: [n_edars, edar_output_dim]
            assert edar_embeddings.shape == (
                edar_data.x.shape[0],
                self.edar_output_dim,
            ), (
                f"EDAR embeddings shape mismatch: expected ({edar_data.x.shape[0]}, {self.edar_output_dim}), got {edar_embeddings.shape}"
            )

            # Apply attention mask to get municipality-specific EDAR signals
            masked_edar_signals, attention_stats = self.attention_processor(
                edar_embeddings=edar_embeddings,  # [n_edars, edar_output_dim]
                contribution_mask=edar_muni_mask,  # [n_municipalities, n_edars]
                municipality_embeddings=mobility_embeddings,  # [n_municipalities, mobility_output_dim]
            )
            # Output shape: [n_municipalities, edar_output_dim]
            assert masked_edar_signals.shape == (
                mobility_embeddings.shape[0],
                self.edar_output_dim,
            ), (
                f"Masked EDAR signals shape mismatch: expected ({mobility_embeddings.shape[0]}, {self.edar_output_dim}), got {masked_edar_signals.shape}"
            )

            # Combine mobility and EDAR signals
            combined_embeddings = self._fuse_embeddings(
                mobility_embeddings,  # [n_municipalities, mobility_output_dim]
                masked_edar_signals,  # [n_municipalities, edar_output_dim]
            )
            # Output shape: [n_municipalities, mobility_output_dim]
            assert combined_embeddings.shape == (
                mobility_embeddings.shape[0],
                self.mobility_output_dim,
            ), (
                f"Combined embeddings shape mismatch: expected {mobility_embeddings.shape}, got {combined_embeddings.shape}"
            )

            outputs.update(
                {
                    "embeddings": combined_embeddings,
                    "edar_signals": masked_edar_signals,
                    "attention_stats": attention_stats,
                }
            )
        else:
            # No EDAR data - just use mobility embeddings
            outputs["embeddings"] = mobility_embeddings

        # Normalize final embeddings if requested
        if self.normalize_embeddings:
            outputs["embeddings"] = F.normalize(outputs["embeddings"], p=2, dim=1)

        # Return separate embeddings if requested
        if not return_separate:
            # Remove intermediate embeddings
            outputs = {
                k: v
                for k, v in outputs.items()
                if k in ["embeddings", "attention_stats"]
            }

        return outputs

    def _fuse_embeddings(
        self, mobility_embeddings: torch.Tensor, edar_signals: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse mobility embeddings with masked EDAR signals.

        Args:
            mobility_embeddings: Municipality mobility embeddings [n_munis, mobility_output_dim]
            edar_signals: Masked EDAR signals for municipalities [n_munis, edar_output_dim]

        Returns:
            Fused embeddings [n_munis, mobility_output_dim]
        """
        # Validate input shapes
        n_munis = mobility_embeddings.shape[0]
        assert edar_signals.shape[0] == n_munis, (
            f"Number of municipalities mismatch: mobility {mobility_embeddings.shape[0]}, EDAR {edar_signals.shape[0]}"
        )
        assert mobility_embeddings.shape[1] == self.mobility_output_dim, (
            f"Mobility embedding dim mismatch: expected {self.mobility_output_dim}, got {mobility_embeddings.shape[1]}"
        )
        assert edar_signals.shape[1] == self.edar_output_dim, (
            f"EDAR signal dim mismatch: expected {self.edar_output_dim}, got {edar_signals.shape[1]}"
        )

        # Concatenate embeddings along feature dimension
        concat_features = torch.cat([mobility_embeddings, edar_signals], dim=1)
        # Shape: [n_munis, mobility_output_dim + edar_output_dim]
        expected_concat_dim = self.mobility_output_dim + self.edar_output_dim
        assert concat_features.shape == (n_munis, expected_concat_dim), (
            f"Concatenated features shape mismatch: expected ({n_munis}, {expected_concat_dim}), got {concat_features.shape}"
        )

        # Compute gating weights
        gate = self.fusion_gate(concat_features)
        # Shape: [n_munis, mobility_output_dim]
        assert gate.shape == (n_munis, self.mobility_output_dim), (
            f"Gate shape mismatch: expected ({n_munis}, {self.mobility_output_dim}), got {gate.shape}"
        )

        # Gated fusion using einops for clarity
        # gate: [n_munis, mobility_output_dim]
        # mobility_embeddings: [n_munis, mobility_output_dim]
        # edar_signals: [n_munis, edar_output_dim] -> need to project to mobility_output_dim
        fused = gate * mobility_embeddings + (1 - gate) * edar_signals

        # Additional non-linear transformation with residual connection
        fusion_output = self.fusion_layer(
            concat_features
        )  # [n_munis, mobility_output_dim]
        fused = fusion_output + fused  # Residual connection

        # Layer normalization
        fused = self.layer_norm(fused)
        # Final shape: [n_munis, mobility_output_dim]
        assert fused.shape == (n_munis, self.mobility_output_dim), (
            f"Fused embeddings shape mismatch: expected ({n_munis}, {self.mobility_output_dim}), got {fused.shape}"
        )

        return fused

    def encode_sequence(
        self,
        mobility_sequence: list[Data],
        edar_sequence: Optional[list[Data]] = None,
        edar_muni_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a temporal sequence of graphs.

        Args:
            mobility_sequence: List of mobility graph snapshots
                Each Data object: x [n_municipalities, mobility_input_dim], edge_index [2, n_edges]
            edar_sequence: Optional list of EDAR graph snapshots
                Each Data object: x [n_edars, edar_input_dim], edge_index [2, n_edar_edges]
            edar_muni_mask: Attention mask (constant across time) [n_municipalities, n_edars]

        Returns:
            Temporal sequence of embeddings [n_municipalities, seq_len, mobility_output_dim]
        """
        # Validate sequence lengths
        seq_len = len(mobility_sequence)
        if edar_sequence is not None:
            assert len(edar_sequence) == seq_len, (
                f"Sequence length mismatch: mobility {seq_len}, EDAR {len(edar_sequence)}"
            )

        temporal_embeddings = []

        for t, mobility_data in enumerate(mobility_sequence):
            edar_data = edar_sequence[t] if edar_sequence else None

            outputs = self.forward(
                mobility_data=mobility_data,
                edar_data=edar_data,
                edar_muni_mask=edar_muni_mask,
                return_separate=False,
            )

            # Each embedding: [n_municipalities, mobility_output_dim]
            embedding = outputs["embeddings"]
            temporal_embeddings.append(embedding)

        # Stack along temporal dimension using einops for clarity
        # Input: list of [n_municipalities, mobility_output_dim] tensors
        # Output: [n_municipalities, seq_len, mobility_output_dim]
        temporal_sequence = torch.stack(temporal_embeddings, dim=1)

        # Validate final shape
        expected_shape = (
            temporal_embeddings[0].shape[0],
            seq_len,
            self.mobility_output_dim,
        )
        assert temporal_sequence.shape == expected_shape, (
            f"Temporal sequence shape mismatch: expected {expected_shape}, got {temporal_sequence.shape}"
        )

        return temporal_sequence


class DualGraphSAGEWithTemporal(nn.Module):
    """
    Extended dual graph model with built-in temporal processing.
    """

    def __init__(
        self,
        dual_graph_encoder: DualGraphSAGE,
        sequence_length: int = 7,
        temporal_hidden_dim: int = 64,
        use_attention: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize temporal dual graph model.

        Args:
            dual_graph_encoder: Base dual graph encoder
            sequence_length: Expected sequence length
            temporal_hidden_dim: Hidden dimension for temporal processing
            use_attention: Whether to use temporal attention
            dropout: Dropout probability
        """
        super().__init__()

        self.dual_graph_encoder = dual_graph_encoder
        self.sequence_length = sequence_length

        embedding_dim = dual_graph_encoder.mobility_output_dim

        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Temporal attention (optional)
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=temporal_hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.temporal_attention = None

        # Output projection
        self.output_proj = nn.Linear(temporal_hidden_dim, embedding_dim)

    def forward(
        self,
        mobility_sequence: list[Data],
        edar_sequence: Optional[list[Data]] = None,
        edar_muni_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Process temporal sequence of dual graphs.

        Args:
            mobility_sequence: Temporal sequence of mobility graphs
            edar_sequence: Optional temporal sequence of EDAR graphs
            edar_muni_mask: EDAR-municipality attention mask

        Returns:
            Dictionary with temporal embeddings and statistics
        """
        # Encode spatial graphs at each time step
        spatial_embeddings = self.dual_graph_encoder.encode_sequence(
            mobility_sequence, edar_sequence, edar_muni_mask
        )

        # Process temporal sequence
        # spatial_embeddings: [n_municipalities, seq_len, embed_dim]
        lstm_out, (hidden, cell) = self.temporal_encoder(spatial_embeddings)
        # lstm_out: [n_municipalities, seq_len, temporal_hidden_dim]
        # hidden: [num_layers, n_municipalities, temporal_hidden_dim]

        # Validate LSTM output shape
        expected_lstm_shape = (
            spatial_embeddings.shape[0],
            spatial_embeddings.shape[1],
            self.temporal_encoder.hidden_size,
        )
        assert lstm_out.shape == expected_lstm_shape, (
            f"LSTM output shape mismatch: expected {expected_lstm_shape}, got {lstm_out.shape}"
        )

        # Apply temporal attention if enabled
        if self.temporal_attention is not None:
            attended, attention_weights = self.temporal_attention(
                lstm_out,
                lstm_out,
                lstm_out,  # Query, Key, Value
            )
            # attended: [n_municipalities, seq_len, temporal_hidden_dim]
            # attention_weights: [n_municipalities, seq_len, seq_len]

            # Extract last time step using einops for clarity
            temporal_embeddings = attended[
                :, -1, :
            ]  # [n_municipalities, temporal_hidden_dim]
        else:
            # Extract last time step from LSTM output
            temporal_embeddings = lstm_out[
                :, -1, :
            ]  # [n_municipalities, temporal_hidden_dim]

        # Validate temporal embeddings shape
        assert temporal_embeddings.shape == (
            spatial_embeddings.shape[0],
            self.temporal_encoder.hidden_size,
        ), (
            f"Temporal embeddings shape mismatch: expected ({spatial_embeddings.shape[0]}, {self.temporal_encoder.hidden_size}), got {temporal_embeddings.shape}"
        )

        # Project back to embedding space
        final_embeddings = self.output_proj(temporal_embeddings)
        # Final shape: [n_municipalities, embedding_dim] (same as mobility_output_dim)
        assert final_embeddings.shape == (
            spatial_embeddings.shape[0],
            self.dual_graph_encoder.mobility_output_dim,
        ), (
            f"Final embeddings shape mismatch: expected ({spatial_embeddings.shape[0]}, {self.dual_graph_encoder.mobility_output_dim}), got {final_embeddings.shape}"
        )

        outputs = {
            "embeddings": final_embeddings,
            "spatial_sequence": spatial_embeddings,
            "temporal_features": lstm_out,
        }

        if self.temporal_attention is not None:
            outputs["temporal_attention"] = attention_weights

        return outputs


if __name__ == "__main__":
    # Test the dual graph model
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    n_municipalities = 200
    n_edars = 59
    mobility_feat_dim = 64
    edar_feat_dim = 32

    # Mobility graph data
    mobility_data = Data(
        x=torch.randn(n_municipalities, mobility_feat_dim),
        edge_index=torch.randint(0, n_municipalities, (2, 500)),
        edge_attr=torch.randn(500, 1),
    )

    # EDAR graph data
    edar_data = Data(
        x=torch.randn(n_edars, edar_feat_dim),
        edge_index=torch.randint(0, n_edars, (2, 100)),
        edge_attr=torch.randn(100, 1),
    )

    # EDAR-municipality mask (sparse)
    edar_muni_mask = torch.zeros(n_municipalities, n_edars)
    for i in range(n_municipalities):
        n_connections = torch.randint(1, 4, (1,)).item()
        connected_edars = torch.randperm(n_edars)[:n_connections]
        weights = torch.rand(n_connections)
        edar_muni_mask[i, connected_edars] = weights / weights.sum()

    # Test basic dual graph model
    model = DualGraphSAGE(
        mobility_input_dim=mobility_feat_dim,
        edar_input_dim=edar_feat_dim,
        use_learnable_attention=True,
        use_multiscale=False,
    )

    outputs = model(
        mobility_data=mobility_data,
        edar_data=edar_data,
        edar_muni_mask=edar_muni_mask,
        return_separate=True,
    )

    print(f"Combined embeddings shape: {outputs['embeddings'].shape}")
    print(f"Mobility embeddings shape: {outputs['mobility_embeddings'].shape}")
    print(f"EDAR signals shape: {outputs['edar_signals'].shape}")
    print(f"Attention stats: {list(outputs['attention_stats'].keys())}")

    # Test temporal version
    temporal_model = DualGraphSAGEWithTemporal(
        dual_graph_encoder=model, sequence_length=7, use_attention=True
    )

    # Create sequence of graphs
    mobility_sequence = [mobility_data] * 7
    edar_sequence = [edar_data] * 7

    temporal_outputs = temporal_model(
        mobility_sequence=mobility_sequence,
        edar_sequence=edar_sequence,
        edar_muni_mask=edar_muni_mask,
    )

    print(f"\nTemporal embeddings shape: {temporal_outputs['embeddings'].shape}")
    print(f"Spatial sequence shape: {temporal_outputs['spatial_sequence'].shape}")
    print(f"Temporal features shape: {temporal_outputs['temporal_features'].shape}")

    print("\nDual Graph SAGE model test completed successfully!")
