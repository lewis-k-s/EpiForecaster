"""
Dual Graph Forecaster

Combines the dual graph SAGE model with temporal forecasting capabilities.
This wrapper integrates the separate mobility and EDAR graphs with prediction heads.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import reduce
from torch_geometric.data import Data

from .dual_graph_sage import DualGraphSAGE

logger = logging.getLogger(__name__)


class DualGraphForecaster(nn.Module):
    """
    Complete forecasting model that combines dual graph encoding with prediction.

    This model processes mobility graphs and optionally EDAR signals through
    separate encoders, applies attention masking, and generates forecasts.
    """

    def __init__(
        self,
        dual_graph_encoder: DualGraphSAGE,
        forecast_horizon: int = 7,
        sequence_length: int = 14,
        hidden_dim: int = 64,
        use_temporal_attention: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize dual graph forecaster.

        Args:
            dual_graph_encoder: Dual graph SAGE encoder
            forecast_horizon: Number of time steps to forecast
            sequence_length: Length of input sequences
            hidden_dim: Hidden dimension for forecasting layers
            use_temporal_attention: Whether to use temporal attention
            dropout: Dropout probability
        """
        super().__init__()

        self.dual_graph_encoder = dual_graph_encoder
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        embedding_dim = dual_graph_encoder.mobility_output_dim

        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Single layer
        )

        # Temporal attention (optional)
        if use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        else:
            self.temporal_attention = None

        # Forecasting heads
        self.case_count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

        self.case_rate_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Sigmoid(),  # Rates should be in [0, 1]
        )

        # Uncertainty estimation
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, forecast_horizon),
            nn.Softplus(),  # Ensure positive
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mobility_sequence: list[Data],
        edar_sequence: Optional[list[Data]] = None,
        edar_muni_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through dual graph forecaster.

        Args:
            mobility_sequence: Temporal sequence of mobility graphs
                Each Data object: x [n_municipalities, mobility_input_dim], edge_index [2, n_edges]
            edar_sequence: Optional temporal sequence of EDAR graphs
                Each Data object: x [n_edars, edar_input_dim], edge_index [2, n_edar_edges]
            edar_muni_mask: EDAR-municipality attention mask [n_municipalities, n_edars]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with forecasting outputs:
                - 'case_count_forecast': [n_municipalities, forecast_horizon]
                - 'case_rate_forecast': [n_municipalities, forecast_horizon]
                - 'temporal_embeddings': [n_municipalities, hidden_dim]
                - 'spatial_sequence': [n_municipalities, sequence_length, embed_dim]
                - 'uncertainty': [n_municipalities, forecast_horizon] (if requested)
        """
        # Validate input sequences
        seq_len = len(mobility_sequence)
        assert seq_len == self.sequence_length, (
            f"Input sequence length mismatch: expected {self.sequence_length}, got {seq_len}"
        )
        if edar_sequence is not None:
            assert len(edar_sequence) == seq_len, (
                f"EDAR sequence length mismatch: expected {seq_len}, got {len(edar_sequence)}"
            )

        # Encode spatial graphs at each time step
        spatial_embeddings = []
        n_municipalities = mobility_sequence[0].x.shape[0]

        for t in range(seq_len):
            mobility_data = mobility_sequence[t]
            edar_data = edar_sequence[t] if edar_sequence else None

            # Validate graph consistency
            assert mobility_data.x.shape[0] == n_municipalities, (
                f"Municipality count mismatch at time {t}: expected {n_municipalities}, got {mobility_data.x.shape[0]}"
            )

            # Get spatial embeddings
            outputs = self.dual_graph_encoder(
                mobility_data=mobility_data,
                edar_data=edar_data,
                edar_muni_mask=edar_muni_mask,
                return_separate=False,
            )

            # Each embedding: [n_municipalities, embed_dim]
            embedding = outputs["embeddings"]
            spatial_embeddings.append(embedding)

        # Stack temporal sequence: [n_municipalities, sequence_length, embed_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)
        expected_temporal_shape = (
            n_municipalities,
            seq_len,
            self.dual_graph_encoder.mobility_output_dim,
        )
        assert temporal_sequence.shape == expected_temporal_shape, (
            f"Temporal sequence shape mismatch: expected {expected_temporal_shape}, got {temporal_sequence.shape}"
        )

        # Temporal encoding through LSTM
        lstm_out, (hidden, cell) = self.temporal_encoder(temporal_sequence)
        # lstm_out: [n_municipalities, sequence_length, hidden_dim]
        # hidden: [num_layers, n_municipalities, hidden_dim]
        expected_lstm_shape = (n_municipalities, seq_len, self.hidden_dim)
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
            # attended: [n_municipalities, sequence_length, hidden_dim]
            # attention_weights: [n_municipalities, sequence_length, sequence_length]

            # Use mean of attended sequence using einops for clarity
            temporal_embeddings = reduce(
                attended,
                "municipalities sequence features -> municipalities features",
                "mean",
            )
        else:
            # Use final time step output
            temporal_embeddings = lstm_out[:, -1, :]  # [n_municipalities, hidden_dim]

        # Validate temporal embeddings shape
        assert temporal_embeddings.shape == (n_municipalities, self.hidden_dim), (
            f"Temporal embeddings shape mismatch: expected ({n_municipalities}, {self.hidden_dim}), got {temporal_embeddings.shape}"
        )

        temporal_embeddings = self.dropout(temporal_embeddings)

        # Generate forecasts with shape validation
        case_count_forecast = self.case_count_predictor(temporal_embeddings)
        # Shape: [n_municipalities, forecast_horizon]
        assert case_count_forecast.shape == (n_municipalities, self.forecast_horizon), (
            f"Case count forecast shape mismatch: expected ({n_municipalities}, {self.forecast_horizon}), got {case_count_forecast.shape}"
        )

        case_rate_forecast = self.case_rate_predictor(temporal_embeddings)
        # Shape: [n_municipalities, forecast_horizon]
        assert case_rate_forecast.shape == (n_municipalities, self.forecast_horizon), (
            f"Case rate forecast shape mismatch: expected ({n_municipalities}, {self.forecast_horizon}), got {case_rate_forecast.shape}"
        )

        outputs = {
            "case_count_forecast": case_count_forecast,
            "case_rate_forecast": case_rate_forecast,
            "temporal_embeddings": temporal_embeddings,
            "spatial_sequence": temporal_sequence,
        }

        # Add uncertainty if requested
        if return_uncertainty:
            uncertainty = self.uncertainty_predictor(temporal_embeddings)
            # Shape: [n_municipalities, forecast_horizon]
            assert uncertainty.shape == (n_municipalities, self.forecast_horizon), (
                f"Uncertainty shape mismatch: expected ({n_municipalities}, {self.forecast_horizon}), got {uncertainty.shape}"
            )
            outputs["uncertainty"] = uncertainty

        return outputs

    def predict_future(
        self,
        mobility_sequence: list[Data],
        edar_sequence: Optional[list[Data]] = None,
        edar_muni_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Generate future predictions with optional confidence intervals.

        Args:
            mobility_sequence: Input mobility sequence
            edar_sequence: Optional EDAR sequence
            edar_muni_mask: EDAR-municipality mask
            return_confidence: Whether to compute confidence intervals

        Returns:
            Predictions with optional confidence intervals
        """
        with torch.no_grad():
            outputs = self.forward(
                mobility_sequence,
                edar_sequence,
                edar_muni_mask,
                return_uncertainty=True,
            )

            predictions = {
                "case_counts": outputs["case_count_forecast"],
                "case_rates": outputs["case_rate_forecast"],
            }

            if return_confidence:
                uncertainty = outputs["uncertainty"]
                case_counts = outputs["case_count_forecast"]

                # 95% confidence intervals
                confidence_multiplier = 1.96
                lower_bound = case_counts - confidence_multiplier * uncertainty
                upper_bound = case_counts + confidence_multiplier * uncertainty

                predictions["confidence_intervals"] = {
                    "lower": torch.clamp(lower_bound, min=0),
                    "upper": upper_bound,
                }

        return predictions

    def forward_subgraph(
        self,
        mobility_sequence: list[Data],
        target_node_indices: torch.Tensor,
        edar_sequence: Optional[list[Data]] = None,
        edar_muni_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for subgraph batches with k=1 sampling.

        Args:
            mobility_sequence: Temporal sequence of mobility subgraphs
                Each Data object contains subgraph with target nodes + neighbors
            target_node_indices: Indices of target nodes in the original graph [num_targets]
            edar_sequence: Optional temporal sequence of EDAR subgraphs
            edar_muni_mask: EDAR-municipality attention mask (subset for targets)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with forecasting outputs for target nodes only:
                - 'case_count_forecast': [num_targets, forecast_horizon]
                - 'case_rate_forecast': [num_targets, forecast_horizon]
                - 'temporal_embeddings': [num_targets, hidden_dim]
                - 'spatial_sequence': [num_targets, sequence_length, embed_dim]
                - 'target_node_indices': [num_targets]
                - 'uncertainty': [num_targets, forecast_horizon] (if requested)
        """
        # Validate inputs
        seq_len = len(mobility_sequence)
        assert seq_len == self.sequence_length, (
            f"Input sequence length mismatch: expected {self.sequence_length}, got {seq_len}"
        )
        num_targets = len(target_node_indices)
        assert num_targets > 0, "No target nodes provided"

        # Encode spatial subgraphs at each time step
        spatial_embeddings = []

        for t in range(seq_len):
            mobility_data = mobility_sequence[t]
            edar_data = edar_sequence[t] if edar_sequence else None

            # Get spatial embeddings for subgraph
            outputs = self.dual_graph_encoder(
                mobility_data=mobility_data,
                edar_data=edar_data,
                edar_muni_mask=edar_muni_mask,
                return_separate=False,
            )

            # Extract embeddings for target nodes only
            # NeighborLoader puts target nodes first in the subgraph
            subgraph_embeddings = outputs["embeddings"]  # [subgraph_size, embed_dim]
            target_embeddings = subgraph_embeddings[
                :num_targets
            ]  # [num_targets, embed_dim]
            assert target_embeddings.shape == (
                num_targets,
                self.dual_graph_encoder.mobility_output_dim,
            ), (
                f"Target embeddings shape mismatch at time {t}: expected ({num_targets}, {self.dual_graph_encoder.mobility_output_dim}), got {target_embeddings.shape}"
            )

            spatial_embeddings.append(target_embeddings)

        # Stack temporal sequence: [num_targets, sequence_length, embed_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)
        expected_temporal_shape = (
            num_targets,
            seq_len,
            self.dual_graph_encoder.mobility_output_dim,
        )
        assert temporal_sequence.shape == expected_temporal_shape, (
            f"Temporal sequence shape mismatch: expected {expected_temporal_shape}, got {temporal_sequence.shape}"
        )

        # Temporal encoding through LSTM
        lstm_out, (hidden, cell) = self.temporal_encoder(temporal_sequence)
        # lstm_out: [num_targets, sequence_length, hidden_dim]
        expected_lstm_shape = (num_targets, seq_len, self.temporal_encoder.hidden_size)
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
            # attended: [num_targets, sequence_length, hidden_dim]

            # Use mean of attended sequence using einops for clarity
            temporal_embeddings = reduce(
                attended, "targets sequence features -> targets features", "mean"
            )
        else:
            # Use final time step output
            temporal_embeddings = lstm_out[:, -1, :]  # [num_targets, hidden_dim]

        # Validate temporal embeddings shape
        assert temporal_embeddings.shape == (num_targets, self.hidden_dim), (
            f"Temporal embeddings shape mismatch: expected ({num_targets}, {self.hidden_dim}), got {temporal_embeddings.shape}"
        )

        temporal_embeddings = self.dropout(temporal_embeddings)

        # Generate forecasts with shape validation
        case_count_forecast = self.case_count_predictor(temporal_embeddings)
        # Shape: [num_targets, forecast_horizon]
        assert case_count_forecast.shape == (num_targets, self.forecast_horizon), (
            f"Case count forecast shape mismatch: expected ({num_targets}, {self.forecast_horizon}), got {case_count_forecast.shape}"
        )

        case_rate_forecast = self.case_rate_predictor(temporal_embeddings)
        # Shape: [num_targets, forecast_horizon]
        assert case_rate_forecast.shape == (num_targets, self.forecast_horizon), (
            f"Case rate forecast shape mismatch: expected ({num_targets}, {self.forecast_horizon}), got {case_rate_forecast.shape}"
        )

        outputs = {
            "case_count_forecast": case_count_forecast,
            "case_rate_forecast": case_rate_forecast,
            "temporal_embeddings": temporal_embeddings,
            "spatial_sequence": temporal_sequence,
            "target_node_indices": target_node_indices,
        }

        # Add uncertainty if requested
        if return_uncertainty:
            uncertainty = self.uncertainty_predictor(temporal_embeddings)
            # Shape: [num_targets, forecast_horizon]
            assert uncertainty.shape == (num_targets, self.forecast_horizon), (
                f"Uncertainty shape mismatch: expected ({num_targets}, {self.forecast_horizon}), got {uncertainty.shape}"
            )
            outputs["uncertainty"] = uncertainty

        return outputs


class SimpleDualGraphForecaster(nn.Module):
    """
    Simplified version for when EDAR data is not available.
    Uses only mobility graphs with a simpler architecture.
    """

    def __init__(
        self,
        mobility_encoder: nn.Module,
        forecast_horizon: int = 7,
        sequence_length: int = 14,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        """
        Initialize simple forecaster.

        Args:
            mobility_encoder: Mobility graph encoder
            forecast_horizon: Number of steps to forecast
            sequence_length: Input sequence length
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.mobility_encoder = mobility_encoder
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length

        embedding_dim = mobility_encoder.output_dim

        logger.info("SimpleDualGraphForecaster initialization:")
        logger.info(f"  - Mobility encoder output dim: {embedding_dim}")
        logger.info(f"  - Temporal LSTM input size: {embedding_dim}")
        logger.info(f"  - Temporal LSTM hidden size: {hidden_dim}")
        logger.info(f"  - Forecast horizon: {forecast_horizon}")

        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Forecasting head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

    def forward(self, mobility_sequence: list[Data]) -> dict[str, torch.Tensor]:
        """
        Forward pass for mobility-only forecasting.

        Args:
            mobility_sequence: Temporal sequence of mobility graphs
                Each Data object: x [n_municipalities, input_dim], edge_index [2, n_edges]

        Returns:
            Forecasting outputs:
                - 'case_count_forecast': [n_municipalities, forecast_horizon]
                - 'temporal_embeddings': [n_municipalities, hidden_dim]
        """
        # Validate input sequence
        seq_len = len(mobility_sequence)
        assert seq_len == self.sequence_length, (
            f"Input sequence length mismatch: expected {self.sequence_length}, got {seq_len}"
        )
        n_municipalities = mobility_sequence[0].x.shape[0]

        # Encode mobility sequence
        spatial_embeddings = []

        for t, mobility_data in enumerate(mobility_sequence):
            # Validate graph consistency
            assert mobility_data.x.shape[0] == n_municipalities, (
                f"Municipality count mismatch at time {t}: expected {n_municipalities}, got {mobility_data.x.shape[0]}"
            )

            embedding = self.mobility_encoder(
                mobility_data.x,  # [n_municipalities, input_dim]
                mobility_data.edge_index,  # [2, n_edges]
                getattr(
                    mobility_data, "edge_attr", None
                ),  # Optional [n_edges, edge_feat_dim]
            )
            # Each embedding: [n_municipalities, output_dim]
            assert embedding.shape == (
                n_municipalities,
                self.mobility_encoder.output_dim,
            ), (
                f"Embedding shape mismatch at time {t}: expected ({n_municipalities}, {self.mobility_encoder.output_dim}), got {embedding.shape}"
            )
            spatial_embeddings.append(embedding)

        # Stack temporal sequence: [n_municipalities, sequence_length, output_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)
        expected_temporal_shape = (
            n_municipalities,
            seq_len,
            self.mobility_encoder.output_dim,
        )
        assert temporal_sequence.shape == expected_temporal_shape, (
            f"Temporal sequence shape mismatch: expected {expected_temporal_shape}, got {temporal_sequence.shape}"
        )

        # Temporal encoding through LSTM
        lstm_out, _ = self.temporal_encoder(temporal_sequence)
        # lstm_out: [n_municipalities, sequence_length, hidden_dim]
        expected_lstm_shape = (n_municipalities, seq_len, self.forecaster_hidden_dim)
        assert lstm_out.shape == expected_lstm_shape, (
            f"LSTM output shape mismatch: expected {expected_lstm_shape}, got {lstm_out.shape}"
        )

        # Use final output for prediction
        final_embeddings = lstm_out[:, -1, :]  # [n_municipalities, hidden_dim]
        assert final_embeddings.shape == (
            n_municipalities,
            self.forecaster_hidden_dim,
        ), (
            f"Final embeddings shape mismatch: expected ({n_municipalities}, {self.forecaster_hidden_dim}), got {final_embeddings.shape}"
        )

        # Generate forecast
        predictions = self.predictor(final_embeddings)
        # Shape: [n_municipalities, forecast_horizon]
        assert predictions.shape == (n_municipalities, self.forecast_horizon), (
            f"Predictions shape mismatch: expected ({n_municipalities}, {self.forecast_horizon}), got {predictions.shape}"
        )

        return {
            "case_count_forecast": predictions,
            "temporal_embeddings": final_embeddings,
        }

    def forward_subgraph(
        self,
        mobility_sequence: list[Data],
        target_node_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for mobility-only subgraph forecasting.

        Args:
            mobility_sequence: Temporal sequence of mobility subgraphs
                Each Data object contains subgraph with target nodes + neighbors
            target_node_indices: Indices of target nodes in the original graph [num_targets]

        Returns:
            Forecasting outputs for target nodes only:
                - 'case_count_forecast': [num_targets, forecast_horizon]
                - 'temporal_embeddings': [num_targets, hidden_dim]
                - 'target_node_indices': [num_targets]
        """
        # Validate inputs
        seq_len = len(mobility_sequence)
        assert seq_len == self.sequence_length, (
            f"Input sequence length mismatch: expected {self.sequence_length}, got {seq_len}"
        )
        num_targets = len(target_node_indices)
        assert num_targets > 0, "No target nodes provided"

        # Encode mobility sequence
        spatial_embeddings = []

        for t, mobility_data in enumerate(mobility_sequence):
            embedding = self.mobility_encoder(
                mobility_data.x,  # [subgraph_size, input_dim]
                mobility_data.edge_index,  # [2, n_edges]
                getattr(
                    mobility_data, "edge_attr", None
                ),  # Optional [n_edges, edge_feat_dim]
            )
            # Each embedding: [subgraph_size, output_dim]

            # Extract embeddings for target nodes only
            # NeighborLoader puts target nodes first in the subgraph
            target_embeddings = embedding[:num_targets]  # [num_targets, output_dim]
            assert target_embeddings.shape == (
                num_targets,
                self.mobility_encoder.output_dim,
            ), (
                f"Target embeddings shape mismatch at time {t}: expected ({num_targets}, {self.mobility_encoder.output_dim}), got {target_embeddings.shape}"
            )

            spatial_embeddings.append(target_embeddings)

        # Stack temporal sequence: [num_targets, sequence_length, output_dim]
        temporal_sequence = torch.stack(spatial_embeddings, dim=1)
        expected_temporal_shape = (
            num_targets,
            seq_len,
            self.mobility_encoder.output_dim,
        )
        assert temporal_sequence.shape == expected_temporal_shape, (
            f"Temporal sequence shape mismatch: expected {expected_temporal_shape}, got {temporal_sequence.shape}"
        )

        # Temporal encoding through LSTM
        lstm_out, _ = self.temporal_encoder(temporal_sequence)
        # lstm_out: [num_targets, sequence_length, hidden_dim]
        expected_lstm_shape = (num_targets, seq_len, self.temporal_encoder.hidden_size)
        assert lstm_out.shape == expected_lstm_shape, (
            f"LSTM output shape mismatch: expected {expected_lstm_shape}, got {lstm_out.shape}"
        )

        # Use final output for prediction
        final_embeddings = lstm_out[:, -1, :]  # [num_targets, hidden_dim]
        assert final_embeddings.shape == (
            num_targets,
            self.temporal_encoder.hidden_size,
        ), (
            f"Final embeddings shape mismatch: expected ({num_targets}, {self.temporal_encoder.hidden_size}), got {final_embeddings.shape}"
        )

        # Generate forecast
        predictions = self.predictor(final_embeddings)
        # Shape: [num_targets, forecast_horizon]
        assert predictions.shape == (num_targets, self.forecast_horizon), (
            f"Predictions shape mismatch: expected ({num_targets}, {self.forecast_horizon}), got {predictions.shape}"
        )

        return {
            "case_count_forecast": predictions,
            "temporal_embeddings": final_embeddings,
            "target_node_indices": target_node_indices,
        }


def create_dual_graph_forecaster(
    config: dict[str, Any],
    mobility_feature_dim: int,
    use_edar_data: bool = False,
    edar_attention_loader=None,
) -> nn.Module:
    """
    Factory function to create dual graph forecaster.

    Args:
        config: Model configuration
        mobility_feature_dim: Input dimension for mobility features
        use_edar_data: Whether to use EDAR data
        edar_attention_loader: EDAR attention loader (if using EDAR data)

    Returns:
        Configured dual graph forecaster
    """
    if use_edar_data and edar_attention_loader is not None:
        # Create full dual graph model
        dual_encoder = DualGraphSAGE(
            mobility_input_dim=mobility_feature_dim,
            mobility_hidden_dim=config.get("hidden_dim", 128),
            mobility_output_dim=config.get("hidden_dim", 128) // 2,
            mobility_num_layers=config.get("num_layers", 2),
            mobility_aggregator=config.get("aggregator", "mean"),
            edar_input_dim=config.get("edar_input_dim", 32),
            edar_hidden_dim=config.get("edar_hidden_dim", 64),
            edar_output_dim=config.get("edar_hidden_dim", 64) // 2,
            edar_num_layers=2,
            edar_aggregator="attention",
            use_learnable_attention=True,
            dropout=config.get("dropout", 0.2),
        )

        forecaster = DualGraphForecaster(
            dual_graph_encoder=dual_encoder,
            forecast_horizon=config.get("forecast_horizon", 7),
            sequence_length=config.get("sequence_length", 14),
            hidden_dim=config.get("hidden_dim", 128) // 2,
            dropout=config.get("dropout", 0.2) * 0.5,
        )

    else:
        # Create mobility-only model
        from .graphsage_od import GraphSAGE_OD

        hidden_dim = config.get("hidden_dim", 128)
        output_dim = hidden_dim // 2
        num_layers = config.get("num_layers", 2)

        logger.info("Creating mobility-only model with dimensions:")
        logger.info(f"  - Input dim: {mobility_feature_dim}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Output dim: {output_dim}")
        logger.info(f"  - Num layers: {num_layers}")
        logger.info(f"  - Aggregator: {config.get('aggregator', 'mean')}")

        mobility_encoder = GraphSAGE_OD(
            input_dim=mobility_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            aggregator_type=config.get("aggregator", "mean"),
            dropout=config.get("dropout", 0.2),
        )

        forecaster_hidden_dim = output_dim
        logger.info("Creating SimpleDualGraphForecaster with:")
        logger.info(f"  - Mobility encoder output dim: {output_dim}")
        logger.info(f"  - Temporal encoder input dim: {output_dim}")
        logger.info(f"  - Temporal encoder hidden dim: {forecaster_hidden_dim}")

        forecaster = SimpleDualGraphForecaster(
            mobility_encoder=mobility_encoder,
            forecast_horizon=config.get("forecast_horizon", 7),
            sequence_length=config.get("sequence_length", 14),
            hidden_dim=forecaster_hidden_dim,
            dropout=config.get("dropout", 0.2) * 0.5,
        )

    logger.info(f"Created dual graph forecaster (use_edar_data={use_edar_data})")
    return forecaster


if __name__ == "__main__":
    # Test the dual graph forecaster
    logging.basicConfig(level=logging.INFO)

    # Test configuration
    config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "forecast_horizon": 7,
        "sequence_length": 14,
        "dropout": 0.2,
        "aggregator": "mean",
    }

    # Test mobility-only forecaster
    mobility_forecaster = create_dual_graph_forecaster(
        config=config, mobility_feature_dim=64, use_edar_data=False
    )

    # Create dummy data
    n_municipalities = 200
    mobility_sequence = []
    for _ in range(14):  # 14 time steps
        mobility_data = Data(
            x=torch.randn(n_municipalities, 64),
            edge_index=torch.randint(0, n_municipalities, (2, 500)),
        )
        mobility_sequence.append(mobility_data)

    # Test forward pass
    outputs = mobility_forecaster(mobility_sequence)
    print(f"Mobility-only forecast shape: {outputs['case_count_forecast'].shape}")

    print("Dual graph forecaster test completed successfully!")
