"""
EpiForecaster: Three-layer epidemiological forecasting architecture.

This module implements the complete three-layer architecture from the design document:
RegionEmbedder (static spatial) → MobilityGNN (per-time-step spatial) → ForecasterHead (temporal)

Key Design Principles:
1. Three-layer pipeline: Clear separation of spatial, mobility, and temporal modeling
2. 4 model variants: Configurable via boolean flags for ablation studies
3. Batch processing: Handle multiple regions and time steps efficiently
4. Data flow orchestration: Manage tensor shapes between components

Architecture:
- Input: cases_hist, biomarkers_hist, mobility_hist, region_ids
- Layer 1: RegionEmbedder creates static spatial embeddings
- Layer 2: MobilityGNN processes per-time-step mobility-weighted case signals
- Layer 3: ForecasterHead uses Transformer to forecast future cases
- Output: forecasts for all variants based on configuration

Data Flow:
1. Extract static region embeddings (if enabled)
2. For each time step in history window:
   - Build per-time node features: concat(cases_t, biomarkers_t, region_embeddings_t)
   - Run MobilityGNN: mobility_embeddings_t = MobilityGNN(features_t, mobility_t)
3. Build final sequences: concat(local_features, mobility_embeddings, static_embeddings)
4. Feed to ForecasterHead: forecasts = ForecasterHead(sequence)
5. Handle variant configuration by zeroing unused components
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .configs import ModelVariant
from .forecaster_head import ForecasterHead, create_forecaster_head
from .mobility_gnn import MobilityGNN, create_mobility_gnn

logger = logging.getLogger(__name__)


class EpiForecaster(nn.Module):
    """
    Three-layer epidemiological forecaster implementing the design document architecture.

    Combines static spatial embeddings (RegionEmbedder), mobility-weighted spatial
    processing (MobilityGNN), and temporal modeling (ForecasterHead) to forecast
    future disease incidence.
    """

    def __init__(
        self,
        variant_type: ModelVariant,
        cases_dim: int = 1,
        biomarkers_dim: int = 4,
        region_embedding_dim: int = 64,
        mobility_embedding_dim: int = 64,
        sequence_length: int = 14,
        forecast_horizon: int = 7,
        device: torch.device | None = None,
    ):
        """
        Initialize EpiForecaster with three-layer architecture.

        Args:
            cases_dim: Dimension of case features (usually 1)
            biomarkers_dim: Dimension of biomarker features
            region_embedding_dim: Dimension of static region embeddings
            mobility_embedding_dim: Dimension of mobility embeddings
            sequence_length: Length of historical context window
            forecast_horizon: Number of future time steps to forecast
            device: Device for tensor operations
        """
        super().__init__()

        self.variant_type = variant_type
        self.cases_dim = cases_dim
        self.biomarkers_dim = biomarkers_dim
        self.region_embedding_dim = region_embedding_dim
        self.mobility_embedding_dim = mobility_embedding_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device or torch.device("cpu")

        # Calculate input dimensions for forecaster head
        self.local_dim = cases_dim + biomarkers_dim  # Local epidemic features

        # Total input dimension for forecaster head
        self.forecaster_input_dim = self.local_dim

        if self.variant_type.regions:
            # load region embeddings from file
            self.region_embedder = torch.load(self.config.region_embeddings_path)

        if self.variant_type.mobility:
            mobility_input_dim = self.local_dim

            self.mobility_gnn = create_mobility_gnn(
                in_dim=mobility_input_dim,
                hidden_dim=64,
                out_dim=mobility_embedding_dim,
                num_layers=2,
                aggregator_type="mean",
                dropout=0.1,
            )
        else:
            self.mobility_gnn = None

        self.forecaster_head = create_forecaster_head(
            in_dim=self.forecaster_input_dim,
            d_model=128,
            n_heads=4,
            num_layers=3,
            horizon=forecast_horizon,
            dropout=0.1,
            device=device,
        )

        # Store parameter counts for logging
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Initialized EpiForecaster with variants: "
            f"regions={self.variant_type.regions}, mobility={self.variant_type.mobility}, "
            f"total_params={total_params:,}"
        )

    def forward(
        self,
        cases_hist: torch.Tensor,  # [batch_size, seq_len, num_regions]
        biomarkers_hist: torch.Tensor,  # [batch_size, seq_len, num_regions, biomarkers_dim]
        mobility_hist: torch.Tensor,  # [batch_size, seq_len, num_regions, num_regions]
        region_id: str = None,  # lookup from region_ids
        region_embeddings: torch.Tensor
        | None = None,  # [num_regions, region_embedding_dim]
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of three-layer EpiForecaster.

        Args:
            cases_hist: Historical case counts [batch_size, seq_len, num_regions]
            biomarkers_hist: Historical biomarker measurements [batch_size, seq_len, num_regions, biomarkers_dim]
            mobility_hist: Historical mobility matrices [batch_size, seq_len, num_regions, num_regions]
            region_ids: Region indices for batch samples [batch_size]
            region_embeddings: Static region embeddings [num_regions, region_embedding_dim]
            return_embeddings: Whether to return intermediate embeddings for analysis

        Returns:
            Forecasts [batch_size, forecast_horizon] for future time steps
            If return_embeddings=True, returns tuple (forecasts, embeddings_dict)
        """
        batch_size, seq_len, num_regions = cases_hist.shape

        # Validate input dimensions
        assert biomarkers_hist.shape == (
            seq_len,
            num_regions,
            self.biomarkers_dim,
        ), (
            f"biomarkers_hist shape mismatch: expected ({batch_size}, {seq_len}, {num_regions}, {self.biomarkers_dim})"
        )

        assert mobility_hist.shape == (batch_size, seq_len, num_regions, num_regions), (
            f"mobility_hist shape mismatch: expected ({batch_size}, {seq_len}, {num_regions}, {num_regions})"
        )

        embeddings_dict = {}

        # Step 1: Extract static region embeddings if enabled
        if self.variant_type.regions:
            assert region_id is not None, (
                "Region ID required when use_region_embeddings=True"
            )
            assert region_embeddings is not None, (
                "Region embeddings required when use_region_embeddings=True"
            )

            batch_region_embeddings = region_embeddings[
                region_id
            ]  # [batch_size, region_embedding_dim]
            embeddings_dict["region_embeddings"] = batch_region_embeddings
        else:
            # Create zero embeddings for disabled components
            batch_region_embeddings = (
                torch.zeros(
                    batch_size,
                    self.region_embedding_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.use_region_embeddings
                else None
            )

        # Step 2: Process each time step with MobilityGNN
        if self.use_mobility_embeddings and self.mobility_gnn is not None:
            # Process mobility for each time step
            mobility_embeddings = self._process_mobility_sequence(
                cases_hist, biomarkers_hist, mobility_hist, batch_region_embeddings
            )  # [batch_size, seq_len, mobility_embedding_dim]
            embeddings_dict["mobility_embeddings"] = mobility_embeddings
        else:
            # Create zero mobility embeddings
            mobility_embeddings = (
                torch.zeros(
                    batch_size,
                    seq_len,
                    self.mobility_embedding_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.use_mobility_embeddings
                else None
            )

        # Step 3: Build sequences for forecaster head
        # For each batch sample, we need sequences of local features
        forecaster_sequences = self._build_forecaster_sequences(
            cases_hist,
            biomarkers_hist,
            batch_region_embeddings,
            mobility_embeddings,
        )  # [batch_size, seq_len, forecaster_input_dim]

        # Step 4: Generate forecasts with Transformer
        forecasts = self.forecaster_head(
            forecaster_sequences
        )  # [batch_size, forecast_horizon]

        if return_embeddings:
            return forecasts, embeddings_dict
        else:
            return forecasts

    def forward_ego_graph(
        self,
        ego_graph_batch: dict[str, Any],
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for ego-graph processing.

        Each batch item contains one target region with its ego-graphs across time steps.
        This ensures true node-level processing where the model never sees the full graph.

        Args:
            ego_graph_batch: Dictionary containing:
                - cases_hist: [batch_size, L] historical cases for target regions
                - biomarkers_hist: [batch_size, L, F] historical biomarkers for target regions
                - node_features_t: List of L ego-graph node features per batch
                - edge_index_t: List of L ego-graph edge indices per batch
                - edge_weight_t: List of L ego-graph edge weights per batch
                - target_local_idx: List of L target local indices per batch
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Forecasts [batch_size, H] for each target region
        """
        # Validate all inputs are on the model device
        device = next(self.parameters()).device
        for key, value in ego_graph_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device, (
                    f"Tensor {key} on {value.device} but expected {device}"
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        assert item.device == device, (
                            f"Tensor {key} item on {item.device} but expected {device}"
                        )

        cases_hist = ego_graph_batch["cases_hist"]  # [batch_size, L]
        biomarkers_hist = ego_graph_batch["biomarkers_hist"]  # [batch_size, L, F]
        node_features_t = ego_graph_batch[
            "node_features_t"
        ]  # List of L [batch_size, N_t, 1]
        edge_index_t = ego_graph_batch["edge_index_t"]  # List of L [batch_size, 2, E_t]
        edge_weight_t = ego_graph_batch["edge_weight_t"]  # List of L [batch_size, E_t]
        target_local_idx = ego_graph_batch["target_local_idx"]  # List of L [batch_size]

        batch_size, L = cases_hist.shape
        H = self.forecast_horizon

        embeddings_dict = {}

        # Step 1: Process ego-graphs through time to get temporal embeddings
        temporal_embeddings = self._process_ego_graph_sequence(
            cases_hist,
            biomarkers_hist,
            node_features_t,
            edge_index_t,
            edge_weight_t,
            target_local_idx,
        )  # [batch_size, L, embed_dim]

        embeddings_dict["temporal_embeddings"] = temporal_embeddings

        # Step 2: Generate forecasts using ForecasterHead
        # ForecasterHead expects [batch_size, seq_len, input_dim]
        # For ego-graph case, input_dim is just the temporal embedding dimension
        forecasts = self.forecaster_head(temporal_embeddings)  # [batch_size, H]

        if return_embeddings:
            return forecasts, embeddings_dict
        else:
            return forecasts

    def _process_ego_graph_sequence(
        self,
        cases_hist: torch.Tensor,  # [batch_size, L]
        biomarkers_hist: torch.Tensor,  # [batch_size, L, F]
        node_features_t: list[torch.Tensor],  # List of L [batch_size, N_t, 1]
        edge_index_t: list[torch.Tensor],  # List of L [batch_size, 2, E_t]
        edge_weight_t: list[torch.Tensor],  # List of L [batch_size, E_t]
        target_local_idx: list[torch.Tensor],  # List of L [batch_idx]
    ) -> torch.Tensor:
        """
        Process ego-graph sequence to generate temporal embeddings.

        Args:
            cases_hist: Historical cases for target regions
            biomarkers_hist: Historical biomarkers for target regions
            node_features_t: Ego-graph node features per time step
            edge_index_t: Ego-graph edge indices per time step
            edge_weight_t: Ego-graph edge weights per time step
            target_local_idx: Target local indices per time step

        Returns:
            Temporal embeddings [batch_size, L, embed_dim]
        """
        batch_size, L = cases_hist.shape
        local_dim = self.cases_dim + self.biomarkers_dim

        # Get the actual device of the model, not the stored self.device
        actual_device = next(self.parameters()).device

        # Initialize temporal embeddings
        temporal_embeddings = torch.zeros(
            batch_size, L, local_dim, device=actual_device, dtype=torch.float32
        )

        # Process each time step
        for t in range(L):
            # Combine local features for target region using einops
            cases_t = rearrange(cases_hist[:, t], "batch -> batch 1")  # [batch_size, 1]
            biomarkers_t = biomarkers_hist[:, t, :]  # [batch_size, F]

            # Concatenate local features: [batch_size, local_dim]
            local_features_t = torch.cat([cases_t, biomarkers_t], dim=-1)

            # Ensure local_features_t is on the correct device
            local_features_t = local_features_t.to(actual_device)

            # Store as temporal embedding for this time step
            temporal_embeddings[:, t, :] = local_features_t

        return temporal_embeddings

    def _process_mobility_sequence(
        self,
        cases_hist: torch.Tensor,  # [batch_size, seq_len, num_regions]
        biomarkers_hist: torch.Tensor,  # [batch_size, seq_len, num_regions, biomarkers_dim]
        mobility_hist: torch.Tensor,  # [batch_size, seq_len, num_regions, num_regions]
        batch_region_embeddings: torch.Tensor
        | None,  # [batch_size, region_embedding_dim] or None
    ) -> torch.Tensor:
        """
        Process mobility-enhanced embeddings for the entire sequence.

        Args:
            cases_hist: Historical case counts
            biomarkers_hist: Historical biomarker measurements
            mobility_hist: Historical mobility matrices
            batch_region_embeddings: Static region embeddings for batch

        Returns:
            Mobility embeddings [batch_size, seq_len, mobility_embedding_dim]
        """
        batch_size, seq_len, num_regions = cases_hist.shape

        # Initialize output tensor
        mobility_embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.mobility_embedding_dim,
            device=self.device,
            dtype=torch.float32,
        )

        # Process each time step
        for t in range(seq_len):
            # Build node features for this time step
            # Cases for all regions at time t using einops
            cases_t = rearrange(
                cases_hist[:, t, :], "batch regions -> batch regions 1"
            )  # [batch_size, num_regions, 1]

            # Biomarkers for all regions at time t
            biomarkers_t = biomarkers_hist[
                :, t, :, :
            ]  # [batch_size, num_regions, biomarkers_dim]

            # Concatenate local features
            local_features_t = torch.cat(
                [cases_t, biomarkers_t], dim=-1
            )  # [batch_size, num_regions, local_dim]

            # Add region embeddings if available
            if batch_region_embeddings is not None and self.use_region_embeddings:
                # Expand region embeddings to all regions using einops
                # This is a simplification - in practice, you might have different region embeddings per region
                batch_expanded_embeddings = repeat(
                    batch_region_embeddings,
                    "batch features -> batch regions features",
                    regions=num_regions,
                )
                node_features_t = torch.cat(
                    [local_features_t, batch_expanded_embeddings], dim=-1
                )
            else:
                node_features_t = local_features_t

            # Process with MobilityGNN for each batch element
            for b in range(batch_size):
                mobility_matrix_t = mobility_hist[
                    b, t, :, :
                ]  # [num_regions, num_regions]
                node_features_b = node_features_t[b, :, :]  # [num_regions, in_dim]

                # Get region embeddings for this batch element
                region_embeddings_b = (
                    batch_region_embeddings[b : b + 1]
                    if batch_region_embeddings is not None
                    else None
                )

                # Apply MobilityGNN (forward processes all regions simultaneously)
                mobility_output_b = self.mobility_gnn(
                    node_features_b, mobility_matrix_t, region_embeddings_b
                )  # [num_regions, mobility_embedding_dim]

                # For now, we'll take the mean across regions to get a single embedding per batch element
                # In practice, you might want to keep per-region embeddings
                mobility_embeddings[b, t, :] = torch.mean(mobility_output_b, dim=0)

        return mobility_embeddings

    def _build_forecaster_sequences(
        self,
        cases_hist: torch.Tensor,  # [batch_size, seq_len, num_regions]
        biomarkers_hist: torch.Tensor,  # [batch_size, seq_len, num_regions, biomarkers_dim]
        batch_region_embeddings: torch.Tensor
        | None,  # [batch_size, region_embedding_dim] or None
        mobility_embeddings: torch.Tensor
        | None,  # [batch_size, seq_len, mobility_embedding_dim] or None
    ) -> torch.Tensor:
        """
        Build sequences for the forecaster head by concatenating features.

        Args:
            cases_hist: Historical case counts
            biomarkers_hist: Historical biomarker measurements
            batch_region_embeddings: Static region embeddings for batch
            mobility_embeddings: Mobility-enhanced embeddings
            region_ids: Region indices for batch samples

        Returns:
            Forecaster sequences [batch_size, seq_len, forecaster_input_dim]
        """
        batch_size, seq_len, num_regions = cases_hist.shape

        # Build local features (cases + biomarkers) for ALL regions using einops

        # cases_hist: [batch_size, seq_len, num_regions] -> [batch_size, seq_len, num_regions, 1]
        cases_with_dim = rearrange(
            cases_hist, "batch seq regions -> batch seq regions 1"
        )
        # biomarkers_hist: [batch_size, seq_len, num_regions, biomarkers_dim] -> keep as is

        # Concatenate along feature dimension: [batch_size, seq_len, num_regions, local_dim]
        local_features = torch.cat([cases_with_dim, biomarkers_hist], dim=-1)

        # Combine all features
        features_to_concat = [local_features]

        # Add mobility embeddings if enabled (need to match region dimension)
        if mobility_embeddings is not None and self.use_mobility_embeddings:
            # mobility_embeddings: [batch_size, seq_len, mobility_embedding_dim]
            # Expand to [batch_size, seq_len, num_regions, mobility_embedding_dim] using einops
            mobility_expanded = repeat(
                mobility_embeddings,
                "batch seq features -> batch seq regions features",
                regions=num_regions,
            )
            features_to_concat.append(mobility_expanded)
        else:
            # Zero embeddings for disabled mobility
            zero_mobility = torch.zeros(
                batch_size,
                seq_len,
                num_regions,
                self.mobility_embedding_dim,
                device=self.device,
                dtype=torch.float32,
            )
            if self.use_mobility_embeddings:
                features_to_concat.append(zero_mobility)

        # Add static region embeddings if enabled
        if batch_region_embeddings is not None and self.use_region_embeddings:
            # batch_region_embeddings: [batch_size, region_embedding_dim]
            # Expand to [batch_size, seq_len, num_regions, region_embedding_dim]
            region_expanded = rearrange(
                batch_region_embeddings, "batch features -> batch 1 features 1"
            ).expand(-1, seq_len, -1, -1)
            features_to_concat.append(region_expanded)
        else:
            # Zero embeddings for disabled regions
            zero_regions = torch.zeros(
                batch_size,
                seq_len,
                num_regions,
                self.region_embedding_dim,
                device=self.device,
                dtype=torch.float32,
            )
            if self.use_region_embeddings:
                features_to_concat.append(zero_regions)

        # Concatenate all features: [batch_size, seq_len, num_regions, forecaster_input_dim]
        forecaster_sequences = torch.cat(features_to_concat, dim=-1)

        # Rearrange for forecaster: [batch_size, num_regions, seq_len, forecaster_input_dim]
        forecaster_sequences = rearrange(
            forecaster_sequences,
            "batch seq regions features -> batch regions seq features",
        )

        return forecaster_sequences

    def get_variant_info(self) -> dict:
        """Get information about the current model variant configuration."""
        return {
            "use_region_embeddings": self.use_region_embeddings,
            "use_mobility_embeddings": self.use_mobility_embeddings,
            "cases_dim": self.cases_dim,
            "biomarkers_dim": self.biomarkers_dim,
            "region_embedding_dim": self.region_embedding_dim,
            "mobility_embedding_dim": self.mobility_embedding_dim,
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "forecaster_input_dim": self.forecaster_input_dim,
        }

    def get_forecast_horizon(self) -> int:
        """Get the forecasting horizon."""
        return self.forecast_horizon


def create_epiforecaster_variant(
    variant_type: str,
    cases_dim: int = 1,
    biomarkers_dim: int = 4,
    region_embedding_dim: int = 64,
    mobility_embedding_dim: int = 64,
    sequence_length: int = 14,
    forecast_horizon: int = 7,
    **kwargs,
) -> EpiForecaster:
    """
    Factory function to create specific EpiForecaster variants.

    Args:
        variant_type: Type of variant ('base', 'regions', 'mobility', 'full')
        cases_dim: Dimension of case features
        biomarkers_dim: Dimension of biomarker features
        region_embedding_dim: Dimension of region embeddings
        mobility_embedding_dim: Dimension of mobility embeddings
        sequence_length: Length of historical context
        forecast_horizon: Forecasting horizon
        **kwargs: Additional arguments

    Returns:
        Configured EpiForecaster instance
    """
    if variant_type == "base":
        return EpiForecaster(
            use_region_embeddings=False,
            use_mobility_embeddings=False,
            cases_dim=cases_dim,
            biomarkers_dim=biomarkers_dim,
            region_embedding_dim=region_embedding_dim,
            mobility_embedding_dim=mobility_embedding_dim,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            **kwargs,
        )
    elif variant_type == "regions":
        return EpiForecaster(
            use_region_embeddings=True,
            use_mobility_embeddings=False,
            cases_dim=cases_dim,
            biomarkers_dim=biomarkers_dim,
            region_embedding_dim=region_embedding_dim,
            mobility_embedding_dim=mobility_embedding_dim,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            **kwargs,
        )
    elif variant_type == "mobility":
        return EpiForecaster(
            use_region_embeddings=False,
            use_mobility_embeddings=True,
            cases_dim=cases_dim,
            biomarkers_dim=biomarkers_dim,
            region_embedding_dim=region_embedding_dim,
            mobility_embedding_dim=mobility_embedding_dim,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            **kwargs,
        )
    elif variant_type == "full":
        return EpiForecaster(
            use_region_embeddings=True,
            use_mobility_embeddings=True,
            cases_dim=cases_dim,
            biomarkers_dim=biomarkers_dim,
            region_embedding_dim=region_embedding_dim,
            mobility_embedding_dim=mobility_embedding_dim,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown variant type: {variant_type}")
