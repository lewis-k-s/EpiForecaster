import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from .configs import ModelVariant
from .forecaster_head import ForecasterHead
from .mobility_gnn import MobilityPyGEncoder

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
        temporal_input_dim: int = 1,
        biomarkers_dim: int = 1,
        region_embedding_dim: int = 64,
        mobility_embedding_dim: int = 64,
        gnn_depth: int = 2,
        sequence_length: int = 14,
        forecast_horizon: int = 7,
        use_population: bool = True,
        population_dim: int = 1,
        device: torch.device | None = None,
        gnn_module: str = "gcn",
    ):
        """
        Initialize EpiForecaster with three-layer architecture.

        Args:
            temporal_input_dim: Dimension of temporal input features (cases)
            biomarkers_dim: Dimension of biomarker features
            region_embedding_dim: Dimension of static region embeddings
            mobility_embedding_dim: Dimension of mobility embeddings
            gnn_module: GNN backbone for mobility graphs ('gcn' or 'gat')
            sequence_length: Length of historical context window
            forecast_horizon: Number of future time steps to forecast
            use_population: Whether to include static population feature
            population_dim: Dimension of the population feature (default 1)
            device: Device for tensor operations
        """
        super().__init__()

        self.variant_type = variant_type
        self.temporal_input_dim = temporal_input_dim
        self.biomarkers_dim = biomarkers_dim
        self.region_embedding_dim = region_embedding_dim
        self.mobility_embedding_dim = mobility_embedding_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.use_population = use_population
        self.population_dim = population_dim
        self.device = device or torch.device("cpu")
        self.gnn_module = gnn_module

        self.temporal_node_dim = 0
        if self.variant_type.cases:
            self.temporal_node_dim += temporal_input_dim
        if self.variant_type.biomarkers:
            self.temporal_node_dim += biomarkers_dim

        self.forecaster_input_dim = 0
        if self.variant_type.cases:
            self.forecaster_input_dim += temporal_input_dim
        if self.variant_type.biomarkers:
            self.forecaster_input_dim += biomarkers_dim
        if self.variant_type.mobility:
            self.forecaster_input_dim += mobility_embedding_dim
        if self.variant_type.regions:
            self.forecaster_input_dim += region_embedding_dim
        if self.use_population:
            self.forecaster_input_dim += population_dim

        # We always include target_mean as a feature now
        self.forecaster_input_dim += 1

        if self.variant_type.mobility:
            assert self.temporal_node_dim > 0, (
                "Mobility GNN requires temporal node features (cases/biomarkers)."
            )
            if gnn_module in {"gcn", "gat"}:
                self.mobility_gnn = MobilityPyGEncoder(
                    in_dim=self.temporal_node_dim,
                    hidden_dim=16,
                    out_dim=self.mobility_embedding_dim,
                    module_type=gnn_module,
                    dropout=0.1,
                    depth=gnn_depth,
                )
            else:
                raise ValueError(f"Unsupported GNN module: {gnn_module}")
                # # Fallback to legacy dense MobilityGNN (kept for backward compatibility)
                # self.mobility_gnn = MobilityGNN(
                #     in_dim=self.temporal_node_dim,
                #     hidden_dim=16,
                #     out_dim=self.mobility_embedding_dim,
                #     num_layers=gnn_depth,
                #     aggregator_type="mean",
                #     dropout=0.1,
                # )
        else:
            self.mobility_gnn = None

        self.forecaster_head = ForecasterHead(
            in_dim=self.forecaster_input_dim,
            d_model=96,
            n_heads=2,
            num_layers=2,
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
        cases_hist: torch.Tensor,
        biomarkers_hist: torch.Tensor,
        mob_graphs: Sequence[Sequence[Data]] | None,
        target_nodes: torch.Tensor,
        region_embeddings: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
        temporal_covariates: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of three-layer EpiForecaster.

        Args:
            cases_hist: Historical case counts [batch_size, seq_len, cases_dim]
            biomarkers_hist: Historical biomarker measurements [batch_size, seq_len, biomarkers_dim]
            mob_graphs: Per-batch list of per-time-step PyG graphs
            target_nodes: Indices of target nodes in the global region list [batch_size]
            region_embeddings: Optional static region embeddings [num_regions, region_embedding_dim]
            population: Optional per-node population [batch_size]
            temporal_covariates: Optional generic temporal covariates [batch_size, seq_len, k]
            target_mean: Optional mean of targets for unscaling hints [batch_size]

        Returns:
            Forecasts [batch_size, forecast_horizon] for future time steps
        """
        B, T, _ = cases_hist.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EpiForecaster.forward: B=%d T=%d cases=%s biomarkers=%s mobility=%s regions=%s",
                B,
                T,
                tuple(cases_hist.shape),
                tuple(biomarkers_hist.shape),
                self.variant_type.mobility,
                self.variant_type.regions,
            )

        features: list[torch.Tensor] = []

        if self.variant_type.cases:
            features.append(cases_hist)
        if self.variant_type.biomarkers:
            features.append(biomarkers_hist)

        if self.variant_type.mobility:
            assert mob_graphs is not None, "Mobility graphs required but not provided."
            if not isinstance(mob_graphs, Batch):
                raise TypeError("Expected mob_graphs to be a PyG Batch after collate.")
            mobility_embeddings = self._process_mobility_sequence_pyg(mob_graphs, B, T)
            features.append(mobility_embeddings)

        if self.variant_type.regions:
            assert region_embeddings is not None, (
                "Region embeddings required but not provided."
            )
            region_emb_batch = region_embeddings[target_nodes]  # (B, region_dim)
            region_emb_seq = region_emb_batch.unsqueeze(1).expand(-1, T, -1)
            features.append(region_emb_seq)

        if self.use_population:
            assert population is not None, (
                "Population feature enabled but not provided."
            )
            pop_seq = population.view(B, 1, -1).expand(-1, T, -1)
            features.append(pop_seq)

        if temporal_covariates is not None:
            features.append(temporal_covariates)

        if target_mean is not None:
            # Expand (B,) -> (B, T, 1)
            target_mean_seq = target_mean.view(B, 1, 1).expand(-1, T, -1)
            features.append(target_mean_seq)
        else:
            # Should practically always be provided, but handle fallback or error?
            # For now, let's assume if it is missing we might have issues if dimensionality expects it.
            # But since we incremented dimensionality in init, we MUST provide it.
            raise ValueError("target_mean is required")

        x_seq = torch.cat(features, dim=-1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EpiForecaster.forward: x_seq=%s", tuple(x_seq.shape))

        forecasts = self.forecaster_head(x_seq)
        assert forecasts.shape == (B, self.forecast_horizon), "forecasts shape mismatch"

        return forecasts

    def _process_mobility_sequence_pyg(
        self, mob_batch: Batch, B: int, T: int
    ) -> torch.Tensor:
        """Process mobility graphs (batched) with MobilityGNN."""

        if not self.variant_type.mobility or self.mobility_gnn is None:
            raise RuntimeError(
                "Mobility processing requested but MobilityGNN is not enabled."
            )

        if mob_batch.x.size(-1) != self.temporal_node_dim:
            raise ValueError(
                f"Mobility graph feature dim {mob_batch.x.size(-1)} "
                f"!= expected {self.temporal_node_dim}"
            )

        mob_batch = mob_batch.to(self.device)

        node_emb = self.mobility_gnn(
            mob_batch.x,
            mob_batch.edge_index,
            getattr(mob_batch, "edge_weight", None),
        )

        # Gather target embeddings via ptr offsets (start index per graph).
        # IMPORTANT: keep this fully tensorized to avoid `.item()` on CUDA tensors
        # which forces device synchronization and DtoH copies.
        ptr = mob_batch.ptr  # shape (num_graphs + 1,)
        num_graphs = ptr.numel() - 1
        expected_graphs = B * T
        if num_graphs != expected_graphs:
            raise ValueError(
                f"Mobility batch has {num_graphs} graphs, expected B*T={expected_graphs} "
                f"(B={B}, T={T})."
            )

        if hasattr(mob_batch, "target_index"):
            target_indices = mob_batch.target_index.reshape(-1)
        else:
            start = ptr[:-1]
            tgt_local = mob_batch.target_node.reshape(-1).to(start.device)
            target_indices = start + tgt_local

        target_embeddings = node_emb[target_indices]
        mobility_embeddings = target_embeddings.view(B, T, self.mobility_embedding_dim)

        return mobility_embeddings

    def __repr__(self) -> str:
        return (
            "EpiForecaster("
            f"gnn_module={self.gnn_module}, "
            f"forecaster_head={self.forecaster_head}, "
            f"device={self.device}, "
            f"variant_type={self.variant_type})"
        )
