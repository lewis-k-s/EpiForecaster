import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

# type: ignore[import-not-found] (PyTorch Geometric has incomplete type stubs)
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
        gnn_hidden_dim: int = 32,
        sequence_length: int = 14,
        forecast_horizon: int = 7,
        use_population: bool = True,
        population_dim: int = 1,
        device: torch.device | None = None,
        gnn_module: str = "gcn",
        head_d_model: int = 96,
        head_n_heads: int = 2,
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
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
            gnn_hidden_dim: Hidden dimension for GNN
            head_d_model: Transformer d_model
            head_n_heads: Transformer heads
            head_num_layers: Transformer layers
            head_dropout: Transformer dropout
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
            # We add mean and std as features
            self.forecaster_input_dim += 1
            self.forecaster_input_dim += 1
        if self.variant_type.biomarkers:
            self.forecaster_input_dim += biomarkers_dim
        if self.variant_type.mobility:
            self.forecaster_input_dim += mobility_embedding_dim
        if self.variant_type.regions:
            self.forecaster_input_dim += region_embedding_dim
        if self.use_population:
            self.forecaster_input_dim += population_dim

        if self.variant_type.mobility:
            assert self.temporal_node_dim > 0, (
                "Mobility GNN requires temporal node features (cases/biomarkers)."
            )
            if gnn_module in {"gcn", "gat"}:
                self.mobility_gnn = MobilityPyGEncoder(
                    in_dim=self.temporal_node_dim,
                    hidden_dim=gnn_hidden_dim,
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
            d_model=head_d_model,
            n_heads=head_n_heads,
            num_layers=head_num_layers,
            horizon=forecast_horizon,
            dropout=head_dropout,
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
        cases_norm: torch.Tensor,
        cases_mean: torch.Tensor,
        cases_std: torch.Tensor,
        biomarkers_hist: torch.Tensor,
        mob_graphs: Sequence[Sequence[Data]] | None,
        target_nodes: torch.Tensor,
        region_embeddings: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
        temporal_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of three-layer EpiForecaster.

        Args:
            cases_norm: Normalized case counts [batch_size, seq_len, cases_dim]
            cases_mean: Rolling mean of cases [batch_size, seq_len, 1]
            cases_std: Rolling std of cases [batch_size, seq_len, 1]
            biomarkers_hist: Historical biomarker measurements [batch_size, seq_len, biomarkers_dim]
            mob_graphs: Per-batch list of per-time-step PyG graphs
            target_nodes: Indices of target nodes in the global region list [batch_size]
            region_embeddings: Optional static region embeddings [num_regions, region_embedding_dim]
            population: Optional per-node population [batch_size]
            temporal_covariates: Optional generic temporal covariates [batch_size, seq_len, k]

        Returns:
            Forecasts [batch_size, forecast_horizon] for future time steps
        """
        B, T, _ = cases_norm.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EpiForecaster.forward: B=%d T=%d cases=%s biomarkers=%s mobility=%s regions=%s",
                B,
                T,
                tuple(cases_norm.shape),
                tuple(biomarkers_hist.shape),
                self.variant_type.mobility,
                self.variant_type.regions,
            )

        features: list[torch.Tensor] = []

        if self.variant_type.cases:
            features.append(cases_norm)
            features.append(cases_mean)
            features.append(cases_std)
        if self.variant_type.biomarkers:
            features.append(biomarkers_hist)

        if self.variant_type.mobility:
            assert mob_graphs is not None, "Mobility graphs required but not provided."
            if not isinstance(mob_graphs, Batch):
                raise TypeError(
                    f"Expected mob_graphs to be PyG Batch, got {type(mob_graphs)}"
                )
            mobility_embeddings = self._process_mobility_sequence_pyg(
                mob_graphs, B, T
            )
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

        x_seq = torch.cat(features, dim=-1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EpiForecaster.forward: x_seq=%s", tuple(x_seq.shape))

        forecasts = self.forecaster_head(x_seq)
        assert forecasts.shape == (B, self.forecast_horizon), "forecasts shape mismatch"

        return forecasts

    def forward_batch(
        self,
        *,
        batch_data: dict[str, Any],
        region_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic device transfers and non-blocking I/O.

        This is the preferred entry point for training/evaluation, as it handles
        all device transfers consistently with non-blocking transfers to reduce
        CPU-GPU sync time.

        Args:
            batch_data: Dict containing batch tensors (CaseNode, CaseMean, CaseStd,
                        BioNode, MobBatch, Population, Target, TargetMean, TargetScale,
                        TargetNode, WindowStart, B, T)
            region_embeddings: Optional static region embeddings [num_regions, region_dim]

        Returns:
            Tuple of (predictions, targets, target_mean, target_scale)
        """
        # Non-blocking transfer for all PyTorch tensors
        batch = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch_data.items()
            if isinstance(v, torch.Tensor)
        }

        # PyG Batch handles its own device transfer (not a torch.Tensor, so not in dict above)
        # Use non_blocking for consistency with tensor transfers
        mob_batch = batch_data["MobBatch"].to(self.device, non_blocking=True)

        target_nodes = batch.get("TargetRegionIndex", batch["TargetNode"])

        predictions = self.forward(
            cases_norm=batch["CaseNode"],
            cases_mean=batch["CaseMean"],
            cases_std=batch["CaseStd"],
            biomarkers_hist=batch["BioNode"],
            mob_graphs=mob_batch,
            target_nodes=target_nodes,
            region_embeddings=region_embeddings,
            population=batch["Population"],
        )

        return predictions, batch["Target"], batch["TargetMean"], batch["TargetScale"]

    def _process_mobility_sequence_pyg_list(
        self, mob_graphs_list: list[Batch], B: int, T: int
    ) -> torch.Tensor:
        """Process list of per-time-step batches (Curriculum format)."""
        if not self.variant_type.mobility or self.mobility_gnn is None:
            raise RuntimeError(
                "Mobility processing requested but MobilityGNN is not enabled."
            )
        
        if len(mob_graphs_list) != T:
            raise ValueError(
                f"Expected {T} mobility batches (one per time step), got {len(mob_graphs_list)}"
            )

        embeddings_list = []
        for t, batch_t in enumerate(mob_graphs_list):
            batch_t = batch_t.to(self.device)
            
            # GNN Forward
            node_emb = self.mobility_gnn(
                batch_t.x, 
                batch_t.edge_index, 
                getattr(batch_t, "edge_weight", None)
            )
            
            # Gather targets
            # Target index should be precomputed in collate_fn
            if hasattr(batch_t, "target_index"):
                 target_indices = batch_t.target_index.reshape(-1)
            else:
                 # Fallback
                 ptr = batch_t.ptr
                 start = ptr[:-1]
                 tgt_local = batch_t.target_node.reshape(-1).to(start.device)
                 target_indices = start + tgt_local
            
            target_emb = node_emb[target_indices]  # (B, dim)
            embeddings_list.append(target_emb)
            
        # Stack -> (B, T, dim)
        mobility_embeddings = torch.stack(embeddings_list, dim=1)
        return mobility_embeddings

    def _process_mobility_sequence_pyg(
        self, mob_batch: Batch, B: int, T: int
    ) -> torch.Tensor:
        """Process mobility graphs (batched) with MobilityGNN."""

        if not self.variant_type.mobility or self.mobility_gnn is None:
            raise RuntimeError(
                "Mobility processing requested but MobilityGNN is not enabled."
            )

        # type: ignore[attr-defined] (PyTorch Geometric Batch class missing type stubs)
        if mob_batch.x.size(-1) != self.temporal_node_dim:  # type: ignore[attr-defined]
            raise ValueError(  # type: ignore[attr-defined]
                f"Mobility graph feature dim {mob_batch.x.size(-1)} "  # type: ignore[attr-defined]
                f"!= expected {self.temporal_node_dim}"
            )

        # Note: mob_batch is already transferred to device in trainer._forward_batch
        # Redundant .to(self.device) call removed to avoid device synchronization

        node_emb = self.mobility_gnn(
            mob_batch.x,  # type: ignore[attr-defined]
            mob_batch.edge_index,  # type: ignore[attr-defined]
            getattr(mob_batch, "edge_weight", None),
        )

        # Gather target embeddings via ptr offsets (start index per graph).
        # IMPORTANT: keep this fully tensorized to avoid `.item()` on CUDA tensors
        # which forces device synchronization and DtoH copies.
        ptr = mob_batch.ptr  # type: ignore[attr-defined]
        num_graphs = ptr.numel() - 1
        expected_graphs = B * T
        if num_graphs != expected_graphs:
            raise ValueError(
                f"Mobility batch has {num_graphs} graphs, expected B*T={expected_graphs} "
                f"(B={B}, T={T})."
            )

        if hasattr(mob_batch, "target_index"):
            target_indices = mob_batch.target_index.reshape(-1)  # type: ignore[attr-defined]
        else:
            start = ptr[:-1]
            tgt_local = mob_batch.target_node.reshape(-1).to(start.device)  # type: ignore[attr-defined]
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
