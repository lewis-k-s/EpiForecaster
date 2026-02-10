import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# type: ignore[import-not-found] (PyTorch Geometric has incomplete type stubs)
from torch_geometric.data import Batch

from .configs import ModelVariant, ObservationHeadConfig, SIRPhysicsConfig
from .mobility_gnn import MobilityPyGEncoder
from .observation_heads import ClinicalObservationHead, WastewaterObservationHead
from .sir_rollforward import SIRRollForward
from .transformer_backbone import TransformerBackbone

logger = logging.getLogger(__name__)


class EpiForecaster(nn.Module):
    """
    Joint Inference-Observation epidemiological forecaster.

    Three-stage architecture:
    1. Encoder (TransformerBackbone): Estimates beta_t, initial states, obs_context
    2. Physics Core (SIRRollForward): Generates latent SIR trajectories
    3. Observation Heads: Map latent I(t) to observable signals (WW, Hosp)
    """

    def __init__(
        self,
        variant_type: ModelVariant,
        sir_physics: SIRPhysicsConfig,
        observation_heads: ObservationHeadConfig,
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
        Initialize EpiForecaster with joint inference architecture.

        Args:
            variant_type: ModelVariant with flags for cases/regions/biomarkers/mobility
            sir_physics: SIR physics configuration
            observation_heads: Observation head configuration
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
        self.sir_physics = sir_physics
        self.observation_heads_config = observation_heads
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

        # Compute dimensions using helpers
        self.temporal_node_dim = self._get_temporal_node_dim()
        self.backbone_input_dim = self._get_backbone_input_dim()

        # Stage 1: Mobility GNN (optional)
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
        else:
            self.mobility_gnn = None

        # Stage 1: Transformer Backbone (encoder)
        self.backbone = TransformerBackbone(
            in_dim=self.backbone_input_dim,
            d_model=head_d_model,
            n_heads=head_n_heads,
            num_layers=head_num_layers,
            horizon=forecast_horizon,
            dropout=head_dropout,
            device=device,
            obs_context_dim=observation_heads.obs_context_dim,
        )

        # Stage 2: SIR Roll-Forward (physics core)
        self.sir_rollforward = SIRRollForward(
            dt=sir_physics.dt,
            enforce_nonnegativity=sir_physics.enforce_nonnegativity,
            enforce_mass_conservation=sir_physics.enforce_mass_conservation,
        )

        # Stage 3: Observation Heads
        # Wastewater head
        self.ww_head = WastewaterObservationHead(
            kernel_length=observation_heads.kernel_length_ww,
            scale_init=1.0,  # Will be learned if learnable_scale_ww=True
            learnable_scale=observation_heads.learnable_scale_ww,
            learnable_kernel=observation_heads.learnable_kernel_ww,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
        )

        # Hospitalization head
        self.hosp_head = ClinicalObservationHead(
            kernel_length=observation_heads.kernel_length_hosp,
            scale_init=0.05,  # ~5% hospitalization rate
            learnable_scale=observation_heads.learnable_scale_hosp,
            learnable_kernel=observation_heads.learnable_kernel_hosp,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
        )

        # Cases head (reported cases observation)
        self.cases_head = ClinicalObservationHead(
            kernel_length=observation_heads.kernel_length_cases,
            scale_init=0.3,  # ~30% ascertainment/reporting rate
            learnable_scale=observation_heads.learnable_scale_cases,
            learnable_kernel=observation_heads.learnable_kernel_cases,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
        )

        # Deaths head (mortality observation)
        # Input is death_flow (fraction dying), so scale is reporting rate (start at 0.5)
        self.deaths_head = ClinicalObservationHead(
            kernel_length=observation_heads.kernel_length_deaths,
            scale_init=0.5,
            learnable_scale=observation_heads.learnable_scale_deaths,
            learnable_kernel=observation_heads.learnable_kernel_deaths,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
        )

        # Store parameter counts for logging
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Initialized EpiForecaster (Joint Inference) with variants: "
            f"regions={self.variant_type.regions}, mobility={self.variant_type.mobility}, "
            f"total_params={total_params:,}"
        )

    def forward(
        self,
        hosp_hist: torch.Tensor,
        deaths_hist: torch.Tensor,
        cases_hist: torch.Tensor,
        biomarkers_hist: torch.Tensor,
        mob_graphs: Batch | None,
        target_nodes: torch.Tensor,
        region_embeddings: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
        temporal_covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of joint inference EpiForecaster.

        Args:
            hosp_hist: Hospitalization history [batch_size, seq_len, 3] (value, mask, age)
            deaths_hist: Deaths history [batch_size, seq_len, 3] (value, mask, age)
            cases_hist: Reported cases history [batch_size, seq_len, 3] (value, mask, age)
            biomarkers_hist: Historical biomarker measurements [batch_size, seq_len, biomarkers_dim]
            mob_graphs: PyG Batch of mobility graphs (B*T graphs)
            target_nodes: Indices of target nodes in the global region list [batch_size]
            region_embeddings: Optional static region embeddings [num_regions, region_embedding_dim]
            population: Optional per-node population [batch_size]
            temporal_covariates: Optional generic temporal covariates [batch_size, seq_len, k]

        Returns:
            Dictionary containing:
                - beta_t: [batch_size, horizon] - transmission rates
                - S_trajectory: [batch_size, horizon+1] - susceptible trajectory
                - I_trajectory: [batch_size, horizon+1] - infected trajectory (latent)
                - R_trajectory: [batch_size, horizon+1] - recovered trajectory
                - physics_residual: [batch_size, horizon] - SIR dynamics residual
                - pred_ww: [batch_size, horizon] - predicted wastewater signal
                - pred_hosp: [batch_size, horizon] - predicted hospitalizations
                - obs_context: [batch_size, horizon, obs_context_dim] - observation context
                - initial_states: [batch_size, 3] - S0, I0, R0 proportions (sum=1)
        """
        B, T, _ = hosp_hist.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EpiForecaster.forward: B=%d T=%d clinical=%s biomarkers=%s mobility=%s regions=%s",
                B,
                T,
                (B, T, 9),  # 3 series x 3 channels
                tuple(biomarkers_hist.shape),
                self.variant_type.mobility,
                self.variant_type.regions,
            )

        # Build input features for backbone
        features: list[torch.Tensor] = []

        if self.variant_type.cases:
            # Concatenate all clinical series: hosp + deaths + cases
            # Each is (B, T, 3) with [value, mask, age]
            features.append(hosp_hist)
            features.append(deaths_hist)
            features.append(cases_hist)
        if self.variant_type.biomarkers:
            features.append(biomarkers_hist)

        if self.variant_type.mobility:
            assert mob_graphs is not None, "Mobility graphs required but not provided."
            if not isinstance(mob_graphs, Batch):
                raise TypeError(
                    f"Expected mob_graphs to be PyG Batch, got {type(mob_graphs)}"
                )
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

        x_seq = torch.cat(features, dim=-1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EpiForecaster.forward: x_seq=%s", tuple(x_seq.shape))

        # Stage 1: Encoder outputs SIR parameters and observation context
        encoder_outputs = self.backbone(x_seq)
        beta_t = encoder_outputs["beta_t"]  # [B, H]
        mortality_t = encoder_outputs["mortality_t"]  # [B, H]
        gamma_t = encoder_outputs["gamma_t"]  # [B, H]
        initial_states_logits = encoder_outputs["initial_states_logits"]  # [B, 3]
        obs_context = encoder_outputs["obs_context"]  # [B, H, C_obs]

        # Convert logits to proportions (softmax ensures sum=1, nonnegative)
        initial_states = F.softmax(initial_states_logits, dim=-1)  # [B, 3]

        # Keep SIR states in fraction space (sum to 1 per sample)
        S0 = initial_states[:, 0]  # [B]
        I0 = initial_states[:, 1]  # [B]
        R0 = initial_states[:, 2]  # [B]

        # Stage 2: SIR Roll-Forward
        sir_outputs = self.sir_rollforward(
            beta_t=beta_t,
            gamma_t=gamma_t,
            mortality_t=mortality_t,
            S0=S0,
            I0=I0,
            R0=R0,
            # SIR is modeled in fraction space (S+I+R=1), so population N=1.0 ensures
            # the standard SIR equations (beta*S*I/N) work correctly with fractions.
            population=torch.ones(B, device=beta_t.device),
        )

        S_traj = sir_outputs["S_trajectory"]  # [B, H+1]
        I_traj = sir_outputs["I_trajectory"]  # [B, H+1]
        R_traj = sir_outputs["R_trajectory"]  # [B, H+1]
        death_flow = sir_outputs["death_flow"]  # [B, H]
        physics_residual = sir_outputs["physics_residual"]  # [B, H]

        # Stage 3: Observation Heads
        # Note: I_traj includes I0 at index 0, so we use all of it
        # The heads expect [B, time_steps] and output [B, time_steps]

        # Observation context needs to align with I trajectory length (H+1).
        # Use the first forecast context as a proxy for t=0.
        obs_context_with_init = torch.cat([obs_context[:, :1, :], obs_context], dim=1)

        # Wastewater prediction
        # SIR is modeled in fraction space (S+I+R=1), so population N=1.0
        pred_ww = self.ww_head(
            I_trajectory=I_traj,
            population=torch.ones(B, device=beta_t.device),
            obs_context=obs_context_with_init,
        )  # [B, H+1]

        # Hospitalization prediction
        pred_hosp = self.hosp_head(
            I_trajectory=I_traj,
            obs_context=obs_context_with_init,
        )  # [B, H+1]

        # Cases prediction (reported cases observation)
        pred_cases = self.cases_head(
            I_trajectory=I_traj,
            obs_context=obs_context_with_init,
        )  # [B, H+1]

        # Deaths prediction (mortality observation)
        # Input is death_flow (fraction dying)
        pred_deaths = self.deaths_head(
            I_trajectory=death_flow,
            obs_context=obs_context,
        )  # [B, H]

        # Trim initial state (t=0) from predictions to match horizon
        pred_ww = self._trim_prediction(pred_ww)
        pred_hosp = self._trim_prediction(pred_hosp)
        pred_cases = self._trim_prediction(pred_cases)
        pred_deaths = self._trim_prediction(pred_deaths)

        return {
            "beta_t": beta_t,
            "S_trajectory": S_traj,
            "I_trajectory": I_traj,
            "R_trajectory": R_traj,
            "physics_residual": physics_residual,
            "pred_ww": pred_ww,
            "pred_hosp": pred_hosp,
            "pred_cases": pred_cases,
            "pred_deaths": pred_deaths,
            "obs_context": obs_context,
            "initial_states": initial_states,
        }

    def forward_batch(
        self,
        *,
        batch_data: dict[str, Any],
        region_embeddings: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """
        Forward pass with automatic device transfers and non-blocking I/O.

        This is the preferred entry point for training/evaluation, as it handles
        all device transfers consistently with non-blocking transfers to reduce
        CPU-GPU sync time.

        Args:
            batch_data: Dict containing batch tensors (HospHist, DeathsHist, CasesHist,
                        BioNode, MobBatch, Population, TargetNode, WindowStart, B, T)
            region_embeddings: Optional static region embeddings [num_regions, region_dim]

        Returns:
            Tuple of (model_outputs, targets_dict) where:
                - model_outputs: Dict from forward() with predictions and latents
                - targets_dict: Dict with target tensors for loss computation
        """
        # Non-blocking transfer for all PyTorch tensors
        batch = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch_data.items()
            if isinstance(v, torch.Tensor)
        }

        # PyG Batch handles its own device transfer
        mob_batch = batch_data["MobBatch"].to(self.device, non_blocking=True)

        target_nodes = batch.get("TargetRegionIndex", batch["TargetNode"])

        # Forward pass
        model_outputs = self.forward(
            hosp_hist=batch["HospHist"],
            deaths_hist=batch["DeathsHist"],
            cases_hist=batch["CasesHist"],
            biomarkers_hist=batch["BioNode"],
            mob_graphs=mob_batch,
            target_nodes=target_nodes,
            region_embeddings=region_embeddings,
            population=batch["Population"],
        )

        # Prepare targets dict
        # Joint inference targets: WW, Hosp, Cases, Deaths with per-target masks
        targets_dict = {
            "ww": batch.get("WWTarget"),
            "hosp": batch.get("HospTarget"),
            "cases": batch.get("CasesTarget"),
            "deaths": batch.get("DeathsTarget"),
            "ww_mask": batch.get("WWTargetMask"),
            "hosp_mask": batch.get("HospTargetMask"),
            "cases_mask": batch.get("CasesTargetMask"),
            "deaths_mask": batch.get("DeathsTargetMask"),
        }

        return model_outputs, targets_dict

    def _process_mobility_sequence_pyg(
        self, mob_batch: Batch, B: int, T: int
    ) -> torch.Tensor:
        """Process mobility graphs (batched) with MobilityGNN."""
        if not self.variant_type.mobility or self.mobility_gnn is None:
            raise RuntimeError(
                "Mobility processing requested but MobilityGNN is not enabled."
            )

        node_emb = self._get_mobility_node_embeddings(mob_batch)
        mobility_embeddings = self._gather_target_mobility_embeddings(
            node_emb, mob_batch, B, T
        )
        return mobility_embeddings

    def _get_temporal_node_dim(self) -> int:
        """Compute temporal node dimension based on enabled variants."""
        dim = 0
        if self.variant_type.cases:
            dim += self.temporal_input_dim
        if self.variant_type.biomarkers:
            dim += self.biomarkers_dim
        return dim

    def _get_backbone_input_dim(self) -> int:
        """Compute backbone input dimension based on enabled variants."""
        dim = 0
        if self.variant_type.cases:
            # 3 clinical series (hosp, deaths, cases) x 3 channels each = 9
            dim += 9
        if self.variant_type.biomarkers:
            dim += self.biomarkers_dim
        if self.variant_type.mobility:
            dim += self.mobility_embedding_dim
        if self.variant_type.regions:
            dim += self.region_embedding_dim
        if self.use_population:
            dim += self.population_dim
        return dim

    def _trim_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Trim prediction to match forecast horizon, removing t=0 if present."""
        if prediction.shape[1] > self.forecast_horizon:
            return prediction[:, 1 : 1 + self.forecast_horizon]
        return prediction

    def _get_mobility_node_embeddings(self, mob_batch: Batch) -> torch.Tensor:
        """Run GNN encoder on mobility batch."""
        # type: ignore[attr-defined]
        if mob_batch.x.size(-1) != self.temporal_node_dim:  # type: ignore[attr-defined]
            raise ValueError(
                f"Mobility graph feature dim {mob_batch.x.size(-1)} "  # type: ignore[attr-defined]
                f"!= expected {self.temporal_node_dim}"
            )

        return self.mobility_gnn(
            mob_batch.x,  # type: ignore[attr-defined]
            mob_batch.edge_index,  # type: ignore[attr-defined]
            getattr(mob_batch, "edge_weight", None),
        )

    def _gather_target_mobility_embeddings(
        self, node_emb: torch.Tensor, mob_batch: Batch, B: int, T: int
    ) -> torch.Tensor:
        """Gather embeddings for target nodes from the full graph batch."""
        # Check batch size consistency
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
        return target_embeddings.view(B, T, self.mobility_embedding_dim)

    def __repr__(self) -> str:
        return (
            "EpiForecaster("
            f"gnn_module={self.gnn_module}, "
            f"backbone={self.backbone}, "
            f"device={self.device}, "
            f"variant_type={self.variant_type})"
        )
