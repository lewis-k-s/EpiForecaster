import logging

import torch
import torch.nn as nn

from data.epi_batch import EpiBatch
import torch.nn.functional as F

# type: ignore[import-not-found] (PyTorch Geometric has incomplete type stubs)
from torch_geometric.data import Batch

from utils.precision_policy import MODEL_PARAM_DTYPE
from .configs import (
    InitWeightsConfig,
    ModelVariant,
    ObservationHeadConfig,
    SIRPhysicsConfig,
)
from .anchor_utils import resolve_last_valid_anchor
from .mobility_gnn import MobilityDenseEncoder
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
        init_weights: InitWeightsConfig | None = None,
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
        head_positional_encoding: str = "sinusoidal",
        temporal_covariates_dim: int = 0,
        strict: bool = True,
    ):
        """
        Initialize EpiForecaster with joint inference architecture.

        Args:
            variant_type: ModelVariant with flags for cases/regions/biomarkers/mobility
            sir_physics: SIR physics configuration
            init_weights: Initialization controls for startup dynamics
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
            head_positional_encoding: Transformer positional encoding
            temporal_covariates_dim: Dimension of temporal covariates (0=disabled, 3=dow_sin/cos+holiday)
        """
        super().__init__()

        self.variant_type = variant_type
        self.sir_physics = sir_physics
        self.init_weights = init_weights or InitWeightsConfig()
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
        self.dtype = MODEL_PARAM_DTYPE
        self.temporal_covariates_dim = temporal_covariates_dim
        self.head_positional_encoding = head_positional_encoding
        self.strict = strict

        # Compute dimensions using helpers
        self.temporal_node_dim = self._get_temporal_node_dim()
        self.backbone_input_dim = self._get_backbone_input_dim()

        # Stage 1: Mobility GNN (optional)
        if self.variant_type.mobility:
            assert self.temporal_node_dim > 0, (
                "Mobility GNN requires temporal node features (cases/biomarkers)."
            )
            gnn_in_dim = self.temporal_node_dim
            if self.variant_type.regions:
                gnn_in_dim += self.region_embedding_dim

            if gnn_module in {"gcn", "gat"}:
                self.mobility_gnn = MobilityDenseEncoder(
                    in_dim=gnn_in_dim,
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
            positional_encoding=head_positional_encoding,
            rezero_init=self.init_weights.rezero_init,
            rate_head_final_gain=self.init_weights.rate_head_final_gain,
            initial_state_final_gain=self.init_weights.initial_state_final_gain,
            obs_context_final_gain=self.init_weights.obs_context_final_gain,
            device=device,
            obs_context_dim=observation_heads.obs_context_dim,
            sir_physics=sir_physics,
        )

        # Stage 2: SIR Roll-Forward (physics core)
        self.sir_rollforward = SIRRollForward(
            dt=sir_physics.dt,
            enforce_nonnegativity=sir_physics.enforce_nonnegativity,
            enforce_mass_conservation=sir_physics.enforce_mass_conservation,
            residual_clip=sir_physics.residual_clip,
            strict=self.strict,
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
            strict=self.strict,
            delta_forecasting=observation_heads.delta_forecasting,
            anchor_mode=observation_heads.anchor_mode,
        )

        # Hospitalization head
        self.hosp_head = ClinicalObservationHead(
            kernel_length=observation_heads.kernel_length_hosp,
            scale_init=0.05,  # ~5% hospitalization rate
            learnable_scale=observation_heads.learnable_scale_hosp,
            learnable_kernel=observation_heads.learnable_kernel_hosp,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
            delta_forecasting=observation_heads.delta_forecasting,
            anchor_mode=observation_heads.anchor_mode,
        )

        # Cases head (reported cases observation)
        self.cases_head = ClinicalObservationHead(
            kernel_length=observation_heads.kernel_length_cases,
            scale_init=0.3,  # ~30% ascertainment/reporting rate
            learnable_scale=observation_heads.learnable_scale_cases,
            learnable_kernel=observation_heads.learnable_kernel_cases,
            residual_dim=observation_heads.obs_context_dim,
            alpha_init=observation_heads.residual_scale,
            delta_forecasting=observation_heads.delta_forecasting,
            anchor_mode=observation_heads.anchor_mode,
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
            delta_forecasting=False, # Do not anchor deaths since death_flow lacks t=0 nowcast
            anchor_mode="disabled",
        )

        # Cast all parameters to the canonical model parameter dtype.
        self.to(MODEL_PARAM_DTYPE)

        # Store parameter counts for logging
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Initialized EpiForecaster (Joint Inference) with variants: "
            f"regions={self.variant_type.regions}, mobility={self.variant_type.mobility}, "
            f"dtype={self.dtype}, total_params={total_params:,}"
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
        ww_hist: torch.Tensor | None = None,
        ww_hist_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of joint inference EpiForecaster.

        Args:
            hosp_hist: Hospitalization history [batch_size, seq_len, 3] (value, mask, age)
            deaths_hist: Deaths history [batch_size, seq_len, 3] (value, mask, age)
            cases_hist: Reported cases history [batch_size, seq_len, 3] (value, mask, age)
            biomarkers_hist: Historical biomarker measurements [batch_size, seq_len, biomarkers_dim]
                using `[value, mask, censor, age] * variants + has_data` layout.
            ww_hist: Optional wastewater history [batch_size, seq_len] in target log1p space.
            ww_hist_mask: Optional wastewater history mask [batch_size, seq_len] in target space.
            mob_graphs: PyG Batch-like mobility container (B*T dense graphs)
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

        if not torch.compiler.is_compiling() and logger.isEnabledFor(logging.DEBUG):
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
            self._validate_biomarker_layout(biomarkers_hist)
            features.append(biomarkers_hist)

        if self.variant_type.mobility:
            assert mob_graphs is not None, "Mobility graphs required but not provided."
            if not isinstance(mob_graphs, Batch):
                raise TypeError(
                    f"Expected mob_graphs to be PyG Batch, got {type(mob_graphs)}"
                )
            mobility_embeddings = self._process_mobility_sequence_pyg(
                mob_graphs, B, T, region_embeddings
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
            # Use log1p scaling to avoid projection instability from large raw counts.
            # Convert to float32 for log1p to handle large values, then to model dtype.
            pop_feature = torch.log1p(population.float()).to(hosp_hist.dtype)
            pop_seq = pop_feature.view(B, 1, -1).expand(-1, T, -1)
            features.append(pop_seq)

        if temporal_covariates is not None:
            features.append(temporal_covariates)

        x_seq = torch.cat(features, dim=-1)

        if not torch.compiler.is_compiling() and logger.isEnabledFor(logging.DEBUG):
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
            population=torch.ones(B, device=beta_t.device, dtype=beta_t.dtype),
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

        hosp_last = hosp_last_mask = None
        cases_last = cases_last_mask = None
        ww_last = ww_last_mask = None

        if (
            self.observation_heads_config.delta_forecasting
            and self.observation_heads_config.anchor_mode != "disabled"
        ):
            if self.variant_type.cases:
                hosp_last, hosp_last_mask = resolve_last_valid_anchor(
                    hosp_hist[:, :, 0], hosp_hist[:, :, 1]
                )
                cases_last, cases_last_mask = resolve_last_valid_anchor(
                    cases_hist[:, :, 0], cases_hist[:, :, 1]
                )

            if self.variant_type.biomarkers:
                if ww_hist is None or ww_hist_mask is None:
                    raise ValueError(
                        "ww_hist and ww_hist_mask are required for wastewater anchoring "
                        "when biomarkers and delta forecasting are enabled."
                    )
                ww_last, ww_last_mask = resolve_last_valid_anchor(ww_hist, ww_hist_mask)

        # Wastewater prediction
        # SIR is modeled in fraction space (S+I+R=1), so population N=1.0
        pred_ww = self.ww_head(
            I_trajectory=I_traj,
            population=torch.ones(B, device=beta_t.device, dtype=beta_t.dtype),
            obs_context=obs_context_with_init,
            last_observed=ww_last,
            last_observed_mask=ww_last_mask,
        )  # [B, H+1]

        # Hospitalization prediction
        pred_hosp = self.hosp_head(
            I_trajectory=I_traj,
            obs_context=obs_context_with_init,
            last_observed=hosp_last,
            last_observed_mask=hosp_last_mask,
        )  # [B, H+1]

        # Cases prediction (reported cases observation)
        pred_cases = self.cases_head(
            I_trajectory=I_traj,
            obs_context=obs_context_with_init,
            last_observed=cases_last,
            last_observed_mask=cases_last_mask,
        )  # [B, H+1]

        # Deaths prediction (mortality observation)
        # Input is death_flow (fraction dying)
        # Note: death_flow lacks t=0 nowcast, so we don't anchor it with delta_forecasting
        pred_deaths = self.deaths_head(
            I_trajectory=death_flow,
            obs_context=obs_context,
            last_observed=None,
        )  # [B, H]

        # Note: Predictions include t=0 (nowcast) and are [B, H+1] or [B, H].
        # Trainer/eval should slice as needed.

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
        batch_data: EpiBatch,
        region_embeddings: torch.Tensor | None = None,
        skip_device_transfer: bool = False,
        mask_cases: bool = False,
        mask_ww: bool = False,
        mask_hosp: bool = False,
        mask_deaths: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """
        Forward pass with automatic device transfers and non-blocking I/O.

        This is the preferred entry point for training/evaluation, as it handles
        all device transfers consistently with non-blocking transfers to reduce
        CPU-GPU sync time.

        Args:
            batch_data: EpiBatch containing batch tensors
            region_embeddings: Optional static region embeddings [num_regions, region_dim]
            skip_device_transfer: If True, assume all tensors are already on the correct
                                 device and dtype. Used with compiled training to avoid
                                 DeviceCopy ops that break CUDA graphs.

        Returns:
            Tuple of (model_outputs, targets_dict) where:
                - model_outputs: Dict from forward() with predictions and latents
                - targets_dict: Dict with target tensors for loss computation
        """
        if not skip_device_transfer:
            batch_data = batch_data.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )

        # Mask ablated inputs directly on the batch object
        if mask_ww and batch_data.bio_node is not None:
            batch_data.bio_node = torch.zeros_like(batch_data.bio_node)
            batch_data.ww_hist = torch.zeros_like(batch_data.ww_hist)
            batch_data.ww_hist_mask = torch.zeros_like(batch_data.ww_hist_mask)
        if mask_hosp and batch_data.hosp_hist is not None:
            batch_data.hosp_hist = torch.zeros_like(batch_data.hosp_hist)
        if mask_cases and batch_data.cases_hist is not None:
            batch_data.cases_hist = torch.zeros_like(batch_data.cases_hist)
        if mask_deaths and batch_data.deaths_hist is not None:
            batch_data.deaths_hist = torch.zeros_like(batch_data.deaths_hist)

        mob_batch = batch_data.mob_batch

        # Convert graph tensors to model dtype.
        if hasattr(mob_batch, "x_dense") and mob_batch.x_dense is not None:
            if mob_batch.x_dense.dtype != self.dtype:
                mob_batch.x_dense = mob_batch.x_dense.to(self.dtype)
        if hasattr(mob_batch, "adj_dense") and mob_batch.adj_dense is not None:
            if mob_batch.adj_dense.dtype != self.dtype:
                mob_batch.adj_dense = mob_batch.adj_dense.to(self.dtype)

        target_nodes = (
            batch_data.target_region_index
            if batch_data.target_region_index is not None
            else batch_data.target_node
        )

        # Extract temporal covariates if present
        temporal_covariates = batch_data.temporal_covariates

        # Forward pass
        model_outputs = self.forward(
            hosp_hist=batch_data.hosp_hist,
            deaths_hist=batch_data.deaths_hist,
            cases_hist=batch_data.cases_hist,
            biomarkers_hist=batch_data.bio_node,
            ww_hist=batch_data.ww_hist,
            ww_hist_mask=batch_data.ww_hist_mask,
            mob_graphs=mob_batch,
            target_nodes=target_nodes,
            region_embeddings=region_embeddings,
            population=batch_data.population,
            temporal_covariates=temporal_covariates,
        )

        # Prepare targets dict
        # Joint inference targets: WW, Hosp, Cases, Deaths with per-target masks
        targets_dict = {
            "ww": batch_data.ww_target,
            "hosp": batch_data.hosp_target,
            "cases": batch_data.cases_target,
            "deaths": batch_data.deaths_target,
            "ww_mask": batch_data.ww_target_mask,
            "hosp_mask": batch_data.hosp_target_mask,
            "cases_mask": batch_data.cases_target_mask,
            "deaths_mask": batch_data.deaths_target_mask,
        }

        return model_outputs, targets_dict

    def _process_mobility_sequence_pyg(
        self,
        mob_batch: Batch,
        B: int,
        T: int,
        region_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process mobility graphs (batched) with MobilityGNN."""
        if not self.variant_type.mobility or self.mobility_gnn is None:
            raise RuntimeError(
                "Mobility processing requested but MobilityGNN is not enabled."
            )

        node_emb = self._get_mobility_node_embeddings(mob_batch, region_embeddings)
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

    @staticmethod
    def _validate_biomarker_layout(biomarkers_hist: torch.Tensor) -> None:
        """Validate `[value, mask, censor, age] * variants + has_data` biomarker layout."""
        if biomarkers_hist.ndim != 3:
            raise ValueError(
                f"Expected biomarkers_hist to have shape [B, T, D], got {biomarkers_hist.shape}"
            )
        feature_dim = biomarkers_hist.shape[-1]
        core_dim = feature_dim - 1
        if core_dim <= 0 or core_dim % 4 != 0:
            raise ValueError(
                "Expected biomarkers_hist feature dim to follow "
                "`[value, mask, censor, age] * variants + has_data`; "
                f"got trailing dimension {feature_dim}"
            )

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
        if self.temporal_covariates_dim > 0:
            dim += self.temporal_covariates_dim
        return dim

    def _trim_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Trim prediction to match forecast horizon, removing t=0 if present."""
        if prediction.shape[1] > self.forecast_horizon:
            return prediction[:, 1 : 1 + self.forecast_horizon]
        return prediction

    def _get_mobility_node_embeddings(
        self, mob_batch: Batch, region_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run dense GNN encoder on mobility batch."""
        if not hasattr(mob_batch, "x_dense") or not hasattr(mob_batch, "adj_dense"):
            raise ValueError(
                "Mobility batch missing dense graph tensors x_dense/adj_dense."
            )

        x_dense = mob_batch.x_dense  # type: ignore[attr-defined]
        adj_dense = mob_batch.adj_dense  # type: ignore[attr-defined]
        if x_dense is None or adj_dense is None:
            raise ValueError(
                "Mobility batch has null dense graph tensors x_dense/adj_dense."
            )

        # Optional: Inject region embeddings into GNN node features
        if self.variant_type.regions and region_embeddings is not None:
            # x_dense is [G, N_ctx, F]
            G, N_ctx, F = x_dense.shape

            # mob_batch.mob_real_node_idx contains the global indices of the context nodes.
            # Its shape should be [N_ctx] (constant across the batch as built by EpiDataset)
            if (
                hasattr(mob_batch, "mob_real_node_idx")
                and mob_batch.mob_real_node_idx is not None
            ):
                real_nodes = mob_batch.mob_real_node_idx  # [N_ctx]
            else:
                raise ValueError(
                    "Mobility batch missing 'mob_real_node_idx' required for region embeddings in GNN."
                )

            if self.strict:
                max_node_idx = region_embeddings.shape[0] - 1
                if (real_nodes < 0).any() or (real_nodes > max_node_idx).any():
                    raise RuntimeError(
                        f"mob_real_node_idx out of bounds for region_embeddings. "
                        f"Max: {max_node_idx}, Min found: {real_nodes.min().item()}, "
                        f"Max found: {real_nodes.max().item()}"
                    )

            # Gather region embeddings: [N_ctx, region_dim]
            gathered_regions = region_embeddings[real_nodes]
            # Broadcast to [G, N_ctx, region_dim]
            gathered_regions = gathered_regions.unsqueeze(0).expand(G, -1, -1)

            # Concatenate to features
            x_dense = torch.cat([x_dense, gathered_regions], dim=-1)

        if self.strict and x_dense.size(-1) != self.mobility_gnn.layers[0].in_channels:
            raise ValueError(
                f"Mobility graph feature dim {x_dense.size(-1)} != expected "
                f"{self.mobility_gnn.layers[0].in_channels}"
            )

        return self.mobility_gnn(x_dense, adj_dense)

    def _gather_target_mobility_embeddings(
        self, node_emb: torch.Tensor, mob_batch: Batch, B: int, T: int
    ) -> torch.Tensor:
        """Gather embeddings for target nodes from dense [G, N, D] node embeddings."""
        if node_emb.ndim != 3:
            raise ValueError(
                f"Expected dense node embeddings [G, N, D], got shape {tuple(node_emb.shape)}"
            )

        num_graphs = node_emb.size(0)
        expected_graphs = B * T
        if num_graphs != expected_graphs:
            raise ValueError(
                f"Mobility batch has {num_graphs} graphs, expected B*T={expected_graphs} "
                f"(B={B}, T={T})."
            )

        if hasattr(mob_batch, "target_node"):
            target_local = mob_batch.target_node.reshape(-1).to(node_emb.device)
        else:
            raise ValueError("Mobility batch missing target indices.")

        if self.strict:
            max_local_idx = node_emb.shape[1] - 1
            if (target_local < 0).any() or (target_local > max_local_idx).any():
                raise RuntimeError(
                    f"target_local indices out of bounds. "
                    f"Max allowed: {max_local_idx}, Min found: {target_local.min().item()}, "
                    f"Max found: {target_local.max().item()}"
                )

        graph_ids = torch.arange(num_graphs, device=node_emb.device)
        target_embeddings = node_emb[graph_ids, target_local]
        return target_embeddings.view(B, T, self.mobility_embedding_dim)

    def to(self, *args, **kwargs):
        """Override to update cached device/dtype when model is moved."""
        result = super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                self.device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                self.dtype = arg
        if "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            self.dtype = kwargs["dtype"]
        return result

    def __repr__(self) -> str:
        return (
            "EpiForecaster("
            f"gnn_module={self.gnn_module}, "
            f"backbone={self.backbone}, "
            f"device={self.device}, "
            f"variant_type={self.variant_type})"
        )
