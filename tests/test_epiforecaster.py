import pytest
import torch
from torch_geometric.data import Batch

from data.epi_batch import EpiBatch
from models.configs import (
    ModelVariant,
    ObservationHeadConfig,
    SIRPhysicsConfig,
)
from models.epiforecaster import EpiForecaster


def _rand_tensor(*shape, dtype=None):
    """Create random tensor with model dtype by default."""
    if dtype is None:
        dtype = torch.float32
    return torch.randn(*shape, dtype=dtype)


class TestEpiForecaster:
    """Tests for EpiForecaster."""

    @pytest.fixture
    def basic_config(self):
        return {
            "variant_type": ModelVariant(
                cases=True, mobility=False, regions=False, biomarkers=False
            ),
            "sir_physics": SIRPhysicsConfig(),
            "observation_heads": ObservationHeadConfig(),
            "temporal_input_dim": 9,  # 3 clinical series × 3 channels each
            "forecast_horizon": 7,
            "sequence_length": 14,
            "head_d_model": 16,
        }

    @pytest.fixture
    def dummy_batch(self):
        B, T = 2, 14
        return {
            "hosp_hist": _rand_tensor(B, T, 3),
            "deaths_hist": _rand_tensor(B, T, 3),
            "cases_hist": _rand_tensor(B, T, 3),
            "biomarkers_hist": _rand_tensor(B, T, 5),
            "ww_hist": _rand_tensor(B, T),
            "ww_hist_mask": torch.ones(B, T, dtype=torch.float32),
            "target_nodes": torch.zeros(B, dtype=torch.long),
            "population": torch.ones(B, dtype=torch.float32) * 1000,
            "region_embeddings": _rand_tensor(10, 8),
        }

    def test_clinical_gamma_priors_reach_delay_kernels(self, basic_config):
        """Configured hosp/cases Gamma priors wire into their DelayKernels."""
        basic_config["observation_heads"] = ObservationHeadConfig(
            kernel_gamma_shape_hosp=6.5,
            kernel_gamma_scale_hosp=1.7,
            kernel_gamma_shape_cases=4.0,
            kernel_gamma_scale_cases=2.5,
            learnable_kernel_hosp=True,
            kernel_parameterization_hosp="gamma",
            learnable_kernel_cases=True,
            kernel_parameterization_cases="gamma",
            learnable_kernel_deaths=True,
            kernel_parameterization_deaths="gamma",
        )
        model = EpiForecaster(**basic_config)
        assert model.hosp_head.delay_kernel.get_gamma_parameters() == (
            pytest.approx(6.5),
            pytest.approx(1.7),
        )
        assert model.cases_head.delay_kernel.get_gamma_parameters() == (
            pytest.approx(4.0),
            pytest.approx(2.5),
        )
        # Deaths keeps its own configured prior.
        assert model.deaths_head.delay_kernel.get_gamma_parameters() == (
            pytest.approx(12.25),
            pytest.approx(4.0 / 7.0),
        )

    def test_init_can_disable_an_observation_residual(self, basic_config):
        baseline_config = basic_config.copy()
        basic_config["observation_heads"] = ObservationHeadConfig(
            residual_disabled_heads=["ww"]
        )

        torch.manual_seed(7)
        baseline = EpiForecaster(**baseline_config)
        torch.manual_seed(7)
        model = EpiForecaster(**basic_config)

        # Static ablation retains the projection structure but never activates
        # its residual location contribution.
        assert model.ww_head._residual_location_enabled is False
        assert model.ww_head.use_residual is True
        assert model.ww_head.residual_proj is not None
        assert model.ww_head.alpha is not None
        assert model.ww_head.alpha.requires_grad is False
        assert model.ww_head.alpha.item() == 0.0
        assert model.hosp_head._residual_location_enabled is True
        for name, parameter in baseline.state_dict().items():
            if not name.startswith("ww_head."):
                assert torch.equal(parameter, model.state_dict()[name])

    def test_init_rejects_unknown_disabled_residual_head(self):
        with pytest.raises(ValueError, match="residual_disabled_heads"):
            ObservationHeadConfig(residual_disabled_heads=["invalid"])

    def test_forward_basic(self, basic_config, dummy_batch):
        """Test forward pass with minimal features (cases only)."""
        model = EpiForecaster(**basic_config)

        # Forward inputs
        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],  # Should be ignored
            mob_graphs=None,
            target_nodes=dummy_batch["target_nodes"],
            population=dummy_batch["population"],
        )

        horizon = basic_config["forecast_horizon"]
        batch_size = dummy_batch["hosp_hist"].shape[0]

        # Predictions include t=0 (nowcast) so they are H+1 shaped
        assert out["pred_cases"].shape == (batch_size, horizon + 1)
        assert out["pred_hosp"].shape == (batch_size, horizon + 1)
        # Deaths uses death_flow which doesn't have nowcast
        assert out["pred_deaths"].shape == (batch_size, horizon)
        for head_name in ("ww", "hosp", "cases", "deaths"):
            pred_key = f"pred_{head_name}"
            assert out[f"dist_{head_name}_loc"].shape == out[pred_key].shape
            assert out[f"dist_{head_name}_scale"].shape == out[pred_key].shape
            assert out[f"mechanistic_{head_name}"].shape == out[pred_key].shape
            assert out[f"residual_{head_name}"].shape == out[pred_key].shape
            assert out[f"residual_input_{head_name}"].shape == out[pred_key].shape
            assert torch.allclose(
                out[pred_key],
                out[f"mechanistic_{head_name}"] + out[f"residual_{head_name}"],
                atol=1e-5,
            )
            assert torch.all(out[f"dist_{head_name}_scale"] > 0)
        source_grad, residual_grad = torch.autograd.grad(
            out["pred_cases"].square().mean(),
            (out["observation_source_cases"], out["residual_input_cases"]),
            retain_graph=True,
        )
        assert torch.all(torch.isfinite(source_grad))
        assert torch.all(torch.isfinite(residual_grad))
        assert out["S_trajectory"].shape == (batch_size, horizon + 1)
        assert out["I_trajectory"].shape == (batch_size, horizon + 1)
        assert out["H_trajectory"].shape == (batch_size, horizon + 1)
        assert out["R_trajectory"].shape == (batch_size, horizon + 1)
        assert out["D_trajectory"].shape == (batch_size, horizon + 1)
        assert out["hospitalization_flow"].shape == (batch_size, horizon)
        assert out["death_flow"].shape == (batch_size, horizon)
        assert out["observation_source_ww"].shape == (batch_size, horizon + 1)
        assert out["observation_source_cases"].shape == (batch_size, horizon + 1)
        assert out["observation_source_hosp"].shape == (batch_size, horizon + 1)
        assert out["observation_source_deaths"].shape == (batch_size, horizon)
        assert out["beta_t"].shape == (batch_size, horizon)
        assert out["gamma_t"].shape == (batch_size, horizon)
        assert out["hospitalization_rate_t"].shape == (batch_size, horizon)
        assert out["hospital_recovery_t"].shape == (batch_size, horizon)
        assert out["hospital_mortality_t"].shape == (batch_size, horizon)

        # Check SIRHD constraints.
        # Initial states sum to 1
        assert torch.allclose(
            out["initial_states"].sum(dim=-1),
            torch.ones(batch_size, dtype=torch.float32),
            atol=1e-5,
        )
        total = (
            out["S_trajectory"]
            + out["I_trajectory"]
            + out["H_trajectory"]
            + out["R_trajectory"]
            + out["D_trajectory"]
        )
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_forward_with_large_population_is_finite(self, basic_config):
        """Large raw population values should remain numerically stable."""
        torch.manual_seed(0)
        model = EpiForecaster(**basic_config)

        B, T = 4, 14
        out = model(
            hosp_hist=_rand_tensor(B, T, 3),
            deaths_hist=_rand_tensor(B, T, 3),
            cases_hist=_rand_tensor(B, T, 3),
            biomarkers_hist=_rand_tensor(B, T, 5),
            mob_graphs=None,
            target_nodes=torch.zeros(B, dtype=torch.long),
            population=torch.full((B,), 1_000_000.0, dtype=torch.float32),
        )

        for key in ["beta_t", "pred_hosp", "pred_cases", "pred_deaths", "pred_ww"]:
            assert torch.all(torch.isfinite(out[key]))

    def test_initial_states_not_one_hot_at_startup(self, basic_config, dummy_batch):
        """Conservative initialization should avoid one-hot collapse at startup."""
        model = EpiForecaster(**basic_config)
        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=None,
            target_nodes=dummy_batch["target_nodes"],
            population=dummy_batch["population"],
        )

        max_prob = out["initial_states"].max(dim=-1).values
        assert torch.all(max_prob < 0.999)

    def test_beta_startup_values_are_not_boundary_clamped(self, basic_config):
        """Seeded random input should produce beta values inside configured bounds."""
        torch.manual_seed(0)
        model = EpiForecaster(**basic_config)

        B, T = 3, 14
        out = model(
            hosp_hist=_rand_tensor(B, T, 3),
            deaths_hist=_rand_tensor(B, T, 3),
            cases_hist=_rand_tensor(B, T, 3),
            biomarkers_hist=_rand_tensor(B, T, 5),
            mob_graphs=None,
            target_nodes=torch.zeros(B, dtype=torch.long),
            population=torch.ones(B, dtype=torch.float32) * 1000,
        )

        cfg = model.sir_physics
        assert torch.all(out["beta_t"] > cfg.beta_min)
        assert torch.all(out["beta_t"] < cfg.beta_max)

    def test_forward_with_mobility_and_regions_uses_collated_local_targets(
        self, basic_config, dummy_batch
    ):
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, mobility=True, regions=True)
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8
        config["region_embedding_dim"] = 8

        model = EpiForecaster(**config)

        B, T = dummy_batch["hosp_hist"].shape[:2]
        num_graphs = B * T
        num_ctx_nodes = 3
        mob_batch = Batch()
        mob_batch.x_dense = _rand_tensor(
            num_graphs, num_ctx_nodes, model.temporal_node_dim
        )
        dense_adj = torch.rand(num_graphs, num_ctx_nodes, num_ctx_nodes)
        eye = torch.eye(num_ctx_nodes, dtype=dense_adj.dtype).unsqueeze(0)
        mob_batch.adj_time_ids = torch.arange(num_graphs)
        mob_batch.adj_table = torch.maximum(dense_adj, eye)

        mob_batch.mob_real_node_idx = torch.tensor([7, 3, 9], dtype=torch.long)
        mob_batch.target_node = torch.tensor(
            [1, 2], dtype=torch.long
        ).repeat_interleave(T)

        target_nodes = torch.tensor([3, 9], dtype=torch.long)
        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=mob_batch,
            target_nodes=target_nodes,
            region_embeddings=torch.randn(16, 8),
            population=dummy_batch["population"],
        )

        assert out["beta_t"].shape == (B, config["forecast_horizon"])

    def test_forward_with_mobility_and_regions_maps_global_targets_without_local_index(
        self, basic_config, dummy_batch
    ):
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, mobility=True, regions=True)
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8
        config["region_embedding_dim"] = 8

        model = EpiForecaster(**config)

        B, T = dummy_batch["hosp_hist"].shape[:2]
        num_graphs = B * T
        num_ctx_nodes = 3
        mob_batch = Batch()
        mob_batch.x_dense = _rand_tensor(
            num_graphs, num_ctx_nodes, model.temporal_node_dim
        )
        dense_adj = torch.rand(num_graphs, num_ctx_nodes, num_ctx_nodes)
        eye = torch.eye(num_ctx_nodes, dtype=dense_adj.dtype).unsqueeze(0)
        mob_batch.adj_time_ids = torch.arange(num_graphs)
        mob_batch.adj_table = torch.maximum(dense_adj, eye)
        mob_batch.mob_real_node_idx = torch.tensor([7, 3, 9], dtype=torch.long)

        target_nodes = torch.tensor([3, 9], dtype=torch.long)
        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=mob_batch,
            target_nodes=target_nodes,
            region_embeddings=torch.randn(16, 8),
            population=dummy_batch["population"],
        )

        assert out["beta_t"].shape == (B, config["forecast_horizon"])

    def test_forward_with_mobility_and_regions_rejects_bad_real_node_idx_shape(
        self, basic_config, dummy_batch
    ):
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, mobility=True, regions=True)
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8
        config["region_embedding_dim"] = 8

        model = EpiForecaster(**config)

        B, T = dummy_batch["hosp_hist"].shape[:2]
        num_graphs = B * T
        num_nodes = 5
        mob_batch = Batch()
        mob_batch.x_dense = _rand_tensor(num_graphs, num_nodes, model.temporal_node_dim)
        dense_adj = torch.rand(num_graphs, num_nodes, num_nodes)
        eye = torch.eye(num_nodes, dtype=dense_adj.dtype).unsqueeze(0)
        mob_batch.adj_time_ids = torch.arange(num_graphs)
        mob_batch.adj_table = torch.maximum(dense_adj, eye)
        mob_batch.target_node = torch.zeros(num_graphs, dtype=torch.long)
        mob_batch.mob_real_node_idx = torch.arange(num_nodes + 1, dtype=torch.long)

        with pytest.raises(RuntimeError, match="shape mismatch"):
            model(
                hosp_hist=dummy_batch["hosp_hist"],
                deaths_hist=dummy_batch["deaths_hist"],
                cases_hist=dummy_batch["cases_hist"],
                biomarkers_hist=dummy_batch["biomarkers_hist"],
                mob_graphs=mob_batch,
                target_nodes=dummy_batch["target_nodes"],
                region_embeddings=torch.randn(10, 8),
                population=dummy_batch["population"],
            )

    def test_gradient_flow(self, basic_config, dummy_batch):
        """Smoke test for parameter gradient flow under conservative initialization."""
        model = EpiForecaster(**basic_config)

        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=None,
            target_nodes=dummy_batch["target_nodes"],
            population=dummy_batch["population"],
        )

        # Conservative init keeps backbone outputs near prior-centered values.
        # Step 1 should propagate gradient through both final rate heads and
        # their projection stems.
        loss_beta = out["beta_t"].sum()
        loss_beta.backward(retain_graph=True)

        beta_stem_weight_grad = model.backbone.beta_projection[0].weight.grad
        assert beta_stem_weight_grad is not None
        assert beta_stem_weight_grad.abs().sum() > 0

        beta_head_weight_grad = model.backbone.beta_projection[2].weight.grad
        assert beta_head_weight_grad is not None
        assert beta_head_weight_grad.abs().sum() > 0

        # Prediction losses should backpropagate to observation head scales.
        model.zero_grad(set_to_none=True)
        loss_cases = out["pred_cases"].sum()
        loss_cases.backward()

        assert model.cases_head.scale.grad is not None
        assert model.cases_head.scale.grad.abs().sum() > 0

    def test_forward_batch_masks_ablated_neighbor_series(self, basic_config):
        """Input ablations must also remove the corresponding neighbor signals."""
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(
            cases=True,
            biomarkers=True,
            mobility=True,
        )
        config["biomarkers_dim"] = 5
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8
        model = EpiForecaster(**config)

        B, T = 1, config["sequence_length"]
        horizon = config["forecast_horizon"]
        num_graphs = B * T
        num_nodes = 3

        mob_batch = Batch()
        mob_batch.x_dense = torch.ones(
            num_graphs,
            num_nodes,
            model.temporal_node_dim,
            dtype=torch.float32,
        )
        mob_batch.adj_time_ids = torch.arange(num_graphs)
        mob_batch.adj_table = torch.eye(num_nodes, dtype=torch.float32).expand(
            num_graphs, -1, -1
        )
        mob_batch.target_node = torch.zeros(num_graphs, dtype=torch.long)

        batch_data = EpiBatch(
            hosp_hist=torch.ones(B, T, 3),
            deaths_hist=torch.ones(B, T, 3),
            cases_hist=torch.ones(B, T, 3),
            bio_node=torch.ones(B, T, config["biomarkers_dim"]),
            mob_batch=mob_batch,
            population=torch.full((B,), 1000.0, dtype=torch.float32),
            b=B,
            t=T,
            target_node=torch.zeros(B, dtype=torch.long),
            target_region_index=None,
            window_start=torch.zeros(B, dtype=torch.long),
            node_labels=["node_0"],
            temporal_covariates=torch.zeros(B, T, 0),
            vaccination_hist=torch.zeros(B, T, 3),
            ww_hist=torch.ones(B, T),
            ww_hist_mask=torch.ones(B, T),
            hosp_target=torch.ones(B, horizon),
            ww_target=torch.ones(B, horizon),
            cases_target=torch.ones(B, horizon),
            deaths_target=torch.ones(B, horizon),
            hosp_target_mask=torch.ones(B, horizon),
            ww_target_mask=torch.ones(B, horizon),
            cases_target_mask=torch.ones(B, horizon),
            deaths_target_mask=torch.ones(B, horizon),
            S_target=None,
            I_target=None,
            R_target=None,
            D_target=None,
        )

        model.forward_batch(
            batch_data=batch_data,
            mask_hosp=True,
            mask_ww=True,
        )

        assert torch.count_nonzero(batch_data.hosp_hist) == 0
        assert torch.count_nonzero(batch_data.bio_node) == 0
        assert torch.count_nonzero(batch_data.ww_hist) == 0
        assert torch.count_nonzero(batch_data.ww_hist_mask) == 0

        graph_features = batch_data.mob_batch.x_dense
        assert torch.count_nonzero(graph_features[..., 0:3]) == 0
        assert torch.count_nonzero(graph_features[..., 3:9]) > 0
        assert torch.count_nonzero(graph_features[..., 9:16]) == 0

    @pytest.mark.device
    def test_forward_batch_cross_device(self, basic_config, accelerator_device):
        """Test that forward_batch works after explicit device transfer."""
        config = basic_config.copy()
        model = EpiForecaster(**config).to(accelerator_device)

        B, T = 2, config["sequence_length"]
        horizon = config["forecast_horizon"]

        batch_data = EpiBatch(
            hosp_hist=_rand_tensor(B, T, 3),
            deaths_hist=_rand_tensor(B, T, 3),
            cases_hist=_rand_tensor(B, T, 3),
            bio_node=torch.zeros(B, T, 1, dtype=torch.float32),
            mob_batch=Batch(),
            population=torch.full((B,), 1000.0, dtype=torch.float32),
            b=B,
            t=T,
            target_node=torch.zeros(B, dtype=torch.long),
            target_region_index=None,
            window_start=torch.zeros(B, dtype=torch.long),
            node_labels=["node_0", "node_1"],
            temporal_covariates=torch.zeros(B, T, 0),
            vaccination_hist=torch.zeros(B, T, 3),
            ww_hist=_rand_tensor(B, T),
            ww_hist_mask=torch.ones(B, T),
            hosp_target=_rand_tensor(B, horizon),
            ww_target=_rand_tensor(B, horizon),
            cases_target=_rand_tensor(B, horizon),
            deaths_target=_rand_tensor(B, horizon),
            hosp_target_mask=torch.ones(B, horizon),
            ww_target_mask=torch.ones(B, horizon),
            cases_target_mask=torch.ones(B, horizon),
            deaths_target_mask=torch.ones(B, horizon),
            S_target=None,
            I_target=None,
            R_target=None,
            D_target=None,
        )
        batch_data = batch_data.to(accelerator_device)

        outputs, targets = model.forward_batch(batch_data=batch_data)

        assert outputs["pred_cases"].device.type == accelerator_device.type
        assert outputs["pred_hosp"].device.type == accelerator_device.type
        assert targets["cases"].device.type == accelerator_device.type

    def test_delta_forecasting_anchors_clinical_heads_from_last_valid_step(
        self, basic_config
    ):
        config = basic_config.copy()
        config["observation_heads"] = ObservationHeadConfig(delta_forecasting=True)
        model = EpiForecaster(**config)

        B, T = 1, config["sequence_length"]
        hist = torch.zeros(B, T, 3, dtype=torch.float32)
        hist[0, -3, 0] = 1.5
        hist[0, -3, 1] = 1.0

        out = model(
            hosp_hist=hist,
            deaths_hist=torch.zeros(B, T, 3),
            cases_hist=hist.clone(),
            biomarkers_hist=torch.zeros(B, T, 5),
            mob_graphs=None,
            target_nodes=torch.zeros(B, dtype=torch.long),
            population=torch.ones(B, dtype=torch.float32) * 1000,
        )

        assert out["pred_hosp"][0, 0].item() == pytest.approx(1.5, abs=1e-5)
        assert out["pred_cases"][0, 0].item() == pytest.approx(1.5, abs=1e-5)

    def test_delta_forecasting_validates_biomarker_layout(self, basic_config):
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(
            cases=True, mobility=False, regions=False, biomarkers=True
        )
        config["observation_heads"] = ObservationHeadConfig(delta_forecasting=True)
        config["biomarkers_dim"] = 6
        model = EpiForecaster(**config)

        B, T = 1, config["sequence_length"]
        with pytest.raises(ValueError, match="biomarkers_hist feature dim"):
            model(
                hosp_hist=torch.zeros(B, T, 3),
                deaths_hist=torch.zeros(B, T, 3),
                cases_hist=torch.zeros(B, T, 3),
                biomarkers_hist=torch.zeros(B, T, 6),
                ww_hist=torch.zeros(B, T),
                ww_hist_mask=torch.ones(B, T),
                mob_graphs=None,
                target_nodes=torch.zeros(B, dtype=torch.long),
                population=torch.ones(B, dtype=torch.float32),
            )

    def test_delta_forecasting_anchors_wastewater_from_target_space_history(
        self, basic_config
    ):
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(
            cases=True, mobility=False, regions=False, biomarkers=True
        )
        config["observation_heads"] = ObservationHeadConfig(delta_forecasting=True)
        config["biomarkers_dim"] = 5
        model = EpiForecaster(**config)

        B, T = 1, config["sequence_length"]
        ww_hist = torch.zeros(B, T, dtype=torch.float32)
        ww_hist_mask = torch.zeros(B, T, dtype=torch.float32)
        ww_hist[0, -2] = 3.25
        ww_hist_mask[0, -2] = 1.0

        out = model(
            hosp_hist=torch.zeros(B, T, 3),
            deaths_hist=torch.zeros(B, T, 3),
            cases_hist=torch.zeros(B, T, 3),
            biomarkers_hist=torch.zeros(B, T, 5),
            ww_hist=ww_hist,
            ww_hist_mask=ww_hist_mask,
            mob_graphs=None,
            target_nodes=torch.zeros(B, dtype=torch.long),
            population=torch.ones(B, dtype=torch.float32),
        )

        assert out["pred_ww"][0, 0].item() == pytest.approx(3.25, abs=1e-5)
