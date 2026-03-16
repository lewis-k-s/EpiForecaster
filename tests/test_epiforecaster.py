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
from utils.precision_policy import MODEL_PARAM_DTYPE


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

    def test_init_basic(self, basic_config):
        model = EpiForecaster(**basic_config)
        assert isinstance(model, EpiForecaster)
        assert model.dtype == MODEL_PARAM_DTYPE

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
        assert out["S_trajectory"].shape == (batch_size, horizon + 1)

        # Check SIR constraints (approximate sum to 1)
        # Note: S+I+R sum to 1.
        # But prediction is trajectories.
        # Initial states sum to 1
        assert torch.allclose(
            out["initial_states"].sum(dim=-1),
            torch.ones(batch_size, dtype=torch.float32),
            atol=1e-5,
        )

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

    def test_forward_with_mobility(self, basic_config, dummy_batch):
        """Test forward pass with mobility enabled."""
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, mobility=True)
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8

        model = EpiForecaster(**config)

        # Create dummy dense mobility batch
        B, T = dummy_batch["hosp_hist"].shape[:2]
        num_graphs = B * T
        num_nodes = 5
        mob_batch = Batch()
        mob_batch.x_dense = _rand_tensor(num_graphs, num_nodes, model.temporal_node_dim)
        dense_adj = torch.rand(num_graphs, num_nodes, num_nodes)
        eye = torch.eye(num_nodes, dtype=dense_adj.dtype).unsqueeze(0)
        mob_batch.adj_dense = torch.maximum(dense_adj, eye)
        mob_batch.target_node = torch.zeros(num_graphs, dtype=torch.long)

        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=mob_batch,
            target_nodes=dummy_batch["target_nodes"],
            population=dummy_batch["population"],
        )

        assert out["beta_t"].shape == (B, config["forecast_horizon"])

    def test_forward_with_regions(self, basic_config, dummy_batch):
        """Test forward pass with static region embeddings."""
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, regions=True)
        config["region_embedding_dim"] = 8

        model = EpiForecaster(**config)

        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=None,
            target_nodes=dummy_batch["target_nodes"],
            region_embeddings=dummy_batch["region_embeddings"],
            population=dummy_batch["population"],
        )

        # Predictions include t=0 (nowcast) so they are H+1 shaped
        assert out["pred_cases"].shape == (2, config["forecast_horizon"] + 1)

    def test_forward_with_mobility_and_regions(self, basic_config, dummy_batch):
        """Test forward pass where region embeddings are injected into the mobility GNN."""
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

        # In this config, the GNN input capacity is temporal_dim + region_dim
        mob_batch.x_dense = _rand_tensor(num_graphs, num_nodes, model.temporal_node_dim)
        dense_adj = torch.rand(num_graphs, num_nodes, num_nodes)
        eye = torch.eye(num_nodes, dtype=dense_adj.dtype).unsqueeze(0)
        mob_batch.adj_dense = torch.maximum(dense_adj, eye)
        mob_batch.target_node = torch.zeros(num_graphs, dtype=torch.long)
        mob_batch.mob_real_node_idx = torch.arange(num_nodes, dtype=torch.long)

        out = model(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=mob_batch,
            target_nodes=dummy_batch["target_nodes"],
            region_embeddings=torch.randn(10, 8),  # num_regions = 10
            population=dummy_batch["population"],
        )

        assert out["beta_t"].shape == (B, config["forecast_horizon"])

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

    def test_forward_batch_helper(self, basic_config, dummy_batch):
        """Test the forward_batch helper method."""
        model = EpiForecaster(**basic_config)

        B, T = 2, 14
        H = basic_config["forecast_horizon"]

        batch_data = EpiBatch(
            hosp_hist=dummy_batch["hosp_hist"],
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            bio_node=dummy_batch["biomarkers_hist"],
            mob_batch=torch.zeros(1),
            population=dummy_batch["population"],
            b=B,
            t=T,
            target_node=dummy_batch["target_nodes"],
            target_region_index=None,
            window_start=torch.zeros(B, dtype=torch.long),
            node_labels=["node_0", "node_1"],
            temporal_covariates=torch.zeros(B, T, 0),
            ww_hist=dummy_batch["ww_hist"],
            ww_hist_mask=dummy_batch["ww_hist_mask"],
            hosp_target=_rand_tensor(B, H),
            ww_target=_rand_tensor(B, H),
            cases_target=_rand_tensor(B, H),
            deaths_target=_rand_tensor(B, H),
            hosp_target_mask=torch.ones(B, H),
            ww_target_mask=torch.ones(B, H),
            cases_target_mask=torch.ones(B, H),
            deaths_target_mask=torch.ones(B, H),
        )

        outputs, targets = model.forward_batch(
            batch_data=batch_data, region_embeddings=dummy_batch["region_embeddings"]
        )

        assert "pred_cases" in outputs
        assert "cases" in targets
        assert targets["cases"].shape == (2, 7)

    def test_forward_batch_casts_half_inputs_with_mobility(self, basic_config):
        """Forward batch should align float inputs with model dtype."""
        config = basic_config.copy()
        config["variant_type"] = ModelVariant(cases=True, mobility=True)
        config["gnn_hidden_dim"] = 8
        config["mobility_embedding_dim"] = 8
        model = EpiForecaster(**config)
        expected_dtype = torch.float32

        B, T = 1, config["sequence_length"]
        horizon = config["forecast_horizon"]

        num_graphs = B * T
        num_nodes = 4
        mob_batch = Batch()
        mob_batch.x_dense = _rand_tensor(
            num_graphs, num_nodes, model.temporal_node_dim, dtype=torch.float64
        )
        dense_adj = torch.rand(num_graphs, num_nodes, num_nodes, dtype=torch.float64)
        eye = torch.eye(num_nodes, dtype=torch.float64).unsqueeze(0)
        mob_batch.adj_dense = torch.maximum(dense_adj, eye)
        mob_batch.target_node = torch.zeros(num_graphs, dtype=torch.long)

        batch_data = EpiBatch(
            hosp_hist=_rand_tensor(B, T, 3),
            deaths_hist=_rand_tensor(B, T, 3),
            cases_hist=_rand_tensor(B, T, 3),
            bio_node=torch.zeros(B, T, 1, dtype=torch.float32),
            mob_batch=mob_batch,
            population=torch.full((B,), 1000.0, dtype=torch.float32),
            b=B,
            t=T,
            target_node=torch.zeros(B, dtype=torch.long),
            target_region_index=None,
            window_start=torch.zeros(B, dtype=torch.long),
            node_labels=["node_0"],
            temporal_covariates=torch.zeros(B, T, 0),
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
        )

        outputs, targets = model.forward_batch(batch_data=batch_data)

        assert outputs["pred_cases"].dtype == expected_dtype
        assert outputs["pred_hosp"].dtype == expected_dtype
        assert targets["cases"] is not None
        assert targets["cases"].dtype == expected_dtype
        assert mob_batch.x_dense.dtype == expected_dtype
        assert mob_batch.adj_dense.dtype == expected_dtype

    @pytest.mark.device
    def test_forward_batch_cross_device(self, basic_config, accelerator_device):
        """Test that forward_batch works when model is on accelerator and batch data on CPU.

        This simulates the real training scenario where:
        - Model is moved to GPU/MPS via .to(device)
        - Batch data comes from DataLoader on CPU

        Regression test for device mismatch bugs in forward_batch.
        """
        config = basic_config.copy()
        model = EpiForecaster(**config).to(accelerator_device)

        B, T = 2, config["sequence_length"]
        horizon = config["forecast_horizon"]

        batch_data = EpiBatch(
            hosp_hist=_rand_tensor(B, T, 3),
            deaths_hist=_rand_tensor(B, T, 3),
            cases_hist=_rand_tensor(B, T, 3),
            bio_node=torch.zeros(B, T, 1, dtype=torch.float32),
            mob_batch=torch.zeros(1),
            population=torch.full((B,), 1000.0, dtype=torch.float32),
            b=B,
            t=T,
            target_node=torch.zeros(B, dtype=torch.long),
            target_region_index=None,
            window_start=torch.zeros(B, dtype=torch.long),
            node_labels=["node_0", "node_1"],
            temporal_covariates=torch.zeros(B, T, 0),
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
        )

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
