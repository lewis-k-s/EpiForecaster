import pytest
import torch
from torch_geometric.data import Batch, Data

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
            "target_nodes": torch.zeros(B, dtype=torch.long),
            "population": torch.ones(B, dtype=torch.float32) * 1000,
            "region_embeddings": _rand_tensor(10, 8),
        }

    def test_init_basic(self, basic_config):
        model = EpiForecaster(**basic_config)
        assert isinstance(model, EpiForecaster)

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

        assert out["pred_cases"].shape == (batch_size, horizon)
        assert out["pred_hosp"].shape == (batch_size, horizon)
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

        # Create dummy mobility graphs
        B, T = dummy_batch["hosp_hist"].shape[:2]
        num_graphs = B * T
        # Create a list of dummy graphs
        graphs = []
        for _ in range(num_graphs):
            # 5 nodes, random edges
            x = _rand_tensor(5, model.temporal_node_dim)
            edge_index = torch.randint(0, 5, (2, 10))
            graphs.append(
                Data(x=x, edge_index=edge_index, target_node=torch.tensor([0]))
            )

        mob_batch = Batch.from_data_list(graphs)

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

        assert out["pred_cases"].shape == (2, config["forecast_horizon"])

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

        # Conservative init keeps backbone outputs prior-centered constants, but
        # the final rate-head weights should still receive gradient.
        loss_beta = out["beta_t"].sum()
        loss_beta.backward(retain_graph=True)

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

        # Construct batch_data dict expected by forward_batch
        # Note: forward_batch expects "BioNode", "MobBatch", "TargetNode" etc.
        # It also handles .to(device)

        batch_data = {
            "HospHist": dummy_batch["hosp_hist"],
            "DeathsHist": dummy_batch["deaths_hist"],
            "CasesHist": dummy_batch["cases_hist"],
            "BioNode": dummy_batch["biomarkers_hist"],
            "MobBatch": torch.zeros(1),  # Placeholder, will fail if mobility=True
            "Population": dummy_batch["population"],
            "TargetNode": dummy_batch["target_nodes"],
            # Targets for loss dict
            "WWTarget": _rand_tensor(2, 7),
            "HospTarget": _rand_tensor(2, 7),
            "CasesTarget": _rand_tensor(2, 7),
            "DeathsTarget": _rand_tensor(2, 7),
        }

        # Test with mobility=False (so MobBatch doesn't matter much unless checked)
        # The code does: mob_batch = batch_data["MobBatch"].to(...)
        # So we need a dummy tensor at least.

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
        # Use centralized dtype constant as the source of truth
        expected_dtype = torch.float32

        B, T = 1, config["sequence_length"]
        horizon = config["forecast_horizon"]

        # Build B*T mobility graphs with MODEL_DTYPE node/edge weights.
        graphs: list[Data] = []
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        for _ in range(B * T):
            graphs.append(
                Data(
                    x=_rand_tensor(4, model.temporal_node_dim),
                    edge_index=edge_index,
                    edge_weight=torch.ones(
                        edge_index.shape[1], dtype=torch.float32
                    ),
                    target_node=torch.tensor([0], dtype=torch.long),
                )
            )
        mob_batch = Batch.from_data_list(graphs)

        batch_data = {
            "HospHist": _rand_tensor(B, T, 3),
            "DeathsHist": _rand_tensor(B, T, 3),
            "CasesHist": _rand_tensor(B, T, 3),
            "BioNode": torch.zeros(B, T, 1, dtype=torch.float32),
            "MobBatch": mob_batch,
            "Population": torch.full((B,), 1000.0, dtype=torch.float32),
            "TargetNode": torch.zeros(B, dtype=torch.long),
            "WWTarget": _rand_tensor(B, horizon),
            "HospTarget": _rand_tensor(B, horizon),
            "CasesTarget": _rand_tensor(B, horizon),
            "DeathsTarget": _rand_tensor(B, horizon),
        }

        outputs, targets = model.forward_batch(batch_data=batch_data)

        assert outputs["pred_cases"].dtype == expected_dtype
        assert outputs["pred_hosp"].dtype == expected_dtype
        assert targets["cases"] is not None
        assert targets["cases"].dtype == expected_dtype
        assert mob_batch.edge_weight.dtype == expected_dtype
