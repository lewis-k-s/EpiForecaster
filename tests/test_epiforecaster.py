import pytest
import torch
from torch_geometric.data import Data, Batch
from models.epiforecaster import EpiForecaster
from models.configs import (
    ModelVariant,
    SIRPhysicsConfig,
    ObservationHeadConfig,
)


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
            "hosp_hist": torch.randn(B, T, 3),
            "deaths_hist": torch.randn(B, T, 3),
            "cases_hist": torch.randn(B, T, 3),
            "biomarkers_hist": torch.randn(B, T, 5),
            "target_nodes": torch.zeros(B, dtype=torch.long),
            "population": torch.ones(B) * 1000,
            "region_embeddings": torch.randn(10, 8),
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
            out["initial_states"].sum(dim=-1), torch.ones(batch_size), atol=1e-5
        )

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
            x = torch.randn(5, model.temporal_node_dim)
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
        """Smoke test for gradient flow."""
        model = EpiForecaster(**basic_config)

        # Enable gradients on inputs that support it
        hosp = dummy_batch["hosp_hist"].clone().requires_grad_(True)

        out = model(
            hosp_hist=hosp,
            deaths_hist=dummy_batch["deaths_hist"],
            cases_hist=dummy_batch["cases_hist"],
            biomarkers_hist=dummy_batch["biomarkers_hist"],
            mob_graphs=None,
            target_nodes=dummy_batch["target_nodes"],
            population=dummy_batch["population"],
        )

        # Test gradient flow to backbone outputs first
        loss_beta = out["beta_t"].sum()
        loss_beta.backward(retain_graph=True)

        assert hosp.grad is not None
        assert hosp.grad.abs().sum() > 0, "Gradient did not flow to beta_t"

        # Clear grads
        hosp.grad.zero_()

        # Test gradient flow to final predictions
        loss_cases = out["pred_cases"].sum()
        loss_cases.backward()

        assert hosp.grad is not None
        assert hosp.grad.abs().sum() > 0, "Gradient did not flow to pred_cases"

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
            "WWTarget": torch.randn(2, 7),
            "HospTarget": torch.randn(2, 7),
            "CasesTarget": torch.randn(2, 7),
            "DeathsTarget": torch.randn(2, 7),
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
