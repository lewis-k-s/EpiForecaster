import torch
from models.configs import (
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    SIRPhysicsConfig,
    ObservationHeadConfig,
)
from models.epiforecaster import EpiForecaster
from torch_geometric.data import Batch


def test_compile():
    # 1. Setup minimal config
    config = EpiForecasterConfig(
        model=ModelConfig(
            type=ModelVariant(cases=True, mobility=True),
            input_window_length=5,
            forecast_horizon=3,
            mobility_embedding_dim=16,
            region_embedding_dim=16,
            max_neighbors=5,
            gnn_module="gcn",
            biomarkers_dim=4,
        )
    )

    # 2. Initialize model
    model = EpiForecaster(
        variant_type=config.model.type,
        sir_physics=SIRPhysicsConfig(dt=0.1),
        observation_heads=ObservationHeadConfig(),
        temporal_input_dim=1,
        biomarkers_dim=config.model.biomarkers_dim,
        region_embedding_dim=config.model.region_embedding_dim,
        mobility_embedding_dim=config.model.mobility_embedding_dim,
        gnn_module=config.model.gnn_module,
        sequence_length=config.model.input_window_length,
        forecast_horizon=config.model.forecast_horizon,
        strict=False,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # 3. Create dummy inputs
    B = 2
    L = config.model.input_window_length
    T = B * L
    N = 10

    hosp = torch.randn(B, L, 3)
    deaths = torch.randn(B, L, 3)
    cases = torch.randn(B, L, 3)
    bio = torch.randn(B, L, config.model.biomarkers_dim)

    # Dummy mobility sequence batch
    mob = Batch()
    mob.x_dense = torch.randn(T, N, 1)
    mob.adj_dense = torch.randn(T, N, N)
    mob.target_node = torch.tensor([0] * L + [1] * L)

    pop = torch.ones(B)

    if torch.cuda.is_available():
        hosp = hosp.cuda()
        deaths = deaths.cuda()
        cases = cases.cuda()
        bio = bio.cuda()
        mob = mob.to("cuda")
        pop = pop.cuda()

    # 4. Compile model
    print("Compiling model (this might take a minute)...")
    compiled_model = torch.compile(model)

    # 5. Run forward passes to trigger tracing
    try:
        print("Running first pass (tracing)...")
        with torch.no_grad():
            _ = compiled_model(
                hosp_hist=hosp,
                deaths_hist=deaths,
                cases_hist=cases,
                biomarkers_hist=bio,
                mob_graphs=mob,
                target_nodes=torch.tensor([0, 1]),
                population=pop,
            )
        print("First pass successful!")

        print("Running second pass (execution)...")
        with torch.no_grad():
            _ = compiled_model(
                hosp_hist=hosp,
                deaths_hist=deaths,
                cases_hist=cases,
                biomarkers_hist=bio,
                mob_graphs=mob,
                target_nodes=torch.tensor([0, 1]),
                population=pop,
            )
        print("Second pass successful! torch.compile works.")

    except Exception:
        print("\nCompilation or execution failed:")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Enable compilation logs
    import logging

    torch._logging.set_logs(dynamo=logging.INFO)
    test_compile()
