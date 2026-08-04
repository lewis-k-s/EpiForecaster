from types import SimpleNamespace

import pytest
import torch

from evaluation.latent_diagnostics import build_latent_diagnostic_protocols
from evaluation.pathway_metrics import TorchLatentProtocolAccumulator
from models.sir_rollforward import SIRRollForward


@pytest.mark.epiforecaster
def test_latent_diagnostic_protocols_recover_exact_known_rollout():
    rollforward = SIRRollForward()
    model = SimpleNamespace(sir_rollforward=rollforward)
    rates = {
        "beta_t": torch.tensor([[0.2, 0.25]]),
        "gamma_t": torch.tensor([[0.1, 0.1]]),
        "mortality_t": torch.zeros(1, 2),
        "hospitalization_rate_t": torch.tensor([[0.02, 0.02]]),
        "hospital_recovery_t": torch.tensor([[0.05, 0.05]]),
        "hospital_mortality_t": torch.zeros(1, 2),
    }
    truth = rollforward(
        **rates,
        S0=torch.tensor([0.9]),
        I0=torch.tensor([0.08]),
        H0=torch.tensor([0.01]),
        R0=torch.tensor([0.01]),
        population=torch.ones(1),
    )
    targets = {
        "S_target": truth["S_trajectory"],
        "I_target": truth["I_trajectory"],
        "H_target": truth["H_trajectory"],
        "R_target": truth["R_trajectory"],
        "death_flow_target": truth["death_flow"],
    }

    protocols = build_latent_diagnostic_protocols(
        model=model,
        model_outputs={**rates},
        targets=targets,
    )

    assert set(protocols) == {"oracle_x0", "teacher_forced_one_step"}
    for protocol, (predictions, protocol_targets) in protocols.items():
        accumulator = TorchLatentProtocolAccumulator(
            protocol=protocol,
            device=torch.device("cpu"),
        )
        accumulator.update(predictions=predictions, targets=protocol_targets)
        metrics = accumulator.finalize()
        assert metrics[f"latent_diag_{protocol}_rmse_i_fraction"] == pytest.approx(
            0.0, abs=1.0e-7
        )
        assert metrics[
            f"latent_diag_{protocol}_rmse_death_flow_fraction"
        ] == pytest.approx(0.0, abs=1.0e-7)


@pytest.mark.epiforecaster
def test_latent_diagnostic_protocols_require_complete_targets():
    protocols = build_latent_diagnostic_protocols(
        model=SimpleNamespace(sir_rollforward=SIRRollForward()),
        model_outputs={},
        targets={},
    )

    assert protocols == {}
