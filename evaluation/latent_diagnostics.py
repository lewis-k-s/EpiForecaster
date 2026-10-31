"""Canonical oracle-X0 and teacher-forced latent transition diagnostics."""

from __future__ import annotations

import torch
from einops import rearrange

from models.epiforecaster import EpiForecaster


_STATE_TASKS = {
    "s": ("S_trajectory", "S_target"),
    "i": ("I_trajectory", "I_target"),
    "h": ("H_trajectory", "H_target"),
    "r": ("R_trajectory", "R_target"),
}
_RATE_KEYS = (
    "beta_t",
    "gamma_t",
    "mortality_t",
    "hospitalization_rate_t",
    "hospital_recovery_t",
    "hospital_mortality_t",
)


def build_latent_diagnostic_protocols(
    *,
    model: EpiForecaster | torch.nn.Module,
    model_outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor | None],
) -> dict[
    str,
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]],
]:
    """Build oracle-X0 rollout and teacher-forced one-step predictions."""
    required_targets = [target_key for _output_key, target_key in _STATE_TASKS.values()]
    required_targets.append("death_flow_target")
    if any(targets.get(key) is None for key in required_targets):
        return {}
    if any(key not in model_outputs for key in _RATE_KEYS):
        return {}

    state_targets = {
        name: targets[target_key]
        for name, (_output_key, target_key) in _STATE_TASKS.items()
    }
    assert all(isinstance(value, torch.Tensor) for value in state_targets.values())
    state_tensors = {
        name: value.float()
        for name, value in state_targets.items()
        if isinstance(value, torch.Tensor)
    }
    death_flow_target = targets["death_flow_target"]
    assert isinstance(death_flow_target, torch.Tensor)

    forward_model = getattr(model, "_orig_mod", model)
    rollforward = forward_model.sir_rollforward
    beta = model_outputs["beta_t"].float()
    batch_size, horizon = beta.shape
    ones = torch.ones(batch_size, device=beta.device, dtype=torch.float32)

    oracle = rollforward(
        beta_t=beta,
        gamma_t=model_outputs["gamma_t"].float(),
        mortality_t=model_outputs["mortality_t"].float(),
        hospitalization_rate_t=model_outputs["hospitalization_rate_t"].float(),
        hospital_recovery_t=model_outputs["hospital_recovery_t"].float(),
        hospital_mortality_t=model_outputs["hospital_mortality_t"].float(),
        S0=state_tensors["s"][:, 0],
        I0=state_tensors["i"][:, 0],
        H0=state_tensors["h"][:, 0],
        R0=state_tensors["r"][:, 0],
        population=ones,
    )
    # Index 0 of each state trajectory is the oracle initial condition (zero
    # error by construction); score only the rolled-forward steps 1..H.
    oracle_outputs = dict(oracle)
    for output_key, _target_key in _STATE_TASKS.values():
        oracle_outputs[output_key] = oracle[output_key][:, 1:]
    oracle_targets = {
        target_key: state_tensors[name][:, 1:]
        for name, (_output_key, target_key) in _STATE_TASKS.items()
    }
    oracle_targets["death_flow_target"] = death_flow_target

    # Treat each (sample, horizon) pair as a one-step batch item.
    teacher = rollforward(
        beta_t=rearrange(beta, "b h -> (b h) 1"),
        gamma_t=rearrange(model_outputs["gamma_t"].float(), "b h -> (b h) 1"),
        mortality_t=rearrange(model_outputs["mortality_t"].float(), "b h -> (b h) 1"),
        hospitalization_rate_t=rearrange(
            model_outputs["hospitalization_rate_t"].float(), "b h -> (b h) 1"
        ),
        hospital_recovery_t=rearrange(
            model_outputs["hospital_recovery_t"].float(), "b h -> (b h) 1"
        ),
        hospital_mortality_t=rearrange(
            model_outputs["hospital_mortality_t"].float(), "b h -> (b h) 1"
        ),
        S0=rearrange(state_tensors["s"][:, :-1], "b h -> (b h)"),
        I0=rearrange(state_tensors["i"][:, :-1], "b h -> (b h)"),
        H0=rearrange(state_tensors["h"][:, :-1], "b h -> (b h)"),
        R0=rearrange(state_tensors["r"][:, :-1], "b h -> (b h)"),
        population=torch.ones(
            batch_size * horizon, device=beta.device, dtype=torch.float32
        ),
    )
    teacher_outputs = {
        output_key: rearrange(teacher[output_key][:, 1], "(b h) -> b h", b=batch_size)
        for output_key, _target_key in _STATE_TASKS.values()
    }
    teacher_outputs["death_flow"] = rearrange(
        teacher["death_flow"], "(b h) 1 -> b h", b=batch_size
    )
    teacher_targets = {
        target_key: state_tensors[name][:, 1:]
        for name, (_output_key, target_key) in _STATE_TASKS.items()
    }
    teacher_targets["death_flow_target"] = death_flow_target

    return {
        "oracle_x0": (oracle_outputs, oracle_targets),
        "teacher_forced_one_step": (teacher_outputs, teacher_targets),
    }
