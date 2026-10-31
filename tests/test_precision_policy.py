"""Precision boundaries that affect numerical stability and checkpoint loading."""

import pytest
import torch

from models.configs import TrainingParams
from utils.precision_policy import (
    MODEL_PARAM_DTYPE,
    PrecisionPolicy,
    create_precision_signature,
    model_scalar,
    resolve_precision_policy,
    validate_old_checkpoint_compatible,
)


def _policy(*, autocast: bool = False) -> PrecisionPolicy:
    return PrecisionPolicy(
        param_dtype=torch.float32,
        autocast_enabled=autocast,
        autocast_dtype=torch.bfloat16 if autocast else None,
        optimizer_eps=1e-8,
        device_type="cuda" if autocast else "cpu",
    )


def test_scalar_and_division_epsilon_follow_precision() -> None:
    assert model_scalar(1.25).dtype == MODEL_PARAM_DTYPE
    assert _policy().division_eps() == 1e-8
    assert _policy().division_eps(torch.bfloat16) == 1e-4
    assert _policy(autocast=True).division_eps() == 1e-4
    assert _policy(autocast=True).division_eps(torch.float32) == 1e-8


def test_resolved_precision_requires_supported_cuda_for_bf16(monkeypatch) -> None:
    cpu_config = TrainingParams(enable_mixed_precision=False, optimizer_eps=1e-6)
    cpu_policy = resolve_precision_policy(cpu_config, torch.device("cpu"))
    assert cpu_policy.param_dtype == torch.float32
    assert cpu_policy.autocast_enabled is False
    assert cpu_policy.optimizer_eps == 1e-6

    bf16_config = TrainingParams(enable_mixed_precision=True)
    with pytest.raises(ValueError, match="BF16 autocast requested but device"):
        resolve_precision_policy(bf16_config, torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_precision_policy(bf16_config, torch.device("cuda:0"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="doesn't support BF16"):
        resolve_precision_policy(bf16_config, torch.device("cuda:0"))

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    policy = resolve_precision_policy(bf16_config, torch.device("cuda:0"))
    assert policy.autocast_enabled is True
    assert policy.autocast_dtype == torch.bfloat16
    assert policy.param_dtype == torch.float32


def test_checkpoint_precision_signature_accepts_supported_state() -> None:
    for policy in (_policy(), _policy(autocast=True)):
        signature = create_precision_signature(policy)
        assert signature["param_dtype"] == "float32"
        assert signature["autocast_dtype"] == (
            "bfloat16" if policy.autocast_enabled else None
        )
        validate_old_checkpoint_compatible({"precision_signature": signature}, policy)
    validate_old_checkpoint_compatible({}, _policy())


def test_checkpoint_precision_rejects_unsupported_state() -> None:
    invalid = [
        ({"scaler_state_dict": {}}, "GradScaler state"),
        ({"precision_signature": {"param_dtype": "float16"}}, "parameter dtype"),
        (
            {
                "precision_signature": {
                    "param_dtype": "float32",
                    "autocast_dtype": "float16",
                }
            },
            "autocast dtype",
        ),
    ]
    for checkpoint, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate_old_checkpoint_compatible(checkpoint, _policy())
