"""Tests for precision policy module."""

import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from utils.precision_policy import (
    PrecisionPolicy,
    create_precision_signature,
    resolve_precision_policy,
    validate_old_checkpoint_compatible,
)


class TestPrecisionPolicy:
    """Test suite for PrecisionPolicy dataclass."""

    def test_default_fp32_policy(self):
        """Default policy should be FP32 with no autocast."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )
        assert policy.param_dtype == torch.float32
        assert not policy.autocast_enabled
        assert policy.autocast_dtype is None

    def test_bf16_policy(self):
        """BF16 policy should have correct settings."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            optimizer_eps=1e-8,
            device_type="cuda",
        )
        assert policy.param_dtype == torch.float32
        assert policy.autocast_enabled
        assert policy.autocast_dtype == torch.bfloat16

    def test_division_eps_fp32(self):
        """Division epsilon should be 1e-8 for FP32."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )
        assert policy.division_eps() == 1e-8
        assert policy.division_eps(torch.float32) == 1e-8

    def test_division_eps_bf16(self):
        """Division epsilon should be 1e-4 for BF16."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            optimizer_eps=1e-8,
            device_type="cuda",
        )
        # When autocast is enabled, uses autocast_dtype
        assert policy.division_eps() == 1e-4
        assert policy.division_eps(torch.bfloat16) == 1e-4

    def test_division_eps_explicit_dtype(self):
        """Division epsilon should respect explicitly provided dtype."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )
        assert policy.division_eps(torch.bfloat16) == 1e-4
        assert policy.division_eps(torch.float32) == 1e-8


class TestResolvePrecisionPolicy:
    """Test suite for resolve_precision_policy function."""

    def test_cpu_no_autocast(self):
        """CPU device should not use autocast."""
        config = MagicMock()
        config.enable_mixed_precision = False
        config.optimizer_eps = None
        device = torch.device("cpu")

        policy = resolve_precision_policy(config, device)

        assert policy.param_dtype == torch.float32
        assert not policy.autocast_enabled
        assert policy.autocast_dtype is None
        assert policy.optimizer_eps == 1e-8
        assert policy.device_type == "cpu"

    def test_bf16_on_cuda(self):
        """BF16 autocast on CUDA when supported."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "bfloat16"
        config.optimizer_eps = None
        device = torch.device("cuda:0")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                with patch("torch.cuda.get_device_name", return_value="Test GPU"):
                    policy = resolve_precision_policy(config, device)

        assert policy.param_dtype == torch.float32
        assert policy.autocast_enabled
        assert policy.autocast_dtype == torch.bfloat16

    def test_fp16_rejected(self):
        """FP16 should raise ValueError."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "float16"
        config.optimizer_eps = None
        device = torch.device("cuda:0")

        with pytest.raises(ValueError, match="FP16 is no longer supported"):
            resolve_precision_policy(config, device)

    def test_bf16_on_cpu_rejected(self):
        """BF16 on CPU should raise ValueError."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "bfloat16"
        config.optimizer_eps = None
        device = torch.device("cpu")

        with pytest.raises(ValueError, match="BF16 autocast requested but device"):
            resolve_precision_policy(config, device)

    def test_bf16_without_cuda_rejected(self):
        """BF16 without CUDA should raise ValueError."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "bfloat16"
        config.optimizer_eps = None
        device = torch.device("cuda:0")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA is not available"):
                resolve_precision_policy(config, device)

    def test_bf16_on_unsupported_gpu_rejected(self):
        """BF16 on GPU without BF16 support should raise ValueError."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "bfloat16"
        config.optimizer_eps = None
        device = torch.device("cuda:0")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=False):
                with pytest.raises(ValueError, match="this GPU doesn't support BF16"):
                    resolve_precision_policy(config, device)

    def test_invalid_mixed_precision_dtype(self):
        """Invalid mixed_precision_dtype should raise ValueError."""
        config = MagicMock()
        config.enable_mixed_precision = True
        config.mixed_precision_dtype = "float64"
        config.optimizer_eps = None
        device = torch.device("cuda:0")

        with pytest.raises(ValueError, match="Unsupported mixed_precision_dtype"):
            resolve_precision_policy(config, device)

    def test_custom_optimizer_eps(self):
        """Custom optimizer_eps should be respected."""
        config = MagicMock()
        config.enable_mixed_precision = False
        config.optimizer_eps = 1e-6
        device = torch.device("cpu")

        policy = resolve_precision_policy(config, device)

        assert policy.optimizer_eps == 1e-6


class TestCheckpointValidation:
    """Test suite for checkpoint compatibility validation."""

    def test_valid_fp32_checkpoint(self):
        """Valid FP32 checkpoint should pass."""
        checkpoint = {
            "precision_signature": {
                "param_dtype": "float32",
                "autocast_enabled": False,
                "autocast_dtype": None,
            }
        }
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )

        # Should not raise
        validate_old_checkpoint_compatible(checkpoint, policy)

    def test_valid_bf16_checkpoint(self):
        """Valid BF16 checkpoint should pass."""
        checkpoint = {
            "precision_signature": {
                "param_dtype": "float32",
                "autocast_enabled": True,
                "autocast_dtype": "bfloat16",
            }
        }
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            optimizer_eps=1e-8,
            device_type="cuda",
        )

        # Should not raise
        validate_old_checkpoint_compatible(checkpoint, policy)

    def test_scaler_state_rejected(self):
        """Checkpoint with scaler state should raise ValueError."""
        checkpoint = {"scaler_state_dict": {}}
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )

        with pytest.raises(ValueError, match="GradScaler state"):
            validate_old_checkpoint_compatible(checkpoint, policy)

    def test_non_fp32_params_rejected(self):
        """Checkpoint with non-FP32 params should raise ValueError."""
        checkpoint = {
            "precision_signature": {
                "param_dtype": "float16",
                "autocast_enabled": False,
                "autocast_dtype": None,
            }
        }
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )

        with pytest.raises(ValueError, match="unsupported parameter dtype"):
            validate_old_checkpoint_compatible(checkpoint, policy)

    def test_fp16_autocast_rejected(self):
        """Checkpoint with FP16 autocast should raise ValueError."""
        checkpoint = {
            "precision_signature": {
                "param_dtype": "float32",
                "autocast_enabled": True,
                "autocast_dtype": "float16",
            }
        }
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            optimizer_eps=1e-8,
            device_type="cuda",
        )

        with pytest.raises(ValueError, match="unsupported autocast dtype"):
            validate_old_checkpoint_compatible(checkpoint, policy)

    def test_no_signature_passes(self):
        """Checkpoint without signature should pass (backward compat)."""
        checkpoint = {}
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )

        # Should not raise
        validate_old_checkpoint_compatible(checkpoint, policy)


class TestPrecisionSignature:
    """Test suite for create_precision_signature function."""

    def test_fp32_signature(self):
        """FP32 policy should create correct signature."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=False,
            autocast_dtype=None,
            optimizer_eps=1e-8,
            device_type="cpu",
        )

        sig = create_precision_signature(policy)

        assert sig["param_dtype"] == "float32"
        assert sig["autocast_enabled"] is False
        assert sig["autocast_dtype"] is None
        assert sig["optimizer_eps"] == 1e-8

    def test_bf16_signature(self):
        """BF16 policy should create correct signature."""
        policy = PrecisionPolicy(
            param_dtype=torch.float32,
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
            optimizer_eps=1e-6,
            device_type="cuda",
        )

        sig = create_precision_signature(policy)

        assert sig["param_dtype"] == "float32"
        assert sig["autocast_enabled"] is True
        assert sig["autocast_dtype"] == "bfloat16"
        assert sig["optimizer_eps"] == 1e-6
