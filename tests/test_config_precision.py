"""Tests for precision-related config validation."""

import pytest

from models.configs import TrainingParams


class TestPrecisionConfigValidation:
    """Test suite for precision configuration validation in TrainingParams."""

    def test_default_precision_config(self):
        """Default config should use FP32 params and BF16 autocast."""
        config = TrainingParams()

        assert config.parameter_dtype == "float32"
        assert config.mixed_precision_dtype == "bfloat16"
        assert config.enable_mixed_precision is True
        assert config.optimizer_eps is None

    def test_fp16_rejected(self):
        """FP16 mixed_precision_dtype should raise ValueError."""
        with pytest.raises(ValueError, match="FP16 is no longer supported"):
            TrainingParams(
                mixed_precision_dtype="float16",
                enable_mixed_precision=True,
            )

    def test_invalid_mixed_precision_dtype(self):
        """Invalid mixed_precision_dtype should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported mixed_precision_dtype"):
            TrainingParams(
                mixed_precision_dtype="float64",
                enable_mixed_precision=True,
            )

    def test_non_float32_parameter_dtype_rejected(self):
        """parameter_dtype other than float32 should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported parameter_dtype"):
            TrainingParams(parameter_dtype="float16")

    def test_bf16_parameter_dtype_rejected(self):
        """bfloat16 parameter_dtype should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported parameter_dtype"):
            TrainingParams(parameter_dtype="bfloat16")

    def test_negative_optimizer_eps_rejected(self):
        """Negative optimizer_eps should raise ValueError."""
        with pytest.raises(ValueError, match="optimizer_eps must be positive"):
            TrainingParams(optimizer_eps=-1e-8)

    def test_zero_optimizer_eps_rejected(self):
        """Zero optimizer_eps should raise ValueError."""
        with pytest.raises(ValueError, match="optimizer_eps must be positive"):
            TrainingParams(optimizer_eps=0.0)

    def test_valid_bf16_config(self):
        """Valid BF16 config should work."""
        config = TrainingParams(
            parameter_dtype="float32",
            mixed_precision_dtype="bfloat16",
            enable_mixed_precision=True,
            optimizer_eps=1e-6,
        )

        assert config.parameter_dtype == "float32"
        assert config.mixed_precision_dtype == "bfloat16"
        assert config.enable_mixed_precision is True
        assert config.optimizer_eps == 1e-6

    def test_disabled_mixed_precision(self):
        """Disabled mixed precision should work with any valid dtype string."""
        # Even with float16 string, if mixed precision is disabled it should work
        # Actually no - the validation checks the string regardless
        config = TrainingParams(
            enable_mixed_precision=False,
            mixed_precision_dtype="bfloat16",
        )

        assert config.enable_mixed_precision is False
        assert config.mixed_precision_dtype == "bfloat16"

    def test_case_insensitive_dtype_check(self):
        """Mixed precision dtype check should be case-insensitive."""
        # BF16 in various cases should all fail validation if it's FP16
        with pytest.raises(ValueError, match="FP16 is no longer supported"):
            TrainingParams(
                mixed_precision_dtype="FLOAT16",
                enable_mixed_precision=True,
            )

        with pytest.raises(ValueError, match="FP16 is no longer supported"):
            TrainingParams(
                mixed_precision_dtype="Float16",
                enable_mixed_precision=True,
            )

    def test_optimizer_eps_none_is_valid(self):
        """None optimizer_eps should use default."""
        config = TrainingParams(optimizer_eps=None)

        assert config.optimizer_eps is None

    def test_optimizer_eps_positive_values(self):
        """Various positive optimizer_eps values should work."""
        # Common epsilon values
        for eps in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
            config = TrainingParams(optimizer_eps=eps)
            assert config.optimizer_eps == eps
