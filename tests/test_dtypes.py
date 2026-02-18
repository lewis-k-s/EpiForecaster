"""Tests for dtype utilities."""

import torch

from utils.dtypes import (
    get_dtype_safe_eps,
    get_model_eps,
)


class TestDtypeSafeEps:
    """Test suite for dtype-safe epsilon utility."""

    def test_float16_returns_larger_eps(self):
        """Float16 should return 1e-4 to avoid underflow."""
        eps = get_dtype_safe_eps(torch.float16)
        assert eps == 1e-4

    def test_float32_returns_base_eps(self):
        """Float32 should return default 1e-8."""
        eps = get_dtype_safe_eps(torch.float32)
        assert eps == 1e-8

    def test_float64_returns_base_eps(self):
        """Float64 should return default 1e-8."""
        eps = get_dtype_safe_eps(torch.float64)
        assert eps == 1e-8

    def test_custom_base_eps(self):
        """Custom base epsilon should be respected for float32."""
        eps = get_dtype_safe_eps(torch.float32, base_eps=1e-6)
        assert eps == 1e-6

    def test_custom_float16_eps(self):
        """Custom float16 epsilon should be respected."""
        eps = get_dtype_safe_eps(torch.float16, float16_eps=1e-3)
        assert eps == 1e-3

    def test_float16_eps_always_larger_than_base(self):
        """Float16 epsilon should always be larger than base to prevent underflow."""
        eps_f16 = get_dtype_safe_eps(torch.float16, base_eps=1e-8, float16_eps=1e-4)
        eps_f32 = get_dtype_safe_eps(torch.float32, base_eps=1e-8, float16_eps=1e-4)
        assert eps_f16 > eps_f32

    def test_bfloat16_returns_base_eps(self):
        """BFloat16 should return base epsilon (has better range than float16)."""
        eps = get_dtype_safe_eps(torch.bfloat16)
        assert eps == 1e-8


class TestGetModelEps:
    """Test suite for device-aware epsilon retrieval."""

    def test_cpu_returns_float32_eps(self):
        """CPU device should return float32 epsilon."""
        device = torch.device("cpu")
        eps = get_model_eps(device)
        assert eps == 1e-8

    def test_cuda_returns_float16_eps(self):
        """CUDA device should return float16-safe epsilon."""
        device = torch.device("cuda:0")
        eps = get_model_eps(device)
        assert eps == 1e-4

    def test_mps_returns_float16_eps(self):
        """MPS device should return float16-safe epsilon."""
        device = torch.device("mps")
        eps = get_model_eps(device)
        assert eps == 1e-4

    def test_string_device_cpu(self):
        """String 'cpu' should work correctly."""
        eps = get_model_eps("cpu")
        assert eps == 1e-8

    def test_string_device_cuda(self):
        """String 'cuda' should work correctly."""
        eps = get_model_eps("cuda")
        assert eps == 1e-4


class TestEpsPreventsDivisionByZero:
    """Integration tests that eps actually prevents numerical issues."""

    def test_division_with_float16_eps(self):
        """Using float16-safe epsilon should prevent division by zero in normalization."""
        # Simulate near-zero denominator scenario
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        y = torch.tensor([1e-7, 1e-7, 1e-7], dtype=torch.float16)  # Small values

        # With 1e-8 epsilon, this would underflow to 0 in float16
        bad_eps = 1e-8
        _ = x / (y + bad_eps)  # May produce inf or nan - intentionally unchecked

        # With 1e-4 epsilon, safe division
        good_eps = get_dtype_safe_eps(torch.float16)
        result_good = x / (y + good_eps)

        # Good result should be finite
        assert torch.isfinite(result_good).all()

    def test_normalization_with_dtype_aware_eps(self):
        """Normalization should work correctly with dtype-aware epsilon."""
        # Simulate mobility matrix normalization
        matrix = torch.randn(10, 10, dtype=torch.float16).abs()
        matrix = matrix / matrix.sum()  # Normalize to small values

        # Sum along dimension (may be very small)
        dim_sum = matrix.sum(dim=0, keepdim=True)

        # Use dtype-safe epsilon
        eps = get_dtype_safe_eps(matrix.dtype)
        normalized = matrix / (dim_sum + eps)

        # Result should be finite
        assert torch.isfinite(normalized).all()
