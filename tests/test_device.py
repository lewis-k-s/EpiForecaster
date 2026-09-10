from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from utils.device import (
    get_dtype_safe_eps,
    get_model_eps,
    setup_tensor_core_optimizations,
)


@pytest.mark.parametrize("enable_tf32", [True, False])
def test_setup_tensor_core_optimizations_uses_one_api(enable_tf32: bool) -> None:
    matmul_backend = MagicMock(spec_set=["allow_tf32"])
    cuda_backend = SimpleNamespace(matmul=matmul_backend)
    cudnn_backend = MagicMock(spec_set=["allow_tf32"])

    with (
        patch("torch.backends.cuda", cuda_backend),
        patch("torch.backends.cudnn", cudnn_backend),
    ):
        setup_tensor_core_optimizations(torch.device("cuda"), enable_tf32=enable_tf32)

    assert matmul_backend.allow_tf32 is enable_tf32
    assert cudnn_backend.allow_tf32 is enable_tf32


def test_setup_tensor_core_optimizations_skips_cpu() -> None:
    with patch("torch.backends.cuda"):
        setup_tensor_core_optimizations(torch.device("cpu"))


class TestDtypeSafeEps:
    """Test the dtype-safe epsilon utility."""

    def test_float16_returns_larger_eps(self):
        assert get_dtype_safe_eps(torch.float16) == 1e-4

    def test_float32_returns_base_eps(self):
        assert get_dtype_safe_eps(torch.float32) == 1e-8

    def test_float64_returns_base_eps(self):
        assert get_dtype_safe_eps(torch.float64) == 1e-8

    def test_custom_base_eps(self):
        assert get_dtype_safe_eps(torch.float32, base_eps=1e-6) == 1e-6

    def test_custom_float16_eps(self):
        assert get_dtype_safe_eps(torch.float16, float16_eps=1e-3) == 1e-3

    def test_float16_eps_is_larger_than_base(self):
        eps_f16 = get_dtype_safe_eps(
            torch.float16,
            base_eps=1e-8,
            float16_eps=1e-4,
        )
        eps_f32 = get_dtype_safe_eps(
            torch.float32,
            base_eps=1e-8,
            float16_eps=1e-4,
        )
        assert eps_f16 > eps_f32

    def test_bfloat16_returns_base_eps(self):
        assert get_dtype_safe_eps(torch.bfloat16) == 1e-8


class TestGetModelEps:
    """Test device-aware epsilon selection."""

    def test_cpu_returns_float32_eps(self):
        assert get_model_eps(torch.device("cpu")) == 1e-8

    def test_cuda_returns_float16_eps(self):
        assert get_model_eps(torch.device("cuda:0")) == 1e-4

    def test_mps_returns_float16_eps(self):
        assert get_model_eps(torch.device("mps")) == 1e-4

    def test_string_device_cpu(self):
        assert get_model_eps("cpu") == 1e-8

    def test_string_device_cuda(self):
        assert get_model_eps("cuda") == 1e-4


class TestEpsPreventsDivisionByZero:
    """Test epsilon behavior in representative normalization operations."""

    def test_division_with_float16_eps_is_finite(self):
        numerator = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        denominator = torch.tensor([1e-7, 1e-7, 1e-7], dtype=torch.float16)

        result = numerator / (
            denominator + get_dtype_safe_eps(torch.float16)
        )

        assert torch.isfinite(result).all()

    def test_normalization_with_dtype_aware_eps_is_finite(self):
        matrix = torch.randn(10, 10, dtype=torch.float16).abs()
        matrix = matrix / matrix.sum()
        dim_sum = matrix.sum(dim=0, keepdim=True)

        normalized = matrix / (dim_sum + get_dtype_safe_eps(matrix.dtype))

        assert torch.isfinite(normalized).all()
