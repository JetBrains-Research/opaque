"""Unit tests for optimizer-oriented engine primitives on the Torch provider."""

from __future__ import annotations

import pytest
import torch

from opaque import ops
from opaque.api.engine.backend import clear_backend, ensure_backend


@pytest.fixture(autouse=True)
def _activate_torch_backend():
    clear_backend()
    ensure_backend(torch.tensor(0.0))
    yield
    clear_backend()


def test_rsqrt_is_reciprocal_square_root() -> None:
    value = torch.tensor([1.0, 4.0, 16.0])
    result = ops.rsqrt(value)
    expected = torch.tensor([1.0, 0.5, 0.25])
    assert torch.allclose(result, expected)


def test_pow_raises_value_to_exponent() -> None:
    value = torch.tensor([2.0, 3.0, 4.0])
    result = ops.pow(value, 3.0)
    expected = torch.tensor([8.0, 27.0, 64.0])
    assert torch.allclose(result, expected)


def test_mean_reduces_over_all_values_by_default() -> None:
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert ops.mean(value).item() == pytest.approx(2.5)


def test_mean_reduces_along_axis() -> None:
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = ops.mean(value, axis=1)
    expected = torch.tensor([1.5, 3.5])
    assert torch.allclose(result, expected)


def test_reciprocal_is_elementwise_inverse() -> None:
    value = torch.tensor([1.0, 2.0, 4.0])
    result = ops.reciprocal(value)
    expected = torch.tensor([1.0, 0.5, 0.25])
    assert torch.allclose(result, expected)


def test_accumulator_dtype_for_float32_is_float64_on_cpu() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert ops.accumulator_dtype(value) is torch.float64
    assert ops.accumulator_dtype(torch.float32) is torch.float64


def test_accumulator_dtype_for_low_precision_is_float32() -> None:
    assert ops.accumulator_dtype(torch.float16) is torch.float32
    assert ops.accumulator_dtype(torch.bfloat16) is torch.float32
    value = torch.tensor([1.0, 2.0], dtype=torch.float16)
    assert ops.accumulator_dtype(value) is torch.float32


def test_accumulator_dtype_uses_float32_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    value = torch.tensor([1.0, 2.0], device="mps", dtype=torch.float32)
    assert ops.accumulator_dtype(value) is torch.float32
