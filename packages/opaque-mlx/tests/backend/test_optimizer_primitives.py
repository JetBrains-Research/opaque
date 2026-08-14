"""Unit tests for optimizer-oriented engine primitives on the MLX provider."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.mlx import mlx_backend

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _values(value):
    mx.eval(value)
    return value.tolist()


def test_rsqrt_is_reciprocal_square_root() -> None:
    backend = mlx_backend()
    value = mx.array([1.0, 4.0, 16.0])

    with use_backend(backend):
        result = ops.rsqrt(value)

    assert _values(result) == pytest.approx([1.0, 0.5, 0.25])


def test_pow_raises_value_to_exponent() -> None:
    backend = mlx_backend()
    value = mx.array([2.0, 3.0, 4.0])

    with use_backend(backend):
        result = ops.pow(value, 3.0)

    assert _values(result) == pytest.approx([8.0, 27.0, 64.0])


def test_mean_reduces_over_all_values_by_default() -> None:
    backend = mlx_backend()
    value = mx.array([[1.0, 2.0], [3.0, 4.0]])

    with use_backend(backend):
        result = ops.mean(value)

    assert result.item() == pytest.approx(2.5)


def test_mean_reduces_along_axis() -> None:
    backend = mlx_backend()
    value = mx.array([[1.0, 2.0], [3.0, 4.0]])

    with use_backend(backend):
        result = ops.mean(value, axis=1)

    assert _values(result) == pytest.approx([1.5, 3.5])


def test_reciprocal_is_elementwise_inverse() -> None:
    backend = mlx_backend()
    value = mx.array([1.0, 2.0, 4.0])

    with use_backend(backend):
        result = ops.reciprocal(value)

    assert _values(result) == pytest.approx([1.0, 0.5, 0.25])


def test_accumulator_dtype_is_widest_stable_float() -> None:
    backend = mlx_backend()

    with use_backend(backend):
        assert (
            ops.accumulator_dtype(mx.array([1.0, 2.0], dtype=mx.float32)) == mx.float32
        )
        assert ops.accumulator_dtype(mx.float32) == mx.float32
        assert ops.accumulator_dtype(mx.float16) == mx.float32
        assert ops.accumulator_dtype(mx.bfloat16) == mx.float32
