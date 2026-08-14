"""Unit tests for optimizer-oriented engine primitives on the JAX provider."""

from __future__ import annotations

import numpy as np
import pytest

from opaque import ops
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.jax import jax_backend

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _values(value):
    return np.asarray(value).tolist()


def test_rsqrt_is_reciprocal_square_root() -> None:
    backend = jax_backend()
    value = jnp.array([1.0, 4.0, 16.0])

    with use_backend(backend):
        result = ops.rsqrt(value)

    assert _values(result) == pytest.approx([1.0, 0.5, 0.25])


def test_pow_raises_value_to_exponent() -> None:
    backend = jax_backend()
    value = jnp.array([2.0, 3.0, 4.0])

    with use_backend(backend):
        result = ops.pow(value, 3.0)

    assert _values(result) == pytest.approx([8.0, 27.0, 64.0])


def test_mean_reduces_over_all_values_by_default() -> None:
    backend = jax_backend()
    value = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    with use_backend(backend):
        result = ops.mean(value)

    assert float(result) == pytest.approx(2.5)


def test_mean_reduces_along_axis() -> None:
    backend = jax_backend()
    value = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    with use_backend(backend):
        result = ops.mean(value, axis=1)

    assert _values(result) == pytest.approx([1.5, 3.5])


def test_reciprocal_is_elementwise_inverse() -> None:
    backend = jax_backend()
    value = jnp.array([1.0, 2.0, 4.0])

    with use_backend(backend):
        result = ops.reciprocal(value)

    assert _values(result) == pytest.approx([1.0, 0.5, 0.25])


def test_accumulator_dtype_follows_x64_configuration() -> None:
    backend = jax_backend()
    low_precision_dtypes = (jnp.float16, jnp.bfloat16)

    with use_backend(backend):
        for dtype in low_precision_dtypes:
            assert ops.accumulator_dtype(dtype) == jnp.float32

        array = jnp.array([1.0, 2.0], dtype=jnp.float32)
        assert ops.accumulator_dtype(array) == (
            jnp.float64 if jax.config.x64_enabled else jnp.float32
        )
        assert ops.accumulator_dtype(jnp.float32) == (
            jnp.float64 if jax.config.x64_enabled else jnp.float32
        )
