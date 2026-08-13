"""Tests for JAX execution transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from opaque.api.engine.autodiff import grad_and_value, vmap
from opaque.api.engine.backend import active_backend, clear_backend
from opaque.api.jax.backend import jax_backend
from opaque.execution import (
    ExecutionProfile,
    checkpoint,
    compile,
    optimize_saved_activations,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


def _square_sum(x: Any) -> Any:
    return jnp.sum(x**2)


def _numpy(value: Any) -> list:
    return np.asarray(value).tolist()


def test_compile_preserves_eager_values() -> None:
    compiled = compile(_square_sum)
    x = jnp.array([1.0, 2.0, 3.0])
    expected = _square_sum(x)
    got = compiled(x)
    assert _numpy(got) == _numpy(expected)
    assert active_backend() is not None
    assert active_backend().name == "jax"


def test_compile_preserves_gradients() -> None:
    compiled = compile(_square_sum)
    x = jnp.array([1.0, 2.0, 3.0])
    expected_grad = jax.grad(_square_sum)(x)
    got_grad = jax.grad(compiled)(x)
    assert _numpy(got_grad) == _numpy(expected_grad)


def test_checkpoint_preserves_eager_values_and_gradients() -> None:
    def block(x: Any, weight: Any) -> Any:
        return jax.nn.relu(x @ weight)

    def loss_fn(x: Any, weight: Any) -> Any:
        return jnp.sum(checkpoint(block)(x, weight))

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 8))
    weight = jax.random.normal(jax.random.PRNGKey(1), (8, 8))

    ref_out = jnp.sum(jax.nn.relu(x @ weight))
    ref_gx, ref_gw = jax.grad(lambda x_, w_: jnp.sum(jax.nn.relu(x_ @ w_)), (0, 1))(
        x, weight
    )

    out = loss_fn(x, weight)
    gx, gw = jax.grad(loss_fn, (0, 1))(x, weight)

    assert _numpy(out) == _numpy(ref_out)
    assert _numpy(gx) == _numpy(ref_gx)
    assert _numpy(gw) == _numpy(ref_gw)


def test_vmap_grad_checkpoint_matches_eager() -> None:
    def block(x: Any) -> Any:
        return jnp.sum(x**2)

    batched = jax.random.normal(jax.random.PRNGKey(2), (4, 3))

    grad_ckpt = grad_and_value(checkpoint(block))
    grad_ref = grad_and_value(block)

    got_grads, got_values = vmap(grad_ckpt)(batched)
    ref_grads, ref_values = vmap(grad_ref)(batched)

    assert _numpy(got_values) == _numpy(ref_values)
    assert _numpy(got_grads) == _numpy(ref_grads)


def test_full_checkpoint_grad_compile_order_matches_eager() -> None:
    def block(x: Any, weight: Any) -> Any:
        return jnp.sum(jax.nn.relu(x @ weight))

    x = jax.random.normal(jax.random.PRNGKey(3), (4, 8))
    weight = jax.random.normal(jax.random.PRNGKey(4), (8, 8))

    transformed = compile(lambda x_, w_: grad_and_value(checkpoint(block))(x_, w_))
    got_grad, got_value = transformed(x, weight)

    ref_grad, ref_value = grad_and_value(block)(x, weight)

    assert _numpy(got_value) == _numpy(ref_value)
    assert _numpy(got_grad) == _numpy(ref_grad)


def test_optimize_saved_activations_preserves_values_and_gradients() -> None:
    def fn(x: Any, weight: Any) -> Any:
        return jnp.sum(x @ weight)

    x = jax.random.normal(jax.random.PRNGKey(5), (4, 8))
    weight = jax.random.normal(jax.random.PRNGKey(6), (8, 8))

    ref_out = fn(x, weight)
    ref_gx, ref_gw = jax.grad(fn, (0, 1))(x, weight)

    optimized = optimize_saved_activations(fn)
    out = optimized(x, weight)
    gx, gw = jax.grad(optimized, (0, 1))(x, weight)

    assert _numpy(out) == _numpy(ref_out)
    assert _numpy(gx) == _numpy(ref_gx)
    assert _numpy(gw) == _numpy(ref_gw)


def test_execution_profiles_report_jax_supported() -> None:
    backend = jax_backend()
    assert ExecutionProfile.COMPILATION.supports(backend)
    assert ExecutionProfile.CHECKPOINTING.supports(backend)
    assert ExecutionProfile.SAVED_ACTIVATIONS.supports(backend)
