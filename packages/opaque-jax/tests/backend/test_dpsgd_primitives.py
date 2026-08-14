"""JAX conformance checks for DP-SGD portable primitives."""

from __future__ import annotations

import numpy as np
import pytest

from opaque import ops, random
from opaque.api.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.distributed import sync
from opaque.jax import jax_backend
from opaque.random import fold_in, key
from opaque.types import PerGroup

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _unselected_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def test_inverse_cdf_math_preserves_native_dtype_and_device() -> None:
    backend = jax_backend()
    value = jnp.array([-0.5, 0.0, 0.5], dtype=jnp.float32)

    with use_backend(backend):
        restored = ops.erfinv(ops.erf(value))
        exponentiated = ops.exp(value)
        epsilon = ops.finfo_eps(value.dtype)

    assert isinstance(restored, jax.Array)
    assert restored.dtype == value.dtype
    assert restored.device == value.device
    assert exponentiated.dtype == value.dtype
    assert exponentiated.device == value.device
    np.testing.assert_allclose(restored, value)
    assert epsilon == pytest.approx(jnp.finfo(value.dtype).eps)


def test_keyed_normal_replays_with_like_dtype_and_device() -> None:
    backend = jax_backend()
    like = jnp.empty(0, dtype=jnp.float32)

    with use_backend(backend):
        first = random.normal(key(7), (2, 3), like=like)
        second = random.normal(key(7), (2, 3), like=like)

    assert isinstance(first, jax.Array)
    assert first.dtype == like.dtype
    assert first.device == like.device
    np.testing.assert_array_equal(first, second)


def test_keyed_normal_accepts_high_bit_folded_keys() -> None:
    backend = jax_backend()
    like = jnp.empty(0, dtype=jnp.float32)
    folded_key = fold_in(key(29), 0, 1)

    with use_backend(backend):
        first = random.normal(folded_key, (), like=like)
        second = random.normal(folded_key, (), like=like)

    assert isinstance(first, jax.Array)
    np.testing.assert_array_equal(first, second)


def test_adaptive_clipping_replays_with_native_arrays() -> None:
    def loss_fn(params, x, y):
        return ((x @ params - y) ** 2).mean()

    params = jnp.ones((2,), dtype=jnp.float32)
    batch_x = jnp.ones((3, 2), dtype=jnp.float32)
    batch_y = jnp.zeros((3,), dtype=jnp.float32)

    grad_fn, state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=1.0,
        key=key(17),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    (grads, aux), updated_state = grad_fn(params, batch_x, batch_y, state=state)

    assert ops.is_array(grads.pytree)
    assert ops.dtype(grads.pytree) == ops.dtype(params)
    assert ops.shape(aux.grad_norms) == (3,)
    assert ops.dtype(aux.grad_norms) == ops.real_dtype(params)
    assert updated_state._step == 1
    assert sync(updated_state) is updated_state

    clear_backend()
    replay_fn, replay_state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=1.0,
        key=key(17),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    (_replay_grads, _replay_aux), replay_updated_state = replay_fn(
        params, batch_x, batch_y, state=replay_state
    )

    assert replay_updated_state._next_clipping_norm == pytest.approx(
        updated_state._next_clipping_norm
    )


def test_per_group_adaptive_clipping_uses_native_arrays() -> None:
    def loss_fn(params, x):
        return (params["first"] * x[:2]).sum() + (params["second"] * x[2:]).sum()

    params = {
        "first": jnp.ones((2,), dtype=jnp.float32),
        "second": jnp.ones((1,), dtype=jnp.float32),
    }
    clipping_norm = PerGroup(
        {"first": "first", "second": "second"}, {"first": 1.0, "second": 0.5}
    )
    grad_fn, state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=clipping_norm,
        key=key(29),
        batch_argnums=1,
        return_aux=True,
    )

    (grads, aux), updated_state = grad_fn(
        params,
        jnp.ones((3, 3), dtype=jnp.float32),
        state=state,
    )

    assert all(ops.is_array(value) for value in grads.pytree.values())
    assert ops.shape(aux.grad_norms) == (3,)
    assert isinstance(updated_state._next_clipping_norm, PerGroup)
    assert all(value > 0 for value in updated_state._next_clipping_norm.values.values())
