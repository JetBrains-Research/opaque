"""MLX conformance checks for DP-SGD portable primitives."""

from __future__ import annotations

import pytest

from opaque import ops, random
from opaque.api.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.distributed import sync
from opaque.mlx import mlx_backend
from opaque.random import key
from opaque.types import PerGroup

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _unselected_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _values(value):
    mx.eval(value)
    return value.tolist()


def test_inverse_cdf_math_preserves_native_dtype() -> None:
    backend = mlx_backend()
    value = mx.array([-0.5, 0.0, 0.5], dtype=mx.float32)

    with use_backend(backend):
        restored = ops.erfinv(ops.erf(value))
        exponentiated = ops.exp(value)
        epsilon = ops.finfo_eps(value.dtype)

    assert ops.is_array(restored)
    assert restored.dtype == value.dtype
    assert exponentiated.dtype == value.dtype
    assert _values(restored) == pytest.approx(_values(value))
    assert epsilon == pytest.approx(mx.finfo(value.dtype).eps)


def test_keyed_normal_replays_with_like_dtype() -> None:
    backend = mlx_backend()
    like = mx.zeros((0,), dtype=mx.float32)

    with use_backend(backend):
        first = random.normal(key(7), (2, 3), like=like)
        second = random.normal(key(7), (2, 3), like=like)

    assert ops.is_array(first)
    assert first.dtype == like.dtype
    assert _values(first) == _values(second)


def test_adaptive_clipping_replays_with_native_arrays() -> None:
    def loss_fn(params, x, y):
        return ((x @ params - y) ** 2).mean()

    params = mx.ones((2,), dtype=mx.float32)
    batch_x = mx.ones((3, 2), dtype=mx.float32)
    batch_y = mx.zeros((3,), dtype=mx.float32)

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
        "first": mx.ones((2,), dtype=mx.float32),
        "second": mx.ones((1,), dtype=mx.float32),
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
        mx.ones((3, 3), dtype=mx.float32),
        state=state,
    )

    assert all(ops.is_array(value) for value in grads.pytree.values())
    assert ops.shape(aux.grad_norms) == (3,)
    assert isinstance(updated_state._next_clipping_norm, PerGroup)
    assert all(value > 0 for value in updated_state._next_clipping_norm.values.values())
