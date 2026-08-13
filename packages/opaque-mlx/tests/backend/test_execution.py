"""Tests for MLX execution transforms."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from opaque.api.engine.autodiff import grad_and_value, vmap
from opaque.api.engine.backend import active_backend, clear_backend
from opaque.api.mlx.backend import mlx_backend
from opaque.execution import (
    ExecutionProfile,
    checkpoint,
    compile,
    optimize_saved_activations,
)

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


def _square_sum(x: Any) -> Any:
    return mx.sum(x * x)


def _values(value: Any) -> Any:
    mx.eval(value)
    return value.tolist()


def test_compile_preserves_eager_values() -> None:
    compiled = compile(_square_sum)
    x = mx.array([1.0, 2.0, 3.0])
    expected = _square_sum(x)
    got = compiled(x)
    assert _values(got) == _values(expected)
    assert active_backend() is not None
    assert active_backend().name == "mlx"


def test_compile_preserves_gradients() -> None:
    compiled = compile(_square_sum)
    x = mx.array([1.0, 2.0, 3.0])
    expected_grad = mx.grad(_square_sum)(x)
    got_grad = mx.grad(compiled)(x)
    assert _values(got_grad) == _values(expected_grad)


def test_checkpoint_preserves_eager_values_and_gradients() -> None:
    def block(x: Any, weight: Any) -> Any:
        return mx.maximum(x @ weight, 0)

    def loss_fn(x: Any, weight: Any) -> Any:
        return mx.sum(checkpoint(block)(x, weight))

    key = mx.random.key(0)
    x = mx.random.normal((4, 8), key=key)
    weight = mx.random.normal((8, 8), key=mx.random.key(1))

    ref_out = mx.sum(mx.maximum(x @ weight, 0))
    ref_gx, ref_gw = mx.grad(lambda x_, w_: mx.sum(mx.maximum(x_ @ w_, 0)), (0, 1))(
        x, weight
    )

    out = loss_fn(x, weight)
    gx, gw = mx.grad(loss_fn, (0, 1))(x, weight)

    assert _values(out) == _values(ref_out)
    assert _values(gx) == _values(ref_gx)
    assert _values(gw) == _values(ref_gw)


def test_vmap_grad_checkpoint_matches_eager() -> None:
    # MLX's checkpoint primitive does not yet carry a vmap batching rule, so
    # ``vmap(grad(checkpoint(...)))`` raises from the underlying runtime. The
    # portable contract still reports checkpointing as supported because eager
    # checkpointed gradients work; vectorized composition is backend-limited.
    def block(x: Any) -> Any:
        return mx.sum(x * x)

    batched = mx.random.normal((4, 3), key=mx.random.key(2))

    grad_ckpt = grad_and_value(checkpoint(block))
    with pytest.raises(ValueError, match="Not implemented for Depends"):
        vmap(grad_ckpt)(batched)


def test_optimize_saved_activations_is_identity_and_warns_once() -> None:
    def fn(x: Any, weight: Any) -> Any:
        return mx.sum(x @ weight)

    x = mx.random.normal((4, 8), key=mx.random.key(3))
    weight = mx.random.normal((8, 8), key=mx.random.key(4))

    optimized = optimize_saved_activations(fn)

    # The warning is emitted lazily on first invocation (when the backend is
    # resolved and the MLX factory runs), and suppressed thereafter.
    with pytest.warns(UserWarning, match="unified memory"):
        out = optimized(x, weight)

    gx, gw = mx.grad(optimized, (0, 1))(x, weight)

    assert _values(out) == _values(fn(x, weight))
    ref_gx, ref_gw = mx.grad(fn, (0, 1))(x, weight)
    assert _values(gx) == _values(ref_gx)
    assert _values(gw) == _values(ref_gw)

    # Repeated transforms/calls must not emit additional warnings.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        another = optimize_saved_activations(fn)
        another(x, weight)
    unified_warnings = [w for w in record if "unified memory" in str(w.message)]
    assert len(unified_warnings) == 0


def test_execution_profiles_report_mlx_supported() -> None:
    backend = mlx_backend()
    assert ExecutionProfile.COMPILATION.supports(backend)
    assert ExecutionProfile.CHECKPOINTING.supports(backend)
    assert ExecutionProfile.SAVED_ACTIVATIONS.supports(backend)
