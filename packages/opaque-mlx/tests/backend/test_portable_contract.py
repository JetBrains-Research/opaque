"""Conformance checks for the MLX portable provider."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from opaque import autodiff, execution, ops, pytree, random, serialization
from opaque.api.engine import pytree as engine_pytree
from opaque.api.engine.backend import active_backend
from opaque.api.engine.primitive import (
    CORE_PRIMITIVES,
    UnsupportedPrimitiveError,
    supports,
)


def test_mlx_satisfies_complete_portable_profile() -> None:
    assert active_backend().name == "mlx"
    assert CORE_PRIMITIVES
    assert all(supports(primitive, "mlx") for primitive in CORE_PRIMITIVES)


def test_ops_preserve_native_arrays_and_capability_aware_precision() -> None:
    value = mx.array([3.0, 4.0], dtype=mx.float16)

    assert ops.is_array(value)
    assert ops.shape(value) == (2,)
    assert ops.dtype(value) == mx.float16
    assert ops.is_low_precision(value)
    assert ops.real_dtype(mx.complex64) == mx.float32
    assert ops.scalar(1.0, dtype=ops.dtype(value), like=value).dtype == mx.float16
    assert ops.zeros((2,), dtype=ops.dtype(value), like=value).dtype == mx.float16
    np.testing.assert_array_equal(ops.to_host(ops.square(value)), [9.0, 16.0])

    assert not ops.float64.supports("mlx")
    with pytest.raises(UnsupportedPrimitiveError, match=r"opaque.ops.float64.*mlx"):
        ops.float64()


def test_keyed_random_and_transforms_are_native_and_deterministic() -> None:
    first = random.normal(random.key((1 << 64) - 1), (2, 3), dtype=mx.float32)
    second = random.normal(random.key((1 << 64) - 1), (2, 3), dtype=mx.float32)
    np.testing.assert_array_equal(ops.to_host(first), ops.to_host(second))

    def loss(value: mx.array) -> mx.array:
        return ops.sum(ops.square(value))

    grads, value = autodiff.grad_and_value(loss)(mx.array([3.0, 4.0]))
    np.testing.assert_array_equal(ops.to_host(grads), [6.0, 8.0])
    assert ops.scalar_item(value) == pytest.approx(25.0)
    np.testing.assert_array_equal(
        ops.to_host(autodiff.vmap(lambda item: item * 2)(mx.array([1, 2]))), [2, 4]
    )

    gradients, (loss_value, auxiliary) = autodiff.grad_and_value(
        lambda item: (ops.sum(ops.square(item)), item * 3), has_aux=True
    )(mx.array([2.0]))
    np.testing.assert_array_equal(ops.to_host(gradients), [4.0])
    assert ops.scalar_item(loss_value) == pytest.approx(4.0)
    np.testing.assert_array_equal(ops.to_host(auxiliary), [6.0])


def test_pytree_norm_guard_serialization_and_execution_profiles() -> None:
    tree = {"nested": [mx.array([3.0, 4.0])], "flat.key": mx.array([12.0])}
    paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
    assert set(paths) == {("nested", 0), ("flat.key",)}
    restored_tree = pytree.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(ops.to_host(restored_tree["flat.key"]), [12.0])

    squared, grouped = engine_pytree._squared_l2_norms(
        leaves, ["all", "all"], dtype=mx.float32
    )
    assert ops.scalar_item(squared) == pytest.approx(169.0)
    assert ops.scalar_item(grouped["all"]) == pytest.approx(169.0)
    assert engine_pytree._squared_l2_norm_roundoff(leaves, dtype=mx.float32) > 0.0

    state = serialization.state_dict(tree)
    restored = serialization.from_state_dict(
        {"nested": [mx.zeros((2,), dtype=mx.float16)], "flat.key": mx.zeros((1,))},
        state,
    )
    assert restored["nested"][0].dtype == mx.float16
    np.testing.assert_array_equal(ops.to_host(restored["nested"][0]), [3.0, 4.0])

    assert execution.ExecutionProfile.COMPILATION.supports("mlx")
    assert execution.ExecutionProfile.CHECKPOINTING.supports("mlx")
    compiled = execution.compile(lambda value: value * 2)
    np.testing.assert_array_equal(ops.to_host(compiled(mx.array([1, 2]))), [2, 4])
