"""Behavioral coverage for the MLX implementation of the backend protocol."""

from __future__ import annotations

from typing import Any

import pytest

from opaque.api.engine.backend import (
    Backend,
    TorchBackend,
    active_backend,
    set_backend,
    use_backend,
)
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.types import PerGroup
from opaque.mlx import mlx_backend
from opaque.random import key

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    yield
    set_backend(TorchBackend())


def _values(value: Any) -> Any:
    mx.eval(value)
    return value.tolist()


def test_satisfies_protocol_and_array_primitives() -> None:
    backend = mlx_backend()
    x = mx.array([3.0, 4.0])

    assert isinstance(backend, Backend)
    assert backend.name == "mlx"
    assert backend.float32 == mx.float32
    assert _values(backend.square(x)) == [9.0, 16.0]
    assert backend.sum(backend.square(x)).item() == pytest.approx(25.0)
    assert backend.sqrt(backend.sum(backend.square(x))).item() == pytest.approx(5.0)
    assert backend.is_array(x)
    assert backend.is_floating(x)
    assert backend.is_floating(mx.float32)
    assert backend.concatenate([x, x]).shape == (4,)
    assert backend.scalar(1.0, dtype=mx.float32, like=x).dtype == mx.float32
    assert backend.promote_dtype(mx.float16, mx.float32) == mx.float32


@pytest.mark.parametrize(
    ("dtype", "low_precision", "floating"),
    [
        (mx.float16, True, True),
        (mx.bfloat16, True, True),
        (mx.float32, False, True),
        (mx.int32, False, False),
    ],
)
def test_dtype_predicates_for_arrays_and_dtypes(dtype, low_precision, floating) -> None:
    backend = mlx_backend()
    array = mx.array([1], dtype=dtype)

    assert backend.is_low_precision(dtype) is low_precision
    assert backend.is_low_precision(array) is low_precision
    assert backend.is_floating(dtype) is floating
    assert backend.is_floating(array) is floating
    assert backend.is_complex(dtype) is False
    assert backend.is_complex(array) is False


def test_value_and_grad_normalizes_value_and_aux_order() -> None:
    backend = mlx_backend()
    x = mx.array([2.0, 3.0])

    grad, value = backend.value_and_grad(lambda t: mx.sum(t * t))(x)
    assert _values(grad) == [4.0, 6.0]
    assert value.item() == pytest.approx(13.0)

    def loss_with_aux(t):
        return mx.sum(t * t), {"mean": mx.mean(t)}

    grad, (value, aux) = backend.value_and_grad(loss_with_aux, has_aux=True)(x)
    assert _values(grad) == [4.0, 6.0]
    assert value.item() == pytest.approx(13.0)
    assert aux["mean"].item() == pytest.approx(2.5)


def test_vmap_composes_with_value_and_grad() -> None:
    backend = mlx_backend()
    weights = mx.array([2.0, 3.0])
    examples = mx.array([[1.0, 4.0], [5.0, 6.0]])

    grad_and_value = backend.value_and_grad(lambda w, x: mx.sum(w * x))
    grads, values = backend.vmap(grad_and_value, in_axes=(None, 0))(weights, examples)

    assert _values(grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(values) == [14.0, 28.0]

    grad_and_value_with_aux = backend.value_and_grad(
        lambda w, x: (mx.sum(w * x), mx.sum(x)), has_aux=True
    )
    grads, (values, aux) = backend.vmap(grad_and_value_with_aux, in_axes=(None, 0))(
        weights, examples
    )
    assert _values(grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(values) == [14.0, 28.0]
    assert _values(aux) == [5.0, 11.0]


def test_pytree_paths_preserve_flat_dotted_keys_for_per_group_clipping() -> None:
    backend = mlx_backend()
    tree = {"layers.0.weight": mx.array([3.0, 4.0]), "bias": mx.array([6.0])}
    paths, leaves, treedef = backend.tree_flatten_with_paths(tree)

    assert set(paths) == {("layers.0.weight",), ("bias",)}
    assert backend.tree_unflatten(treedef, leaves) is not tree

    groups = PerGroup(
        {("layers.0.weight",): "weight", ("bias",): "bias"},
        {"weight": 1.0, "bias": 2.0},
    )
    with use_backend(backend):
        clipped, aux = clip_pytree(tree, groups)

    assert aux.group_norms is not None
    assert _values(clipped["layers.0.weight"]) == pytest.approx([0.6, 0.8])
    assert _values(clipped["bias"]) == pytest.approx([2.0])


def test_empty_tree_and_nonfinite_sanitization() -> None:
    backend = mlx_backend()
    nonfinite = mx.array([float("nan"), float("inf"), -float("inf"), 2.0])

    assert backend.tree_leaves({}) == []
    assert backend.tree_flatten({})[0] == []
    assert _values(backend.nan_to_num(nonfinite)) == [0.0, 0.0, 0.0, 2.0]
    assert _values(backend.clamp(mx.array([-1.0, 0.5, 2.0]), lo=0.0, hi=1.0)) == [
        0.0,
        0.5,
        1.0,
    ]
    with use_backend(backend):
        clipped, aux = clip_pytree({}, 1.0)
    assert clipped == {}
    assert aux.norm.item() == pytest.approx(0.0)


def test_keyed_normal_sampling_is_repeatable() -> None:
    backend = mlx_backend()
    first = backend.normal((2, 3), dtype=mx.float32, generator=backend.generator(key(7)))
    second = backend.normal((2, 3), dtype=mx.float32, generator=backend.generator(key(7)))
    different = backend.normal((2, 3), dtype=mx.float32, generator=backend.generator(key(8)))

    assert _values(first) == _values(second)
    assert _values(first) != _values(different)


def test_registry_swap_and_restore() -> None:
    backend = mlx_backend()
    previous = active_backend()

    with use_backend(backend) as yielded:
        assert yielded is backend
        assert active_backend() is backend

    assert active_backend() is previous
    assert active_backend().name != "mlx"


def test_tree_leaves_filters_out_non_array_leaves() -> None:
    backend = mlx_backend()
    tree = {"w": mx.array([1.0, 2.0]), "meta": 5, "nested": {"b": mx.array([3.0]), "c": "foo"}}
    leaves = backend.tree_leaves(tree)

    assert len(leaves) == 2
    assert all(backend.is_array(leaf) for leaf in leaves)
    assert _values(leaves[0]) == [1.0, 2.0]
    assert _values(leaves[1]) == [3.0]
