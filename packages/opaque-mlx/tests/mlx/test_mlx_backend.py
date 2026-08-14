"""Behavioral coverage for the MLX implementation of the backend protocol."""

from __future__ import annotations

from typing import Any

import pytest

from opaque import autodiff, ops, pytree, random
from opaque.api.engine.backend import (
    Backend,
    active_backend,
    clear_backend,
    use_backend,
)
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.types import PerGroup
from opaque.mlx import mlx_backend
from opaque.random import key

mx = pytest.importorskip("mlx.core")

_SEED_BOUNDARIES = (0, 1, 2**63 - 1, 2**63, 2**64 - 2, 2**64 - 1)


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _values(value: Any) -> Any:
    mx.eval(value)
    return value.tolist()


def test_satisfies_protocol_and_array_primitives() -> None:
    backend = mlx_backend()
    x = mx.array([3.0, 4.0])

    assert isinstance(backend, Backend)
    assert backend.name == "mlx"
    with use_backend(backend):
        assert ops.float32() == mx.float32
        assert _values(ops.square(x)) == [9.0, 16.0]
        assert ops.sum(ops.square(x)).item() == pytest.approx(25.0)
        assert ops.sqrt(ops.sum(ops.square(x))).item() == pytest.approx(5.0)
        assert ops.is_array(x)
        assert ops.is_floating(x)
        assert ops.is_floating(mx.float32)
        assert ops.concatenate([x, x]).shape == (4,)
        assert ops.scalar(1.0, dtype=mx.float32, like=x).dtype == mx.float32
        assert ops.promote_dtype(mx.float16, mx.float32) == mx.float32


def test_factory_returns_identity_with_decorator_registered_implementations() -> None:
    from opaque.api.mlx.backend import _core as provider

    backend = mlx_backend()

    assert not hasattr(backend, "square")
    assert ops.square.resolve(backend) is provider.square


def test_array_automatically_activates_provider() -> None:
    result = ops.square(mx.array([2.0, 3.0]))

    assert _values(result) == [4.0, 9.0]
    assert active_backend() is not None
    assert active_backend().name == "mlx"


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

    with use_backend(backend):
        assert ops.is_low_precision(dtype) is low_precision
        assert ops.is_low_precision(array) is low_precision
        assert ops.is_floating(dtype) is floating
        assert ops.is_floating(array) is floating
        assert ops.is_complex(dtype) is False
        assert ops.is_complex(array) is False


def test_complex_dtype_predicate() -> None:
    backend = mlx_backend()
    complex_value = mx.array([1 + 2j], dtype=mx.complex64)

    with use_backend(backend):
        assert ops.is_complex(mx.complex64)
        assert ops.is_complex(complex_value)


def test_value_and_grad_normalizes_value_and_aux_order() -> None:
    backend = mlx_backend()
    x = mx.array([2.0, 3.0])

    def loss_with_aux(t):
        return mx.sum(t * t), {"mean": mx.mean(t)}

    with use_backend(backend):
        grad, value = autodiff.grad_and_value(lambda t: mx.sum(t * t))(x)
        aux_grad, (aux_value, aux) = autodiff.grad_and_value(
            loss_with_aux, has_aux=True
        )(x)

    assert _values(grad) == [4.0, 6.0]
    assert value.item() == pytest.approx(13.0)
    assert _values(aux_grad) == [4.0, 6.0]
    assert aux_value.item() == pytest.approx(13.0)
    assert aux["mean"].item() == pytest.approx(2.5)


def test_vmap_composes_with_value_and_grad() -> None:
    backend = mlx_backend()
    weights = mx.array([2.0, 3.0])
    examples = mx.array([[1.0, 4.0], [5.0, 6.0]])

    with use_backend(backend):
        grad_and_value = autodiff.grad_and_value(lambda w, x: mx.sum(w * x))
        grads, values = autodiff.vmap(grad_and_value, in_axes=(None, 0))(
            weights, examples
        )
        grad_and_value_with_aux = autodiff.grad_and_value(
            lambda w, x: (mx.sum(w * x), mx.sum(x)), has_aux=True
        )
        aux_grads, (aux_values, aux) = autodiff.vmap(
            grad_and_value_with_aux, in_axes=(None, 0)
        )(weights, examples)

    assert _values(grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(values) == [14.0, 28.0]
    assert _values(aux_grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(aux_values) == [14.0, 28.0]
    assert _values(aux) == [5.0, 11.0]


def test_pytree_paths_preserve_flat_dotted_keys_for_per_group_clipping() -> None:
    backend = mlx_backend()
    tree = {"layers.0.weight": mx.array([3.0, 4.0]), "bias": mx.array([6.0])}
    with use_backend(backend):
        paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
        rebuilt = pytree.tree_unflatten(treedef, leaves)

    assert set(paths) == {("layers.0.weight",), ("bias",)}
    assert rebuilt is not tree

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

    with use_backend(backend):
        assert pytree.tree_leaves({}) == []
        assert pytree.tree_flatten({})[0] == []
        assert _values(ops.nan_to_num(nonfinite)) == [0.0, 0.0, 0.0, 2.0]
        assert _values(ops.clamp(mx.array([-1.0, 0.5, 2.0]), lo=0.0, hi=1.0)) == [
            0.0,
            0.5,
            1.0,
        ]
        clipped, aux = clip_pytree({}, 1.0)
    assert clipped == {}
    assert aux.norm.item() == pytest.approx(0.0)


def test_keyed_normal_sampling_is_repeatable() -> None:
    backend = mlx_backend()
    with use_backend(backend):
        first = random.normal(key(7), (2, 3), dtype=mx.float32)
        second = random.normal(key(7), (2, 3), dtype=mx.float32)
        different = random.normal(key(8), (2, 3), dtype=mx.float32)

    assert _values(first) == _values(second)
    assert _values(first) != _values(different)


@pytest.mark.parametrize("seed", _SEED_BOUNDARIES)
def test_keyed_normal_replays_at_engine_seed_boundaries(seed: int) -> None:
    backend = mlx_backend()
    rng_key = key(seed)

    with use_backend(backend):
        first = random.normal(rng_key, (64,), dtype=mx.float32)
        second = random.normal(rng_key, (64,), dtype=mx.float32)

    assert _values(first) == _values(second)


@pytest.mark.parametrize(
    ("first_seed", "second_seed"),
    [(0, 2**63 - 1), (0, 2**64 - 2), (1, 2**64 - 1)],
)
def test_keyed_normal_distinguishes_engine_seed_boundaries(
    first_seed: int, second_seed: int
) -> None:
    backend = mlx_backend()

    with use_backend(backend):
        first = random.normal(key(first_seed), (64,), dtype=mx.float32)
        second = random.normal(key(second_seed), (64,), dtype=mx.float32)

    assert _values(first) != _values(second)


def test_keyed_normal_ignores_global_rng_draws() -> None:
    backend = mlx_backend()
    rng_key = key(41)

    with use_backend(backend):
        expected = random.normal(rng_key, (64,), dtype=mx.float32)
        mx.random.seed(17)
        unrelated = mx.random.normal((128,))
        mx.eval(unrelated)
        actual = random.normal(rng_key, (64,), dtype=mx.float32)

    assert _values(expected) == _values(actual)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_keyed_normal_honors_shape_and_dtype(dtype: Any) -> None:
    backend = mlx_backend()
    like = mx.zeros((0,), dtype=mx.float32)

    with use_backend(backend):
        sample = random.normal(key(5), (2, 3), dtype=dtype, like=like)

    assert sample.shape == (2, 3)
    assert sample.dtype == dtype


def test_keyed_normal_registration_activates_mlx_provider() -> None:
    from opaque.api.mlx.backend import _core as provider

    clear_backend()
    backend = mlx_backend()

    assert random.normal.resolve(backend) is provider.normal
    with use_backend(backend):
        assert active_backend() is backend
        sample = random.normal(key(5), (1,), dtype=mx.float32)

    assert isinstance(sample, mx.array)


def test_public_core_contract_uses_mlx_registration() -> None:
    backend = mlx_backend()
    tree = {"flat.key": mx.array([2.0]), "nested": [mx.array([3.0])]}

    with use_backend(backend):
        grads, value = autodiff.grad_and_value(lambda x: ops.sum(ops.square(x)))(
            mx.array([2.0])
        )
        assert _values(grads) == [4.0]
        assert value.item() == pytest.approx(4.0)
        assert _values(autodiff.vmap(lambda x: x * 2)(mx.array([1.0, 2.0]))) == [
            2.0,
            4.0,
        ]

        paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
        assert set(paths) == {("flat.key",), ("nested", 0)}
        assert pytree.tree_unflatten(treedef, leaves).keys() == tree.keys()

        first = random.normal(random.key(13), (2,), dtype=mx.float32)
        second = random.normal(random.key(13), (2,), dtype=mx.float32)
        assert _values(first) == _values(second)


def test_registry_swap_and_restore() -> None:
    backend = mlx_backend()
    previous = active_backend()

    with use_backend(backend) as yielded:
        assert yielded is backend
        assert active_backend() is backend

    assert active_backend() is previous


def test_tree_leaves_filters_out_non_array_leaves() -> None:
    backend = mlx_backend()
    tree = {
        "w": mx.array([1.0, 2.0]),
        "meta": 5,
        "nested": {"b": mx.array([3.0]), "c": "foo"},
    }
    with use_backend(backend):
        leaves = pytree.tree_leaves(tree)

    assert len(leaves) == 2
    assert all(isinstance(leaf, mx.array) for leaf in leaves)
    assert _values(leaves[0]) == [1.0, 2.0]
    assert _values(leaves[1]) == [3.0]
