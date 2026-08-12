"""Behavioral coverage for the JAX implementation of the backend protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from opaque import autodiff, ops, pytree, random
from opaque.api.engine.backend import (
    Backend,
    TorchBackend,
    active_backend,
    set_backend,
    use_backend,
)
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.types import PerGroup
from opaque.jax import jax_backend
from opaque.random import key

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    yield
    set_backend(TorchBackend())


def _values(value: Any) -> Any:
    return np.asarray(value).tolist()


def test_satisfies_protocol_and_array_primitives() -> None:
    backend = jax_backend()
    x = jnp.array([3.0, 4.0])

    assert isinstance(backend, Backend)
    assert backend.name == "jax"
    assert backend.float32 == jnp.float32
    assert _values(backend.square(x)) == [9.0, 16.0]
    assert float(backend.sum(backend.square(x))) == pytest.approx(25.0)
    assert float(backend.sqrt(backend.sum(backend.square(x)))) == pytest.approx(5.0)
    assert _values(backend.minimum(x, jnp.array([2.0, 5.0]))) == [2.0, 4.0]
    assert _values(backend.maximum(x, jnp.array([2.0, 5.0]))) == [3.0, 5.0]
    assert _values(backend.where(x > 3.0, x, 0.0)) == [0.0, 4.0]
    assert _values(backend.zeros_like(x)) == [0.0, 0.0]
    assert _values(backend.astype(x, jnp.int32)) == [3, 4]
    assert backend.is_array(x)
    assert not backend.is_array(np.array([1.0]))
    assert backend.is_floating(x)
    assert backend.is_floating(jnp.float32)
    assert backend.concatenate([x, x]).shape == (4,)
    assert backend.scalar(1.0, dtype=jnp.float32, like=x).dtype == jnp.float32
    assert backend.promote_dtype(jnp.float16, jnp.float32) == jnp.float32


@pytest.mark.parametrize(
    ("dtype", "low_precision", "floating", "complex"),
    [
        (jnp.float16, True, True, False),
        (jnp.bfloat16, True, True, False),
        (jnp.float32, False, True, False),
        (jnp.int32, False, False, False),
        (jnp.complex64, False, False, True),
    ],
)
def test_dtype_predicates_for_arrays_and_dtypes(
    dtype: Any,
    low_precision: bool,
    floating: bool,
    complex: bool,
) -> None:
    backend = jax_backend()
    array = jnp.array([1], dtype=dtype)

    assert backend.is_low_precision(dtype) is low_precision
    assert backend.is_low_precision(array) is low_precision
    assert backend.is_floating(dtype) is floating
    assert backend.is_floating(array) is floating
    assert backend.is_complex(dtype) is complex
    assert backend.is_complex(array) is complex


def test_value_and_grad_normalizes_value_and_aux_order() -> None:
    backend = jax_backend()
    x = jnp.array([2.0, 3.0])

    grad, value = backend.value_and_grad(lambda t: jnp.sum(t * t))(x)
    assert _values(grad) == [4.0, 6.0]
    assert float(value) == pytest.approx(13.0)

    def loss_with_aux(t: Any) -> tuple[Any, dict[str, Any]]:
        return jnp.sum(t * t), {"mean": jnp.mean(t)}

    grad, (value, aux) = backend.value_and_grad(loss_with_aux, has_aux=True)(x)
    assert _values(grad) == [4.0, 6.0]
    assert float(value) == pytest.approx(13.0)
    assert float(aux["mean"]) == pytest.approx(2.5)


def test_vmap_composes_with_value_and_grad() -> None:
    backend = jax_backend()
    weights = jnp.array([2.0, 3.0])
    examples = jnp.array([[1.0, 4.0], [5.0, 6.0]])

    grad_and_value = backend.value_and_grad(lambda w, x: jnp.sum(w * x))
    grads, values = backend.vmap(grad_and_value, in_axes=(None, 0))(weights, examples)

    assert _values(grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(values) == [14.0, 28.0]

    grad_and_value_with_aux = backend.value_and_grad(
        lambda w, x: (jnp.sum(w * x), jnp.sum(x)), has_aux=True
    )
    grads, (values, aux) = backend.vmap(grad_and_value_with_aux, in_axes=(None, 0))(
        weights, examples
    )
    assert _values(grads) == [[1.0, 4.0], [5.0, 6.0]]
    assert _values(values) == [14.0, 28.0]
    assert _values(aux) == [5.0, 11.0]

    with pytest.raises(ValueError, match="randomness='error'"):
        backend.vmap(grad_and_value, in_axes=(None, 0), randomness="different")


def test_pytree_paths_support_flat_and_nested_per_group_clipping() -> None:
    backend = jax_backend()
    tree = {
        "layers.0.weight": jnp.array([3.0, 4.0]),
        "blocks": [{"bias": jnp.array([6.0])}],
    }
    paths, leaves, treedef = backend.tree_flatten_with_paths(tree)

    assert set(paths) == {("layers.0.weight",), ("blocks", 0, "bias")}
    assert backend.tree_unflatten(treedef, leaves) is not tree

    groups = PerGroup(
        {("layers.0.weight",): "weight", ("blocks", 0, "bias"): "bias"},
        {"weight": 1.0, "bias": 2.0},
    )
    with use_backend(backend):
        clipped, aux = clip_pytree(tree, groups)

    assert aux.group_norms is not None
    assert _values(clipped["layers.0.weight"]) == pytest.approx([0.6, 0.8])
    assert _values(clipped["blocks"][0]["bias"]) == pytest.approx([2.0])


def test_empty_tree_and_nonfinite_sanitization() -> None:
    backend = jax_backend()
    nonfinite = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 2.0])

    assert backend.tree_leaves({}) == []
    assert backend.tree_leaves({"metadata": 5, "text": "foo"}) == []
    assert backend.tree_flatten({})[0] == []
    assert _values(backend.nan_to_num(nonfinite)) == [0.0, 0.0, 0.0, 2.0]
    assert _values(backend.clamp(jnp.array([-1.0, 0.5, 2.0]), lo=0.0, hi=1.0)) == [
        0.0,
        0.5,
        1.0,
    ]
    with use_backend(backend):
        clipped, aux = clip_pytree({"w": nonfinite}, 1.0)
        empty_clipped, empty_aux = clip_pytree({}, 1.0)

    assert _values(clipped["w"]) == [0.0, 0.0, 0.0, 1.0]
    assert float(aux.norm) == pytest.approx(2.0)
    assert empty_clipped == {}
    assert float(empty_aux.norm) == pytest.approx(0.0)


def test_keyed_normal_sampling_is_repeatable() -> None:
    backend = jax_backend()
    first = backend.normal(
        (2, 3), dtype=jnp.float32, generator=backend.generator(key(7))
    )
    second = backend.normal(
        (2, 3), dtype=jnp.float32, generator=backend.generator(key(7))
    )
    different = backend.normal(
        (2, 3), dtype=jnp.float32, generator=backend.generator(key(8))
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_public_core_contract_uses_jax_registration() -> None:
    backend = jax_backend()
    tree = {"flat.key": jnp.array([2.0]), "nested": [jnp.array([3.0])]}

    with use_backend(backend):
        grads, value = autodiff.grad_and_value(lambda x: ops.sum(ops.square(x)))(
            jnp.array([2.0])
        )
        assert _values(grads) == [4.0]
        assert float(value) == pytest.approx(4.0)
        assert _values(autodiff.vmap(lambda x: x * 2)(jnp.array([1.0, 2.0]))) == [
            2.0,
            4.0,
        ]
        with pytest.raises(ValueError, match="randomness='error'"):
            autodiff.vmap(lambda x: x, randomness="same")

        paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
        assert set(paths) == {("flat.key",), ("nested", 0)}
        assert pytree.tree_unflatten(treedef, leaves).keys() == tree.keys()

        first = random.normal(random.key(13), (2,), dtype=jnp.float32)
        second = random.normal(random.key(13), (2,), dtype=jnp.float32)
        np.testing.assert_array_equal(first, second)


def test_registry_swap_and_restore() -> None:
    backend = jax_backend()
    previous = active_backend()

    with use_backend(backend) as yielded:
        assert yielded is backend
        assert active_backend() is backend

    assert active_backend() is previous
    assert active_backend().name != "jax"


def test_tree_leaves_filters_out_non_array_leaves() -> None:
    backend = jax_backend()
    tree = {
        "w": jnp.array([1.0, 2.0]),
        "meta": 5,
        "nested": {"b": jnp.array([3.0]), "c": "foo"},
    }
    leaves = backend.tree_leaves(tree)

    assert len(leaves) == 2
    assert all(backend.is_array(leaf) for leaf in leaves)
    assert {tuple(_values(leaf)) for leaf in leaves} == {(1.0, 2.0), (3.0,)}
