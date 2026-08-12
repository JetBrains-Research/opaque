"""Conformance checks for the public portable authoring surface."""

from __future__ import annotations

import pytest
import torch

from opaque import autodiff, ops, pytree, random
from opaque.api.engine.backend import active_backend, clear_backend, ensure_backend
from opaque.api.engine.primitive import CORE_PRIMITIVES, supports


@pytest.fixture(autouse=True)
def _activate_torch_backend():
    clear_backend()
    ensure_backend(torch.tensor(0.0))
    yield
    clear_backend()


def test_torch_satisfies_complete_portable_profile() -> None:
    assert active_backend().name == "torch"
    assert CORE_PRIMITIVES
    assert all(supports(primitive, "torch") for primitive in CORE_PRIMITIVES)


def test_ops_use_native_arrays_and_dtype_semantics() -> None:
    value = torch.tensor([3.0, 4.0], dtype=torch.float16)

    assert ops.is_array(value)
    assert ops.shape(value) == (2,)
    assert ops.dtype(value) is torch.float16
    assert ops.is_low_precision(value)
    assert ops.real_dtype(torch.complex64) is torch.float32
    assert ops.scalar(1.0, dtype=ops.dtype(value), like=value).dtype is torch.float16
    assert ops.zeros((2,), dtype=ops.dtype(value), like=value).dtype is torch.float16
    assert torch.equal(
        ops.square(value), torch.tensor([9.0, 16.0], dtype=torch.float16)
    )


def test_autodiff_uses_grads_value_order_and_vmap_error_mode() -> None:
    def loss(value: torch.Tensor) -> torch.Tensor:
        return ops.sum(ops.square(value))

    grads, value = autodiff.grad_and_value(loss)(torch.tensor([3.0, 4.0]))
    assert torch.equal(grads, torch.tensor([6.0, 8.0]))
    assert value.item() == pytest.approx(25.0)
    assert torch.equal(
        autodiff.vmap(lambda value: value * 2)(torch.tensor([1, 2])),
        torch.tensor([2, 4]),
    )


def test_dispatched_pytree_paths_and_keyed_normal_are_stable() -> None:
    tree = {"nested": [torch.tensor(1.0)], "flat.key": torch.tensor(2.0)}
    paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)

    assert set(paths) == {("nested", 0), ("flat.key",)}
    assert torch.equal(
        pytree.tree_unflatten(treedef, leaves)["flat.key"], tree["flat.key"]
    )
    assert torch.equal(
        pytree.tree_map(lambda value: value * 2, tree)["nested"][0], torch.tensor(2.0)
    )

    first = random.normal(random.key(7), (2, 3), dtype=torch.float32)
    second = random.normal(random.key(7), (2, 3), dtype=torch.float32)
    assert torch.equal(first, second)
