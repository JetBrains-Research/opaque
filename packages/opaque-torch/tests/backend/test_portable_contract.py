"""Conformance checks: the Torch provider satisfies the portable core."""

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


def test_autodiff_uses_grads_value_order_and_vmap_randomness_modes() -> None:
    def loss(value: torch.Tensor) -> torch.Tensor:
        return ops.sum(ops.square(value))

    grads, value = autodiff.grad_and_value(loss)(torch.tensor([3.0, 4.0]))
    assert torch.equal(grads, torch.tensor([6.0, 8.0]))
    assert value.item() == pytest.approx(25.0)
    assert torch.equal(
        autodiff.vmap(lambda value: value * 2)(torch.tensor([1, 2])),
        torch.tensor([2, 4]),
    )
    # Default "same": RNG ops inside the mapped function share one draw
    # across batch elements (the pre-split torch.func behavior).
    same_draws = autodiff.vmap(lambda _: torch.rand(3))(torch.arange(4))
    assert same_draws.shape == (4, 3)
    assert torch.equal(same_draws[0], same_draws[1])
    # Explicit "error" rejects RNG ops under the transform.
    with pytest.raises(RuntimeError, match="randomness error mode"):
        autodiff.vmap(lambda _: torch.rand(3), randomness="error")(torch.arange(4))


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


def test_reduction_min_max_are_distinct_from_the_elementwise_pair() -> None:
    """``amin`` / ``amax`` reduce; ``minimum`` / ``maximum`` compare.

    The optimizer-state audit needs the reduction form to fingerprint a leaf,
    and reaching for the elementwise pair by mistake returns the input shape
    rather than a scalar — a fingerprint that silently compares nothing.
    """
    value = torch.tensor([[3.0, 1.0], [2.0, 4.0]])

    assert ops.shape(ops.amin(value)) == ()
    assert ops.shape(ops.amax(value)) == ()
    assert ops.scalar_item(ops.amin(value)) == 1.0
    assert ops.scalar_item(ops.amax(value)) == 4.0

    assert ops.shape(ops.amin(value, axis=0)) == (2,)
    assert ops.amax(value, axis=1).tolist() == [3.0, 4.0]

    # The elementwise pair keeps the shape, which is why they are not
    # interchangeable.
    assert ops.shape(ops.minimum(value, value)) == (2, 2)

    # NaN propagates through the reduction rather than being skipped.
    with_nan = torch.tensor([1.0, float("nan"), 3.0])
    assert torch.isnan(ops.amin(with_nan))
    assert torch.isnan(ops.amax(with_nan))


def test_to_host_returns_a_copy_not_a_view() -> None:
    """`to_host` promises a copy, and the promise has to hold on CPU too.

    `Tensor.numpy()` shares storage and `.cpu()` is a no-op for a tensor
    already there, so the natural spelling aliases on CPU while copying on
    CUDA. A caller normalizing scores in place would then write back into the
    graph on one device and not the other — the kind of device-dependent
    behaviour a portable primitive exists to rule out.
    """
    import numpy as np

    source = torch.ones(4)
    host = ops.to_host(source)

    assert not np.shares_memory(host, source.detach().numpy())

    host[0] = 99.0
    assert source.tolist() == [1.0, 1.0, 1.0, 1.0]

    source[1] = -5.0
    assert host.tolist() == [99.0, 1.0, 1.0, 1.0]


def test_to_host_detaches_from_autograd() -> None:
    source = torch.ones(3, requires_grad=True)
    host = ops.to_host(source * 2)

    assert host.tolist() == [2.0, 2.0, 2.0]


def test_low_precision_draws_keep_full_resolution() -> None:
    """A provider draws at no less than ``float32``, then returns the request.

    Nothing in the surface forces the sample *itself* to be computed in the
    requested dtype, and it must not be: a Gaussian generated natively in
    ``bfloat16`` would be coarser than the same draw made in ``float32`` and
    cast down, which silently shrinks the noise's effective support.  Torch
    already satisfies this, so the check is a conformance pin for the next
    provider rather than a bug report about this one.

    Resolution of the *arithmetic* around the draw is a separate obligation and
    belongs to the mechanism — see ``docs/user-guide/precision.md``.
    """
    for dtype in (torch.bfloat16, torch.float16):
        drawn = random.normal(random.key(11), (20_000,), dtype=dtype)
        upcast = random.normal(random.key(11), (20_000,), dtype=torch.float32)

        assert drawn.dtype is dtype
        assert len(torch.unique(drawn)) >= len(torch.unique(upcast.to(dtype)))


def test_like_supplies_dtype_and_placement_for_draws() -> None:
    leaf = torch.zeros(4, dtype=torch.float64)

    assert random.normal(random.key(3), (4,), like=leaf).dtype is torch.float64
    # An explicit dtype wins over ``like``'s.
    assert (
        random.normal(random.key(3), (4,), dtype=torch.float32, like=leaf).dtype
        is torch.float32
    )


def test_one_key_two_shapes_are_not_independent_draws() -> None:
    """Pins the hazard the ``normal`` docstring warns about.

    Changing only the shape does not produce a fresh sample; it extends the
    same one.  A mechanism that varies shape per leaf without folding first
    hands correlated noise to every leaf.
    """
    k = random.key(5)
    short = random.normal(k, (4,))
    long = random.normal(k, (8,))
    assert torch.equal(short, long[:4])

    folded = random.normal(random.fold_in(k, 0), (8,))
    assert not torch.equal(long, folded)
