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


def test_like_carries_device_and_dtype_for_creation_ops() -> None:
    """``like=`` means the same thing everywhere it appears.

    Taking only the device would put a constant beside a ``float64`` leaf at
    the provider's default ``float32`` without saying so — the class of silent
    precision loss this contract exists to close.
    """
    leaf = torch.zeros(2, dtype=torch.float64)

    assert ops.scalar(1.0, like=leaf).dtype is torch.float64
    assert ops.zeros((2,), like=leaf).dtype is torch.float64
    assert random.normal(random.key(1), (2,), like=leaf).dtype is torch.float64

    # An explicit dtype overrides ``like``'s half of that.
    assert ops.scalar(1.0, dtype=ops.float32(), like=leaf).dtype is torch.float32
    assert ops.zeros((2,), dtype=ops.boolean(), like=leaf).dtype is torch.bool


def test_neutral_dtype_constructors_cover_the_documented_choices() -> None:
    assert ops.float32() is torch.float32
    assert ops.float64() is torch.float64
    assert ops.boolean() is torch.bool

    assert not ops.is_low_precision(ops.float32())
    assert not ops.is_low_precision(ops.float64())
    assert ops.is_low_precision(torch.bfloat16)
    assert ops.is_low_precision(torch.float16)


def test_nan_to_num_substitutes_zero_and_takes_explicit_values() -> None:
    """The default differs from NumPy and Torch, and must stay that way.

    Saturating an infinity to the dtype's largest finite value would hand a DP
    aggregate the biggest number it can hold; zero is the only substitution
    that leaves the sensitivity bound intact.
    """
    value = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0])

    assert ops.nan_to_num(value).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert ops.nan_to_num(value, nan=-1.0, posinf=2.0, neginf=-2.0).tolist() == [
        -1.0,
        2.0,
        -2.0,
        1.0,
    ]
    # ...and saturation stays expressible for callers that want it.
    largest = torch.finfo(torch.float32).max
    assert ops.nan_to_num(value, posinf=largest)[1].item() == largest


def test_clamp_bounds_values_but_not_nan() -> None:
    value = torch.tensor([float("nan"), 5.0, -5.0])
    clamped = ops.clamp(value, lo=0.0, hi=1.0)

    assert clamped[1].item() == 1.0
    assert clamped[2].item() == 0.0
    assert torch.isnan(clamped[0])


def test_reductions_and_predicates_return_arrays_not_python_numbers() -> None:
    """Staying an array is what lets a result survive ``vmap``."""
    assert ops.is_array(ops.sum(torch.ones(3)))
    assert ops.shape(ops.sum(torch.ones(3))) == ()
    assert ops.shape(ops.sum(torch.ones(2, 3), axis=0)) == (3,)
    assert ops.is_array(ops.mean(torch.ones(3)))

    predicate = ops.all(ops.isfinite(torch.ones(3)))
    assert ops.is_array(predicate)
    assert ops.dtype(predicate) is torch.bool
    assert ops.dtype(ops.greater(torch.ones(2), torch.zeros(2))) is torch.bool

    # ``minimum`` on booleans is the portable logical AND for folding
    # per-leaf predicates together.
    left = torch.tensor([True, True])
    right = torch.tensor([True, False])
    assert ops.minimum(left, right).tolist() == [True, False]
    assert ops.maximum(left, right).tolist() == [True, True]

    assert ops.scalar_item(ops.sum(torch.ones(3))) == 3.0


def test_dtype_helpers_accept_an_array_or_a_dtype() -> None:
    leaf = torch.ones(2, dtype=torch.float32)

    assert ops.finfo_eps(leaf) == ops.finfo_eps(torch.float32)
    assert ops.finfo_smallest_normal(leaf) == ops.finfo_smallest_normal(torch.float32)
    assert ops.promote_dtype(leaf, torch.float64) is torch.float64
    assert ops.promote_dtype(torch.float32, torch.float64) is torch.float64
    assert ops.real_dtype(torch.complex64) is torch.float32
    assert ops.real_dtype(torch.float32) is torch.float32
    assert ops.is_floating(leaf)
    assert not ops.is_complex(leaf)


def test_accumulator_dtype_widens_per_array() -> None:
    assert ops.accumulator_dtype(torch.ones(2, dtype=torch.float32)) is torch.float64
    # A low-precision leaf accumulates at float32, not float64.
    assert ops.accumulator_dtype(torch.ones(2, dtype=torch.bfloat16)) is torch.float32


def test_indexing_and_shape_ops_follow_the_documented_semantics() -> None:
    value = torch.ones(3, 2)

    assert ops.shape(ops.slice_array(value, 0)) == (2,)
    assert ops.shape(ops.slice_array(value, slice(0, 2))) == (2, 2)
    # A tuple indexes successive axes, so this drops both.
    assert ops.shape(ops.slice_array(value, (0, 1))) == ()

    assert ops.shape(ops.expand_dims(torch.ones(2), -1)) == (2, 1)
    assert ops.shape(ops.squeeze(torch.ones(1, 2, 1))) == (2,)
    assert ops.shape(ops.squeeze(torch.ones(1, 2, 1), axis=0)) == (2, 1)
    # Squeezing a non-unit axis leaves the array alone rather than raising.
    assert ops.shape(ops.squeeze(torch.ones(1, 2, 1), axis=1)) == (1, 2, 1)

    assert ops.shape(ops.concatenate([torch.ones(2), torch.ones(3)])) == (5,)
    assert ops.shape(ops.concatenate(t for t in (torch.ones(2), torch.ones(3)))) == (5,)


def test_arithmetic_follows_ieee_rather_than_raising() -> None:
    assert ops.divide(torch.tensor([7]), torch.tensor([2])).item() == 3.5
    assert ops.divide(torch.tensor([1.0]), torch.tensor([0.0])).item() == float("inf")
    assert torch.isnan(ops.pow(torch.tensor([-8.0]), 1 / 3.0))


def test_grad_and_value_returns_gradients_first() -> None:
    def loss(params, batch):
        return ops.sum(ops.multiply(params, batch))

    params = torch.ones(3)
    batch = torch.arange(3.0)

    grads, value = autodiff.grad_and_value(loss)(params, batch)
    assert torch.equal(grads, batch)
    assert ops.scalar_item(value) == 3.0

    def loss_with_aux(params, batch):
        return ops.sum(ops.multiply(params, batch)), ops.sum(batch)

    grads, (value, aux) = autodiff.grad_and_value(loss_with_aux, has_aux=True)(
        params, batch
    )
    assert torch.equal(grads, batch)
    assert ops.scalar_item(aux) == 3.0

    both, value = autodiff.grad_and_value(loss, argnums=(0, 1))(params, batch)
    assert len(both) == 2
    assert torch.equal(both[0], batch)
    assert torch.equal(both[1], params)
