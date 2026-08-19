# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Transforms Opaque does not use keep upstream behaviour after patching.

The checkpoint shim conditions two process-global ``torch.func`` behaviours: the
internal ``create_graph`` of a first-order backward, and the saved-tensor-hooks
guard. Both now follow the composition that is actually running, so everything
other than a lone first-order transform must behave as it does on stock torch.
CPU-only, and valid on every torch regime.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd.graph import save_on_cpu
from torch.func import grad, hessian, jacrev, jvp, vjp, vmap
from torch.utils.checkpoint import checkpoint

from opaque.api.engine.clipping import clipped_grad
from opaque.api.patches.torch.checkpoint import native_support
from opaque.patches import apply_runtime_patches

apply_runtime_patches(vmap_checkpointing=True)

# d²/dx² x³ = 6x, so 12.0 at x = 2 for every reverse-over-reverse spelling.
SECOND_DERIVATIVE = 12.0

backport_only = pytest.mark.skipif(
    native_support.native_checkpoint_support(),
    reason="torch conditions create_graph natively; the backport is not applied",
)


def cube(x):
    return (x**3).sum()


@pytest.fixture
def create_graph_flags(monkeypatch):
    """Record the ``create_graph`` each functorch backward asks autograd for."""
    flags: list[bool] = []
    orig = torch.autograd.grad

    def spy(*args, create_graph=False, **kwargs):
        flags.append(create_graph)
        return orig(*args, create_graph=create_graph, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", spy)
    return flags


def test_grad_of_grad_matches_upstream():
    got = grad(grad(cube))(torch.tensor(2.0))
    assert got.item() == pytest.approx(SECOND_DERIVATIVE)


def test_jacrev_of_grad_matches_upstream():
    got = jacrev(grad(cube))(torch.tensor(2.0))
    assert got.item() == pytest.approx(SECOND_DERIVATIVE)


def test_vjp_of_grad_matches_upstream():
    _, backward = vjp(grad(cube), torch.tensor(2.0))
    assert backward(torch.tensor(1.0))[0].item() == pytest.approx(SECOND_DERIVATIVE)


def test_forward_over_reverse_matches_upstream():
    _, tangent = jvp(grad(cube), (torch.tensor(2.0),), (torch.tensor(1.0),))
    assert tangent.item() == pytest.approx(SECOND_DERIVATIVE)
    assert hessian(cube)(torch.tensor(2.0)).item() == pytest.approx(SECOND_DERIVATIVE)


def test_vmap_of_higher_order_matches_upstream():
    got = vmap(grad(grad(cube)))(torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(got, torch.tensor([12.0, 18.0]))


def test_autograd_differentiates_through_grad():
    x = torch.tensor(2.0, requires_grad=True)
    (second,) = torch.autograd.grad(grad(cube)(x), x)
    assert second.item() == pytest.approx(SECOND_DERIVATIVE)


def test_autograd_differentiates_through_vmap_grad():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    (second,) = torch.autograd.grad(vmap(grad(cube))(x).sum(), x)
    torch.testing.assert_close(second, torch.tensor([12.0, 18.0]))


def test_autograd_reaches_parameters_captured_by_the_transform():
    # The gradient-penalty shape: differentiate w.r.t. an input, then update the
    # weights the transformed function closed over.
    weight = torch.nn.Parameter(torch.tensor(2.0))
    penalty = grad(lambda x: (x * x * weight).sum())(torch.tensor(3.0))
    (dweight,) = torch.autograd.grad(penalty, weight)
    assert dweight.item() == pytest.approx(6.0)


@backport_only
def test_no_grad_entry_skips_the_inner_graph(create_graph_flags):
    with torch.no_grad():
        vmap(grad(cube))(torch.tensor([2.0, 3.0]))
    assert create_graph_flags == [False]


@backport_only
def test_grad_mode_entry_keeps_the_inner_graph(create_graph_flags):
    vmap(grad(cube))(torch.tensor([2.0, 3.0]))
    assert create_graph_flags == [True]


@backport_only
def test_clipped_grad_skips_the_inner_graph(create_graph_flags):
    # Opaque's own transform: the saving the shim exists for, on the path the
    # trainers take, regardless of the caller's ambient grad mode.
    grad_fn, state = clipped_grad(lambda w, x: ((x - w) ** 2).sum(), clipping_norm=1.0)
    grad_fn(torch.tensor(3.0), torch.tensor([0.0, 7.0]), state=state)
    assert create_graph_flags == [False]


@backport_only
def test_clipped_grad_keeps_the_inner_graph_under_an_outer_transform(
    create_graph_flags,
):
    # Nested, the clipped gradients are an intermediate the outer transform has
    # to differentiate: both backwards build their graph.  Skipping the inner
    # one here is what would answer 0 instead of 4.
    grad_fn, state = clipped_grad(
        lambda w, x: ((x - w) ** 2).sum(), clipping_norm=100.0
    )

    def clipped_sum(w):
        result, _ = grad_fn(w, torch.tensor([0.0, 7.0]), state=state)
        return result.pytree.sum()

    second = grad(clipped_sum)(torch.tensor(3.0))

    assert create_graph_flags == [True, True]
    assert second.item() == pytest.approx(4.0)


def test_first_order_allows_saved_tensor_hooks():
    with save_on_cpu():
        assert grad(cube)(torch.tensor(3.0)).item() == pytest.approx(27.0)


def test_higher_order_rejects_saved_tensor_hooks():
    with save_on_cpu(), pytest.raises(RuntimeError, match="saved tensor hooks"):
        jacrev(grad(cube))(torch.tensor(2.0))


# ---------------------------------------------------------------------------
# torch.compile: the stack cannot be read from inside a compiled region at all,
# so every scoping decision above has to have a constant to fall back on. Both
# live here rather than with the other compile tests because they only bite
# once the patches are installed, and nothing else in that suite installs them.
# ---------------------------------------------------------------------------


def test_compiles_the_dp_transform_with_the_patches_applied():
    """``fullgraph=True`` over ``clipped_grad``, patches and all.

    Reading the interpreter stack is what a compiled region cannot do: the
    stack itself is a pybind builtin and its interpreters are pybind objects,
    so both the clipping probe and ``prev_grad_mode`` abort the compilation.
    """
    grad_fn, state = clipped_grad(lambda w, x: ((x - w) ** 2).sum(), clipping_norm=1.0)
    args = (torch.tensor(3.0), torch.tensor([0.0, 7.0]))

    eager, _ = grad_fn(*args, state=state)
    compiled_fn = torch.compile(grad_fn, backend="aot_eager", fullgraph=True)
    compiled, _ = compiled_fn(*args, state=state)

    torch.testing.assert_close(compiled.pytree, eager.pytree)


def test_compiled_first_order_keeps_its_saved_tensor_hooks():
    """A compiled first-order transform still gets hooks, so checkpoint runs.

    The guard is what rejects the hooks non-reentrant checkpoint is built on.
    Answering "higher-order" for a compiled composition -- which is the safe
    assumption for the *clipping* probe -- would raise here instead.
    """
    weight = torch.randn(4, 4)

    def f(x):
        h = checkpoint(lambda z: torch.tanh(z @ weight), x, use_reentrant=False)
        return h.sum()

    compiled_fn = torch.compile(vmap(grad(f)), backend="aot_eager", fullgraph=True)
    x = torch.randn(3, 4)

    torch.testing.assert_close(compiled_fn(x), vmap(grad(f))(x))
