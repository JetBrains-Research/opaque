"""Backend-neutral clipping and PyTree behavior."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.clipping._pytree import auto_scale_pytree
from opaque.pytree import (
    global_norm,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from opaque.types import PerGroup, SecondMomentClippingOutput


def _squared_error(params: object, sample: object) -> object:
    return ops.square(ops.subtract(params, sample))


def _vector_squared_error(params: object, sample: object) -> object:
    return ops.sum(ops.square(ops.subtract(params, sample)))


def test_fixed_clipping_preserves_direction_and_enforces_the_bound(
    backend_case,
) -> None:
    grad_fn, state = clipped_grad(
        _squared_error,
        clipping_norm=1.0,
        batch_argnums=1,
    )

    grads, _ = grad_fn(
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        backend_case.array([10.0], dtype=backend_case.dtype("float32")),
        state=state,
    )

    backend_case.assert_allclose(grads.pytree, -1.0)
    assert grads.max_norm == 1.0


def test_fixed_clipping_leaves_small_gradients_unchanged_and_reports_norms(
    backend_case,
) -> None:
    grad_fn, state = clipped_grad(
        _squared_error,
        clipping_norm=100.0,
        batch_argnums=1,
        return_aux=True,
    )

    (grads, aux), _ = grad_fn(
        backend_case.array(1.0, dtype=backend_case.dtype("float32")),
        backend_case.array([1.1, 1.2], dtype=backend_case.dtype("float32")),
        state=state,
    )

    backend_case.assert_allclose(grads.pytree, -0.6)
    backend_case.assert_allclose(aux.grad_norms, [0.2, 0.4], rtol=1e-5)
    backend_case.assert_allclose(aux.loss_values, [0.01, 0.04], rtol=1e-5)


def test_fixed_clipping_handles_empty_batches_and_argument_validation(
    backend_case,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        clipped_grad(_squared_error, clipping_norm=1.0, argnums=0, batch_argnums=0)
    with pytest.raises(ValueError, match="must not be empty"):
        clipped_grad(_squared_error, clipping_norm=1.0, batch_argnums=())

    grad_fn, state = clipped_grad(
        _squared_error,
        clipping_norm=1.0,
        batch_argnums=1,
        return_stats=True,
    )
    (grads, stats), _ = grad_fn(
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        backend_case.array([], dtype=backend_case.dtype("float32")),
        state=state,
    )

    backend_case.assert_allclose(grads.pytree, 0.0)
    assert stats.all_finite is True


def test_microbatching_matches_full_batch_for_pytree_gradients_and_diagnostics(
    backend_case,
) -> None:
    params = {
        "weight": backend_case.array([0.0, 0.0], dtype=backend_case.dtype("float32"))
    }
    samples = backend_case.array(
        [[3.0, 4.0], [1.0, -2.0], [-4.0, 3.0], [2.0, 1.0]],
        dtype=backend_case.dtype("float32"),
    )
    full_fn, full_state = clipped_grad(
        lambda p, sample: _vector_squared_error(p["weight"], sample),
        clipping_norm=2.0,
        batch_argnums=1,
        return_aux=True,
    )
    micro_fn, micro_state = clipped_grad(
        lambda p, sample: _vector_squared_error(p["weight"], sample),
        clipping_norm=2.0,
        batch_argnums=1,
        microbatch_size=2,
        return_aux=True,
    )

    (full_grads, full_aux), _ = full_fn(params, samples, state=full_state)
    (micro_grads, micro_aux), _ = micro_fn(params, samples, state=micro_state)

    backend_case.assert_allclose(
        micro_grads.pytree["weight"], full_grads.pytree["weight"]
    )
    backend_case.assert_allclose(micro_aux.grad_norms, full_aux.grad_norms)
    backend_case.assert_allclose(micro_aux.loss_values, full_aux.loss_values)


def test_second_moment_clipping_uses_per_example_squared_gradients(
    backend_case,
) -> None:
    grad_fn, state = clipped_grad(
        _squared_error,
        clipping_norm=100.0,
        normalize_by=2.0,
        batch_argnums=1,
        second_moment=True,
    )

    output, _ = grad_fn(
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        backend_case.array([1.0, 2.0], dtype=backend_case.dtype("float32")),
        state=state,
    )

    assert isinstance(output, SecondMomentClippingOutput)
    backend_case.assert_allclose(output.grads.pytree, -3.0)
    backend_case.assert_allclose(output.squared_grads.pytree, 10.0)
    assert output.grads.max_norm == 50.0
    assert output.squared_grads.max_norm == 5000.0


def test_auto_scale_preserves_the_formula_and_sanitizes_nonfinite_values(
    backend_case,
) -> None:
    values = backend_case.array(
        [3.0, 4.0, float("nan")], dtype=backend_case.dtype("float32")
    )

    scaled, aux = auto_scale_pytree({"weights": values}, R=2.0, gamma=1.0)

    backend_case.assert_allclose(aux.norm, 5.0)
    backend_case.assert_allclose(scaled["weights"], [1.0, 4.0 / 3.0, 0.0])


def test_auto_scale_applies_per_group_bounds_portably(backend_case) -> None:
    values = {
        "left": backend_case.array([3.0], dtype=backend_case.dtype("float32")),
        "right": backend_case.array([4.0], dtype=backend_case.dtype("float32")),
    }
    bounds = PerGroup(
        groups={"left": "first", "right": "second"},
        values={"first": 1.0, "second": 2.0},
    )

    scaled, aux = auto_scale_pytree(values, R=bounds, gamma=1.0)

    backend_case.assert_allclose(scaled["left"], [0.75])
    backend_case.assert_allclose(scaled["right"], [1.6])
    backend_case.assert_allclose(aux.group_norms["first"], 3.0)
    backend_case.assert_allclose(aux.group_norms["second"], 4.0)


def test_auto_scale_rejects_an_undefined_zero_gamma(backend_case) -> None:
    values = {"weights": backend_case.array([1.0])}

    with pytest.raises(ValueError, match="gamma must be positive"):
        auto_scale_pytree(values, gamma=0.0)


def test_pytree_mapping_round_trip_and_global_norm_are_portable(backend_case) -> None:
    tree = {
        "weight": backend_case.array([3.0, 4.0], dtype=backend_case.dtype("float32")),
        "bias": backend_case.array([12.0], dtype=backend_case.dtype("float32")),
        "metadata": "unchanged",
    }

    leaves, structure = tree_flatten(tree)
    rebuilt = tree_unflatten(structure, leaves)
    doubled = tree_map(
        lambda value: ops.multiply(value, 2) if ops.is_array(value) else value,
        rebuilt,
    )

    assert len(leaves) == 3
    assert len(tree_leaves(tree)) == 2
    backend_case.assert_allclose(global_norm(tree), 13.0)
    backend_case.assert_allclose(doubled["weight"], [6.0, 8.0])
    backend_case.assert_allclose(doubled["bias"], [24.0])
    assert doubled["metadata"] == "unchanged"
