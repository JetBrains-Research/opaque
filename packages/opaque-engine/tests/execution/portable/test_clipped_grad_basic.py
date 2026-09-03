"""Portable clipping behavior exercised through every required provider."""

from __future__ import annotations

from opaque import ops
from opaque.api.engine.clipping import clipped_grad


def _squared_error(params, sample):
    return ops.sum(ops.square(ops.subtract(params, sample)))


def test_fixed_clipped_grad_aggregates_per_example_gradients_portably(
    backend_case,
) -> None:
    params = backend_case.array([1.0, -1.0], dtype=backend_case.dtype("float32"))
    samples = backend_case.array(
        [[0.0, 0.0], [2.0, 0.0]], dtype=backend_case.dtype("float32")
    )
    gradient, state = clipped_grad(
        _squared_error,
        clipping_norm=1.0,
        normalize_by=2.0,
        batch_argnums=1,
    )

    clipped_grads, state = gradient(params, samples, state=state)

    backend_case.assert_allclose(
        clipped_grads.pytree,
        [0.0, -(2**-0.5)],
        rtol=1e-5,
        atol=1e-6,
    )
    assert clipped_grads.max_norm == 0.5
    assert state is not None
