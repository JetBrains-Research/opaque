"""Portable public variants of per-example gradient clipping."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.clipping import clipped_grad
from opaque.functional import with_batch_dim


def _squared_error(params: object, sample: object) -> object:
    return ops.square(ops.subtract(params, sample))


def test_clipped_grad_validates_incompatible_return_and_batch_options(
    backend_case,
) -> None:
    del backend_case

    with pytest.raises(ValueError, match="overlap"):
        clipped_grad(_squared_error, clipping_norm=1.0, argnums=0, batch_argnums=0)
    with pytest.raises(ValueError, match="must not be empty"):
        clipped_grad(_squared_error, clipping_norm=1.0, batch_argnums=())
    with pytest.raises(ValueError, match="cannot be combined"):
        clipped_grad(
            _squared_error,
            clipping_norm=1.0,
            return_aux=True,
            return_stats=True,
        )


@pytest.mark.parametrize("microbatch_size", [None, 1])
def test_clipped_grad_reports_nonfinite_inputs_and_returns_bounded_gradients(
    backend_case, microbatch_size: int | None
) -> None:
    def loss(param: object, data: object) -> object:
        return ops.sqrt(ops.subtract(data, param))

    grad_fn, state = clipped_grad(
        loss,
        clipping_norm=1.0,
        batch_argnums=1,
        microbatch_size=microbatch_size,
        return_stats=True,
    )
    (grads, stats), _ = grad_fn(
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        backend_case.array([1.0, -1.0], dtype=backend_case.dtype("float32")),
        state=state,
    )

    assert stats.all_finite is False
    assert ops.all(ops.isfinite(grads.pytree))


def test_clipped_grad_preserves_loss_aux_and_normalizes_the_aggregate(
    backend_case,
) -> None:
    def loss_with_aux(
        param: object, sample: object
    ) -> tuple[object, dict[str, object]]:
        return _squared_error(param, sample), {"sample": sample}

    grad_fn, state = clipped_grad(
        loss_with_aux,
        clipping_norm=100.0,
        batch_argnums=1,
        has_aux=True,
        return_aux=True,
        normalize_by=2.0,
    )
    (grads, diagnostics), _ = grad_fn(
        backend_case.array(0.0, dtype=backend_case.dtype("float32")),
        backend_case.array([1.0, 2.0], dtype=backend_case.dtype("float32")),
        state=state,
    )

    backend_case.assert_allclose(grads.pytree, -3.0)
    backend_case.assert_allclose(diagnostics.grad_norms, [2.0, 4.0])
    backend_case.assert_allclose(diagnostics.loss_values, [1.0, 4.0])
    backend_case.assert_allclose(diagnostics.loss_aux["sample"], [1.0, 2.0])


def test_with_batch_dim_preserves_per_example_clipped_gradients(backend_case) -> None:
    def scalar_loss(param: object, sample: object) -> object:
        return ops.square(ops.subtract(param, sample))

    def batched_loss(param: object, sample: object) -> object:
        return ops.sum(ops.square(ops.subtract(param, sample)))

    scalar_grad, scalar_state = clipped_grad(
        scalar_loss, clipping_norm=100.0, batch_argnums=1
    )
    batched_grad, batched_state = clipped_grad(
        with_batch_dim(batched_loss, batch_argnums=1),
        clipping_norm=100.0,
        batch_argnums=1,
    )
    param = backend_case.array(0.0, dtype=backend_case.dtype("float32"))
    samples = backend_case.array([1.0, 2.0], dtype=backend_case.dtype("float32"))

    scalar, _ = scalar_grad(param, samples, state=scalar_state)
    batched, _ = batched_grad(param, samples, state=batched_state)

    backend_case.assert_allclose(batched.pytree, scalar.pytree)
